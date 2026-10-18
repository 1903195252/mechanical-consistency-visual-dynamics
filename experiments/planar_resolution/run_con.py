"""Stream the native CON objective at 256/512, with an allocation guard.

The local preflight records a memory block before importing/allocating the model.
On a sufficiently large GPU, benchmark the complete batch before full training.
"""
import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import time
import numpy as np
from .common import ROOT, OUT, dataset, training_records, sha, write

STOP = False


def memory_check(width, folder):
    rows = json.loads((OUT / "model_preflight.json").read_text())["rows"]
    row = next(r for r in rows if r["model"] == "CON" and r["resolution"] == width)
    free = [float(x)/1024 for x in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and all(v.isdigit() for v in visible.split(",")):
        free = [free[int(v)] for v in visible.split(",")]
    minimum = row["fp32_params_grad_adam_lower_bound_GiB"] + row["input_batch_fp32_GiB"]
    report = dict(width=width, available_gpu_GiB=free, known_lower_bound_GiB=minimum,
                  full_objective_peak_memory_measured=False,
                  activation_memory_included=False, native_architecture=True,
                  gpu_memory_not_pooled=True,
                  status="needs_complete_loss_memory_benchmark" if free[0] > minimum else "blocked_by_memory")
    write(folder / "memory_preflight.json", report)
    return report


def run(args):
    folder = OUT / ("preflight" if args.mode != "train" else "runs") / f"con_{args.width}_seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    memory = memory_check(args.width, folder)
    if args.mode == "check":
        print(json.dumps(memory), flush=True)
        return
    if memory["status"] == "blocked_by_memory":
        raise RuntimeError("Native CON training does not fit this GPU; see memory_preflight.json")
    if args.frame_chunk:
        equivalence_path = OUT/"validation/con_memory_equivalence.json"
        if not equivalence_path.exists():
            raise RuntimeError("Run verify_con_memory before using framewise CON execution")
        equivalence = json.loads(equivalence_path.read_text())
        if not (equivalence.get("passed") and equivalence.get("frame_chunk") == args.frame_chunk
                and equivalence.get("matmul_precision") == "highest"
                and equivalence.get("adapter_sha256") == sha(ROOT/"experiments/planar_resolution/con_memory.py")):
            raise RuntimeError("CON framewise objective/gradient equivalence has not passed for this adapter")
    write(folder/"status.json",dict(stage="initializing",width=args.width,seed=args.seed,pid=os.getpid()))
    # Dependency/model imports follow the guard so an unsuitable host never
    # initializes billion-parameter weights just to establish infeasibility.
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from diffrax import Dopri5
    from src.models.autoencoders.vae import VAE
    from src.models.neural_odes.con_iae import ConIaeOde
    from src.models.dynamics_autoencoder import DynamicsAutoencoder
    from src.tasks.dynamics_autoencoder import task_factory
    from src.training.optim import create_learning_rate_fn
    from .con_memory import FramewiseApply
    if all(d.platform != "gpu" for d in jax.devices()):
        raise RuntimeError("CON high-resolution execution requires a GPU runtime")
    data = dataset(args.width)
    rows = training_records(data)
    cfg = json.loads((ROOT / "experiments/con_corpus/protocol.json").read_text())["training"]
    assert len(rows)==813 and cfg["batch_size"]==16 and cfg["epochs"]==50
    report_path = folder / ("benchmark.json" if args.mode == "benchmark" else "training_report.json")
    if report_path.exists():
        raise FileExistsError(report_path)
    model = DynamicsAutoencoder(autoencoder=VAE(img_shape=(args.width,args.width,3), latent_dim=8, norm_layer=nn.LayerNorm),
                               dynamics=ConIaeOde(latent_dim=8,input_dim=14,num_layers=5,hidden_dim=30))
    key = jax.random.PRNGKey(args.seed)
    params = model.init(key, jnp.zeros((1,args.width,args.width,3),dtype=jnp.float32), method=model.forward_all_layers)["params"]
    task_model = FramewiseApply(model, args.frame_chunk) if args.frame_chunk else model
    task, _ = task_factory("planar_closed_chain",task_model,jnp.arange(50)/30,.01,
        loss_weights=cfg["loss_weights"],ae_type="beta_vae",solver=Dopri5(),latent_velocity_source="image-space-finite-differences")
    rate = create_learning_rate_fn(50,51,cfg["base_lr"],warmup_epochs=5)
    tx = optax.adamw(rate,weight_decay=cfg["weight_decay"])
    state = tx.init(params)
    def objective(p,b,k):
        loss,pred = task.loss_fn(b,p,rng=k,training=True)
        return loss,{name:value for name,value in task.compute_metrics_fn(b,pred).items() if name.startswith("mse_")}
    @partial(jax.jit, donate_argnums=(0, 1))
    def step(p,s,b,k):
        (loss,aux),g=jax.value_and_grad(objective,has_aux=True)(p,b,k)
        finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(g)]))
        changes,s=tx.update(g,s,p)
        return optax.apply_updates(p,changes),s,loss,aux,finite
    identity=dict(protocol_sha256=sha(ROOT/"experiments/planar_resolution/protocol.json"),
        frame_chunk=args.frame_chunk, matmul_precision="highest",
        memory_adapter_sha256=sha(ROOT/"experiments/planar_resolution/con_memory.py"),
        native_protocol_sha256=sha(ROOT/"experiments/con_corpus/protocol.json"),
        dataset_sha256=sha(data.root/"dataset_manifest.json"),source_sha256=sha(__file__))
    rng=np.random.RandomState(args.seed)
    epoch,cursor,updates,order=0,0,0,None
    samples=[]
    checkpoint=folder/"resume.p"
    if args.mode=="train" and checkpoint.exists():
        with checkpoint.open("rb") as handle: saved=pickle.load(handle)
        assert saved["identity"]==identity
        params,state,key=(saved[k] for k in ("params","opt_state","key"))
        rng.set_state(saved["rng"])
        epoch,cursor,updates,order=(saved[k] for k in ("epoch","cursor","updates","order"))
    def save(name,obj):
        path=folder/name; temporary=path.with_suffix(".partial")
        with temporary.open("wb") as handle: pickle.dump(jax.device_get(obj),handle,protocol=4)
        temporary.replace(path)
    def checkpoint_now():
        save("resume.p",dict(params=params,opt_state=state,key=key,rng=rng.get_state(),
             epoch=epoch,cursor=cursor,updates=updates,order=order,identity=identity))
    started=time.monotonic()
    last_checkpoint=started
    while epoch<50:
        if STOP:
            if args.mode=="train": checkpoint_now()
            return
        if order is None: order=rng.permutation(len(rows))
        indices=order[cursor:cursor+16]
        tick=time.monotonic()
        images=np.empty((len(indices),50,args.width,args.width,3),dtype=np.float32)
        commands=[]
        for i,index in enumerate(indices):
            item=data.load(rows[int(index)],purpose="training",model="CON")
            images[i]=item["rendering_ts"];commands.append(item["tau"])
        del item
        batch={"rendering_ts":images,"tau":np.stack(commands)}
        key,subkey=jax.random.split(key)
        write(folder/"status.json",dict(stage="compiling" if updates==0 else "updating",updates=updates,
             epoch=epoch+1,pid=os.getpid(),width=args.width,case_ids=[rows[int(i)]["case_id"] for i in indices]))
        params,state,loss,aux,finite=step(params,state,batch,subkey)
        jax.block_until_ready((params,state,loss,aux,finite))
        elapsed=time.monotonic()-tick
        if not bool(finite) or not np.isfinite(float(loss)):
            raise FloatingPointError("Nonfinite complete CON objective or gradient")
        updates+=1;cursor+=len(indices)
        row=dict(update=updates,epoch=epoch+1,seconds=elapsed,batch_size=len(indices),loss=float(loss),metrics={k:float(v) for k,v in aux.items()})
        write(folder/"status.json",row)
        if args.mode=="benchmark":
            samples.append(row)
            if updates==args.steps+2: break
        if cursor==len(rows): epoch+=1;cursor=0;order=None
        if args.mode=="train" and (time.monotonic()-last_checkpoint>=600 or cursor==0):
            checkpoint_now()
            last_checkpoint=time.monotonic()
    report=dict(width=args.width,seed=args.seed,identity=identity,updates=updates,epochs=epoch,
        frame_chunk=args.frame_chunk,
        full_objective=True,native_architecture=True,scientific_training_result=args.mode=="train",
        devices=[str(d) for d in jax.devices()],elapsed_s=time.monotonic()-started)
    if args.mode=="benchmark":
        report.update(samples=samples,steady_mean_s=float(np.mean([r["seconds"] for r in samples[2:]])),
                      device_memory_stats=jax.devices()[0].memory_stats())
    else:
        save("params.p",params)
        report["checkpoint_sha256"]=sha(folder/"params.p")
    write(report_path,report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width",type=int,choices=(256,512),required=True)
    parser.add_argument("--seed",type=int,choices=(42,43,44),default=42)
    parser.add_argument("--mode",choices=("check","benchmark","train"),required=True)
    parser.add_argument("--steps",type=int,default=3)
    parser.add_argument("--frame-chunk",type=int,default=0,
                        help="0: native batched images; positive: rematerialize independent image chunks")
    args=parser.parse_args()
    if args.steps<3: parser.error("At least 3 measured updates required")
    if args.frame_chunk<0: parser.error("Frame chunk must be nonnegative")
    def stop(*unused):
        global STOP
        STOP=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    run(args)
