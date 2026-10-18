"""Native CON with CPU AdamW states; full batch, loss and solver are unchanged."""
import argparse
import gc
import json
import os
import pickle
import signal
import time
import numpy as np
from .common import ROOT, OUT, dataset, training_records, sha, write

STOP = False


def run(args):
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
    from .con_offload import cpu_update

    for filename, source in [("con_memory_equivalence.json","con_memory.py"),
                             ("con_optimizer_equivalence.json","con_offload.py")]:
        report=json.loads((OUT/"validation"/filename).read_text())
        if not report["passed"] or report["adapter_sha256"]!=sha(ROOT/"experiments/planar_resolution"/source):
            raise ValueError("Unverified execution adapter: " + source)
    equivalent=json.loads((OUT/"validation/con_memory_equivalence.json").read_text())
    assert equivalent["frame_chunk"]==args.frame_chunk and equivalent["matmul_precision"]=="highest"
    gpu=jax.devices("gpu")[0];cpu=jax.devices("cpu")[0]
    data=dataset(args.width);rows=training_records(data)
    cfg=json.loads((ROOT/"experiments/con_corpus/protocol.json").read_text())["training"]
    assert len(rows)==813 and cfg["batch_size"]==16 and cfg["epochs"]==50
    folder=OUT/("preflight" if args.mode=="benchmark" else "runs")/f"con_{args.width}_seed{args.seed}"
    folder.mkdir(parents=True,exist_ok=True)
    report_path=folder/("benchmark.json" if args.mode=="benchmark" else "training_report.json")
    if report_path.exists():raise FileExistsError(report_path)
    write(folder/"memory_preflight.json",dict(width=args.width,optimizer_backend="cpu",
        full_objective_peak_memory_measured=False,status="testing_complete_gpu_gradient_with_cpu_adamw",
        gpu_memory_not_pooled=True))
    def status(stage,**extra):
        write(folder/"status.json",dict(stage=stage,pid=os.getpid(),width=args.width,
              optimizer_backend="cpu",**extra))
    status("initializing")
    model=DynamicsAutoencoder(autoencoder=VAE(img_shape=(args.width,args.width,3),latent_dim=8,norm_layer=nn.LayerNorm),
        dynamics=ConIaeOde(latent_dim=8,input_dim=14,num_layers=5,hidden_dim=30))
    key=jax.random.PRNGKey(args.seed)
    params=model.init(key,jnp.zeros((1,args.width,args.width,3),dtype=jnp.float32),method=model.forward_all_layers)["params"]
    task,_=task_factory("planar_closed_chain",FramewiseApply(model,args.frame_chunk),jnp.arange(50)/30,.01,
        loss_weights=cfg["loss_weights"],ae_type="beta_vae",solver=Dopri5(),latent_velocity_source="image-space-finite-differences")
    rate=create_learning_rate_fn(50,51,cfg["base_lr"],warmup_epochs=5)
    tx=optax.adamw(rate,weight_decay=cfg["weight_decay"])
    host_params=jax.device_put(params,cpu)
    with jax.default_device(cpu):state=tx.init(host_params)
    update=cpu_update(tx)
    @jax.jit
    def gradient_step(p,b,k):
        def objective(p):
            loss,pred=task.loss_fn(b,p,rng=k,training=True)
            return loss,{n:v for n,v in task.compute_metrics_fn(b,pred).items() if n.startswith("mse_")}
        (loss,aux),grad=jax.value_and_grad(objective,has_aux=True)(p)
        finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grad)]))
        return loss,aux,grad,finite
    identity=dict(protocol_sha256=sha(ROOT/"experiments/planar_resolution/protocol.json"),
        native_protocol_sha256=sha(ROOT/"experiments/con_corpus/protocol.json"),dataset_sha256=sha(data.root/"dataset_manifest.json"),
        source_sha256=sha(__file__),frame_chunk=args.frame_chunk,matmul_precision="highest",optimizer_backend="cpu",
        memory_adapter_sha256=sha(ROOT/"experiments/planar_resolution/con_memory.py"),
        optimizer_adapter_sha256=sha(ROOT/"experiments/planar_resolution/con_offload.py"))
    rng=np.random.RandomState(args.seed);epoch,cursor,updates,order=0,0,0,None
    checkpoint=folder/"resume.p"
    if args.mode=="train" and checkpoint.exists():
        del host_params,state,params
        gc.collect()
        with checkpoint.open("rb") as stream:saved=pickle.load(stream)
        assert saved["identity"]==identity
        host_params=jax.device_put(saved["params"],cpu);state=jax.device_put(saved["opt_state"],cpu)
        params=jax.device_put(host_params,gpu);key=jax.device_put(saved["key"],gpu)
        rng.set_state(saved["rng"])
        epoch,cursor,updates,order=(saved[k] for k in ("epoch","cursor","updates","order"))
        del saved
    def save(name,value):
        path=folder/name;temporary=path.with_suffix(".partial")
        with temporary.open("wb") as stream:pickle.dump(jax.device_get(value),stream,protocol=4)
        temporary.replace(path)
    def checkpoint_now():
        status("checkpointing",updates=updates,epoch=epoch)
        save("resume.p",dict(params=host_params,opt_state=state,key=key,rng=rng.get_state(),
            epoch=epoch,cursor=cursor,updates=updates,order=order,identity=identity))
    started=time.monotonic();last_checkpoint=started;samples=[]
    while epoch<50:
        if STOP:
            if args.mode=="train":checkpoint_now()
            status("stopped",updates=updates,epoch=epoch)
            return
        if order is None:order=rng.permutation(len(rows))
        indices=order[cursor:cursor+16];tick=time.monotonic()
        images=np.empty((len(indices),50,args.width,args.width,3),np.float32);commands=[]
        for i,index in enumerate(indices):
            item=data.load(rows[int(index)],purpose="training",model="CON")
            images[i]=item["rendering_ts"];commands.append(item["tau"])
        del item
        batch=dict(rendering_ts=images,tau=np.stack(commands))
        key,subkey=jax.random.split(key)
        status("differentiating",updates=updates,epoch=epoch+1)
        loss,aux,grad,finite=gradient_step(params,batch,subkey)
        jax.block_until_ready((loss,aux,grad,finite))
        if not bool(finite) or not np.isfinite(float(loss)):raise FloatingPointError("Nonfinite complete native CON gradient")
        differentiated=time.monotonic()
        host_grad=jax.device_put(grad,cpu);jax.block_until_ready(host_grad)
        # Flax initialization retains references to its parameter dictionaries.
        # Release transferred GPU buffers explicitly; deleting the Python tree
        # alone leaves a complete initial parameter copy live on the device.
        for value in jax.tree.leaves((params,grad)):
            value.delete()
        del grad,params
        status("cpu_adamw",updates=updates,epoch=epoch+1)
        host_params,state=update(host_params,state,host_grad)
        jax.block_until_ready((host_params,state));del host_grad
        params=jax.device_put(host_params,gpu);jax.block_until_ready(params)
        updates+=1;cursor+=len(indices)
        row=dict(update=updates,epoch=epoch+1,batch_size=len(indices),seconds=time.monotonic()-tick,
            load_and_gradient_s=differentiated-tick,transfer_and_optimizer_s=time.monotonic()-differentiated,
            loss=float(loss),metrics={k:float(v) for k,v in aux.items()})
        status("updated",**row)
        print(json.dumps(row),flush=True)
        if args.mode=="benchmark":
            samples.append(row)
            if updates==args.steps+2:break
        if cursor==len(rows):epoch+=1;cursor=0;order=None
        if args.mode=="train" and (time.monotonic()-last_checkpoint>=600 or cursor==0):
            checkpoint_now();last_checkpoint=time.monotonic()
    report=dict(width=args.width,seed=args.seed,identity=identity,updates=updates,epochs=epoch,
        parameters=sum(x.size for x in jax.tree.leaves(host_params)),
        full_objective=True,native_architecture=True,optimizer_backend="cpu",frame_chunk=args.frame_chunk,
        scientific_training_result=args.mode=="train",devices=[str(gpu),str(cpu)],elapsed_s=time.monotonic()-started)
    if args.mode=="benchmark":
        report.update(samples=samples,steady_mean_s=float(np.mean([r["seconds"] for r in samples[2:]])),device_memory_stats=gpu.memory_stats())
    else:
        save("params.p",host_params);report["checkpoint_sha256"]=sha(folder/"params.p")
    write(report_path,report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width",type=int,choices=(256,512),required=True)
    parser.add_argument("--seed",type=int,choices=(42,43,44),default=42)
    parser.add_argument("--mode",choices=("benchmark","train"),required=True)
    parser.add_argument("--frame-chunk",type=int,default=16)
    parser.add_argument("--steps",type=int,default=3)
    args=parser.parse_args()
    if args.steps<3:parser.error("Three measured updates required")
    def stop(*unused):
        global STOP
        STOP=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    run(args)
