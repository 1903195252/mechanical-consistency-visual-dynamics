"""Compare the complete native CON objective and gradients with frame chunking."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from .common import ROOT, OUT, sha, write
from .con_memory import FramewiseApply
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from flax import linen as nn
from diffrax import Dopri5
from src.models.autoencoders.vae import VAE
from src.models.neural_odes.con_iae import ConIaeOde
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.tasks.dynamics_autoencoder import task_factory


def relative_gradient(a, b):
    pairs = list(zip(jax.tree.leaves(a), jax.tree.leaves(b)))
    numerator = sum(np.sum((x.astype(float)-y.astype(float))**2) for x,y in pairs)
    denominator = sum(np.sum(x.astype(float)**2) for x,_ in pairs)
    return float(np.sqrt(numerator/max(denominator,1e-30)))


def run(args):
    manifest = json.loads((ROOT/"output/keycld_corpus/dataset_manifest.json").read_text())
    rows = [r for r in manifest["records"] if r["split"] == "train"][:2]
    images, commands = [], []
    for row in rows:
        with np.load(ROOT/"output/keycld_corpus"/row["observations"]) as z:
            images.append(z["x"].astype(np.float32)/127.5-1)
            commands.append(np.r_[z["action"],z["geometry"]].astype(np.float32)/np.asarray([5,12,12,1,1]+[1]*9,np.float32))
    batch = dict(rendering_ts=np.stack(images), tau=np.stack(commands))
    model = DynamicsAutoencoder(autoencoder=VAE(img_shape=(64,64,3),latent_dim=8,norm_layer=nn.LayerNorm),
                               dynamics=ConIaeOde(latent_dim=8,input_dim=14,num_layers=5,hidden_dim=30))
    params = model.init(jax.random.PRNGKey(42),jnp.zeros((1,64,64,3),jnp.float32),method=model.forward_all_layers)["params"]
    cfg = json.loads((ROOT/"experiments/con_corpus/protocol.json").read_text())["training"]
    values = []
    for task_model in (model,FramewiseApply(model,args.frame_chunk)):
        task,_ = task_factory("planar_closed_chain",task_model,jnp.arange(50)/30,.01,
            loss_weights=cfg["loss_weights"],ae_type="beta_vae",solver=Dopri5(),latent_velocity_source="image-space-finite-differences")
        def objective(p):
            loss,pred = task.loss_fn(batch,p,rng=jax.random.PRNGKey(123),training=True)
            return loss,task.compute_metrics_fn(batch,pred)
        fn = jax.jit(jax.value_and_grad(objective,has_aux=True))
        (loss,metrics),grad = fn(params)
        jax.block_until_ready((loss,metrics,grad))
        values.append((float(loss),jax.device_get(metrics),jax.device_get(grad)))
        del fn, grad, loss, metrics
        jax.clear_caches()
        gc.collect()
    native,chunked = values
    relative_loss = abs(native[0]-chunked[0])/max(abs(native[0]),1e-30)
    groups = {key:relative_gradient(native[2][key],chunked[2][key]) for key in native[2]}
    gradient = relative_gradient(native[2],chunked[2])
    passed = relative_loss < 1e-5 and gradient < 1e-3 and all(v < 1e-3 for v in groups.values())
    report = dict(passed=passed,native_loss=native[0],chunked_loss=chunked[0],relative_loss=relative_loss,
        relative_gradient_l2=gradient,gradient_by_parameter_group=groups,frame_chunk=args.frame_chunk,
        matmul_precision=str(jax.config.jax_default_matmul_precision),
        native_metrics={k:float(v) for k,v in native[1].items()},chunked_metrics={k:float(v) for k,v in chunked[1].items()},
        frames_per_sequence=50,batch_size=2,case_ids=[r["case_id"] for r in rows],width=64,
        full_native_loss=True,scientific_training_result=False,
        adapter_sha256=sha(ROOT/"experiments/planar_resolution/con_memory.py"),
        native_task_sha256=sha(ROOT/"official/src/tasks/dynamics_autoencoder.py"))
    write(args.report,report)
    print(json.dumps(report),flush=True)
    if not passed:
        raise AssertionError("Framewise CON objective or gradients differ from native task")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-chunk",type=int,default=16)
    parser.add_argument("--report",type=Path,default=OUT/"validation/con_memory_equivalence.json")
    run(parser.parse_args())
