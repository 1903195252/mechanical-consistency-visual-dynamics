"""Check CPU and GPU AdamW on all native 64px CON parameter shapes."""
import json
from functools import partial
import numpy as np
from .common import ROOT, OUT, sha, write
import jax
jax.config.update("jax_enable_x64",True)
jax.config.update("jax_default_matmul_precision","highest")
import jax.numpy as jnp
import optax
from flax import linen as nn
from src.models.autoencoders.vae import VAE
from src.models.neural_odes.con_iae import ConIaeOde
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.training.optim import create_learning_rate_fn
from .con_offload import cpu_update


def run():
    cpu=jax.devices("cpu")[0]
    model=DynamicsAutoencoder(autoencoder=VAE(img_shape=(64,64,3),latent_dim=8,norm_layer=nn.LayerNorm),
        dynamics=ConIaeOde(latent_dim=8,input_dim=14,num_layers=5,hidden_dim=30))
    p=model.init(jax.random.PRNGKey(42),jnp.zeros((1,64,64,3),jnp.float32),method=model.forward_all_layers)["params"]
    cfg=json.loads((ROOT/"experiments/con_corpus/protocol.json").read_text())["training"]
    tx=optax.adamw(create_learning_rate_fn(50,51,cfg["base_lr"],warmup_epochs=5),weight_decay=cfg["weight_decay"])
    state=tx.init(p)
    # Evaluate a nonzero learning rate in the warmup schedule.
    state=jax.tree.map(lambda x:jnp.asarray(125,x.dtype) if jnp.issubdtype(x.dtype,jnp.integer) else x,state)
    host_p=jax.device_put(p,cpu);host_state=jax.device_put(state,cpu)
    cpu_step=cpu_update(tx)
    @partial(jax.jit,donate_argnums=(0,1,2))
    def gpu_step(params,s,g):
        delta,s=tx.update(g,s,params)
        return optax.apply_updates(params,delta),s
    for i in range(3):
        g=jax.tree.map(lambda x:jnp.sin(x+jnp.float32(i+1))*jnp.float32(.01),p)
        host_g=jax.device_put(g,cpu);jax.block_until_ready(host_g)
        p,state=gpu_step(p,state,g)
        host_p,host_state=cpu_step(host_p,host_state,host_g)
        jax.block_until_ready((p,state,host_p,host_state))
    a=jax.device_get((p,state));b=jax.device_get((host_p,host_state))
    differences=[]
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        if np.issubdtype(x.dtype,np.integer):assert np.array_equal(x,y)
        else:differences.append(float(np.max(np.abs(x-y))))
    maximum=max(differences)
    report=dict(passed=maximum<1e-6,maximum_absolute_parameter_or_state_difference=maximum,
        updates=3,initial_schedule_count=125,native_parameter_shapes=True,width=64,
        adapter_sha256=sha(ROOT/"experiments/planar_resolution/con_offload.py"),scientific_training_result=False)
    write(OUT/"validation/con_optimizer_equivalence.json",report);print(json.dumps(report),flush=True)
    if not report["passed"]:raise AssertionError(report)


if __name__=="__main__":run()
