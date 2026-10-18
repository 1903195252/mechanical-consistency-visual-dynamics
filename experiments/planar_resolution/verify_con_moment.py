"""Verify streamed second moments against the complete native GPU AdamW."""
import json
from functools import partial
import numpy as np
from .common import ROOT, OUT, sha, write
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import optax
from flax import linen as nn
from src.models.autoencoders.vae import VAE
from src.models.neural_odes.con_iae import ConIaeOde
from src.models.dynamics_autoencoder import DynamicsAutoencoder
from src.training.optim import create_learning_rate_fn
from .con_moment import park_second_moment, moment_update


def run():
    gpu,cpu=jax.devices("gpu")[0],jax.devices("cpu")[0]
    model=DynamicsAutoencoder(autoencoder=VAE(img_shape=(64,64,3),latent_dim=8,norm_layer=nn.LayerNorm),
        dynamics=ConIaeOde(latent_dim=8,input_dim=14,num_layers=5,hidden_dim=30))
    p=model.init(jax.random.PRNGKey(42),jnp.zeros((1,64,64,3),jnp.float32),method=model.forward_all_layers)["params"]
    reference=jax.tree.map(lambda x:x.copy(),p)
    cfg=json.loads((ROOT/"experiments/con_corpus/protocol.json").read_text())["training"]
    tx=optax.adamw(create_learning_rate_fn(50,51,cfg["base_lr"],warmup_epochs=5),weight_decay=cfg["weight_decay"])
    state=tx.init(p)
    state=jax.tree.map(lambda x:jnp.asarray(125,x.dtype) if jnp.issubdtype(x.dtype,jnp.integer) else x,state)
    reference_state=jax.tree.map(lambda x:x.copy(),state)
    state=park_second_moment(state,cpu)
    step=moment_update(tx,gpu,cpu)
    @partial(jax.jit,donate_argnums=(0,1,2))
    def native(p,s,g):
        changes,s=tx.update(g,s,p)
        return optax.apply_updates(p,changes),s
    for i in range(3):
        gradients=jax.tree.map(lambda x:jnp.sin(x+jnp.float32(i+1))*jnp.float32(.01),p)
        reference_gradients=jax.tree.map(lambda x:x.copy(),gradients)
        reference,reference_state=native(reference,reference_state,reference_gradients)
        p,state=step(p,state,gradients)
        jax.block_until_ready((reference,reference_state,p,state))
    differences=[]
    for a,b in zip(jax.tree.leaves(jax.device_get((reference,reference_state))),jax.tree.leaves(jax.device_get((p,state)))):
        if np.issubdtype(a.dtype,np.integer):assert np.array_equal(a,b)
        else:differences.append(float(np.max(np.abs(a-b))))
    maximum=max(differences)
    report=dict(passed=maximum<1e-6,maximum_absolute_parameter_or_state_difference=maximum,
        updates=3,initial_schedule_count=125,native_parameter_shapes=True,width=64,
        adapter_sha256=sha(ROOT/"experiments/planar_resolution/con_moment.py"),scientific_training_result=False)
    write(OUT/"validation/con_moment_equivalence.json",report)
    print(json.dumps(report),flush=True)
    if not report["passed"]:raise AssertionError(report)


if __name__=="__main__":run()
