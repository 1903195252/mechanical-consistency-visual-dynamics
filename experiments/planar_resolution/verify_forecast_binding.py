"""Compare captured and explicit native CON inference parameters at 64px.

This execution check uses an initialized model and synthetic images, not a
scientific forecast or a selected training checkpoint.
"""
import json
import numpy as np
from .common import ROOT, OUT, sha, write


def run():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from flax import linen as nn
    from src.models.autoencoders.vae import VAE
    from src.models.neural_odes.con_iae import ConIaeOde
    from src.models.dynamics_autoencoder import DynamicsAutoencoder
    network = DynamicsAutoencoder(
        autoencoder=VAE(img_shape=(64,64,3), latent_dim=8, norm_layer=nn.LayerNorm),
        dynamics=ConIaeOde(latent_dim=8,input_dim=14,num_layers=5,hidden_dim=30))
    initialized = network.init(jax.random.PRNGKey(42), jnp.zeros((1,64,64,3),jnp.float32),
                               method=network.forward_all_layers)["params"]
    host = jax.device_get(initialized)
    device = jax.device_put(host)
    images = jnp.asarray(np.random.RandomState(7).uniform(-1,1,(3,64,64,3)).astype(np.float32))
    encoder_old = lambda x: network.apply({"params":host}, x, method=network.encode)
    encoder_new = lambda p,x: network.apply({"params":p}, x, method=network.encode)
    old_graph = jax.make_jaxpr(encoder_old)(images)
    new_graph = jax.make_jaxpr(encoder_new)(device,images)
    old_bytes = sum(getattr(x,"nbytes",0) for x in old_graph.consts)
    new_bytes = sum(getattr(x,"nbytes",0) for x in new_graph.consts)
    old_jit, new_jit = jax.jit(encoder_old), jax.jit(encoder_new)
    results = {}

    def compare(name, a, b):
        jax.block_until_ready((a,b))
        differences = [float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
                       for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b))]
        assert all(np.isfinite(differences))
        results[name] = max(differences)

    compare("encoder", old_jit(images), new_jit(device,images))
    tangent=(images[2:3]-images[:1])/(2/30)
    compare("image_space_jvp",jax.jvp(old_jit,(images[1:2],),(tangent,)),
            jax.jvp(lambda x:new_jit(device,x),(images[1:2],),(tangent,)))
    z=new_jit(device,images)
    decode_old=jax.jit(lambda x:network.apply({"params":host},x,method=network.decode))
    decode_new=jax.jit(lambda p,x:network.apply({"params":p},x,method=network.decode))
    compare("decoder",decode_old(z),decode_new(device,z))
    y=jnp.r_[z[0],jnp.ones(8,dtype=jnp.float32)*.1]
    u=jnp.arange(14,dtype=jnp.float32)*.02
    dynamics_old=jax.jit(lambda y,u:network.apply({"params":host},y,u,method=network.forward_dynamics))
    dynamics_new=jax.jit(lambda p,y,u:network.apply({"params":p},y,u,method=network.forward_dynamics))
    compare("dynamics",dynamics_old(y,u),dynamics_new(device,y,u))
    report=dict(passed=max(results.values())<1e-5 and new_bytes<old_bytes/100,
        maximum_absolute_differences=results,captured_encoder_constant_bytes=old_bytes,
        explicit_encoder_constant_bytes=new_bytes,width=64,native_architecture=True,
        synthetic_inputs=True,scientific_training_result=False,
        source_sha256=sha(__file__),forecast_source_sha256=sha(ROOT/"experiments/planar_resolution/forecast.py"))
    write(OUT/"validation/forecast_parameter_binding.json",report)
    print(json.dumps(report),flush=True)
    assert report["passed"]


if __name__=="__main__":
    run()
