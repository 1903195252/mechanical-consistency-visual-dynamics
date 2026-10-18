"""Framewise rematerialization around the unchanged native CON task.

The native beta-VAE samples its entire latent tensor with the original RNG;
the ODE, objectives, loss reductions and batch size are untouched. Only
independent image encoder/decoder calls are partitioned to bound activations.
"""
import jax


class FramewiseApply:
    def __init__(self, model, frame_chunk):
        self.model = model
        self.frame_chunk = frame_chunk

    def __getattr__(self, name):
        return getattr(self.model, name)

    def apply(self, variables, *inputs, method=None, **kwargs):
        if getattr(method, "__name__", "") not in ("encode", "encode_vae", "decode"):
            return self.model.apply(variables, *inputs, method=method, **kwargs)
        if len(inputs) != 1:
            raise ValueError("Image encoder/decoder must receive one native input tensor")
        def frame(value):
            result = self.model.apply(variables, value[None], method=method, **kwargs)
            return jax.tree.map(lambda x: x[0], result)
        return jax.lax.map(jax.checkpoint(frame), inputs[0], batch_size=self.frame_chunk)
