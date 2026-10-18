"""Native KeyCLD at the actual image width, with framewise rematerialization.

The complete 50-frame objective is evaluated before each optimizer update.
Framewise convolutions preserve GroupNorm's per-image statistics. No training
frames, four-step dynamical windows, BCE term or gradient paths are removed.
"""
from . import common
import jax
import jax.numpy as jnp
import optax
from keycld.losses import loss_fn_step
from keycld.models import KeyCLD
from experiments.keycld_corpus.model import ControlledKeyCLD

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map


class ResolutionKeyCLD(ControlledKeyCLD):
    def __init__(self, geometry, width, stream=True, frame_chunk=1):
        super().__init__(geometry)
        KeyCLD.__init__(self, 3, 3, 32, True, self.constraint_fn, image_size=width)
        self.stream = stream
        self.frame_chunk = frame_chunk

    def encoder(self, params, images):
        if not self.stream:
            return super().encoder(params, images)
        def frame(image):
            return self._encoder.apply(params["encoder"], image[None])[0]
        return jax.lax.map(jax.checkpoint(frame), images, batch_size=self.frame_chunk)

    def renderer(self, params, points):
        if not self.stream:
            return super().renderer(params, points)
        def frame(point):
            rgb, maps = self._renderer.apply(params["renderer"], point[None])
            return rgb[0], maps[0]
        return jax.lax.map(jax.checkpoint(frame), points, batch_size=self.frame_chunk)


def objective(params, item, augmentation, width, stream=True, frame_chunk=1):
    model = ResolutionKeyCLD(item["geometry"], width, stream, frame_chunk)
    _, aux = loss_fn_step(1., 1., 4, "euler", model, params, item, augmentation)
    # The original reconstruction term sums pixels; preserve its 64px scale.
    reconstruction = aux["reconstruction"] * (64 / width) ** 2
    loss = reconstruction + aux["dynamics"] + aux["bce_loss"]
    return loss, {**aux, "reconstruction_64px_equivalent": reconstruction}


def optimizer():
    labels = {"encoder": "v", "renderer": "v", "mass_matrix": "d",
              "potential_energy": "d", "input_matrix": "d"}
    return optax.chain(optax.clip(5.), optax.multi_transform(
        {"v": optax.adam(.0003), "d": optax.adam(.00006)}, labels))


def make_step(width, frame_chunk=1):
    tx = optimizer()
    @jax.jit
    def step(params, state, item, augmentation):
        (loss, aux), gradients = jax.value_and_grad(objective, has_aux=True)(
            params, item, augmentation, width, True, frame_chunk)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(gradients)]))
        changes, state = tx.update(gradients, state, params)
        return optax.apply_updates(params, changes), state, loss, aux, finite
    return tx, step
