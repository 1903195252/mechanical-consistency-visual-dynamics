"""Exact Optax AdamW updates on CPU, with model differentiation on GPU."""
from functools import partial
import jax
import optax


def cpu_update(tx):
    @partial(jax.jit, backend="cpu", donate_argnums=(0, 1, 2))
    def update(params, state, gradients):
        changes, state = tx.update(gradients, state, params)
        return optax.apply_updates(params, changes), state
    return update
