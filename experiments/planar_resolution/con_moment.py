"""Native GPU AdamW with only the second moment parked on CPU between updates."""
from functools import partial
import jax
import optax


def park_second_moment(state, cpu):
    device_moment = state[0].nu
    host_moment = jax.device_put(device_moment, cpu)
    jax.block_until_ready(host_moment)
    parked = (state[0]._replace(nu=host_moment), *state[1:])
    for value in jax.tree.leaves(device_moment):
        value.delete()
    return parked


def moment_update(tx, gpu, cpu):
    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def update(params, state, gradients):
        changes, state = tx.update(gradients, state, params)
        return optax.apply_updates(params, changes), state

    def step(params, state, gradients):
        device_state = (state[0]._replace(nu=jax.device_put(state[0].nu, gpu)), *state[1:])
        params, device_state = update(params, device_state, gradients)
        jax.block_until_ready((params, device_state))
        for value in jax.tree.leaves(gradients):
            value.delete()
        return params, park_second_moment(device_state, cpu)
    return step
