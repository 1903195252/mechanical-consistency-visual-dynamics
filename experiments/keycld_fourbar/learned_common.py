"""Mechanism adaptation only; network modules remain the official KeyCLD."""
import json
from pathlib import Path


def load_model(root):
    import jax.numpy as jnp
    from keycld.models import KeyCLD
    geometry=json.loads((Path(root)/'geometry.json').read_text())
    A=jnp.asarray(geometry['A_normalized']);D=jnp.asarray(geometry['D_normalized'])
    lengths=jnp.asarray(geometry['link_lengths_normalized'])
    def constraints(state):
        B,C,P=state.reshape(3,2)
        distance=jnp.array([jnp.linalg.norm(B-A)-lengths[0],
                            jnp.linalg.norm(C-B)-lengths[1],
                            jnp.linalg.norm(C-D)-lengths[2]])
        return jnp.concatenate([distance,P-(B+C)/2])
    return KeyCLD(3,3,32,True,constraints,image_size=64)
