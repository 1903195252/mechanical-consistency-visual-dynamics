"""Check native high-resolution model sizes without allocating their parameters.

The report is a shape/memory feasibility check, never a training result.
Run using the project's JAX environment.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "output/planar_resolution"


def main():
    sys.path[:0] = [str(ROOT / "output/keycld_corpus/runtime"),
                   str(ROOT / "output/closed_chain_eval/visual_calibration/runtime")]
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from keycld.models import KeyCLD
    from experiments.keycld_corpus.model import ControlledKeyCLD
    from src.models.autoencoders.vae import VAE
    from src.models.neural_odes.con_iae import ConIaeOde

    manifest = json.loads((ROOT / "output/keycld_corpus/dataset_manifest.json").read_text())
    first = next(r for r in manifest["records"] if r["split"] == "train")
    with np.load(ROOT / "output/keycld_corpus" / first["observations"]) as data:
        geometry = jnp.asarray(data["geometry"])
    key = jax.random.PRNGKey(42)
    con = ConIaeOde(latent_dim=8, input_dim=14, num_layers=5, hidden_dim=30)
    dynamic = jax.eval_shape(lambda: con.init(key, jnp.zeros(16), jnp.zeros(14),
                                             method=con.forward_all_layers))
    count = lambda p: sum(a.size for a in jax.tree_util.tree_leaves(p))
    reports = []
    for width in (64, 256, 512):
        model = ControlledKeyCLD(geometry)
        KeyCLD.__init__(model, 3, 3, 32, True, model.constraint_fn, image_size=width)
        params = jax.eval_shape(lambda: model.init(key))
        autoencoder = VAE(img_shape=(width, width, 3), latent_dim=8, norm_layer=nn.LayerNorm)
        con_params = jax.eval_shape(lambda: autoencoder.init(key, jnp.zeros((1, width, width, 3))))
        for name, parameters, batch, epochs, updates in (
                ("KeyCLD", count(params), 1, 40, 32520),
                ("CON", count(con_params) + count(dynamic), 16, 50, 2550)):
            reports.append(dict(model=name, resolution=width, parameters=parameters,
                fp32_parameters_GiB=parameters*4/2**30,
                fp32_params_grad_adam_lower_bound_GiB=parameters*16/2**30,
                input_batch_fp32_GiB=batch*50*width*width*3*4/2**30,
                epochs=epochs, effective_batch=batch, updates_per_seed=updates,
                sequence_exposures_per_seed=epochs*813,
                additional_activation_and_solver_memory="not included in lower bound"))
    result = dict(stage="shape_preflight_complete", scientific_training_result=False,
                  devices=[str(d) for d in jax.devices()], rows=reports,
                  high_resolution_video_runs=12, seeds=[42, 43, 44],
                  state_controls="reuse existing exact-state trajectories; no image-resolution dependence",
                  constraints=[
                      "CON native dense encoder/decoder parameter count grows quadratically with width",
                      "KeyCLD renderer's trainable image seed grows quadratically with width",
                      "Streaming batches, rematerialization, and gradient accumulation preserve full objectives",
                      "KeyCLD pixel-sum reconstruction term needs an explicitly registered resolution scaling",
                      "No wall-clock estimate is available before a full-objective GPU update benchmark"])
    (DIRECTORY / "model_preflight.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
