"""Execute frozen KeyCLD predictions and produce calibrated diagnostic scores."""
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

from .core import CORE_METRICS, aggregate
from .continuous import continuous_scorecard, aggregate_continuous
from .evaluate import write_json
from .import_keycld import ROOT
from .scored_metrics import macro_component_scores, measure, points_from_truth

SCALE = .43 * 63 / 64
CENTER = np.array([.2, .37])


def summarize(rows, expected, protocol):
    rows = [continuous_scorecard(row) for row in rows]
    result = aggregate_continuous(rows, expected, calibration_id=protocol["profile_id"])
    result["component_scores"] = macro_component_scores(rows)
    result["joint_passed_trajectories"] = sum(row["joint_pass"] for row in rows)
    result["sensitivity_scores_100"] = {}
    for multiplier in protocol["sensitivity_threshold_multipliers"]:
        alternate = [dict(row, joint_pass=row["sensitivity_joint_pass"][str(multiplier)]) for row in rows]
        result["sensitivity_scores_100"][str(multiplier)] = aggregate(alternate, expected, protocol["profile_id"])["verified_score_100"]
    result["conditions"] = {}
    for condition in protocol["conditions"]:
        subset = [row for row in rows if row["case_key"][1] == condition]
        condition_result = {"passed": sum(row["joint_pass"] for row in subset), "total": len(subset),
                            "complete_measured": sum(row["all_metrics_available"] for row in subset), "metrics": {}}
        for name in CORE_METRICS:
            values = [row["metrics"][name]["value"] for row in subset if row["metrics"][name]["value"] is not None]
            condition_result["metrics"][name] = {
                "n": len(values), "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
                "passed": sum(row["metrics"][name]["passed"] is True for row in subset),
            }
        result["conditions"][condition] = condition_result
    result["rows"] = rows
    return result


def main():
    root = ROOT / "output/keycld_fourbar/scored_evaluation"
    if (root / "scores.json").exists():
        raise RuntimeError("Completed model scores already exist")
    protocol = json.loads((root / "protocol.json").read_text())
    calibration = json.loads((root / "calibration.json").read_text())
    protocol_sha = hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
    if not calibration["measurement_floor_acceptable"] or calibration["protocol_sha256"] != protocol_sha:
        raise RuntimeError("Calibration is not valid for this frozen profile")
    low, high = protocol["test_seeds"]
    expected = [("3R1P" if name == "crank_slider" else "4R", name, seed)
                for name in protocol["conditions"] for seed in range(low, high + 1)]
    source = ROOT / "output/keycld_fourbar/extended_dynamics/local_source"
    sys.path.insert(0, str(source))
    import jax
    import jax.numpy as jnp
    from jax.experimental.ode import odeint
    from keycld.models import KeyCLD
    from keycld import util
    import evaluate_extended
    evaluate_extended.jnp = jnp
    evaluate_extended.KeyCLD = KeyCLD
    files = {"video": ROOT / "output/keycld_fourbar/trained/trained_params.p",
             "state_RK4": ROOT / "output/keycld_fourbar/diagnosis/state_oracle/rk4_4/params.p"}
    parameters = {}
    for name, path in files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol["checkpoint_sha256"][name]:
            raise RuntimeError("Checkpoint hash mismatch: " + name)
        with path.open("rb") as handle:
            parameters[name] = pickle.load(handle)
    start = time.monotonic()
    results = {name: [] for name in files}
    t = np.arange(32, 801, dtype=float) / 1600
    prediction_hashes = {}
    for condition in protocol["conditions"]:
        c = json.loads((root / "truth" / condition / ("run_%04d.json" % low)).read_text())
        model = evaluate_extended.model_for(c)

        @jax.jit
        def integrate(params, position, velocity, action):
            velocity = util.project_velocity(model.constraint_fn, position, velocity)
            initial = jnp.r_[position, velocity]
            history = odeint(partial(model.ode, params), initial, jnp.asarray(t, dtype=jnp.float32), action)
            acceleration = jax.vmap(lambda state: model.euler_lagrange(params, state[:6], state[6:], action))(history)
            return history[:, :6].reshape(-1, 3, 2), history[:, 6:].reshape(-1, 3, 2), acceleration.reshape(-1, 3, 2)

        @jax.jit
        def video_initial(params, images):
            keypoints, _ = util.map_to_keypoints(model.encoder(params, images))
            position, velocity = util.finite_difference(keypoints.reshape(3, 6), jnp.float32(.01))
            return position[0], velocity[0]

        for seed in range(low, high + 1):
            name = "run_%04d" % seed
            with np.load(root / "observations" / condition / (name + ".npz"), allow_pickle=False) as data:
                images = data["x"][:3].astype(np.float32) / 255
                action = data["action"]
            truth_path = root / "truth" / condition / (name + ".npz")
            with np.load(truth_path, allow_pickle=False) as data:
                truth = {key: data[key] for key in data.files}
            c = json.loads(truth_path.with_suffix(".json").read_text())
            truth_x, truth_v, truth_a = points_from_truth(truth, c)
            reference = truth_x[31:]
            for label in ("video", "state_RK4"):
                params = parameters[label]
                if label == "video":
                    initial_x, initial_v = video_initial(params, images)
                else:
                    initial_x = ((truth_x[31] - CENTER) / SCALE).ravel().astype(np.float32)
                    initial_v = (truth_v[31] / SCALE).ravel().astype(np.float32)
                x, v, a = jax.device_get(integrate(params, initial_x, initial_v, action))
                x, v, a = x.astype(float) * SCALE + CENTER, v.astype(float) * SCALE, a.astype(float) * SCALE
                directory = root / "predictions" / label / condition
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / (name + ".npz")
                np.savez_compressed(path, time=t, points=x, velocity=v, acceleration=a)
                prediction_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
                row = measure(t, x, v, a, reference, float(truth["exact_action"][0]), c, protocol, derivative_audit=True)
                row["case_key"] = ["3R1P" if condition == "crank_slider" else "4R", condition, seed]
                results[label].append(row)
            if (seed - low + 1) % 5 == 0:
                status = {"stage": "scoring", "condition": condition, "completed": sum(len(rows) for rows in results.values()),
                          "total": 2 * len(expected), "seconds": time.monotonic() - start}
                write_json(root / "status.json", status)
                print(json.dumps(status), flush=True)
    scorecards = {}
    for label, rows in results.items():
        scorecards[label] = summarize(rows, expected, protocol)
        scorecards[label].update(input_track="video_conditioned_declared_physical_keypoints" if label == "video" else "state_supervised_diagnostic",
                                derivative_track="native_ODE_audited_dense_output",
                                reader_track="declared_B_C_midpoint_with_fixed_camera_calibration")
    report = {"profile_id": protocol["profile_id"], "scope": protocol["score_scope"],
              "new_test_seed_range": protocol["test_seeds"], "thresholds_SI": protocol["limits_SI"],
              "score_definition": protocol["score"], "calibration_controls": calibration["summaries"],
              "scorecards": scorecards, "no_training": True,
              "checkpoint_sha256": protocol["checkpoint_sha256"],
              "protocol_sha256": protocol_sha,
              "calibration_sha256": hashlib.sha256((root / "calibration.json").read_bytes()).hexdigest(),
              "elapsed_seconds": time.monotonic() - start}
    write_json(root / "scores.json", report)
    code_directory = Path(__file__).resolve().parent
    dependencies = [ROOT / "experiments/fourbar/mechanics.py", ROOT / "experiments/keycld_fourbar/extended_mechanics.py",
                    source / "keycld/models.py", source / "keycld/util.py"]
    write_json(root / "execution_manifest.json", {
        "protocol_sha256": protocol_sha, "checkpoint_sha256": protocol["checkpoint_sha256"],
        "calibration_sha256": report["calibration_sha256"],
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(code_directory.glob("*.py")) + dependencies},
        "prediction_sha256": prediction_hashes,
        "packages": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "numpy", "scipy")},
        "devices": [str(device) for device in jax.devices()], "no_training": True,
        "completed_predictions": sum(len(rows) for rows in results.values()), "elapsed_seconds": report["elapsed_seconds"],
    })
    write_json(root / "status.json", {"stage": "complete", "predictions": 2 * len(expected),
                                      "scores": {label: card["verified_score_100"] for label, card in scorecards.items()}})
    print(json.dumps({label: {"score": card["verified_score_100"], "components": card["component_scores"]}
                      for label, card in scorecards.items()}), flush=True)


if __name__ == "__main__":
    main()
