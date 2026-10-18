"""Post-hoc audit of saved state rollouts; never changes the registered scores."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.keycld_corpus.physics_score import MODULES, decode, residuals
from experiments.mechanism_corpus.catalog import ROOT, write_json

OUT = ROOT / "output/con_corpus"
KEYCLD = ROOT / "output/keycld_corpus"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def force_at_state(x, v, a, params):
    """The registered virtual-work torque equation at one unprojected state."""
    config = params["config"]
    q, qd, qdd = decode(x, v, a, config)
    module = MODULES[config["category"]]
    jacobian = module.closure_jacobian(q, config)
    assert np.linalg.cond(jacobian[:, 1:]) <= 1e10
    tangent = np.r_[1.0, np.linalg.solve(jacobian[:, 1:], -jacobian[:, 0])]
    mass, bias, gravity, _, _ = module.mass_bias_energy(q, qd, config, params["bodies"])
    friction = module.friction(qd, config)
    external = module.external(q, config)
    controller = config["actuator"]
    torque = np.clip(
        controller["speed_gain_Nm_s_rad"] * (params["command_speed_rad_s"] - qd[0]),
        -controller["torque_limit_Nm"], controller["torque_limit_Nm"],
    )
    return float(tangent @ (mass @ qdd + bias + gravity - friction - external) - torque)


def macro(rows, value):
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"], row["condition"]].append(value(row))
    return float(np.mean([np.mean(values) for values in groups.values()]))


def summarize(rows):
    crossing = np.array([r["first_geometry_violation_ms"] for r in rows])
    return {
        "test_cases": len(rows),
        "initial_geometry_passes": sum(r["initial_geometry_max_m"] <= 0.001 for r in rows),
        "max_initial_geometry_m": max(r["initial_geometry_max_m"] for r in rows),
        "max_initial_position_conversion_error_m": max(r["initial_position_conversion_error_m"] for r in rows),
        "max_initial_velocity_conversion_error_m_s": max(r["initial_velocity_conversion_error_m_s"] for r in rows),
        "first_geometry_violation_ms_min_median_max": np.percentile(crossing, [0, 50, 100]).tolist(),
        "initial_force_residual_rms_Nm": float(np.sqrt(macro(rows, lambda r: r["initial_force_residual_Nm"] ** 2))),
        "reference_initial_force_residual_max_abs_Nm": max(abs(r["reference_initial_force_residual_Nm"]) for r in rows),
        "initial_acceleration_constraint_max_channel_rms_m_s2": max(
            float(np.sqrt(macro(rows, lambda r: r["initial_acceleration_constraint_m_s2"][i] ** 2)))
            for i in range(5)
        ),
    }


def main():
    manifest = json.loads((OUT / "dataset_manifest.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["evaluation"]["limits_SI"]["geometry_m"] == 0.001
    references = json.loads((OUT / "evaluator/manifest.json").read_text())
    lookup = {r["case_id"]: r for r in references["references"]}
    records = [r for r in manifest["records"] if r["split"] == "test"]
    assert len(records) == 108
    sources = {"experiments/keycld_corpus/physics_score.py": sha(ROOT / "experiments/keycld_corpus/physics_score.py")}
    rows = []
    cache = {}
    for record in records:
        identity = record["case_id"]
        reference = lookup[identity]
        parameters = ROOT / reference["parameters_json"]
        truth = ROOT / reference["reference_npz"]
        labels = KEYCLD / record["state_labels"]
        for path, expected in ((parameters, reference["parameters_sha256"]),
                               (truth, reference["reference_sha256"]),
                               (labels, record["state_labels_sha256"])):
            assert sha(path) == expected, path
        params = json.loads(parameters.read_text())
        with np.load(truth) as z:
            ref_x, ref_v, ref_a = (z[k][0] for k in ("points", "velocity", "acceleration"))
        with np.load(labels) as z:
            expected_x = z["x"][1].astype(float) * 0.65 + [0.2, 0.35]
            expected_v = z["v"][1].astype(float) * 0.65
        oracle_force = force_at_state(ref_x, ref_v, ref_a, params)
        cache[identity] = params, expected_x, expected_v, oracle_force
    for seed in (42, 43, 44):
        folder = OUT / "runs" / f"state_con_iae_seed{seed}"
        trained = json.loads((folder / "training_report.json").read_text())
        sources[str((folder / "training_report.json").relative_to(ROOT))] = sha(folder / "training_report.json")
        for record in records:
            identity = record["case_id"]
            path = folder / "predictions/test" / (identity + ".npz")
            metadata = json.loads(path.with_suffix(".json").read_text())
            assert sha(path) == metadata["prediction_sha256"]
            assert metadata["checkpoint_sha256"] == trained["checkpoint_sha256"]
            params, expected_x, expected_v, oracle_force = cache[identity]
            with np.load(path) as z:
                t, x, v, a = (z[k] for k in ("time", "points", "velocity", "acceleration"))
                assert len(t) == 705 and abs(t[0] - 1 / 30) < 1e-12
                rq, _, ra = residuals(x, v, a, params["config"])
                crossing = np.flatnonzero(np.max(np.abs(rq), axis=1) > 0.001)
                assert len(crossing) and crossing[0] > 0
                force = force_at_state(x[0], v[0], a[0], params)
                rows.append({
                    "seed": seed, **{k: record[k] for k in ("case_id", "assembly_id", "category", "condition")},
                    "prediction_sha256": metadata["prediction_sha256"],
                    "checkpoint_sha256": trained["checkpoint_sha256"],
                    "initial_position_conversion_error_m": float(np.max(np.abs(x[0] - expected_x))),
                    "initial_velocity_conversion_error_m_s": float(np.max(np.abs(v[0] - expected_v))),
                    "initial_geometry_max_m": float(np.max(np.abs(rq[0]))),
                    "first_geometry_violation_ms": float(1000 * (t[crossing[0]] - t[0])),
                    "initial_force_residual_Nm": force,
                    "reference_initial_force_residual_Nm": oracle_force,
                    "initial_acceleration_constraint_m_s2": ra[0].tolist(),
                })
    report = {
        "track": "Post-hoc initial-state and rollout-validity diagnostic; not the registered future-trajectory score",
        "arm": "state_con_iae", "seeds": [42, 43, 44],
        "initialization_time_s": 1 + 1 / 30, "geometry_tolerance_m": 0.001,
        "time_resolution_s": 1 / 240, "geometry_projection": False,
        "registered_scores_changed": False,
        "instantaneous_force_definition": "RMS across initial states of the original crank-normalized virtual-work torque residual; native predicted acceleration at true initial position and velocity",
        "acceleration_scope": "Acceleration constraint residual is reported alongside torque; torque alone does not establish an admissible acceleration",
        "energy_scope": "No accumulated work-energy score is inferred from a single initial state or from invalid future geometry",
        "summary": summarize(rows),
        "per_seed": {str(seed): summarize([r for r in rows if r["seed"] == seed]) for seed in (42, 43, 44)},
        "sources": sources,
        "dataset_manifest_sha256": sha(OUT / "dataset_manifest.json"),
        "evaluator_manifest_sha256": sha(OUT / "evaluator/manifest.json"),
        "audit_source_sha256": sha(Path(__file__)), "rows": rows,
    }
    write_json(OUT / "geometry_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "sources")}, indent=2))


if __name__ == "__main__":
    main()
