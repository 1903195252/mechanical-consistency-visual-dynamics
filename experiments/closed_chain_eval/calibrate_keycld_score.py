"""Verify preregistered engineering thresholds with independent physical controls."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .core import CORE_METRICS
from .evaluate import write_json
from .import_keycld import ROOT
from .planar import physics
from .scored_metrics import measure, points_from_truth


def check_case(task):
    name, seed, root, protocol = task
    root = Path(root)
    file = root / "truth" / name / ("run_%04d.npz" % seed)
    with np.load(file, allow_pickle=False) as data:
        truth = {key: data[key] for key in data.files}
    c = json.loads(file.with_suffix(".json").read_text())
    t, torque = truth["time"], float(truth["exact_action"][0])
    x, v, a = points_from_truth(truth, c)
    oracle = measure(t, x, v, a, x, torque, c, protocol, derivative_audit=True)
    t0 = .03
    initial_index = int(round(t0 * 1600)) - 1
    static = np.repeat(x[initial_index][None], len(t), axis=0)
    static_score = measure(t, static, np.zeros_like(v), np.zeros_like(a), x, torque, c, protocol)
    warped_time = np.maximum(t0, t0 + .8 * (t - t0))
    theta = np.interp(warped_time, t, truth["q"][:, 0])
    omega = .8 * np.interp(warped_time, t, truth["qdot"][:, 0])
    alpha = .64 * np.interp(warped_time, t, truth["qddot"][:, 0])
    points = []
    for angle, speed, acceleration in zip(theta, omega, alpha):
        q = physics.inverse(angle, c)
        gamma, second = physics.derivatives(q, c)
        points.append(physics.point_states(q, gamma * speed, gamma * acceleration + second * speed**2, c))
    wx, wv, wa = [np.asarray(values) for values in zip(*points)]
    warped = measure(t, wx, wv, wa, x, torque, c, protocol)
    family = "3R1P" if name == "crank_slider" else "4R"
    for row in (oracle, static_score, warped):
        row["case_key"] = [family, name, seed]
    return {"oracle": oracle, "static": static_score, "time_scaled_0.8": warped}


def main():
    root = ROOT / "output/keycld_fourbar/scored_evaluation"
    if (root / "calibration.json").exists():
        raise RuntimeError("Calibration already frozen")
    protocol = json.loads((root / "protocol.json").read_text())
    low, high = protocol["calibration_seeds"]
    tasks = [(name, seed, str(root), protocol) for name in protocol["conditions"] for seed in range(low, high + 1)]
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(check_case, task) for task in tasks]):
            results.append(future.result())
            if len(results) % 8 == 0:
                print(json.dumps({"stage": "physical_calibration", "completed": len(results), "total": len(tasks)}), flush=True)
    results.sort(key=lambda item: item["oracle"]["case_key"])
    summaries = {}
    for label in ("oracle", "static", "time_scaled_0.8"):
        rows = [item[label] for item in results]
        summaries[label] = {
            "passed": sum(row["joint_pass"] for row in rows), "total": len(rows),
            "metric_maxima_SI": {name: max(row["metrics"][name]["value"] for row in rows
                                             if row["metrics"][name]["value"] is not None) for name in CORE_METRICS},
        }
    passed = summaries["oracle"]["passed"] == len(results)
    calibration = {"profile_id": protocol["profile_id"], "measurement_floor_acceptable": passed,
                   "uses_model_predictions": False,
                   "protocol_sha256": hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest(),
                   "dataset_manifest_sha256": hashlib.sha256((root / "dataset_manifest.json").read_bytes()).hexdigest(),
                   "elapsed_seconds": time.monotonic() - start, "summaries": summaries, "rows": results}
    write_json(root / "calibration.json", calibration)
    print(json.dumps({"measurement_floor_acceptable": passed, "summaries": summaries}), flush=True)
    if not passed:
        raise RuntimeError("Correct physical controls failed; inspect measurement before scoring models")


if __name__ == "__main__":
    main()
