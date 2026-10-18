"""Improve the measurement chain using development references, not forecasts.

The numerical stage chooses a derivative stencil on original validation
identities and verifies it on the untouched calibration identities. Image
readout optimization is kept separate: a good exact-state derivative is not a
guarantee that the same stencil tolerates noisy coordinates.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from scipy.signal import savgol_filter

from .visual_calibration import ROOT, OUT, load_protocol, sha, write
from .score_visual_calibration import metric_map


DIRECTORY = OUT / "optimization"
CANDIDATES = [(7, 3), (7, 5), (9, 7), (11, 9)]


def reference(row):
    from experiments.keycld_corpus.prepare import point_states
    params = json.loads((ROOT / row["physical_parameters"]).read_text())
    if sha(ROOT / row["physics_reference"]) != row["physics_reference_sha256"]:
        raise ValueError("Reference changed")
    with np.load(ROOT / row["physics_reference"]) as data:
        index = np.arange(240, 960, 8)
        x, v, a = point_states(data["q"][index], data["qdot"][index], data["qddot"][index], params["config"])
        q, qv, qa = data["q"][index], data["qdot"][index], data["qddot"][index]
    return params, x, v, a, q, qv, qa


def numerical_case(task):
    from experiments.keycld_corpus.physics_score import measure
    row, candidates = task
    params, x, v, a, _, _, _ = reference(row)
    limits = json.loads((ROOT / "output/keycld_corpus/protocol.json").read_text())["evaluation"]["limits_SI"]
    # Every candidate uses the same evaluation samples; never win by shortening
    # the hard part of a trajectory or changing geometry/prediction observations.
    s = slice(5, -5)
    t = np.arange(90) / 30
    result = dict(case_id=row["case_id"], assembly_id=row["assembly_id"], category=row["category"], candidates={})
    for window, degree in candidates:
        dv = savgol_filter(x, window, degree, deriv=1, delta=1/30, axis=0)
        da = savgol_filter(x, window, degree, deriv=2, delta=1/30, axis=0)
        measured = measure(t[s], x[s], dv[s], da[s], x[s], params, params["command_speed_rad_s"], limits)
        ratios = {k: None if value is None else value / limits[k] for k, value in measured["metrics"].items()}
        result["candidates"]["%d-%d" % (window, degree)] = dict(metrics=metric_map(measured["metrics"]),
            ratios=metric_map(ratios), passed=measured["joint_pass"],
            velocity_error_rms=float(np.sqrt(np.mean((dv[s] - v[s]) ** 2))),
            acceleration_error_rms=float(np.sqrt(np.mean((da[s] - a[s]) ** 2))))
    return result


def numerical():
    protocol = load_protocol()
    manifest = json.loads((ROOT / "output/keycld_corpus/dataset_manifest.json").read_text())
    development = [r for r in manifest["records"] if r["split"] == "validation"]
    if {r["assembly_id"] for r in development} & {r["assembly_id"] for r in protocol["planar_records"]}:
        raise ValueError("Development/calibration overlap")
    spec = dict(source_sha256=sha(__file__), candidates=CANDIDATES, common_derivative_frames=list(range(5, 85)),
        selection="minimize development worst-case engineering-normalized reference error",
        model_forecasts_used=False, numerical_only=True,
        development_cases=[r["case_id"] for r in development])
    write(DIRECTORY / "numerical_specification.json", spec)
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(numerical_case, [(r, CANDIDATES) for r in development]))
    objective = {}
    for key in rows[0]["candidates"]:
        values = [v for row in rows for v in row["candidates"][key]["ratios"].values() if v is not None]
        objective[key] = max(values)
    selected = min(objective, key=objective.get)
    write(DIRECTORY / "numerical_selection.json", dict(selected=selected, objective=objective,
          specification_sha256=sha(DIRECTORY / "numerical_specification.json"), development=rows))
    window, degree = map(int, selected.split("-"))
    with ProcessPoolExecutor(max_workers=6) as pool:
        verification = list(pool.map(numerical_case, [(r, [(7, 3), (window, degree)]) for r in protocol["planar_records"]]))
    summary = {}
    for key in ("7-3", selected):
        summary[key] = dict(joint_pass=sum(r["candidates"][key]["passed"] for r in verification),
            total=len(verification),
            maxima={k: max(r["candidates"][key]["metrics"][k] for r in verification) for k in metric_map({})})
    write(DIRECTORY / "numerical_verification.json", dict(selected=selected, summary=summary, rows=verification,
          engineering_thresholds_changed=False, no_visual_noise_claim=True))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("numerical",), default="numerical")
    args = parser.parse_args()
    numerical()
