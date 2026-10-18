"""Select a curvature-aware numerical measurement on development references."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from .measurement_derivatives import planar_derivatives
from .optimize_visual_measurement import reference
from .score_visual_calibration import metric_map
from .visual_calibration import ROOT, OUT, load_protocol, sha, write


DIRECTORY = OUT / "optimization/coordinate_derivatives"
CANDIDATES = ("cartesian_sg7d3", "cartesian_sg11d9", "cartesian_spline5", "cartesian_spline7",
              "polar_sg7d5", "polar_sg11d9", "polar_spline5", "polar_spline7")


def evaluate(task):
    from experiments.keycld_corpus.physics_score import measure
    row, candidates = task
    params, x, v, a, _, _, _ = reference(row)
    limits = json.loads((ROOT / "output/keycld_corpus/protocol.json").read_text())["evaluation"]["limits_SI"]
    s, t = slice(5, -5), np.arange(90) / 30
    results = {}
    for method in candidates:
        dv, da = planar_derivatives(t, x, params["config"], method)
        measured = measure(t[s], x[s], dv[s], da[s], x[s], params, params["command_speed_rad_s"], limits)
        ratios = {k: None if value is None else value / limits[k] for k, value in measured["metrics"].items()}
        results[method] = dict(metrics=metric_map(measured["metrics"]), ratios=metric_map(ratios),
            passed=measured["joint_pass"], velocity_error_rms=float(np.sqrt(np.mean((dv[s] - v[s]) ** 2))),
            acceleration_error_rms=float(np.sqrt(np.mean((da[s] - a[s]) ** 2))))
    return dict(case_id=row["case_id"], assembly_id=row["assembly_id"], category=row["category"], candidates=results)


def run():
    protocol = load_protocol()
    manifest = json.loads((ROOT / "output/keycld_corpus/dataset_manifest.json").read_text())
    development = [r for r in manifest["records"] if r["split"] == "validation"]
    spec = dict(candidates=CANDIDATES, development_ids=[r["case_id"] for r in development],
        selection="development maximum engineering-normalized reference error", common_frames=list(range(5, 85)),
        preserves_measured_positions=True, fixes_radii_to_known_lengths=False,
        source_sha256=sha(__file__), derivative_source_sha256=sha(ROOT / "experiments/closed_chain_eval/measurement_derivatives.py"))
    write(DIRECTORY / "specification.json", spec)
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(evaluate, [(r, CANDIDATES) for r in development]))
    objective = {method: max(value for row in rows for value in row["candidates"][method]["ratios"].values()
                             if value is not None) for method in CANDIDATES}
    selected = min(objective, key=objective.get)
    write(DIRECTORY / "selection.json", dict(selected=selected, objective=objective, rows=rows))
    with ProcessPoolExecutor(max_workers=6) as pool:
        verification = list(pool.map(evaluate, [(r, ["cartesian_sg7d3", selected]) for r in protocol["planar_records"]]))
    summary = {method: dict(joint_pass=sum(r["candidates"][method]["passed"] for r in verification),
                   total=len(verification),
                   maxima={k: max(r["candidates"][method]["metrics"][k] for r in verification) for k in metric_map({})})
               for method in ["cartesian_sg7d3", selected]}
    write(DIRECTORY / "verification.json", dict(selected=selected, summary=summary, rows=verification))
    print(json.dumps(dict(selected=selected, objective=objective, verification=summary)), flush=True)


if __name__ == "__main__":
    run()
