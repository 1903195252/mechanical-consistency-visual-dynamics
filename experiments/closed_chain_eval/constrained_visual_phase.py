"""Select a noisy phase fit while requiring correct exact-state controls.

The unconstrained noise-MSE optimum can erase real acceleration. This second
registered selection adds a hard, development-only numerical control gate.
It never uses model forecasts to select a filter.
"""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from . import optimize_visual_phase as phase
from .optimize_visual_measurement import reference
from .visual_calibration import OUT, sha, write


DIRECTORY = OUT / "optimization/constrained_visual_phase"
PREVIOUS = phase.DIRECTORY


def oracle_case(row):
    from experiments.keycld_corpus.physics_score import measure
    params, x, _, _, _, _, _ = reference(row)
    t = np.arange(90)/30
    limits = json.loads((phase.DATA/"protocol.json").read_text())["evaluation"]["limits_SI"]
    results = {}
    for method in phase.CANDIDATES:
        predicted = phase.project_track(x, t, params["config"], method)
        r = measure(t[phase.INTERIOR], *(v[phase.INTERIOR] for v in predicted), x[phase.INTERIOR],
                    params, params["command_speed_rad_s"], limits)
        results[method] = dict(passed=r["joint_pass"], metrics=r["metrics"])
    return dict(case_id=row["case_id"], candidates=results)


def run():
    from . import verify_visual_phase as sensitivity
    from . import remeasure_visual_phase as remeasure
    rows = [r for r in json.loads((phase.DATA/"dataset_manifest.json").read_text())["records"] if r["split"] == "validation"]
    spec = dict(candidates=phase.CANDIDATES, original_selection_sha256=sha(PREVIOUS/"selection.json"),
                rule="minimum development noisy-kinematics MSE subject to all development exact-state SI controls passing",
                development_ids=[r["case_id"] for r in rows], source_sha256=sha(__file__),
                measured_phase_source_sha256=sha(phase.__file__), models_or_test_references_used=False)
    path = DIRECTORY/"specification.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Constrained phase specification changed")
    write(path, spec)
    with ProcessPoolExecutor(max_workers=6) as pool:
        oracles = list(pool.map(oracle_case, rows))
    passed = {m: sum(r["candidates"][m]["passed"] for r in oracles) for m in phase.CANDIDATES}
    objective = json.loads((PREVIOUS/"selection.json").read_text())["objective"]
    eligible = [m for m in phase.CANDIDATES if passed[m] == len(rows)]
    selected = min(eligible, key=objective.get)
    write(DIRECTORY/"selection.json", dict(selected=selected, objective=objective, eligible=eligible,
        oracle_pass_counts=passed, oracle_rows=oracles, specification_sha256=sha(path)))
    print(json.dumps(dict(selected=selected, eligible=eligible, oracle_pass_counts=passed)), flush=True)
    phase.DIRECTORY = DIRECTORY
    phase.verify()
    sensitivity.DIRECTORY = DIRECTORY
    sensitivity.run()
    remeasure.FIT = DIRECTORY
    remeasure.DIRECTORY = OUT/"paired_constrained_visual_phase"
    remeasure.run()


if __name__ == "__main__":
    run()
