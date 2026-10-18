"""Apply the frozen phase measurement to every saved common RGB readout."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from .optimize_visual_phase import DIRECTORY as FIT, DATA, INTERIOR, project_track
from .optimize_visual_measurement import reference
from .visual_calibration import OUT, write, sha


DIRECTORY = OUT / "paired_visual_phase"
SOURCE = OUT / "paired_robust_planar_rgb"


def case(task):
    from experiments.keycld_corpus.physics_score import measure
    row, name, method = task
    prior = json.loads((SOURCE/"measurements"/name/(row["case_id"]+".json")).read_text())
    result = dict(case_id=row["case_id"], assembly_id=row["assembly_id"], category=row["category"],
        model=name, finite=prior["finite"], method=method,
        raw_point_euclidean_rmse_m=prior.get("point_euclidean_rmse_m"),
        raw_geometry_max_m=prior.get("native_metrics", {}).get("geometry"))
    if prior["finite"]:
        source = SOURCE/"readouts"/name/(row["case_id"]+".npz")
        with np.load(source) as data:
            points, t = data["points"], data["time"]
        params, truth, _, _, _, _, _ = reference(row)
        limits = json.loads((DATA/"protocol.json").read_text())["evaluation"]["limits_SI"]
        try:
            reconstructed = project_track(points, t, params["config"], method)
            measured = measure(t[INTERIOR], *(p[INTERIOR] for p in reconstructed), truth[INTERIOR],
                params, params["command_speed_rad_s"], limits)
            result.update(qualified=measured["coverage"] == 1, measurement=measured,
                correction_point_rms_m=float(np.sqrt(np.mean(np.sum((reconstructed[0][INTERIOR]-points[INTERIOR])**2, axis=-1)))),
                readout_sha256=sha(source))
        except (ValueError, np.linalg.LinAlgError) as error:
            result.update(qualified=False, reason=str(error))
    else:
        result.update(qualified=False, reason=prior["reason"])
    write(DIRECTORY/"measurements"/name/(row["case_id"]+".json"), result)
    return result


def run():
    rows = [r for r in json.loads((DATA/"dataset_manifest.json").read_text())["records"] if r["split"] == "test"]
    method = json.loads((FIT/"selection.json").read_text())["selected"]
    names = ["reference"]+[m+"_seed"+str(seed) for m in ("keycld", "con") for seed in (42, 43, 44)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        measured = list(pool.map(case, [(r, n, method) for n in names for r in rows]))
    summary = []
    for name in names:
        group = [r for r in measured if r["model"] == name]
        eligible = [r for r in group if r["qualified"]]
        summary.append(dict(model=name, total=len(group), finite=sum(r["finite"] for r in group),
            projected_qualified=len(eligible), projected_all_six_pass=sum(r["measurement"]["joint_pass"] for r in eligible),
            force_mean_Nm=float(np.mean([r["measurement"]["metrics"]["force_balance_Nm"] for r in eligible])),
            energy_mean_J=float(np.mean([r["measurement"]["metrics"]["work_energy_J"] for r in eligible]))))
    write(DIRECTORY/"summary.json", dict(summary=summary, rows=measured, method=method,
        selection_sha256=sha(FIT/"selection.json"), sensitivity_sha256=sha(FIT/"sensitivity.json"),
        source_sha256=sha(__file__), forecasts_unchanged=True, diagnostic_not_formal_native_scores=True,
        interpretation="bandwidth-limited projection diagnostic with raw-coordinate and unfiltered companions"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
