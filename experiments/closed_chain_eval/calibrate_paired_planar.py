"""Match held-out calibration exactly to the common RGB measurement track."""
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from .visual_calibration import ROOT,OUT,load_protocol,sha,write
from .remeasure_planar_rgb import DIRECTORY,measure_case


def run():
    rows=load_protocol()["planar_records"]
    for row in rows:
        source=OUT/"readouts/shared_planar_64"/(row["case_id"]+".npz")
        with np.load(source) as data:points=data["points"]
        target=DIRECTORY/"readouts/calibration_reference"/source.name;target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,points=points,time=np.arange(90)/30,source_sha256=np.array(sha(source)))
    with ProcessPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(measure_case,[(r,"calibration_reference") for r in rows]))
    by_id={r["case_id"]:r["calibration_role"] for r in rows}
    for r in results:r["calibration_role"]=by_id[r["case_id"]]
    write(DIRECTORY/"calibration.json",dict(pipeline="shared_planar_64 + cartesian_spline5 + full-future raw geometry",
        rows=results,quantile_fit_split="calibration",coverage_check_split="verification",
        test_predictions_used_for_calibration=False,engineering_limits_changed=False,
        point_and_geometry_frames=list(range(3,90)),derivative_frames=list(range(5,85))))
    print(json.dumps(dict(calibration_cases=len(results))))


if __name__=="__main__":run()
