"""Check that numerical improvements preserve known physical defects."""
from concurrent.futures import ProcessPoolExecutor
import copy
import json

import numpy as np

from .measurement_derivatives import planar_derivatives
from .optimize_visual_measurement import reference
from .visual_calibration import ROOT, OUT, load_protocol, write, sha


def case(task):
    from experiments.keycld_corpus.physics_score import measure
    from experiments.keycld_corpus.prepare import point_states
    row, method = task
    params, x, _, _, _, _, _ = reference(row)
    c = params["config"]
    limits = json.loads((ROOT / "output/keycld_corpus/protocol.json").read_text())["evaluation"]["limits_SI"]
    t, selected = np.arange(90)/30, slice(5, -5)
    radial_error = x.copy()
    radial = x[:, 0] - [0, c["pivot_height_m"]]
    radial_error[:, 0] += .005 * radial / np.linalg.norm(radial, axis=1, keepdims=True)
    with np.load(ROOT / row["physics_reference"]) as data:
        warped_time = 1 + .8 * t
        values = [np.stack([np.interp(warped_time, data["time"], data[name][:, j]) for j in range(3)], axis=1)
                  for name in ("q", "qdot", "qddot")]
    warped = point_states(values[0], values[1]*.8, values[2]*.64, c)[0]
    target = params["command_speed_rad_s"]
    wrong_target = target + .5 / c["actuator"]["speed_gain_Nm_s_rad"]
    tracks = dict(correct=(x, target), radial_defect_5mm=(radial_error, target),
                  time_scale_0_8=(warped, target), wrong_actuator_plus_0_5Nm=(x, wrong_target))
    results = {}
    for name, (points, command) in tracks.items():
        v, a = planar_derivatives(t, points, c, method)
        results[name] = measure(t[selected], points[selected], v[selected], a[selected], x[selected], params, command, limits)
    return dict(case_id=row["case_id"], assembly_id=row["assembly_id"], controls=results)


def run():
    selection = OUT / "optimization/coordinate_derivatives/selection.json"
    method = json.loads(selection.read_text())["selected"]
    rows = load_protocol()["planar_records"]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(case, [(row, method) for row in rows]))
    summary = {}
    for track in results[0]["controls"]:
        summary[track] = dict(total=len(results), joint_pass=sum(r["controls"][track]["joint_pass"] for r in results),
            component_failures={key: sum(not r["controls"][track]["passed"][key] for r in results)
                                for key in results[0]["controls"][track]["passed"]})
    write(OUT / "optimization/sensitivity.json", dict(method=method, source_sha256=sha(__file__),
        selection_sha256=sha(selection), summary=summary, rows=results,
        scope="numerical measurement and SI sensitivity; not a substitute for RGB corruption tests"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
