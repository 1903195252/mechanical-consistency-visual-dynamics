"""Check sensitivity and frequency response of the frozen visual phase fit."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from scipy.signal import savgol_filter

from .optimize_visual_phase import DIRECTORY, project_track, INTERIOR
from .optimize_visual_measurement import reference
from .visual_calibration import ROOT, load_protocol, write, sha


def case(task):
    from experiments.keycld_corpus.prepare import point_states
    from experiments.keycld_corpus.physics_score import measure, residuals, MODULES
    row, method = task
    params, truth, _, _, _, _, _ = reference(row)
    c = params["config"]
    t = np.arange(90)/30
    target = params["command_speed_rad_s"]
    limits = json.loads((ROOT / "output/keycld_corpus/protocol.json").read_text())["evaluation"]["limits_SI"]
    with np.load(ROOT / row["physics_reference"]) as data:
        q = np.column_stack([np.interp(1 + .8*t, data["time"], data["q"][:, j]) for j in range(3)])
    warped = point_states(q, np.zeros_like(q), np.zeros_like(q), c)[0]
    phase = np.unwrap(np.arctan2(truth[:, 0, 1]-c["pivot_height_m"], truth[:, 0, 0]))
    perturbed = phase + .1*np.sin(2*np.pi*t)
    module = MODULES[c["category"]]
    q = np.array([module.inverse_kinematics(theta, c) for theta in perturbed])
    oscillating = point_states(q, np.zeros_like(q), np.zeros_like(q), c)[0]
    radial_defect = truth.copy()
    radial = radial_defect[:, 0]-[0., c["pivot_height_m"]]
    radial_defect[:, 0] += .005*radial/np.linalg.norm(radial, axis=1, keepdims=True)
    controls = dict(correct=(truth, target), time_scale_0_8=(warped, target),
                    phase_1Hz_0_1rad=(oscillating, target),
                    wrong_actuator_plus_0_5Nm=(truth, target+.5/c["actuator"]["speed_gain_Nm_s_rad"]))
    results = {}
    for name, (points, command) in controls.items():
        projected = project_track(points, t, c, method)
        results[name] = measure(t[INTERIOR], *(v[INTERIOR] for v in projected), truth[INTERIOR],
                                params, command, limits)
    raw_g = residuals(radial_defect, np.zeros_like(truth), np.zeros_like(truth), c)[0]
    return dict(case_id=row["case_id"], controls=results,
                radial_defect_detected_on_unchanged_raw_coordinates=bool(abs(raw_g[3:]).max() > limits["geometry_m"]))


def run():
    method = json.loads((DIRECTORY/"selection.json").read_text())["selected"]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(case, [(r, method) for r in load_protocol()["planar_records"]]))
    summary = {name: dict(total=len(results), passed=sum(r["controls"][name]["joint_pass"] for r in results),
        force_failed=sum(not r["controls"][name]["passed"]["force_balance_Nm"] for r in results),
        energy_failed=sum(not r["controls"][name]["passed"]["work_energy_J"] for r in results))
        for name in results[0]["controls"]}
    frequency = []
    if method.startswith("phase_sg"):
        w, d = map(int, method[len("phase_sg"):].split("d"))
        t = np.arange(600)/30
        for hz in (.25, .5, 1., 2., 3., 5.):
            phase = .1*np.sin(2*np.pi*hz*t)
            gains = []
            for order in (0, 1, 2):
                true = .1*(2*np.pi*hz)**order*np.sin(2*np.pi*hz*t+order*np.pi/2)
                output = savgol_filter(phase, w, d, deriv=order, delta=1/30)
                s = slice(w, -w)
                gains.append(float(np.dot(output[s], true[s])/np.dot(true[s], true[s])))
            frequency.append(dict(frequency_hz=hz, phase_gain=gains[0], velocity_gain=gains[1], acceleration_gain=gains[2]))
    write(DIRECTORY/"sensitivity.json", dict(method=method, summary=summary, rows=results,
        raw_radial_defects_detected=sum(r["radial_defect_detected_on_unchanged_raw_coordinates"] for r in results),
        frequency_response=frequency, source_sha256=sha(__file__),
        selection_sha256=sha(DIRECTORY/"selection.json"),
        use="additional bandwidth-limited measurement, retain unfiltered companion; not an all-frequency replacement"))
    print(json.dumps(dict(summary=summary, frequency_response=frequency)), flush=True)


if __name__ == "__main__":
    run()
