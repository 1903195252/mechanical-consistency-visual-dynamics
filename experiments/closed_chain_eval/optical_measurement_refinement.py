"""Fuse framewise coordinates with measured optical motion, not rigid dynamics.

The temporal term follows bidirectionally checked image flow. It does not prefer
constant velocity, a closure manifold, or a physically correct force. Its weight
and the numerical derivative are selected only on development reference videos.
"""
from concurrent.futures import ProcessPoolExecutor
import json

import cv2
import numpy as np
from scipy.linalg import solve_banded

from .measurement_derivatives import planar_derivatives
from .optimize_visual_measurement import reference
from .visual_calibration import ROOT, OUT, load_protocol, sha, write, error_statistics
from .score_visual_calibration import metric_map


DATA = ROOT / "output/keycld_corpus"
SHIFT = OUT / "optimization/planar_reader"
DIRECTORY = OUT / "optimization/optical_motion"
WEIGHTS = (0., 1., 4., 16., 64.)
METHODS = ("cartesian_spline5", "cartesian_sg7d5", "cartesian_sg9d5", "polar_sg7d5", "polar_sg9d5", "polar_spline5")


def optical_edges(images, coordinates):
    cv2.setNumThreads(1)
    pixel = (np.asarray(coordinates) - [.2, .35]) / .65 * [31.5, -31.5] + 31.5
    gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images]
    increments, confidence = [], []
    options = dict(winSize=(7, 7), maxLevel=1,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .0001),
                   minEigThreshold=1e-6)
    for i in range(1, len(images)):
        start = pixel[i-1].astype(np.float32).reshape(-1, 1, 2)
        end, status, _ = cv2.calcOpticalFlowPyrLK(gray[i-1], gray[i], start, None, **options)
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[i-1], end, None, **options)
        error = np.linalg.norm(back[:, 0] - start[:, 0], axis=-1)
        valid = (status[:,0] == 1) & (reverse[:,0] == 1) & np.isfinite(end[:,0]).all(-1) & (error <= .5)
        displacement = (end[:, 0] - start[:, 0]) * [.65/31.5, -.65/31.5]
        increments.append(np.where(valid[:,None], displacement, 0.))
        confidence.append(valid.astype(float))
    return np.asarray(increments), np.asarray(confidence)


def fuse(coordinates, increments, confidence, weight):
    y = np.asarray(coordinates, dtype=float)
    x = y.copy()
    for point in range(y.shape[1]):
        w = weight * confidence[:, point]
        diagonal = 1 + np.r_[w,0] + np.r_[0,w]
        system = np.zeros((3, len(y)))
        system[1], system[0,1:], system[2,:-1] = diagonal, -w, -w
        rhs = y[:,point].copy()
        rhs[:-1] -= w[:,None] * increments[:,point]
        rhs[1:] += w[:,None] * increments[:,point]
        x[:,point] = solve_banded((1,1), system, rhs)
    return x


def development_case(task):
    from experiments.keycld_corpus.physics_score import measure
    row, shift = task
    with np.load(SHIFT / "development" / (row["case_id"] + ".npz")) as data:
        points = data[shift]
    with np.load(DATA / row["observations"]) as data:
        images = data["x"]
    params, truth, true_v, true_a, _, _, _ = reference(row)
    increments, confidence = optical_edges(images, points)
    t, s = np.arange(90)/30, slice(5, -5)
    limits = json.loads((DATA / "protocol.json").read_text())["evaluation"]["limits_SI"]
    numerical_pass = {}
    for method in METHODS:
        rv, ra = planar_derivatives(t, truth, params["config"], method)
        numerical_pass[method] = measure(t[s],truth[s],rv[s],ra[s],truth[s],params,
                                        params["command_speed_rad_s"],limits)["joint_pass"]
    results = {}
    for weight in WEIGHTS:
        x = fuse(points, increments, confidence, weight)
        for method in METHODS:
            v, a = planar_derivatives(t, x, params["config"], method)
            values = dict(point_rmse_m=float(np.sqrt(np.mean(np.sum((x[3:] - truth[3:])**2, axis=-1)))),
                          velocity_rmse_m_s=float(np.sqrt(np.mean((v[s]-true_v[s])**2))),
                          acceleration_rmse_m_s2=float(np.sqrt(np.mean((a[s]-true_a[s])**2))))
            results["%g/" % weight + method] = values
    cache = DIRECTORY / "development_edges" / (row["case_id"] + ".npz")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, increments=increments, confidence=confidence)
    return dict(case_id=row["case_id"], category=row["category"], assembly_id=row["assembly_id"], candidates=results,
                numerical_pass=numerical_pass,
                usable_optical_edges=float(confidence.mean()))


def held_out_case(task):
    from experiments.keycld_corpus.physics_score import measure, residuals
    from experiments.con_corpus.crank_readout import reconstruct
    row, weight, method = task
    with np.load(SHIFT / "held_out" / (row["case_id"] + ".npz")) as data:
        points, t = data["points"], data["time"]
    with np.load(DATA / row["observations"]) as data:
        images = data["x"]
    increments, confidence = optical_edges(images, points)
    x = fuse(points, increments, confidence, weight)
    params, truth, true_v, true_a, _, _, _ = reference(row)
    limits = json.loads((DATA / "protocol.json").read_text())["evaluation"]["limits_SI"]
    v, a = planar_derivatives(t, x, params["config"], method)
    s = slice(5,-5)
    native = measure(t[s], x[s], v[s], a[s], truth[s], params, params["command_speed_rad_s"], limits)
    result = dict(reader_id="optimized_planar_64", kind="planar", category=row["category"], assembly_id=row["assembly_id"],
        case_id=row["case_id"], calibration_role=row["calibration_role"],
        identity=dict(protocol_sha256=sha(OUT / "protocol.json"), scorer_sha256=sha(__file__),
                      selection_sha256=sha(DIRECTORY / "selection.json")),
        localization=error_statistics(x[3:]-truth[3:]), native_metrics=metric_map(native["metrics"]),
        native_qualified=native["coverage"] == 1., usable_optical_edges=float(confidence.mean()),
        optical_correction_rms_m=float(np.sqrt(np.mean(np.sum((x-points)**2, axis=-1)))))
    try:
        corrected, _ = reconstruct(x[s], v[s], a[s], params["config"])
        m = measure(t[s], *corrected, truth[s], params, params["command_speed_rad_s"], limits)
        result.update(intervention_metrics=metric_map(m["metrics"]), intervention_reachable=True,
                      intervention_qualified=m["coverage"] == 1.,
                      correction_point_rms_m=float(np.sqrt(np.mean(np.sum((corrected[0]-x[s])**2,axis=-1)))))
    except ValueError as e:
        result.update(intervention_metrics={}, intervention_reachable=False, intervention_qualified=False,
                      intervention_failure=str(e))
    exact = measure(t[s], truth[s], true_v[s], true_a[s], truth[s], params, params["command_speed_rad_s"], limits)
    result["numerical_reference_exact"] = metric_map(exact["metrics"])
    tv, ta = planar_derivatives(t, truth, params["config"], method)
    ref = measure(t[s], truth[s], tv[s], ta[s], truth[s], params, params["command_speed_rad_s"], limits)
    result["numerical_reference_sampled"] = metric_map(ref["metrics"])
    destination = OUT / "readouts/optimized_planar_64" / (row["case_id"] + ".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, time=t, points=x, velocity=v, acceleration=a,
                        selection_sha256=np.array(sha(DIRECTORY / "selection.json")))
    result["identity"]["readout_sha256"] = sha(destination)
    write(OUT / "measurements/optimized_planar_64" / (row["case_id"] + ".json"), result)
    return result


def run():
    shift_selection = json.loads((SHIFT / "selection.json").read_text())
    shift = shift_selection["selected"]
    rows = json.loads((DATA / "dataset_manifest.json").read_text())["records"]
    development = [r for r in rows if r["split"] == "validation"]
    spec = dict(weights=WEIGHTS, derivatives=METHODS, source_sha256=sha(__file__), shift=shift,
        development_ids=[r["case_id"] for r in development],
        selection="maximum of mean normalized position, velocity, acceleration error on development references",
        engineering_thresholds_changed=False, physical_motion_prior=False,
        optimization_scales=dict(point_m=.001, velocity_m_s=.01, acceleration_m_s2=.1))
    write(DIRECTORY / "specification.json", spec)
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(development_case, [(r,shift) for r in development]))
    objectives, components = {}, {}
    for key in results[0]["candidates"]:
        values = np.mean([[r["candidates"][key][m] for m in
                          ("point_rmse_m","velocity_rmse_m_s","acceleration_rmse_m_s2")] for r in results], axis=0)
        valid = all(r["numerical_pass"][key.split("/",1)[1]] for r in results)
        objectives[key] = float(np.max(values / [.001,.01,.1])) if valid else None
        components[key] = values.tolist()
    eligible = {key: value for key,value in objectives.items() if value is not None}
    if not eligible:
        raise ValueError("Every derivative configuration fails the development exact-state control")
    selected = min(eligible, key=eligible.get)
    write(DIRECTORY / "selection.json", dict(selected=selected, objectives=objectives, components=components, rows=results,
         specification_sha256=sha(DIRECTORY / "specification.json")))
    weight, method = selected.split("/")
    print(json.dumps(dict(selected=selected, before=components["0/cartesian_spline5"], after=components[selected])), flush=True)
    protocol = load_protocol()
    with ProcessPoolExecutor(max_workers=6) as pool:
        measurements = list(pool.map(held_out_case, [(r,float(weight),method) for r in protocol["planar_records"]]))
    write(DIRECTORY / "reader.json", dict(id="optimized_planar_64",kind="planar",native_resolution=64,
        encoder_resolution=64, role="external_to_KeyCLD_and_CON; optical-flow and numerical refinement",
        selection_sha256=sha(DIRECTORY / "selection.json"), framewise_shift_ensemble=shift))
    write(DIRECTORY / "complete.json", dict(cases=len(measurements), source_sha256=sha(__file__)))


if __name__ == "__main__":
    run()
