"""Optimize noisy video differentiation on development references only.

This is the explicitly projected measurement track. Raw image coordinates and
their closure failures remain separate. The fit uses measured crank phase,
time and public geometry, without actuator commands or dynamic parameters.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from scipy.signal import savgol_filter

from .measurement_derivatives import planar_derivatives
from .optimize_visual_measurement import reference
from .visual_calibration import ROOT, OUT, load_protocol, sha, write


DIRECTORY = OUT / "optimization/visual_phase"
READER = OUT / "optimization/robust_planar_reader"
DATA = ROOT / "output/keycld_corpus"
BASELINE = "cartesian_spline5"
CANDIDATES = [BASELINE] + ["phase_sg%dd%d" % (w, d) for w, d in
    [(7, 3), (9, 3), (11, 3), (15, 3), (21, 3), (9, 5), (11, 5),
     (15, 5), (21, 5), (31, 5), (15, 7), (21, 7), (31, 7)]]
INTERIOR = slice(5, 85)


def project_track(points, t, config, method):
    """Return one measured phase trajectory and its geometric reconstruction."""
    from experiments.con_corpus.crank_readout import reconstruct
    points = np.asarray(points, dtype=float)
    if method == BASELINE:
        v, a = planar_derivatives(t, points, config, method)
        return reconstruct(points, v, a, config)[0]
    if not method.startswith("phase_sg"):
        raise ValueError("Unknown visual phase measurement")
    window, degree = map(int, method[len("phase_sg"):].split("d"))
    dt = float(np.diff(t).mean())
    if not np.allclose(np.diff(t), dt, atol=1e-8, rtol=1e-5):
        raise ValueError("Phase fit requires uniform timestamps")
    radial = points[:, 0] - [0., config["pivot_height_m"]]
    if np.min(np.linalg.norm(radial, axis=1)) < .001:
        raise ValueError("Singular measured crank direction")
    phase = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
    theta, omega, alpha = [savgol_filter(phase, window, degree, deriv=k, delta=dt) for k in (0, 1, 2)]
    # reconstruct() uses only B's direction and its angular derivatives. Unit
    # vectors pass that measured phase without imposing any force equation.
    radial = np.column_stack((np.cos(theta), np.sin(theta)))
    tangent = np.column_stack((-radial[:, 1], radial[:, 0]))
    x, v, a = [np.zeros_like(points) for _ in range(3)]
    x[:, 0] = radial + [0., config["pivot_height_m"]]
    v[:, 0] = omega[:, None] * tangent
    a[:, 0] = alpha[:, None] * tangent - omega[:, None] ** 2 * radial
    return reconstruct(x, v, a, config)[0]


def development_readouts(rows):
    import torch
    from .planar_visual_reader import CoordinateReader
    from .robust_planar_reader import degradation, QUALITIES
    torch.set_num_threads(4)
    net = CoordinateReader().cuda().eval()
    net.load_state_dict(torch.load(READER / "final.pt", map_location="cpu")["model"])
    for quality, (sigma, contrast) in QUALITIES.items():
        for row in rows:
            destination = DIRECTORY / "development_readouts" / quality / (row["case_id"] + ".npz")
            if destination.exists():
                continue
            with np.load(DATA / row["observations"]) as data:
                images = data["x"]
            predictions = []
            with torch.no_grad():
                for first in range(0, len(images), 45):
                    x = torch.from_numpy(images[first:first+45]).cuda().permute(0, 3, 1, 2).float() / 255
                    predictions.append(net(degradation(x, sigma, contrast))[0].cpu().numpy())
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, points=np.concatenate(predictions) * .65 + [.2, .35],
                                time=np.arange(90)/30, reader_sha256=np.array(sha(READER / "final.pt")))
        print(json.dumps(dict(stage="phase_development_readout", quality=quality, total=len(rows))), flush=True)


def selection_case(task):
    row, quality = task
    params, x, v, a, _, _, _ = reference(row)
    with np.load(DIRECTORY / "development_readouts" / quality / (row["case_id"] + ".npz")) as data:
        points, t = data["points"], data["time"]
    results = {}
    for method in CANDIDATES:
        predicted = project_track(points, t, params["config"], method)
        errors = [float(np.mean((p[INTERIOR] - r[INTERIOR]) ** 2)) for p, r in zip(predicted, (x, v, a))]
        results[method] = dict(position_mse=errors[0], velocity_mse=errors[1], acceleration_mse=errors[2],
                              objective=sum(e / s ** 2 for e, s in zip(errors, (.01, .01, .1))))
    return dict(case_id=row["case_id"], assembly_id=row["assembly_id"], quality=quality, candidates=results)


def score_case(task):
    from experiments.keycld_corpus.physics_score import measure, residuals
    row, quality, method = task
    params, truth, _, _, _, _, _ = reference(row)
    source = READER / "stress_readouts/robust" / quality / (row["case_id"] + ".npz")
    with np.load(source) as data:
        points, t = data["points"], data["time"]
    values = project_track(points, t, params["config"], method)
    limits = json.loads((DATA / "protocol.json").read_text())["evaluation"]["limits_SI"]
    g, _, _ = residuals(points, np.zeros_like(points), np.zeros_like(points), params["config"])
    measured = measure(t[INTERIOR], *(p[INTERIOR] for p in values), truth[INTERIOR], params,
                       params["command_speed_rad_s"], limits)
    return dict(case_id=row["case_id"], assembly_id=row["assembly_id"], category=row["category"],
                calibration_role=row["calibration_role"], quality=quality, method=method,
                raw_geometry_max_m=float(abs(g[3:]).max()), readout_sha256=sha(source),
                correction_point_rms_m=float(np.sqrt(np.mean(np.sum((values[0][INTERIOR]-points[INTERIOR])**2, axis=-1)))),
                measurement=measured)


def select():
    from .robust_planar_reader import QUALITIES
    rows = [r for r in json.loads((DATA / "dataset_manifest.json").read_text())["records"] if r["split"] == "validation"]
    if {r["assembly_id"] for r in rows} & {r["assembly_id"] for r in load_protocol()["planar_records"]}:
        raise ValueError("Development overlaps calibration")
    spec = dict(candidates=CANDIDATES, development_ids=[r["case_id"] for r in rows], qualities=QUALITIES,
                objective="mean projected coordinate/velocity/acceleration MSE normalized by 0.01m/0.01m/s/0.1m/s2",
                model_predictions_used_for_selection=False, common_frames=list(range(5, 85)),
                scope="post-hoc kinematic reconstruction; raw geometric readouts remain unchanged",
                reader_sha256=sha(READER / "final.pt"), source_sha256=sha(__file__))
    path = DIRECTORY / "specification.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Frozen phase specification changed")
    write(path, spec)
    development_readouts(rows)
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(selection_case, [(r, q) for q in QUALITIES for r in rows]))
    objective = {m: float(np.mean([r["candidates"][m]["objective"] for r in results])) for m in CANDIDATES}
    selected = min(objective, key=objective.get)
    write(DIRECTORY / "selection.json", dict(selected=selected, objective=objective, rows=results,
                                             specification_sha256=sha(path)))
    print(json.dumps(dict(selected=selected, objective=objective)), flush=True)


def verify():
    from .robust_planar_reader import QUALITIES
    selected = json.loads((DIRECTORY / "selection.json").read_text())["selected"]
    rows = load_protocol()["planar_records"]
    tasks = [(r, q, m) for q in QUALITIES for r in rows for m in dict.fromkeys((BASELINE, selected))]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(score_case, tasks))
    summaries = []
    for role in ("calibration", "verification"):
        for q in QUALITIES:
            for m in dict.fromkeys((BASELINE, selected)):
                group = [r for r in results if r["calibration_role"] == role and r["quality"] == q and r["method"] == m]
                metrics = group[0]["measurement"]["metrics"]
                summaries.append(dict(role=role, quality=q, method=m, total=len(group),
                    all_six_pass=sum(r["measurement"]["joint_pass"] for r in group),
                    mean={k: float(np.mean([r["measurement"]["metrics"][k] for r in group])) for k in metrics},
                    maximum={k: max(r["measurement"]["metrics"][k] for r in group) for k in metrics}))
    write(DIRECTORY / "verification.json", dict(selected=selected, summary=summaries, rows=results,
        selection_sha256=sha(DIRECTORY / "selection.json"), engineering_limits_unchanged=True,
        qualification_is_post_projection_not_native=True))
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "verify", "all"), default="all")
    args = parser.parse_args()
    if args.stage in ("select", "all"):
        select()
    if args.stage in ("verify", "all"):
        verify()
