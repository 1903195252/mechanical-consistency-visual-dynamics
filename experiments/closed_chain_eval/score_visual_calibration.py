"""Propagate held-out reference-video readout errors through SI measurement.

Raw readout, geometric intervention, numerical reference and qualification are
kept separate. Unreachable/ill-conditioned results remain unavailable.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

import numpy as np

from .core import differentiate
from .visual_calibration import (ROOT, OUT, load_protocol, readout_path, sha,
                                 write, error_statistics)


def metric_map(metrics):
    return {"geometry": metrics.get("geometry_m"), "constraint_velocity": metrics.get("velocity_m_s"),
            "constraint_acceleration": metrics.get("acceleration_m_s2"),
            "dynamics": metrics.get("force_balance_Nm"), "energy": metrics.get("work_energy_J"),
            "prediction": metrics.get("prediction_m")}


def planar_case(row, reader, path):
    from experiments.keycld_corpus.physics_score import measure, residuals
    from experiments.keycld_corpus.prepare import point_states
    from experiments.con_corpus.crank_readout import reconstruct
    with np.load(path) as data:
        x, t = data["points"], data["time"]
    params = json.loads((ROOT / row["physical_parameters"]).read_text())
    with np.load(ROOT / row["physics_reference"]) as data:
        index = np.arange(240, 960, 8)
        truth, truth_v, truth_a = point_states(data["q"][index], data["qdot"][index], data["qddot"][index], params["config"])
    limits = json.loads((ROOT / "output/keycld_corpus/protocol.json").read_text())["evaluation"]["limits_SI"]
    v, a, interior = differentiate(t, x.reshape(len(t), 6))
    v, a = v.reshape(-1, 3, 2), a.reshape(-1, 3, 2)
    tv, ta, _ = differentiate(t, truth.reshape(len(t), 6))
    tv, ta = tv.reshape(-1, 3, 2), ta.reshape(-1, 3, 2)
    c = params["config"]
    target = params["command_speed_rad_s"]

    def evaluate(points, velocity, acceleration):
        points, velocity, acceleration = [value[interior] for value in (points, velocity, acceleration)]
        if not all(np.isfinite(value).all() for value in (points, velocity, acceleration)):
            return dict(metrics={k: None for k in limits}, coverage=0., joint_pass=False,
                        failure_reason="nonfinite_readout")
        g, vg, ag = residuals(points, velocity, acceleration, c)
        rms = lambda values: float(np.sqrt(np.mean(values**2, axis=0)).max())
        # Computing an angle equation off-manifold would not restore qualified
        # mechanics. Preserve the measurable four terms without unnecessary M.
        if np.max(np.abs(g)) > limits["geometry_m"]:
            metrics = dict(geometry_m=float(np.max(np.abs(g))), velocity_m_s=rms(vg),
                           acceleration_m_s2=rms(ag), prediction_m=rms((points - truth[interior]).reshape(-1, 6)),
                           force_balance_Nm=None, work_energy_J=None)
            return dict(metrics=metrics, coverage=0., joint_pass=False, failure_reason="raw_geometry_not_admissible")
        return measure(t[interior], points, velocity, acceleration, truth[interior], params, target, limits)

    native = evaluate(x, v, a)
    ref_exact = evaluate(truth, truth_v, truth_a)
    ref_sampled = evaluate(truth, tv, ta)
    result = dict(localization=error_statistics(x[3:] - truth[3:]),
                  native_metrics=metric_map(native["metrics"]), native_qualified=bool(native["coverage"] == 1),
                  native_failure=native.get("failure_reason"),
                  numerical_reference_exact=metric_map(ref_exact["metrics"]),
                  numerical_reference_sampled=metric_map(ref_sampled["metrics"]),
                  pixel_span_m=1.30 / 63, mechanics_samples=int(interior.sum()),
                  derivative_errors=dict(velocity_rmse_m_s=float(np.sqrt(np.mean((v[interior] - truth_v[interior])**2))),
                      acceleration_rmse_m_s2=float(np.sqrt(np.mean((a[interior] - truth_a[interior])**2))),
                      velocity_same_filter_rmse_m_s=float(np.sqrt(np.mean((v[interior] - tv[interior])**2))),
                      acceleration_same_filter_rmse_m_s2=float(np.sqrt(np.mean((a[interior] - ta[interior])**2)))))
    result["localization"]["point_rmse_pixels"] = result["localization"]["point_euclidean_rmse"] / result["pixel_span_m"]
    try:
        corrected, state = reconstruct(x[interior], v[interior], a[interior], c)
        measured = measure(t[interior], *corrected, truth[interior], params, target, limits)
        result.update(intervention_metrics=metric_map(measured["metrics"]),
                      intervention_reachable=True, intervention_qualified=bool(measured["coverage"] == 1),
                      correction_point_rms_m=float(np.sqrt(np.mean(np.sum((corrected[0] - x[interior])**2, axis=-1)))))
        # Correct reference pixels under the same declared reconstruction are
        # a paired numerical baseline for the reconstructed readout track.
        reference, _ = reconstruct(truth[interior], tv[interior], ta[interior], c)
        reference_measurement = measure(t[interior], *reference, truth[interior], params, target, limits)
        result["numerical_reference_intervention"] = metric_map(reference_measurement["metrics"])
    except (ValueError, np.linalg.LinAlgError) as error:
        result.update(intervention_metrics={}, intervention_reachable=False, intervention_qualified=False,
                      intervention_failure=str(error))
    return result


def spatial_case(row, reader, path):
    from experiments.model_expansion.evaluate_spatial import Audit
    audit = Audit(row)
    with np.load(path) as data:
        q, t = data["qpos"].astype(float), data["time"]
    # LingBot's existing measurement pipeline uses the original 384px observed
    # prefix and 256px future frames. Match that transition before differentiating.
    if reader["native_resolution"] == 256:
        prefix_path = OUT / "readouts/shared_spatial_384" / path.name
        with np.load(prefix_path) as data:
            q[:3] = data["qpos"][:3]
    with np.load(ROOT / row["evaluator_directory"] / "physics.npz") as data:
        truth = data["qpos"][np.arange(120) * 8]
    native = audit.measure(t, q, truth)
    points = np.stack([audit.points(pose) for pose in q])
    ref_points = np.stack([audit.points(pose) for pose in truth])

    def metrics(measured, diagnostic=False):
        return dict(geometry=measured.get("geometry_max_mm", 0.) / 1000,
            constraint_velocity=measured.get("velocity_constraint_max_m_s"),
            constraint_acceleration=measured.get("acceleration_constraint_max_m_s2"),
            dynamics=measured.get("raw_equivalent_effort_rmse" if diagnostic else "effort_rmse"),
            energy=measured.get("raw_work_energy_max_J" if diagnostic else "work_energy_max_J"),
            prediction=measured.get("point_rmse_mm", 0.) / 1000)

    calibration = json.loads((ROOT / "output/model_expansion/spatial" / (row["category"] + "_calibrate.json")).read_text())
    ref = next(r for r in calibration if r["video_id"] == row["video_id"])
    public = json.loads((ROOT / row["public_inputs"]).read_text())
    span = public["observed_camera_frames"][0]["scale_m"]
    result = dict(localization=error_statistics(points[3:] - ref_points[3:]),
                  platform_localization=error_statistics((q[3:, :3] - truth[3:, :3])[:, None, :]),
                  native_metrics=metrics(native), native_qualified=bool(native["mechanical_coverage"]),
                  numerical_reference_exact=metrics(ref["reference_exact"]),
                  numerical_reference_sampled=metrics(ref["reference_30Hz"]),
                  pixel_span_m=span / reader["native_resolution"], effort_unit=audit.unit,
                  mechanics_samples=114,
                  native_raw_equation_diagnostics=metrics(native, diagnostic=True))
    # Pixel equivalence is only a horizontal in-plane scale; it is not a 3D
    # observability claim for this orthographic single-view reader.
    result["localization"]["point_rmse_pixel_equivalent"] = result["localization"]["point_euclidean_rmse"] / result["pixel_span_m"]
    try:
        reconstructed = np.stack([audit.project(pose) for pose in q])
        measured = audit.measure(t, reconstructed, truth)
        corrected_points = np.stack([audit.points(pose) for pose in reconstructed])
        result.update(intervention_metrics=metrics(measured), intervention_reachable=True,
                      intervention_qualified=bool(measured["mechanical_coverage"]),
                      intervention_raw_equation_diagnostics=metrics(measured, diagnostic=True),
                      correction_point_rms_m=float(np.sqrt(np.mean(np.sum((corrected_points[3:] - points[3:])**2, axis=-1)))))
    except (ValueError, np.linalg.LinAlgError) as error:
        result.update(intervention_metrics={}, intervention_reachable=False, intervention_qualified=False,
                      intervention_failure=str(error))
    return result


def case(task):
    row, reader, protocol_sha, source_sha = task
    case_id = row.get("case_id", row.get("video_id"))
    destination = OUT / "measurements" / reader["id"] / (case_id + ".json")
    path = readout_path(reader, row)
    identity = dict(protocol_sha256=protocol_sha, scorer_sha256=source_sha, readout_sha256=sha(path))
    if destination.exists():
        cached = json.loads(destination.read_text())
        if cached["identity"] == identity:
            return cached
        raise ValueError("Existing measurement has different provenance")
    measured = planar_case(row, reader, path) if reader["kind"] == "planar" else spatial_case(row, reader, path)
    result = dict(reader_id=reader["id"], kind=reader["kind"], category=row["category"],
        assembly_id=row["assembly_id"], case_id=case_id, calibration_role=row["calibration_role"],
        identity=identity, **measured)
    write(destination, result)
    return result


def run(kind, workers):
    protocol = load_protocol()
    source_sha, protocol_sha = sha(__file__), sha(OUT / "protocol.json")
    tasks = []
    for reader in protocol["readers"]:
        if kind != "all" and reader["kind"] != kind:
            continue
        rows = protocol[reader["kind"] + "_records"]
        tasks.extend((row, reader, protocol_sha, source_sha) for row in rows)
    missing = [str(readout_path(r, row).relative_to(ROOT)) for row, r, _, _ in tasks if not readout_path(r, row).exists()]
    if missing:
        raise ValueError("Missing readout: " + missing[0])
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(case, task) for task in tasks]
        for index, future in enumerate(as_completed(futures)):
            future.result()
            if (index + 1) % 12 == 0 or index + 1 == len(tasks):
                progress = dict(kind=kind, completed=index + 1, total=len(tasks), seconds=round(time.monotonic() - start, 1))
                write(OUT / (kind + "_measurement_status.json"), progress)
                print(json.dumps(progress), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("planar", "spatial", "all"), default="all")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run(args.kind, args.workers)
