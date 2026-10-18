"""One continuous SI score with independent measurement diagnostics.

This layer consumes the existing six-metric scorecard. It never repairs a
trajectory, changes measurement eligibility, or fits scales to model test errors.
"""
import hashlib
import json

import numpy as np

from .core import CORE_METRICS, aggregate

SCORING_VERSION = "continuous-si-1"


def pixel_span(span_m, width, *, convention):
    """Convert a calibrated horizontal span to metres per pixel.

    Use the actual observation grid before upsampling. ``pixel_centers`` means
    the span joins the first and last pixel centres; ``pixel_edges`` includes
    the outer half pixels. Neither is a localization-uncertainty estimate.
    """
    if not np.isfinite(span_m) or span_m <= 0 or int(width) != width or width < 2:
        raise ValueError("Require a positive span and an integer width >= 2")
    if convention not in ("pixel_centers", "pixel_edges"):
        raise ValueError("Declare the camera's pixel-span convention")
    return float(span_m / (width - 1 if convention == "pixel_centers" else width))


def propagate_covariance(jacobian, covariance):
    """First-order residual covariance J Sigma J^T, retaining correlations.

    The caller supplies the camera/readout/mechanical measurement Jacobian in
    consistent units. A nonlinear trajectory maximum needs pipeline-level
    calibration; this local covariance alone is not its acceptance threshold.
    """
    J = np.asarray(jacobian, dtype=float)
    C = np.asarray(covariance, dtype=float)
    if J.ndim != 2 or C.shape != (J.shape[1], J.shape[1]) or not C.size:
        raise ValueError("Jacobian/covariance dimensions differ")
    if not np.isfinite(J).all() or not np.isfinite(C).all():
        raise ValueError("Measurement covariance and Jacobian must be finite")
    tolerance = 1e-12 * max(float(np.max(np.abs(C))), np.finfo(float).tiny)
    if not np.allclose(C, C.T, rtol=0, atol=tolerance):
        raise ValueError("Measurement covariance must be symmetric")
    if np.linalg.eigvalsh(C).min() < -tolerance:
        raise ValueError("Measurement covariance must be positive semidefinite")
    return J @ C @ J.T


def calibrate_noise_profile(reference_cards, *, profile_id, measurement_id,
                            source_id, split, quantile=0.95):
    """Estimate residual floors from independent reference measurements.

    Inputs must be known-correct reference trajectories passed through the
    declared measurement pipeline, with the same reductions as the scored
    trajectories. This is an empirical quantile, not a confidence guarantee.
    Prediction errors of a candidate model are not calibration measurements.
    """
    if split not in ("calibration", "validation"):
        raise ValueError("Noise profiles require a held-out calibration/validation split")
    if not reference_cards or not 0 < quantile < 1:
        raise ValueError("Require reference samples and a quantile in (0, 1)")
    if any(not isinstance(x, str) or not x.strip()
           for x in (profile_id, measurement_id, source_id)):
        raise ValueError("Declare the profile, measurement pipeline and reference source")
    metrics = {}
    for name in CORE_METRICS:
        rows = [card["metrics"][name] for card in reference_cards]
        units = rows[0]["units"]
        if any(not row["complete"] or row["raw_per_channel"] is None
               or row["units"] != units for row in rows):
            raise ValueError("Incomplete/incompatible calibration measurements: " + name)
        samples = np.asarray([row["raw_per_channel"] for row in rows], dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(units) or not np.isfinite(samples).all():
            raise ValueError("Invalid calibration samples: " + name)
        if np.any(samples < 0):
            raise ValueError("Calibration uses nonnegative metric summaries")
        floor = np.quantile(samples, quantile, axis=0, method="higher")
        metrics[name] = {"units": list(units), "floor_per_channel": floor.tolist()}
    return {"profile_id": profile_id, "measurement_id": measurement_id,
            "source_id": source_id, "split": split, "reference_cases": len(reference_cards),
            "quantile": quantile, "estimator": "empirical_upper_quantile",
            "metrics": metrics}


def _quality(ratio):
    # The reciprocal form avoids overflow when a finite SI error is very large.
    if ratio <= 1:
        return 100. / (1. + ratio * ratio)
    inverse = 1. / ratio
    return 100. * inverse * inverse / (1. + inverse * inverse)


def continuous_scorecard(card, *, noise_profile=None, measurement_id=None):
    """Compute the unique Scomp; calibration never creates a second score.

    Engineering tolerance defines a ratio of one and a score of 50. A matched
    reference-error profile accompanies the result as instrument diagnostics.
    Missing terms contribute zero with the full six-term denominator.
    """
    if card.get("score_status") != "calibrated" or card.get("calibration_id") is None:
        raise ValueError("Engineering scales must be registered/calibrated first")
    if noise_profile is not None:
        if measurement_id is None or noise_profile["measurement_id"] != measurement_id:
            raise ValueError("Noise profile belongs to a different measurement pipeline")
        if noise_profile["split"] not in ("calibration", "validation"):
            raise ValueError("A test-set noise profile is not permitted")
        if set(noise_profile["metrics"]) != set(CORE_METRICS):
            raise ValueError("The noise profile must cover all six metrics")
    metrics, diagnostics = {}, {}
    for name in CORE_METRICS:
        metric = card["metrics"][name]
        scale = np.asarray(metric["scales"], dtype=float) * metric["limit"]
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid engineering scale: " + name)
        if noise_profile is not None:
            reference = noise_profile["metrics"][name]
            floor = np.asarray(reference["floor_per_channel"], dtype=float)
            if reference["units"] != metric["units"] or floor.shape != scale.shape:
                raise ValueError("Calibration channel units/shape differ: " + name)
            if not np.isfinite(floor).all() or np.any(floor < 0):
                raise ValueError("Reference errors must be finite and nonnegative")
            diagnostics[name] = dict(reference_error_per_channel=floor.tolist(),
                reference_error_to_tolerance=(floor/scale).tolist(),
                exceeds_engineering_tolerance=bool(np.any(floor > scale)))
        measured = metric["complete"] and metric["raw_per_channel"] is not None
        ratio, score = None, 0.
        if measured:
            raw = np.asarray(metric["raw_per_channel"], dtype=float)
            if raw.shape != scale.shape or not np.isfinite(raw).all() or np.any(raw < 0):
                raise ValueError("Expected finite nonnegative SI metric summaries")
            with np.errstate(over="ignore"):
                value = float(np.max(raw / scale))
            score = _quality(value)
            ratio = value if np.isfinite(value) else None
        metrics[name] = dict(normalizer_per_channel=scale.tolist(), ratio=ratio,
                             score_100=score, available=bool(measured))
    digest = None if noise_profile is None else hashlib.sha256(
        json.dumps(noise_profile, sort_keys=True, allow_nan=False).encode()).hexdigest()
    result = dict(card)
    result["continuous"] = dict(schema=SCORING_VERSION, rule="100 / (1 + tolerance_ratio ** 2)",
        noise_profile_sha256=digest, measurement_id=measurement_id, metrics=metrics,
        score_100=float(np.mean([m["score_100"] for m in metrics.values()])),
        measurement_diagnostics=diagnostics or None)
    return result


def aggregate_continuous(rows, expected_cases, *, calibration_id):
    """Balance registered cases; retain strict joint rate as an auxiliary."""
    expected = [tuple(key) for key in expected_cases]
    result = aggregate(rows, expected, calibration_id=calibration_id)
    signatures = {(row["continuous"]["schema"], row["continuous"]["noise_profile_sha256"],
                   row["continuous"]["measurement_id"]) for row in rows}
    if len(signatures) > 1:
        raise ValueError("Cannot pool different measurement profiles or scoring versions")
    mapped = {tuple(row["case_key"]): row for row in rows}
    groups = {}
    for key in expected:
        groups.setdefault(key[0], {}).setdefault(key[1], []).append(mapped.get(key))

    def balanced(get_value):
        return float(np.mean([np.mean([np.mean([get_value(row) if row else 0. for row in entries])
                                       for entries in conditions.values()])
                              for conditions in groups.values()]))

    metrics = {name: balanced(lambda r: r["continuous"]["metrics"][name]["score_100"])
               for name in CORE_METRICS}
    result["continuous"] = dict(schema=SCORING_VERSION,
        score_100=float(np.mean(list(metrics.values()))), components_100=metrics)
    result["complete_cases"] = sum(row["all_metrics_available"] for row in rows)
    result["noise_profile_sha256"] = next(iter(signatures))[1] if signatures else None
    return result
