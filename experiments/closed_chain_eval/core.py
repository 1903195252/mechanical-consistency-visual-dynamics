"""Independent physical residuals and coverage-aware scoring; NumPy only.

Mechanism adapters, not learned models, supply M, forces and the tangent map.
No prediction is projected, repaired or fitted to future ground truth here.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


CORE_METRICS = (
    "geometry", "constraint_velocity", "constraint_acceleration",
    "dynamics", "energy", "prediction",
)


def _finite_array(value, name):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("{} must be finite".format(name))
    return result


def _time(t):
    t = _finite_array(t, "time")
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Time must be a strictly increasing vector")
    return t


def differentiate(t, q, periodic=(), window=7, degree=3):
    """Common local-polynomial derivative for Euclidean/unwrapped coordinates.

    Only complete, centred stencils are used. Invalid samples contaminate their
    whole stencil instead of being interpolated. Quaternion poses need a separate
    audited manifold adapter; this routine must not be applied to quaternion q.
    """
    t = _time(t)
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[0] != len(t):
        raise ValueError("q must have shape (time, coordinates)")
    if window % 2 != 1 or not 2 <= degree < window or len(t) < window:
        raise ValueError("Require odd window, 2 <= degree < window <= samples")
    q = q.copy()
    # Unwrap finite segments separately: one failed observation must not destroy
    # every later angle, and must not be filled using future reference states.
    for column in periodic:
        indices = np.flatnonzero(np.isfinite(q[:, column]))
        for segment in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
            if len(segment):
                q[segment, column] = np.unwrap(q[segment, column])
    velocity = np.full_like(q, np.nan)
    acceleration = np.full_like(q, np.nan)
    interior = np.zeros(len(t), dtype=bool)
    half = window // 2
    for i in range(half, len(t) - half):
        interior[i] = True
        indices = slice(i - half, i + half + 1)
        values = q[indices]
        if not np.isfinite(values).all():
            continue
        offset = t[indices] - t[i]
        scale = np.max(np.abs(offset))
        design = np.vander(offset / scale, degree + 1, increasing=True)
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        velocity[i] = coefficients[1] / scale
        acceleration[i] = 2 * coefficients[2] / scale**2
    return velocity, acceleration, interior


@dataclass
class PhysicsFrame:
    mass: np.ndarray
    bias: np.ndarray  # Coriolis + centrifugal + gravity
    actuation: np.ndarray  # B(q) u in generalized force coordinates
    external: np.ndarray
    friction: np.ndarray  # Signed force; opposes velocity
    closure: np.ndarray
    jacobian: np.ndarray
    jacobian_dot_velocity: np.ndarray
    tangent: np.ndarray  # v = N(q) sdot, in registered physical coordinates s


def equation_residuals(frame, velocity, acceleration):
    """Eliminate ideal constraint reactions without fitting their magnitudes.

    Supports multiple independent coordinates and nq != nv: the mechanism
    adapter provides velocity-coordinate J and N, never raw quaternion qdot.
    The registry fixes N and its coordinate units; arbitrary SVD bases cannot
    be compared as motor torques.
    """
    v = _finite_array(velocity, "velocity")
    a = _finite_array(acceleration, "acceleration")
    if v.ndim != 1 or a.shape != v.shape:
        raise ValueError("Velocity and acceleration shapes differ")
    n = len(v)
    M = _finite_array(frame.mass, "mass")
    J = _finite_array(frame.jacobian, "jacobian")
    N = _finite_array(frame.tangent, "tangent")
    phi = _finite_array(frame.closure, "closure")
    curvature = _finite_array(frame.jacobian_dot_velocity, "curvature")
    if M.shape != (n, n) or J.ndim != 2 or J.shape[1] != n:
        raise ValueError("Invalid mass/Jacobian shape")
    if phi.shape != (len(J),) or curvature.shape != phi.shape:
        raise ValueError("Invalid constraint residual shape")
    if N.ndim != 2 or N.shape[0] != n or not 1 <= N.shape[1] <= n:
        raise ValueError("Invalid tangent shape")
    if not np.allclose(M, M.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Mass matrix must be symmetric")
    np.linalg.cholesky(M)
    if np.linalg.matrix_rank(N) != N.shape[1]:
        raise ValueError("Singular tangent chart")
    if np.linalg.matrix_rank(J) != n - N.shape[1]:
        raise ValueError("Constraint rank does not match independent coordinates")
    # Relative row/column normalization avoids an absolute m-vs-rad tolerance.
    normalizer = np.linalg.norm(J, axis=1)[:, None] * np.linalg.norm(N, axis=0)
    if np.any(np.abs(J @ N) > 1e-9 * np.maximum(normalizer, 1e-30)):
        raise ValueError("Tangent does not eliminate constraint reactions")
    vectors = [_finite_array(value, name) for value, name in (
        (frame.bias, "bias"), (frame.actuation, "actuation"),
        (frame.external, "external"), (frame.friction, "friction"),
    )]
    if any(value.shape != (n,) for value in vectors):
        raise ValueError("Generalized force shape mismatch")
    bias, actuation, external, friction = vectors
    return {
        "geometry": phi,
        "constraint_velocity": J @ v,
        "constraint_acceleration": J @ a + curvature,
        "dynamics": N.T @ (M @ a + bias - actuation - external - friction),
    }


def energy_residual(t, energy, signed_power):
    """E(t)-E(t0)-integral[(Q_act+Q_ext+Q_fric).v] dt, in joules.

    For fixed ideal constraints only. All mechanical conservative energy belongs
    in E. Dissipation is already negative signed power; do not subtract it twice.
    """
    t = _time(t)
    energy = np.asarray(energy, dtype=float)
    power = np.asarray(signed_power, dtype=float)
    if energy.shape != t.shape or power.shape != t.shape:
        raise ValueError("Energy/power must have one sample per timestamp")
    work = np.r_[0., np.cumsum(.5 * (power[:-1] + power[1:]) * np.diff(t))]
    return energy - energy[0] - work


@dataclass(frozen=True)
class MetricSpec:
    units: Sequence[str]
    scales: Sequence[float]  # Calibration/training quantities, fixed before test
    reduction: str = "rms"  # Per channel, then maximum across channels
    limit: Optional[float] = None  # Dimensionless, absent until calibrated


def metric_summary(errors, spec, expected_samples):
    """Partial/nonfinite outputs never become zero or a shorter easy test set."""
    scale = _finite_array(spec.scales, "scales")
    if scale.ndim != 1 or not len(scale) or np.any(scale <= 0):
        raise ValueError("Every metric channel needs a positive scale")
    if len(spec.units) != len(scale) or spec.reduction not in ("rms", "max"):
        raise ValueError("Invalid metric units/reduction")
    if expected_samples < 1:
        raise ValueError("Expected samples must be positive")
    if spec.limit is not None and (not np.isfinite(spec.limit) or spec.limit <= 0):
        raise ValueError("A calibrated limit must be finite and positive")
    result = {
        "units": list(spec.units), "scales": scale.tolist(),
        "reduction": spec.reduction, "limit": spec.limit,
        "expected_samples": int(expected_samples), "finite_samples": 0,
        "complete": False, "raw_per_channel": None,
        "normalized_per_channel": None, "value": None, "passed": None,
    }
    if errors is None:
        result["status"] = "unavailable"
        return result
    values = np.asarray(errors, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (expected_samples, len(scale)):
        raise ValueError("Metric shape differs from the registered sample/channel count")
    finite = np.isfinite(values).all(axis=1)
    result["finite_samples"] = int(finite.sum())
    result["complete"] = bool(finite.all())
    if not finite.all():
        result["status"] = "nonfinite_or_unobserved"
        return result
    if spec.reduction == "rms":
        # Scaling before squaring also avoids overflow for very large errors.
        largest = np.max(np.abs(values), axis=0)
        raw = largest * np.sqrt(np.mean((values / np.maximum(largest, 1e-300))**2, axis=0))
    else:
        raw = np.max(np.abs(values), axis=0)
    normalized = raw / scale
    if not np.isfinite(normalized).all():
        result["status"] = "numerical_overflow"
        result["complete"] = False
        return result
    value = float(np.max(normalized))
    result.update(status="measured", raw_per_channel=raw.tolist(),
                  normalized_per_channel=normalized.tolist(), value=value,
                  passed=None if spec.limit is None else value <= spec.limit)
    return result


def score_trajectory(errors, specs, expected_samples, calibration_id=None):
    if set(specs) != set(CORE_METRICS) or set(expected_samples) != set(CORE_METRICS):
        raise ValueError("The six core metrics must all be declared")
    if set(errors) - set(CORE_METRICS):
        raise ValueError("Unknown core metric")
    if calibration_id is not None and (not isinstance(calibration_id, str) or not calibration_id.strip()):
        raise ValueError("Calibration identifier cannot be empty")
    summaries = {name: metric_summary(errors.get(name), specs[name], expected_samples[name])
                 for name in CORE_METRICS}
    calibrated = calibration_id is not None and all(s.limit is not None for s in specs.values())
    complete = all(row["complete"] for row in summaries.values())
    # This is a verified-evidence pass indicator, not a claim that unobserved
    # physics is false. Its denominator stays the full predeclared test suite.
    passed = bool(complete and all(row["passed"] for row in summaries.values())) if calibrated else None
    return {"metrics": summaries, "calibration_id": calibration_id,
            "score_status": "calibrated" if calibrated else "calibration_pending",
            "all_metrics_available": complete, "joint_pass": passed}


COMPARISON_FIELDS = (
    "protocol_version", "suite_id", "test_manifest_sha256", "input_track",
    "training_track", "information_policy_id", "budget_policy_id",
    "reader_id", "derivative_policy_id", "sampling_policy_id", "calibration_id",
)


def assert_comparable(cards):
    """Reject a pooled leaderboard with different inputs/priors/measurements."""
    if not cards:
        raise ValueError("No scorecards")
    for card in cards:
        missing = [key for key in COMPARISON_FIELDS if key not in card]
        if missing:
            raise ValueError("Missing comparison metadata: " + ", ".join(missing))
        for key in COMPARISON_FIELDS:
            if key == "calibration_id" and card[key] is None:
                continue
            if not isinstance(card[key], str) or not card[key].strip():
                raise ValueError("Empty or invalid comparison metadata: " + key)
    for field in COMPARISON_FIELDS:
        if any(card[field] != cards[0][field] for card in cards[1:]):
            raise ValueError("Incompatible comparison field: " + field)


def aggregate(rows, expected_cases, calibration_id=None):
    """Equal mechanism, condition, trajectory weighting; missing rows retained.

    IDs are (mechanism_family, condition, case_id). Aggregate one training seed
    and one registered generation sample at a time; cluster CIs are upstream.
    """
    expected = [tuple(key) for key in expected_cases]
    if not expected or any(len(key) != 3 for key in expected) or len(set(expected)) != len(expected):
        raise ValueError("Expected case IDs must be nonempty and unique")
    mapped = {}
    for row in rows:
        key = tuple(row["case_key"])
        if key not in expected or key in mapped:
            raise ValueError("Unknown or duplicate case ID: " + str(key))
        mapped[key] = row
    if calibration_id is not None:
        for row in mapped.values():
            if row.get("calibration_id") != calibration_id or row.get("score_status") != "calibrated":
                raise ValueError("Cannot aggregate uncalibrated or mixed-calibration results")
    groups = {}
    for key in expected:
        groups.setdefault(key[:2], []).append(mapped.get(key))
    strata = []
    for (family, condition), entries in sorted(groups.items()):
        measured = sum(row is not None and row["all_metrics_available"] for row in entries)
        passed = sum(row is not None and row["joint_pass"] is True for row in entries)
        metrics = {}
        for name in CORE_METRICS:
            values = [row["metrics"][name]["value"] for row in entries
                      if row is not None and row["metrics"][name]["value"] is not None]
            metrics[name] = {"n": len(values), "expected": len(entries),
                             "conditional_mean": float(np.mean(values)) if values else None}
        strata.append({"mechanism_family": family, "condition": condition,
                       "expected": len(entries), "submitted": sum(row is not None for row in entries),
                       "complete": measured, "coverage": measured / len(entries),
                       "verified_pass_rate": passed / len(entries) if calibration_id is not None else None,
                       "metrics": metrics})
    families = sorted({row["mechanism_family"] for row in strata})
    def macro(field):
        return float(np.mean([np.mean([row[field] for row in strata if row["mechanism_family"] == family])
                              for family in families]))
    return {"expected_cases": len(expected), "submitted_cases": len(rows),
            "macro_coverage": macro("coverage"),
            "verified_score_100": 100 * macro("verified_pass_rate") if calibration_id is not None else None,
            "calibration_id": calibration_id, "strata": strata}
