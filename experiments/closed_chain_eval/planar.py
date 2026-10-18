"""Independent 4R / 3R+1P mechanism adapter, reusing audited SI mechanics.

Inputs are physical generalized coordinates, not unlabeled latent keypoints.
Image-to-state calibration belongs to the declared observation reader.
"""
from pathlib import Path
import sys

import numpy as np

from .core import (CORE_METRICS, MetricSpec, PhysicsFrame, differentiate,
                   energy_residual, equation_residuals, score_trajectory)

EXPERIMENTS = Path(__file__).resolve().parents[1]
for directory in (EXPERIMENTS / "fourbar", EXPERIMENTS / "keycld_fourbar"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import extended_mechanics as physics


def physics_frame(q, v, motor_tau, configuration):
    M, C, G = physics.mass_bias(q, v, configuration)
    gamma, _ = physics.derivatives(q, configuration)
    return PhysicsFrame(
        mass=M, bias=C + G, actuation=np.array([motor_tau, 0., 0.]),
        external=physics.external(q, configuration),
        friction=physics.friction(v, configuration),
        closure=physics.closure(q, configuration),
        jacobian=physics.jacobian(q, configuration),
        jacobian_dot_velocity=physics.curvature(q, v, configuration),
        tangent=gamma[:, None],
    )


def mechanical_energy(q, v, configuration):
    c = configuration
    M, _, _ = physics.mass_bias(q, v, c)
    mass, _ = physics.bodies(q, v, c)
    l1, l2, l3 = c["lengths_m"]
    heights = np.array([
        l1 / 2 * np.sin(q[0]),
        l1 * np.sin(q[0]) + l2 / 2 * np.sin(q[1]),
        0. if physics.is_slider(c) else l3 / 2 * np.sin(q[2]),
        l1 * np.sin(q[0]) + c["payload_along_coupler_m"] * np.sin(q[1])
        + c["payload_normal_offset_m"] * np.cos(q[1]),
    ]) + c["pivot_height_m"]
    return float(.5 * v @ M @ v + mass @ heights * c["gravity_m_s2"])


def diagnostic_specs():
    """One SI unit per channel is a transparent diagnostic reference, not a gate."""
    channels = {"geometry": (2, "m"), "constraint_velocity": (2, "m/s"),
                "constraint_acceleration": (2, "m/s^2"), "dynamics": (1, "N*m"),
                "energy": (1, "J"), "prediction": (6, "m")}
    return {name: MetricSpec([unit] * count, [1.] * count,
                            "max" if name in ("geometry", "energy") else "rms")
            for name, (count, unit) in channels.items()}


def evaluate_planar(t, predicted_q, reference_q, motor_tau, configuration,
                    prefix_frames=3, specs=None, calibration_id=None):
    """Score a canonical state trajectory, using the common derivative path.

    The independent evaluator receives truth; no model is called in this function.
    Prediction has observed-prefix positions followed by generated positions.
    Fixed 7-sample cubic stencils exclude three boundary frames for derivatives.
    Geometry/prediction still include ALL future frames.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(predicted_q, dtype=float)
    reference = np.asarray(reference_q, dtype=float)
    tau = np.asarray(motor_tau, dtype=float)
    if q.shape != (len(t), 3) or reference.shape != q.shape or tau.shape != t.shape:
        raise ValueError("Expected q/reference (T,3), motor_tau (T,)")
    if not np.isfinite(reference).all() or not np.isfinite(tau).all():
        raise ValueError("Evaluator reference and applied torques must be finite")
    if not 1 <= prefix_frames < len(t):
        raise ValueError("Invalid observed prefix length")
    periodic = (0, 1) if physics.is_slider(configuration) else (0, 1, 2)
    velocity, acceleration, interior = differentiate(t, q, periodic=periodic)
    future = np.arange(len(t)) >= prefix_frames
    dynamics_indices = np.flatnonzero(future & interior)
    if len(dynamics_indices) < 2:
        raise ValueError("Insufficient future samples for derivative/energy checks")
    errors = {"geometry": np.full((int(future.sum()), 2), np.nan),
              "prediction": np.full((int(future.sum()), 6), np.nan)}
    for row, i in enumerate(np.flatnonzero(future)):
        if not np.isfinite(q[i]).all():
            continue
        errors["geometry"][row] = physics.closure(q[i], configuration)
        predicted_points = physics.point_states(q[i], np.zeros(3), np.zeros(3), configuration)[0]
        reference_points = physics.point_states(reference[i], np.zeros(3), np.zeros(3), configuration)[0]
        errors["prediction"][row] = (predicted_points - reference_points).ravel()
    for name, count in (("constraint_velocity", 2), ("constraint_acceleration", 2), ("dynamics", 1)):
        errors[name] = np.full((len(dynamics_indices), count), np.nan)
    energy = np.full(len(dynamics_indices), np.nan)
    power = energy.copy()
    failures = []
    for row, i in enumerate(dynamics_indices):
        if not all(np.isfinite(value).all() for value in (q[i], velocity[i], acceleration[i])):
            failures.append({"frame": int(i), "reason": "unobserved_or_nonfinite_stencil"})
            continue
        try:
            frame = physics_frame(q[i], velocity[i], tau[i], configuration)
            residuals = equation_residuals(frame, velocity[i], acceleration[i])
            for name in ("constraint_velocity", "constraint_acceleration", "dynamics"):
                errors[name][row] = residuals[name]
            energy[row] = mechanical_energy(q[i], velocity[i], configuration)
            power[row] = (frame.actuation + frame.external + frame.friction) @ velocity[i]
        except (ValueError, np.linalg.LinAlgError) as error:
            failures.append({"frame": int(i), "reason": str(error)})
    errors["energy"] = energy_residual(t[dynamics_indices], energy, power)[:, None]
    expected = {name: len(errors[name]) for name in CORE_METRICS}
    result = score_trajectory(errors, specs or diagnostic_specs(), expected, calibration_id)
    result.update(
        mechanism_family="3R1P" if physics.is_slider(configuration) else "4R",
        reader_id="canonical_physical_coordinates_v1",
        derivative_policy_id="centred_cubic_7_v1",
        prefix_frames=prefix_frames,
        future_frames=int(future.sum()),
        dynamics_frame_indices=dynamics_indices.tolist(),
        numerical_failures=failures,
        interpretation="Dynamics residual must be read together with geometry and derivative constraints; no state projection.",
    )
    return result
