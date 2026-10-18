"""Six SI measurements for declared physical point trajectories and native ODEs."""
import numpy as np

from .core import CORE_METRICS, MetricSpec, differentiate, energy_residual, equation_residuals, score_trajectory
from .planar import mechanical_energy, physics, physics_frame


def point_constraints(x, v, a, c):
    A = np.array([0., c["pivot_height_m"]])
    D = np.array([c["ground_length_m"], c["pivot_height_m"]])
    r = np.stack([x[:, 0] - A, x[:, 1] - x[:, 0], x[:, 1] - D], axis=1)
    rv = np.stack([v[:, 0], v[:, 1] - v[:, 0], v[:, 1]], axis=1)
    ra = np.stack([a[:, 0], a[:, 1] - a[:, 0], a[:, 1]], axis=1)
    radius = np.linalg.norm(r, axis=2)
    safe = np.maximum(radius, 1e-12)
    radial_velocity = np.sum(r * rv, axis=2) / safe
    radial_acceleration = (np.sum(rv * rv + r * ra, axis=2) - radial_velocity**2) / safe
    geometry = radius - np.asarray(c["lengths_m"])
    if physics.is_slider(c):
        geometry[:, 2] = x[:, 1, 1] - c["pivot_height_m"]
        radial_velocity[:, 2] = v[:, 1, 1]
        radial_acceleration[:, 2] = a[:, 1, 1]
    midpoint = lambda values: values[:, 2] - (values[:, 0] + values[:, 1]) / 2
    return {"geometry": np.c_[geometry, midpoint(x)],
            "constraint_velocity": np.c_[radial_velocity, midpoint(v)],
            "constraint_acceleration": np.c_[radial_acceleration, midpoint(a)]}


def specifications(protocol, multiplier=1.):
    counts = {"geometry": 5, "constraint_velocity": 5, "constraint_acceleration": 5,
              "dynamics": 1, "energy": 1, "prediction": 6}
    return {name: MetricSpec([protocol["units"][name]] * count, [1.] * count,
                            "max" if name in ("geometry", "energy") else "rms",
                            protocol["limits_SI"][name] * multiplier)
            for name, count in counts.items()}


def measure(t, x, v, a, reference_points, torque, c, protocol, derivative_audit=False):
    t = np.asarray(t, dtype=float)
    x, v, a, reference_points = [np.asarray(values, dtype=float) for values in (x, v, a, reference_points)]
    if any(values.shape != (len(t), 3, 2) for values in (x, v, a, reference_points)):
        raise ValueError("Expected (T,3,2) physical Cartesian points and derivatives")
    keep = (t >= protocol["future_interval_s"][0] - 1e-10) & (t <= protocol["future_interval_s"][1] + 1e-10)
    selected = np.flatnonzero(keep)
    if len(selected) < 3:
        raise ValueError("Missing future evaluation interval")
    errors = {name: value[keep] for name, value in point_constraints(x, v, a, c).items()}
    errors["prediction"] = (x[keep] - reference_points[keep]).reshape(len(selected), 6)
    errors["dynamics"] = np.full((len(selected), 1), np.nan)
    energy = np.full(len(selected), np.nan)
    power = energy.copy()
    friction_power = energy.copy()
    decoded_q = np.full((len(selected), 3), np.nan)
    # The widest preregistered sensitivity threshold defines the interpretation
    # region. All six stricter/base gates still use unmodified raw measurements.
    readable = np.max(np.abs(errors["geometry"]), axis=1) <= 2 * protocol["physical_interpretation_gate_m"]
    numerical_failures = 0
    for row, i in enumerate(selected):
        if not readable[row] or not all(np.isfinite(values[i]).all() for values in (x, v, a)):
            continue
        decoded = physics.decode(x[i], v[i], a[i], c)
        if decoded is None:
            continue
        q, qv, qa = decoded
        try:
            frame = physics_frame(q, qv, torque, c)
            errors["dynamics"][row] = equation_residuals(frame, qv, qa)["dynamics"]
            energy[row] = mechanical_energy(q, qv, c)
            power[row] = (frame.actuation + frame.external + frame.friction) @ qv
            friction_power[row] = frame.friction @ qv
            decoded_q[row] = q
        except (ValueError, np.linalg.LinAlgError):
            numerical_failures += 1
    angle = np.unwrap(decoded_q[:, 0])
    force_position = (c["lengths_m"][0] * np.c_[np.cos(decoded_q[:, 0]), np.sin(decoded_q[:, 0])]
                      + c["force_application_along_coupler_m"] * np.c_[np.cos(decoded_q[:, 1]), np.sin(decoded_q[:, 1])])
    exact_work = torque * (angle - angle[0]) + (force_position - force_position[0]) @ np.asarray(c["external_force_xz_N"])
    dissipative_balance = energy_residual(t[keep], energy, friction_power)
    errors["energy"] = (dissipative_balance - exact_work)[:, None]
    expected = {name: len(values) for name, values in errors.items()}
    results = {}
    for multiplier in protocol["sensitivity_threshold_multipliers"]:
        results[str(multiplier)] = score_trajectory(errors, specifications(protocol, multiplier), expected,
                                                    protocol["profile_id"])
    main = results["1.0"]
    main.update(physical_interpretation_frames=int(readable.sum()), future_frames=len(selected),
                numerical_failures=numerical_failures,
                sensitivity_joint_pass={key: value["joint_pass"] for key, value in results.items()})
    if np.isfinite(energy).all():
        coarse = energy_residual(t[keep][::2], energy[::2], friction_power[::2]) - exact_work[::2]
        main["energy_quadrature_800_vs_1600_max_J"] = float(np.max(np.abs(coarse - errors["energy"][::2, 0])))
    else:
        main["energy_quadrature_800_vs_1600_max_J"] = None
    if derivative_audit:
        sampled_v, sampled_a, valid = differentiate(t, x.reshape(len(t), 6), window=5, degree=4)
        valid &= keep
        audit = {}
        for name, sampled, native in (("velocity", sampled_v, v), ("acceleration", sampled_a, a)):
            delta = sampled[valid] - native.reshape(len(t), 6)[valid]
            audit[name + "_rms_disagreement"] = float(np.sqrt(np.mean(delta**2))) if np.isfinite(delta).all() else None
        main["native_derivative_consistency"] = audit
    return main


def points_from_truth(truth, c):
    return tuple(np.asarray(values) for values in zip(*[
        physics.point_states(q, v, a, c) for q, v, a in zip(truth["q"], truth["qdot"], truth["qddot"])
    ]))


def macro_component_scores(rows):
    families = sorted({row["case_key"][0] for row in rows})
    result = {}
    for name in CORE_METRICS:
        by_family = []
        for family in families:
            family_rows = [row for row in rows if row["case_key"][0] == family]
            conditions = sorted({row["case_key"][1] for row in family_rows})
            by_family.append(np.mean([np.mean([row["metrics"][name]["passed"] is True
                                               for row in family_rows if row["case_key"][1] == condition])
                                      for condition in conditions]))
        result[name] = {"score_100": float(100 * np.mean(by_family)),
                        "measured_trajectories": sum(row["metrics"][name]["complete"] for row in rows),
                        "passed_trajectories": sum(row["metrics"][name]["passed"] is True for row in rows),
                        "expected_trajectories": len(rows)}
    return result
