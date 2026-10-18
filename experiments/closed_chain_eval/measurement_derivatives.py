"""Coordinate-aware differentiation without enforcing rigid link lengths.

Polar radii are measured, not fixed to mechanism constants. All three measured
points and their closure errors remain unchanged. This removes the need for a
polynomial in Cartesian coordinates to approximate rapid rigid rotation.
"""
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter


def scalar_derivatives(t, values, method):
    values = np.asarray(values, dtype=float)
    if method.startswith("spline"):
        curve = make_interp_spline(t, values, k=int(method[6:]), axis=0)
        return curve.derivative(1)(t), curve.derivative(2)(t)
    window, degree = map(int, method[2:].split("d"))
    if not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-5, atol=1e-8):
        raise ValueError("SG measurement configuration requires uniform timestamps")
    return tuple(savgol_filter(values, window, degree, deriv=order,
                              delta=float(np.diff(t).mean()), axis=0) for order in (1, 2))


def planar_derivatives(t, points, config, method="polar_sg11d9"):
    chart, derivative = method.split("_", 1)
    x = np.asarray(points, dtype=float)
    if chart == "cartesian":
        return scalar_derivatives(t, x, derivative)
    if chart != "polar":
        raise ValueError("Unknown coordinate chart")
    origin = np.array([0., config["pivot_height_m"]])

    def polar(vector):
        radius = np.linalg.norm(vector, axis=1)
        if np.any(radius < 1e-8):
            raise ValueError("Polar measurement chart singular")
        angle = np.unwrap(np.arctan2(vector[:, 1], vector[:, 0]))
        values = np.column_stack((radius, angle))
        v, a = scalar_derivatives(t, values, derivative)
        radial = vector / radius[:, None]
        tangent = np.column_stack((-radial[:, 1], radial[:, 0]))
        velocity = v[:, :1] * radial + (radius * v[:, 1])[:, None] * tangent
        acceleration = (a[:, 0] - radius * v[:, 1] ** 2)[:, None] * radial
        acceleration += (2 * v[:, 0] * v[:, 1] + radius * a[:, 1])[:, None] * tangent
        return velocity, acceleration

    vb, ab = polar(x[:, 0] - origin)
    if config["category"] == "S-B":
        vc, ac = scalar_derivatives(t, x[:, 1], derivative)
    else:
        vc, ac = polar(x[:, 1] - origin - [config["ground_length_m"], 0.])
    # P is also measured independently; its offset from the midpoint is kept.
    residual = x[:, 2] - (x[:, 0] + x[:, 1]) / 2
    vr, ar = scalar_derivatives(t, residual, derivative)
    return np.stack((vb, vc, (vb + vc) / 2 + vr), axis=1), np.stack((ab, ac, (ab + ac) / 2 + ar), axis=1)


def smooth_points(points, window=1, degree=3):
    """Declared temporal measurement filter; no geometric projection."""
    if window == 1:
        return np.asarray(points, dtype=float).copy()
    return savgol_filter(points, window, degree, axis=0)
