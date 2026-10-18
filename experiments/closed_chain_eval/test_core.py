"""Critical scientific invariants, not benchmark evidence or model experiments."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from .core import (COMPARISON_FIELDS, CORE_METRICS, MetricSpec, PhysicsFrame,
                   aggregate, assert_comparable, differentiate, energy_residual,
                   equation_residuals, score_trajectory)
from .planar import mechanical_energy, physics, physics_frame


class PhysicalInvariants(unittest.TestCase):
    def test_reactions_cancel_but_tangent_acceleration_error_remains(self):
        M = np.diag([2., 3., 4.])
        J = np.array([[1., 1., 0.]])
        N = np.array([[1., 0.], [-1., 0.], [0., 1.]])
        a = np.array([2., -2., 3.])
        bias = np.array([.2, .4, .5])
        actuation = M @ a + bias - J.T[:, 0] * 1000.
        frame = PhysicsFrame(M, bias, actuation, np.zeros(3), np.zeros(3),
                             np.zeros(1), J, np.zeros(1), N)
        np.testing.assert_allclose(equation_residuals(frame, np.zeros(3), a)["dynamics"], 0., atol=1e-12)
        changed = equation_residuals(frame, np.zeros(3), a + .1 * N[:, 0])
        np.testing.assert_allclose(changed["dynamics"], [.5, 0.], atol=1e-12)
        frame.tangent = np.ones((3, 2))
        with self.assertRaises(ValueError):
            equation_residuals(frame, np.zeros(3), a)

    def test_signed_friction_power_and_all_prefix_energy(self):
        t = np.linspace(0., 1., 21)
        # 8 W motor input and -3 W friction produce +5 J in one second.
        np.testing.assert_allclose(energy_residual(t, 2. + 5. * t, np.full(21, 5.)), 0., atol=1e-14)
        wrong = energy_residual(t, 2. + 5. * t, np.full(21, 11.))
        self.assertAlmostEqual(wrong[-1], -6.)
        intermediate_error = np.sin(2 * np.pi * t)
        residual = energy_residual(t, 2. + 5. * t + intermediate_error, np.full(21, 5.))
        self.assertAlmostEqual(residual[-1], 0.)
        self.assertGreater(np.max(np.abs(residual)), .9)

    def test_common_derivatives_and_missing_stencil(self):
        t = np.linspace(0., 1., 21)**1.2
        q = np.column_stack([.2 + t**3, 2 * t**2 - t])
        v, a, interior = differentiate(t, q)
        np.testing.assert_allclose(v[interior], np.column_stack([3 * t**2, 4 * t - 1])[interior], atol=1e-12)
        np.testing.assert_allclose(a[interior], np.column_stack([6 * t, np.full(21, 4.)])[interior], atol=1e-11)
        self.assertFalse(interior[:3].any())
        self.assertFalse(interior[-3:].any())
        q[10] = np.nan
        v, a, _ = differentiate(t, q, periodic=(0,))
        self.assertTrue(np.isnan(v[7:14]).all())
        self.assertTrue(np.isfinite(v[14:18]).all())

    def test_planar_equations_and_energy_match_in_both_mechanisms(self):
        path = Path(__file__).resolve().parents[1] / "fourbar/config.json"
        source = json.loads(path.read_text())
        for slider in (False, True):
            c = copy.deepcopy(source)
            if slider:
                c["mechanism"] = "crank_slider"
                c["slider_mass_kg"] = .27
            theta, omega, tau = .7, 2., .5
            q = physics.inverse(theta, c)
            gamma, gamma_prime = physics.derivatives(q, c)
            alpha = physics.acceleration(theta, omega, tau, c)
            v, a = gamma * omega, gamma * alpha + gamma_prime * omega**2
            frame = physics_frame(q, v, tau, c)
            residuals = equation_residuals(frame, v, a)
            for residual in residuals.values():
                np.testing.assert_allclose(residual, 0., atol=2e-12)
            h = 1e-6
            rate = (mechanical_energy(q + h * v, v + h * a, c)
                    - mechanical_energy(q - h * v, v - h * a, c)) / (2 * h)
            power = (frame.actuation + frame.external + frame.friction) @ v
            self.assertAlmostEqual(rate, power, places=7)


def example_row(case_key, prediction_error=0., calibration="test_fixture_only"):
    specs = {name: MetricSpec(["unit"], [1.], limit=1.) for name in CORE_METRICS}
    errors = {name: np.zeros((4, 1)) for name in CORE_METRICS}
    errors["prediction"][:] = prediction_error
    row = score_trajectory(errors, specs, dict.fromkeys(CORE_METRICS, 4), calibration)
    row["case_key"] = case_key
    return row


class FairScoring(unittest.TestCase):
    def test_no_uncalibrated_score_and_no_compensation(self):
        self.assertIsNone(example_row(("4R", "nominal", 1), calibration=None)["joint_pass"])
        row = example_row(("4R", "nominal", 1), prediction_error=2.)
        self.assertFalse(row["joint_pass"])
        self.assertTrue(row["metrics"]["dynamics"]["passed"])

    def test_nan_missing_and_short_submissions_are_not_zero_error(self):
        specs = {name: MetricSpec(["unit"], [1.], limit=1.) for name in CORE_METRICS}
        errors = {name: np.zeros((4, 1)) for name in CORE_METRICS}
        errors["dynamics"][2] = np.nan
        errors["energy"] = None
        row = score_trajectory(errors, specs, dict.fromkeys(CORE_METRICS, 4), "test_fixture_only")
        self.assertFalse(row["joint_pass"])
        self.assertIsNone(row["metrics"]["dynamics"]["value"])
        self.assertEqual(row["metrics"]["dynamics"]["finite_samples"], 3)
        self.assertEqual(row["metrics"]["energy"]["status"], "unavailable")
        errors["dynamics"] = np.zeros((3, 1))
        with self.assertRaises(ValueError):
            score_trajectory(errors, specs, dict.fromkeys(CORE_METRICS, 4))

    def test_missing_cases_and_unequal_family_sizes_cannot_improve_score(self):
        expected = [("4R", "nominal", i) for i in range(3)] + [("3R1P", "nominal", 0)]
        rows = [example_row(key) for key in expected[:3]]
        result = aggregate(rows, expected, "test_fixture_only")
        self.assertEqual(result["verified_score_100"], 50.)
        self.assertEqual(result["macro_coverage"], .5)
        self.assertEqual(result["submitted_cases"], 3)
        with self.assertRaises(ValueError):
            aggregate(rows + rows[:1], expected, "test_fixture_only")
        rows[0]["calibration_id"] = "different"
        with self.assertRaises(ValueError):
            aggregate(rows, expected, "test_fixture_only")

    def test_input_and_derivative_tracks_cannot_be_pooled(self):
        first = dict.fromkeys(COMPARISON_FIELDS, "fixed")
        second = first.copy()
        assert_comparable([first, second])
        with self.assertRaises(ValueError):
            assert_comparable([dict.fromkeys(COMPARISON_FIELDS, None)])
        for key in ("input_track", "derivative_policy_id", "reader_id", "calibration_id"):
            changed = second.copy()
            changed[key] = "different"
            with self.assertRaises(ValueError):
                assert_comparable([first, changed])


if __name__ == "__main__":
    unittest.main()
