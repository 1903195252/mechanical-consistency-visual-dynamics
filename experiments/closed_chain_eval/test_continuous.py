"""Scientific scoring invariants; these checks are not model experiments."""
import copy
import unittest

import numpy as np

from .core import CORE_METRICS, MetricSpec, score_trajectory
from .continuous import (aggregate_continuous, calibrate_noise_profile,
                         continuous_scorecard, pixel_span, propagate_covariance)


def card(error=0., missing=(), scale=1., key=("family", "condition", "case")):
    specs = {name: MetricSpec(["m"], [scale], limit=1.) for name in CORE_METRICS}
    errors = {name: None if name in missing else [error] for name in CORE_METRICS}
    result = score_trajectory(errors, specs, dict.fromkeys(CORE_METRICS, 1), "fixture")
    result["case_key"] = key
    return result


class ContinuousScoringInvariants(unittest.TestCase):
    def test_orders_failures_without_moving_the_acceptance_boundary(self):
        rows = [continuous_scorecard(card(value)) for value in (0., .5, 1., 2., 20., 1e200)]
        scores = [r["continuous"]["score_100"] for r in rows]
        self.assertTrue(all(a > b for a, b in zip(scores, scores[1:])))
        self.assertEqual(scores[0], 100.)
        self.assertEqual(scores[2], 50.)
        self.assertTrue(rows[2]["joint_pass"])
        self.assertFalse(rows[3]["joint_pass"])

    def test_unavailable_physics_is_not_renormalized_away(self):
        scored = continuous_scorecard(card(missing=("dynamics", "energy")))
        branch = scored["continuous"]
        self.assertAlmostEqual(branch["score_100"], 100 * 4 / 6)
        self.assertIsNone(branch["metrics"]["dynamics"]["ratio"])
        self.assertFalse(scored["all_metrics_available"])
        expected = [("a", "c", "1"), ("a", "c", "2"), ("b", "c", "3")]
        rows = [continuous_scorecard(card(key=expected[0])), continuous_scorecard(card(key=expected[2]))]
        result = aggregate_continuous(rows, expected, calibration_id="fixture")
        self.assertEqual(result["continuous"]["score_100"], 75.)
        self.assertEqual(result["expected_cases"], 3)

    def test_units_do_not_change_scores(self):
        metres = continuous_scorecard(card(error=.02, scale=.001))
        millimetre_card = card(error=20., scale=1.)
        for metric in millimetre_card["metrics"].values():
            metric["units"] = ["mm"]
        millimetres = continuous_scorecard(millimetre_card)
        self.assertEqual(metres["continuous"]["score_100"],
                         millimetres["continuous"]["score_100"])

    def test_measurement_profile_never_creates_a_second_scoring_standard(self):
        profile = calibrate_noise_profile([card(2.), card(3.)], profile_id="camera",
            measurement_id="reader", source_id="reference-video", split="calibration")
        source = card(2.)
        original = copy.deepcopy(source)
        scored = continuous_scorecard(source, noise_profile=profile, measurement_id="reader")
        self.assertEqual(scored["continuous"]["score_100"], continuous_scorecard(source)["continuous"]["score_100"])
        self.assertNotIn("branches", scored["continuous"])
        self.assertTrue(scored["continuous"]["measurement_diagnostics"]["geometry"]["exceeds_engineering_tolerance"])
        self.assertFalse(scored["joint_pass"])
        self.assertEqual(source, original)
        with self.assertRaises(ValueError):
            continuous_scorecard(source, noise_profile=profile, measurement_id="other-reader")
        with self.assertRaises(ValueError):
            calibrate_noise_profile([source], profile_id="p", measurement_id="r", source_id="s", split="test")
        with self.assertRaises(ValueError):
            calibrate_noise_profile([card(missing=("energy",))], profile_id="p",
                measurement_id="r", source_id="s", split="calibration")
        other = copy.deepcopy(profile)
        other["metrics"]["geometry"]["units"] = ["mm"]
        with self.assertRaises(ValueError):
            continuous_scorecard(source, noise_profile=other, measurement_id="reader")

    def test_different_noise_profiles_cannot_be_pooled(self):
        profile = calibrate_noise_profile([card(2.)], profile_id="camera",
            measurement_id="reader", source_id="reference", split="validation")
        a = continuous_scorecard(card(key=("f", "c", "1")), noise_profile=profile, measurement_id="reader")
        b = continuous_scorecard(card(key=("f", "c", "2")))
        with self.assertRaises(ValueError):
            aggregate_continuous([a, b], [a["case_key"], b["case_key"]], calibration_id="fixture")

    def test_pixel_scale_is_not_a_closure_uncertainty(self):
        self.assertAlmostEqual(pixel_span(1.30, 64, convention="pixel_edges"), .0203125)
        self.assertAlmostEqual(pixel_span(1.30, 64, convention="pixel_centers"), 1.30 / 63)
        sigma = .002
        closure_jacobian = [[-1., 1.]]
        independent = propagate_covariance(closure_jacobian, np.eye(2) * sigma**2)
        common_translation = propagate_covariance(closure_jacobian, np.ones((2, 2)) * sigma**2)
        self.assertAlmostEqual(independent[0, 0], 2 * sigma**2)
        self.assertAlmostEqual(common_translation[0, 0], 0.)

    def test_derivative_noise_depends_on_sampling_and_correlation(self):
        dt, sigma = .1, .002
        C = np.eye(3) * sigma**2
        J = np.array([[0., -1 / dt, 1 / dt], [1 / dt**2, -2 / dt**2, 1 / dt**2]])
        covariance = propagate_covariance(J, C)
        self.assertAlmostEqual(covariance[0, 0], 2 * sigma**2 / dt**2)
        self.assertAlmostEqual(covariance[1, 1], 6 * sigma**2 / dt**4)
        with self.assertRaises(ValueError):
            propagate_covariance([[1., 1.]], [[1., 2.], [2., 1.]])


if __name__ == "__main__":
    unittest.main()
