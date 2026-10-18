import unittest

import numpy as np

from .visual_calibration import partition, error_statistics
from .summarize_visual_calibration import empirical_bound, verify_bound
from .measurement_derivatives import planar_derivatives


class CalibrationTests(unittest.TestCase):
    def test_partition_keeps_conditions_together_and_rejects_test_fit(self):
        rows = [dict(assembly_id="A%d" % i, category="A", split="calibration", condition=c)
                for i in range(6) for c in ("low", "rated", "load")]
        split = partition(rows)
        roles = {}
        for row in split:
            roles.setdefault(row["assembly_id"], set()).add(row["calibration_role"])
        self.assertTrue(all(len(r) == 1 for r in roles.values()))
        self.assertEqual(sum(r["calibration_role"] == "verification" for r in split), 9)
        with self.assertRaises(ValueError):
            partition([dict(r, split="test") for r in rows])

    def test_bias_variance_and_temporal_correlation_are_separate(self):
        t = np.arange(100)
        error = np.stack((.01 + .002*np.sin(t/10), -.02 + .001*np.cos(t/10)), -1)[:, None]
        stats = error_statistics(error)
        self.assertAlmostEqual(stats["coordinate_rmse"]**2,
                               stats["coordinate_bias_rms"]**2 + stats["centered_coordinate_rmse"]**2, places=12)
        self.assertGreater(np.trace(stats["lag1_cross_covariance"]), 0.)
        self.assertGreater(stats["coordinate_bias_rms"], 10*stats["centered_coordinate_rmse"])

    def test_verification_missing_is_not_dropped_and_cannot_change_bound(self):
        bound = empirical_bound([.001, .002, .003])
        rows = [dict(assembly_id="A", error=.002), dict(assembly_id="A", error=None),
                dict(assembly_id="B", error=.1)]
        result = verify_bound(rows, lambda r: r["error"], bound)
        self.assertEqual(bound, .003)
        self.assertEqual((result["passed"], result["available"], result["total"]), (1, 2, 3))
        self.assertEqual(result["assemblies_passed"], 0)

    def test_curvilinear_derivatives_preserve_radial_and_midpoint_defects(self):
        t = np.arange(180)/30
        angle, omega = .7*t, .7
        r = .15 + .003*t
        e = np.column_stack((np.cos(angle), np.sin(angle)))
        n = np.column_stack((-np.sin(angle), np.cos(angle)))
        b = r[:, None]*e
        c = np.column_stack((.4 + .02*t, .1 + .01*t))
        offset = np.column_stack((.002*t**2, np.full(len(t), .007)))
        p = (b+c)/2 + offset
        x = np.stack((b,c,p),1)
        v, a = planar_derivatives(t,x,dict(category="S-B",pivot_height_m=0.),"polar_sg7d5")
        expected_v = .003*e + r[:,None]*omega*n
        expected_a = 2*.003*omega*n - r[:,None]*omega**2*e
        np.testing.assert_allclose(v[5:-5,0],expected_v[5:-5],atol=1e-10)
        np.testing.assert_allclose(a[5:-5,0],expected_a[5:-5],atol=1e-9)
        np.testing.assert_allclose(a[5:-5,2] - (a[5:-5,0]+a[5:-5,1])/2,
                                   np.tile([.004,0.], (170,1)), atol=1e-9)
        self.assertGreater(np.ptp(np.linalg.norm(x[:,0],axis=1)), .01)
        self.assertAlmostEqual((x[:,2] - (x[:,0]+x[:,1])/2)[0,1], .007)


if __name__ == "__main__":
    unittest.main()
