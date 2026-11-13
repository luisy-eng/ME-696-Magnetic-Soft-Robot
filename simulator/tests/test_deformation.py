"""Numerical / geometric sanity checks for the preliminary sheet model."""
import unittest
import numpy as np

from simulator.models.deformation import (
    SheetParameters, centerline, magnetization_moments,
    energy_joules, solve_equilibrium
)
from simulator.models.magnetic_field import magnet_moment


class DeformationTests(unittest.TestCase):
    def setUp(self):
        self.p = SheetParameters(n_segments=8)
        self.center = np.array([0., 0., 0.060])

    def test_straight_chain_has_correct_length(self):
        theta = np.zeros(self.p.n_segments)
        nodes, mid = centerline(theta, self.p, self.center)
        np.testing.assert_allclose(np.linalg.norm(np.diff(nodes, axis=0), axis=1),
                                   self.p.ds, atol=1e-15)
        np.testing.assert_allclose(mid.mean(axis=0), self.center, atol=1e-15)
        self.assertAlmostEqual(np.linalg.norm(nodes[-1]-nodes[0]), self.p.length_m)

    def test_moment_magnitudes(self):
        moments = magnetization_moments(np.zeros(self.p.n_segments), self.p)
        expected = (self.p.magnetization_a_per_m * self.p.width_m *
                    self.p.thickness_m * self.p.ds)
        np.testing.assert_allclose(np.linalg.norm(moments, axis=1), expected)

    def test_no_field_elastic_energy_prefers_straight_chain(self):
        flat = np.zeros(self.p.n_segments)
        bent = np.linspace(-0.5, 0.5, self.p.n_segments)
        zero_field = np.zeros(3)
        flat_E = energy_joules(flat, self.p, self.center, np.zeros(3), zero_field)
        bent_E = energy_joules(bent, self.p, self.center, np.zeros(3), zero_field)
        self.assertAlmostEqual(flat_E, 0.)
        self.assertGreater(bent_E, flat_E)

    def test_solver_outputs_finite_connected_shape(self):
        result = solve_equilibrium(self.p, distance_mm=60, angle_deg=0)
        self.assertTrue(result['optimization_success'])
        self.assertTrue(np.all(np.isfinite(result['nodes_m'])))
        np.testing.assert_allclose(np.linalg.norm(np.diff(result['nodes_m'], axis=0), axis=1),
                                   self.p.ds, rtol=1e-10)
        self.assertLessEqual(result['end_to_end_mm'], self.p.length_m*1000+1e-10)


if __name__ == '__main__':
    unittest.main()
