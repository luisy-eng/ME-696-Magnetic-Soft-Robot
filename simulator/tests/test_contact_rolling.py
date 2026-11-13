"""Step 2D geometry checks; do not substitute for physical validation."""
import unittest
import numpy as np

from simulator.models.contact_rolling import (
    RollingSettings, normalize_centerline, rotate_xz, roll_kinematically,
)
from simulator.models.deformation import SheetParameters, solve_equilibrium


class ContactRollingTests(unittest.TestCase):
    def test_rotation_preserves_distances(self):
        p = np.array([[-1.,0.], [0., 1.], [2.,-1.]])
        rot = rotate_xz(p, np.pi/3)
        np.testing.assert_allclose(np.linalg.norm(np.diff(rot,axis=0), axis=1),
                                   np.linalg.norm(np.diff(p,axis=0), axis=1), atol=1e-12)

    def test_contact_enforces_floor_and_thickness(self):
        p = np.array([[-.002, 0.], [0., .001], [.002, 0.]])
        s = RollingSettings(n_steps=105, thickness_m=.00018)
        r = roll_kinematically(p, s)
        self.assertGreaterEqual(np.min(r['poses_xz_m'][:,:,1]), s.thickness_m/2-1e-12)
        np.testing.assert_allclose(r['contact_xz_m'][:,1], s.thickness_m/2, atol=1e-12)

    def test_no_rotation_no_horizontal_travel(self):
        p = np.array([[-.001, 0.], [0., .001], [.001, 0.]])
        r = roll_kinematically(p, RollingSettings(rotation_deg=0, n_steps=10))
        np.testing.assert_allclose(r['x_center_m'], 0, atol=1e-15)

    def test_circle_predicts_one_circumference_with_sign(self):
        radius_m = 1e-3
        u = np.linspace(0, 2*np.pi, 361)
        circle = np.column_stack((radius_m * np.cos(u), radius_m * np.sin(u)))
        out = roll_kinematically(circle, RollingSettings(rotation_deg=360, n_steps=721, thickness_m=0))
        self.assertAlmostEqual(out['travel_mm'], -2*np.pi*radius_m*1000, delta=0.002)

    def test_zero_spin_uses_step2b_equilibrium(self):
        p = SheetParameters(n_segments=8)
        eq = solve_equilibrium(p, distance_mm=60., angle_deg=0.)
        profile = normalize_centerline(eq['nodes_m'])
        r = roll_kinematically(profile, RollingSettings(rotation_deg=0, n_steps=2))
        self.assertEqual(r['poses_xz_m'].shape, (2, p.n_segments+1, 2))
        self.assertAlmostEqual(r['travel_mm'], 0.0, places=12)


if __name__ == '__main__':
    unittest.main()
