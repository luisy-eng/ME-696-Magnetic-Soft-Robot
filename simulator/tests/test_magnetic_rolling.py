"""Step 2E tests. Numerical/physical sanity checks, NOT experimental validation."""
import unittest

import numpy as np

from simulator.models.contact_rolling import RollingSettings, normalize_centerline, roll_kinematically
from simulator.models.deformation import SheetParameters, solve_equilibrium
from simulator.models.magnetic_rolling import (
    DriveSettings, build_no_slip_lookup, floor_pose, potential_energy_j,
    generalized_torque_n_m, overdamped_rate_rad_s, simulate_magnetic_rolling,
)


class MagneticRollingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = SheetParameters(n_segments=8)
        cls.eq = solve_equilibrium(cls.p, 60, 0)
        cls.shape = normalize_centerline(cls.eq['nodes_m'])
        cls.lookup = build_no_slip_lookup(cls.shape, cls.p.thickness_m, 9.5, samples=6001)

    def test_floor_enforced_with_both_rotation_signs(self):
        for q in (-3.5, -1.4, 0, .8, 4.7):
            with self.subTest(q=q):
                pose, _ = floor_pose(q, self.shape, self.lookup)
                self.assertAlmostEqual(float(np.min(pose[:, 1])), self.p.thickness_m/2,
                                       delta=2e-8)

    def test_step2d_kinematic_displacement_agrees(self):
        r = roll_kinematically(self.shape,
                              RollingSettings(rotation_deg=190, n_steps=951,
                                              thickness_m=self.p.thickness_m))
        _, center = floor_pose(np.deg2rad(190), self.shape, self.lookup)
        self.assertAlmostEqual(center[0], r['x_center_m'][-1], delta=4e-6)

    def test_flip_magnet_reverses_torque_without_gravity(self):
        settings = DriveSettings(density_kg_m3=0)
        a = generalized_torque_n_m(.33, 47., self.shape, self.eq['theta_rad'],
                                   self.p, settings, self.lookup)
        b = generalized_torque_n_m(.33, 227., self.shape, self.eq['theta_rad'],
                                   self.p, settings, self.lookup)
        self.assertGreater(abs(a),1e-11)
        self.assertAlmostEqual(a, -b, delta=1e-12)

    def test_stronger_magnet_gives_proportional_torque_no_gravity(self):
        base = DriveSettings(density_kg_m3=0, remanence_t=.7)
        doubled = DriveSettings(density_kg_m3=0, remanence_t=1.4)
        a = generalized_torque_n_m(.24, 30, self.shape, self.eq['theta_rad'],
                                   self.p, base, self.lookup)
        b = generalized_torque_n_m(.24, 30, self.shape, self.eq['theta_rad'],
                                   self.p, doubled, self.lookup)
        self.assertAlmostEqual(b, 2*a, delta=5e-13)

    def test_friction_threshold_prevents_motion_below_threshold(self):
        settings = DriveSettings(resistance_n_m=1e-6)
        self.assertEqual(overdamped_rate_rad_s(2e-7, settings), 0.)
        self.assertLess(overdamped_rate_rad_s(-2e-6, settings), 0.)

    def test_no_magnetic_or_gravity_means_no_motion(self):
        p = SheetParameters(n_segments=8, magnetization_a_per_m=0)
        s = DriveSettings(magnet_turn_deg=360, n_steps=26, duration_s=3,
                          density_kg_m3=0, remanence_t=0)
        r = simulate_magnetic_rolling(p,s)
        np.testing.assert_allclose(r['body_angle_deg'], 0, atol=1e-10)
        self.assertAlmostEqual(r['travel_mm'], 0, delta=1e-10)

    def test_magnetic_drive_is_not_prescribed_body_angle(self):
        s = DriveSettings(magnet_turn_deg=180, duration_s=4, n_steps=101,
                          density_kg_m3=1500)
        r = simulate_magnetic_rolling(self.p, s)
        self.assertEqual(len(r['time_s']),101)
        self.assertGreater(abs(r['body_angle_deg'][-1]), 1.)
        self.assertGreater(abs(r['magnet_angle_deg'][-1]-r['body_angle_deg'][-1]), 0.1)
        self.assertGreaterEqual(np.min(r['poses_xz_m'][:,:,1]),self.p.thickness_m/2-2e-8)

    def test_high_resistance_prevents_rotation(self):
        s = DriveSettings(magnet_turn_deg=360, duration_s=3, n_steps=61,
                          resistance_n_m=1e-5)
        r = simulate_magnetic_rolling(self.p, s)
        self.assertAlmostEqual(r['body_angle_deg'][-1],0,delta=1e-12)
        self.assertAlmostEqual(r['travel_mm'],0,delta=1e-12)


if __name__ == '__main__':
    unittest.main()
