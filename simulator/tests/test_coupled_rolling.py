"""Step 2F numerical sanity checks; NOT experimental accuracy tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from simulator.models.deformation import SheetParameters
from simulator.models.coupled_rolling import (
    CoupledSettings, constrained_potential_j, ground_shape,
    orientations_from_joints, relax_shape, simulate_coupled_rolling,
    save_results,
)


class CoupledRollingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = SheetParameters(n_segments=6)
        cls.settings = CoupledSettings(n_steps=15, duration_s=4.,
                                       solver_maxiter=110)

    def test_joint_to_body_orientation_is_not_double_counted(self):
        q = np.array([.1, -.2, .3, -.1, .2])
        phi = .45
        theta = orientations_from_joints(q, phi)
        np.testing.assert_allclose(np.diff(theta), q, atol=1e-13)
        self.assertAlmostEqual(theta.mean(), phi, delta=1e-13)

    def test_ground_placement_respects_half_thickness_clearance(self):
        q = np.array([.1, -.2, .3, -.1, .2])
        for phi in [-2., 0., .3, 2.2]:
            s = ground_shape(q, phi, self.p, center_x_m=.001)
            self.assertAlmostEqual(float(s['nodes_m'][:,2].min()),
                                   self.p.thickness_m/2, delta=1e-12)

    def test_relaxation_decreases_potential(self):
        q = np.zeros(self.p.n_segments-1)
        u0 = constrained_potential_j(q, 0., 0., self.p, self.settings)
        solved = relax_shape(self.p,self.settings,0.,0.,seed=q)
        self.assertLessEqual(solved['energy_j'], u0+1e-12)
        self.assertTrue(np.isfinite(solved['energy_j']))

    def test_no_magnet_no_gravity_gives_no_body_motion(self):
        p = SheetParameters(n_segments=6, magnetization_a_per_m=0)
        s = CoupledSettings(n_steps=9,duration_s=2.,remanence_t=0.,
                            density_kg_m3=0.)
        result = simulate_coupled_rolling(p,s)
        np.testing.assert_allclose(result['body_angle_deg'],0.,atol=1e-9)
        np.testing.assert_allclose(result['end_to_end_mm'],p.length_m*1000,atol=1e-6)
        self.assertAlmostEqual(result['travel_mm'],0.,delta=1e-9)

    def test_shape_is_recomputed_and_can_change_during_drive(self):
        result = simulate_coupled_rolling(self.p,self.settings)
        self.assertGreater(float(np.ptp(result['end_to_end_mm'])),0.03)
        self.assertGreater(float(np.ptp(result['joints_rad'],axis=0).max()),.01)
        self.assertGreaterEqual(float(result['poses_xz_m'][:,:,1].min()),
                                self.p.thickness_m/2-5e-9)
        self.assertGreater(abs(result['magnet_angle_deg'][-1]-result['body_angle_deg'][-1]),1.)

    def test_save_csv_has_one_row_per_step(self):
        r = simulate_coupled_rolling(self.p, CoupledSettings(n_steps=9, duration_s=2.))
        with TemporaryDirectory() as td:
            path = save_results(r, Path(td))
            self.assertEqual(len(path.read_text().splitlines()),10)
            self.assertTrue((Path(td)/'coupled_summary.png').exists())


if __name__=='__main__':
    unittest.main()
