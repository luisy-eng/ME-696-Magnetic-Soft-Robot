"""Small physical sanity checks for the initial dipole model."""
import unittest
import numpy as np
from simulator.models.magnetic_field import (
    MU0, equivalent_dipole_moment, dipole_field, magnet_moment
)


class MagneticFieldTests(unittest.TestCase):
    def test_on_axis_formula(self):
        z = 0.06
        m = magnet_moment()
        B = dipole_field([0, 0, z], moment_A_m2=m)
        expected = MU0 / (4*np.pi) * 2 * np.linalg.norm(m) / z**3
        self.assertAlmostEqual(B[2], expected, places=12)
        self.assertAlmostEqual(B[0], 0., places=12)

    def test_inverse_cube_decay(self):
        m = magnet_moment()
        Bnear = np.linalg.norm(dipole_field([0, 0, 0.06], moment_A_m2=m))
        Bfar = np.linalg.norm(dipole_field([0, 0, 0.12], moment_A_m2=m))
        self.assertAlmostEqual(Bnear / Bfar, 8., places=11)

    def test_field_reverses_if_magnet_flips(self):
        p = [0.012, 0.0, 0.06]
        B1 = dipole_field(p, moment_A_m2=magnet_moment(0))
        B2 = dipole_field(p, moment_A_m2=magnet_moment(180))
        np.testing.assert_allclose(B1, -B2, atol=1e-15)

    def test_center_is_singular(self):
        with self.assertRaises(ValueError):
            dipole_field([0, 0, 0], moment_A_m2=magnet_moment())

    def test_positive_moment(self):
        self.assertGreater(equivalent_dipole_moment(), 0)


if __name__ == '__main__':
    unittest.main()
