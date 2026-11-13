"""ME696 Step 2B: illustrative 2-D reduced-order magneto-elastic sheet model.

Predict planar equilibrium shapes using discrete elastic joints and a harmonic
magnetization profile. Reuses the validated dipole *equation* from Step 2A.

IMPORTANT:
- Preliminary model, NOT validated with the supplied physical sample.
- Paper-reference geometry; assumed elastic modulus and magnetization.
- Center of sheet is constrained to the indicated position. No ground/contact,
  gravity, rolling, hydrodynamics, or collision/finite-size cube magnet model.
- Magnetization is a convenient planar harmonic *hypothesis* until the specific
  sheet's programmed magnetization and Hu Supplementary Info are verified.
- Input distances must be in the regime where a dipole is a sensible start.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from .magnetic_field import (
    SIDE_M, DEFAULT_BR_T, HU_SHEET_LENGTH_M, dipole_field, magnet_moment
)


@dataclass(frozen=True)
class SheetParameters:
    length_m: float = HU_SHEET_LENGTH_M       # Figure 1 of Hu et al., 2018
    width_m: float = 1.5e-3                   # Figure 1, reference geometry
    thickness_m: float = 185e-6               # Figure 1, reference geometry
    youngs_modulus_pa: float = 100_000.0     # ILLUSTRATIVE, not measured
    magnetization_a_per_m: float = 60_000.0  # ILLUSTRATIVE, not measured
    profile_phase_deg: float = 0.0           # ILLUSTRATIVE, must be verified
    n_segments: int = 16

    def __post_init__(self):
        if self.n_segments < 3:
            raise ValueError('n_segments must be at least 3')
        if min(self.length_m, self.width_m, self.thickness_m,
               self.youngs_modulus_pa) <= 0:
            raise ValueError('geometry and elastic modulus must be positive')
        if self.magnetization_a_per_m < 0:
            raise ValueError('magnetization must be nonnegative')

    @property
    def ds(self) -> float:
        return self.length_m / self.n_segments

    @property
    def flexural_rigidity(self) -> float:
        # Rectangular cross section bending about the out-of-plane y axis.
        return self.youngs_modulus_pa * self.width_m * self.thickness_m**3 / 12.0


def centerline(theta_rad: np.ndarray, p: SheetParameters, center_xyz_m) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes (N+1,3), segment centers (N,3) for planar x-z chain.

    Segment angles are measured from +x toward +z. Translation is adjusted
    so the mean of segment centers equals center_xyz_m.
    """
    theta_rad = np.asarray(theta_rad, dtype=float)
    if theta_rad.shape != (p.n_segments,):
        raise ValueError(f'theta must contain {p.n_segments} segment angles')
    center = np.asarray(center_xyz_m, dtype=float)
    if center.shape != (3,):
        raise ValueError('center_xyz_m must be a 3-vector')
    increments = np.column_stack((p.ds*np.cos(theta_rad),
                                  np.zeros(p.n_segments),
                                  p.ds*np.sin(theta_rad)))
    nodes = np.vstack((np.zeros(3), np.cumsum(increments, axis=0)))
    midpoints = 0.5*(nodes[1:] + nodes[:-1])
    offset = center - midpoints.mean(axis=0)
    return nodes + offset, midpoints + offset


def magnetization_moments(theta_rad: np.ndarray, p: SheetParameters) -> np.ndarray:
    """Magnetic moment [A*m^2] on every segment, planar one-cycle profile.

    The profile direction is defined in each segment's material frame.
    The magnetization density is a *hypothesized* parameter, NOT paper data.
    """
    s = (np.arange(p.n_segments) + 0.5) / p.n_segments
    beta = 2*np.pi*s + np.deg2rad(p.profile_phase_deg)
    angle = theta_rad + beta
    moment_per_segment = (p.magnetization_a_per_m * p.width_m *
                          p.thickness_m * p.ds)
    return moment_per_segment*np.column_stack((np.cos(angle),
                                               np.zeros_like(angle),
                                               np.sin(angle)))


def energy_joules(theta_rad: np.ndarray, p: SheetParameters,
                  sheet_center_m: np.ndarray, magnet_center_m: np.ndarray,
                  magnet_moment_a_m2: np.ndarray) -> float:
    """Elastic joint bending energy minus local magnetic dipole energies."""
    theta_rad = np.asarray(theta_rad, dtype=float)
    dtheta = np.diff(theta_rad)
    elastic = 0.5*(p.flexural_rigidity/p.ds)*np.sum(dtheta*dtheta)
    _, midpoints = centerline(theta_rad, p, sheet_center_m)
    B = dipole_field(midpoints, magnet_center_m, magnet_moment_a_m2)
    moments = magnetization_moments(theta_rad, p)
    magnetic = -np.sum(moments * B)
    return float(elastic + magnetic)


def solve_equilibrium(p: SheetParameters, distance_mm: float = 60.0,
                      angle_deg: float = 0.0,
                      br_t: float = DEFAULT_BR_T) -> dict:
    """Minimize planar magneto-elastic potential from multiple initial shapes.

    Returned shape is a local energy minimum. Multiple seeds reduce—but do
    not eliminate—the possibility of missing a lower-energy equilibrium.
    """
    if distance_mm <= 0:
        raise ValueError('distance_mm must be positive')
    magnet_center_m = np.zeros(3)
    sheet_center_m = np.array([0.0, 0.0, distance_mm/1000.0])
    moment = magnet_moment(angle_deg, br_t)
    u = (np.arange(p.n_segments) + 0.5)/p.n_segments
    seeds = [
        np.zeros(p.n_segments),
        np.ones(p.n_segments)*np.pi/2,
        0.7*np.sin(2*np.pi*u),
        -0.7*np.sin(2*np.pi*u),
        1.2*(u-0.5),
        -1.2*(u-0.5),
        2*np.pi*(u-0.5),
        -2*np.pi*(u-0.5),
    ]
    # Scale from Joules to nanojoules for a well-conditioned optimizer.
    objective = lambda x: 1e9*energy_joules(x, p, sheet_center_m,
                                             magnet_center_m, moment)
    solutions = [minimize(objective, seed, method='L-BFGS-B',
                          options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
                 for seed in seeds]
    converged = [s for s in solutions if s.success]
    if not converged:
        raise RuntimeError('No equilibrium candidate converged')
    best = min(converged, key=lambda r: r.fun)
    nodes, midpoints = centerline(best.x, p, sheet_center_m)
    fields = dipole_field(midpoints, magnet_center_m, moment)
    return {
        'theta_rad': best.x,
        'nodes_m': nodes,
        'segment_centers_m': midpoints,
        'B_T': fields,
        'total_energy_J': float(best.fun*1e-9),
        'end_to_end_mm': float(1000*np.linalg.norm(nodes[-1]-nodes[0])),
        'max_joint_angle_deg': float(np.degrees(np.max(np.abs(np.diff(best.x))))),
        'optimization_success': bool(best.success),
        'distance_mm': float(distance_mm),
        'angle_deg': float(angle_deg),
    }


def plot_equilibrium(p: SheetParameters, solution: dict, out_path: Path) -> None:
    nodes = solution['nodes_m']*1000
    x, z = nodes[:, 0], nodes[:, 2]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    # Compare with a straight sheet centered at the same location.
    d = solution['distance_mm']
    ax.plot([-p.length_m*500, p.length_m*500], [d, d],
            ls='--', lw=1.8, color='0.5', label='Undeformed reference')
    ax.plot(x, z, '-o', lw=2.1, ms=3.2, label='Predicted equilibrium')
    ax.scatter([0], [d], s=36, marker='x', color='black', label='Prescribed sheet centroid')
    ax.set(xlabel='x (mm)', ylabel='z (mm)',
           title=f'Hypothetical sheet equilibrium | magnet angle={solution["angle_deg"]:g}°')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.text(0.12, 0.02,
             f'End-to-end: {solution["end_to_end_mm"]:.2f} mm. Assumed E, magnetization and phase; not validated.',
             fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Preliminary planar magnetic sheet model')
    parser.add_argument('--distance-mm', type=float, default=60.)
    parser.add_argument('--angle-deg', type=float, default=0.)
    parser.add_argument('--segments', type=int, default=16)
    parser.add_argument('--modulus-kpa', type=float, default=100.)
    parser.add_argument('--magnetization-ka-m', type=float, default=60.)
    parser.add_argument('--profile-phase-deg', type=float, default=0.)
    parser.add_argument('--br-t', type=float, default=DEFAULT_BR_T)
    parser.add_argument('--output-dir', type=Path, default=Path('results/deformation'))
    args = parser.parse_args()
    p = SheetParameters(n_segments=args.segments,
                        youngs_modulus_pa=args.modulus_kpa*1000,
                        magnetization_a_per_m=args.magnetization_ka_m*1000,
                        profile_phase_deg=args.profile_phase_deg)
    if args.distance_mm < 3*SIDE_M*1000:
        print('WARNING: Dipole approximation may be inaccurate so close to cube.')
    result = solve_equilibrium(p, args.distance_mm, args.angle_deg, args.br_t)
    out = args.output_dir / f'deformation_angle_{args.angle_deg:g}.png'
    plot_equilibrium(p, result, out)
    print(f'Angle: {args.angle_deg:g}°; center-to-magnet distance: {args.distance_mm:g} mm')
    print(f'End-to-end separation: {result["end_to_end_mm"]:.3f} mm')
    print(f'Maximum adjacent-segment angle change: {result["max_joint_angle_deg"]:.2f}°')
    print(f'Minimized total energy: {result["total_energy_J"]:.4e} J')
    print(f'Saved plot: {out.resolve()}')
    print('PRELIMINARY: unmeasured E, magnetization and phase; no contact/rolling.')


if __name__ == '__main__':
    main()
