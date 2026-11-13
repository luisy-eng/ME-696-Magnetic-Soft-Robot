"""ME 696 Topic 1 - initial magnetic field model for the supplied N52 cube.

Physics: point-dipole approximation outside the magnet, not a finite-cube solution.
All internal distances are in meters, B in tesla, magnetic moment in A*m^2.
Run from the repo root: python -m simulator.models.magnetic_field
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle

MU0 = 4.0 * np.pi * 1e-7               # T*m/A (engineering approximation)
SIDE_M = 0.75 * 0.0254                # 3/4-inch cube, project specification
DEFAULT_BR_T = 1.43                   # illustrative N52 remanence, not measured
HU_SHEET_LENGTH_M = 3.7e-3           # Hu et al. Figure 1, NOT measured sample


def equivalent_dipole_moment(br_t: float = DEFAULT_BR_T,
                             side_m: float = SIDE_M) -> float:
    """Approximate moment magnitude [A*m^2]: m ~ (Br/mu0)*V.

    Simplifies the cube into a uniformly magnetized point dipole; this
    neglects the cube's finite geometry and demagnetizing-field effects.
    """
    if br_t <= 0 or side_m <= 0:
        raise ValueError("remanence and cube side must be positive")
    return br_t * side_m ** 3 / MU0


def dipole_field(points_m: np.ndarray,
                 magnet_center_m=(0.0, 0.0, 0.0),
                 moment_A_m2=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Magnetic field B [T] of a point dipole at observation points.

    points_m: (..., 3) array, or a single 3-vector.
    Output has the same shape. The point-dipole formula is invalid at
    the magnet center and is not reliable in the magnet's near-field.
    """
    p = np.asarray(points_m, dtype=float)
    if p.shape[-1] != 3:
        raise ValueError("points_m must have shape (..., 3)")
    center = np.asarray(magnet_center_m, dtype=float)
    m = np.asarray(moment_A_m2, dtype=float)
    if center.shape != (3,) or m.shape != (3,):
        raise ValueError("center and dipole moment must be 3-vectors")

    r = p - center
    distance = np.linalg.norm(r, axis=-1)
    if np.any(distance == 0):
        raise ValueError("dipole field is undefined at the magnet center")
    r_hat = r / distance[..., None]
    m_dot_r = np.sum(m * r_hat, axis=-1)
    return (MU0 / (4.0 * np.pi)) * (
        3.0 * m_dot_r[..., None] * r_hat - m
    ) / distance[..., None] ** 3


def magnet_moment(angle_deg: float = 0.0,
                  br_t: float = DEFAULT_BR_T) -> np.ndarray:
    """Magnetic dipole moment rotated in the x-z plane.

    angle=0 deg -> magnetic axis +z; angle=90 deg -> +x.
    """
    a = np.deg2rad(angle_deg)
    magnitude = equivalent_dipole_moment(br_t)
    return magnitude * np.array([np.sin(a), 0.0, np.cos(a)])


def make_field_map(out_path: Path, distance_mm: float, angle_deg: float,
                   br_t: float) -> None:
    """Save x-z field map with reference sheet position indicated."""
    x = np.linspace(-0.10, 0.10, 161)
    z = np.linspace(-0.10, 0.12, 177)
    X, Z = np.meshgrid(x, z)
    points = np.stack((X, np.zeros_like(X), Z), axis=-1)
    r = np.sqrt(X**2 + Z**2)

    # Plot only outside the rough near-field region. This threshold is a
    # visualization warning, not a quantified accuracy guarantee.
    valid = r >= 0.030
    safe_points = points.copy()
    safe_points[~valid] = np.array([0., 0., 0.030])
    B = dipole_field(safe_points, moment_A_m2=magnet_moment(angle_deg, br_t))
    Bmag_mT = np.linalg.norm(B, axis=-1) * 1e3
    Bmag_mT[~valid] = np.nan

    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    mesh = ax.pcolormesh(1000 * X, 1000 * Z, Bmag_mT,
                         shading='auto', cmap='viridis')
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label('Dipole field magnitude |B| (mT)')

    # Vector arrows, normalized to show field direction clearly.
    skip = (slice(None, None, 12), slice(None, None, 12))
    Bx, Bz = B[..., 0], B[..., 2]
    norms = np.maximum(np.sqrt(Bx**2 + Bz**2), 1e-30)
    qx = np.where(valid, Bx / norms, np.nan)
    qz = np.where(valid, Bz / norms, np.nan)
    ax.quiver(1000 * X[skip], 1000 * Z[skip], qx[skip], qz[skip],
              pivot='mid', scale=26, width=0.003)
    half_mm = SIDE_M * 500
    ax.add_patch(Rectangle((-half_mm, -half_mm), 2*half_mm, 2*half_mm,
                           fill=False, edgecolor='white', linewidth=2))
    ax.add_patch(Circle((0, 0), 30, fill=False, edgecolor='white',
                        linestyle='--', linewidth=1.2, alpha=0.8))
    ax.plot([-HU_SHEET_LENGTH_M*500, HU_SHEET_LENGTH_M*500],
            [distance_mm, distance_mm], linewidth=4, label='Hu reference sheet (3.7 mm)')
    ax.set(xlabel='x (mm)', ylabel='z (mm)',
           title=f'N52 cube dipole approximation | angle {angle_deg:g} deg')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 120)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_sheet_profile(out_path: Path, distance_mm: float,
                       angle_deg: float, br_t: float) -> None:
    x = np.linspace(-HU_SHEET_LENGTH_M / 2,
                    HU_SHEET_LENGTH_M / 2, 101)
    p = np.column_stack((x, np.zeros_like(x),
                         np.full_like(x, distance_mm / 1000)))
    B = dipole_field(p, moment_A_m2=magnet_moment(angle_deg, br_t))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x*1000, B[:, 0]*1000, label='Bx')
    ax.plot(x*1000, B[:, 1]*1000, label='By')
    ax.plot(x*1000, B[:, 2]*1000, label='Bz')
    ax.plot(x*1000, np.linalg.norm(B, axis=-1)*1000,
            linestyle='--', label='|B|')
    ax.set(xlabel='Position along sample, x (mm)', ylabel='Magnetic field (mT)',
           title=f'Field at sample plane, z={distance_mm:g} mm')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='N52 cube magnet dipole model')
    parser.add_argument('--distance-mm', type=float, default=60.,
                        help='Center-to-sheet plane distance (mm), default 60')
    parser.add_argument('--angle-deg', type=float, default=0.,
                        help='Moment angle from +z toward +x in degrees')
    parser.add_argument('--br-t', type=float, default=DEFAULT_BR_T,
                        help='Assumed cube remanence in tesla')
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.distance_mm <= 0:
        parser.error('distance must be positive')
    if args.distance_mm < 3*SIDE_M*1000:
        print('WARNING: distance is close to the cube; dipole errors may be large.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    m = magnet_moment(args.angle_deg, args.br_t)
    midpoint = np.array([0., 0., args.distance_mm / 1000])
    B = dipole_field(midpoint, moment_A_m2=m)

    print(f'Cube side: {SIDE_M*1000:.2f} mm')
    print(f'Assumed Br: {args.br_t:.3f} T (not calibrated)')
    print(f'Equivalent dipole moment magnitude: {np.linalg.norm(m):.3f} A*m^2')
    print(f'Magnet axis angle: {args.angle_deg:g} deg from +z toward +x')
    print(f'Sheet plane z: {args.distance_mm:g} mm above magnet center')
    print(f'Field at sheet midpoint [Bx,By,Bz] (mT): {np.round(B*1000, 4)}')
    print(f'Field magnitude at midpoint: {np.linalg.norm(B)*1000:.4f} mT')

    make_field_map(args.output_dir / 'field_map.png',
                   args.distance_mm, args.angle_deg, args.br_t)
    make_sheet_profile(args.output_dir / 'sheet_field_profile.png',
                       args.distance_mm, args.angle_deg, args.br_t)
    print(f'Saved plots to: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
