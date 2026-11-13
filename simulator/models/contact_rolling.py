"""ME696 Step 2D: ground placement and ideal no-slip *kinematic* rolling.

This is an incremental geometry model, NOT yet a complete predictive rolling model.
The initial C-shape is taken from the Step 2B magneto-elastic equilibrium solver,
then held RIGID while the BODY rotation angle is specified. The cumulative
horizontal travel is derived from the instantaneous contact geometry under
pure-rolling constraints; it is NOT a prescribed travel-distance animation.

Still missing: torque-driven rotation, deformation during rotation, gravitational
and contact forces, finite-width contact, and stick/slip. The experimental
field-to-rolling relationship is NOT validated.

Run from repo root:
  python -m simulator.models.contact_rolling
  python -m simulator.models.contact_rolling --rotation-deg -180 --distance-mm 80

All core geometry uses meters and radians; exported CSV uses mm and degrees.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .deformation import SheetParameters, solve_equilibrium


@dataclass(frozen=True)
class RollingSettings:
    rotation_deg: float = 360.0
    n_steps: int = 721
    floor_z_m: float = 0.0
    thickness_m: float = 185e-6

    def __post_init__(self):
        if self.n_steps < 2:
            raise ValueError('n_steps must be at least 2')
        if self.thickness_m < 0:
            raise ValueError('thickness_m must be nonnegative')
        if not all(np.isfinite(v) for v in (self.rotation_deg, self.floor_z_m, self.thickness_m)):
            raise ValueError('settings must be finite')


def rotate_xz(points_xz: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate an (N,2) set of points in x-z plane, radians from +x to +z."""
    p = np.asarray(points_xz, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError('points_xz must have shape (N,2)')
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]])
    return p @ rot.T


def normalize_centerline(nodes_m: np.ndarray) -> np.ndarray:
    """Use mean of segment-midpoints as reference point, consistent with 2B."""
    nodes = np.asarray(nodes_m, dtype=float)
    if nodes.ndim != 2 or nodes.shape[0] < 3 or nodes.shape[1] != 3:
        raise ValueError('nodes_m must be (N+1,3) for at least two segments')
    chain_xz = nodes[:, [0, 2]]
    centroid = (0.5 * (chain_xz[:-1] + chain_xz[1:])).mean(axis=0)
    return chain_xz - centroid


def roll_kinematically(shape_xz_m: np.ndarray, settings: RollingSettings) -> dict:
    """Roll a FROZEN curve by advancing rigid-body orientation with no slip.

    At every body angle, the lowest centerline node is treated as the point of
    contact. For a non-slipping contact, vx_center = omega * (z_contact - z_center).
    Integrating this kinematic constraint yields the horizontal translation.

    Contact is a discrete node approximation; interpolation error decreases as
    n_steps and shape node density increase. A constant thickness/2 clearance
    keeps the thickness of the modeled strip above the floor in this crude
    geometric representation.
    """
    shape = np.asarray(shape_xz_m, dtype=float)
    if shape.ndim != 2 or shape.shape[1] != 2 or shape.shape[0] < 3:
        raise ValueError('shape_xz_m must be (N,2), N>=3')
    if not np.all(np.isfinite(shape)):
        raise ValueError('shape contains nonfinite values')

    angles = np.linspace(0.0, np.deg2rad(settings.rotation_deg), settings.n_steps)
    rotated = np.array([rotate_xz(shape, a) for a in angles])
    contact_ids = np.argmin(rotated[:, :, 1], axis=1)
    seq = np.arange(settings.n_steps)
    contact_relative = rotated[seq, contact_ids, :]
    x_center = np.zeros(settings.n_steps)
    # Trapezoidal time integration for no-slip constraint dx/dphi = z_contact.
    dphi = np.diff(angles)
    x_center[1:] = np.cumsum(0.5 * (contact_relative[1:, 1] + contact_relative[:-1, 1]) * dphi)
    z_center = settings.floor_z_m + settings.thickness_m / 2 - contact_relative[:, 1]
    placement = rotated + np.stack((x_center, z_center), axis=1)[:, None, :]
    contact_points = placement[seq, contact_ids, :]
    return {
        'angle_deg': np.rad2deg(angles),
        'x_center_m': x_center,
        'z_center_m': z_center,
        'poses_xz_m': placement,
        'contact_node_id': contact_ids,
        'contact_xz_m': contact_points,
        'travel_mm': float((x_center[-1] - x_center[0]) * 1000.0),
        'floor_z_m': settings.floor_z_m,
        'thickness_m': settings.thickness_m,
    }


def plot_snapshots(result: dict, out_path: Path, indices=(0, 1/4, 1/2, 3/4, 1)) -> None:
    poses = result['poses_xz_m'] * 1e3
    xs = result['x_center_m'] * 1e3
    zfloor = result['floor_z_m'] * 1e3
    selected = np.unique(np.round(np.asarray(indices) * (len(poses)-1)).astype(int))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx in selected:
        xy = poses[idx]
        ax.plot(xy[:, 0], xy[:, 1], '-o', markersize=2.3, lw=1.6,
                label=f'{result["angle_deg"][idx]:g}°')
        c = result['contact_xz_m'][idx] * 1e3
        ax.plot(c[0], c[1], 'kx', ms=5)
    allx = poses[selected, :, 0].ravel()
    ax.plot([allx.min()-0.6, allx.max()+0.6], [zfloor, zfloor],
            linestyle='--', lw=1.6, color='0.25', label='Ground z=0')
    ax.set(xlabel='x (mm)', ylabel='z (mm)', title='Step 2D: frozen-shape ideal no-slip rolling — snapshots')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.28)
    ax.legend(fontsize=8, ncol=3)
    fig.text(.07, .015, 'PRELIMINARY: imposed body rotation; no torque, slip, contact forces, or evolving deformation.', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_trajectory(result: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.plot(result['angle_deg'], result['x_center_m'] * 1000, label='Horizontal displacement')
    ax.set(xlabel='Imposed body rotation (degrees)', ylabel='Center displacement x (mm)',
           title='Step 2D: kinematic no-slip displacement prediction')
    ax.grid(alpha=.3)
    ax.legend()
    fig.text(.11, .02, 'Geometry-only baseline; rotation is specified, not predicted from applied magnetic torque.', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_csv(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rotation_deg', 'center_x_mm', 'center_z_mm',
                         'contact_x_mm', 'contact_z_mm', 'contact_node_id'])
        for i, angle in enumerate(result['angle_deg']):
            writer.writerow([f'{angle:.6f}', f'{result["x_center_m"][i]*1000:.6f}',
                             f'{result["z_center_m"][i]*1000:.6f}',
                             f'{result["contact_xz_m"][i,0]*1000:.6f}',
                             f'{result["contact_xz_m"][i,1]*1000:.6f}',
                             int(result['contact_node_id'][i])])


def main() -> None:
    parser = argparse.ArgumentParser(description='Step 2D: ground placement and frozen-shape no-slip rolling')
    parser.add_argument('--shape-angle-deg', type=float, default=0.,
                        help='Magnet angle used to solve the frozen Step 2B shape')
    parser.add_argument('--distance-mm', type=float, default=60.,
                        help='Magnet-center to initial sheet-centroid distance used by Step 2B')
    parser.add_argument('--rotation-deg', type=float, default=360.,
                        help='Imposed BODY rotation (not magnet angle)')
    parser.add_argument('--steps', type=int, default=721)
    parser.add_argument('--output-dir', type=Path, default=Path('results/rolling'))
    parser.add_argument('--modulus-kpa', type=float, default=100.)
    parser.add_argument('--magnetization-ka-m', type=float, default=60.)
    parser.add_argument('--profile-phase-deg', type=float, default=0.)
    args = parser.parse_args()

    p = SheetParameters(youngs_modulus_pa=args.modulus_kpa * 1e3,
                        magnetization_a_per_m=args.magnetization_ka_m * 1e3,
                        profile_phase_deg=args.profile_phase_deg)
    eq = solve_equilibrium(p, distance_mm=args.distance_mm, angle_deg=args.shape_angle_deg)
    profile = normalize_centerline(eq['nodes_m'])
    settings = RollingSettings(rotation_deg=args.rotation_deg,
                               n_steps=args.steps, thickness_m=p.thickness_m)
    result = roll_kinematically(profile, settings)
    plot_snapshots(result, args.output_dir / 'rolling_contact_poses.png')
    plot_trajectory(result, args.output_dir / 'rolling_trajectory.png')
    save_csv(result, args.output_dir / 'rolling_trajectory.csv')
    print(f'Step 2B frozen shape: magnet angle {args.shape_angle_deg:g}°, distance {args.distance_mm:g} mm')
    print(f'Imposed BODY rotation: {args.rotation_deg:g}° ({args.steps} steps)')
    print(f'Kinematic travel (ideal no-slip): {result["travel_mm"]:.3f} mm')
    print(f'Minimum centerline height: {np.min(result["poses_xz_m"][:,:,1])*1000:.4f} mm')
    print(f'Ground: {settings.floor_z_m*1000:.4f} mm, assumed half-thickness clearance: {settings.thickness_m*500:.4f} mm')
    print(f'Saved plots and CSV in {args.output_dir.resolve()}')
    print('WARNING: Body rotation is imposed; model does NOT predict torque-driven rolling or frictional feasibility.')


if __name__ == '__main__':
    main()
