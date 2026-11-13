"""ME 696 Step 2F: operator-split, quasi-static shape + torque-responsive rolling.

Extends Steps 2B and 2E. At each step, the magnet angle is commanded; the
sheet's internal joint curvatures minimize *contact-shifted* magneto-elastic
potential, and a generalized magnetic/gravitational torque advances the
body's rigid rotation through an overdamped resistance law. Center translation
uses the rotational portion of the ideal no-slip constraint.

THIS IS A RESEARCH PROTOTYPE, NOT A VALIDATED PREDICTIVE ROBOT MODEL.
- Plane-strain-style, planar, inextensible, unshearable 16-link chain;
  constant thickness and piecewise-linear centerline.
- Instantaneous internal shape relaxation (no elastic shape dynamics).
- The lowest mesh node is placed thickness/2 above floor (not true continuous
  contact, friction cone, collision, or self-contact).
- Rotational no-slip travel neglects translation caused by joint deformation;
  thus is NOT a full inextensible no-slip contact solver.
- Resisting torque and rotational damping are illustrative and uncalibrated.
- Uniformly magnetized cube approximated as a POINT DIPOLE, which is inaccurate
  near the cube. Material/geometric parameters require actual measurement.
- The commanded magnet rotates about its center FIXED underneath x=0;
  the sheet can leave the stronger-field region as it translates.
- Solves a constrained local equilibrium with bending joints only; branches
  and local minima can make abrupt transitions possible.

Units: SI internally; CLI uses degrees, millimeters, seconds, kPa, kA/m.
Run from project root:
  python -m simulator.models.coupled_rolling
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from .deformation import SheetParameters, centerline, magnetization_moments
from .magnetic_field import DEFAULT_BR_T, SIDE_M, dipole_field, magnet_moment
from .magnetic_rolling import DriveSettings, overdamped_rate_rad_s


@dataclass(frozen=True)
class CoupledSettings(DriveSettings):
    """DriveSettings plus controls for the quasi-static internal shape solver."""
    n_steps: int = 61
    solver_maxiter: int = 120
    max_joint_change_deg: float = 48.0
    finite_difference_rad: float = .001

    def __post_init__(self):
        super().__post_init__()
        if self.solver_maxiter < 10 or not 0 < self.max_joint_change_deg < 180:
            raise ValueError('invalid internal-shape solver limits')
        if not 0 < self.finite_difference_rad < 0.1:
            raise ValueError('finite_difference_rad must be in (0,.1)')


def orientations_from_joints(joints_rad: np.ndarray, body_angle_rad: float) -> np.ndarray:
    """Build N segment orientations from N-1 joints, mean(theta)=body angle.

    Removing the mean avoids double-counting rigid rotation as free bending.
    Each adjacent segment angle difference equals the corresponding joint.
    """
    joints_rad = np.asarray(joints_rad, dtype=float)
    if joints_rad.ndim != 1 or not np.all(np.isfinite(joints_rad)):
        raise ValueError('joints_rad must be a finite 1D vector')
    raw = np.r_[0., np.cumsum(joints_rad)]
    return raw - raw.mean() + float(body_angle_rad)


def ground_shape(joints_rad: np.ndarray, body_angle_rad: float,
                 p: SheetParameters, center_x_m: float = 0.) -> dict:
    """Compute a placed x-z centerline: lowest NODE at thickness/2 above floor.

    Does not enforce distributed contact along segments or self contact.
    """
    if len(joints_rad) != p.n_segments - 1:
        raise ValueError('joint vector length must be n_segments-1')
    if not (np.isfinite(body_angle_rad) and np.isfinite(center_x_m)):
        raise ValueError('nonfinite pose')
    theta = orientations_from_joints(joints_rad, body_angle_rad)
    nodes, midpoints = centerline(theta, p, np.zeros(3))
    relative_contact_z = float(nodes[:, 2].min())
    center_z_m = p.thickness_m / 2 - relative_contact_z
    offset = np.array([center_x_m, 0., center_z_m])
    return {
        'nodes_m': nodes + offset,
        'midpoints_m': midpoints + offset,
        'theta_rad': theta,
        'center_x_m': float(center_x_m),
        'center_z_m': float(center_z_m),
        'relative_contact_z_m': relative_contact_z,
    }


def constrained_potential_j(joints_rad: np.ndarray, body_angle_rad: float,
                            magnet_angle_deg: float, p: SheetParameters,
                            settings: CoupledSettings, center_x_m: float = 0.) -> float:
    """Bending + magnetic + gravity potential for a given body orientation."""
    joints_rad = np.asarray(joints_rad, dtype=float)
    elastic_u = .5 * (p.flexural_rigidity / p.ds) * float(joints_rad @ joints_rad)
    shape = ground_shape(joints_rad, body_angle_rad, p, center_x_m)
    magnetic_u = 0.
    if p.magnetization_a_per_m != 0 and settings.remanence_t != 0:
        source_m = np.array([0., 0., -settings.magnet_distance_mm*1e-3])
        B = dipole_field(shape['midpoints_m'], source_m,
                         magnet_moment(magnet_angle_deg, settings.remanence_t))
        moments = magnetization_moments(shape['theta_rad'], p)
        magnetic_u = -float(np.sum(B * moments))
    mass = settings.density_kg_m3 * p.length_m * p.width_m * p.thickness_m
    gravity_u = mass * 9.81 * shape['center_z_m']
    return float(elastic_u + magnetic_u + gravity_u)


def relax_shape(p: SheetParameters, settings: CoupledSettings,
                body_angle_rad: float, magnet_angle_deg: float,
                center_x_m: float = 0., seed: np.ndarray | None = None) -> dict:
    """Minimize potential over joints while holding mean body orientation fixed.

    A warm start follows the previous local minimum (quasi-static continuation).
    This does not guarantee a GLOBAL energy minimum or stable real dynamics.
    """
    if seed is None:
        seed = np.zeros(p.n_segments - 1)
    seed = np.asarray(seed, dtype=float)
    if seed.shape != (p.n_segments - 1,):
        raise ValueError('seed must have one angle per compliant joint')
    if not np.all(np.isfinite(seed)):
        raise ValueError('seed has nonfinite values')
    limit = np.deg2rad(settings.max_joint_change_deg)
    bound = [(-limit, limit)] * len(seed)
    objective = lambda q: 1e9 * constrained_potential_j(
        q, body_angle_rad, magnet_angle_deg, p, settings, center_x_m)
    result = minimize(objective, np.clip(seed, -limit, limit),
                      method='L-BFGS-B', bounds=bound,
                      options={'maxiter': settings.solver_maxiter,
                               'ftol': 1e-11, 'gtol': 2e-7})
    # Ground contact switches between different nodes and makes the potential
    # only piecewise smooth. L-BFGS-B can terminate ABNORMAL at a contact switch.
    # Retry with a derivative-free optimizer rather than silently trusting a
    # failed gradient-based solve. Require a valid finite, non-increasing energy.
    if not result.success:
        alternate = minimize(objective, result.x, method='Powell', bounds=bound,
                             options={'maxiter': max(settings.solver_maxiter, 150),
                                      'xtol': 1e-5, 'ftol': 1e-8})
        if alternate.success and np.isfinite(alternate.fun) and alternate.fun <= objective(seed) + 1e-5:
            result = alternate
        else:
            raise RuntimeError(f'Shape equilibrium failed: {result.message}; '
                               f'Powell retry: {alternate.message}')
    s = ground_shape(result.x, body_angle_rad, p, center_x_m)
    s.update({'joints_rad': result.x.copy(),
              'energy_j': float(result.fun*1e-9),
              'end_to_end_mm': float(1e3*np.linalg.norm(s['nodes_m'][-1]-s['nodes_m'][0])),
              'max_joint_angle_deg': float(np.rad2deg(np.max(np.abs(result.x))))})
    return s


def generalized_torque_coupled(shape: dict, magnet_angle_deg: float,
                               p: SheetParameters, settings: CoupledSettings) -> float:
    """Partial generalized torque -dU/dphi with shape joints FROZEN.

    Takes a virtual no-slip rotation with dx_center/dphi = contact_z_rel,
    matching the frozen-shape Step 2E constraint locally. Shape is allowed to
    re-equilibrate at each NEXT time step, not within this derivative.
    """
    h = settings.finite_difference_rad
    phi = float(shape['theta_rad'].mean())
    x = shape['center_x_m']
    q = shape['joints_rad']
    def u(delta):
        return constrained_potential_j(q, phi+delta, magnet_angle_deg,
                                       p, settings,
                                       center_x_m=x + shape['relative_contact_z_m']*delta)
    return -(u(h)-u(-h))/(2*h)


def simulate_coupled_rolling(p: SheetParameters, settings: CoupledSettings) -> dict:
    """Sequentially update shape, then drive body rotation with computed torque.

    One local shape equilibrium per time step; Euler rotational rate update.
    Translation integrates only the rigid-roll part of no-slip kinematics.
    This operator-split result has no deformational tangential-velocity term.
    """
    t = np.linspace(0., settings.duration_s, settings.n_steps)
    command_deg = settings.start_magnet_deg + settings.magnet_turn_deg*t/settings.duration_s
    phi = np.zeros(len(t))
    x = np.zeros(len(t))
    rates = np.zeros(len(t))
    torques = np.zeros(len(t))
    max_join = np.zeros(len(t))
    end_to_end = np.zeros(len(t))
    center_z = np.zeros(len(t))
    energies = np.zeros(len(t))
    poses = []
    joints_history = []
    q = np.zeros(p.n_segments-1)
    dt = t[1]-t[0]
    max_body = np.deg2rad(settings.max_body_turn_deg)
    for i in range(len(t)):
        shape = relax_shape(p, settings, phi[i], command_deg[i], x[i], seed=q)
        q = shape['joints_rad']
        joints_history.append(q.copy())
        poses.append(shape['nodes_m'][:, [0,2]])
        center_z[i] = shape['center_z_m']
        max_join[i] = shape['max_joint_angle_deg']
        end_to_end[i] = shape['end_to_end_mm']
        energies[i] = shape['energy_j']
        torques[i] = generalized_torque_coupled(shape, command_deg[i], p, settings)
        rates[i] = overdamped_rate_rad_s(torques[i], settings)
        if i+1 < len(t):
            step_dphi = float(np.clip(dt*rates[i], -0.20, 0.20))
            phi[i+1] = float(np.clip(phi[i] + step_dphi, -max_body, max_body))
            dphi = phi[i+1]-phi[i]
            # Mid-orientation lowest-node z in sheet-centered frame.
            midpoint_shape = ground_shape(q, phi[i]+.5*dphi, p, x[i])
            x[i+1] = x[i] + midpoint_shape['relative_contact_z_m']*dphi

    poses = np.asarray(poses)
    assert poses.shape == (len(t),p.n_segments+1,2)
    if float(poses[:,:,1].min()) < p.thickness_m/2 - 5e-9:
        raise RuntimeError('floor penetration detected')
    return {
        'time_s': t, 'magnet_angle_deg': command_deg,
        'body_angle_deg': np.rad2deg(phi),
        'angular_rate_rad_s': rates,
        'generalized_torque_n_m': torques,
        'center_x_mm': 1e3*x, 'center_z_mm': 1e3*center_z,
        'end_to_end_mm': end_to_end, 'max_joint_angle_deg': max_join,
        'energy_J': energies,
        'poses_xz_m': poses, 'joints_rad': np.asarray(joints_history),
        'travel_mm': float(1e3*(x[-1]-x[0])),
        'thickness_m': p.thickness_m,
    }


def save_results(r: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'coupled_rolling.csv'
    keys = ['time_s', 'magnet_angle_deg', 'body_angle_deg',
            'angular_rate_rad_s', 'generalized_torque_n_m',
            'center_x_mm', 'center_z_mm', 'end_to_end_mm',
            'max_joint_angle_deg', 'energy_J']
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in zip(*(r[key] for key in keys)):
            writer.writerow([f'{float(v):.10g}' for v in row])

    fig, axes = plt.subplots(3, 1, figsize=(9.6, 8), sharex=True)
    axes[0].plot(r['time_s'], r['magnet_angle_deg'], label='Magnet command')
    axes[0].plot(r['time_s'], r['body_angle_deg'], label='Calculated body response')
    axes[0].set(ylabel='Rotation (deg)', title='Step 2F: Coupled (quasi-static) shape / magnetic drive')
    axes[0].legend(loc='best'); axes[0].grid(alpha=.3)
    axes[1].plot(r['time_s'], r['end_to_end_mm'], label='End-to-end separation')
    axes[1].set(ylabel='End-to-end (mm)'); axes[1].grid(alpha=.3)
    axes[2].plot(r['time_s'], r['center_x_mm'], label='Approximate rolling translation')
    axes[2].set(xlabel='Time (s)', ylabel='Center x (mm)'); axes[2].grid(alpha=.3)
    fig.text(.07,.012,'Illustrative: dipole + quasi-static bending, assumed resistance; deformation-driven slip is omitted.',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(out/'coupled_summary.png',dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4,4.8))
    indices = np.unique(np.linspace(0,len(r['time_s'])-1,8,dtype=int))
    selected = r['poses_xz_m'][indices]*1e3
    for j,i in enumerate(indices):
        pose = selected[j]
        ax.plot(pose[:,0],pose[:,1],'-o',markersize=2,lw=1.6,
                label=f't={r["time_s"][i]:.1f}s, bend={r["end_to_end_mm"][i]:.2f}mm')
    ax.axhline(0,color='0.3',lw=1.6,label='Ground')
    ax.set(xlabel='x (mm)',ylabel='z (mm)',title='Step 2F: changing sheet shape while it rolls')
    ax.grid(alpha=.3); ax.set_aspect('equal',adjustable='box')
    ax.legend(fontsize=7,ncol=2)
    fig.text(.07,.015,'Approximate floor placement; not a validated contact/rolling model.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(out/'coupled_poses.png',dpi=150)
    plt.close(fig)
    return csv_path


def main() -> None:
    a = argparse.ArgumentParser(description='Step 2F: preliminary quasi-static deformation + rolling')
    a.add_argument('--magnet-turn-deg', type=float, default=360.)
    a.add_argument('--duration-s', type=float, default=8.)
    a.add_argument('--steps', type=int, default=61)
    a.add_argument('--distance-mm', type=float, default=60.)
    a.add_argument('--damping-nm-s', type=float, default=1.5e-7)
    a.add_argument('--resistance-nm', type=float, default=0.)
    a.add_argument('--density-kg-m3', type=float, default=1500.)
    a.add_argument('--remanence-t', type=float, default=DEFAULT_BR_T)
    a.add_argument('--modulus-kpa', type=float, default=100.)
    a.add_argument('--magnetization-ka-m', type=float, default=60.)
    a.add_argument('--phase-deg', type=float, default=0.)
    a.add_argument('--segments', type=int, default=16)
    a.add_argument('--output-dir', type=Path, default=Path('results/coupled_rolling'))
    args = a.parse_args()
    if args.distance_mm < SIDE_M*1e3*3:
        print('WARNING: Dipole model less reliable close to magnet.')
    p = SheetParameters(n_segments=args.segments,
                        youngs_modulus_pa=args.modulus_kpa*1e3,
                        magnetization_a_per_m=args.magnetization_ka_m*1e3,
                        profile_phase_deg=args.phase_deg)
    s = CoupledSettings(magnet_turn_deg=args.magnet_turn_deg,
                        duration_s=args.duration_s, n_steps=args.steps,
                        magnet_distance_mm=args.distance_mm,
                        damping_n_m_s=args.damping_nm_s,
                        resistance_n_m=args.resistance_nm,
                        density_kg_m3=args.density_kg_m3,
                        remanence_t=args.remanence_t)
    result = simulate_coupled_rolling(p,s)
    csv_path = save_results(result,args.output_dir)
    print(f'Commanded magnet rotation: {args.magnet_turn_deg:g} deg over {args.duration_s:g} s')
    print(f'Calculated body rotation: {result["body_angle_deg"][-1]:.2f} deg')
    print(f'Approximate rolling travel: {result["travel_mm"]:.3f} mm')
    print(f'End-to-end distance range: {result["end_to_end_mm"].min():.3f} to {result["end_to_end_mm"].max():.3f} mm')
    print(f'Minimum node height: {1e3*result["poses_xz_m"][:,:,1].min():.4f} mm')
    print(f'Saved plots/CSV: {csv_path.resolve().parent}')
    print('PRELIMINARY: shape relaxes quasi-statically; rolling ignores deformation-driven tangential slip; no physical validation.')


if __name__ == '__main__':
    main()
