"""ME 696 Step 2E: preliminary magnetically driven, no-slip rolling.

A frozen Step 2B equilibrium contour rolls on a horizontal floor. The cube
magnet orientation is the COMMAND; the sheet body angle is an independently
integrated state. The body angular velocity comes from a generalized magnetic
and gravitational torque derived from potential energy, an assumed static
resistance torque, and an overdamped angular mobility.

This is a reduced-order PROTOTYPE, not experimentally validated or a full
magneto-elastic rolling solution. No inertia, continuous shape adaptation,
finite-cube field, friction-cone check, or stick/slip is included. The no-slip
constraint is imposed, and damping/resistance values are hypothetical.

Units: SI internally; CLI: angles deg, distance mm, time s. Run from repo root:
  python -m simulator.models.magnetic_rolling
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .contact_rolling import normalize_centerline, rotate_xz
from .deformation import SheetParameters, magnetization_moments, solve_equilibrium
from .magnetic_field import DEFAULT_BR_T, SIDE_M, dipole_field, magnet_moment


@dataclass(frozen=True)
class DriveSettings:
    magnet_turn_deg: float = 360.0
    duration_s: float = 8.0
    n_steps: int = 801
    magnet_distance_mm: float = 60.0  # magnet center BELOW the ground by this much
    damping_n_m_s: float = 1.5e-7       # illustrative generalized viscous angular drag
    resistance_n_m: float = 0.0       # illustrative dry resistance threshold
    density_kg_m3: float = 1500.0     # illustrative mass density of composite
    remanence_t: float = DEFAULT_BR_T
    start_magnet_deg: float = 0.0
    max_body_turn_deg: float = 540.0

    def __post_init__(self):
        if self.n_steps < 2 or self.duration_s <= 0 or self.magnet_distance_mm <= 0:
            raise ValueError('n_steps >=2 and duration/distance >0 required')
        if self.damping_n_m_s <= 0 or self.resistance_n_m < 0 or self.density_kg_m3 < 0:
            raise ValueError('damping must be >0; resistance and density must be >=0')
        if self.remanence_t < 0 or self.max_body_turn_deg <= 0:
            raise ValueError('remanence must be >=0; max body turn >0')
        values = [self.magnet_turn_deg, self.duration_s, self.magnet_distance_mm,
                  self.damping_n_m_s, self.resistance_n_m, self.density_kg_m3,
                  self.remanence_t, self.start_magnet_deg, self.max_body_turn_deg]
        if not all(np.isfinite(v) for v in values):
            raise ValueError('all settings must be finite')


def build_no_slip_lookup(shape_xz_m: np.ndarray, thickness_m: float,
                         extent_rad: float, samples: int = 16001) -> dict:
    """Precompute continuous-in-angle interpolation of ideal floor contact.

    dx_center/dphi = z_contact(relative to centroid), where the lowest sampled
    node is the contact. This implements the same geometry as Step 2D for both
    positive and negative body angles. Thickness is a centerline clearance.
    """
    shape = np.asarray(shape_xz_m, dtype=float)
    if shape.ndim != 2 or shape.shape[1] != 2 or shape.shape[0] < 3:
        raise ValueError('shape must be (N,2), with N >= 3')
    if thickness_m < 0 or extent_rad <= 0 or samples < 5:
        raise ValueError('invalid contact lookup settings')
    if not np.all(np.isfinite(shape)):
        raise ValueError('nonfinite shape')
    angles = np.linspace(-extent_rad, extent_rad, samples)
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    rotated_z = shape[None, :, 0] * sin + shape[None, :, 1] * cos
    z_contact = rotated_z.min(axis=1)
    # dx/dphi = lowest point's vertical coordinate relative to centroid.
    dphi = np.diff(angles)
    x = np.zeros(len(angles))
    x[1:] = np.cumsum(0.5 * (z_contact[1:] + z_contact[:-1]) * dphi)
    x -= np.interp(0.0, angles, x)
    z_center = thickness_m / 2 - z_contact
    return {'phi_rad': angles, 'center_x_m': x, 'center_z_m': z_center}


def floor_pose(phi_rad: float, shape_xz_m: np.ndarray, lookup: dict):
    angles = lookup['phi_rad']
    if not (angles[0] <= phi_rad <= angles[-1]):
        raise ValueError('body angle exceeds contact lookup range')
    center = np.array([np.interp(phi_rad, angles, lookup['center_x_m']),
                       np.interp(phi_rad, angles, lookup['center_z_m'])])
    pose = rotate_xz(shape_xz_m, phi_rad) + center
    return pose, center


def potential_energy_j(phi_rad: float, alpha_deg: float, shape_xz_m: np.ndarray,
                       initial_theta_rad: np.ndarray, p: SheetParameters,
                       settings: DriveSettings, lookup: dict) -> float:
    """U = -sum(m_i dot B_i) + Mg z_cm under ideal no-slip constraint.

    Differentiating w.r.t body angle includes magnetic gradient forces from
    the resulting constrained translation, not just free-body magnetic torque.
    """
    pose, center = floor_pose(phi_rad, shape_xz_m, lookup)
    midpoints = (pose[1:] + pose[:-1]) / 2
    points = np.column_stack((midpoints[:, 0],
                              np.zeros(len(midpoints)), midpoints[:, 1]))
    if settings.remanence_t == 0 or p.magnetization_a_per_m == 0:
        magnetic_u = 0.0
    else:
        cube_center = np.array([0., 0., -settings.magnet_distance_mm * 1e-3])
        B = dipole_field(points, magnet_center_m=cube_center,
                         moment_A_m2=magnet_moment(alpha_deg, settings.remanence_t))
        moments = magnetization_moments(initial_theta_rad + phi_rad, p)
        magnetic_u = -float(np.sum(moments * B))
    mass_kg = settings.density_kg_m3 * p.length_m * p.width_m * p.thickness_m
    gravity_u = mass_kg * 9.81 * center[1]
    return float(magnetic_u + gravity_u)


def generalized_torque_n_m(phi_rad: float, alpha_deg: float, shape_xz_m: np.ndarray,
                            initial_theta_rad: np.ndarray, p: SheetParameters,
                            settings: DriveSettings, lookup: dict,
                            h_rad: float = 0.001) -> float:
    """-dU/dphi: the no-slip rolling generalized torque [N m]."""
    if h_rad <= 0:
        raise ValueError('h_rad must be positive')
    high = potential_energy_j(phi_rad+h_rad, alpha_deg, shape_xz_m,
                              initial_theta_rad, p, settings, lookup)
    low = potential_energy_j(phi_rad-h_rad, alpha_deg, shape_xz_m,
                             initial_theta_rad, p, settings, lookup)
    return -(high-low)/(2*h_rad)


def overdamped_rate_rad_s(torque_n_m: float, settings: DriveSettings) -> float:
    """Coulomb-like threshold plus linear damping (phenomenological model)."""
    excess = max(abs(torque_n_m) - settings.resistance_n_m, 0.0)
    return float(np.sign(torque_n_m) * excess / settings.damping_n_m_s)


def simulate_magnetic_rolling(p: SheetParameters, settings: DriveSettings,
                              initial_shape_angle_deg: float = 0.0) -> dict:
    """Integrate torque-driven BODY rotation as the magnet orientation changes.

    The initial contour is found from Step 2B once and FROZEN. The magnet's
    angle is a commanded input; body angle results from overdamped torque
    balance, with a numerical Heun predictor-corrector time step.
    """
    eq = solve_equilibrium(p, distance_mm=settings.magnet_distance_mm,
                           angle_deg=initial_shape_angle_deg,
                           br_t=settings.remanence_t or DEFAULT_BR_T)
    shape = normalize_centerline(eq['nodes_m'])
    theta = eq['theta_rad']
    max_turn = np.deg2rad(settings.max_body_turn_deg)
    lookup = build_no_slip_lookup(shape, p.thickness_m, max_turn + .05)
    t = np.linspace(0, settings.duration_s, settings.n_steps)
    command = settings.start_magnet_deg + settings.magnet_turn_deg * t / settings.duration_s
    body = np.zeros_like(t)
    torques = np.zeros_like(t)
    speeds = np.zeros_like(t)
    dt = t[1] - t[0]

    def torque(q: float, a: float):
        return generalized_torque_n_m(q, a, shape, theta, p, settings, lookup)

    for i in range(len(t)-1):
        torques[i] = torque(body[i], command[i])
        v1 = overdamped_rate_rad_s(torques[i], settings)
        predicted = np.clip(body[i] + dt*v1, -max_turn, max_turn)
        v2 = overdamped_rate_rad_s(torque(predicted, command[i+1]), settings)
        body[i+1] = np.clip(body[i] + dt*0.5*(v1+v2), -max_turn, max_turn)
        speeds[i] = (body[i+1] - body[i])/dt
    torques[-1] = torque(body[-1], command[-1])
    speeds[-1] = overdamped_rate_rad_s(torques[-1], settings)

    poses = []
    centers = []
    for q in body:
        pose, center = floor_pose(q, shape, lookup)
        poses.append(pose)
        centers.append(center)
    poses, centers = np.array(poses), np.array(centers)
    if np.min(poses[:, :, 1]) < p.thickness_m/2-1e-8:
        raise RuntimeError('unexpected penetration through the floor')
    return {
        'time_s':t, 'magnet_angle_deg':command,
        'body_angle_deg':np.rad2deg(body), 'angular_rate_rad_s':speeds,
        'generalized_torque_n_m':torques,
        'center_x_mm':centers[:,0]*1000, 'center_z_mm':centers[:,1]*1000,
        'poses_xz_m':poses, 'travel_mm':float((centers[-1,0]-centers[0,0])*1e3),
        'shape_xz_m':shape, 'initial_shape_angle_deg':initial_shape_angle_deg,
        'reference_end_to_end_mm':eq['end_to_end_mm'],
        'thickness_m':p.thickness_m,
    }


def save_results(result:dict, output_dir:Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path=output_dir/'magnetically_driven_rolling.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['time_s','magnet_angle_deg','body_angle_deg',
                         'angular_rate_rad_s','generalized_torque_n_m',
                         'center_x_mm','center_z_mm'])
        for row in zip(result['time_s'], result['magnet_angle_deg'],
                       result['body_angle_deg'],result['angular_rate_rad_s'],
                       result['generalized_torque_n_m'],
                       result['center_x_mm'],result['center_z_mm']):
            writer.writerow([f'{v:.9g}' for v in row])

    fig, ax=plt.subplots(2,1,figsize=(9,6.5),sharex=True)
    ax[0].plot(result['time_s'],result['magnet_angle_deg'],label='Commanded magnet angle')
    ax[0].plot(result['time_s'],result['body_angle_deg'],label='Predicted body angle')
    ax[0].set(ylabel='Angle (degrees)',title='Step 2E: torque-responsive ideal no-slip rolling')
    ax[0].grid(alpha=.25); ax[0].legend()
    ax[1].plot(result['time_s'],result['center_x_mm'],label='Predicted no-slip x displacement')
    ax[1].set(xlabel='Time (s)',ylabel='Horizontal center x (mm)')
    ax[1].grid(alpha=.25)
    fig.text(.09,.01,'ILLUSTRATIVE: frozen shape, assumed resistance/damping; field-to-rolling not validated.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(output_dir/'drive_vs_body.png',dpi=160)
    plt.close(fig)

    fig, ax=plt.subplots(figsize=(9,4.5))
    indices=np.unique(np.linspace(0,len(result['poses_xz_m'])-1,7,dtype=int))
    for i in indices:
        pose=result['poses_xz_m'][i]*1000
        ax.plot(pose[:,0],pose[:,1],'-o',lw=1.5,markersize=2,
                label=f'body {result["body_angle_deg"][i]:.0f}°')
    xx=result['poses_xz_m'][indices,:,0]*1000
    ax.axhline(0,color='0.35',lw=2,label='Floor')
    ax.set(xlabel='x (mm)',ylabel='z (mm)',title='Step 2E: calculated body poses and floor')
    ax.set_xlim(xx.min()*1.07-.3,xx.max()*1.07+.3)
    ax.set_aspect('equal',adjustable='box'); ax.grid(alpha=.3)
    ax.legend(ncol=3,fontsize=8)
    fig.text(.09,.01,'No-slip assumed; position follows angle solved from generalized torque.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(output_dir/'rolling_poses.png',dpi=160)
    plt.close(fig)
    return csv_path


def main():
    a=argparse.ArgumentParser(description='Step 2E preliminary torque-responsive rolling')
    a.add_argument('--magnet-turn-deg',type=float,default=360.)
    a.add_argument('--duration-s',type=float,default=8.)
    a.add_argument('--steps',type=int,default=801)
    a.add_argument('--distance-mm',type=float,default=60.)
    a.add_argument('--damping-nm-s',type=float,default=1.5e-7)
    a.add_argument('--resistance-nm',type=float,default=0.)
    a.add_argument('--density-kg-m3',type=float,default=1500.)
    a.add_argument('--remanence-t',type=float,default=DEFAULT_BR_T)
    a.add_argument('--modulus-kpa',type=float,default=100.)
    a.add_argument('--magnetization-ka-m',type=float,default=60.)
    a.add_argument('--phase-deg',type=float,default=0.)
    a.add_argument('--output-dir',type=Path,default=Path('results/magnetic_rolling'))
    args=a.parse_args()
    if args.distance_mm < 3*SIDE_M*1e3:
        print('WARNING: point-dipole approximation may be inaccurate this close to the cube.')
    p=SheetParameters(youngs_modulus_pa=args.modulus_kpa*1e3,
                      magnetization_a_per_m=args.magnetization_ka_m*1e3,
                      profile_phase_deg=args.phase_deg)
    settings=DriveSettings(magnet_turn_deg=args.magnet_turn_deg,
                            duration_s=args.duration_s,n_steps=args.steps,
                            magnet_distance_mm=args.distance_mm,
                            damping_n_m_s=args.damping_nm_s,
                            resistance_n_m=args.resistance_nm,
                            density_kg_m3=args.density_kg_m3,
                            remanence_t=args.remanence_t)
    r=simulate_magnetic_rolling(p,settings)
    path=save_results(r,args.output_dir)
    print(f'Commanded magnet rotation: {args.magnet_turn_deg:.1f} deg over {args.duration_s:.2f} s')
    print(f'Simulated body rotation: {r["body_angle_deg"][-1]:.2f} deg')
    print(f'Idealized no-slip displacement: {r["travel_mm"]:.3f} mm')
    print(f'Maximum |generalized torque|: {np.max(np.abs(r["generalized_torque_n_m"])):.3e} N*m')
    print(f'Minimum centerline height: {np.min(r["poses_xz_m"][:,:,1])*1000:.4f} mm')
    print(f'Saved: {path.resolve()} and plots in {args.output_dir.resolve()}')
    print('PRELIMINARY: frozen shape, assumed drag/contact and no slip. Physical motion NOT validated.')


if __name__=='__main__':
    main()
