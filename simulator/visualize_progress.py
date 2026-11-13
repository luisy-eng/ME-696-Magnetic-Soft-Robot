"""Render a narrated-by-labels MP4 of implemented ME696 Steps 2A--2D.

Run from repository root:
    python -m simulator.visualize_progress
    python -m simulator.visualize_progress --output results/video/ME696_Steps_2A_to_2D.mp4

Only the actual equations implemented in the repository are used:
  A. N52 cube POINT DIPOLE magnetic field (not a finite-cube field).
  B. Independent equilibrium snapshots of a planar magneto-elastic chain,
     with assumed geometry/material/magnetization parameters.
  C. Frozen-shape, imposed rotation, ideal no-slip rolling geometry.

The video is NOT a time-domain / dynamic magnetic actuation prediction,
NOT a measured experiment, and NOT validated against the physical sample.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np

from .models.magnetic_field import (
    SIDE_M, DEFAULT_BR_T, HU_SHEET_LENGTH_M, dipole_field, magnet_moment
)
from .models.deformation import SheetParameters, solve_equilibrium
from .models.contact_rolling import RollingSettings, normalize_centerline, roll_kinematically

# These are aesthetic choices for rendering, not model assumptions.
BG = '#0b1524'
PANEL = '#14243a'
WHITE = '#f5f7fb'
MUTED = '#a1b4ce'
CYAN = '#4cbbff'
TEAL = '#49e2c3'
ORANGE = '#ffb45b'
GRID = '#35506b'


def style_axis(ax, xlabel='', ylabel='', equal=False):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color('#4c6682')
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.35, linestyle='--')
    if equal:
        ax.set_aspect('equal', adjustable='box')


def chapter(fig, number, title, subtitle, disclaimer):
    fig.clf()
    fig.patch.set_facecolor(BG)
    fig.text(.055, .938, 'ME 696  /  MAGNETIC SOFT ROBOT', color=TEAL,
             fontsize=11, weight='bold')
    fig.text(.055, .874, f'{number}  /  {title}', color=WHITE,
             fontsize=22, weight='bold')
    fig.text(.055, .826, subtitle, color=MUTED, fontsize=12)
    fig.text(.055, .035, disclaimer, color=MUTED, fontsize=9)
    fig.text(.945, .035, 'PRELIMINARY MODEL  •  NO EXPERIMENTAL VALIDATION',
             color=ORANGE, fontsize=8, ha='right', weight='bold')


def render(output: Path, fps: int = 12, seconds: tuple[int,int,int] = (5, 5, 8)):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    params = SheetParameters()
    # Exact same reduced-order equilibrium solve as Step 2B.
    angles = (0, 45, 90, 135)
    print('Solving Step 2B equilibrium snapshots (0°, 45°, 90°, 135°)...', flush=True)
    solutions = {a: solve_equilibrium(params, distance_mm=60.0, angle_deg=a)
                 for a in angles}
    # Exact same Step 2D kinematic model / settings as in the working demo.
    shape = normalize_centerline(solutions[0]['nodes_m'])
    rolling = roll_kinematically(shape, RollingSettings(n_steps=721,
                              rotation_deg=360.0, thickness_m=params.thickness_m))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = FFMpegWriter(fps=fps, bitrate=2400,
                          metadata={'title': 'ME696 Steps 2A–2D preliminary simulation',
                                    'comment': 'Assumed parameters; rolling rotation imposed; not experimental validation'})
    print('Rendering:', output.resolve(), flush=True)
    with writer.saving(fig, str(output), dpi=100):
        # SCENE 1: the actual step-2A point-dipole vector field.
        chapter(fig, '01', 'EXTERNAL MAGNETIC FIELD',
                'Cube magnet orientation changes the computed B-field at the robot plane.',
                'STEP 2A  •  N52 cube approximated by a point dipole  •  robot plane z = 60 mm')
        ax1 = fig.add_axes([.065, .16, .48, .59])
        ax2 = fig.add_axes([.615, .20, .33, .50])
        style_axis(ax1, 'x (mm)', 'z (mm)', equal=True)
        style_axis(ax2, 'Position along reference sheet (mm)', 'B (mT)')
        ax1.set(xlim=(-75,75), ylim=(-70,95))
        ax2.set(xlim=(-1.85,1.85), ylim=(-18,18))
        x = np.linspace(-70, 70, 67)
        z = np.linspace(-68, 90, 75)
        X, Z = np.meshgrid(x,z)
        p = np.stack((X/1000, np.zeros_like(X), Z/1000), axis=-1)
        radial = np.hypot(X,Z)
        # The point-dipole center / near field is omitted deliberately.
        valid = radial >= 30
        safe = p.copy()
        safe[~valid] = [0.,0.,.03]
        mesh = ax1.imshow(np.zeros_like(X), origin='lower',
                          extent=(x[0],x[-1],z[0],z[-1]),
                          cmap='viridis', vmin=0, vmax=18, aspect='equal')
        # Vector direction markers at a coarse grid.
        xq = np.linspace(-68,68,13)
        zq = np.linspace(-65,88,13)
        Xq,Zq = np.meshgrid(xq,zq)
        qpos = np.stack((Xq/1000,np.zeros_like(Xq),Zq/1000),axis=-1)
        qvalid = np.hypot(Xq,Zq) > 30
        qsafe = qpos.copy(); qsafe[~qvalid] = [0.,0.,.03]
        qv = ax1.quiver(Xq,Zq,np.zeros_like(Xq),np.ones_like(Xq),
                         color=WHITE,alpha=.65,scale=24,width=.003)
        ax1.add_patch(Rectangle((-SIDE_M*500,-SIDE_M*500),
                       SIDE_M*1000,SIDE_M*1000,fill=False,
                       edgecolor=WHITE,lw=1.5))
        ax1.add_patch(Circle((0,0),30,fill=False,ls='--',lw=1,
                             edgecolor=ORANGE,alpha=.8))
        ax1.plot([-HU_SHEET_LENGTH_M*500,HU_SHEET_LENGTH_M*500],
                 [60,60],color=TEAL,lw=5,solid_capstyle='round')
        ax1.annotate('reference robot plane',xy=(3,60),xytext=(16,76),
                     color=TEAL,fontsize=9,arrowprops={'arrowstyle':'->','color':TEAL})
        xp = np.linspace(-HU_SHEET_LENGTH_M*500,HU_SHEET_LENGTH_M*500,101)
        profile_p = np.column_stack((xp/1000,np.zeros_like(xp),np.full_like(xp,.06)))
        line_bx, = ax2.plot([],[],color=CYAN,lw=2,label=r'$B_x$')
        line_bz, = ax2.plot([],[],color=ORANGE,lw=2,label=r'$B_z$')
        line_bmag, = ax2.plot([],[],color=TEAL,lw=2,ls='--',label=r'$|B|$')
        ax2.legend(loc='upper right',facecolor=PANEL,labelcolor=WHITE,edgecolor=GRID)
        angle_text = ax2.text(.04,.08,'',transform=ax2.transAxes,color=WHITE,
                              fontsize=14,weight='bold')
        n = seconds[0]*fps
        for i in range(n):
            angle = 180*i/max(n-1,1)
            mom = magnet_moment(angle)
            field = dipole_field(safe,moment_A_m2=mom)
            magnitude = np.linalg.norm(field,axis=-1)*1000
            magnitude[~valid] = np.nan
            mesh.set_data(magnitude)
            bq = dipole_field(qsafe,moment_A_m2=mom)
            qnorm=np.maximum(np.hypot(bq[...,0],bq[...,2]),1e-20)
            qu = np.where(qvalid,bq[...,0]/qnorm,0)
            qz = np.where(qvalid,bq[...,2]/qnorm,0)
            qv.set_UVC(qu,qz)
            b = dipole_field(profile_p,moment_A_m2=mom)*1000
            line_bx.set_data(xp,b[:,0]);line_bz.set_data(xp,b[:,2])
            line_bmag.set_data(xp,np.linalg.norm(b,axis=1))
            bc=dipole_field(np.array([0.,0.,.06]),moment_A_m2=mom)
            angle_text.set_text(f'Magnet: {angle:5.1f}°\n|B| at center: {np.linalg.norm(bc)*1000:4.2f} mT')
            writer.grab_frame()
        print('  Scene 1/3 field complete', flush=True)

        # SCENE 2: Step-2B equilibrium solutions, rather than animated dynamics.
        chapter(fig, '02', 'MAGNETO-ELASTIC DEFORMATION',
                'Independent equilibrium shapes at four magnet orientations (not real-time dynamics).',
                'STEP 2B  •  16 flexible segments  •  assumed stiffness, magnetization, and phase')
        ax1 = fig.add_axes([.075,.17,.49,.56])
        ax2 = fig.add_axes([.63,.23,.30,.45])
        style_axis(ax1,'x relative to sheet center (mm)','z relative to sheet center (mm)',equal=True)
        style_axis(ax2,'Magnet orientation (degrees)','End-to-end separation (mm)')
        ax1.set(xlim=(-2.8,2.8),ylim=(-2.65,2.65))
        ax2.set(xlim=(-12,147),ylim=(1.25,3.9))
        ax1.plot([-1.85,1.85],[0,0],ls='--',color=MUTED,lw=2,
                 label='Undeformed (3.7 mm)')
        ax1.plot([0],[0],'+',ms=10,color=ORANGE)
        line_sheet, = ax1.plot([],[],'-o',lw=3.8,ms=3.3,color=TEAL,
                               label='Predicted equilibrium')
        ax1.legend(loc='upper right',facecolor=PANEL,labelcolor=WHITE,edgecolor=GRID,fontsize=9)
        end_dists=[solutions[a]['end_to_end_mm'] for a in angles]
        ax2.plot(angles,end_dists,'o-',lw=1.7,ms=6,color=CYAN,alpha=.7)
        active_dot,=ax2.plot([],[],'o',ms=13,color=ORANGE)
        label=ax2.text(.08,.73,'',transform=ax2.transAxes,color=WHITE,
                       fontsize=14,weight='bold')
        ax2.text(.08,.08,'Points are separately solved\nstatic energy minima.',
                 transform=ax2.transAxes,color=MUTED,fontsize=10)
        n=seconds[1]*fps
        for i in range(n):
            idx=min((i*len(angles))//n, len(angles)-1)
            a=angles[idx];sol=solutions[a]
            nodes=sol['nodes_m']*1000
            line_sheet.set_data(nodes[:,0],nodes[:,2]-60)
            active_dot.set_data([a],[sol['end_to_end_mm']])
            label.set_text(f'Magnet: {a}°\nEnd separation: {sol["end_to_end_mm"]:.3f} mm')
            writer.grab_frame()
        print('  Scene 2/3 equilibrium complete', flush=True)

        # SCENE 3: the exact Step-2D imposed-body-rotation kinematics.
        chapter(fig, '03', 'IDEALIZED NO-SLIP ROLLING',
                'A fixed C-shape is rotated by an imposed body angle. Translation follows contact geometry.',
                'STEP 2D  •  rotation IMPOSED, not generated by magnetic torque  •  no slip, friction, or shape evolution')
        ax1=fig.add_axes([.065,.20,.53,.49])
        ax2=fig.add_axes([.66,.23,.29,.44])
        style_axis(ax1,'x (mm)','height above floor (mm)',equal=True)
        style_axis(ax2,'Imposed BODY rotation (degrees)','x displacement (mm)')
        ax1.set(xlim=(-8.0,3.2),ylim=(-.2,4.5))
        ax2.set(xlim=(0,360),ylim=(-6.2,.7))
        ax1.axhline(0,color=WHITE,lw=1.6,alpha=.7)
        poses=rolling['poses_xz_m']*1000
        ax1.plot(poses[0,:,0],poses[0,:,1],ls='--',alpha=.4,
                 color=CYAN,lw=1.7,label='Starting pose')
        current,=ax1.plot([],[],'-o',lw=4,ms=3.2,color=TEAL,label='Current pose')
        contact,=ax1.plot([],[],'o',ms=7,color=ORANGE,label='Contact node')
        ax1.legend(loc='upper right',facecolor=PANEL,labelcolor=WHITE,
                   edgecolor=GRID,fontsize=9)
        xroll=rolling['x_center_m']*1000
        rollang=rolling['angle_deg']
        ax2.plot(rollang,xroll,color=GRID,lw=2,alpha=.8)
        trajectory,=ax2.plot([],[],color=CYAN,lw=3)
        trajectory_dot,=ax2.plot([],[],'o',color=ORANGE,ms=9)
        info=ax2.text(.05,.08,'',transform=ax2.transAxes,color=WHITE,
                      fontsize=13,weight='bold')
        n=seconds[2]*fps
        for i in range(n):
            j=int(round(i/max(n-1,1)*(len(rollang)-1)))
            xy=poses[j]
            current.set_data(xy[:,0],xy[:,1])
            cxz=rolling['contact_xz_m'][j]*1000
            contact.set_data([cxz[0]],[cxz[1]])
            trajectory.set_data(rollang[:j+1],xroll[:j+1])
            trajectory_dot.set_data([rollang[j]],[xroll[j]])
            info.set_text(f'Rotation: {rollang[j]:.0f}°\nDisplacement: {xroll[j]:+.2f} mm')
            writer.grab_frame()
        print('  Scene 3/3 rolling complete', flush=True)
    plt.close(fig)
    print(f'Video completed: {output}  ({sum(seconds)} s at {fps} fps)')


def main():
    parser=argparse.ArgumentParser(description='Render ME696 steps 2A–2D MP4')
    parser.add_argument('--output',type=Path,default=Path('results/video/ME696_Steps_2A_to_2D.mp4'))
    parser.add_argument('--fps',type=int,default=12)
    args=parser.parse_args()
    if args.fps < 2 or args.fps>60:
        parser.error('fps must be between 2 and 60')
    render(args.output,args.fps)

if __name__=='__main__':
    main()
