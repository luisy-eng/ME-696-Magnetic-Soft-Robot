"""Animate Step 2F prototype predictions; not a real-experiment video.

Run from repo root:
  python -m simulator.animate_coupled

Requires ffmpeg installed (e.g. `brew install ffmpeg` on macOS).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

from .models.deformation import SheetParameters
from .models.coupled_rolling import CoupledSettings, simulate_coupled_rolling, save_results


def make_video(result: dict, output_path: Path, fps: int = 18) -> None:
    poses = result['poses_xz_m']*1000
    time = result['time_s']
    fig = plt.figure(figsize=(12.8,7.2), dpi=100)
    grid = fig.add_gridspec(2,2, width_ratios=(1.25,1), hspace=.4, wspace=.3)
    ax_robot = fig.add_subplot(grid[:,0])
    ax_angle = fig.add_subplot(grid[0,1])
    ax_length = fig.add_subplot(grid[1,1])
    fig.suptitle('ME 696 | Step 2F: Changing deformation + magnetic drive',fontsize=16,fontweight='bold')

    ax_robot.axhline(0, color='#454545',lw=2, label='Surface')
    line, = ax_robot.plot([], [], '-o', lw=3, ms=4, color='#158f93',label='Soft sheet centerline')
    ax_robot.set(xlabel='Horizontal position x (mm)',ylabel='Height z (mm)',title='Quasi-static sheet shape on ground')
    xmin = min(-2.3, poses[:,:,0].min()-.35)
    xmax = max(2.3, poses[:,:,0].max()+.35)
    zmax = max(1.55, poses[:,:,1].max()+.25)
    ax_robot.set_xlim(xmin, xmax)
    ax_robot.set_ylim(-.15,zmax)
    ax_robot.set_aspect('equal',adjustable='box'); ax_robot.grid(alpha=.22)
    ax_robot.legend(loc='upper right',fontsize=9)
    label = ax_robot.text(.02,.92,'',transform=ax_robot.transAxes,va='top',fontsize=10,
                          bbox={'facecolor':'white','edgecolor':'0.8','alpha':.85})

    ax_angle.plot(time,result['magnet_angle_deg'],color='#b78728',label='Magnet input')
    ax_angle.plot(time,result['body_angle_deg'],color='#2289b8',label='Body response')
    vline1 = ax_angle.axvline(0,color='0.3',linestyle='--')
    ax_angle.set(xlabel='Time (s)',ylabel='Angle (degrees)',title='Magnet command vs body response')
    ax_angle.grid(alpha=.25); ax_angle.legend(fontsize=9)

    ax_length.plot(time,result['end_to_end_mm'],color='#158f93')
    vline2 = ax_length.axvline(0,color='0.3',linestyle='--')
    ax_length.set(xlabel='Time (s)',ylabel='End-to-end length (mm)',title='Deformation changes with actuation')
    ax_length.grid(alpha=.25)
    ax_length.set_ylim(min(result['end_to_end_mm'])-.1,max(result['end_to_end_mm'])+.1)

    fig.text(.04,.025,'SIMULATION ONLY  |  Point-dipole field, assumed material/drag parameters, approximate floor contact; not experimentally validated.',
             fontsize=10,color='#b14b33')
    fig.subplots_adjust(top=.88,bottom=.105,left=.07,right=.97)

    def update(i):
        line.set_data(poses[i,:,0],poses[i,:,1])
        vline1.set_xdata([time[i],time[i]])
        vline2.set_xdata([time[i],time[i]])
        label.set_text(f't = {time[i]:.2f} s\nMagnet {result["magnet_angle_deg"][i]:+.1f}°\nBody {result["body_angle_deg"][i]:+.1f}°\nEnd-to-end {result["end_to_end_mm"][i]:.2f} mm')
        return line, vline1, vline2, label

    frames = list(range(len(time)))
    animation = FuncAnimation(fig, update, frames=frames,interval=1000/fps,blit=False)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    animation.save(str(output_path), writer=FFMpegWriter(fps=fps, bitrate=2200,
                              extra_args=['-pix_fmt','yuv420p']),dpi=100)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=61)
    ap.add_argument('--fps',type=int,default=10)
    ap.add_argument('--magnet-turn-deg',type=float,default=360.)
    ap.add_argument('--distance-mm',type=float,default=60.)
    ap.add_argument('--output-dir',type=Path,default=Path('results/coupled_rolling'))
    args=ap.parse_args()
    p=SheetParameters()
    s=CoupledSettings(n_steps=args.steps,magnet_turn_deg=args.magnet_turn_deg,
                      magnet_distance_mm=args.distance_mm)
    result=simulate_coupled_rolling(p,s)
    save_results(result,args.output_dir)
    video=args.output_dir/'coupled_rolling_preview.mp4'
    make_video(result,video,fps=args.fps)
    print(f'Saved video: {video.resolve()}')
    print(f'Calculated body rotation: {result["body_angle_deg"][-1]:.2f} deg; '
          f'approx. translation {result["travel_mm"]:.3f} mm')
    print('SIMULATION ONLY: unvalidated coupling and contact assumptions.')


if __name__=='__main__':
    main()
