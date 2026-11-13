# ME 696 — Step 2F: Quasi-static deformation + torque-responsive rolling

## What has changed since Step 2E?

- The soft sheet's **internal curvature is recalculated** at each magnet orientation, rather than using one rigid, frozen shape.
- The robot's overall body orientation still evolves from the **negative derivative of magnetic + gravitational potential** through the Step 2E overdamped angular-rate model.
- The updated shape affects the torque and ground placement; the resulting body rotation affects the next equilibrium calculation.
- A rotational **no-slip approximation** is used to estimate how the sheet's center moves. It does *not* include the contribution of joint deformation to tangential contact velocity, so this is **not** a full no-slip or stick/slip solver.

The approximate sequence is:

`commanded cube-magnet angle → dipole field → elastic-joint equilibrium on floor → generalized torque → body rotation → rotational-only rolling displacement → repeat`

The paper-reference sample dimensions (3.7 mm x 1.5 mm x 0.185 mm), elastic modulus, magnetization, phase, density, damping and resistance are **not measured values from your sample**. Do not call these plots experimental or validated.

## Installation

From inside your existing `ME696-Magnetic-Soft-Robot/` directory:

```bash
unzip -o ~/Downloads/ME696_Step_2F_Coupled_Deformation_Addon.zip -d .
python -m simulator.models.coupled_rolling
python -m unittest discover -s simulator/tests -v
```

Expected: **28 tests pass**, including the original 22. The new files are placed in the existing project folders; do not run from a separate add-on directory.

To try a different magnet command:

```bash
python -m simulator.models.coupled_rolling --magnet-turn-deg -360 --steps 61 \
  --output-dir results/coupled_rolling/reverse
```

For a finer time step (slower run):

```bash
python -m simulator.models.coupled_rolling --steps 121 \
  --output-dir results/coupled_rolling/fine
```

For a video (requires FFmpeg, `brew install ffmpeg` on macOS if missing):

```bash
python -m simulator.animate_coupled
```

Outputs in `results/coupled_rolling/`:

- `coupled_rolling.csv` — commanded angle, body response, end-to-end length, torque, approximate displacement
- `coupled_summary.png` — angle, deformation, and displacement vs time
- `coupled_poses.png` — selected configurations above the floor
- `coupled_rolling_preview.mp4` — optional visual demonstration, simulated only

## Interpreting the difference from Step 2E

Under the **illustrative 60 mm** reference setup, Step 2E predicted approximately one body turn because it *froze* a curled shape computed from the unconstrained free-sheet model. Step 2F instead enforces a floor placement and lets curvature relax at each time step. With the assumed parameters, the body may rotate only a small amount, which is a **model outcome, not proof of physical accuracy**. Directly comparing the numerical distances as if only one physical effect had changed would be misleading: Step 2F also changes the pose parametrization and contact treatment.

## Known limitations and work still required

1. The actual robot's dimensions, stiffness, magnetization amplitude and profile have **not** been measured or verified.
2. Cube field is a far-field **point-dipole approximation**.
3. Sheet shape is an **instantaneous local energy minimum** (no shape inertia or damping).
4. Ground-contact geometry uses the **lowest discretization node** and ignores self-contact and distributed contact mechanics.
5. Rotational damping/resistance are assumed; friction and rolling feasibility are not checked.
6. Horizontal travel accounts for the **rotation-only** contact velocity; tangential contributions due to shape changes are omitted.
7. Numerical tests verify consistency and invariants, **not** experimental prediction accuracy. The design must still be validated against physical motion videos and measurements.

**Next critical task:** test the real sample and magnet, record deformation and rolling, then calibrate and improve the physical model using those measurements. The course also requires a simulation-only obstacle traversal with height at least 10 times the body thickness, an interactive 3D browser simulator, and graduate-level validation/refinement.

## GitHub checkpoint

```bash
git add simulator/models/coupled_rolling.py simulator/tests/test_coupled_rolling.py \
        simulator/animate_coupled.py STEP_2F_README.md results/coupled_rolling/
git commit -m "Add Step 2F quasi-static coupled deformation and rolling"
git push
```
