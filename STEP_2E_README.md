# ME 696 — Step 2E: Torque-responsive Magnetic Rolling

This is an **add-on to the existing** `ME696-Magnetic-Soft-Robot` repository; it is **not** a new standalone project. The ZIP places files directly into the project hierarchy.

## Install on macOS

1. From Terminal, open your **existing** `ME696-Magnetic-Soft-Robot` repository (the one with `simulator/models/magnetic_field.py`, `deformation.py`, and `contact_rolling.py`).
2. From inside that repository run:

   ```bash
   unzip -o ~/Downloads/ME696_Step_2E_Magnetic_Rolling_Addon.zip -d .
   python -m simulator.models.magnetic_rolling
   python -m unittest discover -s simulator/tests -v
   ```

   You should get **22 tests passing** (14 existing + 8 new). Run inside your active Python environment with numpy/scipy/matplotlib installed.

## Example experiments

Base case (magnet rotated 360 degrees over 8 s):

```bash
python -m simulator.models.magnetic_rolling
```

Counter-rotation:

```bash
python -m simulator.models.magnetic_rolling --magnet-turn-deg -360 --output-dir results/magnetic_rolling/reverse
```

Faster actuation (shows lag under assumed damping):

```bash
python -m simulator.models.magnetic_rolling --duration-s 2 --output-dir results/magnetic_rolling/fast
```

Strong phenomenological resistance (may prevent motion):

```bash
python -m simulator.models.magnetic_rolling --resistance-nm 3e-7 --output-dir results/magnetic_rolling/high_resistance
```

Parameters include `--distance-mm`, `--damping-nm-s`, `--density-kg-m3`, `--remanence-t`, `--modulus-kpa`, `--magnetization-ka-m`, and `--phase-deg`. Use `--help` for all inputs.

## Outputs

- `results/magnetic_rolling/drive_vs_body.png`: commanded magnet rotation vs. calculated body rotation; no-slip horizontal motion vs. time
- `results/magnetic_rolling/rolling_poses.png`: successive body poses placed above a floor
- `results/magnetic_rolling/magnetically_driven_rolling.csv`: time, magnet angle, body angle, generalized torque, x and z position

The base illustrative run predicts about **351.70 degrees** of body rotation and **-5.205 mm** travel for a +360-degree magnet command over 8 seconds. These values will change with parameter settings and **are not measured**.

## What changed from Step 2D

Step 2D **prescribed body rotation** and used pure-rolling geometry to calculate translation. Step 2E instead prescribes the **magnet orientation** and integrates the body's rotation under a simplified torque balance:

\[
U(\phi,\alpha) = -\sum_i \mathbf{m}_i(\phi)\cdot \mathbf{B}(\mathbf{r}_i(\phi),\alpha) + Mg\,z_{\mathrm{cm}}(\phi),
\qquad Q_\phi=-\partial U/\partial\phi.
\]

Under a **pre-imposed no-slip constraint** and a **phenomenological overdamped** model, the rotation rate satisfies:

\[
\dot\phi = \operatorname{sgn}(Q_\phi)\frac{\max(|Q_\phi|-T_{\rm resist},0)}{c_\phi}.
\]

The shape is **frozen** from a Step 2B equilibrium, while the field and rigid-body position are recomputed as the body rotates. Numerical differentiation of total potential energy captures magnetic-field gradient effects through the constrained translation. A Heun predictor-corrector integrates the body angle. Horizontal travel is still calculated by the Step 2D no-slip geometry.

## Modeling assumptions and warnings

1. **Magnet field:** point-dipole approximation for the finite N52 cube. Not reliable near cube surfaces. The example uses an assumed 60 mm magnet-center-to-ground separation.
2. **Shape:** frozen, planar, 16-segment equilibrium contour; no deformation during rolling, no contact-induced shape changes, no self-contact.
3. **Magnetization:** hypothetical one-cycle harmonic profile and assumed material parameters. The provided physical sheet's true dimensions, stiffness, magnetization magnitude, and phase are not measured yet.
4. **Contact:** a no-slip kinematic constraint is imposed; no friction feasibility, sliding, sticking transitions, impacts, or contact-force calculations. A fitted resistance torque is **not** a substitute for such tests.
5. **Dynamics:** an overdamped generalized coordinate with assumed damping/resistance (not inertial mechanics); no servo response or hardware field calibration.
6. **Reference conditions:** the material and magnet values in this project are placeholders. Therefore the simulation is an *early predictive-model prototype* and not experimental validation.

## Numerical checks

The eight added tests verify: nonpenetration under rotations, consistency with Step 2D displacement, magnet reversal changes torque sign, magnetic strength scales torque, resistance threshold, zero-field/zero-gravity no-motion case, non-prescribed body motion, and high-resistance locking. Together with previous modules, **22 tests** should pass.

## Commit to GitHub after checking

```bash
git add simulator/models/magnetic_rolling.py simulator/tests/test_magnetic_rolling.py STEP_2E_README.md results/magnetic_rolling/
git commit -m "Add torque-responsive magnetic rolling model and tests"
git push
```

## Next engineering steps

- Add deformation that updates as the magnet rotates; check convergence and possible branch changes.
- Add real contact/friction and then evaluate whether no-slip is feasible.
- Obtain actual robot dimensions and magnetic/material parameters, collect deformation and rolling video under matching magnet settings, and compare to measured motion. These are required for the graduate-level model refinement.
- Implement the separate obstacle-traversal simulation required by the course.
