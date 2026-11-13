# ME 696 — Topic 1: Initial magnetic field model

This is **Step 2A only**: a point-dipole approximation of the professor-supplied 3/4-inch N52 cube magnet. It does **not** yet model the flexible robot's deformation, rolling, or contact dynamics.

## 1. Setup (macOS or Linux)

```bash
cd ME696-Magnetic-Soft-Robot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Run

```bash
python -m simulator.models.magnetic_field
```

The program prints the magnetic field at the robot plane and saves `results/field_map.png` and `results/sheet_field_profile.png`.

Try rotating the magnetic axis by 90 degrees:

```bash
python -m simulator.models.magnetic_field --angle-deg 90 --distance-mm 80
```

All parameters are explicit: `--distance-mm` (center-to-plane), `--angle-deg` (angle from +z toward +x), and `--br-t` (remanent flux density). Internally, distances are in meters and B is in tesla; plots use millimeters and millitesla.

## 3. Tests

```bash
python -m unittest discover -s simulator/tests -v
```

Tests confirm the on-axis formula, inverse-cube decay, moment reversal, and the center singularity.

## 4. Assumptions and limitations

- The magnet is represented as an ideal **point dipole** at the cube center, not an exact finite-cube magnet. Close to the cube the predicted field can differ substantially from measurements.
- Magnet side is 0.75 inch = 19.05 mm (per ME 696 project description).
- Br = 1.43 T is a representative **assumption** for N52 grade, not calibrated for the supplied cube. The approximate moment is `m = (Br/mu0)*side^3`. Real magnet magnetization can be affected by demagnetizing fields.
- Sample length 3.7 mm is from Hu et al. 2018 Figure 1 and **is not the measured dimension of the class sample**.
- Model depicts magnet centered at the origin, sample lying along x at z = +distance. Moment is assumed +z at 0 degrees. Neither the sample's own magnetization nor its effect on the applied field is included.
- The dashed circle in the field map at 30 mm is **only a warning visualization**; it does not establish a validated accuracy threshold. Use care near the cube. The baseline distance 60 mm is a proposed illustrative setup only.

## 5. References

- ME 491/696 Fall 2026 Final Project Description, Topic 1, Section 4.1 (dipole model explicitly permitted).
- Hu, W., Lum, G. Z., Mastrangeli, M., & Sitti, M. (2018). *Small-scale soft-bodied robot with multimodal locomotion*. Nature 554, 81–85. DOI: 10.1038/nature25443.
- Shin-Etsu Chemical, N52 Standard Characteristics (Br ~1.40–1.47 T): https://www.shinetsu.co.jp/serem/e/download/N52sheet.pdf

## 6. Next steps

1. Verify/record the supplied magnet's actual orientation and the sample dimensions; do not treat the illustrative parameters as measurements.
2. Develop the sheet's harmonic magnetization profile and a reduced-order magneto-elastic deformation model.
3. Measure the cube's field when the laboratory sensor is available, and compare errors at different distances.
4. Eventually develop rolling and obstacle contact dynamics for the ME 696 requirements.
