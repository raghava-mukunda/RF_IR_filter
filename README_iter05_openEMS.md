# Iteration 05 LPF — openEMS FDTD files

## Files

- `iter05_openems_design.xml` — parametric geometry and exact Iteration 05 L/C values. This is the **source/config XML consumed by the Python builder**, not a native solver XML.
- `iter05_openems_fdtd_optimize.py` — builds the CSXCAD model, writes both a geometry-only XML and a self-contained openEMS simulation XML, runs a baseline FDTD simulation, exports S-parameters and optionally runs a small deterministic geometry sweep.

## Important modeling decisions

- Physical filter envelope: 65.3 × 5.0 × 4.6 mm.
- Air cavity: 65.3 × 4.6 × 4.2 mm.
- Center strip: 2.2 mm wide × 0.5 mm thick, with eight 0.1 mm series-L gaps.
- Eight inductors and seven shunt capacitors use the supplied Iteration 05 values. Every shunt capacitor is split C/2 to the top and C/2 to the bottom PEC walls.
- 7 mm uniform feed extensions are added outside the physical 65.3 mm filter at the measurement ports so the PML boundaries do not coincide with port planes. These extensions are simulation padding, not part of the physical product length.
- The initial model uses PEC for the center strip and enclosure to isolate electrical/topological behavior. Finite copper loss and skin depth are not included in this first pass.
- The shunt capacitors are ideal lumped elements spanning from the strip to the enclosure walls. This reproduces the *hybrid RLC abstraction*, not a physical discrete capacitor or a manufacturing-ready design.
- The S2P export infers S12=S21 and S22=S11 from reciprocity and the nominally symmetric geometry. The script directly excites Port 1 and directly simulates S11/S21 only.

## Use on the Windows openEMS installation

The reference script points to an older `openEMS_x64_v0.0.36-93-g7b9cd51_msvc` package. Set `OPENEMS_ROOT` to the folder containing the `openEMS` binary and Python bindings, or pass `--openems-root`.

Start with model generation only (no FDTD compute):

```powershell
.\.venv\Scripts\python.exe .\iter05_openems_fdtd_optimize.py --mode build-only
```

This generates, under `iter05_openems_runs\baseline\`:
- `Iter05_openEMS_geometry.xml`
- `Iter05_openEMS_FDTD.xml`

Inspect the XML in AppCSXCAD first. Then run the baseline:

```powershell
.\.venv\Scripts\python.exe .\iter05_openems_fdtd_optimize.py --mode baseline
```

The default band is 0.1–30 GHz to keep the initial LPF test focused and cheaper. To follow the reference script's 0.1–100 GHz bandwidth, use:

```powershell
.\.venv\Scripts\python.exe .\iter05_openems_fdtd_optimize.py --mode baseline --fmax-ghz 100 --n-freq 2001
```

Only after checking the baseline, explicitly request the width sweep:

```powershell
.\.venv\Scripts\python.exe .\iter05_openems_fdtd_optimize.py --mode optimize --max-evals 5
```

Each candidate is a separate full FDTD run, and the optimization tests only the center-strip width (not all design parameters). Default optimization range: 1.8, 2.0, 2.2, 2.4, and 2.6 mm.

## Version/preflight safety

The script performs an API check for series inductors and `LEtype=1` before building/running. Current openEMS documentation says full series-RLC lumped-element support was added in version 0.37. If the installed 0.0.36 build rejects this preflight, the script will stop before an FDTD solve. Do not bypass the check: the 8 series inductors are essential to this 15th-order ladder.

This script was syntax-checked and the parameter XML was parsed in the authoring environment, but openEMS/CSXCAD were not installed there. No FDTD result or optimization result is claimed until the script is run on the target Windows installation.
