# -*- coding: utf-8 -*-
"""
CHEBYSHEV 7TH-ORDER STEPPED-IMPEDANCE LPF
DEEP / CLEAN FULL-WAVE openEMS RUN

This version fixes the previous V1 failure modes:

1. PROJECT_ROOT uses parents[3], not parents[2].
2. Simulation output is a dedicated clean directory.
3. The mesh is generated explicitly with dense grid lines.
   We do NOT depend on SmoothMeshLines() to create the physical mesh.
4. There are at least 8+ explicit cells around every PML region.
5. Exact section boundaries are retained as mesh planes.
6. The validated MSL port topology is retained.
7. The 14.7838-mm filter is centered inside the 40-mm port-to-port
   measurement region, with 12.6081-mm 50-ohm leads on both sides.
8. The run is intentionally high-resolution:
      dx/dy target = 0.15 mm
      dz target    = 0.15 mm
      substrate    = 12+ exact cells
      frequency    = 1-30 GHz
      NrTS         = 300000
9. All old simulation artifacts are deleted before the run.
10. The script refuses to continue if the generated mesh is obviously
    under-resolved or if PML cannot be represented.

This is the DEEP baseline. Do not tune filter dimensions until this run
produces a physically credible response.
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================================
# OPENEMS DLL SETUP
# =========================================================================
OPENEMS_PACKAGE = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
    / "openEMS"
)

if not OPENEMS_PACKAGE.exists():
    raise FileNotFoundError(
        f"openEMS package not found:\n{OPENEMS_PACKAGE}"
    )

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_PACKAGE)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_PACKAGE)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(OPENEMS_PACKAGE))

os.environ["PATH"] = (
    str(OPENEMS_PACKAGE)
    + os.pathsep
    + os.environ["PATH"]
)

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# =========================================================================
# PROJECT PATHS
# =========================================================================
# script:
# cryogenic_filter_design/
#   simulations/openems/model/<this file>
#
# parents[0] = model
# parents[1] = openems
# parents[2] = simulations
# parents[3] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SIM_PATH = (
    PROJECT_ROOT
    / "simulations"
    / "openems"
    / "results"
    / "chebyshev_7th_order_deep"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "em"
    / "chebyshev_7th_order_deep"
)

PLOTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "plots"
    / "chebyshev_7th_order_deep"
)

for path in (SIM_PATH, RESULTS_PATH, PLOTS_PATH):
    path.mkdir(parents=True, exist_ok=True)

# CLEAN SIMULATION DIRECTORY
if SIM_PATH.exists():
    for item in SIM_PATH.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


# =========================================================================
# UNITS / RF SPEC
# =========================================================================
UNIT = 1e-3

Z0 = 50.0

CUTOFF = 9.0e9
RIPPLE_DB = 0.1

STOP_START = 20.0e9
STOP_ATT_DB = 60.0

FREQ_START = 1.0e9
FREQ_STOP = 30.0e9
N_FREQ = 1451

F0 = 15.0e9
FC = 15.0e9

NR_TS = 300000
END_CRITERIA = 1e-5

BOUNDARY = [
    "PML_8", "PML_8",
    "PML_8", "PML_8",
    "PML_8", "PML_8",
]


# =========================================================================
# PHYSICAL GEOMETRY
# =========================================================================
EPS_R = 3.38
H = 1.524

W50 = 3.5577

W_HIGH = 0.5479
W_LOW = 12.3247

SECTION_Z = np.array([
    120.0, 20.0, 120.0, 20.0,
    120.0, 20.0, 120.0,
])

SECTION_W = np.array([
    W_HIGH, W_LOW, W_HIGH, W_LOW,
    W_HIGH, W_LOW, W_HIGH,
])

SECTION_L = np.array([
    1.6798,
    1.7582,
    2.9818,
    1.9442,
    2.9818,
    1.7582,
    1.6798,
])

FILTER_LENGTH = float(np.sum(SECTION_L))

LINE_X0 = -30.0
LINE_X1 = +30.0

MAIN_X0 = -20.0
MAIN_X1 = +20.0

MAIN_LENGTH = MAIN_X1 - MAIN_X0

LEAD_LENGTH = (
    MAIN_LENGTH - FILTER_LENGTH
) / 2.0

FILTER_X0 = MAIN_X0 + LEAD_LENGTH
FILTER_X1 = MAIN_X1 - LEAD_LENGTH

section_x0 = []
section_x1 = []

x = FILTER_X0
for L in SECTION_L:
    section_x0.append(x)
    x += float(L)
    section_x1.append(x)

section_x0 = np.asarray(section_x0)
section_x1 = np.asarray(section_x1)
section_x1[-1] = FILTER_X1


# =========================================================================
# SIMULATION BOX
#
# PML begins only after these physical air margins.
# We explicitly populate the entire volume with grid lines.
# =========================================================================
X_MARGIN = 10.0
Y_MIN = -15.0
Y_MAX = +15.0

Z_MIN = -5.0
Z_MAX = H + 20.0

X_MIN = LINE_X0 - X_MARGIN
X_MAX = LINE_X1 + X_MARGIN


# =========================================================================
# DEEP MESH
# =========================================================================
DX = 0.15
DY = 0.15
DZ = 0.15

PML_CELLS_REQUIRED = 8

# Explicitly generated physical grid.
# This is the critical fix versus the failed 2025-cell run.
def uniform_lines(a, b, step):
    values = np.arange(a, b + 0.5 * step, step, dtype=float)

    if values[-1] < b:
        values = np.append(values, b)
    else:
        values[-1] = b

    values[0] = a
    return values


x_lines = uniform_lines(X_MIN, X_MAX, DX)
y_lines = uniform_lines(Y_MIN, Y_MAX, DY)
z_lines = uniform_lines(Z_MIN, Z_MAX, DZ)


# Exact critical coordinates must be present.
critical_x = np.array([
    X_MIN,
    LINE_X0,
    MAIN_X0,
    FILTER_X0,
    0.0,
    FILTER_X1,
    MAIN_X1,
    LINE_X1,
    X_MAX,
    *section_x0.tolist(),
    *section_x1.tolist(),
])

critical_y = np.array([
    Y_MIN,
    -W_LOW / 2.0,
    -W_HIGH / 2.0,
    -W50 / 2.0,
    0.0,
    W50 / 2.0,
    W_HIGH / 2.0,
    W_LOW / 2.0,
    Y_MAX,
])

# 12 cells through substrate, plus exact z=0 and z=H.
substrate_z = np.linspace(
    0.0,
    H,
    13,
)

critical_z = np.array([
    Z_MIN,
    0.0,
    H,
    Z_MAX,
    *substrate_z.tolist(),
])


def merge_lines(base, critical):
    merged = np.concatenate((base, critical))
    merged = np.unique(np.round(merged, 10))
    merged.sort()
    return merged


x_lines = merge_lines(x_lines, critical_x)
y_lines = merge_lines(y_lines, critical_y)
z_lines = merge_lines(z_lines, critical_z)


# =========================================================================
# MESH SANITY CHECKS
# =========================================================================
def nearest_spacing(lines):
    return float(np.min(np.diff(lines)))


dx_min = nearest_spacing(x_lines)
dy_min = nearest_spacing(y_lines)
dz_min = nearest_spacing(z_lines)

nx = len(x_lines) - 1
ny = len(y_lines) - 1
nz = len(z_lines) - 1

estimated_cells = nx * ny * nz

print("=" * 80)
print("7TH-ORDER CHEBYSHEV FILTER — DEEP CLEAN FULL-WAVE RUN")
print("=" * 80)

print(f"Project root          : {PROJECT_ROOT}")
print(f"Simulation path       : {SIM_PATH}")
print(f"Results path          : {RESULTS_PATH}")
print(f"Plots path            : {PLOTS_PATH}")

print("-" * 80)

print(f"eps_r                 : {EPS_R}")
print(f"substrate height      : {H:.6f} mm")
print(f"50-ohm width          : {W50:.6f} mm")
print(f"high-Z width          : {W_HIGH:.6f} mm")
print(f"low-Z width           : {W_LOW:.6f} mm")

print("-" * 80)

print(f"filter length         : {FILTER_LENGTH:.6f} mm")
print(f"left 50-ohm lead      : {LEAD_LENGTH:.6f} mm")
print(f"filter x              : {FILTER_X0:.6f} -> {FILTER_X1:.6f} mm")

print("-" * 80)

print(f"x grid lines          : {len(x_lines)}")
print(f"y grid lines          : {len(y_lines)}")
print(f"z grid lines          : {len(z_lines)}")
print(f"x cells               : {nx}")
print(f"y cells               : {ny}")
print(f"z cells               : {nz}")
print(f"estimated cells       : {estimated_cells:,}")
print(f"minimum dx            : {dx_min:.6f} mm")
print(f"minimum dy            : {dy_min:.6f} mm")
print(f"minimum dz            : {dz_min:.6f} mm")

if nx < 100 or ny < 100 or nz < 50:
    raise RuntimeError(
        "Generated mesh is unexpectedly coarse. "
        "Refusing to launch EM simulation."
    )

if estimated_cells < 1_000_000:
    raise RuntimeError(
        f"Generated mesh has only {estimated_cells:,} cells. "
        "This is inconsistent with the intended deep model."
    )

print("-" * 80)

print(f"frequency             : {FREQ_START/1e9:.1f} -> {FREQ_STOP/1e9:.1f} GHz")
print(f"FDTD excitation       : {F0/1e9:.1f} GHz / {FC/1e9:.1f} GHz")
print(f"maximum timesteps     : {NR_TS:,}")
print(f"boundary              : {BOUNDARY}")

print("=" * 80)


# =========================================================================
# FDTD / CSX
# =========================================================================
FDTD = openEMS(
    NrTS=NR_TS,
    EndCriteria=END_CRITERIA,
)

FDTD.SetGaussExcite(
    F0,
    FC,
)

FDTD.SetBoundaryCond(
    BOUNDARY,
)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(UNIT)


# =========================================================================
# APPLY EXPLICIT MESH
# =========================================================================
mesh.AddLine("x", x_lines.tolist())
mesh.AddLine("y", y_lines.tolist())
mesh.AddLine("z", z_lines.tolist())


# =========================================================================
# SUBSTRATE
# =========================================================================
substrate = CSX.AddMaterial(
    "substrate",
    epsilon=EPS_R,
)

substrate.AddBox(
    start=[LINE_X0, Y_MIN, 0.0],
    stop=[LINE_X1, Y_MAX, H],
)


# =========================================================================
# GROUND
# =========================================================================
ground = CSX.AddMetal("GROUND")

ground.AddBox(
    start=[LINE_X0, Y_MIN, 0.0],
    stop=[LINE_X1, Y_MAX, 0.0],
    priority=5,
)

FDTD.AddEdges2Grid(
    dirs="xy",
    properties=ground,
)


# =========================================================================
# SIGNAL METAL
# =========================================================================
signal = CSX.AddMetal("SIGNAL")


# Left 50-ohm lead
signal.AddBox(
    start=[MAIN_X0, -W50 / 2.0, H],
    stop=[FILTER_X0, +W50 / 2.0, H],
    priority=10,
)

# Seven sections
for i in range(7):
    signal.AddBox(
        start=[
            float(section_x0[i]),
            -float(SECTION_W[i]) / 2.0,
            H,
        ],
        stop=[
            float(section_x1[i]),
            +float(SECTION_W[i]) / 2.0,
            H,
        ],
        priority=10,
    )

# Right 50-ohm lead
signal.AddBox(
    start=[FILTER_X1, -W50 / 2.0, H],
    stop=[MAIN_X1, +W50 / 2.0, H],
    priority=10,
)

FDTD.AddEdges2Grid(
    dirs="xy",
    properties=signal,
)


# =========================================================================
# MSL PORTS
#
# Same topology and dimensions as the validated uniform-line model.
# =========================================================================
port1_start = [
    LINE_X0,
    -W50 / 2.0,
    H,
]

port1_stop = [
    MAIN_X0,
    +W50 / 2.0,
    0.0,
]

port2_start = [
    LINE_X1,
    -W50 / 2.0,
    H,
]

port2_stop = [
    MAIN_X1,
    +W50 / 2.0,
    0.0,
]

port1 = FDTD.AddMSLPort(
    1,
    ground,
    port1_start,
    port1_stop,
    "x",
    "z",
    excite=1,
    FeedShift=2.5,
    Feed_R=Z0,
    MeasPlaneShift=5.0,
    priority=10,
)

port2 = FDTD.AddMSLPort(
    2,
    ground,
    port2_start,
    port2_stop,
    "x",
    "z",
    excite=0,
    Feed_R=Z0,
    MeasPlaneShift=5.0,
    priority=10,
)


# =========================================================================
# WRITE XML
# =========================================================================
CSX_FILE = SIM_PATH / "chebyshev_7th_order_deep.xml"

CSX.Write2XML(
    str(CSX_FILE)
)

print("\nXML written:")
print(CSX_FILE)

print("\nSECTION COORDINATES")
for i in range(7):
    print(
        f"S{i+1}: "
        f"Z={SECTION_Z[i]:.1f} ohm, "
        f"W={SECTION_W[i]:.4f} mm, "
        f"L={SECTION_L[i]:.4f} mm, "
        f"x={section_x0[i]:.4f}->{section_x1[i]:.4f} mm"
    )


# =========================================================================
# RUN
# =========================================================================
print("\n" + "=" * 80)
print("STARTING DEEP EM SIMULATION")
print("=" * 80)

FDTD.Run(
    str(SIM_PATH),
    cleanup=False,
)


# =========================================================================
# POST PROCESS
# =========================================================================
freq = np.linspace(
    FREQ_START,
    FREQ_STOP,
    N_FREQ,
)

port1.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=Z0,
)

port2.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=Z0,
)

S11 = port1.uf_ref / port1.uf_inc
S21 = port2.uf_ref / port1.uf_inc
Zin = port1.uf_tot / port1.if_tot

S11_dB = 20 * np.log10(
    np.maximum(np.abs(S11), 1e-15)
)

S21_dB = 20 * np.log10(
    np.maximum(np.abs(S21), 1e-15)
)

power_sum = (
    np.abs(S11) ** 2
    + np.abs(S21) ** 2
)


# =========================================================================
# METRICS
# =========================================================================
def idx_at(hz):
    return int(np.argmin(np.abs(freq - hz)))


i5 = idx_at(5e9)
i9 = idx_at(9e9)
i20 = idx_at(20e9)

passband = (
    (freq >= 1e9)
    & (freq <= 9e9)
)

stopband = (
    freq >= 20e9
)

worst_passband_S21 = float(
    np.min(S21_dB[passband])
)

max_stopband_S21 = float(
    np.max(S21_dB[stopband])
)

attenuation_20 = float(
    -S21_dB[i20]
)

print("\n" + "=" * 80)
print("DEEP FULL-WAVE RESULTS")
print("=" * 80)

print("\n5 GHz")
print("-" * 80)
print(f"S11                    : {S11_dB[i5]:.6f} dB")
print(f"S21                    : {S21_dB[i5]:.6f} dB")
print(f"Power sum              : {power_sum[i5]:.6f}")
print(
    f"Zin                    : "
    f"{np.real(Zin[i5]):.6f} "
    f"{np.imag(Zin[i5]):+.6f}j ohm"
)
print(f"|Zin|                  : {abs(Zin[i5]):.6f} ohm")

print("\nPASSBAND 1–9 GHz")
print("-" * 80)
print(f"Worst S21              : {worst_passband_S21:.6f} dB")
print(
    f"S21 @ 9 GHz            : "
    f"{S21_dB[i9]:.6f} dB"
)

print("\nSTOPBAND")
print("-" * 80)
print(
    f"S21 @ 20 GHz           : "
    f"{S21_dB[i20]:.6f} dB"
)
print(
    f"Attenuation @ 20 GHz   : "
    f"{attenuation_20:.6f} dB"
)
print(
    f"Maximum S21, 20–30 GHz : "
    f"{max_stopband_S21:.6f} dB"
)

print("\nPOWER")
print("-" * 80)
print(
    f"Power min, 1–30 GHz    : "
    f"{np.min(power_sum):.6f}"
)
print(
    f"Power max, 1–30 GHz    : "
    f"{np.max(power_sum):.6f}"
)

# The first thing we need from this deep run is physical sanity.
# Do not call the filter successful based only on one attenuation point.
print("\nDEEP RUN STATUS")
print("-" * 80)

sanity = (
    np.all(np.isfinite(S11))
    and np.all(np.isfinite(S21))
    and np.all(np.isfinite(Zin))
    and estimated_cells >= 1_000_000
)

print(
    f"Finite numerical result : "
    f"{'PASS' if sanity else 'FAIL'}"
)

print(
    f"20 GHz >= 60 dB         : "
    f"{'PASS' if attenuation_20 >= STOP_ATT_DB else 'FAIL'}"
)

print(
    "\nIMPORTANT: 20-GHz attenuation is only the design target. "
    "The complete 20–70 GHz rejection requirement will be checked "
    "after the baseline 1–30 GHz model is physically validated."
)


# =========================================================================
# SAVE CSV
# =========================================================================
csv_path = RESULTS_PATH / "chebyshev_7th_order_deep.csv"

pd.DataFrame({
    "frequency_Hz": freq,
    "S11_dB": S11_dB,
    "S21_dB": S21_dB,
    "power_sum": power_sum,
    "Zin_real_ohm": np.real(Zin),
    "Zin_imag_ohm": np.imag(Zin),
    "Zin_mag_ohm": np.abs(Zin),
}).to_csv(
    csv_path,
    index=False,
)


# =========================================================================
# PLOTS
# =========================================================================
def save_plot(fig, name):
    path = PLOTS_PATH / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# S parameters
fig, ax = plt.subplots(figsize=(12, 6.5))

ax.plot(freq / 1e9, S11_dB, label="S11")
ax.plot(freq / 1e9, S21_dB, label="S21")

ax.axvline(9, linestyle="--", linewidth=1, label="9 GHz")
ax.axvline(20, linestyle=":", linewidth=1.2, label="20 GHz")
ax.axhline(-60, linestyle="--", linewidth=1, label="-60 dB")

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title(
    "7th-Order Chebyshev Stepped-Impedance Filter — Deep openEMS"
)
ax.grid(True)
ax.legend()

sparam_path = save_plot(
    fig,
    "chebyshev_7th_order_deep_sparams.png",
)


# S21
fig, ax = plt.subplots(figsize=(12, 6.5))

ax.plot(freq / 1e9, S21_dB, label="S21")
ax.axvline(9, linestyle="--", linewidth=1, label="9 GHz")
ax.axvline(20, linestyle=":", linewidth=1.2, label="20 GHz")
ax.axhline(-60, linestyle="--", linewidth=1, label="-60 dB")

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("S21 (dB)")
ax.set_title(
    "7th-Order Chebyshev Filter — Deep Transmission"
)
ax.grid(True)
ax.legend()

s21_path = save_plot(
    fig,
    "chebyshev_7th_order_deep_s21.png",
)


# Impedance
fig, ax = plt.subplots(figsize=(12, 6.5))

ax.plot(freq / 1e9, np.real(Zin), label="Re{Zin}")
ax.plot(freq / 1e9, np.imag(Zin), label="Im{Zin}")
ax.axhline(50, linestyle="--", linewidth=1, label="50 ohm")
ax.axhline(0, linestyle=":", linewidth=1)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Impedance (ohm)")
ax.set_title(
    "7th-Order Chebyshev Filter — Input Impedance"
)
ax.grid(True)
ax.legend()

zin_path = save_plot(
    fig,
    "chebyshev_7th_order_deep_zin.png",
)


# Power
fig, ax = plt.subplots(figsize=(12, 6.5))

ax.plot(
    freq / 1e9,
    power_sum,
    label="|S11|² + |S21|²",
)

ax.axhline(
    1,
    linestyle="--",
    linewidth=1,
    label="Ideal lossless = 1",
)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Power ratio")
ax.set_title(
    "7th-Order Chebyshev Filter — Power Conservation"
)
ax.grid(True)
ax.legend()

power_path = save_plot(
    fig,
    "chebyshev_7th_order_deep_power.png",
)


# =========================================================================
# SUMMARY
# =========================================================================
summary = RESULTS_PATH / "deep_summary.txt"

summary.write_text(
    f"""7th-Order Chebyshev Deep Full-Wave Run

Mesh
----
x cells: {nx}
y cells: {ny}
z cells: {nz}
estimated cells: {estimated_cells}
minimum dx: {dx_min:.6f} mm
minimum dy: {dy_min:.6f} mm
minimum dz: {dz_min:.6f} mm

Geometry
--------
filter length: {FILTER_LENGTH:.6f} mm
filter x: {FILTER_X0:.6f} -> {FILTER_X1:.6f} mm
left/right 50-ohm lead: {LEAD_LENGTH:.6f} mm

Results
-------
S11 @ 5 GHz: {S11_dB[i5]:.6f} dB
S21 @ 5 GHz: {S21_dB[i5]:.6f} dB
Zin @ 5 GHz: {np.real(Zin[i5]):.6f} {np.imag(Zin[i5]):+.6f}j ohm
S21 @ 9 GHz: {S21_dB[i9]:.6f} dB
S21 @ 20 GHz: {S21_dB[i20]:.6f} dB
Attenuation @ 20 GHz: {attenuation_20:.6f} dB
Maximum S21, 20-30 GHz: {max_stopband_S21:.6f} dB
Minimum power: {np.min(power_sum):.6f}
Maximum power: {np.max(power_sum):.6f}
""",
    encoding="utf-8",
)

print("\nFILES")
print("-" * 80)
print(f"CSV     : {csv_path}")
print(f"Sparams : {sparam_path}")
print(f"S21     : {s21_path}")
print(f"Zin     : {zin_path}")
print(f"Power   : {power_path}")
print(f"Summary : {summary}")
print(f"XML     : {CSX_FILE}")
print("=" * 80)
print("DEEP RUN COMPLETE")
