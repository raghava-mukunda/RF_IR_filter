# -*- coding: utf-8 -*-
"""V14 — 7th-order Chebyshev stepped-impedance filter, stopband tuning.

V14 improved the cutoff behavior substantially, but the physical filter still
does not provide the required stopband attenuation. V14 therefore restores a
stronger stepped-impedance contrast, using the V12 high/low widths, while
shortening the entire stepped section to move the cutoff back upward toward
9 GHz.

V14 physical widths:
    50 ohm reference line : 3.50 mm
    HIGH-Z section        : 0.50 mm
    LOW-Z section         : 12.25 mm

V14 section lengths, all aligned to the 0.25-mm EM mesh:
    [0.75, 1.00, 1.75, 1.25, 1.75, 1.00, 0.75] mm

Total stepped filter length:
    8.25 mm

Symmetry is preserved. The validated MSL/PML/mesh topology from V12/V14 is
kept unchanged except for the physical filter geometry.

This is the final stopband-tuning EM experiment before LC extraction. The
geometry is NOT frozen until the measured full-wave response satisfies the
design acceptance criteria.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Windows openEMS native DLL setup
# ---------------------------------------------------------------------
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

print(f"openEMS native DLL path : {OPENEMS_PACKAGE}")

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SIM_PATH = (
    PROJECT_ROOT
    / "simulations"
    / "openems"
    / "results"
    / "chebyshev_v14"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "em"
    / "chebyshev_v14"
)

PLOTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "plots"
)

for path in (SIM_PATH, RESULTS_PATH, PLOTS_PATH):
    path.mkdir(parents=True, exist_ok=True)

# Clean only this V12 run directory.
if SIM_PATH.exists():
    for p in SIM_PATH.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


# =====================================================================
# ELECTRICAL SPECIFICATION
# =====================================================================

Z0 = 50.0
EPS_R = 3.38

FC = 9.0e9
STOP_F = 20.0e9
ANALYSIS_START = 1.0e9
ANALYSIS_STOP = 30.0e9
N_FREQ = 1451

REQUIRED_STOP_ATTENUATION_DB = 60.0


# =====================================================================
# GRID / PHYSICAL GEOMETRY
# =====================================================================

UNIT = 1e-3

GRID_PITCH = 0.25       # mm

H = 1.50                # mm, V11.1 snapped substrate height

# Snapped widths used by V11.1.
W50 = 3.50              # mm
W_HIGH = 0.50           # mm
W_LOW = 12.25           # mm

# Physical RF structure:
#
# PML | 10 mm MSL feed | main region | 10 mm MSL feed | PML
#
# MSL ports occupy the outer 10 mm feed regions.
LINE_X0 = -30.0
LINE_X1 = +30.0

FEED_LENGTH = 10.0

PORT1_START_X = LINE_X0
PORT1_STOP_X = LINE_X0 + FEED_LENGTH       # -20 mm

PORT2_START_X = LINE_X1
PORT2_STOP_X = LINE_X1 - FEED_LENGTH       # +20 mm

MAIN_X0 = PORT1_STOP_X
MAIN_X1 = PORT2_STOP_X


# =====================================================================
# V12 TUNED SECTION LENGTHS
# =====================================================================
#
# Four independent values because of symmetry:
#
# L1 = L7 = 1.00 mm
# L2 = L6 = 1.25 mm
# L3 = L5 = 2.00 mm
# L4       = 1.50 mm
#
# Total = 10.00 mm
#
# The filter is centered inside the 40 mm main region.
# The remaining main-region length is 15.875 mm of 50-ohm line on each side.
# =====================================================================

SECTION_TYPES = [
    "HIGH",
    "LOW",
    "HIGH",
    "LOW",
    "HIGH",
    "LOW",
    "HIGH",
]

SECTION_WIDTHS = [
    W_HIGH,
    W_LOW,
    W_HIGH,
    W_LOW,
    W_HIGH,
    W_LOW,
    W_HIGH,
]

# V14 snapped tuning lengths.
#
# The previous V12 code normalized the raw lengths to 10.000 mm, which
# produced non-grid-aligned values such as 1.025641 mm. The cumulative
# snap operation then changed the total boundary position and caused:
#
#     RuntimeError: Filter boundaries do not close exactly.
#
# For the 0.25-mm EM grid, every section length is therefore explicitly
# chosen as an integer multiple of 0.25 mm.
#
# Symmetric:
#   L1 = L7 = 1.00 mm
#   L2 = L6 = 1.25 mm
#   L3 = L5 = 2.00 mm
#   L4       = 1.50 mm
#
# Total = 10.00 mm exactly.
SECTION_LENGTHS = [
    0.75,
    1.00,
    1.75,
    1.25,
    1.75,
    1.00,
    0.75,
]

SECTION_Z = [
    120.0,
    20.0,
    120.0,
    20.0,
    120.0,
    20.0,
    120.0,
]

FILTER_LENGTH = sum(SECTION_LENGTHS)

if abs(FILTER_LENGTH - 8.25) > 1e-12:
    raise RuntimeError("V14 section lengths must sum to exactly 8.25 mm.")

if not (W_HIGH < W50 < W_LOW):
    raise RuntimeError(
        "V14 width ordering is invalid: expected W_HIGH < W50 < W_LOW."
    )

FILTER_X0 = -FILTER_LENGTH / 2.0
FILTER_X1 = +FILTER_LENGTH / 2.0

LEFT_50OHM_END = FILTER_X0
RIGHT_50OHM_START = FILTER_X1


# =====================================================================
# TRANSVERSE / AIR BOX
# =====================================================================

Y_MIN = -15.0
Y_MAX = +15.0

Z_MIN = -5.0
Z_MAX = H + 20.0


# =====================================================================
# FDTD
# =====================================================================

F0 = 15.0e9
FGAUSS = 15.0e9

NR_TS = 120000
END_CRITERIA = 1e-5

BOUNDARY = [
    "PML_8",
    "PML_8",
    "PML_8",
    "PML_8",
    "PML_8",
    "PML_8",
]


# =====================================================================
# HELPERS
# =====================================================================

def snap(value_mm):
    """Snap a coordinate to the 0.25 mm geometry grid."""
    return round(value_mm / GRID_PITCH) * GRID_PITCH


def add_uniform_axis(mesh, axis, vmin, vmax, spacing):
    """Add a uniform mesh to one axis."""
    n = int(np.ceil((vmax - vmin) / spacing))
    values = np.linspace(vmin, vmax, n + 1)
    mesh.AddLine(axis, values.tolist())


def db20(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-15))


def nearest_index(freq, target):
    return int(np.argmin(np.abs(freq - target)))


# =====================================================================
# GEOMETRY VALIDATION
# =====================================================================

# The V14 filter is centered at x=0.  Its total length is 8.25 mm,
# so the symmetric endpoints are +/-4.125 mm.  Those endpoints are
# intentionally HALF-GRID positions for a 0.25-mm base mesh.
#
# Do NOT snap the cumulative section boundaries to the 0.25-mm grid:
# doing that changes the physical section lengths.  Instead, use the
# exact section lengths and add every boundary explicitly to CSXCAD.
filter_boundaries = [FILTER_X0]

x_now = FILTER_X0
for L in SECTION_LENGTHS:
    x_now = x_now + L
    filter_boundaries.append(x_now)

# Numerical closure check only.
if abs(filter_boundaries[-1] - FILTER_X1) > 1e-12:
    raise RuntimeError(
        f"Filter boundaries do not close exactly: "
        f"{filter_boundaries[-1]:.12f} vs {FILTER_X1:.12f} mm"
    )

# Every SECTION LENGTH, rather than every absolute coordinate, must be
# aligned to the 0.25-mm design grid.
if any(abs(L / GRID_PITCH - round(L / GRID_PITCH)) > 1e-9
       for L in SECTION_LENGTHS):
    raise RuntimeError("A section length is not aligned to the 0.25-mm grid.")

# Monotonicity check.
if any(filter_boundaries[i+1] <= filter_boundaries[i]
       for i in range(len(filter_boundaries)-1)):
    raise RuntimeError("Filter boundaries are not strictly increasing.")


# =====================================================================
# BANNER
# =====================================================================

print()
print("=" * 82)
print("V14 — 7TH-ORDER CHEBYSHEV FILTER / STOPBAND TUNING")
print("=" * 82)

print()
print("TARGET")
print("-" * 82)
print(f"System impedance       : {Z0:.1f} ohm")
print(f"Passband edge          : {FC/1e9:.3f} GHz")
print(f"Required attenuation   : {REQUIRED_STOP_ATTENUATION_DB:.1f} dB @ {STOP_F/1e9:.1f} GHz")
print(f"Analysis range         : {ANALYSIS_START/1e9:.1f} -> {ANALYSIS_STOP/1e9:.1f} GHz")

print()
print("V14 PHYSICAL STOPBAND RE-SYNTHESIS")
print("-" * 82)
print(f"Grid                   : {GRID_PITCH:.2f} mm")
print(f"Substrate height       : {H:.2f} mm")
print(f"50-ohm width           : {W50:.2f} mm")
print(f"High-Z width           : {W_HIGH:.2f} mm")
print(f"Low-Z width            : {W_LOW:.2f} mm")
print(f"Filter length          : {FILTER_LENGTH:.2f} mm")
print(f"Filter x               : {FILTER_X0:.2f} -> {FILTER_X1:.2f} mm")
print(f"Left 50-ohm lead       : {FILTER_X0 - MAIN_X0:.2f} mm")
print(f"Right 50-ohm lead      : {MAIN_X1 - FILTER_X1:.2f} mm")

print()
print("SECTIONS")
print("-" * 82)

for i in range(7):
    print(
        f"S{i+1}: "
        f"{SECTION_TYPES[i]:>4s}  "
        f"Z={SECTION_Z[i]:6.1f} ohm  "
        f"W={SECTION_WIDTHS[i]:5.2f} mm  "
        f"L={SECTION_LENGTHS[i]:5.2f} mm  "
        f"x={filter_boundaries[i]:6.2f}->{filter_boundaries[i+1]:6.2f} mm"
    )

print()
print("MESH / SOLVER")
print("-" * 82)
print(f"XY/Z mesh target        : {GRID_PITCH:.2f} mm")
print(f"Frequency               : {ANALYSIS_START/1e9:.1f} -> {ANALYSIS_STOP/1e9:.1f} GHz")
print(f"Maximum timesteps       : {NR_TS}")
print(f"Boundary                : PML_8")
print(f"Simulation directory    : {SIM_PATH}")


# =====================================================================
# FDTD + CSXCAD
# =====================================================================

FDTD = openEMS(
    NrTS=NR_TS,
    EndCriteria=END_CRITERIA,
)

FDTD.SetGaussExcite(F0, FGAUSS)
FDTD.SetBoundaryCond(BOUNDARY)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(UNIT)


# =====================================================================
# MESH
# =====================================================================
#
# Build the complete mesh BEFORE AddMSLPort().
# =====================================================================

add_uniform_axis(mesh, "x", -40.0, +40.0, GRID_PITCH)
add_uniform_axis(mesh, "y", Y_MIN, Y_MAX, GRID_PITCH)

# Uniform z mesh with explicit substrate boundaries.
add_uniform_axis(mesh, "z", Z_MIN, Z_MAX, GRID_PITCH)

critical_x = [
    -40.0,
    LINE_X0,
    PORT1_STOP_X,
    MAIN_X0,
    FILTER_X0,
    *filter_boundaries,
    FILTER_X1,
    MAIN_X1,
    PORT2_STOP_X,
    LINE_X1,
    +40.0,
]

critical_y = [
    Y_MIN,
    -W_LOW / 2.0,
    -W50 / 2.0,
    -W_HIGH / 2.0,
    0.0,
    +W_HIGH / 2.0,
    +W50 / 2.0,
    +W_LOW / 2.0,
    Y_MAX,
]

critical_z = [
    Z_MIN,
    0.0,
    H,
    Z_MAX,
]

mesh.AddLine("x", [float(v) for v in critical_x])
mesh.AddLine("y", [float(v) for v in critical_y])
mesh.AddLine("z", [float(v) for v in critical_z])

# Keep the maximum mesh spacing bounded.
mesh.SmoothMeshLines("all", GRID_PITCH, 1.35)

# Reassert critical geometry lines after smoothing.
mesh.AddLine("x", [float(v) for v in critical_x])
mesh.AddLine("y", [float(v) for v in critical_y])
mesh.AddLine("z", [float(v) for v in critical_z])


# =====================================================================
# SUBSTRATE
# =====================================================================

substrate = CSX.AddMaterial(
    "SUBSTRATE",
    epsilon=EPS_R,
)

substrate.AddBox(
    start=[LINE_X0, Y_MIN, 0.0],
    stop=[LINE_X1, Y_MAX, H],
    priority=1,
)


# =====================================================================
# GROUND
# =====================================================================

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


# =====================================================================
# EMPTY PEC PROPERTY FOR MSL PORT FEED METAL
# =====================================================================

pec = CSX.AddMetal("PEC")


# =====================================================================
# SIGNAL CONDUCTOR
# =====================================================================
#
# MSL ports create their own 10-mm feed metal sections.
# Therefore the main signal conductor covers only -20 -> +20 mm.
#
# It is deliberately split at every stepped-width discontinuity.
# =====================================================================

signal = CSX.AddMetal("SIGNAL")

# Left 50-ohm lead: -20 -> filter start
signal.AddBox(
    start=[MAIN_X0, -W50 / 2.0, H],
    stop=[FILTER_X0, +W50 / 2.0, H],
    priority=10,
)

# Seven stepped sections
for i in range(7):
    x0 = filter_boundaries[i]
    x1 = filter_boundaries[i + 1]
    w = SECTION_WIDTHS[i]

    signal.AddBox(
        start=[x0, -w / 2.0, H],
        stop=[x1, +w / 2.0, H],
        priority=10,
    )

# Right 50-ohm lead: filter end -> +20
signal.AddBox(
    start=[FILTER_X1, -W50 / 2.0, H],
    stop=[MAIN_X1, +W50 / 2.0, H],
    priority=10,
)

FDTD.AddEdges2Grid(
    dirs="xy",
    properties=signal,
)


# =====================================================================
# MSL PORT 1
# =====================================================================

port1_start = [
    PORT1_START_X,
    -W50 / 2.0,
    H,
]

port1_stop = [
    PORT1_STOP_X,
    +W50 / 2.0,
    0.0,
]

port1 = FDTD.AddMSLPort(
    1,
    pec,
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


# =====================================================================
# MSL PORT 2
# =====================================================================

port2_start = [
    PORT2_START_X,
    -W50 / 2.0,
    H,
]

port2_stop = [
    PORT2_STOP_X,
    +W50 / 2.0,
    0.0,
]

port2 = FDTD.AddMSLPort(
    2,
    pec,
    port2_start,
    port2_stop,
    "x",
    "z",
    excite=0,
    Feed_R=Z0,
    MeasPlaneShift=5.0,
    priority=10,
)


# =====================================================================
# FINAL MESH SANITY CHECK
# =====================================================================

# ---------------------------------------------------------------------
# IMPORTANT:
# CSXCAD's GetLines() binding can return the mesh-line vector in an
# order that is not suitable for np.diff() after several AddLine() /
# SmoothMeshLines() operations.  A negative value such as -80 mm is
# therefore NOT automatically a negative physical cell size.
#
# Validate the actual coordinate set after sorting and removing any
# duplicate mesh lines.  Do not alter the CSXCAD mesh itself here.
# ---------------------------------------------------------------------
x_lines_raw = np.asarray(mesh.GetLines("x"), dtype=float)
y_lines_raw = np.asarray(mesh.GetLines("y"), dtype=float)
z_lines_raw = np.asarray(mesh.GetLines("z"), dtype=float)

x_lines = np.unique(np.sort(x_lines_raw))
y_lines = np.unique(np.sort(y_lines_raw))
z_lines = np.unique(np.sort(z_lines_raw))

nx = len(x_lines) - 1
ny = len(y_lines) - 1
nz = len(z_lines) - 1

estimated_cells = nx * ny * nz

dx = np.diff(x_lines)
dy = np.diff(y_lines)
dz = np.diff(z_lines)

print()
print("=" * 82)
print("MESH VALIDATION")
print("=" * 82)
print(f"x cells                 : {nx}")
print(f"y cells                 : {ny}")
print(f"z cells                 : {nz}")
print(f"estimated cells         : {estimated_cells:,}")
print(f"min dx                  : {np.min(dx):.6f} mm")
print(f"min dy                  : {np.min(dy):.6f} mm")
print(f"min dz                  : {np.min(dz):.6f} mm")
print(f"x range                 : {x_lines[0]:.6f} -> {x_lines[-1]:.6f} mm")
print(f"y range                 : {y_lines[0]:.6f} -> {y_lines[-1]:.6f} mm")
print(f"z range                 : {z_lines[0]:.6f} -> {z_lines[-1]:.6f} mm")

if estimated_cells < 1_000_000:
    raise RuntimeError(
        "Mesh unexpectedly collapsed below 1 million cells. "
        "Do NOT run the EM simulation."
    )

if np.min(dx) <= 0:
    raise RuntimeError("Invalid x mesh after sorting.")

if np.min(dy) <= 0:
    raise RuntimeError("Invalid y mesh after sorting.")

if np.min(dz) <= 0:
    raise RuntimeError("Invalid z mesh after sorting.")


# =====================================================================
# WRITE XML
# =====================================================================


# =====================================================================
# GEOMETRY SUMMARY
# =====================================================================
#
# Save a human-readable record of the exact V14 physical geometry so that
# the eventual frozen design can be reconstructed without ambiguity.
# =====================================================================

geometry_summary = RESULTS_PATH / "chebyshev_v14_fixed_geometry.txt"

with geometry_summary.open("w", encoding="utf-8") as f:
    f.write("V14 — FINAL CANDIDATE GEOMETRY RECORD\n")
    f.write("=" * 72 + "\n")
    f.write(f"Reference impedance       : {Z0:.3f} ohm\n")
    f.write(f"Substrate material        : lossless dielectric, epsilon_r={EPS_R:.6f}\n")
    f.write(f"Substrate height          : {H:.6f} mm\n")
    f.write("Signal / ground material  : PEC in EM model\n")
    f.write(f"50-ohm line width         : {W50:.6f} mm\n")
    f.write(f"HIGH-Z width              : {W_HIGH:.6f} mm\n")
    f.write(f"LOW-Z width               : {W_LOW:.6f} mm\n")
    f.write(f"Filter length             : {FILTER_LENGTH:.6f} mm\n")
    f.write(f"Filter x                  : {FILTER_X0:.6f} -> {FILTER_X1:.6f} mm\n")
    f.write(f"Full RF line              : {LINE_X0:.6f} -> {LINE_X1:.6f} mm\n")
    f.write(f"Main region               : {MAIN_X0:.6f} -> {MAIN_X1:.6f} mm\n")
    f.write(f"Port-1 feed length        : {FEED_LENGTH:.6f} mm\n")
    f.write(f"Port-2 feed length        : {FEED_LENGTH:.6f} mm\n")
    f.write(f"Simulation Y              : {Y_MIN:.6f} -> {Y_MAX:.6f} mm\n")
    f.write(f"Simulation Z              : {Z_MIN:.6f} -> {Z_MAX:.6f} mm\n")
    f.write(f"Mesh target               : {GRID_PITCH:.6f} mm\n")
    f.write(f"Boundary                  : PML_8\n")
    f.write("\nSECTIONS\n")
    f.write("-" * 72 + "\n")
    f.write("ID  TYPE   WIDTH(mm)  LENGTH(mm)   X0(mm)      X1(mm)\n")
    for i in range(7):
        f.write(
            f"S{i+1:02d} {SECTION_TYPES[i]:>5s} "
            f"{SECTION_WIDTHS[i]:11.6f} "
            f"{SECTION_LENGTHS[i]:11.6f} "
            f"{filter_boundaries[i]:11.6f} "
            f"{filter_boundaries[i+1]:11.6f}\n"
        )
    f.write("\nNOTE: widths are physical CAD/EM dimensions; the historical 120/20-ohm\n")
    f.write("labels are not assumed to be exact characteristic impedances.\n")

CSX_FILE = SIM_PATH / "chebyshev_v14_fixed.xml"
CSX.Write2XML(str(CSX_FILE))

print()
print(f"XML written              : {CSX_FILE}")

print()
print("=" * 82)
print("STARTING V12 FULL-WAVE SIMULATION")
print("=" * 82)
print("Expected:")
print("  * no PML reset warnings")
print("  * no NaN energy")
print("  * multi-million-cell FDTD grid")
print("  * stable energy decay")
print()


# =====================================================================
# RUN OPENEMS
# =====================================================================

FDTD.Run(
    str(SIM_PATH),
    cleanup=False,
)


# =====================================================================
# POST-PROCESSING
# =====================================================================

print()
print("=" * 82)
print("POSTPROCESSING")
print("=" * 82)

freq = np.linspace(
    ANALYSIS_START,
    ANALYSIS_STOP,
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

S11_dB = db20(S11)
S21_dB = db20(S21)

power_sum = (
    np.abs(S11) ** 2
    + np.abs(S21) ** 2
)


# =====================================================================
# NUMERICAL RESULTS
# =====================================================================

i5 = nearest_index(freq, 5.0e9)
i9 = nearest_index(freq, 9.0e9)
i20 = nearest_index(freq, 20.0e9)

passband = (
    (freq >= 1.0e9)
    & (freq <= 9.0e9)
)

stopband = (
    (freq >= 20.0e9)
    & (freq <= 30.0e9)
)

worst_passband_s21 = float(np.max(S21_dB[passband]))
best_stopband_s21 = float(np.max(S21_dB[stopband]))

atten20 = float(-S21_dB[i20])

print()
print("=" * 82)
print("V14 RESULTS")
print("=" * 82)

print(f"S11 @ 5 GHz             : {S11_dB[i5]:.6f} dB")
print(f"S21 @ 5 GHz             : {S21_dB[i5]:.6f} dB")
print(
    f"Zin @ 5 GHz             : "
    f"{np.real(Zin[i5]):.6f} "
    f"{np.imag(Zin[i5]):+.6f}j ohm"
)
print(f"|Zin| @ 5 GHz           : {abs(Zin[i5]):.6f} ohm")

print()
print(f"Worst S21, 1–9 GHz      : {worst_passband_s21:.6f} dB")
print(f"S21 @ 9 GHz             : {S21_dB[i9]:.6f} dB")
print(f"S21 @ 20 GHz            : {S21_dB[i20]:.6f} dB")
print(f"Attenuation @ 20 GHz    : {atten20:.6f} dB")
print(
    f"Maximum S21, 20–30 GHz  : "
    f"{best_stopband_s21:.6f} dB"
)

print()
print(f"Power minimum            : {np.nanmin(power_sum):.6f}")
print(f"Power maximum            : {np.nanmax(power_sum):.6f}")


# =====================================================================
# TUNING ASSESSMENT
# =====================================================================

# These remain diagnostic until the physical response is verified.
# V14 is the stopband-tuning candidate before LC extraction.

PASSBAND_DIAGNOSTIC = worst_passband_s21 > -1.0
CUTOFF_DIAGNOSTIC = S21_dB[i9] > -3.0
STOP20_DIAGNOSTIC = atten20 >= REQUIRED_STOP_ATTENUATION_DB
STOPBAND_DIAGNOSTIC = best_stopband_s21 <= -40.0

print()
print("=" * 82)
print("TUNING DIAGNOSTICS")
print("=" * 82)
print(
    f"Passband transmission > -1 dB : "
    f"{'PASS' if PASSBAND_DIAGNOSTIC else 'FAIL'}"
)
print(
    f"S21 @ 9 GHz > -3 dB           : "
    f"{'PASS' if CUTOFF_DIAGNOSTIC else 'FAIL'}"
)
print(
    f"Attenuation @ 20 GHz >= 60 dB  : "
    f"{'PASS' if STOP20_DIAGNOSTIC else 'FAIL'}"
)
print(
    f"20–30 GHz max S21 <= -40 dB    : "
    f"{'PASS' if STOPBAND_DIAGNOSTIC else 'FAIL'}"
)

print()
print(
    "IMPORTANT: these diagnostics are for deciding the next geometry "
    "iteration. They are not a claim that V12 meets the final specification."
)


# =====================================================================
# CSV
# =====================================================================

csv_path = RESULTS_PATH / "chebyshev_v14_fixed.csv"

pd.DataFrame(
    {
        "frequency_Hz": freq,
        "S11_dB": S11_dB,
        "S21_dB": S21_dB,
        "S11_mag": np.abs(S11),
        "S21_mag": np.abs(S21),
        "power_sum": power_sum,
        "Zin_real_ohm": np.real(Zin),
        "Zin_imag_ohm": np.imag(Zin),
        "Zin_mag_ohm": np.abs(Zin),
    }
).to_csv(
    csv_path,
    index=False,
)


# =====================================================================
# S-PARAMETER PLOT
# =====================================================================

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    freq / 1e9,
    S11_dB,
    label="S11",
)

ax.plot(
    freq / 1e9,
    S21_dB,
    label="S21",
)

ax.axvline(
    FC / 1e9,
    linestyle=":",
    linewidth=1.2,
    label="9 GHz cutoff",
)

ax.axvline(
    STOP_F / 1e9,
    linestyle=":",
    linewidth=1.2,
    label="20 GHz stopband",
)

ax.axhline(
    -REQUIRED_STOP_ATTENUATION_DB,
    linestyle="--",
    linewidth=1.0,
    label="-60 dB requirement",
)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("V14 — 7th-Order Chebyshev Stepped-Impedance Filter")
ax.set_xlim(ANALYSIS_START / 1e9, ANALYSIS_STOP / 1e9)
ax.set_ylim(-80, 5)
ax.grid(True)
ax.legend()

fig.tight_layout()

sparam_plot = PLOTS_PATH / "chebyshev_v14_fixed_sparams.png"
fig.savefig(sparam_plot, dpi=200)
plt.close(fig)


# =====================================================================
# IMPEDANCE PLOT
# =====================================================================

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    freq / 1e9,
    np.real(Zin),
    label="Re{Zin}",
)

ax.plot(
    freq / 1e9,
    np.imag(Zin),
    label="Im{Zin}",
)

ax.axhline(
    50.0,
    linestyle="--",
    linewidth=1.0,
    label="50 ohm",
)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Impedance (ohm)")
ax.set_title("V14 — Input Impedance")
ax.set_xlim(ANALYSIS_START / 1e9, ANALYSIS_STOP / 1e9)
ax.grid(True)
ax.legend()

fig.tight_layout()

zin_plot = PLOTS_PATH / "chebyshev_v14_fixed_zin.png"
fig.savefig(zin_plot, dpi=200)
plt.close(fig)


# =====================================================================
# POWER PLOT
# =====================================================================

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    freq / 1e9,
    power_sum,
    label=r"$|S_{11}|^2 + |S_{21}|^2$",
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
    label="Ideal = 1",
)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Power ratio")
ax.set_title("V14 — S-Parameter Power Check")
ax.set_xlim(ANALYSIS_START / 1e9, ANALYSIS_STOP / 1e9)
ax.grid(True)
ax.legend()

fig.tight_layout()

power_plot = PLOTS_PATH / "chebyshev_v14_fixed_power.png"
fig.savefig(power_plot, dpi=200)
plt.close(fig)


# =====================================================================
# SUMMARY
# =====================================================================

summary_path = RESULTS_PATH / "chebyshev_v14_fixed_summary.txt"

summary_text = f"""
V14 — 7th-order Chebyshev stepped-impedance EM tuning

Geometry
--------
Grid pitch       : {GRID_PITCH:.3f} mm
Substrate height : {H:.3f} mm
W50              : {W50:.3f} mm
W_HIGH           : {W_HIGH:.3f} mm
W_LOW            : {W_LOW:.3f} mm

Section lengths
---------------
{SECTION_LENGTHS}

Filter length
-------------
{FILTER_LENGTH:.3f} mm

Results
-------
S11 @ 5 GHz            : {S11_dB[i5]:.6f} dB
S21 @ 5 GHz            : {S21_dB[i5]:.6f} dB
S21 @ 9 GHz            : {S21_dB[i9]:.6f} dB
S21 @ 20 GHz           : {S21_dB[i20]:.6f} dB
Attenuation @ 20 GHz   : {atten20:.6f} dB
Maximum S21, 20-30 GHz : {best_stopband_s21:.6f} dB

Power
-----
minimum : {np.nanmin(power_sum):.6f}
maximum : {np.nanmax(power_sum):.6f}

This run is an EM tuning iteration.  Do not treat it as final
acceptance unless the actual numerical results satisfy the project
specification.
"""

summary_path.write_text(
    summary_text.strip() + "\n",
    encoding="utf-8",
)


# =====================================================================
# FINAL OUTPUT
# =====================================================================

print()
print("=" * 82)
print("V14 FILES")
print("=" * 82)
print(f"CSV       : {csv_path}")
print(f"S-params  : {sparam_plot}")
print(f"Zin       : {zin_plot}")
print(f"Power     : {power_plot}")
print(f"Summary   : {summary_path}")
print(f"XML       : {CSX_FILE}")

print()
print("=" * 82)
print("V12 COMPLETE")
print("=" * 82)
