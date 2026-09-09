"""
V18 — Clean stepped-impedance low-pass filter, 1–100 GHz EM sweep.

This file is intentionally self-contained and conservative.

Design target
-------------
    Z0                  = 50 ohm
    Desired cutoff      ≈ 9–10 GHz
    Stopband target     >= 40 dB from 20 GHz onward
    Broadband inspection: 1–100 GHz

Important
---------
This is an EM candidate, not a claim of final compliance.

The geometry uses ONLY coordinates that lie on a 0.25 mm Cartesian mesh.
No mesh smoothing is used. No floating-point "snap" operation is used.
Every axis is generated once, deduplicated, and validated before openEMS
is started.

The MSL-port topology follows the validated uniform-microstrip model used
earlier in this project.

V18 physical candidate
----------------------
    Substrate:
        epsilon_r = 3.38
        height    = 1.50 mm

    Conductors:
        PEC, zero-thickness sheets in the EM model

    50-ohm reference width:
        3.50 mm

    HIGH-Z width:
        0.50 mm

    LOW-Z width:
        12.00 mm

    Core stepped sections:
        0.75, 1.00, 1.50, 1.75, 1.50, 1.00, 0.75 mm

    Eight impedance transitions:
        1.00 mm each
        four grid-aligned widths:
            0.50 -> 3.00 -> 6.00 -> 9.00 -> 12.00 mm
        and the reverse on the way back.

    Total physical filter length:
        8.00 + 8.00 = 16.00 mm

    Full RF line:
        -30 to +30 mm

    MSL feeds:
        10 mm each

Simulation
----------
    frequency sweep : 1–100 GHz
    excitation      : 50 GHz Gaussian
    mesh            : 0.25 mm
    boundary        : PML_8
    maximum steps   : 140000
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# OPENEMS NATIVE SETUP
# ============================================================================

OPENEMS_ROOT = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
    / "openEMS"
)

if not OPENEMS_ROOT.exists():
    raise RuntimeError(f"openEMS native directory not found: {OPENEMS_ROOT}")

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["PATH"] = str(OPENEMS_ROOT) + os.pathsep + os.environ.get("PATH", "")

try:
    os.add_dll_directory(str(OPENEMS_ROOT))
except AttributeError:
    pass

print(f"openEMS native DLL path : {OPENEMS_ROOT}")

try:
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
except Exception as exc:
    raise RuntimeError(
        "Could not import CSXCAD/openEMS. "
        "Run this script from .venv-openems."
    ) from exc


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SIM_PATH = PROJECT_ROOT / "simulations" / "openems" / "results" / "chebyshev_v18_clean"
RESULTS_PATH = PROJECT_ROOT / "results" / "em" / "chebyshev_v18_clean"
PLOTS_PATH = PROJECT_ROOT / "results" / "plots"

if SIM_PATH.exists():
    shutil.rmtree(SIM_PATH)

SIM_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DESIGN PARAMETERS
# ============================================================================

UNIT = 1e-3
Z0 = 50.0

EPS_R = 3.38
H = 1.50

W50 = 3.50
W_HIGH = 0.50
W_LOW = 12.00
W_LOW_CENTER = 11.50

GRID = 0.25

ANALYSIS_START = 1.0e9
ANALYSIS_STOP = 100.0e9
N_FREQ = 1981

F0 = 50.0e9
FC = 50.0e9

NR_TS = 140000
END_CRITERIA = 1e-5

BOUNDARY = ["PML_8"] * 6

# Full RF line.
LINE_X0 = -30.0
LINE_X1 = +30.0

# Main signal region between the MSL feeds.
MAIN_X0 = -20.0
MAIN_X1 = +20.0

FEED_LENGTH = 10.0

# Simulation cross-section.
Y_MIN = -10.0
Y_MAX = +10.0
Z_MIN = -3.0
Z_MAX = +11.50


# ============================================================================
# V18 FILTER GEOMETRY
# ============================================================================

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
    W_LOW_CENTER,
    W_HIGH,
    W_LOW,
    W_HIGH,
]

SECTION_LENGTHS = [
    0.75,
    1.00,
    1.50,
    1.50,
    1.50,
    1.00,
    0.75,
]

TRANSITION_LENGTH = 1.00

# All transition widths and transition substeps are exact 0.25-mm values.
TRANSITION_WIDTHS = [0.50, 3.00, 6.00, 9.00, 12.00]

FILTER_LENGTH = (
    sum(SECTION_LENGTHS)
    + 8.0 * TRANSITION_LENGTH
)

FILTER_X0 = -FILTER_LENGTH / 2.0
FILTER_X1 = +FILTER_LENGTH / 2.0


# ============================================================================
# BASIC VALIDATION
# ============================================================================

def assert_grid_aligned(value_mm: float, name: str) -> None:
    q = value_mm / GRID
    if abs(q - round(q)) > 1e-9:
        raise RuntimeError(
            f"{name}={value_mm:.12f} mm is not aligned to "
            f"the {GRID:.3f} mm grid."
        )


assert abs(FILTER_LENGTH - 16.00) < 1e-12
assert_grid_aligned(FILTER_X0, "FILTER_X0")
assert_grid_aligned(FILTER_X1, "FILTER_X1")

for i, value in enumerate(SECTION_LENGTHS, 1):
    assert_grid_aligned(value, f"S{i} length")

assert_grid_aligned(TRANSITION_LENGTH, "transition length")

for i, value in enumerate(TRANSITION_WIDTHS):
    assert_grid_aligned(value, f"transition width {i}")

for value, name in [
    (LINE_X0, "LINE_X0"),
    (LINE_X1, "LINE_X1"),
    (MAIN_X0, "MAIN_X0"),
    (MAIN_X1, "MAIN_X1"),
    (Y_MIN, "Y_MIN"),
    (Y_MAX, "Y_MAX"),
    (Z_MIN, "Z_MIN"),
    (Z_MAX, "Z_MAX"),
    (H, "substrate height"),
]:
    assert_grid_aligned(value, name)


# ============================================================================
# PRINT DESIGN SUMMARY
# ============================================================================

print()
print("=" * 82)
print("V18 — LOCAL PEAK-TUNED STEPPED-IMPEDANCE FILTER / 100 GHz SWEEP")
print("=" * 82)

print()
print("TARGET")
print("-" * 82)
print(f"System impedance       : {Z0:.1f} ohm")
print(f"Desired cutoff         : approximately 9–10 GHz")
print(f"Stopband target        : >=40 dB from 20 GHz onward")
print(f"Analysis range         : {ANALYSIS_START/1e9:.1f} -> {ANALYSIS_STOP/1e9:.1f} GHz")

print()
print("PHYSICAL GEOMETRY")
print("-" * 82)
print(f"Substrate epsilon_r    : {EPS_R:.3f}")
print(f"Substrate height      : {H:.2f} mm")
print(f"50-ohm width           : {W50:.2f} mm")
print(f"HIGH-Z width           : {W_HIGH:.2f} mm")
print(f"LOW-Z width            : {W_LOW:.2f} mm")
print(f"Center LOW-Z width     : {W_LOW_CENTER:.2f} mm  (S4 only)")
print(f"Core length            : {sum(SECTION_LENGTHS):.2f} mm")
print(f"Transition length      : {TRANSITION_LENGTH:.2f} mm each")
print(f"Total filter length    : {FILTER_LENGTH:.2f} mm")
print(f"Filter x               : {FILTER_X0:.2f} -> {FILTER_X1:.2f} mm")
print(f"Left 50-ohm lead       : {MAIN_X0 - FILTER_X0:.2f} mm")
print(f"Right 50-ohm lead      : {FILTER_X1 - MAIN_X0 if False else MAIN_X1 - FILTER_X1:.2f} mm")

print()
print("CORE SECTIONS")
print("-" * 82)

x_core = FILTER_X0 + TRANSITION_LENGTH

for i, (kind, width, length) in enumerate(
    zip(SECTION_TYPES, SECTION_WIDTHS, SECTION_LENGTHS),
    start=1,
):
    x0 = x_core
    x1 = x0 + length
    print(
        f"S{i}: {kind:>4}  "
        f"W={width:6.2f} mm  "
        f"L={length:5.2f} mm  "
        f"x={x0:7.2f}->{x1:7.2f} mm"
    )
    x_core = x1 + TRANSITION_LENGTH


# ============================================================================
# PHYSICAL GEOMETRY ELEMENT GENERATION
# ============================================================================

# Each tuple:
#     (x_start, x_stop, width, type)
#
# There are:
#     8 transitions
#     7 core sections
#
# All x coordinates are exact multiples of 0.25 mm.
geometry_elements: list[tuple[float, float, float, str]] = []


def add_transition(
    x_start: float,
    w_start: float,
    w_end: float,
) -> float:
    """
    Add a 1-mm, four-subsection linear width transition.

    Widths are:
        w_start -> 3.00 -> 6.00 -> 9.00 -> w_end

    for the HIGH->LOW direction, and reversed for LOW->HIGH.

    All transition x boundaries are on the 0.25-mm mesh.
    """
    dx = 0.25

    if abs(w_start - W_HIGH) < 1e-12 and abs(w_end - W_LOW) < 1e-12:
        widths = TRANSITION_WIDTHS
    elif abs(w_start - W_LOW) < 1e-12 and abs(w_end - W_HIGH) < 1e-12:
        widths = list(reversed(TRANSITION_WIDTHS))
    elif abs(w_start - W_HIGH) < 1e-12 and abs(w_end - W_LOW_CENTER) < 1e-12:
        widths = [W_HIGH, 3.00, 6.00, 9.00, W_LOW_CENTER]
    elif abs(w_start - W_LOW_CENTER) < 1e-12 and abs(w_end - W_HIGH) < 1e-12:
        widths = [W_LOW_CENTER, 9.00, 6.00, 3.00, W_HIGH]
    elif abs(w_start - W50) < 1e-12 and abs(w_end - W_HIGH) < 1e-12:
        widths = [W50, 2.50, 1.50, 1.00, W_HIGH]
    elif abs(w_start - W_HIGH) < 1e-12 and abs(w_end - W50) < 1e-12:
        widths = [W_HIGH, 1.00, 1.50, 2.50, W50]
    else:
        raise RuntimeError(
            f"Unsupported transition {w_start} -> {w_end} mm"
        )

    for k in range(4):
        xa = x_start + k * dx
        xb = x_start + (k + 1) * dx
        geometry_elements.append(
            (xa, xb, widths[k], "TRANSITION")
        )

    return x_start + TRANSITION_LENGTH


x = FILTER_X0

# Left 50-ohm -> HIGH-Z transition.
x = add_transition(x, W50, W_HIGH)

for i, (kind, width, length) in enumerate(
    zip(SECTION_TYPES, SECTION_WIDTHS, SECTION_LENGTHS)
):
    # Core section.
    x0 = x
    x1 = x0 + length

    geometry_elements.append(
        (x0, x1, width, kind)
    )

    x = x1

    # Transition to next impedance level.
    next_width = (
        SECTION_WIDTHS[i + 1]
        if i + 1 < len(SECTION_WIDTHS)
        else W50
    )

    x = add_transition(x, width, next_width)

if abs(x - FILTER_X1) > 1e-12:
    raise RuntimeError(
        f"Filter geometry does not close: "
        f"{x:.12f} vs {FILTER_X1:.12f} mm"
    )

for xa, xb, width, kind in geometry_elements:
    assert_grid_aligned(xa, f"{kind} x_start")
    assert_grid_aligned(xb, f"{kind} x_stop")
    assert_grid_aligned(width, f"{kind} width")


# ============================================================================
# MESH GENERATION
# ============================================================================

def uniform_axis(vmin: float, vmax: float, pitch: float) -> list[float]:
    """Generate an exact uniform axis without floating-point duplicates."""
    n = int(round((vmax - vmin) / pitch))

    if abs((vmax - vmin) - n * pitch) > 1e-9:
        raise RuntimeError(
            f"Axis {vmin} -> {vmax} is not an integer multiple "
            f"of {pitch} mm."
        )

    return [
        round(vmin + i * pitch, 10)
        for i in range(n + 1)
    ]


def merge_grid_lines(
    base: list[float],
    critical: list[float],
) -> list[float]:
    """Merge critical lines without creating duplicate coordinates."""
    vals = sorted(base + critical)

    result: list[float] = []

    for value in vals:
        value = round(float(value), 10)

        if not result:
            result.append(value)
            continue

        if abs(value - result[-1]) < 1e-8:
            # Same physical line; keep one copy.
            continue

        result.append(value)

    return result


x_lines = uniform_axis(LINE_X0, LINE_X1, GRID)
y_lines = uniform_axis(Y_MIN, Y_MAX, GRID)
z_lines = uniform_axis(Z_MIN, Z_MAX, GRID)

# Every x geometry boundary is already on the 0.25-mm base grid.
# Keep explicit critical lines for clarity and future geometry changes.
x_critical = [
    LINE_X0,
    LINE_X1,
    MAIN_X0,
    MAIN_X1,
    FILTER_X0,
    FILTER_X1,
    -30.0,
    -20.0,
    20.0,
    30.0,
]

for xa, xb, width, kind in geometry_elements:
    x_critical.extend([xa, xb])

# Port transverse edges and conductor width edges.
y_critical = [
    -W_LOW / 2.0,
    -W_LOW_CENTER / 2.0,
    -W50 / 2.0,
    W50 / 2.0,
    W_LOW_CENTER / 2.0,
    W_LOW / 2.0,
    0.0,
]

# The EM model uses zero-thickness PEC sheets for signal and ground.
# Therefore there is NO copper-thickness offset in the mesh.
# Adding ±0.035 mm here would create a 0.035 mm cell and destroy the
# intended 0.25 mm uniform z-grid.
z_critical = [
    Z_MIN,
    0.0,
    H,
    Z_MAX,
]

x_lines = merge_grid_lines(x_lines, x_critical)
y_lines = merge_grid_lines(y_lines, y_critical)
z_lines = merge_grid_lines(z_lines, z_critical)


def validate_axis(
    name: str,
    lines: list[float],
) -> tuple[float, float]:
    if len(lines) < 2:
        raise RuntimeError(f"{name} axis has too few lines.")

    d = np.diff(np.asarray(lines, dtype=float))

    if np.any(d <= 0):
        raise RuntimeError(f"{name} axis contains duplicate/non-increasing lines.")

    return float(np.min(d)), float(np.max(d))


min_dx, max_dx = validate_axis("x", x_lines)
min_dy, max_dy = validate_axis("y", y_lines)
min_dz, max_dz = validate_axis("z", z_lines)

estimated_cells = (
    (len(x_lines) - 1)
    * (len(y_lines) - 1)
    * (len(z_lines) - 1)
)

print()
print("=" * 82)
print("MESH VALIDATION")
print("=" * 82)
print(f"x cells                 : {len(x_lines)-1}")
print(f"y cells                 : {len(y_lines)-1}")
print(f"z cells                 : {len(z_lines)-1}")
print(f"estimated cells         : {estimated_cells:,}")
print(f"min dx                  : {min_dx:.6f} mm")
print(f"min dy                  : {min_dy:.6f} mm")
print(f"min dz                  : {min_dz:.6f} mm")
print(f"x range                 : {x_lines[0]:.6f} -> {x_lines[-1]:.6f} mm")
print(f"y range                 : {y_lines[0]:.6f} -> {y_lines[-1]:.6f} mm")
print(f"z range                 : {z_lines[0]:.6f} -> {z_lines[-1]:.6f} mm")

if (
    min_dx < GRID - 1e-8
    or min_dy < GRID - 1e-8
    or min_dz < GRID - 1e-8
):
    raise RuntimeError(
        "Unexpected sub-grid cell detected. "
        f"Minimum cells: dx={min_dx}, dy={min_dy}, dz={min_dz} mm"
    )

if estimated_cells < 1_000_000:
    raise RuntimeError("Mesh is unexpectedly small; refusing EM run.")


# ============================================================================
# OPENEMS MODEL
# ============================================================================

CSX = ContinuousStructure()

FDTD = openEMS(
    NrTS=NR_TS,
    EndCriteria=END_CRITERIA,
)

FDTD.SetCSX(CSX)
FDTD.SetBoundaryCond(BOUNDARY)

FDTD.SetGaussExcite(
    F0,
    FC,
)


# ============================================================================
# MATERIALS / SUBSTRATE
# ============================================================================

substrate = CSX.AddMaterial(
    "SUBSTRATE",
    epsilon=EPS_R,
)

substrate.AddBox(
    start=[LINE_X0, Y_MIN, 0.0],
    stop=[LINE_X1, Y_MAX, H],
    priority=0,
)

ground = CSX.AddMetal("GROUND")
signal = CSX.AddMetal("SIGNAL")


# Ground plane.
ground.AddBox(
    start=[LINE_X0, Y_MIN, 0.0],
    stop=[LINE_X1, Y_MAX, 0.0],
    priority=10,
)


# ============================================================================
# SIGNAL CONDUCTOR
# ============================================================================

# Left 50-ohm feed.
signal.AddBox(
    start=[LINE_X0, -W50 / 2.0, H],
    stop=[FILTER_X0, W50 / 2.0, H],
    priority=10,
)

# Stepped filter + transitions.
for xa, xb, width, kind in geometry_elements:
    signal.AddBox(
        start=[xa, -width / 2.0, H],
        stop=[xb, width / 2.0, H],
        priority=10,
    )

# Right 50-ohm feed.
signal.AddBox(
    start=[FILTER_X1, -W50 / 2.0, H],
    stop=[LINE_X1, W50 / 2.0, H],
    priority=10,
)


# Explicitly align conductor edges to the mesh.
FDTD.AddEdges2Grid(
    dirs="xy",
    properties=ground,
)

FDTD.AddEdges2Grid(
    dirs="xy",
    properties=signal,
)


# ============================================================================
# MESH INTO CSXCAD
# ============================================================================

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(UNIT)
mesh.AddLine("x", x_lines)
mesh.AddLine("y", y_lines)
mesh.AddLine("z", z_lines)


# ============================================================================
# MSL PORTS — SAME VALIDATED TOPOLOGY
# ============================================================================

PORT1_X = LINE_X0
PORT2_X = LINE_X1

port1_start = [
    PORT1_X,
    -W50 / 2.0,
    H,
]

port1_stop = [
    PORT1_X + FEED_LENGTH,
    +W50 / 2.0,
    0.0,
]

port2_start = [
    PORT2_X,
    -W50 / 2.0,
    H,
]

port2_stop = [
    PORT2_X - FEED_LENGTH,
    +W50 / 2.0,
    0.0,
]

for value, name in [
    (PORT1_X, "PORT1_X"),
    (PORT2_X, "PORT2_X"),
    (PORT1_X + FEED_LENGTH, "PORT1 feed end"),
    (PORT2_X - FEED_LENGTH, "PORT2 feed end"),
]:
    assert_grid_aligned(value, name)

port1 = FDTD.AddMSLPort(
    1,
    ground,
    port1_start,
    port1_stop,
    "x",
    "z",
    excite=1,
    FeedShift=2.5,
    Feed_R=50.0,
    MeasPlaneShift=5.0,
    priority=20,
)

port2 = FDTD.AddMSLPort(
    2,
    ground,
    port2_start,
    port2_stop,
    "x",
    "z",
    excite=0,
    Feed_R=50.0,
    MeasPlaneShift=5.0,
    priority=20,
)


# ============================================================================
# WRITE XML
# ============================================================================

XML_PATH = SIM_PATH / "chebyshev_v18.xml"
CSX.Write2XML(str(XML_PATH))

print()
print(f"XML written              : {XML_PATH}")


# ============================================================================
# RUN EM
# ============================================================================

print()
print("=" * 82)
print("STARTING V18 FULL-WAVE SIMULATION")
print("=" * 82)
print("Expected:")
print("  * uniform 0.25 mm Cartesian mesh")
print("  * no zero-cell mesh")
print("  * no NaN energy")
print("  * no PML reset warning")
print("  * multi-million-cell FDTD grid")
print("  * broadband S-parameters to 100 GHz")
print()

FDTD.Run(
    str(SIM_PATH),
    cleanup=True,
    verbose=3,
)


# ============================================================================
# POSTPROCESSING
# ============================================================================

print()
print("=" * 82)
print("POSTPROCESSING")
print("=" * 82)

freq = np.linspace(
    ANALYSIS_START,
    ANALYSIS_STOP,
    N_FREQ,
)

# Explicit 50-ohm reference impedance.
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

uf_inc_1 = np.asarray(port1.uf_inc).reshape(-1)
uf_ref_1 = np.asarray(port1.uf_ref).reshape(-1)
uf_ref_2 = np.asarray(port2.uf_ref).reshape(-1)

uf_tot_1 = np.asarray(port1.uf_tot).reshape(-1)
if_tot_1 = np.asarray(port1.if_tot).reshape(-1)

eps = 1e-30

S11 = uf_ref_1 / (uf_inc_1 + eps)
S21 = uf_ref_2 / (uf_inc_1 + eps)

S11_dB = 20.0 * np.log10(np.maximum(np.abs(S11), eps))
S21_dB = 20.0 * np.log10(np.maximum(np.abs(S21), eps))

Zin = uf_tot_1 / (if_tot_1 + eps)

power = np.abs(S11) ** 2 + np.abs(S21) ** 2


def nearest_index(f: float) -> int:
    return int(np.argmin(np.abs(freq - f)))


i5 = nearest_index(5e9)
i9 = nearest_index(9e9)
i10 = nearest_index(10e9)
i20 = nearest_index(20e9)
i100 = nearest_index(100e9)

passband_mask = (
    (freq >= 1e9)
    & (freq <= 9e9)
)

stopband_mask = (
    (freq >= 20e9)
    & (freq <= 100e9)
)

passband_worst = float(np.max(S21_dB[passband_mask]))
stopband_best = float(np.max(S21_dB[stopband_mask]))

attenuation20 = -float(S21_dB[i20])


# ============================================================================
# RESULTS
# ============================================================================

print()
print("=" * 82)
print("V18 RESULTS")
print("=" * 82)

print(f"S11 @ 5 GHz             : {S11_dB[i5]:.6f} dB")
print(f"S21 @ 5 GHz             : {S21_dB[i5]:.6f} dB")
print(
    f"Zin @ 5 GHz             : "
    f"{Zin[i5].real:.6f} {Zin[i5].imag:+.6f}j ohm"
)
print(f"|Zin| @ 5 GHz           : {abs(Zin[i5]):.6f} ohm")

print(f"S21 @ 9 GHz             : {S21_dB[i9]:.6f} dB")
print(f"S21 @ 10 GHz            : {S21_dB[i10]:.6f} dB")
print(f"S21 @ 20 GHz            : {S21_dB[i20]:.6f} dB")
print(f"S21 @ 100 GHz           : {S21_dB[i100]:.6f} dB")

print(f"Worst S21, 1–9 GHz      : {passband_worst:.6f} dB")
print(f"Attenuation @ 20 GHz    : {attenuation20:.6f} dB")
print(f"Maximum S21, 20–100 GHz : {stopband_best:.6f} dB")

print(f"Power minimum            : {np.min(power):.6f}")
print(f"Power maximum            : {np.max(power):.6f}")


# ============================================================================
# ACCEPTANCE DIAGNOSTICS
# ============================================================================

print()
print("=" * 82)
print("TUNING DIAGNOSTICS")
print("=" * 82)

passband_ok = passband_worst >= -1.5
cutoff_ok = S21_dB[i10] <= -3.0
stopband_ok = attenuation20 >= 40.0
broadband_ok = stopband_best <= -40.0

print(
    f"Passband S21 > -1.5 dB : "
    f"{'PASS' if passband_ok else 'FAIL'}"
)
print(
    f"S21 @ 10 GHz <= -3 dB  : "
    f"{'PASS' if cutoff_ok else 'FAIL'}"
)
print(
    f"Attenuation @ 20 GHz >= 40 dB : "
    f"{'PASS' if stopband_ok else 'FAIL'}"
)
print(
    f"20–100 GHz max S21 <= -40 dB : "
    f"{'PASS' if broadband_ok else 'FAIL'}"
)


# ============================================================================
# SAVE CSV
# ============================================================================

CSV_PATH = RESULTS_PATH / "chebyshev_v18.csv"

with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)

    writer.writerow([
        "frequency_Hz",
        "S11_real",
        "S11_imag",
        "S21_real",
        "S21_imag",
        "S11_dB",
        "S21_dB",
        "Zin_real_ohm",
        "Zin_imag_ohm",
        "power_sum",
    ])

    for k in range(len(freq)):
        writer.writerow([
            freq[k],
            S11[k].real,
            S11[k].imag,
            S21[k].real,
            S21[k].imag,
            S11_dB[k],
            S21_dB[k],
            Zin[k].real,
            Zin[k].imag,
            power[k],
        ])


# ============================================================================
# SAVE SUMMARY
# ============================================================================

SUMMARY_PATH = RESULTS_PATH / "chebyshev_v18_summary.txt"

with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
    fh.write("V18 FILTER FULL-WAVE SUMMARY\n")
    fh.write("=" * 72 + "\n")
    fh.write(f"Z0                 : {Z0:.3f} ohm\n")
    fh.write(f"epsilon_r          : {EPS_R:.6f}\n")
    fh.write(f"substrate height   : {H:.6f} mm\n")
    fh.write(f"W50                : {W50:.6f} mm\n")
    fh.write(f"W_HIGH             : {W_HIGH:.6f} mm\n")
    fh.write(f"W_LOW              : {W_LOW:.6f} mm\n")
    fh.write(f"W_LOW_CENTER       : {W_LOW_CENTER:.6f} mm (S4 only)\n")
    fh.write(f"filter length      : {FILTER_LENGTH:.6f} mm\n")
    fh.write(f"analysis stop      : {ANALYSIS_STOP/1e9:.3f} GHz\n")
    fh.write("\nCORE LENGTHS\n")
    fh.write(str(SECTION_LENGTHS) + "\n")
    fh.write("\nRESULTS\n")
    fh.write(f"S11_5GHz_dB        : {S11_dB[i5]:.6f}\n")
    fh.write(f"S21_5GHz_dB        : {S21_dB[i5]:.6f}\n")
    fh.write(f"S21_9GHz_dB        : {S21_dB[i9]:.6f}\n")
    fh.write(f"S21_10GHz_dB       : {S21_dB[i10]:.6f}\n")
    fh.write(f"S21_20GHz_dB       : {S21_dB[i20]:.6f}\n")
    fh.write(f"S21_100GHz_dB      : {S21_dB[i100]:.6f}\n")
    fh.write(f"attenuation_20GHz  : {attenuation20:.6f} dB\n")
    fh.write(f"max_S21_20_100GHz  : {stopband_best:.6f} dB\n")
    fh.write(f"power_min          : {np.min(power):.6f}\n")
    fh.write(f"power_max          : {np.max(power):.6f}\n")


# ============================================================================
# SAVE PLOTS
# ============================================================================

SPLOT = PLOTS_PATH / "chebyshev_v18_sparams_100GHz.png"

plt.figure(figsize=(11, 6))
plt.plot(freq / 1e9, S11_dB, label="S11")
plt.plot(freq / 1e9, S21_dB, label="S21")
plt.axvline(9.0, linestyle="--", linewidth=1.0, label="9 GHz")
plt.axvline(20.0, linestyle="--", linewidth=1.0, label="20 GHz")
plt.axhline(-40.0, linestyle=":", linewidth=1.0, label="-40 dB")
plt.xlim(0, 100)
plt.ylim(-100, 5)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.title("V18 Peak-Tuned Stepped-Impedance Filter — 1–100 GHz")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(SPLOT, dpi=180)
plt.close()


ZPLOT = PLOTS_PATH / "chebyshev_v18_zin.png"

plt.figure(figsize=(11, 6))
plt.plot(freq / 1e9, Zin.real, label="Re{Zin}")
plt.plot(freq / 1e9, Zin.imag, label="Im{Zin}")
plt.axhline(50.0, linestyle="--", linewidth=1.0, label="50 ohm")
plt.xlim(0, 30)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Impedance (ohm)")
plt.title("V18 Input Impedance")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(ZPLOT, dpi=180)
plt.close()


PLOT_POWER = PLOTS_PATH / "chebyshev_v18_power.png"

plt.figure(figsize=(11, 6))
plt.plot(freq / 1e9, power)
plt.axhline(1.0, linestyle="--", linewidth=1.0)
plt.xlim(0, 100)
plt.xlabel("Frequency (GHz)")
plt.ylabel("|S11|² + |S21|²")
plt.title("V18 Power Balance")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(PLOT_POWER, dpi=180)
plt.close()


# ============================================================================
# EXACT GEOMETRY RECORD
# ============================================================================

GEOM_PATH = RESULTS_PATH / "chebyshev_v18_geometry.txt"

with GEOM_PATH.open("w", encoding="utf-8") as fh:
    fh.write("V18 PHYSICAL GEOMETRY (S4 LOCAL DETUNE) RECORD\n")
    fh.write("=" * 72 + "\n")
    fh.write("All dimensions are in mm.\n\n")

    fh.write("MATERIALS\n")
    fh.write("-" * 72 + "\n")
    fh.write(f"Substrate epsilon_r : {EPS_R}\n")
    fh.write(f"Substrate height   : {H}\n")
    fh.write("Signal conductor   : PEC, zero-thickness EM sheet\n")
    fh.write("Ground conductor   : PEC, zero-thickness EM sheet\n\n")

    fh.write("OVERALL RF STRUCTURE\n")
    fh.write("-" * 72 + "\n")
    fh.write(f"RF line             : {LINE_X0} -> {LINE_X1}\n")
    fh.write(f"Main region         : {MAIN_X0} -> {MAIN_X1}\n")
    fh.write(f"Filter              : {FILTER_X0} -> {FILTER_X1}\n")
    fh.write(f"Filter length       : {FILTER_LENGTH}\n")
    fh.write(f"50-ohm width        : {W50}\n")
    fh.write(f"HIGH-Z width        : {W_HIGH}\n")
    fh.write(f"LOW-Z width         : {W_LOW}\n\n")

    fh.write("CORE SECTIONS\n")
    fh.write("-" * 72 + "\n")

    x = FILTER_X0 + TRANSITION_LENGTH

    for i, (kind, width, length) in enumerate(
        zip(SECTION_TYPES, SECTION_WIDTHS, SECTION_LENGTHS),
        start=1,
    ):
        fh.write(
            f"S{i}: {kind:4s}  "
            f"width={width:.3f}  "
            f"length={length:.3f}  "
            f"x={x:.3f}->{x+length:.3f}\n"
        )
        x += length + TRANSITION_LENGTH

    fh.write("\nTRANSITIONS\n")
    fh.write("-" * 72 + "\n")
    fh.write("Length of each transition: 1.000 mm\n")
    fh.write("Transition widths (HIGH -> LOW): 0.50, 3.00, 6.00, 9.00, 12.00 mm\n")
    fh.write("Reverse sequence used for LOW -> HIGH.\n")

    fh.write("\nSIMULATION\n")
    fh.write("-" * 72 + "\n")
    fh.write(f"Frequency range      : {ANALYSIS_START/1e9} -> {ANALYSIS_STOP/1e9} GHz\n")
    fh.write(f"Mesh pitch           : {GRID} mm\n")
    fh.write("Boundary             : PML_8\n")
    fh.write(f"Port impedance       : {Z0} ohm\n")


print()
print("=" * 82)
print("V18 FILES")
print("=" * 82)
print(f"CSV       : {CSV_PATH}")
print(f"S-params  : {SPLOT}")
print(f"Zin       : {ZPLOT}")
print(f"Power     : {PLOT_POWER}")
print(f"Summary   : {SUMMARY_PATH}")
print(f"Geometry  : {GEOM_PATH}")
print(f"XML       : {XML_PATH}")
print("=" * 82)
