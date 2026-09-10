"""
Candidate B3 — V18 Chebyshev + 7th-order Butterworth stepped-impedance cascade, 1–100 GHz EM sweep.

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

Stage 1: V18 physical candidate
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

PROJECT_ROOT = Path(__file__).resolve().parent

_tag = os.environ.get("B3_TAG","cheb_b3_butterworth_cascade_v1")
SIM_PATH = PROJECT_ROOT / "simulations" / "openems" / "results" / _tag
RESULTS_PATH = PROJECT_ROOT / "results" / "em" / _tag
PLOTS_PATH = PROJECT_ROOT / "results" / "plots" / _tag

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
W_HIGH = float(os.environ.get("B3_W_HIGH","0.50"))
W_LOW = float(os.environ.get("B3_W_LOW","12.50"))

GRID = 0.25

ANALYSIS_START = 1.0e9
ANALYSIS_STOP = float(os.environ.get("B3_ANALYSIS_STOP_GHZ","100.0"))*1e9
N_FREQ = int(os.environ.get("B3_N_FREQ","1981"))

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
    W_LOW,
    W_HIGH,
    W_LOW,
    W_HIGH,
]

CHEB_SECTION_LENGTHS = [
    1.00,
    0.75,
    1.25,
    1.00,
    1.25,
    0.75,
    1.00,
]

TRANSITION_LENGTH = 1.00

# All transition widths and transition substeps are exact 0.25-mm values.
TRANSITION_WIDTHS = [0.50, 3.00, 6.00, 9.00, 12.00]

# ============================================================================
# SECOND-STAGE BUTTERWORTH CANDIDATE B3
# ============================================================================

# Stage 1 remains the experimentally optimized V18 7th-order Chebyshev.
CHEB_SECTION_TYPES = [
    "HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH"
]
CHEB_SECTION_WIDTHS = [W_HIGH, W_LOW, W_HIGH, W_LOW, W_HIGH, W_LOW, W_HIGH]
CHEB_SECTION_LENGTHS = [float(v) for v in os.environ.get("B3_CHEB_LENGTHS","1.00,0.75,1.25,1.00,1.25,0.75,1.00").split(",")]
CHEB_FILTER_LENGTH = sum(CHEB_SECTION_LENGTHS) + 8.0 * TRANSITION_LENGTH

# Stage 2: 7th-order Butterworth, target cutoff = 10 GHz.
# Prototype: g_k = 2 sin((2k-1) pi / (2N)).
BUTTERWORTH_ORDER = 7
BUTTERWORTH_FC = 10.0e9
BUTTERWORTH_G = [
    0.4450418679, 1.2469796037, 1.8019377358, 2.0000000000,
    1.8019377358, 1.2469796037, 0.4450418679,
]

# The second stage uses the same high/low impedance levels as the V18
# physical implementation. Section electrical lengths are derived from the
# Butterworth lumped prototype using the small-line stepped-impedance mapping:
#   series L: theta_L = g_k * Z0 / Z_HIGH
#   shunt C : theta_C = g_k * Z_LOW / Z0
# with theta converted to physical length using the microstrip phase velocity.
# The resulting dimensions are then quantized to the existing 0.25-mm mesh.

C_LIGHT = 299792458.0

def microstrip_eps_eff(width_mm: float) -> float:
    wh = width_mm / H
    ee = (EPS_R + 1.0) / 2.0 + (EPS_R - 1.0) / 2.0 * (1.0 + 12.0 / wh) ** -0.5
    if wh < 1.0:
        ee += 0.04 * (1.0 - wh) ** 2
    return ee


def microstrip_z0(width_mm: float) -> float:
    wh = width_mm / H
    ee = microstrip_eps_eff(width_mm)
    if wh <= 1.0:
        return (60.0 / math.sqrt(ee)) * math.log(8.0 / wh + wh / 4.0)
    return (120.0 * math.pi) / (math.sqrt(ee) * (wh + 1.393 + 0.667 * math.log(wh + 1.444)))


def snap_length_mm(value_mm: float) -> float:
    return round(value_mm / GRID) * GRID


def butterworth_section_lengths() -> list[float]:
    z_high = microstrip_z0(W_HIGH)
    z_low = microstrip_z0(W_LOW)
    ee_high = microstrip_eps_eff(W_HIGH)
    ee_low = microstrip_eps_eff(W_LOW)
    lambda_high_mm = C_LIGHT / BUTTERWORTH_FC / math.sqrt(ee_high) * 1e3
    lambda_low_mm = C_LIGHT / BUTTERWORTH_FC / math.sqrt(ee_low) * 1e3

    lengths = []
    for k, gk in enumerate(BUTTERWORTH_G):
        if k % 2 == 0:  # high-Z series inductive section
            theta = gk * Z0 / z_high
            raw = theta / (2.0 * math.pi) * lambda_high_mm
        else:             # low-Z shunt capacitive section
            theta = gk * z_low / Z0
            raw = theta / (2.0 * math.pi) * lambda_low_mm
        lengths.append(snap_length_mm(raw))
    return lengths


BUTTER_SECTION_TYPES = [
    "HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH"
]
BUTTER_SECTION_WIDTHS = [W_HIGH, W_LOW, W_HIGH, W_LOW, W_HIGH, W_LOW, W_HIGH]
if os.environ.get("B3_BUTTER_LENGTHS"):
    BUTTER_SECTION_LENGTHS = [float(v) for v in os.environ["B3_BUTTER_LENGTHS"].split(",")]
else:
    BUTTER_SECTION_LENGTHS = butterworth_section_lengths()
    BUTTER_SECTION_LENGTHS[3] += GRID
BUTTER_FILTER_LENGTH = sum(BUTTER_SECTION_LENGTHS) + 8.0 * TRANSITION_LENGTH

# Two stages are separated by a short 50-ohm interconnect.
CASCADE_GAP = float(os.environ.get("B3_CASCADE_GAP","5.00"))
CASCADE_LENGTH = CHEB_FILTER_LENGTH + CASCADE_GAP + BUTTER_FILTER_LENGTH

LPF1_X0 = -CASCADE_LENGTH / 2.0
LPF1_X1 = LPF1_X0 + CHEB_FILTER_LENGTH
LPF2_X0 = LPF1_X1 + CASCADE_GAP
LPF2_X1 = LPF2_X0 + BUTTER_FILTER_LENGTH

assert abs(LPF2_X1 - CASCADE_LENGTH / 2.0) < 1e-12

# For the stage-independent diagnostics below, keep these names tied to
# stage 1 because the old optimizer machinery expects them.
W_HIGH = float(W_HIGH)
W_LOW = float(W_LOW)


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


assert abs(CASCADE_LENGTH / GRID - round(CASCADE_LENGTH / GRID)) < 1e-9
assert_grid_aligned(LPF1_X0, "LPF1_X0")
assert_grid_aligned(LPF1_X1, "LPF1_X1")
assert_grid_aligned(LPF2_X0, "LPF2_X0")
assert_grid_aligned(LPF2_X1, "LPF2_X1")
assert_grid_aligned(CASCADE_GAP, "CASCADE_GAP")

for i, value in enumerate(CHEB_SECTION_LENGTHS, 1):
    assert_grid_aligned(value, f"Chebyshev S{i} length")
for i, value in enumerate(BUTTER_SECTION_LENGTHS, 1):
    assert_grid_aligned(value, f"Butterworth S{i} length")

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
print("CHEBYSHEV V18 + 7th-ORDER BUTTERWORTH — 1–100 GHz SWEEP")
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
print(f"Substrate height       : {H:.2f} mm")
print(f"50-ohm width           : {W50:.2f} mm")
print(f"HIGH-Z width           : {W_HIGH:.2f} mm")
print(f"LOW-Z width            : {W_LOW:.2f} mm")
print(f"Chebyshev length        : {CHEB_FILTER_LENGTH:.2f} mm")
print(f"Butterworth length      : {BUTTER_FILTER_LENGTH:.2f} mm")
print(f"Butterworth cutoff      : {BUTTERWORTH_FC/1e9:.2f} GHz")
print(f"Butterworth order       : {BUTTERWORTH_ORDER}")
print(f"Butterworth prototype g : {BUTTERWORTH_G}")
print(f"Butterworth core lengths: {BUTTER_SECTION_LENGTHS}")
print(f"Interconnect gap       : {CASCADE_GAP:.2f} mm")
print(f"Total cascade length   : {CASCADE_LENGTH:.2f} mm")
print(f"LPF #1 x               : {LPF1_X0:.2f} -> {LPF1_X1:.2f} mm")
print(f"LPF #2 x               : {LPF2_X0:.2f} -> {LPF2_X1:.2f} mm")
print(f"Interconnect x         : {LPF1_X1:.2f} -> {LPF2_X0:.2f} mm")

print()
print("CORE SECTIONS — LPF #1")
print("-" * 82)

for i, (kind, width, length) in enumerate(
    zip(CHEB_SECTION_TYPES, CHEB_SECTION_WIDTHS, CHEB_SECTION_LENGTHS),
    start=1,
):
    print(f"S{i}: {kind:>4}  W={width:6.2f} mm  L={length:5.2f} mm")

print()
print("CORE SECTIONS — LPF #2")
print("-" * 82)
for i, (kind, width, length) in enumerate(
    zip(BUTTER_SECTION_TYPES, BUTTER_SECTION_WIDTHS, BUTTER_SECTION_LENGTHS),
    start=1,
):
    print(f"S{i}: {kind:>4}  W={width:6.2f} mm  L={length:5.2f} mm")


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

    raw = np.linspace(w_start, w_end, 5)

    # The Cartesian y-grid is 0.25 mm, so conductor width/2 must lie on that
    # grid. Restrict transition widths to 0.50-mm increments.
    #
    # IMPORTANT:
    # np.linspace() itself does not generally land on 0.50-mm values.
    # The previous optimizer runner incorrectly tested the unsnapped values
    # and therefore rejected the very first 3.50 -> 0.50 mm feed transition.
    widths = [
        0.50 * round(float(value) / 0.50)
        for value in raw
    ]

    # Preserve the exact endpoints.
    widths[0] = float(w_start)
    widths[-1] = float(w_end)

    # Check the actual values that will be used.
    for value in widths:
        if abs(value / 0.50 - round(value / 0.50)) > 1e-9:
            raise RuntimeError(
                f"Transition width {value} mm is not 0.50-mm aligned."
            )

    # Ensure the snapped transition remains monotonic.
    direction = 1.0 if w_end >= w_start else -1.0
    for a, b in zip(widths[:-1], widths[1:]):
        if direction * (b - a) < -1e-12:
            raise RuntimeError(
                f"Non-monotonic transition generated: {widths}"
            )

    for k in range(4):
        xa = x_start + k * dx
        xb = x_start + (k + 1) * dx
        geometry_elements.append(
            (xa, xb, widths[k], "TRANSITION")
        )

    return x_start + TRANSITION_LENGTH



def build_stage(
    x_start: float,
    label: str,
    section_types: list[str],
    section_widths: list[float],
    section_lengths: list[float],
) -> float:
    x = x_start
    x = add_transition(x, W50, section_widths[0])

    for i, (kind, width, length) in enumerate(
        zip(section_types, section_widths, section_lengths)
    ):
        x0 = x
        x1 = x0 + length
        geometry_elements.append((x0, x1, width, f"{label}_{kind}"))
        x = x1

        next_width = section_widths[i + 1] if i + 1 < len(section_widths) else W50
        x = add_transition(x, width, next_width)

    return x


end1 = build_stage(
    LPF1_X0, "CHEB",
    CHEB_SECTION_TYPES, CHEB_SECTION_WIDTHS, CHEB_SECTION_LENGTHS
)
if abs(end1 - LPF1_X1) > 1e-12:
    raise RuntimeError(f"Chebyshev stage does not close: {end1} vs {LPF1_X1}")

geometry_elements.append((LPF1_X1, LPF2_X0, W50, "INTERCONNECT_50OHM"))

end2 = build_stage(
    LPF2_X0, "BUTTER",
    BUTTER_SECTION_TYPES, BUTTER_SECTION_WIDTHS, BUTTER_SECTION_LENGTHS
)
if abs(end2 - LPF2_X1) > 1e-12:
    raise RuntimeError(f"Butterworth stage does not close: {end2} vs {LPF2_X1}")

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
    LPF1_X0,
    LPF1_X1,
    LPF2_X0,
    LPF2_X1,
    -30.0,
    -20.0,
    20.0,
    30.0,
]

for xa, xb, width, kind in geometry_elements:
    x_critical.extend([xa, xb])

# Port transverse edges and conductor width edges.
_all_widths = {W50, W_HIGH, W_LOW}
for _xa, _xb, _w, _kind in geometry_elements:
    _all_widths.add(_w)

y_critical = [0.0]
for _w in sorted(_all_widths):
    y_critical.extend([-_w / 2.0, _w / 2.0])

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
    stop=[LPF1_X0, W50 / 2.0, H],
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
    start=[LPF2_X1, -W50 / 2.0, H],
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

XML_PATH = SIM_PATH / "cheb_b3_butterworth_cascade_v1.xml"
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

_res14_18 = (freq >= 14e9) & (freq <= 18e9)
_res20_100 = (freq >= 20e9) & (freq <= min(100e9, ANALYSIS_STOP))
_pass_1_9p5 = (freq >= 1e9) & (freq <= 9.5e9)
metrics = {
    "tag": os.environ.get("B3_TAG", "candidate"),
    "cheb_lengths": [float(v) for v in CHEB_SECTION_LENGTHS],
    "butter_lengths": [float(v) for v in BUTTER_SECTION_LENGTHS],
    "cascade_gap_mm": float(CASCADE_GAP),
    "s21_5GHz_dB": float(S21_dB[i5]),
    "s21_9GHz_dB": float(S21_dB[i9]),
    "s21_10GHz_dB": float(S21_dB[i10]),
    "s21_20GHz_dB": float(S21_dB[i20]),
    "s21_100GHz_dB": float(S21_dB[i100]),
    "passband_min_1_9p5_dB": float(np.min(S21_dB[_pass_1_9p5])),
    "max_s21_14_18_dB": float(np.max(S21_dB[_res14_18])),
    "max_s21_20_100_dB": float(np.max(S21_dB[_res20_100])),
    "power_min": float(np.min(power)),
    "power_max": float(np.max(power)),
}
metrics_path = RESULTS_PATH / "metrics.json"
with metrics_path.open("w", encoding="utf-8") as _mf:
    import json as _json
    _json.dump(metrics, _mf, indent=2)
    _mf.flush()
    os.fsync(_mf.fileno())
print(f"OPTIMIZER_METRICS={metrics_path}")
print("METRICS_EXISTS=" + str(metrics_path.is_file()))


# ============================================================================
# RESULTS
# ============================================================================

print()
print("=" * 82)
print("CHEBYSHEV + BUTTERWORTH CASCADE RESULTS")
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

CSV_PATH = RESULTS_PATH / "cheb_b3_butterworth_cascade_v1.csv"

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

SUMMARY_PATH = RESULTS_PATH / "cheb_b3_butterworth_cascade_v1_summary.txt"

with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
    fh.write("CHEBYSHEV + BUTTERWORTH CASCADE FULL-WAVE SUMMARY\n")
    fh.write("=" * 72 + "\n")
    fh.write(f"Z0                 : {Z0:.3f} ohm\n")
    fh.write(f"epsilon_r          : {EPS_R:.6f}\n")
    fh.write(f"substrate height   : {H:.6f} mm\n")
    fh.write(f"W50                : {W50:.6f} mm\n")
    fh.write(f"W_HIGH             : {W_HIGH:.6f} mm\n")
    fh.write(f"W_LOW              : {W_LOW:.6f} mm\n")
    fh.write(f"Chebyshev length   : {CHEB_FILTER_LENGTH:.6f} mm\n")
    fh.write(f"Butterworth length : {BUTTER_FILTER_LENGTH:.6f} mm\n")
    fh.write(f"cascade length     : {CASCADE_LENGTH:.6f} mm\n")
    fh.write(f"interconnect gap   : {CASCADE_GAP:.6f} mm\n")
    fh.write(f"analysis stop      : {ANALYSIS_STOP/1e9:.3f} GHz\n")
    fh.write("\nCORE LENGTHS\n")
    fh.write(f"Chebyshev lengths   : {CHEB_SECTION_LENGTHS}\n")
    fh.write(f"Butterworth lengths : {BUTTER_SECTION_LENGTHS}\n")
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

SPLOT = PLOTS_PATH / "cheb_b3_butterworth_cascade_v1_sparams_100GHz.png"

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
plt.title("Chebyshev V18 + 7th-Order Butterworth Cascade — 1–100 GHz")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(SPLOT, dpi=180)
plt.close()


ZPLOT = PLOTS_PATH / "cheb_b3_butterworth_cascade_v1_zin.png"

plt.figure(figsize=(11, 6))
plt.plot(freq / 1e9, Zin.real, label="Re{Zin}")
plt.plot(freq / 1e9, Zin.imag, label="Im{Zin}")
plt.axhline(50.0, linestyle="--", linewidth=1.0, label="50 ohm")
plt.xlim(0, 30)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Impedance (ohm)")
plt.title("Two-LPF Cascade Input Impedance")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(ZPLOT, dpi=180)
plt.close()


PLOT_POWER = PLOTS_PATH / "cheb_b3_butterworth_cascade_v1_power.png"

plt.figure(figsize=(11, 6))
plt.plot(freq / 1e9, power)
plt.axhline(1.0, linestyle="--", linewidth=1.0)
plt.xlim(0, 100)
plt.xlabel("Frequency (GHz)")
plt.ylabel("|S11|² + |S21|²")
plt.title("Two-LPF Cascade Power Balance")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(PLOT_POWER, dpi=180)
plt.close()


# ============================================================================
# EXACT GEOMETRY RECORD
# ============================================================================

GEOM_PATH = RESULTS_PATH / "cheb_b3_butterworth_cascade_v1_geometry.txt"

with GEOM_PATH.open("w", encoding="utf-8") as fh:
    fh.write("CHEBYSHEV + BUTTERWORTH CASCADE PHYSICAL GEOMETRY RECORD\n")
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
    fh.write(f"LPF #1              : {LPF1_X0} -> {LPF1_X1}\n")
    fh.write(f"LPF #2              : {LPF2_X0} -> {LPF2_X1}\n")
    fh.write(f"Chebyshev length    : {CHEB_FILTER_LENGTH}\n")
    fh.write(f"Butterworth length  : {BUTTER_FILTER_LENGTH}\n")
    fh.write(f"Butterworth fc     : {BUTTERWORTH_FC/1e9} GHz\n")
    fh.write(f"Butterworth order  : {BUTTERWORTH_ORDER}\n")
    fh.write(f"Interconnect gap     : {CASCADE_GAP}\n")
    fh.write(f"Total cascade length : {CASCADE_LENGTH}\n")
    fh.write(f"50-ohm width        : {W50}\n")
    fh.write(f"HIGH-Z width        : {W_HIGH}\n")
    fh.write(f"LOW-Z width         : {W_LOW}\n\n")

    for label, x_start, types, widths, lengths in [
        ("CHEBYSHEV", LPF1_X0, CHEB_SECTION_TYPES, CHEB_SECTION_WIDTHS, CHEB_SECTION_LENGTHS),
        ("BUTTERWORTH", LPF2_X0, BUTTER_SECTION_TYPES, BUTTER_SECTION_WIDTHS, BUTTER_SECTION_LENGTHS),
    ]:
        fh.write(f"{label} CORE SECTIONS\n")
        fh.write("-" * 72 + "\n")
        x = x_start + TRANSITION_LENGTH
        for i, (kind, width, length) in enumerate(zip(types, widths, lengths), start=1):
            fh.write(
                f"S{i}: {kind:4s}  width={width:.3f}  "
                f"length={length:.3f}  x={x:.3f}->{x+length:.3f}\n"
            )
            x += length + TRANSITION_LENGTH

    fh.write(
        f"\n50-ohm interconnect: {LPF1_X1:.3f}->{LPF2_X0:.3f} mm\n"
    )

    fh.write("\nTRANSITIONS\n")
    fh.write("-" * 72 + "\n")
    fh.write("Length of each transition: 1.000 mm\n")
    fh.write("Transition widths are generated from the actual stage endpoint widths on the 0.25-mm grid.\n")
    fh.write("For 0.50 -> 12.50 mm: 0.50, 3.50, 6.50, 9.50, 12.50 mm.\n")
    fh.write("Reverse sequence used for LOW -> HIGH.\n")

    fh.write("\nSIMULATION\n")
    fh.write("-" * 72 + "\n")
    fh.write(f"Frequency range      : {ANALYSIS_START/1e9} -> {ANALYSIS_STOP/1e9} GHz\n")
    fh.write(f"Mesh pitch           : {GRID} mm\n")
    fh.write("Boundary             : PML_8\n")
    fh.write(f"Port impedance       : {Z0} ohm\n")


# ============================================================================
# END OF STANDALONE CANDIDATE B3 RUNNER
# ============================================================================

print()
print("Results written to:")
print(f"  {RESULTS_PATH}")
print(f"  {SPLOT}")

