"""
Properly synthesized 7th-order Butterworth stepped-impedance LPF.

Purpose
-------
This is a CLEAN standalone validation of the second-stage Butterworth filter
before cascading it with the Chebyshev stage.

Target:
    Z0 = 50 ohm
    fc = 10 GHz
    N  = 7
    1-100 GHz EM simulation

The previous B3 candidate used a linear small-angle mapping:
    theta ~ gk*Z0/ZH
which is not the correct finite-line mapping once the electrical lengths
become large.

This version uses the standard stepped-impedance first-order mapping:
    high-Z series section:
        theta_H = asin(gk*Z0/ZH)

    low-Z shunt section:
        theta_L = asin(gk*ZL/Z0)

and then:
    l = theta * lambda_g / (2*pi)

with the effective dielectric constant of the actual microstrip section.

The resulting lengths are quantized to the existing 0.25 mm EM grid.

IMPORTANT:
-----------
This is still a distributed approximation. The exact openEMS response is the
final authority. The purpose of this run is to establish a physically
correct baseline before any optimization/cascade work.

Unlike the previous B3 code, the impedance steps are ABRUPT. The 1-mm
artificial taper transitions are removed because they add electrical length
and parasitic discontinuities that were not part of the synthesis.

Substrate:
    eps_r = 3.38
    h     = 1.50 mm

Microstrip:
    W50   = 3.50 mm
    WHIGH = 0.50 mm
    WLOW  = 12.50 mm

Expected synthesized sections before grid quantization are approximately:
    H 0.56 mm
    L 1.41 mm
    H 2.53 mm
    L 2.49 mm
    H 2.53 mm
    L 1.41 mm
    H 0.56 mm

Grid-quantized candidate:
    [0.50, 1.50, 2.50, 2.50, 2.50, 1.50, 0.50] mm

The ideal 7th-order Butterworth prototype has:
    S21(fc) = -3.0103 dB
and approximately:
    S21(20 GHz) = -42.14 dB
before distributed/nonideal effects.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
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
        "Could not import CSXCAD/openEMS. Run this from .venv-openems."
    ) from exc


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SIM_PATH = (
    PROJECT_ROOT
    / "simulations"
    / "openems"
    / "results"
    / "butterworth7_proper_synthesis_v1"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "em"
    / "butterworth7_proper_synthesis_v1"
)

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
H = 1.50                         # mm

W50 = 3.50                       # mm
W_HIGH = 0.50                    # mm
W_LOW = 12.50                    # mm

GRID = 0.25                      # mm

FILTER_ORDER = 7
FC = 10.0e9

ANALYSIS_START = 1.0e9
ANALYSIS_STOP = 100.0e9
N_FREQ = 1981

# 50 GHz Gaussian excitation, as in the validated V18 runs.
F0 = 50.0e9
FC_GAUSS = 50.0e9

NR_TS = 140000
END_CRITERIA = 1e-5

BOUNDARY = ["PML_8"] * 6

# RF line / ports.
LINE_X0 = -30.0
LINE_X1 = +30.0
FEED_LENGTH = 10.0

# Simulation cross-section.
Y_MIN = -10.0
Y_MAX = +10.0
Z_MIN = -3.0
Z_MAX = +11.50

C_LIGHT = 299792458.0


# ============================================================================
# BUTTERWORTH PROTOTYPE
# ============================================================================

G = [
    2.0 * math.sin((2.0 * k - 1.0) * math.pi / (2.0 * FILTER_ORDER))
    for k in range(1, FILTER_ORDER + 1)
]

print()
print("=" * 90)
print("7TH-ORDER BUTTERWORTH PROPER STEPPED-IMPEDANCE SYNTHESIS")
print("=" * 90)
print(f"Z0                    : {Z0:.3f} ohm")
print(f"fc                    : {FC/1e9:.3f} GHz")
print(f"order                 : {FILTER_ORDER}")
print(f"epsilon_r             : {EPS_R:.4f}")
print(f"substrate height      : {H:.3f} mm")
print(f"W50                   : {W50:.3f} mm")
print(f"W_HIGH                : {W_HIGH:.3f} mm")
print(f"W_LOW                 : {W_LOW:.3f} mm")
print(f"grid                  : {GRID:.3f} mm")
print()
print("Butterworth g-values:")
print("  " + ", ".join(f"{v:.10f}" for v in G))


# ============================================================================
# MICROSTRIP MODEL
# ============================================================================

def microstrip_eps_eff(width_mm: float) -> float:
    u = width_mm / H
    ee = (
        (EPS_R + 1.0) / 2.0
        + (EPS_R - 1.0) / 2.0
        * (1.0 + 12.0 / u) ** -0.5
    )
    if u < 1.0:
        ee += 0.04 * (1.0 - u) ** 2
    return ee


def microstrip_z0(width_mm: float) -> float:
    u = width_mm / H
    ee = microstrip_eps_eff(width_mm)

    if u <= 1.0:
        return (
            60.0 / math.sqrt(ee)
            * math.log(8.0 / u + u / 4.0)
        )

    return (
        120.0 * math.pi
        / (
            math.sqrt(ee)
            * (u + 1.393 + 0.667 * math.log(u + 1.444))
        )
    )


ZH = microstrip_z0(W_HIGH)
ZL = microstrip_z0(W_LOW)

EE_H = microstrip_eps_eff(W_HIGH)
EE_L = microstrip_eps_eff(W_LOW)

LAMBDA_H_MM = (
    C_LIGHT / FC / math.sqrt(EE_H) * 1e3
)
LAMBDA_L_MM = (
    C_LIGHT / FC / math.sqrt(EE_L) * 1e3
)

print()
print("MICROSTRIP EXTRACTION")
print("-" * 90)
print(f"Z_HIGH                : {ZH:.6f} ohm")
print(f"Z_LOW                 : {ZL:.6f} ohm")
print(f"eps_eff HIGH          : {EE_H:.6f}")
print(f"eps_eff LOW           : {EE_L:.6f}")
print(f"lambda_g HIGH @10GHz  : {LAMBDA_H_MM:.6f} mm")
print(f"lambda_g LOW  @10GHz  : {LAMBDA_L_MM:.6f} mm")


# ============================================================================
# PROPER FIRST-ORDER STEPPED-IMPEDANCE SYNTHESIS
# ============================================================================

def snap(value_mm: float) -> float:
    return GRID * round(value_mm / GRID)


section_types = []
section_widths = []
raw_lengths = []
section_lengths = []
electrical_degrees = []

for k, gk in enumerate(G):
    if k % 2 == 0:
        # Series-inductive high-Z section.
        theta = math.asin(gk * Z0 / ZH)
        width = W_HIGH
        lam = LAMBDA_H_MM
        kind = "HIGH"
    else:
        # Shunt-capacitive low-Z section.
        theta = math.asin(gk * ZL / Z0)
        width = W_LOW
        lam = LAMBDA_L_MM
        kind = "LOW"

    raw_length = theta / (2.0 * math.pi) * lam
    snapped_length = snap(raw_length)

    section_types.append(kind)
    section_widths.append(width)
    raw_lengths.append(raw_length)
    section_lengths.append(snapped_length)
    electrical_degrees.append(math.degrees(theta))

print()
print("SYNTHESIS RESULT")
print("-" * 90)
print(
    f"{'S':>3} {'type':>6} {'gk':>12} {'Z(ohm)':>10} "
    f"{'theta(deg)':>12} {'raw(mm)':>12} {'grid(mm)':>12}"
)

for i, (kind, gk, width, th, raw, q) in enumerate(
    zip(
        section_types,
        G,
        section_widths,
        electrical_degrees,
        raw_lengths,
        section_lengths,
    ),
    1,
):
    z = ZH if kind == "HIGH" else ZL
    print(
        f"{i:3d} {kind:>6} {gk:12.8f} {z:10.4f} "
        f"{th:12.4f} {raw:12.6f} {q:12.6f}"
    )

FILTER_LENGTH = sum(section_lengths)

print()
print(f"Raw section lengths       : {[round(x, 6) for x in raw_lengths]}")
print(f"Grid section lengths      : {section_lengths}")
print(f"Total filter length       : {FILTER_LENGTH:.3f} mm")


# ============================================================================
# GRID VALIDATION
# ============================================================================

def assert_grid(value_mm: float, name: str) -> None:
    q = value_mm / GRID
    if abs(q - round(q)) > 1e-9:
        raise RuntimeError(
            f"{name}={value_mm:.12f} mm is not aligned to "
            f"{GRID:.3f} mm grid."
        )


assert_grid(FILTER_LENGTH, "FILTER_LENGTH")

# Center filter on the grid.
FILTER_X0 = -FILTER_LENGTH / 2.0
FILTER_X1 = +FILTER_LENGTH / 2.0

assert_grid(FILTER_X0, "FILTER_X0")
assert_grid(FILTER_X1, "FILTER_X1")

for i, length in enumerate(section_lengths, 1):
    assert_grid(length, f"S{i} length")

print(f"Filter x extent           : {FILTER_X0:.3f} -> {FILTER_X1:.3f} mm")


# ============================================================================
# GEOMETRY
# ============================================================================

geometry = []

x = FILTER_X0

for i, (kind, width, length) in enumerate(
    zip(section_types, section_widths, section_lengths), 1
):
    x0 = x
    x1 = x + length
    geometry.append((x0, x1, width, f"S{i}_{kind}"))
    x = x1

if abs(x - FILTER_X1) > 1e-12:
    raise RuntimeError("Filter geometry does not close.")

for xa, xb, width, label in geometry:
    assert_grid(xa, f"{label} x0")
    assert_grid(xb, f"{label} x1")
    assert_grid(width, f"{label} width")


# ============================================================================
# IDEAL BUTTERWORTH REFERENCE
# ============================================================================

freq_ref = np.linspace(ANALYSIS_START, ANALYSIS_STOP, N_FREQ)

ideal_power = 1.0 / (
    1.0 + (freq_ref / FC) ** (2 * FILTER_ORDER)
)
ideal_s21_db = 10.0 * np.log10(np.maximum(ideal_power, 1e-300))

print()
print("IDEAL BUTTERWORTH REFERENCE")
print("-" * 90)

def idx_ref(f):
    return int(np.argmin(abs(freq_ref - f)))

for f in [5e9, 9e9, 10e9, 12e9, 20e9, 100e9]:
    print(
        f"S21 ideal @ {f/1e9:6.1f} GHz : "
        f"{ideal_s21_db[idx_ref(f)]:10.4f} dB"
    )


# ============================================================================
# MESH
# ============================================================================

def uniform_axis(vmin: float, vmax: float, pitch: float):
    n = int(round((vmax - vmin) / pitch))
    if abs((vmax - vmin) - n * pitch) > 1e-9:
        raise RuntimeError(
            f"Axis {vmin}->{vmax} is not an integer multiple of {pitch}."
        )
    return [round(vmin + i * pitch, 10) for i in range(n + 1)]


x_lines = uniform_axis(LINE_X0, LINE_X1, GRID)
y_lines = uniform_axis(Y_MIN, Y_MAX, GRID)
z_lines = uniform_axis(Z_MIN, Z_MAX, GRID)

# Explicit geometry boundaries.
x_critical = [LINE_X0, LINE_X1, FILTER_X0, FILTER_X1]
for xa, xb, width, _ in geometry:
    x_critical.extend([xa, xb])

y_critical = [0.0, -W50 / 2.0, W50 / 2.0]
for _, _, width, _ in geometry:
    y_critical.extend([-width / 2.0, width / 2.0])


def merge_lines(base, critical):
    vals = sorted(
        [round(float(v), 10) for v in base + critical]
    )
    result = []
    for v in vals:
        if not result or abs(v - result[-1]) >= 1e-8:
            result.append(v)
    return result


x_lines = merge_lines(x_lines, x_critical)
y_lines = merge_lines(y_lines, y_critical)
z_lines = merge_lines(
    z_lines,
    [Z_MIN, 0.0, H, Z_MAX],
)


def validate_axis(name, lines):
    d = np.diff(np.asarray(lines, dtype=float))
    if np.any(d <= 0):
        raise RuntimeError(f"{name} mesh has duplicate/non-increasing lines.")
    return float(d.min()), float(d.max())


min_dx, max_dx = validate_axis("x", x_lines)
min_dy, max_dy = validate_axis("y", y_lines)
min_dz, max_dz = validate_axis("z", z_lines)

estimated_cells = (
    (len(x_lines) - 1)
    * (len(y_lines) - 1)
    * (len(z_lines) - 1)
)

print()
print("MESH VALIDATION")
print("-" * 90)
print(f"x cells                   : {len(x_lines)-1}")
print(f"y cells                   : {len(y_lines)-1}")
print(f"z cells                   : {len(z_lines)-1}")
print(f"estimated cells           : {estimated_cells:,}")
print(f"min dx                    : {min_dx:.6f} mm")
print(f"min dy                    : {min_dy:.6f} mm")
print(f"min dz                    : {min_dz:.6f} mm")

if min_dx < GRID - 1e-8 or min_dy < GRID - 1e-8 or min_dz < GRID - 1e-8:
    raise RuntimeError("Sub-grid cell detected; refusing EM run.")

if estimated_cells < 1_000_000:
    raise RuntimeError("Unexpectedly small mesh; refusing EM run.")


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
    FC_GAUSS,
)


# ============================================================================
# SUBSTRATE / CONDUCTORS
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

ground.AddBox(
    start=[LINE_X0, Y_MIN, 0.0],
    stop=[LINE_X1, Y_MAX, 0.0],
    priority=10,
)

# Left 50-ohm feed.
signal.AddBox(
    start=[LINE_X0, -W50 / 2.0, H],
    stop=[FILTER_X0, W50 / 2.0, H],
    priority=10,
)

# Proper stepped-impedance filter: ABRUPT width discontinuities.
for xa, xb, width, label in geometry:
    signal.AddBox(
        start=[xa, -width / 2.0, H],
        stop=[xb, width / 2.0, H],
        priority=20,
    )

# Right 50-ohm feed.
signal.AddBox(
    start=[FILTER_X1, -W50 / 2.0, H],
    stop=[LINE_X1, W50 / 2.0, H],
    priority=10,
)

FDTD.AddEdges2Grid(dirs="xy", properties=ground)
FDTD.AddEdges2Grid(dirs="xy", properties=signal)


# ============================================================================
# MESH INTO CSX
# ============================================================================

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(UNIT)
mesh.AddLine("x", x_lines)
mesh.AddLine("y", y_lines)
mesh.AddLine("z", z_lines)


# ============================================================================
# MSL PORTS
# ============================================================================

port1 = FDTD.AddMSLPort(
    1,
    ground,
    [LINE_X0, -W50 / 2.0, H],
    [LINE_X0 + FEED_LENGTH, +W50 / 2.0, 0.0],
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
    [LINE_X1, -W50 / 2.0, H],
    [LINE_X1 - FEED_LENGTH, +W50 / 2.0, 0.0],
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

xml_path = SIM_PATH / "butterworth7_proper_synthesis_v1.xml"
CSX.Write2XML(str(xml_path))

print()
print("=" * 90)
print("STARTING FULL-WAVE OPENEMS RUN")
print("=" * 90)
print(f"XML                       : {xml_path}")
print(f"Frequency                 : 1–100 GHz")
print(f"Grid                      : {GRID:.3f} mm")
print(f"Filter length             : {FILTER_LENGTH:.3f} mm")
print(f"Section lengths           : {section_lengths}")
print("No artificial taper transitions are used.")


# ============================================================================
# RUN
# ============================================================================

FDTD.Run(
    str(SIM_PATH),
    cleanup=True,
    verbose=3,
)


# ============================================================================
# POSTPROCESS
# ============================================================================

print()
print("=" * 90)
print("POSTPROCESSING")
print("=" * 90)

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

eps = 1e-30

uf_inc_1 = np.asarray(port1.uf_inc).reshape(-1)
uf_ref_1 = np.asarray(port1.uf_ref).reshape(-1)
uf_ref_2 = np.asarray(port2.uf_ref).reshape(-1)

uf_tot_1 = np.asarray(port1.uf_tot).reshape(-1)
if_tot_1 = np.asarray(port1.if_tot).reshape(-1)

S11 = uf_ref_1 / (uf_inc_1 + eps)
S21 = uf_ref_2 / (uf_inc_1 + eps)

S11_dB = 20.0 * np.log10(np.maximum(np.abs(S11), eps))
S21_dB = 20.0 * np.log10(np.maximum(np.abs(S21), eps))

Zin = uf_tot_1 / (if_tot_1 + eps)
power = np.abs(S11) ** 2 + np.abs(S21) ** 2


def nearest(f):
    return int(np.argmin(np.abs(freq - f)))


i5 = nearest(5e9)
i8 = nearest(8e9)
i9 = nearest(9e9)
i10 = nearest(10e9)
i12 = nearest(12e9)
i15 = nearest(15e9)
i16 = nearest(16e9)
i17 = nearest(17e9)
i20 = nearest(20e9)
i100 = nearest(100e9)

mask_1_8 = (freq >= 1e9) & (freq <= 8e9)
mask_9_12 = (freq >= 9e9) & (freq <= 12e9)
mask_14_18 = (freq >= 14e9) & (freq <= 18e9)
mask_20_100 = (freq >= 20e9) & (freq <= 100e9)

max_1_8 = float(np.max(S21_dB[mask_1_8]))
min_1_8 = float(np.min(S21_dB[mask_1_8]))
max_9_12 = float(np.max(S21_dB[mask_9_12]))
max_14_18 = float(np.max(S21_dB[mask_14_18]))
max_20_100 = float(np.max(S21_dB[mask_20_100]))

f_max_14_18 = float(freq[mask_14_18][np.argmax(S21_dB[mask_14_18])])
f_max_20_100 = float(freq[mask_20_100][np.argmax(S21_dB[mask_20_100])])

print()
print("MEASURED EM RESULTS")
print("-" * 90)
print(f"S21 @ 5 GHz              : {S21_dB[i5]: .4f} dB")
print(f"S21 @ 8 GHz              : {S21_dB[i8]: .4f} dB")
print(f"S21 @ 9 GHz              : {S21_dB[i9]: .4f} dB")
print(f"S21 @ 10 GHz             : {S21_dB[i10]: .4f} dB")
print(f"S21 @ 12 GHz             : {S21_dB[i12]: .4f} dB")
print(f"S21 @ 15 GHz             : {S21_dB[i15]: .4f} dB")
print(f"S21 @ 16 GHz             : {S21_dB[i16]: .4f} dB")
print(f"S21 @ 17 GHz             : {S21_dB[i17]: .4f} dB")
print(f"S21 @ 20 GHz             : {S21_dB[i20]: .4f} dB")
print(f"S21 @ 100 GHz            : {S21_dB[i100]: .4f} dB")
print()
print(f"1–8 GHz S21 max          : {max_1_8: .4f} dB")
print(f"1–8 GHz S21 min          : {min_1_8: .4f} dB")
print(f"9–12 GHz S21 max         : {max_9_12: .4f} dB")
print(f"14–18 GHz S21 max        : {max_14_18: .4f} dB @ {f_max_14_18/1e9:.3f} GHz")
print(f"20–100 GHz S21 max       : {max_20_100: .4f} dB @ {f_max_20_100/1e9:.3f} GHz")
print(f"Power sum min/max        : {power.min():.6f} / {power.max():.6f}")


# ============================================================================
# SAVE CSV
# ============================================================================

csv_path = RESULTS_PATH / "butterworth7_proper_synthesis_v1.csv"

with csv_path.open("w", newline="", encoding="utf-8") as fh:
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
            float(freq[k]),
            float(S11[k].real),
            float(S11[k].imag),
            float(S21[k].real),
            float(S21[k].imag),
            float(S11_dB[k]),
            float(S21_dB[k]),
            float(Zin[k].real),
            float(Zin[k].imag),
            float(power[k]),
        ])


# ============================================================================
# SAVE METRICS / SYNTHESIS
# ============================================================================

metrics = {
    "order": FILTER_ORDER,
    "fc_Hz": FC,
    "Z0_ohm": Z0,
    "epsilon_r": EPS_R,
    "substrate_height_mm": H,
    "w50_mm": W50,
    "w_high_mm": W_HIGH,
    "w_low_mm": W_LOW,
    "z_high_ohm": ZH,
    "z_low_ohm": ZL,
    "g": G,
    "raw_lengths_mm": raw_lengths,
    "grid_lengths_mm": section_lengths,
    "filter_length_mm": FILTER_LENGTH,
    "s21_5GHz_dB": float(S21_dB[i5]),
    "s21_8GHz_dB": float(S21_dB[i8]),
    "s21_9GHz_dB": float(S21_dB[i9]),
    "s21_10GHz_dB": float(S21_dB[i10]),
    "s21_12GHz_dB": float(S21_dB[i12]),
    "s21_15GHz_dB": float(S21_dB[i15]),
    "s21_16GHz_dB": float(S21_dB[i16]),
    "s21_17GHz_dB": float(S21_dB[i17]),
    "s21_20GHz_dB": float(S21_dB[i20]),
    "s21_100GHz_dB": float(S21_dB[i100]),
    "max_s21_14_18_dB": max_14_18,
    "max_s21_14_18_frequency_Hz": f_max_14_18,
    "max_s21_20_100_dB": max_20_100,
    "max_s21_20_100_frequency_Hz": f_max_20_100,
    "power_min": float(power.min()),
    "power_max": float(power.max()),
}

with (RESULTS_PATH / "metrics.json").open("w", encoding="utf-8") as fh:
    json.dump(metrics, fh, indent=2)


with (RESULTS_PATH / "synthesis.txt").open("w", encoding="utf-8") as fh:
    fh.write("7th-order Butterworth stepped-impedance synthesis\n")
    fh.write("=" * 72 + "\n")
    fh.write(f"fc = {FC/1e9:.6f} GHz\n")
    fh.write(f"Z0 = {Z0:.6f} ohm\n")
    fh.write(f"Z_HIGH = {ZH:.6f} ohm\n")
    fh.write(f"Z_LOW = {ZL:.6f} ohm\n")
    fh.write(f"eps_eff_HIGH = {EE_H:.9f}\n")
    fh.write(f"eps_eff_LOW = {EE_L:.9f}\n")
    fh.write("\n")
    fh.write("k, type, gk, theta_deg, raw_mm, grid_mm\n")
    for i, row in enumerate(
        zip(
            section_types,
            G,
            electrical_degrees,
            raw_lengths,
            section_lengths,
        ),
        1,
    ):
        kind, gk, theta, raw, grid_len = row
        fh.write(
            f"{i}, {kind}, {gk:.10f}, {theta:.8f}, "
            f"{raw:.8f}, {grid_len:.8f}\n"
        )

    fh.write("\nEM results\n")
    for key, value in metrics.items():
        fh.write(f"{key}: {value}\n")


# ============================================================================
# PLOTS
# ============================================================================

plot_path = PLOTS_PATH / "butterworth7_proper_synthesis_v1.png"

plt.figure(figsize=(12, 7))
plt.plot(
    freq / 1e9,
    S11_dB,
    label="EM S11",
)
plt.plot(
    freq / 1e9,
    S21_dB,
    label="EM S21",
)
plt.plot(
    freq_ref / 1e9,
    ideal_s21_db,
    "--",
    label="Ideal 7th-order Butterworth S21",
)
plt.axvline(10.0, linestyle="--", label="10 GHz")
plt.axhline(-3.0103, linestyle=":", label="-3.01 dB")
plt.axhline(-40.0, linestyle=":", label="-40 dB")
plt.xlim(1.0, 100.0)
plt.ylim(-100.0, 5.0)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.title("Proper 7th-Order Butterworth Stepped-Impedance LPF — 1–100 GHz")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=160)
plt.close()

print()
print("=" * 90)
print("FILES")
print("=" * 90)
print(f"CSV                       : {csv_path}")
print(f"Metrics                   : {RESULTS_PATH / 'metrics.json'}")
print(f"Synthesis                 : {RESULTS_PATH / 'synthesis.txt'}")
print(f"Plot                      : {plot_path}")
print()
print("DONE")
