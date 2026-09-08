# -*- coding: utf-8 -*-
"""
7th-Order Chebyshev Stepped-Impedance Low-Pass Filter
FULL-WAVE openEMS V1 — corrected physical layout

IMPORTANT CORRECTION
--------------------
The seven synthesized sections total only 14.7838 mm, while the
validated MSL-port calibration has a 40 mm main-line region between
the two ports (-20 mm to +20 mm).

The previous version incorrectly required:

    sum(section lengths) == 40 mm

That is NOT physically required.

The correct structure is:

    50-ohm feed
       |
       | 10 mm MSL port feed
       |
       | 50-ohm line
       |
       +-- 120 -- 20 -- 120 -- 20 -- 120 -- 20 -- 120 --+
       |                                                   |
       |             7-section filter                     |
       |                                                   |
       +---------------- 50-ohm line --------------------+
       |
       | 10 mm MSL port feed
       |
    50-ohm feed

The 14.7838 mm filter is centered inside the 40 mm region between the
MSL ports. Therefore:

    40.0000 - 14.7838 = 25.2162 mm

and the remaining 50-ohm line is split symmetrically:

    12.6081 mm on each side of the stepped filter.

This preserves the validated MSL port geometry while giving the actual
synthesized filter its correct physical length.

The 50-ohm lead sections do not change the ideal magnitude response;
they mainly introduce phase/reference-plane delay. The MSL ports remain
at exactly the same locations as the passed calibration.

Design target
-------------
Z0                    = 50 ohm
Chebyshev Type-I      = 0.1 dB ripple
Passband edge         = 9 GHz
Stopband start        = 20 GHz
Required attenuation  = >= 60 dB @ 20 GHz
Order                 = 7

Initial stepped-impedance dimensions
------------------------------------
Section 1: 120 ohm, W = 0.5479 mm, L = 1.6798 mm
Section 2:  20 ohm, W = 12.3247 mm, L = 1.7582 mm
Section 3: 120 ohm, W = 0.5479 mm, L = 2.9818 mm
Section 4:  20 ohm, W = 12.3247 mm, L = 1.9442 mm
Section 5: 120 ohm, W = 0.5479 mm, L = 2.9818 mm
Section 6:  20 ohm, W = 12.3247 mm, L = 1.7582 mm
Section 7: 120 ohm, W = 0.5479 mm, L = 1.6798 mm

This is the first physical full-wave baseline. These dimensions are
an initial stepped-impedance approximation, not final fabricated
dimensions.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================================
# WINDOWS openEMS DLL SETUP
# =========================================================================
OPENEMS_PACKAGE = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
    / "openEMS"
)

if not OPENEMS_PACKAGE.exists():
    raise FileNotFoundError(
        f"openEMS package not found at:\n{OPENEMS_PACKAGE}"
    )

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_PACKAGE)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_PACKAGE)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(OPENEMS_PACKAGE))

os.environ["PATH"] = str(OPENEMS_PACKAGE) + os.pathsep + os.environ["PATH"]

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# =========================================================================
# PROJECT PATHS
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SIM_PATH = (
    PROJECT_ROOT
    / "simulations"
    / "openems"
    / "results"
    / "chebyshev_7th_order_v1"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "em"
    / "chebyshev_7th_order_v1"
)

PLOTS_PATH = PROJECT_ROOT / "results" / "plots"

SIM_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Remove stale EM data.
for p in SIM_PATH.iterdir():
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


# =========================================================================
# UNITS
# =========================================================================
UNIT = 1e-3


# =========================================================================
# FILTER SPECIFICATION
# =========================================================================
Z0 = 50.0

CUTOFF_FREQUENCY = 9.0e9
PASSBAND_RIPPLE_DB = 0.1

STOPBAND_START = 20.0e9
STOPBAND_ATTENUATION_DB = 60.0

FILTER_ORDER = 7


# =========================================================================
# SUBSTRATE
# =========================================================================
EPS_R = 3.38
H = 1.524                       # mm


# =========================================================================
# VALIDATED 50-OHM FEED
# =========================================================================
W50 = 3.5577                     # mm


# =========================================================================
# STEPPED-IMPEDANCE INITIAL GEOMETRY
# =========================================================================
W_HIGH = 0.5479                  # mm, approximately 120 ohm
W_LOW = 12.3247                  # mm, approximately 20 ohm

SECTION_LENGTHS = np.array([
    1.6798,
    1.7582,
    2.9818,
    1.9442,
    2.9818,
    1.7582,
    1.6798,
], dtype=float)

SECTION_WIDTHS = np.array([
    W_HIGH,
    W_LOW,
    W_HIGH,
    W_LOW,
    W_HIGH,
    W_LOW,
    W_HIGH,
], dtype=float)

SECTION_Z = np.array([
    120.0,
    20.0,
    120.0,
    20.0,
    120.0,
    20.0,
    120.0,
], dtype=float)


# =========================================================================
# VALIDATED MSL-PORT REFERENCE GEOMETRY
#
# Exactly the same port positions as the passed uniform-line calibration.
# =========================================================================
LINE_X0 = -30.0
LINE_X1 = +30.0

PORT1_X = -30.0
PORT2_X = +30.0

MAIN_REGION_X0 = -20.0
MAIN_REGION_X1 = +20.0

MSL_FEED_LENGTH = 10.0


# =========================================================================
# CORRECT FILTER PLACEMENT
#
# The seven sections occupy 14.7838 mm centered around x=0.
# The remaining 25.2162 mm is 50-ohm line:
#
#       12.6081 mm       14.7838 mm       12.6081 mm
#    50-ohm lead    stepped filter      50-ohm lead
# =========================================================================
FILTER_LENGTH = float(np.sum(SECTION_LENGTHS))
MAIN_REGION_LENGTH = MAIN_REGION_X1 - MAIN_REGION_X0

if FILTER_LENGTH > MAIN_REGION_LENGTH:
    raise ValueError(
        f"Filter length {FILTER_LENGTH:.6f} mm is larger than the "
        f"available main region {MAIN_REGION_LENGTH:.6f} mm."
    )

LEAD_LENGTH = (
    MAIN_REGION_LENGTH - FILTER_LENGTH
) / 2.0

FILTER_START_X = MAIN_REGION_X0 + LEAD_LENGTH
FILTER_STOP_X = MAIN_REGION_X1 - LEAD_LENGTH

if not np.isclose(
    FILTER_STOP_X - FILTER_START_X,
    FILTER_LENGTH,
    atol=1e-9,
):
    raise RuntimeError("Filter placement arithmetic failed.")


# =========================================================================
# BUILD SECTION BOUNDARIES
# =========================================================================
section_x0 = []
section_x1 = []

x = FILTER_START_X

for L in SECTION_LENGTHS:
    section_x0.append(float(x))
    x += float(L)
    section_x1.append(float(x))

section_x0 = np.array(section_x0)
section_x1 = np.array(section_x1)

# Force exact endpoint.
section_x1[-1] = FILTER_STOP_X


# =========================================================================
# TRANSVERSE / VERTICAL SIMULATION DIMENSIONS
# =========================================================================
# W_LOW is 12.3247 mm wide. ±15 mm leaves 2.6753 mm side clearance.
# This is intentionally kept consistent with the validated MSL model.
Y_MIN = -15.0
Y_MAX = +15.0

Z_MIN = -5.0
Z_MAX = H + 20.0


# =========================================================================
# FDTD SETTINGS
# =========================================================================
# Centered at 15 GHz so the broadband pulse contains strong content
# around the 20 GHz design point.
F0 = 15.0e9
FC = 15.0e9

FREQ_START = 1.0e9
FREQ_STOP = 30.0e9
N_FREQ = 1451

NR_TS = 200000
END_CRITERIA = 1e-5

BOUNDARY = [
    "PML_8", "PML_8",
    "PML_8", "PML_8",
    "PML_8", "PML_8",
]


# =========================================================================
# MESH
# =========================================================================
# 0.20 mm target in xy.
#
# The 0.5479 mm high-Z section is only ~2.74 target cells wide before
# exact boundary insertion, so this is intentionally finer than the
# 0.25 mm calibration model.
RES_XY = 0.20

N_SUB_Z = 12
SUB_Z = np.linspace(
    0.0,
    H,
    N_SUB_Z + 1,
)


# =========================================================================
# BANNER
# =========================================================================
print("=" * 78)
print("7TH-ORDER CHEBYSHEV STEPPED-IMPEDANCE FILTER — FULL-WAVE V1")
print("=" * 78)

print(f"Z0                    : {Z0:.1f} ohm")
print(f"Chebyshev ripple      : {PASSBAND_RIPPLE_DB:.3f} dB")
print(f"Passband edge         : {CUTOFF_FREQUENCY/1e9:.3f} GHz")
print(f"Stopband start        : {STOPBAND_START/1e9:.3f} GHz")
print(f"Required attenuation  : {STOPBAND_ATTENUATION_DB:.1f} dB")
print(f"Filter order          : {FILTER_ORDER}")

print("-" * 78)

print(f"eps_r                 : {EPS_R}")
print(f"substrate thickness   : {H:.6f} mm")
print(f"50-ohm feed width     : {W50:.6f} mm")
print(f"high-Z width          : {W_HIGH:.6f} mm")
print(f"low-Z width           : {W_LOW:.6f} mm")

print("-" * 78)

print(f"MSL physical line     : {LINE_X0:.3f} -> {LINE_X1:.3f} mm")
print(f"MSL port separation   : {PORT1_X:.3f} -> {PORT2_X:.3f} mm")
print(f"main region           : {MAIN_REGION_X0:.3f} -> {MAIN_REGION_X1:.3f} mm")
print(f"main-region length    : {MAIN_REGION_LENGTH:.6f} mm")
print(f"filter length         : {FILTER_LENGTH:.6f} mm")
print(f"50-ohm lead each side : {LEAD_LENGTH:.6f} mm")
print(f"filter region         : {FILTER_START_X:.6f} -> {FILTER_STOP_X:.6f} mm")

print("-" * 78)

print(f"xy mesh target        : {RES_XY:.3f} mm")
print(f"substrate z cells     : {N_SUB_Z}")
print(f"frequency             : {FREQ_START/1e9:.1f} -> {FREQ_STOP/1e9:.1f} GHz")
print(f"FDTD excitation       : F0={F0/1e9:.1f} GHz, FC={FC/1e9:.1f} GHz")
print(f"timesteps             : {NR_TS}")
print(f"boundary              : {BOUNDARY}")
print(f"simulation path       : {SIM_PATH}")

print("=" * 78)


# =========================================================================
# SECTION TABLE
# =========================================================================
print("\nSECTION TABLE")
print("-" * 82)
print(
    "Sec   Z_design(ohm)   Width(mm)   Length(mm)"
    "    x_start(mm)    x_stop(mm)"
)
print("-" * 82)

for i in range(FILTER_ORDER):
    print(
        f"{i+1:>3d}"
        f"   {SECTION_Z[i]:>12.1f}"
        f"   {SECTION_WIDTHS[i]:>10.4f}"
        f"   {SECTION_LENGTHS[i]:>10.4f}"
        f"   {section_x0[i]:>11.4f}"
        f"   {section_x1[i]:>10.4f}"
    )

print("-" * 82)


# =========================================================================
# CREATE FDTD
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
# INITIAL MESH
# =========================================================================
mesh.AddLine(
    "x",
    [
        LINE_X0 - 10.0,
        LINE_X0,
        MAIN_REGION_X0,
        FILTER_START_X,
        0.0,
        FILTER_STOP_X,
        MAIN_REGION_X1,
        LINE_X1,
        LINE_X1 + 10.0,
    ],
)

mesh.AddLine(
    "y",
    [
        Y_MIN,
        -W_LOW / 2.0,
        -W_HIGH / 2.0,
        -W50 / 2.0,
        0.0,
        W50 / 2.0,
        W_HIGH / 2.0,
        W_LOW / 2.0,
        Y_MAX,
    ],
)

mesh.AddLine(
    "z",
    [
        Z_MIN,
        0.0,
        H,
        Z_MAX,
    ],
)

mesh.AddLine(
    "z",
    SUB_Z,
)

# Exact mesh planes at every impedance transition.
for xx in np.concatenate(
    (
        section_x0,
        section_x1,
    )
):
    mesh.AddLine(
        "x",
        [float(xx)],
    )


# =========================================================================
# SUBSTRATE
# =========================================================================
substrate = CSX.AddMaterial(
    "substrate",
    epsilon=EPS_R,
)

substrate.AddBox(
    start=[
        LINE_X0,
        Y_MIN,
        0.0,
    ],
    stop=[
        LINE_X1,
        Y_MAX,
        H,
    ],
)


# =========================================================================
# GROUND
#
# Same topology as the validated MSL calibration.
# =========================================================================
ground = CSX.AddMetal(
    "GROUND",
)

ground.AddBox(
    start=[
        LINE_X0,
        Y_MIN,
        0.0,
    ],
    stop=[
        LINE_X1,
        Y_MAX,
        0.0,
    ],
    priority=5,
)

FDTD.AddEdges2Grid(
    dirs="xy",
    properties=ground,
)


# =========================================================================
# SIGNAL METAL
#
# There are THREE kinds of signal regions:
#
# 1. 50-ohm left lead  : -20 -> FILTER_START_X
# 2. seven stepped sections
# 3. 50-ohm right lead : FILTER_STOP_X -> +20
#
# The MSL port itself supplies the -30 -> -20 and +30 -> +20 feed
# sections.
# =========================================================================
signal = CSX.AddMetal(
    "SIGNAL",
)


# -------------------------------------------------------------------------
# Left 50-ohm lead
# -------------------------------------------------------------------------
if LEAD_LENGTH > 0:
    signal.AddBox(
        start=[
            MAIN_REGION_X0,
            -W50 / 2.0,
            H,
        ],
        stop=[
            FILTER_START_X,
            +W50 / 2.0,
            H,
        ],
        priority=10,
    )


# -------------------------------------------------------------------------
# Seven stepped sections
# -------------------------------------------------------------------------
for i in range(FILTER_ORDER):

    signal.AddBox(
        start=[
            section_x0[i],
            -SECTION_WIDTHS[i] / 2.0,
            H,
        ],
        stop=[
            section_x1[i],
            +SECTION_WIDTHS[i] / 2.0,
            H,
        ],
        priority=10,
    )


# -------------------------------------------------------------------------
# Right 50-ohm lead
# -------------------------------------------------------------------------
if LEAD_LENGTH > 0:
    signal.AddBox(
        start=[
            FILTER_STOP_X,
            -W50 / 2.0,
            H,
        ],
        stop=[
            MAIN_REGION_X1,
            +W50 / 2.0,
            H,
        ],
        priority=10,
    )


FDTD.AddEdges2Grid(
    dirs="xy",
    properties=signal,
)


# =========================================================================
# FINAL CRITICAL MESH LINES
# =========================================================================
mesh.AddLine(
    "x",
    [
        LINE_X0,
        MAIN_REGION_X0,
        FILTER_START_X,
        0.0,
        FILTER_STOP_X,
        MAIN_REGION_X1,
        LINE_X1,
    ],
)

for xx in np.concatenate(
    (
        section_x0,
        section_x1,
    )
):
    mesh.AddLine(
        "x",
        [float(xx)],
    )

mesh.AddLine(
    "y",
    [
        Y_MIN,
        -W_LOW / 2.0,
        -W_HIGH / 2.0,
        -W50 / 2.0,
        0.0,
        W50 / 2.0,
        W_HIGH / 2.0,
        W_LOW / 2.0,
        W_LOW / 2.0,
        Y_MAX,
    ],
)

mesh.AddLine(
    "z",
    [
        Z_MIN,
        0.0,
        H,
        Z_MAX,
    ],
)

mesh.AddLine(
    "z",
    SUB_Z,
)


# =========================================================================
# MSL PORT 1
#
# EXACT SAME PORT TOPOLOGY AS THE PASSED UNIFORM-LINE CALIBRATION.
# =========================================================================
port1_start = [
    LINE_X0,
    -W50 / 2.0,
    H,
]

port1_stop = [
    MAIN_REGION_X0,
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


# =========================================================================
# MSL PORT 2
# =========================================================================
port2_start = [
    LINE_X1,
    -W50 / 2.0,
    H,
]

port2_stop = [
    MAIN_REGION_X1,
    +W50 / 2.0,
    0.0,
]

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
CSX_FILE = (
    SIM_PATH
    / "chebyshev_7th_order_v1.xml"
)

CSX.Write2XML(
    str(CSX_FILE)
)

print("\nMSL PORT GEOMETRY")
print("-" * 78)
print(f"P1 start = {port1_start}")
print(f"P1 stop  = {port1_stop}")
print(f"P2 start = {port2_start}")
print(f"P2 stop  = {port2_stop}")

print("\nPHYSICAL FILTER PLACEMENT")
print("-" * 78)
print(f"Left 50-ohm lead  : {MAIN_REGION_X0:.6f} -> {FILTER_START_X:.6f} mm")
print(f"Stepped filter    : {FILTER_START_X:.6f} -> {FILTER_STOP_X:.6f} mm")
print(f"Right 50-ohm lead : {FILTER_STOP_X:.6f} -> {MAIN_REGION_X1:.6f} mm")
print(f"Total stepped     : {FILTER_LENGTH:.6f} mm")

print(f"\nXML = {CSX_FILE}")


# =========================================================================
# RUN
# =========================================================================
print("\n" + "=" * 78)
print("RUNNING OPENEMS")
print("=" * 78)
print("This is the first full-wave simulation of the actual 7-section filter.")
print("The validated MSL port geometry is unchanged.")
print()

FDTD.Run(
    str(SIM_PATH),
    cleanup=False,
)


# =========================================================================
# POST-PROCESSING
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

S11_dB = 20.0 * np.log10(
    np.maximum(
        np.abs(S11),
        1e-15,
    )
)

S21_dB = 20.0 * np.log10(
    np.maximum(
        np.abs(S21),
        1e-15,
    )
)

power_sum = (
    np.abs(S11) ** 2
    + np.abs(S21) ** 2
)


# =========================================================================
# FREQUENCY INDICES
# =========================================================================
def nearest_index(target_hz):
    return int(
        np.argmin(
            np.abs(freq - target_hz)
        )
    )


idx_5 = nearest_index(5e9)
idx_9 = nearest_index(9e9)
idx_20 = nearest_index(20e9)


# =========================================================================
# FILTER REGIONS
# =========================================================================
passband = (
    (freq >= 1e9)
    & (freq <= CUTOFF_FREQUENCY)
)

stopband = (
    freq >= STOPBAND_START
)


# =========================================================================
# FILTER METRICS
# =========================================================================
passband_worst_S21 = float(
    np.min(
        S21_dB[passband]
    )
)

passband_max_loss = float(
    -passband_worst_S21
)

passband_worst_S11 = float(
    np.max(
        S11_dB[passband]
    )
)

attenuation_20GHz = float(
    -S21_dB[idx_20]
)

stopband_max_S21 = float(
    np.max(
        S21_dB[stopband]
    )
)

stopband_min_S21 = float(
    np.min(
        S21_dB[stopband]
    )
)

worst_stopband_attenuation = float(
    -stopband_max_S21
)

power_min_passband = float(
    np.min(
        power_sum[passband]
    )
)

power_max_passband = float(
    np.max(
        power_sum[passband]
    )
)


# =========================================================================
# 5-GHz DIAGNOSTIC
# =========================================================================
print("\n" + "=" * 78)
print("7TH-ORDER CHEBYSHEV FULL-WAVE RESULTS")
print("=" * 78)

print("\n5 GHz diagnostic")
print("-" * 78)

print(
    f"Frequency              : "
    f"{freq[idx_5]/1e9:.6f} GHz"
)

print(
    f"S11                    : "
    f"{S11_dB[idx_5]:.6f} dB"
)

print(
    f"S21                    : "
    f"{S21_dB[idx_5]:.6f} dB"
)

print(
    f"Power sum              : "
    f"{power_sum[idx_5]:.6f}"
)

print(
    f"Zin                    : "
    f"{np.real(Zin[idx_5]):.6f} "
    f"{np.imag(Zin[idx_5]):+.6f}j ohm"
)

print(
    f"|Zin|                  : "
    f"{abs(Zin[idx_5]):.6f} ohm"
)


# =========================================================================
# PASSBAND
# =========================================================================
print("\nPassband: 1–9 GHz")
print("-" * 78)

print(
    f"Worst S21              : "
    f"{passband_worst_S21:.6f} dB"
)

print(
    f"Maximum passband loss  : "
    f"{passband_max_loss:.6f} dB"
)

print(
    f"Worst S11              : "
    f"{passband_worst_S11:.6f} dB"
)

print(
    f"Power min              : "
    f"{power_min_passband:.6f}"
)

print(
    f"Power max              : "
    f"{power_max_passband:.6f}"
)


# =========================================================================
# CRITICAL FREQUENCIES
# =========================================================================
print("\nCritical frequencies")
print("-" * 78)

print(
    f"S21 @ 9 GHz            : "
    f"{S21_dB[idx_9]:.6f} dB"
)

print(
    f"S21 @ 20 GHz           : "
    f"{S21_dB[idx_20]:.6f} dB"
)

print(
    f"Attenuation @ 20 GHz   : "
    f"{attenuation_20GHz:.6f} dB"
)


# =========================================================================
# STOPBAND
# =========================================================================
print("\nStopband: 20–30 GHz")
print("-" * 78)

print(
    f"Maximum S21            : "
    f"{stopband_max_S21:.6f} dB"
)

print(
    f"Minimum S21            : "
    f"{stopband_min_S21:.6f} dB"
)

print(
    f"Worst attenuation      : "
    f"{worst_stopband_attenuation:.6f} dB"
)


# =========================================================================
# INITIAL SANITY CHECKS
#
# These are deliberately NOT called "final filter acceptance".
# The geometry is an initial stepped-impedance approximation.
# =========================================================================
PASS_PASSBAND_SANITY = (
    passband_worst_S21 > -3.0
)

PASS_20DB = (
    attenuation_20GHz >= STOPBAND_ATTENUATION_DB
)

print("\nV1 sanity checks")
print("-" * 78)

print(
    f"Passband S21 > -3 dB   : "
    f"{'PASS' if PASS_PASSBAND_SANITY else 'FAIL'}"
)

print(
    f"Attenuation @20 >=60dB : "
    f"{'PASS' if PASS_20DB else 'FAIL'}"
)

if PASS_20DB and PASS_PASSBAND_SANITY:
    print("\nV1 STATUS: INITIAL EM TARGET PASSED")
    print(
        "Proceed to extended 20–70 GHz verification and final tuning."
    )
else:
    print("\nV1 STATUS: INITIAL EM TARGET NOT YET MET")
    print(
        "This is an initial full-wave realization. "
        "Use the response to tune the stepped sections."
    )


# =========================================================================
# CSV
# =========================================================================
csv_path = (
    RESULTS_PATH
    / "chebyshev_7th_order_v1.csv"
)

df = pd.DataFrame({
    "frequency_Hz": freq,
    "S11_dB": S11_dB,
    "S21_dB": S21_dB,
    "S11_mag": np.abs(S11),
    "S21_mag": np.abs(S21),
    "power_sum": power_sum,
    "Zin_real_ohm": np.real(Zin),
    "Zin_imag_ohm": np.imag(Zin),
    "Zin_mag_ohm": np.abs(Zin),
})

df.to_csv(
    csv_path,
    index=False,
)


# =========================================================================
# PLOT 1 — S11 / S21
# =========================================================================
fig, ax = plt.subplots(
    figsize=(11, 6.5)
)

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
    CUTOFF_FREQUENCY / 1e9,
    linestyle="--",
    linewidth=1.0,
    label="9 GHz cutoff",
)

ax.axvline(
    STOPBAND_START / 1e9,
    linestyle=":",
    linewidth=1.2,
    label="20 GHz stopband",
)

ax.axhline(
    -STOPBAND_ATTENUATION_DB,
    linestyle="--",
    linewidth=1.0,
    label="-60 dB",
)

ax.set_xlabel(
    "Frequency (GHz)"
)

ax.set_ylabel(
    "Magnitude (dB)"
)

ax.set_title(
    "7th-Order Chebyshev Stepped-Impedance Filter — openEMS"
)

ax.grid(True)
ax.legend()
fig.tight_layout()

sparam_plot = (
    PLOTS_PATH
    / "chebyshev_7th_order_v1_sparams.png"
)

fig.savefig(
    sparam_plot,
    dpi=180,
)

plt.close(fig)


# =========================================================================
# PLOT 2 — S21
# =========================================================================
fig, ax = plt.subplots(
    figsize=(11, 6.5)
)

ax.plot(
    freq / 1e9,
    S21_dB,
    label="S21",
)

ax.axvline(
    CUTOFF_FREQUENCY / 1e9,
    linestyle="--",
    linewidth=1.0,
    label="9 GHz",
)

ax.axvline(
    STOPBAND_START / 1e9,
    linestyle=":",
    linewidth=1.2,
    label="20 GHz",
)

ax.axhline(
    -STOPBAND_ATTENUATION_DB,
    linestyle="--",
    linewidth=1.0,
    label="-60 dB",
)

ax.set_xlabel(
    "Frequency (GHz)"
)

ax.set_ylabel(
    "S21 (dB)"
)

ax.set_title(
    "7th-Order Chebyshev Filter — Transmission"
)

ax.set_xlim(
    FREQ_START / 1e9,
    FREQ_STOP / 1e9,
)

ax.grid(True)
ax.legend()
fig.tight_layout()

s21_plot = (
    PLOTS_PATH
    / "chebyshev_7th_order_v1_s21.png"
)

fig.savefig(
    s21_plot,
    dpi=180,
)

plt.close(fig)


# =========================================================================
# PLOT 3 — INPUT IMPEDANCE
# =========================================================================
fig, ax = plt.subplots(
    figsize=(11, 6.5)
)

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
    Z0,
    linestyle="--",
    linewidth=1.0,
    label="50 ohm",
)

ax.axhline(
    0.0,
    linestyle=":",
    linewidth=1.0,
)

ax.set_xlabel(
    "Frequency (GHz)"
)

ax.set_ylabel(
    "Impedance (ohm)"
)

ax.set_title(
    "7th-Order Chebyshev Filter — Input Impedance"
)

ax.grid(True)
ax.legend()
fig.tight_layout()

zin_plot = (
    PLOTS_PATH
    / "chebyshev_7th_order_v1_zin.png"
)

fig.savefig(
    zin_plot,
    dpi=180,
)

plt.close(fig)


# =========================================================================
# PLOT 4 — POWER CONSERVATION
# =========================================================================
fig, ax = plt.subplots(
    figsize=(11, 6.5)
)

ax.plot(
    freq / 1e9,
    power_sum,
    label="|S11|² + |S21|²",
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
    label="Ideal lossless = 1",
)

ax.set_xlabel(
    "Frequency (GHz)"
)

ax.set_ylabel(
    "Power ratio"
)

ax.set_title(
    "7th-Order Chebyshev Filter — Power Conservation"
)

ax.grid(True)
ax.legend()
fig.tight_layout()

power_plot = (
    PLOTS_PATH
    / "chebyshev_7th_order_v1_power.png"
)

fig.savefig(
    power_plot,
    dpi=180,
)

plt.close(fig)


# =========================================================================
# SUMMARY FILE
# =========================================================================
report_path = (
    RESULTS_PATH
    / "chebyshev_7th_order_v1_summary.txt"
)

report = f"""
7th-Order Chebyshev Stepped-Impedance Filter — Full-Wave V1
============================================================

Specification
-------------
Z0                  : {Z0:.6f} ohm
Passband edge       : {CUTOFF_FREQUENCY/1e9:.6f} GHz
Ripple              : {PASSBAND_RIPPLE_DB:.6f} dB
Stopband start      : {STOPBAND_START/1e9:.6f} GHz
Required @20 GHz    : {STOPBAND_ATTENUATION_DB:.6f} dB
Order               : {FILTER_ORDER}

Substrate
---------
epsilon_r           : {EPS_R:.6f}
height              : {H:.6f} mm

Physical placement
------------------
Main MSL region     : {MAIN_REGION_X0:.6f} -> {MAIN_REGION_X1:.6f} mm
Stepped filter      : {FILTER_START_X:.6f} -> {FILTER_STOP_X:.6f} mm
Stepped length      : {FILTER_LENGTH:.6f} mm
50-ohm lead each    : {LEAD_LENGTH:.6f} mm

Initial dimensions
------------------
50-ohm feed width   : {W50:.6f} mm
High-Z width        : {W_HIGH:.6f} mm
Low-Z width         : {W_LOW:.6f} mm

Section lengths [mm]
--------------------
{", ".join(f"{v:.6f}" for v in SECTION_LENGTHS)}

Section widths [mm]
-------------------
{", ".join(f"{v:.6f}" for v in SECTION_WIDTHS)}

Results
-------
S11 @ 5 GHz         : {S11_dB[idx_5]:.6f} dB
S21 @ 5 GHz         : {S21_dB[idx_5]:.6f} dB
Zin @ 5 GHz         : {np.real(Zin[idx_5]):.6f} {np.imag(Zin[idx_5]):+.6f}j ohm

S21 @ 9 GHz         : {S21_dB[idx_9]:.6f} dB
S21 @ 20 GHz        : {S21_dB[idx_20]:.6f} dB
Attenuation @20 GHz : {attenuation_20GHz:.6f} dB

Worst S21, 1-9 GHz  : {passband_worst_S21:.6f} dB
Max S21, 20-30 GHz  : {stopband_max_S21:.6f} dB
Min S21, 20-30 GHz  : {stopband_min_S21:.6f} dB

Power min, 1-9 GHz  : {power_min_passband:.6f}
Power max, 1-9 GHz  : {power_max_passband:.6f}

V1 checks
---------
Passband sanity     : {"PASS" if PASS_PASSBAND_SANITY else "FAIL"}
20-GHz attenuation  : {"PASS" if PASS_20DB else "FAIL"}

This V1 is an initial full-wave EM baseline.
It is not yet the final fabricated geometry.
"""

report_path.write_text(
    report.strip() + "\n",
    encoding="utf-8",
)


# =========================================================================
# FINAL
# =========================================================================
print("\n" + "=" * 78)
print("FILES SAVED")
print("=" * 78)

print(f"CSV       : {csv_path}")
print(f"S11/S21   : {sparam_plot}")
print(f"S21       : {s21_plot}")
print(f"Zin       : {zin_plot}")
print(f"Power     : {power_plot}")
print(f"Summary   : {report_path}")
print(f"XML       : {CSX_FILE}")

print("=" * 78)

print("\nV1 COMPLETE.")
print(
    "The stepped-impedance sections are centered inside the validated "
    "40-mm MSL measurement region."
)
