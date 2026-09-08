# -*- coding: utf-8 -*-
"""
Canonical openEMS MSL-port validation for the cryogenic-filter project.

PURPOSE
-------
This is the final uniform transmission-line calibration model before
building the 7th-order stepped-impedance filter.

There is NO filter discontinuity in this model.

Topology:
    PML | 10 mm MSL feed | 40 mm uniform line | 10 mm MSL feed | PML

The MSL ports follow the openEMS microstrip-port construction:
    - port start is on the signal plane
    - port stop is referenced to the ground plane
    - propagation direction is x
    - excitation direction is z
    - active port has FeedShift and Feed_R
    - measurement plane is shifted into the line

IMPORTANT
---------
The previous version attempted to access port1.ZL / port2.ZL.
That attribute does not exist in the installed Python MSLPort class.

Also, Z_ref is NOT the extracted characteristic impedance when
CalcPort(..., ref_impedance=50) is used: it is the reference impedance
used for S-parameter normalization. Therefore this version does NOT
pretend that Z_ref is a frequency-dependent characteristic impedance.

The physically meaningful validation quantities here are:
    S11
    S21
    |S11|^2 + |S21|^2
    Zin

The MSL object's beta is retained only as an optional diagnostic if
the installed binding exposes it.

The EM simulation itself is unchanged from the successful run.
Only the broken post-processing has been corrected.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Windows openEMS setup
# ---------------------------------------------------------------------
OPENEMS_PACKAGE = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
    / "openEMS"
)

if not OPENEMS_PACKAGE.exists():
    raise FileNotFoundError(
        f"openEMS package not found: {OPENEMS_PACKAGE}"
    )

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_PACKAGE)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_PACKAGE)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(OPENEMS_PACKAGE))

os.environ["PATH"] = (
    str(OPENEMS_PACKAGE) + os.pathsep + os.environ["PATH"]
)

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# This file is expected at:
#   cryogenic_filter_design/
#       simulations/openems/model/<this_file>
#
# parents[2] therefore resolves to cryogenic_filter_design.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SIM_PATH = (
    PROJECT_ROOT
    / "simulations"
    / "openems"
    / "results"
    / "msl_canonical"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "em"
    / "msl_canonical"
)

PLOTS_PATH = PROJECT_ROOT / "results" / "plots"

SIM_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Remove only this validation's old simulation files.
for p in SIM_PATH.iterdir():
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


# ---------------------------------------------------------------------
# Physical geometry
# ---------------------------------------------------------------------
UNIT = 1e-3

Z0 = 50.0
EPS_R = 3.38

H = 1.524                  # substrate thickness [mm]
W = 3.5577                 # nominal 50-ohm width [mm]

# Physical line
LINE_X0 = -30.0
LINE_X1 = +30.0
LINE_LENGTH = LINE_X1 - LINE_X0

# MSL feed sections
FEED_LENGTH = 10.0

PORT1_START_X = LINE_X0
PORT1_STOP_X = LINE_X0 + FEED_LENGTH       # -20 mm

PORT2_START_X = LINE_X1
PORT2_STOP_X = LINE_X1 - FEED_LENGTH       # +20 mm

# Transverse simulation dimensions
Y_MIN = -15.0
Y_MAX = +15.0

# Vertical simulation dimensions
Z_MIN = -5.0
Z_MAX = H + 20.0


# ---------------------------------------------------------------------
# FDTD parameters
# ---------------------------------------------------------------------
F0 = 6.5e9
FC = 5.5e9

FREQ_START = 1.0e9
FREQ_STOP = 12.0e9
N_FREQ = 1101

NR_TS = 180000
END_CRITERIA = 1e-5

BOUNDARY = [
    "PML_8", "PML_8",
    "PML_8", "PML_8",
    "PML_8", "PML_8",
]

# XY mesh target.
RES_XY = 0.25

# Explicit substrate discretization.
N_SUB_Z = 12
SUB_Z = np.linspace(0.0, H, N_SUB_Z + 1)


# ---------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------
print("=" * 76)
print("CANONICAL OPENEMS MSL MICROSTRIP VALIDATION")
print("=" * 76)
print(f"eps_r                 : {EPS_R}")
print(f"substrate thickness   : {H:.6f} mm")
print(f"microstrip width      : {W:.6f} mm")
print(f"physical line         : {LINE_X0:.1f} -> {LINE_X1:.1f} mm")
print(f"main line             : {PORT1_STOP_X:.1f} -> {PORT2_STOP_X:.1f} mm")
print(f"port 1 feed           : {PORT1_START_X:.1f} -> {PORT1_STOP_X:.1f} mm")
print(f"port 2 feed           : {PORT2_START_X:.1f} -> {PORT2_STOP_X:.1f} mm")
print(f"port reference        : {Z0:.1f} ohm")
print(f"xy mesh target        : {RES_XY:.3f} mm")
print(f"substrate z cells     : {N_SUB_Z}")
print(f"frequency             : {FREQ_START/1e9:.1f} -> {FREQ_STOP/1e9:.1f} GHz")
print(f"timesteps             : {NR_TS}")
print(f"boundary              : {BOUNDARY}")
print(f"simulation path       : {SIM_PATH}")
print("=" * 76)


# ---------------------------------------------------------------------
# FDTD / CSX
# ---------------------------------------------------------------------
FDTD = openEMS(
    NrTS=NR_TS,
    EndCriteria=END_CRITERIA
)

FDTD.SetGaussExcite(F0, FC)
FDTD.SetBoundaryCond(BOUNDARY)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(UNIT)


# ---------------------------------------------------------------------
# Mesh
#
# IMPORTANT:
# AddMSLPort requires the mesh to be defined before the port is added.
# ---------------------------------------------------------------------
mesh.AddLine(
    "x",
    [
        -40.0,
        -30.0,
        -20.0,
        0.0,
        20.0,
        30.0,
        40.0,
    ],
)

mesh.AddLine(
    "y",
    [
        Y_MIN,
        -W / 2.0,
        0.0,
        W / 2.0,
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

mesh.AddLine("z", SUB_Z)

mesh.SmoothMeshLines(
    "all",
    RES_XY,
    1.35,
)

# Reassert critical geometry lines after smoothing.
mesh.AddLine(
    "x",
    [
        -40.0,
        -30.0,
        -20.0,
        0.0,
        20.0,
        30.0,
        40.0,
    ],
)

mesh.AddLine(
    "y",
    [
        Y_MIN,
        -W / 2.0,
        0.0,
        W / 2.0,
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

mesh.AddLine("z", SUB_Z)


# ---------------------------------------------------------------------
# Substrate
# ---------------------------------------------------------------------
substrate = CSX.AddMaterial(
    "substrate",
    epsilon=EPS_R,
)

substrate.AddBox(
    start=[LINE_X0, Y_MIN, 0.0],
    stop=[LINE_X1, Y_MAX, H],
)


# ---------------------------------------------------------------------
# PEC property passed to MSLPort
#
# AddMSLPort creates its own feed metal using this metal property.
# ---------------------------------------------------------------------
pec = CSX.AddMetal("PEC")


# ---------------------------------------------------------------------
# Ground plane
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Main uniform microstrip conductor
#
# MSLPort supplies the two 10-mm feed sections.
# Therefore the explicit SIGNAL conductor covers only the central
# 40-mm section between x=-20 and x=+20 mm.
# ---------------------------------------------------------------------
signal = CSX.AddMetal("SIGNAL")

signal.AddBox(
    start=[PORT1_STOP_X, -W / 2.0, H],
    stop=[PORT2_STOP_X, +W / 2.0, H],
    priority=10,
)

FDTD.AddEdges2Grid(
    dirs="xy",
    properties=signal,
)


# ---------------------------------------------------------------------
# MSL PORT 1
#
# Port starts on signal plane and terminates at ground reference.
# Propagation: +x
# Excitation: z
# ---------------------------------------------------------------------
port1_start = [
    PORT1_START_X,
    -W / 2.0,
    H,
]

port1_stop = [
    PORT1_STOP_X,
    +W / 2.0,
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


# ---------------------------------------------------------------------
# MSL PORT 2
#
# Port direction is -x because stop.x < start.x.
# ---------------------------------------------------------------------
port2_start = [
    PORT2_START_X,
    -W / 2.0,
    H,
]

port2_stop = [
    PORT2_STOP_X,
    +W / 2.0,
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


# ---------------------------------------------------------------------
# Write XML before simulation
# ---------------------------------------------------------------------
CSX_FILE = SIM_PATH / "msl_canonical.xml"
CSX.Write2XML(str(CSX_FILE))

print("\nMSL port geometry:")
print(f"P1 start = {port1_start}")
print(f"P1 stop  = {port1_stop}")
print(f"P2 start = {port2_start}")
print(f"P2 stop  = {port2_stop}")
print(f"XML      = {CSX_FILE}")

print("\nRunning openEMS...")
print()


# ---------------------------------------------------------------------
# Run EM simulation
# ---------------------------------------------------------------------
FDTD.Run(
    str(SIM_PATH),
    cleanup=False,
)


# ---------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------
freq = np.linspace(
    FREQ_START,
    FREQ_STOP,
    N_FREQ,
)

# Explicit 50-ohm S-parameter normalization.
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


# ---------------------------------------------------------------------
# S parameters
# ---------------------------------------------------------------------
S11 = port1.uf_ref / port1.uf_inc
S21 = port2.uf_ref / port1.uf_inc

# Input impedance at port 1.
Zin = port1.uf_tot / port1.if_tot

S11_dB = 20.0 * np.log10(
    np.maximum(np.abs(S11), 1e-15)
)

S21_dB = 20.0 * np.log10(
    np.maximum(np.abs(S21), 1e-15)
)

power_sum = (
    np.abs(S11) ** 2
    + np.abs(S21) ** 2
)


# ---------------------------------------------------------------------
# Optional MSL beta diagnostic
#
# The installed Python MSLPort binding exposes beta on versions where
# this diagnostic is available. Do not require it for validation.
# ---------------------------------------------------------------------
beta1 = getattr(port1, "beta", None)
beta2 = getattr(port2, "beta", None)


# ---------------------------------------------------------------------
# Diagnostic region
# ---------------------------------------------------------------------
central = (
    (freq >= 2e9)
    & (freq <= 10e9)
)

idx5 = int(
    np.argmin(np.abs(freq - 5e9))
)


# ---------------------------------------------------------------------
# Main results
# ---------------------------------------------------------------------
print("\n" + "=" * 76)
print("MSL VALIDATION RESULTS")
print("=" * 76)

print(f"5 GHz actual frequency : {freq[idx5]/1e9:.6f} GHz")
print(f"S11 @ 5 GHz            : {S11_dB[idx5]:.6f} dB")
print(f"S21 @ 5 GHz            : {S21_dB[idx5]:.6f} dB")
print(f"power sum @ 5 GHz      : {power_sum[idx5]:.6f}")

print(
    f"Zin @ 5 GHz            : "
    f"{np.real(Zin[idx5]):.6f} "
    f"{np.imag(Zin[idx5]):+.6f}j ohm"
)

print(
    f"|Zin| @ 5 GHz          : "
    f"{abs(Zin[idx5]):.6f} ohm"
)


# ---------------------------------------------------------------------
# MSL diagnostics — safely handled
# ---------------------------------------------------------------------
print("\nMSL diagnostic values")

# With ref_impedance=50 in CalcPort, Z_ref is the S-parameter
# normalization impedance. It is NOT reported as extracted ZL.
zref1 = getattr(port1, "Z_ref", None)
zref2 = getattr(port2, "Z_ref", None)

if zref1 is not None:
    print(f"Port 1 Z_ref           : {zref1}")
else:
    print("Port 1 Z_ref           : unavailable")

if zref2 is not None:
    print(f"Port 2 Z_ref           : {zref2}")
else:
    print("Port 2 Z_ref           : unavailable")

if beta1 is not None:
    beta1_arr = np.asarray(beta1)
    if beta1_arr.ndim == 0:
        print(f"Port 1 beta            : {beta1_arr}")
    else:
        print(
            "Port 1 beta median     : "
            f"{np.median(beta1_arr[central]):.6e}"
        )
else:
    print("Port 1 beta            : unavailable")

if beta2 is not None:
    beta2_arr = np.asarray(beta2)
    if beta2_arr.ndim == 0:
        print(f"Port 2 beta            : {beta2_arr}")
    else:
        print(
            "Port 2 beta median     : "
            f"{np.median(beta2_arr[central]):.6e}"
        )
else:
    print("Port 2 beta            : unavailable")


# ---------------------------------------------------------------------
# Central-band statistics
# ---------------------------------------------------------------------
s11_max = float(np.max(S11_dB[central]))
s11_min = float(np.min(S11_dB[central]))

s21_max = float(np.max(S21_dB[central]))
s21_min = float(np.min(S21_dB[central]))

power_min = float(np.min(power_sum[central]))
power_max = float(np.max(power_sum[central]))

zin_real_median = float(
    np.median(np.real(Zin[central]))
)

zin_imag_median = float(
    np.median(np.imag(Zin[central]))
)

zin_mag_median = float(
    np.median(np.abs(Zin[central]))
)

print("\nCentral-band 2–10 GHz")
print(f"S11 max                : {s11_max:.6f} dB")
print(f"S11 min                : {s11_min:.6f} dB")
print(f"S21 max                : {s21_max:.6f} dB")
print(f"S21 min                : {s21_min:.6f} dB")
print(f"power min              : {power_min:.6f}")
print(f"power max              : {power_max:.6f}")
print(f"median Re(Zin)         : {zin_real_median:.6f} ohm")
print(f"median Im(Zin)         : {zin_imag_median:.6f} ohm")
print(f"median |Zin|           : {zin_mag_median:.6f} ohm")


# ---------------------------------------------------------------------
# Validation criteria
#
# The important point is that this is a calibration line, not the
# actual filter. We therefore require broadband travelling-wave
# behavior, not merely one good frequency point.
# ---------------------------------------------------------------------
PASS_S11 = s11_max < -10.0

PASS_S21 = s21_min > -1.5

PASS_POWER = (
    power_min > 0.90
    and power_max < 1.10
)

PASS_ZIN = (
    abs(zin_real_median - Z0) < 10.0
    and abs(zin_imag_median) < 10.0
)

OVERALL_PASS = (
    PASS_S11
    and PASS_S21
    and PASS_POWER
    and PASS_ZIN
)


print("\nChecks")
print(
    f"S11 < -10 dB           : "
    f"{'PASS' if PASS_S11 else 'FAIL'}"
)
print(
    f"S21 > -1.5 dB          : "
    f"{'PASS' if PASS_S21 else 'FAIL'}"
)
print(
    f"power conservation     : "
    f"{'PASS' if PASS_POWER else 'FAIL'}"
)
print(
    f"Zin ~ 50+j0 ohm        : "
    f"{'PASS' if PASS_ZIN else 'FAIL'}"
)

print("-" * 76)
print(
    f"OVERALL MSL VALIDATION : "
    f"{'PASS' if OVERALL_PASS else 'FAIL'}"
)
print("=" * 76)


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------
csv_path = RESULTS_PATH / "msl_canonical.csv"

csv_data = {
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

# beta is optional.
if beta1 is not None:
    b1 = np.asarray(beta1)
    if b1.ndim == 0:
        csv_data["Port1_beta_real"] = np.full(
            len(freq), np.real(b1)
        )
        csv_data["Port1_beta_imag"] = np.full(
            len(freq), np.imag(b1)
        )
    elif len(b1) == len(freq):
        csv_data["Port1_beta_real"] = np.real(b1)
        csv_data["Port1_beta_imag"] = np.imag(b1)

if beta2 is not None:
    b2 = np.asarray(beta2)
    if b2.ndim == 0:
        csv_data["Port2_beta_real"] = np.full(
            len(freq), np.real(b2)
        )
        csv_data["Port2_beta_imag"] = np.full(
            len(freq), np.imag(b2)
        )
    elif len(b2) == len(freq):
        csv_data["Port2_beta_real"] = np.real(b2)
        csv_data["Port2_beta_imag"] = np.imag(b2)

pd.DataFrame(csv_data).to_csv(
    csv_path,
    index=False,
)


# ---------------------------------------------------------------------
# Plot 1 — S parameters
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

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

ax.axhline(
    -10,
    linestyle="--",
    linewidth=1,
    label="-10 dB",
)

ax.axhline(
    -1.5,
    linestyle=":",
    linewidth=1,
    label="-1.5 dB",
)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title(
    "Canonical openEMS MSL Port — Uniform 50-ohm Microstrip"
)
ax.grid(True)
ax.legend()
fig.tight_layout()

sparam_plot = (
    PLOTS_PATH
    / "msl_canonical_sparams.png"
)

fig.savefig(
    sparam_plot,
    dpi=180,
)

plt.close(fig)


# ---------------------------------------------------------------------
# Plot 2 — Input impedance
#
# This replaces the previous invalid "ZL" plot.
# For a properly terminated uniform line, Zin should remain close
# to its characteristic impedance.
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

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
    50,
    linestyle="--",
    linewidth=1,
    label="50 ohm",
)

ax.axhline(
    0,
    linestyle=":",
    linewidth=1,
)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Impedance (ohm)")
ax.set_title(
    "Canonical openEMS MSL — Input Impedance"
)
ax.grid(True)
ax.legend()
fig.tight_layout()

zin_plot = (
    PLOTS_PATH
    / "msl_canonical_zin.png"
)

fig.savefig(
    zin_plot,
    dpi=180,
)

plt.close(fig)


# ---------------------------------------------------------------------
# Plot 3 — Power conservation
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    freq / 1e9,
    power_sum,
    label="|S11|² + |S21|²",
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1,
    label="Ideal lossless = 1",
)

ax.axhline(
    0.90,
    linestyle=":",
    linewidth=1,
    label="0.90",
)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Power ratio")
ax.set_title(
    "Canonical openEMS MSL — Power Conservation"
)

ax.set_ylim(
    0,
    max(
        1.2,
        float(np.nanmax(power_sum) * 1.05),
    ),
)

ax.grid(True)
ax.legend()
fig.tight_layout()

power_plot = (
    PLOTS_PATH
    / "msl_canonical_power.png"
)

fig.savefig(
    power_plot,
    dpi=180,
)

plt.close(fig)


# ---------------------------------------------------------------------
# Final file report
# ---------------------------------------------------------------------
print("\nSaved:")
print(f"CSV       : {csv_path}")
print(f"S11/S21   : {sparam_plot}")
print(f"Zin       : {zin_plot}")
print(f"Power     : {power_plot}")
print(f"XML       : {CSX_FILE}")

if OVERALL_PASS:
    print("\n" + "=" * 76)
    print("MSL VALIDATION PASSED.")
    print("The uniform microstrip calibration is suitable as the")
    print("EM baseline for the 7th-order stepped-impedance filter.")
    print("=" * 76)
else:
    print("\n" + "=" * 76)
    print("MSL VALIDATION FAILED.")
    print("Do NOT build the filter yet.")
    print("Inspect this uniform-line model and its XML/geometry.")
    print("=" * 76)
