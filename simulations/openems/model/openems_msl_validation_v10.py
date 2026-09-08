# -*- coding: utf-8 -*-
"""
V10 — Canonical openEMS 2-port microstrip validation

Purpose
-------
Validate the openEMS lumped-port topology against the official
Simple_Patch_Antenna construction before using the port in the
7th-order stepped-impedance filter.

Key corrections relative to V9
------------------------------
1. Exact canonical 1-D vertical lumped-port geometry:
       start = [x_port, 0, 0]
       stop  = [x_port, 0, H]
2. Port priority = 5, metal priority = 10, matching the official
   openEMS Simple_Patch_Antenna tutorial.
3. No finite x/y port volume.
4. Explicit z-mesh with 10 cells through the 1.524 mm substrate.
   This is important because an openEMS discussion documents that
   insufficient z discretization can make a lumped port stop working.
5. Explicit mesh lines at both port x positions and y=0.
6. Zero-thickness PEC signal and ground sheets, as in the official
   tutorial.
7. Air exists above and below the ground plane; the ground is finite
   and separated from the outer boundaries.
8. Two identical 50-ohm lumped ports on a uniform 50-ohm microstrip.
9. PML_8 is used with sufficient space for the PML layers.
10. Full S11/S21, impedance, passivity and CSV/PNG output.

Expected physical result
------------------------
This is a uniform, lossless 50-ohm transmission line. In the useful
central band (roughly 2–10 GHz), a successful port implementation
should give:
    - S11 well below 0 dB, preferably < -10 dB
    - S21 close to 0 dB
    - |S11|^2 + |S21|^2 close to 1
    - Zin close to 50 + j0 ohm
    - no unexplained deep transmission nulls

If this V10 still reports:
    "Series\Parallel Lumped RLC load: Active cells : 0"
or produces non-passive/unphysical S-parameters, STOP. Do not make
another arbitrary port geometry variant. At that point the next step
is to inspect the generated CSXCAD XML/grid rather than guessing.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Windows openEMS DLL setup
# ---------------------------------------------------------------------
OPENEMS_PACKAGE = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
    / "openEMS"
)

if not OPENEMS_PACKAGE.exists():
    raise FileNotFoundError(
        f"openEMS package not found at:\n{OPENEMS_PACKAGE}\n"
        "Update OPENEMS_PACKAGE if your installation is elsewhere."
    )

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_PACKAGE)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_PACKAGE)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(OPENEMS_PACKAGE))

os.environ["PATH"] = str(OPENEMS_PACKAGE) + os.pathsep + os.environ["PATH"]

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_PATH = PROJECT_ROOT / "simulations" / "openems" / "results" / "v10_lumped_validation"
RESULTS_PATH = PROJECT_ROOT / "results" / "em" / "v10_lumped_validation"
PLOTS_PATH = PROJECT_ROOT / "results" / "plots"

SIM_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Remove stale simulation data so old UI files cannot contaminate results.
for p in SIM_PATH.iterdir():
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


# ---------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------
UNIT = 1e-3                 # all geometry coordinates are mm
Z0 = 50.0

EPS_R = 3.38
H = 1.524                   # substrate thickness [mm]
W = 3.5577                  # 50-ohm width for eps_r=3.38, h=1.524 mm

LINE_LENGTH = 60.0         # mm
PORT1_X = -20.0             # mm
PORT2_X = +20.0             # mm

# 10 mm of uniform line exists on either side of each port.
# This is intentionally symmetric.
X_MIN = -30.0
X_MAX = +30.0

# Transverse substrate/ground width.
Y_MIN = -15.0
Y_MAX = +15.0

# Air space between structure and PML boundaries.
# PML_8 needs actual mesh cells around the structure.
AIR_BELOW = 5.0             # mm below ground
AIR_ABOVE = 20.0            # mm above substrate

Z_MIN = -AIR_BELOW
Z_MAX = H + AIR_ABOVE

# ---------------------------------------------------------------------
# FDTD settings
# ---------------------------------------------------------------------
F0 = 6.5e9                  # center frequency
FC = 5.5e9                  # broad Gaussian bandwidth

FREQ_START = 1.0e9
FREQ_STOP = 12.0e9
N_FREQ = 1101

NR_TS = 200000
END_CRITERIA = 1e-5         # -50 dB residual energy

BOUNDARY = [
    "PML_8", "PML_8",
    "PML_8", "PML_8",
    "PML_8", "PML_8",
]

# Maximum normal mesh spacing in xy.
MESH_RES_XY = 0.50          # mm

# Explicit substrate discretization:
# 10 cells across 1.524 mm -> 0.1524 mm dz.
SUBSTRATE_CELLS_Z = 10
SUB_Z_LINES = np.linspace(0.0, H, SUBSTRATE_CELLS_Z + 1)

# Coarser air mesh is acceptable, but the substrate/ports stay explicitly
# resolved by the fixed z-lines above.
MESH_RES_Z_AIR = 0.50       # mm


# ---------------------------------------------------------------------
# Basic checks
# ---------------------------------------------------------------------
assert X_MIN < PORT1_X < PORT2_X < X_MAX
assert Y_MIN < -W / 2 < W / 2 < Y_MAX
assert H > 0
assert SUBSTRATE_CELLS_Z >= 8

print("=" * 72)
print("V10 — CANONICAL OPENEMS 2-PORT MICROSTRIP VALIDATION")
print("=" * 72)
print(f"eps_r                 : {EPS_R}")
print(f"substrate thickness   : {H:.6f} mm")
print(f"microstrip width      : {W:.6f} mm")
print(f"line length            : {LINE_LENGTH:.3f} mm")
print(f"port positions        : {PORT1_X:.3f} mm, {PORT2_X:.3f} mm")
print(f"port reference        : {Z0:.1f} ohm")
print(f"xy mesh                : <= {MESH_RES_XY:.3f} mm")
print(f"substrate z cells      : {SUBSTRATE_CELLS_Z}")
print(f"z substrate step       : {H / SUBSTRATE_CELLS_Z:.6f} mm")
print(f"frequency              : {FREQ_START/1e9:.3f}–{FREQ_STOP/1e9:.3f} GHz")
print(f"timesteps              : {NR_TS}")
print(f"boundary               : {BOUNDARY}")
print(f"simulation path        : {SIM_PATH}")
print("=" * 72)


# ---------------------------------------------------------------------
# FDTD + excitation
# ---------------------------------------------------------------------
FDTD = openEMS(NrTS=NR_TS, EndCriteria=END_CRITERIA)
FDTD.SetGaussExcite(F0, FC)
FDTD.SetBoundaryCond(BOUNDARY)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(UNIT)


# ---------------------------------------------------------------------
# Simulation-box mesh
#
# Start with the outer dimensions so the PML has physical space.
# ---------------------------------------------------------------------
mesh.AddLine("x", [X_MIN - 5.0, X_MAX + 5.0])
mesh.AddLine("y", [Y_MIN - 5.0, Y_MAX + 5.0])
mesh.AddLine("z", [Z_MIN, Z_MAX])

# Important fixed lines:
# - port x coordinates
# - strip edges
# - strip center
# - substrate top/bottom
# - all substrate z divisions
mesh.AddLine("x", [
    X_MIN, PORT1_X, PORT2_X, X_MAX,
    -W / 2, 0.0, +W / 2,
])

mesh.AddLine("y", [
    Y_MIN, -W / 2, 0.0, +W / 2, Y_MAX,
])

mesh.AddLine("z", [Z_MIN, 0.0, H, Z_MAX])
mesh.AddLine("z", SUB_Z_LINES)


# ---------------------------------------------------------------------
# Substrate
# ---------------------------------------------------------------------
substrate = CSX.AddMaterial("substrate", epsilon=EPS_R)
substrate.AddBox(
    priority=0,
    start=[X_MIN, Y_MIN, 0.0],
    stop=[X_MAX, Y_MAX, H],
)


# ---------------------------------------------------------------------
# Ground plane — zero-thickness PEC sheet at z=0
#
# This follows the canonical Simple_Patch_Antenna topology.
# ---------------------------------------------------------------------
gnd = CSX.AddMetal("gnd")
gnd.AddBox(
    start=[X_MIN, Y_MIN, 0.0],
    stop=[X_MAX, Y_MAX, 0.0],
    priority=10,
)
FDTD.AddEdges2Grid(dirs="xy", properties=gnd)


# ---------------------------------------------------------------------
# Signal strip — zero-thickness PEC sheet at z=H
# ---------------------------------------------------------------------
signal = CSX.AddMetal("signal")
signal.AddBox(
    start=[X_MIN, -W / 2, H],
    stop=[X_MAX, +W / 2, H],
    priority=10,
)
FDTD.AddEdges2Grid(dirs="xy", properties=signal)


# Explicitly seed all critical signal/port edges.
mesh.AddLine("x", [
    X_MIN, PORT1_X, PORT2_X, X_MAX,
    -W / 2, +W / 2,
])
mesh.AddLine("y", [-W / 2, 0.0, +W / 2])


# ---------------------------------------------------------------------
# PORTS — THIS IS THE CRITICAL V10 CHANGE
#
# Exact canonical vertical 1-D port topology:
#
#       signal z=H
#          |
#          |  lumped port
#          |
#       ground z=0
#
# No finite x/y port box.
# Port priority 5, metal priority 10 — matching the official
# Simple_Patch_Antenna.py tutorial.
# ---------------------------------------------------------------------
PORT1_START = [PORT1_X, 0.0, 0.0]
PORT1_STOP = [PORT1_X, 0.0, H]

PORT2_START = [PORT2_X, 0.0, 0.0]
PORT2_STOP = [PORT2_X, 0.0, H]

# Port 1: excited.
port1 = FDTD.AddLumpedPort(
    1,
    Z0,
    PORT1_START,
    PORT1_STOP,
    "z",
    1.0,
    priority=5,
    edges2grid="xy",
)

# Port 2: passive/load port.
port2 = FDTD.AddLumpedPort(
    2,
    Z0,
    PORT2_START,
    PORT2_STOP,
    "z",
    0.0,
    priority=5,
    edges2grid="xy",
)


# ---------------------------------------------------------------------
# Mesh smoothing
#
# The fixed substrate z-lines are inserted BEFORE smoothing and must
# remain grid lines.  This gives the lumped port multiple active cells
# in the excitation direction instead of a single unresolved z interval.
# ---------------------------------------------------------------------
mesh.SmoothMeshLines("all", MESH_RES_XY, 1.4)


# Re-add the critical z lines after smoothing so they are guaranteed
# to exist exactly at the substrate boundaries and port endpoints.
mesh.AddLine("z", [Z_MIN, 0.0, H, Z_MAX])
mesh.AddLine("z", SUB_Z_LINES)

# Re-add critical x/y lines after smoothing as well.
mesh.AddLine("x", [
    X_MIN, PORT1_X, PORT2_X, X_MAX,
    -W / 2, 0.0, +W / 2,
])
mesh.AddLine("y", [
    Y_MIN, -W / 2, 0.0, +W / 2, Y_MAX,
])


# ---------------------------------------------------------------------
# Write XML before running.
# This makes V10 inspectable in AppCSXCAD if anything is wrong.
# ---------------------------------------------------------------------
CSX_FILE = SIM_PATH / "v10_lumped_validation.xml"
CSX.Write2XML(str(CSX_FILE))

print("\nCSX XML written:")
print(CSX_FILE)

print("\nRunning openEMS...")
print("Do NOT judge the result until the solver reaches the end criterion")
print("or the full timestep limit.")
print()


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
FDTD.Run(str(SIM_PATH), cleanup=False)


# ---------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------
freq = np.linspace(FREQ_START, FREQ_STOP, N_FREQ)

# R=50 in AddLumpedPort already defines the reference impedance.
# We also explicitly pass 50 ohm to CalcPort so the normalization is
# unambiguous.
port1.CalcPort(str(SIM_PATH), freq, ref_impedance=Z0)
port2.CalcPort(str(SIM_PATH), freq, ref_impedance=Z0)

S11 = port1.uf_ref / port1.uf_inc
S21 = port2.uf_ref / port1.uf_inc

Zin = port1.uf_tot / port1.if_tot

S11_dB = 20.0 * np.log10(np.maximum(np.abs(S11), 1e-15))
S21_dB = 20.0 * np.log10(np.maximum(np.abs(S21), 1e-15))

power_sum = np.abs(S11) ** 2 + np.abs(S21) ** 2


# ---------------------------------------------------------------------
# Central-band validation
#
# Ignore the very edges of the broadband Gaussian spectrum for the
# pass/fail check.  The actual design target is in the microwave band,
# and the central 2–10 GHz region has strong excitation.
# ---------------------------------------------------------------------
central = (freq >= 2e9) & (freq <= 10e9)

s11_max_central = float(np.max(S11_dB[central]))
s21_max_central = float(np.max(S21_dB[central]))
s21_min_central = float(np.min(S21_dB[central]))

power_max_central = float(np.max(power_sum[central]))
power_min_central = float(np.min(power_sum[central]))

zin_real = np.real(Zin[central])
zin_imag = np.imag(Zin[central])

# Robust statistics rather than a single frequency point.
zin_real_median = float(np.median(zin_real))
zin_imag_median = float(np.median(zin_imag))

# A line is considered healthy if the central band has:
#   S11 below -10 dB everywhere,
#   S21 never below -1.5 dB,
#   power conservation approximately holds,
#   median impedance is close to 50+j0 ohm.
PASS_S11 = s11_max_central < -10.0
PASS_S21 = s21_min_central > -1.5
PASS_POWER = (
    power_min_central > 0.90
    and power_max_central < 1.10
)
PASS_ZIN = (
    abs(zin_real_median - Z0) < 10.0
    and abs(zin_imag_median) < 10.0
)

OVERALL_PASS = PASS_S11 and PASS_S21 and PASS_POWER and PASS_ZIN


# ---------------------------------------------------------------------
# Point diagnostic near 5 GHz
# ---------------------------------------------------------------------
idx5 = int(np.argmin(np.abs(freq - 5e9)))

print("\n" + "=" * 72)
print("V10 RESULTS")
print("=" * 72)
print(f"5 GHz actual frequency : {freq[idx5]/1e9:.6f} GHz")
print(f"S11 @ 5 GHz            : {S11_dB[idx5]:.6f} dB")
print(f"S21 @ 5 GHz            : {S21_dB[idx5]:.6f} dB")
print(f"power sum @ 5 GHz      : {power_sum[idx5]:.6f}")
print(
    f"Zin @ 5 GHz            : "
    f"{np.real(Zin[idx5]):.6f} "
    f"{np.imag(Zin[idx5]):+.6f}j ohm"
)
print(f"|Zin| @ 5 GHz          : {abs(Zin[idx5]):.6f} ohm")

print("\nCentral-band validation: 2–10 GHz")
print(f"S11 max                : {s11_max_central:.6f} dB")
print(f"S21 min                : {s21_min_central:.6f} dB")
print(f"S21 max                : {s21_max_central:.6f} dB")
print(f"power min              : {power_min_central:.6f}")
print(f"power max              : {power_max_central:.6f}")
print(f"median Re(Zin)         : {zin_real_median:.6f} ohm")
print(f"median Im(Zin)         : {zin_imag_median:.6f} ohm")

print("\nChecks")
print(f"S11 < -10 dB           : {'PASS' if PASS_S11 else 'FAIL'}")
print(f"S21 > -1.5 dB          : {'PASS' if PASS_S21 else 'FAIL'}")
print(f"power conservation     : {'PASS' if PASS_POWER else 'FAIL'}")
print(f"Zin ~ 50+j0 ohm        : {'PASS' if PASS_ZIN else 'FAIL'}")
print("-" * 72)
print(f"OVERALL V10            : {'PASS' if OVERALL_PASS else 'FAIL'}")
print("=" * 72)


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------
csv_path = RESULTS_PATH / "v10_lumped_validation.csv"

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

df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------
# Plot 1 — S parameters
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(freq / 1e9, S11_dB, label="S11")
ax.plot(freq / 1e9, S21_dB, label="S21")
ax.axhline(-10.0, linestyle="--", linewidth=1.0, label="-10 dB")
ax.axhline(-1.5, linestyle=":", linewidth=1.0, label="-1.5 dB")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("V10 — Uniform 50-ohm Microstrip: S-Parameters")
ax.grid(True)
ax.legend()
fig.tight_layout()
sparam_plot = PLOTS_PATH / "v10_lumped_validation_sparams.png"
fig.savefig(sparam_plot, dpi=180)
plt.close(fig)


# ---------------------------------------------------------------------
# Plot 2 — impedance
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(freq / 1e9, np.real(Zin), label="Re{Zin}")
ax.plot(freq / 1e9, np.imag(Zin), label="Im{Zin}")
ax.axhline(50.0, linestyle="--", linewidth=1.0, label="50 ohm")
ax.axhline(0.0, linestyle=":", linewidth=1.0)
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Impedance (ohm)")
ax.set_title("V10 — Input Impedance")
ax.grid(True)
ax.legend()
fig.tight_layout()
zin_plot = PLOTS_PATH / "v10_lumped_validation_zin.png"
fig.savefig(zin_plot, dpi=180)
plt.close(fig)


# ---------------------------------------------------------------------
# Plot 3 — power conservation
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(freq / 1e9, power_sum, label="|S11|² + |S21|²")
ax.axhline(1.0, linestyle="--", linewidth=1.0, label="Ideal lossless = 1")
ax.axhline(0.90, linestyle=":", linewidth=1.0, label="0.90")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Power ratio")
ax.set_title("V10 — Power Conservation")
ax.set_ylim(0, max(1.2, float(np.nanmax(power_sum) * 1.05)))
ax.grid(True)
ax.legend()
fig.tight_layout()
power_plot = PLOTS_PATH / "v10_lumped_validation_power.png"
fig.savefig(power_plot, dpi=180)
plt.close(fig)


print("\nSaved:")
print(f"CSV       : {csv_path}")
print(f"S11/S21   : {sparam_plot}")
print(f"Zin       : {zin_plot}")
print(f"Power     : {power_plot}")
print(f"XML       : {CSX_FILE}")

if not OVERALL_PASS:
    print("\nV10 FAILED validation.")
    print(
        "Do not use this port geometry in the filter yet. "
        "Inspect the V10 XML/grid and solver log."
    )
else:
    print("\nV10 PASSED the uniform-line port validation.")
    print(
        "This port topology is now suitable as the baseline for "
        "the stepped-impedance filter model."
    )
