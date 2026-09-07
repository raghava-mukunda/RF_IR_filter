# ================================================================
# openEMS MSL PORT VALIDATION V4
#
# Purpose:
#   Validate a uniform microstrip transmission line and the
#   openEMS AddMSLPort implementation before building the
#   7th-order cryogenic low-pass filter.
#
# Expected physical result:
#   S11  -> substantially below 0 dB
#   S21  -> approximately 0 dB for an ideal PEC/conducting-sheet line
#
# ================================================================


# ================================================================
# IMPORTS / DLL SETUP
# ================================================================

from pathlib import Path
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ================================================================
# OPENEMS DLL PATH
# ================================================================

OPENEMS_ROOT = os.environ.get("OPENEMS_INSTALL_PATH")

if not OPENEMS_ROOT:
    OPENEMS_ROOT = str(
        Path.home()
        / "Desktop"
        / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
        / "openEMS"
    )

OPENEMS_ROOT = str(Path(OPENEMS_ROOT).resolve())

os.environ["OPENEMS_INSTALL_PATH"] = OPENEMS_ROOT
os.environ["CSXCAD_INSTALL_PATH"] = OPENEMS_ROOT

os.environ["PATH"] = (
    OPENEMS_ROOT
    + os.pathsep
    + os.environ.get("PATH", "")
)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(OPENEMS_ROOT)


# ================================================================
# PARAMETERS
# ================================================================

UNIT = 1e-3


# ------------------------------------------------
# Frequency
# ------------------------------------------------

F_START = 1e9
F_STOP = 12e9

F0 = 6.5e9
FC = 6.5e9


# ------------------------------------------------
# Electrical target
# ------------------------------------------------

Z0 = 50.0


# ------------------------------------------------
# Substrate
# ------------------------------------------------

EPS_R = 3.38
SUBSTRATE_H = 1.524       # mm


# ------------------------------------------------
# Microstrip
# ------------------------------------------------

MSL_WIDTH = 3.5577        # mm


# ------------------------------------------------
# Physical line
# ------------------------------------------------

LINE_LENGTH = 60.0        # mm


# ------------------------------------------------
# Port/feed sections
# ------------------------------------------------

FEED_LENGTH = 10.0         # mm


# ------------------------------------------------
# Lateral/vertical air
#
# Wider than V1 to reduce interaction with boundaries.
# ------------------------------------------------

AIR_SIDE = 20.0           # mm
AIR_HEIGHT = 20.0         # mm


# ------------------------------------------------
# Mesh
# ------------------------------------------------

RESOLUTION = 0.5         # mm


# ------------------------------------------------
# FDTD
# ------------------------------------------------

NR_TS = 200000
END_CRITERIA = 1e-5

BOUNDARY_CONDITIONS = [
    "PML_8",
    "PML_8",
    "PML_8",
    "PML_8",
    "PML_8",
    "PML_8",
]


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SIM_DIR = (
    PROJECT_ROOT
    / "simulations"
    / "openems"
    / "results"
    / "msl_validation_v4"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "em"
)

XML_FILE = SIM_DIR / "msl_validation_v4.xml"
CSV_FILE = RESULTS_DIR / "msl_validation_v4.csv"
PLOT_FILE = RESULTS_DIR / "msl_validation_v4.png"


# ================================================================
# CLEAN PREVIOUS SIMULATION
# ================================================================

if SIM_DIR.exists():
    shutil.rmtree(SIM_DIR)

SIM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# GEOMETRY
# ================================================================

x0 = -LINE_LENGTH / 2.0
x1 = +LINE_LENGTH / 2.0

y0 = -AIR_SIDE
y1 = +AIR_SIDE

z0 = 0.0
z1 = SUBSTRATE_H
z2 = SUBSTRATE_H + AIR_HEIGHT


# ================================================================
# CREATE FDTD
# ================================================================

FDTD = openEMS(
    NrTS=NR_TS,
    EndCriteria=END_CRITERIA,
    OverSampling=15,
)

FDTD.SetGaussExcite(
    F0,
    FC,
)

FDTD.SetBoundaryCond(
    BOUNDARY_CONDITIONS,
)


# ================================================================
# CREATE CSXCAD STRUCTURE
# ================================================================

CSX = ContinuousStructure()

FDTD.SetCSX(CSX)


# ================================================================
# MATERIALS
# ================================================================

GROUND = CSX.AddMetal(
    "GROUND"
)

SIGNAL = CSX.AddConductingSheet(
    "SIGNAL",
    conductivity=5.8e7,
    thickness=35e-6,
)

SUBSTRATE = CSX.AddMaterial(
    "SUBSTRATE",
    epsilon=EPS_R,
)


# ================================================================
# SUBSTRATE
# ================================================================

SUBSTRATE.AddBox(
    [
        x0,
        y0,
        z0,
    ],
    [
        x1,
        y1,
        z1,
    ],
    priority=1,
)


# ================================================================
# GROUND PLANE
# ================================================================

GROUND.AddBox(
    [
        x0,
        y0,
        z0,
    ],
    [
        x1,
        y1,
        z0,
    ],
    priority=10,
)


# ================================================================
# MAIN MICROSTRIP
#
# Signal metal is placed at the top of the substrate.
#
# The port feed sections are created by AddMSLPort.
# Therefore the independent signal line begins after port 1
# and ends before port 2.
# ================================================================

SIGNAL.AddBox(
    [
        x0 + FEED_LENGTH,
        -MSL_WIDTH / 2.0,
        z1,
    ],
    [
        x1 - FEED_LENGTH,
        +MSL_WIDTH / 2.0,
        z1,
    ],
    priority=20,
)


# ================================================================
# MESH
# ================================================================

mesh = CSX.GetGrid()

mesh.SetDeltaUnit(
    UNIT
)


# ------------------------------------------------
# X
# ------------------------------------------------

mesh_x = np.arange(
    x0,
    x1 + RESOLUTION,
    RESOLUTION,
)

mesh_x = np.unique(
    np.concatenate(
        [
            mesh_x,
            [
                x0,
                x0 + FEED_LENGTH,
                x1 - FEED_LENGTH,
                x1,
            ],
        ]
    )
)

mesh_x.sort()


# ------------------------------------------------
# Y
# ------------------------------------------------

mesh_y = np.arange(
    y0,
    y1 + RESOLUTION,
    RESOLUTION,
)

mesh_y = np.unique(
    np.concatenate(
        [
            mesh_y,
            [
                -MSL_WIDTH / 2.0,
                +MSL_WIDTH / 2.0,
            ],
        ]
    )
)

mesh_y.sort()


# ------------------------------------------------
# Z
# ------------------------------------------------

mesh_z = np.arange(
    z0,
    z2 + RESOLUTION,
    RESOLUTION,
)

mesh_z = np.unique(
    np.concatenate(
        [
            mesh_z,
            [
                z0,
                z1,
                z2,
            ],
        ]
    )
)

mesh_z.sort()


mesh.AddLine(
    "x",
    mesh_x,
)

mesh.AddLine(
    "y",
    mesh_y,
)

mesh.AddLine(
    "z",
    mesh_z,
)


# ================================================================
# PORT 1
#
# AddMSLPort convention:
#
#   start = signal plane
#   stop  = ground plane
#
# Port 1 points in +x.
# ================================================================

port1_start = [
    x0,
    -MSL_WIDTH / 2.0,
    z1,
]

port1_stop = [
    x0 + FEED_LENGTH,
    +MSL_WIDTH / 2.0,
    z0,
]

port1 = FDTD.AddMSLPort(
    1,
    SIGNAL,
    port1_start,
    port1_stop,
    "x",
    "z",
    excite=1,
    FeedShift=10 * RESOLUTION,
    Feed_R=Z0,
    MeasPlaneShift=FEED_LENGTH / 2.0,
    priority=50,
)


# ================================================================
# PORT 2
#
# Mirror image of port 1.
# ================================================================

port2_start = [
    x1,
    -MSL_WIDTH / 2.0,
    z1,
]

port2_stop = [
    x1 - FEED_LENGTH,
    +MSL_WIDTH / 2.0,
    z0,
]

port2 = FDTD.AddMSLPort(
    2,
    SIGNAL,
    port2_start,
    port2_stop,
    "x",
    "z",
    excite=0,
    Feed_R=Z0,
    MeasPlaneShift=FEED_LENGTH / 2.0,
    priority=50,
)


# ================================================================
# EDGE ALIGNMENT
# ================================================================

FDTD.AddEdges2Grid(
    dirs="xy",
    properties=GROUND,
)

FDTD.AddEdges2Grid(
    dirs="xy",
    properties=SIGNAL,
)


# ================================================================
# WRITE XML
# ================================================================

CSX.Write2XML(
    str(XML_FILE)
)


# ================================================================
# CONFIGURATION REPORT
# ================================================================

print("=" * 72)
print("openEMS MSL PORT VALIDATION V4")
print("=" * 72)
print("Validation configuration:")
print("  - 50 ohm external S-parameter normalization")
print("  - PML_8 on all six boundaries")
print("  - 200000 maximum timesteps")
print("  - 10 mm MSL feed sections")
print("  - FeedShift = 10 * mesh resolution")
print("=" * 72)

print(
    f"Line length          : "
    f"{LINE_LENGTH:.3f} mm"
)

print(
    f"Line width           : "
    f"{MSL_WIDTH:.4f} mm"
)

print(
    f"Feed length          : "
    f"{FEED_LENGTH:.3f} mm"
)

print(
    f"Substrate            : "
    f"{EPS_R:.3f}"
)

print(
    f"Substrate height     : "
    f"{SUBSTRATE_H:.3f} mm"
)

print(
    f"Air side             : "
    f"{AIR_SIDE:.1f} mm"
)

print(
    f"Air height           : "
    f"{AIR_HEIGHT:.1f} mm"
)

print(
    f"Mesh X/Y/Z           : "
    f"{len(mesh_x)} / "
    f"{len(mesh_y)} / "
    f"{len(mesh_z)}"
)

print(
    f"Timesteps            : "
    f"{NR_TS}"
)

print(
    f"Frequency range      : "
    f"{F_START/1e9:.1f}–"
    f"{F_STOP/1e9:.1f} GHz"
)

print("=" * 72)

print(
    f"XML written          : "
    f"{XML_FILE}"
)

print(
    "Starting openEMS..."
)

print()


# ================================================================
# RUN
# ================================================================

FDTD.Run(
    str(SIM_DIR),
    cleanup=False,
)


# ================================================================
# FREQUENCY VECTOR
# ================================================================

freq = np.linspace(
    F_START,
    F_STOP,
    1101,
)


# ================================================================
# CALCULATE PORT DATA
# ================================================================

port1.CalcPort(
    str(SIM_DIR),
    freq,
    ref_impedance=Z0,
)

port2.CalcPort(
    str(SIM_DIR),
    freq,
    ref_impedance=Z0,
)


# ================================================================
# SELECT 5 GHz POINT
# ================================================================

idx = np.argmin(
    np.abs(freq - 5e9)
)


# ================================================================
# EXTRACTED MSL PARAMETERS
# ================================================================

Zref1 = np.asarray(port1.Z_ref)
Zref2 = np.asarray(port2.Z_ref)
beta1 = np.asarray(port1.beta)
beta2 = np.asarray(port2.beta)

# With an explicit ref_impedance, openEMS may return Z_ref/beta
# as scalars rather than frequency-dependent arrays.
def value_at_index_or_scalar(value, index):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr[index]

Zref1_at_idx = value_at_index_or_scalar(Zref1, idx)
Zref2_at_idx = value_at_index_or_scalar(Zref2, idx)
beta1_at_idx = value_at_index_or_scalar(beta1, idx)
beta2_at_idx = value_at_index_or_scalar(beta2, idx)


print()
print("=" * 72)
print("EXTRACTED MSL TRANSMISSION-LINE PARAMETERS")
print("=" * 72)
print(f"External S-parameter reference impedance : {Z0:.1f} ohm")
print("S11/S21 wave decomposition uses explicit reference impedance.")


print(
    f"Frequency checked : "
    f"{freq[idx]/1e9:.3f} GHz"
)

print(
    f"Port 1 Z_ref      : "
    f"{Zref1_at_idx}"
)

print(
    f"Port 2 Z_ref      : "
    f"{Zref2_at_idx}"
)

print(
    f"Port 1 |Z_ref|    : "
    f"{abs(Zref1_at_idx):.3f} ohm"
)

print(
    f"Port 2 |Z_ref|    : "
    f"{abs(Zref2_at_idx):.3f} ohm"
)

print(
    f"Port 1 beta       : "
    f"{beta1_at_idx}"
)

print(
    f"Port 2 beta       : "
    f"{beta2_at_idx}"
)

print("=" * 72)


# ================================================================
# PORT WAVES
# ================================================================

uf_inc = np.asarray(
    port1.uf_inc
)

uf_ref = np.asarray(
    port1.uf_ref
)

uf_trans = np.asarray(
    port2.uf_ref
)


# ================================================================
# VALID DATA MASK
# ================================================================

valid = (
    np.isfinite(uf_inc)
    & np.isfinite(uf_ref)
    & np.isfinite(uf_trans)
    & (np.abs(uf_inc) > 1e-30)
)


# ================================================================
# S-PARAMETERS
# ================================================================

s11 = np.full(
    freq.shape,
    np.nan + 0j,
    dtype=complex,
)

s21 = np.full(
    freq.shape,
    np.nan + 0j,
    dtype=complex,
)

s11[valid] = (
    uf_ref[valid]
    / uf_inc[valid]
)

s21[valid] = (
    uf_trans[valid]
    / uf_inc[valid]
)


# ================================================================
# S-PARAMETERS IN dB
# ================================================================

s11_db = np.full(
    freq.shape,
    np.nan,
)

s21_db = np.full(
    freq.shape,
    np.nan,
)

s11_db[valid] = (
    20.0
    * np.log10(
        np.maximum(
            np.abs(s11[valid]),
            1e-15,
        )
    )
)

s21_db[valid] = (
    20.0
    * np.log10(
        np.maximum(
            np.abs(s21[valid]),
            1e-15,
        )
    )
)


# ================================================================
# INPUT IMPEDANCE
# ================================================================

Zin = (
    np.asarray(port1.uf_tot)
    / np.asarray(port1.if_tot)
)


# ================================================================
# DIAGNOSTIC REPORT
# ================================================================

print()
print("=" * 72)
print("5 GHz DIAGNOSTICS")
print("=" * 72)

print(
    f"S11 @ 5 GHz       : "
    f"{s11_db[idx]:.3f} dB"
)

print(
    f"S21 @ 5 GHz       : "
    f"{s21_db[idx]:.3f} dB"
)

print(
    f"Zin @ 5 GHz       : "
    f"{Zin[idx]}"
)

print("=" * 72)


# ================================================================
# GLOBAL RESULTS
# ================================================================

finite_ok = (
    np.all(np.isfinite(s11_db))
    and np.all(np.isfinite(s21_db))
)

s11_max = np.nanmax(
    s11_db
)

s11_min = np.nanmin(
    s11_db
)

s21_max = np.nanmax(
    s21_db
)

s21_min = np.nanmin(
    s21_db
)


# ================================================================
# SAVE CSV
# ================================================================

np.savetxt(
    CSV_FILE,
    np.column_stack(
        [
            freq / 1e9,
            s11_db,
            s21_db,
        ]
    ),
    delimiter=",",
    header="frequency_GHz,S11_dB,S21_dB",
    comments="",
)


# ================================================================
# PLOT
# ================================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    freq / 1e9,
    s11_db,
    label="S11",
)

plt.plot(
    freq / 1e9,
    s21_db,
    label="S21",
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1,
)

plt.grid(True)

plt.xlabel(
    "Frequency (GHz)"
)

plt.ylabel(
    "Magnitude (dB)"
)

plt.title(
    "openEMS MSL Port Validation V4"
)

plt.xlim(
    F_START / 1e9,
    F_STOP / 1e9,
)

plt.ylim(
    -40,
    2,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=180,
)

plt.close()


# ================================================================
# FINAL SUMMARY
# ================================================================

print()
print("=" * 72)
print("MSL PORT VALIDATION V4")
print("=" * 72)

print(
    f"max |uf_inc|       : "
    f"{np.max(np.abs(uf_inc)):.6e}"
)

print(
    f"max |uf_ref|       : "
    f"{np.max(np.abs(uf_ref)):.6e}"
)

print(
    f"max |uf_trans|     : "
    f"{np.max(np.abs(uf_trans)):.6e}"
)

print(
    f"S11 max [dB]       : "
    f"{s11_max:.3f}"
)

print(
    f"S11 min [dB]       : "
    f"{s11_min:.3f}"
)

print(
    f"S21 max [dB]       : "
    f"{s21_max:.3f}"
)

print(
    f"S21 min [dB]       : "
    f"{s21_min:.3f}"
)

print(
    f"Finite values      : "
    f"{'PASS' if finite_ok else 'FAIL'}"
)

print("=" * 72)

print(
    f"CSV saved           : "
    f"{CSV_FILE}"
)

print(
    f"Plot saved          : "
    f"{PLOT_FILE}"
)

print("=" * 72)