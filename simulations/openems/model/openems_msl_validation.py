# ================================================================
# openEMS CANONICAL MSL PORT VALIDATION
# Uniform 50-ohm Microstrip Transmission Line
# ================================================================

from pathlib import Path
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ================================================================
# DLL SETUP
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

# Frequency range used for validation.
# Kept below the ~16 GHz Nyquist limit of the 0.5 mm mesh.
F_START = 1e9
F_STOP = 12e9

# Gaussian excitation
F0 = 6.5e9
FC = 6.5e9

Z0 = 50.0

# Substrate
EPS_R = 3.38
SUBSTRATE_H = 1.524       # mm

# Approximate 50-ohm microstrip width
MSL_WIDTH = 3.5577        # mm

# Physical transmission-line length
LINE_LENGTH = 60.0        # mm

# MSL port/feed section
FEED_LENGTH = 5.0         # mm

# Mesh
RESOLUTION = 0.5          # mm

# Air region above and beside substrate
AIR_HEIGHT = 15.0         # mm
AIR_SIDE = 15.0            # mm

# FDTD
NR_TS = 120000
END_CRITERIA = 1e-5

BOUNDARY_CONDITIONS = [
    "MUR",
    "MUR",
    "MUR",
    "MUR",
    "MUR",
    "MUR",
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
    / "msl_validation"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "em"
)

XML_FILE = SIM_DIR / "msl_validation.xml"
CSV_FILE = RESULTS_DIR / "msl_validation.csv"
PLOT_FILE = RESULTS_DIR / "msl_validation.png"


# ================================================================
# CLEAN OUTPUT DIRECTORIES
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
# GEOMETRY COORDINATES
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
    BOUNDARY_CONDITIONS
)


# ================================================================
# CREATE CSXCAD STRUCTURE
# ================================================================

CSX = ContinuousStructure()

FDTD.SetCSX(CSX)


# ================================================================
# MATERIALS
# ================================================================

PEC = CSX.AddMetal("PEC")

SUBSTRATE = CSX.AddMaterial(
    "SUBSTRATE",
    epsilon=EPS_R,
)


# ================================================================
# SUBSTRATE
# ================================================================

SUBSTRATE.AddBox(
    [x0, y0, z0],
    [x1, y1, z1],
    priority=1,
)


# ================================================================
# GROUND PLANE
# ================================================================

PEC.AddBox(
    [x0, y0, z0],
    [x1, y1, z0],
    priority=10,
)


# ================================================================
# MAIN MICROSTRIP
#
# The actual uniform transmission line starts after the
# input MSL port section and ends before the output port section.
# ================================================================

PEC.AddBox(
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
    priority=10,
)


# ================================================================
# MESH
# ================================================================

mesh = CSX.GetGrid()

mesh.SetDeltaUnit(UNIT)


# ------------------------------------------------
# X MESH
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
# Y MESH
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
# Z MESH
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


mesh.AddLine("x", mesh_x)
mesh.AddLine("y", mesh_y)
mesh.AddLine("z", mesh_z)


# ================================================================
# PORT 1
#
# MSLPort convention:
#   start = signal plane
#   stop  = ground plane
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
    PEC,
    port1_start,
    port1_stop,
    "x",
    "z",
    excite=1,
    FeedShift=RESOLUTION * 2,
    Feed_R=Z0,
    MeasPlaneShift=FEED_LENGTH / 2.0,
    priority=50,
)


# ================================================================
# PORT 2
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
    PEC,
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
    properties=PEC,
)


# ================================================================
# WRITE XML
# ================================================================

CSX.Write2XML(
    str(XML_FILE)
)


# ================================================================
# PRINT CONFIGURATION
# ================================================================

print("=" * 72)
print("openEMS CANONICAL MSL PORT VALIDATION")
print("=" * 72)

print(f"Line length          : {LINE_LENGTH:.3f} mm")
print(f"Line width           : {MSL_WIDTH:.4f} mm")
print(f"Feed length          : {FEED_LENGTH:.3f} mm")
print(f"Substrate            : {EPS_R:.3f}")
print(f"Substrate height     : {SUBSTRATE_H:.3f} mm")
print(
    f"Mesh X/Y/Z           : "
    f"{len(mesh_x)} / {len(mesh_y)} / {len(mesh_z)}"
)
print(f"Timesteps            : {NR_TS}")
print(f"Frequency range      : {F_START/1e9:.1f}–{F_STOP/1e9:.1f} GHz")
print("=" * 72)

print(f"XML written          : {XML_FILE}")
print("Starting openEMS...")
print()


# ================================================================
# RUN FDTD
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
# CALCULATE PORT PARAMETERS
#
# IMPORTANT:
# No explicit ref_impedance is supplied here.
#
# This lets the MSL port perform its own transmission-line
# parameter extraction, which we want to inspect.
# ================================================================

port1.CalcPort(
    str(SIM_DIR),
    freq,
)

port2.CalcPort(
    str(SIM_DIR),
    freq,
)


# ================================================================
# EXTRACTED TRANSMISSION-LINE PARAMETERS
# ================================================================

idx = np.argmin(
    np.abs(freq - 5e9)
)

print()
print("=" * 72)
print("EXTRACTED MSL TRANSMISSION-LINE PARAMETERS")
print("=" * 72)

print(
    f"Frequency checked : "
    f"{freq[idx] / 1e9:.3f} GHz"
)

# Current openEMS Python MSLPort stores the extracted
# characteristic impedance as Z_ref.
ZL1 = np.asarray(port1.Z_ref)
ZL2 = np.asarray(port2.Z_ref)

beta1 = np.asarray(port1.beta)
beta2 = np.asarray(port2.beta)

print(
    f"Port 1 Z_ref      : "
    f"{ZL1[idx]}"
)

print(
    f"Port 2 Z_ref      : "
    f"{ZL2[idx]}"
)

print(
    f"Port 1 beta       : "
    f"{beta1[idx]}"
)

print(
    f"Port 2 beta       : "
    f"{beta2[idx]}"
)

print()
print("Characteristic impedance magnitude:")

print(
    f"Port 1 |Z_ref|    : "
    f"{abs(ZL1[idx]):.3f} ohm"
)

print(
    f"Port 2 |Z_ref|    : "
    f"{abs(ZL2[idx]):.3f} ohm"
)

print("=" * 72)


# ================================================================
# INPUT IMPEDANCE
# ================================================================

Zin = (
    port1.uf_tot
    / port1.if_tot
)

print()
print("=" * 72)
print("INPUT IMPEDANCE")
print("=" * 72)

for f_test in [1e9, 5e9, 10e9]:
    i = np.argmin(
        np.abs(freq - f_test)
    )

    print(
        f"Zin @ {freq[i]/1e9:.1f} GHz : "
        f"{Zin[i]}"
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
# VALID FREQUENCY POINTS
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
# RESULTS AT 5 GHz
# ================================================================

print()
print("=" * 72)
print("MSL RESULTS AT 5 GHz")
print("=" * 72)

print(
    f"S11 @ 5 GHz       : "
    f"{s11_db[idx]:.3f} dB"
)

print(
    f"S21 @ 5 GHz       : "
    f"{s21_db[idx]:.3f} dB"
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
# VALID FREQUENCY POINTS
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
# S-PARAMETER MAGNITUDE IN dB
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
# S-PARAMETER CHECK AT 5 GHz
# ================================================================

print()
print("=" * 72)
print("MSL RESULTS AT 5 GHz")
print("=" * 72)

print(
    f"S11 @ 5 GHz       : "
    f"{s11_db[idx]:.3f} dB"
)

print(
    f"S21 @ 5 GHz       : "
    f"{s21_db[idx]:.3f} dB"
)

print("=" * 72)


# ================================================================
# SUMMARY VALUES
# ================================================================

finite_ok = (
    np.all(np.isfinite(s11_db))
    and np.all(np.isfinite(s21_db))
)

s11_max = np.nanmax(s11_db)
s11_min = np.nanmin(s11_db)

s21_max = np.nanmax(s21_db)
s21_min = np.nanmin(s21_db)


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
    "openEMS Canonical MSL Port Validation"
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
# FINAL REPORT
# ================================================================

print()
print("=" * 72)
print("MSL PORT VALIDATION")
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
    f"CSV saved           : {CSV_FILE}"
)

print(
    f"Plot saved          : {PLOT_FILE}"
)