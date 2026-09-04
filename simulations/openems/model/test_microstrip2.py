import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# DLL SETUP
# ================================================================

OPENEMS_ROOT = os.environ.get("OPENEMS_INSTALL_PATH")

if OPENEMS_ROOT:
    os.add_dll_directory(OPENEMS_ROOT)


# ================================================================
# OPENEMS IMPORTS
# ================================================================

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SIM_PATH = (
    PROJECT_ROOT
    / "simulations"
    / "openems"
    / "results"
    / "two_port_line"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "em"
)

PLOT_PATH = (
    RESULTS_PATH
    / "two_port_microstrip_sparams_updated.png"
)

CSV_PATH = (
    RESULTS_PATH
    / "two_port_microstrip_sparams_updated.csv"
)

SIM_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# PHYSICAL CONSTANTS
# ================================================================

C0 = 299792458.0


# ================================================================
# SIMULATION PARAMETERS
# ================================================================

# Broadband test excitation.
f0 = 10e9
fc = 10e9

f_start = 1e9
f_stop = 20e9


# ================================================================
# GEOMETRY UNIT
# ================================================================

# All geometry below is in mm.
unit = 1e-3


# ================================================================
# MICROSTRIP GEOMETRY
# ================================================================

# Transmission-line length.
line_length = 80.0

# Microstrip width.
line_width = 4.0

# Substrate.
substrate_epsr = 3.38
substrate_thickness = 1.524

substrate_width = 80.0
substrate_length = 80.0

# Finite copper thickness.
copper_thickness = 0.035


# ================================================================
# PORT LOCATIONS
# ================================================================

# Ports are placed at the two ends of the microstrip.
port1_x = -30.0
port2_x = +30.0

port_y = 0.0

feed_R = 50.0


# ================================================================
# SIMULATION BOX
# ================================================================

SimBox = np.array(
    [
        100.0,
        50.0,
        40.0,
    ],
    dtype=float,
)


# ================================================================
# FDTD
# ================================================================

FDTD = openEMS(
    NrTS=100000,
    EndCriteria=1e-5,
)


# ================================================================
# GAUSSIAN EXCITATION
# ================================================================

FDTD.SetGaussExcite(
    f0,
    fc,
)


# ================================================================
# BOUNDARY CONDITIONS
# ================================================================

# MUR is retained initially because the successful patch
# validation used the same boundary strategy.
FDTD.SetBoundaryCond(
    [
        "MUR",
        "MUR",
        "MUR",
        "MUR",
        "MUR",
        "MUR",
    ]
)


# ================================================================
# CSXCAD
# ================================================================

CSX = ContinuousStructure()

FDTD.SetCSX(
    CSX
)


# ================================================================
# MATERIAL: SUBSTRATE
# ================================================================

substrate = CSX.AddMaterial(
    "substrate",
    epsilon=substrate_epsr,
)

substrate.AddBox(
    start=[
        -substrate_width / 2.0,
        -substrate_length / 2.0,
        0.0,
    ],

    stop=[
        substrate_width / 2.0,
        substrate_length / 2.0,
        substrate_thickness,
    ],
)


# ================================================================
# METAL: GROUND
# ================================================================

ground = CSX.AddMetal(
    "ground"
)

ground.AddBox(
    start=[
        -substrate_width / 2.0,
        -substrate_length / 2.0,
        -copper_thickness,
    ],

    stop=[
        substrate_width / 2.0,
        substrate_length / 2.0,
        0.0,
    ],
)


# ================================================================
# METAL: MICROSTRIP
# ================================================================

signal = CSX.AddMetal(
    "signal"
)

signal.AddBox(
    start=[
        -line_length / 2.0,
        -line_width / 2.0,
        substrate_thickness,
    ],

    stop=[
        line_length / 2.0,
        line_width / 2.0,
        substrate_thickness + copper_thickness,
    ],
)


# ================================================================
# PORT 1
# ================================================================

# The lumped port spans vertically from the bottom of the ground
# conductor to the top of the signal conductor.

port1 = FDTD.AddLumpedPort(
    port_nr=1,

    R=feed_R,

    start=[
        port1_x,
        port_y,
        -copper_thickness,
    ],

    stop=[
        port1_x,
        port_y,
        substrate_thickness + copper_thickness,
    ],

    p_dir="z",

    excite=1,
)


# ================================================================
# PORT 2
# ================================================================

# Passive receiving port.
port2 = FDTD.AddLumpedPort(
    port_nr=2,

    R=feed_R,

    start=[
        port2_x,
        port_y,
        -copper_thickness,
    ],

    stop=[
        port2_x,
        port_y,
        substrate_thickness + copper_thickness,
    ],

    p_dir="z",

    excite=0,
)


# ================================================================
# MESH
# ================================================================

mesh = CSX.GetGrid()

mesh.SetDeltaUnit(
    unit
)


# ================================================================
# WAVELENGTH-BASED RESOLUTION
# ================================================================

lambda_min = C0 / (f0 + fc)

lambda_min_mm = lambda_min / unit

# First 2-port sanity model:
# approximately lambda/15 in air.
coarse_step = lambda_min_mm / 15.0

# Fine resolution through dielectric.
substrate_step = substrate_thickness / 6.0


# ================================================================
# REQUIRED GEOMETRY BOUNDARIES
# ================================================================

x_edges = np.array(
    [
        -SimBox[0] / 2.0,

        -substrate_width / 2.0,

        port1_x,

        -line_length / 2.0,

        0.0,

        line_length / 2.0,

        port2_x,

        substrate_width / 2.0,

        SimBox[0] / 2.0,
    ],
    dtype=float,
)


y_edges = np.array(
    [
        -SimBox[1] / 2.0,

        -substrate_length / 2.0,

        -line_width / 2.0,

        0.0,

        line_width / 2.0,

        substrate_length / 2.0,

        SimBox[1] / 2.0,
    ],
    dtype=float,
)


z_edges = np.array(
    [
        -SimBox[2] / 3.0,

        -copper_thickness,

        0.0,

        substrate_thickness,

        substrate_thickness + copper_thickness,

        SimBox[2] * 2.0 / 3.0,
    ],
    dtype=float,
)


# ================================================================
# INITIAL GRID
# ================================================================

mesh.AddLine(
    "x",
    x_edges,
)

mesh.AddLine(
    "y",
    y_edges,
)

mesh.AddLine(
    "z",
    z_edges,
)


# ================================================================
# UNIFORM X MESH
# ================================================================

x_dense = np.linspace(
    -SimBox[0] / 2.0,
    SimBox[0] / 2.0,
    int(
        np.ceil(
            SimBox[0] / coarse_step
        )
    ) + 1,
)

x_dense = np.unique(
    np.sort(
        np.concatenate(
            [
                x_dense,
                x_edges,
            ]
        )
    )
)


# ================================================================
# UNIFORM Y MESH
# ================================================================

y_dense = np.linspace(
    -SimBox[1] / 2.0,
    SimBox[1] / 2.0,
    int(
        np.ceil(
            SimBox[1] / coarse_step
        )
    ) + 1,
)

y_dense = np.unique(
    np.sort(
        np.concatenate(
            [
                y_dense,
                y_edges,
            ]
        )
    )
)


# ================================================================
# Z MESH
# ================================================================

z_lower = np.linspace(
    -SimBox[2] / 3.0,
    -copper_thickness,
    max(
        2,
        int(
            np.ceil(
                (
                    SimBox[2] / 3.0
                    - copper_thickness
                )
                / coarse_step
            )
        ) + 1,
    ),
)


z_ground = np.linspace(
    -copper_thickness,
    0.0,
    3,
)


z_substrate = np.linspace(
    0.0,
    substrate_thickness,
    int(
        np.ceil(
            substrate_thickness
            / substrate_step
        )
    ) + 1,
)


z_signal = np.linspace(
    substrate_thickness,
    substrate_thickness + copper_thickness,
    3,
)


z_upper = np.linspace(
    substrate_thickness + copper_thickness,
    SimBox[2] * 2.0 / 3.0,
    max(
        2,
        int(
            np.ceil(
                (
                    SimBox[2] * 2.0 / 3.0
                    - substrate_thickness
                    - copper_thickness
                )
                / coarse_step
            )
        ) + 1,
    ),
)


z_dense = np.unique(
    np.sort(
        np.concatenate(
            [
                z_lower,
                z_ground,
                z_substrate,
                z_signal,
                z_upper,
                z_edges,
            ]
        )
    )
)


# ================================================================
# SET FINAL MESH
# ================================================================

mesh.SetLines(
    "x",
    x_dense,
)

mesh.SetLines(
    "y",
    y_dense,
)

mesh.SetLines(
    "z",
    z_dense,
)


# ================================================================
# REPORT
# ================================================================

print()
print("=" * 72)
print("openEMS TWO-PORT MICROSTRIP TRANSMISSION-LINE TEST")
print("=" * 72)

print()

print(
    f"Simulation directory : {SIM_PATH}"
)

print()

print(
    f"Line length          : "
    f"{line_length:.3f} mm"
)

print(
    f"Line width           : "
    f"{line_width:.3f} mm"
)

print(
    f"Substrate            : "
    f"{substrate_width:.1f} x "
    f"{substrate_length:.1f} x "
    f"{substrate_thickness:.3f} mm"
)

print(
    f"Relative epsilon     : "
    f"{substrate_epsr:.3f}"
)

print(
    f"Copper thickness     : "
    f"{copper_thickness:.3f} mm"
)

print()

print(
    f"Port 1 x             : "
    f"{port1_x:.3f} mm"
)

print(
    f"Port 2 x             : "
    f"{port2_x:.3f} mm"
)

print(
    f"Port impedance       : "
    f"{feed_R:.1f} Ohm"
)

print()

print(
    f"Excitation center    : "
    f"{f0 / 1e9:.2f} GHz"
)

print(
    f"Excitation bandwidth : "
    f"{fc / 1e9:.2f} GHz"
)

print()

print(
    f"Mesh X points        : "
    f"{len(x_dense)}"
)

print(
    f"Mesh Y points        : "
    f"{len(y_dense)}"
)

print(
    f"Mesh Z points        : "
    f"{len(z_dense)}"
)

print()

print(
    f"Minimum wavelength   : "
    f"{lambda_min_mm:.3f} mm"
)

print(
    f"Coarse mesh step     : "
    f"{coarse_step:.3f} mm"
)

print(
    f"Substrate mesh step  : "
    f"{substrate_step:.3f} mm"
)


# ================================================================
# WRITE XML
# ================================================================

xml_path = SIM_PATH / "two_port_microstrip.xml"

if xml_path.exists():
    xml_path.unlink()

CSX.Write2XML(
    str(xml_path)
)

print()
print(
    f"XML written          : {xml_path}"
)

print()
print(
    "Starting openEMS..."
)

print()


# ================================================================
# RUN FDTD
# ================================================================

FDTD.Run(
    str(SIM_PATH),
    cleanup=True,
)


# ================================================================
# FREQUENCY VECTOR
# ================================================================

freq = np.linspace(
    f_start,
    f_stop,
    401,
)


# ================================================================
# PORT POSTPROCESSING
# ================================================================

print()
print(
    "Calculating Port 1..."
)

port1.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=50.0,
)


print(
    "Calculating Port 2..."
)

port2.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=50.0,
)


# ================================================================
# EXTRACT WAVES
# ================================================================

uf_inc_1 = np.asarray(
    port1.uf_inc
)

uf_ref_1 = np.asarray(
    port1.uf_ref
)

uf_ref_2 = np.asarray(
    port2.uf_ref
)


# ================================================================
# VALIDITY CHECK
# ================================================================

if not np.all(
    np.isfinite(uf_inc_1)
):
    raise RuntimeError(
        "Port 1 incident wave contains "
        "NaN or Inf."
    )


if not np.all(
    np.isfinite(uf_ref_1)
):
    raise RuntimeError(
        "Port 1 reflected wave contains "
        "NaN or Inf."
    )


if not np.all(
    np.isfinite(uf_ref_2)
):
    raise RuntimeError(
        "Port 2 response contains "
        "NaN or Inf."
    )


# ================================================================
# INCIDENT WAVE DIAGNOSTIC
# ================================================================

print()
print("=" * 72)
print("PORT WAVE DIAGNOSTIC")
print("=" * 72)

max_uf_inc_1 = np.max(
    np.abs(uf_inc_1)
)

min_uf_inc_1 = np.min(
    np.abs(uf_inc_1)
)

max_uf_ref_1 = np.max(
    np.abs(uf_ref_1)
)

max_uf_ref_2 = np.max(
    np.abs(uf_ref_2)
)

print(
    f"max |uf_inc_1| : "
    f"{max_uf_inc_1:.6e}"
)

print(
    f"min |uf_inc_1| : "
    f"{min_uf_inc_1:.6e}"
)

print(
    f"max |uf_ref_1| : "
    f"{max_uf_ref_1:.6e}"
)

print(
    f"max |uf_ref_2| : "
    f"{max_uf_ref_2:.6e}"
)

if max_uf_inc_1 == 0.0:
    raise RuntimeError(
        "Port 1 incident wave is numerically "
        "exactly zero."
    )

print(
    "Port 1 incident wave exists."
)

print("=" * 72)


# ================================================================
# S-PARAMETERS
# ================================================================

s11 = (
    uf_ref_1
    / uf_inc_1
)

s21 = (
    uf_ref_2
    / uf_inc_1
)


# ================================================================
# CONVERT TO dB
# ================================================================

s11_db = (
    20.0
    * np.log10(
        np.maximum(
            np.abs(s11),
            1e-15,
        )
    )
)


s21_db = (
    20.0
    * np.log10(
        np.maximum(
            np.abs(s21),
            1e-15,
        )
    )
)


# ================================================================
# FINAL FINITE CHECK
# ================================================================

if not np.all(
    np.isfinite(s11_db)
):
    raise RuntimeError(
        "S11 contains NaN or Inf."
    )


if not np.all(
    np.isfinite(s21_db)
):
    raise RuntimeError(
        "S21 contains NaN or Inf."
    )


# ================================================================
# BASIC METRICS
# ================================================================

max_s11 = np.max(
    s11_db
)

min_s11 = np.min(
    s11_db
)

max_s21 = np.max(
    s21_db
)

min_s21 = np.min(
    s21_db
)


# ================================================================
# SAVE CSV
# ================================================================

np.savetxt(
    CSV_PATH,

    np.column_stack(
        [
            freq / 1e9,
            s11_db,
            s21_db,
        ]
    ),

    delimiter=",",

    header=(
        "frequency_GHz,"
        "S11_dB,"
        "S21_dB"
    ),

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
    linewidth=1.5,
    label="S11",
)

plt.plot(
    freq / 1e9,
    s21_db,
    linewidth=1.5,
    label="S21",
)

plt.axhline(
    -10.0,
    linestyle="--",
    linewidth=1.0,
)

plt.xlabel(
    "Frequency (GHz)"
)

plt.ylabel(
    "Magnitude (dB)"
)

plt.title(
    "openEMS 2-Port Microstrip "
    "Transmission Line"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.xlim(
    f_start / 1e9,
    f_stop / 1e9,
)

plt.ylim(
    -60,
    5,
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=200,
)

plt.close()


# ================================================================
# FINAL REPORT
# ================================================================

print()
print("=" * 72)
print("TWO-PORT RESULT")
print("=" * 72)

print()

print(
    f"Maximum S11 : "
    f"{max_s11:.3f} dB"
)

print(
    f"Minimum S11 : "
    f"{min_s11:.3f} dB"
)

print()

print(
    f"Maximum S21 : "
    f"{max_s21:.3f} dB"
)

print(
    f"Minimum S21 : "
    f"{min_s21:.3f} dB"
)

print()

print(
    f"CSV         : "
    f"{CSV_PATH}"
)

print(
    f"Plot        : "
    f"{PLOT_PATH}"
)

print()

print("=" * 72)
print("TWO-PORT SIMULATION COMPLETE")
print("=" * 72)