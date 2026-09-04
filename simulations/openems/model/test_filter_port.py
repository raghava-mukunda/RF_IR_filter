import os
import sys
import shutil
from pathlib import Path

import numpy as np

# ================================================================
# OPENEMS INSTALLATION
# ================================================================

OPENEMS_ROOT = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
    / "openEMS"
)

os.add_dll_directory(str(OPENEMS_ROOT))

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["PATH"] = (
    str(OPENEMS_ROOT)
    + os.pathsep
    + os.environ["PATH"]
)

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ================================================================
# CONSTANTS
# ================================================================

C0 = 299792458.0


# ================================================================
# GEOMETRY
#
# EXACTLY THE SAME STACK AS THE WORKING TWO-PORT TEST
#
# ground     : -0.035 ... 0 mm
# substrate  :  0      ... 1.524 mm
# signal     :  1.524  ... 1.559 mm
# ================================================================

unit = 1e-3

line_length = 80.0
line_width = 4.0

substrate_epsr = 3.38
substrate_thickness = 1.524

substrate_width = 80.0
substrate_length = 80.0

copper_thickness = 0.035

port1_x = -30.0
port2_x = +30.0
port_y = 0.0

feed_R = 50.0


# ================================================================
# SIMULATION
# ================================================================

f0 = 10e9
fc = 10e9

SIM_PATH = Path(
    __file__
).resolve().parents[3] / \
    "simulations" / \
    "openems" / \
    "results" / \
    "filter_port_test"

if SIM_PATH.exists():
    shutil.rmtree(SIM_PATH)

SIM_PATH.mkdir(
    parents=True,
    exist_ok=True
)


FDTD = openEMS(
    NrTS=30000,
    EndCriteria=1e-5,
)

FDTD.SetGaussExcite(
    f0,
    fc,
)

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
# CSX
# ================================================================

CSX = ContinuousStructure()

FDTD.SetCSX(CSX)

GRID = CSX.GetGrid()

GRID.SetDeltaUnit(unit)


# ================================================================
# Z COORDINATES
# ================================================================

z_ground_bottom = -copper_thickness
z_ground_top = 0.0

z_substrate_top = substrate_thickness

z_signal_top = (
    substrate_thickness
    + copper_thickness
)


# ================================================================
# SUBSTRATE
# ================================================================

SUBSTRATE = CSX.AddMaterial(
    "substrate",
    epsilon=substrate_epsr,
)

SUBSTRATE.AddBox(
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
# GROUND
# ================================================================

GROUND = CSX.AddMetal(
    "ground"
)

GROUND.AddBox(
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
# SIGNAL
# ================================================================

SIGNAL = CSX.AddMetal(
    "signal"
)

SIGNAL.AddBox(
    start=[
        -line_length / 2.0,
        -line_width / 2.0,
        substrate_thickness,
    ],
    stop=[
        line_length / 2.0,
        line_width / 2.0,
        z_signal_top,
    ],
)


# ================================================================
# PORT 1
#
# EXACT PORT GEOMETRY FROM THE SUCCESSFUL TWO-PORT TEST
# ================================================================

PORT1 = FDTD.AddLumpedPort(
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

PORT2 = FDTD.AddLumpedPort(
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

lambda_min = (
    C0
    / (f0 + fc)
)

lambda_min_mm = (
    lambda_min / unit
)

coarse_step = lambda_min_mm / 15.0

# X

x_uniform = np.linspace(
    -50.0,
    +50.0,
    int(
        np.ceil(
            100.0 / coarse_step
        )
    ) + 1,
)

x_special = np.array(
    [
        -50.0,
        -40.0,
        -30.0,
        0.0,
        +30.0,
        +40.0,
        +50.0,
    ]
)

x_lines = np.unique(
    np.concatenate(
        [
            x_uniform,
            x_special,
        ]
    )
)


# Y

y_uniform = np.linspace(
    -25.0,
    +25.0,
    int(
        np.ceil(
            50.0 / coarse_step
        )
    ) + 1,
)

y_special = np.array(
    [
        -25.0,
        -line_width / 2.0,
        0.0,
        line_width / 2.0,
        +25.0,
    ]
)

y_lines = np.unique(
    np.concatenate(
        [
            y_uniform,
            y_special,
        ]
    )
)


# Z

z_lower = np.linspace(
    -20.0,
    -copper_thickness,
    30,
)

z_ground = np.linspace(
    -copper_thickness,
    0.0,
    3,
)

z_substrate = np.linspace(
    0.0,
    substrate_thickness,
    7,
)

z_signal = np.linspace(
    substrate_thickness,
    z_signal_top,
    3,
)

z_upper = np.linspace(
    z_signal_top,
    20.0,
    30,
)

z_lines = np.unique(
    np.concatenate(
        [
            z_lower,
            z_ground,
            z_substrate,
            z_signal,
            z_upper,
        ]
    )
)


GRID.SetLines(
    "x",
    x_lines.tolist()
)

GRID.SetLines(
    "y",
    y_lines.tolist()
)

GRID.SetLines(
    "z",
    z_lines.tolist()
)


# ================================================================
# WRITE
# ================================================================

xml_path = (
    SIM_PATH
    / "filter_port_test.xml"
)

CSX.Write2XML(
    str(xml_path)
)

print()
print("=" * 70)
print("PORT DIAGNOSTIC TEST")
print("=" * 70)

print(
    f"Cells: "
    f"{len(x_lines)-1} x "
    f"{len(y_lines)-1} x "
    f"{len(z_lines)-1}"
)

print(
    f"Port 1: "
    f"x={port1_x} mm, "
    f"z={-copper_thickness} -> "
    f"{substrate_thickness + copper_thickness} mm"
)

print()
print("Running openEMS...")
print()


# ================================================================
# RUN
# ================================================================

FDTD.Run(
    str(SIM_PATH),
    cleanup=False,
    verbose=2,
)


# ================================================================
# POSTPROCESS
# ================================================================

freq = np.linspace(
    1e9,
    20e9,
    401,
)

print()
print("Calculating Port 1...")

PORT1.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=50.0,
)

print("Calculating Port 2...")

PORT2.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=50.0,
)


uf_inc = np.asarray(
    PORT1.uf_inc
)

uf_ref = np.asarray(
    PORT1.uf_ref
)

uf_2 = np.asarray(
    PORT2.uf_ref
)


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"max |uf_inc| = "
    f"{np.max(np.abs(uf_inc)):.12e}"
)

print(
    f"max |uf_ref| = "
    f"{np.max(np.abs(uf_ref)):.12e}"
)

print(
    f"max |uf_2|   = "
    f"{np.max(np.abs(uf_2)):.12e}"
)

print("=" * 70)