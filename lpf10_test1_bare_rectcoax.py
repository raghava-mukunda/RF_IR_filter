
import os
from pathlib import Path
import numpy as np

# ============================================================
# TEST 1 — BARE RECTANGULAR COAX
# ============================================================
# Purpose:
# Verify that the rectangular-coax geometry itself can create
# an openEMS FDTD operator and run successfully.
#
# No filter sections.
# No hollow-waveguide absorbers.
# No lossy materials.
# No optimization.
# ============================================================

OPENEMS_ROOT = Path(
    r"C:\Users\Raghava.m\Desktop\openEMS_x64_v0.0.36-93-g7b9cd51_msvc\openEMS"
)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(OPENEMS_ROOT))

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["PATH"] = (
    str(OPENEMS_ROOT)
    + os.pathsep
    + os.environ.get("PATH", "")
)

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ============================================================
# PARAMETERS
# ============================================================

UNIT = 1e-3
Z0 = 50.0

SIM_DIR = Path(
    r"C:\Users\Raghava.m\Desktop\cryogenic_filter_design\_test1_bare_rectcoax"
)

F0 = 10e9
FC = 10e9

NR_TS = 5000
END_CRITERIA = 1e-3

# Rectangular coax
W = 3.0
H = 1.45

CENTER_W = 0.8
CENTER_T = 0.30

WALL_T = 0.20

X0 = -10.0
X1 = 20.0


# ============================================================
# BUILD
# ============================================================

print()
print("==============================================")
print("TEST 1 — BARE RECTANGULAR COAX")
print("==============================================")
print()

CSX = ContinuousStructure()

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(UNIT)

PEC = CSX.AddMetal("PEC")


# ============================================================
# CENTER CONDUCTOR
# ============================================================

PEC.AddBox(
    [X0, -CENTER_W / 2.0, -CENTER_T / 2.0],
    [X1,  CENTER_W / 2.0,  CENTER_T / 2.0],
    priority=30,
)


# ============================================================
# OUTER CONDUCTOR
# ============================================================

# Top
PEC.AddBox(
    [X0, -W / 2.0, H / 2.0 - WALL_T],
    [X1,  W / 2.0, H / 2.0],
    priority=10,
)

# Bottom
PEC.AddBox(
    [X0, -W / 2.0, -H / 2.0],
    [X1,  W / 2.0, -H / 2.0 + WALL_T],
    priority=10,
)

# Left
PEC.AddBox(
    [X0, -W / 2.0, -H / 2.0],
    [X1, -W / 2.0 + WALL_T, H / 2.0],
    priority=10,
)

# Right
PEC.AddBox(
    [X0, W / 2.0 - WALL_T, -H / 2.0],
    [X1, W / 2.0, H / 2.0],
    priority=10,
)


# ============================================================
# MESH
# ============================================================

x_lines = np.arange(X0, X1 + 0.001, 0.5)

y_lines = np.array([
    -W / 2.0,
    -W / 2.0 + WALL_T,
    -CENTER_W / 2.0,
    0.0,
    CENTER_W / 2.0,
    W / 2.0 - WALL_T,
    W / 2.0,
])

z_lines = np.array([
    -H / 2.0,
    -H / 2.0 + WALL_T,
    -CENTER_T / 2.0,
    0.0,
    CENTER_T / 2.0,
    H / 2.0 - WALL_T,
    H / 2.0,
])

mesh.AddLine("x", x_lines.tolist())
mesh.AddLine("y", y_lines.tolist())
mesh.AddLine("z", z_lines.tolist())


# ============================================================
# MESH REPORT
# ============================================================

print("Mesh:")

for axis in ("x", "y", "z"):

    vals = np.asarray(
        mesh.GetLines(axis),
        dtype=float
    ).reshape(-1)

    d = np.diff(vals)

    print(
        f"  {axis}: "
        f"N={len(vals)}, "
        f"cells={len(d)}, "
        f"min={d.min():.9g} mm, "
        f"max={d.max():.9g} mm"
    )

nx = len(np.asarray(mesh.GetLines("x")).reshape(-1)) - 1
ny = len(np.asarray(mesh.GetLines("y")).reshape(-1)) - 1
nz = len(np.asarray(mesh.GetLines("z")).reshape(-1)) - 1

print(f"  TOTAL CELLS = {nx * ny * nz:,}")


# ============================================================
# FDTD
# ============================================================

print()
print("Creating FDTD...")

FDTD = openEMS(
    NrTS=NR_TS,
    EndCriteria=END_CRITERIA,
)

FDTD.SetCSX(CSX)

FDTD.SetBoundaryCond(
    ["PML_8"] * 6
)

FDTD.SetTimeStepMethod(1)

FDTD.SetGaussExcite(
    F0,
    FC,
)


# ============================================================
# LUMPED PORTS
# ============================================================
# Signal is the center conductor.
# Ground is the bottom outer conductor.
# Propagation direction = +x.
#
# The port is a z-directed voltage gap spanning from the
# center conductor to the bottom conductor.

z_sig = CENTER_T / 2.0
z_gnd = -H / 2.0 + WALL_T + 0.05

PORT_X1 = X0 + 2.0
PORT_X2 = X1 - 2.0

print()
print("Port geometry:")
print(f"  Port 1 x = {PORT_X1:.3f} mm")
print(f"  Port 2 x = {PORT_X2:.3f} mm")
print(f"  z ground = {z_gnd:.3f} mm")
print(f"  z signal = {z_sig:.3f} mm")


port1 = FDTD.AddLumpedPort(
    1,
    Z0,
    [PORT_X1, 0.0, z_gnd],
    [PORT_X1, 0.0, z_sig],
    "z",
    excite=1,
    priority=50,
    edges2grid="xz",
)

port2 = FDTD.AddLumpedPort(
    2,
    Z0,
    [PORT_X2, 0.0, z_gnd],
    [PORT_X2, 0.0, z_sig],
    "z",
    excite=0,
    priority=50,
    edges2grid="xz",
)


# ============================================================
# RUN
# ============================================================

SIM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print()
print("Creating FDTD operator...")
print()

FDTD.Run(
    str(SIM_DIR),
    cleanup=True,
    verbose=3,
    numThreads=1,
    disable_dumps=True,
)

print()
print("==============================================")
print("TEST 1 SUCCESS")
print("BARE RECTANGULAR COAX RUN COMPLETED")
print("==============================================")
