import os
from pathlib import Path
import numpy as np

OPENEMS_ROOT = Path(r"C:\Users\Raghava.m\Desktop\openEMS_x64_v0.0.36-93-g7b9cd51_msvc\openEMS")

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(OPENEMS_ROOT))

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["PATH"] = str(OPENEMS_ROOT) + os.pathsep + os.environ.get("PATH", "")

from CSXCAD import ContinuousStructure
from openEMS import openEMS

UNIT = 1e-3
SIM_DIR = Path(r"C:\Users\Raghava.m\Desktop\cryogenic_filter_design\_openems_operator_test")

print()
print("==============================================")
print("CLEAN openEMS OPERATOR TEST")
print("==============================================")

CSX = ContinuousStructure()
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(UNIT)

PEC = CSX.AddMetal("PEC")
PEC.AddBox(
    [-5.0, -2.0, -2.0],
    [25.0, 2.0, -1.5],
    priority=10,
)

x_lines = np.arange(-5.0, 25.001, 1.0)
y_lines = np.arange(-3.0, 3.001, 1.0)
z_lines = np.arange(-3.0, 3.001, 1.0)

mesh.AddLine("x", x_lines.tolist())
mesh.AddLine("y", y_lines.tolist())
mesh.AddLine("z", z_lines.tolist())

print(f"X lines = {len(x_lines)}")
print(f"Y lines = {len(y_lines)}")
print(f"Z lines = {len(z_lines)}")
print(f"Cells   = {(len(x_lines)-1)*(len(y_lines)-1)*(len(z_lines)-1):,}")

for axis in ("x", "y", "z"):
    vals = np.asarray(mesh.GetLines(axis), dtype=float).reshape(-1)
    d = np.diff(vals)
    print(f"{axis}: min={d.min():.12g} mm, max={d.max():.12g} mm")

print()
print("Creating FDTD...")

FDTD = openEMS(NrTS=1000, EndCriteria=1e-3)
FDTD.SetCSX(CSX)
FDTD.SetBoundaryCond(["PML_4"] * 6)
FDTD.SetTimeStepMethod(1)
FDTD.SetGaussExcite(1e9, 1e9)

SIM_DIR.mkdir(parents=True, exist_ok=True)

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
print("SUCCESS")
print("openEMS FDTD operator was created and ran.")
print("==============================================")
