import os
import shutil
import time
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

XML_FILE = Path(
    r"C:\Users\Raghava.m\Desktop\cryogenic_filter_design"
    r"\FINAL_OPENEMS_RIGOROUS\final_filter.xml"
)

RUN_DIR = XML_FILE.parent / "rf_diagnostic"

RUN_DIR.mkdir(parents=True, exist_ok=True)

# Clean ONLY the diagnostic directory
for p in RUN_DIR.iterdir():
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


# ============================================================
# openEMS ENVIRONMENT
# ============================================================

OPENEMS_PATH = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
    / "openEMS"
)

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_PATH)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_PATH)
os.environ["PATH"] = (
    str(OPENEMS_PATH)
    + os.pathsep
    + os.environ["PATH"]
)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(OPENEMS_PATH))


# ============================================================
# IMPORTS
# ============================================================

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ============================================================
# RF PARAMETERS
# ============================================================

Z0 = 50.0

TOTAL = 46.764506724
FEED = 8.0

HO_H = 1.45
WALL = 0.20
CENTER_T = 0.50


# ============================================================
# PORT LOCATIONS
# ============================================================

P1X = -FEED + 1.0
P2X = TOTAL - 1.0

ZG = -HO_H / 2 + WALL
ZS = CENTER_T / 2


# ============================================================
# BROADBAND EXCITATION
# ============================================================

F0 = 35e9
FC = 60e9


# ============================================================
# DIAGNOSTIC SIMULATION LENGTH
#
# DO NOT CHANGE TO 120000 YET.
# ============================================================

NR_TS = 10000


# ============================================================
# LOAD EXISTING FINAL XML
# ============================================================

print()
print("=" * 70)
print("LOADING EXISTING FILTER GEOMETRY")
print("=" * 70)

print("XML:", XML_FILE)

CSX = ContinuousStructure()

CSX.ReadFromXML(
    str(XML_FILE)
)


# ============================================================
# CREATE FDTD
# ============================================================

FDTD = openEMS(
    EndCriteria=1e-5,
    NrTS=NR_TS
)

FDTD.SetGaussExcite(
    F0,
    FC
)

FDTD.SetBoundaryCond(
    ["PML_8"] * 6
)

FDTD.SetCSX(
    CSX
)


# ============================================================
# PORT 1
# ============================================================

p1 = FDTD.AddLumpedPort(
    1,
    Z0,
    [P1X, 0, ZG],
    [P1X, 0, ZS],
    "z",
    excite=1,
    priority=50
)


# ============================================================
# PORT 2
# ============================================================

p2 = FDTD.AddLumpedPort(
    2,
    Z0,
    [P2X, 0, ZG],
    [P2X, 0, ZS],
    "z",
    excite=0,
    priority=50
)


# ============================================================
# PRINT PORT CONFIGURATION
# ============================================================

print()
print("=" * 70)
print("PORT CONFIGURATION")
print("=" * 70)

print(
    f"P1:"
    f" x={P1X:.9f} mm"
    f"  z={ZG:.9f} -> {ZS:.9f} mm"
)

print(
    f"P2:"
    f" x={P2X:.9f} mm"
    f"  z={ZG:.9f} -> {ZS:.9f} mm"
)

print("=" * 70)


# ============================================================
# WRITE DIAGNOSTIC XML
# ============================================================

SIM_XML = RUN_DIR / "rf_diagnostic.xml"

CSX.Write2XML(
    str(SIM_XML)
)

print()
print("Simulation XML:")
print(SIM_XML)


# ============================================================
# RUN FDTD
# ============================================================

print()
print("=" * 70)
print("STARTING DIAGNOSTIC FDTD")
print("=" * 70)

print("Frequency range : 1-70 GHz")
print("Excitation       : 35 GHz / 60 GHz Gaussian")
print("Timesteps        :", NR_TS)
print()
print("This is ONLY a port-output diagnostic.")
print("No S-parameter calculation will be attempted.")
print()

t0 = time.time()

FDTD.Run(
    str(RUN_DIR),
    cleanup=False,
    verbose=2
)

elapsed = time.time() - t0

print()
print("=" * 70)
print("FDTD FINISHED")
print("=" * 70)

print(
    f"Runtime: {elapsed / 60.0:.2f} minutes"
)


# ============================================================
# INSPECT GENERATED FILES
# ============================================================

print()
print("=" * 70)
print("GENERATED FILES")
print("=" * 70)

for p in sorted(RUN_DIR.iterdir()):

    if p.is_file():

        print(
            f"{p.name:35s}"
            f" {p.stat().st_size:12d} bytes"
        )


# ============================================================
# INSPECT PORT FILES
# ============================================================

print()
print("=" * 70)
print("PORT FILE INSPECTION")
print("=" * 70)


for port_name in [
    "port_ut_1",
    "port_ut_2"
]:

    port_file = RUN_DIR / port_name

    print()
    print("-" * 70)
    print("FILE:", port_file)
    print("-" * 70)

    if not port_file.exists():

        print("ERROR: FILE DOES NOT EXIST")

        continue

    print(
        "Size:",
        port_file.stat().st_size,
        "bytes"
    )

    print()
    print("First 10 raw lines:")
    print()

    with open(
        port_file,
        "r",
        errors="replace"
    ) as f:

        for i in range(10):

            line = f.readline()

            if not line:
                break

            print(
                f"{i + 1:02d}:",
                repr(line[:1000])
            )


# ============================================================
# CHECK FOR THE SPECIFIC CORRUPTION WE SAW PREVIOUSLY
# ============================================================

print()
print("=" * 70)
print("PORT FORMAT CHECK")
print("=" * 70)


BAD_PATTERN = (
    "1.31476236814e-12"
)


for port_name in [
    "port_ut_1",
    "port_ut_2"
]:

    port_file = RUN_DIR / port_name

    if not port_file.exists():
        continue

    text = port_file.read_text(
        errors="replace"
    )

    print()
    print(port_name)

    print(
        "Contains expected timestamp:",
        BAD_PATTERN in text
    )

    print(
        "Contains malformed concatenation:",
        "-0.0002143833808081.31476236814e-12"
        in text
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("DIAGNOSTIC RUN COMPLETE")
print("=" * 70)

print()
print("Diagnostic directory:")
print(RUN_DIR)

print()
print(
    "If the port files look correct, "
    "we will run the full 120000-step simulation."
)

print()