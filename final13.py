# ================================================================
# EXACT RUN OF EXISTING COMPACT 10 GHz LPF
# ------------------------------------------------
# NO OPTIMIZATION
# NO GEOMETRY CHANGES
# NO MESH CHANGES
#
# Reads the existing CSXCAD XML exactly as supplied.
# Runs ONE FDTD simulation.
# Frequency range: 0.1 GHz -> 100 GHz
# ================================================================

import os
import sys
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# 1. OPENEMS PATH
# ================================================================

OPENEMS_ROOT = Path(
    r"C:\Users\Raghava.m\Desktop\openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
) / "openEMS"

os.add_dll_directory(str(OPENEMS_ROOT))
sys.path.insert(0, str(OPENEMS_ROOT / "Python"))

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ================================================================
# 2. FILES
# ================================================================

HERE = Path(__file__).resolve().parent

# CHANGE THIS ONLY IF YOUR XML HAS A DIFFERENT NAME
XML_FILE = HERE / "C:/Users/Raghava.m/Desktop/cryogenic_filter_design/LPF10_OPT/BEST/geometry.xml"

RUN_DIR = HERE / "EXACT_100GHz_RUN"

if not XML_FILE.exists():
    raise FileNotFoundError(
        f"\nXML file not found:\n{XML_FILE}\n"
        f"\nPut your existing XML in the same folder and name it:\n"
        f"best_design.xml\n"
    )

RUN_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 3. FREQUENCY RANGE
# ================================================================

FREQ_MIN = 0.1e9
FREQ_MAX = 100e9
N_FREQ = 2001

FREQS = np.linspace(
    FREQ_MIN,
    FREQ_MAX,
    N_FREQ
)


# ================================================================
# 4. FILTER PARAMETERS
# ------------------------------------------------
# These are NOT used to construct geometry.
# They are only documented here for the exact design.
# ================================================================

Z0 = 50.0

TOTAL_LENGTH = 65.3
OUTER_WIDTH = 5.0
OUTER_HEIGHT = 4.6

CENTER_WIDTH = 1.2

H_LOW = 0.30
H_HIGH = 0.80

PORT1_X = 7.0
PORT2_X = 58.3

WALL = 0.20


# ================================================================
# 5. LOAD EXACT XML
# ================================================================

print("=" * 72)
print("EXACT 100 GHz SIMULATION")
print("=" * 72)

print(f"XML       : {XML_FILE}")
print(f"Frequency : {FREQ_MIN/1e9:.3f} -> {FREQ_MAX/1e9:.1f} GHz")
print(f"Points    : {N_FREQ}")
print()
print("Loading EXISTING geometry...")
print("NO geometry generation.")
print("NO optimization.")
print("NO mesh modification.")
print()

CSX = ContinuousStructure()

ok = CSX.ReadFromXML(str(XML_FILE))

print(f"ReadFromXML result: {ok}")


# ================================================================
# 6. CREATE FDTD
# ================================================================

# Important:
# This is ONE simulation.
#
# 100 GHz requires a broadband Gaussian excitation.
# f0 = 50 GHz
# fc = 50 GHz
#
# This is the standard openEMS wideband choice:
# f0 = fmax/2
# fc = fmax/2

F0 = 50e9
FC = 50e9

# Large enough to allow the resonant structure to decay,
# but NOT absurdly large.
#
# If the simulation naturally reaches EndCriteria, it stops early.
# 160000 is a hard ceiling, NOT a target.

NR_TS = 100000

# -40 dB energy decay
END_CRITERIA = 1e-4

print()
print("Creating FDTD...")
print(f"F0           = {F0/1e9:.1f} GHz")
print(f"Bandwidth    = {FC/1e9:.1f} GHz")
print(f"Max timesteps= {NR_TS}")
print(f"EndCriteria  = {END_CRITERIA}")


FDTD = openEMS(
    NrTS=NR_TS,
    EndCriteria=END_CRITERIA
)


# ================================================================
# 7. CONNECT EXISTING CSX TO FDTD
# ================================================================

FDTD.SetCSX(CSX)


# ================================================================
# 8. BOUNDARIES
# ================================================================

# EXACT same intended boundary condition as previous filter runs.
#
# IMPORTANT:
# Your XML has only 7 Y grid lines and 9 Z grid lines.
# PML_8 requires sufficient mesh lines.
#
# openEMS may warn that there are not enough lines and may
# internally reset those directions.
#
# We intentionally do NOT add mesh lines because you requested
# the exact existing XML.

FDTD.SetBoundaryCond([
    "PML_8",   # -X
    "PML_8",   # +X
    "PML_8",   # -Y
    "PML_8",   # +Y
    "PML_8",   # -Z
    "PML_8"    # +Z
])


# ================================================================
# 9. EXCITATION
# ================================================================

FDTD.SetGaussExcite(
    F0,
    FC
)


# ================================================================
# 10. LUMPED PORTS
# ------------------------------------------------
# These reproduce the working port geometry used for the filter.
#
# XML contains the filter geometry but NOT the Python lumped-port
# objects, so these have to be recreated.
#
# Port:
# x = 7.0 mm input
# x = 58.3 mm output
# y = +/- 0.6 mm
# z = +0.15 mm -> +2.10 mm
#
# p_dir = z
# R = 50 ohm
# ================================================================

print()
print("Creating ports...")

port1 = FDTD.AddLumpedPort(
    1,
    Z0,

    [PORT1_X,
     -CENTER_WIDTH / 2,
     H_LOW / 2],

    [PORT1_X,
     CENTER_WIDTH / 2,
     OUTER_HEIGHT / 2 - WALL],

    "z",

    excite=1.0,
    priority=30
)


port2 = FDTD.AddLumpedPort(
    2,
    Z0,

    [PORT2_X,
     -CENTER_WIDTH / 2,
     H_LOW / 2],

    [PORT2_X,
     CENTER_WIDTH / 2,
     OUTER_HEIGHT / 2 - WALL],

    "z",

    excite=0.0,
    priority=30
)


# ================================================================
# 11. SAVE A COPY OF THE EXACT XML
# ================================================================

COPIED_XML = RUN_DIR / "exact_geometry.xml"

shutil.copy2(
    XML_FILE,
    COPIED_XML
)


# ================================================================
# 12. RUN FDTD
# ================================================================

SIM_PATH = RUN_DIR / "simulation"

if SIM_PATH.exists():
    shutil.rmtree(SIM_PATH)

SIM_PATH.mkdir(parents=True, exist_ok=True)

print()
print("=" * 72)
print("STARTING ONE FDTD SIMULATION")
print("=" * 72)
print()
print("This is NOT an optimization.")
print("There will be ONE simulation only.")
print()

FDTD.Run(
    str(SIM_PATH),
    cleanup=True,
    verbose=2
)


# ================================================================
# 13. CALCULATE S-PARAMETERS
# ================================================================

print()
print("=" * 72)
print("CALCULATING S-PARAMETERS")
print("=" * 72)

port1.CalcPort(
    str(SIM_PATH),
    FREQS,
    ref_impedance=Z0
)

port2.CalcPort(
    str(SIM_PATH),
    FREQS,
    ref_impedance=Z0
)


# ================================================================
# 14. S PARAMETERS
# ================================================================

S11 = port1.uf_ref / port1.uf_inc
S21 = port2.uf_ref / port1.uf_inc


S11_dB = 20.0 * np.log10(
    np.maximum(np.abs(S11), 1e-15)
)

S21_dB = 20.0 * np.log10(
    np.maximum(np.abs(S21), 1e-15)
)


# ================================================================
# 15. SAVE CSV
# ================================================================

CSV_FILE = RUN_DIR / "S_parameters_0p1_to_100GHz.csv"

data = np.column_stack([
    FREQS / 1e9,

    np.real(S11),
    np.imag(S11),
    np.abs(S11),
    S11_dB,

    np.real(S21),
    np.imag(S21),
    np.abs(S21),
    S21_dB
])

header = (
    "Frequency_GHz,"
    "S11_real,S11_imag,S11_mag,S11_dB,"
    "S21_real,S21_imag,S21_mag,S21_dB"
)

np.savetxt(
    CSV_FILE,
    data,
    delimiter=",",
    header=header,
    comments=""
)


# ================================================================
# 16. HELPER
# ================================================================

def value_at(freq_ghz, response_db):

    idx = np.argmin(
        np.abs(FREQS / 1e9 - freq_ghz)
    )

    return response_db[idx]


# ================================================================
# 17. REPORT IMPORTANT FREQUENCIES
# ================================================================

print()
print("=" * 72)
print("RESULTS")
print("=" * 72)

for f in [
    1,
    5,
    8,
    10,
    12,
    15,
    20,
    24,
    26,
    28,
    30,
    32,
    35,
    40,
    50,
    60,
    70,
    80,
    90,
    100
]:

    s21 = value_at(f, S21_dB)
    s11 = value_at(f, S11_dB)

    print(
        f"{f:6.1f} GHz : "
        f"S21 = {s21:9.3f} dB    "
        f"S11 = {s11:9.3f} dB"
    )


# ================================================================
# 18. FIND 20 GHz ATTENUATION
# ================================================================

S21_20 = value_at(20, S21_dB)

print()
print(f"S21 @ 20 GHz = {S21_20:.3f} dB")


# ================================================================
# 19. FIND 28 GHz PEAK
# ================================================================

mask_28 = (
    (FREQS >= 27e9) &
    (FREQS <= 29e9)
)

idx28 = np.argmax(
    S21_dB[mask_28]
)

f28 = FREQS[mask_28][idx28] / 1e9
s28 = S21_dB[mask_28][idx28]

print(
    f"S21 peak 27–29 GHz = "
    f"{s28:.3f} dB @ {f28:.3f} GHz"
)


# ================================================================
# 20. 20-40 GHz WORST TRANSMISSION
# ================================================================

mask_20_40 = (
    (FREQS >= 20e9) &
    (FREQS <= 40e9)
)

idx_worst = np.argmax(
    S21_dB[mask_20_40]
)

f_worst = (
    FREQS[mask_20_40][idx_worst]
    / 1e9
)

s_worst = S21_dB[mask_20_40][idx_worst]

print(
    f"20–40 GHz worst S21 = "
    f"{s_worst:.3f} dB @ {f_worst:.3f} GHz"
)


# ================================================================
# 21. 40-100 GHz
# ================================================================

mask_40_100 = (
    (FREQS >= 40e9) &
    (FREQS <= 100e9)
)

idx_high = np.argmax(
    S21_dB[mask_40_100]
)

f_high = (
    FREQS[mask_40_100][idx_high]
    / 1e9
)

s_high = S21_dB[mask_40_100][idx_high]

print(
    f"40–100 GHz worst S21 = "
    f"{s_high:.3f} dB @ {f_high:.3f} GHz"
)


# ================================================================
# 22. PLOT
# ================================================================

plt.figure(
    figsize=(13, 7)
)

plt.plot(
    FREQS / 1e9,
    S21_dB,
    label="S21",
    linewidth=1.3
)

plt.plot(
    FREQS / 1e9,
    S11_dB,
    label="S11",
    linewidth=1.1
)

plt.axvline(
    10,
    linestyle="--",
    linewidth=1
)

plt.axvline(
    20,
    linestyle="--",
    linewidth=1
)

plt.axhline(
    -60,
    linestyle="--",
    linewidth=1
)

plt.xlim(
    0,
    100
)

plt.ylim(
    -100,
    5
)

plt.xlabel(
    "Frequency (GHz)"
)

plt.ylabel(
    "Magnitude (dB)"
)

plt.title(
    "EXACT COMPACT 10 GHz LPF — 0.1–100 GHz"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

PLOT_FILE = RUN_DIR / "S21_S11_0p1_to_100GHz.png"

plt.savefig(
    PLOT_FILE,
    dpi=200
)

plt.show()


# ================================================================
# 23. SAVE SUMMARY
# ================================================================

SUMMARY_FILE = RUN_DIR / "summary.txt"

with open(
    SUMMARY_FILE,
    "w"
) as f:

    f.write(
        "EXACT COMPACT 10 GHz LPF\n"
    )

    f.write(
        "=========================\n\n"
    )

    f.write(
        "Geometry source:\n"
    )

    f.write(
        f"{XML_FILE}\n\n"
    )

    f.write(
        "NO OPTIMIZATION\n"
    )

    f.write(
        "NO GEOMETRY MODIFICATION\n"
    )

    f.write(
        "NO MESH MODIFICATION\n\n"
    )

    f.write(
        f"Frequency range: "
        f"{FREQ_MIN/1e9} - {FREQ_MAX/1e9} GHz\n"
    )

    f.write(
        f"Frequency points: {N_FREQ}\n"
    )

    f.write(
        f"Max timesteps: {NR_TS}\n"
    )

    f.write(
        f"End criterion: {END_CRITERIA}\n\n"
    )

    f.write(
        f"S21 @ 20 GHz: "
        f"{S21_20:.6f} dB\n"
    )

    f.write(
        f"27-29 GHz peak: "
        f"{s28:.6f} dB @ "
        f"{f28:.6f} GHz\n"
    )

    f.write(
        f"20-40 GHz worst: "
        f"{s_worst:.6f} dB @ "
        f"{f_worst:.6f} GHz\n"
    )

    f.write(
        f"40-100 GHz worst: "
        f"{s_high:.6f} dB @ "
        f"{f_high:.6f} GHz\n"
    )


print()
print("=" * 72)
print("DONE")
print("=" * 72)

print()
print(f"CSV     : {CSV_FILE}")
print(f"PLOT    : {PLOT_FILE}")
print(f"SUMMARY : {SUMMARY_FILE}")
print()