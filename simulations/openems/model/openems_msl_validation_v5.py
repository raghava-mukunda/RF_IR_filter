from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[3]

SIM_DIR = (
    ROOT
    / "simulations"
    / "openems"
    / "results"
    / "lumped_validation_v9"
)

CSV_OUT = ROOT / "results" / "em" / "lumped_validation_v9.csv"
PNG_OUT = ROOT / "results" / "em" / "lumped_validation_v9.png"

SIM_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PHYSICAL PARAMETERS
# ============================================================================

# All geometry is specified in mm.
UNIT = 1e-3

EPS_R = 3.38

# Substrate
H = 1.524

# 50-ohm microstrip width previously calculated for eps_r = 3.38
W = 3.5577

# Uniform transmission line
LINE_XMIN = -30.0
LINE_XMAX = +30.0
LINE_LENGTH = LINE_XMAX - LINE_XMIN

# Port centers
PORT1_X = -20.0
PORT2_X = +20.0

# IMPORTANT:
# V8 had zero X extent.
# V9 uses a finite 1-mm-long lumped-port volume.
PORT_LENGTH = 1.0

PORT1_XMIN = PORT1_X - PORT_LENGTH / 2.0
PORT1_XMAX = PORT1_X + PORT_LENGTH / 2.0

PORT2_XMIN = PORT2_X - PORT_LENGTH / 2.0
PORT2_XMAX = PORT2_X + PORT_LENGTH / 2.0

# Simulation air margin
AIR_SIDE = 20.0
AIR_HEIGHT = 20.0

# Port/reference impedance
Z0 = 50.0

# ============================================================================
# FREQUENCY
# ============================================================================

FSTART = 1e9
FSTOP = 12e9

FC = (FSTART + FSTOP) / 2.0
FBW = (FSTOP - FSTART) / 2.0

N_FREQ = 401
FREQ = np.linspace(FSTART, FSTOP, N_FREQ)

# ============================================================================
# FDTD
# ============================================================================

MAX_TIMESTEPS = 200000

# ============================================================================
# MESH
# ============================================================================

# Global transverse mesh scale.
DX_GLOBAL = 0.5
DY_GLOBAL = 0.5
DZ_GLOBAL = 0.25


# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

print("=" * 78)
print("openEMS UNIFORM MICROSTRIP LUMPED-PORT VALIDATION V9")
print("=" * 78)

print("Geometry:")
print(f"  Substrate epsilon_r : {EPS_R}")
print(f"  Substrate height    : {H:.4f} mm")
print(f"  Microstrip width    : {W:.4f} mm")
print(f"  Line length         : {LINE_LENGTH:.4f} mm")

print()
print("Ports:")
print(f"  Port 1 center       : {PORT1_X:.4f} mm")
print(f"  Port 1 X extent     : {PORT1_XMIN:.4f} -> {PORT1_XMAX:.4f} mm")
print(f"  Port 2 center       : {PORT2_X:.4f} mm")
print(f"  Port 2 X extent     : {PORT2_XMIN:.4f} -> {PORT2_XMAX:.4f} mm")
print(f"  Port length         : {PORT_LENGTH:.4f} mm")
print(f"  Port width          : {W:.4f} mm")
print(f"  Port height         : {H:.4f} mm")
print(f"  Port impedance      : {Z0:.2f} ohm")

print()
print("Simulation:")
print(f"  Frequency range     : {FSTART/1e9:.3f} -> {FSTOP/1e9:.3f} GHz")
print(f"  Frequency points    : {N_FREQ}")
print(f"  Center frequency    : {FC/1e9:.3f} GHz")
print(f"  Maximum timesteps   : {MAX_TIMESTEPS}")
print("  Boundary condition  : PML_8")
print("  Port type           : LumpedPort")
print("  Port priority       : 50")
print("  Metal priority      : 10")

print("=" * 78)


# ============================================================================
# FDTD OBJECT
# ============================================================================

FDTD = openEMS(
    NrTS=MAX_TIMESTEPS,
    EndCriteria=1e-5
)

FDTD.SetGaussExcite(
    FC,
    FBW
)

FDTD.SetBoundaryCond([
    "PML_8",  # xmin
    "PML_8",  # xmax
    "PML_8",  # ymin
    "PML_8",  # ymax
    "PML_8",  # zmin
    "PML_8",  # zmax
])


# ============================================================================
# CONTINUOUS STRUCTURE
# ============================================================================

CSX = ContinuousStructure()

FDTD.SetCSX(CSX)

mesh = CSX.GetGrid()

mesh.SetDeltaUnit(UNIT)


# ============================================================================
# MATERIALS
# ============================================================================

substrate = CSX.AddMaterial(
    "SUBSTRATE",
    epsilon=EPS_R
)

signal = CSX.AddMetal(
    "SIGNAL"
)

ground = CSX.AddMetal(
    "GROUND"
)


# ============================================================================
# SUBSTRATE
# ============================================================================

substrate.AddBox(
    start=[
        LINE_XMIN - AIR_SIDE,
        -AIR_SIDE,
        0.0
    ],
    stop=[
        LINE_XMAX + AIR_SIDE,
        +AIR_SIDE,
        H
    ],
    priority=1
)


# ============================================================================
# GROUND PLANE
# ============================================================================

ground.AddBox(
    start=[
        LINE_XMIN - AIR_SIDE,
        -AIR_SIDE,
        0.0
    ],
    stop=[
        LINE_XMAX + AIR_SIDE,
        +AIR_SIDE,
        0.0
    ],
    priority=10
)


# ============================================================================
# SIGNAL MICROSTRIP
# ============================================================================

signal.AddBox(
    start=[
        LINE_XMIN,
        -W / 2.0,
        H
    ],
    stop=[
        LINE_XMAX,
        +W / 2.0,
        H
    ],
    priority=10
)


# ============================================================================
# MESH — GLOBAL
# ============================================================================

# X direction
x_global = np.linspace(
    LINE_XMIN - AIR_SIDE,
    LINE_XMAX + AIR_SIDE,
    int(
        round(
            (LINE_XMAX - LINE_XMIN + 2.0 * AIR_SIDE)
            / DX_GLOBAL
        )
    ) + 1
)

# Add exact port boundaries and centers.
x_port_lines = [
    PORT1_XMIN,
    PORT1_X,
    PORT1_XMAX,
    PORT2_XMIN,
    PORT2_X,
    PORT2_XMAX,
]

# Add exact line endpoints.
x_special = [
    LINE_XMIN,
    LINE_XMAX,
]

x_lines = sorted(
    set(
        np.concatenate([
            x_global,
            np.asarray(x_port_lines),
            np.asarray(x_special),
        ]).tolist()
    )
)

mesh.AddLine(
    "x",
    x_lines
)


# ============================================================================
# MESH — Y
# ============================================================================

y_global = np.linspace(
    -AIR_SIDE,
    +AIR_SIDE,
    int(round(2.0 * AIR_SIDE / DY_GLOBAL)) + 1
)

y_special = [
    -W / 2.0,
    0.0,
    +W / 2.0,
]

y_lines = sorted(
    set(
        np.concatenate([
            y_global,
            np.asarray(y_special),
        ]).tolist()
    )
)

mesh.AddLine(
    "y",
    y_lines
)


# ============================================================================
# MESH — Z
# ============================================================================

z_global = np.linspace(
    0.0,
    H + AIR_HEIGHT,
    int(round((H + AIR_HEIGHT) / DZ_GLOBAL)) + 1
)

z_special = [
    0.0,
    H / 4.0,
    H / 2.0,
    3.0 * H / 4.0,
    H,
]

z_lines = sorted(
    set(
        np.concatenate([
            z_global,
            np.asarray(z_special),
        ]).tolist()
    )
)

mesh.AddLine(
    "z",
    z_lines
)


# ============================================================================
# FORCE METAL EDGES INTO THE MESH
# ============================================================================

FDTD.AddEdges2Grid(
    dirs="all",
    properties=ground
)

FDTD.AddEdges2Grid(
    dirs="all",
    properties=signal
)


# ============================================================================
# LUMPED PORT 1
#
# FINITE 3-D PORT VOLUME:
#
#       x = -20.5 ... -19.5 mm
#       y = -W/2 ... +W/2
#       z = 0 ... H
#
# The excitation direction is z.
#
# ============================================================================

port1_start = [
    PORT1_XMIN,
    -W / 2.0,
    0.0
]

port1_stop = [
    PORT1_XMAX,
    +W / 2.0,
    H
]

port1 = FDTD.AddLumpedPort(
    1,
    Z0,
    port1_start,
    port1_stop,
    "z",
    1.0,
    priority=50,
    edges2grid="xy"
)


# ============================================================================
# LUMPED PORT 2
#
# Passive receiving port.
#
# ============================================================================

port2_start = [
    PORT2_XMIN,
    -W / 2.0,
    0.0
]

port2_stop = [
    PORT2_XMAX,
    +W / 2.0,
    H
]

port2 = FDTD.AddLumpedPort(
    2,
    Z0,
    port2_start,
    port2_stop,
    "z",
    0.0,
    priority=50,
    edges2grid="xy"
)


# ============================================================================
# WRITE XML
# ============================================================================

XML_FILE = SIM_DIR / "lumped_validation_v9.xml"

CSX.Write2XML(
    str(XML_FILE)
)

print()
print(f"XML written : {XML_FILE}")

print()
print("Port geometry:")
print(f"  P1 start : {port1_start}")
print(f"  P1 stop  : {port1_stop}")
print(f"  P2 start : {port2_start}")
print(f"  P2 stop  : {port2_stop}")

print()
print("Starting openEMS...")
print()


# ============================================================================
# CHECK EXECUTABLE
# ============================================================================

if shutil.which("openEMS") is None:
    raise RuntimeError(
        "openEMS executable was not found in PATH."
    )


# ============================================================================
# RUN FDTD
# ============================================================================

FDTD.Run(
    str(SIM_DIR),
    cleanup=False,
    verbose=3
)


# ============================================================================
# POST PROCESS
# ============================================================================

print()
print("=" * 78)
print("POST PROCESSING")
print("=" * 78)

port1.CalcPort(
    str(SIM_DIR),
    FREQ,
    ref_impedance=Z0
)

port2.CalcPort(
    str(SIM_DIR),
    FREQ,
    ref_impedance=Z0
)


# ============================================================================
# S-PARAMETERS
# ============================================================================

S11 = (
    port1.uf_ref
    / port1.uf_inc
)

S21 = (
    port2.uf_ref
    / port1.uf_inc
)


S11_dB = 20.0 * np.log10(
    np.maximum(np.abs(S11), 1e-15)
)

S21_dB = 20.0 * np.log10(
    np.maximum(np.abs(S21), 1e-15)
)


# ============================================================================
# POWER CONSERVATION
# ============================================================================

power_ratio = (
    np.abs(S11) ** 2
    + np.abs(S21) ** 2
)


# ============================================================================
# INPUT IMPEDANCE
# ============================================================================

Zin = (
    port1.uf_tot
    / port1.if_tot
)


# ============================================================================
# 5 GHz RESULT
# ============================================================================

idx5 = np.argmin(
    np.abs(FREQ - 5e9)
)

freq5 = FREQ[idx5]

zin5 = Zin[idx5]

print()
print("=" * 78)
print("V9 RESULT AT 5 GHz")
print("=" * 78)

print(
    f"Frequency       : {freq5/1e9:.6f} GHz"
)

print(
    f"S11             : {S11_dB[idx5]:.6f} dB"
)

print(
    f"S21             : {S21_dB[idx5]:.6f} dB"
)

print(
    f"|S11|^2+|S21|^2 : {power_ratio[idx5]:.6f}"
)

print(
    f"Zin             : "
    f"{zin5.real:.6f} "
    f"{zin5.imag:+.6f}j ohm"
)

print(
    f"|Zin|           : "
    f"{abs(zin5):.6f} ohm"
)

print("=" * 78)


# ============================================================================
# GLOBAL RESULTS
# ============================================================================

print()
print("=" * 78)
print("GLOBAL V9 RESULTS")
print("=" * 78)

print(
    f"S11 maximum : {np.max(S11_dB):.6f} dB"
)

print(
    f"S11 minimum : {np.min(S11_dB):.6f} dB"
)

print(
    f"S21 maximum : {np.max(S21_dB):.6f} dB"
)

print(
    f"S21 minimum : {np.min(S21_dB):.6f} dB"
)

print(
    f"Power max   : {np.max(power_ratio):.6f}"
)

print(
    f"Power min   : {np.min(power_ratio):.6f}"
)

print("=" * 78)


# ============================================================================
# AUTOMATIC VALIDATION
# ============================================================================

# These are deliberately conservative acceptance criteria.
#
# A uniform microstrip should:
#
#   1. have low reflection
#   2. transmit almost all incident power
#   3. remain passive
#
# We do NOT require perfection because this is an FDTD
# discretization of a distributed microstrip.

PASS_S11 = np.max(S11_dB) < -15.0

PASS_S21 = np.max(S21_dB) > -1.0

PASS_POWER = (
    np.max(power_ratio) < 1.05
    and np.min(power_ratio) > 0.90
)

PASS_ZIN = (
    abs(abs(zin5) - Z0) < 10.0
)


print()
print("=" * 78)
print("VALIDATION")
print("=" * 78)

print(
    f"S11 < -15 dB         : "
    f"{'PASS' if PASS_S11 else 'FAIL'}"
)

print(
    f"S21 > -1 dB          : "
    f"{'PASS' if PASS_S21 else 'FAIL'}"
)

print(
    f"0.90 < power < 1.05  : "
    f"{'PASS' if PASS_POWER else 'FAIL'}"
)

print(
    f"|Zin - 50 ohm| < 10  : "
    f"{'PASS' if PASS_ZIN else 'FAIL'}"
)

OVERALL = (
    PASS_S11
    and PASS_S21
    and PASS_POWER
    and PASS_ZIN
)

print()

if OVERALL:
    print("OVERALL VALIDATION : PASS")
else:
    print("OVERALL VALIDATION : FAIL")

print("=" * 78)


# ============================================================================
# SAVE CSV
# ============================================================================

np.savetxt(
    CSV_OUT,
    np.column_stack([
        FREQ / 1e9,
        S11_dB,
        S21_dB,
        power_ratio,
        Zin.real,
        Zin.imag,
        np.abs(Zin),
    ]),
    delimiter=",",
    header=(
        "frequency_GHz,"
        "S11_dB,"
        "S21_dB,"
        "power_ratio,"
        "Zin_real_ohm,"
        "Zin_imag_ohm,"
        "Zin_abs_ohm"
    ),
    comments=""
)

print()
print(f"CSV saved  : {CSV_OUT}")


# ============================================================================
# PLOT
# ============================================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    FREQ / 1e9,
    S11_dB,
    label="S11"
)

plt.plot(
    FREQ / 1e9,
    S21_dB,
    label="S21"
)

plt.axhline(
    -15.0,
    linestyle="--",
    label="-15 dB"
)

plt.axhline(
    -1.0,
    linestyle=":",
    label="-1 dB"
)

plt.xlabel(
    "Frequency (GHz)"
)

plt.ylabel(
    "Magnitude (dB)"
)

plt.title(
    "openEMS Uniform Microstrip Lumped-Port Validation V9"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PNG_OUT,
    dpi=150
)

plt.show()

print(
    f"Plot saved  : {PNG_OUT}"
)

print()
print("=" * 78)
print("V9 COMPLETE")
print("=" * 78)