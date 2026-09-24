# ============================================================
# COMPACT 10 GHz CRYOGENIC LPF
# openEMS V1 — PORT-CORRECTED
#
# Target:
#   Z0              = 50 ohm
#   fc              = 10 GHz
#   Chebyshev-I     = 0.2 dB ripple
#   Stopband        >= 60 dB @ 20 GHz
#
# Form factor:
#   65.3 mm x 5.0 mm x 4.6 mm
#
# Architecture:
#   symmetric 7-section stepped-impedance rectangular coax
#
# IMPORTANT:
#   This is the INITIAL EM DESIGN.
#   Do NOT claim the 60 dB target until full-wave S21 confirms it.
# ============================================================

from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# OPENEMS PATH
# ============================================================

OPENEMS_PATH = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
)

OPENEMS_ROOT = OPENEMS_PATH / "openEMS"

os.add_dll_directory(str(OPENEMS_ROOT))

sys.path.insert(
    0,
    str(OPENEMS_ROOT / "Python")
)

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ============================================================
# OUTPUT
# ============================================================

PROJECT_DIR = (
    Path.home()
    / "Desktop"
    / "cryogenic_filter_design"
    / "LPF10_V1"
)

PROJECT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

XML_FILE = PROJECT_DIR / "lpf10_v1.xml"
SIM_DIR = PROJECT_DIR / "sim"


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

C0 = 299792458.0

F_C = 10e9
F_STOP = 20e9

F_START = 1e9
F_END = 30e9
N_FREQ = 581

Z0 = 50.0


# ============================================================
# FORM FACTOR
# ============================================================

TOTAL_LENGTH = 65.3       # mm
OUTER_WIDTH = 5.0         # mm
OUTER_HEIGHT = 4.6        # mm

WALL = 0.20               # mm

# Center conductor heights
H_LOW = 0.30              # mm
H_HIGH = 0.80             # mm

# Center conductor width
CENTER_WIDTH = 1.20       # mm


# ============================================================
# FILTER LENGTH
# ============================================================

PORT_LENGTH = 8.0

FILTER_LENGTH = (
    TOTAL_LENGTH
    - 2.0 * PORT_LENGTH
)

SECTION_LENGTH = FILTER_LENGTH / 7.0


# ============================================================
# INITIAL IMPEDANCE TARGETS
# ============================================================

Z_LOW = 25.0
Z_HIGH = 85.0


# ============================================================
# CHEBYSHEV PROTOTYPE
# ============================================================

RIPPLE_DB = 0.2

epsilon = np.sqrt(
    10.0 ** (RIPPLE_DB / 10.0) - 1.0
)

N = 7

omega_s = F_STOP / F_C

A_ideal = (
    10.0
    * np.log10(
        1.0
        + epsilon**2
        * np.cosh(
            N * np.arccosh(omega_s)
        )**2
    )
)


# ============================================================
# DESIGN INFORMATION
# ============================================================

print()
print("=" * 70)
print("COMPACT 10 GHz CRYOGENIC LPF")
print("=" * 70)

print(
    f"Form factor       : "
    f"{TOTAL_LENGTH:.2f} x "
    f"{OUTER_WIDTH:.2f} x "
    f"{OUTER_HEIGHT:.2f} mm"
)

print(
    f"Cutoff target      : {F_C/1e9:.2f} GHz"
)

print(
    f"Stopband test      : {F_STOP/1e9:.2f} GHz"
)

print(
    f"Ripple             : {RIPPLE_DB:.2f} dB"
)

print(
    f"Order              : {N}"
)

print(
    f"Ideal attenuation  : {A_ideal:.3f} dB"
)

print(
    f"Section length     : {SECTION_LENGTH:.6f} mm"
)

print(
    f"Low-Z height       : {H_LOW:.3f} mm"
)

print(
    f"High-Z height      : {H_HIGH:.3f} mm"
)

print("=" * 70)


# ============================================================
# FDTD
# ============================================================

FDTD = openEMS(
    EndCriteria=1e-7,
    NrTS=180000
)

FDTD.SetGaussExcite(
    17e9,
    17e9
)

FDTD.SetBoundaryCond(
    [
        "PML_8",
        "PML_8",
        "PML_8",
        "PML_8",
        "PML_8",
        "PML_8",
    ]
)


# ============================================================
# STRUCTURE
# ============================================================

CSX = ContinuousStructure()

FDTD.SetCSX(CSX)


# ============================================================
# MATERIAL
# ============================================================

metal = CSX.AddMetal(
    "Copper"
)

# Background is vacuum.
# No explicit vacuum primitive is required.


# ============================================================
# MESH
# ============================================================

mesh = CSX.GetGrid()

# All dimensions are specified in mm.
mesh.SetDeltaUnit(1e-3)


# ============================================================
# X MESH
# ============================================================

x_lines = [
    0.0,
    TOTAL_LENGTH,
]

# Filter section boundaries
for n in range(8):

    x = (
        PORT_LENGTH
        + n * SECTION_LENGTH
    )

    x_lines.append(x)


# Launch refinement
x_lines.extend([
    PORT_LENGTH - 2.0,
    PORT_LENGTH - 1.0,
    PORT_LENGTH + 1.0,
    PORT_LENGTH + 2.0,

    TOTAL_LENGTH - PORT_LENGTH - 2.0,
    TOTAL_LENGTH - PORT_LENGTH - 1.0,
    TOTAL_LENGTH - PORT_LENGTH + 1.0,
    TOTAL_LENGTH - PORT_LENGTH + 2.0,
])


# Refine around all x discontinuities
for x in list(x_lines):

    for dx in [
        -0.10,
        -0.05,
        0.05,
        0.10,
    ]:

        xx = x + dx

        if 0.0 < xx < TOTAL_LENGTH:
            x_lines.append(xx)


x_lines = sorted(
    set(x_lines)
)

mesh.AddLine(
    "x",
    x_lines
)


# ============================================================
# Y MESH
# ============================================================

y_lines = [
    -OUTER_WIDTH / 2,
    -OUTER_WIDTH / 2 + WALL,

    -CENTER_WIDTH / 2,
    0.0,
    CENTER_WIDTH / 2,

    OUTER_WIDTH / 2 - WALL,
    OUTER_WIDTH / 2,
]


# Explicit port boundaries
y_lines.extend([
    -CENTER_WIDTH / 2,
    0.0,
    CENTER_WIDTH / 2,
])


# Refine
for y in list(y_lines):

    for dy in [
        -0.05,
        0.05,
    ]:

        yy = y + dy

        if (
            -OUTER_WIDTH / 2
            < yy
            < OUTER_WIDTH / 2
        ):
            y_lines.append(yy)


y_lines = sorted(
    set(y_lines)
)

mesh.AddLine(
    "y",
    y_lines
)


# ============================================================
# Z MESH
# ============================================================

z_lines = [
    -OUTER_HEIGHT / 2,
    -OUTER_HEIGHT / 2 + WALL,

    -H_HIGH / 2,
    -H_LOW / 2,

    0.0,

    H_LOW / 2,
    H_HIGH / 2,

    OUTER_HEIGHT / 2 - WALL,
    OUTER_HEIGHT / 2,
]


# Explicit port boundaries
z_lines.extend([
    H_LOW / 2,
    OUTER_HEIGHT / 2 - WALL,
])


# Refine
for z in list(z_lines):

    for dz in [
        -0.05,
        0.05,
    ]:

        zz = z + dz

        if (
            -OUTER_HEIGHT / 2
            < zz
            < OUTER_HEIGHT / 2
        ):
            z_lines.append(zz)


z_lines = sorted(
    set(z_lines)
)

mesh.AddLine(
    "z",
    z_lines
)


# ============================================================
# OUTER CONDUCTOR
# ============================================================

# Bottom wall
metal.AddBox(
    [
        0.0,
        -OUTER_WIDTH / 2,
        -OUTER_HEIGHT / 2,
    ],
    [
        TOTAL_LENGTH,
        OUTER_WIDTH / 2,
        -OUTER_HEIGHT / 2 + WALL,
    ],
    priority=10
)


# Top wall
metal.AddBox(
    [
        0.0,
        -OUTER_WIDTH / 2,
        OUTER_HEIGHT / 2 - WALL,
    ],
    [
        TOTAL_LENGTH,
        OUTER_WIDTH / 2,
        OUTER_HEIGHT / 2,
    ],
    priority=10
)


# Left wall
metal.AddBox(
    [
        0.0,
        -OUTER_WIDTH / 2,
        -OUTER_HEIGHT / 2,
    ],
    [
        TOTAL_LENGTH,
        -OUTER_WIDTH / 2 + WALL,
        OUTER_HEIGHT / 2,
    ],
    priority=10
)


# Right wall
metal.AddBox(
    [
        0.0,
        OUTER_WIDTH / 2 - WALL,
        -OUTER_HEIGHT / 2,
    ],
    [
        TOTAL_LENGTH,
        OUTER_WIDTH / 2,
        OUTER_HEIGHT / 2,
    ],
    priority=10
)


# ============================================================
# CENTER CONDUCTOR
# ============================================================

pattern = [
    "LOW",
    "HIGH",
    "LOW",
    "HIGH",
    "LOW",
    "HIGH",
    "LOW",
]


def section_geometry(n):

    x0 = (
        PORT_LENGTH
        + n * SECTION_LENGTH
    )

    x1 = x0 + SECTION_LENGTH

    if pattern[n] == "LOW":
        h = H_LOW
    else:
        h = H_HIGH

    return x0, x1, h


# ============================================================
# INPUT LAUNCH
# ============================================================

metal.AddBox(
    [
        0.0,
        -CENTER_WIDTH / 2,
        -H_LOW / 2,
    ],
    [
        PORT_LENGTH,
        CENTER_WIDTH / 2,
        H_LOW / 2,
    ],
    priority=20
)


# ============================================================
# FILTER SECTIONS
# ============================================================

for n in range(7):

    x0, x1, h = section_geometry(n)

    metal.AddBox(
        [
            x0,
            -CENTER_WIDTH / 2,
            -h / 2,
        ],
        [
            x1,
            CENTER_WIDTH / 2,
            h / 2,
        ],
        priority=20
    )


# ============================================================
# OUTPUT LAUNCH
# ============================================================

metal.AddBox(
    [
        TOTAL_LENGTH - PORT_LENGTH,
        -CENTER_WIDTH / 2,
        -H_LOW / 2,
    ],
    [
        TOTAL_LENGTH,
        CENTER_WIDTH / 2,
        H_LOW / 2,
    ],
    priority=20
)


# ============================================================
# PORTS
# ============================================================
#
# The previous port was wrong because it had zero transverse
# area and resulted in:
#
#     Voltage excitations : 0
#     Current excitations : 0
#     Active cells        : 0
#
# The ports below span the full center-conductor width in Y
# and the vacuum region from the center conductor to the
# upper outer conductor in Z.
#
# Port dimensions:
#
#     Y : -0.6 ... +0.6 mm
#     Z : +0.15 ... +2.10 mm
#
# The excitation direction is Z.
# ============================================================

PORT1_X = PORT_LENGTH - 1.0

PORT2_X = (
    TOTAL_LENGTH
    - PORT_LENGTH
    + 1.0
)

PORT_Y1 = -CENTER_WIDTH / 2
PORT_Y2 = CENTER_WIDTH / 2

PORT_Z1 = H_LOW / 2
PORT_Z2 = OUTER_HEIGHT / 2 - WALL


# ============================================================
# INPUT PORT
# ============================================================

port1 = FDTD.AddLumpedPort(
    1,
    Z0,

    [
        PORT1_X,
        PORT_Y1,
        PORT_Z1,
    ],

    [
        PORT1_X,
        PORT_Y2,
        PORT_Z2,
    ],

    "z",

    excite=1.0,
    priority=30,
)


# ============================================================
# OUTPUT PORT
# ============================================================

port2 = FDTD.AddLumpedPort(
    2,
    Z0,

    [
        PORT2_X,
        PORT_Y1,
        PORT_Z1,
    ],

    [
        PORT2_X,
        PORT_Y2,
        PORT_Z2,
    ],

    "z",

    excite=0.0,
    priority=30,
)


# ============================================================
# WRITE XML
# ============================================================

CSX.Write2XML(
    str(XML_FILE)
)

print()
print("XML written:")
print(XML_FILE)


# ============================================================
# SIMULATION DIRECTORY
# ============================================================

SIM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RUN SIMULATION
# ============================================================

print()
print("Starting openEMS...")
print()

FDTD.Run(
    str(SIM_DIR),
    cleanup=True,
    verbose=3,
    numThreads=-1,
    disable_dumps=True,
)


# ============================================================
# POST PROCESSING
# ============================================================

print()
print("Simulation finished.")
print()
print("Calculating port data...")


FREQ = np.linspace(
    F_START,
    F_END,
    N_FREQ
)


# ============================================================
# CALCULATE PORT DATA
# ============================================================

port1.CalcPort(
    str(SIM_DIR),
    FREQ,
    ref_impedance=Z0,
)

port2.CalcPort(
    str(SIM_DIR),
    FREQ,
    ref_impedance=Z0,
)


# ============================================================
# S PARAMETERS
# ============================================================

s11 = (
    port1.uf_ref
    / port1.uf_inc
)

s21 = (
    port2.uf_ref
    / port1.uf_inc
)


S11_DB = 20.0 * np.log10(
    np.maximum(
        np.abs(s11),
        1e-15,
    )
)

S21_DB = 20.0 * np.log10(
    np.maximum(
        np.abs(s21),
        1e-15,
    )
)


# ============================================================
# METRICS
# ============================================================

passband = (
    FREQ <= 10e9
)

stopband_20 = np.abs(
    FREQ - 20e9
).argmin()


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"S21 @ 20 GHz       = "
    f"{S21_DB[stopband_20]:.3f} dB"
)

if np.any(passband):

    print(
        f"Worst S21 < 10 GHz = "
        f"{np.max(S21_DB[passband]):.3f} dB"
    )

    print(
        f"Worst S11 < 10 GHz = "
        f"{np.max(S11_DB[passband]):.3f} dB"
    )

print("=" * 70)


# ============================================================
# SAVE TOUCHSTONE
# ============================================================

S2P_FILE = (
    PROJECT_DIR
    / "lpf10_v1.s2p"
)


with open(
    S2P_FILE,
    "w"
) as f:

    f.write(
        "# GHz S RI R 50\n"
    )

    for k, freq in enumerate(FREQ):

        # S11
        f.write(
            f"{freq/1e9:.9f} "
            f"{np.real(s11[k]):.12e} "
            f"{np.imag(s11[k]):.12e} "
        )

        # S21
        f.write(
            f"{np.real(s21[k]):.12e} "
            f"{np.imag(s21[k]):.12e} "
        )

        # S12
        f.write(
            f"{np.real(s21[k]):.12e} "
            f"{np.imag(s21[k]):.12e} "
        )

        # S22
        f.write(
            f"{np.real(s11[k]):.12e} "
            f"{np.imag(s11[k]):.12e}\n"
        )


print()
print("Touchstone written:")
print(S2P_FILE)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.plot(
    FREQ / 1e9,
    S21_DB,
    label="S21",
)

ax.plot(
    FREQ / 1e9,
    S11_DB,
    label="S11",
)

ax.axvline(
    10,
    linestyle="--",
    label="10 GHz",
)

ax.axvline(
    20,
    linestyle="--",
    label="20 GHz",
)

ax.axhline(
    -60,
    linestyle="--",
    label="-60 dB",
)

ax.set_xlabel(
    "Frequency (GHz)"
)

ax.set_ylabel(
    "Magnitude (dB)"
)

ax.set_title(
    "Compact 10 GHz Cryogenic LPF — openEMS V1"
)

ax.set_xlim(
    1,
    30
)

ax.set_ylim(
    -100,
    5
)

ax.grid(True)
ax.legend()

plt.tight_layout()


PLOT_FILE = (
    PROJECT_DIR
    / "lpf10_v1_response.png"
)

plt.savefig(
    PLOT_FILE,
    dpi=200
)

plt.show()


print()
print("Plot written:")
print(PLOT_FILE)

print()
print("DONE.")