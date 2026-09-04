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
copper_thickness = 1
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SIM_PATH = (
    PROJECT_ROOT
    / "simulations"
    / "openems"
    / "results"
    / "patch_reference"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "em"
)

PLOT_PATH = (
    RESULTS_PATH
    / "patch_reference_finite_copper_s11.png"
)

CSV_PATH = (
    RESULTS_PATH
    / "patch_reference_finite_copper_s11.csv"
)

# IMPORTANT:
# Create BOTH directories before Write2XML().
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
EPS0 = 8.8541878128e-12


# ================================================================
# SIMULATION PARAMETERS
# ================================================================

f0 = 2.4e9
fc = 1.0e9

f_start = 1.0e9
f_stop = 4.0e9


# ================================================================
# GEOMETRY UNIT
# ================================================================

# millimetres
unit = 1e-3


# ================================================================
# PATCH
# ================================================================

patch_width = 40.0
patch_length = 40.0


# ================================================================
# SUBSTRATE
# ================================================================

substrate_epsr = 3.38
substrate_thickness = 1.524

substrate_width = 60.0
substrate_length = 60.0


# ================================================================
# FEED
# ================================================================

feed_x = -6.0
feed_R = 50.0


# ================================================================
# SIMULATION BOX
# ================================================================

SimBox = np.array(
    [
        100.0,
        100.0,
        80.0,
    ]
)


# ================================================================
# FDTD
# ================================================================

FDTD = openEMS(
    NrTS=100000,
    EndCriteria=1e-4,
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
#
# The supplied reference uses MUR boundaries.
# ================================================================

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
# PATCH METAL
# ================================================================

patch = CSX.AddMetal(
    "patch"
)

patch.AddBox(
    start=[
        -patch_width / 2,
        -patch_length / 2,
        substrate_thickness,
    ],

    stop=[
        patch_width / 2,
        patch_length / 2,
        substrate_thickness + copper_thickness,
    ],
)


# ================================================================
# SUBSTRATE
# ================================================================

substrate = CSX.AddMaterial(
    "substrate",
    epsilon=substrate_epsr,
)

substrate.AddBox(
    start=[
        -substrate_width / 2,
        -substrate_length / 2,
        0.0,
    ],

    stop=[
        substrate_width / 2,
        substrate_length / 2,
        substrate_thickness,
    ],
)


# ================================================================
# GROUND
# ================================================================

ground = CSX.AddMetal(
    "gnd"
)

ground.AddBox(
    start=[
        -substrate_width / 2,
        -substrate_length / 2,
        -copper_thickness,
    ],

    stop=[
        substrate_width / 2,
        substrate_length / 2,
        0.0,
    ],
)


# ================================================================
# LUMPED FEED PORT
#
# This follows the geometry used in the supplied reference:
#
# start = [feed_x, 0, 0]
# stop  = [feed_x, 0, substrate_thickness]
#
# excitation direction = z
# ================================================================

port = FDTD.AddLumpedPort(
    port_nr=5,
    R=feed_R,

    start=[
        feed_x,
        0.0,
        -copper_thickness,
    ],

    stop=[
        feed_x,
        0.0,
        substrate_thickness + copper_thickness,
    ],

    p_dir="z",

    excite=1,
)


# ================================================================
# MESH
# ================================================================

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

# ----------------------------------------------------------------
# Geometry boundaries MUST be represented by mesh lines.
# ----------------------------------------------------------------
x_edges = np.array([
    -SimBox[0] / 2.0,
    -substrate_width / 2.0,
    -patch_width / 2.0,
    feed_x,
    patch_width / 2.0,
    substrate_width / 2.0,
    SimBox[0] / 2.0,
], dtype=float)

y_edges = np.array([
    -SimBox[1] / 2.0,
    -substrate_length / 2.0,
    -patch_length / 2.0,
    patch_length / 2.0,
    substrate_length / 2.0,
    SimBox[1] / 2.0,
], dtype=float)

z_edges = np.array([
    -SimBox[2] / 3.0,
    -copper_thickness,
    0.0,
    substrate_thickness,
    substrate_thickness + copper_thickness,
    SimBox[2] * 2.0 / 3.0,
], dtype=float)

# ----------------------------------------------------------------
# Wavelength-based resolution.
# Use the highest significant excitation frequency f0+fc.
# ----------------------------------------------------------------
lambda_min = C0 / (f0 + fc)
lambda_min_mm = lambda_min / unit

# For a first sanity run, keep the free-space region modestly
# resolved while explicitly resolving the 1.524 mm dielectric.
coarse_step = lambda_min_mm / 15.0
substrate_step = substrate_thickness / 6.0

def uniform_lines(lo, hi, step):
    n = int(np.ceil((hi - lo) / step))
    return np.linspace(lo, hi, n + 1)

x_dense = uniform_lines(
    -SimBox[0] / 2.0,
    SimBox[0] / 2.0,
    coarse_step,
)
x_dense = np.unique(np.sort(np.concatenate((x_dense, x_edges))))

y_dense = uniform_lines(
    -SimBox[1] / 2.0,
    SimBox[1] / 2.0,
    coarse_step,
)
y_dense = np.unique(np.sort(np.concatenate((y_dense, y_edges))))

# Build z with fine spacing through the dielectric and coarser
# spacing in the surrounding air.
z_lower = uniform_lines(
    -SimBox[2] / 3.0,
    0.0,
    coarse_step,
)

z_ground = np.linspace(
    -copper_thickness,
    0.0,
    3,
)

z_sub = np.linspace(
    0.0,
    substrate_thickness,
    int(np.ceil(substrate_thickness / substrate_step)) + 1,
)

z_patch = np.linspace(
    substrate_thickness,
    substrate_thickness + copper_thickness,
    3,
)

z_upper = uniform_lines(
    substrate_thickness + copper_thickness,
    SimBox[2] * 2.0 / 3.0,
    coarse_step,
)

z_dense = np.unique(
    np.sort(
        np.concatenate(
            (z_lower, z_ground, z_sub, z_patch, z_upper, z_edges)
        )
    )
)

# Set the final mesh using the CSXCAD Python API.
mesh.SetLines("x", x_dense)
mesh.SetLines("y", y_dense)
mesh.SetLines("z", z_dense)

# ----------------------------------------------------------------
# Sanity report
# ----------------------------------------------------------------
print()
print("Mesh resolution:")
print(f"  X points            : {len(x_dense)}")
print(f"  Y points            : {len(y_dense)}")
print(f"  Z points            : {len(z_dense)}")
print(f"  Minimum wavelength  : {lambda_min_mm:.3f} mm")
print(f"  Coarse step         : {coarse_step:.3f} mm")
print(f"  Substrate step      : {substrate_step:.3f} mm")
print()

# ================================================================
# WRITE XML
# ================================================================

xml_path = SIM_PATH / "patch_reference.xml"

if xml_path.exists():
    xml_path.unlink()

CSX.Write2XML(
    str(xml_path)
)


# ================================================================
# REPORT
# ================================================================

print()
print("=" * 72)
print("openEMS REFERENCE PATCH TEST")
print("=" * 72)

print()

print(
    f"Simulation directory : {SIM_PATH}"
)

print(
    f"XML file             : {xml_path}"
)

print()

print(
    f"Patch                : "
    f"{patch_width:.2f} x {patch_length:.2f} mm"
)

print(
    f"Substrate            : "
    f"{substrate_width:.2f} x "
    f"{substrate_length:.2f} x "
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

print(
    f"Feed position        : "
    f"x = {feed_x:.2f} mm"
)

print(
    f"Feed resistance      : "
    f"{feed_R:.2f} Ohm"
)

print()

print(
    f"Excitation           : "
    f"{f0 / 1e9:.2f} GHz"
)

print(
    f"Bandwidth parameter   : "
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
    "Geometry written successfully."
)

print()

print(
    "Starting openEMS..."
)

print()


# ================================================================
# RUN
# ================================================================

FDTD.Run(
    str(SIM_PATH),
    cleanup=True,
)


# ================================================================
# POSTPROCESSING
# ================================================================

freq = np.linspace(
    max(f_start, f0 - fc),
    min(f_stop, f0 + fc),
    501,
)


print()
print(
    "Calculating port response..."
)


port.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=50.0,
)


# ================================================================
# S11
# ================================================================

s11 = (
    port.uf_ref
    / port.uf_inc
)


s11_db = (
    20.0
    * np.log10(
        np.maximum(
            np.abs(s11),
            1e-15,
        )
    )
)


# ================================================================
# NUMERICAL VALIDITY CHECK
# ================================================================

if not np.all(
    np.isfinite(s11_db)
):
    raise RuntimeError(
        "S11 contains NaN or Inf. "
        "The simulation is numerically invalid."
    )


# ================================================================
# FIND RESONANCE
# ================================================================

res_idx = np.argmin(
    s11_db
)

f_res = freq[
    res_idx
]

s11_min = s11_db[
    res_idx
]


# ================================================================
# SAVE CSV
# ================================================================

np.savetxt(
    CSV_PATH,

    np.column_stack(
        [
            freq / 1e9,
            s11_db,
        ]
    ),

    delimiter=",",

    header="frequency_GHz,S11_dB",

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
)

plt.axhline(
    -10.0,
    linestyle="--",
    linewidth=1.0,
)

plt.axvline(
    f_res / 1e9,
    linestyle=":",
    linewidth=1.0,
)

plt.xlabel(
    "Frequency (GHz)"
)

plt.ylabel(
    "S11 (dB)"
)

plt.title(
    "openEMS Reference Patch Antenna"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.xlim(
    freq[0] / 1e9,
    freq[-1] / 1e9,
)

plt.ylim(
    -40,
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
print("RESULT")
print("=" * 72)

print()

print(
    f"Resonance frequency : "
    f"{f_res / 1e9:.4f} GHz"
)

print(
    f"Minimum S11         : "
    f"{s11_min:.3f} dB"
)

print()

print(
    f"CSV                 : "
    f"{CSV_PATH}"
)

print(
    f"Plot                : "
    f"{PLOT_PATH}"
)

print()

print("=" * 72)
print("SIMULATION COMPLETE")
print("=" * 72)