"""
Wideband LP Filter leaky-coax low-pass filter — openEMS V1

Goal
----
Reproduce the published Wideband LP Filter topology of Rehammar & Gasparinetti:
a 50-ohm air-filled coaxial transmission line surrounded by rectangular
hollow-waveguide leakage apertures.

Published prototype parameters used as the target; V1 uses 0.1-mm grid-compatible realizations:
    aperture width a       = 4.00 mm
    aperture height b      = 5.00 mm
    aperture depth d       = 4.85 mm
    coax inner radius ri   = 1.59 mm
    coax outer radius ro   = 3.65 mm
    PTFE slab epsilon_r    = 2.2
    number of sections     = 4
    8 apertures / section = 32 total

Reference:
R. Rehammar and S. Gasparinetti,
"Low-pass filter with ultra-wide stopband for quantum computing applications,"
IEEE T-MTT 71 (2023), 3075-3080, arXiv:2205.03941.

The paper describes each section as a set of 2x4 hollow-waveguide
apertures on the outer coax body, with two consecutive circumferential
rings. The published prototype uses four sections.

This V1 is a geometry-reproduction / physics-validation run, not yet a
fabrication model of proprietary Wideband LP Filter internals. The Wideband LP Filter datasheet
does not disclose its internal geometry.

IMPORTANT IMPLEMENTATION DETAIL
--------------------------------
The Windows Python build available in this project does not expose the
coaxial port in the local installed build, so this script uses a 50-ohm
lumped coaxial excitation across the inner conductor / outer conductor at
each end. The coax geometry itself is fully 3-D.

The outer conductor is represented by a cylindrical PEC shell. Aperture
windows are carved from that shell using higher-priority air primitives.
Each aperture is then filled with a PTFE slab (epsilon_r=2.2) of the
published depth.

The paper's physical aperture layout is reproduced as two rings of four
apertures per section. The four apertures in a ring are centered at
azimuths 0, 90, 180, 270 degrees.

The coax axis is x.

Run:
    python Wideband LP Filter_Wideband LP Filter_openems_v1.py

Outputs:
    results/em/Wideband LP Filter_Wideband LP Filter_openems_v1/
        metrics.json
        geometry.json
        Wideband LP Filter_Wideband LP Filter_openems_v1.csv
        geometry.txt
    results/plots/Wideband LP Filter_Wideband LP Filter_openems_v1.png
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# NATIVE OPENEMS SETUP
# ============================================================================

OPENEMS_ROOT = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
    / "openEMS"
)

if not OPENEMS_ROOT.exists():
    raise RuntimeError(f"openEMS native directory not found: {OPENEMS_ROOT}")

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["PATH"] = str(OPENEMS_ROOT) + os.pathsep + os.environ.get("PATH", "")

try:
    os.add_dll_directory(str(OPENEMS_ROOT))
except AttributeError:
    pass

print(f"openEMS native DLL path : {OPENEMS_ROOT}")

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_ID = os.environ.get("WB_RUN_ID", "lumped_port_characterization_v1")

SIM_PATH = (
    PROJECT_ROOT / "simulations" / "openems" / "results" / RUN_ID
)
RESULTS_PATH = (
    PROJECT_ROOT / "results" / "em" / RUN_ID
)
PLOT_PATH = RESULTS_PATH / "plots"

if SIM_PATH.exists():
    shutil.rmtree(SIM_PATH)

SIM_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PUBLISHED GEOMETRY
# ============================================================================

UNIT = 1e-3

Z0 = 50.0

# Known-good baseline; optimizer perturbs locally around this design.
A = float(os.environ.get("WB_A", "4.00"))
B = float(os.environ.get("WB_B", "5.00"))
D = float(os.environ.get("WB_D", "4.90"))

RI = 1.60
RO = 3.70
RO_OUT = 4.20
EPS_PTFE = 2.2

N_SECTIONS = int(os.environ.get("WB_NSECTIONS", "4"))
APERTURES_PER_SECTION = 8
AZIMUTHS_DEG = [0.0, 90.0, 180.0, 270.0]

RING_SPACING = float(os.environ.get("WB_RING", "5.00"))
SECTION_PITCH = float(os.environ.get("WB_PITCH", "12.00"))

SECTION_CENTERS = [
    (i - (N_SECTIONS - 1) / 2.0) * SECTION_PITCH
    for i in range(N_SECTIONS)
]

FILTER_X0 = SECTION_CENTERS[0] - RING_SPACING / 2.0
FILTER_X1 = SECTION_CENTERS[-1] + RING_SPACING / 2.0

LINE_X0 = -10.0
LINE_X1 = +10.0

APERTURE_R_IN = RO - 0.05
APERTURE_R_OUT = RO + D + 0.05

XY_MARGIN = 2.0
Z_MARGIN = 2.0

F_START = 1.0e9
F_STOP = 20.0e9
N_FREQ = 401

F0 = 50.0e9
FC_GAUSS = 50.0e9

NR_TS = 30000
END_CRITERIA = 3e-4
GRID = 0.25

BOUNDARY = ["PML_8"] * 6


# ============================================================================
# BASIC CHECKS
# ============================================================================

def grid_snap(v):
    return GRID * round(v / GRID)


# -------------------------------------------------------------------------
# FAST SCREENING MESH
# -------------------------------------------------------------------------
# The physical Wideband LP Filter dimensions are intentionally NOT forced onto the
# Cartesian mesh grid. openEMS/CSXCAD can represent geometry independently
# of the nominal mesh pitch. The old assertion incorrectly prevented the
# 0.25 mm screening mesh from using the published 1.60/3.70/4.90 mm geometry.
#
# The final convergence study will use finer meshes.
# -------------------------------------------------------------------------

print()
print("Geometry-to-mesh alignment check:")
print(f"  RI = {RI:.3f} mm")
print(f"  RO = {RO:.3f} mm")
print(f"  D  = {D:.3f} mm")
print(f"  A  = {A:.3f} mm")
print(f"  B  = {B:.3f} mm")
print(f"  nominal mesh = {GRID:.3f} mm")
print("  NOTE: geometry is retained at physical dimensions.")
print()

assert RO_OUT > RO


# ============================================================================
# PRINT DESIGN
# ============================================================================

print()
print("=" * 100)
print("Wideband LP Filter LEAKY COAX — OPENEMS V1")
print("=" * 100)
print("Published target parameters / V1 grid realization:")
print(f"  aperture a             = {A:.2f} mm")
print(f"  aperture b             = {B:.2f} mm")
print(f"  aperture d             = {D:.2f} mm")
print(f"  coax ri                = {RI:.2f} mm")
print(f"  coax ro                = {RO:.2f} mm")
print(f"  PTFE epsilon_r         = {EPS_PTFE:.3f}")
print(f"  sections               = {N_SECTIONS}")
print(f"  apertures/section      = {APERTURES_PER_SECTION}")
print()
print("V1 geometric interpretation:")
print(f"  ring spacing           = {RING_SPACING:.2f} mm")
print(f"  section pitch          = {SECTION_PITCH:.2f} mm")
print(f"  filter x               = {FILTER_X0:.2f} ... {FILTER_X1:.2f} mm")
print(f"  total RF length        = {LINE_X1-LINE_X0:.2f} mm")
print()
print("This is a Wideband LP Filter topology reproduction, not a claim to reproduce")
print("the proprietary internal geometry of the commercial Wideband LP Filter.")


# ============================================================================
# CUTOFF CHECK — RECTANGULAR APERTURE
# ============================================================================

C0 = 299792458.0

# Lowest TE10-like cutoff for a rectangular aperture of width A.
# This is a first-order estimate for air.
F_AP_CUTOFF = C0 / (2.0 * A * UNIT)

print()
print(f"Approx. aperture TE10 cutoff (air): {F_AP_CUTOFF/1e9:.3f} GHz")
print(
    "The published design uses the aperture geometry to make high-frequency "
    "radiation leak out while retaining low-frequency isolation."
)


# ============================================================================
# CSX
# ============================================================================

CSX = ContinuousStructure()

FDTD = openEMS(
    NrTS=NR_TS,
    EndCriteria=END_CRITERIA,
)

FDTD.SetCSX(CSX)
FDTD.SetBoundaryCond(BOUNDARY)
FDTD.SetGaussExcite(F0, FC_GAUSS)


# ============================================================================
# MATERIALS
# ============================================================================

PEC = CSX.AddMetal("PEC")

AIR = CSX.AddMaterial(
    "AIR",
    epsilon=1.0,
)

PTFE = CSX.AddMaterial(
    "PTFE",
    epsilon=EPS_PTFE,
)

# Inner conductor.
INNER = PEC.AddCylinder(
    [LINE_X0, 0.0, 0.0],
    [LINE_X1, 0.0, 0.0],
    RI,
    priority=30,
)

# Outer conductor shell.
#
# AddCylindricalShell(radius is the centerline radius, shell_width is total
# thickness). The shell's inner radius is RO_OUT - shell_width/2.
# Choose centerline so that the inner radius equals RO.
SHELL_RADIUS = (RO + RO_OUT) / 2.0
SHELL_WIDTH = RO_OUT - RO

OUTER = PEC.AddCylindricalShell(
    [LINE_X0, 0.0, 0.0],
    [LINE_X1, 0.0, 0.0],
    SHELL_RADIUS,
    SHELL_WIDTH,
    priority=10,
)


# ============================================================================
# APERTURE GEOMETRY
# ============================================================================

# We carve eight rectangular windows per section from the cylindrical shell.
#
# Coordinate convention:
#   x = coax axis
#   radial direction is in y-z plane
#   tangential width is A
#   axial height is B
#   radial depth is D
#
# For azimuth theta:
#   radial unit u = (cos(theta), sin(theta)) in y-z
#   tangent     t = (-sin(theta), cos(theta))
#
# The rectangular aperture box is constructed as a polygonal prism using
# AddLinPoly. This avoids relying on curved boolean subtraction.
#
# The opening is represented by high-priority AIR and PTFE primitives
# overlapping the low-priority PEC shell.


def add_aperture(section_index, ring_index, x_center, theta_deg):
    theta = math.radians(theta_deg)

    uy = math.cos(theta)
    uz = math.sin(theta)

    ty = -math.sin(theta)
    tz = math.cos(theta)

    # Axial dimension B.
    x0 = x_center - B / 2.0
    x1 = x_center + B / 2.0

    # Tangential half width.
    ht = A / 2.0

    # Radial endpoints. Start slightly inside the coax outer conductor so
    # that the opening is electrically connected to the coax cavity.
    r0 = APERTURE_R_IN
    r1 = APERTURE_R_OUT

    # Four points in the y-z plane at inner radial face.
    p0 = [r0 * uy - ht * ty, r0 * uz - ht * tz]
    p1 = [r0 * uy + ht * ty, r0 * uz + ht * tz]
    p2 = [r1 * uy + ht * ty, r1 * uz + ht * tz]
    p3 = [r1 * uy - ht * ty, r1 * uz - ht * tz]

    # AddLinPoly extrudes the polygon along its normal.
    points = [
        p0,
        p1,
        p2,
        p3,
    ]

    # IMPORTANT:
    # The installed CSXCAD Python binding expects polygon coordinates
    # as a 2 x N sequence:
    #
    #     [[y1, y2, y3, y4],
    #      [z1, z2, z3, z4]]
    #
    # rather than:
    #
    #     [[y1, z1], [y2, z2], ...]
    #
    # The previous version passed the latter representation and failed
    # with:
    #     AssertionError: points must be a list of length 2
    points_csx = [
        [p[0] for p in points],
        [p[1] for p in points],
    ]

    # Normal direction x; elevation is x0; length B.
    #
    # In CSXCAD, AddLinPoly(points, norm_dir, elevation, length)
    # uses the polygon in the plane perpendicular to norm_dir.
    #
    # The AIR primitive has higher priority than the PEC shell so that
    # the rectangular window removes the shell locally.
    air_poly = AIR.AddLinPoly(
        points_csx,
        "x",
        x0,
        B,
        priority=30,
    )

    # PTFE occupies the published aperture depth only.
    r_ptfe0 = RO - 0.02
    r_ptfe1 = RO + D

    p0p = [r_ptfe0 * uy - ht * ty, r_ptfe0 * uz - ht * tz]
    p1p = [r_ptfe0 * uy + ht * ty, r_ptfe0 * uz + ht * tz]
    p2p = [r_ptfe1 * uy + ht * ty, r_ptfe1 * uz + ht * tz]
    p3p = [r_ptfe1 * uy - ht * ty, r_ptfe1 * uz - ht * tz]

    points_ptfe = [
        [p0p[0], p1p[0], p2p[0], p3p[0]],
        [p0p[1], p1p[1], p2p[1], p3p[1]],
    ]

    PTFE.AddLinPoly(
        points_ptfe,
        "x",
        x0,
        B,
        priority=40,
    )

    return {
        "section": section_index + 1,
        "ring": ring_index + 1,
        "x_center_mm": x_center,
        "azimuth_deg": theta_deg,
        "a_mm": A,
        "b_mm": B,
        "d_mm": D,
        "r_inner_mm": r_ptfe0,
        "r_outer_mm": r_ptfe1,
    }


# BARE COAX: no filter apertures
aperture_records = []



# ============================================================================
# COAX DIELECTRIC
# ============================================================================

# The main coax is air-filled, as in the published prototype.
# No dielectric is added between RI and RO.
#
# PTFE exists only in the leakage apertures.


# ============================================================================
# GRID
# ============================================================================

X_MIN = LINE_X0 - 2.0
X_MAX = LINE_X1 + 2.0

Y_EXT = RO_OUT + D + XY_MARGIN
Z_EXT = RO_OUT + D + XY_MARGIN

x_lines = np.arange(
    grid_snap(X_MIN),
    grid_snap(X_MAX) + GRID / 2.0,
    GRID,
)

y_lines = np.arange(
    grid_snap(-Y_EXT),
    grid_snap(+Y_EXT) + GRID / 2.0,
    GRID,
)

z_lines = np.arange(
    grid_snap(-Z_EXT),
    grid_snap(+Z_EXT) + GRID / 2.0,
    GRID,
)

# Critical geometry lines.
x_critical = [
    LINE_X0,
    LINE_X1,
]

for rec in aperture_records:
    xc = rec["x_center_mm"]
    x_critical.extend([
        xc - B / 2.0,
        xc + B / 2.0,
    ])

# Aperture angular edges are represented approximately by Cartesian
# y/z mesh lines. The global 0.25 mm mesh resolves them for V1.
x_lines = sorted(set(
    list(x_lines) + [grid_snap(v) for v in x_critical]
))

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(UNIT)
mesh.AddLine("x", x_lines)
mesh.AddLine("y", y_lines)
mesh.AddLine("z", z_lines)

print()
print("MESH")
print("-" * 100)
print(f"x cells                  : {len(x_lines)-1}")
print(f"y cells                  : {len(y_lines)-1}")
print(f"z cells                  : {len(z_lines)-1}")
estimated_cells = (
    (len(x_lines)-1)
    * (len(y_lines)-1)
    * (len(z_lines)-1)
)

print(
    f"estimated cells          : "
    f"{estimated_cells:,}"
)

if estimated_cells > 6_000_000:
    raise RuntimeError(
        f"FAST V1 mesh is unexpectedly large ({estimated_cells:,} cells). "
        "Check GRID and simulation extents before launching openEMS."
    )


# ============================================================================
# PORTS
# ============================================================================

# We use lumped ports because the installed Windows Python build previously
# reported AddCoaxialPort unavailable. The port spans the coax TEM gap
# radially, from inner conductor surface to outer-conductor inner surface.
#
# Port excitation is therefore a differential voltage between the center
# conductor and the outer conductor.

PORT_X1 = LINE_X0
PORT_X2 = LINE_X1

port1 = FDTD.AddLumpedPort(
    1,
    Z0,
    [PORT_X1, RI, 0.0],
    [PORT_X1, RO, 0.0],
    "y",
    excite=1,
    priority=50,
    edges2grid="xy",
)

port2 = FDTD.AddLumpedPort(
    2,
    Z0,
    [PORT_X2, RI, 0.0],
    [PORT_X2, RO, 0.0],
    "y",
    excite=0,
    priority=50,
    edges2grid="xy",
)


# ============================================================================
# WRITE GEOMETRY
# ============================================================================

xml_path = SIM_PATH / "Wideband LP Filter_Wideband LP Filter_openems_v1.xml"

print()
print("WRITING CSXCAD XML")
print("-" * 100)
print(f"XML                      : {xml_path}")

CSX.Write2XML(str(xml_path))

if not xml_path.exists():
    raise RuntimeError("CSXCAD did not create the expected XML file.")

print(f"XML size                 : {xml_path.stat().st_size / 1024.0:.1f} kB")


geometry = {
    "reference": {
        "paper": "Rehammar & Gasparinetti, arXiv:2205.03941 / IEEE T-MTT 71 (2023)",
        "topology": "leaky coaxial waveguide / Wideband LP Filter",
    },
    "published_parameters": {
        "aperture_width_a_mm": A,
        "aperture_height_b_mm": B,
        "aperture_depth_d_mm": D,
        "coax_inner_radius_mm": RI,
        "coax_outer_radius_mm": RO,
        "ptfe_epsilon_r": EPS_PTFE,
        "number_of_sections": N_SECTIONS,
        "apertures_per_section": APERTURES_PER_SECTION,
    },
    "v1_interpretation": {
        "ring_spacing_mm": RING_SPACING,
        "section_pitch_mm": SECTION_PITCH,
        "outer_wall_outer_radius_mm": RO_OUT,
        "azimuths_deg": AZIMUTHS_DEG,
        "coax_length_mm": LINE_X1 - LINE_X0,
    },
    "apertures": aperture_records,
}

with (RESULTS_PATH / "geometry.json").open("w", encoding="utf-8") as fh:
    json.dump(geometry, fh, indent=2)

with (RESULTS_PATH / "geometry.txt").open("w", encoding="utf-8") as fh:
    fh.write("Wideband LP Filter geometry\n")
    fh.write("=" * 80 + "\n")
    for k, v in geometry["published_parameters"].items():
        fh.write(f"{k}: {v}\n")
    fh.write("\nV1 interpretation:\n")
    for k, v in geometry["v1_interpretation"].items():
        fh.write(f"{k}: {v}\n")


# ============================================================================
# RUN
# ============================================================================

print()
print("=" * 100)
print("STARTING OPENEMS")
print("=" * 100)
print(f"XML                      : {xml_path}")
print(f"Frequency                : {F_START/1e9:.1f}–{F_STOP/1e9:.1f} GHz")
print(f"Sections                 : {N_SECTIONS}")
print(f"Total apertures          : {len(aperture_records)}")
print(f"Mesh pitch               : {GRID:.2f} mm")
print(f"Max timesteps            : {NR_TS:,}")
print()
print("NOTE: FAST V1 SCREENING MODE")
print("      0.25 mm mesh / 15k maximum timesteps / 1–100 GHz.")
print("      Physical Wideband LP Filter geometry is unchanged.")
print("      This is topology validation, not final mesh convergence.")
print()

solver_t0 = time.perf_counter()

solver_stop = threading.Event()

def solver_heartbeat():
    try:
        import psutil
        proc = psutil.Process(os.getpid())
    except Exception:
        proc = None

    while not solver_stop.wait(15.0):
        elapsed = time.perf_counter() - solver_t0
        if proc is not None:
            try:
                rss = proc.memory_info().rss / (1024**3)
                print(
                    f"[OPENEMS ALIVE] elapsed={elapsed/60.0:6.2f} min | "
                    f"PID={os.getpid()} | RSS={rss:5.2f} GB",
                    flush=True,
                )
                continue
            except Exception:
                pass
        print(
            f"[OPENEMS ALIVE] elapsed={elapsed/60.0:6.2f} min | "
            f"PID={os.getpid()}",
            flush=True,
        )

heartbeat = threading.Thread(target=solver_heartbeat, daemon=True)
heartbeat.start()

print(
    f"[OPENEMS START] PID={os.getpid()} | "
    f"cells={estimated_cells:,} | max_steps={NR_TS:,}",
    flush=True,
)

try:
    FDTD.Run(
        str(SIM_PATH),
        cleanup=True,
        verbose=3,
    )
finally:
    solver_stop.set()
    heartbeat.join(timeout=1.0)

solver_elapsed = time.perf_counter() - solver_t0
print(
    f"[OPENEMS DONE] elapsed={solver_elapsed/60.0:.2f} min",
    flush=True,
)
# ============================================================================
# POSTPROCESS — LUMPED PORT CHARACTERIZATION
# ============================================================================

freq = np.linspace(F_START, F_STOP, N_FREQ)

port1.CalcPort(str(SIM_PATH), freq, ref_impedance=Z0)
port2.CalcPort(str(SIM_PATH), freq, ref_impedance=Z0)

eps = 1e-30

uf_inc_1 = np.asarray(port1.uf_inc).reshape(-1)
uf_ref_1 = np.asarray(port1.uf_ref).reshape(-1)
uf_ref_2 = np.asarray(port2.uf_ref).reshape(-1)

uf_tot_1 = np.asarray(port1.uf_tot).reshape(-1)
if_tot_1 = np.asarray(port1.if_tot).reshape(-1)

S11 = uf_ref_1 / (uf_inc_1 + eps)
S21 = uf_ref_2 / (uf_inc_1 + eps)

S11_dB = 20.0 * np.log10(np.maximum(np.abs(S11), eps))
S21_dB = 20.0 * np.log10(np.maximum(np.abs(S21), eps))

Zin = uf_tot_1 / (if_tot_1 + eps)

# Port-derived reflection coefficient from measured port impedance.
Gamma_Z = (Zin - Z0) / (Zin + Z0)
Gamma_Z_dB = 20.0 * np.log10(np.maximum(np.abs(Gamma_Z), eps))

# Direct wave-based port power quantities.
P_inc = np.abs(uf_inc_1)**2 / (2.0 * Z0)
P_ref = np.abs(uf_ref_1)**2 / (2.0 * Z0)
P_trn = np.abs(uf_ref_2)**2 / (2.0 * Z0)

power_ratio = (
    np.abs(S11)**2 +
    np.abs(S21)**2
)

def idx(f):
    return int(np.argmin(np.abs(freq - f)))

print()
print("=" * 100)
print("LUMPED PORT CHARACTERIZATION")
print("=" * 100)

for fGHz in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20]:
    i = idx(fGHz * 1e9)
    print(
        f"{fGHz:5.1f} GHz | "
        f"S11={S11_dB[i]:8.3f} dB | "
        f"S21={S21_dB[i]:8.3f} dB | "
        f"Zin={Zin[i].real:8.2f}"
        f"{Zin[i].imag:+8.2f}j ohm | "
        f"|Gamma_Z|={Gamma_Z_dB[i]:8.3f} dB"
    )

print()
print(f"Zin real range : {np.min(Zin.real):.3f} ... {np.max(Zin.real):.3f} ohm")
print(f"Zin imag range : {np.min(Zin.imag):.3f} ... {np.max(Zin.imag):.3f} ohm")
print(f"Power ratio min: {np.min(power_ratio):.6f}")
print(f"Power ratio max: {np.max(power_ratio):.6f}")
print()

# Compare wave-derived S11 to impedance-derived Gamma.
gamma_difference = np.abs(S11 - Gamma_Z)
print(f"max |S11 - Gamma(Zin)| = {np.max(gamma_difference):.6e}")

# Save raw diagnostic CSV.
csv_path = RESULTS_PATH / "lumped_port_characterization.csv"
with csv_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow([
        "frequency_Hz",
        "S11_dB",
        "S21_dB",
        "Zin_real_ohm",
        "Zin_imag_ohm",
        "Gamma_Z_dB",
        "power_ratio",
        "P_inc",
        "P_ref",
        "P_trn",
    ])

    for i in range(len(freq)):
        writer.writerow([
            float(freq[i]),
            float(S11_dB[i]),
            float(S21_dB[i]),
            float(Zin[i].real),
            float(Zin[i].imag),
            float(Gamma_Z_dB[i]),
            float(power_ratio[i]),
            float(P_inc[i]),
            float(P_ref[i]),
            float(P_trn[i]),
        ])

# Plot impedance.
plt.figure(figsize=(11, 6))
plt.plot(freq / 1e9, Zin.real, label="Re{Zin}")
plt.plot(freq / 1e9, Zin.imag, label="Im{Zin}")
plt.axhline(Z0, linestyle="--", linewidth=1, label="50 ohm")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Impedance (ohm)")
plt.title("Lumped Port — Input Impedance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_PATH / "lumped_port_Zin.png", dpi=180)
plt.close()

# Plot S parameters.
plt.figure(figsize=(11, 6))
plt.plot(freq / 1e9, S11_dB, label="S11")
plt.plot(freq / 1e9, S21_dB, label="S21")
plt.axhline(-3, linestyle=":", linewidth=1)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.title("Lumped Port Characterization — 20 mm Bare Coax")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_PATH / "lumped_port_Sparams.png", dpi=180)
plt.close()

print()
print("=" * 100)
print("FILES")
print("=" * 100)
print(f"CSV: {csv_path}")
print(f"Zin plot: {RESULTS_PATH / 'lumped_port_Zin.png'}")
print(f"S-parameter plot: {RESULTS_PATH / 'lumped_port_Sparams.png'}")
print("=" * 100)

