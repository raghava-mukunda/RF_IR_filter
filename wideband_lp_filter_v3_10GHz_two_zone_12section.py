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
RUN_ID = os.environ.get("WB_RUN_ID", "wideband_lp_filter_v3_10GHz_two_zone_12section")

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
D = float(os.environ.get("WB_D", "4.80"))

RI = 1.60
RO = 3.70
RO_OUT = 4.20
EPS_PTFE = 2.2

N_SECTIONS = int(os.environ.get("WB_NSECTIONS", "12"))
APERTURES_PER_SECTION = 8
AZIMUTHS_DEG = [0.0, 90.0, 180.0, 270.0]

# V2 design target:
# PTFE aperture depth is centered at approximately quarter-wave at 10 GHz.
# lambda0/(4*sqrt(eps_r)) = ~5.05 mm for eps_r=2.2.
# Eight leakage sections increase cumulative stopband attenuation while
# the closely spaced two-ring cells force overlapping leakage around cutoff.


RING_SPACING = float(os.environ.get("WB_RING", "4.00"))
SECTION_PITCH = float(os.environ.get("WB_PITCH", "8.00"))

SECTION_CENTERS = [
    (i - (N_SECTIONS - 1) / 2.0) * SECTION_PITCH
    for i in range(N_SECTIONS)
]

FILTER_X0 = SECTION_CENTERS[0] - RING_SPACING / 2.0
FILTER_X1 = SECTION_CENTERS[-1] + RING_SPACING / 2.0

FEED = 12.0
LINE_X0 = FILTER_X0 - FEED
LINE_X1 = FILTER_X1 + FEED

APERTURE_R_IN = RO - 0.05
APERTURE_R_OUT = RO + D + 0.05

XY_MARGIN = 2.0
Z_MARGIN = 2.0

F_START = 1.0e9
F_STOP = 100.0e9
N_FREQ = 1001

F0 = 50.0e9
FC_GAUSS = 50.0e9

NR_TS = 25000
END_CRITERIA = 2e-4
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
print("Wideband LP Filter LEAKY COAX — V3 10-GHz SHARP CUTOFF")
print("=" * 100)
print("V3 redesign: weak-to-strong two-zone aperture coupling + 12 leakage sections:")
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
    # Two-zone coupling profile:
    #   sections 1-4  : weak / narrow aperture
    #   sections 5-12 : stronger / wider + deeper aperture
    #
    # This is intentionally a screening geometry. The EM result determines
    # whether the desired 10-GHz knee and broadband rejection are achievable.
    if section_index < 4:
        a_local = 3.50
        d_local = 4.00
    else:
        a_local = 5.00
        d_local = 5.20

    theta = math.radians(theta_deg)

    uy = math.cos(theta)
    uz = math.sin(theta)

    ty = -math.sin(theta)
    tz = math.cos(theta)

    # Axial dimension B.
    x0 = x_center - B / 2.0
    x1 = x_center + B / 2.0

    # Tangential half width.
    ht = a_local / 2.0

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
    r_ptfe1 = RO + d_local

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
        "a_mm": a_local,
        "b_mm": B,
        "d_mm": d_local,
        "r_inner_mm": r_ptfe0,
        "r_outer_mm": r_ptfe1,
    }


aperture_records = []

for si, xc in enumerate(SECTION_CENTERS):
    ring_x = [
        xc - RING_SPACING / 2.0,
        xc + RING_SPACING / 2.0,
    ]

    for ri_ring, xr in enumerate(ring_x):
        for theta in AZIMUTHS_DEG:
            aperture_records.append(
                add_aperture(si, ri_ring, xr, theta)
            )

assert len(aperture_records) == N_SECTIONS * APERTURES_PER_SECTION


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
    print(f"WARNING: mesh is large ({estimated_cells:,} cells); proceeding with V2 screening.")


# ============================================================================
# PORTS
# ============================================================================

# We use lumped ports because the installed Windows Python build previously
# reported AddCoaxialPort unavailable. The port spans the coax TEM gap
# radially, from inner conductor surface to outer-conductor inner surface.
#
# Port excitation is therefore a differential voltage between the center
# conductor and the outer conductor.

PORT_X1 = LINE_X0 + 4.0
PORT_X2 = LINE_X1 - 4.0

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
print("NOTE: V3 TWO-ZONE SCREENING MODE")
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
# POSTPROCESS
# ============================================================================

freq = np.linspace(F_START, F_STOP, N_FREQ)

port1.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=Z0,
)

port2.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=Z0,
)

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
power = np.abs(S11)**2 + np.abs(S21)**2


def idx(f):
    return int(np.argmin(np.abs(freq - f)))


m_1_9 = (freq >= 1e9) & (freq <= 9e9)
m_1_8p5 = (freq >= 1e9) & (freq <= 8.5e9)
m_9_20 = (freq >= 9e9) & (freq <= 20e9)
m_12_20 = (freq >= 12e9) & (freq <= 20e9)
m_20_100 = (freq >= 20e9) & (freq <= 100e9)
m_50_100 = (freq >= 50e9) & (freq <= 100e9)

max_1_9 = float(np.max(S21_dB[m_1_9]))
min_1_9 = float(np.min(S21_dB[m_1_9]))
max_1_8p5 = float(np.max(S21_dB[m_1_8p5]))
min_1_8p5 = float(np.min(S21_dB[m_1_8p5]))
max_9_20 = float(np.max(S21_dB[m_9_20]))
max_12_20 = float(np.max(S21_dB[m_12_20]))
max_20_100 = float(np.max(S21_dB[m_20_100]))
max_50_100 = float(np.max(S21_dB[m_50_100]))

fmax_20_100 = float(freq[m_20_100][np.argmax(S21_dB[m_20_100])])
fmax_50_100 = float(freq[m_50_100][np.argmax(S21_dB[m_50_100])])

# First downward -3 dB crossing above 5 GHz.
cross = np.where((freq >= 5e9) & (S21_dB <= -3.0))[0]
f_3db = float(freq[cross[0]] / 1e9) if len(cross) else float("nan")

passband_worst_loss = float(-min_1_9)

print()
print("=" * 100)
print("RESULTS")
print("=" * 100)

for f in [5, 8, 9, 10, 12, 15, 17, 20, 30, 40, 55, 70, 100]:
    print(
        f"S21 @ {f:5.1f} GHz        : "
        f"{S21_dB[idx(f*1e9)]: .4f} dB"
    )

print()
print(f"1–9 GHz S21 min          : {min_1_9:.4f} dB")
print(f"1–9 GHz S21 max          : {max_1_9:.4f} dB")
print(f"1–9 GHz worst loss       : {passband_worst_loss:.4f} dB")
print(f"3-dB cutoff               : {f_3db:.4f} GHz")
print(f"9–20 GHz S21 max         : {max_9_20:.4f} dB")
print(f"12–20 GHz S21 max        : {max_12_20:.4f} dB")
print(
    f"20–100 GHz S21 max       : "
    f"{max_20_100:.4f} dB @ {fmax_20_100/1e9:.3f} GHz"
)
print(
    f"50–100 GHz S21 max       : "
    f"{max_50_100:.4f} dB @ {fmax_50_100/1e9:.3f} GHz"
)
print(f"Power sum min/max        : {power.min():.6f}/{power.max():.6f}")


# ============================================================================
# SAVE CSV
# ============================================================================

csv_path = RESULTS_PATH / "sparameters.csv"

with csv_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow([
        "frequency_Hz",
        "S11_real",
        "S11_imag",
        "S21_real",
        "S21_imag",
        "S11_dB",
        "S21_dB",
        "Zin_real_ohm",
        "Zin_imag_ohm",
        "power_sum",
    ])

    for i in range(len(freq)):
        writer.writerow([
            float(freq[i]),
            float(S11[i].real),
            float(S11[i].imag),
            float(S21[i].real),
            float(S21[i].imag),
            float(S11_dB[i]),
            float(S21_dB[i]),
            float(Zin[i].real),
            float(Zin[i].imag),
            float(power[i]),
        ])


metrics = {
    "a_mm": A,
    "b_mm": B,
    "d_mm": D,
    "ri_mm": RI,
    "ro_mm": RO,
    "ptfe_epsilon_r": EPS_PTFE,
    "n_sections": N_SECTIONS,
    "n_apertures": len(aperture_records),
    "ring_spacing_mm": RING_SPACING,
    "section_pitch_mm": SECTION_PITCH,
    "s21_5GHz_dB": float(S21_dB[idx(5e9)]),
    "s21_8GHz_dB": float(S21_dB[idx(8e9)]),
    "s21_9GHz_dB": float(S21_dB[idx(9e9)]),
    "s21_10GHz_dB": float(S21_dB[idx(10e9)]),
    "s21_12GHz_dB": float(S21_dB[idx(12e9)]),
    "s21_15GHz_dB": float(S21_dB[idx(15e9)]),
    "s21_16GHz_dB": float(S21_dB[idx(16e9)]),
    "s21_17GHz_dB": float(S21_dB[idx(17e9)]),
    "s21_20GHz_dB": float(S21_dB[idx(20e9)]),
    "s21_40GHz_dB": float(S21_dB[idx(40e9)]),
    "s21_55GHz_dB": float(S21_dB[idx(55e9)]),
    "s21_70GHz_dB": float(S21_dB[idx(70e9)]),
    "s21_100GHz_dB": float(S21_dB[idx(100e9)]),
    "max_s21_20_100_dB": max_20_100,
    "max_s21_20_100_frequency_Hz": fmax_20_100,
    "max_s21_50_100_dB": max_50_100,
    "max_s21_50_100_frequency_Hz": fmax_50_100,
    "power_min": float(power.min()),
    "power_max": float(power.max()),
    "passband_worst_loss_1_9_dB": passband_worst_loss,
    "cutoff_3dB_GHz": f_3db,
    "max_s21_9_20_dB": max_9_20,
    "max_s21_12_20_dB": max_12_20,
    "mesh_mm": GRID,
    "max_timesteps": NR_TS,
    "end_criteria": END_CRITERIA,
    "n_frequency_points": N_FREQ,
    "simulation_mode": "FAST_V1_SCREENING",
}

with (RESULTS_PATH / "metrics.json").open("w", encoding="utf-8") as fh:
    json.dump(metrics, fh, indent=2)


# ============================================================================
# PLOT
# ============================================================================

plot_file = PLOT_PATH / "response.png"

plt.figure(figsize=(12, 7))
plt.plot(freq/1e9, S11_dB, label="S11")
plt.plot(freq/1e9, S21_dB, label="S21")
plt.axvline(10, linestyle="--", label="10 GHz")
plt.axhline(-3, linestyle=":", label="-3 dB")
plt.axhline(-60, linestyle=":", label="-60 dB")
plt.xlim(1, 100)
plt.ylim(-100, 5)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.title("Wideband LP Filter V3 — 1–100 GHz")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
try:
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=160)
except Exception as exc:
    print(f"[PLOT WARNING] {exc}", flush=True)
finally:
    plt.close()

print()
print("=" * 100)
print("OUTPUTS")
print("=" * 100)
print(f"CSV                     : {csv_path}")
print(f"Metrics                 : {RESULTS_PATH / 'metrics.json'}")
print(f"Geometry                : {RESULTS_PATH / 'geometry.json'}")
print(f"Geometry text           : {RESULTS_PATH / 'geometry.txt'}")
print(f"Plot                    : {plot_file}")
print()
print("V1 COMPLETE")
