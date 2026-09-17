# -*- coding: utf-8 -*-
"""
V1 — 10 GHz rectangular-coax stepped-impedance cryogenic LPF
based on the architecture/dimensions reported in arXiv:2508.02475.

This is NOT a reproduction of undisclosed proprietary geometry.
It is a research reconstruction using the published topology,
published impedances, published HW dimensions/materials, and
published optimized section-length sequence scaled from 13.5 GHz
to a 10 GHz target.

Stage 1:
    1) synthesize/check the 21-section stepped-impedance response
    2) build the corresponding openEMS 3-D geometry
    3) include hollow-waveguide leakage ports above/below the
       high-impedance sections

Run first with:
    python lpf10_rectcoax_v3.py

Dependencies:
    numpy, scipy, matplotlib, openEMS/CSXCAD local installation
"""

import os
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Native openEMS setup
# ---------------------------------------------------------------------
OPENEMS_ROOT = os.environ.get(
    "OPENEMS_ROOT",
    r"C:\Users\Raghava.m\Desktop\openEMS_x64_v0.0.36-93-g7b9cd51_msvc\openEMS",
)
if os.name == "nt":
    try:
        os.add_dll_directory(OPENEMS_ROOT)
    except Exception:
        pass

from CSXCAD import ContinuousStructure
from openEMS import openEMS

# ---------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------
UNIT = 1e-3
Z0 = 50.0
FC = 10e9
FSTART = 1e9
FSTOP = 100e9
NFREQ = 1001

# ---------------------------------------------------------------------
# Published 2025 architecture
# ---------------------------------------------------------------------
# Published:
#   Z_low  = 22.1 ohm
#   Z_high = 85.9 ohm
#   inner conductor height = 0.8 / 0.3 mm
#   outer conductor width  = 1.5 / 3.0 mm
#   outer conductor height = 0.5 / 1.45 mm
#   HW radii = 1.15 / 1.50 mm
#   HW depth = 2.5 mm
#   MACOR eps_r = 5.64
#
# The paper reports 21 physical stepped-impedance sections.
# The 11 listed lengths are symmetric around the central section.
#
# Published 13.5-GHz prototype lengths:
PUBLISHED_LENGTHS_MM = np.array([
    1.06, 2.30, 2.05, 2.54, 1.91,
    3.16, 1.91, 4.28, 1.77, 4.63, 0.80
], dtype=float)

PUBLISHED_FC = 13.5e9

# First-order frequency scaling starting point.
SCALE = PUBLISHED_FC / FC
HALF_LENGTHS_MM = PUBLISHED_LENGTHS_MM * SCALE

# Prefer the actual V2 differential-evolution result if the optimizer
# has already been run. This keeps the EM model tied to the optimized
# distributed filter rather than the simple frequency-scaled starting point.
OPT_FILE = Path("lpf10_v2_optimized_lengths_mm.txt")

if OPT_FILE.exists():
    HALF_LENGTHS_MM = np.loadtxt(OPT_FILE).reshape(-1)
    if len(HALF_LENGTHS_MM) != 11:
        raise ValueError(
            f"{OPT_FILE} must contain 11 independent section lengths; "
            f"found {len(HALF_LENGTHS_MM)}"
        )
    LENGTH_SOURCE = "V2 differential-evolution optimized lengths"
else:
    LENGTH_SOURCE = "published lengths frequency-scaled from 13.5 to 10 GHz"

# 21 physical sections:
# 1,2,...,11,...,2,1
SECTION_LENGTHS_MM = np.concatenate(
    [HALF_LENGTHS_MM, HALF_LENGTHS_MM[-2::-1]]
)

assert len(SECTION_LENGTHS_MM) == 21

# Alternating H/L sections, with central section 11 low-Z.
SECTION_TYPES = np.array([
    "H" if i % 2 == 0 else "L"
    for i in range(21)
])

# Published physical cross-sections.
Z_HIGH = 85.9
Z_LOW = 22.1

HI_H_MM = 0.8
HI_L_MM = 0.3

WO_H_MM = 3.0
WO_L_MM = 1.5

HO_H_MM = 1.45
HO_L_MM = 0.50

CENTER_THICKNESS_MM = 0.50

# HW radii and MACOR.
HW_R_SMALL_MM = 1.15
HW_R_LARGE_MM = 1.50
HW_DEPTH_MM = 2.50
MACOR_EPS = 5.64
MACOR_TAND = 2.5e-3

# ---------------------------------------------------------------------
# Numerical absorbing termination for the hollow waveguides.
#
# The paper uses carbon-loaded polyethylene foam outside the MACOR.
# Exact foam parameters are not published in the paper, so this is a
# deliberately tunable EM surrogate, NOT a claimed material match.
# The first screening value below is strongly lossy at microwave/mm-wave
# frequencies and lets us test the filter architecture.
# ---------------------------------------------------------------------
ABS_EPS = 10.0
ABS_TAND = 0.30
ABS_LENGTH_MM = 8.0

# Use both radii, as in the paper.
# Larger apertures were experimentally used on selected sections.
HW_RADIUS_BY_H_INDEX = [
    HW_R_SMALL_MM, HW_R_SMALL_MM,
    HW_R_LARGE_MM, HW_R_SMALL_MM,
    HW_R_LARGE_MM, HW_R_SMALL_MM,
    HW_R_LARGE_MM, HW_R_SMALL_MM,
    HW_R_LARGE_MM, HW_R_LARGE_MM,
]

# ---------------------------------------------------------------------
# Ideal stepped-impedance model
# ---------------------------------------------------------------------
def section_abcd(Z, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 1j * Z * s],
                     [1j * s / Z, c]], dtype=complex)

def beta(f, eps_eff=2.1):
    return 2.0 * np.pi * f * np.sqrt(eps_eff) / 299792458.0

def ideal_abcd(f, eps_eff=2.1):
    """
    First-order distributed stepped-line model.

    The published paper uses the physical section lengths as the
    optimization variables and extracts actual inductive-section
    S-parameters from FEM. Here we first use the published 13.5-GHz
    geometry scaled to 10 GHz as the starting point.
    """
    b = beta(f, eps_eff)
    T = np.eye(2, dtype=complex)

    for Lmm, typ in zip(SECTION_LENGTHS_MM, SECTION_TYPES):
        Z = Z_HIGH if typ == "H" else Z_LOW
        th = b * Lmm * UNIT
        T = T @ section_abcd(Z, th)

    return T

def abcd_sparams(T, z0=50.0):
    A, B = T[0, 0], T[0, 1]
    C, D = T[1, 0], T[1, 1]
    den = A + B / z0 + C * z0 + D
    s11 = (A + B / z0 - C * z0 - D) / den
    s21 = 2.0 / den
    return s11, s21

def ideal_response():
    f = np.linspace(FSTART, FSTOP, NFREQ)
    s11 = np.zeros_like(f, dtype=complex)
    s21 = np.zeros_like(f, dtype=complex)

    for k, fk in enumerate(f):
        s11[k], s21[k] = abcd_sparams(ideal_abcd(fk))

    return f, s11, s21

# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def add_box(mat, p1, p2, priority):
    mat.AddBox(
        np.asarray(p1, dtype=float),
        np.asarray(p2, dtype=float),
        priority=priority,
    )

def build_outer_frame(PEC, x0, x1, wo, ho, priority=10):
    """
    Four-wall rectangular outer conductor.
    Coordinates:
        x = propagation
        y = transverse width
        z = transverse height
    """
    wall = 0.20

    # bottom
    add_box(
        PEC,
        [x0, -wo / 2, -ho / 2],
        [x1,  wo / 2, -ho / 2 + wall],
        priority,
    )
    # top
    add_box(
        PEC,
        [x0, -wo / 2, ho / 2 - wall],
        [x1,  wo / 2, ho / 2],
        priority,
    )
    # left
    add_box(
        PEC,
        [x0, -wo / 2, -ho / 2 + wall],
        [x1, -wo / 2 + wall, ho / 2 - wall],
        priority,
    )
    # right
    add_box(
        PEC,
        [x0, wo / 2 - wall, -ho / 2 + wall],
        [x1, wo / 2, ho / 2 - wall],
        priority,
    )

def add_hollow_waveguide_pair(
    AIR, MACOR, ABSORBER, x, wo, ho, radius, depth, priority=30
):
    """
    Two circular HW apertures, one above and one below the center strip.

    The cylinder axis is z. The opening is represented as an air/MACOR
    cylinder extending outward from the coax cavity through the outer wall.
    """
    # Put the aperture center laterally near the center of the high-Z wall.
    y0 = 0.0

    # Top and bottom HW centers.
    z_top = ho / 2.0
    z_bot = -ho / 2.0

    # Start slightly inside the outer wall and extend outward.
    z1_top = z_top - 0.15
    z2_top = z_top + depth
    z1_bot = z_bot + 0.15
    z2_bot = z_bot - depth

    # Air openings.
    AIR.AddCylinder(
        [x, y0, z1_top],
        [x, y0, z2_top],
        radius,
        priority=priority,
    )
    AIR.AddCylinder(
        [x, y0, z1_bot],
        [x, y0, z2_bot],
        radius,
        priority=priority,
    )

    # MACOR plugs occupy the published HW depth.
    plug_start_top = z_top + 0.10
    plug_end_top = z_top + depth
    plug_start_bot = z_bot - 0.10
    plug_end_bot = z_bot - depth

    MACOR.AddCylinder(
        [x, y0, plug_start_top],
        [x, y0, plug_end_top],
        radius * 0.98,
        priority=priority + 1,
    )
    MACOR.AddCylinder(
        [x, y0, plug_start_bot],
        [x, y0, plug_end_bot],
        radius * 0.98,
        priority=priority + 1,
    )

    # Lossy termination beyond the MACOR. The outer end is left open to
    # the PML so radiated power is removed rather than re-reflected.
    abs_top_0 = plug_end_top
    abs_top_1 = plug_end_top + ABS_LENGTH_MM

    abs_bot_0 = plug_end_bot
    abs_bot_1 = plug_end_bot - ABS_LENGTH_MM

    ABSORBER.AddCylinder(
        [x, y0, abs_top_0],
        [x, y0, abs_top_1],
        radius * 0.98,
        priority=priority + 2,
    )
    ABSORBER.AddCylinder(
        [x, y0, abs_bot_0],
        [x, y0, abs_bot_1],
        radius * 0.98,
        priority=priority + 2,
    )

# ---------------------------------------------------------------------
# Full 3-D model
# ---------------------------------------------------------------------
def build_model(sim_path):
    CSX = ContinuousStructure()

    PEC = CSX.AddMetal("PEC")
    AIR = CSX.AddMaterial("AIR", epsilon=1.0)
    MACOR = CSX.AddMaterial(
        "MACOR",
        epsilon=MACOR_EPS,
        kappa=2.0 * np.pi * 50e9 * 8.8541878128e-12 * MACOR_EPS * MACOR_TAND,
    )
    ABSORBER = CSX.AddMaterial(
        "ABSORBER",
        epsilon=ABS_EPS,
        kappa=2.0 * np.pi * 50e9 * 8.8541878128e-12 * ABS_EPS * ABS_TAND,
    )

    # Global envelope.
    total_length = np.sum(SECTION_LENGTHS_MM)
    feed = 8.0
    x_start = -feed
    x_end = total_length + feed

    # Build section-by-section rectangular coax.
    x = 0.0
    high_index = 0

    section_centers = []

    for i, (Lmm, typ) in enumerate(
        zip(SECTION_LENGTHS_MM, SECTION_TYPES)
    ):
        x0 = x
        x1 = x + Lmm

        if typ == "H":
            wo = WO_H_MM
            ho = HO_H_MM
            hi = HI_H_MM
        else:
            wo = WO_L_MM
            ho = HO_L_MM
            hi = HI_L_MM

        build_outer_frame(PEC, x0, x1, wo, ho)

        # TEM cavity / dielectric-free first-pass model.
        # Center conductor is centered vertically.
        add_box(
            PEC,
            [x0, -wo / 2 + 0.20, -CENTER_THICKNESS_MM / 2],
            [x1,  wo / 2 - 0.20,  CENTER_THICKNESS_MM / 2],
            priority=30,
        )

        section_centers.append((0.5 * (x0 + x1), typ))

        if typ == "H":
            r = HW_RADIUS_BY_H_INDEX[
                min(high_index, len(HW_RADIUS_BY_H_INDEX) - 1)
            ]
            add_hollow_waveguide_pair(
                AIR,
                MACOR,
                ABSORBER,
                0.5 * (x0 + x1),
                wo,
                ho,
                r,
                HW_DEPTH_MM,
            )
            high_index += 1

        x = x1

    # Straight input/output 50-ohm-ish sections.
    # These are deliberately kept separate from the filter optimization.
    feed_wo = WO_L_MM
    feed_ho = HO_L_MM

    build_outer_frame(PEC, x_start, 0.0, feed_wo, feed_ho)
    build_outer_frame(PEC, total_length, x_end, feed_wo, feed_ho)

    add_box(
        PEC,
        [x_start, -feed_wo / 2 + 0.20, -CENTER_THICKNESS_MM / 2],
        [0.0, feed_wo / 2 - 0.20, CENTER_THICKNESS_MM / 2],
        priority=30,
    )
    add_box(
        PEC,
        [total_length, -feed_wo / 2 + 0.20, -CENTER_THICKNESS_MM / 2],
        [x_end, feed_wo / 2 - 0.20, CENTER_THICKNESS_MM / 2],
        priority=30,
    )

    # Mesh.
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    # Longitudinal section boundaries.
    x_lines = [x_start, 0.0]
    xx = 0.0
    for Lmm in SECTION_LENGTHS_MM:
        xx += Lmm
        x_lines.append(xx)
    x_lines.extend([total_length, x_end])

    # Add local refinement around every section boundary and HW center.
    mesh.AddLine("x", sorted(set(x_lines)))

    y_lines = [
        -WO_H_MM / 2,
        -WO_L_MM / 2,
        -CENTER_THICKNESS_MM / 2,
        0.0,
        CENTER_THICKNESS_MM / 2,
        WO_L_MM / 2,
        WO_H_MM / 2,
    ]
    mesh.AddLine("y", sorted(set(y_lines)))

    z_lines = [
        -HW_DEPTH_MM - ABS_LENGTH_MM - HO_H_MM / 2,
        -HW_DEPTH_MM - HO_H_MM / 2,
        -HO_H_MM / 2,
        -CENTER_THICKNESS_MM / 2,
        0.0,
        CENTER_THICKNESS_MM / 2,
        HO_H_MM / 2,
        HW_DEPTH_MM + HO_H_MM / 2,
        HW_DEPTH_MM + ABS_LENGTH_MM + HO_H_MM / 2,
    ]
    mesh.AddLine("z", sorted(set(z_lines)))

    # Fine global refinement. Increase/decrease after the first successful run.
    mesh.AddLine("x", np.arange(x_start, x_end + 0.25, 0.25))
    mesh.AddLine("y", np.arange(-2.0, 2.01, 0.25))
    z_extent = HW_DEPTH_MM + ABS_LENGTH_MM + 2.0
    mesh.AddLine("z", np.arange(-z_extent, z_extent + 0.25, 0.25))

    # FDTD.
    FDTD = openEMS(
        NrTS=50000,
        EndCriteria=1e-4,
    )
    FDTD.SetCSX(CSX)
    FDTD.SetBoundaryCond(["PML_8"] * 6)

    # Broadband Gaussian excitation.
    FDTD.SetGaussExcite(
        50e9,
        50e9,
    )

    # Lumped ports at the ends.
    # The exact port calibration will be refined after the first geometry run.
    p1 = FDTD.AddLumpedPort(
        1,
        Z0,
        [x_start + 1.0, -feed_wo / 2 + 0.20, 0.0],
        [x_start + 1.0,  feed_wo / 2 - 0.20, 0.0],
        "y",
        excite=1,
        priority=50,
        edges2grid="xy",
    )

    p2 = FDTD.AddLumpedPort(
        2,
        Z0,
        [x_end - 1.0, -feed_wo / 2 + 0.20, 0.0],
        [x_end - 1.0,  feed_wo / 2 - 0.20, 0.0],
        "y",
        excite=0,
        priority=50,
        edges2grid="xy",
    )

    Path(sim_path).mkdir(parents=True, exist_ok=True)
    CSX.Write2XML(str(Path(sim_path) / "model.xml"))

    return FDTD, p1, p2, total_length

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("10 GHz RECTANGULAR-COAX LPF + HOLLOW-WAVEGUIDE ABSORBERS — V3")
    print("=" * 80)
    print(f"Target cutoff       : {FC/1e9:.3f} GHz")
    print(f"Published cutoff    : {PUBLISHED_FC/1e9:.3f} GHz")
    print(f"Length source       : {LENGTH_SOURCE}")
    print(f"Sections            : {len(SECTION_LENGTHS_MM)}")
    print(f"Z_high / Z_low     : {Z_HIGH:.1f} / {Z_LOW:.1f} ohm")
    print(f"Total filter length : {SECTION_LENGTHS_MM.sum():.3f} mm")
    print(f"HW absorber length  : {ABS_LENGTH_MM:.3f} mm (numerical surrogate)")
    print()
    print("Section lengths [mm]:")
    for i, (L, typ) in enumerate(zip(SECTION_LENGTHS_MM, SECTION_TYPES), 1):
        print(f"  {i:02d}  {typ}  {L:.4f}")
    print()

    # Ideal distributed model.
    f, s11, s21 = ideal_response()
    s11_db = 20 * np.log10(np.maximum(np.abs(s11), 1e-15))
    s21_db = 20 * np.log10(np.maximum(np.abs(s21), 1e-15))

    print("IDEAL DISTRIBUTED STARTING MODEL")
    for fg in [5, 8, 9, 10, 11, 12, 15, 20, 30, 50, 70, 100]:
        k = np.argmin(np.abs(f / 1e9 - fg))
        print(
            f"S21 @ {fg:5.1f} GHz : {s21_db[k]:8.3f} dB   "
            f"S11 : {s11_db[k]:8.3f} dB"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(f / 1e9, s21_db, label="S21 ideal stepped-line")
    plt.plot(f / 1e9, s11_db, label="S11 ideal stepped-line")
    plt.axvline(10, linestyle="--", label="10 GHz target")
    plt.axhline(-3, linestyle=":", label="-3 dB")
    plt.axhline(-60, linestyle=":", label="-60 dB")
    plt.xlim(0, 100)
    plt.ylim(-100, 2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("10 GHz rectangular-coax LPF + hollow-waveguide absorbers")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lpf10_v3_ideal_response.png", dpi=180)
    plt.show()

    # Build the EM model.
    sim_dir = Path("sim_lpf10_v3")
    sim_dir.mkdir(exist_ok=True)

    print()
    print("Building openEMS geometry...")
    _, _, _, total = build_model(str(sim_dir))
    print(f"openEMS model written to: {sim_dir}")
    print(f"Total filter length: {total:.3f} mm")
    print("NEXT: run the full-wave model and inspect S11/S21.")
    print("NOTE: ABSORBER material is a tunable numerical surrogate;")
    print("      replace with measured absorber properties before fabrication.")
