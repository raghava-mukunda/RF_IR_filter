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
    python lpf10_rectcoax_v4.py

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

OPT_FILE = Path(
    r"C:\Users\Raghava.m\Desktop\cryogenic_filter_design\lpf10_v2_optimized_lengths_mm.txt"
)
FSTART = 1e9
FSTOP = 40e9
NFREQ = 401

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

# Load the actual V2 differential-evolution result.
if not OPT_FILE.exists():
    raise FileNotFoundError(
        f"Optimized length file not found: {OPT_FILE}"
    )

HALF_LENGTHS_MM = np.loadtxt(OPT_FILE).reshape(-1)

if len(HALF_LENGTHS_MM) != 11:
    raise ValueError(
        f"Expected 11 optimized lengths, got {len(HALF_LENGTHS_MM)}"
    )

LENGTH_SOURCE = f"V2 optimized: {OPT_FILE}"

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
    AIR, MACOR, x, wo, ho, radius, depth, priority=30
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

    # MACOR plugs occupying the outer part of each HW.
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
    # IMPORTANT: cumulative floating-point addition can produce two
    # numerically distinct coordinates for the same physical plane.
    # That creates a zero-width cell and can break the FDTD operator.
    x_lines_raw = [x_start, 0.0]
    xx = 0.0
    for Lmm in SECTION_LENGTHS_MM:
        xx += Lmm
        x_lines_raw.append(xx)
    x_lines_raw.extend([total_length, x_end])

    # Quantize coordinates before deduplication.
    x_lines = sorted({
        round(float(v), 9)
        for v in x_lines_raw
    })

    # Add local refinement around every section boundary and HW center.
    mesh.AddLine("x", x_lines)

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
        -HW_DEPTH_MM - HO_H_MM / 2,
        -HO_H_MM / 2,
        -CENTER_THICKNESS_MM / 2,
        0.0,
        CENTER_THICKNESS_MM / 2,
        HO_H_MM / 2,
        HW_DEPTH_MM + HO_H_MM / 2,
    ]
    mesh.AddLine("z", sorted(set(z_lines)))

    # Fine global refinement. Increase/decrease after the first successful run.
    mesh.AddLine("x", np.arange(x_start, x_end + 0.50, 0.50))
    mesh.AddLine("y", np.arange(-2.0, 2.01, 0.50))
    mesh.AddLine("z", np.arange(-4.0, 4.01, 0.50))

    # FDTD.
    FDTD = openEMS(
        NrTS=20000,
        EndCriteria=3e-4,
    )
    FDTD.SetCSX(CSX)
    FDTD.SetBoundaryCond(["PML_6"] * 6)

    # Use the standard CFL timestep for the screening run.
    FDTD.SetTimeStepMethod(1)

    # Broadband Gaussian excitation.
    FDTD.SetGaussExcite(
        50e9,
        50e9,
    )

    # Lumped ports at the ends.
    # The exact port calibration will be refined after the first geometry run.
    # Rectangular-coax TEM port: E-field is vertical (z), from the
    # center conductor to the lower outer conductor. The previous V4
    # runner used a y-directed port, which is parallel to the strip and
    # resulted in zero S21.
    z_sig = CENTER_THICKNESS_MM / 2.0
    z_gnd = -feed_ho / 2.0 + 0.05

    p1 = FDTD.AddLumpedPort(
        1,
        Z0,
        [x_start + 1.0, 0.0, z_gnd],
        [x_start + 1.0, 0.0, z_sig],
        "z",
        excite=1,
        priority=50,
        edges2grid="xz",
    )

    p2 = FDTD.AddLumpedPort(
        2,
        Z0,
        [x_end - 1.0, 0.0, z_gnd],
        [x_end - 1.0, 0.0, z_sig],
        "z",
        excite=0,
        priority=50,
        edges2grid="xz",
    )

    Path(sim_path).mkdir(parents=True, exist_ok=True)
    CSX.Write2XML(str(Path(sim_path) / "model.xml"))

    return FDTD, p1, p2, total_length

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 90)
    print("10 GHz RECTANGULAR-COAX LPF + HW ABSORBERS — V4 SCREENING 40 GHz")
    print("=" * 90)

    if not OPT_FILE.exists():
        raise FileNotFoundError(
            f"Optimized length file not found: {OPT_FILE}"
        )

    print(f"Length source       : {OPT_FILE}")
    print(f"Sections            : {len(SECTION_LENGTHS_MM)}")
    print(f"Total filter length : {SECTION_LENGTHS_MM.sum():.4f} mm")
    print(f"Frequency range     : {FSTART/1e9:.1f}–{FSTOP/1e9:.1f} GHz (screening)")
    print(f"Frequency points    : {NFREQ}")

    sim_dir = Path(__file__).resolve().parent / "sim_lpf10_v4_screening40"
    sim_dir.mkdir(parents=True, exist_ok=True)

    print("\nBuilding openEMS geometry...")
    FDTD, port1, port2, total = build_model(str(sim_dir))

    print(f"Model directory     : {sim_dir}")
    print(f"Total filter length : {total:.4f} mm")
    print("Port orientation    : z (signal conductor -> outer conductor)")

    print("\nRunning openEMS...")
    print("This is the expensive step.")
    print("Do not close this terminal until the solver finishes.")

    FDTD.Run(
        str(sim_dir),
        verbose=3,
        numThreads=1,
        disable_dumps=True,
    )

    print("\nSimulation finished.")
    print("Calculating S-parameters...")

    freq = np.linspace(FSTART, FSTOP, NFREQ)

    port1.CalcPort(
        str(sim_dir),
        freq,
        ref_impedance=Z0,
    )
    port2.CalcPort(
        str(sim_dir),
        freq,
        ref_impedance=Z0,
    )

    uf_inc = port1.uf_inc
    uf_ref = port1.uf_ref
    uf_tot_2 = port2.uf_tot

    S11 = uf_ref / uf_inc
    S21 = uf_tot_2 / uf_inc

    S11_dB = 20.0 * np.log10(np.maximum(np.abs(S11), 1e-15))
    S21_dB = 20.0 * np.log10(np.maximum(np.abs(S21), 1e-15))

    def at(fGHz, arr):
        return float(np.interp(fGHz, freq / 1e9, arr))

    print("\n" + "=" * 90)
    print("FULL-WAVE RESULTS")
    print("=" * 90)

    for fg in [1, 5, 7, 8, 9, 9.5, 10, 11, 12, 15, 20, 30, 35, 40]:
        print(
            f"{fg:6.1f} GHz : "
            f"S21 = {at(fg, S21_dB):8.3f} dB   "
            f"S11 = {at(fg, S11_dB):8.3f} dB"
        )

    pb = freq <= 8.0e9
    sb20 = (freq >= 20.0e9) & (freq <= 40.0e9)
    sb50 = (freq >= 35.0e9) & (freq <= 40.0e9)

    cutoff_idx = np.where(S21_dB <= -3.0)[0]
    cutoff = freq[cutoff_idx[0]] / 1e9 if cutoff_idx.size else np.nan

    print("\nMetrics:")
    print(f"3-dB cutoff              : {cutoff:.4f} GHz")
    print(f"0–8 GHz max insertion loss: {np.max(-S21_dB[pb]):.4f} dB")
    print(f"0–8 GHz worst S11        : {np.max(S11_dB[pb]):.4f} dB")
    print(f"20–100 GHz worst S21     : {np.max(S21_dB[sb20]):.4f} dB")
    print(f"35–40 GHz worst S21     : {np.max(S21_dB[sb50]):.4f} dB")

    # Locate the strongest out-of-band transmission windows.
    for lo, hi in [(10, 20), (20, 30), (30, 40)]:
        m = (freq >= lo*1e9) & (freq <= hi*1e9)
        k = np.argmax(S21_dB[m])
        ff = freq[m][k] / 1e9
        vv = S21_dB[m][k]
        print(f"{lo:2d}–{hi:3d} GHz max S21 : {vv:8.3f} dB @ {ff:8.3f} GHz")

    np.savez(
        sim_dir / "sparams_v4_screening40.npz",
        freq_Hz=freq,
        S11=S11,
        S21=S21,
    )

    plt.figure(figsize=(12, 7))
    plt.plot(freq / 1e9, S21_dB, label="S21")
    plt.plot(freq / 1e9, S11_dB, label="S11")
    plt.axvline(10.0, linestyle="--", label="10 GHz target")
    plt.axhline(-3.0, linestyle=":", label="-3 dB")
    plt.axhline(-60.0, linestyle=":", label="-60 dB")
    plt.xlim(1, 100)
    plt.ylim(-100, 2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("10 GHz rectangular-coax LPF + hollow-waveguide absorbers — FULL WAVE")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(sim_dir / "lpf10_v4_screening40_sparameters.png", dpi=180)
    plt.show()

    print("\nSaved:")
    print(sim_dir / "sparams_v4_screening40.npz")
    print(sim_dir / "lpf10_v4_screening40_sparameters.png")
