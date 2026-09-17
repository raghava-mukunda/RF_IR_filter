# -*- coding: utf-8 -*-
"""
RIGOROUS OPENEMS RECONSTRUCTION
21-section stepped-impedance + circular hollow-waveguide cryogenic LPF.

Design basis:
    Andersson et al., "Co-designed reflective and leaky-waveguide
    low-pass filter for superconducting circuits", arXiv:2508.02475 (2025).

This is a research reconstruction, NOT an exact copy of any proprietary
commercial filter.

The 11 optimized section lengths are fixed exactly to the user's optimized
values. The physical sequence is symmetric:
    1,2,3,...,10,11,10,...,3,2,1

Important:
    - Circular HWs are retained. No rectangular-slot surrogate.
    - Section-aware X mesh prevents cumulative floating-point zero cells.
    - Cylinder bounding coordinates are explicitly inserted into the mesh.
    - A controlled Cartesian mesh is used.
    - The model is written to XML before the simulation.
    - Forward and reverse runs provide S11/S21/S12/S22.
    - The full-wave sweep is 1-70 GHz in this run; the paper's optimization
      itself was performed over 15-40 GHz, while the HWs are intended to
      control higher-frequency re-transmission.
    - The HW outer termination is represented by a lossy MACOR/absorber
      reconstruction. Final absorber foam properties must be replaced by
      measured/vendor data in HFSS before fabrication.

Run:
    python final_openems_rigorous_cylindrical_hw.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 0. NATIVE OPENEMS
# ============================================================================

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


# ============================================================================
# 1. OUTPUT / RUN CONTROL
# ============================================================================

PROJECT_DIR = Path(
    r"C:\Users\Raghava.m\Desktop\cryogenic_filter_design"
).resolve()

OUT_DIR = PROJECT_DIR / "FINAL_OPENEMS_RIGOROUS"
FWD_DIR = OUT_DIR / "forward"
REV_DIR = OUT_DIR / "reverse"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FWD_DIR.mkdir(parents=True, exist_ok=True)
REV_DIR.mkdir(parents=True, exist_ok=True)

XML_FILE = OUT_DIR / "final_filter.xml"
S2P_FILE = OUT_DIR / "final_filter.s2p"
S_PARAM_PNG = OUT_DIR / "final_s_parameters.png"
IDEAL_PNG = OUT_DIR / "ideal_response.png"
MESH_PNG = OUT_DIR / "mesh_summary.png"

# Full validation range. The 2025 paper's optimization target is 15-40 GHz;
# the extra range is included to inspect higher-frequency retransmission.
FSTART = 1e9
FSTOP = 70e9
NFREQ = 691
FREQ = np.linspace(FSTART, FSTOP, NFREQ)

Z0 = 50.0
UNIT = 1e-3

# Maximum Cartesian cell target.
# 0.10 mm resolves the 0.20 mm numerical PEC wall with two cells and the
# 0.50 mm copper strip with five cells.
DX_MAX_MM = 0.10
DY_MAX_MM = 0.10
DZ_MAX_MM = 0.10

NR_TS = 120000
END_CRITERIA = 1e-5

GAUSS_F0 = 35e9
GAUSS_FC = 35e9

BOUNDARY = ["PML_8"] * 6

RUN_REVERSE = True
OPEN_3D_VIEWER = True

# One-thread execution is deliberate: it is stable on the user's current
# native Windows build and avoids multiplying peak memory.
NTHREADS = 1


# ============================================================================
# 2. EXACT OPTIMIZED LENGTHS
# ============================================================================

HALF_LENGTHS_MM = np.array([
    1.549074106459433109,
    2.999395271442262700,
    2.658444742966871388,
    3.233620473026813436,
    2.625457253425803383,
    2.877294426583933529,
    1.854230813896529328,
    1.590614543218636046,
    1.654978191688255551,
    1.641163741887338157,
    1.395959595206971660,
], dtype=float)

SECTION_LENGTHS_MM = np.concatenate(
    [HALF_LENGTHS_MM, HALF_LENGTHS_MM[-2::-1]]
)

SECTION_TYPES = np.array(
    ["H" if i % 2 == 0 else "L" for i in range(21)]
)

TOTAL_LENGTH_MM = float(np.sum(SECTION_LENGTHS_MM))

assert len(SECTION_LENGTHS_MM) == 21


# ============================================================================
# 3. PAPER-BASED GEOMETRY PARAMETERS
# ============================================================================

# Values reported in arXiv:2508.02475.
Z_HIGH = 85.9
Z_LOW = 22.1

HI_H_MM = 0.8
HI_L_MM = 0.3

WO_H_MM = 3.0
WO_L_MM = 1.5

HO_H_MM = 1.45
HO_L_MM = 0.50

HW_R_SMALL_MM = 1.15
HW_R_LARGE_MM = 1.50
HW_DEPTH_MM = 2.50

MACOR_EPS = 5.64
MACOR_TAND = 2.5e-3

# Paper reports 0.5 mm copper sheet for center conductor.
CENTER_THICKNESS_MM = 0.50

# Numerical reconstruction of the machined outer-conductor wall.
WALL_MM = 0.20

# Feed length.
FEED_MM = 8.0

# HW diameter choices used in the reported prototype:
# sections 6, 8 and 10 were enlarged to 3 mm in the final prototype.
#
# Here the ten high-Z sections are indexed 0...9.
# Use small radius by default, large radius on 3/5/7/8/9 equivalent
# positions from the earlier reconstruction, while explicitly preserving
# the paper's final enlarged apertures for sections 6, 8, 10.
#
# We keep the earlier research reconstruction's mixed-radius sequence so
# this run remains comparable to the previously optimized model.
HW_RADIUS_BY_H_INDEX = [
    HW_R_SMALL_MM,
    HW_R_SMALL_MM,
    HW_R_LARGE_MM,
    HW_R_SMALL_MM,
    HW_R_LARGE_MM,
    HW_R_SMALL_MM,
    HW_R_LARGE_MM,
    HW_R_SMALL_MM,
    HW_R_LARGE_MM,
    HW_R_LARGE_MM,
]

# Numerical absorbing termination surrogate.
# The paper uses MACOR in the HW and carbon-loaded polyethylene foam farther
# outward. We explicitly model MACOR and use a lossy outer absorber region.
ABS_EPS = 10.0
ABS_TAND = 0.30

EPS0 = 8.8541878128e-12
OMEGA_REF = 2*np.pi*35e9
ABS_KAPPA = OMEGA_REF * EPS0 * ABS_EPS * ABS_TAND

# MACOR conductivity representation.
MACOR_KAPPA = (
    OMEGA_REF
    * EPS0
    * MACOR_EPS
    * MACOR_TAND
)


# ============================================================================
# 4. TIMING / PROGRESS
# ============================================================================

T0 = time.perf_counter()


def stage(name):
    elapsed = time.perf_counter() - T0
    print()
    print("=" * 78)
    print(f"[{elapsed:8.1f} s] {name}")
    print("=" * 78)


def progress_bar(label, fraction, width=42):
    fraction = max(0.0, min(1.0, float(fraction)))
    n = int(round(width * fraction))
    bar = "#" * n + "-" * (width - n)
    print(
        f"\r{label:<22} [{bar}] {100*fraction:6.1f}%",
        end="",
        flush=True,
    )
    if fraction >= 1.0:
        print()


# ============================================================================
# 5. IDEAL DISTRIBUTED MODEL
# ============================================================================

def section_abcd(Z, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 1j*Z*s],
        [1j*s/Z, c],
    ], dtype=complex)


def beta(f, eps_eff=2.1):
    return (
        2*np.pi*f*np.sqrt(eps_eff)
        / 299792458.0
    )


def ideal_abcd(f, eps_eff=2.1):
    b = beta(f, eps_eff)
    T = np.eye(2, dtype=complex)

    for Lmm, typ in zip(
        SECTION_LENGTHS_MM,
        SECTION_TYPES,
    ):
        Z = Z_HIGH if typ == "H" else Z_LOW
        T = T @ section_abcd(
            Z,
            b*Lmm*UNIT,
        )

    return T


def abcd_sparams(T):
    A, B = T[0, 0], T[0, 1]
    C, D = T[1, 0], T[1, 1]

    den = A + B/Z0 + C*Z0 + D

    return (
        (A + B/Z0 - C*Z0 - D)/den,
        2.0/den,
    )


def ideal_response():
    s11 = np.zeros(NFREQ, dtype=complex)
    s21 = np.zeros(NFREQ, dtype=complex)

    for k, f in enumerate(FREQ):
        s11[k], s21[k] = abcd_sparams(
            ideal_abcd(f)
        )

    return s11, s21


# ============================================================================
# 6. CSXCAD GEOMETRY HELPERS
# ============================================================================

def add_box(mat, p1, p2, priority):
    mat.AddBox(
        np.asarray(p1, dtype=float),
        np.asarray(p2, dtype=float),
        priority=priority,
    )


def build_outer_frame(PEC, x0, x1, wo, ho):
    wall = WALL_MM

    # Bottom
    add_box(
        PEC,
        [x0, -wo/2, -ho/2],
        [x1,  wo/2, -ho/2 + wall],
        10,
    )

    # Top
    add_box(
        PEC,
        [x0, -wo/2, ho/2 - wall],
        [x1,  wo/2, ho/2],
        10,
    )

    # Left
    add_box(
        PEC,
        [x0, -wo/2, -ho/2 + wall],
        [x1, -wo/2 + wall, ho/2 - wall],
        10,
    )

    # Right
    add_box(
        PEC,
        [x0, wo/2 - wall, -ho/2 + wall],
        [x1, wo/2, ho/2 - wall],
        10,
    )


def add_center_conductor(PEC, x0, x1, wo):
    add_box(
        PEC,
        [x0, -wo/2 + WALL_MM, -CENTER_THICKNESS_MM/2],
        [x1,  wo/2 - WALL_MM,  CENTER_THICKNESS_MM/2],
        30,
    )


def add_hw_pair(
    AIR,
    MACOR,
    ABS,
    x,
    ho,
    radius,
):
    """
    Circular HW pair, axis along z.

    Geometry:
        air cavity starts just inside the outer-conductor wall;
        MACOR fills the HW;
        lossy absorber occupies the outer termination.

    This follows the physical architecture described in the 2025 paper:
    circular HWs are orthogonal to the main rectangular-coax axis, MACOR
    is inserted in the HWs, and an absorptive termination is placed farther
    outward.
    """

    y0 = 0.0

    z_top = ho/2
    z_bot = -ho/2

    # Extend air slightly through the PEC wall so priority 20 cuts the
    # aperture through the metal.
    overlap = 0.05

    # ------------------------------------------------------------
    # Top air bore
    # ------------------------------------------------------------
    AIR.AddCylinder(
        [x, y0, z_top - overlap],
        [x, y0, z_top + HW_DEPTH_MM],
        radius,
        priority=20,
    )

    # ------------------------------------------------------------
    # Bottom air bore
    # ------------------------------------------------------------
    AIR.AddCylinder(
        [x, y0, z_bot + overlap],
        [x, y0, z_bot - HW_DEPTH_MM],
        radius,
        priority=20,
    )

    # ------------------------------------------------------------
    # MACOR inner section
    # ------------------------------------------------------------
    macor_start = 0.10
    macor_end = HW_DEPTH_MM * 0.62

    MACOR.AddCylinder(
        [x, y0, z_top + macor_start],
        [x, y0, z_top + macor_end],
        radius*0.98,
        priority=30,
    )

    MACOR.AddCylinder(
        [x, y0, z_bot - macor_start],
        [x, y0, z_bot - macor_end],
        radius*0.98,
        priority=30,
    )

    # ------------------------------------------------------------
    # Outer lossy termination
    # ------------------------------------------------------------
    ABS.AddCylinder(
        [x, y0, z_top + macor_end],
        [x, y0, z_top + HW_DEPTH_MM],
        radius*0.98,
        priority=40,
    )

    ABS.AddCylinder(
        [x, y0, z_bot - macor_end],
        [x, y0, z_bot - HW_DEPTH_MM],
        radius*0.98,
        priority=40,
    )


# ============================================================================
# 7. ROBUST MESH
# ============================================================================

def quantized_unique(values, quantum=1e-4):
    """
    Remove floating-point duplicates without materially moving any geometry.

    1e-5 mm = 10 nm, much smaller than the 0.1 mm numerical mesh target.
    """

    a = np.asarray(values, dtype=float)

    if a.size == 0:
        return a

    a = np.round(a/quantum)*quantum
    a = np.unique(a)
    a.sort()

    return a


def segment_grid(a, b, max_step):
    """
    Generate a mesh whose endpoints are EXACTLY a and b and whose largest
    interval is <= max_step.
    """

    length = float(b-a)

    if length <= 0:
        return np.array([a], dtype=float)

    n = max(
        1,
        int(np.ceil(length/max_step)),
    )

    return np.linspace(
        a,
        b,
        n+1,
    )


def piecewise_grid(boundaries, max_step):
    """Build a strictly monotonic mesh from quantized boundaries."""
    q = 1e-4  # mm
    b = np.asarray(boundaries, dtype=float)
    b = np.round(b / q) * q
    b = np.unique(b)
    b.sort()

    out = [float(b[0])]
    for a, c in zip(b[:-1], b[1:]):
        length = float(c - a)
        if length <= q:
            continue
        n = max(1, int(np.ceil(length / max_step)))
        seg = np.round(np.linspace(a, c, n + 1) / q) * q
        for x in seg[1:]:
            x = float(x)
            if x - out[-1] >= q:
                out.append(x)

    return np.asarray(out, dtype=float)


def build_model(
    excitation_port,
    sim_dir,
    write_xml,
):

    stage(
        f"BUILDING FULL 3-D MODEL — PORT {excitation_port}"
    )

    sim_dir = Path(sim_dir)
    sim_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    CSX = ContinuousStructure()

    PEC = CSX.AddMetal("PEC")

    AIR = CSX.AddMaterial(
        "AIR",
        epsilon=1.0,
    )

    MACOR = CSX.AddMaterial(
        "MACOR",
        epsilon=MACOR_EPS,
        kappa=MACOR_KAPPA,
    )

    ABS = CSX.AddMaterial(
        "ABSORBER",
        epsilon=ABS_EPS,
        kappa=ABS_KAPPA,
    )

    # ------------------------------------------------------------
    # Overall x coordinates
    # ------------------------------------------------------------

    x_start = -FEED_MM
    x_end = TOTAL_LENGTH_MM + FEED_MM

    section_boundaries = [
        x_start,
        0.0,
    ]

    xx = 0.0

    for L in SECTION_LENGTHS_MM:
        xx += L
        section_boundaries.append(xx)

    section_boundaries += [
        TOTAL_LENGTH_MM,
        x_end,
    ]

    section_boundaries = quantized_unique(
        section_boundaries,
        quantum=1e-4,
    )

    # ------------------------------------------------------------
    # Build stepped coax
    # ------------------------------------------------------------

    x = 0.0
    high_index = 0

    hw_data = []

    for i, (Lmm, typ) in enumerate(
        zip(
            SECTION_LENGTHS_MM,
            SECTION_TYPES,
        )
    ):

        x0 = x
        x1 = x + Lmm

        if typ == "H":
            wo = WO_H_MM
            ho = HO_H_MM
        else:
            wo = WO_L_MM
            ho = HO_L_MM

        build_outer_frame(
            PEC,
            x0,
            x1,
            wo,
            ho,
        )

        add_center_conductor(
            PEC,
            x0,
            x1,
            wo,
        )

        if typ == "H":

            radius = HW_RADIUS_BY_H_INDEX[
                min(
                    high_index,
                    len(HW_RADIUS_BY_H_INDEX)-1,
                )
            ]

            xc = 0.5*(x0+x1)

            add_hw_pair(
                AIR,
                MACOR,
                ABS,
                xc,
                ho,
                radius,
            )

            hw_data.append(
                (
                    xc,
                    radius,
                    ho,
                )
            )

            high_index += 1

        x = x1

        progress_bar(
            "Geometry",
            (i+1)/len(SECTION_LENGTHS_MM),
        )

    # ------------------------------------------------------------
    # Feeds
    # ------------------------------------------------------------

    build_outer_frame(
        PEC,
        x_start,
        0.0,
        WO_L_MM,
        HO_L_MM,
    )

    build_outer_frame(
        PEC,
        TOTAL_LENGTH_MM,
        x_end,
        WO_L_MM,
        HO_L_MM,
    )

    add_center_conductor(
        PEC,
        x_start,
        0.0,
        WO_L_MM,
    )

    add_center_conductor(
        PEC,
        TOTAL_LENGTH_MM,
        x_end,
        WO_L_MM,
    )

    # ------------------------------------------------------------
    # X mesh
    # ------------------------------------------------------------
    # Longitudinal mesh comes ONLY from physical section boundaries.
    # Do not inject HW radial coordinates into X.
    # ------------------------------------------------------------

    x_mesh = piecewise_grid(
        section_boundaries,
        DX_MAX_MM,
    )

    x_mesh = quantized_unique(
        x_mesh,
        quantum=1e-4,
    )

    # ------------------------------------------------------------
    # Y mesh
    # ------------------------------------------------------------

    y_boundaries = [
        -2.0,
        -WO_H_MM/2,
        -WO_L_MM/2,
        -CENTER_THICKNESS_MM/2,
        0.0,
        CENTER_THICKNESS_MM/2,
        WO_L_MM/2,
        WO_H_MM/2,
        2.0,
    ]

    # Cylinder circular cross-sections require y = +/- r lines.
    for _, radius, _ in hw_data:
        y_boundaries += [
            -radius,
            0.0,
            radius,
        ]

    y_boundaries = quantized_unique(
        y_boundaries,
        quantum=1e-4,
    )

    y_mesh = piecewise_grid(
        y_boundaries,
        DY_MAX_MM,
    )

    # ------------------------------------------------------------
    # Z mesh
    # ------------------------------------------------------------

    z_boundaries = [
        -4.0,
        -HW_DEPTH_MM-HO_H_MM/2,
        -HO_H_MM/2,
        -HO_L_MM/2,
        -CENTER_THICKNESS_MM/2,
        0.0,
        CENTER_THICKNESS_MM/2,
        HO_L_MM/2,
        HO_H_MM/2,
        HW_DEPTH_MM+HO_H_MM/2,
        4.0,
    ]

    z_mesh = piecewise_grid(
        z_boundaries,
        DZ_MAX_MM,
    )

    # ------------------------------------------------------------
    # Mesh checks
    # ------------------------------------------------------------

    dx = np.diff(x_mesh)
    dy = np.diff(y_mesh)
    dz = np.diff(z_mesh)

    min_dx = float(np.min(dx))
    min_dy = float(np.min(dy))
    min_dz = float(np.min(dz))

    max_dx = float(np.max(dx))
    max_dy = float(np.max(dy))
    max_dz = float(np.max(dz))

    print()
    print("MESH QUALITY")
    print("-"*78)
    print(
        f"X lines={len(x_mesh):,} "
        f"Y lines={len(y_mesh):,} "
        f"Z lines={len(z_mesh):,}"
    )

    print(
        f"X cell: min={min_dx:.6f} mm "
        f"max={max_dx:.6f} mm"
    )

    print(
        f"Y cell: min={min_dy:.6f} mm "
        f"max={max_dy:.6f} mm"
    )

    print(
        f"Z cell: min={min_dz:.6f} mm "
        f"max={max_dz:.6f} mm"
    )

    # Hard safety limit. A sub-10-micron cell is not justified by the
    # geometry here and indicates another floating-point/mesh bug.
    if min_dx < 0.02 or min_dy < 0.02 or min_dz < 0.02:
        print()
        print("FATAL MESH CHECK")
        print("-" * 78)
        print(f"dx_min = {min_dx:.12g} mm")
        print(f"dy_min = {min_dy:.12g} mm")
        print(f"dz_min = {min_dz:.12g} mm")

        if min_dx < 0.02:
            j = int(np.argmin(dx))
            print(
                f"X offending interval: "
                f"{x_mesh[j]:.12f} -> {x_mesh[j+1]:.12f} mm"
            )
        if min_dy < 0.02:
            j = int(np.argmin(dy))
            print(
                f"Y offending interval: "
                f"{y_mesh[j]:.12f} -> {y_mesh[j+1]:.12f} mm"
            )
        if min_dz < 0.02:
            j = int(np.argmin(dz))
            print(
                f"Z offending interval: "
                f"{z_mesh[j]:.12f} -> {z_mesh[j+1]:.12f} mm"
            )

        raise RuntimeError(
            "Pathological mesh cell detected in cleaned mesh."
        )


    # ------------------------------------------------------------
    # Install mesh
    # ------------------------------------------------------------

    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    mesh.AddLine(
        "x",
        x_mesh,
    )

    mesh.AddLine(
        "y",
        y_mesh,
    )

    mesh.AddLine(
        "z",
        z_mesh,
    )

    # ------------------------------------------------------------
    # FDTD
    # ------------------------------------------------------------

    stage("CREATING FDTD OPERATOR")

    FDTD = openEMS(
        NrTS=NR_TS,
        EndCriteria=END_CRITERIA,
    )

    FDTD.SetCSX(CSX)

    FDTD.SetBoundaryCond(
        BOUNDARY
    )

    FDTD.SetGaussExcite(
        GAUSS_F0,
        GAUSS_FC,
    )

    # ------------------------------------------------------------
    # Ports
    # ------------------------------------------------------------

    port_x1 = x_start + 1.0
    port_x2 = x_end - 1.0

    z_ground = -HO_L_MM/2 + WALL_MM
    z_signal = CENTER_THICKNESS_MM/2

    p1 = FDTD.AddLumpedPort(
        1,
        Z0,
        [
            port_x1,
            0.0,
            z_ground,
        ],
        [
            port_x1,
            0.0,
            z_signal,
        ],
        "z",
        excite=1 if excitation_port == 1 else 0,
        priority=50,
        edges2grid="xz",
    )

    p2 = FDTD.AddLumpedPort(
        2,
        Z0,
        [
            port_x2,
            0.0,
            z_ground,
        ],
        [
            port_x2,
            0.0,
            z_signal,
        ],
        "z",
        excite=1 if excitation_port == 2 else 0,
        priority=50,
        edges2grid="xz",
    )

    # ------------------------------------------------------------
    # Save geometry BEFORE running
    # ------------------------------------------------------------

    if write_xml:

        stage("WRITING CSXCAD XML")

        CSX.Write2XML(
            str(XML_FILE)
        )

        if not XML_FILE.is_file():
            raise RuntimeError(
                "XML was not created:\n"
                + str(XML_FILE)
            )

        print(
            f"XML: {XML_FILE}"
        )

        print(
            f"Size: "
            f"{XML_FILE.stat().st_size:,} bytes"
        )

    return FDTD, p1, p2


# ============================================================================
# 9. APPCSXCAD
# ============================================================================

def find_appcsxcad():

    candidates = [
        Path(OPENEMS_ROOT)/"AppCSXCAD.exe",
        Path(OPENEMS_ROOT).parent/"AppCSXCAD.exe",
    ]

    for p in candidates:
        if p.is_file():
            return p

    root = Path(OPENEMS_ROOT)

    if root.exists():
        for p in root.parent.rglob(
            "AppCSXCAD.exe"
        ):
            return p

    return None


def launch_viewer():

    if not OPEN_3D_VIEWER:
        return

    if not XML_FILE.is_file():
        print(
            "3-D viewer skipped: XML does not exist."
        )
        return

    app = find_appcsxcad()

    if app is None:
        print(
            "AppCSXCAD.exe not found."
        )
        return

    print()
    print(
        f"Launching AppCSXCAD: {XML_FILE}"
    )

    subprocess.Popen(
        [
            str(app),
            str(XML_FILE),
        ],
        cwd=str(app.parent),
    )

    time.sleep(2)


# ============================================================================
# 10. RUN FDTD
# ============================================================================

def run_one(
    excitation_port,
    sim_dir,
    write_xml=False,
):

    FDTD, p1, p2 = build_model(
        excitation_port,
        sim_dir,
        write_xml,
    )

    stage(
        f"RUNNING FDTD — PORT {excitation_port}"
    )

    print(
        f"Frequency sweep: "
        f"{FSTART/1e9:.1f}–{FSTOP/1e9:.1f} GHz"
    )

    print(
        f"Timesteps: {NR_TS:,}"
    )

    print(
        "The native openEMS solver will print its timestep/iteration "
        "progress below."
    )

    FDTD.Run(
        str(sim_dir),
        verbose=2,
        numThreads=NTHREADS,
    )

    stage(
        f"CALCULATING PORT S-PARAMETERS — PORT {excitation_port}"
    )

    p1.CalcPort(
        str(sim_dir),
        FREQ,
        ref_impedance=Z0,
    )

    p2.CalcPort(
        str(sim_dir),
        FREQ,
        ref_impedance=Z0,
    )

    if excitation_port == 1:

        S11 = p1.uf_ref/p1.uf_inc
        S21 = p2.uf_ref/p1.uf_inc

        return S11, S21

    S12 = p1.uf_ref/p2.uf_inc
    S22 = p2.uf_ref/p2.uf_inc

    return S12, S22


# ============================================================================
# 11. TOUCHSTONE
# ============================================================================

def write_touchstone(
    filename,
    S11,
    S21,
    S12,
    S22,
):

    with open(
        filename,
        "w",
        encoding="ascii",
    ) as fh:

        fh.write(
            "! 21-section rectangular-coax + circular-HW LPF\n"
        )
        fh.write(
            "! openEMS full-wave reconstruction\n"
        )
        fh.write(
            "# GHz S DB R 50\n"
        )

        for k, f in enumerate(FREQ):

            row = [f/1e9]

            for s in [
                S11[k],
                S21[k],
                S12[k],
                S22[k],
            ]:

                row += [
                    20*np.log10(
                        max(abs(s), 1e-15)
                    ),
                    np.angle(
                        s,
                        deg=True,
                    ),
                ]

            fh.write(
                " ".join(
                    f"{v:.12g}"
                    for v in row
                )
                + "\n"
            )


# ============================================================================
# 12. MAIN
# ============================================================================

if __name__ == "__main__":

    stage("INITIALIZING FINAL RIGOROUS RUN")

    print(
        "21-section stepped-impedance rectangular-coax LPF"
    )

    print(
        "with circular hollow-waveguide leakage/absorber structures"
    )

    print()
    print(
        f"Total length : {TOTAL_LENGTH_MM:.9f} mm"
    )
    print(
        f"Frequency    : {FSTART/1e9:.1f}–{FSTOP/1e9:.1f} GHz"
    )
    print(
        f"Mesh target  : <= {DX_MAX_MM:.2f} mm"
    )
    print(
        f"Timesteps    : {NR_TS:,}"
    )

    print()
    print("OPTIMIZED 11 LENGTHS")
    print("-"*78)

    for i, L in enumerate(
        HALF_LENGTHS_MM,
        1,
    ):
        print(
            f"L{i:02d} = {L:.12f} mm"
        )

    print()
    print("21 PHYSICAL SECTIONS")
    print("-"*78)

    for i, (L, typ) in enumerate(
        zip(
            SECTION_LENGTHS_MM,
            SECTION_TYPES,
        ),
        1,
    ):
        print(
            f"{i:02d}  {typ}  {L:.9f} mm"
        )

    # ------------------------------------------------------------
    # Ideal reference
    # ------------------------------------------------------------

    stage("CALCULATING IDEAL DISTRIBUTED REFERENCE")

    ideal_s11, ideal_s21 = ideal_response()

    ideal_s11_db = 20*np.log10(
        np.maximum(abs(ideal_s11), 1e-15)
    )

    ideal_s21_db = 20*np.log10(
        np.maximum(abs(ideal_s21), 1e-15)
    )

    for fg in [
        5, 8, 9, 10, 11, 12,
        15, 20, 30, 40, 50, 60, 70
    ]:

        k = int(
            np.argmin(
                abs(FREQ/1e9-fg)
            )
        )

        print(
            f"{fg:5.1f} GHz | "
            f"S21={ideal_s21_db[k]:9.3f} dB | "
            f"S11={ideal_s11_db[k]:9.3f} dB"
        )

    plt.figure(
        figsize=(11, 6.5)
    )

    plt.plot(
        FREQ/1e9,
        ideal_s21_db,
        label="Ideal S21",
    )

    plt.plot(
        FREQ/1e9,
        ideal_s11_db,
        label="Ideal S11",
    )

    plt.axvline(
        10,
        linestyle="--",
        label="10 GHz target",
    )

    plt.axhline(
        -60,
        linestyle=":",
        label="-60 dB",
    )

    plt.xlim(
        1,
        70,
    )

    plt.ylim(
        -100,
        5,
    )

    plt.xlabel(
        "Frequency (GHz)"
    )

    plt.ylabel(
        "Magnitude (dB)"
    )

    plt.title(
        "Optimized 21-section LPF — ideal distributed reference"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(
        IDEAL_PNG,
        dpi=180,
    )
    plt.close()

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    S11, S21 = run_one(
        1,
        FWD_DIR,
        write_xml=True,
    )

    launch_viewer()

    # ------------------------------------------------------------
    # Reverse
    # ------------------------------------------------------------

    if RUN_REVERSE:

        S12, S22 = run_one(
            2,
            REV_DIR,
            write_xml=False,
        )

    else:

        S12 = S21.copy()
        S22 = np.full_like(
            S11,
            np.nan + 1j*np.nan,
        )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    stage("POST-PROCESSING")

    S11_db = 20*np.log10(
        np.maximum(abs(S11), 1e-15)
    )

    S21_db = 20*np.log10(
        np.maximum(abs(S21), 1e-15)
    )

    S12_db = 20*np.log10(
        np.maximum(abs(S12), 1e-15)
    )

    S22_db = 20*np.log10(
        np.maximum(abs(S22), 1e-15)
    )

    print()
    print("="*78)
    print("FULL-WAVE RESULTS")
    print("="*78)

    for fg in [
        5, 8, 9, 10, 11, 12,
        15, 20, 30, 40, 50, 60, 70
    ]:

        k = int(
            np.argmin(
                abs(FREQ/1e9-fg)
            )
        )

        print(
            f"{fg:5.1f} GHz | "
            f"S11={S11_db[k]:9.3f} dB | "
            f"S21={S21_db[k]:9.3f} dB | "
            f"S12={S12_db[k]:9.3f} dB | "
            f"S22={S22_db[k]:9.3f} dB"
        )

    # ------------------------------------------------------------
    # Stopband metrics
    # ------------------------------------------------------------

    passband = (
        (FREQ >= 1e9)
        & (FREQ <= 8e9)
    )

    transition = (
        (FREQ >= 8e9)
        & (FREQ <= 20e9)
    )

    stopband = (
        (FREQ >= 20e9)
        & (FREQ <= 70e9)
    )

    print()
    print("="*78)
    print("KEY METRICS")
    print("="*78)

    print(
        f"Worst S21, 1-8 GHz      : "
        f"{np.max(S21_db[passband]):.3f} dB"
    )

    print(
        f"Highest S21, 8-20 GHz   : "
        f"{np.max(S21_db[transition]):.3f} dB"
    )

    print(
        f"Highest S21, 20-70 GHz  : "
        f"{np.max(S21_db[stopband]):.3f} dB"
    )

    print(
        f"Minimum attenuation, 20-70 GHz: "
        f"{-np.max(S21_db[stopband]):.3f} dB"
    )

    # ------------------------------------------------------------
    # Plot all S parameters
    # ------------------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        FREQ/1e9,
        S11_db,
        label="S11",
    )

    plt.plot(
        FREQ/1e9,
        S21_db,
        label="S21",
    )

    plt.plot(
        FREQ/1e9,
        S12_db,
        label="S12",
    )

    plt.plot(
        FREQ/1e9,
        S22_db,
        label="S22",
    )

    plt.axvline(
        10,
        linestyle="--",
        label="10 GHz target",
    )

    plt.axhline(
        -60,
        linestyle=":",
        label="-60 dB",
    )

    plt.xlim(
        1,
        70,
    )

    plt.ylim(
        -100,
        5,
    )

    plt.xlabel(
        "Frequency (GHz)"
    )

    plt.ylabel(
        "Magnitude (dB)"
    )

    plt.title(
        "Final 21-section cryogenic LPF — openEMS full-wave"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend(
        ncol=2,
    )

    plt.tight_layout()

    plt.savefig(
        S_PARAM_PNG,
        dpi=200,
    )

    plt.show()

    # ------------------------------------------------------------
    # Touchstone
    # ------------------------------------------------------------

    write_touchstone(
        S2P_FILE,
        S11,
        S21,
        S12,
        S22,
    )

    stage("RUN COMPLETE")

    print(
        f"XML       : {XML_FILE}"
    )

    print(
        f"S2P       : {S2P_FILE}"
    )

    print(
        f"S plot    : {S_PARAM_PNG}"
    )

    print(
        f"Ideal plot: {IDEAL_PNG}"
    )

    print()
    print(
        "Do NOT treat the openEMS result as fabrication approval yet."
    )
    print(
        "Use this result to select the geometry, then independently"
    )
    print(
        "validate the same geometry/material stack in HFSS."
    )
