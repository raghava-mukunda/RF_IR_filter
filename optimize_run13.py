from __future__ import annotations

import os
import sys
import csv
import shutil
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# OPENEMS
# ============================================================

OPENEMS_ROOT = Path(
    r"C:\Users\Raghava.m\Desktop\openEMS_x64_v0.0.36-93-g7b9cd51_msvc\openEMS"
)

os.add_dll_directory(str(OPENEMS_ROOT))
sys.path.insert(0, str(OPENEMS_ROOT / "Python"))

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ============================================================
# OUTPUT
# ============================================================

ROOT = (
    Path(r"C:\Users\Raghava.m\Desktop\cryogenic_filter_design")
    / "LPF10_FINE_OPT_30GHz"
)

ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIXED RUN-13 ARCHITECTURE
# ============================================================

TOTAL_LENGTH = 65.3
TOTAL_WIDTH = 5.0
TOTAL_HEIGHT = 4.6

WALL = 0.2

CENTER_WIDTH = 1.2

INPUT_LAUNCH = 8.0
OUTPUT_LAUNCH = 8.0

FILTER_LENGTH = (
    TOTAL_LENGTH
    - INPUT_LAUNCH
    - OUTPUT_LAUNCH
)


# ============================================================
# RUN-13 HEIGHTS
# ============================================================

H_THIN = 0.28671928
H_THICK = 0.76349166


# ============================================================
# RUN-13 STARTING GEOMETRY
# ============================================================

RUN13 = np.array([
    4.07365691,
    7.53238661,
    8.07101208,
    9.94588880
])


# ============================================================
# SEARCH RANGE
# ============================================================

L_MIN = np.array([
    3.50,
    6.50,
    7.00,
    8.50
])

L_MAX = np.array([
    5.00,
    8.50,
    9.20,
    11.00
])


# ============================================================
# OPTIMIZATION
# ============================================================

N_CANDIDATES = 50

RNG_SEED = 20260925

rng = np.random.default_rng(
    RNG_SEED
)


# ============================================================
# FREQUENCY
#
# IMPORTANT:
# Optimization stops at 30 GHz.
#
# We will NOT operate the FDTD solver at its numerical
# Nyquist boundary.
# ============================================================

FREQ_MIN = 0.5e9
FREQ_MAX = 30.0e9

N_FREQ = 601

FREQS = np.linspace(
    FREQ_MIN,
    FREQ_MAX,
    N_FREQ
)


# ============================================================
# FDTD EXCITATION
#
# Broad enough to cover the useful 0.5–30 GHz band,
# but not centered at the numerical ceiling.
# ============================================================

F0 = 15e9
FC = 15e9


# ============================================================
# FDTD RUNTIME
#
# Optimization does NOT need 160000 timesteps.
#
# -30 dB is sufficient for screening candidates.
# Final winner gets a separate long/high-resolution run.
# ============================================================

NR_TS = 12000

END_CRITERIA = 1e-3


# ============================================================
# MESH
#
# REAL controlled mesh = 0.20 mm.
#
# We deliberately do NOT add arbitrary ±0.05/±0.10 mm
# refinement planes because those were creating very small
# cells and collapsing the numerical Nyquist frequency.
# ============================================================

MESH_DX = 0.20
MESH_DY = 0.20
MESH_DZ = 0.20


# ============================================================
# PORT
# ============================================================

Z0 = 50.0


# ============================================================
# ADD BOX
# ============================================================

def add_box(
    material,
    p1,
    p2
):

    material.AddBox(
        p1,
        p2,
        priority=10
    )


# ============================================================
# SNAP TO MESH
#
# This prevents accidental tiny cells.
# ============================================================

def snap(
    value,
    step=MESH_DX
):

    return round(
        value / step
    ) * step


# ============================================================
# NORMALIZE LENGTHS
#
# The physical filter length remains fixed.
# Then section boundaries are snapped to the controlled mesh.
#
# This produces a tiny geometric quantization (~0.1 mm),
# which is completely appropriate for this optimization stage.
# ============================================================

def normalize_lengths(
    raw
):

    raw = np.asarray(
        raw,
        dtype=float
    )

    raw = np.clip(
        raw,
        L_MIN,
        L_MAX
    )

    # --------------------------------------------------------
    # First normalize exactly to FILTER_LENGTH
    # --------------------------------------------------------

    lengths = np.array([
        raw[0],
        raw[1],
        raw[2],
        raw[3],
        raw[2],
        raw[1],
        raw[0]
    ])

    scale = (
        FILTER_LENGTH
        /
        np.sum(lengths)
    )

    lengths *= scale

    # --------------------------------------------------------
    # Construct cumulative section boundaries
    # --------------------------------------------------------

    boundaries = [
        0.0
    ]

    for L in lengths:

        boundaries.append(
            boundaries[-1] + L
        )

    # --------------------------------------------------------
    # Snap boundaries to 0.2 mm mesh
    #
    # The first and final boundary are fixed.
    # --------------------------------------------------------

    snapped = [
        0.0
    ]

    for x in boundaries[1:-1]:

        snapped.append(
            snap(
                x,
                MESH_DX
            )
        )

    snapped.append(
        FILTER_LENGTH
    )

    snapped = np.asarray(
        snapped,
        dtype=float
    )

    # --------------------------------------------------------
    # Guarantee strictly increasing boundaries
    # --------------------------------------------------------

    for i in range(
        1,
        len(snapped)
    ):

        if snapped[i] <= snapped[i - 1]:

            snapped[i] = (
                snapped[i - 1]
                +
                MESH_DX
            )

    # --------------------------------------------------------
    # Force final boundary
    # --------------------------------------------------------

    snapped[-1] = FILTER_LENGTH

    # --------------------------------------------------------
    # Convert boundaries to lengths
    # --------------------------------------------------------

    lengths = np.diff(
        snapped
    )

    return lengths


# ============================================================
# BUILD CONTROLLED MESH
# ============================================================

def uniform_lines(
    start,
    stop,
    step
):

    values = []

    x = start

    while x < stop - 1e-12:

        values.append(
            x
        )

        x += step

    values.append(
        stop
    )

    return values


# ============================================================
# BUILD GEOMETRY
# ============================================================

def build_geometry(
    lengths,
    geometry_xml
):

    L1, L2, L3, L4, L5, L6, L7 = lengths

    # --------------------------------------------------------
    # SECTION POSITIONS
    # --------------------------------------------------------

    x0 = INPUT_LAUNCH

    x1 = x0 + L1
    x2 = x1 + L2
    x3 = x2 + L3
    x4 = x3 + L4
    x5 = x4 + L5
    x6 = x5 + L6
    x7 = x6 + L7

    expected = (
        TOTAL_LENGTH
        - OUTPUT_LAUNCH
    )

    if abs(
        x7 - expected
    ) > 1e-8:

        raise RuntimeError(
            f"Section length error: "
            f"x7={x7}, expected={expected}"
        )

    # --------------------------------------------------------
    # CSX
    # --------------------------------------------------------

    CSX = ContinuousStructure()

    mesh = CSX.GetGrid()

    mesh.SetDeltaUnit(
        1e-3
    )

    # ========================================================
    # X MESH
    #
    # Uniform 0.20 mm mesh.
    #
    # Section boundaries are already snapped to this mesh.
    # ========================================================

    x_lines = uniform_lines(
        0.0,
        TOTAL_LENGTH,
        MESH_DX
    )

    # Ensure important planes are present.
    #
    # These should already lie very close to mesh planes.
    # Add only if they are not already represented.
    #
    # We DO NOT add ±0.05 / ±0.10 refinement planes.

    important_x = [
        INPUT_LAUNCH,
        TOTAL_LENGTH - OUTPUT_LAUNCH,
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        INPUT_LAUNCH - 1.0,
        TOTAL_LENGTH - OUTPUT_LAUNCH + 1.0
    ]

    for xx in important_x:

        if (
            xx >= 0.0
            and
            xx <= TOTAL_LENGTH
        ):

            # Only add if sufficiently separated
            # from existing mesh lines.
            nearest = min(
                abs(xx - v)
                for v in x_lines
            )

            if nearest > 1e-8:

                # DO NOT insert arbitrary tiny cells.
                #
                # Snap the important plane instead.
                snapped_x = snap(
                    xx,
                    MESH_DX
                )

                if (
                    snapped_x >= 0.0
                    and
                    snapped_x <= TOTAL_LENGTH
                ):

                    if all(
                        abs(
                            snapped_x - v
                        ) > 1e-8
                        for v in x_lines
                    ):

                        x_lines.append(
                            snapped_x
                        )

    x_lines = sorted(
        set(
            round(v, 10)
            for v in x_lines
        )
    )

    mesh.SetLines(
        "x",
        x_lines
    )

    # ========================================================
    # Y MESH
    #
    # Uniform 0.20 mm.
    # ========================================================

    y_lines = uniform_lines(
        -TOTAL_WIDTH / 2,
        TOTAL_WIDTH / 2,
        MESH_DY
    )

    # Explicit geometry planes
    for yy in [
        -2.3,
        -CENTER_WIDTH / 2,
        0.0,
        CENTER_WIDTH / 2,
        2.3
    ]:

        yy = snap(
            yy,
            MESH_DY
        )

        if yy not in y_lines:

            y_lines.append(
                yy
            )

    y_lines = sorted(
        set(
            round(v, 10)
            for v in y_lines
        )
    )

    mesh.SetLines(
        "y",
        y_lines
    )

    # ========================================================
    # Z MESH
    #
    # Uniform 0.20 mm.
    #
    # The center conductor heights are represented using
    # the nearest controlled mesh planes.
    # ========================================================

    z_lines = uniform_lines(
        -TOTAL_HEIGHT / 2,
        TOTAL_HEIGHT / 2,
        MESH_DZ
    )

    for zz in [
        -2.1,
        2.1,
        -H_THICK / 2,
        -H_THIN / 2,
        0.0,
        H_THIN / 2,
        H_THICK / 2
    ]:

        zz = snap(
            zz,
            MESH_DZ
        )

        if (
            zz >= -TOTAL_HEIGHT / 2
            and
            zz <= TOTAL_HEIGHT / 2
        ):

            if zz not in z_lines:

                z_lines.append(
                    zz
                )

    z_lines = sorted(
        set(
            round(v, 10)
            for v in z_lines
        )
    )

    mesh.SetLines(
        "z",
        z_lines
    )

    # ========================================================
    # COPPER
    # ========================================================

    copper = CSX.AddMetal(
        "Copper"
    )

    # ========================================================
    # OUTER CONDUCTOR
    # ========================================================

    # Bottom
    add_box(
        copper,
        [
            0.0,
            -2.5,
            -2.3
        ],
        [
            TOTAL_LENGTH,
            2.5,
            -2.1
        ]
    )

    # Top
    add_box(
        copper,
        [
            0.0,
            -2.5,
            2.1
        ],
        [
            TOTAL_LENGTH,
            2.5,
            2.3
        ]
    )

    # Left
    add_box(
        copper,
        [
            0.0,
            -2.5,
            -2.3
        ],
        [
            TOTAL_LENGTH,
            -2.3,
            2.3
        ]
    )

    # Right
    add_box(
        copper,
        [
            0.0,
            2.3,
            -2.3
        ],
        [
            TOTAL_LENGTH,
            2.5,
            2.3
        ]
    )

    # ========================================================
    # CENTER CONDUCTOR
    # ========================================================

    sections = [

        (x0, x1, H_THIN),

        (x1, x2, H_THICK),

        (x2, x3, H_THIN),

        (x3, x4, H_THICK),

        (x4, x5, H_THIN),

        (x5, x6, H_THICK),

        (x6, x7, H_THIN)
    ]

    for xa, xb, h in sections:

        # ----------------------------------------------------
        # Use the physical geometry values.
        #
        # They are already aligned to the controlled mesh
        # within the optimizer's quantization.
        # ----------------------------------------------------

        add_box(
            copper,
            [
                xa,
                -CENTER_WIDTH / 2,
                -h / 2
            ],
            [
                xb,
                CENTER_WIDTH / 2,
                h / 2
            ]
        )

    # ========================================================
    # SAVE
    # ========================================================

    CSX.Write2XML(
        str(geometry_xml)
    )

    return CSX


# ============================================================
# SIMULATION
# ============================================================

def simulate(
    lengths,
    run_dir
):

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    geometry_xml = (
        run_dir
        /
        "geometry.xml"
    )

    print(
        "  [1/7] Building geometry..."
    )

    CSX = build_geometry(
        lengths,
        geometry_xml
    )

    print(
        "  [2/7] Creating FDTD..."
    )

    FDTD = openEMS(
        NrTS=NR_TS,
        EndCriteria=END_CRITERIA
    )

    print(
        "  [3/7] Connecting CSX to FDTD..."
    )

    FDTD.SetCSX(
        CSX
    )

    print(
        "  [4/7] Setting excitation..."
    )

    FDTD.SetGaussExcite(
        F0,
        FC
    )

    # ========================================================
    # PML
    # ========================================================

    FDTD.SetBoundaryCond(
        [
            "PML_8",
            "PML_8",
            "PML_8",
            "PML_8",
            "PML_8",
            "PML_8"
        ]
    )

    # ========================================================
    # PORTS
    #
    # SAME WORKING PORT GEOMETRY.
    # ========================================================

    print(
        "  [5/7] Creating ports..."
    )

    PORT1_X = (
        INPUT_LAUNCH
        -
        1.0
    )

    PORT2_X = (
        TOTAL_LENGTH
        -
        OUTPUT_LAUNCH
        +
        1.0
    )

    port1 = FDTD.AddLumpedPort(
        1,
        Z0,
        [
            PORT1_X,
            -CENTER_WIDTH / 2,
            H_THIN / 2
        ],
        [
            PORT1_X,
            CENTER_WIDTH / 2,
            TOTAL_HEIGHT / 2 - WALL
        ],
        "z",
        excite=1
    )

    port2 = FDTD.AddLumpedPort(
        2,
        Z0,
        [
            PORT2_X,
            -CENTER_WIDTH / 2,
            H_THIN / 2
        ],
        [
            PORT2_X,
            CENTER_WIDTH / 2,
            TOTAL_HEIGHT / 2 - WALL
        ],
        "z",
        excite=0
    )

    # ========================================================
    # RUN
    # ========================================================

    sim_dir = (
        run_dir
        /
        "sim"
    )

    if sim_dir.exists():

        shutil.rmtree(
            sim_dir
        )

    sim_dir.mkdir()

    print(
        "  [6/7] Running FDTD..."
    )

    FDTD.Run(
        str(sim_dir),
        cleanup=True
    )

    print(
        "  [7/7] Calculating S-parameters..."
    )

    # ========================================================
    # PORT DATA
    # ========================================================

    port1.CalcPort(
        str(sim_dir),
        FREQS
    )

    port2.CalcPort(
        str(sim_dir),
        FREQS
    )

    # ========================================================
    # CHECK FOR NON-FINITE DATA
    # ========================================================

    arrays_to_check = [
        port1.uf_inc,
        port1.uf_ref,
        port2.uf_ref
    ]

    for arr in arrays_to_check:

        if not np.all(
            np.isfinite(arr)
        ):

            raise RuntimeError(
                "Non-finite port data detected "
                "(NaN/Inf). Candidate rejected."
            )

    # ========================================================
    # S PARAMETERS
    # ========================================================

    denominator = (
        port1.uf_inc
    )

    denominator = np.where(
        np.abs(denominator) < 1e-20,
        1e-20,
        denominator
    )

    s11 = (
        port1.uf_ref
        /
        denominator
    )

    s21 = (
        port2.uf_ref
        /
        denominator
    )

    if not (
        np.all(
            np.isfinite(s11)
        )
        and
        np.all(
            np.isfinite(s21)
        )
    ):

        raise RuntimeError(
            "Non-finite S-parameters detected."
        )

    # ========================================================
    # dB
    # ========================================================

    s11_db = (
        20
        *
        np.log10(
            np.maximum(
                np.abs(s11),
                1e-15
            )
        )
    )

    s21_db = (
        20
        *
        np.log10(
            np.maximum(
                np.abs(s21),
                1e-15
            )
        )
    )

    # ========================================================
    # CSV
    # ========================================================

    with open(
        run_dir / "result.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "Frequency_GHz",
            "S11_dB",
            "S21_dB",
            "S11_real",
            "S11_imag",
            "S21_real",
            "S21_imag"
        ])

        for i in range(
            len(FREQS)
        ):

            writer.writerow([
                FREQS[i] / 1e9,
                s11_db[i],
                s21_db[i],
                np.real(s11[i]),
                np.imag(s11[i]),
                np.real(s21[i]),
                np.imag(s21[i])
            ])

    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.plot(
        FREQS / 1e9,
        s21_db,
        label="S21"
    )

    ax.plot(
        FREQS / 1e9,
        s11_db,
        label="S11"
    )

    ax.axvline(
        10,
        linestyle="--",
        label="10 GHz"
    )

    ax.axvline(
        20,
        linestyle="--",
        label="20 GHz"
    )

    ax.axvline(
        28,
        linestyle=":",
        label="28 GHz"
    )

    ax.axhline(
        -60,
        linestyle="--",
        label="-60 dB"
    )

    ax.axvspan(
        24,
        30,
        alpha=0.12,
        label="28 GHz search region"
    )

    ax.set_xlim(
        0,
        30
    )

    ax.set_ylim(
        -100,
        5
    )

    ax.set_xlabel(
        "Frequency (GHz)"
    )

    ax.set_ylabel(
        "Magnitude (dB)"
    )

    ax.set_title(
        "Run-13 Fine Optimization — 0.5–30 GHz"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        run_dir / "response.png",
        dpi=160
    )

    plt.close(
        fig
    )

    return (
        FREQS,
        s11_db,
        s21_db
    )


# ============================================================
# METRICS
# ============================================================

def calculate_score(
    freq,
    s11,
    s21
):

    # ========================================================
    # PASSBAND
    # ========================================================

    pb = (
        (freq >= 1e9)
        &
        (freq <= 8e9)
    )

    worst_pb = np.min(
        s21[pb]
    )

    avg_il = np.mean(
        -s21[pb]
    )

    # ========================================================
    # 10 GHz
    # ========================================================

    s21_10 = np.interp(
        10e9,
        freq,
        s21
    )

    # ========================================================
    # 20 GHz
    # ========================================================

    s21_20 = np.interp(
        20e9,
        freq,
        s21
    )

    # ========================================================
    # 24–30 GHz
    #
    # THIS IS THE MAIN SPUR OBJECTIVE.
    # ========================================================

    mask28 = (
        (freq >= 24e9)
        &
        (freq <= 30e9)
    )

    peak28 = np.max(
        s21[mask28]
    )

    freq28 = (
        freq[mask28][
            np.argmax(
                s21[mask28]
            )
        ]
    )

    # ========================================================
    # 20–30 GHz GLOBAL STOPBAND
    # ========================================================

    mask_global = (
        (freq >= 20e9)
        &
        (freq <= 30e9)
    )

    global_peak = np.max(
        s21[mask_global]
    )

    freq_global = (
        freq[mask_global][
            np.argmax(
                s21[mask_global]
            )
        ]
    )

    # ========================================================
    # SCORE
    # ========================================================

    score = 0.0

    # --------------------------------------------------------
    # Passband insertion loss
    # --------------------------------------------------------

    score += (
        avg_il
        *
        2
    )

    # Penalize very poor passband

    if worst_pb < -1.5:

        score += (
            -1.5
            -
            worst_pb
        ) * 5

    # --------------------------------------------------------
    # 10 GHz transition
    # --------------------------------------------------------

    if s21_10 > -3:

        score += (
            s21_10
            +
            3
        ) * 8

    # --------------------------------------------------------
    # 20 GHz
    #
    # Strong penalty until -60 dB is achieved.
    # --------------------------------------------------------

    if s21_20 > -60:

        score += (
            s21_20
            +
            60
        ) * 20

    else:

        # Small reward for going deeper than -60 dB
        score += (
            s21_20
            +
            60
        ) * 0.5

    # --------------------------------------------------------
    # 28 GHz spur
    #
    # Strongest optimization term.
    # --------------------------------------------------------

    if peak28 > -70:

        score += (
            peak28
            +
            70
        ) * 35

    else:

        score += (
            peak28
            +
            70
        ) * 0.5

    # --------------------------------------------------------
    # General stopband
    # --------------------------------------------------------

    if global_peak > -45:

        score += (
            global_peak
            +
            45
        ) * 5

    return {
        "score": score,
        "worst_pb": worst_pb,
        "avg_il": avg_il,
        "s21_10": s21_10,
        "s21_20": s21_20,
        "peak28": peak28,
        "freq28": freq28,
        "global_peak": global_peak,
        "freq_global": freq_global
    }


# ============================================================
# CANDIDATES
# ============================================================

def generate_candidates():

    candidates = [
        RUN13.copy()
    ]

    sigma = np.array([
        0.30,
        0.30,
        0.30,
        0.40
    ])

    for _ in range(
        N_CANDIDATES - 1
    ):

        raw = (
            RUN13
            +
            rng.normal(
                0,
                sigma
            )
        )

        raw = np.clip(
            raw,
            L_MIN,
            L_MAX
        )

        candidates.append(
            raw
        )

    return candidates


# ============================================================
# SAVE PARAMETERS
# ============================================================

def save_parameters(
    run_dir,
    run_number,
    lengths,
    metrics
):

    with open(
        run_dir / "parameters.txt",
        "w"
    ) as f:

        f.write(
            "RUN-13 FINE OPTIMIZATION\n"
        )

        f.write(
            "========================\n\n"
        )

        f.write(
            f"Run = {run_number}\n"
        )

        f.write(
            f"Score = "
            f"{metrics['score']:.8f}\n\n"
        )

        f.write(
            "FIXED GEOMETRY\n"
        )

        f.write(
            "--------------\n"
        )

        f.write(
            f"Overall length = "
            f"{TOTAL_LENGTH:.8f} mm\n"
        )

        f.write(
            f"Overall width = "
            f"{TOTAL_WIDTH:.8f} mm\n"
        )

        f.write(
            f"Overall height = "
            f"{TOTAL_HEIGHT:.8f} mm\n"
        )

        f.write(
            f"Wall = "
            f"{WALL:.8f} mm\n"
        )

        f.write(
            f"Center width = "
            f"{CENTER_WIDTH:.8f} mm\n"
        )

        f.write(
            f"h_thin = "
            f"{H_THIN:.8f} mm\n"
        )

        f.write(
            f"h_thick = "
            f"{H_THICK:.8f} mm\n\n"
        )

        f.write(
            "SECTION LENGTHS\n"
        )

        for i, L in enumerate(
            lengths,
            1
        ):

            f.write(
                f"L{i} = "
                f"{L:.8f} mm\n"
            )

        f.write(
            "\nPERFORMANCE\n"
        )

        f.write(
            f"Worst PB S21 = "
            f"{metrics['worst_pb']:.4f} dB\n"
        )

        f.write(
            f"Average PB IL = "
            f"{metrics['avg_il']:.4f} dB\n"
        )

        f.write(
            f"S21 @ 10 GHz = "
            f"{metrics['s21_10']:.4f} dB\n"
        )

        f.write(
            f"S21 @ 20 GHz = "
            f"{metrics['s21_20']:.4f} dB\n"
        )

        f.write(
            f"Peak 24–30 GHz = "
            f"{metrics['peak28']:.4f} dB\n"
        )

        f.write(
            f"Peak frequency = "
            f"{metrics['freq28']/1e9:.4f} GHz\n"
        )

        f.write(
            f"Global peak 20–30 GHz = "
            f"{metrics['global_peak']:.4f} dB\n"
        )

        f.write(
            f"Global peak frequency = "
            f"{metrics['freq_global']/1e9:.4f} GHz\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 76)
    print(
        "RUN-13 FINE OPTIMIZATION"
    )
    print(
        "0.5–30 GHz / 10 GHz LPF / 20 GHz -60 dB / 28 GHz SPUR"
    )
    print("=" * 76)
    print()

    print(
        f"Candidates       : {N_CANDIDATES}"
    )

    print(
        "Frequency        : 0.5–30.0 GHz"
    )

    print(
        f"Max timesteps    : {NR_TS}"
    )

    print(
        f"End criterion    : {END_CRITERIA}"
    )

    print(
        f"Mesh             : {MESH_DX:.2f} mm uniform"
    )

    print(
        "Architecture     : Run 13"
    )

    print()

    candidates = (
        generate_candidates()
    )

    best_score = np.inf

    best_data = None

    summary = []

    # ========================================================
    # CANDIDATE LOOP
    # ========================================================

    for run_number, raw in enumerate(
        candidates
    ):

        print()
        print("-" * 76)

        print(
            f"RUN {run_number:03d}"
        )

        print(
            "Raw:",
            np.round(
                raw,
                6
            )
        )

        lengths = (
            normalize_lengths(
                raw
            )
        )

        print(
            "Final:",
            np.round(
                lengths,
                6
            )
        )

        run_dir = (
            ROOT
            /
            f"run_{run_number:03d}"
        )

        try:

            (
                freq,
                s11,
                s21
            ) = simulate(
                lengths,
                run_dir
            )

            metrics = (
                calculate_score(
                    freq,
                    s11,
                    s21
                )
            )

            save_parameters(
                run_dir,
                run_number,
                lengths,
                metrics
            )

            print()

            print(
                f"S21 @ 10 GHz : "
                f"{metrics['s21_10']:.3f} dB"
            )

            print(
                f"S21 @ 20 GHz : "
                f"{metrics['s21_20']:.3f} dB"
            )

            print(
                f"24–30 GHz    : "
                f"{metrics['peak28']:.3f} dB "
                f"@ "
                f"{metrics['freq28']/1e9:.3f} GHz"
            )

            print(
                f"20–30 GHz    : "
                f"{metrics['global_peak']:.3f} dB "
                f"@ "
                f"{metrics['freq_global']/1e9:.3f} GHz"
            )

            print(
                f"PB worst     : "
                f"{metrics['worst_pb']:.3f} dB"
            )

            print(
                f"Score        : "
                f"{metrics['score']:.4f}"
            )

            summary.append({
                "run": run_number,
                **metrics,
                "L1": lengths[0],
                "L2": lengths[1],
                "L3": lengths[2],
                "L4": lengths[3]
            })

            # ------------------------------------------------
            # BEST
            # ------------------------------------------------

            if (
                metrics["score"]
                <
                best_score
            ):

                best_score = (
                    metrics["score"]
                )

                best_data = {
                    "run": run_number,
                    "lengths": lengths.copy(),
                    "metrics": metrics,
                    "run_dir": run_dir
                }

                print()
                print(
                    "******** NEW BEST ********"
                )

        except Exception as e:

            print()
            print(
                "SIMULATION FAILED"
            )

            print(
                repr(e)
            )

            run_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                run_dir / "ERROR.txt",
                "w"
            ) as f:

                traceback.print_exc(
                    file=f
                )

            # ------------------------------------------------
            # IMPORTANT:
            # Continue immediately to next candidate.
            # ------------------------------------------------

            continue

    # ========================================================
    # SUMMARY CSV
    # ========================================================

    if summary:

        with open(
            ROOT / "optimization_summary.csv",
            "w",
            newline=""
        ) as f:

            fields = [
                "run",
                "score",
                "worst_pb",
                "avg_il",
                "s21_10",
                "s21_20",
                "peak28",
                "freq28",
                "global_peak",
                "freq_global",
                "L1",
                "L2",
                "L3",
                "L4"
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )

            writer.writeheader()

            writer.writerows(
                summary
            )

    # ========================================================
    # NO SUCCESSFUL RUN
    # ========================================================

    if best_data is None:

        print()
        print("=" * 76)

        print(
            "NO SUCCESSFUL SIMULATIONS"
        )

        print("=" * 76)

        return

    # ========================================================
    # COPY BEST
    # ========================================================

    best_dir = (
        ROOT
        /
        "BEST"
    )

    if best_dir.exists():

        shutil.rmtree(
            best_dir
        )

    best_dir.mkdir()

    source = (
        best_data["run_dir"]
    )

    for item in source.iterdir():

        dest = (
            best_dir
            /
            item.name
        )

        if item.is_dir():

            shutil.copytree(
                item,
                dest
            )

        else:

            shutil.copy2(
                item,
                dest
            )

    # ========================================================
    # BEST DESIGN FILE
    # ========================================================

    lengths = (
        best_data["lengths"]
    )

    metrics = (
        best_data["metrics"]
    )

    with open(
        best_dir / "BEST_DESIGN.txt",
        "w"
    ) as f:

        f.write(
            "RUN-13 FINE OPTIMIZATION — BEST DESIGN\n"
        )

        f.write(
            "=======================================\n\n"
        )

        f.write(
            f"Run = "
            f"{best_data['run']}\n"
        )

        f.write(
            f"Score = "
            f"{metrics['score']:.8f}\n\n"
        )

        f.write(
            "OUTER DIMENSIONS\n"
        )

        f.write(
            "65.300000 x 5.000000 x 4.600000 mm\n\n"
        )

        f.write(
            "CENTER CONDUCTOR\n"
        )

        f.write(
            f"Width = "
            f"{CENTER_WIDTH:.8f} mm\n"
        )

        f.write(
            f"Thin height = "
            f"{H_THIN:.8f} mm\n"
        )

        f.write(
            f"Thick height = "
            f"{H_THICK:.8f} mm\n\n"
        )

        f.write(
            "SECTION LENGTHS\n"
        )

        for i, L in enumerate(
            lengths,
            1
        ):

            f.write(
                f"L{i} = "
                f"{L:.8f} mm\n"
            )

        f.write(
            "\nPERFORMANCE\n"
        )

        f.write(
            f"S21 @ 10 GHz = "
            f"{metrics['s21_10']:.4f} dB\n"
        )

        f.write(
            f"S21 @ 20 GHz = "
            f"{metrics['s21_20']:.4f} dB\n"
        )

        f.write(
            f"Peak 24–30 GHz = "
            f"{metrics['peak28']:.4f} dB\n"
        )

        f.write(
            f"Peak frequency = "
            f"{metrics['freq28']/1e9:.4f} GHz\n"
        )

        f.write(
            f"Global peak 20–30 GHz = "
            f"{metrics['global_peak']:.4f} dB\n"
        )

        f.write(
            f"Global peak frequency = "
            f"{metrics['freq_global']/1e9:.4f} GHz\n"
        )

    # ========================================================
    # FINAL CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 76)
    print(
        "OPTIMIZATION COMPLETE"
    )
    print("=" * 76)

    print()

    print(
        "BEST RUN:",
        best_data["run"]
    )

    print()

    for i, L in enumerate(
        lengths,
        1
    ):

        print(
            f"L{i} = {L:.8f} mm"
        )

    print()

    print(
        f"S21 @10 GHz = "
        f"{metrics['s21_10']:.3f} dB"
    )

    print(
        f"S21 @20 GHz = "
        f"{metrics['s21_20']:.3f} dB"
    )

    print(
        f"24–30 GHz peak = "
        f"{metrics['peak28']:.3f} dB "
        f"@ "
        f"{metrics['freq28']/1e9:.3f} GHz"
    )

    print()

    print(
        "BEST DIRECTORY:"
    )

    print(
        best_dir
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()