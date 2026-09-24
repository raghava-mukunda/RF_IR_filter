# ============================================================
# AUTOMATED EM OPTIMIZATION
# COMPACT 10 GHz CRYOGENIC LPF
#
# openEMS + Python
#
# Every candidate produces:
#
#   geometry.xml   -> complete physical geometry
#   openems.xml    -> geometry + FDTD setup
#   result.csv     -> S-parameters
#   response.png   -> candidate response
#
# Best candidate is copied to:
#
#   LPF10_OPT/BEST/
#
# ============================================================

from pathlib import Path
import os
import sys
import shutil
import csv

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# OPENEMS
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
# PROJECT
# ============================================================

PROJECT_DIR = (
    Path.home()
    / "Desktop"
    / "cryogenic_filter_design"
    / "LPF10_OPT"
)

PROJECT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIXED DIMENSIONS
# ============================================================

TOTAL_LENGTH = 65.3

OUTER_WIDTH = 5.0
OUTER_HEIGHT = 4.6

WALL = 0.20

CENTER_WIDTH = 1.20

PORT_LENGTH = 8.0

FILTER_LENGTH = (
    TOTAL_LENGTH
    - 2.0 * PORT_LENGTH
)

Z0 = 50.0


# ============================================================
# FREQUENCY
# ============================================================

F_START = 1e9
F_END = 30e9

N_FREQ = 301

FREQ = np.linspace(
    F_START,
    F_END,
    N_FREQ
)


# ============================================================
# SIMULATION SETTINGS
#
# Optimization runs are intentionally shorter than the
# final verification simulation.
#
# Final candidate should be rerun with:
#
# NrTS = 180000
# EndCriteria = 1e-7
#
# ============================================================

OPT_NR_TS = 100000
OPT_END_CRITERIA = 1e-6


# ============================================================
# OPTIMIZATION BOUNDS
# ============================================================
#
# x =
#
# [h_thin,
#  h_thick,
#  L1,
#  L2,
#  L3,
#  L4]
#
# Full section lengths:
#
# [L1,L2,L3,L4,L3,L2,L1]
#
# The four lengths are normalized so that their
# total always equals 49.3 mm.
#
# ============================================================

H_THIN_MIN = 0.20
H_THIN_MAX = 0.40

H_THICK_MIN = 0.60
H_THICK_MAX = 1.20

L_MIN = 4.0
L_MAX = 10.5


# ============================================================
# NUMBER OF RUNS
# ============================================================
#
# Start conservatively.
#
# Each EM evaluation is a real FDTD simulation.
#
# Increase this after the workflow is confirmed.
#
# ============================================================

MAX_RUNS = 20


# ============================================================
# INITIAL DESIGN
# ============================================================

INITIAL_X = np.array([
    0.30,       # thin height
    0.80,       # thick height
    7.042857,   # L1
    7.042857,   # L2
    7.042857,   # L3
    7.042857,   # L4
])


# ============================================================
# RANDOM SEARCH SETTINGS
# ============================================================

RNG = np.random.default_rng(20260924)


# ============================================================
# SECTION LENGTH NORMALIZATION
# ============================================================

def make_section_lengths(x):

    raw = np.array([
        x[2],
        x[3],
        x[4],
        x[5],
    ])

    # Mirror symmetry:
    #
    # L1 L2 L3 L4 L3 L2 L1
    #
    weights = np.array([
        raw[0],
        raw[1],
        raw[2],
        raw[3],
        raw[2],
        raw[1],
        raw[0],
    ])

    lengths = (
        weights
        / np.sum(weights)
        * FILTER_LENGTH
    )

    return lengths


# ============================================================
# CANDIDATE PARAMETER GENERATOR
# ============================================================

def generate_candidate(run_number):

    if run_number == 0:

        return INITIAL_X.copy()

    x = np.zeros(6)

    x[0] = RNG.uniform(
        H_THIN_MIN,
        H_THIN_MAX
    )

    x[1] = RNG.uniform(
        H_THICK_MIN,
        H_THICK_MAX
    )

    x[2:] = RNG.uniform(
        L_MIN,
        L_MAX,
        4
    )

    return x


# ============================================================
# MESH
# ============================================================

def build_mesh(CSX, section_lengths, h_thin, h_thick):

    mesh = CSX.GetGrid()

    mesh.SetDeltaUnit(
        1e-3
    )

    # --------------------------------------------------------
    # X
    # --------------------------------------------------------

    x_lines = [
        0.0,
        TOTAL_LENGTH,
    ]

    xpos = PORT_LENGTH

    x_lines.append(xpos)

    for L in section_lengths:

        xpos += L

        x_lines.append(xpos)

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

    # Refine discontinuities
    for x in list(x_lines):

        for dx in [
            -0.10,
            -0.05,
            0.05,
            0.10,
        ]:

            xx = x + dx

            if (
                0.0 < xx < TOTAL_LENGTH
            ):
                x_lines.append(xx)

    mesh.AddLine(
        "x",
        sorted(set(x_lines))
    )


    # --------------------------------------------------------
    # Y
    # --------------------------------------------------------

    y_lines = [
        -OUTER_WIDTH / 2,
        -OUTER_WIDTH / 2 + WALL,

        -CENTER_WIDTH / 2,
        0.0,
        CENTER_WIDTH / 2,

        OUTER_WIDTH / 2 - WALL,
        OUTER_WIDTH / 2,
    ]

    mesh.AddLine(
        "y",
        sorted(set(y_lines))
    )


    # --------------------------------------------------------
    # Z
    # --------------------------------------------------------

    z_lines = [
        -OUTER_HEIGHT / 2,
        -OUTER_HEIGHT / 2 + WALL,

        -h_thick / 2,
        -h_thin / 2,

        0.0,

        h_thin / 2,
        h_thick / 2,

        OUTER_HEIGHT / 2 - WALL,
        OUTER_HEIGHT / 2,
    ]

    mesh.AddLine(
        "z",
        sorted(set(z_lines))
    )

    return mesh


# ============================================================
# BUILD GEOMETRY
# ============================================================

def build_geometry(
    CSX,
    section_lengths,
    h_thin,
    h_thick
):

    metal = CSX.AddMetal(
        "Copper"
    )


    # ========================================================
    # OUTER CONDUCTOR
    # ========================================================

    # Bottom
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


    # Top
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


    # Left
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


    # Right
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


    # ========================================================
    # INPUT LAUNCH
    # ========================================================

    metal.AddBox(
        [
            0.0,
            -CENTER_WIDTH / 2,
            -h_thin / 2,
        ],
        [
            PORT_LENGTH,
            CENTER_WIDTH / 2,
            h_thin / 2,
        ],
        priority=20
    )


    # ========================================================
    # FILTER
    # ========================================================

    xpos = PORT_LENGTH

    for n, L in enumerate(section_lengths):

        x0 = xpos
        x1 = xpos + L

        # Alternating geometry
        if n % 2 == 0:
            h = h_thin
        else:
            h = h_thick

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

        xpos = x1


    # ========================================================
    # OUTPUT LAUNCH
    # ========================================================

    metal.AddBox(
        [
            TOTAL_LENGTH - PORT_LENGTH,
            -CENTER_WIDTH / 2,
            -h_thin / 2,
        ],
        [
            TOTAL_LENGTH,
            CENTER_WIDTH / 2,
            h_thin / 2,
        ],
        priority=20
    )


    return metal


# ============================================================
# BUILD PORTS
# ============================================================

def add_ports(
    FDTD,
    h_thin
):

    PORT1_X = PORT_LENGTH - 1.0

    PORT2_X = (
        TOTAL_LENGTH
        - PORT_LENGTH
        + 1.0
    )

    PORT_Y1 = -CENTER_WIDTH / 2
    PORT_Y2 = CENTER_WIDTH / 2

    PORT_Z1 = h_thin / 2
    PORT_Z2 = OUTER_HEIGHT / 2 - WALL


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


    return port1, port2


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    freq,
    s21_db,
    s11_db
):

    # --------------------------------------------------------
    # Passband
    # --------------------------------------------------------

    pb = (
        freq >= 1e9
    ) & (
        freq <= 8e9
    )

    worst_pb = np.max(
        s21_db[pb]
    )

    # --------------------------------------------------------
    # 3 dB cutoff
    # --------------------------------------------------------

    target_level = -3.0

    # First frequency below -3 dB
    indices = np.where(
        s21_db <= target_level
    )[0]

    if len(indices) == 0:

        f3db = 30e9

    else:

        f3db = freq[
            indices[0]
        ]

    f3db_ghz = f3db / 1e9


    # --------------------------------------------------------
    # S21 @ 20 GHz
    # --------------------------------------------------------

    idx20 = np.argmin(
        np.abs(
            freq - 20e9
        )
    )

    s21_20 = s21_db[idx20]


    # --------------------------------------------------------
    # Ripple below 8 GHz
    # --------------------------------------------------------

    pb_values = s21_db[pb]

    ripple = (
        np.max(pb_values)
        - np.min(pb_values)
    )


    # --------------------------------------------------------
    # S11 below 8 GHz
    # --------------------------------------------------------

    s11_pb = np.max(
        s11_db[pb]
    )


    # ========================================================
    # OBJECTIVE
    # ========================================================

    score = 0.0

    # Strongly target 10 GHz cutoff
    score += (
        12.0
        * abs(f3db_ghz - 10.0)
    )


    # Penalize passband insertion loss
    if worst_pb < -1.0:

        score += (
            (-1.0 - worst_pb)
            * 5.0
        )


    # Penalize passband ripple
    score += (
        max(0.0, ripple - 1.0)
        * 2.0
    )


    # Strong stopband requirement
    if s21_20 > -60.0:

        score += (
            s21_20 + 60.0
        ) * 8.0


    # Reflection penalty
    if s11_pb > -10.0:

        score += (
            s11_pb + 10.0
        ) * 1.0


    return (
        score,
        f3db_ghz,
        s21_20,
        worst_pb,
        ripple,
        s11_pb
    )


# ============================================================
# RUN ONE CANDIDATE
# ============================================================

def run_candidate(
    run_number,
    x
):

    run_dir = (
        PROJECT_DIR
        / f"run_{run_number:03d}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sim_dir = (
        run_dir
        / "sim"
    )

    sim_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    h_thin = float(x[0])
    h_thick = float(x[1])

    section_lengths = (
        make_section_lengths(x)
    )


    print()
    print("=" * 70)
    print(
        f"OPTIMIZATION RUN {run_number:03d}"
    )
    print("=" * 70)

    print(
        f"h_thin  = {h_thin:.5f} mm"
    )

    print(
        f"h_thick = {h_thick:.5f} mm"
    )

    print(
        "Sections = "
        + ", ".join(
            f"{v:.5f}"
            for v in section_lengths
        )
    )


    # ========================================================
    # FDTD
    # ========================================================

    FDTD = openEMS(
        EndCriteria=OPT_END_CRITERIA,
        NrTS=OPT_NR_TS
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


    # ========================================================
    # CSX
    # ========================================================

    CSX = ContinuousStructure()

    FDTD.SetCSX(CSX)


    # ========================================================
    # MESH
    # ========================================================

    build_mesh(
        CSX,
        section_lengths,
        h_thin,
        h_thick
    )


    # ========================================================
    # GEOMETRY
    # ========================================================

    build_geometry(
        CSX,
        section_lengths,
        h_thin,
        h_thick
    )


    # ========================================================
    # SAVE GEOMETRY-ONLY XML
    #
    # This is the file to open in AppCSXCAD.
    # It contains the complete filter body including
    # the outer conductor.
    # ========================================================

    geometry_xml = (
        run_dir
        / "geometry.xml"
    )

    CSX.Write2XML(
        str(geometry_xml)
    )


    # ========================================================
    # PORTS
    # ========================================================

    port1, port2 = add_ports(
        FDTD,
        h_thin
    )


    # ========================================================
    # SAVE COMPLETE OPENEMS XML
    # ========================================================

    openems_xml = (
        run_dir
        / "openems.xml"
    )

    try:

        FDTD.Write2XML(
            str(openems_xml)
        )

    except Exception as exc:

        print(
            "WARNING: could not write "
            "full openEMS XML:"
        )

        print(exc)


    # ========================================================
    # RUN
    # ========================================================

    FDTD.Run(
        str(sim_dir),
        cleanup=True,
        verbose=1,
        numThreads=-1,
        disable_dumps=True,
    )


    # ========================================================
    # PORT PROCESSING
    # ========================================================

    port1.CalcPort(
        str(sim_dir),
        FREQ,
        ref_impedance=Z0
    )

    port2.CalcPort(
        str(sim_dir),
        FREQ,
        ref_impedance=Z0
    )


    # ========================================================
    # S PARAMETERS
    # ========================================================

    s11 = (
        port1.uf_ref
        / port1.uf_inc
    )

    s21 = (
        port2.uf_ref
        / port1.uf_inc
    )


    s11_db = (
        20.0
        * np.log10(
            np.maximum(
                np.abs(s11),
                1e-15
            )
        )
    )

    s21_db = (
        20.0
        * np.log10(
            np.maximum(
                np.abs(s21),
                1e-15
            )
        )
    )


    # ========================================================
    # SCORE
    # ========================================================

    (
        score,
        f3db,
        s21_20,
        worst_pb,
        ripple,
        s11_pb
    ) = calculate_score(
        FREQ,
        s21_db,
        s11_db
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_file = (
        run_dir
        / "result.csv"
    )

    with open(
        csv_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Frequency_Hz",
            "S11_dB",
            "S21_dB",
        ])

        for k in range(
            len(FREQ)
        ):

            writer.writerow([
                FREQ[k],
                s11_db[k],
                s21_db[k],
            ])


    # ========================================================
    # SAVE CANDIDATE PLOT
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        FREQ / 1e9,
        s21_db,
        label="S21"
    )

    plt.plot(
        FREQ / 1e9,
        s11_db,
        label="S11"
    )

    plt.axvline(
        10,
        linestyle="--",
        label="10 GHz"
    )

    plt.axvline(
        20,
        linestyle="--",
        label="20 GHz"
    )

    plt.axhline(
        -60,
        linestyle="--",
        label="-60 dB"
    )

    plt.xlabel(
        "Frequency (GHz)"
    )

    plt.ylabel(
        "Magnitude (dB)"
    )

    plt.title(
        f"LPF Optimization Run {run_number:03d}"
    )

    plt.xlim(
        1,
        30
    )

    plt.ylim(
        -100,
        5
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        run_dir / "response.png",
        dpi=180
    )

    plt.close()


    # ========================================================
    # PARAMETERS FILE
    # ========================================================

    with open(
        run_dir / "parameters.txt",
        "w"
    ) as f:

        f.write(
            "COMPACT 10 GHz LPF\n"
        )

        f.write(
            "===================\n\n"
        )

        f.write(
            f"Run = {run_number}\n"
        )

        f.write(
            f"Score = {score:.8f}\n"
        )

        f.write(
            f"h_thin = {h_thin:.8f} mm\n"
        )

        f.write(
            f"h_thick = {h_thick:.8f} mm\n"
        )

        f.write(
            "\nSection lengths:\n"
        )

        for n, L in enumerate(
            section_lengths,
            start=1
        ):

            f.write(
                f"L{n} = {L:.8f} mm\n"
            )

        f.write(
            "\nPerformance:\n"
        )

        f.write(
            f"f3dB = {f3db:.4f} GHz\n"
        )

        f.write(
            f"S21_20GHz = "
            f"{s21_20:.4f} dB\n"
        )

        f.write(
            f"Worst PB S21 = "
            f"{worst_pb:.4f} dB\n"
        )

        f.write(
            f"PB ripple = "
            f"{ripple:.4f} dB\n"
        )

        f.write(
            f"Worst PB S11 = "
            f"{s11_pb:.4f} dB\n"
        )


    print()
    print(
        f"f3dB       = {f3db:.3f} GHz"
    )

    print(
        f"S21@20GHz = {s21_20:.3f} dB"
    )

    print(
        f"PB S21     = {worst_pb:.3f} dB"
    )

    print(
        f"PB ripple  = {ripple:.3f} dB"
    )

    print(
        f"S11 PB     = {s11_pb:.3f} dB"
    )

    print(
        f"SCORE      = {score:.5f}"
    )

    print(
        f"Geometry   = {geometry_xml}"
    )

    return {
        "run": run_number,
        "score": score,
        "f3db": f3db,
        "s21_20": s21_20,
        "worst_pb": worst_pb,
        "ripple": ripple,
        "s11_pb": s11_pb,
        "x": x.copy(),
        "section_lengths": section_lengths.copy(),
        "run_dir": run_dir,
    }


# ============================================================
# MAIN OPTIMIZATION
# ============================================================

results = []

best = None


print()
print("=" * 70)
print("STARTING EM OPTIMIZATION")
print("=" * 70)

print(
    f"Maximum candidate simulations: "
    f"{MAX_RUNS}"
)

print(
    "Every candidate will have its own XML."
)

print("=" * 70)


for run_number in range(
    MAX_RUNS
):

    x = generate_candidate(
        run_number
    )

    try:

        result = run_candidate(
            run_number,
            x
        )

        results.append(
            result
        )

        # ----------------------------------------------------
        # Best candidate
        # ----------------------------------------------------

        if (
            best is None
            or result["score"]
            < best["score"]
        ):

            best = result

            print()
            print(
                "******** NEW BEST DESIGN ********"
            )

            print(
                f"Run       : "
                f"{result['run']}"
            )

            print(
                f"Score     : "
                f"{result['score']:.5f}"
            )

            print(
                f"f3dB      : "
                f"{result['f3db']:.3f} GHz"
            )

            print(
                f"S21@20GHz: "
                f"{result['s21_20']:.3f} dB"
            )

            print(
                "**********************************"
            )


    except Exception as exc:

        print()
        print(
            f"RUN {run_number:03d} FAILED"
        )

        print(exc)

        continue


# ============================================================
# SAVE OPTIMIZATION SUMMARY
# ============================================================

summary_file = (
    PROJECT_DIR
    / "optimization_results.csv"
)

with open(
    summary_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "run",
        "score",
        "f3db_GHz",
        "S21_20GHz_dB",
        "worst_passband_S21_dB",
        "passband_ripple_dB",
        "worst_passband_S11_dB",
        "h_thin_mm",
        "h_thick_mm",
        "L1_mm",
        "L2_mm",
        "L3_mm",
        "L4_mm",
        "L5_mm",
        "L6_mm",
        "L7_mm",
    ])

    for r in results:

        writer.writerow([
            r["run"],
            r["score"],
            r["f3db"],
            r["s21_20"],
            r["worst_pb"],
            r["ripple"],
            r["s11_pb"],
            r["x"][0],
            r["x"][1],
            *r["section_lengths"],
        ])


# ============================================================
# COPY BEST DESIGN
# ============================================================

if best is not None:

    BEST_DIR = (
        PROJECT_DIR
        / "BEST"
    )

    BEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    source_dir = best["run_dir"]


    for filename in [
        "geometry.xml",
        "openems.xml",
        "result.csv",
        "response.png",
        "parameters.txt",
    ]:

        source = (
            source_dir
            / filename
        )

        if source.exists():

            shutil.copy2(
                source,
                BEST_DIR / filename
            )


    # ========================================================
    # COPY S2P
    # ========================================================

    source_csv = (
        source_dir
        / "result.csv"
    )

    data = np.loadtxt(
        source_csv,
        delimiter=",",
        skiprows=1
    )

    freq = data[:, 0]
    s11db = data[:, 1]
    s21db = data[:, 2]


    # Approximate complex S parameters are not recovered
    # from dB-only data. The final verification run should
    # generate the definitive Touchstone file.
    #
    # Therefore do NOT create a fake S2P here.


    # ========================================================
    # BEST SUMMARY
    # ========================================================

    with open(
        BEST_DIR / "BEST_DESIGN.txt",
        "w"
    ) as f:

        f.write(
            "OPTIMIZED LPF CANDIDATE\n"
        )

        f.write(
            "========================\n\n"
        )

        f.write(
            f"Run: {best['run']}\n"
        )

        f.write(
            f"Score: {best['score']:.8f}\n\n"
        )

        f.write(
            f"h_thin  = "
            f"{best['x'][0]:.8f} mm\n"
        )

        f.write(
            f"h_thick = "
            f"{best['x'][1]:.8f} mm\n\n"
        )

        for n, L in enumerate(
            best["section_lengths"],
            start=1
        ):

            f.write(
                f"L{n} = "
                f"{L:.8f} mm\n"
            )

        f.write("\n")

        f.write(
            f"f3dB = "
            f"{best['f3db']:.5f} GHz\n"
        )

        f.write(
            f"S21 @ 20 GHz = "
            f"{best['s21_20']:.5f} dB\n"
        )

        f.write(
            f"Worst passband S21 = "
            f"{best['worst_pb']:.5f} dB\n"
        )

        f.write(
            f"Passband ripple = "
            f"{best['ripple']:.5f} dB\n"
        )

        f.write(
            f"Worst passband S11 = "
            f"{best['s11_pb']:.5f} dB\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print()
print("=" * 70)
print("OPTIMIZATION FINISHED")
print("=" * 70)

print(
    f"Summary:"
)

print(
    summary_file
)

if best is not None:

    print()
    print(
        "BEST CANDIDATE"
    )

    print(
        f"Run        : "
        f"{best['run']}"
    )

    print(
        f"Score      : "
        f"{best['score']:.5f}"
    )

    print(
        f"f3dB       : "
        f"{best['f3db']:.3f} GHz"
    )

    print(
        f"S21 @20GHz: "
        f"{best['s21_20']:.3f} dB"
    )

    print()
    print(
        "Best geometry:"
    )

    print(
        f"h_thin  = "
        f"{best['x'][0]:.5f} mm"
    )

    print(
        f"h_thick = "
        f"{best['x'][1]:.5f} mm"
    )

    print(
        "Sections:"
    )

    for n, L in enumerate(
        best["section_lengths"],
        start=1
    ):

        print(
            f"  L{n} = {L:.5f} mm"
        )

    print()
    print(
        f"BEST folder:"
    )

    print(
        PROJECT_DIR / "BEST"
    )

print()
print("=" * 70)