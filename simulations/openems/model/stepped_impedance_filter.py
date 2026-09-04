import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ================================================================
# IMPORT USER PARAMETERS
# ================================================================

import parameters as P


# ================================================================
# OPENEMS DLL SETUP
# ================================================================

OPENEMS_ROOT = os.environ.get(
    "OPENEMS_INSTALL_PATH"
)

if OPENEMS_ROOT:
    os.add_dll_directory(
        OPENEMS_ROOT
    )


# ================================================================
# OPENEMS IMPORTS
# ================================================================

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ================================================================
# PATHS
# ================================================================

SIM_PATH = (
    PROJECT_ROOT
    / "simulations"
    / "openems"
    / "results"
    / P.OUTPUT_SUBDIRECTORY
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "em"
)

SIM_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


PLOT_PATH = (
    RESULTS_PATH
    / "stepped_impedance_filter_sparams.png"
)

CSV_PATH = (
    RESULTS_PATH
    / "stepped_impedance_filter_sparams.csv"
)


# ================================================================
# CONSTANTS
# ================================================================

C0 = 299792458.0


# ================================================================
# UNIT
# ================================================================

# Geometry is entered in millimetres.
UNIT = 1e-3


# ================================================================
# CONVERT PARAMETERS
# ================================================================

er = P.SUBSTRATE_EPSR

h = P.SUBSTRATE_THICKNESS_MM

copper_t = P.COPPER_THICKNESS_MM

fc = P.CUTOFF_FREQUENCY

Z0 = P.Z0

Z_HIGH = P.Z_HIGH

Z_LOW = P.Z_LOW


# ================================================================
# MICROSTRIP EFFECTIVE PERMITTIVITY
# ================================================================

def effective_epsilon(
    width_mm,
):
    """
    First-order Hammerstad-style effective
    dielectric constant approximation.
    """

    w_h = width_mm / h

    return (
        (er + 1.0) / 2.0
        +
        (er - 1.0) / 2.0
        *
        (
            1.0
            /
            np.sqrt(
                1.0
                +
                12.0 / w_h
            )
        )
    )


# ================================================================
# MICROSTRIP IMPEDANCE
# ================================================================

def microstrip_impedance(
    width_mm,
):
    """
    Approximate characteristic impedance
    using standard closed-form microstrip
    equations.

    This is used only for initial geometry synthesis.

    The final impedance is determined by
    full-wave EM simulation.
    """

    w_h = width_mm / h

    eps_eff = effective_epsilon(
        width_mm
    )

    if w_h <= 1.0:

        Z = (
            60.0
            /
            np.sqrt(eps_eff)
            *
            np.log(
                8.0 / w_h
                +
                w_h / 4.0
            )
        )

    else:

        Z = (
            120.0
            *
            np.pi
            /
            (
                np.sqrt(eps_eff)
                *
                (
                    w_h
                    +
                    1.393
                    +
                    0.667
                    *
                    np.log(
                        w_h + 1.444
                    )
                )
            )
        )

    return Z


# ================================================================
# SOLVE MICROSTRIP WIDTH
# ================================================================

def solve_width_for_impedance(
    target_Z,
):
    """
    Numerically determine microstrip width
    corresponding to target characteristic
    impedance.
    """

    low = 0.02

    high = 100.0

    Z_low_test = microstrip_impedance(
        low
    )

    Z_high_test = microstrip_impedance(
        high
    )

    if not (
        Z_high_test
        <
        target_Z
        <
        Z_low_test
    ):
        raise RuntimeError(
            "Target impedance is outside "
            "the numerical search range."
        )

    for _ in range(100):

        mid = (
            low + high
        ) / 2.0

        Z_mid = microstrip_impedance(
            mid
        )

        if Z_mid > target_Z:

            low = mid

        else:

            high = mid

    return (
        low + high
    ) / 2.0


# ================================================================
# CALCULATE WIDTHS
# ================================================================

if P.AUTO_CALCULATE_WIDTHS:

    width_high = (
        solve_width_for_impedance(
            Z_HIGH
        )
    )

    width_low = (
        solve_width_for_impedance(
            Z_LOW
        )
    )

else:

    width_high = (
        P.HIGH_Z_WIDTH_MM
    )

    width_low = (
        P.LOW_Z_WIDTH_MM
    )


if P.AUTO_CALCULATE_50OHM_WIDTH:

    width_50 = (
        solve_width_for_impedance(
            Z0
        )
    )

else:

    width_50 = (
        P.MANUAL_50OHM_WIDTH_MM
    )


# ================================================================
# DISPLAY WIDTHS
# ================================================================

print()
print("=" * 72)
print("STEPPED-IMPEDANCE FILTER DESIGN")
print("=" * 72)

print()

print(
    f"50-ohm width       : "
    f"{width_50:.4f} mm"
)

print(
    f"High-Z width       : "
    f"{width_high:.4f} mm"
)

print(
    f"Low-Z width        : "
    f"{width_low:.4f} mm"
)

print()

print(
    f"Calculated Z(50)   : "
    f"{microstrip_impedance(width_50):.2f} ohm"
)

print(
    f"Calculated Z(high) : "
    f"{microstrip_impedance(width_high):.2f} ohm"
)

print(
    f"Calculated Z(low)  : "
    f"{microstrip_impedance(width_low):.2f} ohm"
)


# ================================================================
# SIMPLE 3-SECTION PROTOTYPE
# ================================================================

# Normalized prototype values for a simple
# demonstration filter.
#
# These are NOT the final 7th-order Chebyshev values.
#
# g1 -> series inductive section
# g2 -> shunt capacitive section
# g3 -> series inductive section

g = [
    1.0,
    1.0,
    1.0,
]


# ================================================================
# CUTOFF ANGULAR FREQUENCY
# ================================================================

omega_c = (
    2.0
    *
    np.pi
    *
    fc
)


# ================================================================
# SECTION LENGTHS
# ================================================================

section_lengths = []


for index, gk in enumerate(g):

    section_number = (
        index + 1
    )

    if (
        section_number
        % 2
        == 1
    ):
        # --------------------------------------------------------
        # High impedance section.
        #
        # Short high-Z line:
        #
        #       L ≈ Z_H * theta / omega_c
        #
        # Therefore:
        #
        #       theta ≈ g_k * Z0 / Z_H
        # --------------------------------------------------------

        theta = (
            gk
            *
            Z0
            /
            Z_HIGH
        )

        width = width_high

    else:
        # --------------------------------------------------------
        # Low impedance section.
        #
        # Short low-Z line:
        #
        #       C ≈ theta / (omega_c Z_L)
        #
        # Therefore:
        #
        #       theta ≈ g_k * Z_L / Z0
        # --------------------------------------------------------

        theta = (
            gk
            *
            Z_LOW
            /
            Z0
        )

        width = width_low

    # Effective dielectric constant.
    eps_eff = (
        effective_epsilon(
            width
        )
    )

    # Guided wavelength.
    lambda_g = (
        C0
        /
        (
            fc
            *
            np.sqrt(
                eps_eff
            )
        )
        /
        UNIT
    )

    # Physical section length.
    length = (
        theta
        /
        (2.0 * np.pi)
        *
        lambda_g
    )

    section_lengths.append(
        length
    )


# ================================================================
# MANUAL SECTION LENGTH OVERRIDE
# ================================================================

if P.MANUAL_SECTION_LENGTHS:

    if len(
        P.MANUAL_SECTION_LENGTHS_MM
    ) != len(g):

        raise ValueError(
            "MANUAL_SECTION_LENGTHS_MM "
            "must contain exactly "
            "three values."
        )

    section_lengths = [
        float(x)
        for x in
        P.MANUAL_SECTION_LENGTHS_MM
    ]


# ================================================================
# PRINT SECTION DATA
# ================================================================

print()

print(
    "Distributed sections:"
)

for index, length in enumerate(
    section_lengths
):

    section_number = (
        index + 1
    )

    if (
        section_number
        % 2
        == 1
    ):

        width = width_high
        impedance = Z_HIGH

    else:

        width = width_low
        impedance = Z_LOW

    print(
        f"  Section {section_number}: "
        f"Z ≈ {impedance:.1f} ohm, "
        f"W = {width:.4f} mm, "
        f"L = {length:.4f} mm"
    )


# ================================================================
# TOTAL DUT LENGTH
# ================================================================

dut_length = (
    P.INPUT_LINE_LENGTH_MM
    +
    sum(section_lengths)
    +
    P.OUTPUT_LINE_LENGTH_MM
)


# ================================================================
# FDTD
# ================================================================

FDTD = openEMS(
    NrTS=P.MAX_TIMESTEPS,
    EndCriteria=P.END_CRITERIA,
)


# ================================================================
# EXCITATION
# ================================================================

if P.AUTO_CALCULATE_EXCITATION:

    excitation_f0 = (
        P.ANALYSIS_START_HZ
        +
        P.ANALYSIS_STOP_HZ
    ) / 2.0

    excitation_fc = (
        P.ANALYSIS_STOP_HZ
        -
        P.ANALYSIS_START_HZ
    ) / 2.0

else:

    excitation_f0 = (
        P.EXCITATION_F0_HZ
    )

    excitation_fc = (
        P.EXCITATION_FC_HZ
    )


FDTD.SetGaussExcite(
    excitation_f0,
    excitation_fc,
)


# ================================================================
# BOUNDARY CONDITIONS
# ================================================================

FDTD.SetBoundaryCond(
    P.BOUNDARY_CONDITIONS
)


# ================================================================
# CSXCAD
# ================================================================

CSX = ContinuousStructure()

FDTD.SetCSX(
    CSX
)


# ================================================================
# SUBSTRATE
# ================================================================

substrate = CSX.AddMaterial(
    "substrate",
    epsilon=er,
)

substrate.AddBox(
    start=[
        -P.SIMBOX_X_MM / 2.0,
        -P.SUBSTRATE_WIDTH_MM / 2.0,
        0.0,
    ],

    stop=[
        P.SIMBOX_X_MM / 2.0,
        P.SUBSTRATE_WIDTH_MM / 2.0,
        h,
    ],
)


# ================================================================
# GROUND
# ================================================================

ground = CSX.AddMetal(
    "ground"
)

ground.AddBox(
    start=[
        -P.SIMBOX_X_MM / 2.0,
        -P.SUBSTRATE_WIDTH_MM / 2.0,
        -copper_t,
    ],

    stop=[
        P.SIMBOX_X_MM / 2.0,
        P.SUBSTRATE_WIDTH_MM / 2.0,
        0.0,
    ],
)


# ================================================================
# FILTER METAL
# ================================================================

signal = CSX.AddMetal(
    "signal"
)


# ================================================================
# X POSITION TRACKING
# ================================================================

x = (
    -dut_length / 2.0
)


# ================================================================
# INPUT 50-OHM LINE
# ================================================================

signal.AddBox(
    start=[
        x,
        -width_50 / 2.0,
        h,
    ],

    stop=[
        x
        + P.INPUT_LINE_LENGTH_MM,
        width_50 / 2.0,
        h
        + copper_t,
    ],
)


x += (
    P.INPUT_LINE_LENGTH_MM
)


# ================================================================
# FILTER SECTIONS
# ================================================================

for index, length in enumerate(
    section_lengths
):

    section_number = (
        index + 1
    )

    if (
        section_number
        % 2
        == 1
    ):

        width = width_high

    else:

        width = width_low

    signal.AddBox(
        start=[
            x,
            -width / 2.0,
            h,
        ],

        stop=[
            x + length,
            width / 2.0,
            h + copper_t,
        ],
    )

    x += length


# ================================================================
# OUTPUT 50-OHM LINE
# ================================================================

signal.AddBox(
    start=[
        x,
        -width_50 / 2.0,
        h,
    ],

    stop=[
        x
        + P.OUTPUT_LINE_LENGTH_MM,
        width_50 / 2.0,
        h
        + copper_t,
    ],
)


# ================================================================
# PORT LOCATIONS
# ================================================================

port1_x = (
    -dut_length / 2.0
    +
    P.INPUT_LINE_LENGTH_MM / 2.0
)

port2_x = (
    dut_length / 2.0
    -
    P.OUTPUT_LINE_LENGTH_MM / 2.0
)


# ================================================================
# PORT 1
# ================================================================

port1 = FDTD.AddLumpedPort(
    port_nr=1,

    R=P.REFERENCE_IMPEDANCE,

    start=[
        port1_x,
        0.0,
        -copper_t,
    ],

    stop=[
        port1_x,
        0.0,
        h + copper_t,
    ],

    p_dir="z",

    excite=1,
)


# ================================================================
# PORT 2
# ================================================================

port2 = FDTD.AddLumpedPort(
    port_nr=2,

    R=P.REFERENCE_IMPEDANCE,

    start=[
        port2_x,
        0.0,
        -copper_t,
    ],

    stop=[
        port2_x,
        0.0,
        h + copper_t,
    ],

    p_dir="z",

    excite=0,
)


# ================================================================
# MESH
# ================================================================

mesh = CSX.GetGrid()

mesh.SetDeltaUnit(
    UNIT
)


# ================================================================
# WAVELENGTH RESOLUTION
# ================================================================

lambda_min = (
    C0
    /
    (
        excitation_f0
        +
        excitation_fc
    )
)

lambda_min_mm = (
    lambda_min
    /
    UNIT
)

coarse_step = (
    lambda_min_mm
    /
    P.MESH_WAVELENGTH_DIVISOR
)


# ================================================================
# X GEOMETRY EDGES
# ================================================================

x_edges = [
    -P.SIMBOX_X_MM / 2.0,
    -dut_length / 2.0,
    port1_x,
]


x_cursor = (
    -dut_length / 2.0
    +
    P.INPUT_LINE_LENGTH_MM
)

x_edges.append(
    x_cursor
)


for length in section_lengths:

    x_cursor += length

    x_edges.append(
        x_cursor
    )


x_edges.extend(
    [
        port2_x,
        dut_length / 2.0,
        P.SIMBOX_X_MM / 2.0,
    ]
)


x_edges = np.array(
    x_edges,
    dtype=float,
)


# ================================================================
# Y GEOMETRY EDGES
# ================================================================

y_edges = np.array(
    [
        -P.SIMBOX_Y_MM / 2.0,
        -P.SUBSTRATE_WIDTH_MM / 2.0,
        -width_low / 2.0,
        -width_50 / 2.0,
        0.0,
        width_50 / 2.0,
        width_low / 2.0,
        P.SUBSTRATE_WIDTH_MM / 2.0,
        P.SIMBOX_Y_MM / 2.0,
    ],
    dtype=float,
)


# ================================================================
# Z GEOMETRY EDGES
# ================================================================

z_edges = np.array(
    [
        -P.SIMBOX_Z_MM / 3.0,
        -copper_t,
        0.0,
        h,
        h + copper_t,
        P.SIMBOX_Z_MM * 2.0 / 3.0,
    ],
    dtype=float,
)


# ================================================================
# UNIFORM X GRID
# ================================================================

x_dense = np.linspace(
    -P.SIMBOX_X_MM / 2.0,
    P.SIMBOX_X_MM / 2.0,
    int(
        np.ceil(
            P.SIMBOX_X_MM
            /
            coarse_step
        )
    ) + 1,
)

x_dense = np.unique(
    np.sort(
        np.concatenate(
            [
                x_dense,
                x_edges,
            ]
        )
    )
)


# ================================================================
# UNIFORM Y GRID
# ================================================================

y_dense = np.linspace(
    -P.SIMBOX_Y_MM / 2.0,
    P.SIMBOX_Y_MM / 2.0,
    int(
        np.ceil(
            P.SIMBOX_Y_MM
            /
            coarse_step
        )
    ) + 1,
)

y_dense = np.unique(
    np.sort(
        np.concatenate(
            [
                y_dense,
                y_edges,
            ]
        )
    )
)


# ================================================================
# Z GRID
# ================================================================

z_lower = np.linspace(
    -P.SIMBOX_Z_MM / 3.0,
    -copper_t,
    max(
        2,
        int(
            np.ceil(
                (
                    P.SIMBOX_Z_MM / 3.0
                    -
                    copper_t
                )
                /
                coarse_step
            )
        )
        + 1,
    ),
)


z_ground = np.linspace(
    -copper_t,
    0.0,
    P.COPPER_MESH_CELLS + 1,
)


z_substrate = np.linspace(
    0.0,
    h,
    P.SUBSTRATE_MESH_CELLS + 1,
)


z_signal = np.linspace(
    h,
    h + copper_t,
    P.COPPER_MESH_CELLS + 1,
)


z_upper = np.linspace(
    h + copper_t,
    P.SIMBOX_Z_MM * 2.0 / 3.0,
    max(
        2,
        int(
            np.ceil(
                (
                    P.SIMBOX_Z_MM * 2.0 / 3.0
                    -
                    h
                    -
                    copper_t
                )
                /
                coarse_step
            )
        )
        + 1,
    ),
)


z_dense = np.unique(
    np.sort(
        np.concatenate(
            [
                z_lower,
                z_ground,
                z_substrate,
                z_signal,
                z_upper,
                z_edges,
            ]
        )
    )
)


# ================================================================
# SET GRID
# ================================================================

mesh.SetLines(
    "x",
    x_dense,
)

mesh.SetLines(
    "y",
    y_dense,
)

mesh.SetLines(
    "z",
    z_dense,
)


# ================================================================
# REPORT
# ================================================================

print()

print(
    f"Simulation directory : "
    f"{SIM_PATH}"
)

print(
    f"Cutoff frequency     : "
    f"{fc / 1e9:.3f} GHz"
)

print(
    f"Analysis range       : "
    f"{P.ANALYSIS_START_HZ / 1e9:.3f}"
    f" – "
    f"{P.ANALYSIS_STOP_HZ / 1e9:.3f} GHz"
)

print()

print(
    f"Excitation f0        : "
    f"{excitation_f0 / 1e9:.3f} GHz"
)

print(
    f"Excitation fc        : "
    f"{excitation_fc / 1e9:.3f} GHz"
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
    "Writing geometry..."
)


# ================================================================
# WRITE XML
# ================================================================

xml_path = (
    SIM_PATH
    /
    "stepped_impedance_filter.xml"
)

if xml_path.exists():

    xml_path.unlink()


CSX.Write2XML(
    str(xml_path)
)


print(
    f"XML                  : "
    f"{xml_path}"
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
# FREQUENCY VECTOR
# ================================================================

freq = np.linspace(
    P.ANALYSIS_START_HZ,
    P.ANALYSIS_STOP_HZ,
    P.NUMBER_OF_FREQUENCY_POINTS,
)


# ================================================================
# CALCULATE PORTS
# ================================================================

print()
print(
    "Calculating Port 1..."
)

port1.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=P.REFERENCE_IMPEDANCE,
)


print(
    "Calculating Port 2..."
)

port2.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=P.REFERENCE_IMPEDANCE,
)


# ================================================================
# EXTRACT WAVES
# ================================================================

uf_inc_1 = np.asarray(
    port1.uf_inc
)

uf_ref_1 = np.asarray(
    port1.uf_ref
)

uf_ref_2 = np.asarray(
    port2.uf_ref
)


# ================================================================
# VALIDITY
# ================================================================

if not np.all(
    np.isfinite(
        uf_inc_1
    )
):

    raise RuntimeError(
        "Port 1 incident wave "
        "contains NaN or Inf."
    )


if not np.all(
    np.isfinite(
        uf_ref_1
    )
):

    raise RuntimeError(
        "Port 1 reflected wave "
        "contains NaN or Inf."
    )


if not np.all(
    np.isfinite(
        uf_ref_2
    )
):

    raise RuntimeError(
        "Port 2 wave "
        "contains NaN or Inf."
    )


# ================================================================
# INCIDENT-WAVE DIAGNOSTIC
# ================================================================

print()

print(
    "=" * 72
)

print(
    "PORT WAVE DIAGNOSTIC"
)

print(
    "=" * 72
)

print(
    f"max |uf_inc_1| : "
    f"{np.max(np.abs(uf_inc_1)):.6e}"
)

print(
    f"max |uf_ref_1| : "
    f"{np.max(np.abs(uf_ref_1)):.6e}"
)

print(
    f"max |uf_ref_2| : "
    f"{np.max(np.abs(uf_ref_2)):.6e}"
)

print(
    "=" * 72
)


if np.max(
    np.abs(
        uf_inc_1
    )
) == 0.0:

    raise RuntimeError(
        "Port 1 incident wave "
        "is exactly zero."
    )


# ================================================================
# S-PARAMETERS
# ================================================================

s11 = (
    uf_ref_1
    /
    uf_inc_1
)

s21 = (
    uf_ref_2
    /
    uf_inc_1
)


# ================================================================
# MAGNITUDE dB
# ================================================================

s11_db = (
    20.0
    *
    np.log10(
        np.maximum(
            np.abs(s11),
            1e-15,
        )
    )
)


s21_db = (
    20.0
    *
    np.log10(
        np.maximum(
            np.abs(s21),
            1e-15,
        )
    )
)


# ================================================================
# FINITE CHECK
# ================================================================

if not np.all(
    np.isfinite(
        s11_db
    )
):

    raise RuntimeError(
        "S11 contains NaN or Inf."
    )


if not np.all(
    np.isfinite(
        s21_db
    )
):

    raise RuntimeError(
        "S21 contains NaN or Inf."
    )


# ================================================================
# SAVE CSV
# ================================================================

np.savetxt(
    CSV_PATH,

    np.column_stack(
        [
            freq / 1e9,
            s11_db,
            s21_db,
        ]
    ),

    delimiter=",",

    header=(
        "frequency_GHz,"
        "S11_dB,"
        "S21_dB"
    ),

    comments="",
)


# ================================================================
# PLOT
# ================================================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    freq / 1e9,
    s11_db,
    linewidth=1.5,
    label="S11",
)

plt.plot(
    freq / 1e9,
    s21_db,
    linewidth=1.5,
    label="S21",
)

plt.axvline(
    fc / 1e9,
    linestyle=":",
    linewidth=1.2,
    label="Nominal cutoff",
)

plt.axhline(
    -10.0,
    linestyle="--",
    linewidth=1.0,
)

plt.xlabel(
    "Frequency (GHz)"
)

plt.ylabel(
    "Magnitude (dB)"
)

plt.title(
    "2-Port Stepped-Impedance "
    "Distributed Low-Pass Filter"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.xlim(
    P.ANALYSIS_START_HZ / 1e9,
    P.ANALYSIS_STOP_HZ / 1e9,
)

plt.ylim(
    -80,
    5,
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=200,
)

plt.close()


# ================================================================
# SUMMARY
# ================================================================

print()

print(
    "=" * 72
)

print(
    "FILTER RESULT"
)

print(
    "=" * 72
)

print()

print(
    f"Maximum S11 : "
    f"{np.max(s11_db):.3f} dB"
)

print(
    f"Minimum S11 : "
    f"{np.min(s11_db):.3f} dB"
)

print()

print(
    f"Maximum S21 : "
    f"{np.max(s21_db):.3f} dB"
)

print(
    f"Minimum S21 : "
    f"{np.min(s21_db):.3f} dB"
)

print()

print(
    f"CSV         : "
    f"{CSV_PATH}"
)

print(
    f"Plot        : "
    f"{PLOT_PATH}"
)

print()

print(
    "=" * 72
)

print(
    "SIMULATION COMPLETE"
)

print(
    "=" * 72
)