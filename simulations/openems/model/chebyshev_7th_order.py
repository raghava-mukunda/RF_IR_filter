# ================================================================
# 7th-Order Chebyshev Stepped-Impedance Distributed Low-Pass Filter
#
# Full-wave 3-D FDTD simulation using openEMS
#
# IMPORTANT
# ----------
# This is an INITIAL distributed realization of the validated
# 7th-order Chebyshev Type-I prototype.
#
# The stepped-impedance dimensions are an initial synthesis.
# They must be tuned after the first full-wave simulation.
#
# Coordinate convention
# ---------------------
# openEMS geometry is expressed in MILLIMETRES, using:
#     GRID.SetDeltaUnit(1e-3)
#
# This intentionally follows the unit convention of the known-good
# two-port microstrip diagnostic used to validate the lumped ports.
#
# RF electrical quantities remain SI units (Hz, metres, ohms).
# ================================================================

import os
import sys
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# OPENEMS / CSXCAD DLL SETUP
# ================================================================

OPENEMS_ROOT = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
    / "openEMS"
)

if not OPENEMS_ROOT.exists():
    raise FileNotFoundError(
        f"openEMS installation not found:\n{OPENEMS_ROOT}"
    )

os.add_dll_directory(str(OPENEMS_ROOT))

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["PATH"] = (
    str(OPENEMS_ROOT)
    + os.pathsep
    + os.environ["PATH"]
)


# ================================================================
# IMPORTS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import parameters

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ================================================================
# CONSTANTS
# ================================================================

C0 = 299792458.0
MM = 1e3


# ================================================================
# MICROSTRIP CALCULATIONS
# ================================================================

def effective_epsilon(width_m, height_m, eps_r):
    """Hammerstad-type approximation for microstrip effective epsilon."""

    u = width_m / height_m

    if u <= 1.0:
        return (
            (eps_r + 1.0) / 2.0
            + (eps_r - 1.0) / 2.0
            * (
                1.0 / np.sqrt(1.0 + 12.0 / u)
                + 0.04 * (1.0 - u) ** 2
            )
        )

    return (
        (eps_r + 1.0) / 2.0
        + (eps_r - 1.0) / 2.0
        * (1.0 / np.sqrt(1.0 + 12.0 / u))
    )


def microstrip_impedance(width_m, height_m, eps_r):
    """Approximate characteristic impedance of a microstrip."""

    u = width_m / height_m
    eps_eff = effective_epsilon(width_m, height_m, eps_r)

    if u <= 1.0:
        return (
            60.0 / np.sqrt(eps_eff)
            * np.log(8.0 / u + u / 4.0)
        )

    return (
        120.0 * np.pi
        / (
            np.sqrt(eps_eff)
            * (
                u
                + 1.393
                + 0.667 * np.log(u + 1.444)
            )
        )
    )


def solve_microstrip_width(target_impedance, height_m, eps_r):
    """Solve microstrip width using bisection."""

    low = 0.01 * height_m
    high = 100.0 * height_m

    z_low = microstrip_impedance(low, height_m, eps_r)
    z_high = microstrip_impedance(high, height_m, eps_r)

    if not (z_low > target_impedance and z_high < target_impedance):
        raise RuntimeError(
            "Microstrip impedance solver bracket is invalid."
        )

    for _ in range(100):
        mid = 0.5 * (low + high)
        z_mid = microstrip_impedance(mid, height_m, eps_r)

        if z_mid > target_impedance:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def uniform_mesh_lines(minimum_mm, maximum_mm, maximum_spacing_mm):
    """Generate approximately uniform mesh lines in millimetres."""

    intervals = int(
        np.ceil(
            (maximum_mm - minimum_mm)
            / maximum_spacing_mm
        )
    )

    return np.linspace(
        minimum_mm,
        maximum_mm,
        intervals + 1,
    )


# ================================================================
# VALIDATE PARAMETERS
# ================================================================

if parameters.FILTER_ORDER != 7:
    raise ValueError("This model requires FILTER_ORDER = 7.")

if len(parameters.CHEBYSHEV_G) != 7:
    raise ValueError(
        "CHEBYSHEV_G must contain exactly 7 coefficients."
    )

if parameters.Z_HIGH <= parameters.Z0:
    raise ValueError("Z_HIGH must be greater than Z0.")

if parameters.Z_LOW >= parameters.Z0:
    raise ValueError("Z_LOW must be less than Z0.")

if parameters.ANALYSIS_STOP <= parameters.ANALYSIS_START:
    raise ValueError("Invalid analysis frequency range.")


# ================================================================
# INITIAL MICROSTRIP WIDTHS
# ================================================================
# These are calculated in metres, then converted to mm only for
# openEMS geometry.

W50_m = solve_microstrip_width(
    parameters.Z0,
    parameters.SUBSTRATE_HEIGHT,
    parameters.EPS_R,
)

W_HIGH_m = solve_microstrip_width(
    parameters.Z_HIGH,
    parameters.SUBSTRATE_HEIGHT,
    parameters.EPS_R,
)

W_LOW_m = solve_microstrip_width(
    parameters.Z_LOW,
    parameters.SUBSTRATE_HEIGHT,
    parameters.EPS_R,
)

W50 = W50_m * MM
W_HIGH = W_HIGH_m * MM
W_LOW = W_LOW_m * MM

H = parameters.SUBSTRATE_HEIGHT * MM
T_CU = parameters.COPPER_THICKNESS * MM


# ================================================================
# DISTRIBUTED CHEBYSHEV SYNTHESIS
# ================================================================

section_types = []
section_impedances = []
section_widths = []
section_lengths = []
section_theta = []
section_eps_eff = []

for index, g in enumerate(parameters.CHEBYSHEV_G):

    if index % 2 == 0:
        # Series-inductive sections are implemented as HIGH-Z lines.
        section_type = "HIGH"
        impedance = parameters.Z_HIGH
        width_m = W_HIGH_m

        theta = (
            g
            * parameters.Z0
            / parameters.Z_HIGH
        )

    else:
        # Shunt-capacitive sections are implemented as LOW-Z lines.
        section_type = "LOW"
        impedance = parameters.Z_LOW
        width_m = W_LOW_m

        theta = (
            g
            * parameters.Z_LOW
            / parameters.Z0
        )

    eps_eff = effective_epsilon(
        width_m,
        parameters.SUBSTRATE_HEIGHT,
        parameters.EPS_R,
    )

    lambda_g_m = (
        C0
        / (
            parameters.CUTOFF_FREQUENCY
            * np.sqrt(eps_eff)
        )
    )

    physical_length_m = (
        theta
        / (2.0 * np.pi)
        * lambda_g_m
    )

    section_types.append(section_type)
    section_impedances.append(impedance)
    section_widths.append(width_m * MM)
    section_lengths.append(physical_length_m * MM)
    section_theta.append(theta)
    section_eps_eff.append(eps_eff)


FILTER_LENGTH = sum(section_lengths)

INPUT_LENGTH = parameters.INPUT_LINE_LENGTH * MM
OUTPUT_LENGTH = parameters.OUTPUT_LINE_LENGTH * MM

TOTAL_LENGTH = (
    INPUT_LENGTH
    + FILTER_LENGTH
    + OUTPUT_LENGTH
)


# ================================================================
# X GEOMETRY -- MILLIMETRES
# ================================================================

x0 = -TOTAL_LENGTH / 2.0
x1 = +TOTAL_LENGTH / 2.0

x_input_end = x0 + INPUT_LENGTH
x_filter_start = x_input_end

x_current = x_filter_start
section_x_boundaries = [x_filter_start]

for length_mm in section_lengths:
    x_current += length_mm
    section_x_boundaries.append(x_current)

x_filter_end = x_current
x_output_start = x_filter_end


# ================================================================
# PORT LOCATIONS
# ================================================================
# Ports are deliberately placed inside the 50-ohm input/output
# feed sections, away from the impedance discontinuities.
#
# This is the same physical construction that was validated by
# test_filter_port.py.

port1_x = x0 + 0.5 * INPUT_LENGTH
port2_x = x_output_start + 0.5 * OUTPUT_LENGTH
port_y = 0.0


# ================================================================
# Z GEOMETRY -- MILLIMETRES
# ================================================================
#
# ground     : -T_CU ... 0
# substrate  :  0     ... H
# signal     :  H     ... H+T_CU
#
# The lumped ports span the entire dielectric/metal stack from the
# bottom of the ground conductor to the top of the signal conductor.

z_ground_bottom = -T_CU
z_ground_top = 0.0
z_substrate_top = H
z_signal_top = H + T_CU


# ================================================================
# SIMULATION BOX -- MILLIMETRES
# ================================================================

SIM_X = (
    TOTAL_LENGTH
    + 2.0 * parameters.SIM_BOX_X_MARGIN * MM
)

SIM_Y = (
    parameters.SUBSTRATE_WIDTH * MM
    + 2.0 * parameters.SIM_BOX_Y_MARGIN * MM
)

SIM_Z = parameters.SIM_BOX_Z * MM

z_sim_min = -SIM_Z / 2.0
z_sim_max = +SIM_Z / 2.0


# ================================================================
# SIMULATION DIRECTORY
# ================================================================

SIM_PATH = Path(parameters.SIMULATION_DIRECTORY)

if parameters.CLEAN_SIMULATION_DIRECTORY:
    if SIM_PATH.exists():
        shutil.rmtree(SIM_PATH)

SIM_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# PRINT DESIGN INFORMATION
# ================================================================

print()
print("=" * 78)
print("7th-ORDER CHEBYSHEV STEPPED-IMPEDANCE LOW-PASS")
print("=" * 78)

print()
print("ELECTRICAL SPECIFICATION")
print("-" * 78)
print(f"Z0                     : {parameters.Z0:.2f} ohm")
print(
    f"Cutoff frequency       : "
    f"{parameters.CUTOFF_FREQUENCY / 1e9:.3f} GHz"
)
print(
    f"Passband ripple        : "
    f"{parameters.PASSBAND_RIPPLE_DB:.3f} dB"
)
print(
    f"Stopband start         : "
    f"{parameters.STOPBAND_START / 1e9:.3f} GHz"
)
print(
    f"Required attenuation   : "
    f"{parameters.STOPBAND_ATTENUATION_DB:.1f} dB"
)
print(f"Filter order            : {parameters.FILTER_ORDER}")

print()
print("MICROSTRIP DIMENSIONS")
print("-" * 78)
print(f"50-ohm width           : {W50:.4f} mm")
print(
    f"High-Z width           : {W_HIGH:.4f} mm"
    f"  ({parameters.Z_HIGH:.1f} ohm)"
)
print(
    f"Low-Z width            : {W_LOW:.4f} mm"
    f"  ({parameters.Z_LOW:.1f} ohm)"
)

print()
print("FILTER SECTIONS")
print("-" * 78)
print(
    f"{'Sec':>4}"
    f"{'Type':>8}"
    f"{'g':>12}"
    f"{'Z':>10}"
    f"{'W(mm)':>12}"
    f"{'theta':>12}"
    f"{'L(mm)':>12}"
)

print("-" * 78)

for i in range(7):
    print(
        f"{i + 1:4d}"
        f"{section_types[i]:>8}"
        f"{parameters.CHEBYSHEV_G[i]:12.6f}"
        f"{section_impedances[i]:10.2f}"
        f"{section_widths[i]:12.4f}"
        f"{section_theta[i]:12.6f}"
        f"{section_lengths[i]:12.4f}"
    )

print("-" * 78)
print(f"Filter length          : {FILTER_LENGTH:.4f} mm")
print(f"Total RF length        : {TOTAL_LENGTH:.4f} mm")
print(f"Port 1 x               : {port1_x:.4f} mm")
print(f"Port 2 x               : {port2_x:.4f} mm")


# ================================================================
# OPENEMS SOLVER
# ================================================================

FDTD = openEMS(
    NrTS=parameters.NR_TS,
    EndCriteria=1e-5,
)

FDTD.SetGaussExcite(
    parameters.F0,
    parameters.FC,
)

FDTD.SetBoundaryCond(
    parameters.BOUNDARY_CONDITIONS
)


# ================================================================
# CSXCAD STRUCTURE
# ================================================================

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

GRID = CSX.GetGrid()

# IMPORTANT:
# Geometry values above are in millimetres.
# openEMS interprets one geometry unit as 1e-3 metres.
GRID.SetDeltaUnit(1e-3)


# ================================================================
# SUBSTRATE
# ================================================================

SUBSTRATE = CSX.AddMaterial(
    "Substrate",
    epsilon=parameters.EPS_R,
)

SUBSTRATE.AddBox(
    priority=0,
    start=[
        x0,
        -SIM_Y / 2.0,
        z_ground_top,
    ],
    stop=[
        x1,
        +SIM_Y / 2.0,
        z_substrate_top,
    ],
)


# ================================================================
# GROUND
# ================================================================

GROUND = CSX.AddMetal("Ground")

GROUND.AddBox(
    priority=1,
    start=[
        x0,
        -SIM_Y / 2.0,
        z_ground_bottom,
    ],
    stop=[
        x1,
        +SIM_Y / 2.0,
        z_ground_top,
    ],
)


# ================================================================
# SIGNAL METAL
# ================================================================

SIGNAL = CSX.AddMetal("Signal")


# ================================================================
# INPUT 50-OHM LINE
# ================================================================

SIGNAL.AddBox(
    priority=10,
    start=[
        x0,
        -W50 / 2.0,
        z_substrate_top,
    ],
    stop=[
        x_input_end,
        +W50 / 2.0,
        z_signal_top,
    ],
)


# ================================================================
# SEVEN STEPPED SECTIONS
# ================================================================

x_current = x_filter_start

for i in range(7):

    x_next = x_current + section_lengths[i]

    SIGNAL.AddBox(
        priority=10,
        start=[
            x_current,
            -section_widths[i] / 2.0,
            z_substrate_top,
        ],
        stop=[
            x_next,
            +section_widths[i] / 2.0,
            z_signal_top,
        ],
    )

    x_current = x_next


# ================================================================
# OUTPUT 50-OHM LINE
# ================================================================

SIGNAL.AddBox(
    priority=10,
    start=[
        x_output_start,
        -W50 / 2.0,
        z_substrate_top,
    ],
    stop=[
        x1,
        +W50 / 2.0,
        z_signal_top,
    ],
)


# ================================================================
# PORT GRID ALIGNMENT
#
# openEMS' documented lumped-port examples explicitly promote the
# port priority and add its transverse edges to the FDTD grid.
# This is important here because the port overlaps the substrate
# and both conductors.
# ================================================================
FDTD.AddEdges2Grid(
    dirs="xy",
    properties=GROUND,
)

FDTD.AddEdges2Grid(
    dirs="xy",
    properties=SIGNAL,
)


# ================================================================
# PORT 1
# ================================================================
# Proven working two-port construction:
#   x = inside 50-ohm feed
#   y = 0
#   z = bottom of ground -> top of signal
#   p_dir = z
# ================================================================

# Use a finite transverse port face covering the complete
# 50-ohm microstrip width.  This is more robust than a degenerate
# line-like port and still represents the same vertical voltage
# excitation between signal and ground.
PORT1 = FDTD.AddLumpedPort(
    port_nr=1,
    R=parameters.PORT_IMPEDANCE,
    start=[
        port1_x,
        -W50 / 2.0,
        z_ground_bottom,
    ],
    stop=[
        port1_x,
        +W50 / 2.0,
        z_signal_top,
    ],
    p_dir="z",
    excite=1,
    priority=50,
    edges2grid="xy",
)


# ================================================================
# PORT 2
# ================================================================

PORT2 = FDTD.AddLumpedPort(
    port_nr=2,
    R=parameters.PORT_IMPEDANCE,
    start=[
        port2_x,
        -W50 / 2.0,
        z_ground_bottom,
    ],
    stop=[
        port2_x,
        +W50 / 2.0,
        z_signal_top,
    ],
    p_dir="z",
    excite=0,
    priority=50,
    edges2grid="xy",
)


# ================================================================
# X MESH
# ================================================================

MAX_MESH_MM = parameters.MAX_MESH_XY * MM

x_uniform = uniform_mesh_lines(
    -SIM_X / 2.0,
    +SIM_X / 2.0,
    MAX_MESH_MM,
)

x_special = np.array(
    [
        -SIM_X / 2.0,
        x0,
        port1_x,
        x_input_end,
        x_filter_start,
        *section_x_boundaries,
        x_filter_end,
        x_output_start,
        port2_x,
        x1,
        +SIM_X / 2.0,
    ],
    dtype=float,
)

x_lines = np.unique(
    np.round(
        np.concatenate([x_uniform, x_special]),
        9,
    )
)

GRID.SetLines("x", x_lines.tolist())


# ================================================================
# Y MESH
# ================================================================

y_uniform = uniform_mesh_lines(
    -SIM_Y / 2.0,
    +SIM_Y / 2.0,
    MAX_MESH_MM,
)

y_special = np.array(
    [
        -SIM_Y / 2.0,
        -W_LOW / 2.0,
        -W50 / 2.0,
        -W_HIGH / 2.0,
        0.0,
        +W_HIGH / 2.0,
        +W50 / 2.0,
        +W_LOW / 2.0,
        +SIM_Y / 2.0,
    ],
    dtype=float,
)

y_lines = np.unique(
    np.round(
        np.concatenate([y_uniform, y_special]),
        9,
    )
)

GRID.SetLines("y", y_lines.tolist())


# ================================================================
# Z MESH
# ================================================================

z_ground_lines = np.linspace(
    z_ground_bottom,
    z_ground_top,
    parameters.COPPER_CELLS_Z + 1,
)

z_substrate_lines = np.linspace(
    z_ground_top,
    z_substrate_top,
    parameters.SUBSTRATE_CELLS_Z + 1,
)

z_signal_lines = np.linspace(
    z_substrate_top,
    z_signal_top,
    parameters.COPPER_CELLS_Z + 1,
)

# Air below ground
z_air_lower = uniform_mesh_lines(
    z_sim_min,
    z_ground_bottom,
    MAX_MESH_MM,
)

# Air above signal
z_air_upper = uniform_mesh_lines(
    z_signal_top,
    z_sim_max,
    MAX_MESH_MM,
)

z_lines = np.unique(
    np.round(
        np.concatenate(
            [
                z_air_lower,
                z_ground_lines,
                z_substrate_lines,
                z_signal_lines,
                z_air_upper,
            ]
        ),
        9,
    )
)

GRID.SetLines("z", z_lines.tolist())


# ================================================================
# WRITE XML
# ================================================================

SIM_CSX = SIM_PATH / "chebyshev_7th_order.xml"

print()
print("Writing openEMS XML...")

CSX.Write2XML(str(SIM_CSX))


# ================================================================
# SIMULATION REPORT
# ================================================================

print()
print("=" * 78)
print("SIMULATION")
print("=" * 78)
print(f"Simulation directory : {SIM_PATH}")
print(
    f"Simulation box       : "
    f"{SIM_X:.3f} x {SIM_Y:.3f} x {SIM_Z:.3f} mm"
)
print(
    f"Mesh lines           : "
    f"{len(x_lines)} x {len(y_lines)} x {len(z_lines)}"
)
print(f"Maximum timesteps    : {parameters.NR_TS}")

print()
print("Starting openEMS...")
print()


# ================================================================
# RUN
# ================================================================

if parameters.RUN_SIMULATION:
    FDTD.Run(
        str(SIM_PATH),
        cleanup=False,
        verbose=2,
    )


# ================================================================
# FREQUENCY AXIS
# ================================================================

freq = np.linspace(
    parameters.ANALYSIS_START,
    parameters.ANALYSIS_STOP,
    1001,
)


# ================================================================
# PORT POSTPROCESSING
# ================================================================

print()
print("=" * 78)
print("POST-PROCESSING")
print("=" * 78)

print("Calculating Port 1...")

PORT1.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=parameters.PORT_IMPEDANCE,
)

print("Calculating Port 2...")

PORT2.CalcPort(
    str(SIM_PATH),
    freq,
    ref_impedance=parameters.PORT_IMPEDANCE,
)


# ================================================================
# EXTRACT WAVES
# ================================================================

uf_inc_1 = np.asarray(PORT1.uf_inc)
uf_ref_1 = np.asarray(PORT1.uf_ref)
uf_ref_2 = np.asarray(PORT2.uf_ref)


# ================================================================
# PORT DIAGNOSTICS
# ================================================================

max_incident = np.max(np.abs(uf_inc_1))
max_reflected_1 = np.max(np.abs(uf_ref_1))
max_reflected_2 = np.max(np.abs(uf_ref_2))

print()
print("PORT DIAGNOSTICS")
print("-" * 78)
print(f"Maximum |Port 1 incident| : {max_incident:.6e}")
print(f"Maximum |Port 1 reflected|: {max_reflected_1:.6e}")
print(f"Maximum |Port 2 response| : {max_reflected_2:.6e}")

if not np.all(np.isfinite(uf_inc_1)):
    raise RuntimeError("Port 1 incident wave contains NaN or Inf.")

if not np.all(np.isfinite(uf_ref_1)):
    raise RuntimeError("Port 1 reflected wave contains NaN or Inf.")

if not np.all(np.isfinite(uf_ref_2)):
    raise RuntimeError("Port 2 response contains NaN or Inf.")

if max_incident == 0.0:
    raise RuntimeError(
        "Port 1 incident wave is exactly zero. "
        "The excitation was not activated. "
        "Check the port face against the signal/ground mesh."
    )

print("Port 1 incident wave exists.")


# ================================================================
# S-PARAMETERS
# ================================================================

S11 = uf_ref_1 / uf_inc_1
S21 = uf_ref_2 / uf_inc_1

S11_dB = 20.0 * np.log10(
    np.maximum(np.abs(S11), 1e-15)
)

S21_dB = 20.0 * np.log10(
    np.maximum(np.abs(S21), 1e-15)
)


# ================================================================
# SAVE RESULTS
# ================================================================

results_dir = Path(parameters.RESULTS_DIRECTORY)
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / "chebyshev_7th_order_sparams.csv"

np.savetxt(
    csv_path,
    np.column_stack(
        [
            freq,
            np.abs(S11),
            np.abs(S21),
            S11_dB,
            S21_dB,
        ]
    ),
    delimiter=",",
    header=(
        "frequency_Hz,"
        "S11_abs,"
        "S21_abs,"
        "S11_dB,"
        "S21_dB"
    ),
    comments="",
)


# ================================================================
# PERFORMANCE METRICS
# ================================================================

passband_mask = freq <= parameters.CUTOFF_FREQUENCY
stopband_mask = freq >= parameters.STOPBAND_START

passband_s21 = S21_dB[passband_mask]
stopband_s21 = S21_dB[stopband_mask]

# Loss is positive when S21 is below 0 dB.
max_passband_loss = -np.min(passband_s21)

S21_at_20 = np.interp(
    parameters.STOPBAND_START,
    freq,
    S21_dB,
)

attenuation_at_20 = -S21_at_20

worst_stopband_s21 = np.max(stopband_s21)
worst_stopband_attenuation = -worst_stopband_s21


# ================================================================
# APPROXIMATE -3 dB CUTOFF
# ================================================================

cutoff_indices = np.where(S21_dB <= -3.0)[0]

if len(cutoff_indices) > 0:
    measured_cutoff = freq[cutoff_indices[0]]
else:
    measured_cutoff = np.nan


# ================================================================
# SPECIFICATION CHECK
# ================================================================

passband_ok = (
    max_passband_loss
    <= parameters.PASSBAND_RIPPLE_DB
)

attenuation_20_ok = (
    attenuation_at_20
    >= parameters.STOPBAND_ATTENUATION_DB
)

worst_stopband_ok = (
    worst_stopband_attenuation
    >= parameters.STOPBAND_ATTENUATION_DB
)


# ================================================================
# REPORT
# ================================================================

print()
print("=" * 78)
print("FULL-WAVE PERFORMANCE")
print("=" * 78)

print(
    f"Maximum passband loss    : "
    f"{max_passband_loss:.3f} dB"
)

print(
    f"S21 at 20 GHz            : "
    f"{S21_at_20:.3f} dB"
)

print(
    f"Attenuation at 20 GHz    : "
    f"{attenuation_at_20:.3f} dB"
)

print(
    f"Worst S21, "
    f"{parameters.STOPBAND_START / 1e9:.1f}-"
    f"{parameters.ANALYSIS_STOP / 1e9:.1f} GHz"
    f" : {worst_stopband_s21:.3f} dB"
)

print(
    f"Worst stopband attenuation: "
    f"{worst_stopband_attenuation:.3f} dB"
)

if np.isfinite(measured_cutoff):
    print(
        f"First -3 dB frequency     : "
        f"{measured_cutoff / 1e9:.3f} GHz"
    )
else:
    print(
        "First -3 dB frequency     : "
        "not reached"
    )


# ================================================================
# SPECIFICATION CHECK
# ================================================================

print()
print("SPECIFICATION CHECK")
print("-" * 78)

print(
    f"Passband loss <= "
    f"{parameters.PASSBAND_RIPPLE_DB:.2f} dB : "
    f"{'PASS' if passband_ok else 'FAIL'}"
)

print(
    f"Attenuation @ 20 GHz >= "
    f"{parameters.STOPBAND_ATTENUATION_DB:.1f} dB : "
    f"{'PASS' if attenuation_20_ok else 'FAIL'}"
)

print(
    f"Worst stopband attenuation >= "
    f"{parameters.STOPBAND_ATTENUATION_DB:.1f} dB : "
    f"{'PASS' if worst_stopband_ok else 'FAIL'}"
)


# ================================================================
# PLOT
# ================================================================

plot_dir = Path(parameters.PLOT_DIRECTORY)
plot_dir.mkdir(parents=True, exist_ok=True)

plot_path = plot_dir / "chebyshev_7th_order_em.png"

plt.figure(figsize=(11, 6))

plt.plot(
    freq / 1e9,
    S11_dB,
    label="S11",
)

plt.plot(
    freq / 1e9,
    S21_dB,
    label="S21",
)

plt.axvline(
    parameters.CUTOFF_FREQUENCY / 1e9,
    linestyle=":",
    label="9 GHz cutoff",
)

plt.axvline(
    parameters.STOPBAND_START / 1e9,
    linestyle=":",
    label="20 GHz stopband",
)

plt.axhline(
    -parameters.STOPBAND_ATTENUATION_DB,
    linestyle="--",
    label="-60 dB requirement",
)

plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.title(
    "7th-Order Chebyshev Stepped-Impedance "
    "Distributed Low-Pass Filter"
)

plt.grid(True, alpha=0.3)
plt.legend()

plt.xlim(
    parameters.ANALYSIS_START / 1e9,
    parameters.ANALYSIS_STOP / 1e9,
)

plt.ylim(-80, 5)

plt.tight_layout()
plt.savefig(plot_path, dpi=200)
plt.close()


# ================================================================
# FINAL
# ================================================================

print()
print("=" * 78)
print("OUTPUT FILES")
print("=" * 78)
print(f"S-parameter CSV : {csv_path}")
print(f"EM plot         : {plot_path}")
print(f"Simulation XML  : {SIM_CSX}")

print()
print("=" * 78)
print("SIMULATION COMPLETE")
print("=" * 78)
