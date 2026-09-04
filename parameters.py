from pathlib import Path


# ============================================================
# FILTER ELECTRICAL SPECIFICATION
# ============================================================

Z0 = 50.0

CUTOFF_FREQUENCY = 9e9

PASSBAND_RIPPLE_DB = 0.1

STOPBAND_START = 20e9

STOPBAND_ATTENUATION_DB = 60.0

ANALYSIS_START = 1e9

ANALYSIS_STOP = 30e9


# ============================================================
# FILTER ORDER
# ============================================================

FILTER_ORDER = 7


# ============================================================
# 7th-ORDER CHEBYSHEV TYPE-I PROTOTYPE
#
# g0 = 1
#
# g1 ... g7 below
#
# g8 = 1
# ============================================================

CHEBYSHEV_G = [
    1.1811783386,
    1.4228062176,
    2.0966713394,
    1.5734010555,
    2.0966713394,
    1.4228062176,
    1.1811783386,
]


# ============================================================
# STEPPED-IMPEDANCE IMPLEMENTATION
# ============================================================

# High impedance sections
# Represent approximately series-inductive behavior.

Z_HIGH = 120.0


# Low impedance sections
# Represent approximately shunt-capacitive behavior.

Z_LOW = 20.0


# ============================================================
# SUBSTRATE
# ============================================================

EPS_R = 3.38

SUBSTRATE_HEIGHT = 1.524e-3

SUBSTRATE_WIDTH = 80e-3


# ============================================================
# CONDUCTOR
# ============================================================

COPPER_THICKNESS = 35e-6

COPPER_SIGMA = 5.8e7


# ============================================================
# 50-OHM FEED LINES
# ============================================================

INPUT_LINE_LENGTH = 10e-3

OUTPUT_LINE_LENGTH = 10e-3


# ============================================================
# PORTS
# ============================================================

PORT_IMPEDANCE = 50.0


# ============================================================
# MESH
# ============================================================

# Maximum X/Y cell dimension.

MAX_MESH_XY = 0.25e-3


# Number of cells through the substrate.

SUBSTRATE_CELLS_Z = 6


# Number of cells through copper.

COPPER_CELLS_Z = 2


# ============================================================
# SIMULATION BOX
# ============================================================

SIM_BOX_X_MARGIN = 10e-3

SIM_BOX_Y_MARGIN = 10e-3

SIM_BOX_Z = 40e-3


# ============================================================
# FDTD EXCITATION
# ============================================================

F0 = 15e9

FC = 15e9


# ============================================================
# FDTD SIMULATION
# ============================================================

NR_TS = 150000


# ============================================================
# BOUNDARY CONDITIONS
#
# x_min
# x_max
# y_min
# y_max
# z_min
# z_max
# ============================================================

BOUNDARY_CONDITIONS = [
    'MUR',
    'MUR',
    'MUR',
    'MUR',
    'MUR',
    'MUR',
]


# ============================================================
# SIMULATION CONTROL
# ============================================================

RUN_SIMULATION = True

SHOW_GEOMETRY = False

CLEAN_SIMULATION_DIRECTORY = True


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent


SIMULATION_DIRECTORY = (
    PROJECT_ROOT
    / "simulations"
    / "openems"
    / "results"
)


RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "em"
)


PLOT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "plots"
)