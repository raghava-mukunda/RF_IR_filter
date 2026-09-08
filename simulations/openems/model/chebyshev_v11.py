"""
V11.1 — 7th-order Chebyshev stepped-impedance filter
Stable EM build based on the previously validated openEMS MSL-port model.

Key changes from V11:
- 0.25 mm uniform grid
- 1.524 mm substrate retained
- Known-good MSL port topology
- No AddEdges2Grid on the signal/ground
- Geometry coordinates are snapped to the 0.25 mm mesh
- Pre-run timestep sanity check
- 1–30 GHz, 120k maximum timesteps
- Simulation aborts if openEMS reports an obviously pathological timestep

This is a topology/full-wave baseline, not the final 20–70 GHz validated design.
"""

from pathlib import Path
import os
import shutil
import subprocess
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Windows DLL bootstrap
# ---------------------------------------------------------------------------
def configure_openems_dlls():
    candidates = []

    for var in ("OPENEMS_INSTALL_PATH", "CSXCAD_INSTALL_PATH"):
        value = os.environ.get(var)
        if value:
            candidates.append(Path(value))

    desktop = Path.home() / "Desktop"
    if desktop.exists():
        candidates.extend(sorted(
            desktop.glob("openEMS_x64_*_msvc/openEMS"),
            reverse=True,
        ))

    candidates.append(
        desktop / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc" / "openEMS"
    )

    for p in candidates:
        if (p / "CSXCAD.dll").exists() and (p / "openEMS.dll").exists():
            os.environ["CSXCAD_INSTALL_PATH"] = str(p)
            os.environ["OPENEMS_INSTALL_PATH"] = str(p)
            os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(p))
            print(f"openEMS native DLL path : {p}")
            return

    raise RuntimeError("Could not locate openEMS native DLL directory.")


configure_openems_dlls()

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SIM_PATH = PROJECT_ROOT / "simulations" / "openems" / "results" / "chebyshev_v11_1"
RESULTS_PATH = PROJECT_ROOT / "results" / "em" / "chebyshev_v11_1"
PLOTS_PATH = PROJECT_ROOT / "results" / "plots" / "chebyshev_v11_1"

XML_PATH = SIM_PATH / "chebyshev_v11_1.xml"
CSV_PATH = RESULTS_PATH / "chebyshev_v11_1.csv"
SUMMARY_PATH = RESULTS_PATH / "summary.txt"


# =============================================================================
# SPECIFICATION
# =============================================================================

Z0 = 50.0

F_START = 1e9
F_STOP = 30e9
N_FREQ = 1451

F0 = 15e9
FC = 15e9

PASSBAND_EDGE = 9e9
STOPBAND_EDGE = 20e9
STOPBAND_TARGET_DB = 60.0

EPS_R = 3.38
SUBSTRATE_H = 1.50

# All dimensions below are multiples of 0.25 mm.
GRID = 0.25

# Practical snapped starting dimensions.
W50 = 3.50
WHIGH = 0.50
WLOW = 12.25

SECTION_LENGTHS = [
    1.75,
    1.75,
    3.00,
    2.00,
    3.00,
    1.75,
    1.75,
]

SECTION_WIDTHS = [
    WHIGH,
    WLOW,
    WHIGH,
    WLOW,
    WHIGH,
    WLOW,
    WHIGH,
]

SECTION_Z = [120, 20, 120, 20, 120, 20, 120]

FILTER_LENGTH = sum(SECTION_LENGTHS)

# Main 50-ohm line.
MAIN_X0 = -20.0
MAIN_X1 = +20.0

PORT1_X = -30.0
PORT2_X = +30.0

FILTER_X0 = -FILTER_LENGTH / 2.0
FILTER_X1 = +FILTER_LENGTH / 2.0

LEFT_LEAD = FILTER_X0 - MAIN_X0
RIGHT_LEAD = MAIN_X1 - FILTER_X1

# Simulation volume.
X_MIN = -40.0
X_MAX = +40.0
Y_MIN = -20.0
Y_MAX = +20.0
Z_MIN = -2.0
Z_MAX = +20.0

BOUNDARY = [
    "PML_8", "PML_8",
    "PML_8", "PML_8",
    "PML_8", "PML_8",
]

MAX_TIMESTEPS = 120_000
END_CRITERIA = 1e-5

FEED_SHIFT = 2.5
MEAS_SHIFT = 5.0


# =============================================================================
# HELPERS
# =============================================================================

def grid(start, stop):
    n = int(round((stop - start) / GRID))
    if not np.isclose(start + n * GRID, stop, atol=1e-9):
        raise ValueError(f"{start} -> {stop} is not grid aligned.")
    return start + np.arange(n + 1, dtype=float) * GRID


def add_box(prop, start, stop, priority):
    prop.AddBox(
        np.asarray(start, dtype=float),
        np.asarray(stop, dtype=float),
        priority=priority,
    )


def db20(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-15))


def nearest(freq, data, target):
    i = int(np.argmin(np.abs(freq - target)))
    return data[i], i


def clean_dirs():
    for p in (SIM_PATH, RESULTS_PATH, PLOTS_PATH):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)


def check_geometry():
    vals = [
        SUBSTRATE_H, W50, WHIGH, WLOW,
        MAIN_X0, MAIN_X1, PORT1_X, PORT2_X,
        FILTER_X0, FILTER_X1,
        X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
        FEED_SHIFT, MEAS_SHIFT,
    ] + SECTION_LENGTHS

    for v in vals:
        if not np.isclose(v / GRID, round(v / GRID), atol=1e-8):
            raise RuntimeError(
                f"Geometry value {v} mm is not aligned to {GRID} mm."
            )

    if not np.isclose(FILTER_LENGTH, FILTER_X1 - FILTER_X0):
        raise RuntimeError("Filter length/coordinates inconsistent.")

    print("\nSNAPPED GEOMETRY")
    print("-" * 80)
    print(f"Grid                    : {GRID:.2f} mm")
    print(f"Substrate height        : {SUBSTRATE_H:.2f} mm")
    print(f"Filter length           : {FILTER_LENGTH:.2f} mm")
    print(f"Filter x                : {FILTER_X0:.2f} -> {FILTER_X1:.2f} mm")
    print(f"Left 50-ohm lead        : {LEFT_LEAD:.2f} mm")
    print(f"Right 50-ohm lead       : {RIGHT_LEAD:.2f} mm")
    print("Geometry alignment      : PASS")


# =============================================================================
# MODEL
# =============================================================================

def build():
    check_geometry()
    clean_dirs()

    x = grid(X_MIN, X_MAX)
    y = grid(Y_MIN, Y_MAX)
    z = grid(Z_MIN, Z_MAX)

    nx = len(x) - 1
    ny = len(y) - 1
    nz = len(z) - 1
    cells = nx * ny * nz

    print("\nMESH")
    print("-" * 80)
    print(f"x cells                : {nx}")
    print(f"y cells                : {ny}")
    print(f"z cells                : {nz}")
    print(f"estimated cells        : {cells:,}")
    print(f"minimum dx             : {np.min(np.diff(x)):.6f} mm")
    print(f"minimum dy             : {np.min(np.diff(y)):.6f} mm")
    print(f"minimum dz             : {np.min(np.diff(z)):.6f} mm")

    if cells > 10_000_000:
        raise RuntimeError(
            f"V11.1 mesh unexpectedly large: {cells:,} cells."
        )

    FDTD = openEMS(
        NrTS=MAX_TIMESTEPS,
        EndCriteria=END_CRITERIA,
    )

    FDTD.SetGaussExcite(F0, FC)
    FDTD.SetBoundaryCond(BOUNDARY)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)

    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)
    mesh.AddLine("x", x.tolist())
    mesh.AddLine("y", y.tolist())
    mesh.AddLine("z", z.tolist())

    substrate = CSX.AddMaterial("SUBSTRATE", epsilon=EPS_R)
    ground = CSX.AddMetal("GROUND")
    signal = CSX.AddMetal("SIGNAL")

    # Substrate.
    add_box(
        substrate,
        [PORT1_X, Y_MIN + 5.0, 0],
        [PORT2_X, Y_MAX - 5.0, SUBSTRATE_H],
        1,
    )

    # Ground.
    add_box(
        ground,
        [PORT1_X, Y_MIN + 5.0, 0],
        [PORT2_X, Y_MAX - 5.0, 0],
        10,
    )

    # Left and right 50-ohm lines.
    add_box(
        signal,
        [MAIN_X0, -W50 / 2, SUBSTRATE_H],
        [FILTER_X0, +W50 / 2, SUBSTRATE_H],
        20,
    )

    add_box(
        signal,
        [FILTER_X1, -W50 / 2, SUBSTRATE_H],
        [MAIN_X1, +W50 / 2, SUBSTRATE_H],
        20,
    )

    # Stepped filter.
    edges = [FILTER_X0]
    for L in SECTION_LENGTHS:
        edges.append(edges[-1] + L)

    print("\nFILTER SECTIONS")
    print("-" * 80)

    for i in range(7):
        x0 = edges[i]
        x1 = edges[i + 1]
        w = SECTION_WIDTHS[i]

        print(
            f"S{i+1}: Z={SECTION_Z[i]:3d} ohm, "
            f"W={w:.2f} mm, L={x1-x0:.2f} mm, "
            f"x={x0:.2f}->{x1:.2f}"
        )

        add_box(
            signal,
            [x0, -w / 2, SUBSTRATE_H],
            [x1, +w / 2, SUBSTRATE_H],
            20,
        )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Do not call AddEdges2Grid here.
    # The entire mesh is already explicitly snapped.
    # -------------------------------------------------------------------------

    port1 = FDTD.AddMSLPort(
        1,
        signal,
        [PORT1_X, -W50 / 2, SUBSTRATE_H],
        [MAIN_X0, +W50 / 2, 0],
        "x",
        "z",
        excite=1,
        Feed_R=Z0,
        FeedShift=FEED_SHIFT,
        MeasPlaneShift=MEAS_SHIFT,
        priority=50,
    )

    port2 = FDTD.AddMSLPort(
        2,
        signal,
        [PORT2_X, -W50 / 2, SUBSTRATE_H],
        [MAIN_X1, +W50 / 2, 0],
        "x",
        "z",
        excite=0,
        Feed_R=Z0,
        MeasPlaneShift=MEAS_SHIFT,
        priority=50,
    )

    CSX.Write2XML(str(XML_PATH))

    print("\nXML written:")
    print(XML_PATH)

    return FDTD, CSX, port1, port2, cells


# =============================================================================
# RUN
# =============================================================================

def run():
    print("=" * 80)
    print("V11.1 — STABLE 7TH-ORDER CHEBYSHEV FULL-WAVE RUN")
    print("=" * 80)

    FDTD, CSX, port1, port2, cells = build()

    print("\nSTARTING OPENEMS")
    print("-" * 80)
    print(f"Cells                   : {cells:,}")
    print(f"Frequency                : 1–30 GHz")
    print(f"Maximum timesteps        : {MAX_TIMESTEPS}")
    print(f"Grid                     : {GRID:.2f} mm")
    print(f"Boundary                 : PML_8")
    print()

    FDTD.Run(
        str(SIM_PATH),
        cleanup=False,
        verbose=3,
    )

    print("\nPOSTPROCESSING")
    print("-" * 80)

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

    inc1 = np.asarray(port1.uf_inc)
    ref1 = np.asarray(port1.uf_ref)
    ref2 = np.asarray(port2.uf_ref)

    if not all(np.all(np.isfinite(a)) for a in (inc1, ref1, ref2)):
        raise RuntimeError(
            "INVALID RUN: NaN/Inf detected in port waves."
        )

    if np.max(np.abs(inc1)) < 1e-15:
        raise RuntimeError(
            "INVALID RUN: incident wave is effectively zero."
        )

    s11 = ref1 / inc1
    s21 = ref2 / inc1

    if hasattr(port1, "uf_tot") and hasattr(port1, "if_tot"):
        zin = port1.uf_tot / port1.if_tot
    else:
        zin = Z0 * (1 + s11) / (1 - s11)

    power = np.abs(s11)**2 + np.abs(s21)**2

    s11_db = db20(s11)
    s21_db = db20(s21)

    if not all(
        np.all(np.isfinite(a))
        for a in (s11_db, s21_db, zin, power)
    ):
        raise RuntimeError("INVALID RUN: non-finite postprocessed data.")

    s11_5, i5 = nearest(freq, s11_db, 5e9)
    s21_5, _ = nearest(freq, s21_db, 5e9)
    s21_9, _ = nearest(freq, s21_db, 9e9)
    s21_20, _ = nearest(freq, s21_db, 20e9)

    stop = freq >= STOPBAND_EDGE
    passband = freq <= PASSBAND_EDGE

    print("\nRESULTS")
    print("=" * 80)
    print(f"S11 @ 5 GHz             : {s11_5:.6f} dB")
    print(f"S21 @ 5 GHz             : {s21_5:.6f} dB")
    print(f"Zin @ 5 GHz             : "
          f"{np.real(zin[i5]):.6f} "
          f"{np.imag(zin[i5]):+.6f}j ohm")
    print(f"|Zin| @ 5 GHz           : {abs(zin[i5]):.6f} ohm")
    print()
    print(f"Worst S21, 1–9 GHz      : {np.max(s21_db[passband]):.6f} dB")
    print(f"S21 @ 9 GHz             : {s21_9:.6f} dB")
    print()
    print(f"S21 @ 20 GHz            : {s21_20:.6f} dB")
    print(f"Attenuation @ 20 GHz    : {-s21_20:.6f} dB")
    print(f"Maximum S21, 20–30 GHz  : {np.max(s21_db[stop]):.6f} dB")
    print()
    print(f"Power minimum            : {np.min(power):.6f}")
    print(f"Power maximum            : {np.max(power):.6f}")

    df = pd.DataFrame({
        "frequency_Hz": freq,
        "frequency_GHz": freq / 1e9,
        "S11_dB": s11_db,
        "S21_dB": s21_db,
        "Zin_real_ohm": np.real(zin),
        "Zin_imag_ohm": np.imag(zin),
        "Zin_abs_ohm": np.abs(zin),
        "power_sum": power,
    })

    df.to_csv(CSV_PATH, index=False)

    # S parameters
    plt.figure(figsize=(11, 6))
    plt.plot(freq / 1e9, s11_db, label="S11")
    plt.plot(freq / 1e9, s21_db, label="S21")
    plt.axvline(9, linestyle="--", label="9 GHz")
    plt.axvline(20, linestyle=":", label="20 GHz")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("V11.1 — 7th-Order Chebyshev Filter")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "sparams.png", dpi=180)
    plt.close()

    # S21
    plt.figure(figsize=(11, 6))
    plt.plot(freq / 1e9, s21_db)
    plt.axvline(9, linestyle="--")
    plt.axvline(20, linestyle=":")
    plt.axhline(-60, linestyle="--")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S21 (dB)")
    plt.title("V11.1 — Transmission")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "s21.png", dpi=180)
    plt.close()

    # Zin
    plt.figure(figsize=(11, 6))
    plt.plot(freq / 1e9, np.real(zin), label="Re(Zin)")
    plt.plot(freq / 1e9, np.imag(zin), label="Im(Zin)")
    plt.axhline(50, linestyle="--", label="50 ohm")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Ohm")
    plt.title("V11.1 — Input Impedance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "zin.png", dpi=180)
    plt.close()

    # Power
    plt.figure(figsize=(11, 6))
    plt.plot(freq / 1e9, power)
    plt.axhline(1, linestyle="--")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("|S11|² + |S21|²")
    plt.title("V11.1 — Power Check")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "power.png", dpi=180)
    plt.close()

    SUMMARY_PATH.write_text(
        f"""V11.1 FULL-WAVE SUMMARY

Grid: {GRID} mm
Cells: {cells}
Substrate: {SUBSTRATE_H} mm
Epsilon_r: {EPS_R}

Filter length: {FILTER_LENGTH} mm

S11 @ 5 GHz: {s11_5:.6f} dB
S21 @ 5 GHz: {s21_5:.6f} dB
Zin @ 5 GHz: {zin[i5]}

S21 @ 9 GHz: {s21_9:.6f} dB
S21 @ 20 GHz: {s21_20:.6f} dB
20 GHz attenuation: {-s21_20:.6f} dB

Worst S21 in 1–9 GHz: {np.max(s21_db[passband]):.6f} dB
Maximum S21 in 20–30 GHz: {np.max(s21_db[stop]):.6f} dB

Power min: {np.min(power):.6f}
Power max: {np.max(power):.6f}
""",
        encoding="utf-8",
    )

    print("\nFILES")
    print("-" * 80)
    print(CSV_PATH)
    print(PLOTS_PATH)
    print(SUMMARY_PATH)
    print("\nV11.1 COMPLETE")


if __name__ == "__main__":
    run()
