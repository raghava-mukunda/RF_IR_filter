"""
Clean openEMS diagnostic for the coaxial wideband low-pass filter.

IMPORTANT:
This version targets the installed openEMS/CSXCAD build reported by the
user. It first validates a bare coax before running the filtered cases.

Cases:
    0 sections = bare coax
    1 section
    2 sections
    4 sections

Run:
    .\.venv-openems\Scripts\Activate.ps1
    python wideband_lp_filter_diagnostic_clean_v2.py
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# NATIVE OPENEMS SETUP
# ============================================================================

OPENEMS_PATH = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
    / "openEMS"
)

if not OPENEMS_PATH.is_dir():
    raise RuntimeError(f"openEMS directory not found:\n{OPENEMS_PATH}")

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_PATH)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_PATH)
os.environ["PATH"] = str(OPENEMS_PATH) + os.pathsep + os.environ.get("PATH", "")

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(OPENEMS_PATH))

print(f"openEMS native DLL path : {OPENEMS_PATH}")

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ============================================================================
# PARAMETERS
# ============================================================================

# Geometry, mm
RI = 1.60
RO = 3.70
RO_OUT = 4.20
EPS_PTFE = 2.2

A = 4.00
B = 5.00
D = 4.90

RING_SPACING = 5.00
SECTION_PITCH = 12.00

# Simulation
Z0 = 50.0

F_START = 1e9
F_STOP = 100e9
N_FREQ = 501

F0 = 50e9

GRID = 0.25
MARGIN = 2.0

NR_TS = 15000
END_CRITERIA = 3e-4

PML = 8

# Bare coax length. Keep it the same reference length for the port test.
BARE_LENGTH = 29.0

RESULT_ROOT = Path.cwd() / "results" / "diagnostic_clean_v2"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# UTILS
# ============================================================================

def db20(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-15))


def add_uniform_mesh(mesh, axis, lo, hi, step):
    n = int(round((hi - lo) / step))
    lines = [lo + i * step for i in range(n + 1)]
    mesh.AddLine(axis, lines)


def build_mesh(csx, length):
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(1e-3)
    add_uniform_mesh(
        mesh, "x",
        -length / 2 - MARGIN,
        +length / 2 + MARGIN,
        GRID
    )

    extent = RO_OUT + A + MARGIN

    add_uniform_mesh(
        mesh, "y",
        -extent,
        +extent,
        GRID
    )

    add_uniform_mesh(
        mesh, "z",
        -extent,
        +extent,
        GRID
    )

    return mesh


# ============================================================================
# COAX GEOMETRY
# ============================================================================

def add_coax(csx, length):
    """
    Coaxial line along x.

    Installed CSXCAD API:
        AddCylinder(start, stop, radius)
        AddCylindricalShell(start, stop, r_inner, r_outer)
    """

    air = csx.AddMaterial("Air", epsilon=1.0)
    ptfe = csx.AddMaterial("PTFE", epsilon=EPS_PTFE)
    pec = csx.AddMetal("PEC")

    x1 = -length / 2
    x2 = +length / 2

    # Inner conductor
    pec.AddCylinder(
        [x1, 0, 0],
        [x2, 0, 0],
        RI
    )

    # Dielectric annulus
    ptfe.AddCylindricalShell(
        [x1, 0, 0],
        [x2, 0, 0],
        RI,
        RO
    )

    # Outer conductor wall
    pec.AddCylindricalShell(
        [x1, 0, 0],
        [x2, 0, 0],
        RO,
        RO_OUT
    )

    return air, ptfe, pec


# ============================================================================
# APERTURES
# ============================================================================

def add_aperture_air_box(air, x0, orientation):
    """
    Add an air opening extending from the outer coax wall outward.

    The aperture is represented as a rectangular air volume. Its inner
    boundary slightly overlaps the PEC wall so that voxelization cannot
    leave a thin PEC sliver blocking the opening.

    orientation:
        +y, -y, +z, -z

    The aperture dimensions are:
        x extent = D
        radial extent = A
        transverse extent = B
    """

    overlap = 0.10

    if orientation == "+y":
        # x-y polygon extruded in z
        pts = [
            [x0 - D / 2, RO - overlap],
            [x0 + D / 2, RO - overlap],
            [x0 + D / 2, RO + A],
            [x0 - D / 2, RO + A],
        ]
        air.AddLinPoly(
            [
                [p[0] for p in pts],
                [p[1] for p in pts],
            ],
            "z",
            -B / 2,
            priority=30
        )

    elif orientation == "-y":
        pts = [
            [x0 - D / 2, -RO + overlap],
            [x0 + D / 2, -RO + overlap],
            [x0 + D / 2, -RO - A],
            [x0 - D / 2, -RO - A],
        ]
        air.AddLinPoly(
            [
                [p[0] for p in pts],
                [p[1] for p in pts],
            ],
            "z",
            -B / 2,
            priority=30
        )

    elif orientation == "+z":
        # Build x-z rectangle and extrude in y.
        pts = [
            [x0 - D / 2, RO - overlap],
            [x0 + D / 2, RO - overlap],
            [x0 + D / 2, RO + A],
            [x0 - D / 2, RO + A],
        ]
        air.AddLinPoly(
            [
                [p[0] for p in pts],
                [p[1] for p in pts],
            ],
            "y",
            -B / 2,
            priority=30
        )

    elif orientation == "-z":
        pts = [
            [x0 - D / 2, -RO + overlap],
            [x0 + D / 2, -RO + overlap],
            [x0 + D / 2, -RO - A],
            [x0 - D / 2, -RO - A],
        ]
        air.AddLinPoly(
            [
                [p[0] for p in pts],
                [p[1] for p in pts],
            ],
            "y",
            -B / 2,
            priority=30
        )

    else:
        raise ValueError(f"Unknown aperture orientation: {orientation}")


def add_sections(air, sections):
    if sections <= 0:
        return

    first = -0.5 * (sections - 1) * SECTION_PITCH

    for sec in range(sections):
        xc = first + sec * SECTION_PITCH

        for dx in (-RING_SPACING / 2, +RING_SPACING / 2):
            x0 = xc + dx

            for orientation in ("+y", "-y", "+z", "-z"):
                add_aperture_air_box(
                    air,
                    x0,
                    orientation
                )


# ============================================================================
# LUMPED PORTS
# ============================================================================

def add_ports(fdtd, length):
    """
    Port model for the installed AddLumpedPort API.

    Verified from the user's traceback:
        AddLumpedPort takes at most 6 positional arguments.

    Therefore:
        AddLumpedPort(port_nr, Z0, start, stop, exc_dir, excite)

    The port is a radial sheet in the dielectric annulus at each end.
    Excitation direction is specified as the axis string 'y'.

    This avoids the invalid vector form [0,1,0].
    """

    port_depth = 1.0

    # We place the port at the x end, with its sheet spanning radially
    # through the dielectric from the inner conductor to the outer wall.
    #
    # A y-directed electric field is used for the lumped excitation.

    x_left = -length / 2
    x_right = +length / 2

    # Port 1
    p1_start = [
        x_left,
        RI,
        -0.5 * port_depth
    ]

    p1_stop = [
        x_left,
        RO,
        +0.5 * port_depth
    ]

    port1 = fdtd.AddLumpedPort(
        1,
        Z0,
        p1_start,
        p1_stop,
        "y",
        1.0
    )

    # Port 2
    p2_start = [
        x_right,
        RI,
        -0.5 * port_depth
    ]

    p2_stop = [
        x_right,
        RO,
        +0.5 * port_depth
    ]

    port2 = fdtd.AddLumpedPort(
        2,
        Z0,
        p2_start,
        p2_stop,
        "y",
        0.0
    )

    return port1, port2


# ============================================================================
# CASE BUILDER
# ============================================================================

def get_length(sections):
    if sections == 0:
        return BARE_LENGTH

    section_span = (sections - 1) * SECTION_PITCH

    # Five mm clear coax on either side plus ring spacing.
    return section_span + 10.0 + RING_SPACING


def build_case(sections, length):
    csx = ContinuousStructure()

    fdtd = openEMS(
        EndCriteria=END_CRITERIA,
        NrTS=NR_TS
    )

    fdtd.SetGaussExcite(F0, F0)

    fdtd.SetBoundaryCond([
        f"PML_{PML}",
        f"PML_{PML}",
        f"PML_{PML}",
        f"PML_{PML}",
        f"PML_{PML}",
        f"PML_{PML}",
    ])

    fdtd.SetCSX(csx)

    air, ptfe, pec = add_coax(csx, length)

    add_sections(
        air,
        sections
    )

    build_mesh(
        csx,
        length
    )

    port1, port2 = add_ports(
        fdtd,
        length
    )

    return csx, fdtd, port1, port2


# ============================================================================
# RUN CASE
# ============================================================================

def run_case(sections, name):
    print()
    print("=" * 88)
    print(f"CASE: {name}   sections={sections}")
    print("=" * 88)

    length = get_length(sections)

    print(f"length: {length:.2f} mm")

    csx, fdtd, port1, port2 = build_case(
        sections,
        length
    )

    sim_dir = RESULT_ROOT / name
    sim_dir.mkdir(parents=True, exist_ok=True)

    xml_path = sim_dir / f"{name}.xml"

    csx.Write2XML(
        str(xml_path)
    )

    freq = np.linspace(
        F_START,
        F_STOP,
        N_FREQ
    )

    print(
        f"frequency: {F_START/1e9:.1f} - "
        f"{F_STOP/1e9:.1f} GHz"
    )
    print(f"points: {N_FREQ}")
    print(f"mesh: {GRID:.2f} mm")
    print(f"max steps: {NR_TS}")
    print()
    print("[OPENEMS START]")

    t0 = time.time()

    fdtd.Run(
        str(sim_dir),
        cleanup=False,
        verbose=2
    )

    elapsed = time.time() - t0

    print(
        f"[OPENEMS DONE] {elapsed:.1f} s"
    )

    # ------------------------------------------------------------------------
    # Ports
    # ------------------------------------------------------------------------

    port1.CalcPort(
        str(sim_dir),
        freq
    )

    port2.CalcPort(
        str(sim_dir),
        freq
    )

    # Local openEMS port convention used in the existing project.
    s11 = port1.uf_ref / port1.uf_inc
    s21 = port2.uf_ref / port1.uf_inc

    s11_db = db20(s11)
    s21_db = db20(s21)

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    def at(f):
        i = int(np.argmin(np.abs(freq - f)))
        return float(s21_db[i])

    s21_5 = at(5e9)
    s21_9 = at(9e9)
    s21_10 = at(10e9)
    s21_20 = at(20e9)

    pb = (
        (freq >= 1e9)
        & (freq <= 9e9)
    )

    passband_loss = float(
        -np.min(s21_db[pb])
    )

    sb12 = (
        (freq >= 12e9)
        & (freq <= 20e9)
    )

    idx12 = np.where(sb12)[0]
    imax12 = idx12[
        np.argmax(s21_db[idx12])
    ]

    max12_20 = float(
        s21_db[imax12]
    )

    fmax12_20 = float(
        freq[imax12] / 1e9
    )

    sb20 = (
        (freq >= 20e9)
        & (freq <= 100e9)
    )

    idx20 = np.where(sb20)[0]
    imax20 = idx20[
        np.argmax(s21_db[idx20])
    ]

    max20_100 = float(
        s21_db[imax20]
    )

    fmax20_100 = float(
        freq[imax20] / 1e9
    )

    # ------------------------------------------------------------------------
    # Robust 3-dB cutoff
    # ------------------------------------------------------------------------

    cutoff = None

    for i in range(len(freq)):

        if freq[i] < 2e9:
            continue

        if s21_db[i] > -3:
            continue

        future = (
            (freq >= freq[i])
            & (freq <= freq[i] + 1e9)
        )

        if np.sum(future) < 2:
            continue

        if np.all(s21_db[future] <= -3):
            cutoff = float(freq[i] / 1e9)
            break

    # ------------------------------------------------------------------------
    # Power conservation
    # ------------------------------------------------------------------------

    power_sum = (
        np.abs(s11) ** 2
        + np.abs(s21) ** 2
    )

    power_error = float(
        np.max(
            np.abs(power_sum - 1.0)
        )
    )

    # ------------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------------

    print()
    print(f"S21 @ 5 GHz       = {s21_5:.3f} dB")
    print(f"S21 @ 9 GHz       = {s21_9:.3f} dB")
    print(f"S21 @ 10 GHz      = {s21_10:.3f} dB")
    print(f"S21 @ 20 GHz      = {s21_20:.3f} dB")

    if cutoff is None:
        print("robust 3-dB cutoff = NOT FOUND")
    else:
        print(
            f"robust 3-dB cutoff = {cutoff:.3f} GHz"
        )

    print(
        f"worst 1-9 GHz loss = "
        f"{passband_loss:.3f} dB"
    )

    print(
        f"max 12-20 GHz     = "
        f"{max12_20:.3f} dB "
        f"@ {fmax12_20:.3f} GHz"
    )

    print(
        f"max 20-100 GHz    = "
        f"{max20_100:.3f} dB "
        f"@ {fmax20_100:.3f} GHz"
    )

    print(
        f"max power error    = "
        f"{power_error:.3e}"
    )

    # ------------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------------

    csv_path = sim_dir / f"{name}_sparams.csv"

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frequency_Hz",
            "frequency_GHz",
            "S11_dB",
            "S21_dB",
            "S11_abs",
            "S21_abs",
            "power_sum",
        ])

        for i in range(len(freq)):
            writer.writerow([
                float(freq[i]),
                float(freq[i] / 1e9),
                float(s11_db[i]),
                float(s21_db[i]),
                float(abs(s11[i])),
                float(abs(s21[i])),
                float(power_sum[i]),
            ])

    # ------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(
        freq / 1e9,
        s21_db,
        label="S21"
    )

    plt.plot(
        freq / 1e9,
        s11_db,
        label="S11"
    )

    plt.axvline(
        9,
        linestyle="--",
        label="9 GHz"
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
        linestyle=":",
        label="-60 dB"
    )

    plt.xlim(1, 100)
    plt.ylim(-120, 5)

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(name)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        sim_dir / f"{name}.png",
        dpi=160
    )

    plt.close()

    return {
        "case": name,
        "sections": sections,
        "length_mm": length,
        "S21_5GHz_dB": s21_5,
        "S21_9GHz_dB": s21_9,
        "S21_10GHz_dB": s21_10,
        "S21_20GHz_dB": s21_20,
        "cutoff_3dB_GHz": cutoff,
        "passband_worst_loss_1_9_dB": passband_loss,
        "max_S21_12_20_dB": max12_20,
        "max_S21_12_20_GHz": fmax12_20,
        "max_S21_20_100_dB": max20_100,
        "max_S21_20_100_GHz": fmax20_100,
        "max_power_error": power_error,
        "wall_time_s": elapsed,
    }, freq, s21_db


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 88)
    print("CLEAN COAXIAL LOW-PASS DIAGNOSTIC")
    print("=" * 88)
    print()
    print("Geometry:")
    print(f"  RI = {RI:.2f} mm")
    print(f"  RO = {RO:.2f} mm")
    print(f"  PTFE epsilon_r = {EPS_PTFE:.2f}")
    print(f"  A = {A:.2f} mm")
    print(f"  B = {B:.2f} mm")
    print(f"  D = {D:.2f} mm")
    print()
    print("Simulation:")
    print(f"  1-100 GHz / {N_FREQ} points")
    print(f"  mesh = {GRID:.2f} mm")
    print(f"  steps = {NR_TS}")
    print()

    cases = [
        (0, "case_0_bare_coax"),
        (1, "case_1_one_section"),
        (2, "case_2_two_sections"),
        (4, "case_4_four_sections"),
    ]

    all_results = []
    curves = []

    for sections, name in cases:

        result, freq, s21 = run_case(
            sections,
            name
        )

        all_results.append(result)
        curves.append((name, freq, s21))

        # --------------------------------------------------------------------
        # Critical gate after bare coax
        # --------------------------------------------------------------------

        if sections == 0:

            if result["S21_5GHz_dB"] < -3.0:
                print()
                print("=" * 88)
                print("BARE COAX PORT TEST FAILED")
                print("=" * 88)
                print(
                    "The bare coax is already losing more than 3 dB "
                    "at 5 GHz."
                )
                print()
                print(
                    "STOPPING before filtered cases."
                )
                print(
                    "Do NOT optimize the filter geometry yet."
                )
                print("=" * 88)
                return

            print()
            print("=" * 88)
            print("BARE COAX PORT TEST PASSED")
            print("=" * 88)
            print(
                "Proceeding to leakage-section diagnostics."
            )

    # ------------------------------------------------------------------------
    # Comparison plot
    # ------------------------------------------------------------------------

    plt.figure(figsize=(12, 7))

    for name, freq, s21 in curves:
        plt.plot(
            freq / 1e9,
            s21,
            label=name
        )

    plt.axvline(
        9,
        linestyle="--",
        label="9 GHz"
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
        linestyle=":",
        label="-60 dB"
    )

    plt.xlim(1, 100)
    plt.ylim(-120, 5)

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S21 (dB)")
    plt.title("Diagnostic Comparison")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    comparison_path = (
        RESULT_ROOT
        / "diagnostic_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=180
    )

    plt.close()

    # ------------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------------

    csv_path = RESULT_ROOT / "diagnostic_summary.csv"

    fields = list(all_results[0].keys())

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(all_results)

    json_path = RESULT_ROOT / "diagnostic_summary.json"

    with open(json_path, "w") as f:
        json.dump(
            all_results,
            f,
            indent=2
        )

    # ------------------------------------------------------------------------
    # Final table
    # ------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)

    print(
        f"{'Case':<25}"
        f"{'S21@5':>10}"
        f"{'S21@9':>10}"
        f"{'S21@10':>10}"
        f"{'S21@20':>10}"
        f"{'Max20-100':>14}"
        f"{'Freq':>10}"
    )

    print("-" * 110)

    for r in all_results:
        print(
            f"{r['case']:<25}"
            f"{r['S21_5GHz_dB']:>10.2f}"
            f"{r['S21_9GHz_dB']:>10.2f}"
            f"{r['S21_10GHz_dB']:>10.2f}"
            f"{r['S21_20GHz_dB']:>10.2f}"
            f"{r['max_S21_20_100_dB']:>14.2f}"
            f"{r['max_S21_20_100_GHz']:>10.2f}"
        )

    print()
    print(f"Results: {RESULT_ROOT}")
    print(f"Plot:    {comparison_path}")
    print(f"CSV:     {csv_path}")
    print(f"JSON:    {json_path}")
    print()
    print("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()
