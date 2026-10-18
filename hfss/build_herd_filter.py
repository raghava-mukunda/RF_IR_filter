"""
3D HFSS realization candidate for the calculated 10 GHz low-pass filter (iteration 07).

Architecture:
  * Rectangular coax-like shielded line, interior 4.6 x 4.2 mm
    (outer envelope 5.0 x 4.6 mm with 0.2 mm assumed walls).
  * 15th-order, 0.1 dB-ripple Chebyshev ladder, equal 50-ohm references.
  * Eight series inductors and seven shunt capacitors, calculated from the
    Chebyshev prototype and represented by HFSS lumped-RLC boundary sheets.
  * Shunt capacitors are split equally between the upper and lower enclosure
    walls (C/2 in parallel = C).
  * RLC boundaries use the raw AEDT COM API with explicit units on every R/L/C field.
  * Disabled R/L/C fields receive valid unit-bearing dummy values and Use* flags remain false.
  * A pre-solve audit checks that all expected RLC boundaries exist before meshing.
  * Report creation is attempted before Touchstone export.
  * Creates a new project; never overwrites existing projects.

IMPORTANT:
This is a 3D electromagnetic HIBRID model with ideal RLC boundaries, not yet a
manufacturable all-metal stepped-impedance/leaky-waveguide filter. It is meant
to place the proven electrical prototype into a shielded 3D EM environment,
check ports/geometry, and expose parasitic coupling around gaps. Replace the
ideal RLC boundaries with physical component or distributed structures for a
manufacturing-ready design. The outer enclosure is represented by PEC boundary
conditions on the cavity walls; finite wall thickness/loss is not yet meshed.

Run from repository root:
  .venv\\Scripts\\python.exe build_10GHz_LPF_3D_HFSS_hybrid_iter07.py
"""
from pathlib import Path
import csv
import json
import math
import re
import traceback

from ansys.aedt.core import Desktop, Hfss

# ----------------------------- paths ---------------------------------
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "hfss" / "chebyshev_lpf_10GHz" / "3d_hybrid_iter07"
PROJECT_NAME = "Chebyshev15_10GHz_3D_Hybrid_Iter07"
DESIGN_NAME = "Chebyshev15_10GHz_3D_Hybrid_Iter07"
AEDT_VERSION = "2025.2"
STUDENT_VERSION = True

# Safety gate: build and audit RLC boundaries first; no HFSS solve by default.
RUN_ANALYSIS = False  # Deliberately preflight-only; do not spend solver time until boundary checks pass.

# ----------------------- enclosure / geometry -------------------------
LENGTH_MM = 65.3
OUTER_W_MM = 5.0
OUTER_H_MM = 4.6
WALL_MM = 0.2
INNER_W_MM = OUTER_W_MM - 2.0 * WALL_MM
INNER_H_MM = OUTER_H_MM - 2.0 * WALL_MM

# Center-strip width is not arbitrary. Paper 1 gives r_o/r_i ~= 2.302 for a
# 50-ohm air-filled circular coax. Using an area-equivalent outer radius
# r_o ~= sqrt(INNER_W*INNER_H/pi) ~= 2.48 mm gives r_i ~= 1.08 mm and
# an equivalent inner diameter ~= 2.16 mm. We start with 2.2 mm for this
# rectangular-coax approximation, then confirm modal Z0 in HFSS.
CENTER_STRIP_W_MM = 2.2
CENTER_STRIP_T_MM = 0.5
COPPER_MATERIAL = "copper"

# Iteration 07 change: distribute all seven shunt-C nodes through the body.
# Iteration 02 clustered the capacitor nodes over 2.1 mm between ~31.2 mm
# end sections. Here both end feeds are 5 mm; all seven internal nodes share
# the remaining package length evenly. Synthesized L/C values remain fixed.
SERIES_GAP_MM = 0.10
END_FEED_MM = 5.0
INTERNAL_NODE_MM = (LENGTH_MM - 2.0 * END_FEED_MM - 8.0 * SERIES_GAP_MM) / 7.0
# Nine conductor nodes (two end feeds + seven capacitor nodes) and eight gaps.
F_ADAPT_GHZ = 10.0
F_START_GHZ = 0.1
F_STOP_GHZ = 30.0
N_SWEEP_POINTS = 600
MAX_PASSES = 12
DELTA_S = 0.02
PORT_MODES = 1

# ------------------------ filter synthesis -----------------------------
ORDER = 15
RIPPLE_DB = 0.1
ZREF_OHM = 50.0
TARGET_F3_DB_GHZ = 10.0


def chebyshev_g(n, ripple_db):
    """Normalized prototype coefficients g1..gn and g[n+1]."""
    beta = math.log(1.0 / math.tanh(ripple_db / 17.37))
    gamma = math.sinh(beta / (2.0 * n))
    a = [math.sin((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1)]
    b = [gamma**2 + math.sin(k * math.pi / n)**2 for k in range(1, n + 1)]
    g = [0.0] * (n + 1)
    g[0] = 2.0 * a[0] / gamma
    for k in range(1, n):
        g[k] = 4.0 * a[k - 1] * a[k] / (b[k - 1] * g[k - 1])
    g[n] = 1.0 if n % 2 else (1.0 / math.tanh(beta / 4.0))**2
    return g


def synthesize_ladder():
    """
    Set the Chebyshev passband edge so that the ideal LC ladder is -3 dB
    at 10 GHz. Return element specs in ladder order.
    """
    eps = math.sqrt(10.0 ** (RIPPLE_DB / 10.0) - 1.0)
    t_for_3db = math.sqrt((10.0 ** (3.0 / 10.0) - 1.0) / eps**2)
    omega_ratio = math.cosh(math.acosh(t_for_3db) / ORDER)
    f_passband_edge = TARGET_F3_DB_GHZ * 1e9 / omega_ratio
    g = chebyshev_g(ORDER, RIPPLE_DB)

    elems = []
    for index, gi in enumerate(g[:ORDER], start=1):
        if index % 2:
            value = ZREF_OHM * gi / (2.0 * math.pi * f_passband_edge)
            elems.append({
                "index": index, "type": "L", "g": gi, "value_si": value,
                "value_display": value * 1e9, "unit": "nH",
                "boundary_name": f"SeriesL_{index:02d}",
            })
        else:
            value = gi / (ZREF_OHM * 2.0 * math.pi * f_passband_edge)
            elems.append({
                "index": index, "type": "C", "g": gi, "value_si": value,
                "value_display": value * 1e12, "unit": "pF",
                "boundary_name": f"ShuntC_{index:02d}",
            })
    return f_passband_edge, elems, g[ORDER]


def require(obj, label):
    if obj is None or obj is False:
        raise RuntimeError(f"AEDT failed to create {label}")
    return obj


def parse_touchstone_ma(path):
    """Read a two-port Touchstone file exported as S MA, returning f, S11 dB, S21 dB."""
    freqs, s11_db, s21_db = [], [], []
    scale = 1.0
    in_header = True
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            header = line.upper().split()
            if len(header) >= 3 and header[1] in {"HZ", "KHZ", "MHZ", "GHZ"}:
                scale = {"HZ": 1e-9, "KHZ": 1e-6, "MHZ": 1e-3, "GHZ": 1.0}[header[1]]
                if "MA" not in header:
                    raise ValueError(f"Expected S MA Touchstone data, got header: {line}")
            in_header = False
            continue
        fields = line.split()
        if len(fields) < 9:
            continue
        try:
            f_ghz = float(fields[0]) * scale
            # Two-port order: S11, S21, S12, S22. Magnitudes are linear in MA format.
            m11 = max(float(fields[1]), 1e-15)
            m21 = max(float(fields[3]), 1e-15)
        except (ValueError, IndexError):
            continue
        freqs.append(f_ghz)
        s11_db.append(20.0 * math.log10(m11))
        s21_db.append(20.0 * math.log10(m21))
    if not freqs:
        raise ValueError(f"No S MA data rows parsed from {path}")
    return freqs, s11_db, s21_db


def write_response_diagnostics(s2p_path, out_dir):
    """Save a CSV, response PNG, and concise target checks after HFSS export."""
    freqs, s11_db, s21_db = parse_touchstone_ma(s2p_path)
    csv_path = out_dir / "sparams_3d_hybrid_iter07.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Frequency_GHz", "S11_dB", "S21_dB"])
        writer.writerows(zip(freqs, s11_db, s21_db))

    passband = [(f, s) for f, s in zip(freqs, s21_db) if 0.1 <= f <= 9.0]
    stopband = [(f, s) for f, s in zip(freqs, s21_db) if 12.0 <= f <= 30.0]
    crossing = next(((f, s) for f, s in zip(freqs, s21_db) if f >= 0.1 and s <= -3.0), None)
    summary_lines = [
        "ITERATION 05 POST-SOLVE DIAGNOSTICS",
        f"Touchstone: {s2p_path}",
        f"Samples: {len(freqs)}; frequency range: {min(freqs):.6g} to {max(freqs):.6g} GHz",
        f"Passband S21 range, 0.1–9 GHz: {min(s for _, s in passband):.3f} to {max(s for _, s in passband):.3f} dB" if passband else "No samples in 0.1–9 GHz passband window.",
        f"Worst stopband S21, 12–30 GHz: {max(stopband, key=lambda item: item[1])[1]:.3f} dB at {max(stopband, key=lambda item: item[1])[0]:.4f} GHz" if stopband else "No samples in 12–30 GHz stopband window.",
        f"Best stopband S21, 12–30 GHz: {min(stopband, key=lambda item: item[1])[1]:.3f} dB at {min(stopband, key=lambda item: item[1])[0]:.4f} GHz" if stopband else "",
        f"First sampled point at/below -3 dB: {crossing[1]:.3f} dB at {crossing[0]:.4f} GHz" if crossing else "No -3 dB crossing found in sampled range.",
        "Targets: approximately -3 dB at 10 GHz and S21 <= -60 dB throughout 12–30 GHz.",
        "This diagnostic evaluates the simulated 3D data; it does not assume the ideal circuit response.",
    ]
    summary_path = out_dir / "analysis_summary_iter07.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    png_path = out_dir / "LP10_Iter07_SParameters.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(freqs, s21_db, label="S21 (transmission)", linewidth=1.6)
        ax.plot(freqs, s11_db, label="S11 (input reflection)", linewidth=1.2)
        ax.axhline(-3.0, linestyle="--", linewidth=1.0, label="-3 dB cutoff reference")
        ax.axhline(-60.0, linestyle=":", linewidth=1.1, label="-60 dB stopband target")
        ax.axvline(10.0, linestyle="--", linewidth=0.9, label="10 GHz target cutoff")
        ax.axvline(12.0, linestyle=":", linewidth=0.9, label="12 GHz stopband starts")
        ax.set(xlabel="Frequency (GHz)", ylabel="Magnitude (dB)", title="HFSS Iteration 07 — S-Parameters")
        ax.set_xlim(0, 30)
        ax.set_ylim(-100, 2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        print(f"Response plot: {png_path.resolve()}")
    except Exception as exc:
        print(f"WARNING: local response PNG was not created (HFSS report is still attempted): {exc}")
    print("\n" + "\n".join(summary_lines))
    print(f"S-parameter CSV: {csv_path.resolve()}")
    print(f"Analysis summary: {summary_path.resolve()}")


def make_box(hfss, name, origin, sizes, material):
    return require(
        hfss.modeler.create_box(
            origin=origin, sizes=sizes, name=name, material=material
        ),
        name,
    )


def _com_point(point):
    # Explicit mm strings match the model geometry and AEDT COM scripting syntax.
    return [f"{float(v):.12g}mm" for v in point]


def create_rlc(hfss, sheet, name, current_line, rlc_type, inductance=None, capacitance=None):
    # Bypass PyAEDT's RLC wrapper. AEDT's COM scripting interface accepts units
    # explicitly (e.g. 1.2e-13F); every field gets a valid unit even when disabled.
    if not isinstance(current_line, (list, tuple)) or len(current_line) != 2:
        raise TypeError(f"{name}: current_line must be [start_point, end_point]")
    start, end = current_line
    if len(start) != 3 or len(end) != 3:
        raise ValueError(f"{name}: endpoints must each contain x, y, z")

    use_induct = inductance is not None
    use_cap = capacitance is not None
    if use_induct and (not math.isfinite(float(inductance)) or float(inductance) <= 0):
        raise ValueError(f"{name}: inductance must be positive and finite")
    if use_cap and (not math.isfinite(float(capacitance)) or float(capacitance) <= 0):
        raise ValueError(f"{name}: capacitance must be positive and finite")

    r_string = "1ohm"       # inactive because UseResist=False
    l_string = f"{float(inductance):.15g}H" if use_induct else "1H"
    c_string = f"{float(capacitance):.15g}F" if use_cap else "1F"
    args = [
        f"NAME:{name}",
        "Objects:=", [sheet.name],
        ["NAME:CurrentLine", "Start:=", _com_point(start), "End:=", _com_point(end)],
        "RLC Type:=", rlc_type,
        "UseResist:=", False, "Resistance:=", r_string,
        "UseInduct:=", use_induct, "Inductance:=", l_string,
        "UseCap:=", use_cap, "Capacitance:=", c_string,
    ]
    try:
        hfss.oboundary.AssignLumpedRLC(args)
    except Exception as exc:
        print(f"\nRLC ASSIGNMENT FAILED: {name}")
        print(f"  sheet={sheet.name!r}; type={rlc_type!r}; UseInduct={use_induct}; L={l_string!r}")
        print(f"  UseCap={use_cap}; C={c_string!r}; current_line={current_line!r}")
        print(f"  COM argument array={args!r}")
        raise RuntimeError(
            f"AEDT rejected RLC boundary {name}. Project construction stopped before meshing/analysis."
        ) from exc
    print(f"Assigned via AEDT COM: {name}; type={rlc_type}; L={l_string} (enabled={use_induct}); C={c_string} (enabled={use_cap})")
    return name


def audit_rlc_boundaries(hfss):
    expected = [f"SeriesL_{i:02d}" for i in range(1, 16, 2)]
    expected += [f"ShuntC_{i:02d}_{side}" for i in range(2, 15, 2) for side in ("Top", "Bottom")]
    try:
        actual = {str(name) for name in hfss.oboundary.GetBoundaries()}
    except Exception as exc:
        raise RuntimeError(f"PRE-SOLVE ABORT: could not query AEDT boundary names: {exc}") from exc
    missing = [name for name in expected if name not in actual]
    if missing:
        raise RuntimeError("PRE-SOLVE ABORT: missing RLC boundaries: " + ", ".join(missing))
    print(f"Pre-solve boundary audit passed: {len(expected)} RLC boundaries are present.")
    for name in expected:
        print("  " + name)

def create_hfss_sparameter_report(hfss, output_dir):
    """Create a native AEDT S-parameter report through ReportSetup COM API."""
    report_name = "LP10_Iter07_SParameters"
    snapshot = str(output_dir / "LP10_Iter07_HFSS_Report.png")
    expressions = ["dB(S(Port1:1,Port1:1))", "dB(S(Port2:1,Port1:1))"]
    try:
        # Call AEDT's ReportSetup module directly. This avoids the failing
        # PyAEDT 1.9.0 create_report() wrapper route seen in the previous log.
        report_module = hfss.odesign.GetModule("ReportSetup")
        report_module.CreateReport(
            report_name,
            "Standard",
            "Rectangular Plot",
            "Setup1 : Sweep1",
            ["Domain:=", "Sweep"],
            ["Freq:=", ["All"]],
            ["X Component:=", "Freq"],
            ["Y Component:=", expressions],
        )
        hfss.save_project()
        print(f"Native HFSS report created: {report_name}")
        # The native report persists in the AEDT project. PNG from the local
        # Touchstone plotter is produced separately after export.
        return report_name
    except Exception as exc:
        print(f"WARNING: native AEDT ReportSetup creation failed: {exc}")
        traceback.print_exc()
        print("The solve remains valid if S-parameter export succeeds; diagnostics will still be exported.")
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    project_path = (OUT / f"{PROJECT_NAME}.aedt").resolve()
    if project_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing project: {project_path}\n"
            "Change PROJECT_NAME or move the existing iteration directory."
        )

    fp_hz, elements, g_load = synthesize_ladder()
    L_values = [e for e in elements if e["type"] == "L"]
    C_values = [e for e in elements if e["type"] == "C"]
    if len(L_values) != 8 or len(C_values) != 7:
        raise RuntimeError("Expected 8 series inductors and 7 shunt capacitors.")

    # Use short matching end feeds and seven distributed internal nodes.
    if END_FEED_MM <= 0 or INTERNAL_NODE_MM <= 0:
        raise RuntimeError("Chosen package is too short for the node/gap dimensions.")
    node_lengths = [END_FEED_MM] + [INTERNAL_NODE_MM] * 7 + [END_FEED_MM]
    total_geom_len = sum(node_lengths) + 8 * SERIES_GAP_MM
    if abs(total_geom_len - LENGTH_MM) > 1e-6:
        raise RuntimeError(f"Internal length accounting mismatch: {total_geom_len:.6f} mm")

    desktop = None
    hfss = None
    try:
        desktop = Desktop(
            version=AEDT_VERSION,
            student_version=STUDENT_VERSION,
            non_graphical=False,
            new_desktop=True,
        )
        hfss = Hfss(
            project=str(project_path),
            design=DESIGN_NAME,
            solution_type="Modal",
            version=AEDT_VERSION,
            new_desktop=False,
        )
        hfss.modeler.model_units = "mm"

        # The air volume represents the inside of the shielded line. The outer
        # walls are ideal PEC boundary conditions on the four longitudinal faces,
        # which avoids overlapping wall solids and leaves only the solved cavity.
        air = make_box(
            hfss, "AirInterior",
            [0, -INNER_W_MM / 2.0, -INNER_H_MM / 2.0],
            [LENGTH_MM, INNER_W_MM, INNER_H_MM],
            "air",
        )
        try:
            air.solve_inside = True
        except Exception:
            pass

        # Build 9 disjoint centre-conductor nodes. Each adjacent node is separated
        # by SERIES_GAP_MM and coupled through an ideal series-inductor RLC sheet.
        # The metal solids never overlap; node and gap positions are explicit.
        conductor_nodes = []
        node_ranges = []
        x = 0.0
        for node_idx, node_len in enumerate(node_lengths):
            node_start, node_end = x, x + node_len
            node = make_box(
                hfss, f"CenterNode_{node_idx:02d}",
                [node_start, -CENTER_STRIP_W_MM / 2.0, -CENTER_STRIP_T_MM / 2.0],
                [node_len, CENTER_STRIP_W_MM, CENTER_STRIP_T_MM],
                COPPER_MATERIAL,
            )
            conductor_nodes.append(node)
            node_ranges.append((node_start, node_end))
            x = node_end
            if node_idx < 8:
                gap_start = x
                x += SERIES_GAP_MM

        # Subtract conductor nodes from the cavity volume while retaining copper.
        # This avoids volume intersections between the solved air domain and metal.
        subtract_result = hfss.modeler.subtract(
            blank_list=[air],
            tool_list=conductor_nodes,
            keep_originals=True,
        )
        if subtract_result is False:
            raise RuntimeError("Failed subtracting center conductor from AirInterior.")
        try:
            air.solve_inside = True
        except Exception:
            pass

        # Identify the four long faces of the cavity and apply PEC. The end faces
        # remain open for the two wave ports.
        wall_face_ids = []
        for face in air.faces:
            try:
                cx, cy, cz = [float(v) for v in face.center]
            except Exception:
                continue
            if abs(cx - LENGTH_MM / 2.0) > 1e-3:
                continue
            on_side = abs(abs(cy) - INNER_W_MM / 2.0) < 1e-4
            on_top_bottom = abs(abs(cz) - INNER_H_MM / 2.0) < 1e-4
            if on_side or on_top_bottom:
                wall_face_ids.append(face.id)

        if len(wall_face_ids) != 4:
            raise RuntimeError(
                f"Expected 4 longitudinal cavity wall faces; found {len(wall_face_ids)}. "
                "Geometry saved state may need inspection."
            )
        wall_boundary = hfss.assign_perfect_e(
            assignment=wall_face_ids, name="ShieldWalls_PEC"
        )
        require(wall_boundary, "ShieldWalls_PEC")

        # Place the eight series-inductor RLC sheets in the gaps. The sheet is
        # horizontal and located at the centerline of the conductor thickness.
        # The integration line runs from one conductor node to the next along x.
        x = 0.0
        node_centres = []
        gaps = []
        for node_idx, node_len in enumerate(node_lengths):
            node_start, node_end = node_ranges[node_idx]
            if 1 <= node_idx <= 7:
                node_centres.append((node_idx, 0.5 * (node_start + node_end)))
            x = node_end
            if node_idx < 8:
                gap_start, gap_end = x, x + SERIES_GAP_MM
                gap_mid = 0.5 * (gap_start + gap_end)
                z_mid = 0.0
                sheet = require(
                    hfss.modeler.create_rectangle(
                        orientation="XY",
                        origin=[gap_start, -CENTER_STRIP_W_MM / 2.0, z_mid],
                        sizes=[SERIES_GAP_MM, CENTER_STRIP_W_MM],
                        name=f"SeriesL_Sheet_{node_idx + 1:02d}",
                    ),
                    f"SeriesL_Sheet_{node_idx + 1:02d}",
                )
                elem = L_values[node_idx]
                create_rlc(
                    hfss,
                    sheet,
                    elem["boundary_name"],
                    [[gap_start, 0.0, z_mid], [gap_end, 0.0, z_mid]],
                    "Serial",
                    inductance=elem["value_si"],
                )
                gaps.append({
                    "boundary": elem["boundary_name"],
                    "element": elem["index"],
                    "L_nH": elem["value_display"],
                    "gap_start_mm": gap_start,
                    "gap_end_mm": gap_end,
                })
                x = gap_end

        # Each shunt capacitor is split into two equal ideal capacitors to the
        # top and bottom PEC walls. Since the walls are connected, C/2 + C/2 = C.
        cap_sheet_width = 0.15
        z_strip_top = CENTER_STRIP_T_MM / 2.0
        z_wall_top = INNER_H_MM / 2.0
        z_wall_bottom = -INNER_H_MM / 2.0
        z_strip_bottom = -CENTER_STRIP_T_MM / 2.0
        cap_records = []

        for cap_idx, elem in enumerate(C_values):
            node_idx = cap_idx + 1  # C2 is on node 1, ... C14 on node 7.
            xcap = node_centres[cap_idx][1]
            if cap_sheet_width >= INTERNAL_NODE_MM:
                raise RuntimeError("Capacitor boundary width must be smaller than each internal node.")
            c_half = elem["value_si"] / 2.0

            top_sheet = require(
                hfss.modeler.create_rectangle(
                    orientation="XZ",
                    origin=[xcap - cap_sheet_width / 2.0, 0.0, z_strip_top],
                    sizes=[cap_sheet_width, z_wall_top - z_strip_top],
                    name=f"ShuntC_Top_Sheet_{elem['index']:02d}",
                ),
                f"ShuntC_Top_Sheet_{elem['index']:02d}",
            )
            create_rlc(
                hfss,
                top_sheet,
                f"{elem['boundary_name']}_Top",
                [[xcap, 0.0, z_strip_top], [xcap, 0.0, z_wall_top]],
                "Parallel",
                capacitance=float(c_half),
            )

            bottom_sheet = require(
                hfss.modeler.create_rectangle(
                    orientation="XZ",
                    origin=[xcap - cap_sheet_width / 2.0, 0.0, z_wall_bottom],
                    sizes=[cap_sheet_width, z_strip_bottom - z_wall_bottom],
                    name=f"ShuntC_Bottom_Sheet_{elem['index']:02d}",
                ),
                f"ShuntC_Bottom_Sheet_{elem['index']:02d}",
            )
            create_rlc(
                hfss,
                bottom_sheet,
                f"{elem['boundary_name']}_Bottom",
                [[xcap, 0.0, z_wall_bottom], [xcap, 0.0, z_strip_bottom]],
                "Parallel",
                capacitance=float(c_half),
            )
            cap_records.append({
                "boundary": elem["boundary_name"],
                "element": elem["index"],
                "C_pF_total": elem["value_display"],
                "C_pF_each_wall": elem["value_display"] / 2.0,
                "x_mm": xcap,
            })

        # Wave ports are on the two ends of the rectangular coax cavity.
        # The 1-mode coax/TEM response is renormalized to 50 ohms.
        port1_sheet = require(
            hfss.modeler.create_rectangle(
                orientation="YZ",
                origin=[0.0, -INNER_W_MM / 2.0, -INNER_H_MM / 2.0],
                sizes=[INNER_W_MM, INNER_H_MM],
                name="Port1Sheet",
            ),
            "Port1Sheet",
        )
        port2_sheet = require(
            hfss.modeler.create_rectangle(
                orientation="YZ",
                origin=[LENGTH_MM, -INNER_W_MM / 2.0, -INNER_H_MM / 2.0],
                sizes=[INNER_W_MM, INNER_H_MM],
                name="Port2Sheet",
            ),
            "Port2Sheet",
        )
        p1 = hfss.wave_port(
            assignment=port1_sheet.name,
            name="Port1",
            modes=PORT_MODES,
            impedance=ZREF_OHM,
            renormalize=True,
            create_pec_cap=False,
            port_on_plane=True,
        )
        p2 = hfss.wave_port(
            assignment=port2_sheet.name,
            name="Port2",
            modes=PORT_MODES,
            impedance=ZREF_OHM,
            renormalize=True,
            create_pec_cap=False,
            port_on_plane=True,
        )
        require(p1, "Port1")
        require(p2, "Port2")

        # Full-wave solution setup and sweep.
        setup = hfss.create_setup("Setup1")
        setup.props["Frequency"] = f"{F_ADAPT_GHZ:g}GHz"
        setup.props["MaximumPasses"] = MAX_PASSES
        setup.props["MinimumPasses"] = 3
        setup.props["MinimumConvergedPasses"] = 2
        setup.props["DeltaS"] = DELTA_S
        setup.update()

        sweep = hfss.create_linear_count_sweep(
            setup="Setup1",
            unit="GHz",
            start_frequency=F_START_GHZ,
            stop_frequency=F_STOP_GHZ,
            num_of_freq_points=N_SWEEP_POINTS,
            sweep_type="Interpolating",
            name="Sweep1",
            save_fields=False,
        )
        require(sweep, "Sweep1")

        # Write reproducible synthesis and geometry records.
        component_csv = OUT / "lumped_elements_iter07.csv"
        with component_csv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["Element", "Type", "Normalized_g", "Value", "Unit"])
            for e in elements:
                writer.writerow([e["index"], e["type"], f'{e["g"]:.12g}',
                                 f'{e["value_display"]:.9g}', e["unit"]])

        manifest = {
            "status": "3D hybrid EM candidate; ideal RLC boundaries; not manufacturing-ready",
            "source_electrical_reference": "15th-order, 0.1 dB Chebyshev LC prototype",
            "target_cutoff_at_minus3dB_GHz": TARGET_F3_DB_GHZ,
            "calculated_passband_edge_GHz": fp_hz / 1e9,
            "ideal_circuit_target_S21_at_12GHz_dB": -62.1,
            "envelope_mm": [LENGTH_MM, OUTER_W_MM, OUTER_H_MM],
            "assumed_wall_thickness_mm": WALL_MM,
            "interior_cavity_mm": [LENGTH_MM, INNER_W_MM, INNER_H_MM],
            "center_conductor_mm": [LENGTH_MM, CENTER_STRIP_W_MM, CENTER_STRIP_T_MM],
            "port_modes": PORT_MODES,
            "reference_impedance_ohm": ZREF_OHM,
            "node_lengths_mm": node_lengths,
            "layout_inherited_from_iter03": {
                "end_feed_mm_each": END_FEED_MM,
                "internal_node_mm_each": INTERNAL_NODE_MM,
                "series_gap_mm_each": SERIES_GAP_MM,
                "reason": "Seven shunt-capacitor nodes remain distributed as in Iteration 03; Iteration 07 specifically tests direct COM assignment with explicit units to avoid PyAEDT wrapper serialization."
            },
            "series_inductors": gaps,
            "shunt_capacitors": cap_records,
            "known_limitations": [
                "R, L, C are ideal boundary values; no discrete component package parasitics are modeled.",
                "The outer enclosure is represented by ideal PEC boundaries on the cavity surfaces, not finite-thickness copper solids.",
                "The coax-like center conductor cross-section is a first approximation; confirm the extracted modal impedance.",
                "Full-wave fringing/coupling around component gaps can create deviations from the ideal LC S2P target.",
                "The model is not a faithful CAD reproduction of the paper's stepped-impedance line and hollow-waveguide absorbers.",
                "The requested <= -60 dB stopband must be checked in the 3D solution, not inferred from the ideal circuit."
            ],
            "element_values": elements,
        }
        (OUT / "geometry_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        hfss.save_project()
        print(f"PROJECT: {project_path}")
        print(f"MANIFEST: {(OUT / 'geometry_manifest.json').resolve()}")
        print(f"ELEMENT CSV: {component_csv.resolve()}")
        print(f"Envelope: {LENGTH_MM} x {OUTER_W_MM} x {OUTER_H_MM} mm")
        print(f"Interior: {INNER_W_MM} x {INNER_H_MM} mm")
        print(f"Center conductor width x thickness: {CENTER_STRIP_W_MM} x {CENTER_STRIP_T_MM} mm")
        print(f"Calculated prototype passband edge: {fp_hz/1e9:.6f} GHz")
        print(f"Node lengths: {[round(v, 6) for v in node_lengths]}")
        print(f"Iteration 07 layout: {END_FEED_MM:.3f} mm end feeds; {INTERNAL_NODE_MM:.3f} mm each internal node; {SERIES_GAP_MM:.3f} mm series gaps")
        print("Auditing direct AEDT COM RLC assignments before geometry validation/solve...")
        try:
            audit_rlc_boundaries(hfss)
        except Exception:
            # Save the model for inspection, but critically DO NOT consume solver time.
            hfss.save_project()
            raise

        print("Validating before solve...")

        messages, is_valid = hfss.validate_full_design(
            output_dir=str(OUT), ports=PORT_MODES * 2
        )
        validation_path = OUT / "validation_report.txt"
        validation_path.write_text("\n".join(str(m) for m in messages), encoding="utf-8")
        print(f"Validation passed: {is_valid}")
        print(f"Validation report: {validation_path.resolve()}")
        if not is_valid:
            print("Validation failed; solve skipped. First validation messages:")
            for message in messages[:50]:
                print(message)
            hfss.save_project()
            input("Press Enter to close AEDT.")
            return

        if not RUN_ANALYSIS:
            hfss.save_project()
            print("\nPREFLIGHT-ONLY MODE: no HFSS analysis was launched.")
            print("Inspect AEDT Message Manager for any remaining Capacitance unit warnings.")
            print(f"Saved project for inspection: {project_path}")
            input("Press Enter to close AEDT.")
            return

        print("Starting HFSS solve...")
        ok = hfss.analyze_setup("Setup1")
        print(f"analyze_setup returned: {ok}")
        hfss.save_project()
        if not ok:
            print("HFSS solve failed. Read the first red error in AEDT Message Manager.")
            input("Press Enter to close AEDT.")
            return

        # Create the native report immediately after solve, before data export.
        report = create_hfss_sparameter_report(hfss, OUT)
        hfss.save_project()

        # Export Touchstone; retain project and CSV if the installed PyAEDT
        # version does not accept this export signature.
        s2p_path = OUT / "Chebyshev15_10GHz_3D_Hybrid.s2p"
        try:
            exported = hfss.export_touchstone(
                setup="Setup1",
                sweep="Sweep1",
                output_file=str(s2p_path),
                renormalization=True,
                impedance=ZREF_OHM,
            )
            print(f"Touchstone export return: {exported}")
            print(f"Touchstone exists: {s2p_path.exists()}")
        except Exception as exc:
            print(f"WARNING: Touchstone export failed: {exc}")

        # Native AEDT report was attempted immediately after solve (before Touchstone export).
        # The independent S2P-derived PNG/CSV below remain available even if report API fails.

        if s2p_path.exists():
            try:
                write_response_diagnostics(s2p_path, OUT)
            except Exception as exc:
                print(f"WARNING: post-solve S2P diagnostics failed: {exc}")
                traceback.print_exc()

        # Save again after report creation so the S-parameter plot persists in the project.
        try:
            hfss.save_project()
            print(f"Saved AEDT project with report: {project_path}")
        except Exception as exc:
            print(f"WARNING: final save after report creation failed: {exc}")

        print("3D hybrid solve completed. Compare S21 to the ideal LC reference.")
        print("This is not yet a manufacturing-ready physical component model.")
        input("Press Enter to close AEDT.")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if desktop is not None:
            try:
                desktop.release_desktop(close_projects=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()
