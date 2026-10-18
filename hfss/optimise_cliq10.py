"""
CliQ10 Iteration 01 closed-loop HFSS optimiser
================================================

Runs on the user's Windows machine with PyAEDT 1.9.x and AEDT Student 2025 R2.
Does NOT modify hfss/cliq10_iter01/CliQ10_Iter01.aedt.

Loop:
  1. Score existing Iteration 01 Touchstone as the baseline.
  2. Coarse design-space search (block positions/length profile, strip width,
     and section thickness scale).
  3. Two coordinate-search refinement rounds with smaller steps.
  4. Every candidate gets a fresh project, a complete HFSS solve, Touchstone
     export, score, and a record in the run ledger.

Run from repository root:
  .\\.venv\\Scripts\\python.exe hfss\\optimise_cliq10.py

IMPORTANT:
- One full-wave solve can take several minutes. Default budget: 43 evaluations
  including the existing baseline; expect a long run.
- Do not run multiple instances of this script at once.
- The 0.01-40 GHz sweep does not verify a 55 GHz rejection requirement.
- This is an engineering search of the Iteration 01 proxy geometry, not a
  validated reconstruction of the commercial CliQ10 filter.
"""

from __future__ import annotations

import csv
import json
import itertools
import math
import traceback
from pathlib import Path

import numpy as np
from ansys.aedt.core import Desktop, Hfss

# ---------------- Paths / AEDT ----------------
ROOT = Path.cwd()
BASELINE_DIR = ROOT / "hfss" / "cliq10_iter01"
BASELINE_S2P = BASELINE_DIR / "CliQ10_Iter01.s2p"
OUT_DIR = ROOT / "hfss" / "cliq10_optimizer"
AEDT_VERSION = "2025.2"
STUDENT_VERSION = True

# ---------------- Fixed package geometry (mm) ----------------
L = 65.3
OUTER_W = 5.0
OUTER_H = 4.6
SIDE_WALL = 0.20
TOP_BOTTOM_WALL = 0.20
INNER_W = OUTER_W - 2 * SIDE_WALL
INNER_H = OUTER_H - 2 * TOP_BOTTOM_WALL
FEED_LENGTH = 4.0
FEED_THICKNESS = 0.20

# ---------------- Baseline design variables ----------------
BASE_WIDTH = 1.20
BASE_THIN = 0.12
BASE_THICK = 0.35
BASE_WEIGHTS = np.array([1.0, 1.2, 1.5, 1.8, 2.0, 1.8, 1.5, 1.2, 1.0], dtype=float)

# Each candidate is parameterized by:
# center_bias: redistributes length toward centre (+) or ends (-)
# alternation: changes odd/even section lengths (moves individual boundaries)
# strip_width_mm: width of all centre-conductor blocks
# thickness_scale: scales both thin/thick section heights from Iteration 01
BASE_PARAMS = {
    "center_bias": 0.0,
    "alternation": 0.0,
    "strip_width_mm": BASE_WIDTH,
    "thickness_scale": 1.0,
}

# ---------------- Solve configuration ----------------
F_ADAPT = "40GHz"
F_START_GHZ = 0.01
F_STOP_GHZ = 40.0
MAX_PASSES = 10
DELTA_S = 0.02
PORT_MODES = 2

# ---------------- Search budget ----------------
# Baseline is scored from the already-exported S2P, so it costs no solve.
MAX_EVALUATIONS = 43
REFINEMENT_ROUNDS = 2
COARSE_CENTER_BIAS = 0.70
COARSE_ALTERNATION = 0.50
COARSE_WIDTH_DELTA = 0.30
COARSE_THICKNESS_SCALE_DELTA = 0.30
REFINE_STEPS = {
    "center_bias": 0.35,
    "alternation": 0.25,
    "strip_width_mm": 0.15,
    "thickness_scale": 0.15,
}
MIN_WIDTH = 0.50
MAX_WIDTH = 2.50
MIN_THICKNESS_SCALE = 0.35
MAX_THICKNESS_SCALE = 2.00
MIN_SECTION_LENGTH = 1.0

# Target criteria. S-parameter magnitudes are in dB.
PASS1_END_GHZ = 7.5
PASS2_END_GHZ = 9.3
CUTOFF_TARGET_GHZ = 10.0
STOP_START_GHZ = 11.0
PASS1_S21_MIN_DB = -0.5
PASS2_S21_MIN_DB = -1.5
RETURN_S11_MAX_DB = -15.0
STOP_S21_MAX_DB = -30.0
CUTOFF_S21_TARGET_DB = -3.0

LEDGER_CSV = OUT_DIR / "optimisation_history.csv"
BEST_JSON = OUT_DIR / "best_design.json"
BEST_S2P_COPY = OUT_DIR / "best_design.s2p"
BEST_PARAMS_JSON = OUT_DIR / "best_geometry.json"


def create_box(hfss, name, origin, sizes, material):
    obj = hfss.modeler.create_box(
        origin=origin, sizes=sizes, name=name, material=material
    )
    if obj is None or obj is False:
        raise RuntimeError(f"HFSS failed to create object {name}")
    return obj


def clamp_params(p):
    return {
        "center_bias": float(np.clip(p["center_bias"], -1.5, 1.5)),
        "alternation": float(np.clip(p["alternation"], -1.0, 1.0)),
        "strip_width_mm": float(np.clip(p["strip_width_mm"], MIN_WIDTH, MAX_WIDTH)),
        "thickness_scale": float(np.clip(
            p["thickness_scale"], MIN_THICKNESS_SCALE, MAX_THICKNESS_SCALE
        )),
    }


def section_lengths(params):
    """Create a normalized, continuous section profile; total active length stays fixed."""
    weights = BASE_WEIGHTS.copy()
    n = len(weights)
    positions = np.linspace(-1.0, 1.0, n)
    # Positive center_bias increases the centre-section lengths.
    center_profile = 1.0 - np.abs(positions)
    weights *= np.exp(params["center_bias"] * center_profile)
    # Alternation shifts boundaries by changing the lengths of alternating blocks.
    signs = np.array([1.0 if i % 2 else -1.0 for i in range(n)])
    weights *= np.exp(params["alternation"] * signs)
    weights = np.maximum(weights, 1e-6)
    active_length = L - 2 * FEED_LENGTH
    lengths = active_length * weights / np.sum(weights)
    if np.min(lengths) < MIN_SECTION_LENGTH:
        raise ValueError(
            f"Candidate creates a section shorter than {MIN_SECTION_LENGTH} mm."
        )
    return lengths.tolist()


def parse_touchstone_4port(path):
    """Parse HFSS four-modal-port S MA/RI/DB data, including wrapped records."""
    unit = "ghz"
    fmt = "ma"
    z0 = 50.0
    values = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("!") or line.startswith(";"):
                continue
            if line.startswith("#"):
                tokens = line[1:].lower().split()
                for t in tokens:
                    if t in ("hz", "khz", "mhz", "ghz"):
                        unit = t
                    elif t in ("ma", "ri", "db"):
                        fmt = t
                if "r" in tokens:
                    try:
                        z0 = float(tokens[tokens.index("r") + 1])
                    except (IndexError, ValueError):
                        pass
                continue
            line = line.split("!")[0].strip()
            if not line:
                continue
            try:
                values.extend(float(v) for v in line.split())
            except ValueError:
                continue

    record = 1 + 2 * 4 * 4  # frequency + 16 complex S-parameters
    if len(values) % record:
        raise ValueError(
            f"{path.name}: {len(values)} numeric values are not divisible by "
            f"{record}; expected four-port Touchstone data."
        )
    if not values:
        raise ValueError(f"No numeric Touchstone data in {path}")

    data = np.asarray(values, dtype=float).reshape(-1, record)
    fscale = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0}[unit]
    freq = data[:, 0] * fscale
    pairs = data[:, 1:].reshape(-1, 16, 2)

    a = pairs[:, :, 0]
    b = pairs[:, :, 1]
    if fmt == "ma":
        sflat = a * np.exp(1j * np.deg2rad(b))
    elif fmt == "ri":
        sflat = a + 1j * b
    elif fmt == "db":
        sflat = 10 ** (a / 20.0) * np.exp(1j * np.deg2rad(b))
    else:
        raise ValueError(f"Unsupported Touchstone format {fmt}")

    # Touchstone ordering is column-major:
    # S11,S21,S31,S41,S12,S22,...; modal port order comes from HFSS header:
    # 1=Port1:1, 2=Port1:2, 3=Port2:1, 4=Port2:2.
    smat = np.empty((len(freq), 4, 4), dtype=complex)
    for k in range(16):
        smat[:, k % 4, k // 4] = sflat[:, k]

    # Physical mode-1 input/output terms: S11, S31, S13, S33 in 4-port notation.
    s11 = smat[:, 0, 0]
    s21_mode1 = smat[:, 2, 0]
    s12_mode1 = smat[:, 0, 2]
    s22_mode1 = smat[:, 2, 2]
    return freq, s11, s21_mode1, s12_mode1, s22_mode1, z0, fmt


def db20(v):
    return 20.0 * np.log10(np.maximum(np.abs(v), 1e-15))


def score_touchstone(path):
    f, s11, s21, s12, s22, z0, fmt = parse_touchstone_4port(path)
    s11db, s21db, s12db, s22db = [db20(x) for x in (s11, s21, s12, s22)]

    def band(lo, hi):
        mask = (f >= lo) & (f <= hi)
        if not np.any(mask):
            raise ValueError(f"No sweep samples in required band {lo}-{hi} GHz")
        return mask

    m1 = band(F_START_GHZ, PASS1_END_GHZ)
    m2 = band(PASS1_END_GHZ, PASS2_END_GHZ)
    mr = band(F_START_GHZ, PASS2_END_GHZ)
    ms = band(STOP_START_GHZ, min(F_STOP_GHZ, float(np.max(f))))

    # Normalized squared violations. Lower is better; zero means every sampled
    # point satisfies all envelope constraints and the 10 GHz -3 dB anchor.
    p_pass1 = float(np.mean((np.maximum(0.0, PASS1_S21_MIN_DB - s21db[m1]) / 0.5) ** 2))
    p_pass2 = float(np.mean((np.maximum(0.0, PASS2_S21_MIN_DB - s21db[m2]) / 1.0) ** 2))
    p_return = float(np.mean((np.maximum(0.0, s11db[mr] - RETURN_S11_MAX_DB) / 5.0) ** 2))
    p_stop = float(np.mean((np.maximum(0.0, s21db[ms] - STOP_S21_MAX_DB) / 10.0) ** 2))
    s21_cutoff = float(np.interp(CUTOFF_TARGET_GHZ, f, s21db))
    p_cutoff = float(((s21_cutoff - CUTOFF_S21_TARGET_DB) / 3.0) ** 2)

    # Heavier emphasis on stopband and passband specifications than the single cutoff anchor.
    score = 8.0 * p_pass1 + 5.0 * p_pass2 + 5.0 * p_return + 8.0 * p_stop + 2.0 * p_cutoff

    metrics = {
        "score": float(score),
        "penalty_passband_0_7p5": p_pass1,
        "penalty_passband_7p5_9p3": p_pass2,
        "penalty_return_loss_0_9p3": p_return,
        "penalty_stopband_11_40": p_stop,
        "penalty_cutoff_10GHz": p_cutoff,
        "s21_at_10GHz_dB": s21_cutoff,
        "worst_s21_0_7p5_dB": float(np.min(s21db[m1])),
        "worst_s21_7p5_9p3_dB": float(np.min(s21db[m2])),
        "worst_s11_0_9p3_dB": float(np.max(s11db[mr])),
        "worst_s21_11_40_dB": float(np.max(s21db[ms])),
        "frequency_min_GHz": float(np.min(f)),
        "frequency_max_GHz": float(np.max(f)),
        "frequency_points": int(len(f)),
        "reference_impedance_ohm": float(z0),
        "touchstone_format": fmt,
    }

    # Save full curves for every successful candidate.
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "Frequency_GHz", "S11_dB", "S21_mode1_dB", "S12_mode1_dB", "S22_mode1_dB",
            "S11_mag", "S21_mode1_mag", "S12_mode1_mag", "S22_mode1_mag",
        ])
        for i in range(len(f)):
            w.writerow([
                f"{f[i]:.9g}", f"{s11db[i]:.8g}", f"{s21db[i]:.8g}",
                f"{s12db[i]:.8g}", f"{s22db[i]:.8g}", f"{abs(s11[i]):.8g}",
                f"{abs(s21[i]):.8g}", f"{abs(s12[i]):.8g}", f"{abs(s22[i]):.8g}",
            ])
    return metrics, csv_path


def build_and_solve(params, eval_dir, eval_id):
    """Build one independent HFSS project, solve, and export its Touchstone file."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    project_name = f"CliQ10_Opt_{eval_id:03d}"
    project_path = (eval_dir / f"{project_name}.aedt").resolve()
    s2p_path = (eval_dir / f"{project_name}.s2p").resolve()
    lengths = section_lengths(params)
    thin = BASE_THIN * params["thickness_scale"]
    thick = BASE_THICK * params["thickness_scale"]
    width = params["strip_width_mm"]

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
            design=project_name,
            solution_type="Modal",
            version=AEDT_VERSION,
            new_desktop=False,
        )
        hfss.modeler.model_units = "mm"

        # Enclosure walls (same as Iteration 01).
        create_box(hfss, "WallBottom", [0, -OUTER_W/2, -OUTER_H/2],
                   [L, OUTER_W, TOP_BOTTOM_WALL], "copper")
        create_box(hfss, "WallTop", [0, -OUTER_W/2, OUTER_H/2-TOP_BOTTOM_WALL],
                   [L, OUTER_W, TOP_BOTTOM_WALL], "copper")
        create_box(hfss, "WallYMinus", [0, -OUTER_W/2, -OUTER_H/2+TOP_BOTTOM_WALL],
                   [L, SIDE_WALL, OUTER_H-2*TOP_BOTTOM_WALL], "copper")
        create_box(hfss, "WallYPlus", [0, OUTER_W/2-SIDE_WALL, -OUTER_H/2+TOP_BOTTOM_WALL],
                   [L, SIDE_WALL, OUTER_H-2*TOP_BOTTOM_WALL], "copper")

        conductors = []
        conductors.append(create_box(
            hfss, "InputFeed",
            [0, -width/2, -FEED_THICKNESS/2],
            [FEED_LENGTH, width, FEED_THICKNESS], "copper"
        ))
        x = FEED_LENGTH
        section_rows = []
        for i, seg_len in enumerate(lengths):
            thickness = thin if i % 2 == 0 else thick
            name = f"CoreSection{i+1:02d}"
            obj = create_box(
                hfss, name,
                [x, -width/2, -thickness/2],
                [seg_len, width, thickness], "copper"
            )
            conductors.append(obj)
            section_rows.append({
                "name": name, "x_start_mm": x, "length_mm": seg_len,
                "thickness_mm": thickness,
            })
            x += seg_len

        conductors.append(create_box(
            hfss, "OutputFeed",
            [L-FEED_LENGTH, -width/2, -FEED_THICKNESS/2],
            [FEED_LENGTH, width, FEED_THICKNESS], "copper"
        ))

        air = create_box(
            hfss, "AirInterior",
            [0, -INNER_W/2, -INNER_H/2], [L, INNER_W, INNER_H], "air"
        )
        sub = hfss.modeler.subtract(
            blank_list=air, tool_list=conductors, keep_originals=True
        )
        if sub is False:
            raise RuntimeError("Subtracting conductor solids from AirInterior failed")
        air.solve_inside = True

        p1sheet = hfss.modeler.create_rectangle(
            orientation="YZ", origin=[0, -INNER_W/2, -INNER_H/2],
            sizes=[INNER_W, INNER_H], name="Port1Sheet"
        )
        p2sheet = hfss.modeler.create_rectangle(
            orientation="YZ", origin=[L, -INNER_W/2, -INNER_H/2],
            sizes=[INNER_W, INNER_H], name="Port2Sheet"
        )
        if p1sheet is None or p2sheet is None:
            raise RuntimeError("Failed to create wave-port sheets")

        p1 = hfss.wave_port(
            assignment=p1sheet.name, name="Port1", modes=PORT_MODES,
            impedance=50, renormalize=True
        )
        p2 = hfss.wave_port(
            assignment=p2sheet.name, name="Port2", modes=PORT_MODES,
            impedance=50, renormalize=True
        )
        if p1 is None or p2 is None or p1 is False or p2 is False:
            raise RuntimeError("Failed to create wave ports")

        setup = hfss.create_setup("Setup1")
        setup.props["Frequency"] = F_ADAPT
        setup.props["MaximumPasses"] = MAX_PASSES
        setup.props["MinimumPasses"] = 2
        setup.props["MinimumConvergedPasses"] = 1
        setup.props["DeltaS"] = DELTA_S
        if not setup.update():
            raise RuntimeError("Failed to update Setup1")
        sweep = setup.create_frequency_sweep(
            unit="GHz", name="Sweep1",
            start_frequency=F_START_GHZ, stop_frequency=F_STOP_GHZ,
            sweep_type="Interpolating", save_fields=False,
        )
        if sweep is None or sweep is False:
            raise RuntimeError("Failed to create Sweep1")

        geometry = {
            "eval_id": eval_id,
            "project": str(project_path),
            "params": params,
            "section_lengths_mm": lengths,
            "section_table": section_rows,
            "section_thin_mm": thin,
            "section_thick_mm": thick,
            "strip_width_mm": width,
        }
        (eval_dir / "geometry.json").write_text(
            json.dumps(geometry, indent=2), encoding="utf-8"
        )
        hfss.save_project()

        print(f"\n[EVAL {eval_id:03d}] Solving {project_name}")
        print(f"  parameters={json.dumps(params)}")
        result = hfss.analyze_setup("Setup1")
        print(f"  analyze_setup returned: {result}")
        if not result:
            raise RuntimeError("HFSS analyze_setup('Setup1') returned false")
        hfss.save_project()

        exported = hfss.export_touchstone(
            setup="Setup1",
            sweep="Sweep1",
            output_file=str(s2p_path),
            renormalization=False,
        )
        if not s2p_path.exists() or s2p_path.stat().st_size < 100:
            raise RuntimeError(f"Touchstone export failed; return={exported}")
        print(f"  Touchstone exported: {s2p_path}")
        return s2p_path
    finally:
        if desktop is not None:
            try:
                desktop.release_desktop(close_projects=True)
            except Exception:
                pass


def append_ledger(row):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exists = LEDGER_CSV.exists()
    columns = [
        "eval_id", "stage", "status", "score", "center_bias", "alternation",
        "strip_width_mm", "thickness_scale", "s21_at_10GHz_dB",
        "worst_s21_0_7p5_dB", "worst_s21_7p5_9p3_dB",
        "worst_s11_0_9p3_dB", "worst_s21_11_40_dB", "project_or_s2p", "error",
    ]
    with LEDGER_CSV.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def write_best(best):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "BEST FOUND BY SEARCH; NOT YET VALIDATED AT 55 GHz",
        "score": best["score"],
        "params": best["params"],
        "metrics": best["metrics"],
        "project_or_s2p": best["s2p"],
    }
    BEST_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    BEST_PARAMS_JSON.write_text(
        json.dumps(best["geometry"], indent=2), encoding="utf-8"
    )
    source = Path(best["s2p"])
    if source.exists():
        import shutil
        shutil.copy2(source, BEST_S2P_COPY)


def evaluate_candidate(params, eval_id, stage):
    params = clamp_params(params)
    # Reuse the baseline S2P for the exact baseline geometry.
    if all(abs(params[k] - BASE_PARAMS[k]) < 1e-12 for k in BASE_PARAMS):
        s2p = BASELINE_S2P
        if not s2p.exists():
            raise FileNotFoundError(
                f"Baseline Touchstone missing: {s2p}. Run the existing export script first."
            )
        geometry = {
            "eval_id": 0, "params": params, "section_lengths_mm": section_lengths(params),
            "section_thin_mm": BASE_THIN, "section_thick_mm": BASE_THICK,
            "strip_width_mm": BASE_WIDTH, "project": str(BASELINE_DIR / "CliQ10_Iter01.aedt"),
        }
    else:
        eval_dir = OUT_DIR / f"eval_{eval_id:03d}"
        s2p = build_and_solve(params, eval_dir, eval_id)
        geometry = json.loads((eval_dir / "geometry.json").read_text(encoding="utf-8"))

    metrics, curve_csv = score_touchstone(s2p)
    metrics["curve_csv"] = str(curve_csv)
    print(f"[EVAL {eval_id:03d}] score={metrics['score']:.6f} "
          f"S21@10GHz={metrics['s21_at_10GHz_dB']:.3f} dB")
    row = {
        "eval_id": eval_id,
        "stage": stage,
        "status": "OK",
        "score": metrics["score"],
        **params,
        **metrics,
        "project_or_s2p": str(s2p),
        "error": "",
    }
    append_ledger(row)
    return {
        "eval_id": eval_id, "stage": stage, "params": params,
        "score": metrics["score"], "metrics": metrics, "s2p": str(s2p),
        "geometry": geometry,
    }


def candidate_key(params):
    return tuple(round(params[k], 6) for k in (
        "center_bias", "alternation", "strip_width_mm", "thickness_scale"
    ))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BASELINE_S2P.exists():
        raise FileNotFoundError(
            f"Missing baseline S2P: {BASELINE_S2P}\n"
            "First export the solved Iteration 01 Touchstone file."
        )

    print("=" * 72)
    print("CliQ10 closed-loop geometry optimiser")
    print("Iteration 01 project will NOT be modified.")
    print(f"Output: {OUT_DIR}")
    print(f"Evaluation budget: {MAX_EVALUATIONS} (baseline included, no solve for baseline)")
    print("Sweep: 0.01-40 GHz; 55 GHz rejection is outside this sweep.")
    print("=" * 72)

    evaluated = {}
    best = None
    next_id = 1

    # Score the existing baseline data without spending an HFSS solve.
    baseline = evaluate_candidate(dict(BASE_PARAMS), 0, "baseline")
    evaluated[candidate_key(baseline["params"])] = baseline
    best = baseline
    write_best(best)
    next_id = 1

    # Stage 1 is a genuine coarse 3-D factorial grid over section placement
    # (center_bias and alternation) and section thickness scale: 3 x 3 x 3 = 27
    # coarse points, including the already-scored baseline. Width is held at
    # baseline during coarse screening and refined locally in Stage 2.
    coarse_values = list(itertools.product(
        [-COARSE_CENTER_BIAS, 0.0, +COARSE_CENTER_BIAS],
        [-COARSE_ALTERNATION, 0.0, +COARSE_ALTERNATION],
        [1.0-COARSE_THICKNESS_SCALE_DELTA, 1.0, 1.0+COARSE_THICKNESS_SCALE_DELTA],
    ))
    print("\\n--- STAGE 1: COARSE 3-D FACTORIAL SEARCH (27 GRID POINTS) ---")
    for center_bias, alternation, thickness_scale in coarse_values:
        if next_id >= MAX_EVALUATIONS:
            break
        p = dict(BASE_PARAMS)
        p.update({
            "center_bias": center_bias,
            "alternation": alternation,
            "thickness_scale": thickness_scale,
        })
        p = clamp_params(p)
        key = candidate_key(p)
        if key in evaluated:
            continue
        try:
            result = evaluate_candidate(p, next_id, "coarse")
            evaluated[key] = result
            if result["score"] < best["score"]:
                best = result
                write_best(best)
                print(f"  NEW BEST: eval {best['eval_id']} score={best['score']:.6f}")
        except Exception as exc:
            print(f"[EVAL {next_id:03d}] FAILED: {exc}")
            append_ledger({
                "eval_id": next_id, "stage": "coarse", "status": "FAILED",
                **p, "project_or_s2p": str(OUT_DIR / f"eval_{next_id:03d}"),
                "error": repr(exc),
            })
        next_id += 1

    # Coordinate/pattern refinement: evaluate +/- step for each variable, then
    # halve all steps between rounds. This is derivative-free and robust for
    # expensive, resonant EM simulations.
    print("\n--- STAGE 2: COARSE-TO-FINE COORDINATE REFINEMENT ---")
    steps = dict(REFINE_STEPS)
    variable_names = list(REFINE_STEPS.keys())

    for round_index in range(REFINEMENT_ROUNDS):
        if next_id >= MAX_EVALUATIONS:
            break
        print(f"\nRefinement round {round_index + 1}/{REFINEMENT_ROUNDS}; steps={steps}")
        improved_this_round = False

        # Start each coordinate from the current best, test negative and positive
        # steps, then immediately accept a better point before moving to next axis.
        for name in variable_names:
            if next_id >= MAX_EVALUATIONS:
                break
            coordinate_best = best
            for direction in (-1.0, +1.0):
                if next_id >= MAX_EVALUATIONS:
                    break
                p = dict(best["params"])
                p[name] += direction * steps[name]
                p = clamp_params(p)
                key = candidate_key(p)
                if key in evaluated:
                    continue
                try:
                    result = evaluate_candidate(p, next_id, f"refine_{round_index+1}")
                    evaluated[key] = result
                    if result["score"] < best["score"]:
                        best = result
                        coordinate_best = result
                        improved_this_round = True
                        write_best(best)
                        print(f"  NEW BEST: eval {best['eval_id']} score={best['score']:.6f}")
                except Exception as exc:
                    print(f"[EVAL {next_id:03d}] FAILED: {exc}")
                    append_ledger({
                        "eval_id": next_id, "stage": f"refine_{round_index+1}",
                        "status": "FAILED", **p,
                        "project_or_s2p": str(OUT_DIR / f"eval_{next_id:03d}"),
                        "error": repr(exc),
                    })
                next_id += 1
            # `best` may have changed during the +/- trials; the next variable
            # starts from that best candidate.
        if not improved_this_round:
            print("No improvement in this round; halving refinement steps.")
        steps = {k: v / 2.0 for k, v in steps.items()}

    # Final best summary and full geometry manifest.
    write_best(best)
    final = {
        "evaluations_attempted": next_id,
        "unique_successful_candidates": len(evaluated),
        "best_eval_id": best["eval_id"],
        "best_score": best["score"],
        "best_params": best["params"],
        "best_metrics": best["metrics"],
        "best_touchstone": str(BEST_S2P_COPY),
        "best_geometry_json": str(BEST_PARAMS_JSON),
        "history_csv": str(LEDGER_CSV),
        "important_limit": "This sweep ends at 40 GHz; no conclusion at 55 GHz is possible.",
    }
    (OUT_DIR / "final_report.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("OPTIMISATION CYCLE FINISHED")
    print(json.dumps(final, indent=2))
    print("=" * 72)
    print("Inspect optimisation_history.csv and best_design.s2p.")
    print("The score is a constraint-violation objective; lower is better.")
    print("A low score does not itself validate the design; inspect the curves.")
    input("Press Enter to close.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("Optimizer stopped with an error. Press Enter to close.")
