
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNNER = PROJECT_ROOT / "wideband_lp_filter_v2.py"

OPT_ROOT = PROJECT_ROOT / "results" / "optimizer_v3"
LOG_ROOT = OPT_ROOT / "candidate_logs"
OPT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)

# This is the known-good design. Candidate 000 is always this exact baseline.
BASELINE = {
    "A": 4.00,
    "B": 5.00,
    "D": 4.90,
    "RING": 5.00,
    "PITCH": 12.00,
    "N": 4,
}

# Local search around the baseline only.
BOUNDS = {
    "A": (3.50, 4.80),
    "B": (4.25, 5.75),
    "D": (4.00, 6.00),
    "RING": (4.00, 6.00),
    "PITCH": (10.0, 14.0),
    "N": (4, 5),
}

SEED = 20260911
rng = random.Random(SEED)
N_INITIAL = 10
N_LOCAL = 6


def clamp(c):
    c = dict(c)
    for k in ["A", "B", "D", "RING", "PITCH"]:
        lo, hi = BOUNDS[k]
        c[k] = max(lo, min(hi, c[k]))
    c["N"] = int(max(4, min(5, round(c["N"]))))
    c["PITCH"] = max(c["PITCH"], c["RING"] + 4.0)
    return c


def local_candidate(scale=1.0):
    c = dict(BASELINE)
    steps = {
        "A": 0.45,
        "B": 0.50,
        "D": 0.70,
        "RING": 0.60,
        "PITCH": 1.10,
    }
    for k, step in steps.items():
        c[k] += rng.uniform(-step * scale, step * scale)
    return clamp(c)


def env_for(c, run_id):
    return {
        **os.environ,
        "WB_A": f"{c['A']:.5f}",
        "WB_B": f"{c['B']:.5f}",
        "WB_D": f"{c['D']:.5f}",
        "WB_RING": f"{c['RING']:.5f}",
        "WB_PITCH": f"{c['PITCH']:.5f}",
        "WB_NSECTIONS": str(int(c["N"])),
        "WB_RUN_ID": run_id,
    }


def objective(m):
    pb = float(m["passband_worst_loss_1_9_dB"])
    cut = float(m["cutoff_3dB_GHz"])
    stop = float(m["max_s21_20_100_dB"])
    mid = float(m["max_s21_12_20_dB"])

    # Passband is a hard requirement.
    if pb > 3.0:
        return 1e7 + 1e5 * (pb - 3.0)

    # Catastrophic stopband candidates are rejected.
    if stop > -20.0:
        return 5e6 + 1e4 * (stop + 20.0)

    pb_pen = max(0.0, pb - 1.0)
    cut_pen = abs(cut - 10.0) if math.isfinite(cut) else 1000.0
    mid_pen = max(0.0, mid + 45.0)
    stop_pen = max(0.0, stop + 60.0)

    return (
        300.0 * pb_pen
        + 12.0 * cut_pen
        + 2.0 * mid_pen
        + 6.0 * stop_pen
    )


def run_candidate(c, number):
    c = clamp(c)
    run_id = f"opt_{number:03d}"

    print("\n" + "=" * 100, flush=True)
    print(f"CANDIDATE {number:03d}", flush=True)
    print(json.dumps(c, indent=2), flush=True)
    print("=" * 100, flush=True)

    t0 = time.perf_counter()

    proc = subprocess.Popen(
        [sys.executable, str(RUNNER)],
        cwd=str(PROJECT_ROOT),
        env=env_for(c, run_id),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    lines = []
    assert proc.stdout is not None

    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"[CAND {number:03d}] {line}", flush=True)
            lines.append(line)

    proc.wait()
    elapsed = time.perf_counter() - t0

    (LOG_ROOT / f"candidate_{number:03d}.log").write_text(
        "\n".join(lines), encoding="utf-8", errors="replace"
    )

    metrics_file = (
        PROJECT_ROOT / "results" / "em" / run_id / "metrics.json"
    )

    if not metrics_file.exists():
        print(
            f"[CAND {number:03d}] FAILED — metrics.json not found",
            flush=True,
        )
        return None

    with metrics_file.open("r", encoding="utf-8") as f:
        m = json.load(f)

    m["candidate"] = c
    m["objective"] = objective(m)
    m["wall_time_s"] = elapsed
    m["process_return_code"] = proc.returncode

    # Strict feasibility flag.
    m["feasible"] = (
        m["passband_worst_loss_1_9_dB"] <= 1.0
        and math.isfinite(m["cutoff_3dB_GHz"])
        and abs(m["cutoff_3dB_GHz"] - 10.0) <= 1.5
        and m["max_s21_20_100_dB"] <= -60.0
    )

    print(
        f"[RESULT {number:03d}] "
        f"score={m['objective']:.2f} | "
        f"PBworst={m['passband_worst_loss_1_9_dB']:.2f} dB | "
        f"f3dB={m['cutoff_3dB_GHz']:.3f} GHz | "
        f"S21@10={m['s21_10GHz_dB']:.2f} dB | "
        f"max20-100={m['max_s21_20_100_dB']:.2f} dB | "
        f"FEASIBLE={m['feasible']} | "
        f"time={elapsed/60:.2f} min",
        flush=True,
    )
    return m


def save(results):
    ordered = sorted(results, key=lambda x: x["objective"])

    with (OPT_ROOT / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)

    fields = [
        "objective", "feasible",
        "a_mm", "b_mm", "d_mm",
        "ring_spacing_mm", "section_pitch_mm", "n_sections",
        "passband_worst_loss_1_9_dB",
        "cutoff_3dB_GHz",
        "s21_10GHz_dB", "s21_20GHz_dB",
        "max_s21_12_20_dB",
        "max_s21_20_100_dB",
        "max_s21_20_100_frequency_Hz",
        "wall_time_s",
    ]

    with (OPT_ROOT / "all_results.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for m in ordered:
            f.write(",".join(str(m.get(k, "")) for k in fields) + "\n")

    feasible = [m for m in ordered if m.get("feasible")]
    best = min(feasible, key=lambda x: x["objective"]) if feasible else ordered[0]

    with (OPT_ROOT / "best_so_far.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    with (OPT_ROOT / "best_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(best["candidate"], f, indent=2)

    return best


def progress(results):
    best = save(results)
    print("-" * 100, flush=True)
    print(
        f"[OPTIMIZER] completed={len(results)} | "
        f"best={best['objective']:.2f} | "
        f"feasible={best['feasible']}",
        flush=True,
    )
    print(
        f"[BEST] A={best['a_mm']:.3f} B={best['b_mm']:.3f} "
        f"D={best['d_mm']:.3f} ring={best['ring_spacing_mm']:.3f} "
        f"pitch={best['section_pitch_mm']:.3f} N={best['n_sections']}",
        flush=True,
    )
    print(
        f"[BEST RESPONSE] PB={best['passband_worst_loss_1_9_dB']:.2f} dB | "
        f"f3dB={best['cutoff_3dB_GHz']:.3f} GHz | "
        f"max20-100={best['max_s21_20_100_dB']:.2f} dB",
        flush=True,
    )
    print("-" * 100, flush=True)


results = []

# Always establish the baseline first.
m = run_candidate(BASELINE, 0)
if m:
    results.append(m)
    progress(results)

# Local search.
for i in range(1, N_INITIAL + 1):
    m = run_candidate(local_candidate(1.0), i)
    if m:
        results.append(m)
        progress(results)

# Refine around the best candidate.
for j in range(1, N_LOCAL + 1):
    best = min(results, key=lambda x: x["objective"])
    c = dict(best["candidate"])
    scale = 0.45 if j <= 2 else 0.25

    for k, step in {
        "A": 0.30,
        "B": 0.30,
        "D": 0.45,
        "RING": 0.40,
        "PITCH": 0.70,
    }.items():
        c[k] += rng.uniform(-step * scale, step * scale)

    m = run_candidate(clamp(c), N_INITIAL + j)
    if m:
        results.append(m)
        progress(results)

if results:
    best = save(results)
    print("\n" + "=" * 100)
    print("WIDEBAND LP FILTER V3 OPTIMIZATION COMPLETE")
    print("=" * 100)
    print(json.dumps(best, indent=2))
    print()
    print(f"All results     : {OPT_ROOT / 'all_results.csv'}")
    print(f"Best result     : {OPT_ROOT / 'best_so_far.json'}")
    print(f"Best parameters : {OPT_ROOT / 'best_parameters.json'}")
    print(f"Logs            : {LOG_ROOT}")
else:
    raise SystemExit("No candidate completed.")
