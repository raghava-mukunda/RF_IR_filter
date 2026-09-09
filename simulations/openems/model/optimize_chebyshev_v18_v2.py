"""
V18 COARSE-TO-FINE FULL-WAVE GEOMETRY OPTIMIZER

Every candidate is a real openEMS FDTD simulation.

The optimizer is intentionally conservative:
  - 7-section symmetric topology is retained.
  - 50-ohm feed remains fixed.
  - transition length remains fixed at 1.00 mm.
  - only six independent geometry variables are searched:
        W_HIGH, W_LOW, L1, L2, L3, L4
  - lengths map as [L1,L2,L3,L4,L3,L2,L1].
  - widths are restricted to 0.50-mm increments.
  - lengths are restricted to 0.25-mm increments.

Search strategy:
  coordinate descent, coarse -> fine.

For each parameter:
  evaluate current - step
  evaluate current
  evaluate current + step
  move only if the objective improves.

Optimization simulations use 1–30 GHz. The winner is then verified at
1–100 GHz using the full-resolution V18 simulation.

This avoids the mistake of using an analytical TL model as a substitute for
the actual EM result.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "simulations" / "openems" / "model" / "chebyshev_v18_optimizer_runner_fixed.py"

LOG_DIR = PROJECT_ROOT / "results" / "em" / "optimizer_campaign"
LOG_DIR.mkdir(parents=True, exist_ok=True)

HISTORY = LOG_DIR / "history.jsonl"
BEST = LOG_DIR / "best_geometry.json"

# ============================================================================
# STARTING GEOMETRY
# ============================================================================

START = {
    "w_high": 0.50,
    "w_low": 12.00,
    "l1": 0.75,
    "l2": 1.00,
    "l3": 1.50,
    "l4": 1.50,
}

BOUNDS = {
    "w_high": (0.50, 1.50),
    "w_low": (8.00, 14.00),
    "l1": (0.50, 1.50),
    "l2": (0.50, 2.00),
    "l3": (1.00, 2.50),
    "l4": (1.00, 2.50),
}

# We deliberately begin with larger moves, then refine.
STAGES = [
    ("coarse", {
        "w_high": 0.50,
        "w_low": 1.00,
        "l1": 0.50,
        "l2": 0.50,
        "l3": 0.50,
        "l4": 0.50,
    }),
    ("fine", {
        "w_high": 0.50,
        "w_low": 0.50,
        "l1": 0.25,
        "l2": 0.25,
        "l3": 0.25,
        "l4": 0.25,
    }),
]


if not RUNNER.exists():
    raise FileNotFoundError(
        f"Optimizer runner not found: {RUNNER}\n"
        "Place the corrected runner beside this optimizer script."
    )

def objective(m: dict) -> float:
    """
    Lower is better.

    The strongest term is the unwanted 12–20 GHz transmission band seen in
    the measured V18 result. Passband and 20-GHz attenuation are also kept
    in the objective so the optimizer cannot simply destroy the passband.
    """

    pass_min = float(m["passband_min_1_8p5_dB"])
    s10 = float(m["s21_10GHz_dB"])
    peak12_20 = float(m["max_s21_12_20_dB"])
    s20 = float(m["s21_20GHz_dB"])
    peak20_30 = float(m["max_s21_20_30_dB"])

    # Target passband floor: -1 dB.
    p_pass = max(0.0, -1.0 - pass_min) ** 2

    # Encourage transition by 10 GHz.
    p_cutoff = max(0.0, s10 + 3.0) ** 2

    # Main target: suppress the observed secondary passband to <= -25 dB.
    p_resonance = max(0.0, peak12_20 + 25.0) ** 2

    # Desired stopband at 20 GHz.
    p_20 = max(0.0, s20 + 40.0) ** 2

    # Keep the 20–30 GHz region moving downward as well.
    p_20_30 = max(0.0, peak20_30 + 35.0) ** 2

    return (
        1.0 * p_pass
        + 1.5 * p_cutoff
        + 4.0 * p_resonance
        + 2.0 * p_20
        + 1.5 * p_20_30
    )


counter = 0


def normalize_candidate(c: dict) -> dict:
    out = dict(c)

    for name, (lo, hi) in BOUNDS.items():
        step = 0.50 if name in ("w_high", "w_low") else 0.25
        out[name] = round(out[name] / step) * step
        out[name] = max(lo, min(hi, out[name]))

    return out


def make_candidate(current: dict, name: str, delta: float) -> dict | None:
    c = dict(current)
    c[name] += delta
    c = normalize_candidate(c)

    if c["w_high"] >= c["w_low"]:
        return None

    # The filter is centered at x=0. Therefore the total filter length must
    # be an integer multiple of 0.50 mm so that ±L/2 remain on the 0.25-mm
    # Cartesian x-grid used by the EM runner.
    total_length = (
        2.0 * (c["l1"] + c["l2"] + c["l3"])
        + c["l4"]
        + 8.0
    )
    if abs(total_length / 0.50 - round(total_length / 0.50)) > 1e-9:
        return None

    return c


def run_candidate(c: dict, mode: str) -> dict:
    global counter
    counter += 1

    tag = (
        f"{mode}_{counter:04d}_"
        f"wh{c['w_high']:.2f}_wl{c['w_low']:.2f}_"
        f"l1{c['l1']:.2f}_l2{c['l2']:.2f}_"
        f"l3{c['l3']:.2f}_l4{c['l4']:.2f}"
    )

    payload = dict(c)
    payload["tag"] = tag

    env = os.environ.copy()
    env["V18_CANDIDATE_JSON"] = json.dumps(payload)
    env["V18_OPTIMIZER_MODE"] = "coarse"

    log_file = LOG_DIR / f"{tag}.log"

    print()
    print("=" * 100)
    print(f"RUN {counter}: {tag}")
    print("=" * 100)

    t0 = time.time()

    with log_file.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, str(RUNNER)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )

    elapsed = time.time() - t0

    if proc.returncode != 0:
        try:
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            log_text = "<could not read candidate log>"

        tail = "\n".join(log_text.splitlines()[-80:])

        print()
        print("=" * 100)
        print("ACTUAL CANDIDATE FAILURE LOG (LAST 80 LINES)")
        print("=" * 100)
        print(tail)
        print("=" * 100)

        raise RuntimeError(
            f"Candidate failed.\n"
            f"Geometry: {c}\n"
            f"Log: {log_file}"
        )

    # IMPORTANT:
    # Do not reconstruct the metrics path independently from the EM runner.
    # The runner is the authority because it prints the exact file it created.
    #
    # This avoids Windows/path/version mismatches between the controller and
    # the runner. The runner now prints:
    #     OPTIMIZER_METRICS=<absolute path>
    log_text = log_file.read_text(encoding="utf-8", errors="replace")

    metrics_matches = re.findall(
        r"^OPTIMIZER_METRICS=(.+)$",
        log_text,
        flags=re.MULTILINE,
    )

    if not metrics_matches:
        raise RuntimeError(
            "EM runner completed but did not report OPTIMIZER_METRICS.\n"
            f"Log: {log_file}"
        )

    metrics_file = Path(metrics_matches[-1].strip())

    if not metrics_file.is_file():
        raise RuntimeError(
            "EM runner reported a metrics file, but the controller cannot "
            "read it.\n"
            f"Reported path: {metrics_file}\n"
            f"Log: {log_file}"
        )

    print(f"CONTROLLER_USING_METRICS={metrics_file}")

    try:
        m = json.loads(metrics_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "metrics.json exists but could not be parsed as JSON.\n"
            f"Path: {metrics_file}\n"
            f"Error: {exc}"
        ) from exc
    m["objective"] = objective(m)
    m["elapsed_s"] = elapsed
    m["geometry"] = c

    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(m) + "\n")

    print(
        f"objective={m['objective']:.3f} | "
        f"S21@5={m['s21_5GHz_dB']:.2f} dB | "
        f"S21@10={m['s21_10GHz_dB']:.2f} dB | "
        f"S21@20={m['s21_20GHz_dB']:.2f} dB | "
        f"max12–20={m['max_s21_12_20_dB']:.2f} dB | "
        f"max20–30={m['max_s21_20_30_dB']:.2f} dB | "
        f"time={elapsed/60:.1f} min"
    )

    return m


def coordinate_pass(current: dict, step_map: dict) -> tuple[dict, dict]:
    # Evaluate the current geometry once for this pass.
    best = run_candidate(current, "base")

    for name, step in step_map.items():

        options = [("current", current)]

        minus = make_candidate(current, name, -step)
        plus = make_candidate(current, name, +step)

        if minus is not None and minus != current:
            options.append(("minus", minus))

        if plus is not None and plus != current:
            options.append(("plus", plus))

        local_best_c = current
        local_best_m = best

        for label, candidate in options[1:]:
            m = run_candidate(candidate, "trial")

            if m["objective"] < local_best_m["objective"]:
                local_best_c = candidate
                local_best_m = m

        if local_best_c != current:
            print(
                f"  MOVE {name}: "
                f"{current[name]:.2f} -> {local_best_c[name]:.2f}"
            )
            current = local_best_c
            best = local_best_m
        else:
            print(f"  KEEP {name}: {current[name]:.2f}")

    return current, best


def final_100ghz(current: dict) -> dict:
    global counter
    counter += 1

    tag = (
        f"FINAL_100GHz_{counter:04d}_"
        f"wh{current['w_high']:.2f}_wl{current['w_low']:.2f}_"
        f"l1{current['l1']:.2f}_l2{current['l2']:.2f}_"
        f"l3{current['l3']:.2f}_l4{current['l4']:.2f}"
    )

    payload = dict(current)
    payload["tag"] = tag

    env = os.environ.copy()
    env["V18_CANDIDATE_JSON"] = json.dumps(payload)
    env["V18_OPTIMIZER_MODE"] = "final"

    log_file = LOG_DIR / f"{tag}.log"

    print()
    print("=" * 100)
    print("FINAL 1–100 GHz FULL-WAVE VERIFICATION")
    print("=" * 100)

    with log_file.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, str(RUNNER)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if proc.returncode != 0:
        raise RuntimeError(f"Final simulation failed. Log: {log_file}")

    metrics_file = (
        PROJECT_ROOT / "results" / "em" / "optimizer"
        / tag / "metrics.json"
    )

    m = json.loads(metrics_file.read_text(encoding="utf-8"))
    m["geometry"] = current
    m["log"] = str(log_file)

    return m


def main() -> None:
    print("=" * 100)
    print("V18 FULL-WAVE COARSE-TO-FINE OPTIMIZATION")
    print("=" * 100)

    HISTORY.write_text("", encoding="utf-8")

    current = dict(START)
    best = None

    print("\nStarting geometry:")
    print(json.dumps(current, indent=2))

    for stage_name, step_map in STAGES:
        print()
        print("#" * 100)
        print(f"STAGE: {stage_name.upper()}")
        print("#" * 100)

        current, best = coordinate_pass(current, step_map)

        print("\nStage winner:")
        print(json.dumps(current, indent=2))
        print(f"Objective = {best['objective']:.6f}")

    BEST.write_text(
        json.dumps(
            {
                "geometry": current,
                "coarse_metrics": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    final = final_100ghz(current)

    final_file = LOG_DIR / "final_100GHz.json"
    final_file.write_text(
        json.dumps(final, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("OPTIMIZATION FINISHED")
    print("=" * 100)
    print("BEST GEOMETRY")
    print(json.dumps(current, indent=2))
    print()
    print("FINAL 1–100 GHz METRICS")
    print(json.dumps(final, indent=2))
    print()
    print(f"History       : {HISTORY}")
    print(f"Best geometry : {BEST}")
    print(f"Final result  : {final_file}")


if __name__ == "__main__":
    main()
