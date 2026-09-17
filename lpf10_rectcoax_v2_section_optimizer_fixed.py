
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# ============================================================
# 10-GHz RECTANGULAR-COAX LPF — V2 SECTION-LENGTH OPTIMIZER
#
# Based on the architecture in:
# Andersson et al., arXiv:2508.02475
#
# 21 physical stepped-impedance sections:
#   11 independent lengths + mirror symmetry
#   odd sections  = high impedance / inductive
#   even sections = low impedance / capacitive
#
# This stage optimizes the distributed LPF BEFORE adding the
# hollow-waveguide absorbers.
# ============================================================

C0 = 299_792_458.0

Z0 = 50.0
ZH = 85.9
ZL = 22.1

EPS_EFF = 2.10          # first-order PTFE-filled TEM estimate
FC_TARGET = 10e9

# Optimization frequency grid
F = np.linspace(0.05e9, 70e9, 3501)

# Weighting bands
PB = F <= 8.5e9
EDGE = (F > 8.5e9) & (F <= 14e9)
SB1 = (F >= 15e9) & (F <= 40e9)
SB2 = (F >= 40e9) & (F <= 70e9)

# Published 13.5-GHz prototype starting lengths, mm.
# 21-section physical structure is formed by mirroring the first
# 10 sections around the central 11th section.
PUBLISHED_HALF = np.array([
    1.06, 2.30, 2.05, 2.54, 1.91,
    3.16, 1.91, 4.28, 1.77, 4.63, 0.80
])

# Simple first-order frequency scaling from 13.5 -> 10 GHz.
START = PUBLISHED_HALF * (13.5 / 10.0)

# Reasonable fabrication/optimization bounds.
BOUNDS = [(0.35, 7.00)] * 11


def section_matrix(z, length_mm, freq_hz):
    """Lossless TEM transmission-line ABCD matrix."""
    beta = 2.0 * np.pi * freq_hz * np.sqrt(EPS_EFF) / C0
    theta = beta * length_mm * 1e-3

    ct = np.cos(theta)
    st = np.sin(theta)

    A = ct
    B = 1j * z * st
    C = 1j * st / z
    D = ct

    return A, B, C, D


def cascade(T, M):
    A, B, C, D = T
    a, b, c, d = M

    return (
        A*a + B*c,
        A*b + B*d,
        C*a + D*c,
        C*b + D*d,
    )


def s21_from_abcd(A, B, C, D):
    den = A + B/Z0 + C*Z0 + D
    s21 = 2.0 / den
    s11 = (A + B/Z0 - C*Z0 - D) / den
    return s11, s21


def full_lengths(x):
    # 21 sections:
    # 1..11 then mirror 10..1
    return np.r_[x, x[-2::-1]]


def full_impedances():
    # odd = high-Z, even = low-Z
    return np.array([
        ZH if (i % 2 == 0) else ZL
        for i in range(21)
    ])


ZSEC = full_impedances()


def response(x):
    lengths = full_lengths(x)

    s11 = np.empty(F.size, dtype=complex)
    s21 = np.empty(F.size, dtype=complex)

    for k, f in enumerate(F):
        T = (1+0j, 0j, 0j, 1+0j)

        for z, ell in zip(ZSEC, lengths):
            T = cascade(T, section_matrix(z, ell, f))

        s11[k], s21[k] = s21_from_abcd(*T)

    return s11, s21


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-14))


def objective(x):
    s11, s21 = response(x)

    s11_db = db(s11)
    s21_db = db(s21)

    # ---- Passband ----
    # Want S21 close to 0 dB and S11 below -20 dB.
    pb_loss = np.maximum(-s21_db[PB], 0.0)
    pb_ripple = np.ptp(s21_db[PB])
    pb_rl_penalty = np.maximum(s11_db[PB] + 20.0, 0.0)

    # ---- Transition ----
    # Encourage rapid attenuation after 8.5 GHz.
    edge_penalty = np.maximum(s21_db[EDGE] + 3.0, 0.0)

    # ---- Stopband ----
    # Distributed stepped-impedance stage target.
    sb1_penalty = np.maximum(s21_db[SB1] + 45.0, 0.0)
    sb2_penalty = np.maximum(s21_db[SB2] + 35.0, 0.0)

    # Penalize strong resonant windows particularly heavily.
    sb1_peak = np.max(s21_db[SB1])
    sb2_peak = np.max(s21_db[SB2])

    # Mild length regularization to avoid unnecessarily huge geometry.
    length_penalty = 0.010 * np.sum(x)

    cost = (
        18.0 * np.mean(pb_loss**2)
        + 10.0 * pb_ripple**2
        + 5.0 * np.mean(pb_rl_penalty**2)
        + 8.0 * np.mean(edge_penalty**2)
        + 4.0 * np.mean(sb1_penalty**2)
        + 2.0 * np.mean(sb2_penalty**2)
        + 8.0 * max(sb1_peak + 45.0, 0.0)**2
        + 4.0 * max(sb2_peak + 35.0, 0.0)**2
        + length_penalty
    )

    return float(cost)


def metrics(x):
    s11, s21 = response(x)
    s11_db = db(s11)
    s21_db = db(s21)

    cutoff_idx = np.where(s21_db <= -3.0)[0]
    cutoff = F[cutoff_idx[0]] / 1e9 if cutoff_idx.size else np.nan

    return {
        "cutoff_3dB_GHz": cutoff,
        "PB_max_loss_dB": float(np.max(-s21_db[PB])),
        "PB_ripple_dB": float(np.ptp(s21_db[PB])),
        "PB_worst_S11_dB": float(np.max(s11_db[PB])),
        "S21_10GHz_dB": float(np.interp(10e9, F, s21_db)),
        "S21_15GHz_dB": float(np.interp(15e9, F, s21_db)),
        "S21_20GHz_dB": float(np.interp(20e9, F, s21_db)),
        "S21_30GHz_dB": float(np.interp(30e9, F, s21_db)),
        "S21_40GHz_dB": float(np.interp(40e9, F, s21_db)),
        "S21_70GHz_dB": float(np.interp(70e9, F, s21_db)),
        "SB15_40_max_dB": float(np.max(s21_db[SB1])),
        "SB40_70_max_dB": float(np.max(s21_db[SB2])),
    }


def plot_response(x, filename):
    s11, s21 = response(x)

    plt.figure(figsize=(12, 7))
    plt.plot(F/1e9, db(s21), label="S21")
    plt.plot(F/1e9, db(s11), label="S11")

    plt.axvline(10, linestyle="--", label="10 GHz target")
    plt.axhline(-3, linestyle=":", label="-3 dB")
    plt.axhline(-60, linestyle=":", label="-60 dB")

    plt.xlim(0, 70)
    plt.ylim(-100, 2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("10 GHz rectangular-coax stepped-impedance LPF — optimized section model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.show()


# ============================================================
# INITIAL MODEL
# ============================================================

print("=" * 90)
print("10 GHz RECTANGULAR-COAX LPF — V2 SECTION OPTIMIZATION")
print("=" * 90)

print("\nStarting independent lengths (mm):")
print(np.array2string(START, precision=4))

print("\nStarting full 21-section lengths (mm):")
print(np.array2string(full_lengths(START), precision=4))

print("\nEvaluating starting model...")
print(metrics(START))

plot_response(START, "lpf10_v2_starting_stepped_model.png")


# ============================================================
# DIFFERENTIAL EVOLUTION
# ============================================================

print("\nOptimizer bounds [mm]:")
for i, (lo, hi) in enumerate(BOUNDS, 1):
    print(f"  Section {i:2d}: {lo:.2f} .. {hi:.2f}")

print("\nChecking starting point against bounds...")
for i, (x0, (lo, hi)) in enumerate(zip(START, BOUNDS), 1):
    if not (lo <= x0 <= hi):
        raise ValueError(
            f"START section {i} = {x0:.4f} mm is outside "
            f"bounds {lo:.4f} .. {hi:.4f} mm"
        )
print("Starting point is feasible.")

print("\nRunning differential evolution...")
print("This can take several minutes depending on CPU.")

result = differential_evolution(
    objective,
    BOUNDS,
    x0=START,
    strategy="best1bin",
    maxiter=120,
    popsize=10,
    tol=2e-3,
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=True,
    workers=1,
    updating="immediate",
    seed=10,
    disp=True,
)

OPT = result.x

print("\n" + "=" * 90)
print("OPTIMIZATION COMPLETE")
print("=" * 90)

print("Success:", result.success)
print("Message:", result.message)
print("Objective:", result.fun)

print("\nOptimized independent lengths (mm):")
for i, value in enumerate(OPT, 1):
    print(f"Section {i:2d}: {value:8.4f} mm")

print("\nFull 21-section lengths (mm):")
for i, value in enumerate(full_lengths(OPT), 1):
    print(f"Section {i:2d}: {value:8.4f} mm")

print("\nMetrics:")
for key, value in metrics(OPT).items():
    print(f"{key:24s}: {value:10.4f}")

RESULT_FILE = Path(__file__).resolve().parent / "lpf10_v2_optimized_lengths_mm.txt"
np.savetxt(
    RESULT_FILE,
    OPT,
    header="Optimized independent section lengths, mm"
)

plot_response(OPT, "lpf10_v2_optimized_stepped_model.png")

print("\nSaved:")
print("  lpf10_v2_starting_stepped_model.png")
print("  lpf10_v2_optimized_stepped_model.png")
print(f"  {RESULT_FILE}")
