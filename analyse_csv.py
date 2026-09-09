from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# FINAL 100 GHz CSV
# ============================================================

CSV_PATH = Path(
    r"C:\Users\Raghava.m\Desktop\cryogenic_filter_design\results\em\optimizer"
    r"\FINAL_100GHz_0021_wh0_50_wl12_50_l11_00_l20_75_l31_25_l41_00"
    r"\chebyshev_v18.csv"
)

# ============================================================
# LOAD
# ============================================================

print("=" * 100)
print("FINAL 100 GHz RESPONSE ANALYSIS")
print("=" * 100)

print(f"\nLoading:")
print(CSV_PATH)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"\nCSV not found:\n{CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("\nCSV columns:")
for col in df.columns:
    print(f"  {col}")

print(f"\nNumber of frequency points: {len(df)}")

# ============================================================
# FIND FREQUENCY / S21 COLUMNS
# ============================================================

def find_column(columns, candidates):
    for candidate in candidates:
        for col in columns:
            if col.lower() == candidate.lower():
                return col
    return None


freq_col = find_column(
    df.columns,
    [
        "frequency_Hz",
        "freq_Hz",
        "frequency",
        "freq",
        "Frequency",
    ],
)

s21_col = find_column(
    df.columns,
    [
        "S21_dB",
        "s21_dB",
        "S21",
        "s21",
    ],
)

if freq_col is None:
    raise RuntimeError(
        f"Could not identify frequency column.\n"
        f"Available columns: {list(df.columns)}"
    )

if s21_col is None:
    raise RuntimeError(
        f"Could not identify S21 column.\n"
        f"Available columns: {list(df.columns)}"
    )

print(f"\nFrequency column : {freq_col}")
print(f"S21 column       : {s21_col}")

freq = df[freq_col].to_numpy(dtype=float)
s21 = df[s21_col].to_numpy(dtype=float)

# Automatically convert frequency to GHz if necessary
if np.max(freq) > 1e6:
    freq_GHz = freq / 1e9
elif np.max(freq) > 1e3:
    freq_GHz = freq / 1e3
else:
    freq_GHz = freq

# ============================================================
# GLOBAL MAXIMA
# ============================================================

print("\n" + "=" * 100)
print("STRONGEST TRANSMISSION WINDOWS")
print("=" * 100)

# Sort by S21 descending
idx_sorted = np.argsort(s21)[::-1]

print("\nTop 20 highest-S21 frequency points:")
print("-" * 60)

for rank, idx in enumerate(idx_sorted[:20], start=1):
    print(
        f"{rank:2d}. "
        f"{freq_GHz[idx]:9.4f} GHz   "
        f"S21 = {s21[idx]:9.4f} dB"
    )

# ============================================================
# REGION ANALYSIS
# ============================================================

def analyze_region(f_low, f_high, name):
    mask = (freq_GHz >= f_low) & (freq_GHz <= f_high)

    if not np.any(mask):
        print(f"\n{name}: no data")
        return

    indices = np.where(mask)[0]
    local_idx = indices[np.argmax(s21[mask])]

    print(f"\n{name}")
    print("-" * 60)
    print(f"Frequency range : {f_low:.1f} – {f_high:.1f} GHz")
    print(f"Maximum S21     : {s21[local_idx]:.4f} dB")
    print(f"At frequency    : {freq_GHz[local_idx]:.4f} GHz")

    # Minimum S21 in region
    local_min_idx = indices[np.argmin(s21[mask])]

    print(f"Minimum S21     : {s21[local_min_idx]:.4f} dB")
    print(f"At frequency    : {freq_GHz[local_min_idx]:.4f} GHz")


analyze_region(1, 8.5, "PASSBAND: 1–8.5 GHz")
analyze_region(9, 12, "TRANSITION: 9–12 GHz")
analyze_region(12, 20, "CRITICAL RESONANCE REGION: 12–20 GHz")
analyze_region(20, 30, "STOPBAND: 20–30 GHz")
analyze_region(30, 50, "HIGH FREQUENCY: 30–50 GHz")
analyze_region(50, 75, "HIGH FREQUENCY: 50–75 GHz")
analyze_region(75, 100, "HIGH FREQUENCY: 75–100 GHz")

# ============================================================
# LOCAL PEAK DETECTION
# ============================================================

print("\n" + "=" * 100)
print("LOCAL TRANSMISSION PEAKS")
print("=" * 100)

# A point is a local maximum if S21[i] > neighbors
local_maxima = []

for i in range(1, len(s21) - 1):
    if s21[i] > s21[i - 1] and s21[i] >= s21[i + 1]:
        local_maxima.append(i)

# Sort local peaks by S21
local_maxima = sorted(
    local_maxima,
    key=lambda i: s21[i],
    reverse=True
)

print("\nStrongest local peaks:")
print("-" * 60)

count = 0

for idx in local_maxima:
    # Only show meaningful transmission peaks
    if s21[idx] > -40:
        print(
            f"{freq_GHz[idx]:9.4f} GHz   "
            f"S21 = {s21[idx]:9.4f} dB"
        )
        count += 1

        if count >= 30:
            break

# ============================================================
# SAVE CLEAN ANALYSIS
# ============================================================

output_path = CSV_PATH.parent / "resonance_analysis.txt"

with output_path.open("w", encoding="utf-8") as f:

    f.write("FINAL 100 GHz RESPONSE ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"CSV: {CSV_PATH}\n")
    f.write(f"Frequency points: {len(df)}\n\n")

    f.write("TOP TRANSMISSION POINTS\n")
    f.write("-" * 80 + "\n")

    for rank, idx in enumerate(idx_sorted[:50], start=1):
        f.write(
            f"{rank:2d}. "
            f"{freq_GHz[idx]:10.5f} GHz   "
            f"S21 = {s21[idx]:10.5f} dB\n"
        )

    f.write("\n\nLOCAL PEAKS\n")
    f.write("-" * 80 + "\n")

    for idx in local_maxima:
        if s21[idx] > -40:
            f.write(
                f"{freq_GHz[idx]:10.5f} GHz   "
                f"S21 = {s21[idx]:10.5f} dB\n"
            )

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

print(f"\nAnalysis saved to:")
print(output_path)