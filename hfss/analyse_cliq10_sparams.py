from pathlib import Path
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

# Reads the exported HFSS Touchstone file without requiring scikit-rf.
ROOT = Path(__file__).resolve().parent
S2P = ROOT / "cliq10_iter01" / "CliQ10_Iter01.s2p"
OUT_CSV = S2P.with_name("CliQ10_Iter01_Sparams.csv")
OUT_PNG = S2P.with_name("CliQ10_Iter01_Sparams.png")

def parse_touchstone(path):
    unit = "ghz"
    fmt = "ma"
    z0 = 50.0
    numeric_values = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("!") or line.startswith(";"):
                continue
            if line.startswith("#"):
                tokens = line[1:].lower().split()
                for token in tokens:
                    if token in ("hz", "khz", "mhz", "ghz"):
                        unit = token
                    elif token in ("ri", "ma", "db"):
                        fmt = token
                if "r" in tokens:
                    try:
                        z0 = float(tokens[tokens.index("r") + 1])
                    except (ValueError, IndexError):
                        pass
                continue
            line = line.split("!")[0].strip()
            if not line:
                continue
            try:
                numeric_values.extend(float(x) for x in line.split())
            except ValueError:
                continue

    # HFSS exports this file as a 4-port modal network: 1 frequency plus
    # 16 complex S-parameters (32 values) per frequency = 33 values.
    nports = 4
    record_size = 1 + 2 * nports * nports
    if len(numeric_values) % record_size != 0:
        raise ValueError(
            f"Unexpected Touchstone numeric count ({len(numeric_values)} values; "
            f"not divisible by {record_size} for a {nports}-port file). "
            "Inspect the header and file format before changing the parser."
        )

    scale = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0}[unit]
    data = np.asarray(numeric_values, dtype=float).reshape(-1, record_size)
    freq_ghz = data[:, 0] * scale
    raw_s = data[:, 1:].reshape(-1, nports * nports, 2)

    def convert(pair):
        a, b = pair[:, 0], pair[:, 1]
        if fmt == "ri":
            return a + 1j * b
        if fmt == "ma":
            return a * np.exp(1j * np.deg2rad(b))
        if fmt == "db":
            return 10 ** (a / 20.0) * np.exp(1j * np.deg2rad(b))
        raise ValueError(f"Unsupported Touchstone format: {fmt}")

    s_flat = convert(raw_s.reshape(-1, 2)).reshape(-1, nports * nports)
    # Touchstone 1.x order is column-major: S11, S21, ... S(N,1), S12, ...
    smat = np.empty((len(freq_ghz), nports, nports), dtype=complex)
    for k in range(nports * nports):
        row = k % nports
        col = k // nports
        smat[:, row, col] = s_flat[:, k]

    # Modal order from the HFSS header:
    # index 0 = Port1:1, 1 = Port1:2, 2 = Port2:1, 3 = Port2:2.
    # Mode-1 input at Port1 is column 0; corresponding output mode-1 at Port2 is row 2.
    s11 = smat[:, 0, 0]
    s21 = smat[:, 2, 0]
    s12 = smat[:, 0, 2]
    s22 = smat[:, 2, 2]
    return freq_ghz, s11, s21, s12, s22, z0, fmt

def db20(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-15))

def band_value(freq, values_db, target_ghz):
    idx = int(np.argmin(np.abs(freq - target_ghz)))
    return float(freq[idx]), float(values_db[idx])

def main():
    if not S2P.exists():
        raise FileNotFoundError(
            f"Could not find {S2P}\n"
            "Save this script in the hfss folder next to the cliq10_iter01 folder."
        )

    f, s11, s21, s12, s22, z0, fmt = parse_touchstone(S2P)
    s11_db, s21_db, s12_db, s22_db = map(db20, (s11, s21, s12, s22))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["Frequency_GHz", "S11_dB", "S21_dB", "S12_dB", "S22_dB",
                    "S11_mag", "S21_mag", "S12_mag", "S22_mag"])
        for i in range(len(f)):
            w.writerow([f"{f[i]:.9g}", f"{s11_db[i]:.8g}", f"{s21_db[i]:.8g}",
                        f"{s12_db[i]:.8g}", f"{s22_db[i]:.8g}",
                        f"{abs(s11[i]):.8g}", f"{abs(s21[i]):.8g}",
                        f"{abs(s12[i]):.8g}", f"{abs(s22[i]):.8g}"])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(f, s11_db, label="S11 (return)")
    ax.plot(f, s21_db, label="S21 (transmission)")
    ax.plot(f, s12_db, label="S12 (reverse transmission)", alpha=0.75)
    ax.plot(f, s22_db, label="S22 (output return)", alpha=0.75)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("CliQ10 Iteration 01 — HFSS S-parameters")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_xlim(float(np.min(f)), float(np.max(f)))
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)

    print(f"Touchstone: {S2P}")
    print(f"Rows: {len(f)} | Modal ports: 4 | Using Port1:1 ↔ Port2:1 | Frequency range: {f.min():.6g} to {f.max():.6g} GHz")
    print(f"Reference impedance: {z0:g} ohm | Data format: {fmt.upper()}")
    print(f"CSV: {OUT_CSV}")
    print(f"Plot: {OUT_PNG}")
    print("\nSampled S-parameters (nearest exported point):")
    for ghz in (7.5, 9.3, 9.8, 11.0, 40.0):
        if f.min() <= ghz <= f.max():
            actual, v21 = band_value(f, s21_db, ghz)
            _, v11 = band_value(f, s11_db, ghz)
            print(f"  {actual:8.4f} GHz: S21={v21:8.3f} dB, S11={v11:8.3f} dB")

    print("\nPreliminary target checks (sampled points only; not a full band-extrema check):")
    bands = [
        ("Insertion loss <= 0.5 dB through 7.5 GHz", 0.01, 7.5, s21_db, -0.5),
        ("Insertion loss <= 1.5 dB from 7.5 to 9.3 GHz", 7.5, 9.3, s21_db, -1.5),
        ("Return loss > 15 dB through 9.3 GHz", 0.01, 9.3, s11_db, -15.0),
        ("Rejection >= 30 dB from 11 GHz onward", 11.0, float(f.max()), s21_db, -30.0),
    ]
    for label, lo, hi, curve, threshold in bands:
        mask = (f >= lo) & (f <= hi)
        if not np.any(mask):
            print(f"  NOT CHECKED: {label} (no samples in band)")
            continue
        segment = curve[mask]
        if "Return loss" in label:
            # S11 magnitude must be below -15 dB.
            ok = np.all(segment < threshold)
            worst = float(np.max(segment))
            print(f"  {'PASS' if ok else 'FAIL'}: {label}; worst S11={worst:.3f} dB")
        else:
            # S21 must be above threshold in passband, below threshold in stopband.
            if "Rejection" in label:
                ok = np.all(segment <= threshold)
                worst = float(np.max(segment))
                print(f"  {'PASS' if ok else 'FAIL'}: {label}; highest S21 in band={worst:.3f} dB")
            else:
                ok = np.all(segment >= threshold)
                worst = float(np.min(segment))
                print(f"  {'PASS' if ok else 'FAIL'}: {label}; lowest S21 in band={worst:.3f} dB")

    print("\nNote: the sweep only reaches 40 GHz, so the >=55 dB rejection target at 55 GHz cannot be checked.")
    print("Also inspect the 7.5–9.3 GHz transition band separately; this script checks the stated envelope.")

if __name__ == "__main__":
    main()
