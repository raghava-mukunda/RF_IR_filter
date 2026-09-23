from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


RUN_DIR = Path(
    r"C:\Users\Raghava.m\Desktop\cryogenic_filter_design"
    r"\FINAL_OPENEMS_RIGOROUS\rf_characterization"
)

PORT1 = RUN_DIR / "port_ut_1"
PORT2 = RUN_DIR / "port_ut_2"


# ============================================================
# PORT OUTPUT
# ============================================================

def load_port(path):

    text = path.read_text(errors="replace")

    # Remove comment/header lines
    text = "\n".join(
        line for line in text.splitlines()
        if not line.startswith("%")
    )

    # Remove column header
    text = text.replace("t/s", "")
    text = text.replace("voltag0", "")

    return text


# ============================================================
# PARSER
#
# The openEMS file has concatenated fields such as:
#
#   -0.0002143833808081.31476236814e-12
#
# which means:
#
#   voltage = -0.000214383380808
#   time    =  1.31476236814e-12
#
# The timestamps are uniquely recognizable because they have
# the expected e-12 exponent.
# ============================================================

def parse_port(path):

    text = load_port(path)

    # Match timestamps specifically.
    #
    # The port sampling interval is ~1.314762e-12 s, so all
    # timestamps in this file are e-12.
    timestamp_pattern = re.compile(
        r"\d+\.\d+e-12"
    )

    matches = list(
        timestamp_pattern.finditer(text)
    )

    print(
        path.name,
        "timestamp-like fields:",
        len(matches)
    )

    if len(matches) < 50:
        raise RuntimeError(
            f"Could not identify enough timestamps in {path}"
        )

    times = []
    voltages = []

    # The voltage immediately before each timestamp is the
    # port voltage for that timestamp.
    #
    # Use the region between the previous timestamp and the
    # current timestamp.
    previous_end = 0

    for m in matches:

        ts_string = m.group(0)

        try:
            t = float(ts_string)
        except ValueError:
            continue

        # Text between previous timestamp and this timestamp.
        segment = text[previous_end:m.start()]

        # Extract numeric values from the segment.
        numbers = re.findall(
            r"[-+]?(?:"
            r"\d+\.\d*|"
            r"\.\d+|"
            r"\d+"
            r")(?:[eE][-+]?\d+)?",
            segment
        )

        if not numbers:
            previous_end = m.end()
            continue

        # Usually the final number is the voltage.
        try:
            v = float(numbers[-1])
        except ValueError:
            previous_end = m.end()
            continue

        times.append(t)
        voltages.append(v)

        previous_end = m.end()

    times = np.asarray(times)
    voltages = np.asarray(voltages)

    print(
        path.name,
        "recovered:",
        len(times),
        "samples"
    )

    if len(times) < 50:
        raise RuntimeError(
            f"Too few samples recovered from {path}"
        )

    return times, voltages


# ============================================================
# READ BOTH PORTS
# ============================================================

print("Reading existing FDTD port files...")

t1, v1 = parse_port(PORT1)
t2, v2 = parse_port(PORT2)


# ============================================================
# CHECK TIME AXIS
# ============================================================

print()
print("PORT 1")
print("first:", t1[:5])
print("last :", t1[-5:])

print()
print("PORT 2")
print("first:", t2[:5])
print("last :", t2[-5:])


# ============================================================
# COMMON TIME AXIS
# ============================================================

t_start = max(
    t1[0],
    t2[0]
)

t_end = min(
    t1[-1],
    t2[-1]
)

dt = np.median(
    np.diff(t1)
)

print()
print("Recovered port dt =", dt)
print("Recovered port rate =", 1 / dt / 1e9, "GHz")


t = np.arange(
    t_start,
    t_end + dt / 2,
    dt
)

v1i = np.interp(
    t,
    t1,
    v1
)

v2i = np.interp(
    t,
    t2,
    v2
)


# ============================================================
# SAVE RECOVERED TIME DOMAIN
# ============================================================

np.savetxt(
    RUN_DIR / "recovered_port_data.csv",
    np.column_stack(
        (t, v1i, v2i)
    ),
    delimiter=",",
    header="time_s,port1_voltage_V,port2_voltage_V",
    comments=""
)


# ============================================================
# FFT
# ============================================================

window = np.hanning(len(t))

V1 = np.fft.rfft(
    v1i * window
)

V2 = np.fft.rfft(
    v2i * window
)

freq = np.fft.rfftfreq(
    len(t),
    dt
)


# ============================================================
# V2/V1
# ============================================================

valid = (
    (freq >= 1e9)
    &
    (freq <= 70e9)
    &
    (
        np.abs(V1)
        >
        1e-12 * np.max(np.abs(V1))
    )
)

f = freq[valid]

H = V2[valid] / V1[valid]


# ============================================================
# PRINT AVAILABLE FREQUENCY RESOLUTION
# ============================================================

df = 1 / (len(t) * dt)

print()
print(
    "FFT frequency resolution:",
    df / 1e9,
    "GHz"
)


# ============================================================
# SAVE RAW FFT TRANSMISSION
# ============================================================

H_dB = 20 * np.log10(
    np.maximum(
        np.abs(H),
        1e-15
    )
)

np.savetxt(
    RUN_DIR / "recovered_transmission_raw.csv",
    np.column_stack(
        (
            f / 1e9,
            np.abs(H),
            H_dB
        )
    ),
    delimiter=",",
    header="Frequency_GHz,abs_V2_over_V1,V2_over_V1_dB",
    comments=""
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    f / 1e9,
    H_dB,
    label="|V2/V1|"
)

plt.axvline(
    10,
    linestyle="--",
    label="10 GHz target"
)

plt.axhline(
    -60,
    linestyle=":",
    label="-60 dB"
)

plt.xlim(1, 70)
plt.ylim(-100, 5)

plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")

plt.title(
    "Existing FDTD run — recovered voltage transmission"
)

plt.grid(
    True,
    alpha=0.25
)

plt.legend()
plt.tight_layout()

plt.savefig(
    RUN_DIR / "recovered_transmission.png",
    dpi=200
)

plt.show()


print()
print("=" * 60)
print("DONE")
print("=" * 60)
print(
    "Recovered data:",
    RUN_DIR / "recovered_port_data.csv"
)
print(
    "Spectrum:",
    RUN_DIR / "recovered_transmission_raw.csv"
)
print(
    "Plot:",
    RUN_DIR / "recovered_transmission.png"
)