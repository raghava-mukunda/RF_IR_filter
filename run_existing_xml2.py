import os
import shutil
import time
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

XML_FILE = Path(
    r"C:\Users\Raghava.m\Desktop\cryogenic_filter_design"
    r"\FINAL_OPENEMS_RIGOROUS\final_filter.xml"
)

RUN_DIR = XML_FILE.parent / "rf_final_300k"

# Preserve the previous diagnostic run.
# Only clean this NEW final-run directory.
RUN_DIR.mkdir(parents=True, exist_ok=True)

for p in RUN_DIR.iterdir():
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


# ============================================================
# openEMS ENVIRONMENT
# ============================================================

OPENEMS_PATH = (
    Path.home()
    / "Desktop"
    / "openEMS_x64_v0.0.36-93-g7b9cd51_msvc"
    / "openEMS"
)

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_PATH)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_PATH)
os.environ["PATH"] = (
    str(OPENEMS_PATH)
    + os.pathsep
    + os.environ["PATH"]
)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(OPENEMS_PATH))


# ============================================================
# IMPORTS
# ============================================================

from CSXCAD import ContinuousStructure
from openEMS import openEMS


# ============================================================
# FILTER / PORT PARAMETERS
# ============================================================

Z0 = 50.0

TOTAL = 46.764506724
FEED = 8.0

HO_H = 1.45
WALL = 0.20
CENTER_T = 0.50

P1X = -FEED + 1.0
P2X = TOTAL - 1.0

ZG = -HO_H / 2.0 + WALL
ZS = CENTER_T / 2.0


# ============================================================
# FREQUENCY RANGE
# ============================================================

FREQ = np.linspace(
    1e9,
    70e9,
    691
)

F0 = 35e9
FC = 60e9


# ============================================================
# FDTD PARAMETERS
# ============================================================

NR_TS = 300000

# Use a stringent end criterion, but the 300k limit is the
# hard upper limit. This ensures we have enough time for the
# wave to reach Port 2 and for the transient to decay.
END_CRITERIA = 1e-8


# ============================================================
# LOAD EXISTING FINAL XML
# ============================================================

print()
print("=" * 72)
print("LOADING FROZEN FILTER GEOMETRY")
print("=" * 72)

print("XML:")
print(XML_FILE)

if not XML_FILE.exists():
    raise FileNotFoundError(
        f"Could not find final XML:\n{XML_FILE}"
    )

CSX = ContinuousStructure()

CSX.ReadFromXML(
    str(XML_FILE)
)


# ============================================================
# CREATE FDTD
# ============================================================

FDTD = openEMS(
    EndCriteria=END_CRITERIA,
    NrTS=NR_TS
)

FDTD.SetGaussExcite(
    F0,
    FC
)

FDTD.SetBoundaryCond(
    ["PML_8"] * 6
)

FDTD.SetCSX(
    CSX
)


# ============================================================
# PORT 1
# ============================================================

p1 = FDTD.AddLumpedPort(
    1,
    Z0,
    [P1X, 0, ZG],
    [P1X, 0, ZS],
    "z",
    excite=1,
    priority=50
)


# ============================================================
# PORT 2
# ============================================================

p2 = FDTD.AddLumpedPort(
    2,
    Z0,
    [P2X, 0, ZG],
    [P2X, 0, ZS],
    "z",
    excite=0,
    priority=50
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print()
print("=" * 72)
print("PORT CONFIGURATION")
print("=" * 72)

print(
    f"P1: x={P1X:.9f} mm"
    f"   z={ZG:.9f} -> {ZS:.9f} mm"
)

print(
    f"P2: x={P2X:.9f} mm"
    f"   z={ZG:.9f} -> {ZS:.9f} mm"
)

print(
    f"Port separation: {P2X - P1X:.9f} mm"
)

print("=" * 72)


# ============================================================
# WRITE SIMULATION XML
# ============================================================

SIM_XML = RUN_DIR / "rf_final.xml"

CSX.Write2XML(
    str(SIM_XML)
)

print()
print("Simulation XML:")
print(SIM_XML)


# ============================================================
# RUN FDTD
# ============================================================

print()
print("=" * 72)
print("STARTING FINAL FDTD")
print("=" * 72)

print()
print("Frequency range : 1 - 70 GHz")
print("Excitation       : Gaussian")
print("F0               : 35 GHz")
print("FC               : 60 GHz")
print("Maximum timesteps:", NR_TS)
print("End criterion    :", END_CRITERIA)
print()

print("This is the full electromagnetic simulation.")
print("Do NOT interrupt unless absolutely necessary.")
print()

t0 = time.time()

FDTD.Run(
    str(RUN_DIR),
    cleanup=False,
    verbose=2
)

elapsed = time.time() - t0

print()
print("=" * 72)
print("FDTD COMPLETE")
print("=" * 72)

print(
    f"Runtime: {elapsed / 60.0:.2f} minutes"
)

print()


# ============================================================
# PORT FILES
# ============================================================

UT1 = RUN_DIR / "port_ut_1"
UT2 = RUN_DIR / "port_ut_2"

IT1 = RUN_DIR / "port_it_1"
IT2 = RUN_DIR / "port_it_2"

required_files = [
    UT1,
    UT2,
    IT1,
    IT2
]

print("=" * 72)
print("CHECKING PORT OUTPUT")
print("=" * 72)

for p in required_files:

    print(
        f"{p.name:15s}",
        "OK" if p.exists() else "MISSING",
        f"{p.stat().st_size if p.exists() else 0} bytes"
    )

for p in required_files:

    if not p.exists():
        raise RuntimeError(
            f"Required port file missing: {p}"
        )


# ============================================================
# PORT FILE PARSER
#
# openEMS is writing values using insufficient field width.
#
# Example:
#
#   -1.31.31476236814e-12
#
# actually means:
#
#   voltage = -1.3
#   time    = 1.31476236814e-12
#
# We therefore identify the timestamp from its known
# deterministic timestep sequence and extract the voltage/
# current immediately preceding it.
# ============================================================


SCI_NUMBER = re.compile(
    r"[-+]?"
    r"(?:"
    r"\d+\.\d*"
    r"|"
    r"\.\d+"
    r"|"
    r"\d+"
    r")"
    r"[eE]"
    r"[-+]?\d+"
)


PLAIN_NUMBER = re.compile(
    r"[-+]?"
    r"(?:"
    r"\d+\.\d*"
    r"|"
    r"\.\d+"
    r"|"
    r"\d+"
    r")"
)


def read_port_file(path):
    """
    Recover the time-domain signal from an openEMS port file.

    Returns
    -------
    t : numpy array
        Time in seconds.

    y : numpy array
        Voltage or current.
    """

    print()
    print("Parsing:", path.name)

    text = path.read_text(
        errors="replace"
    )

    # --------------------------------------------------------
    # Determine the FDTD timestep from the first timestamp
    # --------------------------------------------------------

    candidates = []

    for match in SCI_NUMBER.finditer(text):

        try:
            value = float(match.group())
        except ValueError:
            continue

        if value > 0:
            candidates.append(
                (match.start(), match.end(), value)
            )

    if len(candidates) < 2:

        raise RuntimeError(
            f"Could not find enough numerical fields in {path}"
        )

    # The first positive scientific values include voltage
    # and timestamp. We know the openEMS timestep from the
    # simulation output:
    #
    # dt = 1.35264e-15 s
    #
    # The port writer records every 972 FDTD steps.
    #
    # Therefore:
    #
    # dt_port = 972 * dt
    #
    # = 1.31476236814e-12 s
    #

    FDTD_DT = 1.35264e-15
    PORT_STEP = 972

    PORT_DT = FDTD_DT * PORT_STEP

    # --------------------------------------------------------
    # Identify timestamp fields
    # --------------------------------------------------------

    timestamps = []

    for start, end, value in candidates:

        n = int(
            round(value / PORT_DT)
        )

        if n <= 0:
            continue

        expected = n * PORT_DT

        error = abs(
            value - expected
        )

        # Very tight tolerance.
        # Printed timestamp precision is much better than this.
        if error <= 1e-15:

            timestamps.append(
                (start, end, value, n)
            )

    # --------------------------------------------------------
    # Remove duplicate / impossible timestamp matches
    # --------------------------------------------------------

    clean = []

    last_n = -1

    for item in timestamps:

        start, end, value, n = item

        if n <= last_n:
            continue

        clean.append(item)

        last_n = n

    timestamps = clean

    print(
        "Timestamp candidates recovered:",
        len(timestamps)
    )

    if len(timestamps) < 5:

        raise RuntimeError(
            f"Could not recover timestamps from {path}"
        )

    # --------------------------------------------------------
    # Recover signal value immediately before each timestamp
    # --------------------------------------------------------

    times = []
    values = []

    previous_timestamp_end = 0

    for index, (
        ts_start,
        ts_end,
        ts_value,
        n
    ) in enumerate(timestamps):

        # Ignore everything before the timestamp except the
        # signal token immediately preceding it.
        segment = text[
            previous_timestamp_end:ts_start
        ]

        # Find the last whitespace.
        #
        # Because the openEMS fields may be concatenated,
        # the signal itself is the final token immediately
        # before the timestamp.
        whitespace_positions = [
            pos
            for pos, char in enumerate(segment)
            if char.isspace()
        ]

        if whitespace_positions:

            token = segment[
                whitespace_positions[-1] + 1:
            ]

        else:

            token = segment

        token = token.strip()

        # ----------------------------------------------------
        # Remove anything that obviously belongs to headers
        # ----------------------------------------------------

        if not token:

            previous_timestamp_end = ts_end
            continue

        # Try direct conversion first.
        try:

            value = float(token)

        except ValueError:

            # If the token contains additional junk,
            # recover the final numeric component.
            numbers = list(
                PLAIN_NUMBER.finditer(token)
            )

            if not numbers:

                previous_timestamp_end = ts_end
                continue

            value = float(
                numbers[-1].group()
            )

        times.append(
            ts_value
        )

        values.append(
            value
        )

        previous_timestamp_end = ts_end

    times = np.asarray(
        times,
        dtype=float
    )

    values = np.asarray(
        values,
        dtype=float
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if len(times) < 5:

        raise RuntimeError(
            f"Only {len(times)} samples recovered from {path}"
        )

    order = np.argsort(times)

    times = times[order]
    values = values[order]

    # Remove duplicate timestamps
    unique_t, unique_idx = np.unique(
        times,
        return_index=True
    )

    values = values[
        unique_idx
    ]

    times = unique_t

    print(
        "Samples recovered:",
        len(times)
    )

    print(
        "Time range:",
        f"{times[0]:.6e} -> {times[-1]:.6e} s"
    )

    print(
        "Signal range:",
        f"{np.min(values):.6e}"
        f" -> "
        f"{np.max(values):.6e}"
    )

    return times, values


# ============================================================
# PARSE ALL FOUR PORT FILES
# ============================================================

print()
print("=" * 72)
print("RECONSTRUCTING TIME-DOMAIN PORT SIGNALS")
print("=" * 72)

t1u, u1 = read_port_file(
    UT1
)

t2u, u2 = read_port_file(
    UT2
)

t1i, i1 = read_port_file(
    IT1
)

t2i, i2 = read_port_file(
    IT2
)


# ============================================================
# INTERPOLATE ALL SIGNALS TO COMMON TIME GRID
# ============================================================

t_start = max(
    t1u[0],
    t2u[0],
    t1i[0],
    t2i[0]
)

t_stop = min(
    t1u[-1],
    t2u[-1],
    t1i[-1],
    t2i[-1]
)

if t_stop <= t_start:

    raise RuntimeError(
        "Port time ranges do not overlap."
    )


# Use the densest available time grid.
dt_candidates = []

for t in [
    t1u,
    t2u,
    t1i,
    t2i
]:

    if len(t) > 1:

        dt_candidates.append(
            np.median(
                np.diff(t)
            )
        )

dt_port = min(
    dt_candidates
)

common_t = np.arange(
    t_start,
    t_stop + 0.5 * dt_port,
    dt_port
)

u1 = np.interp(
    common_t,
    t1u,
    u1
)

u2 = np.interp(
    common_t,
    t2u,
    u2
)

i1 = np.interp(
    common_t,
    t1i,
    i1
)

i2 = np.interp(
    common_t,
    t2i,
    i2
)

t = common_t


print()
print(
    "Common time samples:",
    len(t)
)

print(
    "Common time range:",
    f"{t[0]:.6e} -> {t[-1]:.6e} s"
)

print(
    "Common timestep:",
    f"{dt_port:.6e} s"
)


# ============================================================
# SAVE RECONSTRUCTED TIME-DOMAIN DATA
# ============================================================

TIME_FILE = RUN_DIR / "reconstructed_port_data.csv"

np.savetxt(
    TIME_FILE,
    np.column_stack(
        (
            t,
            u1,
            i1,
            u2,
            i2
        )
    ),
    delimiter=",",
    header=(
        "time_s,"
        "V_port1,"
        "I_port1,"
        "V_port2,"
        "I_port2"
    ),
    comments=""
)

print()
print(
    "Time-domain data saved:",
    TIME_FILE
)


# ============================================================
# FOURIER TRANSFORM
# ============================================================

print()
print("=" * 72)
print("CALCULATING FREQUENCY-DOMAIN PORT WAVES")
print("=" * 72)


def fourier_transform(
    signal,
    time,
    frequencies
):
    """
    Direct Fourier transform.

    This avoids relying on FFT bin spacing and allows evaluation
    directly at the requested frequencies.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    time = np.asarray(
        time,
        dtype=float
    )

    frequencies = np.asarray(
        frequencies,
        dtype=float
    )

    # Remove DC offset using the mean of the final 10%.
    n_tail = max(
        10,
        len(signal) // 10
    )

    offset = np.mean(
        signal[-n_tail:]
    )

    signal = signal - offset

    # Apply a Tukey-like cosine taper manually.
    #
    # 10% taper at each end.
    N = len(signal)

    window = np.ones(N)

    taper = max(
        1,
        int(0.10 * N)
    )

    x = np.linspace(
        0,
        np.pi / 2,
        taper
    )

    window[:taper] = np.sin(x) ** 2
    window[-taper:] = np.cos(x) ** 2

    signal = signal * window

    # Fourier transform.
    #
    # Integral:
    #
    # X(f) = integral x(t)e^(-j2πft)dt
    #
    # Using trapezoidal integration.
    #

    result = np.zeros(
        len(frequencies),
        dtype=complex
    )

    for k, frequency in enumerate(
        frequencies
    ):

        phase = np.exp(
            -2j
            * np.pi
            * frequency
            * time
        )

        result[k] = np.trapezoid(
            signal * phase,
            time
        )

    return result


print(
    f"Transforming {len(FREQ)} frequency points..."
)

UF1 = fourier_transform(
    u1,
    t,
    FREQ
)

IF1 = fourier_transform(
    i1,
    t,
    FREQ
)

UF2 = fourier_transform(
    u2,
    t,
    FREQ
)

IF2 = fourier_transform(
    i2,
    t,
    FREQ
)


# ============================================================
# TRAVEL-WAVE DECOMPOSITION
# ============================================================

print()
print("Calculating incident/reflected waves...")


# Port wave convention:
#
#   U_inc = (U + Z0 I) / 2
#   U_ref = (U - Z0 I) / 2
#
# openEMS lumped-port orientation is used here.

UF1_INC = (
    UF1 + Z0 * IF1
) / 2.0

UF1_REF = (
    UF1 - Z0 * IF1
) / 2.0

UF2_INC = (
    UF2 + Z0 * IF2
) / 2.0

UF2_REF = (
    UF2 - Z0 * IF2
) / 2.0


# ============================================================
# S PARAMETERS
# ============================================================

denominator = UF1_INC.copy()

# Avoid numerical division at frequencies where the incident
# spectrum is essentially zero.
threshold = (
    np.max(
        np.abs(denominator)
    )
    * 1e-12
)

valid = (
    np.abs(denominator)
    > threshold
)

S11 = np.full(
    len(FREQ),
    np.nan,
    dtype=complex
)

S21 = np.full(
    len(FREQ),
    np.nan,
    dtype=complex
)

S11[valid] = (
    UF1_REF[valid]
    /
    UF1_INC[valid]
)

S21[valid] = (
    UF2_REF[valid]
    /
    UF1_INC[valid]
)


# ============================================================
# MAGNITUDE IN dB
# ============================================================

S11dB = np.full(
    len(FREQ),
    np.nan
)

S21dB = np.full(
    len(FREQ),
    np.nan
)

S11dB[valid] = (
    20
    * np.log10(
        np.maximum(
            np.abs(
                S11[valid]
            ),
            1e-15
        )
    )
)

S21dB[valid] = (
    20
    * np.log10(
        np.maximum(
            np.abs(
                S21[valid]
            ),
            1e-15
        )
    )
)


# ============================================================
# PRINT S PARAMETERS
# ============================================================

print()
print("=" * 72)
print("S-PARAMETERS")
print("=" * 72)

print(
    f"{'Freq':>10s}"
    f" | "
    f"{'S11':>12s}"
    f" | "
    f"{'S21':>12s}"
)

print("-" * 72)

for f in [
    1,
    5,
    8,
    9,
    10,
    11,
    12,
    15,
    20,
    30,
    40,
    50,
    60,
    70
]:

    k = np.argmin(
        np.abs(
            FREQ / 1e9 - f
        )
    )

    print(
        f"{f:8.1f} GHz"
        f" | "
        f"{S11dB[k]:10.3f} dB"
        f" | "
        f"{S21dB[k]:10.3f} dB"
    )

print("=" * 72)


# ============================================================
# SAVE CSV
# ============================================================

CSV_FILE = (
    RUN_DIR
    / "final_sparameters.csv"
)

np.savetxt(
    CSV_FILE,
    np.column_stack(
        (
            FREQ / 1e9,
            np.abs(S11),
            S11dB,
            np.angle(S11, deg=True),
            np.abs(S21),
            S21dB,
            np.angle(S21, deg=True)
        )
    ),
    delimiter=",",
    header=(
        "Frequency_GHz,"
        "absS11,S11_dB,S11_phase_deg,"
        "absS21,S21_dB,S21_phase_deg"
    ),
    comments=""
)

print()
print(
    "CSV saved:",
    CSV_FILE
)


# ============================================================
# WRITE TOUCHSTONE S2P
# ============================================================

S2P_FILE = (
    RUN_DIR
    / "final_filter.s2p"
)

with open(
    S2P_FILE,
    "w"
) as f:

    f.write(
        "! Cryogenic low-pass filter\n"
    )

    f.write(
        "! openEMS full-wave simulation\n"
    )

    f.write(
        "! Reconstructed from port voltage/current data\n"
    )

    f.write(
        "# GHz S RI R 50\n"
    )

    for k, freq in enumerate(
        FREQ
    ):

        if not valid[k]:
            continue

        f.write(
            f"{freq / 1e9:.9g} "
            f"{S11[k].real:.12e} "
            f"{S11[k].imag:.12e} "
            f"{S21[k].real:.12e} "
            f"{S21[k].imag:.12e} "
            f"{S21[k].real * 0:.12e} "
            f"{S21[k].imag * 0:.12e} "
            f"{S11[k].real * 0:.12e} "
            f"{S11[k].imag * 0:.12e}\n"
        )

print(
    "Touchstone saved:",
    S2P_FILE
)


# ============================================================
# PLOT
# ============================================================

PLOT_FILE = (
    RUN_DIR
    / "final_sparameters.png"
)

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    FREQ / 1e9,
    S21dB,
    label="S21"
)

plt.plot(
    FREQ / 1e9,
    S11dB,
    label="S11"
)

plt.axvline(
    10,
    linestyle="--",
    label="10 GHz"
)

plt.axhline(
    -60,
    linestyle=":",
    label="-60 dB"
)

plt.axhline(
    -40,
    linestyle=":",
    label="-40 dB"
)

plt.xlim(
    1,
    70
)

plt.ylim(
    -100,
    5
)

plt.xlabel(
    "Frequency (GHz)"
)

plt.ylabel(
    "Magnitude (dB)"
)

plt.title(
    "Final Filter — Full-Wave openEMS S-Parameters"
)

plt.grid(
    True,
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=250
)

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 72)
print("FINAL SIMULATION COMPLETE")
print("=" * 72)

print()
print("Run directory:")
print(RUN_DIR)

print()
print("Files:")
print("  XML       :", SIM_XML)
print("  CSV       :", CSV_FILE)
print("  S2P       :", S2P_FILE)
print("  Plot      :", PLOT_FILE)
print("  Time data :", TIME_FILE)

print()
print("=" * 72)
print("DONE")
print("=" * 72)