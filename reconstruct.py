from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RUN_DIR = Path(
    r"C:\Users\Raghava.m\Desktop\cryogenic_filter_design"
    r"\FINAL_OPENEMS_RIGOROUS\rf_final_300k"
)

FILES = {
    "u1": RUN_DIR / "port_ut_1",
    "u2": RUN_DIR / "port_ut_2",
    "i1": RUN_DIR / "port_it_1",
    "i2": RUN_DIR / "port_it_2",
}


# ============================================================
# OPENEMS SAMPLING INTERVALS
# ============================================================

DT_U = 1.31476236814e-12
DT_I = 1.31543868623e-12


# ============================================================
# REGEX
# ============================================================

# Ordinary scientific number
SCIENTIFIC_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)"
)

# Ordinary decimal number
DECIMAL_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)"
)

# Normal timestamp such as:
#
# 1.31476236814e-12
# 11.18328613133e-11
# 1.05180989451e-10
#
TIMESTAMP_NORMAL_RE = re.compile(
    r"[-+]?\d+\.\d+(?:[eE])-\d{2}"
)

# Malformed timestamp such as:
#
# 1.311.31543868623e-12
# 2.632.63020105437e-12
#
# Meaning:
#
# 1.31543868623e-12
# 2.63020105437e-12
#
TIMESTAMP_MANGLED_RE = re.compile(
    r"(?P<first>\d)\.(?P<junk>\d+)\.(?P<rest>\d+(?:[eE])-\d{2})"
)


# ============================================================
# LOAD RAW OPENEMS FILE
# ============================================================

def load_raw(path):

    with open(path, "r", errors="replace") as f:
        text = f.read()

    # Remove OpenEMS header lines.
    lines = text.splitlines()

    data_lines = []

    for line in lines:

        if line.startswith("%"):
            continue

        data_lines.append(line)

    return "\n".join(data_lines)


# ============================================================
# FIND TIMESTAMPS
# ============================================================

def find_timestamps(text, current_file=False):

    candidates = []

    # --------------------------------------------------------
    # CURRENT FILES
    # --------------------------------------------------------
    #
    # Current files contain malformed timestamps like:
    #
    # 1.311.31543868623e-12
    #
    # We process these FIRST.
    # --------------------------------------------------------

    if current_file:

        mangled_spans = []

        for m in TIMESTAMP_MANGLED_RE.finditer(text):

            first = m.group("first")
            rest = m.group("rest")

            # Example:
            #
            # first = "1"
            # rest  = "31543868623e-12"
            #
            # => 1.31543868623e-12

            ts_text = first + "." + rest

            try:
                t = float(ts_text)
            except ValueError:
                continue

            candidates.append({
                "start": m.start(),
                "end": m.end(),
                "time": t,
            })

            mangled_spans.append(
                (m.start(), m.end())
            )

        # ----------------------------------------------------
        # Also find ordinary timestamps.
        #
        # Ignore ordinary matches which occur INSIDE a
        # malformed timestamp.
        # ----------------------------------------------------

        for m in TIMESTAMP_NORMAL_RE.finditer(text):

            inside_mangled = False

            for s, e in mangled_spans:

                if m.start() >= s and m.end() <= e:
                    inside_mangled = True
                    break

            if inside_mangled:
                continue

            try:
                t = float(m.group())
            except ValueError:
                continue

            candidates.append({
                "start": m.start(),
                "end": m.end(),
                "time": t,
            })

    # --------------------------------------------------------
    # VOLTAGE FILES
    # --------------------------------------------------------

    else:

        for m in TIMESTAMP_NORMAL_RE.finditer(text):

            try:
                t = float(m.group())
            except ValueError:
                continue

            candidates.append({
                "start": m.start(),
                "end": m.end(),
                "time": t,
            })

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["time"]
    )

    return candidates


# ============================================================
# SELECT PHYSICAL TIMESTAMP SEQUENCE
# ============================================================

def select_timestamp_sequence(
    candidates,
    expected_dt
):

    if not candidates:
        return []

    # --------------------------------------------------------
    # We know approximately what the timestamp sequence must be:
    #
    # t[n+1] - t[n] ≈ expected_dt
    #
    # Start with the earliest physically plausible timestamp.
    # --------------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda x: x["time"]
    )

    selected = []

    # First timestamp should be close to DT.
    first = min(
        candidates,
        key=lambda x: abs(
            x["time"] - expected_dt
        )
    )

    selected.append(first)

    current_time = first["time"]

    used = {id(first)}

    # --------------------------------------------------------
    # Keep walking forward.
    # --------------------------------------------------------

    while True:

        target = current_time + expected_dt

        available = [
            c
            for c in candidates
            if id(c) not in used
            and c["time"] > current_time
        ]

        if not available:
            break

        best = min(
            available,
            key=lambda x: abs(
                x["time"] - target
            )
        )

        error = abs(
            best["time"] - target
        )

        # 5% tolerance.
        if error > expected_dt * 0.05:

            # Look for an exact-ish candidate anyway.
            close = [
                c
                for c in available
                if abs(
                    c["time"] - target
                ) <= expected_dt * 0.10
            ]

            if not close:
                break

            best = min(
                close,
                key=lambda x: abs(
                    x["time"] - target
                )
            )

        selected.append(best)

        used.add(id(best))

        current_time = best["time"]

    return selected


# ============================================================
# EXTRACT SIGNAL BETWEEN TWO TIMESTAMPS
# ============================================================

def extract_signal_between(
    fragment,
    current_file=False
):

    fragment = fragment.strip()

    if not fragment:
        return np.nan

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The signal is BEFORE the next timestamp.
    #
    # Examples:
    #
    # voltage:
    #
    # 0.00182337109436
    #
    # current:
    #
    # 1.11746478737e-2
    #
    # --------------------------------------------------------

    # Remove tabs/newlines from the edges only.
    fragment = fragment.strip()

    # --------------------------------------------------------
    # Split by tabs first.
    # --------------------------------------------------------

    fields = re.split(
        r"[\t\r\n]+",
        fragment
    )

    candidates = []

    for field in fields:

        field = field.strip()

        if not field:
            continue

        # Exact scientific number
        m = SCIENTIFIC_RE.fullmatch(field)

        if m:

            try:
                candidates.append(
                    float(m.group())
                )
            except ValueError:
                pass

            continue

        # Exact decimal
        m = DECIMAL_RE.fullmatch(field)

        if m:

            try:
                candidates.append(
                    float(m.group())
                )
            except ValueError:
                pass

    # --------------------------------------------------------
    # If exactly one clean number exists, use it.
    # --------------------------------------------------------

    if len(candidates) == 1:

        return candidates[0]

    # --------------------------------------------------------
    # The fields can be concatenated.
    #
    # Example:
    #
    # -0.0002143833808081
    #
    # We need the signal at the beginning of the fragment,
    # NOT an arbitrary number later in the fragment.
    # --------------------------------------------------------

    # Try scientific notation at the beginning.
    m = re.match(
        r"^([-+]?(?:\d+\.\d*|\.\d+|\d+)"
        r"(?:[eE][-+]?\d+)?)",
        fragment
    )

    if m:

        token = m.group(1)

        try:
            return float(token)
        except ValueError:
            pass

    # --------------------------------------------------------
    # Try ordinary decimal at the beginning.
    # --------------------------------------------------------

    m = re.match(
        r"^([-+]?(?:\d+\.\d*|\.\d+|\d+))",
        fragment
    )

    if m:

        token = m.group(1)

        try:
            return float(token)
        except ValueError:
            pass

    # --------------------------------------------------------
    # Explicit zero.
    # --------------------------------------------------------

    if fragment == "0":
        return 0.0

    if fragment == "-0":
        return -0.0

    return np.nan


# ============================================================
# RECONSTRUCT ONE PORT
# ============================================================

def reconstruct(
    path,
    expected_dt
):

    print()
    print("=" * 70)
    print(path.name)
    print("=" * 70)

    text = load_raw(path)

    current_file = (
        "port_it" in path.name
    )

    # --------------------------------------------------------
    # Find timestamp candidates.
    # --------------------------------------------------------

    candidates = find_timestamps(
        text,
        current_file=current_file
    )

    print(
        "timestamp candidates:",
        len(candidates)
    )

    if len(candidates) < 10:

        raise RuntimeError(
            f"Too few timestamps found in {path.name}"
        )

    # --------------------------------------------------------
    # Select actual physical sequence.
    # --------------------------------------------------------

    timestamps = select_timestamp_sequence(
        candidates,
        expected_dt
    )

    print(
        "selected timestamps:",
        len(timestamps)
    )

    if len(timestamps) < 5:

        raise RuntimeError(
            f"Could not reconstruct timestamp sequence "
            f"for {path.name}"
        )

    # --------------------------------------------------------
    # Extract signal values.
    # --------------------------------------------------------

    times = []
    values = []

    for n in range(
        len(timestamps) - 1
    ):

        a = timestamps[n]
        b = timestamps[n + 1]

        t0 = a["time"]
        t1 = b["time"]

        actual_dt = t1 - t0

        # Reject bad gaps.
        if actual_dt <= 0:
            continue

        if abs(
            actual_dt - expected_dt
        ) > expected_dt * 0.05:

            continue

        # ----------------------------------------------------
        # Signal is physically between timestamp n and n+1.
        # ----------------------------------------------------

        fragment = text[
            a["end"]:
            b["start"]
        ]

        value = extract_signal_between(
            fragment,
            current_file=current_file
        )

        times.append(t0)
        values.append(value)

    times = np.asarray(
        times,
        dtype=float
    )

    values = np.asarray(
        values,
        dtype=float
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    valid = np.isfinite(values)

    print(
        "reconstructed samples:",
        len(values)
    )

    print(
        "valid samples:",
        np.sum(valid)
    )

    print(
        "invalid samples:",
        np.sum(~valid)
    )

    if np.any(valid):

        print(
            "value range:",
            np.nanmin(values),
            "to",
            np.nanmax(values)
        )

        print()
        print("first 20 valid samples:")

        valid_indices = np.where(valid)[0]

        for k in valid_indices[:20]:

            print(
                f"{times[k] * 1e12:12.6f} ps   "
                f"{values[k]: .12e}"
            )

    return times, values


# ============================================================
# RECONSTRUCT ALL FOUR PORT SIGNALS
# ============================================================

tu1, u1 = reconstruct(
    FILES["u1"],
    DT_U
)

tu2, u2 = reconstruct(
    FILES["u2"],
    DT_U
)

ti1, i1 = reconstruct(
    FILES["i1"],
    DT_I
)

ti2, i2 = reconstruct(
    FILES["i2"],
    DT_I
)


# ============================================================
# SAVE RECONSTRUCTED DATA
# ============================================================

np.savetxt(
    RUN_DIR / "reconstructed_u1.csv",
    np.column_stack([
        tu1,
        u1
    ]),
    delimiter=",",
    header="time_s,voltage_V",
    comments=""
)

np.savetxt(
    RUN_DIR / "reconstructed_u2.csv",
    np.column_stack([
        tu2,
        u2
    ]),
    delimiter=",",
    header="time_s,voltage_V",
    comments=""
)

np.savetxt(
    RUN_DIR / "reconstructed_i1.csv",
    np.column_stack([
        ti1,
        i1
    ]),
    delimiter=",",
    header="time_s,current_A",
    comments=""
)

np.savetxt(
    RUN_DIR / "reconstructed_i2.csv",
    np.column_stack([
        ti2,
        i2
    ]),
    delimiter=",",
    header="time_s,current_A",
    comments=""
)


# ============================================================
# PLOT VOLTAGES
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.plot(
    tu1 * 1e12,
    u1,
    label="Port 1 voltage"
)

ax.plot(
    tu2 * 1e12,
    u2,
    label="Port 2 voltage"
)

ax.set_xlabel(
    "Time (ps)"
)

ax.set_ylabel(
    "Voltage (V)"
)

ax.set_title(
    "Reconstructed openEMS port voltages"
)

ax.grid(True)

ax.legend()

plt.tight_layout()

plt.show()


# ============================================================
# PLOT CURRENTS
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.plot(
    ti1 * 1e12,
    i1,
    label="Port 1 current"
)

ax.plot(
    ti2 * 1e12,
    i2,
    label="Port 2 current"
)

ax.set_xlabel(
    "Time (ps)"
)

ax.set_ylabel(
    "Current (A)"
)

ax.set_title(
    "Reconstructed openEMS port currents"
)

ax.grid(True)

ax.legend()

plt.tight_layout()

plt.show()