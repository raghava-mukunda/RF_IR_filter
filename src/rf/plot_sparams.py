from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_s_parameters(
    frequency_hz,
    s11_db,
    s21_db,
    output_file=None,
    title="S-Parameters",
):

    frequency_ghz = np.asarray(
        frequency_hz
    ) / 1e9

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        frequency_ghz,
        s21_db,
        label="S21",
        linewidth=1.5,
    )

    ax.plot(
        frequency_ghz,
        s11_db,
        label="S11",
        linewidth=1.5,
    )

    ax.axvline(
        9.0,
        linestyle="--",
        linewidth=1.0,
        label="9 GHz passband edge",
    )

    ax.axvline(
        20.0,
        linestyle="--",
        linewidth=1.0,
        label="20 GHz stopband requirement",
    )

    ax.axhline(
        -60.0,
        linestyle=":",
        linewidth=1.0,
        label="-60 dB attenuation",
    )

    ax.set_xlabel(
        "Frequency (GHz)"
    )

    ax.set_ylabel(
        "Magnitude (dB)"
    )

    ax.set_title(title)

    ax.set_xlim(
        frequency_ghz.min(),
        frequency_ghz.max(),
    )

    ax.set_ylim(
        -160,
        5,
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    if output_file is not None:

        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_file,
            dpi=200,
        )

        print(
            f"Saved plot: {output_file}"
        )

    return fig, ax
