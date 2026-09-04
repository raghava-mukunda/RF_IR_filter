from pathlib import Path

import numpy as np

from src.filter.spec import FilterSpec
from src.filter.chebyshev import design_filter
from src.rf.golden_model import (
    chebyshev_s21_db,
    s11_db_from_s21_db,
)
from src.rf.plot_sparams import (
    plot_s_parameters,
)


def main():

    spec = FilterSpec()

    design = design_filter(
        spec
    )

    print()
    print("======================================")
    print("CRYOGENIC FILTER V1")
    print("CHEBYSHEV REFERENCE DESIGN")
    print("======================================")

    print()
    print("Specification")
    print("--------------------------------------")

    print(
        f"Z0             : {spec.z0:.1f} ohm"
    )

    print(
        f"Passband edge  : "
        f"{spec.passband_edge / 1e9:.3f} GHz"
    )

    print(
        f"Ripple         : "
        f"{spec.passband_ripple_db:.3f} dB"
    )

    print(
        f"Stopband       : "
        f"{spec.stopband_start / 1e9:.3f} GHz"
    )

    print(
        f"Required As    : "
        f"{spec.stopband_attenuation_db:.1f} dB"
    )

    print()
    print("Synthesis")
    print("--------------------------------------")

    print(
        f"Exact order    : "
        f"{design['exact_order']:.10f}"
    )

    print(
        f"Chosen order   : "
        f"{design['order']}"
    )

    print()
    print("g-values")
    print("--------------------------------------")

    for k, value in enumerate(
        design["g"]
    ):

        print(
            f"g{k} = {value:.10f}"
        )

    print()
    print("Equivalent L/C prototype")
    print("--------------------------------------")

    for index, value in enumerate(
        design["L"],
        start=1,
    ):

        print(
            f"L{2 * index - 1} = "
            f"{value * 1e12:.6f} pH"
        )

    for index, value in enumerate(
        design["C"],
        start=1,
    ):

        print(
            f"C{2 * index} = "
            f"{value * 1e15:.6f} fF"
        )

    # --------------------------------------------------
    # Frequency grid
    # --------------------------------------------------

    frequency = np.linspace(
        spec.analysis_start,
        spec.analysis_stop,
        10001,
    )

    # --------------------------------------------------
    # Golden analytical response
    # --------------------------------------------------

    s21_db = chebyshev_s21_db(
        frequency,
        spec.passband_edge,
        spec.passband_ripple_db,
        design["order"],
    )

    s11_db = s11_db_from_s21_db(
        s21_db
    )

    # --------------------------------------------------
    # Specification checks
    # --------------------------------------------------

    passband_mask = (
        frequency <= spec.passband_edge
    )

    stopband_mask = (
        frequency >= spec.stopband_start
    )

    maximum_passband_loss = (
        -np.min(
            s21_db[passband_mask]
        )
    )

    minimum_stopband_attenuation = (
        -np.max(
            s21_db[stopband_mask]
        )
    )

    print()
    print("Golden-model validation")
    print("--------------------------------------")

    print(
        f"Maximum passband loss : "
        f"{maximum_passband_loss:.6f} dB"
    )

    print(
        f"Minimum stopband loss : "
        f"{minimum_stopband_attenuation:.6f} dB"
    )

    if maximum_passband_loss <= (
        spec.passband_ripple_db + 1e-9
    ):
        print(
            "PASS: passband ripple requirement"
        )
    else:
        print(
            "FAIL: passband ripple requirement"
        )

    if minimum_stopband_attenuation >= (
        spec.stopband_attenuation_db
    ):
        print(
            "PASS: 20 GHz attenuation requirement"
        )
    else:
        print(
            "FAIL: 20 GHz attenuation requirement"
        )

    # --------------------------------------------------
    # Save reference data
    # --------------------------------------------------

    results_dir = Path(
        "results/reference"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = np.column_stack(
        (
            frequency,
            s11_db,
            s21_db,
        )
    )

    csv_file = (
        results_dir
        / "chebyshev_reference.csv"
    )

    np.savetxt(
        csv_file,
        data,
        delimiter=",",
        header="frequency_hz,s11_db,s21_db",
        comments="",
    )

    print()
    print(
        f"Saved reference data: {csv_file}"
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    plot_s_parameters(
        frequency,
        s11_db,
        s21_db,
        output_file=(
            "results/plots/"
            "chebyshev_reference.png"
        ),
        title=(
            "7th-Order 0.1 dB Chebyshev-I "
            "Cryogenic Filter Reference"
        ),
    )

    print()
    print("Reference design complete.")


if __name__ == "__main__":
    main()
