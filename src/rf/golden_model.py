import numpy as np


def chebyshev_s21_db(
    frequency_hz,
    passband_edge_hz,
    ripple_db,
    order,
):
    """
    Ideal lossless Chebyshev-I magnitude response.

    |S21|^2 = 1 / (1 + epsilon^2 T_N(Omega)^2)

    where:
        Omega = f / fc
    """

    frequency_hz = np.asarray(
        frequency_hz,
        dtype=float,
    )

    epsilon = np.sqrt(
        10.0 ** (ripple_db / 10.0) - 1.0
    )

    omega = frequency_hz / passband_edge_hz

    Tn = np.zeros_like(omega)

    inside = np.abs(omega) <= 1.0

    Tn[inside] = np.cos(
        order * np.arccos(omega[inside])
    )

    outside = ~inside

    Tn[outside] = np.cosh(
        order * np.arccosh(np.abs(omega[outside]))
    )

    magnitude_squared = (
        1.0
        / (1.0 + epsilon ** 2 * Tn ** 2)
    )

    magnitude = np.sqrt(
        magnitude_squared
    )

    return 20.0 * np.log10(
        np.maximum(magnitude, 1e-300)
    )


def s11_db_from_s21_db(s21_db):

    s21_linear_squared = (
        10.0 ** (s21_db / 10.0)
    )

    reflection_squared = (
        1.0 - s21_linear_squared
    )

    reflection_squared = np.maximum(
        reflection_squared,
        1e-300,
    )

    return 10.0 * np.log10(
        reflection_squared
    )


if __name__ == "__main__":

    f = np.linspace(
        10e6,
        70e9,
        5001,
    )

    s21 = chebyshev_s21_db(
        f,
        9e9,
        0.1,
        7,
    )

    s11 = s11_db_from_s21_db(
        s21
    )

    print("Golden model")
    print("============")

    for freq in [
        1e9,
        5e9,
        7e9,
        9e9,
        10e9,
        12e9,
        15e9,
        20e9,
        30e9,
        40e9,
        50e9,
        60e9,
        70e9,
    ]:

        index = np.argmin(
            np.abs(f - freq)
        )

        print(
            f"{freq / 1e9:6.1f} GHz : "
            f"S21 = {s21[index]:8.3f} dB, "
            f"S11 = {s11[index]:8.3f} dB"
        )
