import numpy as np

from .spec import FilterSpec


def ripple_factor(ripple_db: float) -> float:
    """
    Chebyshev ripple factor:

        epsilon = sqrt(10^(Rp/10) - 1)
    """
    return np.sqrt(10.0 ** (ripple_db / 10.0) - 1.0)


def chebyshev_order(
    passband_edge: float,
    stopband_frequency: float,
    ripple_db: float,
    attenuation_db: float,
) -> tuple[float, int]:

    epsilon = ripple_factor(ripple_db)

    omega_s = stopband_frequency / passband_edge

    numerator = np.arccosh(
        np.sqrt(
            (10.0 ** (attenuation_db / 10.0) - 1.0)
            / (epsilon ** 2)
        )
    )

    denominator = np.arccosh(omega_s)

    exact_order = numerator / denominator

    order = int(np.ceil(exact_order))

    return exact_order, order


def chebyshev_g_values(
    order: int,
    ripple_db: float,
) -> np.ndarray:

    if order < 1:
        raise ValueError("Filter order must be >= 1.")

    epsilon = ripple_factor(ripple_db)

    beta = np.arcsinh(1.0 / epsilon) / order
    alpha = np.sinh(beta)

    g = np.ones(order + 2)

    a = np.zeros(order + 1)

    for k in range(1, order + 1):
        a[k] = np.sin(
            (2.0 * k - 1.0) * np.pi / (2.0 * order)
        )

    g[1] = 2.0 * a[1] / alpha

    for k in range(2, order + 1):

        b_previous = (
            alpha ** 2
            + np.sin((k - 1) * np.pi / order) ** 2
        )

        g[k] = (
            4.0
            * a[k - 1]
            * a[k]
            / (b_previous * g[k - 1])
        )

    if order % 2 == 1:
        g[order + 1] = 1.0

    else:
        g[order + 1] = 1.0 / np.tanh(
            beta
        ) ** 2

    return g


def equivalent_lc_values(
    g: np.ndarray,
    z0: float,
    fc: float,
) -> tuple[np.ndarray, np.ndarray]:

    omega_c = 2.0 * np.pi * fc

    order = len(g) - 2

    inductors = []
    capacitors = []

    for k in range(1, order + 1):

        if k % 2 == 1:
            L = z0 * g[k] / omega_c
            inductors.append(L)

        else:
            C = g[k] / (z0 * omega_c)
            capacitors.append(C)

    return (
        np.array(inductors),
        np.array(capacitors),
    )


def design_filter(spec: FilterSpec):

    spec.validate()

    exact_order, order = chebyshev_order(
        spec.passband_edge,
        spec.stopband_start,
        spec.passband_ripple_db,
        spec.stopband_attenuation_db,
    )

    g = chebyshev_g_values(
        order,
        spec.passband_ripple_db,
    )

    L, C = equivalent_lc_values(
        g,
        spec.z0,
        spec.passband_edge,
    )

    return {
        "exact_order": exact_order,
        "order": order,
        "g": g,
        "L": L,
        "C": C,
    }


if __name__ == "__main__":

    spec = FilterSpec()

    design = design_filter(spec)

    print()
    print("CHEBYSHEV TYPE-I SYNTHESIS")
    print("===========================")

    print(
        f"Exact order = "
        f"{design['exact_order']:.10f}"
    )

    print(
        f"Selected order = "
        f"{design['order']}"
    )

    print()
    print("g-values:")

    for i, value in enumerate(design["g"]):
        print(
            f"g{i} = {value:.10f}"
        )

    print()
    print("Equivalent series inductors:")

    for i, value in enumerate(design["L"], start=1):
        print(
            f"L{i} = {value * 1e12:.6f} pH"
        )

    print()
    print("Equivalent shunt capacitors:")

    for i, value in enumerate(design["C"], start=1):
        print(
            f"C{i} = {value * 1e15:.6f} fF"
        )
