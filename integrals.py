import numpy as np


def simpson_integral(f, a, b, m):
    """
    Calculates integral of given f function in [a, b] range.
    Args:
        f: float -> float callable
        a: start of a range
        b: end of a range
        m: number of sub-ranges to use, has to be a multpile of 2
    Returns:
        Approximation of a integral
    """
    assert m % 2 == 0, 'm must be even'
    assert a < b, f'invalid range [{a}, {b}]'

    xs = np.linspace(a, b, m + 1)
    ys = np.array(list(map(f, xs)))

    constants = np.fromfunction(lambda i: 2 + 2 * (i % 2), shape=(m + 1,))
    constants[[0, -1]] = 1

    return (b - a) / m / 3 * (constants * ys).sum()
