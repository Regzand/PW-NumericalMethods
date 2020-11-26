import numpy as np

from solution.integrals import simpson_integral


def tabularize(f, start, stop, samples):
    xs = np.linspace(start, stop, samples)
    ys = np.array(list(map(f, xs)))
    return np.column_stack([xs, ys])


def functions_difference(f, g, a, b, m):
    """
    Calculates averaged difference between given two functions in provided range
    Args:
        f: first function as a float -> float callable
        g: second function as a float -> float callable
        a: start of the test range
        b: end of the test range
        m: number of sub-ranges used in integral calculation
    Returns:
        Averaged difference between functions
    """
    assert a < b, f'invalid range [{a}, {b}]'

    diff = lambda x: abs(f(x) - g(x))
    integral = simpson_integral(diff, a, b, m)
    return 1.0 / (b - a) * integral
