from functools import cached_property

import numpy as np


def approximation_error(f, samples):
    xs = samples[:, 0]
    ys = samples[:, 1]
    approx_ys = np.array(list(map(f, xs)))
    return np.sqrt(((ys - approx_ys) ** 2).sum() / len(ys))


class PolynomialApproximation:

    def __init__(self, samples, degree):
        self._samples = samples
        self._degree = degree

    def __call__(self, x):
        return (self._matrix_A * x ** np.arange(self._degree + 1)).sum()

    @cached_property
    def _matrix_A(self):
        M = self._matrix_M
        Y = self._matrix_Y
        return np.linalg.inv(M.T.dot(M)).dot(M.T).dot(Y)

    @cached_property
    def _matrix_M(self):
        return self._samples[:, :1] ** np.arange(self._degree + 1)

    @cached_property
    def _matrix_Y(self):
        return self._samples[:, 1]
