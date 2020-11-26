from functools import cached_property

import numpy as np


class CubicSpline:

    def __init__(self, samples):
        self._n = len(samples) - 1
        self._samples = samples

    def __call__(self, x):
        for i in range(1, self._n + 1):
            if x <= self._samples[i, 0]:
                return (self._coefficients[i - 1, :] * x ** np.array([3, 2, 1, 0])).sum()
        return (self._coefficients[-1, :] * x ** np.array([3, 2, 1, 0])).sum()

    @cached_property
    def _coefficients(self):
        return self._matrix_X.reshape(-1, 4)

    @cached_property
    def _matrix_X(self):
        return np.linalg.solve(self._matrix_A, self._matrix_B)

    @cached_property
    def _matrix_A(self):
        return np.concatenate([
            self._matrix_A1,
            self._matrix_A2,
            self._matrix_A3,
            self._matrix_A4
        ])

    @cached_property
    def _matrix_B(self):
        return np.concatenate([
            self._matrix_B1,
            self._matrix_B2,
            self._matrix_B3,
            self._matrix_B4
        ])

    @cached_property
    def _matrix_A1(self):
        A1 = np.zeros((2 * self._n, 4 * self._n))
        for k in range(self._n):
            A1[2*k+0, 4*k:4*k+4] = self._samples[k + 0, 0] ** np.array([3, 2, 1, 0])
            A1[2*k+1, 4*k:4*k+4] = self._samples[k + 1, 0] ** np.array([3, 2, 1, 0])
        return A1

    @cached_property
    def _matrix_B1(self):
        B1 = np.repeat(self._samples[:, 1], 2)[1:-1]
        B1 = B1.reshape(-1, 1)
        return B1

    @cached_property
    def _matrix_A2(self):
        A2 = np.zeros((self._n - 1, 4 * self._n))
        for k in range(self._n - 1):
            A2[k, 4*k+0:4*k+3] = self._samples[k + 1, 0] ** np.array([2, 1, 0]) * np.array([3, 2, 1])
            A2[k, 4*k+4:4*k+7] = -1 * self._samples[k + 1, 0] ** np.array([2, 1, 0]) * np.array([3, 2, 1])
        return A2

    @cached_property
    def _matrix_B2(self):
        return np.zeros((self._n - 1, 1))

    @cached_property
    def _matrix_A3(self):
        A3 = np.zeros((self._n - 1, 4 * self._n))
        for k in range(self._n - 1):
            A3[k, 4*k+0:4*k+2] = self._samples[k + 1, 0] ** np.array([1, 0]) * np.array([6, 2])
            A3[k, 4*k+4:4*k+6] = self._samples[k + 1, 0] ** np.array([1, 0]) * np.array([-6, -2])
        return A3

    @cached_property
    def _matrix_B3(self):
        return np.zeros((self._n - 1, 1))

    @cached_property
    def _matrix_A4(self):
        A4 = np.zeros((2, 4 * self._n))
        A4[0, :4] = [6 * self._samples[0, 0], 2, 1, 0]
        A4[1, -4:] = [6 * self._samples[-1, 0], 2, 1, 0]
        return A4

    @cached_property
    def _matrix_B4(self):
        return np.zeros((2, 1))


class LinearSpline:

    def __init__(self, samples):
        self._samples = samples

    def __call__(self, x):
        if not self._samples[0, 0] <= x <= self._samples[-1, 0]:
            return np.nan

        i = np.searchsorted(self._samples[:, 0], x)
        dx = self._samples[i, 0] - self._samples[i - 1, 0]
        dy = self._samples[i, 1] - self._samples[i - 1, 1]
        a = dy / dx
        return self._samples[i - 1, 1] + a * (x - self._samples[i - 1, 0])
