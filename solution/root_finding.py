import numpy as np


class NewtonRaphson:

    def __init__(self, f, x0, dx, scale=1.0):
        """
        Args:
            f: function R->R that meets requirements of Newton-Raphson method.
            x0: starting value
            dx: step used in approximation of derivative
            scale: value that scales values of f in order to overcome numerical errors in some cases
        """
        self._f = f
        self._x = x0
        self._dx = dx
        self._scale = scale

        self._history = [(self.x, self.y)]

    def step(self):
        """ Performs one iteration. """
        y1 = self._f(self._x)
        y2 = self._f(self._x + self._dx)
        dy = (y2 - y1) * self._scale
        df = dy / self._dx

        self._x -= y1 / df * self._scale

        # update history
        self._history.append((self.x, self.y))

    def run(self, epsilon=1e-4, max_steps=None):
        """ Finds function root with precision of at least epsilon. """
        max_steps = max_steps or float('inf')

        while abs(self.y) > epsilon and len(self._history) < max_steps:
            self.step()

    @property
    def x(self):
        """ Current value of x """
        return self._x

    @property
    def y(self):
        """ Current value of y [=f(x)]. """
        return self._f(self._x)

    @property
    def history(self):
        """ History of the iterations consisting of entries in format (x, y). """
        return np.array(self._history)
