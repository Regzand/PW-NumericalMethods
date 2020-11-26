import numpy as np


class HeatExchangeSimulation:

    def __init__(self, Tb, Tw, mb, mw, cb, cw, h, a, dt=0.1, improved=True):
        """

        Args:
            Tb: initial temperature of the bar
            Tw: initial temperature of the water
            mb: mass of the bar
            mw: mass of the water
            cb: specific heat capacity of the bar
            cw: specific heat capacity of the water
            h: thermal conductivity, can be a callable that accepts temp delta and returns h
            a: area of contact between the bodies
            dt: time step of simulation
            improved: whether improved Euler's method will be used
        """
        self.improved = improved

        self._t = 0.0
        self._T = np.array([Tb, Tw])
        self._m = np.array([mb, mw])
        self._c = np.array([cb, cw])
        self._a = a
        self._dt = dt
        self._h = h if callable(h) else (lambda dT: h)

        self._history = [
            (self.t, self.Tb, self.Tw, self.h)
        ]

    def step(self, dt=None):
        """ Calculates next step of the simulation. """
        dt = dt or self._dt

        # update time
        self._t += dt

        # update temperatures
        if self.improved:
            yp = self._T + dt / 2 * self._dy(self._T)
            self._T += dt * self._dy(yp)
        else:
            self._T += dt * self._dy(self._T)

        # update history
        self._history.append((self.t, self.Tb, self.Tw, self.h))

    def _dy(self, T):
        """ Derivative for given temperatures. """
        return self._h(np.diff(T)) * self._a / self._m / self._c * np.diff(T) * np.array([1, -1])

    def simulate(self, n, dt=None):
        """ Runs simulation for n steps. """
        for _ in range(n):
            self.step(dt)

    def simulate_until(self, stop_time, dt=None):
        """ Runs simulation up to given time. """
        while self.t < stop_time:
            self.step(dt)

    @property
    def t(self):
        """ Current simulation time. """
        return self._t

    @property
    def Tb(self):
        """ Current bar temperature. """
        return self._T[0]

    @property
    def Tw(self):
        """ Current water temperature. """
        return self._T[1]

    @property
    def h(self):
        """ Current thermal conductivity. """
        return self._h(np.diff(self._T))

    @property
    def history(self):
        """ History of the simulation consisting of entries in format (t, Tb, Tw). """
        return np.array(self._history)

    @property
    def t_history(self):
        """ Simulation time history. """
        return self.history[:, 0]

    @property
    def Tb_history(self):
        """ Bar temperature history. """
        return self.history[:, 1]

    @property
    def Tw_history(self):
        """ Water temperature history. """
        return self.history[:, 2]

    @property
    def h_history(self):
        """ Thermal conductivity history. """
        return self.history[:, 3]
