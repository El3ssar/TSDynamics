import numpy as np

from tsdynamics.families import DiscreteMap


class Gauss(DiscreteMap):
    params = {"a": 4.9, "b": -0.5}
    dim = 1

    @staticmethod
    def _step(X, a, b):
        x = X
        return np.exp(-a * x**2) + b

    @staticmethod
    def _jacobian(X, a, b):
        x = X
        return [-2 * a * x * np.exp(-a * x**2)]


class DeJong(DiscreteMap):
    params = {"a": 1.641, "b": 1.902, "c": 0.316, "d": 1.525}
    dim = 2

    @staticmethod
    def _step(X, a, b, c, d):
        x, y = X
        xp = np.sin(a * y) - np.cos(b * x)
        yp = np.sin(c * x) - np.cos(d * y)
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b, c, d):
        x, y = X
        row1 = [b * np.sin(b * x), a * np.cos(a * y)]
        row2 = [c * np.cos(c * x), d * np.sin(d * y)]
        return row1, row2


class KaplanYorke(DiscreteMap):
    params = {"alpha": 0.2}
    dim = 2

    @staticmethod
    def _step(X, alpha):
        x, y = X
        # Doubling map on the unit circle. The canonical wrap is mod 1, but the
        # pure doubling map x -> frac(2x) drains a float mantissa one bit per
        # step (2x is an exact shift) and collapses to the x=0 fixed point in
        # ~52 iterations, killing the chaos. Wrapping just below 1 injects a
        # ~5e-8 offset at each fold that keeps the orbit non-degenerate; the
        # dynamics (and Lyapunov exponent ln 2) are unchanged to that tolerance.
        xp = (2 * x) % 0.99999995
        yp = alpha * y + np.cos(4 * np.pi * x)
        return xp, yp

    @staticmethod
    def _jacobian(X, alpha):
        x, y = X
        row1 = [2, 0]
        row2 = [-4 * np.pi * np.sin(4 * np.pi * x), alpha]
        return row1, row2
