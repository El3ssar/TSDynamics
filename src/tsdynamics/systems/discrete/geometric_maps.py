import numpy as np

from tsdynamics.base import DiscreteMap
from tsdynamics.utils import staticjit


class Tent(DiscreteMap):
    params = {"mu": 0.95}
    dim = 1

    @staticjit
    def _step(X, mu):
        x = X
        return mu * (1 - 2 * np.abs(x - 0.5))

    @staticjit
    def _jacobian(X, mu):
        x = X
        return [np.where(x < 0.5, -2 * mu, 2 * mu)]


class Baker(DiscreteMap):
    params = {"alpha": 0.5}
    dim = 2

    @staticjit
    def _step(X, alpha):
        x, y = X
        in_first = (y >= 0) & (y < alpha)
        xp = np.where(in_first, (2 * x) % 1, (2 * x - 1) % 1)
        yp = np.where(in_first, y / alpha, (y - alpha) / (1 - alpha))
        return xp, yp

    @staticjit
    def _jacobian(X, alpha):
        x, y = X
        in_first = (y >= 0) & (y < alpha)
        j11 = np.where(in_first, 1 / alpha, 1 / (1 - alpha))
        return (2.0, 0.0), (0.0, j11)


class Circle(DiscreteMap):
    """
    Arnold's circle map.

    Parameter order matches ``params`` insertion order (``omega`` then ``k``);
    the previous implementation had the two swapped in the function signature,
    silently using ``omega`` as ``k`` and vice versa.
    """

    params = {"omega": 0.333, "k": 5.7}
    dim = 1

    @staticjit
    def _step(X, omega, k):
        theta = X
        thetap = theta + omega + (k / (2 * np.pi)) * np.sin(2 * np.pi * theta)
        return thetap % 1

    @staticjit
    def _jacobian(X, omega, k):
        theta = X
        return [1 + k * np.cos(2 * np.pi * theta)]


class Chebyshev(DiscreteMap):
    params = {"a": 6.0}
    dim = 1

    @staticjit
    def _step(X, a):
        x = X
        return np.cos(a * np.arccos(x))

    @staticjit
    def _jacobian(X, a):
        x = X
        return [-a * np.sin(a * np.arccos(x)) / np.sqrt(1 - x**2)]
