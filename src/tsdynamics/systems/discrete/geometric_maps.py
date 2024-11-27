from tsdynamics.base import DynMap
from tsdynamics.utils import staticjit
import numpy as np


class Tent(DynMap):
    params = {
            "mu": 0.95
        }
    n_dim = 1
    @staticjit
    def _rhs(X, mu):
        x = X
        return mu * (1 - 2 * np.abs(x - 0.5))


class Baker(DynMap):
    params = {"alpha": 0.5}
    n_dim = 2
    @staticmethod
    def _rhs(X, alpha):
        """
        Right-hand side of the Baker map.

        x, y: Current state variables
        alpha: Fraction determining the fold (0 < alpha < 1)
        """
        x, y = X
        if 0 <= y < alpha:
            xp = (2 * x) % 1  # Stretch in x
            yp = y / alpha  # Fold in y
        else:
            xp = (2 * x - 1) % 1  # Shift after stretch
            yp = (y - alpha) / (1 - alpha)  # Fold in y
        return xp, yp


class Circle(DynMap):
    params = {
            "omega": 0.333,
            "k": 5.7
        }
    n_dim = 1
    @staticjit
    def _rhs(X, k, omega):
        theta = X
        thetap = theta + omega + (k / (2 * np.pi)) * np.sin(2 * np.pi * theta)
        thetap = thetap % 1
        return thetap


class Chebyshev(DynMap):
    params = {
            "a": 6.0
        }
    n_dim = 1
    @staticjit
    def _rhs(X, a):
        x = X
        return np.cos(a * np.arccos(x))

