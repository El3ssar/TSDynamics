from tsdynamics.base import DynMap
from tsdynamics.utils import staticjit
import numpy as np


class Gauss(DynMap):
    params = {
            "a": 4.9,
            "b": -0.5
        }
    n_dim = 1
    @staticjit
    def _rhs(X, a, b):
        x = X
        return np.exp(-a * x**2) + b


class DeJong(DynMap):
    params = {
            "a": 1.641,
            "b": 1.902, 
            "c": 0.316, 
            "d": 1.525
        }
    n_dim = 2
    @staticjit
    def _rhs(X, a, b, c, d):
        x, y = X
        xp = np.sin(a * y) - np.cos(b * x)
        yp = np.sin(c * x) - np.cos(d * y)
        return xp, yp


class KaplanYorke(DynMap):
    params = {
            "alpha": 0.2
        }
    n_dim = 2
    @staticjit
    def _rhs(X, alpha):
        x, y = X
        xp = (2 * x) % 0.99999995
        yp = alpha * y + np.cos(4 * np.pi * x)
        return xp, yp

