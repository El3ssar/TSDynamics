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
    
    @staticjit
    def _jac(X, a, b):
        x = X
        return [-2 * a * x * np.exp(-a * x**2)]


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
    
    @staticjit
    def _jac(X, a, b, c, d):
        x, y = X
        row1 = [b * np.sin(b * x), a * np.cos(a * y)]
        row2 = [c * np.cos(c * x), d * np.sin(d * y)]
        return row1, row2


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

    @staticjit
    def _jac(X, alpha):
        x, y = X
        row1 = [2, 0]
        row2 = [-4 * np.pi * np.sin(4 * np.pi * x), alpha]
        return row1, row2
