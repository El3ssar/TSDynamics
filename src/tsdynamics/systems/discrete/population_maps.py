from tsdynamics.base import DynMap
from tsdynamics.utils import staticjit
import numpy as np


class Logistic(DynMap):
    params = {
        "r": 3.9
        }
    n_dim = 1
    @staticjit
    def _rhs(X, r):
        x = X
        return r * x * (1 - x)

    @staticjit
    def _jac(X, r):
        x = X
        return [r - 2 * r * x]


class Ricker(DynMap):
    params = {
        "a": 3.3
        }
    n_dim = 1
    @staticjit
    def _rhs(X, a):
        x = X
        return x * np.exp(a - x)

    @staticjit
    def _jac(X, a):
        x = X
        return [np.exp(a - x) - x * np.exp(a - x)]


class MaynardSmith(DynMap):
    params = {
            "a": 0.87,
            "b": 0.75
        }
    n_dim = 2
    @staticjit
    def _rhs(X, a, b):
        x, y = X
        xp = y
        yp = a * y + b - x**2
        return xp, yp

    @staticjit
    def _jac(X, a, b):
        x, y = X
        row1 = [0, 1]
        row2 = [-2 * x, a]
        return row1, row2

