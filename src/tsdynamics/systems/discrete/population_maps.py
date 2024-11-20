from tsdynamics.base import DynMap
from tsdynamics.utils import staticjit
import numpy as np


class Logistic(DynMap):
    params = {
        "r": 3.9
        }
    n_dim = 1
    @staticjit
    def _rhs(x, r):
        return r * x * (1 - x)


class Ricker(DynMap):
    params = {
        "a": 3.3
        }
    @staticjit
    def _rhs(x, a):
        return x * np.exp(a - x)


class MaynardSmith(DynMap):
    params = {
            "a": 0.87,
            "b": 0.75
        }
    n_dim = 2
    @staticjit
    def _rhs(x, y, a, b):
        xp = y
        yp = a * y + b - x**2
        return xp, yp

