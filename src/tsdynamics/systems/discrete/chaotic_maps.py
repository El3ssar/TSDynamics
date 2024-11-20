from tsdynamics.base import DynMap
from tsdynamics.utils import staticjit
import numpy as np


class Henon(DynMap):
    params = {
            "a": 1.4,
            "b": 0.3
        }
    n_dim = 2
    @staticjit
    def _rhs(x, y, a, b):
        xp = 1 - a * x**2 + y
        yp = b * x
        return xp, yp


class Ikeda(DynMap):
    params = {
            "u": 0.9
        }
    n_dim = 2
    @staticjit
    def _rhs(x, y, u):
        t = 0.4 - 6 / (1 + x**2 + y**2)
        xp = 1 + u * (x * np.cos(t) - y * np.sin(t))
        yp = u * (x * np.sin(t) + y * np.cos(t))
        return xp, yp


class Tinkerbell(DynMap):
    params = {
            "a": 0.9,
            "b": -0.6013,
            "c": 2.0,
            "d": 0.5
        }
    n_dim = 2
    @staticjit
    def _rhs(x, y, a, b, c, d):
        xp = x**2 - y**2 + a * x + b * y
        yp = 2 * x * y + c * x + d * y
        return xp, yp


class Gingerbreadman(DynMap):
    params = {}
    n_dim = 2
    @staticjit
    def _rhs(x, y):
        xp = 1 - y + np.abs(x)
        yp = x
        return xp, yp


class Zaslavskii(DynMap):
    params = {
            "eps": 5.0,
            "nu": 0.2,
            "r": 2.0
        }
    n_dim = 2
    @staticjit
    def _rhs(x, y, eps, nu, r):
        mu = (1 - np.exp(-r)) / r
        xp = x + nu * (1 + mu * y) + eps * nu * mu * np.cos(2 * np.pi * x)
        xp = xp % 0.99999995
        yp = np.exp(-r) * (y + eps * np.cos(2 * np.pi * x))
        return xp, yp


class Chirikov(DynMap):
    params = {
            "k": 0.971635
        }
    n_dim = 2
    @staticjit
    def _rhs(p, x, k):
        pp = p + k * np.sin(x)
        xp = x + pp
        return pp, xp

    @staticjit
    def _rhs_inv(pp, xp, k):
        x = xp - pp
        p = pp - k * np.sin(xp - pp)
        return p, x

