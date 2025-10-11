from tsdynamics.base import DynMap
from tsdynamics.utils import staticjit
import numpy as np


class Henon(DynMap):
    params = {
            "a": 1.4,
            "b": 0.3
        }
    n_dim = 1
    @staticjit
    def _rhs(X, a, b):
        x = X
        xp = 1 - a * x**2 + b * x
        return xp

    @staticjit
    def _jac(X, a, b):
        x = X
        row1 = [-2 * a * x + b]
        return row1


class Ulam(DynMap):
    params = {
        "a": 1.0,
        "b": 2.0
    }
    n_dim = 1
    @staticjit
    def _rhs(X, a, b):
        x = X
        return a - b * x**2

    @staticjit
    def _jac(X, a, b):
        x = X
        return [-2 * b * x]

class Ikeda(DynMap):
    params = {
            "a": 0.4,
            "b": 6.0,
            "u": 0.9
        }
    n_dim = 2
    @staticjit
    def _rhs(X, a, b, u):
        x, y = X
        t = a - b / (1 + x**2 + y**2)
        xp = 1 + u * (x * np.cos(t) - y * np.sin(t))
        yp = u * (x * np.sin(t) + y * np.cos(t))
        return xp, yp

    @staticjit
    def _jac(X, a, b, u):
        x, y = X
        t = a - b / (1 + x**2 + y**2)
        dtdx = -12 * x / (1 + x**2 + y**2)**2
        dtdy = -12 * y / (1 + x**2 + y**2)**2

        dxpdx = u * np.cos(t) - u * x * np.sin(t) * dtdx - u * y * np.cos(t) * dtdx
        dxpdy = -u * x * np.sin(t) * dtdy - u * y * np.cos(t) * dtdy - u * np.sin(t)

        dypdx = u * np.sin(t) + u * x * np.cos(t) * dtdx - u * y * np.sin(t) * dtdx
        dypdy = u * x * np.cos(t) * dtdy + u * np.cos(t) - u * y * np.sin(t) * dtdy

        row1 = [dxpdx, dxpdy]
        row2 = [dypdx, dypdy]

        return row1, row2


class Tinkerbell(DynMap):
    params = {
            "a": 0.9,
            "b": -0.6013,
            "c": 2.0,
            "d": 0.5
        }
    # params = {
    #         "a": 0.3,
    #         "b": 0.6,
    #         "c": 2.0,
    #         "d": 0.27
    #     }


    n_dim = 2
    @staticjit
    def _rhs(X, a, b, c, d):
        x, y = X
        xp = x**2 - y**2 + a * x + b * y
        yp = 2 * x * y + c * x + d * y
        return xp, yp

    @staticjit
    def _jac(X, a, b, c, d):
        x, y = X
        row1 = [2 * x + a, -2 * y + b]
        row2 = [2 * y + c, 2 * x + d]
        return row1, row2


class Gingerbreadman(DynMap):
    params = {}
    n_dim = 2
    @staticjit
    def _rhs(X):
        x, y = X
        xp = 1 - y + np.abs(x)
        yp = x
        return xp, yp

    @staticjit
    def _jac(X):
        x, y = X
        row1 = [np.sign(x), -1]
        row2 = [1, 0]
        return row1, row2


class Zaslavskii(DynMap):
    params = {
            "eps": 5.0,
            "nu": 0.2,
            "r": 2.0
        }
    n_dim = 2
    @staticjit
    def _rhs(X, eps, nu, r):
        x, y = X
        mu = (1 - np.exp(-r)) / r
        xp = x + nu * (1 + mu * y) + eps * nu * mu * np.cos(2 * np.pi * x)
        xp = xp % 0.99999995
        yp = np.exp(-r) * (y + eps * np.cos(2 * np.pi * x))
        return xp, yp

    @staticjit
    def _jac(X, eps, nu, r):
        x, y = X
        mu = (1 - np.exp(-r)) / r

        dxpdx = nu * (1 + mu * y) - 2 * np.pi * eps * nu * mu * np.sin(2 * np.pi * x)
        dxpdy = nu * mu

        dypdx = -2 * np.pi * eps * np.exp(-r) * np.sin(2 * np.pi * x)
        dypdy = np.exp(-r)

        row1 = [dxpdx, dxpdy]
        row2 = [dypdx, dypdy]
        return row1, row2


class Chirikov(DynMap):
    params = {
            "k": 0.971635
        }
    n_dim = 2
    @staticjit
    def _rhs(X, k):
        p, x = X
        pp = p + k * np.sin(x)
        xp = x + pp
        return pp, xp

    @staticjit
    def _jac(X, k):
        p, x = X
        row1 = [1, k * np.cos(x)]
        row2 = [1, 1]
        return row1, row2

# Hyperchaotic maps
class FoldedTowel(DynMap):
    params = {
            "a": 3.8,
            "b": 0.05,
            "c": 0.35,
            "d": 0.1,
            "e": 1.9,
            "f": 3.78,
            "g": 0.2
        }
    n_dim = 3
    @staticjit
    def _rhs(X, a, b, c, d, e, f, g):
        x, y, z = X
        xp = a * x * (1 - x) - b * (y + c) * (1 - 2 * z)
        yp = d * ((y + c) * (1 + 2 * z) - 1) * (1 - e * x)
        zp = f * z * (1 - z) + g * y
        return xp, yp, zp

    @staticjit
    def _jac(X, a, b, c, d, e, f, g):
        x, y, z = X
        row1 = [
                a * (1 - 2 * x),
                -b * (1 - 2 * z),
                2 * b * (y - c)
                ]

        row2 = [
                -d * e * ((y - c) * (1 + 2 * z) - 1),
                d * (1 + 2 * z) * (1 - e * x),
                2 * d * (y - c) * (1 - e * x)
                ]

        row3 = [0,
                g,
                f * (1 - 2 * z)
                ]

        return row1, row2, row3

class GeneralizedHenon(DynMap):
    params = {
            "a": 1.9,
            "b": 0.03,
        }
    n_dim = 3
    @staticjit
    def _rhs(X, a, b):
        x, y, z = X
        xp = a - y**2 - b * z
        yp = x
        zp = y
        return xp, yp, zp

    @staticjit
    def _jac(X, a, b):
        x, y, z = X
        row1 = [0, -2 * y, b]
        row2 = [1, 0, 0]
        row3 = [0, 1, 0]

        return row1, row2, row3