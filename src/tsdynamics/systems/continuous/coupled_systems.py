from tsdynamics.base import DynSys
from tsdynamics.utils import staticjit
import numpy as np


class Sakarya(DynSys):
    params = {
      "a": -1.0,
      "b": 1.0,
      "c": 1.0,
      "h": 1.0,
      "p": 1.0,
      "q": 0.4,
      "r": 0.3,
      "s": 1.0
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c, h, p, q, r, s):
        x, y, z = X
        xdot = a * x + h * y + s * y * z
        ydot = -b * y - p * x + q * x * z
        zdot = c * z - r * x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c, h, p, q, r, s):
        x, y, z = X
        row1 = [a, h + s * z, s * y]
        row2 = [-p + q * z, -b, q * x]
        row3 = [-r * y, -r * x, c]
        return row1, row2, row3


class Bouali2(DynSys):
    params = {
      "a": 1.0,
      "b": -0.3,
      "bb": 1.0,
      "c": 0.05,
      "g": 1.0,
      "m": 1,
      "y0": 4.0
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, bb, c, g, m, y0):
        x, y, z = X
        xdot = a * y0 * x - a * x * y - b * z
        ydot = -g * y + g * y * x**2
        zdot = -1.5 * m * x + m * bb * x * z - c * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, bb, c, g, m, y0):
        x, y, z = X
        row1 = [a * y0 - a * y, -a * x, -b]
        row2 = [2 * g * y * x, g * x ** 2 - g, 0]
        row3 = [-1.5 * m + m * bb * z, 0, m * bb * x - c ]
        return row1, row2, row3


class LuChenCheng(DynSys):
    params = {
      "a": -10,
      "b": -4,
      "c": 18.1
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c):
        x, y, z = X
        xdot = -(a * b) / (a + b) * x - y * z + c
        ydot = a * y + x * z
        zdot = b * z + x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c):
        x, y, z = X
        row1 = [-(a * b) / (a + b), -z, -y]
        row2 = [z, a, x]
        row3 = [y, x, b]
        return row1, row2, row3


class LuChen(DynSys):
    params = {
      "a": 36,
      "b": 3,
      "c": 18
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c):
        x, y, z = X
        xdot = a * y - a * x
        ydot = -x * z + c * y
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c):
        x, y, z = X
        row1 = [-a, a, 0]
        row2 = [-z, c, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class QiChen(DynSys):
    params = {
      "a": 38,
      "b": 2.666,
      "c": 80
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c):
        x, y, z = X
        xdot = a * y - a * x + y * z
        ydot = c * x + y - x * z
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c):
        x, y, z = X
        row1 = [-a, a + z, y]
        row2 = [c - z, 1, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class ZhouChen(DynSys):
    params = {
      "a": 2.97,
      "b": 0.15,
      "c": -3.0,
      "d": 1,
      "e": -8.78
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c, d, e):
        x, y, z = X
        xdot = a * x + b * y + y * z
        ydot = c * y - x * z + d * y * z
        zdot = e * z - x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c, d, e):
        x, y, z = X
        row1 = [a, b + z, y]
        row2 = [-z, c + d * z, -x + d * y]
        row3 = [-y, -x, e]
        return row1, row2, row3


class BurkeShaw(DynSys):
    params = {
      "e": 13,
      "n": 10
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, e, n):
        x, y, z = X
        xdot = -n * x - n * y
        ydot = y - n * x * z
        zdot = n * x * y + e
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, e, n):
        x, y, z = X
        row1 = [-n, -n, 0]
        row2 = [-n * z, 1, -n * x]
        row3 = [n * y, n * x, 0]
        return row1, row2, row3


class Chen(DynSys):
    params = {
      "a": 35,
      "b": 3,
      "c": 28
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c):
        x, y, z = X
        xdot = a * y - a * x
        ydot = (c - a) * x - x * z + c * y
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c):
        x, y, z = X
        row1 = [-a, a, 0]
        row2 = [c - a - z, c, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class ChenLee(DynSys):
    params = {
      "a": 5,
      "b": -10,
      "c": -0.38
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c):
        x, y, z = X
        xdot = a * x - y * z
        ydot = b * y + x * z
        zdot = c * z + x * y / 3
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c):
        x, y, z = X
        row1 = [a, -z, -y]
        row2 = [z, b, x]
        row3 = [y / 3, x / 3, c]
        return row1, row2, row3


class WangSun(DynSys):
    params = {
      "a": 0.2,
      "b": -0.01,
      "d": -0.4,
      "e": -1.0,
      "f": -1.0,
      "q": 1.0
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, d, e, f, q):
        x, y, z = X
        xdot = a * x + q * y * z
        ydot = b * x + d * y - x * z
        zdot = e * z + f * x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, d, e, f, q):
        x, y, z = X
        row1 = [a, q * z, q * y]
        row2 = [b - z, d, -x]
        row3 = [f * y, f * x, e]
        return row1, row2, row3


class YuWang(DynSys):
    params = {
      "a": 10,
      "b": 40,
      "c": 2,
      "d": 2.5
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c, d):
        x, y, z = X
        xdot = a * (y - x)
        ydot = b * x - c * x * z
        zdot = np.exp(x * y) - d * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c, d):
        x, y, z = X
        row1 = [-a, a, 0]
        row2 = [b - c * z, 0, -c * x]
        row3 = [y * np.exp(x * y), x * np.exp(x * y), -d]
        return row1, row2, row3


class YuWang2(DynSys):
    params = {
      "a": 10,
      "b": 30,
      "c": 2,
      "d": 2.5
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c, d):
        x, y, z = X
        xdot = a * (y - x)
        ydot = b * x - c * x * z
        zdot = np.cosh(x * y) - d * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c, d):
        x, y, z = X
        row1 = [-a, a, 0]
        row2 = [b - c * z, 0, -c * x]
        row3 = [y * np.sinh(x * y), x * np.sinh(x * y), -d]
        return row1, row2, row3


class SanUmSrisuchinwong(DynSys):
    params = {
      "a": 2
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a):
        x, y, z = X
        xdot = y - x
        ydot = -z * np.tanh(x)
        zdot = -a + x * y + np.abs(y)
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a):
        x, y, z = X
        row1 = [-1, 1, 0]
        row2 = [-z * (1 - np.tanh(x) ** 2), 0, -np.tanh(x)]
        row3 = [y, x + np.sign(y), 0]
        return row1, row2, row3


class DequanLi(DynSys):
    params = {
      "a": 40,
      "c": 1.833,
      "d": 0.16,
      "eps": 0.65,
      "f": 20,
      "k": 55
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, c, d, eps, f, k):
        x, y, z = X
        xdot = a * y - a * x + d * x * z
        ydot = k * x + f * y - x * z
        zdot = c * z + x * y - eps * x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, c, d, eps, f, k):
        x, y, z = X
        row1 = [-a + d * z, a, d * x]
        row2 = [k - z, f, -x]
        row3 = [y - 2 * eps * x, x, c]
        return row1, row2, row3

