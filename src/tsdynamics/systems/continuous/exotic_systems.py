from tsdynamics.base import DynSys
from symengine import sqrt, cos, sin


class NuclearQuadrupole(DynSys):
    params = {
      "a": 1.0,
      "b": 0.55,
      "d": 0.4
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, a, b, d):
        q1, q2, p1, p2 = Y(0), Y(1), Y(2), Y(3)
        q1dot = a * p1
        q2dot = a * p2
        p1dot = (
            -a * q1
            + 3 / sqrt(2) * b * q1**2
            - 3 / sqrt(2) * b * q2**2
            - d * q1**3
            - d * q1 * q2**2
        )
        p2dot = -a * q2 - 3 * sqrt(2) * b * q1 * q2 - d * q2 * q1**2 - d * q2**3
        return q1dot, q2dot, p1dot, p2dot


class HyperCai(DynSys):
    params = {
      "a": 27.5,
      "b": 3,
      "c": 19.3,
      "d": 2.9,
      "e": 3.3
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, a, b, c, d, e):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x
        ydot = b * x + c * y - x * z + w
        zdot = -d * z + y**2
        wdot = -e * x
        return xdot, ydot, zdot, wdot


class HyperBao(DynSys):
    params = {
      "a": 36,
      "b": 3,
      "c": 20,
      "d": 0.1,
      "e": 21
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, a, b, c, d, e):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + w
        ydot = c * y - x * z
        zdot = x * y - b * z
        wdot = e * x + d * y * z
        return xdot, ydot, zdot, wdot


class HyperJha(DynSys):
    params = {
      "a": 10,
      "b": 28,
      "c": 2.667,
      "d": 1.3
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + w
        ydot = -x * z + b * x - y
        zdot = x * y - c * z
        wdot = -x * z + d * w
        return xdot, ydot, zdot, wdot


class HyperQi(DynSys):
    params = {
      "a": 50,
      "b": 24,
      "c": 13,
      "d": 8,
      "e": 33,
      "f": 30
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, a, b, c, d, e, f):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + y * z
        ydot = b * x + b * y - x * z
        zdot = -c * z - e * w + x * y
        wdot = -d * w + f * z + x * y
        return xdot, ydot, zdot, wdot


class HyperXu(DynSys):
    params = {
      "a": 10,
      "b": 40,
      "c": 2.5,
      "d": 2,
      "e": 16
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, a=10, b=40, c=2.5, d=2, e=16):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + w
        ydot = b * x + e * x * z
        zdot = -c * z - x * y
        wdot = x * z - d * y
        return xdot, ydot, zdot, wdot


class HyperWang(DynSys):
    params = {
      "a": 10,
      "b": 40,
      "c": 2.5,
      "d": 10.6,
      "e": 4
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, a=10, b=40, c=2.5, d=10.6, e=4):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x
        ydot = -x * z + b * x + w
        zdot = -c * z + e * x**2
        wdot = -d * x
        return xdot, ydot, zdot, wdot


class HyperPang(DynSys):
    params = {
      "a": 36,
      "b": 3,
      "c": 20,
      "d": 2
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, a=36, b=3, c=20, d=2):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x
        ydot = -x * z + c * y + w
        zdot = x * y - b * z
        wdot = -d * x - d * y
        return xdot, ydot, zdot, wdot


class HyperLu(DynSys):
    params = {
      "a": 36,
      "b": 3,
      "c": 20,
      "d": 1.3
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, a=36, b=3, c=20, d=1.3):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + w
        ydot = -x * z + c * y
        zdot = x * y - b * z
        wdot = d * w + x * z
        return xdot, ydot, zdot, wdot


class LorenzStenflo(DynSys):
    params = {
      "a": 2,
      "b": 0.7,
      "c": 26,
      "d": 1.5
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + d * w
        ydot = c * x - x * z - y
        zdot = x * y - b * z
        wdot = -x - a * w
        return xdot, ydot, zdot, wdot


class Qi(DynSys):
    params = {
      "a": 45,
      "b": 10,
      "c": 1,
      "d": 10
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + y * z * w
        ydot = b * x + b * y - x * z * w
        zdot = -c * z + x * y * w
        wdot = -d * w + x * y * z
        return xdot, ydot, zdot, wdot


class ArnoldWeb(DynSys):
    params = {
      "mu": 0.01,
      "w": 1
    }
    n_dim = 5
    @staticmethod
    def _rhs(Y, t, mu, w):
        p1, p2, x1, x2, z = Y(0), Y(1), Y(2), Y(3), Y(4)
        denom = 4 + cos(z) + cos(x1) + cos(x2)
        p1dot = -mu * sin(x1) / denom**2
        p2dot = -mu * sin(x2) / denom**2
        x1dot = p1
        x2dot = p2
        zdot = w
        return p1dot, p2dot, x1dot, x2dot, zdot

    @staticmethod
    def _postprocessing(p1, p2, x1, x2, z):
        return p1, p2, sin(x1), sin(x2), cos(z)


class NewtonLiepnik(DynSys):
    params = {
      "a": 0.4,
      "b": 0.175
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -a * x + y + 10 * y * z
        ydot = -x - 0.4 * y + 5 * x * z
        zdot = b * z - 5 * x * y
        return xdot, ydot, zdot


class Robinson(DynSys):
    params = {"a": 0.5, "b": 0.2, "c": 0.1, "d": 0.3, "v": 0.05}
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, b, c, d, v):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = x - 2 * x**3 - a * y + b * x**2 * y - v * y * z
        zdot = -c * z + d * x**2
        return (xdot, ydot, zdot)

