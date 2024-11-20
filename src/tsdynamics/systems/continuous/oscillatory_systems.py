from tsdynamics.base import DynSys
from tsdynamics.utils import staticjit
import numpy as np



class ShimizuMorioka(DynSys):
    params = {
      "a": 0.85,
      "b": 0.5
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b):
        x, y, z = X
        xdot = y
        ydot = x - a * y - x * z
        zdot = -b * z + x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b):
        x, y, z = X
        row1 = [0, 1, 0]
        row2 = [1 - z, -a, -x]
        row3 = [2 * x, 0, -b]
        return row1, row2, row3


class GenesioTesi(DynSys):
    params = {
      "a": 0.44,
      "b": 1.1,
      "c": 1
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c):
        x, y, z = X
        xdot = y
        ydot = z
        zdot = -c * x - b * y - a * z + x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c):
        x, y, z = X
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-c + 2 * x, -b, -a]
        return row1, row2, row3


class MooreSpiegel(DynSys):
    params = {
      "a": 10,
      "b": 4,
      "eps": 9
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, eps):
        x, y, z = X
        xdot = y
        ydot = a * z
        zdot = -z + eps * y - y * x**2 - b * x
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, eps):
        x, y, z = X
        row1 = [0, 1, 0]
        row2 = [0, 0, a]
        row3 = [-2 * x * y - b, eps - x**2, -1]
        return row1, row2, row3


class AnishchenkoAstakhov(DynSys):
    params = {
      "eta": 0.5,
      "mu": 1.2
    }
    n_dim = 3
    @staticjit
    def rhs(X, t, mu, eta):
        x, y, z = X
        xdot = mu * x + y - x * z
        ydot = -x
        zdot = -eta * z + eta * np.heaviside(x, 0) * x**2
        return xdot, ydot, zdot


class Aizawa(DynSys):
    params = {
      "a": 0.95,
      "b": 0.7,
      "c": 0.6,
      "d": 3.5,
      "e": 0.25,
      "f": 0.1
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c, d, e, f):
        x, y, z = X
        xdot = x * z - b * x - d * y
        ydot = d * x + y * z - b * y
        zdot = (
            c
            + a * z
            - (z**3) / 3
            - x**2
            - y**2
            - e * z * x**2
            - e * z * y**2
            + f * z * x**3
        )
        return xdot, ydot, zdot


class StickSlipOscillator(DynSys):
    params = {
      "a": 1,
      "alpha": 0.3,
      "b": 1,
      "beta": 0.3,
      "eps": 0.05,
      "gamma": 1.0,
      "t0": 0.3,
      "vs": 0.4,
      "w": 2
    }
    n_dim = 3
    def _t(self, v):
        return self.t0 * np.sign(v) - self.alpha * v + self.beta * v**3

    @staticjit
    def _rhs(X, t, a, alpha, b, beta, eps, gamma, t0, vs, w):
        x, v, th = X
        tq = t0 * np.sign(v - vs) - alpha * v + beta * (v - vs) ** 3
        xdot = v
        vdot = eps * (gamma * np.cos(th) - tq) + a * x - b * x**3
        thdot = w
        return xdot, vdot, thdot

    @staticjit
    def _postprocessing(x, v, th):
        return x, v, np.cos(th)


class Torus(DynSys):
    params = {
      "a": 0.5,
      "n": 15.3,
      "r": 1
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, n, r):
        x, y, z = X
        xdot = (-a * n * np.sin(n * t)) * np.cos(t) - (r + a * np.cos(n * t)) * np.sin(
            t
        )
        ydot = (-a * n * np.sin(n * t)) * np.sin(t) + (r + a * np.cos(n * t)) * np.cos(
            t
        )
        zdot = a * n * np.cos(n * t)
        return xdot, ydot, zdot

