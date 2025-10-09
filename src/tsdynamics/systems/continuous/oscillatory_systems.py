from tsdynamics.base import DynSys
import numpy as np
from symengine import pi, sin, cos, sign



class ShimizuMorioka(DynSys):
    params = {
      "a": 0.85,
      "b": 0.5
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = x - a * y - x * z
        zdot = -b * z + x**2
        return xdot, ydot, zdot

    @staticmethod
    def _jac(Y, t, a, b):
        x, y, z = Y(0), Y(1), Y(2)
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
    @staticmethod
    def _rhs(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = z
        zdot = -c * x - b * y - a * z + x**2
        return xdot, ydot, zdot

    @staticmethod
    def _jac(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
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
    @staticmethod
    def _rhs(Y, t, a, b, eps):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = a * z
        zdot = -z + eps * y - y * x**2 - b * x
        return xdot, ydot, zdot

    @staticmethod
    def _jac(Y, t, a, b, eps):
        x, y, z = Y(0), Y(1), Y(2)
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
    @staticmethod
    def rhs(Y, t, mu, eta):
        x, y, z = Y(0), Y(1), Y(2)
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
    @staticmethod
    def _rhs(Y, t, a, b, c, d, e, f):
        x, y, z = Y(0), Y(1), Y(2)
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
        return self.t0 * sign(v) - self.alpha * v + self.beta * v**3

    @staticmethod
    def _rhs(Y, t, a, alpha, b, beta, eps, gamma, t0, vs, w):
        x, v, th = Y(0), Y(1), Y(2)
        tq = t0 * sign(v - vs) - alpha * v + beta * (v - vs) ** 3
        xdot = v
        vdot = eps * (gamma * cos(th) - tq) + a * x - b * x**3
        thdot = w
        return xdot, vdot, thdot

    @staticmethod
    def _postprocessing(x, v, th):
        return x, v, cos(th)


class Torus(DynSys):
    params = {
      "a": 0.5,
      "n": 15.3,
      "r": 1
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, n, r):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = (-a * n * sin(n * t)) * cos(t) - (r + a * cos(n * t)) * sin(
            t
        )
        ydot = (-a * n * sin(n * t)) * sin(t) + (r + a * cos(n * t)) * cos(
            t
        )
        zdot = a * n * cos(n * t)
        return xdot, ydot, zdot


class Lissajous3D(DynSys):
    params = {"A": 1, "B": 1, "C": 1, "a": 3, "b": 2, "c": 5, "delta_y": pi / 2, "delta_z": pi / 4}
    n_dim = 3

    @staticmethod
    def _rhs(Y, t, A, B, C, a, b, c, delta_y, delta_z):
        """
        RHS of the 3D Lissajous system.
        Parameters:
            X : ndarray
                Current state [x, y, z].
            t : float
                Current time.
            A, B, C : float
                Amplitudes along x, y, z axes.
            a, b, c : float
                Frequencies along x, y, z axes.
            delta_y, delta_z : float
                Phase shifts along y and z axes.
        Returns:
            Derivatives [dx/dt, dy/dt, dz/dt].
        """
        x, y, z = Y(0), Y(1), Y(2)
        dxdt = A * (-a * sin(a * t))
        dydt = B * (-b * sin(b * t + delta_y))
        dzdt = C * (-c * sin(c * t + delta_z))
        return dxdt, dydt, dzdt


class Lissajous2D(DynSys):
    params = {"A": 1, "B": 1, "a": 3, "b": 2, "delta": pi / 2}
    n_dim = 2

    @staticmethod
    def _rhs(Y, t, A, B, a, b, delta):
        """
        RHS of the 2D Lissajous system.
        Parameters:
            X : ndarray
                Current state [x, y].
            t : float
                Current time.
            A, B : float
                Amplitudes along x and y axes.
            a, b : float
                Frequencies along x and y axes.
            delta : float
                Phase shift along y axis.
        Returns:
            Derivatives [dx/dt, dy/dt].
        """
        x, y = Y(0), Y(1)
        dxdt = A * (-a * sin(a * t))
        dydt = B * (-b * sin(b * t + delta))
        return dxdt, dydt
