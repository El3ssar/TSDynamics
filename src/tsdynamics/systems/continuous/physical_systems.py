from tsdynamics.base import DynSys
import numpy as np
from symengine import cos, sin, exp, tanh, pi


class DoublePendulum(DynSys):
    params = {
      "d": 1.0,
      "m": 1.0
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, d, m):
        th1, th2, p1, p2 = Y(0), Y(1), Y(2), Y(3)
        g = 9.82
        pre = 6 / (m * d**2)
        denom = 16 - 9 * cos(th1 - th2) ** 2
        th1_dot = pre * (2 * p1 - 3 * cos(th1 - th2) * p2) / denom
        th2_dot = pre * (8 * p2 - 3 * cos(th1 - th2) * p1) / denom
        p1_dot = (
            -0.5
            * (m * d**2)
            * (th1_dot * th2_dot * sin(th1 - th2) + 3 * (g / d) * sin(th1))
        )
        p2_dot = (
            -0.5
            * (m * d**2)
            * (-th1_dot * th2_dot * sin(th1 - th2) + 3 * (g / d) * sin(th2))
        )
        return th1_dot, th2_dot, p1_dot, p2_dot

    @staticmethod
    def _postprocessing(th1, th2, p1, p2):
        return sin(th1), sin(th2), p1, p2


class SwingingAtwood(DynSys):
    params = {
      "m1": 1.0,
      "m2": 4.5
    }
    n_dim = 4
    @staticmethod
    def _rhs(Y, t, m1, m2):
        r, th, pr, pth = Y(0), Y(1), Y(2), Y(3)
        g = 9.82
        rdot = pr / (m1 + m2)
        thdot = pth / (m1 * r**2)
        prdot = pth**2 / (m1 * r**3) - m2 * g + m1 * g * cos(th)
        pthdot = -m1 * g * r * sin(th)
        return rdot, thdot, prdot, pthdot

    @staticmethod
    def _postprocessing(r, th, pr, pth):
        return r, sin(th), pr, pth


class Colpitts(DynSys):
    params = {
      "a": 30,
      "b": 0.8,
      "c": 20,
      "d": 0.08,
      "e": 10
    }
    n_dim = 3

    @staticmethod
    def _rhs(Y, t, a, b, c, d, e):
        x, y, z = Y(0), Y(1), Y(2)
        u = z - (e - 1)
        fz = -u * (1 - np.heaviside(u, 0))
        xdot = y - a * fz
        ydot = c - x - b * y - z
        zdot = y - d * z
        return (xdot, ydot, zdot)


class Laser(DynSys):
    params = {
      "a": 10.0,
      "b": 1.0,
      "c": 5.0,
      "d": -1.0,
      "h": -5.0,
      "k": -6.0
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, b, c, d, h, k):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * y - a * x + b * y * z**2
        ydot = c * x + d * x * z**2
        zdot = h * z + k * x**2
        return xdot, ydot, zdot

    @staticmethod
    def _jac(Y, t, a, b, c, d, h, k):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, a + b * z**2, 2 * b * y * z]
        row2 = [c + d * z**2, 0, 2 * d * x * z]
        row3 = [2 * k * x, 0, h]
        return row1, row2, row3


class Blasius(DynSys):
    params = {
      "a": 1,
      "alpha1": 0.2,
      "alpha2": 1,
      "b": 1,
      "c": 10,
      "k1": 0.05,
      "k2": 0,
      "zs": 0.006
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, alpha1, alpha2, b, c, k1, k2, zs):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * x - alpha1 * x * y / (1 + k1 * x)
        ydot = -b * y + alpha1 * x * y / (1 + k1 * x) - alpha2 * y * z / (1 + k2 * y)
        zdot = -c * (z - zs) + alpha2 * y * z / (1 + k2 * y)
        return xdot, ydot, zdot


class FluidTrampoline(DynSys):
    params = {
      "gamma": 1.82,
      "psi": 0.01019,
      "w": 1.21
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, gamma, psi, w):
        x, y, th = Y(0), Y(1), Y(2)
        xdot = y
        ydot = -1 - np.heaviside(-x, 0) * (x + psi * y * abs(y)) + gamma * cos(th)
        thdot = w
        return (xdot, ydot, thdot)

    @staticmethod
    def _postprocessing(x, y, th):
        return x, y, cos(th)


class JerkCircuit(DynSys):
    params = {
      "eps": 1e-9,
      "y0": 0.026
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, eps, y0):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = z
        zdot = -z - x - eps * (exp(y / y0) - 1)
        return xdot, ydot, zdot

    @staticmethod
    def _jac(Y, t, eps, y0):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-1, -eps * exp(y / y0) / y0, -1]
        return row1, row2, row3


class InteriorSquirmer(DynSys):
    params = {
      "a": [0.5, 0.5, 0.5, 0.5, 0.5],
      "g": [0.5, 0.5, 0.5, 0.5, 0.5],
      "n": 5,
      "tau": 3
    }
    n_dim = 2
    @staticmethod
    def _rhs_static(r, th, t, a, g, n):
        nvals = np.arange(1, n + 1)
        sinvals, cosvals = sin(th * nvals), cos(th * nvals)
        rnvals = r**nvals

        vrn = g * cosvals + a * sinvals
        vrn *= (nvals * rnvals * (r**2 - 1)) / r

        vth = 2 * r + (r**2 - 1) * nvals / r
        vth *= a * cosvals - g * sinvals
        vth *= rnvals

        return np.sum(vrn), np.sum(vth) / r

    @staticmethod
    def _jac_static(r, th, t, a, g, n):
        nvals = np.arange(1, n + 1)
        sinvals, cosvals = sin(th * nvals), cos(th * nvals)
        rnvals = r**nvals
        trigsum = a * sinvals + g * cosvals
        trigskew = a * cosvals - g * sinvals

        j11 = np.copy(trigsum)
        j11 *= nvals * rnvals * (2 * r**2 + (r**2 - 1) * (nvals - 1))
        j11 = (1 / r**2) * np.sum(j11)

        j12 = np.copy(trigskew)
        j12 *= -(nvals**2) * rnvals * (1 - r**2) / r
        j12 = np.sum(j12)

        j21 = 2 * rnvals * (2 * nvals + 1) * (-np.copy(trigskew))
        j21 += (n * (1 - r**2) * rnvals * (nvals - 1) / r**2) * np.copy(
            g * sinvals + a * cosvals
        )
        j21 = -np.sum(j21)

        j22 = np.copy(trigsum)
        j22 *= -nvals * rnvals * (2 * r + (r**2 - 1) * nvals / r)
        j22 = np.sum(j22)
        # (1 / r**2) *

        ## Correct for polar coordinates
        vth = np.copy(trigskew)
        vth *= 2 * r + (r**2 - 1) * nvals / r
        vth *= rnvals
        vth = np.sum(vth) / r
        j21 = j21 / r - vth / r
        j22 /= r

        return np.array([[j11, j12], [j21, j22]])

    @staticmethod
    def _protocol(t, tau, stiffness=20):
        return 0.5 + 0.5 * tanh(tau * stiffness * sin(2 * pi * t / tau))


    @staticmethod
    def _rhs(Y, t, a, g, n, tau):
        r, th, tt = Y(0), Y(1), Y(2)
        phase = InteriorSquirmer._protocol(tt, tau)
        dtt = 1
        dr, dth = InteriorSquirmer._rhs_static(
            r, th, t, a * phase, g * (1 - phase), n
        )
        return dr, dth, dtt


class WindmiReduced(DynSys):
    params = {
      "a1": 0.247,
      "b1": 10.8,
      "b2": 0.0752,
      "b3": 1.06,
      "d1": 2200,
      "vsw": 5
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a1, b1, b2, b3, d1, vsw):
        i, v, p = Y(0), Y(1), Y(2)
        idot = a1 * (vsw - v)
        vdot = b1 * i - b2 * p**1 / 2 - b3 * v
        pdot = vsw**2 - p ** (5 / 4) * vsw ** (1 / 2) * (1 + tanh(d1 * (i - 1))) / 2
        return idot, vdot, pdot
