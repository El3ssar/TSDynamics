from tsdynamics.base import DynSys
# import numpy as np
from symengine import sin, cos, exp, pi



class GlycolyticOscillation(DynSys):
    params = {
      "d": 0.0,
      "k": 4.422,
      "l1": 500000000.0,
      "l2": 100,
      "nu": 1.0,
      "q1": 50,
      "q2": 0.02,
      "s1": 22.2222,
      "s2": 22.2222
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, d, k, l1, l2, nu, q1, q2, s1, s2):
        a, b, c = Y(0), Y(1), Y(2)
        phi = (a * (1 + a) * (1 + b) ** 2) / (l1 + (1 + a) ** 2 * (1 + b) ** 2)
        eta = b * (1 + d * b) * (1 + c) ** 2 / (l2 + (1 + d * b) ** 2 * (1 + c) ** 2)
        adot = nu - s1 * phi
        bdot = q1 * s1 * phi - s2 * eta
        cdot = q2 * s2 * eta - k * c
        return adot, bdot, cdot


class Oregonator(DynSys):
    params = {
        "q": 2e-4,
        "f": 1,
        "mu": 1e-6,
        "epsilon": 1e-2,
    }
    n_dim = 3  # Three variables: X, Y, Z (reduced forms of the chemical species)

    @staticmethod
    def _rhs(Y, t, q, f, mu, epsilon):
        """
        Right-hand side of the Belousov-Zhabotinsky (Oregonator) model.

        X: State vector [X, Y, Z]
        t: Time (not explicitly used in autonomous systems)
        q, f, epsilon: Model parameters
        """
        # Variables
        x, y, z = Y(0), Y(1), Y(2)

        # Oregonator equations
        xdot = y - x
        ydot = (q * z - z * y + y * (1 - y)) / epsilon
        zdot = (-q * z - y * z + f * x) / mu

        return xdot, ydot, zdot


class IsothermalChemical(DynSys):
    params = {
      "delta": 1.0,
      "kappa": 2.5,
      "mu": 0.29786,
      "sigma": 0.013
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, delta, kappa, mu, sigma):
        alpha, beta, gamma = Y(0), Y(1), Y(2)
        alphadot = mu * (kappa + gamma) - alpha * beta**2 - alpha
        betadot = (alpha * beta**2 + alpha - beta) / sigma
        gammadot = (beta - gamma) / delta
        return alphadot, betadot, gammadot


class ForcedBrusselator(DynSys):
    params = {
      "a": 0.4,
      "b": 1.2,
      "f": 0.05,
      "w": 0.81
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, b, f, w):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a + x**2 * y - (b + 1) * x + f * cos(z)
        ydot = b * x - x**2 * y
        zdot = w
        return xdot, ydot, zdot

    @staticmethod
    def _postprocessing(x, y, z):
        return x, y, sin(z)


class CircadianRhythm(DynSys):
    params = {
      "Ki": 1,
      "k": 0.5,
      "k1": 0.3,
      "k2": 0.15,
      "kd": 1.4,
      "kdn": 0.4,
      "km": 0.4,
      "ks": 1,
      "n": 4,
      "vd": 6,
      "vdn": 1.5,
      "vm": 0.7,
      "vmax": 4.7,
      "vmin": 1.0,
      "vs": 6
    }
    n_dim = 5
    @staticmethod
    def _rhs(
        Y,
        t,
        Ki,
        k,
        k1,
        k2,
        kd,
        kdn,
        km,
        ks,
        n,
        vd,
        vdn,
        vm,
        vmax,
        vmin,
        v,
    ):
        m, fc, fs, fn, th = Y(0), Y(1), Y(2), Y(3), Y(4)
        vs = 2.5 * ((0.5 + 0.5 * cos(th)) + vmin) * (vmax - vmin)
        mdot = vs * (Ki**n) / (Ki**n + fn**n) - vm * m / (km + m)
        fcdot = ks * m - k1 * fc + k2 * fn - k * fc
        fsdot = k * fc - vd * fs / (kd + fs)
        fndot = k1 * fc - k2 * fn - vdn * fn / (kdn + fn)
        thdot = 2 * pi / 24
        return mdot, fcdot, fsdot, fndot, thdot

    @staticmethod
    def _postprocessing(m, fc, fs, fn, th):
        return m, fc, fs, fn, cos(th)


class CaTwoPlus(DynSys):
    params = {
      "K2": 0.1,
      "K5": 0.3194,
      "Ka": 0.1,
      "Kd": 1,
      "Ky": 0.3,
      "Kz": 0.6,
      "V0": 2,
      "V1": 2,
      "V4": 3,
      "Vm2": 6,
      "Vm3": 30,
      "Vm5": 50,
      "beta": 0.65,
      "eps": 13,
      "k": 10,
      "kf": 1,
      "m": 2,
      "n": 4,
      "p": 1
    }
    n_dim = 3

    @staticmethod
    def _rhs(Y, t, K2, K5, Ka, Kd, Ky, Kz, V0, V1, V4, Vm2, Vm3, Vm5, beta, eps, k, kf, m, n, p):
        z, y, a = Y(0), Y(1), Y(2)
        Vin = V0 + V1 * beta
        V2 = Vm2 * (z**2) / (K2**2 + z**2)
        V3 = (
            (Vm3 * (z**m) / (Kz**m + z**m))
            * (y**2 / (Ky**2 + y**2))
            * (a**4 / (Ka**4 + a**4))
        )
        V5 = (
            Vm5
            * (a**p / (K5**p + a**p))
            * (z**n / (Kd**n + z**n))
        )
        zdot = Vin - V2 + V3 + kf * y - k * z
        ydot = V2 - V3 - kf * y
        adot = beta * V4 - V5 - eps * a
        return zdot, ydot, adot


class ExcitableCell(DynSys):
    params = {
      "gi": 1800,
      "gkc": 11,
      "gkv": 1700,
      "gl": 7,
      "kc": 0.183333,
      "rho": 0.27,
      "vc": 100,
      "vi": 100,
      "vk": -75,
      "vl": -40,
      "vm": -50,
      "vn": -30
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, gi, gkc, gkv, gl, kc, rho, vc, vi, vk, vl, vm, vn):
        v, n, c = Y(0), Y(1), Y(2)

        alpham = 0.1 * (25 + v) / (1 - exp(-0.1 * v - 2.5))
        betam = 4 * exp(-(v + 50) / 18)
        minf = alpham / (alpham + betam)

        alphah = 0.07 * exp(-0.05 * v - 2.5)
        betah = 1 / (1 + exp(-0.1 * v - 2))
        hinf = alphah / (alphah + betah)

        alphan = 0.01 * (20 + v) / (1 - exp(-0.1 * v - 2))
        betan = 0.125 * exp(-(v + 30) / 80)
        ninf = alphan / (alphan + betan)
        tau = 1 / (230 * (alphan + betan))

        ca = c / (1 + c)

        vdot = (
            gi * minf**3 * hinf * (vi - v)
            + gkv * n**4 * (vk - v)
            + gkc * ca * (vk - v)
            + gl * (vl - v)
        )
        ndot = (ninf - n) / tau
        cdot = rho * (minf**3 * hinf * (vc - v) - kc * c)
        return vdot, ndot, cdot


class CellCycle(DynSys):
    params = {
      "K": 0.01,
      "Kc": 0.5,
      "Kd1": 0.02,
      "Kim": 0.65,
      "V2": 0.15,
      "V4": 0.05,
      "Vm1": 0.3,
      "Vm3": 0.1,
      "kd1": 0.001,
      "vd": 0.025,
      "vi": 0.05
    }
    n_dim = 6

    @staticmethod
    def rhs(Y, t, K, Kc, Kd1, Kim, V2, V4, Vm1, Vm3, kd1, vd, vi):
        c1, m1, x1, c2, m2, x2 = Y(0), Y(1), Y(2), Y(3), Y(4), Y(5)
        Vm1, Um1 = 2 * [Vm1]
        vi1, vi2 = 2 * [vi]
        H1, H2, H3, H4 = 4 * [K]
        K1, K2, K3, K4 = 4 * [K]
        V2, U2 = 2 * [V2]
        Vm3, Um3 = 2 * [Vm3]
        V4, U4 = 2 * [V4]
        Kc1, Kc2 = 2 * [Kc]
        vd1, vd2 = 2 * [vd]
        Kd1, Kd2 = 2 * [Kd1]
        kd1, kd2 = 2 * [kd1]
        Kim1, Kim2 = 2 * [Kim]
        V1 = Vm1 * c1 / (Kc1 + c1)
        U1 = Um1 * c2 / (Kc2 + c2)
        V3 = m1 * Vm3
        U3 = m2 * Um3
        c1dot = vi1 * Kim1 / (Kim1 + m2) - vd1 * x1 * c1 / (Kd1 + c1) - kd1 * c1
        c2dot = vi2 * Kim2 / (Kim2 + m1) - vd2 * x2 * c2 / (Kd2 + c2) - kd2 * c2
        m1dot = V1 * (1 - m1) / (K1 + (1 - m1)) - V2 * m1 / (K2 + m1)
        m2dot = U1 * (1 - m2) / (H1 + (1 - m2)) - U2 * m2 / (H2 + m2)
        x1dot = V3 * (1 - x1) / (K3 + (1 - x1)) - V4 * x1 / (K4 + x1)
        x2dot = U3 * (1 - x2) / (H3 + (1 - x2)) - U4 * x2 / (H4 + x2)
        return c1dot, m1dot, x1dot, c2dot, m2dot, x2dot


class HindmarshRose(DynSys):
    params = {
      "a": 0.49,
      "b": 1.0,
      "c": 0.0322,
      "d": 1.0,
      "s": 1.0,
      "tx": 0.03,
      "tz": 0.8
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, b, c, d, s, tx, tz):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -x + 1 / tx * y - a / tx * x**3 + b / tx * x**2 + 1 / tx * z
        ydot = -a * x**3 - (d - b) * x**2 + z
        zdot = -s / tz * x - 1 / tz * z + c / tz
        return xdot, ydot, zdot

    @staticmethod
    def _jac(Y, t, a, b, c, d, s, tx, tz):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-1 / tx - 3 * a / tx * x ** 2 + 2 * b / tx * x, 1 / tx, 1 / tx]
        row2 = [-3 * a * x ** 2 - 2 * (d - b) * x, 0, 1]
        row3 = [-s / tz, 0, -1 / tz - c / tz]
        return row1, row2, row3


class ForcedVanDerPol(DynSys):
    params = {
      "a": 1.2,
      "mu": 8.53,
      "w": 0.63
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, mu, w):
        x, y, z = Y(0), Y(1), Y(2)
        ydot = mu * (1 - x**2) * y - x + a * sin(z)
        xdot = y
        zdot = w
        return xdot, ydot, zdot

    @staticmethod
    def _postprocessing(x, y, z):
        return x, y, sin(z)


class ForcedFitzHughNagumo(DynSys):
    params = {
      "a": 0.7,
      "b": 0.8,
      "curr": 0.965,
      "f": 0.4008225,
      "gamma": 0.08,
      "omega": 0.043650793650793655
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, b, curr, f, gamma, omega):
        v, w, z = Y(0), Y(1), Y(2)
        vdot = v - v**3 / 3 - w + curr + f * sin(z)
        wdot = gamma * (v + a - b * w)
        zdot = omega
        return vdot, wdot, zdot

    @staticmethod
    def _postprocessing(x, y, z):
        return x, y, sin(z)


class TurchinHanski(DynSys):
    params = {
      "a": 8,
      "d": 0.04,
      "e": 0.5,
      "g": 0.1,
      "h": 0.8,
      "r": 8.12,
      "s": 1.25
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, d, e, g, h, r, s):
        n, p, z = Y(0), Y(1), Y(2)
        ndot = (
            r * (1 - e * sin(z)) * n
            - r * (n**2)
            - g * (n**2) / (n**2 + h**2)
            - a * n * p / (n + d)
        )
        pdot = s * (1 - e * sin(z)) * p - s * (p**2) / n
        zdot = 2 * pi
        return ndot, pdot, zdot

    @staticmethod
    def _postprocessing(x, y, z):
        return x, y, sin(z)


class HastingsPowell(DynSys):
    params = {
      "a1": 5.0,
      "a2": 0.1,
      "b1": 3.0,
      "b2": 2.0,
      "d1": 0.4,
      "d2": 0.01
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a1, a2, b1, b2, d1, d2):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = x * (1 - x) - y * a1 * x / (1 + b1 * x)
        ydot = y * a1 * x / (1 + b1 * x) - z * a2 * y / (1 + b2 * y) - d1 * y
        zdot = z * a2 * y / (1 + b2 * y) - d2 * z
        return xdot, ydot, zdot


class ItikBanksTumor(DynSys):
    params = {
      "a12": 1,
      "a13": 2.5,
      "a21": 1.5,
      "a31": 0.2,
      "d3": 0.5,
      "k3": 1,
      "r2": 0.6,
      "r3": 4.5
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a12, a13, a21, a31, d3, k3, r2, r3):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = x * (1 - x) - a12 * x * y - a13 * x * z
        ydot = r2 * y * (1 - y) - a21 * x * y
        zdot = r3 * x * z / (x + k3) - a31 * x * z - d3 * z
        return xdot, ydot, zdot

