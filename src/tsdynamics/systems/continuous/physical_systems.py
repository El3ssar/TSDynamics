from symengine import cos, exp, sign, sin, tanh

from tsdynamics.base import ContinuousSystem


class DoublePendulum(ContinuousSystem):
    params = {"d": 1.0, "m": 1.0}
    dim = 4

    @staticmethod
    def _equations(Y, t, *, d, m):
        th1, th2, p1, p2 = Y(0), Y(1), Y(2), Y(3)
        g = 9.82
        pre = 6 / (m * d**2)
        denom = 16 - 9 * cos(th1 - th2) ** 2
        th1_dot = pre * (2 * p1 - 3 * cos(th1 - th2) * p2) / denom
        th2_dot = pre * (8 * p2 - 3 * cos(th1 - th2) * p1) / denom
        p1_dot = -0.5 * (m * d**2) * (th1_dot * th2_dot * sin(th1 - th2) + 3 * (g / d) * sin(th1))
        p2_dot = -0.5 * (m * d**2) * (-th1_dot * th2_dot * sin(th1 - th2) + 3 * (g / d) * sin(th2))
        return th1_dot, th2_dot, p1_dot, p2_dot


class SwingingAtwood(ContinuousSystem):
    params = {"m1": 1.0, "m2": 4.5}
    dim = 4

    @staticmethod
    def _equations(Y, t, *, m1, m2):
        r, th, pr, pth = Y(0), Y(1), Y(2), Y(3)
        g = 9.82
        rdot = pr / (m1 + m2)
        thdot = pth / (m1 * r**2)
        prdot = pth**2 / (m1 * r**3) - m2 * g + m1 * g * cos(th)
        pthdot = -m1 * g * r * sin(th)
        return rdot, thdot, prdot, pthdot


class Colpitts(ContinuousSystem):
    params = {"a": 30, "b": 0.8, "c": 20, "d": 0.08, "e": 10}
    dim = 3

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, e):
        x, y, z = Y(0), Y(1), Y(2)
        u = z - (e - 1)
        fz = -u * (1 - sign(u)) / 2
        xdot = y - a * fz
        ydot = c - x - b * y - z
        zdot = y - d * z
        return (xdot, ydot, zdot)


class Laser(ContinuousSystem):
    params = {"a": 10.0, "b": 1.0, "c": 5.0, "d": -1.0, "h": -5.0, "k": -6.0}
    dim = 3

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, h, k):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * y - a * x + b * y * z**2
        ydot = c * x + d * x * z**2
        zdot = h * z + k * x**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c, d, h, k):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, a + b * z**2, 2 * b * y * z]
        row2 = [c + d * z**2, 0, 2 * d * x * z]
        row3 = [2 * k * x, 0, h]
        return row1, row2, row3


class Blasius(ContinuousSystem):
    """Three-level food chain with Holling type-II functional responses.

    The chaotic "teacup" attractor lives on a finite basin around a coexistence
    equilibrium near ``x ≈ 20`` — which is also the prey-runaway threshold of
    the type-II response, where predation saturates below the resource's growth
    rate. Random ``U[0, 1)^3`` ICs (small ``y``, sizeable ``z``) fall outside
    the basin and ``x`` blows up exponentially, so a known on-attractor point is
    set as ``default_ic`` to keep default integration bounded.
    """

    reference = "Blasius, Huppert & Stone (1999), Nature 399, 354-359"
    params = {
        "a": 1,
        "alpha1": 0.2,
        "alpha2": 1,
        "b": 1,
        "c": 10,
        "k1": 0.05,
        "k2": 0,
        "zs": 0.006,
    }
    dim = 3
    default_ic = [4.031713, 5.1113788, 0.016508812]

    @staticmethod
    def _equations(Y, t, *, a, alpha1, alpha2, b, c, k1, k2, zs):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * x - alpha1 * x * y / (1 + k1 * x)
        ydot = -b * y + alpha1 * x * y / (1 + k1 * x) - alpha2 * y * z / (1 + k2 * y)
        zdot = -c * (z - zs) + alpha2 * y * z / (1 + k2 * y)
        return xdot, ydot, zdot


class FluidTrampoline(ContinuousSystem):
    params = {"gamma": 1.82, "psi": 0.01019, "w": 1.21}
    dim = 3

    @staticmethod
    def _equations(Y, t, *, gamma, psi, w):
        x, y, th = Y(0), Y(1), Y(2)
        xdot = y
        ydot = -1 - (1 - sign(x)) / 2 * (x + psi * y * abs(y)) + gamma * cos(th)
        thdot = w
        return (xdot, ydot, thdot)


class JerkCircuit(ContinuousSystem):
    params = {"eps": 1e-9, "y0": 0.026}
    dim = 3

    @staticmethod
    def _equations(Y, t, *, eps, y0):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = z
        zdot = -z - x - eps * (exp(y / y0) - 1)
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, eps, y0):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-1, -eps * exp(y / y0) / y0, -1]
        return row1, row2, row3


class WindmiReduced(ContinuousSystem):
    params = {"a1": 0.247, "b1": 10.8, "b2": 0.0752, "b3": 1.06, "d1": 2200, "vsw": 5}
    dim = 3

    @staticmethod
    def _equations(Y, t, *, a1, b1, b2, b3, d1, vsw):
        i, v, p = Y(0), Y(1), Y(2)
        idot = a1 * (vsw - v)
        vdot = b1 * i - b2 * p**1 / 2 - b3 * v
        pdot = vsw**2 - p ** (5 / 4) * vsw ** (1 / 2) * (1 + tanh(d1 * (i - 1))) / 2
        return idot, vdot, pdot
