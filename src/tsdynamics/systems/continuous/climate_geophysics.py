from symengine import cos, pi, sin, tanh

from tsdynamics.base import ContinuousSystem


class VallisElNino(ContinuousSystem):
    params = {"b": 102.0, "c": 3.0, "p": 0.0}
    dim = 3

    @staticmethod
    def _equations(Y, t, b, c, p):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = b * y - c * x - c * p
        ydot = -y + x * z
        zdot = -z - x * y + 1
        return xdot, ydot, zdot


class RayleighBenard(ContinuousSystem):
    params = {"a": 30, "b": 5, "r": 18}
    dim = 3

    @staticmethod
    def _equations(Y, t, a, b, r):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * y - a * x
        ydot = r * y - x * z
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, r):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, a, 0]
        row2 = [-z, r, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class Hadley(ContinuousSystem):
    params = {"a": 0.2, "b": 4.0, "f": 9.0, "g": 1.0}
    dim = 3

    @staticmethod
    def _equations(Y, t, a, b, f, g):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -(y**2) - z**2 - a * x + a * f
        ydot = x * y - b * x * z - y + g
        zdot = b * x * y + x * z - z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, f, g):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, -2 * y, -2 * z]
        row2 = [y - b * z, x - 1, -b * x]
        row3 = [b * y + z, b * x, x - 1]
        return row1, row2, row3


class DoubleGyre(ContinuousSystem):
    params = {"alpha": 0.1, "eps": 0.1, "omega": 0.62832}
    dim = 3

    @staticmethod
    def _equations(Y, t, alpha, eps, omega):
        x, y, z = Y(0), Y(1), Y(2)
        a = eps * sin(z)
        b = 1 - 2 * eps * sin(z)
        f = a * x**2 + b * x
        dx = -alpha * pi * sin(pi * f) * cos(pi * y)
        dy = alpha * pi * cos(pi * f) * sin(pi * y) * (2 * a * x + b)
        dz = omega
        return dx, dy, dz


class BlinkingRotlet(ContinuousSystem):
    params = {
        "a": 1.0,
        "b": 0.5298833894399929,
        "bc": 1.0,
        "sigma": -1.0,
        "tau": 3.0,
    }
    dim = 3

    @staticmethod
    def _rotlet(r, theta, a, b, bc):
        kappa = a**2 + (b**2 * r**2) / a**2 - 2 * b * r * cos(theta)
        gamma = (1 - r**2 / a**2) * (a**2 - (b**2 * r**2) / a**2)
        iota = (b**2 * r) / a**2 - b * cos(theta)
        zeta = b**2 + r**2 - 2 * b * r * cos(theta)
        nu = a**2 + b**2 - (2 * b**2 * r**2) / a**2

        vr = b * sin(theta) * (-bc * (gamma / kappa**2) - 1 / kappa + 1 / zeta)

        vth = (
            bc * (gamma * iota) / kappa**2
            + bc * r * nu / (a**2 * kappa)
            + iota / kappa
            - (r - b * cos(theta)) / zeta
        )
        return vr, vth

    @staticmethod
    def _protocol(t, tau, stiffness=20):
        return 0.5 + 0.5 * tanh(tau * stiffness * sin(2 * pi * t / tau))

    @staticmethod
    def _equations(y, t, a, b, bc, sigma, tau):
        r = y(0)
        theta = y(1)
        tt = y(2)

        weight = BlinkingRotlet._protocol(tt, tau)

        dr1, dth1 = BlinkingRotlet._rotlet(r, theta, a, b, bc)
        dr2, dth2 = BlinkingRotlet._rotlet(r, theta, a, -b, bc)

        dr = weight * dr1 + (1 - weight) * dr2
        dth = (weight * dth1 + (1 - weight) * dth2) / r
        dtt = 1

        return (
            sigma * dr,
            sigma * dth,
            dtt,
        )


class OscillatingFlow(ContinuousSystem):
    params = {"b": 0.48, "k": 1.0, "omega": 0.49, "u": 0.72}
    dim = 3

    @staticmethod
    def _equations(Y, t, b, k, omega, u):
        x, y, z = Y(0), Y(1), Y(2)
        f = x + b * sin(z)
        dx = u * cos(k * y) * sin(k * f)
        dy = -u * sin(k * y) * cos(k * f)
        dz = omega
        return dx, dy, dz


class ArnoldBeltramiChildress(ContinuousSystem):
    params = {"a": 1.73205, "b": 1.41421, "c": 1}
    dim = 3

    @staticmethod
    def _equations(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        dx = a * sin(z) + c * cos(y)
        dy = b * sin(x) + a * cos(z)
        dz = c * sin(y) + b * cos(x)
        return dx, dy, dz


class AtmosphericRegime(ContinuousSystem):
    params = {
        "alpha": -2.0,
        "beta": -5.0,
        "mu1": 0.05,
        "mu2": -0.01,
        "omega": 3.0,
        "sigma": 1.1,
    }
    dim = 3

    @staticmethod
    def _equations(Y, t, alpha, beta, mu1, mu2, omega, sigma):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = mu1 * x + sigma * x * y
        ydot = mu2 * y + omega * z + alpha * y * z + beta * z**2 - sigma * x**2
        zdot = mu2 * z - omega * y - alpha * y**2 - beta * y * z
        return xdot, ydot, zdot


class SaltonSea(ContinuousSystem):
    params = {
        "a": 15,
        "d": 8.3,
        "k": 400,
        "lam": 0.06,
        "m": 15.5,
        "mu": 3.4,
        "r": 22,
        "th": 10.0,
    }
    dim = 3

    @staticmethod
    def _equations(Y, t, a, d, k, lam, m, mu, r, th):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = r * x * (1 - (x + y) / k) - lam * x * y
        ydot = lam * x * y - m * y * z / (y + a) - mu * y
        zdot = th * y * z / (y + a) - d * z
        return xdot, ydot, zdot
