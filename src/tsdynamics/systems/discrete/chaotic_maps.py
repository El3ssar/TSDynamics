import numpy as np

from tsdynamics.base import DiscreteMap
from tsdynamics.utils import staticjit


class Henon(DiscreteMap):
    """
    Hénon (1976) map.

    A 2D dissipative map with a strange attractor for the default parameters.
    Lyapunov exponents at default params ≈ [0.42, −1.62].

    Parameters
    ----------
    a, b : float
        Map parameters.  ``a=1.4, b=0.3`` gives the classical attractor.
    """

    params = {"a": 1.4, "b": 0.3}
    dim = 2
    variables = ("x", "y")
    reference = "Hénon (1976), Commun. Math. Phys. 50, 69-77"
    known_lyapunov = {
        "spectrum": (0.419, -1.623),
        "atol": (0.2, 0.4),
        "kwargs": {"steps": 10_000},
        "source": "Sprott (2003), Chaos and Time-Series Analysis",
    }

    @staticjit
    def _step(X, a, b):
        x, y = X[0], X[1]
        return (1.0 - a * x**2 + y, b * x)

    @staticjit
    def _jacobian(X, a, b):
        x, y = X[0], X[1]
        return ((-2.0 * a * x, 1.0), (b, 0.0))


class Ulam(DiscreteMap):
    params = {"a": 1.0, "b": 2.0}
    dim = 1

    @staticjit
    def _step(X, a, b):
        x = X
        return a - b * x**2

    @staticjit
    def _jacobian(X, a, b):
        x = X
        return [-2 * b * x]


class Ikeda(DiscreteMap):
    r"""
    Ikeda (1979) map — laser ring cavity model.

    .. math::

        t   &= a - b / (1 + x^2 + y^2) \\
        x'  &= 1 + u \, (x \cos t - y \sin t) \\
        y'  &= u \, (x \sin t + y \cos t)
    """

    params = {"a": 0.4, "b": 6.0, "u": 0.9}
    dim = 2

    @staticjit
    def _step(X, a, b, u):
        x, y = X
        t = a - b / (1 + x**2 + y**2)
        xp = 1 + u * (x * np.cos(t) - y * np.sin(t))
        yp = u * (x * np.sin(t) + y * np.cos(t))
        return xp, yp

    @staticjit
    def _jacobian(X, a, b, u):
        x, y = X
        denom = 1.0 + x**2 + y**2
        t = a - b / denom
        # ∂t/∂x = 2bx / (1+x²+y²)²    (positive sign — earlier code had this wrong)
        dt_dx = 2.0 * b * x / denom**2
        dt_dy = 2.0 * b * y / denom**2

        cos_t, sin_t = np.cos(t), np.sin(t)
        # P = x cos t - y sin t  ;  Q = x sin t + y cos t
        Q = x * sin_t + y * cos_t
        P = x * cos_t - y * sin_t

        dxp_dx = u * cos_t - u * Q * dt_dx
        dxp_dy = -u * sin_t - u * Q * dt_dy
        dyp_dx = u * sin_t + u * P * dt_dx
        dyp_dy = u * cos_t + u * P * dt_dy

        return [[dxp_dx, dxp_dy], [dyp_dx, dyp_dy]]


class Tinkerbell(DiscreteMap):
    """
    Tinkerbell map.

    Random ``U[0, 1)^2`` ICs almost always escape the basin, so a known-good
    attractor point is set as ``default_ic``.
    """

    params = {"a": 0.9, "b": -0.6013, "c": 2.0, "d": 0.5}
    dim = 2
    default_ic = np.array([-0.72, -0.64])
    variables = ("x", "y")
    reference = "Nusse & Yorke (1994), Dynamics: Numerical Explorations"
    known_lyapunov = {
        "n_positive": 1,
        "kwargs": {"steps": 20_000},
        "source": "chaotic at default parameters",
    }

    @staticjit
    def _step(X, a, b, c, d):
        x, y = X
        xp = x**2 - y**2 + a * x + b * y
        yp = 2 * x * y + c * x + d * y
        return xp, yp

    @staticjit
    def _jacobian(X, a, b, c, d):
        x, y = X
        row1 = [2 * x + a, -2 * y + b]
        row2 = [2 * y + c, 2 * x + d]
        return row1, row2


class Gingerbreadman(DiscreteMap):
    params = {}
    dim = 2

    @staticjit
    def _step(X):
        x, y = X
        xp = 1 - y + np.abs(x)
        yp = x
        return xp, yp

    @staticjit
    def _jacobian(X):
        x, y = X
        row1 = [np.sign(x), -1]
        row2 = [1, 0]
        return row1, row2


class Zaslavskii(DiscreteMap):
    params = {"eps": 5.0, "nu": 0.2, "r": 2.0}
    dim = 2

    @staticjit
    def _step(X, eps, nu, r):
        x, y = X
        mu = (1 - np.exp(-r)) / r
        xp = x + nu * (1 + mu * y) + eps * nu * mu * np.cos(2 * np.pi * x)
        xp = xp % 0.99999995
        yp = np.exp(-r) * (y + eps * np.cos(2 * np.pi * x))
        return xp, yp

    @staticjit
    def _jacobian(X, eps, nu, r):
        x, y = X
        mu = (1 - np.exp(-r)) / r

        dxpdx = nu * (1 + mu * y) - 2 * np.pi * eps * nu * mu * np.sin(2 * np.pi * x)
        dxpdy = nu * mu

        dypdx = -2 * np.pi * eps * np.exp(-r) * np.sin(2 * np.pi * x)
        dypdy = np.exp(-r)

        row1 = [dxpdx, dxpdy]
        row2 = [dypdx, dypdy]
        return row1, row2


class Chirikov(DiscreteMap):
    params = {"k": 0.971635}
    dim = 2

    @staticjit
    def _step(X, k):
        p, x = X
        pp = p + k * np.sin(x)
        xp = x + pp
        return pp, xp

    @staticjit
    def _jacobian(X, k):
        p, x = X
        row1 = [1, k * np.cos(x)]
        row2 = [1, 1]
        return row1, row2


# Hyperchaotic maps
class FoldedTowel(DiscreteMap):
    params = {"a": 3.8, "b": 0.05, "c": 0.35, "d": 0.1, "e": 1.9, "f": 3.78, "g": 0.2}
    dim = 3
    reference = "Rössler (1979), Phys. Lett. A 71, 155-157"
    known_lyapunov = {
        "n_positive": 2,
        "kwargs": {"steps": 20_000},
        "source": "hyperchaotic map: two positive exponents",
    }

    @staticjit
    def _step(X, a, b, c, d, e, f, g):
        x, y, z = X
        xp = a * x * (1 - x) - b * (y + c) * (1 - 2 * z)
        yp = d * ((y + c) * (1 + 2 * z) - 1) * (1 - e * x)
        zp = f * z * (1 - z) + g * y
        return xp, yp, zp

    @staticjit
    def _jacobian(X, a, b, c, d, e, f, g):
        x, y, z = X
        row1 = [a * (1 - 2 * x), -b * (1 - 2 * z), 2 * b * (y - c)]

        row2 = [
            -d * e * ((y - c) * (1 + 2 * z) - 1),
            d * (1 + 2 * z) * (1 - e * x),
            2 * d * (y - c) * (1 - e * x),
        ]

        row3 = [0, g, f * (1 - 2 * z)]

        return row1, row2, row3


class GeneralizedHenon(DiscreteMap):
    params = {
        "a": 1.9,
        "b": 0.03,
    }
    dim = 3

    @staticjit
    def _step(X, a, b):
        x, y, z = X
        xp = a - y**2 - b * z
        yp = x
        zp = y
        return xp, yp, zp

    @staticjit
    def _jacobian(X, a, b):
        x, y, z = X
        row1 = [0, -2 * y, b]
        row2 = [1, 0, 0]
        row3 = [0, 1, 0]

        return row1, row2, row3
