import numpy as np

from tsdynamics.families import DiscreteMap


class Bogdanov(DiscreteMap):
    params = {"eps": 0.0, "k": 1.2, "mu": 0.0}
    dim = 2
    reference = "Arrowsmith et al. (1993), Int. J. Bifurcation Chaos 3, 803-842"

    @staticmethod
    def _step(X, eps, k, mu):
        x, y = X
        yp = (1 + eps) * y + k * x * (x - 1) + mu * x * y
        xp = x + yp
        return xp, yp

    @staticmethod
    def _jacobian(X, eps, k, mu):
        # x' = x + y' inherits the y'-row derivatives
        x, y = X
        dyp_dx = k * (2 * x - 1) + mu * y
        dyp_dy = 1 + eps + mu * x
        row1 = [1 + dyp_dx, dyp_dy]
        row2 = [dyp_dx, dyp_dy]
        return row1, row2


class Svensson(DiscreteMap):
    params = {"a": 1.5, "b": -1.8, "c": 1.6, "d": 0.9}
    dim = 2

    @staticmethod
    def _step(X, a, b, c, d):
        x, y = X
        xp = d * np.sin(a * x) - np.sin(b * y)
        yp = c * np.cos(a * x) + np.cos(b * y)
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b, c, d):
        x, y = X
        row1 = [a * d * np.cos(a * x), -b * np.cos(b * y)]
        row2 = [-a * c * np.sin(a * x), -b * np.sin(b * y)]
        return row1, row2


class Bedhead(DiscreteMap):
    params = {"a": -0.67, "b": 0.83}
    dim = 2

    @staticmethod
    def _step(X, a, b):
        x, y = X
        xp = np.sin(x * y / b) * y + np.cos(a * x - y)
        yp = x + np.sin(y) / b
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b):
        x, y = X
        row1 = [
            y**2 * np.cos(x * y / b) / b - a * np.sin(a * x - y),
            np.sin(x * y / b) + x * y * np.cos(x * y / b) / b + np.sin(a * x - y),
        ]
        row2 = [1, np.cos(y) / b]
        return row1, row2


class ZeraouliaSprott(DiscreteMap):
    params = {"a": 2.7, "b": 0.35}
    dim = 2

    @staticmethod
    def _step(X, a, b):
        x, y = X
        xp = (-a * x) / (1 + y**2)
        yp = x + b * y
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b):
        x, y = X
        # Partial derivatives for the rational map
        df1_dx = -a / (1 + y**2)
        df1_dy = (2 * a * x * y) / (1 + y**2) ** 2
        df2_dx = 1.0
        df2_dy = b
        row1 = [df1_dx, df1_dy]
        row2 = [df2_dx, df2_dy]
        return row1, row2


class GumowskiMira(DiscreteMap):
    params = {"a": -1.1, "b": -0.2}
    dim = 2
    reference = "Gumowski & Mira (1980), Recurrences and Discrete Dynamic Systems"

    @staticmethod
    def _step(X, a, b):
        x, y = X
        fx = a * x + 2 * (1 - a) * x**2 / (1 + x**2)
        xp = b * y + fx
        fx1 = a * xp + 2 * (1 - a) * xp**2 / (1 + xp**2)
        yp = fx1 - x
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b):
        x, y = X

        fx = a * x + 2 * (1 - a) * x**2 / (1 + x**2)
        xp = b * y + fx

        dxdx = a + (4 * (1 - a) * x) / (1 + x**2) - (4 * (1 - a) * x**3) / (1 + x**2) ** 2

        dxdy = b

        # y' = G(x') - x  →  dy'/dx = G'(x')·dx'/dx - 1,  dy'/dy = G'(x')·b
        gprime_xp = a + (4 * (1 - a) * xp) / (1 + xp**2) - (4 * (1 - a) * xp**3) / (1 + xp**2) ** 2
        dydx = dxdx * gprime_xp - 1

        dydy = b * gprime_xp

        row1 = [dxdx, dxdy]
        row2 = [dydx, dydy]

        return row1, row2


class Hopalong(DiscreteMap):
    """
    Hopalong (Martin) attractor map.

    This is a shifted variant of Barry Martin's "Hopalong" map: the step carries
    several ``- 1`` offsets and a ``sign(x - 1)`` / ``b*x - 1 - c`` argument
    rather than the bare canonical ``x' = y - sign(x)*sqrt(|b*x - c|)``,
    ``y' = a - x``. The offsets are intentional and self-consistent (the hand
    Jacobian matches ``_step``).
    """

    params = {"a": 3.1, "b": 2.5, "c": 4.2}
    dim = 2
    reference = "Dewdney (1986), Scientific American 255(3), 14-20"

    @staticmethod
    def _step(X, a, b, c):
        x, y = X
        xp = y - 1 - np.sqrt(np.abs(b * x - 1 - c)) * np.sign(x - 1)
        yp = a - x - 1
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b, c):
        x, y = X
        eps = 1e-30
        denom = np.sqrt(np.abs(b * x - 1 - c) + eps)
        j00 = -0.5 * b * np.sign(x - 1) * np.sign(b * x - 1 - c) / denom
        row1 = [j00, 1.0]
        row2 = [-1.0, 0.0]
        return row1, row2


class Pickover(DiscreteMap):
    params = {"a": -1.4, "b": 1.6, "c": 1.0, "d": 0.7}
    dim = 2

    @staticmethod
    def _step(X, a, b, c, d):
        x, y = X
        xp = np.sin(a * y) + c * np.cos(a * x)
        yp = np.sin(b * x) + d * np.cos(b * y)
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b, c, d):
        x, y = X
        row1 = [-a * c * np.sin(a * x), a * np.cos(a * y)]
        row2 = [b * np.cos(b * x), -b * d * np.sin(b * y)]
        return row1, row2
