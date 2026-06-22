import numpy as np

from tsdynamics.families import DiscreteMap


class Logistic(DiscreteMap):
    params = {"r": 3.9}
    dim = 1
    variables = ("x",)
    reference = "May (1976), Nature 261, 459-467"
    known_lyapunov = {
        "params": {"r": 4.0},
        "spectrum": (0.6931,),  # exactly ln 2 at r = 4
        "atol": (0.15,),
        "kwargs": {"steps": 10_000},
        "source": "exact result at r = 4",
    }

    @staticmethod
    def _step(X, r):
        x = X
        return r * x * (1 - x)

    @staticmethod
    def _jacobian(X, r):
        x = X
        return [r - 2 * r * x]


class Ricker(DiscreteMap):
    params = {"a": 3.3}
    dim = 1
    reference = "Ricker (1954), J. Fish. Res. Board Can. 11, 559-623"

    @staticmethod
    def _step(X, a):
        x = X
        return x * np.exp(a - x)

    @staticmethod
    def _jacobian(X, a):
        x = X
        return [np.exp(a - x) - x * np.exp(a - x)]


class MaynardSmith(DiscreteMap):
    params = {"a": 0.87, "b": 0.75}
    dim = 2

    @staticmethod
    def _step(X, a, b):
        x, y = X
        xp = y
        yp = a * y + b - x**2
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b):
        x, y = X
        row1 = [0, 1]
        row2 = [-2 * x, a]
        return row1, row2
