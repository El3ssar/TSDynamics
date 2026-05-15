import numpy as np

from tsdynamics.base import DiscreteMap
from tsdynamics.utils import staticjit


class Logistic(DiscreteMap):
    params = {"r": 3.9}
    dim = 1

    @staticjit
    def _step(X, r):
        x = X
        return r * x * (1 - x)

    @staticjit
    def _jacobian(X, r):
        x = X
        return [r - 2 * r * x]


class Ricker(DiscreteMap):
    params = {"a": 3.3}
    dim = 1

    @staticjit
    def _step(X, a):
        x = X
        return x * np.exp(a - x)

    @staticjit
    def _jacobian(X, a):
        x = X
        return [np.exp(a - x) - x * np.exp(a - x)]


class MaynardSmith(DiscreteMap):
    params = {"a": 0.87, "b": 0.75}
    dim = 2

    @staticjit
    def _step(X, a, b):
        x, y = X
        xp = y
        yp = a * y + b - x**2
        return xp, yp

    @staticjit
    def _jacobian(X, a, b):
        x, y = X
        row1 = [0, 1]
        row2 = [-2 * x, a]
        return row1, row2
