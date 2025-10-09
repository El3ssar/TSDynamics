from tsdynamics.base import DynSys

class CoevolvingPredatorPrey(DynSys):
    params = {
      "a1": 2.5,
      "a2": 0.05,
      "a3": 0.4,
      "b1": 6.0,
      "b2": 1.333,
      "d1": 0.16,
      "d2": 0.004,
      "delta": 1,
      "k1": 6.0,
      "k2": 9.0,
      "k4": 9.0,
      "vv": 0.33333
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a1, a2, a3, b1, b2, d1, d2, delta, k1, k2, k4, vv):
        x, y, alpha = Y(0), Y(1), Y(2)
        xdot = x * (
            -((a3 * y) / (1 + b2 * x))
            + (a1 * alpha * (1 - k1 * x * (-alpha + alpha * delta))) / (1 + b1 * alpha)
            - d1
            * (
                1
                - k2 * (-(alpha**2) + (alpha * delta) ** 2)
                + k4 * (-(alpha**4) + (alpha * delta) ** 4)
            )
        )
        ydot = (-d2 + (a2 * x) / (1 + b2 * x)) * y
        alphadot = vv * (
            -((a1 * k1 * x * alpha * delta) / (1 + b1 * alpha))
            - d1 * (-2 * k2 * alpha * delta**2 + 4 * k4 * alpha**3 * delta**4)
        )
        return xdot, ydot, alphadot


class KawczynskiStrizhak(DynSys):
    params = {
      "beta": -0.4,
      "gamma": 0.49,
      "kappa": 0.2,
      "mu": 2.1
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, beta, gamma, kappa, mu):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = gamma * y - gamma * x**3 + 3 * mu * gamma * x
        ydot = -2 * mu * x - y - z + beta
        zdot = kappa * x - kappa * z
        return xdot, ydot, zdot

    @staticmethod
    def _jac(Y, t, beta, gamma, kappa, mu):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-3 * gamma * x**2 + 3 * mu * gamma, gamma, 0]
        row2 = [-2 * mu, -1, -1]
        row3 = [kappa, 0, -kappa]
        return row1, row2, row3


class Finance(DynSys):
    params = {
      "a": 0.001,
      "b": 0.2,
      "c": 1.1
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = (1 / b - a) * x + z + x * y
        ydot = -b * y - x**2
        zdot = -x - c * z
        return xdot, ydot, zdot

    @staticmethod
    def _jac(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [(1 / b - a) + y, x, 1]
        row2 = [-2 * x, -b, 0]
        row3 = [-1, 0, -c]
        return row1, row2, row3

