from tsdynamics.base import DynSysDelay
from symengine import sin, tanh

class MackeyGlass(DynSysDelay):
    params = {
        "beta": 0.2,
        "gamma": 0.1,
        "tau": 17.0,
        "n": 10
        }
    n_dim = 1  # One-dimensional system

    @staticmethod
    def _rhs(Y, t, beta, gamma, tau, n):
        return [
            beta * Y(0, t - tau) / (1.0 + Y(0, t - tau) ** n)
            - gamma * Y(0, t)
        ]


class IkedaDelay(DynSysDelay):
    params = {
        "c": 1.0,
        "mu": -20,
        "tau": 2.0,
        "x0": 0.0
        }
    n_dim = 1

    @staticmethod
    def _rhs(Y, t, c, mu, tau, x0):
        return [mu * sin(Y(0, t - tau) - x0) - c * Y(0, t)]


class SprottDelay(DynSysDelay):
    params = {
      "tau": 5.1
    }
    n_dim = 1
    @staticmethod
    def _rhs(Y, t, tau):
        return [sin(Y(0, t - tau))]


class ScrollDelay(DynSysDelay):
    params={
      "alpha": 0.2,
      "beta": 0.2,
      "tau": 10.0
    }
    n_dim = 1
    @staticmethod
    def _rhs(Y, t, alpha, beta, tau):
        xt = Y(0, t - tau)
        f = tanh(10 * xt)
        return [-alpha * xt + beta * f]


class PiecewiseCircuit(DynSysDelay):
    params={
      "alpha": 1.0,
      "beta": 1.0,
      "c": 2.24,
      "tau": 4.9
    }
    n_dim = 1
    @staticmethod
    def _rhs(Y, t, alpha, beta, c, tau):
        xt = Y(0, t - tau)
        f = -((xt / c) ** 3) + 3 * xt / c
        return [-alpha * xt + beta * f]


class ENSODelay(DynSysDelay):
    params={
      "alpha": 0.2,
      "beta": 0.4,
      "gamma": 0.5,
      "tau": 5.0
    }
    n_dim = 1
    @staticmethod
    def _rhs(Y, t, alpha, beta, gamma, tau):
        x = Y(0, t)
        xt = Y(0, t - tau)
        return [-alpha * x - beta * xt + gamma]

