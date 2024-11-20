from tsdynamics.base import DynSysDelay
from tsdynamics.utils import staticjit
import numpy as np

class MackeyGlass(DynSysDelay):
    params = {
        "beta": 0.2, 
        "gamma": 0.1, 
        "tau": 17.0, 
        "n": 10
        }
    delays = [params["tau"]]
    n_dim = 1  # One-dimensional system

    @staticjit
    def _rhs(X_current, X_delayed, t, beta, gamma, tau, n):
        x = X_current
        xt = X_delayed[0]
        xdot = beta * xt / (1 + xt ** n) - gamma * x
        return xdot


class IkedaDelay(DynSysDelay):
    params = {
        "c": 1.0,
        "mu": -20,
        "tau": 2.0,
        "x0": 0.0
        }
    delays = [params["tau"]]
    n_dim = 1
    
    @staticjit
    def _rhs(X_current, X_delayed, t, c, mu, tau, x0):
        x = X_current
        xt = X_delayed[0]
        xdot = mu * np.sin(xt - x0) - c * x
        return xdot


class SprottDelay(DynSysDelay):
    params = {
      "tau": 5.1
    }
    delays = [params["tau"]]
    n_dim = 1
    @staticjit
    def _rhs(X_current, X_delayed, t, tau):
        xt = X_delayed[0]
        return np.sin(xt)


class VossDelay(DynSysDelay):
    params = {
        "alpha": 3.24,
        "tau": 13.28
        }
    delays = [params["tau"]]
    n_dim = 1
    @staticjit
    def _rhs(X_current, X_delayed, t, alpha, tau):
        x = X_current
        xt = X_delayed[0]
        f = -10.44 * xt**3 - 13.95 * xt**2 - 3.63 * xt + 0.85
        xdot = -alpha * x + f
        return xdot


class ScrollDelay(DynSysDelay):
    params={
      "alpha": 0.2,
      "beta": 0.2,
      "tau": 10.0
    }
    delays = [params["tau"]]
    n_dim = 1
    @staticjit
    def _rhs(X_current, X_delayed, t, alpha, beta, tau):
        xt = X_delayed[0]
        f = np.tanh(10 * xt)
        xdot = -alpha * xt + beta * f
        return xdot


class PiecewiseCircuit(DynSysDelay):
    params={
      "alpha": 1.0,
      "beta": 1.0,
      "c": 2.24,
      "tau": 4.9
    }
    delays = [params["tau"]]
    n_dim = 1

    @staticjit
    def _rhs(X_current, X_delayed, t, alpha, beta, c, tau):
        xt = X_delayed[0]
        f = -((xt / c) ** 3) + 3 * xt / c
        xdot = -alpha * xt + beta * f
        return xdot


class ENSODelay(DynSysDelay):
    params={
      "alpha": 0.2,
      "beta": 0.4,
      "gamma": 0.5,
      "tau": 5.0
    }
    delays = [params["tau"]]
    n_dim = 1
    @staticjit
    def _rhs(X_current, X_delayed, t, alpha, beta, gamma, tau):
        x = X_current
        xt = X_delayed[0]
        xdot = -alpha * x - beta * xt + gamma
        return xdot

