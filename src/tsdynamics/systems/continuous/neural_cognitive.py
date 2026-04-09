from tsdynamics.base import DynSys
from symengine import tanh, pi, sin
import numpy as np


class Hopfield(DynSys):
    params = {
        "n_neurons": 3,
        "tau": 1.0,
        "beta": 1.0,
    }
    n_dim = params["n_neurons"]

    def __init__(self, *args, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        n = int(self.n_neurons)
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n, n))
        W = 0.5 * (W + W.T)   # symmetrize
        np.fill_diagonal(W, 0)
        self._W = W

    @staticmethod
    def _rhs(Y, t, **params):
        # Hopfield overrides rhs() directly (needs stored weight matrix).
        raise NotImplementedError("Hopfield overrides rhs() directly.")

    def rhs(self, Y, t):
        n = int(self.n_neurons)
        result = []
        for i in range(n):
            wx_i = sum(float(self._W[i, j]) * Y(j) for j in range(n))
            result.append((-Y(i) + tanh(float(self.beta) * wx_i)) / float(self.tau))
        return result


class CellularNeuralNetwork(DynSys):
    params = {
      "a": 4.4,
      "b": 3.21,
      "c": 1.1,
      "d": 1.24
    }
    n_dim = 3
    @staticmethod
    def _rhs(Y, t, a, b, c, d):
        x, y, z = Y(0), Y(1), Y(2)

        def f(x):
            return 0.5 * (abs(x + 1) - abs(x - 1))

        xdot = -x + d * f(x) - b * f(y) - b * f(z)
        ydot = -y - b * f(x) + c * f(y) - a * f(z)
        zdot = -z - b * f(x) + a * f(y) + f(z)
        return (xdot, ydot, zdot)


class BeerRNN(DynSys):
    params = {
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 0.01,
        "tau": 10.0,
        "n_neurons": 100,
    }
    n_dim = params["n_neurons"]

    def __init__(self, *args, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        n = int(self.n_neurons)
        rng = np.random.default_rng(seed)
        self._W = rng.standard_normal((n, n)) * float(self.beta)

    @staticmethod
    def _rhs(Y, t, **params):
        # BeerRNN overrides rhs() directly (needs stored weight matrix).
        raise NotImplementedError("BeerRNN overrides rhs() directly.")

    def rhs(self, Y, t):
        n = int(self.n_neurons)
        I = sin(2 * pi * t / float(self.tau)) * float(self.gamma)  # noqa: E741
        result = []
        for i in range(n):
            wx_i = sum(float(self._W[i, j]) * Y(j) for j in range(n))
            result.append(-float(self.alpha) * Y(i) + tanh(wx_i + I))
        return result