from tsdynamics.base import DynSys
from tsdynamics.utils import staticjit
import numpy as np


class Hopfield(DynSys):
    params = {
        "n_neurons": 3, 
        "tau": 1.0, 
        "beta": 1.0
        }
    n_dim = params["n_neurons"]

    @staticjit
    def _rhs(X, t, tau, beta, n_neurons):
        """
        Right-hand side of the Hopfield model.

        X: State vector [neuron activities]
        t: Time (not explicitly used in autonomous systems)
        tau: Timescale of neuron dynamics
        beta: Scaling factor for weights
        n_neurons: Number of neurons in the network
        """
        # Generate symmetric weight matrix (W) inside _rhs
        W = np.random.randn(n_neurons, n_neurons)
        W = 0.5 * (W + W.T)  # Symmetrize weights
        np.fill_diagonal(W, 0)  # Zero diagonal

        # Hopfield dynamics
        dXdt = (-X + np.tanh(beta * W @ X)) / tau
        return dXdt


class CellularNeuralNetwork(DynSys):
    params = {
      "a": 4.4,
      "b": 3.21,
      "c": 1.1,
      "d": 1.24
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c, d):
        x, y, z = X

        def f(x):
            return 0.5 * (np.abs(x + 1) - np.abs(x - 1))

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
        "n_neurons": 100
        }
    n_dim = params["n_neurons"]  # Number of neurons in the RNN
    @staticjit
    def _rhs(X, t, alpha, beta, gamma, tau, n_neurons):
        """
        Right-hand side of the BeerRNN model.

        X: State vector [neuron activities]
        t: Time (not explicitly used in autonomous systems)
        alpha, beta, gamma, tau: Model parameters
        n_neurons: Number of neurons in the network
        """
        W = np.random.randn(n_neurons, n_neurons) * beta  # Recurrent weights
        I = np.sin(2 * np.pi * t / tau) * gamma  # External input (periodic forcing)  # noqa: E741
        dXdt = -alpha * X + np.tanh(W @ X + I)
        return dXdt