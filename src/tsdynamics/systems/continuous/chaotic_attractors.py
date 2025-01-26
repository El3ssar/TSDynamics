from tsdynamics.base import DynSys
from tsdynamics.utils import staticjit
import numpy as np


class Lorenz(DynSys):
    params = {"sigma": 10, "rho": 28, "beta": 8 / 3}
    n_dim = 3

    @staticjit
    def _rhs(X, t, beta, rho, sigma):
        x, y, z = X
        xdot = sigma * y - sigma * x
        ydot = rho * x - x * z - y
        zdot = x * y - beta * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, beta, rho, sigma):
        x, y, z = X
        row1 = [-sigma, sigma, 0]
        row2 = [rho - z, -1, -x]
        row3 = [y, x, -beta]
        return row1, row2, row3


class LorenzBounded(DynSys):
    params = {
      "beta": 2.667,
      "r": 64,
      "rho": 28,
      "sigma": 10
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, beta, r, rho, sigma):
        x, y, z = X
        xdot = (
            sigma * y
            - sigma * x
            - sigma / r**2 * y * x**2
            - sigma / r**2 * y**3
            - sigma / r**2 * y * z**2
            + sigma / r**2 * x**3
            + sigma / r**2 * x * y**2
            + sigma / r**2 * x * z**2
        )
        ydot = (
            rho * x
            - x * z
            - y
            - rho / r**2 * x**3
            - rho / r**2 * x * y**2
            - rho / r**2 * x * z**2
            + 1 / r**2 * z * x**3
            + 1 / r**2 * x * z * y**2
            + 1 / r**2 * x * z**3
            + 1 / r**2 * y * x**2
            + 1 / r**2 * y**3
            + 1 / r**2 * y * z**2
        )
        zdot = (
            x * y
            - beta * z
            - 1 / r**2 * y * x**3
            - 1 / r**2 * x * y**3
            - 1 / r**2 * x * y * z**2
            + beta / r**2 * z * x**2
            + beta / r**2 * z * y**2
            + beta / r**2 * z**3
        )
        return xdot, ydot, zdot


class LorenzCoupled(DynSys):
    params = {
      "beta": 8/3,
      "kappa": 2.85,
      "rho": 28,
      "sigma": 10
    }
    n_dim = 6
    @staticjit
    def _rhs(X, t, beta, kappa, rho, sigma):
        x1, y1, z1, x2, y2, z2 = X
        x1dot = sigma * (y1 - x1) + kappa * (x2 - x1)
        y1dot = rho * x1 - x1 * z1 - y1
        z1dot = x1 * y1 - beta * z1
        x2dot = sigma * (y2 - x2) + kappa * (x1 - x2)
        y2dot = rho * x2 - x2 * z2 - y2
        z2dot = x2 * y2 - beta * z2
        return x1dot, y1dot, z1dot, x2dot, y2dot, z2dot


class Lorenz96(DynSys):
    params = {
      "f": 8
    }
    n_dim = 5

    @staticmethod
    def _rhs(X, t, f):

        X_plus_2 = np.roll(X, 2)
        X_minus_1 = np.roll(X, -1)
        X_plus_1 = np.roll(X, 1)
        Xdot = (X_plus_1 * (X_minus_1 - X_plus_2) - X + f)

        return Xdot


class Lorenz84(DynSys):
    params = {
      "a": 1.32,
      "b": 7.91,
      "f": 4.83,
      "g": 4.194
    }
    n_dim = 3

    @staticjit
    def _rhs(X, t, a, b, f, g):
        x, y, z = X
        xdot = -a * x - y**2 - z**2 + a * f
        ydot = -y + x * y - b * x * z + g
        zdot = -z + b * x * y + x * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, f, g):
        x, y, z = X
        row1 = [-a, -2 * y, -2 * z]
        row2 = [y - b * z, x - 1, -b * x]
        row3 = [b * y + z, b * x, -1 + x]
        return row1, row2, row3


class Rossler(DynSys):
    params = {"a": 0.2, "b": 0.2, "c": 5.7}
    n_dim = 3

    @staticjit
    def _rhs(X, t, a, b, c):
        x, y, z = X
        xdot = -y - z
        ydot = x + a * y
        zdot = b + z * x - c * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c):
        x, y, z = X
        row1 = [0, -1, -1]
        row2 = [1, a, 0]
        row3 = [z, 0, x - c]
        return row1, row2, row3


class Thomas(DynSys):
    params = {"a": 1.85, "b": 10}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b):
        x, y, z = X
        xdot = -a * x + b * np.sin(y)
        ydot = -a * y + b * np.sin(z)
        zdot = -a * z + b * np.sin(x)
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b):
        x, y, z = X
        row1 = [-a, b * np.cos(y), 0]
        row2 = [0, -a, b * np.cos(z)]
        row3 = [b * np.cos(x), 0, -a]
        return row1, row2, row3


class Halvorsen(DynSys):
    params = {
      "a": 1.4,
      "b": 4
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b):
        x, y, z = X
        xdot = -a * x - b * y - b * z - y**2
        ydot = -a * y - b * z - b * x - z**2
        zdot = -a * z - b * x - b * y - x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b):
        x, y, z = X
        row1 = [-a, -b - 2 * y, -b]
        row2 = [-b, -a, -b - 2 * z]
        row3 = [-b - 2 * x, -b, -a]
        return row1, row2, row3


class Chua(DynSys):
    params = {
      "alpha": 15.6,
      "beta": 28.0,
      "m0": -1.142857,
      "m1": -0.71429
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, alpha, beta, m0, m1):
        x, y, z = X
        ramp_x = m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))
        xdot = alpha * (y - x - ramp_x)
        ydot = x - y + z
        zdot = -beta * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, alpha, beta, m0, m1):
        x, y, z = X
        dramp_xdx = m1 + 0.5 * (m0 - m1) * (np.sign(x + 1) - np.sign(x - 1))
        row1 = [-alpha - alpha * dramp_xdx, alpha, 0]
        row2 = [1, -1, 1]
        row3 = [0, -beta, 0]
        return row1, row2, row3


class MultiChua(DynSys):
    params = {
        "alpha": 15.6,
        "beta": 28.0,
        "m0": -1.143,
        "m1": -0.714,
        "kappa": 0.1,  # Coupling strength
        "n_circuits": 3  # Number of coupled Chua circuits
    }
    n_dim = 3 * params["n_circuits"]  # 3 variables per circuit

    @staticmethod
    def _rhs(X, t, alpha, beta, m0, m1, kappa, n_circuits):
        """
        Right-hand side of the MultiChua model.

        X: State vector [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn]
        """
        n_dim = 3 * n_circuits
        dXdt = np.zeros(n_dim)

        for i in range(n_circuits):
            # Extract indices for the current circuit
            x_idx = 3 * i
            y_idx = x_idx + 1
            z_idx = x_idx + 2

            # State variables for this circuit
            x = X[x_idx]
            y = X[y_idx]
            z = X[z_idx]

            # Coupled neighbor indices (periodic boundary conditions)
            x_prev = X[(x_idx - 3) % n_dim]  # Previous x (cyclic indexing)
            x_next = X[(x_idx + 3) % n_dim]  # Next x (cyclic indexing)

            # Nonlinear Chua diode function
            ramp_x = m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))

            # Chua equations with coupling
            xdot = alpha * (y - x - ramp_x) + kappa * (x_next - x_prev)
            ydot = x - y + z
            zdot = -beta * y

            # Assign derivatives
            dXdt[x_idx] = xdot
            dXdt[y_idx] = ydot
            dXdt[z_idx] = zdot

        return dXdt


class Duffing(DynSys):
    params = {
      "alpha": 1.0,
      "beta": -1.0,
      "delta": 0.1,
      "gamma": 0.35,
      "omega": 1.4
    }
    n_dim = 3

    @staticjit
    def _rhs(X, t, alpha, beta, delta, gamma, omega):
        x, y, z = X
        xdot = y
        ydot = -delta * y - beta * x - alpha * x**3 + gamma * np.cos(z)
        zdot = omega
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, alpha, beta, delta, gamma, omega):
        x, y, z = X
        row1 = [0, 1, 0]
        row2 = [-3 * alpha * x**2 - beta, -delta, -gamma * np.sin(z)]
        row3 = [0, 0, 0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.cos(z)


class RabinovichFabrikant(DynSys):
    params = {
      "a": 1.1,
      "g": 0.87
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, g):
        x, y, z = X
        xdot = y * (z - 1 + x**2) + g * x
        ydot = x * (3 * z + 1 - x**2) + g * y
        zdot = -2 * z * (a + x * y)
        return (xdot, ydot, zdot)

    @staticjit
    def _jac(X, t, a, g):
        x, y, z = X
        row1 = [2 * x * y + g, z - 1 + x**2, y]
        row2 = [3 * z + 1 - x**2, g, 3 * x]
        row3 = [-2 * y * z, -2 * x * z, -2 * (a + x * y)]
        return row1, row2, row3

class Dadras(DynSys):
    params = {
      "c": 2.0,
      "e": 9.0,
      "o": 2.7,
      "p": 3.0,
      "r": 1.7
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, c, e, o, p, r):
        x, y, z = X
        xdot = y - p * x + o * y * z
        ydot = r * y - x * z + z
        zdot = c * x * y - e * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, c, e, o, p, r):
        x, y, z = X
        row1 = [-p, 1 + o * z, o * y]
        row2 = [-z, r, -x]
        row3 = [c * y, c * x, -e]
        return row1, row2, row3

class PehlivanWei(DynSys):
    params = {}
    n_dim = 3
    @staticjit
    def _rhs(X, t):
        x, y, z = X
        xdot = y - y * z
        ydot = y + y * z - 2 * x
        zdot = 2 - x * y - y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t):
        x, y, z = X
        row1 = [0, 1 - z, -y]
        row2 = [-2, 1 + z, y]
        row3 = [-y, -x - 2 * y, 0]
        return row1, row2, row3


# region Sprott Attractors

class SprottTorus(DynSys):
    params = {}
    n_dim = 3
    @staticjit
    def _rhs(X, t):
        x, y, z = X
        xdot = y + 2 * x * y + x * z
        ydot = 1 - 2 * x**2 + y * z
        zdot = x - x**2 - y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t):
        x, y, z = X
        row1 = [2 * y + z, 2 * x + 1, x]
        row2 = [-4 * x, z, y]
        row3 = [1 - 2 * x, -2 * y, 0]
        return row1, row2, row3


class SprottA(DynSys):
    params = {}
    n_dim = 3
    @staticjit
    def _rhs(X, t):
        x, y, z = X
        xdot = y
        ydot = -x + y * z
        zdot = 1 - y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t):
        x, y, z = X
        row1 = [0, 1, 0]
        row2 = [-1, z, y]
        row3 = [0, -2 * y, 0]
        return row1, row2, row3


class SprottB(DynSys):
    params = {}
    n_dim = 3
    @staticjit
    def _rhs(X, t):
        x, y, z = X
        xdot = y * z
        ydot = x - y
        zdot = 1 - x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t):
        x, y, z = X
        row1 = [0, z, y]
        row2 = [1, -1, 0]
        row3 = [-y, -x, 0]
        return row1, row2, row3


class SprottC(DynSys):
    params = {}
    n_dim = 3
    @staticjit
    def _rhs(X, t):
        x, y, z = X
        xdot = y * z
        ydot = x - y
        zdot = 1 - x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t):
        x, y, z = X
        row1 = [0, z, y]
        row2 = [1, -1, 0]
        row3 = [-2 * x, 0, 0]
        return row1, row2, row3


class SprottD(DynSys):
    params = {}
    n_dim = 3
    @staticjit
    def _rhs(X, t):
        x, y, z = X
        xdot = -y
        ydot = x + z
        zdot = x * z + 3 * y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t):
        x, y, z = X
        row1 = [0, -1, 0]
        row2 = [1, 0, 1]
        row3 = [z, 6 * y, x]
        return row1, row2, row3


class SprottE(DynSys):
    params = {}
    n_dim = 3
    @staticjit
    def _rhs(X, t):
        x, y, z = X
        xdot = y * z
        ydot = x**2 - y
        zdot = 1 - 4 * x
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t):
        x, y, z = X
        row1 = [0, z, y]
        row2 = [2 * x, -1, 0]
        row3 = [-4, 0, 0]
        return row1, row2, row3


class SprottF(DynSys):
    params = {"a": 0.5}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a):
        x, y, z = X
        xdot = y + z
        ydot = -x + a * y
        zdot = x**2 - z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a):
        x, y, z = X
        row1 = [0, 1, 1]
        row2 = [-1, a, 0]
        row3 = [2 * x, 0, -1]
        return row1, row2, row3


class SprottG(DynSys):
    params = {"a": 0.4}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a):
        x, y, z = X
        xdot = a * x + z
        ydot = x * z - y
        zdot = -x + y
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a):
        x, y, z = X
        row1 = [a, 0, 1]
        row2 = [z, -1, x]
        row3 = [-1, 1, 0]
        return row1, row2, row3


class SprottH(DynSys):
    params = {"a": 0.5}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a):
        x, y, z = X
        xdot = -y + z**2
        ydot = x + a * y
        zdot = x - z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a):
        x, y, z = X
        row1 = [0, -1, 2 * z]
        row2 = [1, a, 0]
        row3 = [1, 0, -1]
        return row1, row2, row3


class SprottI(DynSys):
    params = {"a": 0.2}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a):
        x, y, z = X
        xdot = -a * y
        ydot = x + z
        zdot = x + y**2 - z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a):
        x, y, z = X
        row1 = [0, -a, 0]
        row2 = [1, 0, 1]
        row3 = [1, 2 * y, -1]
        return row1, row2, row3


class SprottJ(DynSys):
    params = {}
    n_dim = 3
    @staticjit
    def _rhs(X, t):
        x, y, z = X
        xdot = 2 * z
        ydot = -2 * y + z
        zdot = -x + y + y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t):
        x, y, z = X
        row1 = [0, 0, 2]
        row2 = [0, -2, 1]
        row3 = [-1, 1 + 2 * y, 0]
        return row1, row2, row3


class SprottK(DynSys):
    params = {"a": 0.3}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a):
        x, y, z = X
        xdot = x * y - z
        ydot = x - y
        zdot = x + a * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a):
        x, y, z = X
        row1 = [y, x, -1]
        row2 = [1, -1, 0]
        row3 = [1, 0, a]
        return row1, row2, row3


class SprottL(DynSys):
    params = {"a": 0.9, "b": 3.9}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b):
        x, y, z = X
        xdot = y + b * z
        ydot = a * x**2 - y
        zdot = 1 - x
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b):
        x, y, z = X
        row1 = [0, 1, b]
        row2 = [2 * a * x, -1, 0]
        row3 = [-1, 0, 0]
        return row1, row2, row3


class SprottM(DynSys):
    params = {"a": 1.7}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a):
        x, y, z = X
        xdot = -z
        ydot = -(x**2) - y
        zdot = a + a * x + y
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a):
        x, y, z = X
        row1 = [0, 0, -1]
        row2 = [-2 * x, -1, 0]
        row3 = [a, 1, 0]
        return row1, row2, row3


class SprottN(DynSys):
    params = {}
    n_dim = 3
    @staticjit
    def _rhs(X, t):
        x, y, z = X
        xdot = -2 * y
        ydot = x + z**2
        zdot = 1 + y - 2 * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t):
        x, y, z = X
        row1 = [0, -2, 0]
        row2 = [1, 0, 2 * z]
        row3 = [0, 1, -2]
        return row1, row2, row3


class SprottO(DynSys):
    params = {"a": 2.7}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a):
        x, y, z = X
        xdot = y
        ydot = x - z
        zdot = x + x * z + a * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a):
        x, y, z = X
        row1 = [0, 1, 0]
        row2 = [1, 0, -1]
        row3 = [1 + z, a, x]
        return row1, row2, row3


class SprottP(DynSys):
    params = {"a": 2.7}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a):
        x, y, z = X
        xdot = a * y + z
        ydot = -x + y**2
        zdot = x + y
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a):
        x, y, z = X
        row1 = [0, a, 1]
        row2 = [-1, 2 * y, 0]
        row3 = [1, 1, 0]
        return row1, row2, row3


class SprottQ(DynSys):
    params = {"a": 3.1, "b": 0.5}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b):
        x, y, z = X
        xdot = -z
        ydot = x - y
        zdot = a * x + y**2 + b * z
        return (xdot, ydot, zdot)

    @staticjit
    def _jac(X, t, a, b):
        x, y, z = X
        row1 = [0, 0, -1]
        row2 = [1, -1, 0]
        row3 = [a, 2 * y, b]
        return row1, row2, row3


class SprottR(DynSys):
    params = {"a": 0.9, "b": 0.4}
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b):
        x, y, z = X
        xdot = a - y
        ydot = b + z
        zdot = x * y - z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b):
        x, y, z = X
        row1 = [0, -1, 0]
        row2 = [0, 0, 1]
        row3 = [y, x, -1]
        return row1, row2, row3


class SprottS(DynSys):
    params = {}
    n_dim = 3
    @staticjit
    def _rhs(X, t):
        x, y, z = X
        xdot = -x - 4 * y
        ydot = x + z**2
        zdot = 1 + x
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t):
        x, y, z = X
        row1 = [-1, -4, 0]
        row2 = [1, 0, 2 * z]
        row3 = [1, 0, 0]
        return row1, row2, row3


class SprottMore(DynSys):
    params = {}
    n_dim = 3
    @staticjit
    def _rhs(X, t):
        x, y, z = X
        xdot = y
        ydot = -x - np.sign(z) * y
        zdot = y**2 - np.exp(-(x**2))
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t):
        x, y, z = X
        row1 = [0, 1, 0]
        row2 = [-1, -np.sign(z), 0]
        row3 = [-2 * x * np.exp(-x ** 2), 2 * y, 0]
        return row1, row2, row3


class SprottJerk(DynSys):
    params = {
      "mu": 2.017
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, mu):
        x, y, z = X
        xdot = y
        ydot = z
        zdot = -x + y**2 - mu * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, mu):
        x, y, z = X
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-1, 2 * y, -mu]
        return row1, row2, row3

# endregion


class Arneodo(DynSys):
    params = {
      "a": -5.5,
      "b": 4.5,
      "c": 1.0,
      "d": -1.0
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c, d):
        x, y, z = X
        xdot = y
        ydot = z
        zdot = -a * x - b * y - c * z + d * x**3
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b, c, d):
        x, y, z = X
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-a + 3 * d * x**2, -b, -c]
        return row1, row2, row3


class Rucklidge(DynSys):
    params = {
      "a": 2.0,
      "b": 6.7
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b):
        x, y, z = X
        xdot = -a * x + b * y - y * z
        ydot = x
        zdot = -z + y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, b):
        x, y, z = X
        row1 = [-a, b - z, -y]
        row2 = [1, 0, 0]
        row3 = [0, 2 * y, -1]
        return row1, row2, row3


class HyperRossler(DynSys):
    params = {
      "a": 0.25,
      "b": 3.0,
      "c": 0.5,
      "d": 0.05
    }
    n_dim = 4
    @staticjit
    def _rhs(X, t, a, b, c, d):
        x, y, z, w = X
        xdot = -y - z
        ydot = x + a * y + w
        zdot = b + x * z
        wdot = -c * z + d * w
        return xdot, ydot, zdot, wdot
    @staticjit
    def _jac(X, t, a, b, c, d):
        x, y, z, w = X
        row1 = [0, -1, -1, 0]
        row2 = [1, a, 0, 1]
        row3 = [z, 0, x, 0]
        row4 = [0, 0, -c, d]
        return row1, row2, row3, row4


class HyperLorenz(DynSys):
    params = {
      "a": 10,
      "b": 2.667,
      "c": 28,
      "d": 1.1
    }
    n_dim = 4
    @staticjit
    def _rhs(X, t, a, b, c, d):
        x, y, z, w = X
        xdot = a * y - a * x + w
        ydot = -x * z + c * x - y
        zdot = -b * z + x * y
        wdot = d * w - x * z
        return xdot, ydot, zdot, wdot


class HyperYangChen(DynSys):
    params = {
      "a": 30,
      "b": 3,
      "c": 35,
      "d": 8
    }
    n_dim = 4
    @staticjit
    def _rhs(X, t, a=30, b=3, c=35, d=8):
        x, y, z, w = X
        xdot = a * y - a * x
        ydot = c * x - x * z + w
        zdot = -b * z + x * y
        wdot = -d * x
        return xdot, ydot, zdot, wdot


class HyperYan(DynSys):
    params = {
      "a": 37,
      "b": 3,
      "c": 26,
      "d": 38
    }
    n_dim = 4
    @staticjit
    def _rhs(X, t, a=37, b=3, c=26, d=38):
        x, y, z, w = X
        xdot = a * y - a * x
        ydot = (c - a) * x - x * z + c * y
        zdot = -b * z + x * y - y * z + x * z - w
        wdot = -d * w + y * z - x * z
        return xdot, ydot, zdot, wdot


class GuckenheimerHolmes(DynSys):
    params = {
      "a": 0.4,
      "b": 20.25,
      "c": 3,
      "d": 1.6,
      "e": 1.7,
      "f": 0.44
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, b, c, d, e, f):
        x, y, z = X
        xdot = a * x - b * y + c * z * x + d * z * x**2 + d * z * y**2
        ydot = a * y + b * x + c * z * y
        zdot = e - z**2 - f * x**2 - f * y**2 - a * z**3
        return xdot, ydot, zdot


class HenonHeiles(DynSys):
    params = {
      "lam": 1
    }
    n_dim = 4
    @staticjit
    def _rhs(X, t, lam):
        x, y, px, py = X
        xdot = px
        ydot = py
        pxdot = -x - 2 * lam * x * y
        pydot = -y - lam * x**2 + lam * y**2
        return xdot, ydot, pxdot, pydot

    @staticjit
    def _jac(X, t, lam):
        x, y, px, py = X
        row1 = [0, 0, 1, 0]
        row2 = [0, 0, 0, 1]
        row3 = [-1 - 2 * lam * y, -2 * lam * x, 0, 0]
        row4 = [-2 * lam * x, 2 * lam * y - 1, 0, 0]
        return row1, row2, row3, row4


class NoseHoover(DynSys):
    params = {
      "a": 1.5
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a):
        x, y, z = X
        xdot = y
        ydot = -x + y * z
        zdot = a - y**2
        return xdot, ydot, zdot


class RikitakeDynamo(DynSys):
    params = {
      "a": 1.0,
      "mu": 1.0
    }
    n_dim = 3
    @staticjit
    def _rhs(X, t, a, mu):
        x, y, z = X
        xdot = -mu * x + y * z
        ydot = -mu * y - a * x + x * z
        zdot = 1 - x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(X, t, a, mu):
        x, y, z = X
        row1 = [-mu, z, y]
        row2 = [-a + z, -mu, x]
        row3 = [-y, -x, 0]
        return row1, row2, row3
