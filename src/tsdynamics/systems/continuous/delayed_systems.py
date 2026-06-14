from symengine import sin, tanh

from tsdynamics.families import DelaySystem


class MackeyGlass(DelaySystem):
    """
    Mackey-Glass (1977) delay differential equation.

    A canonical model of physiological control exhibiting a transition from
    periodic to chaotic behaviour as the delay ``tau`` increases.

    Parameters
    ----------
    beta, gamma : float
        Production and degradation rates.
    tau : float
        Delay.  Larger values increase attractor complexity.
    n : float
        Nonlinearity exponent.
    """

    params = {"beta": 0.2, "gamma": 0.1, "tau": 17.0, "n": 10.0}
    dim = 1
    variables = ("x",)
    reference = "Mackey & Glass (1977), Science 197, 287-289"
    known_lyapunov = {
        "n_positive": 1,
        "kwargs": {
            "n_exp": 1,
            "dt": 0.5,
            "burn_in": 100.0,
            "final_time": 1000.0,
            "rtol": 1e-4,
            "atol": 1e-4,
        },
        "source": "chaotic at tau = 17",
    }

    @staticmethod
    def _equations(y, t, *, beta, gamma, tau, n):
        x_tau = y(0, t - tau)
        return [beta * x_tau / (1.0 + x_tau**n) - gamma * y(0)]


class IkedaDelay(DelaySystem):
    params = {"c": 1.0, "mu": -20, "tau": 2.0, "x0": 0.0}
    dim = 1

    @staticmethod
    def _equations(Y, t, *, c, mu, tau, x0):
        return [mu * sin(Y(0, t - tau) - x0) - c * Y(0, t)]


class SprottDelay(DelaySystem):
    params = {"tau": 5.1}
    dim = 1

    @staticmethod
    def _equations(Y, t, *, tau):
        return [sin(Y(0, t - tau))]


class ScrollDelay(DelaySystem):
    params = {"alpha": 0.2, "beta": 0.2, "tau": 10.0}
    dim = 1

    @staticmethod
    def _equations(Y, t, *, alpha, beta, tau):
        xt = Y(0, t - tau)
        f = tanh(10 * xt)
        return [-alpha * xt + beta * f]


class PiecewiseCircuit(DelaySystem):
    params = {"alpha": 1.0, "beta": 1.0, "c": 2.24, "tau": 4.9}
    dim = 1

    @staticmethod
    def _equations(Y, t, *, alpha, beta, c, tau):
        xt = Y(0, t - tau)
        f = -((xt / c) ** 3) + 3 * xt / c
        return [-alpha * xt + beta * f]
