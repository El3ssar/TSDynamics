import numpy as np

from tsdynamics.families import DiscreteMap


class Logistic(DiscreteMap):
    r"""Logistic map — the archetypal 1-D route to chaos.

    The quadratic recurrence ``x_{n+1} = r x_n (1 - x_n)`` on the unit interval,
    introduced as a population-growth model. As the growth parameter ``r`` is
    increased it undergoes a period-doubling cascade (the universal Feigenbaum
    scenario) accumulating at ``r ≈ 3.5699`` and giving way to chaos interleaved
    with periodic windows.

    Parameters
    ----------
    r : float
        Growth rate, the single bifurcation parameter (meaningful on
        ``0 <= r <= 4``, which maps ``[0, 1]`` into itself). The default
        ``r = 3.9`` lies in the chaotic regime; ``r = 4`` is fully chaotic with
        Lyapunov exponent exactly ``ln 2``.
    """

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
    r"""Ricker map — a discrete single-species population model.

    The recurrence ``x_{n+1} = x_n exp(a - x_n)``, derived by Ricker from
    stock–recruitment relationships in fisheries. Unlike the logistic map its
    overcompensatory exponential nonlinearity keeps iterates non-negative for any
    state, so it does not escape to infinity. Increasing the intrinsic growth
    parameter ``a`` drives a period-doubling cascade into chaos.

    Parameters
    ----------
    a : float
        Intrinsic growth rate (the bifurcation parameter). Fixed points are
        stable for small ``a``; period doubling sets in near ``a = 2`` and chaos
        emerges around ``a >= 2.7``. The default ``a = 3.3`` is chaotic.
    """

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
    r"""Maynard Smith map — a 2-D second-order population recurrence.

    The invertible quadratic map ``x_{n+1} = y_n``,
    ``y_{n+1} = a y_n + b - x_n^2``, written in the delay form
    ``z_{n+1} = a z_n + b - z_{n-1}^2``. It arises as a discrete second-order
    density-dependent population model and exhibits a Neimark–Sacker (secondary
    Hopf) bifurcation, producing invariant closed curves, mode locking and chaos
    as the parameters vary.

    Parameters
    ----------
    a : float
        Damping / linear feedback coefficient on the current state.
    b : float
        Additive recruitment (forcing) term.

    Notes
    -----
    The default ``a = 0.87``, ``b = 0.75`` sits in a chaotic regime.
    """

    params = {"a": 0.87, "b": 0.75}
    dim = 2
    reference = "Maynard Smith (1968), Mathematical Ideas in Biology (Cambridge University Press)"

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
