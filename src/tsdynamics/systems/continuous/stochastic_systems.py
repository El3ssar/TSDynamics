"""Built-in stochastic differential equations (diagonal-Itô SDEs).

The first three catalogue members of the ``sde`` family — the canonical
processes every stochastic-dynamics course opens with:

* :class:`OrnsteinUhlenbeck` — additive-noise mean reversion;
* :class:`GeometricBrownianMotion` — multiplicative noise, the asset-price model;
* :class:`DoubleWell` — gradient flow in a bistable potential (Kramers escape).

Each is a :class:`~tsdynamics.families.StochasticSystem`: a symbolic
``_drift`` (the deterministic skeleton, like an ODE's ``_equations``) plus a
symbolic ``_diffusion`` (one diagonal noise coefficient per component), so the
SDE is ``dX_k = f_k(X, t) dt + g_k(X, t) dW_k`` with independent ``dW_k``.
"""

from tsdynamics.families import StochasticSystem


class OrnsteinUhlenbeck(StochasticSystem):
    r"""Ornstein-Uhlenbeck process — additive-noise mean reversion.

    The canonical mean-reverting diffusion: a linear restoring drift pulls the
    state back towards the long-run mean ``mu`` at rate ``theta`` while constant
    (additive) Gaussian noise of intensity ``sigma`` perturbs it,

    .. math::

        dX = \theta\,(\mu - X)\,dt + \sigma\,dW .

    It is the continuous-time analogue of an AR(1) process and the only
    non-trivial diffusion that is simultaneously Gaussian, Markovian, and
    stationary. Its stationary distribution is normal with mean ``mu`` and
    variance ``sigma**2 / (2*theta)``, and the autocovariance decays as
    ``exp(-theta*|s|)`` — the model behind everything from interest-rate
    (Vasicek) dynamics to the velocity of a Brownian particle.

    Parameters
    ----------
    theta : float
        Mean-reversion rate (stiffness of the restoring drift); must be positive
        for a stationary process.
    mu : float
        Long-run mean the process reverts to.
    sigma : float
        Noise intensity (the constant diagonal diffusion coefficient).
    """

    reference = "Uhlenbeck & Ornstein (1930), Phys. Rev. 36, 823-841"
    doi = "10.1103/PhysRev.36.823"
    params = {"theta": 1.0, "mu": 0.0, "sigma": 0.3}
    dim = 1
    variables = ("x",)
    default_ic = [2.0]

    @staticmethod
    def _drift(y, t, theta, mu, sigma):
        return [theta * (mu - y(0))]

    @staticmethod
    def _diffusion(y, t, theta, mu, sigma):
        return [sigma]


class GeometricBrownianMotion(StochasticSystem):
    r"""Geometric Brownian motion — multiplicative noise, the asset-price model.

    A positive process whose drift and diffusion both scale with the state, so
    its *logarithm* is a Brownian motion with drift,

    .. math::

        dX = \mu\,X\,dt + \sigma\,X\,dW .

    Starting from ``X_0 > 0`` the path stays strictly positive, and the
    multiplicative noise makes the relative (percentage) change stationary —
    which is why it is the standard model of a stock price and the asset process
    underlying the Black-Scholes option theory. The mean grows as
    ``X_0 * exp(mu*t)`` while the median grows as
    ``X_0 * exp((mu - sigma**2/2)*t)``; for ``mu < sigma**2/2`` almost every path
    decays to zero even though the mean diverges.

    Parameters
    ----------
    mu : float
        Drift rate (expected log-growth plus the ``sigma**2/2`` Itô correction).
    sigma : float
        Volatility — the multiplicative diffusion coefficient.
    """

    reference = "Osborne (1959), Oper. Res. 7, 145-173"
    doi = "10.1287/opre.7.2.145"
    params = {"mu": 0.1, "sigma": 0.3}
    dim = 1
    variables = ("x",)
    default_ic = [1.0]

    @staticmethod
    def _drift(y, t, mu, sigma):
        return [mu * y(0)]

    @staticmethod
    def _diffusion(y, t, mu, sigma):
        return [sigma * y(0)]


class DoubleWell(StochasticSystem):
    r"""Double-well diffusion — noise-driven switching (Kramers escape).

    Overdamped gradient flow in the symmetric bistable potential
    ``U(x) = -a*x**2/2 + b*x**4/4``, whose drift ``-U'(x)`` has two stable
    equilibria at ``x = ±sqrt(a/b)`` separated by an unstable barrier at the
    origin,

    .. math::

        dX = (a\,X - b\,X^{3})\,dt + \sigma\,dW .

    Without noise the state relaxes into whichever well it starts in; additive
    noise lets it occasionally hop the barrier, so the long-time dynamics is a
    random telegraph between the two wells. The mean residence time follows
    Kramers' law — exponential in the barrier-height-to-noise ratio
    (``~ exp(2*ΔU/sigma**2)``) — making this the textbook model of
    noise-induced transitions, stochastic resonance, and metastability.

    Parameters
    ----------
    a : float
        Curvature of the potential at the origin (sets the barrier and the
        ``±sqrt(a/b)`` well locations).
    b : float
        Quartic stiffness confining the wells.
    sigma : float
        Noise intensity (the constant diagonal diffusion coefficient).
    """

    reference = "Kramers (1940), Physica 7, 284-304"
    doi = "10.1016/S0031-8914(40)90098-2"
    params = {"a": 1.0, "b": 1.0, "sigma": 0.5}
    dim = 1
    variables = ("x",)
    default_ic = [-1.0]

    @staticmethod
    def _drift(y, t, a, b, sigma):
        return [a * y(0) - b * y(0) ** 3]

    @staticmethod
    def _diffusion(y, t, a, b, sigma):
        return [sigma]
