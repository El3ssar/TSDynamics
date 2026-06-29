from tsdynamics.families import ContinuousSystem


class CoevolvingPredatorPrey(ContinuousSystem):
    """Eco-evolutionary predator-prey model with an evolving prey trait.

    A three-variable extension of competitive predator-prey dynamics in which the
    prey density ``x`` and predator density ``y`` are coupled to a mean prey trait
    ``alpha`` evolving on a density-dependent fitness landscape. Natural selection
    acting fast enough to feed back on the population dynamics drives the trait
    between stabilizing and disruptive regimes, and this coupling generates chaos
    even without external forcing.

    Parameters
    ----------
    a1, a2, a3 : float
        Interaction (predation / conversion) rate coefficients.
    b1, b2 : float
        Saturation constants of the trait- and prey-dependent functional responses.
    d1, d2 : float
        Prey and predator intrinsic death rates.
    delta : float
        Trait-asymmetry parameter of the fitness landscape.
    k1, k2, k4 : float
        Coupling strengths of the polynomial fitness landscape in the prey trait.
    vv : float
        Relative timescale of trait evolution versus population dynamics.

    Notes
    -----
    The default parameters lie in the chaotic regime reported by the authors.
    """

    reference = "Gilpin & Feldman (2017), PLoS Comput. Biol. 13, e1005644"
    doi = "10.1371/journal.pcbi.1005644"
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
        "vv": 0.33333,
    }
    dim = 3

    @staticmethod
    def _equations(Y, t, *, a1, a2, a3, b1, b2, d1, d2, delta, k1, k2, k4, vv):
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


class KawczynskiStrizhak(ContinuousSystem):
    """Kawczynski-Strizhak model of complex Belousov-Zhabotinsky oscillations.

    A three-variable polynomial reduction abstracting the chaotic and mixed-mode
    transient oscillations observed in the Belousov-Zhabotinsky reaction in a batch
    reactor. The fast cubic variable ``x`` is coupled to two slower recovery
    variables ``y`` and ``z``, reproducing the bursting and period-doubling routes
    to chaos seen in the chemistry.

    Parameters
    ----------
    beta : float
        Constant drive (offset) of the slow ``y`` recovery variable.
    gamma : float
        Timescale / gain of the fast cubic ``x`` nullcline.
    kappa : float
        Relaxation rate of the slowest variable ``z``.
    mu : float
        Bifurcation parameter setting the shape of the cubic nullcline.

    Notes
    -----
    The default parameters lie in the chaotic regime.
    """

    reference = "Strizhak & Kawczynski (1995), J. Phys. Chem. 99, 10830-10833"
    doi = "10.1021/j100027a024"
    params = {"beta": -0.4, "gamma": 0.49, "kappa": 0.2, "mu": 2.1}
    dim = 3

    @staticmethod
    def _equations(Y, t, *, beta, gamma, kappa, mu):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = gamma * y - gamma * x**3 + 3 * mu * gamma * x
        ydot = -2 * mu * x - y - z + beta
        zdot = kappa * x - kappa * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, beta, gamma, kappa, mu):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-3 * gamma * x**2 + 3 * mu * gamma, gamma, 0]
        row2 = [-2 * mu, -1, -1]
        row3 = [kappa, 0, -kappa]
        return row1, row2, row3


class Finance(ContinuousSystem):
    """Ma-Chen nonlinear finance system.

    A three-dimensional macroeconomic model whose state collects the interest rate
    ``x``, the investment demand ``y``, and the price index ``z``. The quadratic
    couplings between these variables produce Hopf bifurcations and a chaotic
    attractor, making it a canonical low-dimensional model of irregular economic
    dynamics.

    Parameters
    ----------
    a : float
        Savings amount (damping of the interest rate).
    b : float
        Per-unit investment cost (damping of investment demand).
    c : float
        Elasticity of demand of commercial markets (damping of the price index).

    Notes
    -----
    The default parameters lie in the chaotic regime.
    """

    reference = "Cai & Huang (2007), Int. J. Nonlinear Sci. 3, 235-241"
    params = {"a": 0.001, "b": 0.2, "c": 1.1}
    dim = 3

    @staticmethod
    def _equations(Y, t, *, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = (1 / b - a) * x + z + x * y
        ydot = -b * y - x**2
        zdot = -x - c * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [(1 / b - a) + y, x, 1]
        row2 = [-2 * x, -b, 0]
        row3 = [-1, 0, -c]
        return row1, row2, row3
