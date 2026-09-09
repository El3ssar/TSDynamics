from symengine import Min

from tsdynamics.families import ContinuousSystem


class LotkaVolterra(ContinuousSystem):
    """Lotka-Volterra predator-prey equations - the classic planar ecology model.

    The original two-species model of coupled population oscillations: prey
    ``x`` grow exponentially and are eaten on encounter, predators ``y`` die
    off exponentially and are born from what they eat.

    .. code-block:: text

        x' = alpha x - beta x y
        y' = delta x y - gamma y

    The system is **conservative**, not dissipative: away from the axes it has
    the constant of motion ``V = delta x - gamma ln x + beta y - alpha ln y``,
    so the coexistence equilibrium ``(gamma/delta, alpha/beta)`` is a neutrally
    stable *center* surrounded by a one-parameter family of closed orbits - one
    through every initial condition, each with its own amplitude and period.
    There is no limit cycle and no attractor; the prey peak always leads the
    predator peak by a quarter cycle.  At the defaults the center sits at
    ``(4.0, 2.75)`` and the orbit through the default initial condition has
    period ~9.926.

    Because every orbit is closed, both Lyapunov exponents are **exactly zero**:
    one along the flow, and the other because the time average of the divergence
    ``(alpha - beta y) + (delta x - gamma)`` over a closed orbit is the average
    of ``d/dt [ln x + ln y]``, which vanishes over a period.

    Parameters
    ----------
    alpha : float
        Intrinsic per-capita growth rate of the prey.
    beta : float
        Predation rate - prey lost per predator-prey encounter.
    delta : float
        Conversion efficiency of consumed prey into new predators.
    gamma : float
        Per-capita death rate of the predators.

    Notes
    -----
    Formulated independently by Lotka (1920, cited here) and by Volterra
    (*Nature* **118**, 558-560, 1926; doi:10.1038/118558a0) to explain the
    post-war fluctuations in Adriatic fish catches.
    """

    reference = "Lotka (1920), Proc. Natl. Acad. Sci. U.S.A. 6, 410-415"
    doi = "10.1073/pnas.6.7.410"
    params = {"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4}
    dim = 2
    variables = ("x", "y")
    default_ic = [4.0, 1.5]
    #: Analytic: every orbit is closed, so the spectrum is exactly (0, 0) - one
    #: exponent along the flow and one from the vanishing average divergence
    #: (see the class docstring).  Not a measured value.
    known_lyapunov = {
        "spectrum": [0.0, 0.0],
        "atol": 0.02,
        "source": "analytic: closed orbits of a conservative planar flow",
        "kwargs": {"final_time": 2000.0, "dt": 0.05},
    }

    @staticmethod
    def _equations(Y, t, *, alpha, beta, delta, gamma):
        x, y = Y(0), Y(1)
        xdot = alpha * x - beta * x * y
        ydot = delta * x * y - gamma * y
        return xdot, ydot

    @staticmethod
    def _jacobian(Y, t, alpha, beta, delta, gamma):
        x, y = Y(0), Y(1)
        row1 = [alpha - beta * y, -beta * x]
        row2 = [delta * y, delta * x - gamma]
        return row1, row2


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
    variables = ("x", "y", "alpha")

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
    variables = ("x", "y", "z")

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
    variables = ("x", "y", "z")

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


class MacArthur(ContinuousSystem):
    r"""MacArthur consumer–resource model (five species, five resources).

    A ten-dimensional consumer–resource competition model in which five
    consumer species ``N_i`` grow on five substitutable resources ``R_j``
    following Liebig's law of the minimum — each species' per-capita growth is
    set by its single scarcest resource:

    .. code-block:: text

        N_i' = N_i (mu_i - m)
        R_j' = d (S_j - R_j) - sum_i c_ji mu_i N_i
        mu_i = min_j [ r R_j / (K_ji + R_j) ]

    With the phytoplankton parameters of Huisman & Weissing the model exhibits
    sustained chaotic competition of more species than resources ("the paradox
    of the plankton").  State is ``[N_1..N_5, R_1..R_5]``.

    Parameters
    ----------
    d : float
        Resource turnover (supply/dilution) rate.
    m : float
        Consumer mortality rate.
    r : float
        Maximum per-capita resource-limited growth rate.
    """

    reference = "MacArthur (1969), Proc. Natl. Acad. Sci. USA 64, 1369-1371"
    doi = "10.1073/pnas.64.4.1369"
    params = {"d": 0.25, "m": 0.25, "r": 1.0}
    dim = 10
    variables = ("N1", "N2", "N3", "N4", "N5", "R1", "R2", "R3", "R4", "R5")
    default_ic = [
        9.43233,
        18.83865,
        34.37572,
        15.08545,
        42.22843,
        0.17031,
        0.10144,
        0.3148,
        0.21254,
        0.22599,
    ]
    #: Consumption matrix c[j][i], half-saturation constants K[j][i], and the
    #: resource supply concentrations S[j] (Huisman & Weissing 1999).
    _C = (
        (0.04, 0.04, 0.07, 0.04, 0.04),
        (0.08, 0.08, 0.08, 0.1, 0.08),
        (0.1, 0.1, 0.1, 0.1, 0.14),
        (0.05, 0.03, 0.03, 0.03, 0.03),
        (0.07, 0.09, 0.07, 0.07, 0.07),
    )
    _K = (
        (0.39, 0.34, 0.3, 0.24, 0.23),
        (0.22, 0.39, 0.34, 0.3, 0.27),
        (0.27, 0.22, 0.39, 0.34, 0.3),
        (0.3, 0.24, 0.22, 0.39, 0.34),
        (0.34, 0.3, 0.22, 0.2, 0.39),
    )
    _S = (6.0, 10.0, 14.0, 4.0, 9.0)

    @staticmethod
    def _equations(Y, t, *, d, m, r):
        c, kmat, s = MacArthur._C, MacArthur._K, MacArthur._S
        nn = [Y(i) for i in range(5)]
        rr = [Y(5 + j) for j in range(5)]
        # Liebig minimum: species i is limited by its scarcest resource.
        mu = [Min(*[r * rr[j] / (kmat[j][i] + rr[j]) for j in range(5)]) for i in range(5)]
        nndot = [nn[i] * (mu[i] - m) for i in range(5)]
        rrdot = [
            d * (s[j] - rr[j]) - sum(c[j][i] * mu[i] * nn[i] for i in range(5)) for j in range(5)
        ]
        return tuple(nndot + rrdot)
