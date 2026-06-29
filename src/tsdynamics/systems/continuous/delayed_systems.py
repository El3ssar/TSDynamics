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
    reference = "Glass & Mackey (1979), Ann. N.Y. Acad. Sci. 316, 214-235"
    doi = "10.1111/j.1749-6632.1979.tb29471.x"
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
    """
    Ikeda (1979) time-delay optical-bistability equation.

    A scalar delay differential equation derived by Ikeda from the
    Maxwell-Bloch equations for a nonlinear absorbing medium of two-level
    atoms in a passive ring cavity driven by a constant input.  The sinusoidal
    delayed feedback makes it a foundational model of "optical turbulence" --
    deterministic chaos in optics -- and a benchmark high-dimensional chaotic
    delay system.

    Parameters
    ----------
    c : float
        Linear relaxation (damping) rate of the cavity field.
    mu : float
        Feedback gain; larger magnitude drives the cascade to chaos.
    tau : float
        Round-trip feedback delay.
    x0 : float
        Phase offset (detuning) of the nonlinear feedback term.
    """

    params = {"c": 1.0, "mu": -20, "tau": 2.0, "x0": 0.0}
    dim = 1
    reference = "Ikeda & Matsumoto (1987), Physica D 29, 223-235"
    doi = "10.1016/0167-2789(87)90058-3"

    @staticmethod
    def _equations(Y, t, *, c, mu, tau, x0):
        return [mu * sin(Y(0, t - tau) - x0) - c * Y(0, t)]


class SprottDelay(DelaySystem):
    """
    Sprott (2007) simplest chaotic delay differential equation.

    The scalar equation ``x' = sin(x(t - tau))``, proposed by Sprott as one of
    the simplest chaotic delay differential equations: a single sinusoidal
    nonlinearity of the delayed state with no explicit damping term.  As the
    delay grows it passes through a limit cycle, period doubling, and the onset
    of chaos near ``tau ~ 5``, and is prototypical of many high-dimensional
    chaotic systems (it also exhibits chaotic diffusion).

    Parameters
    ----------
    tau : float
        Feedback delay; the sole control parameter.  Chaos sets in around
        ``tau ~ 5`` (default ``tau = 5.1``).
    """

    params = {"tau": 5.1}
    dim = 1
    reference = "Sprott (2007), Physics Letters A 366, 397-402"
    doi = "10.1016/j.physleta.2007.01.083"

    @staticmethod
    def _equations(Y, t, *, tau):
        return [sin(Y(0, t - tau))]


class ScrollDelay(DelaySystem):
    """
    Scalar time-delay chaotic oscillator with hyperbolic-tangent feedback.

    A first-order delay differential equation with linear decay of the delayed
    state plus a saturating ``tanh`` nonlinearity of the same delayed state.
    Such scalar delay feedback systems possess an infinite-dimensional phase
    space and can generate scroll-type chaotic attractors despite their single
    state variable.

    Parameters
    ----------
    alpha : float
        Linear decay rate of the delayed state.
    beta : float
        Gain of the saturating ``tanh`` feedback term.
    tau : float
        Feedback delay; larger values raise the attractor dimension.
    """

    params = {"alpha": 0.2, "beta": 0.2, "tau": 10.0}
    dim = 1
    reference = "Driver (1977), Ordinary and Delay Differential Equations, Springer"
    doi = "10.1007/978-1-4684-9467-9_5"

    @staticmethod
    def _equations(Y, t, *, alpha, beta, tau):
        xt = Y(0, t - tau)
        f = tanh(10 * xt)
        return [-alpha * xt + beta * f]


class PiecewiseCircuit(DelaySystem):
    """
    Scalar time-delay chaotic oscillator with cubic feedback.

    A first-order delay differential equation with linear decay of the delayed
    state plus a cubic feedback nonlinearity ``-(x/c)**3 + 3 x/c`` of the same
    delayed state -- a smooth one-hump characteristic of the kind used in
    delay-feedback chaotic circuits.  As a scalar delay system it carries an
    infinite-dimensional phase space and produces chaotic (scroll-like)
    attractors.

    Parameters
    ----------
    alpha : float
        Linear decay rate of the delayed state.
    beta : float
        Gain of the cubic feedback term.
    c : float
        Scale of the cubic nonlinearity (sets the feedback's hump width).
    tau : float
        Feedback delay.
    """

    params = {"alpha": 1.0, "beta": 1.0, "c": 2.24, "tau": 4.9}
    dim = 1
    reference = "Tamasevicius, Mykolaitis & Bumeliene (2006), Electron. Lett. 42, 13"
    doi = "10.5755/j01.eie.21.5.13324"

    @staticmethod
    def _equations(Y, t, *, alpha, beta, c, tau):
        xt = Y(0, t - tau)
        f = -((xt / c) ** 3) + 3 * xt / c
        return [-alpha * xt + beta * f]
