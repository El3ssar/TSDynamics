import numpy as np

from tsdynamics.families import DiscreteMap


# Classical map: the tent map is a textbook construction with no single
# originating paper; it is covered by any standard dynamical-systems text rather
# than attributed to a primary research source.
class Tent(DiscreteMap):
    r"""
    Tent map, a piecewise-linear unimodal map on the unit interval.

    The map :math:`x_{n+1} = \mu\,(1 - 2\,|x_n - 0.5|)` folds the interval
    :math:`[0, 1]` onto itself with a single peak at :math:`x = 0.5`. As the
    simplest piecewise-linear chaotic map it is a classical model for stretching-
    and-folding dynamics: each branch has constant slope :math:`\pm 2\mu`, so the
    Lyapunov exponent is :math:`\ln(2\mu)` wherever the orbit stays on the
    attractor. At the full-height value :math:`\mu = 1` the map is conjugate to
    the logistic map at :math:`r = 4` and to the Bernoulli shift, giving fully
    developed chaos with an absolutely continuous invariant measure.

    Parameters
    ----------
    mu : float
        Peak height (slope magnitude is :math:`2\mu`). Chaotic for
        :math:`\mu` near 1; the default ``0.95`` sits in the chaotic regime.

    Note
    ----
    A classical textbook map with no single originating paper; see any standard
    introduction to nonlinear dynamics (e.g. Strogatz, *Nonlinear Dynamics and
    Chaos*) rather than a primary research citation.
    """

    reference = "Classical map; see e.g. Strogatz, Nonlinear Dynamics and Chaos"
    params = {"mu": 0.95}
    dim = 1
    variables = ("x",)

    @staticmethod
    def _step(X, mu):
        x = X
        return mu * (1 - 2 * np.abs(x - 0.5))

    @staticmethod
    def _jacobian(X, mu):
        x = X
        if x < 0.5:
            return [2 * mu]
        else:
            return [-2 * mu]


class Baker(DiscreteMap):
    r"""
    Baker's map, the stretch-cut-stack transformation of the unit square.

    Modelled on a baker kneading dough, the map stretches the unit square to
    twice its width, cuts it at :math:`x = 1/2`, and stacks the right half on
    top of the left:

    .. math::

        x' = 2x \bmod 1, \qquad
        y' = \begin{cases}
            \alpha y                & x < 1/2 \\
            \alpha y + (1 - \alpha) & x \ge 1/2 .
        \end{cases}

    The :math:`x` direction is uniformly expanding with slope 2 and the
    :math:`y` direction uniformly contracting with slope :math:`\alpha`, so the
    Lyapunov exponents are exactly :math:`(\ln 2,\ \ln\alpha)` and the Jacobian
    determinant is the constant :math:`2\alpha`. With ``alpha = 0.5`` the map is
    the classic **measure-preserving** baker's transformation (:math:`|\det J| =
    1`) — uniformly hyperbolic, mixing, and conjugate to a two-sided Bernoulli
    shift, making it a canonical example in ergodic theory. For
    ``alpha < 0.5`` it is the dissipative (asymmetric) baker's map, whose
    attractor is the product of the unit interval with a Cantor set of
    dimension :math:`\ln 2 / \ln(1/\alpha)`.

    Parameters
    ----------
    alpha : float
        Contraction ratio of the fold, ``0 < alpha <= 0.5``. The symmetric,
        area-preserving case is ``alpha = 0.5`` (the default); smaller values
        give a dissipative map with a fractal attractor. (The two branch images
        overlap for ``alpha > 0.5``, which is no longer a baker's map.)

    Notes
    -----
    The ``x`` update is a doubling map wrapped just below 1 rather than exactly
    mod 1, for the same reason as :class:`~tsdynamics.systems.KaplanYorke` — see
    ``_step``.
    """

    params = {"alpha": 0.5}
    dim = 2
    variables = ("x", "y")
    reference = "Hopf (1937), Ergodentheorie (Springer, Berlin)"
    doi = "10.1007/978-3-642-86630-2"

    @staticmethod
    def _step(X, alpha):
        """
        Right-hand side of the Baker map.

        x, y: Current state variables
        alpha: Contraction ratio of the fold (0 < alpha <= 0.5)

        The canonical ``x`` fold is ``(2 * x) % 1``, but the pure doubling map
        drains a float mantissa one bit per step (``2 * x`` is an exact binary
        shift), so every orbit collapses onto the ``x = 0`` fixed point in ~52
        iterations and the chaos dies. Wrapping just below 1 injects a ~5e-8
        offset at each fold that keeps the orbit non-degenerate; the dynamics
        (and the exponents ``ln 2`` / ``ln alpha``) are unchanged to that
        tolerance. This is the same guard :class:`KaplanYorke` carries.

        The ``y`` branch is written with ``np.where`` (rather than a Python
        ``if`` on the state) so the step traces to a straight-line tape and runs
        on the Rust engine.
        """
        x, y = X
        xp = (2 * x) % 0.99999995  # stretch in x (wrapped just below 1)
        yp = np.where(x < 0.5, alpha * y, alpha * y + (1 - alpha))  # contract + stack
        return xp, yp

    @staticmethod
    def _jacobian(X, alpha):
        x, y = X
        # Constant on each branch: expand by 2 in x, contract by alpha in y.
        # The branch shift is additive, so the Jacobian is the same either side.
        row1 = [2, 0]
        row2 = [0, alpha]
        return row1, row2


class Circle(DiscreteMap):
    r"""
    Arnold's (standard) circle map of the unit circle onto itself.

    The map :math:`\theta_{n+1} = \theta_n + \omega + (k/2\pi)\,
    \sin(2\pi\theta_n) \pmod 1` is the paradigm for mode locking and the
    transition to chaos via quasiperiodicity. For :math:`k < 1` it is an
    invertible circle diffeomorphism whose rotation number locks onto rationals
    over Arnold tongues (the devil's-staircase structure). At :math:`k = 1` the
    map develops a cubic inflection and the tongues fill the parameter axis; for
    :math:`k > 1` it is non-invertible and can be chaotic. The default
    ``omega = 0.333``, ``k = 5.7`` sits well inside the chaotic regime.

    Parameters
    ----------
    omega : float
        Bare winding number (the rotation in the absence of coupling).
    k : float
        Nonlinearity / coupling strength; the critical value is ``k = 1``.

    Listed in ``params`` insertion order (``omega`` then ``k``).
    """

    params = {"omega": 0.333, "k": 5.7}
    dim = 1
    variables = ("theta",)
    reference = "Arnold (1965), Amer. Math. Soc. Transl. 46, 213-284"

    @staticmethod
    def _step(X, omega, k):
        theta = X
        thetap = theta + omega + (k / (2 * np.pi)) * np.sin(2 * np.pi * theta)
        return thetap % 1

    @staticmethod
    def _jacobian(X, omega, k):
        theta = X
        return [1 + k * np.cos(2 * np.pi * theta)]


class Chebyshev(DiscreteMap):
    r"""
    Chebyshev map, the Chebyshev polynomial iterated as a dynamical system.

    The map :math:`x_{n+1} = \cos(a\,\arccos x_n)` on :math:`[-1, 1]` is the
    degree-:math:`a` Chebyshev polynomial :math:`T_a(x)` when :math:`a` is an
    integer. For integer degree :math:`a \ge 2` it is an exact, strongly mixing
    map with the explicit invariant density
    :math:`1/(\pi\sqrt{1 - x^2})` and constant Lyapunov exponent
    :math:`\ln a`, which makes it a clean generator of fully developed chaos
    (the quadratic case :math:`a = 2` is conjugate to the logistic map at
    :math:`r = 4`). The ergodic and mixing properties were established by Adler
    and Rivlin (1964).

    Parameters
    ----------
    a : float
        Map degree; integer ``a >= 2`` gives chaos with Lyapunov exponent
        ``ln(a)``. The default ``a = 6.0`` is strongly chaotic.
    """

    params = {"a": 6.0}
    dim = 1
    variables = ("x",)
    reference = "Adler & Rivlin (1964), Proc. Amer. Math. Soc. 15, 794-796"
    doi = "10.1090/s0002-9939-1964-0202968-3"

    @staticmethod
    def _step(X, a):
        x = X
        return np.cos(a * np.arccos(x))

    @staticmethod
    def _jacobian(X, a):
        # chain rule: d/dx arccos(x) = -1/sqrt(1-x^2), the two minuses cancel
        x = X
        return [a * np.sin(a * np.arccos(x)) / np.sqrt(1 - x**2)]


__all__ = [
    "Baker",
    "Chebyshev",
    "Circle",
    "Tent",
]


def __dir__() -> list[str]:
    """Expose only the catalogue classes (``__all__``) to ``dir()`` / autocomplete.

    ``__all__`` governs ``import *`` and nothing else, so without this the module
    also offers every helper it imported — SymEngine's ``sin``/``cos``/``exp``,
    ``numpy`` — as though they were part of this library's surface.
    """
    return sorted(__all__)
