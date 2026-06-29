import numpy as np

from tsdynamics.families import DiscreteMap


# TODO(reference): unverified — needs a primary citation
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
    """

    params = {"mu": 0.95}
    dim = 1

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
    Baker's map, an area-preserving stretch-cut-stack map of the unit square.

    Modelled on a baker kneading dough, the map stretches the unit square to
    twice its width and half its height, cuts it in two, and stacks the halves
    back onto :math:`[0, 1]^2`. With ``alpha = 0.5`` it is the classic
    measure-preserving baker's transformation — uniformly hyperbolic, mixing,
    and conjugate to a two-sided Bernoulli shift, making it a canonical example
    in ergodic theory. For ``alpha`` other than 0.5 the fold is asymmetric (the
    generalized / asymmetric baker's map), expanding in :math:`x` while
    contracting unevenly in :math:`y`.

    Parameters
    ----------
    alpha : float
        Fold fraction ``0 < alpha < 1`` splitting the height of the square. The
        symmetric, area-preserving case is ``alpha = 0.5`` (the default).
    """

    params = {"alpha": 0.5}
    dim = 2
    reference = "Hopf (1937), Ergodentheorie (Springer, Berlin)"
    # Float doubling collapses the orbit onto the x = 0 discontinuity after
    # ~53 iterations, so finite differences always straddle the jump there.
    _jacobian_fd_check = False

    @staticmethod
    def _step(X, alpha):
        """
        Right-hand side of the Baker map.

        x, y: Current state variables
        alpha: Fraction determining the fold (0 < alpha < 1)

        Written branchlessly with ``np.where`` (rather than a Python ``if`` on
        the state) so the step traces to a straight-line tape and runs on the
        Rust engine.  On the attractor ``y in [0, 1)`` the single test
        ``y < alpha`` is equivalent to ``0 <= y < alpha``.
        """
        x, y = X
        lower = y < alpha
        xp = np.where(lower, (2 * x) % 1, (2 * x - 1) % 1)  # stretch / shift in x
        yp = np.where(lower, y / alpha, (y - alpha) / (1 - alpha))  # fold in y
        return xp, yp

    @staticmethod
    def _jacobian(X, alpha):
        x, y = X
        # Same branch test as ``_step`` (``y < alpha``, equivalent to
        # ``0 <= y < alpha`` on the attractor) so the hand Jacobian matches the
        # symbolic derivative of the lowered step.
        if y < alpha:
            row1 = [2, 0]
            row2 = [0, 1 / alpha]
        else:
            row1 = [2, 0]
            row2 = [0, 1 / (1 - alpha)]
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
    reference = "Adler & Rivlin (1964), Proc. Amer. Math. Soc. 15, 794-796"

    @staticmethod
    def _step(X, a):
        x = X
        return np.cos(a * np.arccos(x))

    @staticmethod
    def _jacobian(X, a):
        # chain rule: d/dx arccos(x) = -1/sqrt(1-x^2), the two minuses cancel
        x = X
        return [a * np.sin(a * np.arccos(x)) / np.sqrt(1 - x**2)]
