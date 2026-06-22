import numpy as np

from tsdynamics.families import DiscreteMap


class Tent(DiscreteMap):
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
    params = {"alpha": 0.5}
    dim = 2
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
    """
    Arnold's circle map.

    Parameters ``omega`` (winding number) and ``k`` (nonlinearity strength), in
    that order (matching ``params`` insertion order).
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
    params = {"a": 6.0}
    dim = 1

    @staticmethod
    def _step(X, a):
        x = X
        return np.cos(a * np.arccos(x))

    @staticmethod
    def _jacobian(X, a):
        # chain rule: d/dx arccos(x) = -1/sqrt(1-x^2), the two minuses cancel
        x = X
        return [a * np.sin(a * np.arccos(x)) / np.sqrt(1 - x**2)]
