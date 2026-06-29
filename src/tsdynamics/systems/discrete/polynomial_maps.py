import numpy as np

from tsdynamics.families import DiscreteMap


class Gauss(DiscreteMap):
    r"""
    Gauss (Gaussian / "mouse") map :math:`x' = e^{-a x^2} + b`.

    A noninvertible one-dimensional iterated map built from the Gaussian
    bell curve.  Sweeping the shift parameter ``b`` produces a rich
    bifurcation structure — period doubling, reverse period doubling,
    period adding and chaos — whose diagram famously resembles a mouse,
    earning it the nickname *mouse map*.  Unlike the logistic map its
    bifurcation tree is two-sided (forward and reverse cascades), making it a
    standard textbook example of non-unimodal one-dimensional chaos.

    Parameters
    ----------
    a : float
        Width of the Gaussian (larger ``a`` ⇒ narrower peak).
    b : float
        Vertical shift controlling the bifurcation cascade.  ``a=4.9,
        b=-0.5`` lies in a chaotic regime.
    """

    params = {"a": 4.9, "b": -0.5}
    dim = 1
    reference = "Hilborn (2000), Chaos and Nonlinear Dynamics, 2nd ed. (Oxford University Press)"

    @staticmethod
    def _step(X, a, b):
        x = X
        return np.exp(-a * x**2) + b

    @staticmethod
    def _jacobian(X, a, b):
        x = X
        return [-2 * a * x * np.exp(-a * x**2)]


class DeJong(DiscreteMap):
    r"""
    Peter de Jong attractor map.

    .. math::

        x' &= \sin(a\,y) - \cos(b\,x) \\
        y' &= \sin(c\,x) - \cos(d\,y)

    A four-parameter trigonometric map of the plane whose orbits fill out
    intricate, often visually striking strange attractors.  It is iterated from
    a single seed point and the accumulated orbit is plotted as a point set;
    different ``(a, b, c, d)`` tuples yield dramatically different filigree
    structures, which is why it is a staple of generative / aesthetic
    chaos.  It was popularised by Peter de Jong through A. K. Dewdney's
    *Computer Recreations* column.

    Parameters
    ----------
    a, b, c, d : float
        Map parameters; each tuple selects a distinct attractor geometry.
    """

    params = {"a": 1.641, "b": 1.902, "c": 0.316, "d": 1.525}
    dim = 2
    reference = "Dewdney (1987), Scientific American 257(1), 108-111"

    @staticmethod
    def _step(X, a, b, c, d):
        x, y = X
        xp = np.sin(a * y) - np.cos(b * x)
        yp = np.sin(c * x) - np.cos(d * y)
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b, c, d):
        x, y = X
        row1 = [b * np.sin(b * x), a * np.cos(a * y)]
        row2 = [c * np.cos(c * x), d * np.sin(d * y)]
        return row1, row2


class KaplanYorke(DiscreteMap):
    r"""
    Kaplan–Yorke map.

    .. math::

        x' &= 2x \bmod 1 \\
        y' &= \alpha\,y + \cos(4\pi x)

    A two-dimensional map driving a contracting ``y`` coordinate by a chaotic
    doubling map in ``x``.  Kaplan and Yorke introduced it as a concrete
    example motivating the Lyapunov (Kaplan–Yorke) dimension: ``x`` carries the
    Bernoulli doubling map (Lyapunov exponent ``ln 2``) while ``y`` relaxes at
    rate ``ln α``, so the attractor's fractal dimension follows directly from
    the two exponents.

    Parameters
    ----------
    alpha : float
        Contraction rate of the ``y`` coordinate (``0 < alpha < 1``); smaller
        ``alpha`` gives stronger transverse contraction.

    Notes
    -----
    The ``x`` update is implemented as a doubling map wrapped just below 1 to
    avoid mantissa drain to the ``x=0`` fixed point; the dynamics and Lyapunov
    exponent ``ln 2`` are unchanged to that tolerance (see ``_step``).
    """

    params = {"alpha": 0.2}
    dim = 2
    reference = (
        "Kaplan & Yorke (1979), Functional Differential Equations and "
        "Approximation of Fixed Points, Lecture Notes in Mathematics 730, 204-227"
    )

    @staticmethod
    def _step(X, alpha):
        x, y = X
        # Doubling map on the unit circle. The canonical wrap is mod 1, but the
        # pure doubling map x -> frac(2x) drains a float mantissa one bit per
        # step (2x is an exact shift) and collapses to the x=0 fixed point in
        # ~52 iterations, killing the chaos. Wrapping just below 1 injects a
        # ~5e-8 offset at each fold that keeps the orbit non-degenerate; the
        # dynamics (and Lyapunov exponent ln 2) are unchanged to that tolerance.
        xp = (2 * x) % 0.99999995
        yp = alpha * y + np.cos(4 * np.pi * x)
        return xp, yp

    @staticmethod
    def _jacobian(X, alpha):
        x, y = X
        row1 = [2, 0]
        row2 = [-4 * np.pi * np.sin(4 * np.pi * x), alpha]
        return row1, row2
