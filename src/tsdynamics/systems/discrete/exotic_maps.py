import numpy as np

from tsdynamics.families import DiscreteMap


class Bogdanov(DiscreteMap):
    """Bogdanov map -- discrete-time normal form of the Bogdanov-Takens bifurcation.

    A planar quadratic map that arises as the Euler/time-discretisation of the
    Bogdanov-Takens (double-zero eigenvalue) unfolding, used to study the
    interplay of the saddle-node, Hopf and homoclinic bifurcations near a
    cusp. The map exhibits invariant circles, Arnold tongues and chaotic
    attractors as the parameters are varied.

    Parameters
    ----------
    eps : float
        Linear damping/detuning of the ``y`` recurrence (the Hopf parameter).
    k : float
        Strength of the quadratic ``x(x-1)`` nonlinearity.
    mu : float
        Coefficient of the mixed ``x*y`` term.
    """

    params = {"eps": 0.0, "k": 1.2, "mu": 0.0}
    dim = 2
    reference = "Bogdanov (1981), Selecta Math. Soviet. 1, 389-421"

    @staticmethod
    def _step(X, eps, k, mu):
        x, y = X
        yp = (1 + eps) * y + k * x * (x - 1) + mu * x * y
        xp = x + yp
        return xp, yp

    @staticmethod
    def _jacobian(X, eps, k, mu):
        # x' = x + y' inherits the y'-row derivatives
        x, y = X
        dyp_dx = k * (2 * x - 1) + mu * y
        dyp_dy = 1 + eps + mu * x
        row1 = [1 + dyp_dx, dyp_dy]
        row2 = [dyp_dx, dyp_dy]
        return row1, row2


# Classical compendium attractor: the Svensson map circulates as a "popular"
# trigonometric strange attractor (attributed to Johnny Svensson, documented on
# Paul Bourke's site and in fractal-art galleries) with no single primary
# academic source, so it is carried without a literature citation.
class Svensson(DiscreteMap):
    """Svensson attractor -- a sinusoidal planar strange-attractor map.

    A two-dimensional iterated map built from sines and cosines of the scaled
    coordinates that produces an intricate, often filamentary chaotic
    attractor. It belongs to the family of trigonometric "strange attractor"
    maps popular in generative art and used as visual test cases for attractor
    reconstruction and Lyapunov-exponent estimation.

    Parameters
    ----------
    a, b : float
        Angular frequencies scaling ``x`` and ``y`` inside the trig terms.
    c, d : float
        Amplitudes of the cosine (``c``) and sine (``d``) contributions.

    Notes
    -----
    Default parameters ``(a, b, c, d) = (1.5, -1.8, 1.6, 0.9)`` give one of the
    commonly illustrated chaotic attractors.

    This is a classical compendium ("strange attractor art") map with no single
    primary research source; it is intentionally carried without a citation.
    """

    params = {"a": 1.5, "b": -1.8, "c": 1.6, "d": 0.9}
    dim = 2

    @staticmethod
    def _step(X, a, b, c, d):
        x, y = X
        xp = d * np.sin(a * x) - np.sin(b * y)
        yp = c * np.cos(a * x) + np.cos(b * y)
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b, c, d):
        x, y = X
        row1 = [a * d * np.cos(a * x), -b * np.cos(b * y)]
        row2 = [-a * c * np.sin(a * x), -b * np.sin(b * y)]
        return row1, row2


# Classical compendium attractor: the Bedhead map circulates as a "popular"
# trigonometric strange attractor (attributed to Ivan Emrich, documented on Paul
# Bourke's site and in fractal-art galleries) with no single primary academic
# source, so it is carried without a literature citation.
class Bedhead(DiscreteMap):
    """Bedhead attractor -- a sinusoidal planar strange-attractor map.

    A two-dimensional iterated map combining a ``sin(x*y/b)`` cross term with
    additive sine/cosine terms; for suitable parameters it generates a
    delicate chaotic attractor with a characteristic swept, "bedhead" shape.
    Like the Svensson map it is a trigonometric strange-attractor map used as a
    visual benchmark for chaotic-attractor analysis.

    Parameters
    ----------
    a : float
        Phase coefficient in the ``cos(a*x - y)`` term.
    b : float
        Scale dividing the ``x*y`` product and the ``sin(y)`` term.

    Notes
    -----
    Default parameters ``(a, b) = (-0.67, 0.83)``, iterated from ``x = y = 1``,
    give the commonly illustrated attractor.

    This is a classical compendium ("strange attractor art") map with no single
    primary research source; it is intentionally carried without a citation.
    """

    params = {"a": -0.67, "b": 0.83}
    dim = 2

    @staticmethod
    def _step(X, a, b):
        x, y = X
        xp = np.sin(x * y / b) * y + np.cos(a * x - y)
        yp = x + np.sin(y) / b
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b):
        x, y = X
        row1 = [
            y**2 * np.cos(x * y / b) / b - a * np.sin(a * x - y),
            np.sin(x * y / b) + x * y * np.cos(x * y / b) / b + np.sin(a * x - y),
        ]
        row2 = [1, np.cos(y) / b]
        return row1, row2


class ZeraouliaSprott(DiscreteMap):
    """Zeraoulia-Sprott map -- a minimal 2-D rational chaotic map.

    A two-dimensional discrete map whose first component is a rational fraction
    ``-a*x / (1 + y**2)`` with a non-vanishing denominator. It was introduced
    as one of the simplest rational planar maps that still produces chaotic
    attractors, reached via a quasi-periodic route to chaos, and is studied for
    the boundedness of its attractors.

    Parameters
    ----------
    a : float
        Gain of the rational ``x``-recurrence (the primary control parameter).
    b : float
        Linear feedback coefficient of ``y`` in the second component.

    Notes
    -----
    Default parameters ``(a, b) = (2.7, 0.35)`` lie in the chaotic regime
    reported by Zeraoulia & Sprott.
    """

    params = {"a": 2.7, "b": 0.35}
    dim = 2
    reference = "Zeraoulia & Sprott (2011), Int. J. Bifurcation Chaos 21, 155-160"

    @staticmethod
    def _step(X, a, b):
        x, y = X
        xp = (-a * x) / (1 + y**2)
        yp = x + b * y
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b):
        x, y = X
        # Partial derivatives for the rational map
        df1_dx = -a / (1 + y**2)
        df1_dy = (2 * a * x * y) / (1 + y**2) ** 2
        df2_dx = 1.0
        df2_dy = b
        row1 = [df1_dx, df1_dy]
        row2 = [df2_dx, df2_dy]
        return row1, row2


class GumowskiMira(DiscreteMap):
    """Gumowski-Mira map -- a recurrence with a rational nonlinearity.

    A planar map introduced in the study of particle-accelerator beam dynamics
    that iterates a nonlinear function ``G(x) = a*x + 2*(1-a)*x**2 / (1+x**2)``.
    It is celebrated for producing an extraordinary variety of organic,
    ornamental attractors -- closed invariant curves, island chains and chaotic
    sea-and-island structures -- as the parameters are tuned.

    Parameters
    ----------
    a : float
        Coefficient controlling the shape of the rational nonlinearity ``G``.
    b : float
        Linear feedback gain mixing the previous ``y`` into the recurrence.
    """

    params = {"a": -1.1, "b": -0.2}
    dim = 2
    reference = "Gumowski & Mira (1980), Recurrences and Discrete Dynamic Systems"

    @staticmethod
    def _step(X, a, b):
        x, y = X
        fx = a * x + 2 * (1 - a) * x**2 / (1 + x**2)
        xp = b * y + fx
        fx1 = a * xp + 2 * (1 - a) * xp**2 / (1 + xp**2)
        yp = fx1 - x
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b):
        x, y = X

        fx = a * x + 2 * (1 - a) * x**2 / (1 + x**2)
        xp = b * y + fx

        dxdx = a + (4 * (1 - a) * x) / (1 + x**2) - (4 * (1 - a) * x**3) / (1 + x**2) ** 2

        dxdy = b

        # y' = G(x') - x  →  dy'/dx = G'(x')·dx'/dx - 1,  dy'/dy = G'(x')·b
        gprime_xp = a + (4 * (1 - a) * xp) / (1 + xp**2) - (4 * (1 - a) * xp**3) / (1 + xp**2) ** 2
        dydx = dxdx * gprime_xp - 1

        dydy = b * gprime_xp

        row1 = [dxdx, dxdy]
        row2 = [dydx, dydy]

        return row1, row2


class Hopalong(DiscreteMap):
    """Hopalong (Martin) attractor map -- a square-root strange-attractor map.

    Barry Martin's "Hopalong" map, popularised by A. K. Dewdney's Computer
    Recreations column. Its square-root-and-sign nonlinearity sprays iterates
    into layered, often quasi-symmetric chaotic attractors that were a staple
    of early home-computer fractal art.

    This implementation is a shifted variant: the step carries several ``- 1``
    offsets and a ``sign(x - 1)`` / ``b*x - 1 - c`` argument rather than the
    bare canonical ``x' = y - sign(x)*sqrt(|b*x - c|)``, ``y' = a - x``. The
    offsets are intentional and self-consistent (the hand Jacobian matches
    ``_step``).

    Parameters
    ----------
    a : float
        Additive constant in the ``y``-recurrence (sets the overall offset).
    b : float
        Scale inside the square-root argument.
    c : float
        Shift inside the square-root argument.
    """

    params = {"a": 3.1, "b": 2.5, "c": 4.2}
    dim = 2
    reference = "Dewdney (1986), Scientific American 255(3), 14-20"

    @staticmethod
    def _step(X, a, b, c):
        x, y = X
        xp = y - 1 - np.sqrt(np.abs(b * x - 1 - c)) * np.sign(x - 1)
        yp = a - x - 1
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b, c):
        x, y = X
        eps = 1e-30
        denom = np.sqrt(np.abs(b * x - 1 - c) + eps)
        j00 = -0.5 * b * np.sign(x - 1) * np.sign(b * x - 1 - c) / denom
        row1 = [j00, 1.0]
        row2 = [-1.0, 0.0]
        return row1, row2


class Pickover(DiscreteMap):
    """Pickover (Clifford) attractor -- a sinusoidal planar strange-attractor map.

    A two-dimensional iterated map of the form
    ``x' = sin(a*y) + c*cos(a*x)``, ``y' = sin(b*x) + d*cos(b*y)`` introduced
    in Clifford Pickover's explorations of computer-generated chaos. Like other
    trigonometric strange-attractor maps it converges to the same intricate
    attractor irrespective of the starting point, making it a popular subject
    in mathematical art and a benchmark for attractor visualisation.

    Parameters
    ----------
    a, b : float
        Angular frequencies scaling the coordinates inside the trig terms.
    c, d : float
        Amplitudes of the cosine contributions in the two components.

    Notes
    -----
    Default parameters ``(a, b, c, d) = (-1.4, 1.6, 1.0, 0.7)`` produce a
    commonly illustrated chaotic attractor.
    """

    params = {"a": -1.4, "b": 1.6, "c": 1.0, "d": 0.7}
    dim = 2
    reference = "Pickover (1990), Computers, Pattern, Chaos and Beauty (St. Martin's Press)"

    @staticmethod
    def _step(X, a, b, c, d):
        x, y = X
        xp = np.sin(a * y) + c * np.cos(a * x)
        yp = np.sin(b * x) + d * np.cos(b * y)
        return xp, yp

    @staticmethod
    def _jacobian(X, a, b, c, d):
        x, y = X
        row1 = [-a * c * np.sin(a * x), a * np.cos(a * y)]
        row2 = [b * np.cos(b * x), -b * d * np.sin(b * y)]
        return row1, row2
