from symengine import cosh, exp, sign, sinh, tanh

from tsdynamics.families import ContinuousSystem


class Sakarya(ContinuousSystem):
    """Sakarya system — a Lorenz-family three-dimensional chaotic flow.

    A six-term autonomous quadratic system drawn from the general Lorenz system
    family, introduced (and realised as an electronic circuit) by Pehlivan and
    Uyaroğlu. Its two quadratic cross-product nonlinearities (``y*z`` and
    ``x*z``) produce a butterfly-like attractor reminiscent of the Lorenz and
    Chen systems.

    Parameters
    ----------
    a, b, c : float
        Linear self-feedback gains on the ``x``, ``y`` and ``z`` channels.
    h, p : float
        Linear cross-coupling gains between ``x`` and ``y``.
    q, r, s : float
        Strengths of the quadratic cross-product terms ``x*z``, ``x*y`` and
        ``y*z`` respectively.

    Notes
    -----
    Chaotic at the default parameters.
    """

    reference = "Li et al. (2015), IEICE Electron. Express 12(4), 20141116"
    doi = "10.1587/elex.12.20141116"
    params = {
        "a": -1.0,
        "b": 1.0,
        "c": 1.0,
        "h": 1.0,
        "p": 1.0,
        "q": 0.4,
        "r": 0.3,
        "s": 1.0,
    }
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, c, h, p, q, r, s):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * x + h * y + s * y * z
        ydot = -b * y - p * x + q * x * z
        zdot = c * z - r * x * y
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c, h, p, q, r, s):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [a, h + s * z, s * y]
        row2 = [-p + q * z, -b, q * x]
        row3 = [-r * y, -r * x, c]
        return row1, row2, row3


class Bouali2(ContinuousSystem):
    """Bouali economic-cycle chaotic system (second parameter regime).

    A three-dimensional flow obtained by adding a feedback loop to an extended
    Van der Pol oscillator, proposed by Bouali as an idealised macroeconomic
    model of business cycles. The 2-D oscillator core (``x``, ``y``) supplies
    the relaxation cycle and the ``z`` feedback channel drives it chaotic for
    suitable gains, producing a stretched-loop strange attractor. These defaults
    are the second regime (weak, slow ``z`` feedback with ``bb = c = 0``); the
    :class:`Bouali` sibling holds the first regime (strong ``x*z`` feedback).

    Parameters
    ----------
    a : float
        Self-excitation gain of the oscillator core.
    y0 : float
        Reference (target) level of the ``x`` channel.
    b : float
        Feedback gain of the ``z`` channel into ``x``.
    g : float
        Nonlinear damping coefficient of the ``y`` channel.
    m, bb, c : float
        Gains of the ``z`` feedback loop (drive, ``x*z`` cross-product, and
        linear decay respectively).

    Notes
    -----
    Chaotic at the default parameters.
    """

    reference = "Bouali (1999), Int. J. Bifurcation Chaos 9, 745-756"
    doi = "10.1142/s0218127499000535"
    params = {"a": 3.0, "b": 2.2, "bb": 0.0, "c": 0.0, "g": 1.0, "m": -0.0026667, "y0": 1.0}
    dim = 3
    variables = ("x", "y", "z")
    default_ic = [-0.7939, 1.3618, -0.0306]

    @staticmethod
    def _equations(Y, t, *, a, b, bb, c, g, m, y0):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * y0 * x - a * x * y - b * z
        ydot = -g * y + g * y * x**2
        zdot = -1.5 * m * x + m * bb * x * z - c * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, bb, c, g, m, y0):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [a * y0 - a * y, -a * x, -b]
        row2 = [2 * g * y * x, g * x**2 - g, 0]
        row3 = [-1.5 * m + m * bb * z, 0, m * bb * x - c]
        return row1, row2, row3


class LuChenCheng(ContinuousSystem):
    """Lü–Chen–Cheng generalized Lorenz-like system with constant input.

    A three-dimensional quadratic flow from the generalized Lorenz-like family of
    Lü, Chen and Cheng, which contains the Lorenz- and Chen-type attractors as
    special cases and stays chaotic under a constant forcing input. This
    parameterisation uses the harmonic combination ``-(a*b)/(a+b)*x`` on the ``x``
    channel together with a constant forcing ``c``.

    Parameters
    ----------
    a, b : float
        Linear self-feedback gains of the ``y`` and ``z`` channels; together
        they set the ``x``-channel decay ``-(a*b)/(a+b)``.
    c : float
        Constant forcing on the ``x`` channel.

    Notes
    -----
    Chaotic at the default parameters.
    """

    reference = "Lü, Chen & Cheng (2004), Int. J. Bifurcation Chaos 14, 1507-1537"
    doi = "10.1142/s021812740401014x"
    params = {"a": -10, "b": -4, "c": 18.1}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -(a * b) / (a + b) * x - y * z + c
        ydot = a * y + x * z
        zdot = b * z + x * y
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-(a * b) / (a + b), -z, -y]
        row2 = [z, a, x]
        row3 = [y, x, b]
        return row1, row2, row3


class LuChen(ContinuousSystem):
    """Lü system — the chaotic attractor "coined" between Lorenz and Chen.

    A three-dimensional autonomous quadratic flow that sits at the transition
    between the Lorenz and the Chen attractors, sharing their Lorenz-like
    structure (two quadratic cross-product terms) but belonging to neither.
    Introduced by Lü and Chen; widely known simply as the Lü attractor.

    Parameters
    ----------
    a : float
        Prandtl-like coupling gain of the ``x``–``y`` channel.
    c : float
        Self-feedback gain of the ``y`` channel.
    b : float
        Linear decay rate of the ``z`` channel.

    Notes
    -----
    Chaotic at the default parameters (a=36, b=3, c=18).
    """

    reference = "Lü & Chen (2002), Int. J. Bifurcation Chaos 12, 659-661"
    doi = "10.1142/s0218127402004620"
    params = {"a": 36, "b": 3, "c": 18}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * y - a * x
        ydot = -x * z + c * y
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, a, 0]
        row2 = [-z, c, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class QiChen(ContinuousSystem):
    """Qi four-wing chaotic system.

    A three-dimensional quadratic autonomous flow, due to Qi, Chen, Du, Chen and
    Yuan, that carries an extra ``y*z`` cross-product term in the ``x`` equation
    beyond the Lorenz/Chen template. With this term the system has multiple
    equilibria and produces a four-wing chaotic attractor, distinguishing it from
    the two-wing Lorenz-family flows.

    Parameters
    ----------
    a : float
        Coupling gain of the ``x``–``y`` channel.
    c : float
        Forcing gain of the ``x`` channel into ``y``.
    b : float
        Linear decay rate of the ``z`` channel.

    Notes
    -----
    Chaotic at the default parameters, exhibiting a four-wing attractor.
    """

    reference = "Qi et al. (2008), Chaos Solitons Fractals 38, 705-721"
    doi = "10.1016/j.chaos.2006.09.012"
    params = {"a": 38, "b": 2.666, "c": 80}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * y - a * x + y * z
        ydot = c * x + y - x * z
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, a + z, y]
        row2 = [c - z, 1, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class ZhouChen(ContinuousSystem):
    """Zhou–Chen three-dimensional chaotic system.

    A three-dimensional quadratic autonomous flow carrying two quadratic
    cross-product nonlinearities (``y*z`` in both the ``x`` and ``y`` equations)
    plus an ``x*z`` and ``x*y`` term, producing a chaotic attractor at the
    default parameters.

    Parameters
    ----------
    a, c, e : float
        Linear self-feedback gains on the ``x``, ``y`` and ``z`` channels.
    b : float
        Linear ``y``-into-``x`` coupling gain.
    d : float
        Strength of the ``y*z`` cross-product term in the ``y`` equation.

    Notes
    -----
    Chaotic at the default parameters.
    """

    reference = "Zhou & Chen (2004), Int. J. Bifurcation Chaos"
    doi = "10.1142/s0218127404010175"
    params = {"a": 2.97, "b": 0.15, "c": -3.0, "d": 1, "e": -8.78}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, e):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * x + b * y + y * z
        ydot = c * y - x * z + d * y * z
        zdot = e * z - x * y
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c, d, e):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [a, b + z, y]
        row2 = [-z, c + d * z, -x + d * y]
        row3 = [-y, -x, e]
        return row1, row2, row3


class BurkeShaw(ContinuousSystem):
    """Burke–Shaw system — a symmetric Lorenz-class chaotic flow.

    A three-dimensional autonomous flow derived from the Lorenz system by Shaw,
    sharing the same broad organisation but rearranged so the two quadratic
    cross-product terms (``x*z`` and ``x*y``) are scaled by a common coupling
    ``n``. The attractor has the rotational symmetry of a four-branch template
    and is a standard test case for information-flow / entropy studies.

    Parameters
    ----------
    n : float
        Common coupling/timescale parameter scaling the linear and quadratic
        cross terms.
    e : float
        Constant forcing on the ``z`` channel.

    Notes
    -----
    Chaotic at the default parameters (n=10, e=13).
    """

    reference = "Shaw (1981), Z. Naturforsch. A 36, 80-112"
    doi = "10.1515/zna-1981-0115"
    params = {"e": 13, "n": 10}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, e, n):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -n * x - n * y
        ydot = y - n * x * z
        zdot = n * x * y + e
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, e, n):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-n, -n, 0]
        row2 = [-n * z, 1, -n * x]
        row3 = [n * y, n * x, 0]
        return row1, row2, row3


class Chen(ContinuousSystem):
    """Chen system — a Lorenz-dual double-scroll chaotic attractor.

    A three-dimensional autonomous quadratic flow introduced by Chen and Ueta,
    obtained from the Lorenz system via anti-control. It is topologically related
    to but not equivalent to Lorenz (it is the "dual" of Lorenz in the sense of
    Vaněček–Čelikovský), producing a more strongly folded double-scroll
    attractor.

    Parameters
    ----------
    a : float
        Coupling gain of the ``x``–``y`` channel.
    c : float
        Self-feedback gain of the ``y`` channel; the ``(c - a)`` term couples
        ``x`` into ``y``.
    b : float
        Linear decay rate of the ``z`` channel.

    Notes
    -----
    Chaotic at the default parameters (a=35, b=3, c=28). The divergence is the
    constant ``trace(J) = -a + c - b = -10``, so the Lyapunov spectrum sums to
    ``-10`` (see ``known_lyapunov``).
    """

    params = {"a": 35, "b": 3, "c": 28}
    dim = 3
    variables = ("x", "y", "z")
    reference = "Chen & Ueta (1999), Int. J. Bifurcation Chaos 9, 1465-1466"
    doi = "10.1142/s0218127499001024"
    # Canonical Chen attractor (a=35, b=3, c=28). The Lyapunov spectrum is
    # widely reported as (≈2.03, 0, ≈-12.03); the negative exponent is pinned by
    # the constant divergence trace(J) = -a + c - b = -10, so the spectrum must
    # sum to -10 — a hard analytic constraint the finite-time estimate respects.
    known_lyapunov = {
        "spectrum": (2.03, 0.0, -12.03),
        "atol": (0.5, 0.2, 0.6),
        "ic": (-0.1, 0.5, -0.6),
        "kwargs": {
            "dt": 0.02,
            "transient": 50.0,
            "final_time": 300.0,
            "method": "dop853",
            "rtol": 1e-9,
            "atol": 1e-12,
        },
        "source": "Lü, Chen, Cheng & Čelikovský (2002), Int. J. Bifurcation Chaos 12, 2917-2926",
    }

    @staticmethod
    def _equations(Y, t, *, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * y - a * x
        ydot = (c - a) * x - x * z + c * y
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, a, 0]
        row2 = [c - a - z, c, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class ChenLee(ContinuousSystem):
    """Chen–Lee system — chaotic rigid-body / gyro motion.

    A three-dimensional flow derived by Chen and Lee from the Euler equations for
    the rotation of a rigid body subject to feedback torque; it is the governing
    set of equations for a gyro with internal-torque feedback control rather than
    an abstract Lorenz-like model. Each equation carries one quadratic
    cross-product term and the system generates a symmetric two-scroll attractor.

    Parameters
    ----------
    a : float
        Self-feedback gain of the ``x`` channel.
    b : float
        Self-feedback gain of the ``y`` channel.
    c : float
        Self-feedback gain of the ``z`` channel.

    Notes
    -----
    Chaotic at the default parameters (a=5, b=-10, c=-0.38).
    """

    reference = "Chen & Lee (2004), Chaos Solitons Fractals 21, 957-965"
    doi = "10.1016/j.chaos.2003.12.034"
    params = {"a": 5, "b": -10, "c": -0.38}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * x - y * z
        ydot = b * y + x * z
        zdot = c * z + x * y / 3
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [a, -z, -y]
        row2 = [z, b, x]
        row3 = [y / 3, x / 3, c]
        return row1, row2, row3


class WangSun(ContinuousSystem):
    """Wang–Sun three-dimensional chaotic system.

    A three-dimensional quadratic autonomous flow with three quadratic
    cross-product nonlinearities (``y*z``, ``x*z`` and ``x*y``), one per
    equation, producing a chaotic attractor at the default parameters.

    Parameters
    ----------
    a, d, e : float
        Linear self-feedback gains on the ``x``, ``y`` and ``z`` channels.
    b : float
        Linear ``x``-into-``y`` coupling gain.
    q, f : float
        Strengths of the ``y*z`` (in ``x``) and ``x*y`` (in ``z``) cross-product
        terms.

    Notes
    -----
    Chaotic at the default parameters.
    """

    reference = "Wang, Sun, van Wyk, Qi & van Wyk (2009), Braz. J. Phys. 39"
    doi = "10.1590/s0103-97332009000500007"
    params = {"a": 0.2, "b": -0.01, "d": -0.4, "e": -1.0, "f": -1.0, "q": 1.0}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, d, e, f, q):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * x + q * y * z
        ydot = b * x + d * y - x * z
        zdot = e * z + f * x * y
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, d, e, f, q):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [a, q * z, q * y]
        row2 = [b - z, d, -x]
        row3 = [f * y, f * x, e]
        return row1, row2, row3


class YuWang(ContinuousSystem):
    """Yu–Wang system — a chaotic flow with a quadratic-exponential term.

    A three-dimensional autonomous chaotic system, due to Yu and Wang, whose
    ``z`` equation carries a quadratic-exponential nonlinearity ``exp(x*y)``
    (in addition to a quadratic ``x*z`` cross-product in the ``y`` equation).
    The exponential term distinguishes it from the purely quadratic Lorenz-family
    flows and yields a compound (mirror-merged) attractor.

    Parameters
    ----------
    a : float
        Coupling gain of the ``x``–``y`` channel.
    b : float
        Forcing gain of the ``x`` channel into ``y``.
    c : float
        Strength of the ``x*z`` cross-product term in the ``y`` equation.
    d : float
        Linear decay rate of the ``z`` channel.

    Notes
    -----
    Chaotic at the default parameters (a=10, b=40, c=2, d=2.5).
    """

    reference = "Yu & Wang (2012), Eng. Technol. Appl. Sci. Res. 2, 209-215"
    doi = "10.48084/etasr.86"
    params = {"a": 10, "b": 40, "c": 2, "d": 2.5}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * (y - x)
        ydot = b * x - c * x * z
        zdot = exp(x * y) - d * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c, d):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, a, 0]
        row2 = [b - c * z, 0, -c * x]
        row3 = [y * exp(x * y), x * exp(x * y), -d]
        return row1, row2, row3


class YuWang2(ContinuousSystem):
    """Yu–Wang system (hyperbolic-cosine variant).

    A variant of the Yu–Wang chaotic flow in which the ``z`` equation's
    quadratic-exponential nonlinearity is replaced by a hyperbolic cosine
    ``cosh(x*y)`` term, retaining the ``x*z`` quadratic cross-product in the
    ``y`` equation. The result is again a compound chaotic attractor.

    Parameters
    ----------
    a : float
        Coupling gain of the ``x``–``y`` channel.
    b : float
        Forcing gain of the ``x`` channel into ``y``.
    c : float
        Strength of the ``x*z`` cross-product term in the ``y`` equation.
    d : float
        Linear decay rate of the ``z`` channel.

    Notes
    -----
    Chaotic at the default parameters (a=10, b=30, c=2, d=2.5).
    """

    reference = "Yu & Wang (2012), Eng. Technol. Appl. Sci. Res. 2, 209-215"
    doi = "10.48084/etasr.86"
    params = {"a": 10, "b": 30, "c": 2, "d": 2.5}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * (y - x)
        ydot = b * x - c * x * z
        zdot = cosh(x * y) - d * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c, d):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, a, 0]
        row2 = [b - c * z, 0, -c * x]
        row3 = [y * sinh(x * y), x * sinh(x * y), -d]
        return row1, row2, row3


class SanUmSrisuchinwong(ContinuousSystem):
    """San-Um–Srisuchinwong simple chaotic flow.

    A single-parameter three-dimensional chaotic flow whose only nonlinearities
    are a hyperbolic-tangent term ``z*tanh(x)``, a quadratic ``x*y`` product, and
    an absolute-value term ``abs(y)``. The small parameter count and smooth/
    piecewise-smooth nonlinearities make it convenient for analogue-circuit
    realisation.

    Parameters
    ----------
    a : float
        Constant forcing on the ``z`` channel (the single tunable parameter).

    Notes
    -----
    Chaotic at the default parameter (a=2).
    """

    reference = "San-Um & Srisuchinwong (2012), J. Comput. 7, 1041-1047"
    doi = "10.4304/jcp.7.4.1041-1047"
    params = {"a": 2}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y - x
        ydot = -z * tanh(x)
        zdot = -a + x * y + abs(y)
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-1, 1, 0]
        row2 = [-z * (1 - tanh(x) ** 2), 0, -tanh(x)]
        row3 = [y, x + sign(y), 0]
        return row1, row2, row3


class DequanLi(ContinuousSystem):
    """Dequan Li system — a three-scroll chaotic attractor.

    A three-dimensional smooth autonomous quadratic flow, due to Dequan Li, in
    the Lorenz family but with an extra ``x*z`` term in the ``x`` equation and an
    ``x**2`` term in the ``z`` equation. These break the simple two-scroll
    structure and produce a three-scroll attractor: two outer scrolls symmetric
    about the ``z``-axis (as in Lorenz) plus a third scroll encircling it.

    Parameters
    ----------
    a : float
        Coupling gain of the ``x``–``y`` channel.
    d : float
        Strength of the ``x*z`` cross-product term in the ``x`` equation.
    k, f : float
        Forcing gain and self-feedback gain of the ``y`` channel.
    c : float
        Self-feedback gain of the ``z`` channel.
    eps : float
        Strength of the ``x**2`` term in the ``z`` equation.

    Notes
    -----
    Chaotic at the default parameters, exhibiting a three-scroll attractor.
    """

    reference = "Li (2008), Phys. Lett. A 372, 387-393"
    doi = "10.1016/j.physleta.2007.07.045"
    params = {"a": 40, "c": 1.833, "d": 0.16, "eps": 0.65, "f": 20, "k": 55}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, c, d, eps, f, k):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * y - a * x + d * x * z
        ydot = k * x + f * y - x * z
        zdot = c * z + x * y - eps * x**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, c, d, eps, f, k):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a + d * z, a, d * x]
        row2 = [k - z, f, -x]
        row3 = [y - 2 * eps * x, x, c]
        return row1, row2, row3


class Bouali(Bouali2):
    """Bouali economic-cycle chaotic system (first parameter regime).

    The same feedback-augmented Van der Pol flow as :class:`Bouali2`
    (``x' = a y0 x - a x y - b z``; ``y' = -g y + g y x^2``;
    ``z' = -1.5 m x + m bb x z - c z``), at Bouali's original parameter set with
    strong ``x*z`` feedback (``bb = 1``) and linear decay (``c = 0.05``). It
    produces the stretched-loop "business-cycle" strange attractor.

    Parameters
    ----------
    a, y0, b, g, m, bb, c : float
        As in :class:`Bouali2`.
    """

    reference = "Bouali (1999), Int. J. Bifurcation Chaos 9, 745-756"
    doi = "10.1142/s0218127499000535"
    params = {"a": 1.0, "b": -0.3, "bb": 1.0, "c": 0.05, "g": 1.0, "m": 1.0, "y0": 4.0}
    default_ic = [0.3867, 3.0544, -0.0068]


class LiuChen(Sakarya):
    """Liu–Chen three-dimensional chaotic system.

    A member of the generalized Lorenz / Sakarya family (same six-term quadratic
    form ``x' = a x + h y + s y z``; ``y' = -b y - p x + q x z``;
    ``z' = c z - r x y``) at the Liu–Chen parameters, which collapse the linear
    cross-coupling (``h = p = 0``) and leave two quadratic cross-products,
    yielding a double-scroll attractor.

    Parameters
    ----------
    a, b, c, h, p, q, r, s : float
        As in :class:`Sakarya`.
    """

    reference = "Liu & Chen (2004), Int. J. Bifurc. Chaos 14, 1395-1403"
    doi = "10.1142/s0218127404009880"
    params = {"a": 0.4, "b": 12.0, "c": -5.0, "h": 0.0, "p": 0.0, "q": -1.0, "r": 1.0, "s": 1.0}
    default_ic = [4.6723, 0.01, -0.01]


class PanXuZhou(DequanLi):
    r"""Pan–Xu–Zhou (Pan) three-dimensional chaotic attractor.

    A Lorenz-family quadratic flow sharing the :class:`DequanLi` functional form
    (``x' = a y - a x + d x z``; ``y' = k x + f y - x z``;
    ``z' = c z + x y - eps x^2``).  At the Pan–Xu–Zhou parameters the extra
    ``x*z`` / ``x**2`` terms vanish (``d = eps = f = 0``), leaving the compact
    Lorenz-like form

    .. math::

        x' = a (y - x), \quad y' = k x - x z, \quad z' = c z + x y ,

    whose attractor has two scrolls symmetric under
    :math:`(x, y, z) \mapsto (-x, -y, z)`.  It differs from Lorenz only in the
    missing ``-y`` damping of the ``y`` channel, which the original paper argues
    makes it topologically non-equivalent to the Lorenz system.

    Parameters
    ----------
    a, c, d, eps, f, k : float
        As in :class:`DequanLi`.

    Notes
    -----
    **Deviation from the cited source.**  This class ships ``k = 28``.  The
    cited paper — and the ``dysts`` catalogue entry derived from it — use
    ``k = 16``.  **At the published ``k = 16`` this system is not chaotic**, so
    the default was moved rather than shipping a system that contradicts its own
    description.  Three independent checks agree:

    1. *Linear stability.*  With ``b = -c`` the two non-trivial equilibria sit
       at :math:`(\pm\sqrt{bk},\, \pm\sqrt{bk},\, k)`, and the Routh–Hurwitz
       condition on the characteristic polynomial
       :math:`\lambda^3 + (a+b)\lambda^2 + (ab + bk)\lambda + 2abk` makes them
       *stable* for

       .. math::

           k < \frac{a\,(a + b)}{a - b} = 17.2729 \quad (a = 10,\ b = 8/3),

       so ``k = 16`` lies **below** the Hopf threshold.
    2. *Eigenvalues.*  At ``k = 16`` the equilibria are
       :math:`(\pm 6.532, \pm 6.532, 16)` with spectrum
       :math:`-12.557,\; -0.0548 \pm 8.243 i` — a stable focus.
    3. *Measured Lyapunov spectrum.*  A variational-QR run at ``k = 16``
       (``final_time=3000``, ``dt=0.005``, ``transient=1000``) returns
       :math:`(-0.0548,\, -0.0548,\, -12.557)` — no positive exponent and no
       zero exponent; it has converged onto the focus above, not onto an
       attractor.

    At the shipped ``k = 28`` the same measurement gives
    :math:`(0.999,\, 0.000,\, -13.666)` — one positive, one zero, and the sum
    matching the constant divergence :math:`-(a - f) + c = -12.667` exactly —
    from every initial condition tried (five random draws from
    :math:`[-10, 10]^3`, all landing on the same attractor).  Pass
    ``params={"k": 16.0}`` to recover the published, non-chaotic parameters.
    """

    reference = "Zhou, Wuneng et al. (2008), Phys. Lett. A 372, 5773-5777"
    doi = "10.1016/j.physleta.2008.07.032"
    # k = 28 (not the paper's 16, which is below the Hopf threshold ~17.27 and
    # decays to a stable focus — see Notes).
    params = {"a": 10.0, "c": -2.6667, "d": 0.0, "eps": 0.0, "f": 0.0, "k": 28.0}
    default_ic = [-3.038, -1.9805, 14.6567]
    known_lyapunov = {
        "spectrum": (1.0, 0.0, -13.67),
        "atol": (0.15, 0.05, 0.2),
        "kwargs": {"final_time": 3000.0, "dt": 0.005, "transient": 1000.0},
        "source": (
            "measured; the sum is pinned analytically by the constant divergence "
            "-(a - f) + c = -12.667"
        ),
    }


class Tsucs2(DequanLi):
    """Three-Scroll Unified Chaotic System 2 (TSUCS-2).

    A member of the :class:`DequanLi` family
    (``x' = a y - a x + d x z``; ``y' = k x + f y - x z``;
    ``z' = c z + x y - eps x^2``) at the TSUCS-2 parameters (``k = 0``), a
    unified model that contains several three-scroll attractors as special
    cases.

    Parameters
    ----------
    a, c, d, eps, f, k : float
        As in :class:`DequanLi`.

    Warning
    -------
    **At the published parameters this system is quasi-periodic, not chaotic.**
    Unlike :class:`PanXuZhou`, the shipped values here are **unchanged** from the
    reference, and the attractor they produce is bounded, large
    (``|x| <~ 68.5``, ``|y| <~ 66.6``, ``|z| <~ 74.8``) and visually
    three-scroll-like — but it is a 2-torus, not a strange attractor.  The
    measured Lyapunov spectrum is ``(0.000, 0.000, -1.497)`` — two zero
    exponents — from every initial
    condition tried (``final_time=3000``, ``dt=0.002``, ``transient=1000``), on
    both the engine QR estimator and an independent two-trajectory Benettin run
    at ``rtol = 1e-10``.  Lowering ``f`` restores chaos (``f = 10`` measures
    ``(0.492, 0.000, -2.727)``), but no literature source for a chaotic TSUCS-2
    parameter set could be verified, so the published values are kept rather
    than silently replaced.  Treat any "chaotic" quantity computed from the
    defaults with suspicion.
    """

    reference = "Pan, Zhou & Li (2013), Nonlinear Dyn. 73, 1965-1976"
    doi = "10.1007/s11071-013-0922-8"
    params = {"a": 40.0, "c": 0.833, "d": 0.5, "eps": 0.65, "f": 20.0, "k": 0.0}
    default_ic = [1.297, 1.1214, 50.029]
