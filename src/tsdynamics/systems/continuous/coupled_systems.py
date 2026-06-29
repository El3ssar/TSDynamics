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
    """Bouali economic-cycle chaotic system.

    A three-dimensional flow obtained by adding a feedback loop to an extended
    Van der Pol oscillator, proposed by Bouali as an idealised macroeconomic
    model of business cycles. The 2-D oscillator core (``x``, ``y``) supplies
    the relaxation cycle and the ``z`` feedback channel drives it chaotic for
    suitable gains, producing a stretched-loop strange attractor.

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
    params = {"a": 1.0, "b": -0.3, "bb": 1.0, "c": 0.05, "g": 1.0, "m": 1, "y0": 4.0}
    dim = 3

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
    """Lü–Chen–Cheng unified system bridging the Lorenz and Chen attractors.

    A one-parameter family of three-dimensional quadratic flows that interpolates
    continuously between the Lorenz and the Chen system, remaining chaotic across
    the whole transition. Introduced by Lü, Chen, Cheng and Čelikovský to "bridge
    the gap" between the two canonical attractors; this parameterisation uses the
    harmonic combination ``-(a*b)/(a+b)*x`` on the ``x`` channel.

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

    reference = "Lü, Chen, Cheng & Čelikovský (2002), Int. J. Bifurcation Chaos 12, 2917-2926"
    doi = "10.1142/s021812740401014x"
    params = {"a": -10, "b": -4, "c": 18.1}
    dim = 3

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
    doi = "10.1515/zna-1985-0102"
    params = {"e": 13, "n": 10}
    dim = 3

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
    reference = "Chen (1997), Proc. 1st Int. Conf. Control of Oscillations and Chaos"
    doi = "10.1109/coc.1997.631323"
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
            "burn_in": 50.0,
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

    reference = "San-Um & Srisuchinwong (2012), J. Comput."
    doi = "10.1109/apcc.2007.4433503"
    params = {"a": 2}
    dim = 3

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
