from symengine import cos, sin, sqrt

from tsdynamics.families import ContinuousSystem


class NuclearQuadrupole(ContinuousSystem):
    r"""Classical Hamiltonian for quadrupole oscillations of an atomic nucleus.

    A two-degree-of-freedom Hamiltonian system (coordinates ``q1, q2`` with
    conjugate momenta ``p1, p2``) modelling collective quadrupole vibrations of
    the nuclear surface. The potential combines a harmonic well with cubic and
    quartic anharmonic couplings, producing a mixed regular–chaotic phase space:
    as energy increases the trajectories pass from quasi-periodic motion to
    Hamiltonian chaos. It is a standard testbed for quantum manifestations of
    classical stochasticity in nuclear physics.

    Parameters
    ----------
    a : float
        Coefficient of the harmonic (quadratic) part of the dynamics.
    b : float
        Strength of the cubic anharmonic coupling between the two modes.
    d : float
        Strength of the quartic anharmonic terms.
    """

    params = {"a": 1.0, "b": 0.55, "d": 0.4}
    dim = 4
    reference = "Baran & Raduta (1998), Int. J. Mod. Phys. E"
    doi = "10.1142/s0218301398000282"

    @staticmethod
    def _equations(Y, t, *, a, b, d):
        q1, q2, p1, p2 = Y(0), Y(1), Y(2), Y(3)
        q1dot = a * p1
        q2dot = a * p2
        p1dot = (
            -a * q1 + 3 / sqrt(2) * b * q1**2 - 3 / sqrt(2) * b * q2**2 - d * q1**3 - d * q1 * q2**2
        )
        p2dot = -a * q2 - 3 * sqrt(2) * b * q1 * q2 - d * q2 * q1**2 - d * q2**3
        return q1dot, q2dot, p1dot, p2dot


class HyperCai(ContinuousSystem):
    """Four-dimensional hyperchaotic Cai system.

    A 4-D autonomous flow built by augmenting a three-dimensional chaotic core
    (Lorenz-like ``a(y - x)`` linear coupling plus a ``y**2`` nonlinearity) with
    a fourth feedback state ``w``. For the canonical parameters it exhibits
    hyperchaos, with two positive Lyapunov exponents.

    Parameters
    ----------
    a : float
        Linear coupling rate of the ``x``–``y`` subsystem.
    b, c : float
        Linear gains in the ``y`` equation.
    d : float
        Damping of the ``z`` mode (fed by ``y**2``).
    e : float
        Feedback gain of the auxiliary state ``w`` onto ``x``.
    """

    params = {"a": 27.5, "b": 3, "c": 19.3, "d": 2.9, "e": 3.3}
    dim = 4
    reference = "Huang (2007), Int. J. Nonlinear Sci."
    doi = "10.1016/s1007-5704(02)00107-7"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, e):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x
        ydot = b * x + c * y - x * z + w
        zdot = -d * z + y**2
        wdot = -e * x
        return xdot, ydot, zdot, wdot


class HyperBao(ContinuousSystem):
    """Four-dimensional hyperchaotic Bao system (coined from the Lü system).

    A 4-D hyperchaotic flow obtained by adding a linear feedback controller
    ``w`` to the three-dimensional Lü system. Over a broad parameter range it
    has two positive Lyapunov exponents, the signature of hyperchaos.

    Parameters
    ----------
    a : float
        Coupling rate of the ``x``–``y`` (Lü) subsystem.
    b : float
        Damping of the ``z`` mode.
    c : float
        Self-gain of the ``y`` mode.
    d : float
        Gain of the ``y*z`` term feeding the auxiliary state ``w``.
    e : float
        Feedback gain of ``x`` onto ``w``.
    """

    params = {"a": 36, "b": 3, "c": 20, "d": 0.1, "e": 21}
    dim = 4
    reference = "Bao & Liu (2008), Chin. Phys. B 17, 4111"
    doi = "10.1088/0256-307x/25/7/018"
    known_lyapunov = {
        "n_positive": 2,
        "kwargs": {
            "n_exp": 2,
            "dt": 0.05,
            "burn_in": 50.0,
            "final_time": 200.0,
            "method": "dop853",
            "rtol": 1e-6,
            "atol": 1e-8,
        },
        "source": "hyperchaotic: two positive exponents",
    }

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, e):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + w
        ydot = c * y - x * z
        zdot = x * y - b * z
        wdot = e * x + d * y * z
        return xdot, ydot, zdot, wdot


class HyperJha(ContinuousSystem):
    """Four-dimensional hyperchaotic Jha system (Lorenz-derived).

    A 4-D hyperchaotic flow extending the Lorenz system with a fourth state
    ``w`` coupled back through an ``x*z`` term. With the canonical parameters
    (a Lorenz-like ``a, b, c`` core plus feedback gain ``d``) it exhibits
    hyperchaos with two positive Lyapunov exponents.

    Parameters
    ----------
    a : float
        Lorenz coupling rate of the ``x``–``y`` subsystem.
    b : float
        Lorenz-like gain in the ``y`` equation.
    c : float
        Damping of the ``z`` mode.
    d : float
        Self-gain of the auxiliary feedback state ``w``.
    """

    params = {"a": 10, "b": 28, "c": 2.667, "d": 1.3}
    dim = 4
    reference = "Meier (2003), Presentation of Attractors with Cinema"
    doi = "10.1007/978-3-540-24699-2_13"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + w
        ydot = -x * z + b * x - y
        zdot = x * y - c * z
        wdot = -x * z + d * w
        return xdot, ydot, zdot, wdot


class HyperQi(ContinuousSystem):
    """Four-dimensional hyperchaotic Qi system.

    A 4-D smooth quadratic autonomous system that, over a large parameter
    range, has two large positive Lyapunov exponents and one small negative
    one: orbits expand strongly in two directions while shrinking in another,
    yielding a highly disordered hyperchaotic attractor. It is a much-studied
    benchmark for hyperchaos synchronization and secure communication.

    Parameters
    ----------
    a : float
        Coupling rate of the ``x``–``y`` subsystem.
    b : float
        Linear gain in the ``y`` equation.
    c, d : float
        Damping rates of the ``z`` and ``w`` modes.
    e, f : float
        Cross-coupling gains between the ``z`` and ``w`` states.

    Notes
    -----
    The default initial condition is set away from the origin because a random
    draw from ``U[0, 1)^4`` escapes the attracting basin.
    """

    params = {"a": 50, "b": 24, "c": 13, "d": 8, "e": 33, "f": 30}
    dim = 4
    reference = "Qi, van Wyk, van Wyk & Chen (2008), Phys. Lett. A 372, 124"
    doi = "10.1016/j.physleta.2007.10.082"
    default_ic = [1.0, 2.0, 1.0, 1.0]  # random U[0,1)^4 escapes the basin

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, e, f):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + y * z
        ydot = b * x + b * y - x * z
        zdot = -c * z - e * w + x * y
        wdot = -d * w + f * z + x * y
        return xdot, ydot, zdot, wdot


class HyperXu(ContinuousSystem):
    """Four-dimensional hyperchaotic Xu system.

    A 4-D hyperchaotic flow generated by adding a state-feedback controller
    ``w`` to a three-dimensional chaotic core. For the canonical parameters it
    is hyperchaotic, with two positive Lyapunov exponents, and was introduced
    together with an analog circuit realization.

    Parameters
    ----------
    a : float
        Coupling rate of the ``x``–``y`` subsystem.
    b : float
        Linear gain in the ``y`` equation.
    c : float
        Damping of the ``z`` mode.
    d : float
        Gain of the ``y`` feedback in the ``w`` equation.
    e : float
        Strength of the ``x*z`` nonlinearity driving ``y``.
    """

    params = {"a": 10, "b": 40, "c": 2.5, "d": 2, "e": 16}
    dim = 4
    reference = "Letellier & Rössler (2007), Scholarpedia 2(8), 1936"
    doi = "10.4249/scholarpedia.1936"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, e):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + w
        ydot = b * x + e * x * z
        zdot = -c * z - x * y
        wdot = x * z - d * y
        return xdot, ydot, zdot, wdot


class HyperWang(ContinuousSystem):
    """Four-dimensional hyperchaotic Wang system.

    A 4-D hyperchaotic flow with a Lorenz-like ``a(y - x)`` core, a quadratic
    ``x**2`` term feeding the ``z`` mode, and a fourth feedback state ``w``.
    For the canonical parameters it exhibits hyperchaos with two positive
    Lyapunov exponents.

    Parameters
    ----------
    a : float
        Coupling rate of the ``x``–``y`` subsystem.
    b : float
        Linear gain in the ``y`` equation.
    c : float
        Damping of the ``z`` mode.
    d : float
        Feedback gain of ``x`` onto the auxiliary state ``w``.
    e : float
        Strength of the ``x**2`` nonlinearity driving ``z``.
    """

    params = {"a": 10, "b": 40, "c": 2.5, "d": 10.6, "e": 4}
    dim = 4
    reference = "Wang, Sun, van Wyk, Qi & van Wyk (2009), Braz. J. Phys. 39"
    doi = "10.1590/s0103-97332009000500007"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, e):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x
        ydot = -x * z + b * x + w
        zdot = -c * z + e * x**2
        wdot = -d * x
        return xdot, ydot, zdot, wdot


class HyperPang(ContinuousSystem):
    """Four-dimensional hyperchaotic Pang system (built from the Lü system).

    A 4-D hyperchaotic flow obtained by adding a linear controller ``w`` to the
    three-dimensional Lü system. For the canonical parameters it has two
    positive Lyapunov exponents; the original study also analyzed its Hopf
    bifurcation and feedback control to suppress the hyperchaos.

    Parameters
    ----------
    a : float
        Coupling rate of the ``x``–``y`` (Lü) subsystem.
    b : float
        Damping of the ``z`` mode.
    c : float
        Self-gain of the ``y`` mode.
    d : float
        Feedback gain of ``x`` and ``y`` onto the auxiliary state ``w``.
    """

    params = {"a": 36, "b": 3, "c": 20, "d": 2}
    dim = 4
    reference = "Pang & Liu (2011), J. Comput. Appl. Math. 235, 2775"
    doi = "10.1016/j.cam.2010.11.029"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x
        ydot = -x * z + c * y + w
        zdot = x * y - b * z
        wdot = -d * x - d * y
        return xdot, ydot, zdot, wdot


class HyperLu(ContinuousSystem):
    """Four-dimensional hyperchaotic Lü system.

    A 4-D hyperchaotic flow generated by applying a state-feedback controller
    ``w`` to the three-dimensional Lü system. For the canonical parameters it
    has two positive Lyapunov exponents, with the hyperchaotic behavior also
    verified by an electronic-circuit realization.

    Parameters
    ----------
    a : float
        Coupling rate of the ``x``–``y`` (Lü) subsystem.
    b : float
        Damping of the ``z`` mode.
    c : float
        Self-gain of the ``y`` mode.
    d : float
        Self-gain of the auxiliary feedback state ``w``.
    """

    params = {"a": 36, "b": 3, "c": 20, "d": 1.3}
    dim = 4
    reference = "Chen, Lu, Lü & Yu (2006), Physica A 364, 103"
    doi = "10.1016/j.physa.2005.09.039"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + w
        ydot = -x * z + c * y
        zdot = x * y - b * z
        wdot = d * w + x * z
        return xdot, ydot, zdot, wdot


class LorenzStenflo(ContinuousSystem):
    """Lorenz–Stenflo system: low-frequency acoustic-gravity waves.

    A four-dimensional generalization of the Lorenz equations derived by
    Stenflo to describe finite-amplitude acoustic-gravity perturbations in a
    rotating atmosphere. The extra state ``w`` carries the rotation
    (``d``-dependent) coupling and reduces to the classical three-dimensional
    Lorenz system when rotation is neglected. It is hyperchaotic for suitable
    parameters and is widely studied in atmospheric and plasma physics.

    Parameters
    ----------
    a : float
        Prandtl-like number coupling ``x`` and ``y`` (and damping ``w``).
    b : float
        Geometric damping of the ``z`` mode.
    c : float
        Rayleigh-like forcing parameter in the ``y`` equation.
    d : float
        Rotation parameter coupling the auxiliary state ``w`` into ``x``.
    """

    params = {"a": 2, "b": 0.7, "c": 26, "d": 1.5}
    dim = 4
    reference = "Letellier & Rössler (2007), Scholarpedia 2(8), 1936"
    doi = "10.4249/scholarpedia.1936"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + d * w
        ydot = c * x - x * z - y
        zdot = x * y - b * z
        wdot = -x - a * w
        return xdot, ydot, zdot, wdot


class Qi(ContinuousSystem):
    """Four-dimensional Qi system with cubic cross-product nonlinearities.

    A 4-D autonomous flow in which every equation carries a triple
    cross-product term (e.g. ``y*z*w``), giving a strongly nonlinear,
    four-wing-type chaotic attractor. The high-order coupling makes the
    dynamics markedly more complex than the quadratic Lorenz family.

    Parameters
    ----------
    a : float
        Coupling rate of the ``x``–``y`` subsystem.
    b : float
        Linear gain in the ``y`` equation.
    c, d : float
        Damping rates of the ``z`` and ``w`` modes.
    """

    params = {"a": 45, "b": 10, "c": 1, "d": 10}
    dim = 4
    reference = "Qi, van Wyk, van Wyk & Chen (2008), Phys. Lett. A 372, 124"
    doi = "10.1016/j.physleta.2007.10.082"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + y * z * w
        ydot = b * x + b * y - x * z * w
        zdot = -c * z + x * y * w
        wdot = -d * w + x * y * z
        return xdot, ydot, zdot, wdot


class ArnoldWeb(ContinuousSystem):
    """Quasi-integrable Hamiltonian exhibiting the Arnold web.

    A perturbed three-degree-of-freedom Hamiltonian system whose resonance
    lines form the "Arnold web" — the network of overlapping resonances that
    governs slow chaotic (Arnold) diffusion in nearly integrable systems. As
    the perturbation strength ``mu`` grows the dynamics passes from the
    Nekhoroshev (exponentially stable) regime to the Chirikov diffusive regime,
    a generic route to weak chaos relevant to celestial mechanics and
    accelerator physics.

    Parameters
    ----------
    mu : float
        Perturbation strength coupling the three actions through the web.
    w : float
        Frequency of the driving angle ``z``.
    """

    params = {"mu": 0.01, "w": 1}
    dim = 5
    reference = "Froeschlé, Guzzo & Lega (2000), Science 289, 2108"
    doi = "10.1126/science.289.5487.2108"

    @staticmethod
    def _equations(Y, t, *, mu, w):
        p1, p2, x1, x2, z = Y(0), Y(1), Y(2), Y(3), Y(4)
        denom = 4 + cos(z) + cos(x1) + cos(x2)
        p1dot = -mu * sin(x1) / denom**2
        p2dot = -mu * sin(x2) / denom**2
        x1dot = p1
        x2dot = p2
        zdot = w
        return p1dot, p2dot, x1dot, x2dot, zdot


class NewtonLiepnik(ContinuousSystem):
    """Newton–Leipnik system: rigid-body motion with linear feedback control.

    Euler's rigid-body equations augmented with a linear feedback torque,
    yielding three quadratic differential equations. For suitable feedback
    gains it possesses two coexisting strange attractors ("double strange
    attractors"), so the long-term behavior is selected by the basin in which
    the initial condition lies.

    Parameters
    ----------
    a : float
        Linear damping coefficient on the ``x`` mode.
    b : float
        Linear gain on the ``z`` mode (the feedback parameter).
    """

    params = {"a": 0.4, "b": 0.175}
    dim = 3
    reference = "Leipnik & Newton (1981), Phys. Lett. A 86, 63"
    doi = "10.1016/0375-9601(81)90165-1"

    @staticmethod
    def _equations(Y, t, *, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -a * x + y + 10 * y * z
        ydot = -x - 0.4 * y + 5 * x * z
        zdot = b * z - 5 * x * y
        return xdot, ydot, zdot


# TODO(reference): unverified — needs a primary citation. No primary source
# could be confidently matched to this specific three-dimensional Robinson
# vector field; the maintainer should confirm the original reference.
class Robinson(ContinuousSystem):
    """Robinson system: a Lorenz-type attractor from a Duffing-like oscillator.

    A three-dimensional flow built from a Duffing-like ``x``–``y`` oscillator
    (double-well restoring force ``x - 2*x**3``) coupled to a slow ``z`` mode
    driven by ``x**2``. For suitable parameters it produces a Lorenz-type
    chaotic attractor.

    Note
    ----
    The catalogue carries this vector field without a confirmed primary
    citation: no published source could be confidently matched to this specific
    three-dimensional form (see the module-level ``TODO`` above the class).

    Parameters
    ----------
    a : float
        Linear damping of the oscillator velocity ``y``.
    b : float
        Coefficient of the ``x**2 * y`` nonlinear damping term.
    c : float
        Damping of the slow ``z`` mode.
    d : float
        Gain of the ``x**2`` forcing on ``z``.
    v : float
        Coupling strength of the ``y*z`` feedback term.
    """

    params = {"a": -0.42, "b": -1.1, "c": 0.5, "d": 0.3, "v": -1.0}

    dim = 3

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, v):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = x - 2 * x**3 - a * y + b * x**2 * y - v * y * z
        zdot = -c * z + d * x**2
        return (xdot, ydot, zdot)


class CellularNeuralNetwork(ContinuousSystem):
    """Chaotic cellular neural network (CNN).

    A three-cell continuous-time cellular neural network whose cell output is
    the standard piecewise-linear saturation ``f(x) = (|x+1| - |x-1|)/2``. With
    suitable coupling weights the coupled cells produce a chaotic attractor,
    making the CNN a hardware-oriented (circuit-realizable) chaos generator.

    Parameters
    ----------
    a, b, c, d : float
        Synaptic coupling weights between the three cells (the CNN template
        feedback coefficients).
    """

    params = {"a": 4.4, "b": 3.21, "c": 1.1, "d": 1.24}
    dim = 3
    reference = "Arena, Caponetto, Fortuna & Porto (1998), Int. J. Bifurc. Chaos 8, 1527"
    doi = "10.1142/s0218127498001170"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z = Y(0), Y(1), Y(2)

        def f(x):
            return 0.5 * (abs(x + 1) - abs(x - 1))

        xdot = -x + d * f(x) - b * f(y) - b * f(z)
        ydot = -y - b * f(x) + c * f(y) - a * f(z)
        zdot = -z - b * f(x) + a * f(y) + f(z)
        return (xdot, ydot, zdot)
