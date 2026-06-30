from symengine import cos, pi, sign, sin

from tsdynamics.families import ContinuousSystem


class ShimizuMorioka(ContinuousSystem):
    """
    Shimizu–Morioka model — a Lorenz-like flow with simplified nonlinearity.

    A three-dimensional autonomous flow introduced to study the pitchfork
    bifurcation of the symmetric figure-eight limit cycle and the resulting
    cascade of bifurcations seen in the Lorenz equations at large Rayleigh
    number.  It retains the Z2 symmetry ``(x, y) -> (-x, -y)`` of the Lorenz
    system and supports a Lorenz-type strange attractor.

    Parameters
    ----------
    a, b : float
        Positive bifurcation parameters (damping ``a`` and dissipation ``b``).
        The defaults ``a = 0.85``, ``b = 0.5`` lie in the Lorenz-attractor
        region of the parameter plane.
    """

    params = {"a": 0.85, "b": 0.5}
    dim = 3
    reference = "Shimizu & Morioka (1980), Phys. Lett. A 76, 201-204"
    doi = "10.1016/0375-9601(80)90466-1"

    @staticmethod
    def _equations(Y, t, *, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = x - a * y - x * z
        zdot = -b * z + x**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 0]
        row2 = [1 - z, -a, -x]
        row3 = [2 * x, 0, -b]
        return row1, row2, row3


class MooreSpiegel(ContinuousSystem):
    """
    Moore–Spiegel oscillator — a thermally excited aperiodic convection model.

    A three-dimensional flow for a fluid element oscillating vertically in a
    temperature gradient with a linear restoring force, exchanging heat with its
    surroundings so that its temperature-dependent buoyancy can drive
    overstable convection.  It was one of the earliest models to exhibit
    aperiodic (chaotic) variability and was proposed to explain the irregular
    luminosity of variable stars.

    Parameters
    ----------
    a, b, eps : float
        Coupling/stiffness (``a``), linear restoring (``b``) and tension/heat-
        transfer (``eps``) parameters.  The defaults ``a = 10``, ``b = 4``,
        ``eps = 9`` produce chaotic motion.
    """

    params = {"a": 10, "b": 4, "eps": 9}
    dim = 3
    reference = "Moore & Spiegel (1966), Astrophys. J. 143, 871-887"
    doi = "10.1086/148562"

    @staticmethod
    def _equations(Y, t, *, a, b, eps):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = a * z
        zdot = -z + eps * y - y * x**2 - b * x
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, eps):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 0]
        row2 = [0, 0, a]
        row3 = [-2 * x * y - b, eps - x**2, -1]
        return row1, row2, row3


class AnishchenkoAstakhov(ContinuousSystem):
    """
    Anishchenko–Astakhov oscillator — a radiophysical generator of spiral chaos.

    A three-dimensional autonomous flow modelling a self-sustained oscillator
    with inertial nonlinearity (a radiophysical chaos generator).  It is a
    canonical model of spiral (saddle-focus) chaos: the onset of chaotic
    self-oscillations is tied to a saddle-focus separatrix loop, and the route
    to chaos follows the Feigenbaum period-doubling scenario.  The discontinuous
    ``sign(x)`` switch implements the inertial (half-wave rectifying)
    nonlinearity of the generator.

    Parameters
    ----------
    eta : float
        Inertial-inertia (relaxation) parameter of the nonlinear element.
    mu : float
        Excitation parameter controlling the negative damping of the
        oscillatory circuit.  The defaults ``eta = 0.5``, ``mu = 1.2`` lie in
        the chaotic regime.
    """

    params = {"eta": 0.5, "mu": 1.2}
    dim = 3
    reference = "Anishchenko et al. (2007), Nonlinear Dynamics of Chaotic and Stochastic Systems"
    doi = "10.1007/978-3-540-38168-6"

    @staticmethod
    def _equations(Y, t, *, eta, mu):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = mu * x + y - x * z
        ydot = -x
        zdot = -eta * z + eta * (1 + sign(x)) / 2 * x**2
        return xdot, ydot, zdot


class Aizawa(ContinuousSystem):
    """
    Aizawa attractor — a six-parameter chaotic flow with symmetric lobes.

    A three-dimensional flow with rotational symmetry about the z-axis whose
    trajectories wind into a sphere-like attractor pierced by a narrow tube,
    producing distinctive symmetric lobes.  It is widely used as a visually
    striking benchmark attractor.

    Parameters
    ----------
    a, b, c, d, e, f : float
        Shape parameters of the attractor; the defaults
        ``(a, b, c, d, e, f) = (0.95, 0.7, 0.6, 3.5, 0.25, 0.1)`` give the
        canonical lobed chaotic attractor.
    """

    params = {"a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5, "e": 0.25, "f": 0.1}
    dim = 3
    reference = "Aizawa & Uezu (1982), Prog. Theor. Phys. 67, 982-985"
    doi = "10.1143/PTP.67.982"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, e, f):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = x * z - b * x - d * y
        ydot = d * x + y * z - b * y
        zdot = c + a * z - (z**3) / 3 - x**2 - y**2 - e * z * x**2 - e * z * y**2 + f * z * x**3
        return xdot, ydot, zdot


class StickSlipOscillator(ContinuousSystem):
    """
    Stick–slip oscillator — a dry-friction flow with harmonic forcing.

    A forced single-degree-of-freedom oscillator with a Duffing-type restoring
    force and a velocity-dependent (Stribeck-like) dry-friction torque.  The
    ``sign(v - vs)`` switch about the relative sliding velocity makes the system
    non-smooth and produces alternating stick and slip phases that can become
    chaotic under the periodic drive ``gamma * cos(th)``.

    Parameters
    ----------
    a, b : float
        Linear and cubic stiffness of the Duffing restoring force.
    alpha, beta : float
        Linear and cubic velocity coefficients shaping the friction torque.
    t0 : float
        Static (Coulomb) friction level of the ``sign``-switched torque.
    vs : float
        Reference (Stribeck) sliding velocity at which the switch acts.
    eps : float
        Coupling weight of the friction/forcing term in the velocity equation.
    gamma : float
        Amplitude of the harmonic drive.
    w : float
        Angular frequency of the harmonic drive (``th`` advances at rate ``w``).
    """

    params = {
        "a": 1,
        "alpha": 0.3,
        "b": 1,
        "beta": 0.3,
        "eps": 0.05,
        "gamma": 1.0,
        "t0": 0.3,
        "vs": 0.4,
        "w": 2,
    }
    dim = 3
    reference = "Awrejcewicz & Holicke (1999), Int. J. Bifurc. Chaos"
    doi = "10.1142/s0218127499000341"

    @staticmethod
    def _equations(Y, t, *, a, alpha, b, beta, eps, gamma, t0, vs, w):
        x, v, th = Y(0), Y(1), Y(2)
        tq = t0 * sign(v - vs) - alpha * v + beta * (v - vs) ** 3
        xdot = v
        vdot = eps * (gamma * cos(th) - tq) + a * x - b * x**3
        thdot = w
        return xdot, vdot, thdot


class Torus(ContinuousSystem):
    """
    Parametric torus winding — a quasiperiodic curve on a torus surface.

    Not a chaotic system: the right-hand side is purely time-driven (independent
    of the state), so integrating it traces the parametric curve of a winding
    that wraps ``n`` times around the tube while circling the central hole once.
    Useful as a non-chaotic, quasiperiodic reference trajectory.

    Parameters
    ----------
    r : float
        Major radius (distance from the torus centre to the tube centre).
    a : float
        Minor radius (tube radius).
    n : float
        Number of tube windings per revolution about the central axis; an
        irrational ratio gives a dense quasiperiodic winding.
    """

    params = {"a": 0.5, "n": 15.3, "r": 1}
    dim = 3
    reference = "Strogatz (1994), Nonlinear Dynamics and Chaos"
    doi = "10.1201/9780429492563"

    @staticmethod
    def _equations(Y, t, *, a, n, r):
        # Parametric torus: derivatives are purely time-driven, independent of state.
        xdot = (-a * n * sin(n * t)) * cos(t) - (r + a * cos(n * t)) * sin(t)
        ydot = (-a * n * sin(n * t)) * sin(t) + (r + a * cos(n * t)) * cos(t)
        zdot = a * n * cos(n * t)
        return xdot, ydot, zdot


# Intentionally citation-free: a standard Lissajous (Bowditch) figure is a
# classical parametric curve, not a research-introduced dynamical system, so no
# primary reference is attached (and none should be invented).
class Lissajous3D(ContinuousSystem):
    """
    Three-dimensional Lissajous (Bowditch) figure.

    Not a chaotic system: the right-hand side is purely time-driven (independent
    of the state), so integrating it traces the classical parametric curve
    ``(A cos(a t), B cos(b t + delta_y), C cos(c t + delta_z))``.  The shape is
    closed and periodic when the frequency ratios are rational.

    Note
    ----
    A Lissajous (Bowditch) figure is a classical parametric construction, not a
    system introduced by a single research paper; it is deliberately carried
    without a literature citation.

    Parameters
    ----------
    A, B, C : float
        Amplitudes along the x, y and z axes.
    a, b, c : float
        Frequencies along the x, y and z axes.
    delta_y, delta_z : float
        Phase shifts of the y and z components relative to x.
    """

    params = {
        "A": 1,
        "B": 1,
        "C": 1,
        "a": 3,
        "b": 2,
        "c": 5,
        "delta_y": pi / 2,
        "delta_z": pi / 4,
    }
    dim = 3

    @staticmethod
    def _equations(Y, t, *, A, B, C, a, b, c, delta_y, delta_z):
        """
        RHS of the 3D Lissajous system.

        Parameters
        ----------
        Y : callable
            Symbolic state accessor (unused — Lissajous is purely parametric).
        t : symbol
            Symbolic time variable.
        A, B, C : float
            Amplitudes along x, y, z axes.
        a, b, c : float
            Frequencies along x, y, z axes.
        delta_y, delta_z : float
            Phase shifts along y and z axes.

        Returns
        -------
        tuple
            Derivatives [dx/dt, dy/dt, dz/dt].
        """
        # Parametric Lissajous: derivatives are purely time-driven, independent of state.
        dxdt = A * (-a * sin(a * t))
        dydt = B * (-b * sin(b * t + delta_y))
        dzdt = C * (-c * sin(c * t + delta_z))
        return dxdt, dydt, dzdt


# Intentionally citation-free: a standard Lissajous (Bowditch) figure is a
# classical parametric curve, not a research-introduced dynamical system, so no
# primary reference is attached (and none should be invented).
class Lissajous2D(ContinuousSystem):
    """
    Two-dimensional Lissajous (Bowditch) figure.

    Not a chaotic system: the right-hand side is purely time-driven (independent
    of the state), so integrating it traces the classical parametric curve
    ``(A cos(a t), B cos(b t + delta))``.  The figure is closed and periodic
    when the frequency ratio ``a : b`` is rational.

    Note
    ----
    A Lissajous (Bowditch) figure is a classical parametric construction, not a
    system introduced by a single research paper; it is deliberately carried
    without a literature citation.

    Parameters
    ----------
    A, B : float
        Amplitudes along the x and y axes.
    a, b : float
        Frequencies along the x and y axes.
    delta : float
        Phase shift of the y component relative to x.
    """

    params = {"A": 1, "B": 1, "a": 3, "b": 2, "delta": pi / 2}
    dim = 2

    @staticmethod
    def _equations(Y, t, *, A, B, a, b, delta):
        """
        RHS of the 2D Lissajous system.

        Parameters
        ----------
        Y : callable
            Symbolic state accessor (unused — Lissajous is purely parametric).
        t : symbol
            Symbolic time variable.
        A, B : float
            Amplitudes along x and y axes.
        a, b : float
            Frequencies along x and y axes.
        delta : float
            Phase shift along y axis.

        Returns
        -------
        tuple
            Derivatives [dx/dt, dy/dt].
        """
        # Parametric Lissajous: derivatives are purely time-driven, independent of state.
        dxdt = A * (-a * sin(a * t))
        dydt = B * (-b * sin(b * t + delta))
        return dxdt, dydt
