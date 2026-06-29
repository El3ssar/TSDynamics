from symengine import cos, exp, sign, sin, tanh

from tsdynamics.families import ContinuousSystem


class DoublePendulum(ContinuousSystem):
    r"""Planar double pendulum — a canonical Hamiltonian chaotic system.

    Two identical rigid arms (each of length ``d`` and mass ``m``) swing under
    gravity in a vertical plane, the lower hinged to the tip of the upper. The
    state ``(θ₁, θ₂, p₁, p₂)`` collects the two arm angles and their conjugate
    momenta; the equations are the canonical Hamiltonian flow, and the energy is
    conserved (no dissipation), so the motion is quasi-periodic at low energy and
    chaotic — exquisitely sensitive to initial conditions — once the energy is
    large enough for the arms to flip over.

    Parameters
    ----------
    d : float
        Arm length (both arms equal). Sets the gravitational frequency scale.
    m : float
        Arm mass (both arms equal).
    """

    reference = "Marion (2013), Classical Dynamics of Particles and Systems"
    doi = "10.1016/b978-1-4832-5676-4.50003-2"
    params = {"d": 1.0, "m": 1.0}
    dim = 4

    @staticmethod
    def _equations(Y, t, *, d, m):
        th1, th2, p1, p2 = Y(0), Y(1), Y(2), Y(3)
        g = 9.82
        pre = 6 / (m * d**2)
        denom = 16 - 9 * cos(th1 - th2) ** 2
        th1_dot = pre * (2 * p1 - 3 * cos(th1 - th2) * p2) / denom
        th2_dot = pre * (8 * p2 - 3 * cos(th1 - th2) * p1) / denom
        p1_dot = -0.5 * (m * d**2) * (th1_dot * th2_dot * sin(th1 - th2) + 3 * (g / d) * sin(th1))
        p2_dot = -0.5 * (m * d**2) * (-th1_dot * th2_dot * sin(th1 - th2) + 3 * (g / d) * sin(th2))
        return th1_dot, th2_dot, p1_dot, p2_dot


class SwingingAtwood(ContinuousSystem):
    r"""Swinging Atwood's machine — a pendulum coupled to a hanging counterweight.

    A mass ``m1`` is free to swing as a pendulum on a string that runs over
    pulleys to a non-swinging counterweight ``m2`` that only moves vertically;
    the radial length ``r`` and swing angle ``θ`` (with conjugate momenta
    ``pr``, ``pθ``) are thus coupled through the shared string. The conservative
    dynamics is integrable at the special mass ratio ``m2/m1 = 3`` (every bounded
    orbit closes) and generically chaotic otherwise, producing the intricate
    "teardrop" and "smile" orbits for which the system is known.

    Parameters
    ----------
    m1 : float
        Swinging pendulum mass.
    m2 : float
        Non-swinging counterweight mass; the mass ratio ``m2/m1`` controls the
        dynamics (``= 3`` is the integrable case, the default ``4.5`` is chaotic).
    """

    reference = "Tufillaro, Abbott & Griffiths (1984), Am. J. Phys. 52, 895-903"
    doi = "10.1119/1.13791"
    params = {"m1": 1.0, "m2": 4.5}
    dim = 4

    @staticmethod
    def _equations(Y, t, *, m1, m2):
        r, th, pr, pth = Y(0), Y(1), Y(2), Y(3)
        g = 9.82
        rdot = pr / (m1 + m2)
        thdot = pth / (m1 * r**2)
        prdot = pth**2 / (m1 * r**3) - m2 * g + m1 * g * cos(th)
        pthdot = -m1 * g * r * sin(th)
        return rdot, thdot, prdot, pthdot


class Colpitts(ContinuousSystem):
    r"""Chaotic Colpitts oscillator — a sinusoidal LC oscillator driven into chaos.

    A dimensionless third-order model of the Colpitts electronic oscillator (a
    bipolar transistor with an LC tank), whose transistor nonlinearity is the
    piecewise-linear exponential-diode characteristic captured here by the
    ``sign``-gated rectifier term. For suitable component values the normally
    sinusoidal oscillator undergoes a period-doubling route to a chaotic
    attractor, making it a standard hardware chaos generator.

    Parameters
    ----------
    a : float
        Loop-gain / nonlinearity strength of the transistor term.
    b : float
        Damping (loss) coefficient on the voltage variable ``y``.
    c : float
        Bias / supply drive term.
    d : float
        Inductor-branch loss coefficient on ``z``.
    e : float
        Transistor turn-on (pinch-off) threshold of the rectifying nonlinearity.
    """

    reference = "Kennedy (1994), IEEE Trans. Circuits Syst. I 41, 771-774"
    doi = "10.1109/81.331536"
    params = {"a": 30, "b": 0.8, "c": 20, "d": 0.08, "e": 10}
    dim = 3

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, e):
        x, y, z = Y(0), Y(1), Y(2)
        u = z - (e - 1)
        fz = -u * (1 - sign(u)) / 2
        xdot = y - a * fz
        ydot = c - x - b * y - z
        zdot = y - d * z
        return (xdot, ydot, zdot)


class Laser(ContinuousSystem):
    r"""Abooee–Yaghini-Bonabi–Jahed-Motlagh three-dimensional chaotic system.

    A three-variable quadratic/cubic flow proposed as a semiconductor-laser-type
    chaotic model, with cubic cross-coupling (``y·z²``, ``x·z²``) feeding a
    strange attractor that the original authors realised as an analog electronic
    circuit. The coordinates ``(x, y, z)`` are the model's abstract state
    variables (field/inversion-like quantities) rather than calibrated physical
    laser observables.

    Parameters
    ----------
    a : float
        Linear coupling/decay rate between ``x`` and ``y``.
    b : float
        Strength of the cubic ``y·z²`` coupling driving ``x``.
    c, d : float
        Linear and cubic (``x·z²``) coupling coefficients driving ``y``.
    h : float
        Linear damping of ``z``.
    k : float
        Strength of the quadratic ``x²`` feedback into ``z``.
    """

    reference = (
        "Abooee, Yaghini-Bonabi & Jahed-Motlagh (2013), "
        "Commun. Nonlinear Sci. Numer. Simul. 18, 1235-1245"
    )
    doi = "10.1016/j.cnsns.2012.08.036"
    params = {"a": 10.0, "b": 1.0, "c": 5.0, "d": -1.0, "h": -5.0, "k": -6.0}
    dim = 3

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, h, k):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * y - a * x + b * y * z**2
        ydot = c * x + d * x * z**2
        zdot = h * z + k * x**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c, d, h, k):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, a + b * z**2, 2 * b * y * z]
        row2 = [c + d * z**2, 0, 2 * d * x * z]
        row3 = [2 * k * x, 0, h]
        return row1, row2, row3


class Blasius(ContinuousSystem):
    """Three-level food chain with Holling type-II functional responses.

    The chaotic "teacup" attractor lives on a finite basin around a coexistence
    equilibrium near ``x ≈ 20`` — which is also the prey-runaway threshold of
    the type-II response, where predation saturates below the resource's growth
    rate. Random ``U[0, 1)^3`` ICs (small ``y``, sizeable ``z``) fall outside
    the basin and ``x`` blows up exponentially, so a known on-attractor point is
    set as ``default_ic`` to keep default integration bounded.
    """

    reference = "Blasius, Huppert & Stone (1999), Nature 399, 354-359"
    doi = "10.1038/20676"
    params = {
        "a": 1,
        "alpha1": 0.2,
        "alpha2": 1,
        "b": 1,
        "c": 10,
        "k1": 0.05,
        "k2": 0,
        "zs": 0.006,
    }
    dim = 3
    default_ic = [4.031713, 5.1113788, 0.016508812]

    @staticmethod
    def _equations(Y, t, *, a, alpha1, alpha2, b, c, k1, k2, zs):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * x - alpha1 * x * y / (1 + k1 * x)
        ydot = -b * y + alpha1 * x * y / (1 + k1 * x) - alpha2 * y * z / (1 + k2 * y)
        zdot = -c * (z - zs) + alpha2 * y * z / (1 + k2 * y)
        return xdot, ydot, zdot


class FluidTrampoline(ContinuousSystem):
    r"""Fluid trampoline — a droplet bouncing on a vibrated soap film.

    A reduced model of a liquid droplet bouncing on a sinusoidally driven soap
    film, which acts as a nonlinear vertical spring. The state ``(x, y, θ)`` is
    the droplet height, its velocity, and the forcing phase; contact with the
    film is gated by the ``sign(x)`` switch (a restoring force plus quadratic
    drag acting only while ``x < 0``), and the periodic forcing ``γ·cos(θ)``
    drives the bounce. As the forcing is increased the system passes through
    simple and multi-periodic bouncing states into chaos.

    Parameters
    ----------
    gamma : float
        Dimensionless forcing acceleration amplitude.
    psi : float
        Quadratic-drag (energy-loss) coefficient of the film contact.
    w : float
        Dimensionless forcing angular frequency (``θ`` advances at rate ``w``).
    """

    reference = "Gilet & Bush (2009), J. Fluid Mech. 625, 167-203"
    doi = "10.1017/s0022112008005442"
    params = {"gamma": 1.82, "psi": 0.01019, "w": 1.21}
    dim = 3

    @staticmethod
    def _equations(Y, t, *, gamma, psi, w):
        x, y, th = Y(0), Y(1), Y(2)
        xdot = y
        ydot = -1 - (1 - sign(x)) / 2 * (x + psi * y * abs(y)) + gamma * cos(th)
        thdot = w
        return (xdot, ydot, thdot)


class JerkCircuit(ContinuousSystem):
    r"""Sprott's chaotic jerk circuit with an exponential-diode nonlinearity.

    A "jerk" system — a single third-order scalar ODE ``x''' = f(x, x', x'')``
    written in companion form ``(x, y, z)`` — realising one of the algebraically
    simplest known chaotic electronic circuits. The only nonlinearity is the
    diode's exponential current–voltage law ``ε·(exp(y/y0) − 1)``; three
    integrators and a single diode suffice to generate the chaotic attractor.

    Parameters
    ----------
    eps : float
        Diode saturation-current scale (strength of the exponential nonlinearity).
    y0 : float
        Diode thermal-voltage scale setting the exponential's steepness.
    """

    reference = "Sprott (2011), IEEE Trans. Circuits Syst. II 58, 240-243"
    doi = "10.1109/tcsii.2011.2124490"
    params = {"eps": 1e-9, "y0": 0.026}
    dim = 3

    @staticmethod
    def _equations(Y, t, *, eps, y0):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = z
        zdot = -z - x - eps * (exp(y / y0) - 1)
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, eps, y0):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-1, -eps * exp(y / y0) / y0, -1]
        return row1, row2, row3


class WindmiReduced(ContinuousSystem):
    r"""Reduced WINDMI model of the solar-wind–magnetosphere–ionosphere coupling.

    Three-variable reduction (field-aligned current ``i``, voltage ``v``,
    pressure ``p``) of the WINDMI energy-conserving model.  The pressure-loss
    term carries a fractional power ``p**(5/4)`` gated by a steep
    ``tanh(d1*(i-1))`` switch (``d1 = 2200`` makes it a near-step at ``i = 1``,
    the substorm-onset threshold).

    Numerical guards (no change to the dynamics)
    --------------------------------------------
    Two robustness guards keep the orbit reproducible across the engine's
    explicit and implicit solver kernels:

    * ``abs(p)`` under the fractional power.  On the attractor ``p >= 0``, but
      adaptive and implicit solvers probe trial states with ``p < 0`` where the
      real ``p**(5/4)`` is complex.  ``abs(p)`` is identical to ``p`` on the
      physical trajectory and real, ``C^1`` everywhere.

    * The ``tanh`` argument is clamped to ``[-_TANH_CLAMP, _TANH_CLAMP]``.
      ``tanh`` saturates to ``±1`` to machine precision past ``|arg| ≈ 20``, so
      clamping leaves the switch value unchanged, but it stops the implicit
      kernels' automatic differentiation from overflowing on ``exp`` of an
      enormous ``d1*(i-1)`` (≈ ±1500 on the orbit) — the actual cause of their
      Newton/error-test failures.  The clamp uses ``abs`` only.
    """

    reference = "Smith, Thiffeault & Horton (2000), J. Geophys. Res. 105, 12983-12996"
    doi = "10.1029/1999ja000218"
    params = {"a1": 0.247, "b1": 10.8, "b2": 0.0752, "b3": 1.06, "d1": 2200, "vsw": 5}
    dim = 3
    variables = ("i", "v", "p")

    #: Bound on the ``tanh`` switch argument; ``tanh(±20)`` already equals ``±1``
    #: to ~1e-18, so clamping is dynamically invisible but autodiff-safe.
    _TANH_CLAMP = 25.0

    @staticmethod
    def _equations(Y, t, *, a1, b1, b2, b3, d1, vsw):
        i, v, p = Y(0), Y(1), Y(2)
        # Clamp d1*(i-1) to [-C, C] via the abs identity
        # clamp(z, -C, C) = (|z + C| - |z - C|) / 2 — smooth-enough (abs only)
        # and lowers cleanly to the engine tape.
        c = WindmiReduced._TANH_CLAMP
        z = d1 * (i - 1)
        z_clamped = (abs(z + c) - abs(z - c)) / 2
        idot = a1 * (vsw - v)
        vdot = b1 * i - b2 * abs(p) ** (1 / 2) - b3 * v
        pdot = vsw**2 - abs(p) ** (5 / 4) * vsw ** (1 / 2) * (1 + tanh(z_clamped)) / 2
        return idot, vdot, pdot
