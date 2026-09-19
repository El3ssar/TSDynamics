from symengine import cos, cosh, pi, sin, sinh, tanh

from tsdynamics.families import ContinuousSystem


class VallisElNino(ContinuousSystem):
    r"""Vallis low-order model of El Niño–Southern Oscillation chaos.

    A three-variable conceptual model of the coupled tropical-Pacific
    ocean–atmosphere system: ``x`` is the surface wind / zonal current,
    ``y`` and ``z`` are the eastern and western sea-surface temperatures.
    It is a Lorenz-like system — adding the external forcing parameter ``p``
    to a Lorenz-type core — and Vallis showed that even without stochastic
    forcing the deterministic dynamics can be chaotic, suggesting El Niño's
    aperiodicity may be intrinsic.

    Parameters
    ----------
    b
        Strength of the wind-driven ocean coupling (advective feedback gain).
    c
        Damping/relaxation rate of the wind anomaly toward equilibrium.
    p
        Steady external (e.g. seasonal mean) forcing of the wind.

    With ``b=102, c=3, p=0`` the model settles onto a chaotic attractor.
    """

    reference = "Vallis (1988), J. Geophys. Res. 93, 13979-13991"
    doi = "10.1029/jc093ic11p13979"
    params = {"b": 102.0, "c": 3.0, "p": 0.0}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, b, c, p):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = b * y - c * x - c * p
        ydot = -y + x * z
        zdot = -z - x * y + 1
        return xdot, ydot, zdot


class RayleighBenard(ContinuousSystem):
    r"""Low-order Rayleigh–Bénard convection model (Saltzman truncation).

    A three-mode spectral truncation of two-dimensional thermal convection in
    a fluid layer heated from below, of the same Saltzman/Lorenz convection
    family: ``x`` is the convective overturning intensity, ``y`` and ``z`` the
    horizontal and vertical temperature-perturbation modes.  Saltzman derived
    these ordinary differential equations from the Boussinesq equations as
    double-Fourier coefficients; the closely related three-mode subset is the
    source of the Lorenz attractor.

    Parameters
    ----------
    a
        Prandtl-number-like ratio controlling momentum vs. thermal diffusion.
    b
        Geometric aspect-ratio factor damping the vertical temperature mode.
    r
        Reduced Rayleigh number — the convective driving / buoyancy forcing.

    Increasing ``r`` past the convective threshold drives the layer through
    steady, periodic and chaotic overturning.
    """

    reference = "Yanagita & Kaneko (1995), Physica D 82, 288-313"
    doi = "10.1016/0167-2789(94)00233-g"
    params = {"a": 30, "b": 5, "r": 18}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, r):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * y - a * x
        ydot = r * y - x * z
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, r):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, a, 0]
        row2 = [-z, r, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class Hadley(ContinuousSystem):
    r"""Lorenz-84 model of the atmospheric Hadley/westerly circulation.

    A three-variable caricature of the large-scale mid-latitude circulation:
    ``x`` is the intensity of the symmetric, globe-encircling westerly current
    (and the poleward temperature gradient), while ``y`` and ``z`` are the
    cosine and sine amplitudes of a chain of superposed large-scale eddies.
    Despite only three modes it reproduces irregular, intransitive behaviour
    and is a standard low-order testbed for atmospheric predictability.

    Parameters
    ----------
    a
        Damping rate of the zonal flow ``x`` (relative to the eddy damping,
        which is normalised to 1).
    b
        Strength of the nonlinear coupling that displaces the eddies relative
        to the westerly current.
    f
        Symmetric (cross-latitude) thermal forcing driving the zonal flow.
    g
        Asymmetric thermal forcing acting on the eddies.

    With ``a=0.25, b=4, f=8, g=1`` (and nearby) the model is chaotic; the
    catalogue default ``f=9`` likewise yields aperiodic circulation.

    Note
    ----
    Despite the catalogue name, this is the **Lorenz-84** model (it is
    algebraically the same system as ``Lorenz84`` in ``chaotic_attractors.py``,
    here at a different parameter regime), not a model due to George Hadley.
    """

    reference = (
        "Lorenz (1984), 'Irregularity: a fundamental property of the "
        "atmosphere', Tellus 36A, 98-110"
    )
    doi = "10.1111/j.1600-0870.1984.tb00230.x"
    params = {"a": 0.2, "b": 4.0, "f": 9.0, "g": 1.0}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, f, g):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -(y**2) - z**2 - a * x + a * f
        ydot = x * y - b * x * z - y + g
        zdot = b * x * y + x * z - z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, f, g):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, -2 * y, -2 * z]
        row2 = [y - b * z, x - 1, -b * x]
        row3 = [b * y + z, b * x, x - 1]
        return row1, row2, row3


class DoubleGyre(ContinuousSystem):
    r"""Time-dependent double-gyre flow (a kinematic transport benchmark).

    A canonical kinematic model of two counter-rotating gyres whose dividing
    streamline oscillates periodically, popularised by Shadden, Lekien &
    Marsden as a testbed for Lagrangian coherent structures and chaotic
    advection.  The state ``(x, y, z)`` is a tracer position in the unit cell
    augmented with an explicit phase ``z = omega t``, so the autonomous flow
    reproduces the original non-autonomous stream function.

    Parameters
    ----------
    alpha
        Velocity-field amplitude (overall advection speed).
    eps
        Amplitude of the periodic lateral perturbation of the gyre boundary;
        ``eps = 0`` gives two static, non-mixing gyres.
    omega
        Angular frequency of the boundary oscillation (the clock rate).

    For finite ``eps`` the oscillating separatrix produces a tangle that mixes
    tracers chaotically across the two gyres.
    """

    reference = "Shadden, Lekien & Marsden (2005), Physica D 212, 271-304"
    doi = "10.1016/j.physd.2005.10.007"
    params = {"alpha": 0.1, "eps": 0.1, "omega": 0.62832}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, alpha, eps, omega):
        x, y, z = Y(0), Y(1), Y(2)
        a = eps * sin(z)
        b = 1 - 2 * eps * sin(z)
        f = a * x**2 + b * x
        dx = -alpha * pi * sin(pi * f) * cos(pi * y)
        dy = alpha * pi * cos(pi * f) * sin(pi * y) * (2 * a * x + b)
        dz = omega
        return dx, dy, dz


class BlinkingRotlet(ContinuousSystem):
    r"""Blinking-rotlet flow — a model of chaotic advection in Stokes mixing.

    A passive tracer in a circular cell stirred by two off-centre rotlets at
    radius ``+b`` and ``-b`` that alternate being active with period ``tau``,
    the switch being a steep ``tanh`` ramp.  The periodic blinking folds and
    stretches material lines, producing chaotic advection (Aref-type *blinking*
    flow).  State is ``(r, theta, t)`` in polar coordinates with an explicit
    clock; ``dtheta`` carries a ``1/r`` factor singular at the cell centre, so
    ``default_ic`` sits on a bounded orbit away from ``r = 0``.

    The rotlet velocity field is a large rational expression; the engine lowers
    it to an IR tape and integrates the flow in well under a millisecond at every
    tolerance, even at the original steep ``tanh`` switch.
    """

    reference = "Meleshko & Aref (1996), Phys. Fluids 8, 3215-3217"
    doi = "10.1063/1.869128"
    params = {
        "a": 1.0,
        "b": 0.5298833894399929,
        "bc": 1.0,
        "sigma": -1.0,
        "tau": 3.0,
    }
    dim = 3
    variables = ("r", "theta", "t")
    #: On a bounded orbit (``r`` stays in ``[0.8, 0.91]``) away from the
    #: ``r = 0`` angular singularity, so default sweeps stay bounded.
    default_ic = [0.8, 4.887, 0.0]

    @staticmethod
    def _rotlet(r, theta, a, b, bc):
        kappa = a**2 + (b**2 * r**2) / a**2 - 2 * b * r * cos(theta)
        gamma = (1 - r**2 / a**2) * (a**2 - (b**2 * r**2) / a**2)
        iota = (b**2 * r) / a**2 - b * cos(theta)
        zeta = b**2 + r**2 - 2 * b * r * cos(theta)
        nu = a**2 + b**2 - (2 * b**2 * r**2) / a**2

        vr = b * sin(theta) * (-bc * (gamma / kappa**2) - 1 / kappa + 1 / zeta)

        vth = (
            bc * (gamma * iota) / kappa**2
            + bc * r * nu / (a**2 * kappa)
            + iota / kappa
            - (r - b * cos(theta)) / zeta
        )
        return vr, vth

    @staticmethod
    def _protocol(t, tau, stiffness=20):
        return 0.5 + 0.5 * tanh(tau * stiffness * sin(2 * pi * t / tau))

    @staticmethod
    def _equations(y, t, *, a, b, bc, sigma, tau):
        r = y(0)
        theta = y(1)
        tt = y(2)

        weight = BlinkingRotlet._protocol(tt, tau)

        dr1, dth1 = BlinkingRotlet._rotlet(r, theta, a, b, bc)
        dr2, dth2 = BlinkingRotlet._rotlet(r, theta, a, -b, bc)

        dr = weight * dr1 + (1 - weight) * dr2
        dth = (weight * dth1 + (1 - weight) * dth2) / r
        dtt = 1

        return (
            sigma * dr,
            sigma * dth,
            dtt,
        )


class OscillatingFlow(ContinuousSystem):
    r"""Oscillating convection-roll flow (chaotic advection model).

    A kinematic stream-function model of a periodic array of two-dimensional
    convection rolls whose boundaries oscillate laterally in time, after the
    time-dependent Rayleigh–Bénard experiments of Solomon & Gollub.  The
    state ``(x, y, z)`` is a tracer position together with an explicit phase
    ``z = omega t``; the periodic ``b sin(z)`` modulation of the roll pattern
    opens the cell boundaries and produces chaotic transport of passive
    tracers between adjacent rolls.

    Parameters
    ----------
    b
        Amplitude of the lateral oscillation of the roll boundaries.
    k
        Spatial wavenumber setting the roll size.
    omega
        Angular frequency of the roll oscillation.
    u
        Overall flow (advection) speed.

    Non-zero ``b`` makes the roll separatrices time-dependent, so tracers
    chaotically hop between cells (enhanced diffusive transport).
    """

    reference = "Solomon & Gollub (1988), Phys. Rev. A 38, 6280-6286"
    doi = "10.1103/physreva.38.6280"
    params = {"b": 0.48, "k": 1.0, "omega": 0.49, "u": 0.72}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, b, k, omega, u):
        x, y, z = Y(0), Y(1), Y(2)
        f = x + b * sin(z)
        dx = u * cos(k * y) * sin(k * f)
        dy = -u * sin(k * y) * cos(k * f)
        dz = omega
        return dx, dy, dz


class ArnoldBeltramiChildress(ContinuousSystem):
    r"""Arnold–Beltrami–Childress (ABC) flow.

    The ABC flow is an exact, spatially-periodic steady solution of Euler's
    equations for an inviscid incompressible fluid in three dimensions, and a
    Beltrami flow (vorticity everywhere parallel to velocity).  Although the
    velocity field is steady, its streamlines — here integrated as tracer
    trajectories ``(x, y, z)`` — are generically chaotic, making it a classic
    example of Lagrangian (kinematic) chaos and fast-dynamo theory.

    Parameters
    ----------
    a, b, c
        Amplitudes of the three Beltrami components.  When all three are
        non-zero the streamlines develop resonant overlaps and chaotic
        regions interleaved with integrable tubes; the standard chaotic case
        is ``a = sqrt(3), b = sqrt(2), c = 1``.
    """

    reference = "Arnold (1966), J. Appl. Math. Mech. 30, 223-226"
    doi = "10.1016/0021-8928(66)90070-0"
    params = {"a": 1.73205, "b": 1.41421, "c": 1}
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        dx = a * sin(z) + c * cos(y)
        dy = b * sin(x) + a * cos(z)
        dz = c * sin(y) + b * cos(x)
        return dx, dy, dz


class AtmosphericRegime(ContinuousSystem):
    r"""Coupled-oscillator model of atmospheric regime transitions.

    A model of regime transition in atmospheric flows, due to Tuwankotta, built
    as two coupled oscillators with widely spaced natural frequencies and an
    energy-preserving quadratic nonlinearity.  The slow mode ``x`` is coupled
    to a fast oscillator ``(y, z)``; the timescale separation produces
    intermittent bursts in which the system jumps between quasi-stationary
    regimes, a caricature of weather-regime (e.g. zonal/blocked) transitions.

    Parameters
    ----------
    alpha, beta
        Coefficients of the quadratic coupling between the slow mode and the
        fast oscillator (the energy-preserving nonlinearity).
    mu1, mu2
        Linear growth/damping rates of the slow mode and the fast oscillator.
    omega
        Natural frequency of the fast oscillator (the widely-spaced frequency).
    sigma
        Strength of the bilinear slow–fast feedback.

    For the catalogue defaults the timescale gap drives chaotic regime
    switching.
    """

    reference = "Tuwankotta (2006), Int. J. Non-Linear Mech. 41, 180-191"
    doi = "10.1016/j.ijnonlinmec.2005.02.007"
    params = {
        "alpha": -2.0,
        "beta": -5.0,
        "mu1": 0.05,
        "mu2": -0.01,
        "omega": 3.0,
        "sigma": 1.1,
    }
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, alpha, beta, mu1, mu2, omega, sigma):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = mu1 * x + sigma * x * y
        ydot = mu2 * y + omega * z + alpha * y * z + beta * z**2 - sigma * x**2
        zdot = mu2 * z - omega * y - alpha * y**2 - beta * y * z
        return xdot, ydot, zdot


class SaltonSea(ContinuousSystem):
    r"""Salton Sea eco-epidemiological (fish–bird disease) model.

    A three-variable eco-epidemiological model of the Salton Sea fish–bird
    system, in which susceptible tilapia ``x`` grow logistically and become
    infected (``y``) at rate ``lam``, and pelicans ``z`` prey on the infected
    fish through a Holling type-II response.  Upadhyay et al. showed the model
    exhibits chaos (a period-doubling route) for critical parameter values,
    a candidate explanation for the observed mass die-offs of birds and fish.

    Parameters
    ----------
    a
        Half-saturation constant of the predator's functional response.
    d
        Natural death rate of the pelican predator.
    k
        Carrying capacity of the susceptible fish population.
    lam
        Disease transmission (infection) rate among the fish.
    m
        Maximum predation rate of pelicans on infected fish.
    mu
        Death rate of infected fish (disease-induced mortality).
    r
        Intrinsic growth rate of the susceptible fish.
    th
        Conversion efficiency of consumed infected fish into new predators.

    Varying the fish growth rate ``r`` drives the system through a
    period-doubling cascade to a chaotic attractor.
    """

    reference = "Upadhyay, Bairagi, Kundu & Chattopadhyay (2008), Appl. Math. Comput. 196, 392-401"
    doi = "10.1016/j.amc.2007.06.007"
    params = {
        "a": 15,
        "d": 8.3,
        "k": 400,
        "lam": 0.06,
        "m": 15.5,
        "mu": 3.4,
        "r": 22,
        "th": 10.0,
    }
    dim = 3
    variables = ("x", "y", "z")

    @staticmethod
    def _equations(Y, t, *, a, d, k, lam, m, mu, r, th):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = r * x * (1 - (x + y) / k) - lam * x * y
        ydot = lam * x * y - m * y * z / (y + a) - mu * y
        zdot = th * y * z / (y + a) - d * z
        return xdot, ydot, zdot


class BlinkingVortex(BlinkingRotlet):
    """Blinking-vortex flow — Aref's paradigm of chaotic advection.

    The blinking-flow model of Aref: a passive tracer in a circular cell stirred
    by two agitators at ``±b`` that alternately switch on and off with period
    ``tau``.  It shares the rotlet velocity field and blinking machinery of
    :class:`BlinkingRotlet`, but at the vortex parameters (``bc = 0``), and is
    the classic demonstration that a time-periodic two-dimensional Stokes flow
    can advect tracers chaotically.  State is ``(r, theta, t)`` in polar
    coordinates with an explicit clock.

    Parameters
    ----------
    a, b, bc, sigma, tau : float
        As in :class:`BlinkingRotlet`; ``bc = 0`` selects the blinking-vortex
        limit.
    """

    reference = "Aref (1984), J. Fluid Mech. 143, 1-21"
    doi = "10.1017/s0022112084001233"
    params = {"a": 1.0, "b": 0.5, "bc": 0.0, "sigma": -1.0, "tau": 3.0}
    variables = ("r", "theta", "t")
    default_ic = [0.8, 4.887, 0.0]


class LidDrivenCavityFlow(ContinuousSystem):
    r"""Time-periodic lid-driven cavity flow (chaotic advection).

    Advection of a passive tracer in a two-dimensional cavity whose walls drive
    the interior with two spatial Fourier modes, the driving alternating in
    direction with period ``tau`` (a steep ``tanh`` blinking protocol).  The
    Stokes stream function is a closed-form combination of hyperbolic profiles;
    the periodic reversal folds and stretches material lines into a chaotic
    braid.  State is ``(x, y, t)`` — the tracer position plus an explicit clock.

    Parameters
    ----------
    a, b : float
        Cavity width and half-height.
    u1, u2 : float
        Amplitudes of the first and second driving Fourier modes.
    tau : float
        Period of the wall-driving reversal.
    """

    reference = "Grover, Ross, Stremler & Kumar (2012), Chaos 22, 043135"
    doi = "10.1063/1.4768666"
    params = {"a": 6.0, "b": 1.0, "tau": 1.1, "u1": 9.92786, "u2": 8.34932}
    dim = 3
    variables = ("x", "y", "t")
    default_ic = [3.9668, 0.1843, 0.0]

    @staticmethod
    def _lid(x, y, a, b, u1, u2):
        """Interior velocity field driven from one wall (Stokes stream function)."""
        prefactor1 = 2 * u1 * sin(pi * x / a) / (2 * b * pi + a * sinh(2 * pi * b / a))
        prefactor2 = 2 * u2 * sin(2 * pi * x / a) / (4 * b * pi + a * sinh(4 * pi * b / a))
        vx1 = -b * pi * sinh(pi * b / a) * sinh(pi * y / a) + cosh(pi * b / a) * (
            pi * y * cosh(pi * y / a) + a * sinh(pi * y / a)
        )
        vx2 = -2 * b * pi * sinh(2 * pi * b / a) * sinh(2 * pi * y / a) + cosh(2 * pi * b / a) * (
            2 * pi * y * cosh(2 * pi * y / a) + a * sinh(2 * pi * y / a)
        )
        vx = prefactor1 * vx1 + prefactor2 * vx2

        prefactor1 = 2 * pi * u1 * cos(pi * x / a) / (2 * b * pi + a * sinh(2 * pi * b / a))
        prefactor2 = 4 * pi * u2 * cos(2 * pi * x / a) / (4 * b * pi + a * sinh(4 * pi * b / a))
        vy1 = b * sinh(pi * b / a) * cosh(pi * y / a) - cosh(pi * b / a) * y * sinh(pi * y / a)
        vy2 = b * sinh(2 * pi * b / a) * cosh(2 * pi * y / a) - cosh(2 * pi * b / a) * y * sinh(
            2 * pi * y / a
        )
        vy = prefactor1 * vy1 + prefactor2 * vy2
        return vx, vy

    @staticmethod
    def _protocol(tt, tau, stiffness=20):
        return 0.5 + 0.5 * tanh(tau * stiffness * sin(2 * pi * tt / tau))

    @staticmethod
    def _equations(Y, t, *, a, b, tau, u1, u2):
        x, y, tt = Y(0), Y(1), Y(2)
        weight = LidDrivenCavityFlow._protocol(tt, tau)
        dx1, dy1 = LidDrivenCavityFlow._lid(x, y, a, b, u1, u2)
        dx2, dy2 = LidDrivenCavityFlow._lid(x, y, a, b, -u1, u2)
        dx = weight * dx1 + (1 - weight) * dx2
        dy = weight * dy1 + (1 - weight) * dy2
        return dx, dy, 1


class BickleyJet(ContinuousSystem):
    r"""Bickley jet — a kinematic model of a meandering geophysical jet.

    A time-dependent streamfunction model of a Bickley (``sech^2``) zonal jet
    perturbed by three Rossby-wave modes, used as a benchmark for Lagrangian
    coherent structures in the atmosphere and ocean.  The state is
    ``(y, x, z)``: the cross-stream and along-stream tracer coordinates plus a
    slow phase ``z`` advancing the wave train.  The three wave amplitudes,
    wavenumbers and phase speeds are fixed at their standard values.

    Parameters
    ----------
    ell : float
        Jet half-width.
    u : float
        Characteristic jet speed.
    omega : float
        Phase-advance rate of the wave train.
    """

    reference = "Hadjighasem, Karrasch, Teramoto & Haller (2016), Phys. Rev. E 93, 063107"
    doi = "10.1103/physreve.93.063107"
    params = {"ell": 1.77, "omega": 1.0, "u": 6.266e-05}
    dim = 3
    variables = ("y", "x", "z")
    default_ic = [-0.4, 0.3, 0.0]
    #: Fixed Rossby-wave amplitudes, wavenumbers and phase speeds (three modes).
    _EPS = (0.0075, 0.15, 0.3)
    _K = (0.313922, 0.627845, 0.941767)
    _SIGMA = (9.05854e-06, 1.28453e-05, 2.88863e-05)

    @staticmethod
    def _equations(Y, t, *, ell, omega, u):
        y, x, z = Y(0), Y(1), Y(2)
        sechy = 1 / cosh(y / ell)
        eps, k, sig = BickleyJet._EPS, BickleyJet._K, BickleyJet._SIGMA
        un = [k[i] * (x - z * sig[i]) for i in range(3)]
        dy = ell * u * sechy**2 * sum(eps[i] * k[i] * sin(un[i]) for i in range(3))
        dx = u * sechy**2 * (-1 - 2 * sum(cos(un[i]) * eps[i] for i in range(3)) * tanh(y / ell))
        dz = omega
        return dy, dx, dz


class InteriorSquirmer(ContinuousSystem):
    r"""Streamlines interior to an oscillating squirmer (low-Reynolds swimmer).

    The unsteady interior Stokes flow generated by a cylindrical squirmer whose
    surface actuation oscillates in time (a steep ``tanh`` protocol switching
    between two mode sets with period ``tau``).  A passive tracer in polar
    coordinates ``(r, theta)`` — plus an explicit clock ``t`` — is advected
    chaotically by the five-mode velocity field.

    Parameters
    ----------
    tau : float
        Period of the surface-actuation protocol.
    """

    reference = "Blake (1971), Bull. Aust. Math. Soc. 5, 255-264"
    doi = "10.1017/s0004972700047134"
    params = {"tau": 3.0}
    dim = 3
    variables = ("r", "theta", "t")
    default_ic = [0.1, 0.1, 0.1]
    #: The five radial (a) and tangential (g) surface-actuation mode amplitudes.
    _A = (0.5, 0.5, 0.5, 0.5, 0.5)
    _G = (0.5, 0.5, 0.5, 0.5, 0.5)

    @staticmethod
    def _protocol(tt, tau, stiffness=20):
        return 0.5 + 0.5 * tanh(tau * stiffness * sin(2 * pi * tt / tau))

    @staticmethod
    def _equations(Y, t, *, tau):
        r, th, tt = Y(0), Y(1), Y(2)
        phase = InteriorSquirmer._protocol(tt, tau)
        dr = 0
        vth = 0
        for n in range(1, 6):
            an = InteriorSquirmer._A[n - 1] * phase
            gn = InteriorSquirmer._G[n - 1] * (1 - phase)
            cn, sn, rn = cos(n * th), sin(n * th), r**n
            dr = dr + (gn * cn + an * sn) * (n * rn * (r**2 - 1)) / r
            vth = vth + (an * cn - gn * sn) * (2 * r + (r**2 - 1) * n / r) * rn
        return dr, vth / r, 1


__all__ = [
    "ArnoldBeltramiChildress",
    "AtmosphericRegime",
    "BickleyJet",
    "BlinkingRotlet",
    "BlinkingVortex",
    "DoubleGyre",
    "Hadley",
    "InteriorSquirmer",
    "LidDrivenCavityFlow",
    "OscillatingFlow",
    "RayleighBenard",
    "SaltonSea",
    "VallisElNino",
]


def __dir__() -> list[str]:
    """Expose only the catalogue classes (``__all__``) to ``dir()`` / autocomplete.

    ``__all__`` governs ``import *`` and nothing else, so without this the module
    also offers every helper it imported — SymEngine's ``sin``/``cos``/``exp``,
    ``numpy`` — as though they were part of this library's surface.
    """
    return sorted(__all__)
