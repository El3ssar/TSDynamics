from symengine import cos, pi, sin, tanh

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

    reference = "Vallis (1986), Science 232, 243-245"
    params = {"b": 102.0, "c": 3.0, "p": 0.0}
    dim = 3

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

    reference = "Saltzman (1962), J. Atmos. Sci. 19, 329-342"
    params = {"a": 30, "b": 5, "r": 18}
    dim = 3

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
    """

    reference = "Lorenz (1984), Tellus A 36, 98-110"
    params = {"a": 0.2, "b": 4.0, "f": 9.0, "g": 1.0}
    dim = 3

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
    params = {"alpha": 0.1, "eps": 0.1, "omega": 0.62832}
    dim = 3

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
    params = {"b": 0.48, "k": 1.0, "omega": 0.49, "u": 0.72}
    dim = 3

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

    reference = "Dombre, Frisch, Greene, Hénon, Mehr & Soward (1986), J. Fluid Mech. 167, 353-391"
    params = {"a": 1.73205, "b": 1.41421, "c": 1}
    dim = 3

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
    params = {
        "alpha": -2.0,
        "beta": -5.0,
        "mu1": 0.05,
        "mu2": -0.01,
        "omega": 3.0,
        "sigma": 1.1,
    }
    dim = 3

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

    @staticmethod
    def _equations(Y, t, *, a, d, k, lam, m, mu, r, th):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = r * x * (1 - (x + y) / k) - lam * x * y
        ydot = lam * x * y - m * y * z / (y + a) - mu * y
        zdot = th * y * z / (y + a) - d * z
        return xdot, ydot, zdot
