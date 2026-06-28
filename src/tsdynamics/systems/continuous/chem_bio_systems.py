from symengine import cos, exp, pi, sin

from tsdynamics.families import ContinuousSystem


class GlycolyticOscillation(ContinuousSystem):
    """
    Decroly–Goldbeter multiply-regulated biochemical oscillator.

    Two allosteric enzymes coupled in series, each activated by its own
    product, modelling a glycolytic-type biochemical system.  The successive
    product activations supply two nested feedback loops whose interaction
    produces a rich variety of temporal self-organization: simple limit-cycle
    oscillations, birhythmicity (coexisting stable cycles), and deterministic
    chaos.

    State variables are the (normalised) concentrations of the substrate
    ``a`` and the two product species ``b`` and ``c``.

    Parameters
    ----------
    nu : float
        Constant input rate of the substrate.
    s1, s2 : float
        Maximum rates of the first and second allosteric enzymes.
    q1, q2 : float
        Coupling / stoichiometric ratios linking the two enzymatic stages.
    l1, l2 : float
        Allosteric constants of the two enzymes (ratio of the inactive to the
        active conformation).
    k : float
        Removal (sink) rate of the final product ``c``.
    d : float
        Cooperativity modulation of the second enzyme's substrate binding.
    """

    params = {
        "d": 0.0,
        "k": 4.422,
        "l1": 500000000.0,
        "l2": 100,
        "nu": 1.0,
        "q1": 50,
        "q2": 0.02,
        "s1": 22.2222,
        "s2": 22.2222,
    }
    dim = 3
    reference = "Decroly & Goldbeter (1982), Proc. Natl. Acad. Sci. U.S.A. 79, 6917-6921"

    @staticmethod
    def _equations(Y, t, *, d, k, l1, l2, nu, q1, q2, s1, s2):
        a, b, c = Y(0), Y(1), Y(2)
        phi = (a * (1 + a) * (1 + b) ** 2) / (l1 + (1 + a) ** 2 * (1 + b) ** 2)
        eta = b * (1 + d * b) * (1 + c) ** 2 / (l2 + (1 + d * b) ** 2 * (1 + c) ** 2)
        adot = nu - s1 * phi
        bdot = q1 * s1 * phi - s2 * eta
        cdot = q2 * s2 * eta - k * c
        return adot, bdot, cdot


class Oregonator(ContinuousSystem):
    """
    Oregonator model of the Belousov–Zhabotinsky oscillating reaction.

    Field and Noyes's three-variable reduction of the FKN mechanism for the
    Belousov–Zhabotinsky reaction — the canonical model of an oscillating
    chemical reaction.  The variables ``X``, ``Y`` and ``Z`` are scaled
    concentrations of the key intermediates (HBrO₂, Br⁻ and the oxidised
    catalyst Ce⁴⁺).  The small parameters ``epsilon`` and ``mu`` separate the
    timescales, making the system **stiff** with a stable limit cycle; an
    explicit solver cannot integrate it, so the default method is the engine's
    variable-order BDF.

    Parameters
    ----------
    q : float
        Scaled rate-constant ratio of the bromide-consuming steps.
    f : float
        Stoichiometric factor for catalyst regeneration (the principal
        bifurcation parameter; oscillations occur for a range around ``f≈1``).
    epsilon, mu : float
        Timescale-separation parameters (``mu ≪ epsilon ≪ 1``) controlling the
        relaxation character and stiffness of the dynamics.
    """

    params = {
        "q": 2e-4,
        "f": 1,
        "mu": 1e-6,
        "epsilon": 1e-2,
    }
    dim = 3  # Three variables: X, Y, Z (reduced forms of the chemical species)
    reference = "Field & Noyes (1974), J. Chem. Phys. 60, 1877-1884"
    # Classic stiff system (Field–Noyes); an explicit solver cannot integrate
    # it, so default to the engine's variable-order BDF.
    _default_method = "bdf"

    @staticmethod
    def _equations(Y, t, *, q, f, mu, epsilon):
        """
        Right-hand side of the Belousov-Zhabotinsky (Oregonator) model.

        X: State vector [X, Y, Z]
        t: Time (not explicitly used in autonomous systems)
        q, f, epsilon: Model parameters
        """
        # Variables
        x, y, z = Y(0), Y(1), Y(2)

        # Oregonator equations
        xdot = y - x
        ydot = (q * z - z * y + y * (1 - y)) / epsilon
        zdot = (-q * z - y * z + f * x) / mu

        return xdot, ydot, zdot


class IsothermalChemical(ContinuousSystem):
    """
    Three-variable isothermal autocatalator (Petrov–Scott–Showalter).

    A prototype model of an isothermal chemical reaction with cubic
    autocatalytic feedback (the ``alpha·beta²`` term) and a slow third
    species ``gamma``.  As a bifurcation parameter is varied the system
    displays a supercritical Hopf bifurcation followed by period doubling,
    mixed-mode oscillations and chaos interleaved with periodic windows — the
    canonical mechanism for complex oscillations in unforced isothermal
    chemistry.

    State variables are the scaled concentrations of the autocatalyst
    ``alpha``, the intermediate ``beta`` and the slow species ``gamma``.

    Parameters
    ----------
    mu : float
        Scaled inflow / precursor-decay rate feeding the autocatalytic loop.
    kappa : float
        Coupling of the slow species ``gamma`` back into the ``alpha`` balance.
    sigma : float
        Timescale ratio of the fast autocatalyst ``beta`` (small ``sigma`` ⇒
        relaxation oscillations).
    delta : float
        Timescale ratio of the slow species ``gamma``.
    """

    params = {"delta": 1.0, "kappa": 2.5, "mu": 0.29786, "sigma": 0.013}
    dim = 3
    reference = "Petrov, Scott & Showalter (1992), J. Chem. Phys. 97, 6191-6198"

    @staticmethod
    def _equations(Y, t, *, delta, kappa, mu, sigma):
        alpha, beta, gamma = Y(0), Y(1), Y(2)
        alphadot = mu * (kappa + gamma) - alpha * beta**2 - alpha
        betadot = (alpha * beta**2 + alpha - beta) / sigma
        gammadot = (beta - gamma) / delta
        return alphadot, betadot, gammadot


class ForcedBrusselator(ContinuousSystem):
    """
    Periodically forced Brusselator.

    The Brusselator (Prigogine–Lefever) is the classic two-variable model of
    an autocatalytic chemical reaction exhibiting symmetry-breaking
    instabilities and limit-cycle oscillations.  Here it is driven by a weak
    periodic forcing ``f·cos(z)`` (the phase ``z`` advances at constant
    frequency ``w``), so the autonomous limit cycle competes with the external
    drive and the system can phase-lock, quasi-periodically modulate, or
    become chaotic.

    State variables are the activator ``x``, the inhibitor ``y`` and the
    forcing phase ``z``.

    Parameters
    ----------
    a : float
        Constant feed of the activator ``x``.
    b : float
        Control parameter; the unforced Brusselator passes through a Hopf
        bifurcation at ``b = 1 + a²``.
    f : float
        Amplitude of the periodic forcing.
    w : float
        Angular frequency of the forcing.
    """

    params = {"a": 0.4, "b": 1.2, "f": 0.05, "w": 0.81}
    dim = 3
    reference = "Prigogine & Lefever (1968), J. Chem. Phys. 48, 1695-1700"

    @staticmethod
    def _equations(Y, t, *, a, b, f, w):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a + x**2 * y - (b + 1) * x + f * cos(z)
        ydot = b * x - x**2 * y
        zdot = w
        return xdot, ydot, zdot


class CircadianRhythm(ContinuousSystem):
    """
    Goldbeter molecular model of circadian oscillations (Drosophila PER).

    A model for circadian rhythms based on the negative feedback exerted by
    the PER protein on the transcription of its own (``per``) gene, together
    with multiple phosphorylation of PER.  The negative feedback plus the
    phosphorylation-induced delays generate self-sustained limit-cycle
    oscillations with a near-24 h period.  This variant carries an explicit
    circadian phase ``th`` (advancing at ``2π/24`` per hour) that drives a
    periodically modulated transcription rate.

    State variables are the ``per`` mRNA ``m``, the cytosolic PER protein
    ``fc``, its phosphorylated form ``fs``, the nuclear PER ``fn``, and the
    driving phase ``th``.

    Parameters
    ----------
    vs, vm : float
        Maximum rates of mRNA synthesis and degradation.
    vmax, vmin : float
        Bounds of the phase-modulated transcription rate driven by ``th``.
    ks : float
        Rate of PER synthesis from mRNA.
    vd, vdn : float
        Maximum degradation rates of cytosolic and nuclear PER.
    k1, k2 : float
        Rate constants for PER transport into and out of the nucleus.
    k : float
        Phosphorylation/turnover rate of cytosolic PER.
    Ki : float
        Threshold constant for repression of transcription by nuclear PER.
    n : float
        Hill coefficient of the transcriptional repression.
    km, kd, kdn : float
        Michaelis constants of mRNA degradation and the two PER degradations.
    """

    params = {
        "Ki": 1,
        "k": 0.5,
        "k1": 0.3,
        "k2": 0.15,
        "kd": 1.4,
        "kdn": 0.4,
        "km": 0.4,
        "ks": 1,
        "n": 4,
        "vd": 6,
        "vdn": 1.5,
        "vm": 0.7,
        "vmax": 4.7,
        "vmin": 1.0,
        "vs": 6,
    }
    dim = 5
    reference = "Goldbeter (1995), Proc. R. Soc. Lond. B 261, 319-324"

    @staticmethod
    def _equations(
        Y,
        t,
        *,
        Ki,
        k,
        k1,
        k2,
        kd,
        kdn,
        km,
        ks,
        n,
        vd,
        vdn,
        vm,
        vmax,
        vmin,
        vs,
    ):
        m, fc, fs, fn, th = Y(0), Y(1), Y(2), Y(3), Y(4)
        # The transcription rate is driven by the circadian phase ``th``; the
        # scalar ``vs`` parameter is overridden here and has no effect on the
        # dynamics (kept in ``params`` for catalogue-contract stability).
        vs_t = 2.5 * ((0.5 + 0.5 * cos(th)) + vmin) * (vmax - vmin)
        mdot = vs_t * (Ki**n) / (Ki**n + fn**n) - vm * m / (km + m)
        fcdot = ks * m - k1 * fc + k2 * fn - k * fc
        fsdot = k * fc - vd * fs / (kd + fs)
        fndot = k1 * fc - k2 * fn - vdn * fn / (kdn + fn)
        thdot = 2 * pi / 24
        return mdot, fcdot, fsdot, fndot, thdot


class CaTwoPlus(ContinuousSystem):
    """
    Intracellular Ca²⁺ oscillator with self-modulated InsP₃ (Houart et al.).

    A three-variable model of cytosolic calcium signalling based on
    Ca²⁺-induced Ca²⁺ release, extended by self-modulation of the inositol
    1,4,5-trisphosphate (InsP₃) signal through Ca²⁺-stimulated InsP₃
    degradation.  Beyond simple periodic Ca²⁺ spikes this feedback yields
    complex behaviour: bursting, chaos and birhythmicity.

    State variables are the cytosolic Ca²⁺ ``z``, the Ca²⁺ in the
    InsP₃-insensitive store ``y``, and the InsP₃ concentration ``a``.

    Parameters
    ----------
    V0, V1 : float
        Constant and stimulus-dependent Ca²⁺ influx into the cytosol.
    beta : float
        Saturation of the external (InsP₃-generating) stimulus, ``0 ≤ beta ≤ 1``.
    Vm2, Vm3, Vm5 : float
        Maximum rates of Ca²⁺ pumping into the store, CICR-mediated release,
        and the InsP₃-degrading kinase.
    V4 : float
        Maximum rate of stimulus-induced InsP₃ synthesis.
    K2, Kz, Ky, Ka, K5, Kd : float
        Threshold (Michaelis) constants of the corresponding fluxes.
    m, n, p : float
        Hill coefficients of the CICR and InsP₃-degradation terms.
    kf : float
        Passive Ca²⁺ leak from the store into the cytosol.
    k : float
        Linear cytosolic Ca²⁺ removal (e.g. efflux across the plasma membrane).
    eps : float
        Linear degradation rate of InsP₃.
    """

    params = {
        "K2": 0.1,
        "K5": 0.3194,
        "Ka": 0.1,
        "Kd": 1,
        "Ky": 0.3,
        "Kz": 0.6,
        "V0": 2,
        "V1": 2,
        "V4": 3,
        "Vm2": 6,
        "Vm3": 30,
        "Vm5": 50,
        "beta": 0.65,
        "eps": 13,
        "k": 10,
        "kf": 1,
        "m": 2,
        "n": 4,
        "p": 1,
    }
    dim = 3
    reference = "Houart, Dupont & Goldbeter (1999), Bull. Math. Biol. 61, 507-530"

    @staticmethod
    def _equations(
        Y,
        t,
        *,
        K2,
        K5,
        Ka,
        Kd,
        Ky,
        Kz,
        V0,
        V1,
        V4,
        Vm2,
        Vm3,
        Vm5,
        beta,
        eps,
        k,
        kf,
        m,
        n,
        p,
    ):
        z, y, a = Y(0), Y(1), Y(2)
        Vin = V0 + V1 * beta
        V2 = Vm2 * (z**2) / (K2**2 + z**2)
        V3 = (Vm3 * (z**m) / (Kz**m + z**m)) * (y**2 / (Ky**2 + y**2)) * (a**4 / (Ka**4 + a**4))
        V5 = Vm5 * (a**p / (K5**p + a**p)) * (z**n / (Kd**n + z**n))
        zdot = Vin - V2 + V3 + kf * y - k * z
        ydot = V2 - V3 - kf * y
        adot = beta * V4 - V5 - eps * a
        return zdot, ydot, adot


class ExcitableCell(ContinuousSystem):
    """
    Chay three-variable model of an excitable (pancreatic β-) cell.

    A Hodgkin–Huxley-style conductance model of a bursting excitable cell.
    The membrane potential ``v`` and the fast voltage-gated K⁺ activation
    ``n`` provide the spiking, while the slow intracellular Ca²⁺
    concentration ``c`` modulates a Ca²⁺-activated K⁺ current and forms the
    slow wave that organises bursting.  Across the beating–bursting transition
    the model shows period doubling and deterministic chaos.

    State variables are the membrane potential ``v``, the K⁺ activation
    gate ``n`` and the intracellular Ca²⁺ ``c``.

    Parameters
    ----------
    gi, gkv, gkc, gl : float
        Maximum conductances of the mixed Na⁺/Ca²⁺ inward current, the
        voltage-gated K⁺ current, the Ca²⁺-activated K⁺ current, and the leak.
    vi, vk, vl : float
        Reversal potentials of the inward, K⁺ and leak currents.
    vc : float
        Reversal potential driving Ca²⁺ entry in the Ca²⁺ balance.
    vm, vn : float
        Half-activation voltages of the gating rate functions.
    kc : float
        Rate of intracellular Ca²⁺ removal.
    rho : float
        Slow timescale factor of the Ca²⁺ dynamics.
    """

    params = {
        "gi": 1800,
        "gkc": 11,
        "gkv": 1700,
        "gl": 7,
        "kc": 0.183333,
        "rho": 0.27,
        "vc": 100,
        "vi": 100,
        "vk": -75,
        "vl": -40,
        "vm": -50,
        "vn": -30,
    }
    dim = 3
    reference = "Chay (1985), Physica D 16, 233-242"

    @staticmethod
    def _equations(Y, t, *, gi, gkc, gkv, gl, kc, rho, vc, vi, vk, vl, vm, vn):
        v, n, c = Y(0), Y(1), Y(2)

        alpham = 0.1 * (25 + v) / (1 - exp(-0.1 * v - 2.5))
        betam = 4 * exp(-(v + 50) / 18)
        minf = alpham / (alpham + betam)

        alphah = 0.07 * exp(-0.05 * v - 2.5)
        betah = 1 / (1 + exp(-0.1 * v - 2))
        hinf = alphah / (alphah + betah)

        alphan = 0.01 * (20 + v) / (1 - exp(-0.1 * v - 2))
        betan = 0.125 * exp(-(v + 30) / 80)
        ninf = alphan / (alphan + betan)
        tau = 1 / (230 * (alphan + betan))

        ca = c / (1 + c)

        vdot = (
            gi * minf**3 * hinf * (vi - v)
            + gkv * n**4 * (vk - v)
            + gkc * ca * (vk - v)
            + gl * (vl - v)
        )
        ndot = (ninf - n) / tau
        cdot = rho * (minf**3 * hinf * (vc - v) - kc * c)
        return vdot, ndot, cdot


class CellCycle(ContinuousSystem):
    """
    Two coupled biochemical oscillators driving the cell cycle (Romond et al.).

    Two minimal Goldbeter-type cyclin/cdk oscillators — one for the G2/M
    transition (cdk1) and one for the G1/S transition (cdk2) — coupled through
    mutual inhibition, so they fire in alternation to drive successive phases
    of the cell cycle.  Each subsystem is a three-variable cascade (cyclin,
    active cdk, and the cyclin-degrading enzyme); together they produce
    alternating oscillations and, in parameter regimes, chaos.

    State variables are ``(c1, m1, x1)`` for the first oscillator and
    ``(c2, m2, x2)`` for the second: cyclin concentration ``c``, active cdk
    fraction ``m`` and active protease fraction ``x``.

    Parameters
    ----------
    vi : float
        Cyclin synthesis rate (gated by the partner oscillator's cdk).
    Kim : float
        Inhibition constant of the mutual coupling between the two oscillators.
    vd, Kd1, kd1 : float
        Maximum rate, Michaelis constant and basal rate of cyclin degradation.
    Kc : float
        Threshold for cdk activation by cyclin.
    Vm1, V2 : float
        Maximum rates of cdk activation and inactivation.
    Vm3, V4 : float
        Maximum rates of protease activation and inactivation.
    K : float
        Common Michaelis constant of the cdk- and protease-modification cycles.
    """

    params = {
        "K": 0.01,
        "Kc": 0.5,
        "Kd1": 0.02,
        "Kim": 0.65,
        "V2": 0.15,
        "V4": 0.05,
        "Vm1": 0.3,
        "Vm3": 0.1,
        "kd1": 0.001,
        "vd": 0.025,
        "vi": 0.05,
    }
    dim = 6
    reference = "Romond, Rustici, Gonze & Goldbeter (1999), Ann. N.Y. Acad. Sci. 879, 180-193"

    @staticmethod
    def _equations(Y, t, *, K, Kc, Kd1, Kim, V2, V4, Vm1, Vm3, kd1, vd, vi):
        c1, m1, x1, c2, m2, x2 = Y(0), Y(1), Y(2), Y(3), Y(4), Y(5)
        Vm1, Um1 = 2 * [Vm1]
        vi1, vi2 = 2 * [vi]
        H1, H2, H3, H4 = 4 * [K]
        K1, K2, K3, K4 = 4 * [K]
        V2, U2 = 2 * [V2]
        Vm3, Um3 = 2 * [Vm3]
        V4, U4 = 2 * [V4]
        Kc1, Kc2 = 2 * [Kc]
        vd1, vd2 = 2 * [vd]
        Kd1, Kd2 = 2 * [Kd1]
        kd1, kd2 = 2 * [kd1]
        Kim1, Kim2 = 2 * [Kim]
        V1 = Vm1 * c1 / (Kc1 + c1)
        U1 = Um1 * c2 / (Kc2 + c2)
        V3 = m1 * Vm3
        U3 = m2 * Um3
        c1dot = vi1 * Kim1 / (Kim1 + m2) - vd1 * x1 * c1 / (Kd1 + c1) - kd1 * c1
        c2dot = vi2 * Kim2 / (Kim2 + m1) - vd2 * x2 * c2 / (Kd2 + c2) - kd2 * c2
        m1dot = V1 * (1 - m1) / (K1 + (1 - m1)) - V2 * m1 / (K2 + m1)
        m2dot = U1 * (1 - m2) / (H1 + (1 - m2)) - U2 * m2 / (H2 + m2)
        x1dot = V3 * (1 - x1) / (K3 + (1 - x1)) - V4 * x1 / (K4 + x1)
        x2dot = U3 * (1 - x2) / (H3 + (1 - x2)) - U4 * x2 / (H4 + x2)
        return c1dot, m1dot, x1dot, c2dot, m2dot, x2dot


class HindmarshRose(ContinuousSystem):
    """
    Hindmarsh–Rose model of neuronal spiking and bursting.

    A three-variable reduction of neuronal dynamics capturing the
    spiking–bursting behaviour of a single neuron's membrane potential.  The
    membrane potential ``x`` and the fast recovery (spiking) variable ``y``
    produce the spikes, while the slow adaptation current ``z`` increments at
    each spike and modulates the firing rate, giving rise to bursts and,
    for suitable parameters, chaotic bursting.

    State variables are the membrane potential ``x``, the spiking variable
    ``y`` and the slow adaptation current ``z``.

    Parameters
    ----------
    a, b : float
        Coefficients of the cubic and quadratic terms shaping the fast (spike)
        nonlinearity.
    c, d : float
        Constants of the fast recovery dynamics.
    s : float
        Strength of the slow adaptation feedback.
    tx, tz : float
        Timescale constants of the fast (``x``) and slow (``z``) variables
        (small ``tx`` ⇒ fast spiking, larger ``tz`` ⇒ slow bursting envelope).
    """

    params = {
        "a": 0.49,
        "b": 1.0,
        "c": 0.0322,
        "d": 1.0,
        "s": 1.0,
        "tx": 0.03,
        "tz": 0.8,
    }
    dim = 3
    reference = "Hindmarsh & Rose (1984), Proc. R. Soc. Lond. B 221, 87-102"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, s, tx, tz):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -x + 1 / tx * y - a / tx * x**3 + b / tx * x**2 + 1 / tx * z
        ydot = -a * x**3 - (d - b) * x**2 + z
        zdot = -s / tz * x - 1 / tz * z + c / tz
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c, d, s, tx, tz):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-1 - 3 * a / tx * x**2 + 2 * b / tx * x, 1 / tx, 1 / tx]
        row2 = [-3 * a * x**2 - 2 * (d - b) * x, 0, 1]
        row3 = [-s / tz, 0, -1 / tz]
        return row1, row2, row3


class ForcedVanDerPol(ContinuousSystem):
    """
    Periodically forced van der Pol oscillator.

    The van der Pol oscillator is the archetypal self-sustained relaxation
    oscillator: a damped oscillator with nonlinear (amplitude-dependent)
    damping ``mu·(1 − x²)`` that pumps energy into small oscillations and
    dissipates it from large ones, producing a stable limit cycle.  Driving it
    with a sinusoidal force ``a·sin(z)`` (phase ``z`` advancing at frequency
    ``w``) yields the classic forced system whose interplay of intrinsic and
    driving frequencies gives entrainment, quasi-periodicity and chaos.

    State variables are the position ``x``, the velocity ``y`` and the
    forcing phase ``z``.

    Parameters
    ----------
    mu : float
        Nonlinear damping strength (large ``mu`` ⇒ strongly relaxational).
    a : float
        Amplitude of the periodic forcing.
    w : float
        Angular frequency of the forcing.
    """

    params = {"a": 1.2, "mu": 8.53, "w": 0.63}
    dim = 3
    reference = "van der Pol (1926), London Edinburgh Dublin Philos. Mag. J. Sci. 2, 978-992"

    @staticmethod
    def _equations(Y, t, *, a, mu, w):
        x, y, z = Y(0), Y(1), Y(2)
        ydot = mu * (1 - x**2) * y - x + a * sin(z)
        xdot = y
        zdot = w
        return xdot, ydot, zdot


class ForcedFitzHughNagumo(ContinuousSystem):
    """
    Periodically forced FitzHugh–Nagumo excitable system.

    The FitzHugh–Nagumo model is a two-variable reduction of the
    Hodgkin–Huxley nerve-membrane equations: a fast excitable voltage
    variable ``v`` with cubic nonlinearity ``v − v³/3`` and a slow linear
    recovery variable ``w``.  Adding a periodic stimulus ``curr + f·sin(z)``
    (phase ``z`` advancing at frequency ``omega``) makes the excitable
    element fire in response to the drive, producing entrainment and chaotic
    spike trains.

    State variables are the fast voltage ``v``, the recovery variable ``w``
    and the forcing phase ``z``.

    Parameters
    ----------
    curr : float
        Constant baseline stimulus current.
    f : float
        Amplitude of the periodic stimulus.
    omega : float
        Angular frequency of the periodic stimulus.
    a, b : float
        Constants of the slow recovery dynamics.
    gamma : float
        Recovery (slow-variable) timescale factor.
    """

    params = {
        "a": 0.7,
        "b": 0.8,
        "curr": 0.965,
        "f": 0.4008225,
        "gamma": 0.08,
        "omega": 0.043650793650793655,
    }
    dim = 3
    reference = "FitzHugh (1961), Biophys. J. 1, 445-466"

    @staticmethod
    def _equations(Y, t, *, a, b, curr, f, gamma, omega):
        v, w, z = Y(0), Y(1), Y(2)
        vdot = v - v**3 / 3 - w + curr + f * sin(z)
        wdot = gamma * (v + a - b * w)
        zdot = omega
        return vdot, wdot, zdot


class TurchinHanski(ContinuousSystem):
    """
    Turchin–Hanski predator–prey model of vole population cycles.

    An empirically grounded model of boreal vole dynamics: a prey population
    ``n`` (voles) regulated by an interaction with a specialist predator ``p``
    (weasels), under a seasonally varying (sinusoidal) environment carried by
    the phase ``z``.  The model reproduces the latitudinal gradient of vole
    dynamics — stable at low latitudes, oscillatory and ultimately chaotic at
    high latitudes — through generalist-predation, self-limitation and
    seasonal forcing.

    State variables are the prey density ``n``, the predator density ``p``
    and the seasonal phase ``z``.

    Parameters
    ----------
    r : float
        Intrinsic growth rate of the prey.
    e : float
        Amplitude of the seasonal modulation of the growth rates.
    g, h : float
        Maximum rate and half-saturation of generalist (saturating) predation
        on the prey.
    a, d : float
        Attack rate and half-saturation of the specialist predator's functional
        response.
    s : float
        Intrinsic growth/self-limitation rate of the specialist predator.
    """

    params = {"a": 8, "d": 0.04, "e": 0.5, "g": 0.1, "h": 0.8, "r": 8.12, "s": 1.25}
    dim = 3
    reference = "Turchin & Hanski (1997), Am. Nat. 149, 842-874"

    @staticmethod
    def _equations(Y, t, *, a, d, e, g, h, r, s):
        n, p, z = Y(0), Y(1), Y(2)
        ndot = (
            r * (1 - e * sin(z)) * n - r * (n**2) - g * (n**2) / (n**2 + h**2) - a * n * p / (n + d)
        )
        pdot = s * (1 - e * sin(z)) * p - s * (p**2) / n
        zdot = 2 * pi
        return ndot, pdot, zdot


class HastingsPowell(ContinuousSystem):
    """
    Hastings–Powell three-species food chain.

    A continuous-time food-chain model — resource ``x``, consumer ``y`` and
    top predator ``z`` — with Holling type-II (saturating) functional
    responses between trophic levels.  For biologically reasonable parameters
    the long-term dynamics are chaotic, with the characteristic "tea-cup"
    strange attractor; it is a classic demonstration that chaos may be common
    in natural food webs.

    State variables are the resource ``x``, the consumer ``y`` and the top
    predator ``z``.

    Parameters
    ----------
    a1, b1 : float
        Maximum attack rate and half-saturation of the consumer feeding on the
        resource.
    a2, b2 : float
        Maximum attack rate and half-saturation of the predator feeding on the
        consumer.
    d1, d2 : float
        Death rates of the consumer and the top predator.
    """

    params = {"a1": 5.0, "a2": 0.1, "b1": 3.0, "b2": 2.0, "d1": 0.4, "d2": 0.01}
    dim = 3
    reference = "Hastings & Powell (1991), Ecology 72, 896-903"

    @staticmethod
    def _equations(Y, t, *, a1, a2, b1, b2, d1, d2):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = x * (1 - x) - y * a1 * x / (1 + b1 * x)
        ydot = y * a1 * x / (1 + b1 * x) - z * a2 * y / (1 + b2 * y) - d1 * y
        zdot = z * a2 * y / (1 + b2 * y) - d2 * z
        return xdot, ydot, zdot


class ItikBanksTumor(ContinuousSystem):
    """
    Itik–Banks three-dimensional cancer model.

    A Lotka–Volterra-type model of tumour growth describing the competitive
    interactions between tumour cells ``x``, healthy host cells ``y`` and
    effector immune cells ``z``.  Tumour and healthy cells grow logistically
    and compete, while the immune response is recruited by (and acts on) the
    tumour.  For a range of parameters the interactions are chaotic, giving a
    strange attractor in the three-population space.

    State variables are the tumour-cell ``x``, healthy-cell ``y`` and
    effector immune-cell ``z`` populations.

    Parameters
    ----------
    a12, a13 : float
        Competition coefficients of healthy cells and immune cells on the
        tumour.
    a21 : float
        Competition coefficient of the tumour on the healthy cells.
    a31 : float
        Inactivation rate of immune cells by the tumour.
    r2 : float
        Growth rate of the healthy host cells.
    r3, k3 : float
        Maximum recruitment rate and half-saturation of the immune response to
        the tumour.
    d3 : float
        Natural death rate of the effector immune cells.
    """

    params = {
        "a12": 1,
        "a13": 2.5,
        "a21": 1.5,
        "a31": 0.2,
        "d3": 0.5,
        "k3": 1,
        "r2": 0.6,
        "r3": 4.5,
    }
    dim = 3
    reference = "Itik & Banks (2010), Int. J. Bifurcation Chaos 20, 71-79"

    @staticmethod
    def _equations(Y, t, *, a12, a13, a21, a31, d3, k3, r2, r3):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = x * (1 - x) - a12 * x * y - a13 * x * z
        ydot = r2 * y * (1 - y) - a21 * x * y
        zdot = r3 * x * z / (x + k3) - a31 * x * z - d3 * z
        return xdot, ydot, zdot
