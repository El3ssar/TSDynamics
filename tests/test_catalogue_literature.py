"""
Catalogue literature / analytic correctness checks across categories.

The bulk per-system sweeps only smoke-test the catalogue (shape + finiteness).
This module pins a curated set of *catalogue* systems to **analytically known**
quantities — exact Lyapunov exponents of conjugate / piecewise-linear maps,
energy conservation of a Hamiltonian flow, a closed-form parametric solution, a
constant phase-space-contraction identity, and an analytic equilibrium — so a
transcription bug or a backend regression that a smoke test would miss is caught
with a defensible expected number.

Every check cites its source inline.  None duplicates ``test_known_values.py``
(literature Lyapunov spectra via ``known_lyapunov`` metadata),
``test_known_quantifiers.py`` (estimator identities on synthetic signals),
``test_dimensions.py`` or ``test_fixed_points.py`` (Hénon / Logistic / Lorenz /
Rössler fixed points).  In particular the Logistic-r=4 ``ln 2`` exponent is
already covered there, so the map-Lyapunov checks here use *different* maps
(Tent, Chebyshev, Ulam, Gingerbreadman).

All randomness is seeded and horizons are short (fast tier).  Tolerances are
sized for finite-time / finite-step estimates and documented at each assertion.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

import tsdynamics as ts
from tsdynamics.data import Box

# ---------------------------------------------------------------------------
# Exact map Lyapunov exponents (chaotic_maps / geometric_maps / population_maps)
#
# Piecewise-linear and Chebyshev maps have a *constant* slope magnitude on the
# attractor, so their Lyapunov exponent is exact (no invariant-measure average
# is needed).  These are not the Logistic-r=4 value already pinned in
# test_known_values.py.
# ---------------------------------------------------------------------------

_IC1D = [0.123456789]  # a generic non-periodic seed on the interval


def test_tent_full_height_lyapunov_is_ln2() -> None:
    """Tent map at mu=1 has slope magnitude 2 everywhere → lambda = ln 2.

    The full-height tent is conjugate to the Bernoulli shift; each branch has
    constant slope +/-2, so lambda = ln 2 exactly (Ott, *Chaos in Dynamical
    Systems*, 2nd ed., Sec. 2.2).
    """
    le = ts.analysis.lyapunov_spectrum(ts.systems.Tent(params={"mu": 1.0}), n=10_000, ic=_IC1D)[0]
    # Slope is constant so the estimate is exact up to float roundoff.
    assert le == pytest.approx(np.log(2.0), abs=1e-4)


def test_tent_general_slope_lyapunov_is_ln_2mu() -> None:
    """Tent map at mu=0.7 has slope magnitude 2*mu → lambda = ln(2*mu).

    The branch slope of x' = mu*(1-2|x-1/2|) is +/-2*mu, giving the analytic
    exponent ln(2*mu) wherever the orbit stays on the attractor (the docstring's
    own statement, and the standard piecewise-linear result).
    """
    le = ts.analysis.lyapunov_spectrum(ts.systems.Tent(params={"mu": 0.7}), n=10_000, ic=_IC1D)[0]
    assert le == pytest.approx(np.log(2.0 * 0.7), abs=1e-4)


def test_chebyshev_degree_two_lyapunov_is_ln2() -> None:
    """Chebyshev map T_2 has constant Lyapunov exponent ln 2.

    For integer degree a >= 2 the Chebyshev map x' = cos(a*arccos x) is exact
    with Lyapunov exponent ln a (Adler & Rivlin 1964, Proc. AMS 15, 794-796).
    The a=2 case is conjugate to the logistic map at r=4 → ln 2.
    """
    le = ts.analysis.lyapunov_spectrum(ts.systems.Chebyshev(params={"a": 2.0}), n=10_000, ic=[0.3])[
        0
    ]
    assert le == pytest.approx(np.log(2.0), abs=1e-3)


def test_chebyshev_degree_six_lyapunov_is_ln6() -> None:
    """Chebyshev map at a=6 has constant Lyapunov exponent ln 6 (Adler-Rivlin 1964)."""
    le = ts.analysis.lyapunov_spectrum(ts.systems.Chebyshev(params={"a": 6.0}), n=10_000, ic=[0.3])[
        0
    ]
    assert le == pytest.approx(np.log(6.0), abs=1e-3)


def test_ulam_map_lyapunov_is_ln2() -> None:
    """Ulam-von Neumann map x' = 1 - 2x^2 is conjugate to logistic r=4 → ln 2.

    Ergodic with a smooth invariant density and Lyapunov exponent ln 2
    (Ulam & von Neumann 1947, Bull. AMS 53, 1120).  A finite-step average over a
    smooth measure converges more slowly than the piecewise-linear maps, so the
    tolerance is looser.
    """
    le = ts.analysis.lyapunov_spectrum(ts.systems.Ulam(), n=20_000, ic=[0.1])[0]
    assert le == pytest.approx(np.log(2.0), abs=2e-2)


def test_gingerbreadman_is_area_preserving() -> None:
    """The Gingerbreadman map is area-preserving → the two exponents sum to 0.

    x' = 1 - y + |x|, y' = x has Jacobian determinant
    sign(x)*0 - (-1)*1 = 1 everywhere, so it is conservative (Devaney 1984,
    Physica D 10, 387-393): lambda_1 + lambda_2 = ln|det J| = 0 exactly.
    """
    spec = ts.analysis.lyapunov_spectrum(ts.systems.Gingerbreadman(), n=10_000, ic=[0.5, 3.7])
    assert spec.shape == (2,)
    # det J == 1 identically, so the sum is zero to estimator roundoff.
    assert spec.sum() == pytest.approx(0.0, abs=1e-6)
    # ...and it is genuinely chaotic (a positive leading exponent), not a fixed
    # point with two zeros — guards against a degenerate "0 = 0" tautology.
    assert spec[0] > 0.01


def test_gingerbreadman_default_ic_is_in_the_chaotic_sea() -> None:
    """The class default lands on the chaotic sea, not on a periodic island.

    The map is conservative, so the invariant set is chosen entirely by the
    initial condition and much of the unit square (e.g. [0.5, 0.5]) sits on a
    period-6 island. ``default_ic`` must therefore pick the sea.
    """
    spec = ts.analysis.lyapunov_spectrum(ts.systems.Gingerbreadman(), n=10_000)
    assert spec[0] > 0.01, f"default IC is not chaotic: {spec}"


@pytest.mark.parametrize("alpha", [0.5, 0.3])
def test_baker_exponents_are_ln2_and_ln_alpha(alpha: float) -> None:
    """Baker's map has exact exponents (ln 2, ln alpha) and det J = 2*alpha.

    The classical stretch-cut-stack map x' = 2x mod 1,
    y' = alpha*y (+ 1-alpha on the right half) expands x by exactly 2 and
    contracts y by exactly alpha at every point (Hopf 1937), so both exponents
    are constant — no invariant-measure average is needed — and their sum is the
    constant ln|det J| = ln(2*alpha). At alpha = 0.5 the map is the classic
    measure-preserving baker's transformation and the sum is exactly 0.

    This is the identity that the pre-v6 implementation violated: it branched on
    ``y`` and expanded *both* coordinates, giving |det J| = 4 and two positive
    exponents in an invertible 2-D map.
    """
    spec = ts.analysis.lyapunov_spectrum(
        ts.systems.Baker(params={"alpha": alpha}), steps=20_000, ic=[0.31415926535, 0.2718281828]
    )
    assert spec.shape == (2,)
    # Constant slopes → the estimate is exact up to float roundoff.
    assert spec[0] == pytest.approx(np.log(2.0), abs=1e-6)
    assert spec[1] == pytest.approx(np.log(alpha), abs=1e-6)
    assert spec.sum() == pytest.approx(np.log(2.0 * alpha), abs=1e-6)


def test_baker_orbits_do_not_collapse() -> None:
    """A Baker orbit stays non-degenerate for thousands of iterations.

    The exact doubling map ``(2*x) % 1`` drains a float mantissa one bit per
    step and every orbit reaches the (0, 0) fixed point in ~53 iterations; the
    kernel wraps just below 1 to avoid it (as :class:`KaplanYorke` does). Check
    a spread of initial conditions really do survive.
    """
    rng = np.random.default_rng(0)
    baker = ts.systems.Baker()
    for _ in range(25):
        traj = baker.run(steps=5_000, ic=rng.random(2))
        uniques = len(np.unique(np.round(traj.y, 9), axis=0))
        assert uniques > 4_000, f"orbit collapsed to {uniques} distinct points"


def test_zaslavskii_exponent_sum_is_minus_r() -> None:
    """The Zaslavsky map contracts phase-space area at the constant rate ``-r``.

    det J = exp(-r) identically (the kick's two off-diagonal contributions
    cancel), so lambda_1 + lambda_2 = -r exactly (Zaslavsky 1978, Phys. Lett. A
    69, 145-147) — whatever the orbit does. Pinned together with a positive
    leading exponent, since the pre-v6 default parameters (eps=5) collapsed the
    orbit onto a stable period-2 cycle where the sum identity also holds.
    """
    r = ts.systems.Zaslavskii().params["r"]
    spec = ts.analysis.lyapunov_spectrum(ts.systems.Zaslavskii(), n=20_000)
    assert spec.shape == (2,)
    assert spec.sum() == pytest.approx(-r, abs=1e-6)
    assert spec[0] > 0.5, f"default parameters are not chaotic: {spec}"


# ---------------------------------------------------------------------------
# Hamiltonian energy conservation (chaotic_attractors: HenonHeiles)
# ---------------------------------------------------------------------------


def test_henon_heiles_energy_is_conserved() -> None:
    """Energy is conserved along a Hénon-Heiles orbit to integration tolerance.

    The Hénon-Heiles Hamiltonian (Hénon & Heiles 1964, Astron. J. 69, 73-79;
    with the catalogue's lam=1) is

        H = 1/2 (px^2 + py^2) + 1/2 (x^2 + y^2) + x^2 y - y^3/3.

    On a bounded low-energy orbit H must stay constant; a symplectic-energy drift
    far above the integrator tolerance would signal a sign/transcription bug in
    the force law.
    """

    def energy(state: np.ndarray) -> float:
        x, y, px, py = state
        return 0.5 * (px**2 + py**2) + 0.5 * (x**2 + y**2) + x**2 * y - y**3 / 3.0

    ic = [0.0, 0.1, 0.4, 0.0]  # bounded sub-escape orbit (E ~ 0.085)
    traj = ts.systems.HenonHeiles().run(final_time=100.0, dt=0.05, ic=ic, rtol=1e-10, atol=1e-10)
    e = np.array([energy(s) for s in traj.y])
    e0 = energy(np.asarray(ic, dtype=float))
    # Tolerant of solver drift (rtol/atol 1e-10), strict enough to catch a wrong
    # force term (which would drift by O(1) over 2000 steps).
    assert np.max(np.abs(e - e0)) < 1e-6


def test_double_pendulum_conserves_the_compound_rod_hamiltonian() -> None:
    """The double pendulum conserves the *compound-rod* energy, not some other one.

    For two identical uniform rods (mass m, length d) the Hamiltonian is

        H = (6/(m d^2)) (2 p1^2 + 8 p2^2 - 6 p1 p2 cos(th1-th2))
            / (2 (16 - 9 cos^2(th1-th2)))
            - (1/2) m g d (3 cos th1 + cos th2)

    (Marion, *Classical Dynamics*).  The factor 3 sits on cos(th1) alone — the
    upper rod carries its own weight plus the whole weight of the rod below it.
    A kernel with the wrong torque coefficient is still Hamiltonian, just for a
    *different* potential, so this exact H is the discriminating check: with the
    pre-v6 spurious factor 3 on sin(th2) it drifts by ~0.2 on |H| ~ 19.
    """
    system = ts.systems.DoublePendulum()
    d, m = system.params["d"], system.params["m"]
    g = 9.82  # the value DoublePendulum._equations uses

    def energy(state: np.ndarray) -> np.ndarray:
        th1, th2, p1, p2 = state.T
        c = np.cos(th1 - th2)
        kinetic = (6.0 / (m * d**2)) * (2 * p1**2 + 8 * p2**2 - 6 * p1 * p2 * c)
        kinetic /= 2.0 * (16.0 - 9.0 * c**2)
        potential = -0.5 * m * g * d * (3.0 * np.cos(th1) + np.cos(th2))
        return kinetic + potential

    ic = [0.3, 0.2, 0.0, 0.0]
    traj = system.run(final_time=50.0, dt=0.01, ic=ic, rtol=1e-11, atol=1e-12)
    e = energy(traj.y)
    # Machine-precision conservation at rtol 1e-11; 1e-8 is a comfortable ceiling
    # and four orders of magnitude below the drift a wrong torque term produces.
    assert np.max(np.abs(e - e[0])) < 1e-8


def test_double_pendulum_normal_modes_match_the_textbook() -> None:
    """Small-oscillation frequencies equal the equal-rod textbook values.

    Linearising about the hanging equilibrium gives det(K - w^2 A) = 0 with
    A = (m d^2 / 6) [[8, 3], [3, 2]] and K = m g d diag(3/2, 1/2), whose roots
    are 2.6815 and 7.1923 rad/s at g = 9.82, d = m = 1.  These frequencies read
    the potential's *curvature*, so they pin the gravitational torque
    coefficients directly (a spurious factor 3 on the lower arm moves them to
    3.0923 and 10.8025 — 15% and 50% high).
    """
    system = ts.systems.DoublePendulum()
    eigs = np.linalg.eigvals(system.jacobian(np.zeros(4), 0.0))
    # A conservative linearisation: two conjugate pairs on the imaginary axis.
    assert np.max(np.abs(eigs.real)) < 1e-9
    freqs = np.unique(np.round(np.abs(eigs.imag), 6))
    assert freqs.shape == (2,)
    assert freqs[0] == pytest.approx(2.68147968, abs=1e-6)
    assert freqs[1] == pytest.approx(7.19233389, abs=1e-6)


# ---------------------------------------------------------------------------
# Closed-form parametric solution (oscillatory_systems: Lissajous2D)
# ---------------------------------------------------------------------------


def test_lissajous2d_matches_closed_form() -> None:
    """Lissajous2D integrates to its exact parametric curve.

    The RHS is purely time-driven, so the solution is the closed form
    x(t) = A cos(a t), y(t) = B cos(b t + delta) when started from
    (A cos 0, B cos delta).  This pins the *whole trajectory* against an
    analytic function — a stringent end-to-end integrator check.
    """
    s = ts.systems.Lissajous2D()
    a_amp, b_amp = 1.0, 1.0
    fa, fb = 3.0, 2.0
    delta = np.pi / 2
    ic = [a_amp * np.cos(0.0), b_amp * np.cos(delta)]
    traj = s.run(final_time=10.0, dt=0.01, ic=ic, rtol=1e-11, atol=1e-11)
    t = traj.t
    x_exact = a_amp * np.cos(fa * t)
    y_exact = b_amp * np.cos(fb * t + delta)
    err = np.max(np.abs(traj.y[:, 0] - x_exact)) + np.max(np.abs(traj.y[:, 1] - y_exact))
    # A high-order adaptive integrator on a smooth analytic RHS lands near
    # machine precision; 1e-9 is a comfortable, falsifiable ceiling.
    assert err < 1e-9


# ---------------------------------------------------------------------------
# Constant phase-space contraction identity (chaotic_attractors: Halvorsen)
# ---------------------------------------------------------------------------


def test_halvorsen_exponent_sum_equals_trace() -> None:
    """Halvorsen's flow has constant divergence -3a → sum of exponents = -3a.

    The Jacobian diagonal is (-a, -a, -a) independent of state, so the
    phase-space contraction rate is the constant trace -3a; by the standard
    identity sum_i lambda_i = <div f> this equals -3a exactly (Sprott 2010,
    *Elegant Chaos*).  This is a different identity from the Lorenz divergence
    pinned in test_known_values.py.
    """
    spec = ts.analysis.lyapunov_spectrum(
        ts.systems.Halvorsen(ic=[-5.0, 0.0, 0.0]), dt=0.02, transient=50.0, final_time=400.0
    )
    assert spec.shape == (3,)
    assert spec.sum() == pytest.approx(-3.0 * 1.4, abs=2e-3)
    # Chaotic-flow signature: one positive, one near-zero, one negative.
    assert spec[0] > 0.0
    assert spec[2] < 0.0


def test_duffing_is_a_bounded_double_well_with_divergence_minus_delta() -> None:
    """The forced Duffing oscillator is bounded, chaotic, and contracts at -delta.

    In the autonomous 3-D form (x, y, z=drive phase) the Jacobian trace is the
    state-independent -delta, so sum_i lambda_i = -delta exactly, and the drive
    phase contributes an exponent that is identically zero.

    The catalogue's defaults (beta = -1 linear, alpha = +1 cubic) give the
    double-well potential V = beta x^2/2 + alpha x^4/4, whose orbit hops between
    wells at |x| ~ 1.5 and is chaotic. Swapping the two coefficients gives
    V = x^2/2 - x^4/4, unbounded below: the orbit escapes and integrate() raises.
    Both facts are pinned here.
    """
    system = ts.systems.Duffing()
    delta = system.params["delta"]

    traj = system.run(final_time=500.0, dt=0.01)
    assert np.all(np.isfinite(traj.y))
    # The two wells sit at x = +/-1; the attractor spans both and stays O(1).
    assert 1.2 < np.max(np.abs(traj.y[:, 0])) < 3.0
    assert np.min(traj.y[:, 0]) < -0.5 < 0.5 < np.max(traj.y[:, 0])

    spec = ts.analysis.lyapunov_spectrum(system, final_time=4000.0, dt=0.02, transient=400.0)
    assert spec.shape == (3,)
    # Trace(J) = -delta identically, so the sum is exact to estimator roundoff.
    assert spec.sum() == pytest.approx(-delta, abs=1e-6)
    # Genuinely chaotic (an independent scipy variational computation of the
    # same vector field gives lambda_1 = 0.123), and the drive phase is neutral.
    assert spec.max() == pytest.approx(0.125, abs=0.03)
    assert np.min(np.abs(spec)) < 1e-6


def test_pan_xu_zhou_default_parameters_are_above_the_hopf_threshold() -> None:
    """PanXuZhou's default ``k`` is in the chaotic regime, not on a stable focus.

    For x' = a(y-x), y' = kx - xz, z' = -bz + xy the non-trivial equilibria are
    (+/-sqrt(bk), same, k) and the Routh-Hurwitz criterion on
    lam^3 + (a+b)lam^2 + (ab + bk)lam + 2abk makes them **stable** while
    k < a(a+b)/(a-b).  Below that threshold every orbit spirals into a focus
    after a long chaotic transient (the regime the pre-v6 default k=16 shipped).
    Pin both halves: the analytic threshold, and a genuinely positive exponent
    at the default parameters.
    """
    system = ts.systems.PanXuZhou()
    a = system.params["a"]
    b = -system.params["c"]  # z' = c z + x y with c < 0, so b = -c
    k = system.params["k"]
    threshold = a * (a + b) / (a - b)
    assert k > threshold, f"k={k} is below the Hopf threshold {threshold:.3f}"

    spec = ts.analysis.lyapunov_spectrum(system, final_time=3000.0, dt=0.005, transient=1000.0)
    assert spec.shape == (3,)
    assert spec[0] > 0.5, f"default parameters are not chaotic: {spec}"
    # Divergence is the constant -a + f + c = -(a + b), so the sum is exact.
    assert spec.sum() == pytest.approx(-(a + b), abs=5e-2)


# ---------------------------------------------------------------------------
# Analytic equilibrium of a flow (chaotic_attractors: Thomas)
# ---------------------------------------------------------------------------


def test_thomas_origin_is_an_equilibrium() -> None:
    """The origin is an exact equilibrium of Thomas' cyclically symmetric flow.

    With xdot = -a x + b sin y (and cyclic), f(0,0,0) = 0 since sin 0 = 0, so the
    origin is a fixed point of the flow (Thomas 1999, Int. J. Bifurc. Chaos 9,
    1889-1905).  ``fixed_points`` over a box containing the origin must recover
    it.  (Lorenz/Rössler equilibria are covered in test_fixed_points.py; Thomas
    is not.)
    """
    sys = ts.systems.Thomas()
    # Sanity: the RHS really vanishes at the origin (defends the expected value).
    rhs = sys._rhs_numeric()
    assert np.allclose(rhs(np.zeros(3), 0.0), 0.0, atol=1e-12)

    # A tight box around the origin: it is the *only* equilibrium of the lattice
    # inside [-1, 1]^3, so every multi-start lands in its Newton basin.  (A wide
    # box is sensitive to which random starts fall in the origin's small basin,
    # which depends on the process-global RNG state — order-dependent in a full
    # suite run; the tight box makes the recovery deterministic.)
    region = Box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    fps = ts.analysis.fixed_points(sys, region=region, n_seeds=200, seed=0)
    locations = np.array([fp.x for fp in fps])
    nearest = float(np.min(np.linalg.norm(locations, axis=1)))
    # Newton converges to the root to full tolerance; 1e-6 is the dedup scale.
    assert nearest < 1e-6


# ---------------------------------------------------------------------------
# Planar classics (chem_bio_systems / population_dynamics / oscillatory_systems)
#
# The unforced two-dimensional systems are exactly the ones whose qualitative
# claims are analytically checkable: a Hopf threshold in closed form, an exact
# limit cycle, a conserved quantity.  Each check below re-derives its expected
# value from the cited equations here in the test, never from the kernel.
# ---------------------------------------------------------------------------


def test_stuart_landau_cycle_is_the_analytic_circle() -> None:
    """Stuart-Landau relaxes onto r = sqrt(mu) turning at omega - b*mu.

    The normal form A' = (mu + i omega) A - (1 + i b) |A|^2 A decouples exactly
    into r' = mu r - r^3 and theta' = omega - b r^2 (Stuart 1960, J. Fluid Mech.
    9, 353-370), so the limit cycle is the *circle* of radius sqrt(mu) traversed
    at the constant angular velocity omega - b*mu.  Both numbers are closed
    form, which makes this the sharpest available end-to-end check of a
    limit-cycle integration: a mis-expanded real form (the b term attached to
    the wrong component) breaks the constant radius immediately.
    """
    mu, omega, b = 1.4, 0.9, 0.5
    s = ts.systems.StuartLandau(params={"mu": mu, "omega": omega, "b": b})
    traj = s.run(final_time=200.0, dt=0.005, ic=[0.3, 0.1], rtol=1e-11, atol=1e-13)
    settled = traj.y[traj.t > 100.0]
    t_settled = traj.t[traj.t > 100.0]

    r = np.linalg.norm(settled, axis=1)
    assert np.max(np.abs(r - np.sqrt(mu))) < 1e-8

    theta = np.unwrap(np.arctan2(settled[:, 1], settled[:, 0]))
    rate = np.polyfit(t_settled, theta, 1)[0]
    assert rate == pytest.approx(omega - b * mu, abs=1e-8)


def test_stuart_landau_spectrum_is_zero_and_minus_two_mu() -> None:
    """The transverse exponent is exactly -2*mu, for any mu > 0.

    d/dr (mu r - r^3) at r = sqrt(mu) is mu - 3 mu = -2 mu, and the tangential
    exponent of any limit cycle is 0.  Sweeping mu turns the identity into a
    *line* rather than a single number, so a coincidence cannot pass it.
    """
    for mu in (0.6, 1.0, 1.7):
        spec = ts.analysis.lyapunov_spectrum(
            ts.systems.StuartLandau(params={"mu": mu}), final_time=2000.0, dt=0.05
        )
        assert spec.shape == (2,)
        assert spec[0] == pytest.approx(0.0, abs=1e-3)
        assert spec[1] == pytest.approx(-2.0 * mu, abs=1e-3)


def test_stuart_landau_below_the_hopf_bifurcation_decays_to_the_origin() -> None:
    """For mu < 0 the origin is globally attracting (r' = mu r - r^3 < 0)."""
    traj = ts.systems.StuartLandau(params={"mu": -0.3}).run(final_time=200.0, dt=0.5, ic=[0.8, 0.2])
    assert np.linalg.norm(traj.y[-1]) < 1e-9


def test_lotka_volterra_conserves_its_first_integral() -> None:
    """Lotka-Volterra orbits are closed level sets of V, not a limit cycle.

    V = delta x - gamma ln x + beta y - alpha ln y is a constant of motion
    (Volterra 1926, Nature 118, 558-560): dV/dt = (delta - gamma/x) x' +
    (beta - alpha/y) y' = 0 identically.  So the coexistence equilibrium is a
    neutrally stable *center* and each initial condition has its own orbit —
    the two halves pinned here.  A dissipative transcription error (a sign slip,
    or a self-limitation term that does not belong) would make V drift.
    """
    s = ts.systems.LotkaVolterra()
    p = s.params
    traj = s.run(final_time=500.0, dt=0.01, rtol=1e-11, atol=1e-13)
    x, y = traj.y[:, 0], traj.y[:, 1]
    v = p["delta"] * x - p["gamma"] * np.log(x) + p["beta"] * y - p["alpha"] * np.log(y)
    assert np.max(np.abs(v - v[0])) < 1e-7

    # The center is exactly (gamma/delta, alpha/beta) — the RHS vanishes there.
    center = np.array([p["gamma"] / p["delta"], p["alpha"] / p["beta"]])
    assert np.allclose(s._rhs_numeric()(center, 0.0), 0.0, atol=1e-12)

    # A different initial condition traces a *different* closed orbit (a center,
    # not an attractor): the amplitudes must not coincide.
    other = s.run(final_time=500.0, dt=0.01, ic=[4.0, 2.2])
    assert np.ptp(other.y[:, 0]) < 0.5 * np.ptp(x)


def test_brusselator_hopf_threshold_is_one_plus_a_squared() -> None:
    """The Brusselator oscillates exactly for b > 1 + a^2.

    At the equilibrium (a, b/a) the Jacobian is [[b-1, a^2], [-b, -a^2]], with
    determinant a^2 > 0 and trace b - 1 - a^2 (Prigogine & Lefever 1968,
    J. Chem. Phys. 48, 1695), so the steady state loses stability in a Hopf
    bifurcation precisely at b = 1 + a^2.  Straddle it.
    """
    a = 1.0
    threshold = 1.0 + a**2
    s = ts.systems.Brusselator(params={"a": a, "b": threshold})
    # The equilibrium claim itself, checked against the RHS.
    eq = np.array([a, threshold / a])
    assert np.allclose(s._rhs_numeric()(eq, 0.0), 0.0, atol=1e-12)

    below = ts.systems.Brusselator(params={"a": a, "b": threshold - 0.5}).run(
        final_time=400.0, dt=0.05, ic=[1.3, 2.0]
    )
    above = ts.systems.Brusselator(params={"a": a, "b": threshold + 1.0}).run(
        final_time=400.0, dt=0.05, ic=[1.3, 2.0]
    )
    assert np.ptp(below.y[below.t > 350.0][:, 0]) < 1e-4  # decays to the focus
    assert np.ptp(above.y[above.t > 350.0][:, 0]) > 1.0  # sustained limit cycle


def test_selkov_oscillates_only_inside_its_analytic_hopf_window() -> None:
    """Sel'kov's limit cycle exists exactly for b^2 between the two Hopf roots.

    At the equilibrium (b, b/(a+b^2)) the Jacobian trace is
    (b^2 - a - b^2 (a + b^2)) / (a + b^2)... equivalently the steady state is
    unstable for b^2 in ((1 - 2a -+ sqrt(1 - 8a)) / 2) (Sel'kov 1968, Eur. J.
    Biochem. 4, 79-86, in the dimensionless form of Strogatz §7.3).  The roots
    are computed here in closed form and straddled on both sides.
    """
    a = 0.1
    disc = np.sqrt(1.0 - 8.0 * a)
    lo, hi = np.sqrt((1.0 - 2.0 * a - disc) / 2.0), np.sqrt((1.0 - 2.0 * a + disc) / 2.0)
    assert lo < ts.systems.Selkov().params["b"] < hi  # the shipped default oscillates

    for b, expect_cycle in ((lo - 0.12, False), (0.5 * (lo + hi), True), (hi + 0.12, False)):
        s = ts.systems.Selkov(params={"a": a, "b": float(b)})
        # The equilibrium claim, checked against the RHS.
        eq = np.array([b, b / (a + b**2)])
        assert np.allclose(s._rhs_numeric()(eq, 0.0), 0.0, atol=1e-12)
        traj = s.run(final_time=800.0, dt=0.05, ic=[0.6, 0.8])
        amplitude = float(np.ptp(traj.y[traj.t > 700.0][:, 0]))
        assert (amplitude > 0.1) is expect_cycle, f"b={b}: amplitude {amplitude:.2e}"


def test_van_der_pol_has_a_unique_globally_attracting_limit_cycle() -> None:
    """Every non-equilibrium van der Pol orbit lands on the *same* cycle.

    Uniqueness of the limit cycle is the defining property of the van der Pol
    oscillator (van der Pol 1926; Liénard's theorem).  Two initial conditions
    started far apart — one inside the cycle, one well outside — must converge
    to a single closed curve, so the Hausdorff distance between their settled
    orbits is ~0.  A system with a spurious second attractor, or one whose
    "cycle" is really a slow spiral, fails this.

    The two orbits are sampled polylines, so the achievable floor is set by the
    sampling: a point of one orbit can sit up to half a sample-spacing from the
    nearest *sampled* point of the other even when the curves coincide exactly.
    The bound is therefore derived from the measured spacing rather than being a
    magic constant — and is additionally required to be small in absolute terms,
    so it cannot become vacuous if the sampling is coarsened.
    """
    s = ts.systems.VanDerPol()
    tails = []
    for ic in ([0.05, 0.0], [3.0, 3.0]):
        traj = s.run(final_time=300.0, dt=0.0005, ic=ic, rtol=1e-11, atol=1e-13)
        # Two limit-cycle periods (~6.66 each) is a whole closed curve, twice.
        tails.append(traj.y[traj.t > 286.0])
    a_pts, b_pts = tails
    d_ab = float(np.max(cKDTree(b_pts).query(a_pts)[0]))
    d_ba = float(np.max(cKDTree(a_pts).query(b_pts)[0]))
    spacing = max(
        float(np.max(np.linalg.norm(np.diff(a_pts, axis=0), axis=1))),
        float(np.max(np.linalg.norm(np.diff(b_pts, axis=0), axis=1))),
    )
    assert spacing < 5e-3, f"sampling too coarse to be a meaningful test: {spacing}"
    assert max(d_ab, d_ba) < spacing, (
        f"the two orbits are not the same curve: Hausdorff {max(d_ab, d_ba):.2e} "
        f"exceeds the {spacing:.2e} sample spacing"
    )

    # ...and the cycle really is a cycle: the orbit returns to its own start.
    assert np.min(np.linalg.norm(a_pts[200:] - a_pts[0], axis=1)) < spacing


def test_van_der_pol_becomes_relaxational_at_large_mu() -> None:
    """Large mu gives slow-fast structure; mu ~ 1 does not.

    The relaxation regime is the *reason* van der Pol's 1926 paper is titled
    "On relaxation-oscillations": for mu >> 1 the orbit crawls along the outer
    branches of the cubic Lienard nullcline and jumps between them, so the peak
    speed |x'| towers over its median.  At mu = 1 the cycle is nearly harmonic
    and that ratio is small.  Pinned as an order-of-magnitude separation, which
    is what the docstring claims.
    """
    ratios = {}
    for mu in (1.0, 10.0):
        traj = ts.systems.VanDerPol(params={"mu": mu}).run(
            final_time=300.0, dt=0.002, ic=[2.0, 0.0], rtol=1e-10, atol=1e-12
        )
        x = traj.y[traj.t > 260.0, 0]
        speed = np.abs(np.gradient(x, 0.002))
        ratios[mu] = float(np.max(speed) / np.median(speed))
    assert ratios[1.0] < 5.0, ratios
    assert ratios[10.0] > 50.0, ratios


def test_fitzhugh_nagumo_is_excitable_at_zero_current_and_oscillatory_at_half() -> None:
    """The applied current selects excitable rest vs sustained spiking.

    The equilibrium is the intersection of the cubic v-nullcline
    w = v - v^3/3 + curr with the line w = (v + a)/b (FitzHugh 1961, Biophys. J.
    1, 445-466).  At curr = 0 it lies on the stable left branch, so the model
    rests; at curr = 0.5 it has moved onto the unstable middle branch and the
    model fires periodically.  The resting state is located here by solving the
    nullcline intersection independently of the integration.
    """
    p = dict(ts.systems.FitzHughNagumo().params)
    a, b = p["a"], p["b"]

    # Independent root of the two nullclines at curr = 0: v - v^3/3 = (v + a)/b.
    roots = np.roots([-1.0 / 3.0, 0.0, 1.0 - 1.0 / b, -a / b])
    v_rest = float(np.min(roots[np.abs(roots.imag) < 1e-9].real))
    w_rest = (v_rest + a) / b

    rest = ts.systems.FitzHughNagumo(params={**p, "curr": 0.0}).run(
        final_time=600.0, dt=0.05, ic=[0.0, 0.0]
    )
    assert np.ptp(rest.y[rest.t > 500.0][:, 0]) < 1e-4
    assert rest.y[-1] == pytest.approx([v_rest, w_rest], abs=1e-4)

    firing = ts.systems.FitzHughNagumo(params={**p, "curr": 0.5}).run(
        final_time=600.0, dt=0.05, ic=[0.0, 0.0]
    )
    assert np.ptp(firing.y[firing.t > 500.0][:, 0]) > 3.0
