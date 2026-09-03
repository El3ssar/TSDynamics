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
    le = ts.systems.Tent(params={"mu": 1.0}).lyapunov_spectrum(steps=10_000, ic=_IC1D)[0]
    # Slope is constant so the estimate is exact up to float roundoff.
    assert le == pytest.approx(np.log(2.0), abs=1e-4)


def test_tent_general_slope_lyapunov_is_ln_2mu() -> None:
    """Tent map at mu=0.7 has slope magnitude 2*mu → lambda = ln(2*mu).

    The branch slope of x' = mu*(1-2|x-1/2|) is +/-2*mu, giving the analytic
    exponent ln(2*mu) wherever the orbit stays on the attractor (the docstring's
    own statement, and the standard piecewise-linear result).
    """
    le = ts.systems.Tent(params={"mu": 0.7}).lyapunov_spectrum(steps=10_000, ic=_IC1D)[0]
    assert le == pytest.approx(np.log(2.0 * 0.7), abs=1e-4)


def test_chebyshev_degree_two_lyapunov_is_ln2() -> None:
    """Chebyshev map T_2 has constant Lyapunov exponent ln 2.

    For integer degree a >= 2 the Chebyshev map x' = cos(a*arccos x) is exact
    with Lyapunov exponent ln a (Adler & Rivlin 1964, Proc. AMS 15, 794-796).
    The a=2 case is conjugate to the logistic map at r=4 → ln 2.
    """
    le = ts.systems.Chebyshev(params={"a": 2.0}).lyapunov_spectrum(steps=10_000, ic=[0.3])[0]
    assert le == pytest.approx(np.log(2.0), abs=1e-3)


def test_chebyshev_degree_six_lyapunov_is_ln6() -> None:
    """Chebyshev map at a=6 has constant Lyapunov exponent ln 6 (Adler-Rivlin 1964)."""
    le = ts.systems.Chebyshev(params={"a": 6.0}).lyapunov_spectrum(steps=10_000, ic=[0.3])[0]
    assert le == pytest.approx(np.log(6.0), abs=1e-3)


def test_ulam_map_lyapunov_is_ln2() -> None:
    """Ulam-von Neumann map x' = 1 - 2x^2 is conjugate to logistic r=4 → ln 2.

    Ergodic with a smooth invariant density and Lyapunov exponent ln 2
    (Ulam & von Neumann 1947, Bull. AMS 53, 1120).  A finite-step average over a
    smooth measure converges more slowly than the piecewise-linear maps, so the
    tolerance is looser.
    """
    le = ts.systems.Ulam().lyapunov_spectrum(steps=20_000, ic=[0.1])[0]
    assert le == pytest.approx(np.log(2.0), abs=2e-2)


def test_gingerbreadman_is_area_preserving() -> None:
    """The Gingerbreadman map is area-preserving → the two exponents sum to 0.

    x' = 1 - y + |x|, y' = x has Jacobian determinant
    sign(x)*0 - (-1)*1 = 1 everywhere, so it is conservative (Devaney 1984,
    Physica D 10, 387-393): lambda_1 + lambda_2 = ln|det J| = 0 exactly.
    """
    spec = ts.systems.Gingerbreadman().lyapunov_spectrum(steps=10_000, ic=[0.5, 3.7])
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
    spec = ts.systems.Gingerbreadman().lyapunov_spectrum(steps=10_000)
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
    spec = ts.systems.Baker(params={"alpha": alpha}).lyapunov_spectrum(
        steps=20_000, ic=[0.31415926535, 0.2718281828]
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
        traj = baker.iterate(steps=5_000, ic=rng.random(2))
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
    spec = ts.systems.Zaslavskii().lyapunov_spectrum(steps=20_000)
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
    traj = ts.systems.HenonHeiles().integrate(
        final_time=100.0, dt=0.05, ic=ic, rtol=1e-10, atol=1e-10
    )
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
    traj = system.integrate(final_time=50.0, dt=0.01, ic=ic, rtol=1e-11, atol=1e-12)
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
    traj = s.integrate(final_time=10.0, dt=0.01, ic=ic, rtol=1e-11, atol=1e-11)
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
    spec = ts.systems.Halvorsen(ic=[-5.0, 0.0, 0.0]).lyapunov_spectrum(
        dt=0.02, burn_in=50.0, final_time=400.0
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

    traj = system.integrate(final_time=500.0, dt=0.01)
    assert np.all(np.isfinite(traj.y))
    # The two wells sit at x = +/-1; the attractor spans both and stays O(1).
    assert 1.2 < np.max(np.abs(traj.y[:, 0])) < 3.0
    assert np.min(traj.y[:, 0]) < -0.5 < 0.5 < np.max(traj.y[:, 0])

    spec = system.lyapunov_spectrum(final_time=4000.0, dt=0.02, burn_in=400.0)
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

    spec = system.lyapunov_spectrum(final_time=3000.0, dt=0.005, burn_in=1000.0)
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
    fps = ts.fixed_points(sys, region=region, n_seeds=200, seed=0)
    locations = np.array([fp.x for fp in fps])
    nearest = float(np.min(np.linalg.norm(locations, axis=1)))
    # Newton converges to the root to full tolerance; 1e-6 is the dedup scale.
    assert nearest < 1e-6
