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
