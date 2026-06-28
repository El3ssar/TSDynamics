"""Regression: DDE ``lyapunov_spectrum`` averaging-window semantics.

Ticket **FIX-DDELYAP-WINDOW** (part 1).  The documented contract — in both
``DelaySystem.lyapunov_spectrum`` and
:func:`tsdynamics.families._dde_lyapunov.dde_lyapunov_spectrum` — is that
``final_time`` is the **post-burn-in averaging window**, so the total integration
is ``burn_in + final_time``.

The pre-fix implementation instead treated ``final_time`` as the *total* length
and carved ``burn_in`` out of it (``n_chunks = round(final_time / chunk)``;
``n_burn = min(n_chunks - 1, round(burn_in / chunk))``).  When ``burn_in`` was
comparable to ``final_time`` the averaging window **collapsed to a single delay
window** — a single finite-time Lyapunov exponent (FTLE) of one ``τ``-chunk.  That
FTLE fluctuates wildly (and in sign), so the estimator reported values an order of
magnitude off the converged exponent — and, for some on-attractor states, a
**negative** leading exponent for the demonstrably chaotic Mackey–Glass attractor
(the audit's ``-0.0125`` artifact).

These tests assert the *semantic invariant* rather than a single fragile FTLE
sign (which is genuinely initial-state dependent): with ``final_time`` held fixed
the estimate must be stable as ``burn_in`` grows past the window, and a
large-``burn_in`` run must converge to the well-averaged reference.  Both are
**failing-first** against the pre-fix code (see the inline OLD/NEW numbers).

All checks are reference-free and deterministic (a seeded on-attractor initial
state); ``interp == jit`` bit-for-bit is re-asserted under a window config.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from _sampling import DDE_HISTORIES

import tsdynamics as ts

pytest.importorskip("tsdynamics._rust")


@functools.lru_cache(maxsize=1)
def _mackeyglass_on_attractor_ic() -> tuple[float, ...]:
    """A deterministic on-attractor Mackey–Glass state (end of a seeded run)."""
    traj = ts.MackeyGlass().integrate(
        final_time=500.0, dt=0.2, history=DDE_HISTORIES["MackeyGlass"]
    )
    return tuple(np.asarray(traj.y[-1], dtype=np.float64).ravel().tolist())


def _lyap(burn_in: float, final_time: float) -> float:
    """Leading Mackey–Glass DDE exponent from the seeded on-attractor state."""
    ic = np.asarray(_mackeyglass_on_attractor_ic(), dtype=np.float64)
    spec = ts.MackeyGlass().lyapunov_spectrum(
        backend="interp",
        n_exp=1,
        dt=0.5,
        burn_in=burn_in,
        final_time=final_time,
        ic=ic,
        rtol=1e-4,
        atol=1e-4,
    )
    return float(spec[0])


def test_estimate_is_invariant_to_burn_in_for_a_fixed_window() -> None:
    """``final_time`` is the averaging window: growing ``burn_in`` past it is stable.

    With ``final_time = 400`` the leading exponent must barely move when
    ``burn_in`` grows from 50 (window not yet reached) to 400 (>= the window) —
    the extra ``burn_in`` only discards more transient.

    Failing-first (pre-fix, ``final_time`` = total length):
        burn_in=50  -> +0.001889   (21 chunks averaged)
        burn_in=400 -> +0.037988   (collapsed to a single FTLE)
        |Δ| = 0.036  >> 3e-3   -> FAIL
    After the fix:
        burn_in=50  -> +0.003715
        burn_in=400 -> +0.004710
        |Δ| = 0.001  <  3e-3   -> PASS
    """
    lam_small_burn = _lyap(burn_in=50.0, final_time=400.0)
    lam_large_burn = _lyap(burn_in=400.0, final_time=400.0)
    assert lam_small_burn > 0.0, lam_small_burn
    assert lam_large_burn > 0.0, lam_large_burn
    assert abs(lam_large_burn - lam_small_burn) < 3e-3, (lam_small_burn, lam_large_burn)


def test_large_burn_in_converges_to_the_well_averaged_reference() -> None:
    """A large-``burn_in`` estimate matches the long, well-averaged reference.

    The pre-fix collapse made a large ``burn_in`` report a single-window FTLE an
    order of magnitude off the converged value.

    Failing-first (pre-fix):
        large burn  (burn_in=400, final_time=400) -> +0.037988
        reference   (burn_in=100, final_time=1000) -> +0.003460
        |Δ| = 0.0345  >> 2.5e-3   -> FAIL
    After the fix:
        large burn  -> +0.004710 ; reference -> +0.003772 ; |Δ| = 0.0009 -> PASS
    """
    lam_large_burn = _lyap(burn_in=400.0, final_time=400.0)
    lam_reference = _lyap(burn_in=100.0, final_time=1000.0)
    assert lam_large_burn > 0.0, lam_large_burn
    assert abs(lam_large_burn - lam_reference) < 2.5e-3, (lam_large_burn, lam_reference)


def test_mackeyglass_positive_leading_exponent_with_large_burn_in() -> None:
    """Acceptance guard: Mackey–Glass(burn_in=180, final_time=200) has λ₁ > 0.

    This is the literal acceptance criterion ("MackeyGlass(180,200) positive
    leading exponent, n_positive=1") — Mackey–Glass at ``tau = 17`` is chaotic, so
    its leading exponent must be positive (matching its ``known_lyapunov``
    ``n_positive = 1``).  Post-fix this is +0.00124.

    It is a *contract* guard, **not** the failing-first discriminator: the audit's
    negative ``-0.0125`` was a single-chunk FTLE for one particular on-attractor
    state, and the pre-fix single-chunk sign is genuinely initial-state dependent
    (for this test's deterministic IC the pre-fix value happened to be positive
    too, ``+0.0246``).  The deterministic regressions that fail on the pre-fix
    code are the two window tests above (``invariant_to_burn_in`` and
    ``converges_to_the_well_averaged_reference``); this one pins the documented
    chaotic-sign acceptance under the fixed window semantics.
    """
    lam = _lyap(burn_in=180.0, final_time=200.0)
    assert lam > 0.0, f"chaotic Mackey-Glass leading exponent must be positive, got {lam}"


def test_interp_equals_jit_bit_for_bit_under_window_semantics() -> None:
    """The D2 contract holds under the fixed window arithmetic: interp == jit.

    The chunk-count change is deterministic and backend-independent, so both
    engines must still return an identical spectrum.
    """
    ic = np.asarray(_mackeyglass_on_attractor_ic(), dtype=np.float64)
    kw = dict(n_exp=2, dt=0.5, burn_in=180.0, final_time=200.0, ic=ic, rtol=1e-4, atol=1e-4)
    interp = ts.MackeyGlass().lyapunov_spectrum(backend="interp", **kw)
    jit = ts.MackeyGlass().lyapunov_spectrum(backend="jit", **kw)
    np.testing.assert_array_equal(interp, jit)


def _mackeyglass_chunk() -> float:
    """The renormalisation chunk (= one delay window) the estimator uses."""
    from tsdynamics.families._dde_lyapunov import _build_extended_tape

    _, slots, _ = _build_extended_tape(ts.MackeyGlass(), 1)
    return max(s.delay for s in slots)


def test_small_positive_burn_in_still_discards_one_window() -> None:
    """A ``burn_in`` that rounds to zero windows must NOT skip the transient.

    For Mackey–Glass the renormalisation chunk is one delay window
    (``max_delay = tau * 1.01 ≈ 17.17``).  A small but positive ``burn_in`` such
    as 5.0 has ``round(5.0 / 17.17) = round(0.29) = 0`` — and the pre-fix code
    then discarded **zero** windows, silently averaging the transient back in
    (the exact failure the module docstring warns against).

    Failing-first contract: a small positive ``burn_in`` (rounds to 0) must
    behave like discarding exactly **one** window — so its spectrum equals the
    ``burn_in == chunk`` run and **differs** from the ``burn_in = 0`` run (which
    legitimately discards nothing).  Pre-fix, the small-``burn_in`` run was
    identical to ``burn_in = 0`` (both discarded zero windows) → FAIL.
    """
    ic = np.asarray(_mackeyglass_on_attractor_ic(), dtype=np.float64)
    chunk = _mackeyglass_chunk()
    assert round(5.0 / chunk) == 0, "test premise: 5.0 must round to zero windows"

    def run(burn_in: float) -> np.ndarray:
        return ts.MackeyGlass().lyapunov_spectrum(
            backend="interp",
            n_exp=1,
            dt=0.5,
            burn_in=burn_in,
            final_time=200.0,
            ic=ic,
            rtol=1e-4,
            atol=1e-4,
        )

    no_burn = run(0.0)  # discards nothing
    small_burn = run(5.0)  # rounds to 0 → must round UP to 1 window post-fix
    one_window = run(chunk)  # discards exactly one window

    # Post-fix: a positive burn_in rounding to zero discards one window, matching
    # the explicit one-window run and differing from the zero-burn-in run.
    np.testing.assert_array_equal(small_burn, one_window)
    assert not np.array_equal(small_burn, no_burn), (small_burn, no_burn)
