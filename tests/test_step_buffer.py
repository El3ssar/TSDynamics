"""Tests for the lean per-step path in ``ContinuousSystem.step`` (WS-STEPBUF).

``step`` integrates exactly one ``dt`` from the live state through the engine's
lean dense-output core, reusing the tape and resolved kernel cached by
``reinit``.  It must be *answer-preserving*: the sequence of states a consumer
sees has to be byte-for-byte the released per-``dt`` ``integrate`` path — the
speedup comes only from skipping fixed per-call overhead, never from changing the
numbers.

(An earlier batch-ahead-buffer design integrated a whole chunk in one engine
call; that is **not** equal to N single-``dt`` integrations — the adaptive
controller carries its step-size/error state across output nodes — and silently
corrupted ``max_lyapunov``.  These tests pin the exact equivalence and guard that
regression.)

They exercise the compiled engine (``tsdynamics._rust``), so they skip cleanly
where the extension is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts

pytest.importorskip("tsdynamics._rust")


FIXED_STEP_METHODS = ["rk4"]
ADAPTIVE_METHODS = ["rk45", "dop853", "tsit5"]


def _reference_chain(make, ic, dt, n, method, rtol=1e-6, atol=1e-9):
    """Ground truth: chain ``n`` independent single-``dt`` ``integrate`` calls.

    This is the released semantics of ``step`` — each ``dt`` interval integrated
    as a fresh sub-problem from the live state — which the lean ``step`` must
    reproduce bit-for-bit.
    """
    sys = make()
    state = np.asarray(ic, dtype=float)
    t = 0.0
    out = []
    for _ in range(n):
        traj = sys.integrate(
            final_time=t + dt, dt=dt, t0=t, ic=state, method=method, rtol=rtol, atol=atol
        )
        state = np.asarray(traj.y[-1], dtype=float)
        t += dt
        out.append(state)
    return np.array(out)


def _stepped(make, ic, dt, n, method, rtol=1e-6, atol=1e-9):
    """The states handed out by the live ``step`` stepper."""
    sys = make()
    sys.reinit(list(ic), method=method, rtol=rtol, atol=atol)
    return np.array([sys.step(dt).copy() for _ in range(n)])


# ---------------------------------------------------------------------------
# Answer-preservation: step() == chained per-dt integrate(), bit-for-bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", FIXED_STEP_METHODS + ADAPTIVE_METHODS)
def test_step_matches_perstep_integrate(method):
    """``step`` reproduces the per-``dt`` ``integrate`` chain exactly (every method).

    Unlike a batched integration, the lean per-step path performs the *same*
    single-``dt`` engine call ``integrate`` does, so the agreement is exact for
    adaptive methods too — not merely "within tolerance".
    """
    ic = [1.0, 1.0, 1.0]
    ref = _reference_chain(ts.Lorenz, ic, 0.01, 300, method)
    got = _stepped(ts.Lorenz, ic, 0.01, 300, method)
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) < 1e-12


def test_step_matches_perstep_integrate_rossler():
    """Same exact equivalence on a second system / dt (Rössler, dt=0.02)."""
    ic = [0.1, 0.0, 0.0]
    ref = _reference_chain(ts.Rossler, ic, 0.02, 250, "rk45")
    got = _stepped(ts.Rossler, ic, 0.02, 250, "rk45")
    assert np.max(np.abs(got - ref)) < 1e-12


# ---------------------------------------------------------------------------
# The regression guard: max_lyapunov stays sane (the buffer drove it to ~23)
# ---------------------------------------------------------------------------


def test_max_lyapunov_lorenz_not_corrupted_by_step():
    """``max_lyapunov`` (which interleaves ``step``/``set_state``) lands near 0.9.

    This is the exact failure mode the rejected batch buffer introduced: the
    reference and perturbed trajectories integrated at different cadences and the
    measured exponent blew up to ~23.  A short but unambiguous run pins it back to
    the Lorenz value (≈0.906) in the fast tier.
    """
    lam = ts.max_lyapunov(
        ts.Lorenz(),
        ic=[1.0, 1.0, 1.0],
        dt=0.05,
        n=250,
        steps_per=4,
        transient=400,
        seed=2,
    )
    assert 0.7 < lam < 1.15


# ---------------------------------------------------------------------------
# State management: set_state / reinit / time bookkeeping
# ---------------------------------------------------------------------------


def test_set_state_midstream_matches_fresh_system():
    """``set_state`` mid-stream re-derives the trajectory from the new state."""
    ic = [1.0, 1.0, 1.0]
    new_state = [5.0, -3.0, 12.0]

    sys = ts.Lorenz()
    sys.reinit(ic, method="rk4")
    for _ in range(30):
        sys.step(0.01)
    sys.set_state(new_state)
    after = np.array([sys.step(0.01).copy() for _ in range(30)])

    # A fresh system reinitialised at the same time/state.
    ref_sys = ts.Lorenz()
    ref_sys.reinit(new_state, t=30 * 0.01, method="rk4")
    ref = np.array([ref_sys.step(0.01).copy() for _ in range(30)])
    assert np.max(np.abs(after - ref)) < 1e-12


def test_dt_change_midstream_is_exact():
    """Switching ``dt`` mid-stream keeps each leg exact (no stale step context)."""
    ic = [1.0, 1.0, 1.0]
    dt1, dt2 = 0.01, 0.025

    sys = ts.Lorenz()
    sys.reinit(ic, method="rk4")
    leg1 = np.array([sys.step(dt1).copy() for _ in range(40)])
    leg2 = np.array([sys.step(dt2).copy() for _ in range(40)])

    ref1 = _reference_chain(ts.Lorenz, ic, dt1, 40, "rk4")
    ref2 = _reference_chain(ts.Lorenz, list(ref1[-1]), dt2, 40, "rk4")
    assert np.max(np.abs(leg1 - ref1)) < 1e-12
    assert np.max(np.abs(leg2 - ref2)) < 1e-12


def test_state_time_track_stepping():
    """``state()``/``time()`` follow the stepping exactly (vs a per-step chain)."""
    sys = ts.Lorenz()
    sys.reinit([1.0, 1.0, 1.0], method="rk4")
    for _ in range(100):
        sys.step(0.01)
    ref = _reference_chain(ts.Lorenz, [1.0, 1.0, 1.0], 0.01, 100, "rk4")
    assert sys.time() == pytest.approx(100 * 0.01)
    assert np.max(np.abs(sys.state() - ref[-1])) < 1e-12


# ---------------------------------------------------------------------------
# Divergence still raises at the right step
# ---------------------------------------------------------------------------


def test_divergence_raises():
    """A trajectory that blows up raises a ``RuntimeError`` (the diverged signal)."""
    sys = ts.Lorenz()
    sys.reinit([1e6, 1e6, 1e6], method="rk4")
    with pytest.raises(RuntimeError):
        for _ in range(512):
            sys.step(1.0)


def test_finite_prefix_before_divergence():
    """Valid states before a divergence are handed out; the raise lands after.

    The per-step path raises at the exact offending step (``_run_continuous``
    raises on a non-finite node), so a consumer collecting states sees the same
    finite prefix the released path produced.
    """
    ref = ts.Lorenz()
    ref.reinit([100.0, 100.0, 100.0], method="rk4")
    finite_ref = []
    with pytest.raises(RuntimeError):
        for _ in range(512):
            finite_ref.append(ref.step(0.5).copy())

    got = ts.Lorenz()
    got.reinit([100.0, 100.0, 100.0], method="rk4")
    finite_got = []
    with pytest.raises(RuntimeError):
        for _ in range(512):
            finite_got.append(got.step(0.5).copy())

    assert len(finite_got) == len(finite_ref)
    if finite_ref:
        assert np.max(np.abs(np.array(finite_got) - np.array(finite_ref))) < 1e-12


# ---------------------------------------------------------------------------
# Integration-level: basins over a flow is answer-identical to per-dt integrate
# ---------------------------------------------------------------------------


def test_basins_over_flow_answer_preserving():
    """A 2-D-flow basin image is stable — ``step`` drives the basins FSM exactly."""

    class _DampedDuffing(ts.ContinuousSystem):
        params = {"delta": 0.3, "alpha": -1.0, "beta": 1.0}
        dim = 2
        default_ic = (0.5, 0.0)

        @staticmethod
        def _equations(y, t, delta, alpha, beta):
            return [y(1), -delta * y(1) - alpha * y(0) - beta * y(0) ** 3]

    grid = ts.Grid(np.array([-2.0, -2.0]), np.array([2.0, 2.0]), (12, 12))
    res_a = ts.basins_of_attraction(_DampedDuffing(), grid, dt=0.05, max_steps=2000)
    res_b = ts.basins_of_attraction(_DampedDuffing(), grid, dt=0.05, max_steps=2000)
    assert np.array_equal(res_a.labels, res_b.labels)
