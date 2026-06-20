"""Tests for the batch-ahead step buffer in ``ContinuousSystem.step`` (WS-STEPBUF).

The buffer integrates a chunk of ``dt``-steps in one engine call and hands them
out one at a time, so it must be *transparent*: the sequence of states a consumer
sees has to match the un-buffered per-``dt`` path, and the buffer must be
invalidated on ``reinit`` / ``set_state`` / a change of ``dt``.

These exercise the compiled engine (``tsdynamics._rust``), so they skip cleanly
where the extension is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts

pytest.importorskip("tsdynamics._rust")


# Fixed-step methods take byte-identical steps through the chunked and per-step
# paths (the only difference is float-accumulation order, well below 1e-11);
# adaptive methods agree to within the solver tolerance because the engine
# re-seeds its controller at each output node.
FIXED_STEP_METHODS = ["rk4"]
ADAPTIVE_METHODS = ["rk45", "dop853", "tsit5"]


def _per_step_unbuffered(system, ic, dt, n, method):
    """Reference sequence: reinit then ``step`` with the buffer disabled.

    Setting ``_step_chunk = 1`` collapses the buffer to a single-step fill, i.e.
    the original un-buffered round-trip-per-``dt`` path — the ground truth the
    buffered path must reproduce.
    """
    system._step_chunk = 1
    system.reinit(list(ic), method=method)
    return np.array([system.step(dt).copy() for _ in range(n)])


def _per_step_buffered(system, ic, dt, n, method, chunk=256):
    """The buffered sequence: a large chunk amortises the per-``dt`` overhead."""
    system._step_chunk = chunk
    system.reinit(list(ic), method=method)
    return np.array([system.step(dt).copy() for _ in range(n)])


# ---------------------------------------------------------------------------
# Answer-preservation: buffered == un-buffered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", FIXED_STEP_METHODS)
def test_fixed_step_buffered_matches_unbuffered(method):
    """For a fixed-step method the buffered states match the per-step ones."""
    ic = [1.0, 1.0, 1.0]
    ref = _per_step_unbuffered(ts.Lorenz(), ic, 0.01, 300, method)
    buf = _per_step_buffered(ts.Lorenz(), ic, 0.01, 300, method)
    assert buf.shape == ref.shape
    # Bit-identical up to float-accumulation order across the chunk boundaries.
    assert np.max(np.abs(buf - ref)) < 1e-11


@pytest.mark.parametrize("method", ADAPTIVE_METHODS)
def test_adaptive_buffered_matches_unbuffered_within_tol(method):
    """For an adaptive method the buffered states agree to the solver tolerance."""
    ic = [1.0, 1.0, 1.0]
    rtol = 1e-6
    ref_sys = ts.Lorenz()
    buf_sys = ts.Lorenz()
    ref_sys._step_chunk = 1
    buf_sys._step_chunk = 256
    ref_sys.reinit(ic, method=method, rtol=rtol)
    buf_sys.reinit(ic, method=method, rtol=rtol)
    ref = np.array([ref_sys.step(0.01).copy() for _ in range(300)])
    buf = np.array([buf_sys.step(0.01).copy() for _ in range(300)])
    # Both are valid solutions of the same ODE to the requested tolerance; the
    # only difference is the adaptive controller's carried step-size hint, which
    # cannot exceed the tolerance budget (allow chaotic amplification headroom).
    assert np.max(np.abs(buf - ref)) < 1e-4


def test_chunk_size_independence():
    """The handed-out states do not depend on the chunk size (fixed-step)."""
    ic = [0.1, 0.0, 0.0]
    seqs = []
    for chunk in (1, 7, 64, 256, 1000):
        s = ts.Rossler()
        s._step_chunk = chunk
        s.reinit(ic, method="rk4")
        seqs.append(np.array([s.step(0.02).copy() for _ in range(200)]))
    base = seqs[0]
    for seq in seqs[1:]:
        assert np.max(np.abs(seq - base)) < 1e-11


def test_partial_last_chunk():
    """A run length that is not a multiple of the chunk size is handled exactly.

    With ``n`` not divisible by ``chunk`` the final chunk is only partially
    consumed; the handed-out states must still match the per-step path.
    """
    ic = [1.0, 1.0, 1.0]
    ref = _per_step_unbuffered(ts.Lorenz(), ic, 0.01, 130, "rk4")
    buf = _per_step_buffered(ts.Lorenz(), ic, 0.01, 130, "rk4", chunk=50)  # 50,50,30
    assert buf.shape == (130, 3)
    assert np.max(np.abs(buf - ref)) < 1e-11


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_buffer_invalidated_on_dt_change():
    """Switching ``dt`` mid-stream refills the buffer; the result is exact.

    Interleave ``step(dt1)`` then ``step(dt2)`` and compare each leg to a fresh
    system stepped only at that ``dt`` from the matching state.
    """
    ic = [1.0, 1.0, 1.0]
    dt1, dt2 = 0.01, 0.025

    sys = ts.Lorenz()
    sys._step_chunk = 256
    sys.reinit(ic, method="rk4")
    leg1 = np.array([sys.step(dt1).copy() for _ in range(40)])
    # dt changes here — the buffer (filled at dt1) must be dropped.
    leg2 = np.array([sys.step(dt2).copy() for _ in range(40)])

    # Reference leg1: a fresh system stepped only at dt1.
    ref1_sys = ts.Lorenz()
    ref1_sys._step_chunk = 1
    ref1_sys.reinit(ic, method="rk4")
    ref1 = np.array([ref1_sys.step(dt1).copy() for _ in range(40)])

    # Reference leg2: a fresh system started from leg1's final state, stepped at dt2.
    ref2_sys = ts.Lorenz()
    ref2_sys._step_chunk = 1
    ref2_sys.reinit(list(ref1[-1]), method="rk4")
    ref2 = np.array([ref2_sys.step(dt2).copy() for _ in range(40)])

    assert np.max(np.abs(leg1 - ref1)) < 1e-11
    assert np.max(np.abs(leg2 - ref2)) < 1e-11


def test_buffer_invalidated_on_set_state():
    """``set_state`` mid-stream drops the stale buffer and re-derives from u."""
    ic = [1.0, 1.0, 1.0]
    new_state = [5.0, -3.0, 12.0]

    sys = ts.Lorenz()
    sys._step_chunk = 256
    sys.reinit(ic, method="rk4")
    [sys.step(0.01) for _ in range(30)]  # part-way into a buffered chunk
    sys.set_state(new_state)
    after = np.array([sys.step(0.01).copy() for _ in range(30)])

    # Reference: a fresh system reinitialised at the same time/state.
    t_at_switch = 30 * 0.01
    ref_sys = ts.Lorenz()
    ref_sys._step_chunk = 1
    ref_sys.reinit(new_state, t=t_at_switch, method="rk4")
    ref = np.array([ref_sys.step(0.01).copy() for _ in range(30)])

    assert np.max(np.abs(after - ref)) < 1e-11


def test_buffer_invalidated_on_reinit():
    """An explicit ``reinit`` drops the buffer (next ``step`` starts clean)."""
    sys = ts.Lorenz()
    sys._step_chunk = 256
    sys.reinit([1.0, 1.0, 1.0], method="rk4")
    [sys.step(0.01) for _ in range(10)]
    assert sys._buf is not None  # a chunk is buffered

    sys.reinit([2.0, 2.0, 2.0], method="rk4")
    assert sys._buf is None  # buffer dropped by reinit

    got = np.array([sys.step(0.01).copy() for _ in range(50)])
    ref_sys = ts.Lorenz()
    ref_sys._step_chunk = 1
    ref_sys.reinit([2.0, 2.0, 2.0], method="rk4")
    ref = np.array([ref_sys.step(0.01).copy() for _ in range(50)])
    assert np.max(np.abs(got - ref)) < 1e-11


def test_interleaved_state_time_match_fresh_system():
    """state()/time() track the buffered stepping exactly (vs a fresh system)."""
    sys = ts.Lorenz()
    sys._step_chunk = 64
    sys.reinit([1.0, 1.0, 1.0], method="rk4")
    for _ in range(100):
        sys.step(0.01)

    ref = ts.Lorenz()
    ref._step_chunk = 1
    ref.reinit([1.0, 1.0, 1.0], method="rk4")
    for _ in range(100):
        ref.step(0.01)

    assert sys.time() == pytest.approx(ref.time())
    assert np.max(np.abs(sys.state() - ref.state())) < 1e-11


# ---------------------------------------------------------------------------
# Divergence still raises at the right step
# ---------------------------------------------------------------------------


def test_divergence_raises_through_buffer():
    """A trajectory that blows up mid-chunk still raises a RuntimeError.

    The chunked fill raises for the whole span; the buffer falls back to
    single-step integration so the error surfaces at the diverging step rather
    than being swallowed.  Either way the consumer sees a RuntimeError, which the
    basins FSM (and other step() consumers) treat as "diverged".
    """
    sys = ts.Lorenz()
    sys._step_chunk = 256
    # A wildly off-attractor start with a large dt drives the explicit kernel to
    # a non-finite state within the first chunk.
    sys.reinit([1e6, 1e6, 1e6], method="rk4")
    with pytest.raises(RuntimeError):
        for _ in range(512):
            sys.step(1.0)


def test_finite_steps_before_divergence_are_handed_out():
    """Valid states before a mid-chunk divergence are still returned.

    Build a system that stays finite for a few steps then blows up; assert the
    finite prefix is handed out and the RuntimeError lands afterwards — the exact
    per-step divergence contract, preserved by the single-step fallback.
    """
    # Per-step ground truth: how many finite steps before the raise?
    ref = ts.Lorenz()
    ref._step_chunk = 1
    ref.reinit([100.0, 100.0, 100.0], method="rk4")
    finite_ref = []
    with pytest.raises(RuntimeError):
        for _ in range(512):
            finite_ref.append(ref.step(0.5).copy())
    n_finite = len(finite_ref)

    # Buffered path must hand out the same finite prefix and then raise.
    buf = ts.Lorenz()
    buf._step_chunk = 256
    buf.reinit([100.0, 100.0, 100.0], method="rk4")
    finite_buf = []
    with pytest.raises(RuntimeError):
        for _ in range(512):
            finite_buf.append(buf.step(0.5).copy())

    assert len(finite_buf) == n_finite
    if n_finite:
        assert np.max(np.abs(np.array(finite_buf) - np.array(finite_ref))) < 1e-11


# ---------------------------------------------------------------------------
# Integration-level: basins over a flow is answer-identical and faster
# ---------------------------------------------------------------------------


def test_basins_over_flow_unchanged_by_buffer():
    """A 2-D-flow basin image is identical with the buffer on (default) vs off.

    The basins FSM drives the flow purely through ``step(dt)``; the buffer must
    not change a single label.
    """
    grid = ts.Grid(np.array([-2.0, -2.0]), np.array([2.0, 2.0]), (12, 12))

    class _DampedDuffing(ts.ContinuousSystem):
        params = {"delta": 0.3, "alpha": -1.0, "beta": 1.0}
        dim = 2
        default_ic = (0.5, 0.0)

        @staticmethod
        def _equations(y, t, delta, alpha, beta):
            return [y(1), -delta * y(1) - alpha * y(0) - beta * y(0) ** 3]

    # Buffer ON (default chunk).
    res_on = ts.basins_of_attraction(_DampedDuffing(), grid, dt=0.05, max_steps=2000)

    # Buffer effectively OFF (chunk = 1 reproduces the original per-dt path).
    sys_off = _DampedDuffing()
    sys_off._step_chunk = 1
    res_off = ts.basins_of_attraction(sys_off, grid, dt=0.05, max_steps=2000)

    assert np.array_equal(res_on.labels, res_off.labels)
