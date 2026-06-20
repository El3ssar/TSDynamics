"""Loop-invariant hoisting in the per-step path (WS-INVHOIST).

``ContinuousSystem.step`` advances exactly one ``dt`` through the engine's lean
dense-output core.  WS-STEPBUF already moved the solver resolve, the Jacobian
decision, provenance assembly and the ``Trajectory`` wrap out of that hot loop;
WS-INVHOIST hoists the rest of the per-call orchestration that is invariant
across a constant-``dt`` stepping loop:

* the two-node output grid is assembled directly (``[t0, t0 + dt]``) instead of
  through ``make_output_grid`` (which pays an ``arange``/append and an
  error-module import every call);
* the tape's engine wire arrays are marshalled **once** at ``reinit`` and reused,
  rather than rebuilt per step;
* the state/time advance writes bypass the param-typo-guarding ``__setattr__``.

Every hoist is a **pure speedup**: the numbers ``step`` hands out must stay
byte-for-byte what the released per-``dt`` ``integrate`` path produces.  These
tests pin (a) the exact grid-construction equivalence, (b) that the hot loop no
longer builds the grid through the helper, (c) that the marshalled arrays are
cached and reused, and (d) bit-for-bit agreement of ``step`` with a per-``dt``
``integrate`` chain on the paths the marshalling cache could plausibly disturb —
the Jacobian-carrying stiff tape and the structural-parameter tape.

They exercise the compiled engine (``tsdynamics._rust``), so they skip cleanly
where the extension is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts

pytest.importorskip("tsdynamics._rust")


# ---------------------------------------------------------------------------
# (a) The two-node grid special-case is bit-identical to make_output_grid
# ---------------------------------------------------------------------------


def _step_grid(t0, dt):
    """Mirror of the inline output-grid branch in ``ContinuousSystem.step``.

    Kept in lock-step with the source so the test exercises the *exact* decision
    ``step`` makes: the direct two-node shortcut above the helper's append
    threshold, the helper itself at/below it.
    """
    tf = t0 + dt
    if tf - t0 > 1e-9:
        return np.array([t0, tf], dtype=np.float64)
    from tsdynamics.utils.grids import make_output_grid

    return make_output_grid(t0, tf, dt)


def test_step_grid_matches_helper_across_dt_threshold():
    """``step``'s inline grid equals ``make_output_grid`` for every ``dt``.

    Above the ``1e-9`` cutover the shortcut ``[t0, tf]`` is used — that sits a
    thousandfold above the helper's ``1e-12`` append tolerance and above the
    float-subtraction rounding even at large ``t0``, and a single-``dt`` span never
    yields more than two nodes — so the grids are byte-identical there.  At or below
    the cutover ``step`` defers to the helper itself.  Sweeping ``dt`` log-uniformly
    from ``1e-13`` to ``5`` across a range of ``t0`` magnitudes crosses the cutover in
    both directions, pinning byte-for-byte agreement over the whole range (including
    the degenerate single-node band the naive direct shortcut gets wrong — the exact
    bug an earlier ``1e-12`` cutover shipped, caught here).
    """
    from tsdynamics.utils.grids import make_output_grid

    rng = np.random.default_rng(20240620)
    n_checked = 0
    for scale in (1.0, 1e2, 1e4, 1e6):
        for _ in range(60_000):
            t0 = float(rng.uniform(-scale, scale))
            dt = float(10.0 ** rng.uniform(-13.0, np.log10(5.0)))
            tf = t0 + dt
            if not tf > t0:  # sub-ULP step: helper *and* step both raise; not a grid compare
                continue
            helper = make_output_grid(t0, tf, dt)
            got = _step_grid(t0, dt)
            assert got.shape == helper.shape
            assert np.array_equal(got, helper)
            n_checked += 1
    assert n_checked > 100_000  # the sweep genuinely exercised the comparison


# ---------------------------------------------------------------------------
# (b) step() no longer routes its grid through make_output_grid
# ---------------------------------------------------------------------------


def test_step_does_not_call_make_output_grid(monkeypatch):
    """A warm ``step`` succeeds even when the grid helper is sabotaged.

    ``step`` builds its two-node grid inline, so breaking
    ``run.make_output_grid`` must not affect it — while the full ``integrate``
    entry point, which *does* build the grid through the helper, still raises.
    This pins the grid hoist structurally (not merely by timing).
    """

    def _boom(*_a, **_k):
        raise AssertionError("make_output_grid must not be called from step()")

    sys = ts.Rossler()
    sys.reinit([1.0, 1.0, 1.0])
    sys.step(0.01)  # warm: tape + step context cached by reinit

    monkeypatch.setattr("tsdynamics.engine.run.make_output_grid", _boom)
    # step keeps working: it never touches the helper…
    state = sys.step(0.01)
    assert state.shape == (3,)
    assert np.all(np.isfinite(state))
    # …but integrate still routes through it (sanity: the sabotage is effective).
    with pytest.raises(AssertionError):
        ts.Rossler().integrate(final_time=0.1, dt=0.01, ic=[1.0, 1.0, 1.0])


def test_step_keeps_make_output_grid_footgun_guards():
    """The direct grid build still raises on ``dt <= 0`` / a non-forward window.

    The pre-hoist ``step`` validated its span by calling ``make_output_grid``,
    which raises ``InvalidParameterError`` (a ``ValueError`` subclass) for a
    non-positive ``dt`` or a window that does not run forward.  The direct two-node
    construction must keep that loud-footgun contract — the happy path skips the
    helper, but a bad span still defers to it for the identical error.
    """
    from tsdynamics.errors import InvalidParameterError

    sys = ts.Rossler()
    sys.reinit([1.0, 1.0, 1.0])
    for bad in (0.0, -1.0):
        with pytest.raises(InvalidParameterError, match="dt must be > 0"):
            sys.step(bad)

    # A start time so large that ``t0 + dt == t0`` is a non-forward window — the
    # same case ``make_output_grid``'s second guard catches.
    far = ts.Rossler()
    far.reinit([1.0, 1.0, 1.0], t=1e16)
    assert 1e16 + 1.0 == 1e16  # guards the premise of the test
    with pytest.raises(InvalidParameterError, match="run forward in time"):
        far.step(1.0)


# ---------------------------------------------------------------------------
# (c) The tape wire arrays are marshalled once at reinit and reused
# ---------------------------------------------------------------------------


def test_tape_arrays_cached_and_reused():
    """``reinit`` caches ``Tape.to_arrays()``; the hot loop reuses the same tuple."""
    sys = ts.Rossler()
    sys.reinit([1.0, 1.0, 1.0])

    cached = sys._step_tape_arrays
    assert cached is not None
    # Equal, element-for-element, to a fresh marshalling of the live tape.
    fresh = sys._engine_problem.tape.to_arrays()
    assert len(cached) == len(fresh)
    for c, f in zip(cached, fresh, strict=True):
        assert np.array_equal(c, f) if isinstance(c, np.ndarray) else c == f

    # Identity is stable across steps — the loop does not re-marshal per call.
    before = sys._step_tape_arrays
    for _ in range(5):
        sys.step(0.01)
    assert sys._step_tape_arrays is before


def test_reinit_refreshes_cached_arrays_for_relowered_tape():
    """A structural-parameter change re-lowers the tape; ``reinit`` re-caches it.

    ``Lorenz96``'s ``N`` is structural (baked into the tape), so two different
    ``N`` lower to different tapes with different wire-array shapes.  The per-step
    cache must follow the tape it currently steps, or the stepper would feed the
    engine a stale tape.
    """
    small = ts.Lorenz96(N=5)
    small.reinit(np.ones(5))
    big = ts.Lorenz96(N=8)
    big.reinit(np.ones(8))
    # outputs[] length == dim, so the cached arrays track the re-lowered tape.
    assert small._step_tape_arrays[4].size == 5
    assert big._step_tape_arrays[4].size == 8


# ---------------------------------------------------------------------------
# (d) Answer-preservation on the marshalling-sensitive paths
# ---------------------------------------------------------------------------


def _reference_chain(make, ic, dt, n, **kw):
    """Per-``dt`` ``integrate`` chain — the released ``step`` semantics."""
    sys = make()
    state = np.asarray(ic, dtype=float)
    t = 0.0
    out = []
    for _ in range(n):
        traj = sys.integrate(final_time=t + dt, dt=dt, t0=t, ic=state, **kw)
        state = np.asarray(traj.y[-1], dtype=float)
        t += dt
        out.append(state)
    return np.array(out)


def _stepped(make, ic, dt, n, **kw):
    """The states the live ``step`` stepper hands out."""
    sys = make()
    sys.reinit(list(ic), **kw)
    return np.array([sys.step(dt).copy() for _ in range(n)])


def test_step_exact_with_jacobian_carrying_stiff_tape():
    """``step`` == per-``dt`` ``integrate`` bit-for-bit on a ``bdf`` (Jacobian) tape.

    The stiff implicit kernel needs ∂f/∂u on the tape — exactly the tape whose
    wire arrays WS-INVHOIST now caches.  Agreement to machine zero confirms the
    cached marshalling feeds the engine the identical Jacobian-carrying tape.
    """
    ic = [1.0, 1.0, 1.0]
    ref = _reference_chain(ts.Oregonator, ic, 0.01, 200, method="bdf")
    got = _stepped(ts.Oregonator, ic, 0.01, 200, method="bdf")
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


def test_step_exact_with_structural_parameter_tape():
    """``step`` == per-``dt`` ``integrate`` bit-for-bit on a structural-``N`` tape."""
    ic = [2.0, 2.0, 2.0, 2.0, 2.0, 5.0]
    ref = _reference_chain(lambda: ts.Lorenz96(N=6), ic, 0.01, 200)
    got = _stepped(lambda: ts.Lorenz96(N=6), ic, 0.01, 200)
    assert np.array_equal(got, ref)


def test_step_exact_on_jit_backend():
    """The hoist preserves the answer on the Cranelift JIT backend too."""
    ic = [1.0, 1.0, 1.0]
    ref = _reference_chain(ts.Rossler, ic, 0.01, 200, backend="jit")
    got = _stepped(ts.Rossler, ic, 0.01, 200, backend="jit")
    assert np.array_equal(got, ref)


def test_step_exact_non_autonomous_nonzero_t0():
    """``step`` == per-``dt`` ``integrate`` bit-for-bit on a *time-driven* RHS at ``t0 != 0``.

    The inline grid's first node is the live ``t0``; only a non-autonomous system
    (``t`` in the RHS) started at a non-zero time can catch a start-time-handling
    slip.  ``Lissajous2D`` is explicitly time-driven.
    """
    ic = [0.3, -0.4]
    t_start = 13.37
    sys = ts.Lissajous2D()
    sys.reinit(list(ic), t=t_start, method="rk45")
    got = np.array([sys.step(0.02).copy() for _ in range(150)])

    # Reference: chain single-``dt`` integrate() from the same live (state, time).
    ref_sys = ts.Lissajous2D()
    state = np.asarray(ic, dtype=float)
    t = t_start
    ref = []
    for _ in range(150):
        traj = ref_sys.integrate(final_time=t + 0.02, dt=0.02, t0=t, ic=state, method="rk45")
        state = np.asarray(traj.y[-1], dtype=float)
        t += 0.02
        ref.append(state)
    assert np.array_equal(got, np.array(ref))


def test_step_exact_tiny_dt_across_threshold():
    """``step`` == per-``dt`` ``integrate`` for ``dt`` straddling the ``1e-12`` grid band.

    Below the helper's append threshold the grid degenerates to a single node and
    ``step`` returns the un-advanced state — the same as ``integrate`` over that same
    span — so the two must still agree exactly on both sides of the threshold.
    """
    for dt in (5e-13, 1e-12, 2e-12, 1e-9, 1e-6, 1e-2):
        sys = ts.Lorenz()
        sys.reinit([1.0, 1.0, 1.0], method="rk45")
        got = sys.step(dt)
        ref = ts.Lorenz().integrate(final_time=dt, dt=dt, t0=0.0, ic=[1.0, 1.0, 1.0], method="rk45")
        assert np.array_equal(got, np.asarray(ref.y[-1], dtype=float)), f"mismatch at dt={dt}"


def test_step_reads_params_live_mid_loop():
    """A control-parameter change between steps still takes effect (live-stepper contract).

    WS-INVHOIST caches the *tape* at ``reinit`` but reads the parameter vector live
    each step, so mutating a control parameter mid-loop changes subsequent steps —
    exactly as the pre-hoist path did.  The over-eager version of this refactor would
    snapshot params into ``reinit`` and silently freeze them; this pins against that.
    """
    sys = ts.Rossler()
    sys.reinit([1.0, 1.0, 1.0], method="rk45")
    for _ in range(20):
        sys.step(0.05)
    state, t = sys.state(), sys.time()

    sys.c = 9.0  # mutate a control parameter mid-loop
    after = sys.step(0.05)

    # It must equal a fresh integrate() from the live (state, time) WITH the new c…
    changed = (
        ts.Rossler()
        .with_params(c=9.0)
        .integrate(final_time=t + 0.05, dt=0.05, t0=t, ic=state, method="rk45")
    )
    assert np.array_equal(after, np.asarray(changed.y[-1], dtype=float))
    # …and (negative control) differ from the unchanged-parameter step.
    unchanged = ts.Rossler().integrate(final_time=t + 0.05, dt=0.05, t0=t, ic=state, method="rk45")
    assert not np.array_equal(after, np.asarray(unchanged.y[-1], dtype=float))


# ---------------------------------------------------------------------------
# (e) The refactored engine seam: _step_continuous == _run_continuous
# ---------------------------------------------------------------------------


def test_step_continuous_matches_run_continuous():
    """The lean per-step seam returns exactly what the integrate seam does.

    Both call the same ``integrate_dense`` FFI over the same span; the only
    difference is that ``_step_continuous`` takes pre-marshalled tape arrays while
    ``_run_continuous`` marshals them from the ``Problem``.  Identical output
    confirms the split changed nothing numerically.
    """
    from tsdynamics.engine.problem import ode_problem
    from tsdynamics.engine.run import _run_continuous, _step_continuous

    sys = ts.Rossler()
    prob = ode_problem(sys, ic=[0.3, -0.2, 0.1], t0=2.0)
    t_eval = np.array([2.0, 2.0 + 0.02], dtype=np.float64)

    via_run = _run_continuous(prob, t_eval, method="rk45", rtol=1e-6, atol=1e-9, backend="interp")
    via_step = _step_continuous(
        prob.tape.to_arrays(),
        prob.ic,
        prob.params_vec(),
        t_eval,
        method="rk45",
        rtol=1e-6,
        atol=1e-9,
        jit=False,
        name="Rossler",
    )
    assert np.array_equal(via_run, via_step)


def test_step_continuous_diverges_loudly():
    """A non-finite span surfaces as ``RuntimeError`` — the loud-divergence contract.

    Divergence may raise from the engine FFI directly (it refuses to return a
    non-finite trajectory) or from ``_step_continuous``'s own finiteness guard
    (defense-in-depth, mirroring :func:`_run_continuous`); either way a blow-up is
    never silently handed back.
    """
    from tsdynamics.engine.problem import ode_problem
    from tsdynamics.engine.run import _step_continuous

    sys = ts.Lorenz()
    prob = ode_problem(sys, ic=[1e6, 1e6, 1e6], t0=0.0)
    arrays = prob.tape.to_arrays()
    params = prob.params_vec()
    span = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(RuntimeError):
        # Compounding huge fixed steps blow the explicit kernel up to non-finite.
        ic = prob.ic
        for _ in range(512):
            y = _step_continuous(
                arrays,
                ic,
                params,
                span,
                method="rk4",
                rtol=1e-6,
                atol=1e-9,
                jit=False,
                name="Lorenz",
            )
            ic = y[-1]


def test_step_continuous_finiteness_guard_message(monkeypatch):
    """``_step_continuous``'s own guard raises its named message on a non-finite return.

    The engine FFI normally refuses to *return* a non-finite trajectory (it raises
    first), so the defense-in-depth ``np.isfinite(y).all()`` guard is exercised here
    by forcing the engine call to hand back a poisoned array — confirming the guard
    is wired and names the system, mirroring :func:`_run_continuous`.
    """
    import tsdynamics.engine.run as run
    from tsdynamics.engine.run import _step_continuous

    monkeypatch.setattr(run, "_engine_integrate_dense", lambda *a, **k: np.array([[0.0], [np.inf]]))
    with pytest.raises(RuntimeError, match="diverged or the step collapsed"):
        _step_continuous(
            (),
            np.zeros(1),
            np.empty(0),
            np.array([0.0, 1.0]),
            method="rk4",
            rtol=1e-6,
            atol=1e-9,
            jit=False,
            name="PoisonSystem",
        )
