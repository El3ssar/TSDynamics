"""Dense output and the ``max_step`` ceiling (v6, audit ids ``eng-no-dense-output`` /
``perf-no-max-step``).

Two engine defects are fixed here and pinned by this file:

1. ``integrate_grid`` had **no dense output**, so the adaptive stepper was forced
   to land exactly on every requested output sample.  The consequence was that
   *the answer depended on the output grid* — Lorenz to ``T=10`` at ``rtol=1e-6``
   returned an error of ``1.3e-3`` at ``dt=10`` but ``2.8e-5`` at ``dt=0.01``, a
   47× spread — which silently contradicted the documented contract for ``dt``
   ("output sampling interval; the internal stepper is adaptive").  ``dt`` was
   secretly an accuracy knob, and ``rtol`` was correspondingly inert on a fine
   grid.
2. No adaptive kernel exposed a ``max_step`` ceiling, so the only way to bound the
   internal step was to bound the *output* grid.

The fix is deliberately narrow, and the tests below are organised around the
discriminator that defines its blast radius.  **A number may move only when all
three hold:** the call came from ``ContinuousSystem.integrate`` / ``.run``, the
output grid has **three or more** points, and the kernel is one carrying a native
continuous extension (``rk45`` / ``tsit5`` / ``dop853``).  Everything else — every two-node
grid, every other kernel, every other family — is bit-for-bit unchanged, and
``test_the_must_not_move_set_is_unchanged`` automates exactly that check.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidParameterError

pytest.importorskip("tsdynamics._rust")

IC = [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _reference_lorenz(final_time: float, ic=IC) -> np.ndarray:
    """A tight, wholly independent reference: SciPy DOP853 at ``rtol=1e-13``.

    Written against the Lorenz equations directly (not through the lowered tape),
    so it shares no code path with the engine — it is the "closer to truth" yard-
    stick this file measures accuracy against, rather than re-baselining.
    """
    from scipy.integrate import solve_ivp

    s = ts.Lorenz()
    sigma, rho, beta = s.sigma, s.rho, s.beta

    def f(t, y):
        return [sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]]

    sol = solve_ivp(f, (0.0, final_time), list(ic), method="DOP853", rtol=1e-13, atol=1e-15)
    assert sol.success
    return np.asarray(sol.y[:, -1])


def _run_in_subprocess(code: str, **env_extra: str) -> str:
    """Run ``code`` in a fresh interpreter with extra env vars set.

    The dense-output bypass is a process-wide env var read at call time, but a
    subprocess keeps the parent's environment (and any cached engine state)
    completely out of it.
    """
    env = dict(os.environ, **env_extra)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout}\n{out.stderr}"
    return out.stdout


# ---------------------------------------------------------------------------
# P1 — THE contract test: the answer no longer depends on the output grid
# ---------------------------------------------------------------------------


def test_the_answer_no_longer_depends_on_the_output_grid():
    """``dt`` is an output sampling interval, not an accuracy knob.

    Integrating the same span at wildly different output resolutions must give
    the same trajectory to within the accuracy ``rtol`` actually bought.  Before
    v6 this failed: the forced landing made a fine ``dt`` silently more accurate.
    """
    lorenz = ts.Lorenz()
    finals = {}
    for dt in (10.0, 5.0, 1.0, 0.1, 0.01):
        traj = lorenz.integrate(final_time=10.0, dt=dt, ic=IC, method="rk45", rtol=1e-8, atol=1e-11)
        finals[dt] = traj.y[-1]

    coarse = finals[10.0]
    for dt, y in finals.items():
        rel = np.max(np.abs(y - coarse)) / np.max(np.abs(coarse))
        assert rel < 1e-6, f"dt={dt} disagrees with dt=10 by {rel:.2e} relative"

    # And the shared *interior* samples of two different grids agree too, not
    # merely the endpoints.
    fine = lorenz.integrate(final_time=10.0, dt=0.01, ic=IC, method="rk45", rtol=1e-8, atol=1e-11)
    coarse_grid = lorenz.integrate(
        final_time=10.0, dt=0.1, ic=IC, method="rk45", rtol=1e-8, atol=1e-11
    )
    shared = fine.y[::10]
    assert shared.shape == coarse_grid.y.shape
    assert np.max(np.abs(shared - coarse_grid.y)) < 1e-6


def test_grid_dependence_is_small_compared_to_the_requested_tolerance():
    """The residual grid spread must be far *below* the delivered error.

    Stated as a ratio so it is meaningful without a magic constant: how much the
    answer moves when only the output resolution changes, against how far the
    answer is from truth at the requested tolerance.  Grid dependence should be a
    rounding detail of the requested accuracy, not comparable to it.
    """
    truth = _reference_lorenz(10.0)
    lorenz = ts.Lorenz()
    errs = {}
    for dt in (10.0, 0.1, 0.01):
        traj = lorenz.integrate(final_time=10.0, dt=dt, ic=IC, method="rk45", rtol=1e-6, atol=1e-9)
        errs[dt] = np.max(np.abs(traj.y[-1] - truth))
    spread = max(errs.values()) / min(errs.values())
    assert spread < 1.05, f"the answer still depends on the grid: error ratio {spread:.2f} ({errs})"


# ---------------------------------------------------------------------------
# P2 — rtol is live
# ---------------------------------------------------------------------------


def test_rtol_is_live_on_a_fine_output_grid():
    """A knob that silently does nothing is worse than a missing knob.

    On an output grid finer than the natural step the forced landing made the
    step size a function of ``dt`` alone, so tightening ``rtol`` changed nothing.
    Now it must change both the numbers and the delivered accuracy.
    """
    truth = _reference_lorenz(5.0)
    lorenz = ts.Lorenz()
    loose = lorenz.integrate(final_time=5.0, dt=0.001, ic=IC, method="rk45", rtol=1e-4, atol=1e-7)
    tight = lorenz.integrate(final_time=5.0, dt=0.001, ic=IC, method="rk45", rtol=1e-11, atol=1e-13)
    assert not np.array_equal(loose.y, tight.y), "rtol is inert on a fine grid"
    e_loose = np.max(np.abs(loose.y[-1] - truth))
    e_tight = np.max(np.abs(tight.y[-1] - truth))
    assert e_tight < e_loose / 100, f"tightening rtol barely helped: {e_loose:.2e} -> {e_tight:.2e}"


# ---------------------------------------------------------------------------
# P3 — the engine and its reference oracle now agree *in kind*
# ---------------------------------------------------------------------------


def test_engine_and_reference_oracle_agree_in_kind():
    """``backend="reference"`` is ``solve_ivp(t_eval=...)``, which has always
    produced its samples by interpolation.  Before v6 the engine produced them by
    forced landing, so the two implemented *different output semantics* and the
    cross-validation was structurally weaker than it looked.  They now agree in
    kind, so the residual is genuine solver disagreement only.
    """
    lorenz = ts.Lorenz()
    kw = dict(final_time=5.0, dt=0.01, ic=IC, method="rk45", rtol=1e-9, atol=1e-12)
    engine = lorenz.integrate(**kw)
    ref = lorenz.integrate(**kw, backend="reference")
    assert np.max(np.abs(engine.y - ref.y)) < 1e-6


# ---------------------------------------------------------------------------
# P4 — interp == jit bit-for-bit, on a dense grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [0.05, 0.01, 0.002])
@pytest.mark.parametrize("method", ["rk45", "tsit5", "dop853"])
def test_interp_equals_jit_bit_for_bit_on_a_dense_grid(method, dt):
    """The load-bearing engine contract, re-checked on the dense path.

    The interpolant is pure arithmetic on stage buffers both evaluators fill
    identically, so bit-identity must survive.  The catalogue cross-validation
    only ever exercised the *landing* path (``dt=0.05`` over ``T=0.5``), so this
    is the dense-path leg it did not have.
    """
    lorenz = ts.Lorenz()
    kw = dict(final_time=2.0, dt=dt, ic=IC, method=method, rtol=1e-8, atol=1e-11)
    a = lorenz.integrate(**kw, backend="interp")
    b = lorenz.integrate(**kw, backend="jit")
    np.testing.assert_array_equal(a.y, b.y)


# ---------------------------------------------------------------------------
# P5 — the bypass gate (mirrors test_lowering_cache / test_jit_cache)
# ---------------------------------------------------------------------------

_BYPASS_PROBE = """
import numpy as np, tsdynamics as ts
lorenz = ts.Lorenz()
traj = lorenz.integrate(final_time=3.0, dt={dt}, ic=[1.0, 1.0, 1.0],
                        method="rk45", rtol=1e-7, atol=1e-10)
print(traj.y.tobytes().hex())
print(traj.meta["dense_output"])
"""


def test_the_bypass_env_var_reproduces_the_pre_v6_numbers():
    """``TSDYNAMICS_NO_DENSE_OUTPUT=1`` restores the forced-landing march.

    The escape hatch a user mid-migration needs, and the control this whole file
    is measured against.  On a many-point grid the two must differ (dense output
    is doing something); the flag must be reported in provenance either way.
    """
    on = _run_in_subprocess(_BYPASS_PROBE.format(dt=0.01)).splitlines()
    off = _run_in_subprocess(_BYPASS_PROBE.format(dt=0.01), TSDYNAMICS_NO_DENSE_OUTPUT="1")
    off = off.splitlines()
    assert on[1] == "True" and off[1] == "False"
    assert on[0] != off[0], "the bypass changed nothing on a 301-point grid"


def test_a_two_node_grid_is_identical_with_and_without_the_bypass():
    """No interior sample ⇒ nothing to interpolate ⇒ bit-for-bit unchanged.

    This is the structural reason the whole two-node ecosystem (the stepping
    protocol, the Lyapunov chunk loop, the basin cell march) is out of the blast
    radius — the engine gates dense output on ``t_eval.len() > 2``.
    """
    on = _run_in_subprocess(_BYPASS_PROBE.format(dt=3.0)).splitlines()[0]
    off = _run_in_subprocess(
        _BYPASS_PROBE.format(dt=3.0), TSDYNAMICS_NO_DENSE_OUTPUT="1"
    ).splitlines()[0]
    assert on == off


# ---------------------------------------------------------------------------
# P6 — the must-not-move set
# ---------------------------------------------------------------------------

_MUST_NOT_MOVE = """
import numpy as np, tsdynamics as ts

out = {}

# (a) the stepping protocol (two-node spans through OdeStepper.advance)
lor = ts.Lorenz()
lor.reinit([1.0, 1.0, 1.0])
out["step"] = np.array([lor.step(0.01) for _ in range(200)])

# (b) lyapunov_spectrum (the engine chunk loop; t_eval = [t, tf])
out["lyap"] = ts.Lorenz().lyapunov_spectrum(final_time=30.0, dt=0.1, transient=5.0,
                                            ic=[1.0, 1.0, 1.0])

# (c) PoincareMap (pinned to the fixed-step rk4 march, which carries no Caps::dense)
pm = ts.PoincareMap(ts.Rossler(ic=[1.0, 1.0, 0.0]), plane=("y", 0.0, "up"))
out["poincare"] = pm.trajectory(60).y

# (d) a map (a different family entirely)
out["map"] = ts.Henon().iterate(steps=500, ic=[0.1, 0.1]).y

# (e) a kernel with no continuous extension, on a fine grid
out["rk4"] = ts.Lorenz().integrate(final_time=3.0, dt=0.01, ic=[1.0, 1.0, 1.0],
                                   method="rk4").y
out["bdf"] = ts.Lorenz().integrate(final_time=3.0, dt=0.01, ic=[1.0, 1.0, 1.0],
                                   method="bdf").y

for k in sorted(out):
    print(k, np.ascontiguousarray(out[k], dtype=np.float64).tobytes().hex())
"""


def test_the_must_not_move_set_is_unchanged():
    """Everything outside the discriminator must be **byte-for-byte** identical.

    Run the same six probes with dense output on and with the bypass set; any
    difference is a bug, not churn.  This automates the discriminator: only
    ``integrate``/``run``, ≥3 output points, ``rk45``/``tsit5``/``dop853`` may move.
    """
    on = dict(line.split(" ", 1) for line in _run_in_subprocess(_MUST_NOT_MOVE).splitlines())
    off = dict(
        line.split(" ", 1)
        for line in _run_in_subprocess(_MUST_NOT_MOVE, TSDYNAMICS_NO_DENSE_OUTPUT="1").splitlines()
    )
    assert set(on) == set(off)
    moved = [k for k in on if on[k] != off[k]]
    assert not moved, f"these must not move but did: {moved}"


def test_poincare_is_structurally_out_of_the_blast_radius():
    """The guard behind the ``poincare`` leg above.

    The crossing march resolves ``method="rk4"``, and ``rk4`` carries no native
    continuous extension — so the engine's dense branch is unreachable from
    ``PoincareMap`` *by construction*, not by numerical coincidence.  Both halves
    are asserted: the pin is still in the source, and ``rk4`` really is inert
    under the dense flag while ``rk45`` really is not (so the observable is not
    vacuous).
    """
    import inspect

    from tsdynamics.derived import _crossings

    src = inspect.getsource(_crossings)
    assert 'method="rk4"' in src, "the crossing march is no longer pinned to rk4"

    probe = (
        "import tsdynamics as ts;"
        "t = ts.Lorenz().integrate(final_time=3.0, dt=0.01, ic=[1.0,1.0,1.0], method={m!r});"
        "print(t.y.tobytes().hex())"
    )
    rk4_on = _run_in_subprocess(probe.format(m="rk4")).strip()
    rk4_off = _run_in_subprocess(probe.format(m="rk4"), TSDYNAMICS_NO_DENSE_OUTPUT="1").strip()
    assert rk4_on == rk4_off, "rk4 must be inert under the dense flag"

    rk45_on = _run_in_subprocess(probe.format(m="rk45")).strip()
    rk45_off = _run_in_subprocess(probe.format(m="rk45"), TSDYNAMICS_NO_DENSE_OUTPUT="1").strip()
    assert rk45_on != rk45_off, "the rk4 check above would be vacuous if rk45 were inert too"


# ---------------------------------------------------------------------------
# P7 — max_step
# ---------------------------------------------------------------------------


def test_max_step_reproduces_the_pre_v6_forced_landing():
    """``max_step=dt`` states the old *implicit* step bound explicitly.

    Before v6 the forced landing meant no step could exceed ``dt``.  Setting the
    ceiling to ``dt`` restores that bound, so the run reproduces the bypassed
    (pre-v6) numbers.  **Not** bit-for-bit: the pre-v6 march additionally *landed*
    on every sample, so its step sequence is `dt, dt, dt, …` exactly, whereas the
    ceiling only caps the step the controller asks for — and on a chaotic flow
    that ULP-scale difference in step placement amplifies over the span.  The
    agreement is therefore asserted at a level far tighter than the delivered
    accuracy (both runs are checked against an independent DOP853@1e-13 reference
    below), which is what "the ceiling restores the old step regime" means
    operationally.
    """
    lorenz = ts.Lorenz()
    kw = dict(final_time=3.0, dt=0.01, ic=IC, method="rk45", rtol=1e-9, atol=1e-12)
    capped = lorenz.integrate(**kw, max_step=0.01)
    bypassed = _run_in_subprocess(
        "import numpy as np, tsdynamics as ts;"
        "t = ts.Lorenz().integrate(final_time=3.0, dt=0.01, ic=[1.0,1.0,1.0],"
        " method='rk45', rtol=1e-9, atol=1e-12);"
        "print(t.y.tobytes().hex())",
        TSDYNAMICS_NO_DENSE_OUTPUT="1",
    ).strip()
    want = np.frombuffer(bytes.fromhex(bypassed), dtype=np.float64).reshape(capped.y.shape)
    drift = np.max(np.abs(capped.y - want))
    truth = _reference_lorenz(3.0)
    delivered = max(
        np.max(np.abs(capped.y[-1] - truth)),
        np.max(np.abs(want[-1] - truth)),
    )
    assert drift < 1e-6, f"max_step={kw['dt']} did not restore the old step regime ({drift:.2e})"
    # The two paths differ by *less* than either differs from truth: the residual
    # is step placement, not accuracy.
    assert drift < delivered


class _Oscillator(ts.ContinuousSystem):
    """``x'' = -x`` — analytic solution ``x = cos t``, ``v = -sin t`` from ``(1, 0)``.

    Used by the ``max_step`` order measurement below, which needs a problem whose
    exact answer is known in closed form so the observable is truth, not another
    integration.
    """

    dim = 2
    variables = ("x", "v")
    params: dict[str, float] = {}

    @staticmethod
    def _equations(y, t):  # noqa: D102
        return [y(1), -y(0)]


def test_max_step_bounds_every_internal_step():
    """The ceiling really caps *every* step — measured through the error's order in it.

    The naive form of this test ("a capped run differs from a free one") is
    nearly worthless: it passes for any ceiling that binds even once.  So drive
    the observable instead.  With ``rtol``/``atol`` set absurdly loose the error
    controller accepts whatever it is handed, so ``max_step`` becomes the *only*
    thing setting the step size and ``rk45`` degenerates to a fixed-step method at
    exactly ``max_step``.  Its global error must then fall as ``O(max_step^5)``.

    Measured order ≈ 5.2 over ``max_step ∈ {0.4, 0.2, 0.1}``; the assertion floor
    is 4.0.  If the ceiling were ignored — or applied to only some steps — the
    three runs would be identical (order 0) or the fit would collapse, so this
    cannot pass vacuously.  It is also machine-independent: it compares an error
    *ratio* against a closed-form solution, never a wall time.

    (The Rust side pins the same property directly by counting RHS evaluations:
    ``integrate::tests::max_step_is_honoured_and_infinity_is_inert`` and
    ``event::tests::max_step_bounds_the_event_march``.)
    """
    caps = (0.4, 0.2, 0.1)
    errs = []
    for cap in caps:
        traj = _Oscillator().integrate(
            final_time=4.0,
            dt=4.0,
            ic=[1.0, 0.0],
            method="rk45",
            rtol=1e9,
            atol=1e9,
            max_step=cap,
        )
        errs.append(abs(float(traj.y[-1, 0]) - math.cos(4.0)))
    assert errs[0] > errs[1] > errs[2] > 0.0, f"the ceiling did not bind: {errs}"
    order = math.log(errs[0] / errs[2]) / math.log(caps[0] / caps[2])
    assert order > 4.0, f"error falls at only order {order:.2f} in max_step ({errs})"

    # And the cost consequence, on a real system: a ceiling far below the natural
    # step forces strictly more work and so a materially different step sequence.
    kw = dict(final_time=20.0, dt=0.5, ic=IC, method="rk45", rtol=1e-6, atol=1e-9)
    free = ts.Lorenz().integrate(**kw)
    capped = ts.Lorenz().integrate(**kw, max_step=1e-3)
    assert np.isfinite(capped.y).all()
    assert free.y.shape == capped.y.shape
    assert not np.array_equal(free.y, capped.y), "max_step changed nothing at all"


@pytest.mark.parametrize("backend", ["interp", "jit", "reference"])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_a_bad_max_step_is_a_typed_invalid_parameter_error(bad, backend):
    """A scalar option value, so it joins ``rtol``/``atol`` on
    :class:`~tsdynamics.errors.InvalidParameterError` (a ``ValueError``).

    Swept over **every** backend deliberately.  The engine validates at the FFI
    boundary, but ``reference`` delegates to SciPy, which raises a bare
    ``ValueError`` for a non-positive ceiling and *silently ignores* ``nan`` — so
    without an explicit guard the same typo would raise on two backends and
    quietly do nothing on the third.  This sweep is what pins the three together.
    """
    with pytest.raises(InvalidParameterError, match="max_step"):
        ts.Lorenz().integrate(final_time=1.0, dt=0.1, ic=IC, backend=backend, max_step=bad)


def test_max_step_is_recorded_in_provenance():
    traj = ts.Lorenz().integrate(final_time=1.0, dt=0.1, ic=IC, max_step=0.05)
    assert traj.meta["max_step"] == 0.05
    assert traj.meta["dense_output"] is True
    free = ts.Lorenz().integrate(final_time=1.0, dt=0.1, ic=IC)
    assert math.isinf(free.meta["max_step"])


def test_max_step_is_honoured_through_the_stepping_protocol():
    """``reinit(max_step=...)`` is stored and applied to every later ``step``."""
    lorenz = ts.Lorenz()
    lorenz.reinit(IC, max_step=1e-3)
    u = lorenz.step(0.1)
    assert np.isfinite(u).all()
    # An out-of-range ceiling is rejected at the first step, through the same type.
    lorenz.reinit(IC, max_step=-1.0)
    with pytest.raises(InvalidParameterError, match="max_step"):
        lorenz.step(0.1)


# ---------------------------------------------------------------------------
# P8 — the registry-driven catalogue sweep
# ---------------------------------------------------------------------------


def test_every_ode_integrates_on_a_dense_grid(ode_entry):
    """Registry sweep: every catalogue ODE still integrates, with dense output on.

    Costs no maintenance (it rides the existing ``ode_entry`` fixture), and is
    the guard that the change did not break some corner of the catalogue that no
    hand-written case covers.
    """
    system = ode_entry.cls()
    traj = system.integrate(final_time=1.0, dt=0.05, method="rk45", rtol=1e-6, atol=1e-9)
    assert traj.y.shape == (traj.t.size, system.dim)
    assert traj.t.size > 2, "the sweep must exercise a grid with interior points"
    assert np.isfinite(traj.y).all()
    # Row 0 is the initial condition, exactly (never an interpolant).
    np.testing.assert_array_equal(traj.y[0], np.asarray(traj.meta["ic"], dtype=float))
