"""WS-CROSSKERNEL: the wired Rust event engine behind Poincaré sections.

``PoincareMap.trajectory`` (and the ``trajectory``-driven consumers built on it —
``poincare_section``, ``return_map(kind="poincare")``) now marches the whole
attractor and refines every crossing in **one engine call** via the wired
:func:`tsdyn_engine::integrate_events`, instead of the per-``dt`` Python loop.
These tests prove the wiring is fast *and* answer-preserving.

Answer-preservation note
------------------------
The engine crossing path marches the **fixed-step** ``rk4`` kernel at the
detection step ``dt`` (the engine's adaptive kernels carry no step ceiling, so an
adaptive march would grow the step, skip crossings and degrade the O(h⁴) Hermite
refinement).  So the right reference is the Python ``PoincareMap`` loop driven at
the *same* discretisation (inner system at ``method="rk4"``): against it the engine
agrees to ~machine precision per crossing.  Over many crossings of a *chaotic*
flow the two floating-point-distinct computations necessarily diverge (Rössler's
positive Lyapunov exponent amplifies roundoff) — so per-crossing equality is
asserted only over the horizon before roundoff amplifies past tolerance.  The full
section is the same attractor either way (asserted statistically).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.derived._crossings import engine_eligible, section_crossings

_rust = pytest.importorskip("tsdynamics._rust")

_NORMAL_X = np.array([1.0, 0.0, 0.0])
_IC = np.array([1.0, 1.0, 1.0])


def _python_rk4_reference(n: int, dt: float = 0.05):
    """Crossings from the Python PoincareMap loop at the engine's discretisation.

    Forcing the inner system to ``method="rk4"`` makes the per-``dt`` Python march
    use the same fixed-step kernel as the engine event path, so the two refine the
    identical bracket — the faithful parity oracle.
    """
    ros = ts.Rossler()
    ros.reinit(_IC.copy(), t=0.0, method="rk4")
    pmap = ts.PoincareMap(ros, plane=(0, 0.0), direction=+1, dt=dt)
    times = np.empty(n)
    states = np.empty((n, 3))
    for k in range(n):
        pmap._advance_to_crossing()
        times[k] = pmap._t_cross
        states[k] = pmap._u_cross
    return times, states


# --------------------------------------------------------------------------- #
# Correctness — the engine reproduces the Python refinement
# --------------------------------------------------------------------------- #


def test_first_crossing_matches_python_to_machine_precision() -> None:
    """The very first crossing (identical bracket) matches to ~1e-11.

    Isolates the *refinement* from chaotic trajectory divergence: both paths march
    the same first bracket, so any discrepancy is the bracketed root solver alone
    (engine Illinois vs ``brentq``, both to ``xtol=1e-14``).
    """
    te, ye, *_ = section_crossings(
        ts.Rossler(),
        _NORMAL_X,
        0.0,
        direction=+1,
        n_crossings=1,
        dt=0.05,
        max_time=1e4,
        ic=_IC.copy(),
    )
    tp, yp = _python_rk4_reference(1)
    assert abs(te[0] - tp[0]) < 1e-11
    np.testing.assert_allclose(ye[0], yp[0], atol=1e-11, rtol=0)


def test_section_matches_python_rk4_over_roundoff_horizon() -> None:
    """The engine section equals the Python ``rk4`` refinement to <1e-9.

    Asserted over the first 12 crossings — well inside the horizon where Rössler's
    roundoff amplification stays below 1e-9 (measured: ~1e-12 there).
    """
    k = 12
    te, ye, *_ = section_crossings(
        ts.Rossler(),
        _NORMAL_X,
        0.0,
        direction=+1,
        n_crossings=k,
        dt=0.05,
        max_time=1e4,
        ic=_IC.copy(),
    )
    tp, yp = _python_rk4_reference(k)
    np.testing.assert_allclose(te, tp, atol=1e-9, rtol=0)
    np.testing.assert_allclose(ye, yp, atol=1e-9, rtol=0)


def test_crossings_lie_on_the_plane() -> None:
    """Refined crossings sit on the section to far better than ``dt``."""
    sec = ts.PoincareMap(ts.Rossler(), plane=(0, 0.0), direction=+1, dt=0.05).trajectory(200)
    assert np.max(np.abs(sec.y[:, 0])) < 1e-10
    assert np.all(np.diff(sec.t) > 0)
    assert sec.y.shape == (200, 3)


def test_direction_filter_selects_the_right_branch() -> None:
    """``+1`` and ``-1`` collect distinct, finite, on-plane crossings."""
    up = ts.PoincareMap(ts.Rossler(), plane=(0, 0.0), direction=+1, dt=0.05).trajectory(50)
    down = ts.PoincareMap(ts.Rossler(), plane=(0, 0.0), direction=-1, dt=0.05).trajectory(50)
    assert np.isfinite(up.y).all() and np.isfinite(down.y).all()
    assert np.max(np.abs(up.y[:, 0])) < 1e-10
    assert np.max(np.abs(down.y[:, 0])) < 1e-10
    # The two branches are genuinely different sets of points.
    assert not np.allclose(up.y[:10], down.y[:10])


def test_jit_equals_interp_bit_for_bit() -> None:
    """The Cranelift JIT and the interpreter produce identical crossings."""
    kw = dict(direction=+1, n_crossings=80, dt=0.05, max_time=1e4, ic=_IC.copy())
    ti, yi, *_ = section_crossings(ts.Rossler(), _NORMAL_X, 0.0, backend="interp", **kw)
    tj, yj, *_ = section_crossings(ts.Rossler(), _NORMAL_X, 0.0, backend="jit", **kw)
    assert np.array_equal(ti, tj)
    assert np.array_equal(yi, yj)


def test_full_section_is_the_same_attractor_as_the_python_path() -> None:
    """Engine (``rk4``) and old Python (``rk45``) sections share the attractor.

    They cannot be point-equal over 2000 chaotic crossings (different kernels →
    roundoff diverges), but they image the *same* section — equal bounding boxes.
    """
    new = ts.PoincareMap(ts.Rossler(), plane=(0, 0.0), direction=+1, dt=0.05).trajectory(
        2000, transient=50
    )
    old = ts.PoincareMap(ts.Rossler(), plane=(0, 0.0), direction=+1, dt=0.05).trajectory(
        2000, transient=50, backend="reference"
    )
    np.testing.assert_allclose(new.y.min(0), old.y.min(0), atol=0.05)
    np.testing.assert_allclose(new.y.max(0), old.y.max(0), atol=0.05)


# --------------------------------------------------------------------------- #
# Routing — consumers inherit; non-ODE keeps the Python fallback
# --------------------------------------------------------------------------- #


def test_poincare_section_routes_through_the_engine() -> None:
    sec = ts.poincare_section(ts.Rossler(), (0, 0.0), n=300, skip_crossings=10, dt=0.05)
    assert sec.y.shape == (300, 3)
    assert np.max(np.abs(sec.y[:, 0])) < 1e-10


def test_return_map_poincare_routes_through_the_engine() -> None:
    rm = ts.return_map(ts.Rossler(), 1, method="poincare", plane=(0, 0.0), n=200)
    assert rm.current.shape == rm.successor.shape
    assert rm.current.size >= 150


def test_trajectory_zero_returns_empty() -> None:
    """``trajectory(0)`` returns correctly-shaped empties, not a crash."""
    sec = ts.PoincareMap(ts.Rossler(), plane=(0, 0.0), dt=0.05).trajectory(0)
    assert sec.t.shape == (0,)
    assert sec.y.shape == (0, 3)


def test_trajectory_then_step_continues_forward() -> None:
    """A ``step()`` after an engine ``trajectory()`` advances past it (no re-yield).

    The engine path advances the inner flow to the marched span end, so the next
    ``step()`` continues forward rather than restarting the section and silently
    re-returning crossings already collected.
    """
    pmap = ts.PoincareMap(ts.Rossler(), plane=(0, 0.0), direction=+1, dt=0.05)
    sec = pmap.trajectory(40)
    nxt = pmap.step()
    assert pmap.time() > sec.t[-1]  # strictly forward in continuous time
    # The next crossing is not a duplicate of any collected crossing.
    assert np.min(np.linalg.norm(sec.y - nxt, axis=1)) > 1e-6


def test_repeated_trajectory_advances() -> None:
    """Two successive engine ``trajectory()`` calls march forward, not in place."""
    pmap = ts.PoincareMap(ts.Rossler(), plane=(0, 0.0), direction=+1, dt=0.05)
    first = pmap.trajectory(30)
    second = pmap.trajectory(30)
    assert second.t[0] > first.t[-1]


def test_dde_keeps_the_python_fallback() -> None:
    """A DDE has no numeric RHS → not engine-eligible → Python crossing loop."""
    mg = ts.MackeyGlass()
    assert engine_eligible(mg, "interp") is False
    tr = mg.integrate(final_time=300.0, dt=0.5, history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)])
    sec = ts.PoincareMap(mg, plane=(0, 1.0), direction=+1, dt=0.5, max_time=2000.0).trajectory(
        10, ic=tr.y[-1]
    )
    assert sec.y.shape == (10, 1)


def test_stiff_default_keeps_the_python_fallback() -> None:
    """A stiff default (an implicit kernel) must not take the ``rk4`` engine path."""

    class _Stiff(ts.Rossler):  # type: ignore[misc, valid-type]
        _default_method = "bdf"

    assert engine_eligible(_Stiff(), "interp") is False


def test_reference_backend_forces_the_python_loop() -> None:
    assert engine_eligible(ts.Rossler(), "reference") is False


# --------------------------------------------------------------------------- #
# Robustness — divergence still raises loudly
# --------------------------------------------------------------------------- #


def test_no_crossing_within_max_time_raises() -> None:
    """A plane that misses the attractor raises, as the Python loop did."""
    with pytest.raises(RuntimeError):
        ts.PoincareMap(
            ts.Rossler(), plane=(0, 1e6), direction=+1, dt=0.05, max_time=20.0
        ).trajectory(5)


# --------------------------------------------------------------------------- #
# Performance — the per-dt FFI tax is gone (regression guard, not a benchmark)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_engine_path_is_far_faster_than_the_python_loop() -> None:
    """A 500-crossing section finishes in well under the old per-``dt`` wall time.

    The Python per-``dt`` loop took ~10 s+ for this; the engine call is sub-second.
    A loose ceiling (3 s) catches a regression to the slow path without being a
    flaky micro-benchmark.
    """
    ros = ts.Rossler()
    pmap = ts.PoincareMap(ros, plane=(0, 0.0), direction=+1, dt=0.05)
    t0 = time.perf_counter()
    sec = pmap.trajectory(500)
    elapsed = time.perf_counter() - t0
    assert sec.y.shape == (500, 3)
    assert elapsed < 3.0, f"engine Poincaré path took {elapsed:.2f}s (>3s ⇒ slow-path regression)"
