"""WS-MAPITER: map ``orbit_diagram`` routes through the Rust iterate engine.

A genuine :class:`~tsdynamics.families.DiscreteMap` runs the whole
``transient + record`` span for one parameter value in a single engine
``iterate`` call (one FFI round-trip) instead of a ``transient + n`` Python
``step()`` loop.  These tests pin the two load-bearing properties:

* **answer preservation** — the result is *byte-identical* to the old per-step
  path wherever the engine and NumPy agree bit-for-bit (the logistic map, across
  every regime), and records the *same attractor* for a chaotic map whose lowered
  IR differs from NumPy at the ULP level; and
* **the mechanism** — the engine path issues no ``step()`` calls, while flow
  wrappers (``PoincareMap`` / ``StroboscopicMap``) and engine-less fallbacks keep
  the per-step protocol path.

The reference is :func:`_old_orbit_diagram`, a faithful copy of the per-step
algorithm this stream replaced.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("tsdynamics._rust")  # engine-marked: routes through iterate

import tsdynamics as ts
from tsdynamics.engine.compile import TapeCompileError
from tsdynamics.engine.run import EngineNotAvailableError
from tsdynamics.families import DiscreteMap
from tsdynamics.systems import Henon, Logistic


def _old_orbit_diagram(
    sys,
    param,
    values,
    *,
    n=200,
    transient=500,
    carry_state=True,
    components=0,
    ic=None,
):
    """The pre-WS-MAPITER per-step algorithm, verbatim, as the byte-identity oracle."""
    comp = (components,) if isinstance(components, int | str) else tuple(components)
    names = getattr(sys, "variables", None)
    idx = [(names.index(c) if isinstance(c, str) else int(c)) for c in comp]
    values_arr = np.asarray(list(values), dtype=float)
    points: list[np.ndarray] = []
    state: np.ndarray | None = None
    for v in values_arr:
        current = sys.with_params(**{param: v})
        start = state if (carry_state and state is not None) else ic
        try:
            current.reinit(start)
            for _ in range(transient):
                current.step()
            rec = np.empty((n, len(idx)))
            for i in range(n):
                rec[i] = current.step()[idx]
        except RuntimeError:
            points.append(np.empty((0, len(idx))))
            state = None
            continue
        points.append(rec)
        if carry_state:
            state = current.state()
    return points


def _assert_points_equal(new, old, *, exact: bool):
    assert len(new) == len(old)
    for a, b in zip(new, old, strict=True):
        assert a.shape == b.shape
        if exact:
            assert np.array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------------------
# Byte-identity vs the old per-step path (the named acceptance)
# ---------------------------------------------------------------------------


def test_logistic_600x120_byte_identical():
    """The Logistic 600×120 sweep is byte-identical to the old step-loop path."""
    vals = np.linspace(2.5, 4.0, 600)
    new = ts.orbit_diagram(Logistic(), "r", vals, n=120, transient=500, ic=[0.3])
    old = _old_orbit_diagram(Logistic(), "r", vals, n=120, transient=500, ic=[0.3])
    _assert_points_equal(new.points, old, exact=True)


def test_carry_state_byte_identical():
    """``carry_state`` (final row → next IC) propagates byte-identically."""
    vals = np.linspace(2.8, 4.0, 200)
    new = ts.orbit_diagram(Logistic(), "r", vals, n=64, transient=200, ic=[0.123])
    old = _old_orbit_diagram(Logistic(), "r", vals, n=64, transient=200, ic=[0.123])
    _assert_points_equal(new.points, old, exact=True)


def test_no_carry_state_byte_identical():
    """With ``carry_state=False`` every value restarts from ``ic`` — still identical."""
    vals = np.linspace(2.8, 3.9, 120)
    new = ts.orbit_diagram(Logistic(), "r", vals, n=48, transient=150, ic=[0.4], carry_state=False)
    old = _old_orbit_diagram(
        Logistic(), "r", vals, n=48, transient=150, ic=[0.4], carry_state=False
    )
    _assert_points_equal(new.points, old, exact=True)


def test_multidim_convergent_window_agrees():
    """A 2-D map in a convergent (stable fixed point) window agrees to ~1e-11.

    Exact byte-identity is only claimed for maps whose lowering shares NumPy's
    arithmetic bit-for-bit (the logistic map, pinned above); a map with squares or
    transcendentals may differ at the ULP level — and that difference is platform
    dependent (libm vs the engine's host shims), so it must not be a CI gate.  In a
    *convergent* window both paths contract onto the same fixed point regardless,
    so they agree to a tight tolerance on every platform.
    """
    vals = np.linspace(0.05, 0.2, 40)  # stable fixed point across this range (b=0.3)
    kw = dict(n=60, transient=400)
    new = ts.orbit_diagram(
        Henon().with_params(b=0.3), "a", vals, ic=[0.0, 0.0], component=(0, 1), **kw
    )
    old = _old_orbit_diagram(
        Henon().with_params(b=0.3), "a", vals, ic=[0.0, 0.0], components=(0, 1), **kw
    )
    _assert_points_equal(new.points, old, exact=False)


# ---------------------------------------------------------------------------
# Divergence handling preserved
# ---------------------------------------------------------------------------


def test_divergence_records_empty_set_and_warns():
    """A divergent value records an empty set and warns — exactly as before."""
    with pytest.warns(RuntimeWarning, match="diverged"):
        od = ts.orbit_diagram(Logistic(), "r", [4.5], n=50, transient=50, ic=[0.5])
    assert od.points[0].shape == (0, 1)
    assert od.periods()[0] == -1


def test_divergence_then_recovery_byte_identical():
    """A diverged value resets the carry state; the mixed sweep stays identical."""
    vals = [3.7, 4.5, 3.2]  # middle value escapes [0, 1]
    with pytest.warns(RuntimeWarning, match="diverged"):
        new = ts.orbit_diagram(Logistic(), "r", vals, n=40, transient=80, ic=[0.5])
    old = _old_orbit_diagram(Logistic(), "r", vals, n=40, transient=80, ic=[0.5])
    _assert_points_equal(new.points, old, exact=True)


def test_zero_transient_and_n_records_empty_without_crashing():
    """``transient + n == 0`` records empty sets (regression: a zero-length engine
    iterate must not index ``y[-1]`` and abort the sweep)."""
    new = ts.orbit_diagram(Logistic(), "r", [3.2, 3.5, 3.8], n=0, transient=0, ic=[0.3])
    old = _old_orbit_diagram(Logistic(), "r", [3.2, 3.5, 3.8], n=0, transient=0, ic=[0.3])
    _assert_points_equal(new.points, old, exact=True)
    assert all(p.shape == (0, 1) for p in new.points)


# ---------------------------------------------------------------------------
# Chaotic map: same attractor (engine arithmetic ≠ NumPy at the ULP level, so a
# chaotic window is NOT pointwise-identical — it is the same recurrent set).
# ---------------------------------------------------------------------------


def test_chaotic_map_same_attractor():
    """Hénon's chaotic attractor is reproduced as a *set* (bounds + occupancy).

    A chaotic window is not pointwise-identical: the engine's lowered IR differs
    from NumPy at the ULP level and chaos amplifies it.  What is preserved is the
    invariant set — the support and the occupied region of state space.
    """
    ic = [0.1, 0.2]
    new = ts.orbit_diagram(Henon(), "a", [1.4], n=4000, transient=2000, ic=ic, component=(0, 1))
    old = _old_orbit_diagram(Henon(), "a", [1.4], n=4000, transient=2000, ic=ic, components=(0, 1))
    a, b = new.points[0], old[0]
    # Same support: matching min/max on each coordinate.
    np.testing.assert_allclose(a.min(axis=0), b.min(axis=0), atol=2e-2)
    np.testing.assert_allclose(a.max(axis=0), b.max(axis=0), atol=2e-2)
    # Same occupied region: high overlap of the populated histogram cells.
    rng = [[-1.5, 1.5], [-0.45, 0.45]]
    ha, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=24, range=rng)
    hb, _, _ = np.histogram2d(b[:, 0], b[:, 1], bins=24, range=rng)
    occ_a, occ_b = ha > 0, hb > 0
    jaccard = np.sum(occ_a & occ_b) / np.sum(occ_a | occ_b)
    assert jaccard > 0.9, f"attractor occupancy diverged (Jaccard={jaccard:.3f})"


# ---------------------------------------------------------------------------
# The mechanism: engine path issues no step(); flows keep the step path
# ---------------------------------------------------------------------------


def test_map_path_issues_no_step_calls(monkeypatch):
    """A DiscreteMap sweep takes the engine path — it never calls ``step()``."""
    calls = {"n": 0}
    real_step = DiscreteMap.step

    def counting_step(self, *a, **k):
        calls["n"] += 1
        return real_step(self, *a, **k)

    monkeypatch.setattr(DiscreteMap, "step", counting_step)
    ts.orbit_diagram(Logistic(), "r", np.linspace(2.8, 3.9, 50), n=64, transient=200, ic=[0.4])
    assert calls["n"] == 0


def test_flow_wrapper_keeps_step_path(monkeypatch):
    """A flow wrapped in StroboscopicMap stays on the per-step protocol path."""
    calls = {"n": 0}
    real_step = ts.StroboscopicMap.step

    def counting_step(self, *a, **k):
        calls["n"] += 1
        return real_step(self, *a, **k)

    monkeypatch.setattr(ts.StroboscopicMap, "step", counting_step)
    strobo = ts.StroboscopicMap(ts.Rossler(), period=2 * np.pi)
    ts.orbit_diagram(strobo, "c", [5.7], n=10, transient=10, component=0, ic=[1.0, 1.0, 0.0])
    assert calls["n"] > 0


# ---------------------------------------------------------------------------
# Fallback: a non-lowerable map or a wheel-free env uses the step path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [EngineNotAvailableError("no wheel"), TapeCompileError("no lower")])
def test_engine_unavailable_falls_back_byte_identical(monkeypatch, exc):
    """If ``iterate`` cannot run, orbit_diagram falls back to step — same answer."""

    def boom(self, *a, **k):
        raise exc

    monkeypatch.setattr(DiscreteMap, "iterate", boom)
    vals = np.linspace(2.8, 3.9, 80)
    new = ts.orbit_diagram(Logistic(), "r", vals, n=48, transient=150, ic=[0.31])
    old = _old_orbit_diagram(Logistic(), "r", vals, n=48, transient=150, ic=[0.31])
    _assert_points_equal(new.points, old, exact=True)


# ---------------------------------------------------------------------------
# Performance guard (generous, so it never flakes but catches a regression to
# the step loop — the measured local speedup is ~7–9×).
# ---------------------------------------------------------------------------


def test_map_orbit_diagram_is_faster_than_step_loop():
    """The engine path beats the reconstructed step loop by a wide margin."""
    vals = np.linspace(2.5, 4.0, 400)
    kw = dict(n=100, transient=400, ic=[0.3])

    ts.orbit_diagram(Logistic(), "r", vals, **kw)  # warm any one-time costs
    t0 = time.perf_counter()
    ts.orbit_diagram(Logistic(), "r", vals, **kw)
    t_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    _old_orbit_diagram(Logistic(), "r", vals, **kw)
    t_old = time.perf_counter() - t0

    assert t_old > 2.0 * t_new, f"expected a clear speedup, got {t_old / t_new:.1f}×"
