"""Map ``orbit_diagram`` sweeps the whole parameter array in one engine call.

A genuine :class:`~tsdynamics.families.DiscreteMap` is lowered **once** keeping
the swept parameter as the tape's single runtime input, then the Rust sweep
kernel (stream ``perf/param-sweep-kernel``) iterates ``transient + record`` for
every value in **one** FFI round-trip — superseding the WS-MAPITER path's one
``iterate`` call per value (a 1000-value sweep was ~1000 round-trips).  These
tests pin the two load-bearing properties:

* **answer preservation** — the result is *byte-identical* to the old per-step
  path wherever the engine and NumPy agree bit-for-bit (the logistic map, across
  every regime), and records the *same attractor* for a chaotic map whose lowered
  IR differs from NumPy at the ULP level; and
* **the mechanism** — the engine sweep issues no ``step()`` calls, while
  ``StroboscopicMap`` and engine-less fallbacks keep the per-step protocol path.

A second stream (``perf/poincare-orbit-diagram``) closed the other half of the
gap: a ``PoincareMap`` sweep — the flagship *bifurcation diagram of a flow* —
now collects each value's section through ``PoincareMap.trajectory`` (the wired
Rust event march, WS-CROSSKERNEL) instead of the per-``dt`` ``step()`` loop.
That path is discretised with the fixed-step ``rk4`` kernel, so it is *not*
pointwise-equal to the step loop on a chaotic band; what is pinned below is the
**diagram** (``periods()`` / ``bifurcation_points()``), the point *set* in a
periodic window, and byte-identity wherever the engine march declines.

The reference is :func:`_old_orbit_diagram`, a faithful copy of the per-step
algorithm these streams replaced.
"""

from __future__ import annotations

import importlib
import time

import numpy as np
import pytest

pytest.importorskip("tsdynamics._rust")  # engine-marked: routes through the sweep kernel

import tsdynamics as ts
from tsdynamics.engine.compile import TapeCompileError
from tsdynamics.engine.run import EngineNotAvailableError
from tsdynamics.errors import BackendError
from tsdynamics.families import DiscreteMap
from tsdynamics.systems import Henon, Logistic

# The package re-exports the ``orbit_diagram`` *function* under the submodule's
# dotted name, shadowing the module on attribute access; reach the real module
# object (whose ``_sweep_via_kernel`` the fallback tests patch) via importlib.
od_mod = importlib.import_module("tsdynamics.analysis.orbits.orbit_diagram")


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
    new = ts.orbit_diagram(Logistic(), "r", vals, points_per_value=120, transient=500, ic=[0.3])
    old = _old_orbit_diagram(Logistic(), "r", vals, n=120, transient=500, ic=[0.3])
    _assert_points_equal(new.points, old, exact=True)


def test_carry_state_byte_identical():
    """``carry_state`` (final row → next IC) propagates byte-identically."""
    vals = np.linspace(2.8, 4.0, 200)
    new = ts.orbit_diagram(Logistic(), "r", vals, points_per_value=64, transient=200, ic=[0.123])
    old = _old_orbit_diagram(Logistic(), "r", vals, n=64, transient=200, ic=[0.123])
    _assert_points_equal(new.points, old, exact=True)


def test_no_carry_state_byte_identical():
    """With ``carry_state=False`` every value restarts from ``ic`` — still identical."""
    vals = np.linspace(2.8, 3.9, 120)
    new = ts.orbit_diagram(
        Logistic(), "r", vals, points_per_value=48, transient=150, ic=[0.4], carry_state=False
    )
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
    kw = dict(points_per_value=60, transient=400)
    old_kw = dict(n=60, transient=400)
    new = ts.orbit_diagram(
        Henon().with_params(b=0.3), "a", vals, ic=[0.0, 0.0], component=(0, 1), **kw
    )
    old = _old_orbit_diagram(
        Henon().with_params(b=0.3), "a", vals, ic=[0.0, 0.0], components=(0, 1), **old_kw
    )
    _assert_points_equal(new.points, old, exact=False)


# ---------------------------------------------------------------------------
# Divergence handling preserved
# ---------------------------------------------------------------------------


def test_divergence_records_empty_set_and_warns():
    """A divergent value records an empty set and warns — exactly as before."""
    with pytest.warns(RuntimeWarning, match="diverged"):
        od = ts.orbit_diagram(Logistic(), "r", [4.5], points_per_value=50, transient=50, ic=[0.5])
    assert od.points[0].shape == (0, 1)
    assert od.periods()[0] == -1


def test_divergence_then_recovery_byte_identical():
    """A diverged value resets the carry state; the mixed sweep stays identical."""
    vals = [3.7, 4.5, 3.2]  # middle value escapes [0, 1]
    with pytest.warns(RuntimeWarning, match="diverged"):
        new = ts.orbit_diagram(Logistic(), "r", vals, points_per_value=40, transient=80, ic=[0.5])
    old = _old_orbit_diagram(Logistic(), "r", vals, n=40, transient=80, ic=[0.5])
    _assert_points_equal(new.points, old, exact=True)


def test_zero_transient_and_n_records_empty_without_crashing():
    """``transient + n == 0`` records empty sets (regression: a zero-length engine
    iterate must not index ``y[-1]`` and abort the sweep)."""
    new = ts.orbit_diagram(
        Logistic(), "r", [3.2, 3.5, 3.8], points_per_value=0, transient=0, ic=[0.3]
    )
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
    new = ts.orbit_diagram(
        Henon(), "a", [1.4], points_per_value=4000, transient=2000, ic=ic, component=(0, 1)
    )
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
    """A DiscreteMap sweep takes the engine sweep path — it never calls ``step()``."""
    calls = {"n": 0}
    real_step = DiscreteMap.step

    def counting_step(self, *a, **k):
        calls["n"] += 1
        return real_step(self, *a, **k)

    monkeypatch.setattr(DiscreteMap, "step", counting_step)
    ts.orbit_diagram(
        Logistic(), "r", np.linspace(2.8, 3.9, 50), points_per_value=64, transient=200, ic=[0.4]
    )
    assert calls["n"] == 0


def test_map_sweep_is_a_single_engine_call(monkeypatch):
    """The whole DiscreteMap sweep is ONE engine call (the named win of this stream).

    The sweep kernel marches every parameter value internally, so
    ``orbit_diagram`` over a genuine map invokes the engine sweep entry point
    exactly once for the entire diagram — not once per value (the WS-MAPITER
    path this stream supersedes) and not the per-step protocol loop.
    """
    from tsdynamics.engine import run as run_mod

    calls = {"n": 0}
    real_sweep = run_mod.map_param_sweep

    def counting_sweep(*a, **k):
        calls["n"] += 1
        return real_sweep(*a, **k)

    # Patch the name the orbit-diagram module resolves at call time (it imports
    # ``map_param_sweep`` inside ``_sweep_via_kernel``), so patching the run module
    # is what the wiring sees.
    monkeypatch.setattr(run_mod, "map_param_sweep", counting_sweep)
    ts.orbit_diagram(
        Logistic(), "r", np.linspace(2.5, 4.0, 200), points_per_value=64, transient=300, ic=[0.4]
    )
    assert calls["n"] == 1, f"expected one sweep call for the whole diagram, got {calls['n']}"


def test_flow_wrapper_keeps_step_path(monkeypatch):
    """A flow wrapped in StroboscopicMap stays on the per-step protocol path."""
    calls = {"n": 0}
    real_step = ts.StroboscopicMap.step

    def counting_step(self, *a, **k):
        calls["n"] += 1
        return real_step(self, *a, **k)

    monkeypatch.setattr(ts.StroboscopicMap, "step", counting_step)
    strobo = ts.StroboscopicMap(ts.Rossler(), period=2 * np.pi)
    ts.orbit_diagram(
        strobo, "c", [5.7], points_per_value=10, transient=10, component=0, ic=[1.0, 1.0, 0.0]
    )
    assert calls["n"] > 0


# ---------------------------------------------------------------------------
# Fallback: a non-lowerable map or a wheel-free env uses the per-step path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [EngineNotAvailableError("no wheel"), TapeCompileError("no lower")])
def test_engine_unavailable_falls_back_byte_identical(monkeypatch, exc):
    """If the sweep kernel cannot run, orbit_diagram falls back to step — same answer.

    A non-lowerable ``_step`` (``TapeCompileError`` → ``NotImplementedError``) or a
    wheel-free environment (``EngineNotAvailableError`` → ``BackendError``) raised
    from the engine sweep must drop to the per-value/per-step protocol loop, which
    is byte-identical on the logistic map.
    """

    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(od_mod, "_sweep_via_kernel", boom)
    vals = np.linspace(2.8, 3.9, 80)
    new = ts.orbit_diagram(Logistic(), "r", vals, points_per_value=48, transient=150, ic=[0.31])
    old = _old_orbit_diagram(Logistic(), "r", vals, n=48, transient=150, ic=[0.31])
    _assert_points_equal(new.points, old, exact=True)


@pytest.mark.parametrize(
    "exc",
    [
        BackendError("backend down"),  # any BackendError, not just the leaf
        NotImplementedError("cannot lower"),  # any NotImplementedError, not just TapeCompileError
    ],
)
def test_engine_fallback_catches_public_bases(monkeypatch, exc):
    """The sweep→step fallback catches the PUBLIC bases, not engine-internal leaves.

    The fallback catches ``(NotImplementedError, BackendError)`` — the public bases
    ``TapeCompileError`` and ``EngineNotAvailableError`` subclass — so a bare
    ``BackendError`` and a bare ``NotImplementedError`` raised from the sweep kernel
    must still trigger the per-step fallback (a narrower leaf-only catch would let
    these propagate).
    """

    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(od_mod, "_sweep_via_kernel", boom)
    vals = np.linspace(2.8, 3.9, 60)
    new = ts.orbit_diagram(Logistic(), "r", vals, points_per_value=40, transient=120, ic=[0.27])
    old = _old_orbit_diagram(Logistic(), "r", vals, n=40, transient=120, ic=[0.27])
    _assert_points_equal(new.points, old, exact=True)


# ---------------------------------------------------------------------------
# Performance guard (generous, so it never flakes but catches a regression to
# the step loop — the measured local speedup is ~7–9×).
# ---------------------------------------------------------------------------


def test_map_orbit_diagram_is_faster_than_step_loop():
    """The engine path beats the reconstructed step loop by a wide margin.

    Measured locally at ~370x for this 400x100 Logistic sweep; the gate asks for
    20x, which is an order of magnitude of headroom against a loaded runner while
    still failing loudly if the sweep ever falls back to the step loop (which is
    what the pre-kernel path cost).
    """
    vals = np.linspace(2.5, 4.0, 400)
    kw = dict(points_per_value=100, transient=400, ic=[0.3])
    old_kw = dict(n=100, transient=400, ic=[0.3])

    ts.orbit_diagram(Logistic(), "r", vals, **kw)  # warm any one-time costs
    t0 = time.perf_counter()
    ts.orbit_diagram(Logistic(), "r", vals, **kw)
    t_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    _old_orbit_diagram(Logistic(), "r", vals, **old_kw)
    t_old = time.perf_counter() - t0

    assert t_old > 20.0 * t_new, f"expected a clear speedup, got {t_old / t_new:.1f}×"


# ---------------------------------------------------------------------------
# PoincareMap: the bifurcation-diagram-of-a-flow path routes through
# ``PoincareMap.trajectory`` (the wired Rust event march), not the step loop.
# ---------------------------------------------------------------------------


def _pmap():
    """A fresh Rössler Poincaré map on the ``y = 0`` upward section."""
    return ts.PoincareMap(ts.systems.Rossler(), plane=("y", 0.0, "up"))


def test_poincare_orbit_diagram_issues_no_step_calls(monkeypatch):
    """A ``PoincareMap`` sweep collects each section via ``trajectory`` — no ``step()``.

    The mechanism check for this stream: before it, every recorded crossing cost
    one ``PoincareMap.step()`` (which re-entered the flow integrator once per
    detection ``dt``); now the whole per-value section is one ``trajectory`` call.
    """
    step_calls = {"n": 0}
    traj_calls = {"n": 0}
    real_step = ts.PoincareMap.step
    real_traj = ts.PoincareMap.trajectory

    def counting_step(self, *a, **k):
        step_calls["n"] += 1
        return real_step(self, *a, **k)

    def counting_traj(self, *a, **k):
        traj_calls["n"] += 1
        return real_traj(self, *a, **k)

    monkeypatch.setattr(ts.PoincareMap, "step", counting_step)
    monkeypatch.setattr(ts.PoincareMap, "trajectory", counting_traj)
    ts.orbit_diagram(
        _pmap(), "c", [4.0, 5.0], points_per_value=20, transient=20, ic=[1.0, 1.0, 1.0]
    )

    assert step_calls["n"] == 0, f"expected no step() calls, got {step_calls['n']}"
    assert traj_calls["n"] == 2, f"expected one trajectory() call per value, got {traj_calls['n']}"


def test_poincare_orbit_diagram_reproduces_the_branch_structure():
    """The cascade's ``periods()`` and ``bifurcation_points()`` are unchanged.

    The engine march is discretised with the fixed-step ``rk4`` kernel at the
    detection ``dt`` (see :mod:`tsdynamics.derived._crossings`), whereas the step
    loop drove the flow's adaptive default — so the two are not pointwise equal on
    a chaotic band.  What must be preserved is the *diagram*: the same asymptotic
    branch counts at every parameter value and therefore the same detected
    bifurcation onsets.
    """
    vals = np.linspace(2.5, 6.0, 24)
    kw = dict(points_per_value=60, transient=150, ic=[1.0, 1.0, 1.0])
    new = ts.orbit_diagram(_pmap(), "c", vals, **kw)
    old_points = _old_orbit_diagram(_pmap(), "c", vals, n=60, transient=150, ic=[1.0, 1.0, 1.0])
    old = od_mod.OrbitDiagram(param="c", values=vals, points=old_points, components=(0,))

    np.testing.assert_array_equal(new.periods(), old.periods())
    np.testing.assert_array_equal(new.bifurcation_points(), old.bifurcation_points())


def test_poincare_periodic_window_agrees_pointwise_as_a_set():
    """In a periodic window the two paths record the *same* point set.

    A period-``p`` window is recorded cyclically, so the two paths may start on a
    different branch of the cycle — the sequences are phase-shifted.  Comparing the
    sorted values removes the phase and leaves only the ``rk4``-vs-adaptive
    discretisation difference, which is ~5e-8 here.
    """
    vals = np.linspace(3.0, 3.6, 4)
    kw = dict(points_per_value=40, transient=200, ic=[1.0, 1.0, 1.0])
    new = ts.orbit_diagram(_pmap(), "c", vals, **kw)
    old = _old_orbit_diagram(_pmap(), "c", vals, n=40, transient=200, ic=[1.0, 1.0, 1.0])
    for a, b in zip(new.points, old, strict=True):
        np.testing.assert_allclose(np.sort(a.ravel()), np.sort(b.ravel()), atol=1e-6, rtol=0)


def test_non_eligible_poincare_map_is_byte_identical():
    """A ``PoincareMap`` the engine march declines stays byte-identical.

    A DDE has no ``_rhs_numeric``, so ``trajectory`` falls back to
    ``_python_trajectory`` — the very ``_advance_to_crossing`` loop ``step()``
    drives.  The recorded points must therefore match bit-for-bit, including the
    divergence contract (an empty set for a value with no crossing).
    """
    kwargs = dict(plane=(0, 1.0, "up"), dt=0.5, max_time=2000.0)
    vals = [17.0, 18.0]  # the second value finds no crossing → empty set + warning
    with pytest.warns(RuntimeWarning, match="diverged"):
        new = ts.orbit_diagram(
            ts.PoincareMap(ts.MackeyGlass(), **kwargs),
            "tau",
            vals,
            points_per_value=8,
            transient=3,
            ic=[1.1],
        )
    old = _old_orbit_diagram(
        ts.PoincareMap(ts.MackeyGlass(), **kwargs), "tau", vals, n=8, transient=3, ic=[1.1]
    )
    _assert_points_equal(new.points, old, exact=True)


def test_zero_n_poincare_keeps_the_step_loop(monkeypatch):
    """``n == 0`` keeps the step loop, so ``carry_state`` semantics do not drift.

    With nothing recorded, ``trajectory`` cannot leave ``state()`` on the last
    *discarded* transient crossing the way the step loop does, so the degenerate
    case is deliberately excluded from the fast path.
    """
    calls = {"n": 0}
    real_step = ts.PoincareMap.step

    def counting_step(self, *a, **k):
        calls["n"] += 1
        return real_step(self, *a, **k)

    monkeypatch.setattr(ts.PoincareMap, "step", counting_step)
    new = ts.orbit_diagram(
        _pmap(), "c", [4.0, 5.0], points_per_value=0, transient=3, ic=[1.0, 1.0, 1.0]
    )
    monkeypatch.undo()
    old = _old_orbit_diagram(_pmap(), "c", [4.0, 5.0], n=0, transient=3, ic=[1.0, 1.0, 1.0])
    assert calls["n"] > 0
    _assert_points_equal(new.points, old, exact=True)


def test_poincare_orbit_diagram_is_faster_than_step_loop():
    """The flagship "bifurcation diagram of a flow" beats the step loop by ≫10x.

    Measured locally at ~72x for this 8-value Rössler sweep (7.2 s → 0.10 s); the
    gate asks for 10x, leaving a 7x margin against a loaded runner while still
    failing if the sweep regresses to the per-``dt`` step loop.
    """
    vals = np.linspace(3.0, 6.0, 8)
    kw = dict(points_per_value=60, transient=100, ic=[1.0, 1.0, 1.0])

    ts.orbit_diagram(_pmap(), "c", vals, **kw)  # warm any one-time costs
    t0 = time.perf_counter()
    ts.orbit_diagram(_pmap(), "c", vals, **kw)
    t_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    _old_orbit_diagram(_pmap(), "c", vals, n=60, transient=100, ic=[1.0, 1.0, 1.0])
    t_old = time.perf_counter() - t0

    assert t_old > 10.0 * t_new, f"expected a clear speedup, got {t_old / t_new:.1f}×"


def test_poincare_sweep_into_a_stiff_regime_fails_loudly():
    """A value the fixed-step ``rk4`` march cannot hold is loud, never silently wrong.

    ``rk4`` has a bounded stability region (``|λh| ≲ 2.785``), so a parameter value
    that makes the flow stiff relative to the detection ``dt`` blows the engine
    march up where the old adaptive ``step()`` loop would simply have shrunk its
    step.  That is an accepted cost of routing through ``trajectory`` — but it must
    stay *visible*: the sweep has to record an empty point set and warn for that
    value rather than emit points from a diverged march.  On the Rössler
    ``c``-sweep at ``dt=0.01`` the threshold is ``c ≈ 200`` (measured); ``c=1e6``
    is far past it.
    """
    vals = [5.7, 1.0e6]
    with pytest.warns(RuntimeWarning, match="diverged"):
        od = ts.orbit_diagram(
            _pmap(), "c", vals, points_per_value=5, transient=5, ic=[1.0, 1.0, 1.0]
        )

    assert od.points[0].shape == (5, 1)  # the ordinary value is unaffected
    assert od.points[1].size == 0  # the stiff value records nothing, loudly
    assert np.all(np.isfinite(od.points[0]))
