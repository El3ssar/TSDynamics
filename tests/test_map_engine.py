"""Cross-validation for the discrete-map engine path (streams E-MAP, E-OPS).

Every built-in map is swept once (via the registry-driven ``map_entry`` fixture).
Since stream **E-OPS** added the non-smooth / piecewise opcodes (comparison,
``min``/``max``, ``floor``/``ceil``, ``mod``) and taught the map tracer to model
NumPy array math symbolically, **every** built-in map now lowers to a
straight-line tape and iterates on the engine:

* the lowered next-state map equals the pure-Python ``_step`` *pointwise* to a
  tight tolerance (the chaos-free signal that the lowering is arithmetically
  faithful);
* a short engine-iterated trajectory tracks the reference-backend trajectory (the
  loop's bookkeeping — ordering, shape, step-index axis).

Two earlier obstacles are now handled: NumPy ufuncs on the symbolic state (the
tracer rebinds ``np`` to a SymEngine-backed shim) and genuine discontinuities —
modular reduction ``%`` lowers via ``floor`` and a state branch is written
branchlessly with ``np.where`` (→ ``Piecewise`` → comparison-masked blend).

The engine's native loop (the Rust ``tsdyn-engine`` map iterator, stream E-MAP)
is exercised here through the ``reference`` backend: it iterates the *same*
lowered tape the interpreter/JIT evaluate, in pure Python, so these checks hold
today without the compiled wheel (stream E7).  The interpreter already matches
the reference evaluator to ~1e-15 (stream E1), so reference-vs-``_step`` agreement
is native-vs-``_step`` agreement up to that bound.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.engine import run
from tsdynamics.engine.compile import TapeCompileError
from tsdynamics.engine.problem import map_problem
from tsdynamics.families.discrete import _unwrap_static

pytest.importorskip("tsdynamics._rust")

#: Maps that do not lower to the frozen straight-line IR.  Empty since E-OPS:
#: the non-smooth opcode block plus the symbolic-NumPy tracer cover the whole
#: built-in catalogue.  Kept as a named guard so a future regression (or a new
#: map using a construct the tracer cannot model) trips
#: :func:`test_non_lowerable_set_is_exhaustive` loudly.
NON_LOWERABLE: frozenset[str] = frozenset()


def _attractor_states(cls, *, n_warm: int = 60, drop: int = 40, take: int = 5) -> np.ndarray:
    """Return a few finite, on-attractor states for ``cls`` (deterministically).

    Iterates the reference path from a seeded initial condition and keeps a tail
    slice, so the test points are finite (the reference path only returns once the
    whole buffer is finite) and sit on the orbit rather than in the transient.

    The seed goes to ``run``, **not** to the global ``numpy.random`` stream: a
    system's random-IC draw comes from its own private ``Generator``
    (``SystemBase._ic_generator``), seeded from OS entropy precisely so a plain
    ``run()`` cannot disturb a caller's ``np.random.seed(0)``.  So the global
    reseed this used to do had no effect, and "deterministically" was untrue —
    which showed up as a ~1-in-3 failure on ``Bogdanov``, whose random draws
    escape to infinity.
    """
    warm = cls().run(steps=n_warm, backend="reference", seed=0)
    finite = warm.y[np.isfinite(warm.y).all(axis=1)]
    # Guard the slice that follows, not just the row count: finite[drop:drop+take]
    # is only non-degenerate when there are at least drop + take finite rows.
    assert finite.shape[0] >= drop + take, f"{cls.__name__}: too few finite warm-up states"
    return np.ascontiguousarray(finite[drop : drop + take])


def test_map_lowering_boundary(map_entry) -> None:
    """Each map either lowers, or is a *recorded* non-lowerable case.

    Pins the boundary in both directions: a map outside :data:`NON_LOWERABLE`
    must lower, and one inside it must raise :class:`TapeCompileError`.  Either a
    newly-lowerable map or a regression in coverage trips this.
    """
    cls = map_entry.cls
    if map_entry.name in NON_LOWERABLE:
        with pytest.raises(TapeCompileError):
            map_problem(cls())
    else:
        prob = map_problem(cls())  # must not raise
        assert prob.dim == cls().dim


def test_map_pointwise_matches_step(map_entry) -> None:
    """The lowered next-state map equals the pure-Python ``_step`` pointwise.

    The tight, chaos-free check: evaluated at the *same* on-attractor states, the
    engine's reference next-state and the pure-Python ``_step`` agree to a small
    tolerance.  Differences here are pure lowering errors, not sensitivity to
    initial conditions.
    """
    if map_entry.name in NON_LOWERABLE:
        pytest.skip(f"{map_entry.name} does not lower to the straight-line IR")

    cls = map_entry.cls
    m = cls()
    step = _unwrap_static(type(m)._step)
    params = m.params.as_tuple()
    for s in _attractor_states(cls):
        expected = np.asarray(step(s, *params), dtype=float).ravel()
        got = run.eval_rhs(m, s, backend="reference")
        np.testing.assert_allclose(
            got, expected, rtol=1e-9, atol=1e-12, err_msg=f"{map_entry.name} next-state mismatch"
        )


def test_map_short_trajectory_matches_step(map_entry) -> None:
    """A short engine-iterated trajectory tracks the reference-backend trajectory.

    Validates the iterate *loop* (output ordering, shape, the step-index time
    axis) rather than per-step arithmetic — that is covered tightly by
    :func:`test_map_pointwise_matches_step`.  The horizon is deliberately short:
    a chaotic map amplifies the ~1e-15 per-step lowering difference exponentially,
    so this only asks that the two trajectories stay close over a few steps (a
    structurally wrong loop diverges immediately and is caught).
    """
    if map_entry.name in NON_LOWERABLE:
        pytest.skip(f"{map_entry.name} does not lower to the straight-line IR")

    cls = map_entry.cls
    ic = _attractor_states(cls, take=1)[0]
    steps = 8
    interp = cls().run(steps=steps, ic=ic, backend="interp")
    ref = cls().run(steps=steps, ic=ic, backend="reference")

    # ``steps + 1``: a map run returns its INITIAL CONDITION followed by the
    # ``steps`` iterates, exactly as a flow returns ``t0`` followed by its
    # samples — so ``y[k]`` is the state at ``t[k]`` on both families.
    assert ref.y.shape == interp.y.shape == (steps + 1, cls().dim)
    np.testing.assert_array_equal(ref.t, interp.t)
    np.testing.assert_allclose(
        ref.y, interp.y, rtol=1e-6, atol=1e-8, err_msg=f"{map_entry.name} trajectory drift"
    )


def test_engine_path_diverges_loudly() -> None:
    """The engine ``iterate`` raises on a diverging orbit, never returns NaN rows.

    The family enforces the "diverge loudly" contract uniformly: the Rust engine
    path raises on a non-finite iterate, and the pure-Python reference iterator's
    silently-returned inf/NaN rows are caught at the family boundary and turned
    into the same loud failure (no quietly poisoned trajectory).
    """
    import tsdynamics as ts

    # Logistic with an initial condition outside [0, 1] escapes to -inf.
    with pytest.raises(RuntimeError, match="diverged"):
        ts.systems.Logistic().run(steps=60, ic=[2.0], backend="reference")


def test_non_lowerable_set_is_exhaustive() -> None:
    """Guard the bookkeeping: exactly the maps in :data:`NON_LOWERABLE` fail to lower.

    A single sweep that asserts the recorded set matches reality — so the set
    cannot silently drift out of date as maps are added or the lowering is
    extended.
    """
    from tsdynamics import registry

    actual_nonlowerable = set()
    for entry in registry.all_systems(family="map"):
        try:
            map_problem(entry.cls())
        except TapeCompileError:
            actual_nonlowerable.add(entry.name)
    assert actual_nonlowerable == set(NON_LOWERABLE)


# ---------------------------------------------------------------------------
# The map iterate path runs exactly ONE full-array finiteness scan
# (perf: the duplicate row-wise scan in ``DiscreteMap._iterate_engine`` cost
# ~12 ms of a 25 ms 1e6-step Hénon run — a 48% Python tax over the Rust kernel)
# ---------------------------------------------------------------------------


def _count_full_array_scans(monkeypatch, shape: tuple[int, int], fn) -> int:
    """Run ``fn`` counting ``np.isfinite`` calls over an array of exactly ``shape``."""
    calls = {"n": 0}
    real = np.isfinite

    def counting(x, *args, **kwargs):
        arr = x if isinstance(x, np.ndarray) else None
        if arr is not None and arr.shape == shape:
            calls["n"] += 1
        return real(x, *args, **kwargs)

    monkeypatch.setattr(np, "isfinite", counting)
    fn()
    return calls["n"]


@pytest.mark.parametrize("backend", ["interp", "jit", "reference"])
def test_map_iterate_runs_exactly_one_full_orbit_finiteness_scan(monkeypatch, backend) -> None:
    """One scan of the whole orbit block, not two.

    Every backend already diverges loudly *before* returning (the Rust map loop
    raises ``EngineError::Diverged`` → ``ConvergenceError`` at the first
    non-finite iterate; ``_reference_map`` raises per-iterate), so the single
    remaining scan in :func:`tsdynamics.engine._families._run_map` is
    defense-in-depth.  A second scan at the family boundary was unreachable and
    was pure O(steps x dim) tax — this counting test pins it at one and cannot
    flake.  (``reference`` also scans per-iterate on a ``(dim,)`` state; those
    are not full-orbit scans and are not counted.)
    """
    import tsdynamics as ts

    henon = ts.systems.Henon()
    steps = 400
    n = _count_full_array_scans(
        monkeypatch,
        (steps, 2),
        lambda: henon.run(steps=steps, ic=[0.1, 0.1], backend=backend),
    )
    assert n == 1, f"{backend}: {n} full-orbit scans (expected exactly 1)"


# ---------------------------------------------------------------------------
# Divergence behaviour is unchanged by the scan removal
# ---------------------------------------------------------------------------


class _Blowup(__import__("tsdynamics").DiscreteMap):
    """``x -> 10 x`` in 2-D: overflows to ``inf`` within a few hundred iterates."""

    dim = 2
    params = {"a": 10.0}  # noqa: RUF012
    default_ic = [1.0, 1.0]  # noqa: RUF012

    @staticmethod
    def _step(x, a):
        return [a * x[0], a * x[1]]

    @staticmethod
    def _jacobian(x, a):
        return [[a, 0.0], [0.0, a]]


@pytest.mark.parametrize("backend", ["interp", "jit"])
def test_engine_map_divergence_message_is_the_engine_seam_message(backend) -> None:
    """A diverging engine map raises ``ConvergenceError`` from ``_run_map``.

    Pins the exact message the user sees, so removing the (unreachable) family
    boundary scan cannot silently change the divergence report.
    """
    from tsdynamics.errors import ConvergenceError

    with pytest.raises(ConvergenceError) as exc:
        _Blowup().run(steps=1000, ic=[1.0, 1.0], backend=backend)
    assert str(exc.value) == (
        "_Blowup: map diverged or produced a non-finite state before reaching 1000 iterations."
    )


def test_reference_map_divergence_message_is_the_per_iterate_message() -> None:
    """The reference oracle keeps its own per-iterate divergence report."""
    from tsdynamics.errors import ConvergenceError

    with pytest.raises(ConvergenceError, match=r"non-finite state at iteration \d+ \(0-based"):
        _Blowup().run(steps=1000, ic=[1.0, 1.0], backend="reference")


def test_diverging_map_with_explicit_ic_does_not_retry(monkeypatch) -> None:
    """An explicit ``ic`` raises on the first attempt — retry policy unchanged."""
    from tsdynamics.errors import ConvergenceError

    calls = {"n": 0}
    real = type(_Blowup())._iterate_engine

    def counting(self, **kwargs):
        calls["n"] += 1
        return real(self, **kwargs)

    monkeypatch.setattr(_Blowup, "_iterate_engine", counting)
    with pytest.raises(ConvergenceError):
        _Blowup().run(steps=1000, ic=[1.0, 1.0], max_retries=5)
    assert calls["n"] == 1


def test_diverging_map_without_explicit_ic_still_retries(monkeypatch) -> None:
    """No explicit ``ic`` → the random-IC retry budget is still spent, then raises."""
    from tsdynamics.errors import ConvergenceError

    calls = {"n": 0}
    real = type(_Blowup())._iterate_engine

    def counting(self, **kwargs):
        calls["n"] += 1
        return real(self, **kwargs)

    monkeypatch.setattr(_Blowup, "_iterate_engine", counting)
    m = _Blowup()
    m.ic = None
    with (
        pytest.warns(RuntimeWarning, match="Retrying from a new random"),
        pytest.raises(ConvergenceError),
    ):
        m.run(steps=1000, max_retries=3)
    assert calls["n"] == 3
