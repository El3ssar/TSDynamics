"""Cross-validation for the discrete-map engine path (stream E-MAP).

Every built-in map is swept once (via the registry-driven ``map_entry`` fixture)
and falls into one of two buckets:

* **Lowerable** — ``_step`` traces to a straight-line tape, so the map iterates
  on the engine.  We check the lowered next-state map equals the v2 Numba
  ``_step`` *pointwise* to a tight tolerance (the chaos-free signal that the
  lowering is arithmetically faithful), and that a short engine-iterated
  trajectory tracks the Numba trajectory (the loop's bookkeeping — ordering,
  shape, step-index axis).
* **Non-lowerable** — ``_step`` either branches on the state / uses ``%`` (a
  discontinuity the straight-line IR has no opcode for) or calls a NumPy ufunc
  that does not dispatch onto symbolic operands.  These are pinned to raise
  :class:`~tsdynamics.engine.compile.TapeCompileError`, so the boundary is
  explicit and this test fails loudly if the lowering coverage ever changes.

The engine's native loop (the Rust ``tsdyn-engine`` map iterator, stream E-MAP)
is exercised here through the ``reference`` backend: it iterates the *same*
lowered tape the interpreter/JIT evaluate, in pure Python, so these checks hold
today without the compiled wheel (stream E7).  The interpreter already matches
the reference evaluator to ~1e-15 (stream E1), so reference-vs-Numba agreement is
native-vs-Numba agreement up to that bound.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.engine import run
from tsdynamics.engine.compile import TapeCompileError
from tsdynamics.engine.problem import map_problem
from tsdynamics.families.discrete import _unwrap_static

# ---------------------------------------------------------------------------
# The non-lowerable catalogue, split by *why* it cannot become a straight-line
# tape.  Both classes iterate on the v2 Numba path until the lowering grows to
# cover them; the split records what each would need.
# ---------------------------------------------------------------------------

#: Steps written with NumPy ufuncs (``np.sin``/``np.cos``/``np.exp``/...), which
#: the IR *could* represent (it has Sin/Cos/Exp/Sqrt/Abs/Sign/... opcodes) — they
#: fail only because the ufunc does not dispatch onto a symbolic operand during
#: tracing.  Folding a ufunc-dispatching trace into the lowering (a compile-layer
#: change) would let these lower unchanged.
_NONLOWERABLE_UFUNC = frozenset(
    {
        "Ikeda",
        "Chirikov",
        "Svensson",
        "Bedhead",
        "Hopalong",
        "Pickover",
        "Chebyshev",
        "Gauss",
        "DeJong",
        "Ricker",
    }
)

#: Steps with a genuine discontinuity — modular reduction (``%``) or a state
#: branch — that the straight-line IR cannot represent at all (no mod/floor
#: opcode); these need a new IR primitive, not just a richer trace.
_NONLOWERABLE_DISCONTINUOUS = frozenset({"Zaslavskii", "Baker", "Circle", "KaplanYorke"})

#: Maps that do not lower to the frozen straight-line IR today.
NON_LOWERABLE = _NONLOWERABLE_UFUNC | _NONLOWERABLE_DISCONTINUOUS


def _attractor_states(cls, *, n_warm: int = 60, drop: int = 40, take: int = 5) -> np.ndarray:
    """Return a few finite, on-attractor states for ``cls`` (deterministically).

    Iterates the Numba path from a seeded initial condition and keeps a tail
    slice, so the test points are finite (the Numba path only returns once the
    whole buffer is finite) and sit on the orbit rather than in the transient.
    """
    np.random.seed(0)
    warm = cls().iterate(steps=n_warm, backend="numba")
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


def test_map_pointwise_matches_numba(map_entry) -> None:
    """The lowered next-state map equals the v2 Numba ``_step`` pointwise.

    The tight, chaos-free check: evaluated at the *same* on-attractor states, the
    engine's reference next-state and the Numba ``_step`` agree to a small
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


def test_map_short_trajectory_matches_numba(map_entry) -> None:
    """A short engine-iterated trajectory tracks the Numba trajectory.

    Validates the iterate *loop* (output ordering, shape, the step-index time
    axis) rather than per-step arithmetic — that is covered tightly by
    :func:`test_map_pointwise_matches_numba`.  The horizon is deliberately short:
    a chaotic map amplifies the ~1e-15 per-step lowering difference exponentially,
    so this only asks that the two trajectories stay close over a few steps (a
    structurally wrong loop diverges immediately and is caught).
    """
    if map_entry.name in NON_LOWERABLE:
        pytest.skip(f"{map_entry.name} does not lower to the straight-line IR")

    cls = map_entry.cls
    ic = _attractor_states(cls, take=1)[0]
    steps = 8
    numba = cls().iterate(steps=steps, ic=ic, backend="numba")
    ref = cls().iterate(steps=steps, ic=ic, backend="reference")

    assert ref.y.shape == numba.y.shape == (steps, cls().dim)
    np.testing.assert_array_equal(ref.t, numba.t)
    np.testing.assert_allclose(
        ref.y, numba.y, rtol=1e-6, atol=1e-8, err_msg=f"{map_entry.name} trajectory drift"
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
        ts.Logistic().iterate(steps=60, ic=[2.0], backend="reference")


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
