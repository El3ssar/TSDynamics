"""Tests for the compiled-evaluator (JIT) cache — v6 WP3-perf.

``backend="jit"`` hands the lowered tape to Cranelift.  Before this cache, the
bridge compiled the whole tape on **every** FFI call, so the compile was a
per-call cost: ~0.13 ms for Lorenz but ~0.3 s for a Gray–Scott field, which made
``"jit"`` *slower* than ``"interp"`` for short runs and made a parameter sweep
re-compile a byte-identical tape once per value.  The engine now memoises the
compiled evaluator on the tape's identity (``crates/tsdyn-jit/src/cache.rs``),
the exact analogue of the lowered-tape cache in
:mod:`tsdynamics.engine.compile`.

A stale hit would silently return wrong numbers, so the tests here assert both
halves:

(a) a repeat run on an unchanged tape is a **hit**, and a control-parameter sweep
    keys on one entry (the tape does not depend on control-parameter values);
(b) a changed tape is a **miss** — the three things that change generated code
    are the immediates, the opcode sequence, and the ``with_jacobian`` lowering,
    and each is exercised through the real FFI entry point;
(c) results are **bit-for-bit identical** with the cache on, with it bypassed via
    ``TSDYNAMICS_NO_JIT_CACHE``, and against the interpreter (the documented
    ``interp == jit`` contract, which a wrongly-keyed cache would break).
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.engine import run as runmod
from tsdynamics.engine.compile import lower_ode_cached
from tsdynamics.engine.problem import ode_problem

_rust = pytest.importorskip("tsdynamics._rust")

#: The env var that turns the cache into a straight passthrough.
_NO_JIT_CACHE = "TSDYNAMICS_NO_JIT_CACHE"


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Each test starts from an empty cache with the bypass cleared."""
    monkeypatch.delenv(_NO_JIT_CACHE, raising=False)
    runmod.clear_jit_cache()
    yield
    runmod.clear_jit_cache()


def _integrate_tape(tape, ic, p, t_eval, *, jit: bool) -> np.ndarray:
    """Drive ``integrate_dense`` directly, so the test owns the exact tape."""
    return np.asarray(
        _rust.integrate_dense(
            *tape.to_arrays(),
            np.asarray(ic, dtype=np.float64),
            np.asarray(p, dtype=np.float64),
            np.asarray(t_eval, dtype=np.float64),
            "rk4",
            1e-8,
            1e-8,
            jit,
        )
    )


# ---------------------------------------------------------------------------
# (a) an unchanged tape is served from the cache
# ---------------------------------------------------------------------------


def test_repeat_jit_run_is_a_cache_hit():
    lor = ts.systems.Lorenz()
    lor.integrate(final_time=0.1, dt=0.01, backend="jit", ic=[1.0, 1.0, 1.0])
    first = runmod.jit_cache_stats()
    assert (first["hits"], first["misses"], first["size"]) == (0, 1, 1)

    lor.integrate(final_time=0.1, dt=0.01, backend="jit", ic=[1.0, 1.0, 1.0])
    second = runmod.jit_cache_stats()
    assert second["hits"] == 1
    assert second["misses"] == 1
    assert second["size"] == 1


def test_control_param_sweep_reuses_one_compiled_evaluator():
    """The tape ignores control-parameter values, so the compile must too."""
    for rho in (28.0, 35.0, 40.0, 45.0):
        ts.systems.Lorenz(params={"rho": rho}).integrate(
            final_time=0.1, dt=0.01, backend="jit", ic=[1.0, 1.0, 1.0]
        )
    stats = runmod.jit_cache_stats()
    assert stats["size"] == 1, "a control-parameter sweep must key on one entry"
    assert (stats["misses"], stats["hits"]) == (1, 3)


def test_interp_run_never_touches_the_jit_cache():
    ts.systems.Lorenz().integrate(final_time=0.1, dt=0.01, backend="interp")
    stats = runmod.jit_cache_stats()
    assert (stats["hits"], stats["misses"], stats["size"]) == (0, 0, 0)


def test_clear_resets_counters_and_store():
    lor = ts.systems.Lorenz()
    lor.integrate(final_time=0.1, dt=0.01, backend="jit")
    lor.integrate(final_time=0.1, dt=0.01, backend="jit")
    assert runmod.jit_cache_stats()["hits"] == 1
    runmod.clear_jit_cache()
    assert runmod.jit_cache_stats() == {
        "hits": 0,
        "misses": 0,
        "size": 0,
        "maxsize": runmod.jit_cache_stats()["maxsize"],
    }
    # …and the next call therefore compiles again.
    lor.integrate(final_time=0.1, dt=0.01, backend="jit")
    assert runmod.jit_cache_stats()["misses"] == 1


# ---------------------------------------------------------------------------
# (b) every input to codegen is part of the key — a change is a miss
# ---------------------------------------------------------------------------


def test_changed_immediates_are_a_miss():
    """Same shape, different constants: `du/dt = -c·u` for two values of c.

    A cache keyed on anything coarser than the tape's own contents would serve
    the first compile for the second system and silently integrate the wrong
    equation, so the test checks the *numbers*, not just the counters.
    """

    class Decay(ts.ContinuousSystem):
        """du/dt = -k u with k baked in as a structural constant."""

        params = {"k": 1.0}
        _structural_params = frozenset({"k"})
        dim = 1

        @staticmethod
        def _equations(y, t, k):
            return [-k * y(0)]

    t_eval = np.linspace(0.0, 1.0, 201)  # fine enough for rk4 to hit exp() to 1e-8
    slow = lower_ode_cached(Decay(params={"k": 1.0}))
    fast = lower_ode_cached(Decay(params={"k": 3.0}))
    # The two tapes differ ONLY in their immediates — the one thing a
    # shape-based key would miss.
    slow_arrays, fast_arrays = slow.to_arrays(), fast.to_arrays()
    for i in (0, 1, 2, 4, 5):  # ops, a, b, outputs, jac_outputs
        assert np.array_equal(slow_arrays[i], fast_arrays[i])
    assert not np.array_equal(slow_arrays[3], fast_arrays[3])  # imm

    y_slow = _integrate_tape(slow, [1.0], [], t_eval, jit=True)
    y_fast = _integrate_tape(fast, [1.0], [], t_eval, jit=True)
    stats = runmod.jit_cache_stats()
    assert (stats["misses"], stats["size"]) == (2, 2)

    # Each ran its own equation (a stale hit would make these identical).
    assert y_slow[-1, 0] == pytest.approx(np.exp(-1.0), rel=1e-6)
    assert y_fast[-1, 0] == pytest.approx(np.exp(-3.0), rel=1e-6)


def test_changed_opcode_sequence_is_a_miss():
    """Two systems with the same dimension and no constants, different math."""
    t_eval = np.linspace(0.0, 0.5, 6)
    lor = ode_problem(ts.systems.Lorenz())
    ross = ode_problem(ts.systems.Rossler())
    _integrate_tape(lor.tape, [1.0, 1.0, 1.0], lor.params_vec(), t_eval, jit=True)
    _integrate_tape(ross.tape, [1.0, 1.0, 1.0], ross.params_vec(), t_eval, jit=True)
    stats = runmod.jit_cache_stats()
    assert (stats["misses"], stats["hits"], stats["size"]) == (2, 0, 2)


def test_with_jacobian_lowering_is_a_miss():
    """The Jacobian-bearing tape compiles a second function — a distinct key."""
    lor = ts.systems.Lorenz()
    plain = lower_ode_cached(lor, with_jacobian=False)
    with_jac = lower_ode_cached(lor, with_jacobian=True)
    assert not plain.has_jacobian
    assert with_jac.has_jacobian

    t_eval = np.linspace(0.0, 0.5, 6)
    p = ode_problem(lor).params_vec()
    y_plain = _integrate_tape(plain, [1.0, 1.0, 1.0], p, t_eval, jit=True)
    y_jac = _integrate_tape(with_jac, [1.0, 1.0, 1.0], p, t_eval, jit=True)
    stats = runmod.jit_cache_stats()
    assert (stats["misses"], stats["hits"], stats["size"]) == (2, 0, 2)
    # Same RHS either way — the extra outputs must not perturb the trajectory.
    assert np.array_equal(y_plain, y_jac)


# ---------------------------------------------------------------------------
# (c) the cache is answer-preserving
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Lorenz", "Rossler", "Thomas"])
def test_cached_equals_bypassed_bit_for_bit(name, monkeypatch):
    """WITH-cache == WITHOUT-cache, to the last bit."""
    cls = getattr(ts.systems, name)
    kw = {"final_time": 5.0, "dt": 0.01, "backend": "jit", "ic": [0.5, 0.5, 0.5]}
    cached = cls().integrate(**kw).y

    monkeypatch.setenv(_NO_JIT_CACHE, "1")
    runmod.clear_jit_cache()
    bypassed = cls().integrate(**kw).y
    # Nothing was stored while the bypass was on.
    assert runmod.jit_cache_stats()["size"] == 0

    assert cached.tobytes() == bypassed.tobytes()


def test_repeat_hits_stay_bit_identical():
    """A hit must reproduce the first compile exactly, not merely closely."""
    lor = ts.systems.Lorenz()
    kw = {"final_time": 5.0, "dt": 0.01, "backend": "jit", "ic": [0.5, 0.5, 0.5]}
    first = lor.integrate(**kw).y
    second = ts.systems.Lorenz(params={"rho": 28.0}).integrate(**kw).y
    assert runmod.jit_cache_stats()["hits"] == 1
    assert first.tobytes() == second.tobytes()


def test_interp_equals_jit_bit_for_bit_through_the_cache():
    """The documented ``interp == jit`` contract, re-checked on a cache hit."""
    lor = ts.systems.Lorenz()
    kw = {"final_time": 5.0, "dt": 0.01, "ic": [0.5, 0.5, 0.5]}
    interp = lor.integrate(**kw, backend="interp").y
    lor.integrate(**kw, backend="jit")  # miss (compiles)
    jit_hit = lor.integrate(**kw, backend="jit").y  # hit (cached)
    assert runmod.jit_cache_stats()["hits"] == 1
    assert interp.tobytes() == jit_hit.tobytes()
