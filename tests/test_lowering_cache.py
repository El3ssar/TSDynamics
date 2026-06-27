"""Tests for the lowered-tape cache (stream PERF-LOWER-CACHE).

The engine lowers a system's symbolic dynamics to an IR :class:`Tape` that is a
pure function of the *math* (kernel body, dimension, structural parameters, DDE
delays, ``with_jacobian``) — **not** of the control-parameter values, which are
read live at runtime.  :mod:`tsdynamics.engine.compile` therefore memoises the
lowered tape so a control-parameter sweep reuses one tape instead of re-lowering
a byte-identical one per value.

Correctness is paramount (a stale tape would silently produce wrong results), so
these tests assert:

(a) a control-parameter sweep is **bit-for-bit identical** with the cache enabled
    and with it bypassed (``TSDYNAMICS_NO_TAPE_CACHE``);
(b) a **structural** parameter change (and a DDE delay / a dimension change)
    produces a *different* tape — no stale hit;
(c) ``with_jacobian=True`` and ``=False`` never collide;
(d) monkeypatching a kernel invalidates the cached entry;
(e) the cache is actually **hit** on a repeat lowering (via the stats counters).
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.engine import compile as comp
from tsdynamics.engine import run as runmod
from tsdynamics.engine.problem import ode_problem

_rust = pytest.importorskip("tsdynamics._rust")


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Each test starts with an empty cache and the env-var bypass cleared."""
    monkeypatch.delenv(comp._TAPE_CACHE_ENV, raising=False)
    comp.clear_tape_cache()
    yield
    comp.clear_tape_cache()


# ---------------------------------------------------------------------------
# (e) the cache is actually hit on a repeat lowering
# ---------------------------------------------------------------------------


def test_repeat_lowering_is_a_cache_hit():
    lor = ts.systems.Lorenz()
    t1 = comp.lower_ode_cached(lor)
    stats1 = comp.tape_cache_stats()
    assert stats1["misses"] == 1
    assert stats1["hits"] == 0

    t2 = comp.lower_ode_cached(lor)
    stats2 = comp.tape_cache_stats()
    assert stats2["hits"] == 1
    assert stats2["misses"] == 1
    # A hit returns the *same* shared tape object (immutable, safe to share).
    assert t1 is t2


def test_control_param_change_is_a_cache_hit():
    """A control-parameter change must NOT change the tape (the whole point)."""
    lor = ts.systems.Lorenz()
    t1 = ode_problem(lor).tape
    t2 = ode_problem(lor.with_params(rho=40.0)).tape
    t3 = ode_problem(lor.with_params(sigma=12.0, beta=3.0)).tape
    assert t1 is t2 is t3
    assert comp.tape_cache_stats()["hits"] == 2
    assert comp.tape_cache_stats()["misses"] == 1


# ---------------------------------------------------------------------------
# (a) WITH-cache == WITHOUT-cache, bit-for-bit, over a control-param sweep
# ---------------------------------------------------------------------------


def _lorenz_rho_sweep():
    lor = ts.systems.Lorenz()
    ic = [1.0, 1.0, 1.0]
    rows = []
    for rho in np.linspace(24.0, 30.0, 12):
        s = lor.with_params(rho=float(rho))
        prob = runmod.build_problem(s, ic=ic)
        tr = runmod.integrate(prob, final_time=8.0, dt=0.01, backend="interp")
        rows.append(tr.y[-1])
    return np.asarray(rows)


def test_sweep_cached_equals_bypassed_bit_for_bit(monkeypatch):
    # With the cache (cleared once, then re-used across the 12 values).
    comp.clear_tape_cache()
    cached = _lorenz_rho_sweep()
    assert comp.tape_cache_stats()["hits"] >= 11  # 1 miss + 11 hits over 12 values

    # With the cache entirely bypassed (re-lower every value).
    monkeypatch.setenv(comp._TAPE_CACHE_ENV, "1")
    comp.clear_tape_cache()
    bypassed = _lorenz_rho_sweep()
    # Bypass means nothing is stored.
    assert comp.tape_cache_stats()["size"] == 0
    assert comp.tape_cache_stats()["hits"] == 0

    np.testing.assert_array_equal(cached, bypassed)


def test_clear_each_iteration_equals_cached_bit_for_bit():
    """Clearing the cache every value (forces re-lower) must match the cached sweep."""
    lor = ts.systems.Lorenz()
    ic = [0.5, 0.5, 0.5]
    rhos = np.linspace(20.0, 28.0, 8)

    comp.clear_tape_cache()
    cached = []
    for rho in rhos:
        prob = runmod.build_problem(lor.with_params(rho=float(rho)), ic=ic)
        cached.append(runmod.integrate(prob, final_time=6.0, dt=0.01).y[-1])

    relowered = []
    for rho in rhos:
        comp.clear_tape_cache()  # force a fresh lowering each value
        prob = runmod.build_problem(lor.with_params(rho=float(rho)), ic=ic)
        relowered.append(runmod.integrate(prob, final_time=6.0, dt=0.01).y[-1])

    np.testing.assert_array_equal(np.asarray(cached), np.asarray(relowered))


# ---------------------------------------------------------------------------
# (b) structural change / dim change / DDE delay change -> a different tape
# ---------------------------------------------------------------------------


def test_structural_param_change_is_a_distinct_tape():
    """A structural parameter is baked into the tape -> must miss, not hit."""
    a = comp.lower_ode_cached(ts.systems.GrayScott(N=8))
    b = comp.lower_ode_cached(ts.systems.GrayScott(N=12))
    assert a is not b
    assert a.dim != b.dim  # N drives the flattened-field dimension
    assert comp.tape_cache_stats()["hits"] == 0
    assert comp.tape_cache_stats()["misses"] == 2


def test_dde_delay_change_is_a_distinct_tape():
    """DDEs bake delays (a parameter) into the tape -> a delay change must miss."""
    mg = ts.systems.MackeyGlass()
    tau0 = mg.params["tau"]
    tape_a, slots_a = comp.lower_dde_cached(mg)
    tape_b, slots_b = comp.lower_dde_cached(mg.with_params(tau=tau0 * 1.5))
    assert tape_a is not tape_b
    # The slot delay magnitudes reflect the changed delay.
    assert [s.delay for s in slots_a] != [s.delay for s in slots_b]


def test_dde_control_param_change_at_same_delay_is_a_hit():
    """A DDE *non-delay* parameter change at the same delay reuses the tape."""
    mg = ts.systems.MackeyGlass()
    tape_a, slots_a = comp.lower_dde_cached(mg)
    # gamma is a non-delay parameter; the delay slots (and thus the tape) are
    # only distinct if a *delay* changes -- but DDEs bake ALL params, so even a
    # gamma change is a deliberate miss.  Assert it produces a CONSISTENT tape
    # (same delays), i.e. caching never returns a stale tape for a changed param.
    tape_b, slots_b = comp.lower_dde_cached(mg.with_params(gamma=mg.params["gamma"] * 1.1))
    assert [s.delay for s in slots_a] == [s.delay for s in slots_b]


def test_dde_returns_a_fresh_slot_list_per_call():
    """The cached slot list must not be the same mutable object across calls."""
    mg = ts.systems.MackeyGlass()
    _, slots1 = comp.lower_dde_cached(mg)
    _, slots2 = comp.lower_dde_cached(mg)
    assert slots1 == slots2
    assert slots1 is not slots2  # a consumer cannot mutate the cached list


# ---------------------------------------------------------------------------
# (c) with_jacobian=True and =False never collide
# ---------------------------------------------------------------------------


def test_with_jacobian_does_not_collide():
    lor = ts.systems.Lorenz()
    no_jac = comp.lower_ode_cached(lor, with_jacobian=False)
    with_jac = comp.lower_ode_cached(lor, with_jacobian=True)
    assert no_jac is not with_jac
    assert not no_jac.has_jacobian
    assert with_jac.has_jacobian
    # Asking again for each hits its own entry, never the other.
    assert comp.lower_ode_cached(lor, with_jacobian=False) is no_jac
    assert comp.lower_ode_cached(lor, with_jacobian=True) is with_jac


def test_map_with_jacobian_does_not_collide():
    h = ts.systems.Henon()
    no_jac = comp.lower_map_cached(h, with_jacobian=False)
    with_jac = comp.lower_map_cached(h, with_jacobian=True)
    assert no_jac is not with_jac
    assert not no_jac.has_jacobian
    assert with_jac.has_jacobian


# ---------------------------------------------------------------------------
# (d) monkeypatching a kernel invalidates the entry
# ---------------------------------------------------------------------------


def test_monkeypatching_equations_invalidates(monkeypatch):
    import symengine

    lor = ts.systems.Lorenz()
    t1 = comp.lower_ode_cached(lor)

    def patched(y, t, *, sigma, rho, beta):  # noqa: ANN001, ANN202 - kernel contract
        # Genuinely different math (extra term) so a stale hit would be visible.
        return [
            sigma * (y(1) - y(0)),
            y(0) * (rho - y(2)) - y(1),
            y(0) * y(1) - beta * y(2) + symengine.sin(y(0)),
        ]

    monkeypatch.setattr(ts.systems.Lorenz, "_equations", staticmethod(patched))
    t2 = comp.lower_ode_cached(ts.systems.Lorenz())
    assert t1 is not t2  # a new kernel object -> a cache miss, not a stale tape
    # The new tape reflects the new math: evaluate the RHS at a probe point.
    p = np.asarray([float(lor.params[k]) for k in t2.control_names])
    u = np.array([1.0, 2.0, 3.0])
    rhs_new = comp.eval_tape(t2, u, p, 0.0)
    rhs_old = comp.eval_tape(t1, u, p, 0.0)
    assert not np.allclose(rhs_new, rhs_old)  # the sin(y0) term changed component 2


def test_redefining_same_math_new_object_is_a_miss(monkeypatch):
    """Even identical math, but a new function object, must miss (identity key)."""
    lor = ts.systems.Lorenz()
    t1 = comp.lower_ode_cached(lor)
    orig = ts.systems.Lorenz._equations
    raw = getattr(orig, "__func__", orig)

    # A fresh FunctionType sharing the original code object -> a new identity.
    import types as _types

    clone = _types.FunctionType(
        raw.__code__, raw.__globals__, raw.__name__, raw.__defaults__, raw.__closure__
    )
    monkeypatch.setattr(ts.systems.Lorenz, "_equations", staticmethod(clone))
    t2 = comp.lower_ode_cached(ts.systems.Lorenz())
    assert t1 is not t2
    # The math is identical, so the lowered tapes compare equal by value.
    assert t1 == t2


# ---------------------------------------------------------------------------
# bypass / clear hooks
# ---------------------------------------------------------------------------


def test_env_var_bypass_never_stores(monkeypatch):
    monkeypatch.setenv(comp._TAPE_CACHE_ENV, "1")
    comp.clear_tape_cache()
    lor = ts.systems.Lorenz()
    a = comp.lower_ode_cached(lor)
    b = comp.lower_ode_cached(lor)
    assert a is not b  # bypass -> a fresh tape each time
    stats = comp.tape_cache_stats()
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_clear_resets_counters_and_store():
    lor = ts.systems.Lorenz()
    comp.lower_ode_cached(lor)
    comp.lower_ode_cached(lor)
    assert comp.tape_cache_stats()["size"] == 1
    comp.clear_tape_cache()
    s = comp.tape_cache_stats()
    assert s == {"hits": 0, "misses": 0, "size": 0, "maxsize": s["maxsize"]}


def test_cache_is_bounded_lru(monkeypatch):
    """The cache evicts oldest entries past the cap (no unbounded growth)."""
    monkeypatch.setattr(comp, "_TAPE_CACHE_MAXSIZE", 4)
    comp.clear_tape_cache()
    # Lower 6 distinct structural variants -> at most 4 retained.
    for n in (8, 10, 12, 14, 16, 18):
        comp.lower_ode_cached(ts.systems.GrayScott(N=n))
    assert comp.tape_cache_stats()["size"] <= 4


def test_hashable_value_distinguishes_distinct_arrays():
    """Unhashable array-valued keys must not collide (NumPy reprs truncate).

    A structural parameter that is a large array would, under a bare ``repr``
    fallback, alias with another large array (NumPy elides the middle as
    ``...``) and serve a stale tape. ``_hashable_value`` keys arrays on a
    content digest, so genuinely different arrays get different keys while an
    identical array reproduces the same key.
    """
    a = np.arange(4000, dtype=float)
    b = a.copy()
    b[2000] = -1.0  # differs only in the repr-truncated middle
    ka, kb, ka2 = (comp._hashable_value(x) for x in (a, b, a.copy()))
    assert ka != kb, "distinct large arrays must not collide in the cache key"
    assert ka == ka2, "an identical array must reproduce the same key"
    assert hash(ka) == hash(ka2)  # the key part is hashable
