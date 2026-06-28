"""Regression tests for focused family-base fixes (deep-audit follow-ups).

Covers three fixes in :mod:`tsdynamics.families.continuous` /
:mod:`tsdynamics.families.discrete`:

1. ``ContinuousSystem.lyapunov_spectrum`` now takes a ``backend=`` keyword and
   forwards it to :class:`~tsdynamics.derived.tangent.TangentSystem` instead of
   hard-coding ``"interp"`` (the ODE variational path is backend-neutral).
2. The per-class ``_lambdified`` numeric-evaluator cache is a bounded LRU, so a
   long-lived process that lowers many distinct structural variants of one
   system cannot grow it without bound.
3. ``DiscreteMap.iterate``'s divergence-retry loop catches only divergence
   (:class:`~tsdynamics.errors.ConvergenceError` / arithmetic blow-ups), so a
   missing-engine :class:`~tsdynamics.engine.run.EngineNotAvailableError` (a
   ``RuntimeError`` subclass that is *not* a ``ConvergenceError``) propagates
   loudly instead of silently burning the whole retry budget.

These run on the dependency-light ``reference`` backend or on pure-Python
fixtures, so none requires the compiled ``tsdynamics._rust`` extension.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, ClassVar

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.engine.run import EngineNotAvailableError
from tsdynamics.errors import ConvergenceError, InvalidParameterError

# ---------------------------------------------------------------------------
# Fix 1 — lyapunov_spectrum forwards backend=
# ---------------------------------------------------------------------------


def test_lyapunov_spectrum_accepts_reference_backend() -> None:
    """``backend="reference"`` is accepted and returns a real spectrum.

    This is the wheel-free path: before the fix the method hard-coded
    ``backend="interp"`` and ``backend=`` was not even a keyword, so this call
    raised ``TypeError``.  The reference oracle needs no compiled engine, so the
    test runs anywhere.
    """
    lor = ts.Lorenz()
    exps = lor.lyapunov_spectrum(
        final_time=12.0,
        dt=0.1,
        burn_in=3.0,
        backend="reference",
    )
    exps = np.asarray(exps, dtype=float)
    assert exps.shape == (lor.dim,)
    assert np.all(np.isfinite(exps))
    # The strongly-contracting direction and the dissipative trace converge fast and
    # are reliable even on this short window, while the leading (expanding) direction
    # is not strongly negative.  Full descending/literature convergence needs a long
    # run; here we only assert that the reference path produces a sane Lorenz spectrum.
    assert exps.min() < -5.0
    assert exps.sum() < 0.0
    assert exps.max() > -1.0


def test_lyapunov_spectrum_rejects_unknown_backend() -> None:
    """An unknown ODE tangent backend is rejected (forwarding is real)."""
    with pytest.raises(ValueError):
        ts.Lorenz().lyapunov_spectrum(final_time=5.0, dt=0.1, backend="gpu")


def test_lyapunov_spectrum_rejects_nonpositive_n_exp() -> None:
    """``n_exp`` must be a positive integer (unchanged contract, guard intact)."""
    with pytest.raises(InvalidParameterError):
        ts.Lorenz().lyapunov_spectrum(n_exp=0)


# ---------------------------------------------------------------------------
# Fix 2 — the _lambdified cache is a bounded LRU
# ---------------------------------------------------------------------------


class _CacheProbe(ts.ContinuousSystem):
    """A trivial structural-param ODE with its OWN small LRU cache.

    Overriding ``_lambdified`` (a fresh ``OrderedDict``) and the cap keeps the
    test from touching the shared ``ContinuousSystem`` cache.  Each distinct
    ``N`` is a distinct cache key (``N`` is structural), so lowering a range of
    ``N`` drives the LRU eviction.
    """

    params: ClassVar[dict[str, Any]] = {"N": 3, "a": 1.0}
    dim = 3
    _structural_params = frozenset({"N"})
    _lambdified: ClassVar[OrderedDict[str, tuple[Any, Any, list[str]]]] = OrderedDict()
    _LAMBDIFIED_CACHE_MAXSIZE: ClassVar[int] = 4

    @staticmethod
    def _equations(y: Any, t: Any, N: int, a: float) -> list[Any]:  # noqa: N803
        return [a * y((i + 1) % N) - y(i) for i in range(N)]


def _variant(n: int) -> _CacheProbe:
    """A ``_CacheProbe`` with structural dimension ``N = n`` (a fresh cache key)."""
    return _CacheProbe(params={"N": n}, dim=n)


def test_lambdified_cache_is_bounded_lru() -> None:
    """Lowering more structural variants than the cap never overflows the cache."""
    _CacheProbe._lambdified.clear()
    cap = _CacheProbe._LAMBDIFIED_CACHE_MAXSIZE

    for n in range(2, 2 + cap + 6):  # well past the cap
        _variant(n)._build_lambdified()
        assert len(_CacheProbe._lambdified) <= cap

    # The cache holds exactly the cap-many most-recent keys after the sweep.
    assert len(_CacheProbe._lambdified) == cap


def test_lambdified_cache_hit_is_moved_to_mru() -> None:
    """A cache hit is moved to the most-recently-used end (true LRU, not FIFO)."""
    _CacheProbe._lambdified.clear()
    cap = _CacheProbe._LAMBDIFIED_CACHE_MAXSIZE

    # Fill the cache exactly to the cap with variants N = 2 .. 2+cap-1.
    first = _variant(2)
    first_key = first._cache_key()
    first._build_lambdified()
    for n in range(3, 2 + cap):
        _variant(n)._build_lambdified()
    assert first_key in _CacheProbe._lambdified

    # Touch the oldest entry (N=2) so it becomes most-recently-used.
    first._build_lambdified()

    # Inserting one more distinct variant must now evict the NEW oldest entry
    # (N=3), not the just-touched N=2.
    _variant(2 + cap)._build_lambdified()
    assert first_key in _CacheProbe._lambdified  # survived because it was touched
    assert _variant(3)._cache_key() not in _CacheProbe._lambdified


# ---------------------------------------------------------------------------
# Fix 3 — iterate()'s retry loop catches divergence only
# ---------------------------------------------------------------------------


class _RetryMap(ts.DiscreteMap):
    """A 1-D map whose ``_iterate_engine`` is monkeypatched in each test."""

    params: ClassVar[dict[str, Any]] = {"a": 1.0}
    dim = 1

    @staticmethod
    def _step(X: np.ndarray, a: float) -> Any:  # noqa: N803
        return a * X

    @staticmethod
    def _jacobian(X: np.ndarray, a: float) -> Any:  # noqa: N803
        return np.array([[a]])


def test_iterate_propagates_engine_not_available(monkeypatch: Any) -> None:
    """A missing-engine error propagates immediately — it is NOT divergence.

    Before the fix the retry loop caught the broad ``RuntimeError``, so an
    :class:`EngineNotAvailableError` (a ``RuntimeError`` subclass) was mistaken
    for divergence: the loop swallowed it, retried ``max_retries`` times, and
    finally raised a misleading ``ConvergenceError``.  Now it surfaces the real
    fault on the first attempt.
    """
    calls = {"n": 0}

    def _boom(**kwargs: Any) -> Any:
        calls["n"] += 1
        raise EngineNotAvailableError("the compiled tsdynamics._rust is not built")

    m = _RetryMap()
    monkeypatch.setattr(_RetryMap, "_iterate_engine", staticmethod(_boom))

    with pytest.raises(EngineNotAvailableError):
        m.iterate(steps=10, max_retries=5)
    # Surfaced on the FIRST attempt — no retry budget burned.
    assert calls["n"] == 1


def test_iterate_still_retries_on_divergence(monkeypatch: Any) -> None:
    """A genuine divergence (random IC) still triggers the random-IC retry."""
    calls = {"n": 0}

    def _diverge_then_ok(self: Any, *, steps: int, ic: Any, backend: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConvergenceError("map diverged at iteration 3")
        # Second attempt: hand back a finite trajectory.
        t = np.arange(steps)
        y = np.zeros((steps, self.dim))
        return ts.Trajectory(t=t, y=y, system=self, meta={})

    monkeypatch.setattr(_RetryMap, "_iterate_engine", _diverge_then_ok)

    m = _RetryMap()
    traj = m.iterate(steps=4, max_retries=5)  # no explicit ic → retry allowed
    assert calls["n"] == 2  # diverged once, then succeeded
    assert np.isfinite(traj.y).all()


def test_iterate_explicit_ic_divergence_raises(monkeypatch: Any) -> None:
    """An explicit ``ic`` that diverges raises immediately (no retry)."""
    calls = {"n": 0}

    def _always_diverge(self: Any, *, steps: int, ic: Any, backend: str) -> Any:
        calls["n"] += 1
        raise ConvergenceError("map diverged at iteration 1")

    monkeypatch.setattr(_RetryMap, "_iterate_engine", _always_diverge)

    m = _RetryMap()
    with pytest.raises(ConvergenceError):
        m.iterate(steps=4, ic=[0.5], max_retries=5)
    assert calls["n"] == 1  # explicit ic → no retry
