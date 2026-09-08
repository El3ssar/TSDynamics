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


# ---------------------------------------------------------------------------
# v6 api-core: the map family contract, IC honesty, and the plot front door
# ---------------------------------------------------------------------------


class TestDiscreteMapIsAbstract:
    """``DiscreteMap`` is an ``abc.ABC`` like every other family base.

    It used to be the only one that was not, so its ``@abstractmethod`` markers
    on ``_step`` / ``_jacobian`` were inert: a subclass missing a kernel
    instantiated happily and failed much later, during lowering, as a
    ``TapeCompileError`` complaining that the step "cannot be traced
    symbolically" — a thoroughly misleading diagnosis of a missing method.
    """

    def test_missing_step_cannot_be_instantiated(self) -> None:
        class _NoStep(ts.DiscreteMap):
            params: ClassVar[dict[str, Any]] = {"a": 1.0}
            dim = 1

            @staticmethod
            def _jacobian(X, a):
                return ((a,),)

        with pytest.raises(TypeError, match="_step"):
            _NoStep()

    def test_missing_jacobian_cannot_be_instantiated(self) -> None:
        class _NoJacobian(ts.DiscreteMap):
            params: ClassVar[dict[str, Any]] = {"a": 1.0}
            dim = 1

            @staticmethod
            def _step(X, a):
                return (a * X[0],)

        with pytest.raises(TypeError, match="_jacobian"):
            _NoJacobian()

    def test_every_catalogue_map_still_instantiates(self) -> None:
        """The contract is met by all built-in maps (the ABC is not a regression)."""
        from tsdynamics import registry

        for entry in registry.all_systems(family="map"):
            entry.cls()


class TestExplicitICIsNeverSwapped:
    """A user-chosen initial condition must not be replaced by a random one.

    ``iterate`` retried from a fresh random IC whenever the orbit diverged and no
    ``ic=`` argument was passed — but an IC set on the **constructor** looks
    exactly like that case, so ``Henon(ic=[1e6, 1e6]).iterate()`` silently
    returned the orbit of a completely different, randomly drawn initial state.
    """

    def test_constructor_ic_that_diverges_raises(self) -> None:
        h = ts.Henon(ic=[1e6, 1e6])
        with pytest.raises(ConvergenceError):
            h.iterate(steps=100)
        # ... and the IC the user set is still there, unswapped.
        np.testing.assert_array_equal(h.ic, [1e6, 1e6])

    def test_argument_ic_that_diverges_raises(self) -> None:
        h = ts.Henon(ic=[0.1, 0.1])
        with pytest.raises(ConvergenceError):
            h.iterate(steps=100, ic=[1e6, 1e6])

    def test_a_good_constructor_ic_is_honoured(self) -> None:
        h = ts.Henon(ic=[0.1, 0.1])
        # The orbit starts from the IC the user gave (``y[0]`` is its first image).
        np.testing.assert_array_equal(h.iterate(steps=50).meta["ic"], [0.1, 0.1])


class TestFailedRunLeavesTheSystemUntouched:
    """A run that raises must not latch its bad IC onto the instance.

    ``resolve_ic`` commits the resolved IC to ``self.ic`` *before* the run, so a
    divergence used to leave the offending state on the object — and every later,
    unrelated analysis silently started from it, returning wrong answers with no
    warning at all.
    """

    def test_ode_integrate_failure_restores_ic(self) -> None:
        lor = ts.Lorenz(ic=[1.0, 1.0, 1.0])
        with pytest.raises(ConvergenceError):
            lor.integrate(final_time=10.0, dt=0.1, ic=[1e300, 1e300, 1e300])
        np.testing.assert_array_equal(lor.ic, [1.0, 1.0, 1.0])
        # The later, unrelated run is unaffected.
        np.testing.assert_array_equal(
            lor.integrate(final_time=0.1, dt=0.1).meta["ic"], [1.0, 1.0, 1.0]
        )

    def test_map_iterate_failure_restores_ic(self) -> None:
        h = ts.Henon(ic=[0.1, 0.1])
        with pytest.raises(ConvergenceError):
            h.iterate(steps=100, ic=[1e6, 1e6])
        np.testing.assert_array_equal(h.ic, [0.1, 0.1])
        np.testing.assert_array_equal(h.iterate(steps=10).meta["ic"], [0.1, 0.1])


class TestSeededIntegration:
    """A run from a random IC is reproducible and leaves the global RNG alone."""

    def test_map_iterate_seed_is_reproducible(self) -> None:
        a = ts.Henon(seed=5).iterate(steps=50)
        b = ts.Henon(seed=5).iterate(steps=50)
        np.testing.assert_array_equal(a.y, b.y)

    def test_map_iterate_seed_keyword(self) -> None:
        a = ts.Henon().iterate(steps=50, seed=5)
        b = ts.Henon().iterate(steps=50, seed=5)
        np.testing.assert_array_equal(a.y, b.y)

    def test_run_does_not_perturb_the_global_rng(self) -> None:
        np.random.seed(0)
        expected = np.random.rand(3)
        np.random.seed(0)
        ts.SprottB().integrate(final_time=0.5, dt=0.1)
        np.testing.assert_array_equal(expected, np.random.rand(3))

    def test_meta_carries_the_seed_needed_to_reproduce_the_run(self) -> None:
        first = ts.SprottB().integrate(final_time=0.5, dt=0.1)
        replay = ts.SprottB(seed=first.meta["ic_seed"]).integrate(final_time=0.5, dt=0.1)
        np.testing.assert_array_equal(first.y, replay.y)


class TestSystemPlotForwardsIntegrationKeywords:
    """``system.plot(final_time=..., dt=...)`` must not be a silent no-op.

    Every keyword that was neither plot-shaping nor an inline tweak used to be
    handed to the renderer, whose ``**kwargs`` swallowed it — so the integration
    keywords were dropped on the floor and a typo was silently accepted.
    """

    def test_integration_keywords_reach_the_integrator(self) -> None:
        spec = ts.Lorenz(ic=[1.0, 1.0, 1.0]).to_plot_spec(final_time=2.0, dt=0.1, components="x")
        assert spec.layers[0].data["x"].shape == (21,)

    def test_plot_honours_integration_keywords(self) -> None:
        pytest.importorskip("matplotlib")
        fig = ts.Lorenz(ic=[1.0, 1.0, 1.0]).plot(final_time=2.0, dt=0.1, components="x")
        assert [len(line.get_xdata()) for line in fig.axes[0].lines] == [21]

    def test_plot_rejects_an_unknown_keyword(self) -> None:
        pytest.importorskip("matplotlib")
        with pytest.raises(InvalidParameterError, match="finaltime"):
            ts.Lorenz(ic=[1.0, 1.0, 1.0]).plot(finaltime=2.0)

    def test_plot_still_accepts_tweaks_and_renderer_options(self) -> None:
        pytest.importorskip("matplotlib")
        fig = ts.Lorenz(ic=[1.0, 1.0, 1.0]).plot(
            final_time=2.0, dt=0.1, components="x", title="T", figsize=(4.0, 3.0)
        )
        assert fig.axes[0].get_title() == "T"
        assert tuple(fig.get_size_inches()) == (4.0, 3.0)

    def test_backend_kwargs_escape_hatch(self) -> None:
        pytest.importorskip("matplotlib")
        fig = ts.Lorenz(ic=[1.0, 1.0, 1.0]).plot(
            final_time=1.0, dt=0.1, components="x", backend_kwargs={"figsize": (5.0, 2.0)}
        )
        assert tuple(fig.get_size_inches()) == (5.0, 2.0)


# ---------------------------------------------------------------------------
# v6 api-core, adversarial follow-up: an internally *drawn* IC must not be
# promoted to a user choice.
#
# ``resolve_ic(ic)`` marks ``self.ic`` as user-chosen, and the engine problem
# builders (``ode_problem`` / ``map_problem``) hand the array ``resolve_ic`` just
# returned straight back into ``resolve_ic``.  That round-trip flipped the flag
# on every run, with two visible consequences on a map:
#
#   * the random-IC retry was silently dead from the SECOND ``iterate()`` call
#     onwards (at HEAD it was alive on every call), and
#   * the divergence diagnostic asserted the initial condition "was supplied
#     explicitly" about an IC the library itself had drawn at random.
# ---------------------------------------------------------------------------


class _Divergent(ts.DiscreteMap):
    """``x -> 4 a x``: diverges from any non-zero state once ``a`` reaches 1."""

    params: ClassVar[dict[str, Any]] = {"a": 1.0}
    dim = 1

    @staticmethod
    def _step(X, a):  # type: ignore[no-untyped-def]
        return (4.0 * a * X[0],)

    @staticmethod
    def _jacobian(X, a):  # type: ignore[no-untyped-def]
        return ((4.0 * a,),)


def test_a_drawn_ic_stays_non_explicit_across_runs() -> None:
    """A run must not promote its own random draw to "user-chosen"."""
    m = _Divergent(params={"a": 0.05}, seed=3)
    m.iterate(steps=20)
    assert m._ic_explicit is False


def test_the_random_ic_retry_survives_the_first_run() -> None:
    """The second ``iterate`` still retries from a fresh random IC."""
    m = _Divergent(params={"a": 0.05}, seed=3)
    m.iterate(steps=20)
    m.a = 1.0  # now divergent from the drawn IC
    with (
        pytest.warns(RuntimeWarning, match="diverged") as rec,
        pytest.raises(ConvergenceError) as exc,
    ):
        m.iterate(steps=2000, max_retries=3)
    # Two fresh random ICs were tried before the final attempt raised, i.e. the
    # retry loop is alive on a second run (it was dead: ``ic_explicit`` was True).
    assert len(rec) == 2
    # ... and the diagnostic does NOT claim the user supplied the IC.
    assert not any("supplied explicitly" in n for n in getattr(exc.value, "__notes__", []))


def test_an_explicit_ic_stays_explicit_across_runs() -> None:
    """The genuine user choice is still recorded (and still never swapped)."""
    h = ts.Henon(ic=[0.1, 0.1])
    h.iterate(steps=10)
    assert h._ic_explicit is True


def test_system_plot_accepts_the_in_tree_renderer_keywords() -> None:
    """``system.plot`` takes the same backend options as ``Trajectory.plot``.

    The routing sends every unrecognised keyword to the integration, so a
    renderer option that was not in the allow-list used to raise on a system
    while working on a trajectory.
    """
    pytest.importorskip("plotly")
    lor = ts.Lorenz(ic=[1.0, 1.0, 1.0])
    traj_fig = lor.integrate(final_time=1.0, dt=0.05).plot(backend="plotly", html=True)
    sys_fig = lor.plot(backend="plotly", html=True, final_time=1.0, dt=0.05)
    assert type(sys_fig) is type(traj_fig)
