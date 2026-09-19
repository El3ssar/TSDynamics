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
   missing-engine :class:`~tsdynamics._engine.run.EngineNotAvailableError` (a
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
from tsdynamics._engine.run import EngineNotAvailableError
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
    lor = ts.systems.Lorenz()
    exps = lor._lyapunov_spectrum(
        final_time=12.0,
        dt=0.1,
        transient=3.0,
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
        ts.systems.Lorenz()._lyapunov_spectrum(final_time=5.0, dt=0.1, backend="gpu")


def test_lyapunov_spectrum_rejects_nonpositive_n_exp() -> None:
    """``n_exp`` must be a positive integer (unchanged contract, guard intact)."""
    with pytest.raises(InvalidParameterError):
        ts.systems.Lorenz()._lyapunov_spectrum(k=0)


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
        m.run(steps=10, max_retries=5)
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
    traj = m.run(steps=4, max_retries=5)  # no explicit ic → retry allowed
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
        m.run(steps=4, ic=[0.5], max_retries=5)
    assert calls["n"] == 1  # explicit ic → no retry


# ---------------------------------------------------------------------------
# v6 api-core: the map family contract, IC honesty, and the plot front door
# ---------------------------------------------------------------------------


class TestDiscreteMapIsAbstract:
    """``DiscreteMap`` is an ``abc.ABC`` like every other family base.

    It used to be the only one that was not, so its ``@abstractmethod`` marker on
    ``_step`` was inert: a subclass missing the kernel instantiated happily and
    failed much later, during lowering, as a ``TapeCompileError`` complaining
    that the step "cannot be traced symbolically" — a thoroughly misleading
    diagnosis of a missing method.

    ``_jacobian`` is deliberately **not** abstract (see
    :class:`TestMapJacobianIsAutogenerated`): a map's Jacobian is the derivative
    of its own ``_step``, so the library derives it, exactly as
    ``ContinuousSystem`` derives one from ``_equations``.
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

    def test_every_catalogue_map_still_instantiates(self) -> None:
        """The contract is met by all built-in maps (the ABC is not a regression)."""
        from tsdynamics import registry

        for entry in registry.all_systems(family="map"):
            entry.cls()


class TestMapJacobianIsAutogenerated:
    """Defining a map means writing ``_step`` — nothing else.

    A map's Jacobian is the symbolic derivative of its own ``_step``, which the
    lowering pass already computes for the engine tape (``lower_map`` documents
    the traced derivative as "the single source of truth").  Requiring the user
    to *also* transcribe it by hand was therefore asking for work the library
    does anyway, on the exact surface the audit found transcription bugs in.
    ``ContinuousSystem`` has autogenerated its Jacobian from ``_equations``
    since v1; this makes the two families answer the same way.
    """

    @staticmethod
    def _user_henon() -> Any:
        class _UserHenon(ts.DiscreteMap):
            params: ClassVar[dict[str, Any]] = {"a": 1.4, "b": 0.3}
            dim = 2

            @staticmethod
            def _step(u, a, b):
                x, y = u
                return [1 - a * x**2 + y, b * x]

        return _UserHenon

    def test_a_map_with_no_jacobian_instantiates_and_runs(self) -> None:
        m = self._user_henon()()
        traj = m.run(500, ic=[0.1, 0.1], transient=100)
        assert traj.y.shape == (501, 2)  # the start, then 500 iterates
        assert np.all(np.isfinite(traj.y))

    def test_the_autogenerated_jacobian_equals_the_hand_written_one(self) -> None:
        """Against the catalogue Hénon, whose ``_jacobian`` is hand-written."""
        cls = self._user_henon()
        for state in ([0.1, 0.2], [-0.4, 0.31], [0.0, 0.0]):
            auto = np.asarray(cls._jacobian(state, 1.4, 0.3), dtype=float)
            hand = np.asarray(ts.systems.Henon._jacobian(np.asarray(state), 1.4, 0.3), dtype=float)
            assert np.allclose(auto, hand, rtol=0, atol=1e-14)

    def test_lyapunov_agrees_with_the_hand_written_catalogue_map(self) -> None:
        auto = self._user_henon()()._lyapunov_spectrum(n=20_000, ic=[0.1, 0.1])
        hand = ts.systems.Henon()._lyapunov_spectrum(n=20_000, ic=[0.1, 0.1])
        assert np.allclose(np.asarray(auto), np.asarray(hand), rtol=1e-9, atol=1e-9)

    def test_a_hand_written_jacobian_still_wins(self) -> None:
        """An override is honoured verbatim — the autogen never shadows it."""

        class _Overridden(ts.DiscreteMap):
            params: ClassVar[dict[str, Any]] = {"a": 1.4, "b": 0.3}
            dim = 2
            _jacobian_fd_check = False  # deliberately wrong, to prove it is used

            @staticmethod
            def _step(u, a, b):
                x, y = u
                return [1 - a * x**2 + y, b * x]

            @staticmethod
            def _jacobian(u, a, b):
                return ((42.0, 0.0), (0.0, 42.0))

        assert np.asarray(_Overridden._jacobian([0.1, 0.2], 1.4, 0.3))[0][0] == 42.0

    def test_an_untraceable_step_says_to_write_the_jacobian(self) -> None:
        """A map that branches on the state cannot be differentiated — say so."""
        from tsdynamics._engine.compile import TapeCompileError

        class _Branchy(ts.DiscreteMap):
            params: ClassVar[dict[str, Any]] = {"a": 0.5}
            dim = 1

            @staticmethod
            def _step(u, a):
                if u[0] > 0:  # a Python branch on the state: untraceable
                    return [a * u[0]]
                return [-a * u[0]]

        with pytest.raises(TapeCompileError) as excinfo:
            _Branchy._jacobian([0.3], 0.5)
        message = str(excinfo.value)
        assert "_Branchy" in message, "the error must name the user's class"
        assert "`_jacobian`" in message, "…and name the way out"

    def test_the_a_e_derivative_nodes_resolve(self) -> None:
        """``np.abs`` in a step leaves an unevaluated ``Derivative`` — resolve it."""

        class _Tent(ts.DiscreteMap):
            params: ClassVar[dict[str, Any]] = {"mu": 1.0}
            dim = 1

            @staticmethod
            def _step(u, mu):
                return [mu * (1.0 - np.abs(2.0 * u[0] - 1.0))]

        # d/du [mu(1 - |2u-1|)] = -2 mu sign(2u-1)
        assert float(np.asarray(_Tent._jacobian([0.3], 1.0))[0][0]) == pytest.approx(2.0)
        assert float(np.asarray(_Tent._jacobian([0.7], 1.0))[0][0]) == pytest.approx(-2.0)


class TestExplicitICIsNeverSwapped:
    """A user-chosen initial condition must not be replaced by a random one.

    ``iterate`` retried from a fresh random IC whenever the orbit diverged and no
    ``ic=`` argument was passed — but an IC set on the **constructor** looks
    exactly like that case, so ``Henon(ic=[1e6, 1e6]).run()`` silently
    returned the orbit of a completely different, randomly drawn initial state.
    """

    def test_constructor_ic_that_diverges_raises(self) -> None:
        h = ts.systems.Henon(ic=[1e6, 1e6])
        with pytest.raises(ConvergenceError):
            h.run(steps=100)
        # ... and the IC the user set is still there, unswapped.
        np.testing.assert_array_equal(h.ic, [1e6, 1e6])

    def test_argument_ic_that_diverges_raises(self) -> None:
        h = ts.systems.Henon(ic=[0.1, 0.1])
        with pytest.raises(ConvergenceError):
            h.run(steps=100, ic=[1e6, 1e6])

    def test_a_good_constructor_ic_is_honoured(self) -> None:
        h = ts.systems.Henon(ic=[0.1, 0.1])
        # The orbit starts from the IC the user gave (``y[0]`` is its first image).
        np.testing.assert_array_equal(h.run(steps=50).meta["ic"], [0.1, 0.1])


class TestFailedRunLeavesTheSystemUntouched:
    """A run that raises must not latch its bad IC onto the instance.

    ``resolve_ic`` commits the resolved IC to ``self.ic`` *before* the run, so a
    divergence used to leave the offending state on the object — and every later,
    unrelated analysis silently started from it, returning wrong answers with no
    warning at all.
    """

    def test_ode_integrate_failure_restores_ic(self) -> None:
        lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
        with pytest.raises(ConvergenceError):
            lor.run(final_time=10.0, dt=0.1, ic=[1e300, 1e300, 1e300])
        np.testing.assert_array_equal(lor.ic, [1.0, 1.0, 1.0])
        # The later, unrelated run is unaffected.
        np.testing.assert_array_equal(lor.run(final_time=0.1, dt=0.1).meta["ic"], [1.0, 1.0, 1.0])

    def test_map_iterate_failure_restores_ic(self) -> None:
        h = ts.systems.Henon(ic=[0.1, 0.1])
        with pytest.raises(ConvergenceError):
            h.run(steps=100, ic=[1e6, 1e6])
        np.testing.assert_array_equal(h.ic, [0.1, 0.1])
        np.testing.assert_array_equal(h.run(steps=10).meta["ic"], [0.1, 0.1])


class TestSeededIntegration:
    """A run from a random IC is reproducible and leaves the global RNG alone."""

    def test_map_iterate_seed_is_reproducible(self) -> None:
        a = ts.systems.Henon(seed=5).run(steps=50)
        b = ts.systems.Henon(seed=5).run(steps=50)
        np.testing.assert_array_equal(a.y, b.y)

    def test_map_iterate_seed_keyword(self) -> None:
        a = ts.systems.Henon().run(steps=50, seed=5)
        b = ts.systems.Henon().run(steps=50, seed=5)
        np.testing.assert_array_equal(a.y, b.y)

    def test_run_does_not_perturb_the_global_rng(self) -> None:
        np.random.seed(0)
        expected = np.random.rand(3)
        np.random.seed(0)
        ts.systems.SprottB().run(final_time=0.5, dt=0.1)
        np.testing.assert_array_equal(expected, np.random.rand(3))

    def test_meta_carries_the_seed_needed_to_reproduce_the_run(self) -> None:
        first = ts.systems.SprottB().run(final_time=0.5, dt=0.1)
        replay = ts.systems.SprottB(seed=first.meta["ic_seed"]).run(final_time=0.5, dt=0.1)
        np.testing.assert_array_equal(first.y, replay.y)


class TestSystemPlotForwardsIntegrationKeywords:
    """``system.plot(final_time=..., dt=...)`` must not be a silent no-op.

    Every keyword that was neither plot-shaping nor an inline tweak used to be
    handed to the renderer, whose ``**kwargs`` swallowed it — so the integration
    keywords were dropped on the floor and a typo was silently accepted.
    """

    def test_integration_keywords_reach_the_integrator(self) -> None:
        spec = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).__plot_spec__(
            final_time=2.0, dt=0.1, components="x"
        )
        assert spec.layers[0].data["x"].shape == (21,)

    def test_plot_honours_integration_keywords(self) -> None:
        pytest.importorskip("matplotlib")
        # ``.plot()`` returns the PlotSpec since v6; ``.render()`` draws it.
        fig = (
            ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
            .plot(final_time=2.0, dt=0.1, components="x")
            .render()
        )
        assert [len(line.get_xdata()) for line in fig.axes[0].lines] == [21]

    def test_plot_rejects_an_unknown_keyword(self) -> None:
        pytest.importorskip("matplotlib")
        with pytest.raises(InvalidParameterError, match="finaltime"):
            ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).plot(finaltime=2.0)

    def test_plot_still_accepts_tweaks_and_renderer_options(self) -> None:
        pytest.importorskip("matplotlib")
        fig = (
            ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
            .plot(final_time=2.0, dt=0.1, components="x", title="T")
            .render(figsize=(4.0, 3.0))
        )
        assert fig.axes[0].get_title() == "T"
        assert tuple(fig.get_size_inches()) == (4.0, 3.0)

    def test_backend_kwargs_escape_hatch(self) -> None:
        pytest.importorskip("matplotlib")
        fig = (
            ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
            .plot(final_time=1.0, dt=0.1, components="x")
            .render(figsize=(5.0, 2.0))
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
    m.run(steps=20)
    assert m._ic_explicit is False


def test_the_random_ic_retry_survives_the_first_run() -> None:
    """The second ``iterate`` still retries from a fresh random IC."""
    m = _Divergent(params={"a": 0.05}, seed=3)
    m.run(steps=20)
    m.a = 1.0  # now divergent from the drawn IC
    with (
        pytest.warns(RuntimeWarning, match="diverged") as rec,
        pytest.raises(ConvergenceError) as exc,
    ):
        m.run(steps=2000, max_retries=3)
    # Two fresh random ICs were tried before the final attempt raised, i.e. the
    # retry loop is alive on a second run (it was dead: ``ic_explicit`` was True).
    assert len(rec) == 2
    # ... and the diagnostic does NOT claim the user supplied the IC.
    assert not any("supplied explicitly" in n for n in getattr(exc.value, "__notes__", []))


def test_an_explicit_ic_stays_explicit_across_runs() -> None:
    """The genuine user choice is still recorded (and still never swapped)."""
    h = ts.systems.Henon(ic=[0.1, 0.1])
    h.run(steps=10)
    assert h._ic_explicit is True


def test_system_plot_accepts_the_in_tree_renderer_keywords() -> None:
    """``system.plot`` takes the same backend options as ``Trajectory.plot``.

    The routing sends every unrecognised keyword to the integration, so a
    renderer option that was not in the allow-list used to raise on a system
    while working on a trajectory.
    """
    pytest.importorskip("plotly")
    lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
    traj_fig = lor.run(final_time=1.0, dt=0.05).plot().render("plotly", html=True)
    sys_fig = lor.plot(final_time=1.0, dt=0.05).render("plotly", html=True)
    assert type(sys_fig) is type(traj_fig)


class TestAFieldSystemRefusesAParameterGivenTwice:
    """A variable-dimension system's custom ``__init__`` obeys the front door.

    ``SystemBase.__init__`` refuses a parameter passed both in ``params=`` and as
    a keyword — "there is deliberately no precedence rule".  All **five** systems
    that resolve their own ``dim`` from a structural parameter merged their
    arguments *before* calling ``super()``, so the base never saw the duplication
    and a silent precedence applied instead (qodo #9).

    Two of the five are not field systems at all — ``Lorenz96`` and ``MultiChua``
    size themselves from ``N`` / ``n_circuits`` — which is why the shared merge
    lives in ``tsdynamics.systems._declared_params`` rather than in the spatial
    fields module.
    """

    CASES: ClassVar[list[tuple[str, str, float, float]]] = [
        ("GrayScott", "Du", 0.20, 0.99),
        ("GrayScott", "F", 0.03, 0.09),
        ("SwiftHohenberg", "r", 0.5, 0.9),
        ("SwiftHohenberg", "L", 20.0, 50.0),
        ("KuramotoSivashinsky", "L", 40.0, 22.0),
        ("Lorenz96", "f", 9.0, 8.0),
        ("MultiChua", "alpha", 16.0, 15.6),
    ]

    @pytest.mark.parametrize(("cls_name", "name", "v1", "v2"), CASES)
    def test_the_named_argument_and_params_cannot_both_carry_it(
        self, cls_name: str, name: str, v1: float, v2: float
    ) -> None:
        cls = getattr(ts.systems, cls_name)
        with pytest.raises(InvalidParameterError, match="given twice"):
            cls(**{name: v1}, params={name: v2})

    @pytest.mark.parametrize(("cls_name", "name", "v1", "v2"), CASES)
    def test_a_free_keyword_and_params_cannot_both_carry_it(
        self, cls_name: str, name: str, v1: float, v2: float
    ) -> None:
        cls = getattr(ts.systems, cls_name)
        with pytest.raises(InvalidParameterError, match="given twice"):
            cls(params={name: v2}, **{name: v1})

    @pytest.mark.parametrize(("cls_name", "name", "v1", "_v2"), CASES)
    def test_naming_it_exactly_once_still_works(
        self, cls_name: str, name: str, v1: float, _v2: float
    ) -> None:
        cls = getattr(ts.systems, cls_name)
        assert cls(**{name: v1}).params[name] == pytest.approx(v1)
        assert cls(params={name: v1}).params[name] == pytest.approx(v1)


@pytest.mark.parametrize(
    ("cls_name", "name", "value"),
    [
        ("GrayScott", "F", 0.07),
        ("SwiftHohenberg", "r", 0.4),
        ("KuramotoSivashinsky", "L", 40.0),
        ("Lorenz96", "f", 9.0),
        ("MultiChua", "alpha", 16.0),
    ],
)
def test_a_variable_dimension_system_can_be_reparametrised(
    cls_name: str, name: str, value: float
) -> None:
    """``with_params`` / ``copy`` rebuild a system that sizes itself.

    ``SystemBase.with_params`` forwards ``dim=`` and ``field_shape=`` on every
    rebuild, but these constructors swallowed them into ``**param_kwargs`` and
    rejected them as unknown parameters — so continuation, orbit diagrams and
    every other sweep over any of the five raised.  CLAUDE.md states this exact
    case is supposed to work ("a ``Sys(dim=2)``-constructed system can be
    re-parametrised").
    """
    cls = getattr(ts.systems, cls_name)
    sys_ = cls()
    rebuilt = sys_.with_params(**{name: value})
    assert rebuilt.params[name] == pytest.approx(value)
    assert rebuilt.dim == sys_.dim
    assert sys_.copy().dim == sys_.dim


def test_no_catalogue_system_raises_on_reparametrisation() -> None:
    """The sweep that found the three the review did not mention.

    The filed finding (qodo #9) named the field systems.  Driving ``with_params``
    over the whole registry is what showed the same constructor defect in
    ``Lorenz96`` and ``MultiChua``, neither of which is a field system.
    """
    from tsdynamics import registry

    broken: list[str] = []
    for entry in registry.all_systems():
        try:
            system = entry.cls()
        except Exception:  # pragma: no cover - construction is not what is tested
            continue
        first = next(iter(system.params.items()), None)
        if first is None:
            continue
        try:
            system.with_params(**{first[0]: first[1]})
            system.copy()
        except Exception as exc:  # noqa: BLE001 - the failure IS the finding
            broken.append(f"{entry.name}: {type(exc).__name__}: {exc}")
    assert not broken, "with_params/copy raises on:\n  " + "\n  ".join(broken)
