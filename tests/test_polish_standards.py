"""Enforce-forever polish standards (the v4 'lock it in' gates).

This module is the shared home for the registry-driven P4 **standards gates**
that keep the v4 redesign from silently rotting back into inconsistency.  Each
gate is biased to be *registry-driven* — it sweeps every registered analysis (or
every discovered result class), so a new analysis/result joins the gate with zero
test edits and a regression fails loudly instead of slipping through.

Sections (one per P4 ``POLISH`` gate stream):

- **Result-object contract** (stream WS-RESULT-GATE, this file's content) — every
  registered analysis returns a self-describing
  :class:`~tsdynamics.analysis._result.AnalysisResult` carrying the full result
  surface (``.meta`` / ``.summary()`` / ``.to_dict()`` / the ``.plot`` seam), and
  every scaling-curve result exposes the one canonical
  ``(estimate, stderr, abscissa, ordinate, fit_region, intercept)`` schema.  The
  contract is checked three ways: a *static* sweep over every registered
  analysis's return annotation (so the whole surface is enforced without
  constructing 47 inputs), a *runtime* sweep over a representative analysis of
  every result kind (so the surface is proven to actually fire), and a
  *foundation* check on the base/wrapper classes the rest inherit from.

(The sibling P4 gates — WS-NAMEGATE's naming gate and WS-ERRGATE's error-message
gate — append their own sections to this file.)
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import types as _types
import typing
from collections.abc import Mapping

import numpy as np
import pytest

import tsdynamics as ts
import tsdynamics.transforms as _tx  # noqa: F401  (import populates registry.transforms)
from tsdynamics import registry
from tsdynamics.analysis._result import (
    AnalysisResult,
    ArrayResult,
    CollectionResult,
    CountResult,
    ScalarResult,
    ScalingResult,
    VisualizationNotInstalled,
)
from tsdynamics.derived.poincare import PoincareSection
from tsdynamics.viz.spec import PlotKind

# ===========================================================================
# Result-object contract gate (stream WS-RESULT-GATE)
# ===========================================================================

#: The four affordances every result must carry (the acceptance contract).  A
#: result is self-describing when it can report its provenance (``meta``), render
#: a human readout (``summary``), serialize itself (``to_dict``), and expose the
#: deferred visualization seam (``plot``).
_CONTRACT_METHODS = ("summary", "to_dict")

#: Registered analyses whose return is a deliberate carve-out from
#: ``AnalysisResult`` — they return a richer type that *still* carries the result
#: surface.  ``poincare_section`` returns a
#: :class:`~tsdynamics.derived.PoincareSection` (a
#: :class:`~tsdynamics.data.Trajectory` subclass) so the section keeps every
#: trajectory affordance while adding ``summary``/``to_dict``/``plot`` on top.
_RESULT_CARVE_OUTS: dict[str, type] = {
    "poincare_section": PoincareSection,
}


def _return_classes(fn: object) -> tuple[type, ...]:
    """Resolve a callable's return annotation into its component *classes*.

    Unwraps ``Optional`` / ``X | Y`` unions (so ``ScalarResult | tuple[...]``
    yields ``ScalarResult``) and keeps only the members that are concrete classes
    — non-class annotation members (a bare ``tuple``/``ndarray`` overload that
    rides alongside the result object) are dropped.  Returns an empty tuple when
    there is no resolvable return annotation.
    """
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - an unresolved hint is a contract failure, surfaced below
        return ()
    annotation = hints.get("return")
    if annotation is None:
        return ()
    flat: list[type] = []

    def _walk(node: object) -> None:
        origin = typing.get_origin(node)
        if origin in (typing.Union, _types.UnionType):
            for arg in typing.get_args(node):
                _walk(arg)
        elif isinstance(node, type):
            flat.append(node)

    _walk(annotation)
    return tuple(flat)


def _carries_core_surface(cls: type, *, require_to_frame: bool) -> list[str]:
    """Return the contract affordances ``cls`` is *missing* (empty when complete).

    Checks that ``summary``/``to_dict`` (and, for ``AnalysisResult`` proper,
    ``to_frame``) resolve as callables, that ``plot`` resolves (the seam), and
    that ``meta`` is a declared field — read from :func:`dataclasses.fields` for
    dataclass results (``meta`` has a ``default_factory`` so it is not a class
    attribute), and assumed present on the non-dataclass carve-out (a
    ``Trajectory`` sets ``meta`` per instance).
    """
    missing: list[str] = []
    for name in _CONTRACT_METHODS:
        if not callable(getattr(cls, name, None)):
            missing.append(name)
    if require_to_frame and not callable(getattr(cls, "to_frame", None)):
        missing.append("to_frame")
    if getattr(cls, "plot", None) is None:
        missing.append("plot")
    if dataclasses.is_dataclass(cls):
        if "meta" not in {f.name for f in dataclasses.fields(cls)}:
            missing.append("meta")
    elif not hasattr(cls, "meta") and "meta" not in getattr(cls, "__slots__", ()):
        # Trajectory carries ``meta`` per instance (a slot/attribute), not on the
        # class; only flag it if neither a class attr nor a declared slot exists.
        missing.append("meta")
    return missing


# ---------------------------------------------------------------------------
# Foundation — the base + wrapper classes everything inherits from
# ---------------------------------------------------------------------------

#: Every result-wrapper base.  These define the surface the per-stream result
#: dataclasses inherit; a refactor dropping a method here would silently strip it
#: from the whole analysis layer, so they are guarded directly.
_BASE_RESULT_CLASSES = (
    AnalysisResult,
    ScalarResult,
    CountResult,
    ArrayResult,
    CollectionResult,
    ScalingResult,
)


@pytest.mark.parametrize("cls", _BASE_RESULT_CLASSES, ids=lambda c: c.__name__)
def test_base_result_classes_expose_contract(cls):
    """Each result base/wrapper exposes the full self-describing surface.

    The bases carry the canonical implementations; if one loses ``summary`` /
    ``to_dict`` / ``to_frame`` / ``plot`` / a ``meta`` field, every result that
    inherits from it loses it too — so they are the load-bearing guard.
    """
    missing = _carries_core_surface(cls, require_to_frame=True)
    assert not missing, f"{cls.__name__} is missing contract affordances: {missing}"


def test_analysis_result_plot_seam_raises_until_a_backend_lands():
    """The ``.plot`` seam exists on every result but raises until a backend ships.

    Visualization is deferred in v4: ``result.plot`` resolves to the accessor
    (callable *and* a namespace of typed kind methods), but every entry point —
    the bare call and each typed method — raises
    :class:`VisualizationNotInstalled` while no renderer is registered.
    """
    result = ScalarResult(1.23, meta={"system": "probe"})
    accessor = result.plot
    assert accessor is not None
    with pytest.raises(VisualizationNotInstalled):
        result.plot()
    for kind_method in ("scaling", "diagnostic", "time_series", "phase", "image"):
        with pytest.raises(VisualizationNotInstalled):
            getattr(accessor, kind_method)()


# ---------------------------------------------------------------------------
# Static, registry-driven — every registered analysis's *return type*
# ---------------------------------------------------------------------------


def test_registered_analysis_return_carries_full_contract(analysis_entry):
    """Every registered analysis declares a return type that carries the contract.

    Registry-driven (so a new analysis is swept with zero edits) and *static* —
    it reads the resolved return annotation rather than constructing inputs for
    all ~47 analyses.  Fails loudly if an analysis returns a bare
    ``float``/``ndarray``/``list`` (no annotated result class), or a hand-rolled
    result object that does not subclass :class:`AnalysisResult` and lacks the
    ``summary``/``to_dict``/``plot``/``meta`` surface.  The runtime sweep below
    proves the surface actually fires.
    """
    name = analysis_entry.name
    classes = _return_classes(analysis_entry.obj)

    if name in _RESULT_CARVE_OUTS:
        expected = _RESULT_CARVE_OUTS[name]
        assert expected in classes, (
            f"carve-out {name!r} should return {expected.__name__}, got {classes}"
        )
        missing = _carries_core_surface(expected, require_to_frame=False)
        assert not missing, f"carve-out {expected.__name__} missing contract: {missing}"
        return

    result_classes = [c for c in classes if issubclass(c, AnalysisResult)]
    assert result_classes, (
        f"analysis {name!r} must return an AnalysisResult subclass "
        f"(got return annotation {classes or 'none'})"
    )
    for cls in result_classes:
        missing = _carries_core_surface(cls, require_to_frame=True)
        assert not missing, (
            f"analysis {name!r} returns {cls.__name__}, which is missing "
            f"contract affordances: {missing}"
        )


# ---------------------------------------------------------------------------
# Runtime, representative — the surface actually fires on live objects
# ---------------------------------------------------------------------------


def _henon():
    """A small Hénon map for runtime result-contract checks."""
    return ts.systems.Henon()


def _logistic_series(n: int = 600) -> np.ndarray:
    """A deterministic chaotic series (logistic, r=3.9) for data-consuming analyses."""
    x = np.empty(n)
    x[0] = 0.4
    for i in range(1, n):
        x[i] = 3.9 * x[i - 1] * (1.0 - x[i - 1])
    return x


def _synthetic_basin_labels() -> np.ndarray:
    """A small 2-D basin label image for the (integration-free) basin metrics."""
    labels = np.zeros((24, 24), dtype=int)
    labels[:, 12:] = 1
    labels[::3, ::3] = 1  # a roughened boundary so the metrics are well-defined
    return labels


def _basin_box():
    """A box bounding the Hénon attractor, for the cheap basin/expansion probes."""
    from tsdynamics.data import Box

    return Box(np.array([-1.6, -0.45]), np.array([1.6, 0.45]))


#: Registered analyses whose ``to_dict()`` is a valid mapping but not yet
#: JSON-serializable, because a hand-rolled ``to_dict`` embeds a raw object rather
#: than recursing into its serializable form: ``recurrence_matrix`` carries a
#: SciPy sparse ``csr_matrix``, ``windowed_rqa`` carries nested ``RQAResult``
#: objects, and ``find_attractors`` carries nested ``Attractor`` objects.  All
#: three are real gaps against the ``to_dict`` "JSON-friendly" promise, tracked
#: for a fix in the owning analysis modules (out of this gate's owned paths); the
#: core contract (``to_dict`` returns a mapping) is still asserted for them below.
_TODICT_NOT_JSON = frozenset({"recurrence_matrix", "windowed_rqa", "find_attractors"})


def _runtime_cases() -> list[tuple[str, object]]:
    """Build the representative runtime cases — one cheap call per result class.

    Each entry exercises a distinct result class (or wrapper kind) end-to-end so
    the static surface above is proven to fire on a live object.  Calls are kept
    small/deterministic (fast tier) and warning-clean under ``filterwarnings=
    ['error']``.  The expensive or multistability-requiring analyses
    (``periodic_orbits``/``periodic_orbit``/``basins_of_attraction``/
    ``basin_fractions``/``continuation``/``tipping_points``/``resilience``) are
    covered by the static annotation sweep, not re-run here.
    """
    series = _logistic_series()
    traj = _henon().iterate(steps=600, ic=[0.1, 0.1])
    sine = np.sin(np.linspace(0.0, 40.0, 400))
    labels = _synthetic_basin_labels()
    box = _basin_box()
    return [
        # -- ArrayResult family --
        ("lyapunov_spectrum", lambda: ts.lyapunov_spectrum(_henon(), k=2, n=1500, ic=[0.1, 0.1])),
        ("mutual_information", lambda: ts.mutual_information(series, max_delay=20)),
        ("multiscale_entropy", lambda: ts.multiscale_entropy(series[:400], scales=6)),
        ("embed", lambda: ts.embed(series, 3, 1)),
        ("surrogates", lambda: ts.surrogates(series, "shuffle", 4, seed=0)),
        # -- ScalarResult family --
        ("max_lyapunov", lambda: ts.max_lyapunov(_henon(), n=150, ic=[0.1, 0.1])),
        ("kaplan_yorke_dimension", lambda: ts.kaplan_yorke_dimension([0.42, -1.62])),
        ("estimate_period", lambda: ts.estimate_period(sine)),
        ("zero_one_test", lambda: ts.zero_one_test(series)),
        ("permutation_entropy", lambda: ts.permutation_entropy(series)),
        ("sample_entropy", lambda: ts.sample_entropy(series[:300])),
        ("lz76_complexity", lambda: ts.lz76_complexity(series)),
        # -- CountResult --
        ("optimal_delay", lambda: ts.optimal_delay(series, max_delay=20)),
        # -- ScalingResult family (canonical scaling-curve schema) --
        ("correlation_dimension", lambda: ts.correlation_dimension(traj)),
        ("generalized_dimension", lambda: ts.generalized_dimension(traj)),
        ("lyapunov_from_data", lambda: ts.lyapunov_from_data(series)),
        ("expansion_entropy", lambda: ts.expansion_entropy(_henon(), box, n_samples=150, n=8)),
        # -- CollectionResult family --
        ("fixed_points", lambda: ts.fixed_points(_henon(), seed=0)),
        # -- rich per-stream result dataclasses --
        ("embedding_dimension", lambda: ts.embedding_dimension(series, max_dim=6)),
        ("recurrence_matrix", lambda: ts.recurrence_matrix(traj, recurrence_rate=0.05)),
        ("rqa", lambda: ts.rqa(traj, recurrence_rate=0.05)),
        ("windowed_rqa", lambda: ts.windowed_rqa(traj, window=200, step=100, recurrence_rate=0.05)),
        ("gali", lambda: ts.gali(_henon(), k=2, n=300, ic=[0.1, 0.1])),
        ("return_map", lambda: ts.return_map(series)),
        (
            "orbit_diagram",
            lambda: ts.orbit_diagram(
                ts.systems.Logistic(), "r", np.linspace(3.4, 4.0, 40), transient=100, n=60
            ),
        ),
        ("surrogate_test", lambda: ts.surrogate_test(series, n=19, seed=0)),
        (
            "find_attractors",
            lambda: ts.find_attractors(
                _henon(), box, resolution=30, n_seeds=80, max_steps=400, seed=0
            ),
        ),
        ("basin_entropy", lambda: ts.basin_entropy(labels)),
        ("uncertainty_exponent", lambda: ts.uncertainty_exponent(labels)),
        ("wada_property", lambda: ts.wada_property(labels)),
        # -- carve-out: PoincareSection (a Trajectory, not an AnalysisResult) --
        (
            "poincare_section",
            lambda: ts.poincare_section(ts.systems.Rossler(), plane=("y", 0.0, "up"), n=20, seed=0),
        ),
    ]


_RUNTIME_CASES = _runtime_cases()


@pytest.mark.parametrize("name,thunk", _RUNTIME_CASES, ids=[c[0] for c in _RUNTIME_CASES])
def test_runtime_result_contract(name, thunk):
    """A representative analysis of each result class fires the full contract live.

    Asserts on the returned object: it is an ``AnalysisResult`` (or the
    ``PoincareSection`` carve-out); ``.meta`` is a populated mapping; ``repr`` and
    ``.summary()`` produce non-empty text; ``.to_dict()`` is a mapping (and
    JSON-serializable, except the two tracked gaps in :data:`_TODICT_NOT_JSON`);
    and the ``.plot`` seam raises :class:`VisualizationNotInstalled`.

    The carve-out (``PoincareSection``) carries the same ``meta``/``summary``/
    ``to_dict``/``plot`` surface as an ``AnalysisResult``, but its ``plot`` is a
    ``Trajectory`` *method* rather than the typed-method accessor namespace — so
    the ``.plot.<kind>()`` assertion runs only for ``AnalysisResult`` instances.
    """
    result = thunk()
    assert isinstance(result, (AnalysisResult, PoincareSection)), (
        f"{name} returned {type(result).__name__}, not a result object"
    )

    # provenance
    assert isinstance(result.meta, Mapping) and result.meta, f"{name} has no provenance .meta"

    # human readouts
    assert repr(result).strip(), f"{name} has an empty repr"
    summary = result.summary()
    assert isinstance(summary, str) and summary.strip(), f"{name}.summary() is empty"

    # export
    data = result.to_dict()
    assert isinstance(data, dict), f"{name}.to_dict() returned {type(data).__name__}, not a dict"
    if name not in _TODICT_NOT_JSON:
        try:
            json.dumps(data)
        except TypeError as exc:  # pragma: no cover - a failure here is the gate firing
            pytest.fail(f"{name}.to_dict() is not JSON-serializable: {exc}")

    # the deferred visualization seam (raises until a rendering backend registers)
    with pytest.raises(VisualizationNotInstalled):
        result.plot()
    if isinstance(result, AnalysisResult):
        with pytest.raises(VisualizationNotInstalled):
            result.plot.scaling()


@pytest.mark.parametrize("name", sorted(_TODICT_NOT_JSON))
def test_known_non_json_to_dict_is_still_a_mapping(name):
    """The tracked non-JSON ``to_dict`` results still satisfy the mapping contract.

    ``recurrence_matrix``/``windowed_rqa`` embed a sparse matrix / nested results,
    so ``json.dumps`` does not yet round-trip them — a known gap against the
    ``to_dict`` "JSON-friendly" promise, tracked for a fix in the owning modules.
    This pins the *structural* half of the contract (``to_dict`` is a mapping) so
    the gap cannot widen into "no ``to_dict`` at all".
    """
    thunk = dict(_RUNTIME_CASES)[name]
    data = thunk().to_dict()
    assert isinstance(data, dict) and data, f"{name}.to_dict() is not a populated mapping"


# ---------------------------------------------------------------------------
# Scaling-curve results — the one canonical schema
# ---------------------------------------------------------------------------

#: The canonical scaling-curve field set every ``ScalingResult`` must expose so a
#: single generic ``result.plot.scaling()`` renders any of them.
_SCALING_FIELDS = ("estimate", "stderr", "abscissa", "ordinate", "fit_region", "intercept")


def _concrete_scaling_subclasses() -> list[type]:
    """Return every concrete ``ScalingResult`` subclass, auto-discovered."""
    seen: set[type] = set()
    out: list[type] = [ScalingResult]
    stack = list(ScalingResult.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        out.append(cls)
        stack.extend(cls.__subclasses__())
    return out


_SCALING_SUBCLASSES = _concrete_scaling_subclasses()


@pytest.mark.parametrize("cls", _SCALING_SUBCLASSES, ids=lambda c: c.__name__)
def test_scaling_result_exposes_canonical_schema(cls):
    """Every ``ScalingResult`` subclass declares the one canonical scaling schema.

    The whole scaling-curve family (every fractal dimension, the data-Lyapunov
    exponent, expansion entropy, …) shares ``(estimate, stderr, abscissa,
    ordinate, fit_region, intercept)`` plus the ``local_slopes`` / ``scaling_window``
    diagnostics and ``float(result)`` — so one ``result.plot.scaling()`` renders
    any of them.  Discovered via ``__subclasses__``, so a new scaling result is
    swept automatically.
    """
    field_names = {f.name for f in dataclasses.fields(cls)}
    missing = [name for name in _SCALING_FIELDS if name not in field_names]
    assert not missing, f"{cls.__name__} is missing canonical scaling fields: {missing}"
    for prop in ("local_slopes", "scaling_window"):
        assert hasattr(cls, prop), f"{cls.__name__} is missing scaling diagnostic {prop!r}"
    assert callable(getattr(cls, "__float__", None)), f"{cls.__name__} must define __float__"


#: Runtime cases that return a ``ScalingResult`` — used to prove the schema holds
#: behaviorally (not just structurally).
_SCALING_RUNTIME_CASES = [
    (name, thunk)
    for name, thunk in _RUNTIME_CASES
    if name
    in {"correlation_dimension", "generalized_dimension", "lyapunov_from_data", "expansion_entropy"}
]


@pytest.mark.parametrize(
    "name,thunk", _SCALING_RUNTIME_CASES, ids=[c[0] for c in _SCALING_RUNTIME_CASES]
)
def test_scaling_result_runtime_schema(name, thunk):
    """A live scaling result satisfies the canonical schema and emits a SCALING_FIT spec.

    ``float(result)`` returns ``estimate``; the curve arrays are equal-length
    ndarrays; ``fit_region`` is an in-bounds ``(lo, hi)`` index pair; and
    ``to_plot_spec()`` carries the ``SCALING_FIT`` plot intent so the generic
    scaling renderer can find the curve and the fit.
    """
    result = thunk()
    assert isinstance(result, ScalingResult)
    assert float(result) == pytest.approx(float(result.estimate))

    abscissa = np.asarray(result.abscissa)
    ordinate = np.asarray(result.ordinate)
    assert abscissa.ndim == 1 and ordinate.ndim == 1
    assert abscissa.size == ordinate.size and abscissa.size > 0

    lo, hi = result.fit_region
    assert isinstance(lo, int) and isinstance(hi, int)
    assert 0 <= lo <= hi < abscissa.size, f"{name} fit_region {result.fit_region} out of bounds"
    assert isinstance(result.intercept, float)

    spec = result.to_plot_spec()
    assert spec.kind == PlotKind.SCALING_FIT


# ---------------------------------------------------------------------------
# Coverage guard — keep the runtime sweep representative
# ---------------------------------------------------------------------------


def test_runtime_sweep_covers_every_result_kind():
    """The runtime sweep exercises every wrapper kind and every scaling subclass.

    Resolves each runtime case to its result class via the registry's return
    annotation (no extra runs) and asserts the sweep covers each wrapper-kind base
    (``ScalarResult``/``CountResult``/``ArrayResult``/``CollectionResult``/
    ``ScalingResult``) and *every* concrete ``ScalingResult`` subclass.  So adding
    a new scaling-curve estimator forces a runtime case here — the canonical
    schema can never ship un-exercised.
    """
    covered: set[type] = set()
    for name, _thunk in _RUNTIME_CASES:
        fn = registry.analyses.get(name)
        assert fn is not None, f"runtime case {name!r} is not a registered analysis"
        covered.update(_return_classes(fn))

    def _covers(base: type) -> bool:
        return any(isinstance(c, type) and issubclass(c, base) for c in covered)

    for kind in (ScalarResult, CountResult, ArrayResult, CollectionResult, ScalingResult):
        assert _covers(kind), f"runtime sweep covers no {kind.__name__}"

    for cls in _SCALING_SUBCLASSES:
        if cls is ScalingResult:
            continue
        assert cls in covered, (
            f"{cls.__name__} (a ScalingResult subclass) has no runtime contract case"
        )


# ===========================================================================
# Naming gate (stream WS-NAMEGATE)
# ===========================================================================
#
# The registry-driven enforcement of the *frozen* naming glossary
# (``docs/contributing/glossary.md``, stream WS-VOCAB) over the whole public
# callable surface — every function in ``registry.analyses`` *and*
# ``registry.transforms``.  Two rules are decidable from ``inspect.signature``
# alone (glossary §7) and so are CI-enforced here:
#
#   1. the **first positional argument** of every public callable is ``system``
#      (a System it integrates/iterates) or ``data`` (a measured series it
#      consumes), unless the ``(function, first-arg)`` pair names a *prior result*
#      on the §5 whitelist; and
#   2. **no parameter** uses a banned spelling from glossary §2 — built straight
#      from the §2 "Bans" column — unless the ``(function, parameter)`` pair is a
#      §5 homonym carve-out.
#
# ``tests/test_calling_convention.py`` (stream WS-CONV) is the *focused
# precursor* this gate generalises; both read from the same glossary, so a banned
# spelling can never re-enter a public signature.  This module is the **forever
# home** the new-analysis checklist (glossary §8) points at: it carries the full
# §2 ban set (including the ``method``-selector row — ``kind``/``mode``/
# ``estimator``/``scheme``), and it *self-validates* its own tables against the
# live registry (the whitelists can never silently go stale and mask a
# regression).  The sweep is pure introspection — no engine, fast tier.

# ── glossary §1: the two canonical first-argument roles ────────────────────
_NAMEGATE_FIRST_ARGS = frozenset({"system", "data"})

# Banned first-arg spellings (glossary §1) — every one collapses to system/data.
_NAMEGATE_BANNED_FIRST_ARGS = frozenset(
    {"sys", "sys_or_traj", "map_sys", "observable", "source", "x", "series"}
)

# §1 / §5: the first argument of a function that consumes a *prior result* is
# named by the *kind* of result, not unified onto system/data.  Whitelisted as
# exact ``(function, first-arg)`` pairs; liveness is asserted below so a rename
# cannot leave a stale entry masking a real first-arg regression.
_NAMEGATE_PRIOR_RESULT_FIRST_ARG = {
    "kaplan_yorke_dimension": "spectrum",  # a Lyapunov spectrum
    "uncertainty_exponent": "basins",  # a BasinsResult
    "wada_property": "basins",
    "basin_entropy": "basins",
    "resilience": "result",  # a BasinsResult / ContinuationResult
    "tipping_points": "result",
}

# ── glossary §2: banned parameter spellings → their canonical concept ───────
# Built straight from the §2 "Bans" column (incl. the pre-emptive † spellings,
# kept so a concept can never drift into them).  A parameter whose name is a key
# here fails unless its ``(function, parameter)`` pair is on the §5 whitelist.
# ``method`` (the canonical variant/kernel selector) is *never* a key — it is the
# allowed spelling, so it needs no per-site whitelist.
_NAMEGATE_BANNED_PARAMS: dict[str, str] = {
    # initial condition → ic
    "x0": "ic",
    "initial": "ic",
    "u0": "ic",
    "y0": "ic",
    # RNG seed → seed
    "random_state": "seed",
    "rng": "seed",
    # discard-transient amount → transient
    "burn_in": "transient",
    "n_transient": "transient",
    "warmup": "transient",
    # integration horizon (flows) → final_time
    "T": "final_time",
    "t_final": "final_time",
    "tmax": "final_time",
    # iteration / sampling horizon → n
    "steps": "n",
    "n_rescale": "n",
    # step size → dt
    "h": "dt",
    # observed component → component
    "components": "component",
    "observable": "component",
    "coord": "component",
    "col": "component",
    # embedding dimension → dimension
    "m": "dimension",
    "emb_dim": "dimension",
    "dim": "dimension",
    # embedding delay → delay; the delay-search ceiling → max_delay
    "tau": "delay",
    "lag": "delay",
    "max_lag": "max_delay",
    # Theiler window → theiler
    "theiler_window": "theiler",
    "w": "theiler",
    # nearest-neighbour count → n_neighbors
    "min_neighbors": "n_neighbors",
    "num_neighbors": "n_neighbors",
    # spatial region → region
    "grid": "region",
    "box": "region",
    "domain": "region",
    "bounds": "region",
    # algorithm / kernel selector → method  (method itself is never banned)
    "kind": "method",
    "mode": "method",
    "estimator": "method",
    "scheme": "method",
}

# §5 homonym carve-outs: exact ``(function, parameter)`` pairs that may use a
# token banned elsewhere.  None of the canonical homonym tokens (``k``/``k_max``/
# ``step``/``horizon``/``max_steps``/``max_delay``/``fs``) collide with a §2 ban
# under exact-name matching (``step`` ≠ ``steps``, ``max_delay`` ≠ ``max_lag``,
# ``max_steps`` ≠ ``steps``), so this whitelist is empty today — kept as the
# documented extension point.  ``test_naming_gate_homonym_whitelist_is_sound``
# guards that any future entry references a real banned token on a real function.
_NAMEGATE_HOMONYM_WHITELIST: frozenset[tuple[str, str]] = frozenset()

# §5 / §6: the homonym tokens, with the functions that legitimately carry them.
# Each token names a *different* concept here (a GALI order, a stride, a search
# ceiling, …), so the gate must never ban it — asserted below.  Functions that
# are not registered (so the gate never sweeps them) are listed for completeness;
# their liveness is checked only when they are present in the registry.
_NAMEGATE_HOMONYM_CARVE_OUTS: dict[str, tuple[str, ...]] = {
    "k": ("gali", "lyapunov_spectrum"),  # GALI order / count of exponents
    "k_max": ("lyapunov_from_data",),  # scaling-curve abscissa horizon
    "step": ("windowed_rqa",),  # window stride (not the time step dt)
    "horizon": ("nonlinear_prediction_error",),  # prediction lead-time
    "max_steps": (  # integration safety cap (not the run length n)
        "find_attractors",
        "basins_of_attraction",
        "continuation",
        "basin_fractions",
    ),
    "max_delay": (  # delay-search ceiling (supersedes max_lag)
        "optimal_delay",
        "mutual_information",
        "estimate_period",
        "autocorrelation",
    ),
    "fs": (  # sampling frequency (Hz), alongside dt
        "power_spectral_density",
        "spectral_entropy",
        "spectral_centroid",
        "dominant_frequency",
        "butter_filter",
        "extract_features",
    ),
    "skip_crossings": ("poincare_section", "return_map"),  # discarded crossings
}

# §5 / §6: ``n_cut`` (``zero_one_test``) is a domain-owned mean-square-displacement
# *lag ceiling* (default ``N // 10``) — explicitly **not** a transient and
# **neither renamed nor banned** (glossary §6 note).  It must never enter the ban
# set; the roadmap's older "n_cut beyond canonical" sketch is superseded by the
# frozen glossary.
_NAMEGATE_DOMAIN_OWNED = {"zero_one_test": "n_cut"}


def _namegate_public_callables() -> list[tuple[str, object]]:
    """Every registered analysis + transform as sorted ``(name, callable)`` pairs.

    Sweeps **both** ``registry.analyses`` and ``registry.transforms`` (glossary
    §7 rule 4).  Evaluated at import time over the live registries, so a new
    analysis/transform joins the gate with zero test edits.
    """
    pairs: list[tuple[str, object]] = []
    for reg in (registry.analyses, registry.transforms):
        for entry in reg.all():
            pairs.append((entry.name, entry.obj))
    return sorted(pairs, key=lambda p: p[0])


_NAMEGATE_CALLABLES = _namegate_public_callables()
_NAMEGATE_BY_NAME = dict(_NAMEGATE_CALLABLES)


def _namegate_positional(fn: object) -> list[inspect.Parameter]:
    """The positional parameters of ``fn`` (positional-only or positional-or-keyword)."""
    return [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]


def _namegate_first_arg(fn: object) -> str | None:
    """Name of ``fn``'s first positional argument, or ``None`` if it is keyword-only."""
    positional = _namegate_positional(fn)
    return positional[0].name if positional else None


@pytest.mark.parametrize("name,fn", _NAMEGATE_CALLABLES, ids=[n for n, _ in _NAMEGATE_CALLABLES])
def test_naming_gate_first_argument_is_canonical(name: str, fn: object) -> None:
    """First positional arg is ``system`` / ``data`` (or a §5 prior-result).

    Registry-driven over both ``registry.analyses`` and ``registry.transforms``;
    a pure keyword-only consumer (no positional parameter) has no first-arg role
    and is skipped.
    """
    first = _namegate_first_arg(fn)
    if first is None:
        return
    if name in _NAMEGATE_PRIOR_RESULT_FIRST_ARG:
        expected = _NAMEGATE_PRIOR_RESULT_FIRST_ARG[name]
        assert first == expected, (
            f"{name}: prior-result first arg should be {expected!r}, got {first!r}."
        )
        return
    assert first not in _NAMEGATE_BANNED_FIRST_ARGS, (
        f"{name}: first argument {first!r} is a banned spelling (glossary §1) — "
        f"use 'system' (a System) or 'data' (a measured series)."
    )
    assert first in _NAMEGATE_FIRST_ARGS, (
        f"{name}: first argument {first!r} is neither 'system' nor 'data' (and is "
        f"not a whitelisted prior-result consumer)."
    )


@pytest.mark.parametrize("name,fn", _NAMEGATE_CALLABLES, ids=[n for n, _ in _NAMEGATE_CALLABLES])
def test_naming_gate_no_banned_parameter_spellings(name: str, fn: object) -> None:
    """No parameter uses a glossary §2 banned spelling (outside §5 carve-outs).

    Registry-driven over both registries.  Reports every offender on a function
    at once, each with its canonical replacement, so a regression names the fix.
    """
    offenders = [
        f"{p.name!r} (use {_NAMEGATE_BANNED_PARAMS[p.name]!r})"
        for p in inspect.signature(fn).parameters.values()
        if p.name in _NAMEGATE_BANNED_PARAMS and (name, p.name) not in _NAMEGATE_HOMONYM_WHITELIST
    ]
    assert not offenders, f"{name}: banned parameter spelling(s): {', '.join(offenders)}."


def test_naming_gate_tables_are_glossary_faithful() -> None:
    """The gate's own ban/canonical tables are internally consistent.

    A corrupted table could silently neuter the gate (e.g. a canonical concept
    accidentally added to the ban set, or ``method`` banned).  These invariants
    fail loudly if the glossary-derived constants ever drift into nonsense.
    """
    # The two first-arg roles are never themselves banned first-args.
    assert _NAMEGATE_FIRST_ARGS.isdisjoint(_NAMEGATE_BANNED_FIRST_ARGS)
    # No banned token maps onto another banned token — every canonical target is
    # a clean landing spelling.
    for banned, canonical in _NAMEGATE_BANNED_PARAMS.items():
        assert canonical not in _NAMEGATE_BANNED_PARAMS, (
            f"canonical target {canonical!r} (for banned {banned!r}) is itself banned."
        )
    # The locked / always-allowed spellings can never be banned: the two input
    # roles, the universal seed/ic, and the method selector (glossary §2 note).
    for locked in ("system", "data", "ic", "seed", "method"):
        assert locked not in _NAMEGATE_BANNED_PARAMS, f"{locked!r} must never be banned."
    # Every homonym carve-out token is kept out of the ban set, so the gate can
    # never over-ban a legitimate different-concept use (glossary §5).
    for token in (*_NAMEGATE_HOMONYM_CARVE_OUTS, *_NAMEGATE_DOMAIN_OWNED.values()):
        assert token not in _NAMEGATE_BANNED_PARAMS, (
            f"homonym/domain token {token!r} must not be in the ban set (glossary §5/§6)."
        )


def test_naming_gate_prior_result_whitelist_is_live() -> None:
    """Every §5 prior-result whitelist entry matches a registered function's real first arg.

    A stale whitelist entry (a renamed function, or a first-arg that changed)
    would silently mask a genuine first-arg violation, so each pair is checked
    against the live signature.
    """
    for fn_name, expected in _NAMEGATE_PRIOR_RESULT_FIRST_ARG.items():
        fn = _NAMEGATE_BY_NAME.get(fn_name)
        assert fn is not None, (
            f"prior-result whitelist names {fn_name!r}, which is not a registered analysis."
        )
        first = _namegate_first_arg(fn)
        assert first == expected, (
            f"{fn_name}: whitelist expects first arg {expected!r} but the live "
            f"signature has {first!r} — update the whitelist or the signature."
        )


def test_naming_gate_homonym_carve_outs_are_honored() -> None:
    """The §5 homonym carve-outs are honored: never banned, and live where registered.

    For every carve-out token the gate must (a) keep it out of the ban set — so a
    legitimate ``gali.k`` / ``windowed_rqa.step`` / ``optimal_delay.max_delay``
    never trips the gate — and (b) where the named function is *registered* (and
    thus actually swept), carry the documented parameter, so the §5 table stays a
    faithful description of the live surface rather than dead documentation.
    """
    for token, functions in _NAMEGATE_HOMONYM_CARVE_OUTS.items():
        assert token not in _NAMEGATE_BANNED_PARAMS, (
            f"carve-out token {token!r} must never be banned (glossary §5)."
        )
        for fn_name in functions:
            fn = _NAMEGATE_BY_NAME.get(fn_name)
            if fn is None:
                continue  # not registered → outside the gate's sweep; nothing to honor
            params = set(inspect.signature(fn).parameters)
            assert token in params, (
                f"{fn_name}: §5 documents a {token!r} parameter, but the live "
                f"signature has none — the carve-out is stale."
            )
    # The domain-owned non-transient (zero_one_test.n_cut) is likewise un-banned
    # and present (glossary §5/§6).
    for fn_name, token in _NAMEGATE_DOMAIN_OWNED.items():
        fn = _NAMEGATE_BY_NAME.get(fn_name)
        if fn is None:
            continue
        assert token not in _NAMEGATE_BANNED_PARAMS
        assert token in set(inspect.signature(fn).parameters), (
            f"{fn_name}: §5 documents a {token!r} parameter that is absent from the signature."
        )


def test_naming_gate_homonym_whitelist_is_sound() -> None:
    """Any §5 homonym whitelist entry references a real banned token on a real function.

    The whitelist is empty today (no canonical homonym collides with a §2 ban),
    but this keeps it honest as an extension point: a future ``(function,
    parameter)`` entry must name a *registered* function that actually has the
    parameter and whose parameter is a *banned* token — otherwise the entry is
    dead and should be removed rather than silently doing nothing.
    """
    for fn_name, param in _NAMEGATE_HOMONYM_WHITELIST:
        assert param in _NAMEGATE_BANNED_PARAMS, (
            f"whitelist pair ({fn_name!r}, {param!r}) is pointless: {param!r} is not banned."
        )
        fn = _NAMEGATE_BY_NAME.get(fn_name)
        assert fn is not None, (
            f"whitelist names {fn_name!r}, which is not a registered analysis/transform."
        )
        assert param in set(inspect.signature(fn).parameters), (
            f"{fn_name}: whitelist carves out {param!r}, absent from the live signature."
        )
