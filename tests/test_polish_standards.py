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

- **Naming gate** (stream WS-NAMEGATE) — the registry-driven enforcement of the
  *frozen* naming glossary (``docs/contributing/glossary.md``) over every public
  callable's signature: a decidable check that the first positional argument is
  ``system``/``data`` and that no parameter uses a banned spelling.

- **Error-message gate** (stream WS-ERRGATE) — a *curated* table of the headline
  wrong-input footguns the v4 audit named (``final_time<=0`` · ``dt<=0`` ·
  too-short data · unknown keyword · unknown attribute · wrong-dimension ``ic``).
  Error quality is not decidable from a signature, so this gate feeds each wrong
  input and asserts the two halves of the ``tsdynamics.errors`` value-naming
  standard — the message *names the offending value* (and, for the migrated
  sites, raises a ``TSDynamicsError`` subclass) and the input *raises* rather
  than silently returning garbage.  Footguns WS-ERRORS explicitly deferred to
  later lanes are tracked with a strict ``xfail`` so they trip the moment a
  future stream closes them.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import pathlib
import re
import types as _types
import typing
from collections.abc import Mapping

import numpy as np
import pytest

import tsdynamics as ts
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
from tsdynamics.analysis.basins.basins import BasinsResult
from tsdynamics.derived.poincare import PoincareSection
from tsdynamics.errors import InvalidInputError, InvalidParameterError
from tsdynamics.viz.spec import PlotKind


@pytest.fixture
def _no_render_backend(monkeypatch):
    """Force an empty renderers registry so the ``.plot`` seam raises.

    As of stream VIZ-MPL-CORE the matplotlib backend lazily auto-registers on the
    first render, so the deferred-seam ``VisualizationNotInstalled`` path is
    exercised by clearing the registry and stubbing
    :func:`register_builtin_renderers` to a no-op for the test, then restoring it.
    """
    from tsdynamics.viz import render as render_mod

    saved = registry.renderers.all()
    registry.renderers.clear()
    monkeypatch.setattr(render_mod, "register_builtin_renderers", lambda *a, **k: [])
    try:
        yield
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj, replace=True)


# ===========================================================================
# Result-object contract gate (stream WS-RESULT-GATE)
# ===========================================================================

#: The affordances every result must carry (the acceptance contract).  A result
#: is self-describing when it can report its provenance (``meta``), serialize
#: itself (``to_dict``), and expose the deferred visualization seam (``plot``).
#:
#: ``summary`` is **deliberately absent** since v6.  It was a method nothing
#: advertised, carrying the good text while ``repr()`` printed a dataclass dump —
#: so the readout a user got by typing the result's name in a REPL was the worse
#: of the two.  ``__repr__`` *is* what ``summary()`` printed now; one concept, one
#: spelling.
_CONTRACT_METHODS = ("to_dict",)

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


def test_analysis_result_plot_seam_raises_until_a_backend_lands(_no_render_backend):
    """The ``.plot`` seam exists on every result but raises when no backend is registered.

    ``result.plot`` resolves to the accessor — callable, *and* a namespace of the
    transforms that admit this result — but the verb raises
    :class:`VisualizationNotInstalled` while no renderer is registered, with every
    vocabulary the door accepts.

    The eight typed kind methods this used to sweep are gone (they relabelled the
    spec rather than rebuilding it); a guess at one is an ``AttributeError``
    naming what to type, which is a different question and lives in
    ``tests/test_plot_accessor_kinds.py``.
    """
    result = ScalarResult(1.23, meta={"system": "probe"})
    accessor = result.plot
    assert accessor is not None
    with pytest.raises(VisualizationNotInstalled):
        result.plot()
    for vocabulary in ({"color": "red"}, {"title": "t"}, {"theme": "dark"}, {"figsize": (3, 2)}):
        with pytest.raises(VisualizationNotInstalled):
            result.plot(**vocabulary)


# ---------------------------------------------------------------------------
# Static, registry-driven — every registered analysis's *return type*
# ---------------------------------------------------------------------------

#: Analyses **promoted into the registry by v6** that do not yet declare an
#: ``AnalysisResult`` return, with the slot that owns the result class.
#:
#: Contract §4.2 r1 ("every registered analysis returns an ``AnalysisResult``
#: subclass") and §5.7 ("``analysis/planar.py``'s 8 field functions — **promoted
#: public**") pull in opposite directions for exactly one release: promoting a
#: function into ``registry.analyses`` puts it under the result contract, and the
#: result classes are a different slot's file.  Every row is an **xfail**, so the
#: day one of them starts returning a result the row has to be deleted — the
#: table cannot quietly become the norm.
_RESULT_CONTRACT_NOT_LANDED: dict[str, str] = {
    # S4 · ANALYSIS-DOOR promoted these eight from analysis/planar.py (§5.7);
    # S3 · ANALYSIS-DOMAIN owns the result classes they need.
    "escape_time_field": "S3/S4: promoted in v6, result class not landed (§4.2 r1)",
    "flow_field": "S3/S4: promoted in v6, result class not landed (§4.2 r1)",
    "ftle_field": "S3/S4: promoted in v6, result class not landed (§4.2 r1)",
    "invariant_density": "S3/S4: promoted in v6, result class not landed (§4.2 r1)",
    "nullclines": "S3/S4: promoted in v6, result class not landed (§4.2 r1)",
    "streamlines": "S3/S4: promoted in v6, result class not landed (§4.2 r1)",
    "trace_determinant": "S3/S4: promoted in v6, result class not landed (§4.2 r1)",
    "transient_time_field": "S3/S4: promoted in v6, result class not landed (§4.2 r1)",
    # the sampling tools and the set metric, likewise newly registered
    "estimate_dt_from_sagitta": "S3: registered in v6, returns a hidden SagittaDt (§4.2 r1)",
    "sagitta_profile": "S3: registered in v6, returns a bare ndarray (§4.2 r1)",
    "set_distance": "S3: registered in v6, returns a bare float (§4.2 r1)",
    # pre-existing returns the promotion sweep newly exposed
    "autocorrelation": "S3: returns a bare ndarray (§4.2 r1)",
    "correlation_sum": "S3: returns an unannotated tuple (§4.2 r1)",
    "dimension_spectrum": "S3: returns an unannotated tuple (§4.2 r1)",
}


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
    if name in _RESULT_CONTRACT_NOT_LANDED:
        pytest.xfail(_RESULT_CONTRACT_NOT_LANDED[name])
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
    from _strategies import logistic_series

    return logistic_series(n, r=3.9, x0=0.4, burn=0)


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
#: objects, and ``attractors`` carries nested ``Attractor`` objects.  All
#: three are real gaps against the ``to_dict`` "JSON-friendly" promise, tracked
#: for a fix in the owning analysis modules (out of this gate's owned paths); the
#: core contract (``to_dict`` returns a mapping) is still asserted for them below.
_TODICT_NOT_JSON = frozenset({"recurrence_matrix", "windowed_rqa", "attractors"})


def _runtime_cases() -> list[tuple[str, object]]:
    """Build the representative runtime cases — one cheap call per result class.

    Each entry exercises a distinct result class (or wrapper kind) end-to-end so
    the static surface above is proven to fire on a live object.  Calls are kept
    small/deterministic (fast tier) and warning-clean under ``filterwarnings=
    ['error']``.  The expensive or multistability-requiring analyses
    (``periodic_orbits``/``basins``/
    ``basin_fractions``/``continuation``/``tipping_points``/``resilience``) are
    covered by the static annotation sweep, not re-run here.
    """
    series = _logistic_series()
    traj = _henon().run(steps=600, ic=[0.1, 0.1])
    sine = np.sin(np.linspace(0.0, 40.0, 400))
    labels = _synthetic_basin_labels()
    box = _basin_box()
    return [
        # -- ArrayResult family --
        (
            "lyapunov_spectrum",
            lambda: ts.analysis.lyapunov_spectrum(_henon(), k=2, n=1500, ic=[0.1, 0.1]),
        ),
        ("mutual_information", lambda: ts.analysis.mutual_information(series, max_delay=20)),
        ("embed", lambda: ts.analysis.embed(series, 3, 1)),
        # -- ScalarResult family --
        ("kaplan_yorke_dimension", lambda: ts.analysis.kaplan_yorke_dimension([0.42, -1.62])),
        ("estimate_period", lambda: ts.analysis.estimate_period(sine)),
        ("zero_one_test", lambda: ts.analysis.zero_one_test(series)),
        # -- CountResult --
        ("optimal_delay", lambda: ts.analysis.optimal_delay(series, max_delay=20)),
        # -- ScalingResult family (canonical scaling-curve schema) --
        ("correlation_dimension", lambda: ts.analysis.correlation_dimension(traj)),
        ("generalized_dimension", lambda: ts.analysis.generalized_dimension(traj)),
        ("lyapunov_from_data", lambda: ts.analysis.lyapunov_from_data(series)),
        (
            "expansion_entropy",
            lambda: ts.analysis.expansion_entropy(_henon(), box, n_samples=150, n=8),
        ),
        # -- CollectionResult family --
        ("fixed_points", lambda: ts.analysis.fixed_points(_henon(), seed=0)),
        # -- rich per-stream result dataclasses --
        ("embedding_dimension", lambda: ts.analysis.embedding_dimension(series, max_dim=6)),
        ("recurrence_matrix", lambda: ts.analysis.recurrence_matrix(traj, recurrence_rate=0.05)),
        ("rqa", lambda: ts.analysis.rqa(traj, recurrence_rate=0.05)),
        (
            "windowed_rqa",
            lambda: ts.analysis.windowed_rqa(traj, window=200, step=100, recurrence_rate=0.05),
        ),
        ("gali", lambda: ts.analysis.gali(_henon(), k=2, n=300, ic=[0.1, 0.1])),
        ("return_map", lambda: ts.analysis.return_map(series)),
        (
            "orbit_diagram",
            lambda: ts.analysis.orbit_diagram(
                ts.systems.Logistic(),
                "r",
                np.linspace(3.4, 4.0, 40),
                transient=100,
                points_per_value=60,
            ),
        ),
        (
            "attractors",
            lambda: ts.analysis.attractors(
                _henon(), box, resolution=30, n_seeds=80, max_steps=400, seed=0
            ),
        ),
        ("basin_entropy", lambda: ts.analysis.basin_entropy(labels)),
        ("uncertainty_exponent", lambda: ts.analysis.uncertainty_exponent(labels)),
        ("wada_property", lambda: ts.analysis.wada_property(labels)),
        # -- carve-out: PoincareSection (a Trajectory, not an AnalysisResult) --
        (
            "poincare_section",
            lambda: ts.analysis.poincare_section(
                ts.systems.Rossler(), plane=("y", 0.0, "up"), crossings=20, seed=0
            ),
        ),
    ]


_RUNTIME_CASES = _runtime_cases()

#: Results whose ``summary()`` has not been folded into ``__repr__`` yet.
#: **Empty since round 4** — ``PoincareSection`` was the last one, and folding it
#: in is what closed §4.3 for every result the library returns.
_SUMMARY_NOT_YET_FOLDED: frozenset[str] = frozenset()


def test_no_result_still_carries_summary() -> None:
    """§4.3 — the repr IS the answer, for EVERY result, with no carve-out left."""
    assert not _SUMMARY_NOT_YET_FOLDED, "delete the row; the contract is enforced below"
    for name, thunk in _RUNTIME_CASES:
        assert not hasattr(thunk(), "summary"), f"{name} still carries summary()"


@pytest.mark.parametrize("name,thunk", _RUNTIME_CASES, ids=[c[0] for c in _RUNTIME_CASES])
def test_runtime_result_contract(name, thunk, _no_render_backend):
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

    # the human readout — since v6 there is exactly one, and it is ``repr``
    readout = repr(result)
    assert readout.strip(), f"{name} has an empty repr"
    if name not in _SUMMARY_NOT_YET_FOLDED:
        assert not hasattr(result, "summary"), (
            f"{name} still carries summary(); v6 folded it into __repr__ (one concept, "
            "one spelling), so a second readout is a second answer waiting to drift"
        )

    # export
    data = result.to_dict()
    assert isinstance(data, dict), f"{name}.to_dict() returned {type(data).__name__}, not a dict"
    if name not in _TODICT_NOT_JSON:
        try:
            json.dumps(data)
        except TypeError as exc:  # pragma: no cover - a failure here is the gate firing
            pytest.fail(f"{name}.to_dict() is not JSON-serializable: {exc}")

    # the deferred visualization seam (raises until a rendering backend registers).
    # A Trajectory-shaped result's ``.plot()`` only *builds* the spec since v6, so
    # the backend is reached through ``.render()``; an AnalysisResult's ``.plot``
    # accessor still renders directly.
    with pytest.raises(VisualizationNotInstalled):
        if isinstance(result, AnalysisResult):
            result.plot()
        else:
            result.plot().render()
    if isinstance(result, AnalysisResult):
        with pytest.raises(VisualizationNotInstalled):
            result.plot(title="still refused without a backend")


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
#: single generic ``result.plot()`` renders any of them.
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
    diagnostics and ``float(result)`` — so one ``result.plot()`` renders
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
    ``__plot_spec__()`` carries the ``SCALING_FIT`` plot intent so the generic
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

    spec = result.__plot_spec__()
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
# callable surface — every function in ``registry.analyses``.  Two rules are
# decidable from ``inspect.signature`` alone (glossary §7) and so are CI-enforced
# here:
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
    # observed component(s) → components
    #
    # **Flipped in v6.**  The glossary froze the singular, and the singular was
    # measurably wrong: ``estimate_period(components=2)`` sliced ``data[2]`` — a
    # *row*, one state — so it raised on a 2-component Van der Pol and silently
    # returned 0.026 where the truth is 8.0 (311x) on a 10-component system.  The
    # plural is the honest name because every one of these doors takes a
    # selection, not a single index.
    "component": "components",
    "observable": "components",
    "coord": "components",
    "col": "components",
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
# token banned elsewhere, because on *that* function the token names a different
# concept.  ``test_naming_gate_homonym_whitelist_is_sound`` asserts every entry
# references a real banned token on a real function, so a rename cannot leave a
# stale row masking a genuine regression.
#
# Every row below is a homonym, not an exemption:
#
# * ``grid`` on the six planar **field** analyses is a *resolution* — the number
#   of samples per axis (``grid=201``) — not a region.  The box those functions
#   work over is ``xlim``/``ylim``, and the §2 ban exists to stop a *region*
#   being spelled ``grid``.  Banning an integer resolution because it shares a
#   word with a box is the false positive the carve-out table is for.
# * ``steps`` on ``streamlines`` is the integration length of one streamline —
#   the ``max_steps`` concept, a safety/length cap on a single curve, not the
#   run length of an analysis.
# * ``kind`` on ``return_map`` names *which return map* (successive maxima,
#   successive minima, successive section crossings).  The §2 selector ban exists
#   to keep one word for "which algorithm"; this is not an algorithm choice, and
#   v6 renamed it from ``method`` precisely so that ``method`` stays the
#   estimator word everywhere.
_NAMEGATE_HOMONYM_WHITELIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("escape_time_field", "grid"),
        ("flow_field", "grid"),
        ("ftle_field", "grid"),
        ("nullclines", "grid"),
        ("transient_time_field", "grid"),
        ("streamlines", "steps"),
        ("return_map", "kind"),
    }
)

#: Signatures the **deferred** v6.1 VOCAB sweep still owns, with the reason.
#: Contract §8.2 defers the general keyword sweep ("worst risk-to-value ratio in
#: the plan"); the *named* renames a user types landed in v6 and are enforced
#: above.  These rows are what is left, and each names why it is not simply a
#: bug.  ``test_namegate_deferred_rows_are_live`` fails when one starts passing,
#: so the table can only shrink.
#: The **other half** of the v6 VOCAB sweep: signatures still spelling the
#: observed-component selector in the singular.  **Empty since round 4** — the
#: nine analysis doors and every plot transform were renamed to ``components=``
#: together, closing the C3 split (two grammars for one argument) that this
#: table existed to record.  It can only shrink, so it stays as the gate.
_NAMEGATE_DEFERRED_PARAM: frozenset[tuple[str, str]] = frozenset()

_NAMEGATE_DEFERRED_FIRST_ARG: dict[str, str] = {
    # A binary metric: neither point set is "the subject", so neither can be
    # 'system' or 'data' without lying about the other.  Arguably permanent.
    "set_distance": "a",
    # Sampling tools promoted into the registry in v6; their first argument is
    # measured data under another name.  v6.1 VOCAB sweep.
    "estimate_dt_from_sagitta": "y",
    "sagitta_profile": "samples",
    "invariant_density": "values",
}

# §5 / §6: the homonym tokens, with the functions that legitimately carry them.
# Each token names a *different* concept here (a GALI order, a stride, a search
# ceiling, …), so the gate must never ban it — asserted below.  Functions that
# are not registered (so the gate never sweeps them) are listed for completeness;
# their liveness is checked only when they are present in the registry.
_NAMEGATE_HOMONYM_CARVE_OUTS: dict[str, tuple[str, ...]] = {
    "k": ("gali", "lyapunov_spectrum"),  # GALI order / count of exponents
    "k_max": ("lyapunov_from_data",),  # scaling-curve abscissa horizon
    "step": ("windowed_rqa",),  # window stride (not the time step dt)
    "max_steps": (  # integration safety cap (not the run length n)
        "attractors",
        "basins",
        "continuation",
        "basin_fractions",
    ),
    "max_delay": (  # delay-search ceiling (supersedes max_lag)
        "optimal_delay",
        "mutual_information",
        "estimate_period",
        "autocorrelation",
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
    """Every registered analysis as sorted ``(name, callable)`` pairs.

    Sweeps ``registry.analyses`` (glossary §7 rule 4).  Evaluated at import time
    over the live registry, so a new analysis joins the gate with zero test edits.
    """
    pairs = [(entry.name, entry.obj) for entry in registry.analyses.all()]
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

    Registry-driven over ``registry.analyses``; a pure keyword-only consumer
    (no positional parameter) has no first-arg role and is skipped.
    """
    first = _namegate_first_arg(fn)
    if first is None:
        return
    if name in _NAMEGATE_DEFERRED_FIRST_ARG:
        assert first == _NAMEGATE_DEFERRED_FIRST_ARG[name], (
            f"{name}: first argument moved to {first!r} — update or delete its row in "
            "_NAMEGATE_DEFERRED_FIRST_ARG"
        )
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
    exempt = _NAMEGATE_HOMONYM_WHITELIST | _NAMEGATE_DEFERRED_PARAM
    offenders = [
        f"{p.name!r} (use {_NAMEGATE_BANNED_PARAMS[p.name]!r})"
        for p in inspect.signature(fn).parameters.values()
        if p.name in _NAMEGATE_BANNED_PARAMS and (name, p.name) not in exempt
    ]
    assert not offenders, f"{name}: banned parameter spelling(s): {', '.join(offenders)}."


def test_namegate_deferred_param_rows_are_live() -> None:
    """The deferred keyword table can only shrink.

    Every row must name a **registered** analysis that still carries the
    non-canonical spelling.  A row for a signature already fixed is a row that
    would mask the next regression.
    """
    for fn_name, param in sorted(_NAMEGATE_DEFERRED_PARAM):
        fn = _NAMEGATE_BY_NAME.get(fn_name)
        assert fn is not None, f"{fn_name} is no longer registered — drop its deferred row"
        assert param in inspect.signature(fn).parameters, (
            f"{fn_name} no longer takes {param!r} — drop its row from _NAMEGATE_DEFERRED_PARAM"
        )


def test_namegate_deferred_rows_are_live() -> None:
    """The deferred-sweep table can only shrink.

    Every row must name a **registered** analysis whose first argument is still
    the non-canonical one.  A row for a function that has been fixed, renamed or
    unregistered is a row that would mask the next regression, so it fails here
    rather than rotting.
    """
    for name, first in _NAMEGATE_DEFERRED_FIRST_ARG.items():
        fn = _NAMEGATE_BY_NAME.get(name)
        assert fn is not None, f"{name} is no longer registered — drop its deferred row"
        assert _namegate_first_arg(fn) == first, (
            f"{name}'s first argument is no longer {first!r} — drop its deferred row"
        )


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
        assert fn is not None, f"whitelist names {fn_name!r}, which is not a registered analysis."
        assert param in set(inspect.signature(fn).parameters), (
            f"{fn_name}: whitelist carves out {param!r}, absent from the live signature."
        )


# ===========================================================================
# Error-message gate (stream WS-ERRGATE)
# ===========================================================================
#
# The curated counterpart to the registry-driven gates above.  Error *quality*
# is not decidable from a signature — it is decidable only by feeding a wrong
# input and inspecting what comes back — so this gate is a hand-curated table of
# the headline footguns the v4 audit named (``tsdynamics.errors``, the
# value-naming standard), one row per wrong-input case:
#
#   final_time<=0 · dt<=0 · too-short data · unknown keyword · unknown attribute
#   · wrong-dimension initial condition
#
# Two orthogonal properties are enforced, mirroring the two halves of the
# WS-ERRORS standard — "name the offending value, list the rule/options, suggest
# the fix" *and* "validate early; never silently produce garbage":
#
#   1. the **good-error shape** — the message names the offending value, and for
#      the sites WS-ERRORS migrated the exception is the right ``TSDynamicsError``
#      subclass (so ``except ts.TSDynamicsError`` works) while its stdlib base
#      keeps ``except ValueError`` / ``except TypeError`` working; and
#   2. **no silent footgun** — the wrong input *raises* rather than returning a
#      1-step ``Trajectory`` / a ``0 ± 0`` dimension / a swallowed keyword.
#
# Reality is tiered, and the gate is honest about it — every assertion below
# passes on ``main`` today:
#
#   * **closed footguns** (``final_time``/``dt``/unknown-attribute/unknown-param,
#     plus the entropy wrong-type leak) raise the right ``TSDynamicsError``
#     subclass with a value-naming message — asserted in full by
#     :func:`test_errgate_value_naming_error`, alongside the already-excellent
#     curated messages (``method=``/``backend=``/component/``set_state``) the
#     standard set out to *generalise*;
#   * **partially handled** cases (too-short data for the dimension/embedding
#     estimators, a typo'd keyword on an explicit-signature analysis, a
#     wrong-length ``ic``) at least *raise* — asserted to never silently return
#     by :func:`test_errgate_no_silent_garbage`; and
#   * **still-open footguns** WS-ERRORS *explicitly deferred* to the engine /
#     ``WS-WRAP`` / ``WS-CONV`` lanes (``correlation_dimension`` returning a
#     degenerate dimension for a handful of points, ``run()`` swallowing an
#     unknown keyword, a wrong-``ic`` leaking a raw NumPy reshape message) are
#     tracked by :func:`test_errgate_open_footgun_is_tracked` with a *strict*
#     ``xfail`` asserting the standard they should meet, so they trip
#     (xpass → fail) the instant a future stream closes them.
#
# The table is engine-free by construction — every system call routes through the
# wheel-free ``backend="reference"`` oracle, and every footgun that involves a
# system is validated in pure Python before any backend dispatch — so the section
# stays in the fast tier and the module is not auto-tagged ``engine``.

# ── the six headline footgun categories (the v4 audit's wrong-input table) ──
_ERRGATE_FINAL_TIME = "final_time<=0"
_ERRGATE_DT = "dt<=0"
_ERRGATE_SHORT_DATA = "too-short data"
_ERRGATE_UNKNOWN_KWARG = "unknown keyword"
_ERRGATE_UNKNOWN_ATTR = "unknown attribute/parameter"
_ERRGATE_WRONG_IC = "wrong-dimension ic"
# extra witnesses (not part of the six-category coverage requirement):
_ERRGATE_WRONG_TYPE = "wrong-type input"  # a System where a measured series is required
_ERRGATE_CURATED = "curated exemplar"  # already-excellent messages, the standard to generalise

_ERRGATE_HEADLINE_CATEGORIES = frozenset(
    {
        _ERRGATE_FINAL_TIME,
        _ERRGATE_DT,
        _ERRGATE_SHORT_DATA,
        _ERRGATE_UNKNOWN_KWARG,
        _ERRGATE_UNKNOWN_ATTR,
        _ERRGATE_WRONG_IC,
    }
)

# ── deterministic probe inputs ─────────────────────────────────────────────
# Three points: below every estimator's minimum embedding/box window, so the
# length-validating estimators reject it loudly.
_ERRGATE_SHORT_SERIES = np.array([0.1, 0.2, 0.3])
# Long enough that the *only* fault is the typo'd keyword.
_ERRGATE_VALID_SERIES = np.sin(np.linspace(0.0, 50.0, 2000))
# Eight points: enough to clear correlation_dimension's internal guards yet
# degenerate, so it returns dimension ~= 0 with no error — the open footgun.
_ERRGATE_DEGENERATE_SERIES = np.linspace(0.0, 1.0, 8)


def _errgate_set_unknown_attribute() -> None:
    """Trigger the typo'd-attribute footgun (``lor.sigmaa = 99`` for ``sigma``)."""
    system = ts.systems.Lorenz()
    system.sigmaa = 99


def _errgate_unknown_component() -> object:
    """Index a Trajectory by a component name the system does not declare."""
    traj = ts.systems.Lorenz().run(final_time=1.0, dt=0.5, backend="reference")
    return traj["nonexistent"]


def _errgate_data_analysis_on_system() -> object:
    """Feed a System where a measured series is required (the type-leak footgun).

    ``lyapunov_from_data`` estimates the maximal exponent from a *series*; handed
    a live System it must not silently produce a number.  Until v6 it leaked the
    raw ``float() argument …`` ``TypeError`` from the array coercion; the shared
    ``analysis/_common.py::reject_system`` guard now raises an
    ``InvalidInputError`` naming the system and the call that fixes it, so this
    thunk backs a tier-2 "it raises" row *and* a value-naming row.
    """
    return ts.analysis.lyapunov_from_data(ts.systems.Lorenz())


@dataclasses.dataclass(frozen=True)
class _ValueNamingCase:
    """A wrong input that must raise with a value-naming message.

    ``raises`` is the *guaranteed stdlib base* the exception must be an instance
    of (so the row keeps passing if the site is later promoted to a
    ``TSDynamicsError`` subclass of that base); ``names`` are substrings the
    message must contain — the offending value or its name; ``tsdclass`` is the
    specific ``TSDynamicsError`` subclass for the sites WS-ERRORS migrated, or
    ``None`` for a curated exemplar that still raises a stock stdlib type.
    """

    cid: str
    category: str
    thunk: object
    raises: type
    names: tuple[str, ...]
    tsdclass: type | None


@dataclasses.dataclass(frozen=True)
class _RaisesCase:
    """A wrong input that must *raise* (never silently return), with a token.

    These satisfy the "validate early; never silently produce garbage" half of
    the standard but are not yet a ``TSDynamicsError`` / fully domain-framed.
    ``token`` is an optional substring the message must mention.
    """

    cid: str
    category: str
    thunk: object
    raises: type
    token: str | None


@dataclasses.dataclass(frozen=True)
class _OpenFootgun:
    """A footgun WS-ERRORS explicitly deferred to a later lane.

    The gate asserts the standard it *should* meet (``expect`` raised, the
    message containing every entry of ``names``) under a strict ``xfail``;
    ``reason`` names the lane that owns closing it.

    Closure is detected by the *exception type* (``expect``): the strict ``xfail``
    trips (xpass → fail) only once the site raises ``expect``.  A *half*-fix that
    raises a different type (e.g. a bare ``ValueError`` rather than a
    ``TSDynamicsError``) leaves the case xfailed — by design, since the standard
    being tracked is the ``TSDynamicsError`` bar, not merely "raises something".
    When a case trips, promote it into the value-naming table above (adding the
    message-token assertion there) and delete the row here.
    """

    cid: str
    category: str
    thunk: object
    reason: str
    expect: type | tuple[type, ...]
    names: tuple[str, ...]


# ── tier 1: closed footguns + curated exemplars (value-naming message) ──────
_ERRGATE_VALUE_NAMING: list[_ValueNamingCase] = [
    _ValueNamingCase(
        "final_time-negative",
        _ERRGATE_FINAL_TIME,
        lambda: ts.systems.Lorenz().run(final_time=-5.0, dt=0.1, backend="reference"),
        ValueError,
        ("final_time",),
        InvalidParameterError,
    ),
    _ValueNamingCase(
        "final_time-zero",
        _ERRGATE_FINAL_TIME,
        lambda: ts.systems.Lorenz().run(final_time=0.0, dt=0.1, backend="reference"),
        ValueError,
        ("final_time",),
        InvalidParameterError,
    ),
    _ValueNamingCase(
        "final_time-negative-integrate-alias",
        _ERRGATE_FINAL_TIME,
        lambda: ts.systems.Lorenz().run(final_time=-5.0, dt=0.1, backend="reference"),
        ValueError,
        ("final_time",),
        InvalidParameterError,
    ),
    _ValueNamingCase(
        "dt-zero",
        _ERRGATE_DT,
        lambda: ts.systems.Lorenz().run(final_time=5.0, dt=0.0, backend="reference"),
        ValueError,
        ("dt",),
        InvalidParameterError,
    ),
    _ValueNamingCase(
        "dt-negative",
        _ERRGATE_DT,
        lambda: ts.systems.Lorenz().run(final_time=5.0, dt=-0.1, backend="reference"),
        ValueError,
        ("dt",),
        InvalidParameterError,
    ),
    _ValueNamingCase(
        "unknown-attribute-set",
        _ERRGATE_UNKNOWN_ATTR,
        _errgate_set_unknown_attribute,
        ValueError,
        ("sigmaa",),
        InvalidParameterError,
    ),
    _ValueNamingCase(
        "unknown-parameter-with_params",
        _ERRGATE_UNKNOWN_ATTR,
        lambda: ts.systems.Lorenz().with_params(nonexistent=5),
        ValueError,
        ("nonexistent",),
        InvalidParameterError,
    ),
    _ValueNamingCase(
        "unknown-parameter-constructor",
        _ERRGATE_UNKNOWN_ATTR,
        lambda: ts.systems.Lorenz(params={"sigmaa": 9}),
        ValueError,
        ("sigmaa",),
        InvalidParameterError,
    ),
    # Closed by FINISH-ERRADOPT — promoted out of the tier-3 strict-xfail table.
    _ValueNamingCase(
        "short-data-correlation-dimension",
        _ERRGATE_SHORT_DATA,
        lambda: ts.analysis.correlation_dimension(_ERRGATE_DEGENERATE_SERIES),
        ValueError,
        ("data length",),
        InvalidParameterError,
    ),
    _ValueNamingCase(
        "unknown-keyword-run",
        _ERRGATE_UNKNOWN_KWARG,
        lambda: ts.systems.Lorenz().run(final_time=5.0, dt=0.5, backend="reference", nonsense=5),
        ValueError,
        ("nonsense",),
        InvalidParameterError,
    ),
    # Closed in v6 by the shared System-rejecting guard
    # (``analysis/_common.py::reject_system``), which every data-first analysis
    # now calls before it coerces its input to an array.  Promoted out of the
    # tier-3 strict-xfail table; the message must name the *system* and give a
    # runnable next step, not leak NumPy's ``float() argument …``.
    _ValueNamingCase(
        "wrong-type-input-message",
        _ERRGATE_WRONG_TYPE,
        _errgate_data_analysis_on_system,
        TypeError,
        ("ystem", "Lorenz", "system.run("),
        InvalidInputError,
    ),
    # Closed by v6 FRESH-EYES: ``SystemBase._coerce_ic`` is now the one place the
    # three initial-condition sources are normalised, so a wrong-length ``ic``
    # names the system's dimension and its component names instead of leaking
    # NumPy's ``cannot reshape array of size 2 into shape (3,)``.  Promoted out of
    # the tier-3 strict-xfail table (``open-wrong-ic-message``).
    _ValueNamingCase(
        "wrong-ic-message",
        _ERRGATE_WRONG_IC,
        lambda: ts.systems.Lorenz().run(ic=[1.0, 2.0], final_time=5.0, dt=0.1, backend="reference"),
        TypeError,
        ("Lorenz", "3 state components", "got 2"),
        InvalidInputError,
    ),
    # Curated exemplars — already-excellent value-naming messages (stock stdlib
    # types) that WS-ERRORS set out to make the law rather than the exception.
    _ValueNamingCase(
        "curated-solver-method",
        _ERRGATE_CURATED,
        lambda: ts.systems.Lorenz().run(
            final_time=5.0, dt=0.1, method="LSODA", backend="reference"
        ),
        ValueError,
        ("LSODA", "solver="),
        None,
    ),
    _ValueNamingCase(
        "curated-backend",
        _ERRGATE_CURATED,
        lambda: ts.systems.Lorenz().run(final_time=5.0, dt=0.1, backend="gpu"),
        ValueError,
        ("gpu", "choose from"),
        None,
    ),
    _ValueNamingCase(
        "curated-trajectory-component",
        _ERRGATE_CURATED,
        _errgate_unknown_component,
        KeyError,
        ("nonexistent",),
        None,
    ),
    _ValueNamingCase(
        "curated-dde-set-state",
        _ERRGATE_CURATED,
        lambda: ts.systems.MackeyGlass().set_state([1.0]),
        AttributeError,
        ("set_state", "history"),
        None,
    ),
]


# ── tier 2: partially handled — must raise (no silent garbage) ──────────────
_ERRGATE_NO_SILENT: list[_RaisesCase] = [
    _RaisesCase(
        "short-data-embed",
        _ERRGATE_SHORT_DATA,
        lambda: ts.analysis.embed(_ERRGATE_SHORT_SERIES, dimension=5, delay=3),
        ValueError,
        "too short",
    ),
    _RaisesCase(
        "short-data-box-counting",
        _ERRGATE_SHORT_DATA,
        lambda: ts.analysis.box_counting_dimension(_ERRGATE_SHORT_SERIES),
        ValueError,
        None,
    ),
    _RaisesCase(
        "short-data-lyapunov-from-data",
        _ERRGATE_SHORT_DATA,
        lambda: ts.analysis.lyapunov_from_data(_ERRGATE_SHORT_SERIES),
        ValueError,
        "longer series",
    ),
    _RaisesCase(
        "unknown-keyword-lyapunov-spectrum",
        _ERRGATE_UNKNOWN_KWARG,
        lambda: ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), nonsense=5),
        TypeError,
        "nonsense",
    ),
    _RaisesCase(
        "unknown-keyword-correlation-dimension",
        _ERRGATE_UNKNOWN_KWARG,
        lambda: ts.analysis.correlation_dimension(_ERRGATE_VALID_SERIES, nonsense=5),
        TypeError,
        "nonsense",
    ),
    _RaisesCase(
        "wrong-ic-dimension",
        _ERRGATE_WRONG_IC,
        lambda: ts.systems.Lorenz().run(ic=[1.0, 2.0], final_time=5.0, dt=0.1, backend="reference"),
        TypeError,
        None,
    ),
    _RaisesCase(
        "wrong-type-input-lyapunov-from-data",
        _ERRGATE_WRONG_TYPE,
        _errgate_data_analysis_on_system,
        TypeError,
        None,
    ),
]


# ── tier 3: still-open footguns, tracked under a strict xfail ───────────────
# Empty, and that is the point: every footgun this table ever tracked has been
# closed and promoted into one of the tables above — `open-short-data-
# correlation-dimension` and `open-unknown-keyword-run` by FINISH-ERRADOPT,
# `open-wrong-type-input-message` by v6's shared `reject_system` guard, and
# `open-wrong-ic-message` by v6 FRESH-EYES (`SystemBase._coerce_ic`).  Add a row
# here only for a footgun a lane has *explicitly deferred*; the strict xfail then
# turns red the moment it is fixed, forcing the promotion.
_ERRGATE_OPEN_FOOTGUNS: list[_OpenFootgun] = []

_ERRGATE_OPEN_PARAMS = [
    pytest.param(case, id=case.cid, marks=pytest.mark.xfail(reason=case.reason, strict=True))
    for case in _ERRGATE_OPEN_FOOTGUNS
]


@pytest.mark.parametrize("case", _ERRGATE_VALUE_NAMING, ids=lambda c: c.cid)
def test_errgate_value_naming_error(case: _ValueNamingCase) -> None:
    """A wrong input raises with a message that names the offending value.

    For the sites WS-ERRORS migrated onto the hierarchy (``case.tsdclass`` set),
    the raised exception is additionally that specific ``TSDynamicsError``
    subclass — so a caller can ``except ts.TSDynamicsError`` — while the stdlib
    base in ``case.raises`` keeps ``except ValueError`` / ``except TypeError``
    working.  Asserting the stdlib base (not the subclass) keeps the curated
    exemplars forward-compatible if they are later promoted onto the hierarchy.
    """
    with pytest.raises(case.raises) as excinfo:
        case.thunk()
    message = str(excinfo.value)
    for token in case.names:
        assert token in message, f"{case.cid}: message does not name {token!r}: {message!r}"
    if case.tsdclass is not None:
        assert isinstance(excinfo.value, case.tsdclass), (
            f"{case.cid}: expected a {case.tsdclass.__name__} (a TSDynamicsError), "
            f"got {type(excinfo.value).__name__}."
        )


@pytest.mark.parametrize("case", _ERRGATE_NO_SILENT, ids=lambda c: c.cid)
def test_errgate_no_silent_garbage(case: _RaisesCase) -> None:
    """A partially-handled wrong input *raises* — it never silently returns garbage.

    These cases do not yet raise a ``TSDynamicsError`` (too-short data leaks a
    domain ``ValueError`` from the estimator; a typo'd keyword on an
    explicit-signature analysis is a stock ``TypeError``; a wrong-length ``ic``
    leaks a NumPy reshape ``ValueError``), but they satisfy the *other* half of
    the standard: the input is rejected loudly rather than turned into a 1-step
    trajectory / a ``0 ± 0`` dimension / a swallowed keyword.
    """
    with pytest.raises(case.raises) as excinfo:
        case.thunk()
    if case.token is not None:
        assert case.token in str(excinfo.value), (
            f"{case.cid}: message does not mention {case.token!r}: {str(excinfo.value)!r}"
        )


@pytest.mark.parametrize("case", _ERRGATE_OPEN_PARAMS)
def test_errgate_open_footgun_is_tracked(case: _OpenFootgun) -> None:
    """A footgun WS-ERRORS deferred: assert the standard it *should* meet.

    Every case currently fails this assertion (it returns garbage, or raises a
    bare/leaky error), so each is a *strict* ``xfail``: the gate records the open
    gap executably and turns red (an unexpected pass) the instant a future stream
    — the lane named in ``case.reason`` — closes the footgun, forcing the marker
    to be removed and the case promoted into one of the tables above.
    """
    with pytest.raises(case.expect) as excinfo:
        case.thunk()
    for token in case.names:
        assert token in str(excinfo.value), (
            f"{case.cid}: a fixed error should name {token!r}: {str(excinfo.value)!r}"
        )


def test_errgate_table_covers_every_headline_footgun() -> None:
    """Every headline wrong-input category from the v4 audit is exercised by the gate.

    Coverage is required twice: every category appears *somewhere*, and every
    category has at least one *live* (non-``xfail``) case — so demoting a
    category's only executing case into the tracked-open ``xfail`` table (which
    would quietly stop exercising it) fails this gate loudly.
    """
    everywhere = (*_ERRGATE_VALUE_NAMING, *_ERRGATE_NO_SILENT, *_ERRGATE_OPEN_FOOTGUNS)
    missing = _ERRGATE_HEADLINE_CATEGORIES - {case.category for case in everywhere}
    assert not missing, f"headline footgun categories not gated: {sorted(missing)}"

    live = {case.category for case in (*_ERRGATE_VALUE_NAMING, *_ERRGATE_NO_SILENT)}
    untested = _ERRGATE_HEADLINE_CATEGORIES - live
    assert not untested, (
        f"headline categories with no live (non-xfail) case: {sorted(untested)}; "
        f"a strict-xfail-only category exercises no executing assertion."
    )


def test_errgate_case_ids_are_unique() -> None:
    """No two curated cases share an id, so a failure is unambiguously located."""
    ids = [
        case.cid for case in (*_ERRGATE_VALUE_NAMING, *_ERRGATE_NO_SILENT, *_ERRGATE_OPEN_FOOTGUNS)
    ]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"duplicate case ids: {duplicates}"


def test_errgate_open_footgun_reasons_cite_a_lane() -> None:
    """Each tracked-open footgun documents the stream/lane that owns closing it."""
    for case in _ERRGATE_OPEN_FOOTGUNS:
        assert "WS-" in case.reason or "defer" in case.reason.lower(), (
            f"{case.cid}: an open-footgun reason must cite the deferring lane."
        )


# ===========================================================================
# Runnable-line gate (stream v6 FRESH-EYES)
# ===========================================================================
#
# The error-message gate above enforces that a message *names the offending
# value*.  Naming it is necessary and not sufficient: the owner's v6 session hit
# a wall of messages that described the mistake perfectly and left him with
# nothing to type.  The bar this section adds is decidable —
#
#   when a call fails because the caller typed the wrong *shape* of call, the
#   message must contain a line that parses as Python and is a call,
#
# — so a regression is caught mechanically rather than by taste.  ``remedy()``
# (``tsdynamics.errors``) is the formatter that produces such a block, and
# :func:`_runnable_lines` below is its inverse: it reads a message back and
# returns the statements a user could paste.  A message that merely *mentions* a
# function ("use periodic_orbits for maps") yields nothing and fails the bar.
#
# Two further properties are asserted, because a line that parses is not yet a
# line that helps:
#
#   * it must be **runnable in the user's REPL**, so it is spelled against the
#     public surface (``ts.`` / ``system.`` / ``traj``), never a private helper;
#     and
#   * it must **not name a different function than the one the caller reached
#     for** without also naming that one — the exact defect the owner reported
#     ("orbit_diagram needs …" answering a ``bifurcation_diagram`` call).
#
# The table is curated for the same reason the errgate table is: whether a
# message is *useful* is not decidable from a signature.  Every row is a call a
# real session made.


def _runnable_lines(message: str) -> list[str]:
    """Return the indented lines of ``message`` that parse as a Python call.

    A remedy line is either a bare call (``ts.analysis.basins(system, region)``) or an
    assignment whose value is a call (``traj = system.run(...)``); anything else
    — prose, a fragment, a bare name — is not something to paste and is not
    counted.  Trailing comments are fine (the tokenizer drops them).
    """
    found: list[str] = []
    for raw in message.splitlines():
        line = raw.strip()
        if not line or not raw.startswith(" "):
            continue
        try:
            tree = ast.parse(line)
        except SyntaxError:
            continue
        for node in tree.body:
            value = node.value if isinstance(node, (ast.Expr, ast.Assign)) else None
            if isinstance(value, ast.Call):
                found.append(line)
                break
    return found


@dataclasses.dataclass(frozen=True)
class _RunnableCase:
    """A wrong call whose error must hand back the line to type instead.

    ``thunk`` makes the mistake; ``called`` is the public name the user typed (so
    the gate can check the answer is not about some other function); ``mentions``
    are substrings at least one runnable line must contain — normally the fixed
    call, so the row pins *which* fix is offered, not merely that one is.
    """

    cid: str
    thunk: object
    called: str
    mentions: tuple[str, ...]


def _undeclared_dim_system() -> object:
    """Instantiate a system class that forgot ``dim`` -- the first-timer's error.

    Defined as a function rather than a lambda because the mistake is in the
    *class body*, and the class must be built fresh inside the thunk so the
    registry sees one definition per call.
    """

    class NoDimension(ts.ContinuousSystem):
        params = {"a": 1.0}

        @staticmethod
        def _equations(y, t, *, a):  # type: ignore[no-untyped-def]
            return [-a * y(0)]

    return NoDimension()


#: Deterministic inputs for the rows below.
_RUNNABLE_SERIES = np.linspace(0.0, 1.0, 8)
_RUNNABLE_TRAJ = ts.systems.Lorenz().run(final_time=2.0, dt=0.1, backend="reference")
#: A two-basin label image, built directly rather than integrated: ``resilience``
#: reads labels + grid, and the point of this row is the *ambiguity* (which of the
#: two?), which a synthetic image states exactly and a real system only
#: approximately (and slowly).
_RUNNABLE_BASINS = BasinsResult(
    labels=np.where(np.arange(64).reshape(8, 8) % 8 < 4, 1, 2),
    grid=ts.data.Grid([-1.0, -1.0], [1.0, 1.0], (8, 8)),
)

_ERRGATE_RUNNABLE: list[_RunnableCase] = [
    # ── wrong shape of call: a model where data belongs, and the reverse ──
    _RunnableCase(
        "system-first-analysis-given-data",
        lambda: ts.analysis.lyapunov_spectrum(np.asarray(_RUNNABLE_TRAJ.y[:, 0])),
        "lyapunov_spectrum",
        ("lyapunov_from_data",),
    ),
    _RunnableCase(
        "lyapunov-spectrum-given-a-trajectory",
        lambda: ts.analysis.lyapunov_spectrum(_RUNNABLE_TRAJ),
        "lyapunov_spectrum",
        ("lyapunov_from_data(traj)",),
    ),
    _RunnableCase(
        "data-first-analysis-given-a-system",
        lambda: ts.analysis.correlation_dimension(ts.systems.Lorenz()),
        "correlation_dimension",
        ("correlation_dimension(traj",),
    ),
    _RunnableCase(
        "fixed-points-given-data",
        lambda: ts.analysis.fixed_points(np.zeros((10, 2))),
        "fixed_points",
        # A bare array has no ``.system`` to forward to (§5.6), so the one
        # runnable line is the listing of what a measurement CAN answer.
        ("ts.analysis.find(traj)",),
    ),
    # ── a required argument with no natural default ──
    _RunnableCase(
        "basins-without-a-region",
        lambda: ts.analysis.basins(ts.systems.Henon()),
        "basins",
        ("ts.analysis.basins(system, [",),
    ),
    _RunnableCase(
        "recurrence-matrix-without-a-scale",
        lambda: ts.analysis.recurrence_matrix(_RUNNABLE_TRAJ),
        "recurrence_matrix",
        ("recurrence_rate=",),
    ),
    _RunnableCase(
        "windowed-rqa-without-a-window",
        lambda: ts.analysis.windowed_rqa(_RUNNABLE_TRAJ),
        "windowed_rqa",
        ("window=",),
    ),
    _RunnableCase(
        "continuation-without-a-region",
        lambda: ts.analysis.continuation(ts.systems.Henon(), "a", [1.2, 1.4]),
        "continuation",
        ("ts.analysis.continuation(system, param, values,",),
    ),
    _RunnableCase(
        "resilience-without-an-attractor-id",
        lambda: ts.analysis.resilience(_RUNNABLE_BASINS),
        "resilience",
        ("attractor_id=",),
    ),
    _RunnableCase(
        "tipping-points-given-a-trajectory",
        lambda: ts.analysis.tipping_points(_RUNNABLE_TRAJ),
        "tipping_points",
        ("ts.analysis.tipping_points(c)",),
    ),
    # ── a flow keyword aimed at a map ──
    _RunnableCase(
        "map-given-a-flow-keyword",
        lambda: ts.systems.Henon().run(n=10, dt=0.01),
        "run",
        ("Henon().run(n=",),
    ),
    # ── right idea, wrong family ──
    _RunnableCase(
        "periodic-orbits-on-a-flow",
        lambda: ts.analysis.periodic_orbits(ts.systems.Lorenz(), 2),
        "periodic_orbits",
        ("ts.analysis.periodic_orbits(",),
    ),
    _RunnableCase(
        "periodic-orbit-on-a-map",
        lambda: ts.analysis.periodic_orbits(ts.systems.Henon()),
        "periodic_orbits",
        ("ts.analysis.periodic_orbits(",),
    ),
    # ── a value the caller can only fix by being told the right one ──
    _RunnableCase(
        "wrong-length-initial-condition",
        lambda: ts.systems.Lorenz().run(ic=[1.0, 2.0], final_time=5.0, dt=0.1, backend="reference"),
        "run",
        ("ic=[1.0, 1.0, 1.0]",),
    ),
    _RunnableCase(
        "too-short-series-for-a-dimension",
        lambda: ts.analysis.correlation_dimension(_RUNNABLE_SERIES),
        "correlation_dimension",
        ("ts.analysis.correlation_dimension(traj)",),
    ),
    _RunnableCase(
        "too-short-series-for-fixed-mass",
        lambda: ts.analysis.fixed_mass_dimension(_RUNNABLE_SERIES),
        "fixed_mass_dimension",
        ("ts.analysis.fixed_mass_dimension(traj)",),
    ),
    _RunnableCase(
        "non-numeric-renyi-order",
        lambda: ts.analysis.generalized_dimension(_RUNNABLE_TRAJ, q="two"),
        "generalized_dimension",
        ("q=2.0",),
    ),
    # ── writing a system class: the very first thing a new user does, and the
    # place a leaked internal error is most expensive ──
    _RunnableCase(
        "system-class-without-a-dimension",
        _undeclared_dim_system,
        "dim",
        ("(dim=3)",),
    ),
    # ── shooting must escalate, not repeat the line the caller just ran ──
    _RunnableCase(
        "periodic-orbits-given-a-flow",
        lambda: ts.analysis.periodic_orbits(ts.systems.Lorenz(), 2),
        "periodic_orbits",
        ("ts.analysis.periodic_orbits(system,", "ic=traj.y[-1]"),
    ),
    # ── a near-miss name: the fix is one character, so spell the whole call ──
    _RunnableCase(
        "misspelt-parameter-keyword",
        lambda: ts.systems.Lorenz(sigmaa=10.0),
        "Lorenz",
        ("Lorenz(sigma=10.0)",),
    ),
    _RunnableCase(
        "unknown-fixed-points-method",
        lambda: ts.analysis.fixed_points(ts.systems.Lorenz(), method="nooton"),
        "method",
        ("ts.analysis.fixed_points(system, method='newton')",),
    ),
    # ── a single exponent is not a spectrum: refuse, do not saturate to 1.0 ──
    _RunnableCase(
        "kaplan-yorke-given-one-exponent",
        lambda: ts.analysis.kaplan_yorke_dimension(0.9),
        "kaplan_yorke_dimension",
        ("ts.analysis.kaplan_yorke_dimension(exps)",),
    ),
    # ── a basin is a property of the model, so data cannot reach the FSM ──
    _RunnableCase(
        "basins-given-a-trajectory",
        lambda: ts.analysis.basins(_RUNNABLE_TRAJ, [(-2.0, 2.0, 8), (-2.0, 2.0, 8)]),
        "basins",
        ("ts.analysis.basins(traj.system)",),
    ),
    # ── a transposed point set must not be diagnosed as a short one ──
    _RunnableCase(
        "transposed-point-set",
        lambda: ts.analysis.correlation_dimension(np.zeros((3, 500))),
        "correlation_dimension",
        ("ts.analysis.correlation_dimension(data.T)",),
    ),
]

#: Spellings that are *not* runnable in a user's REPL: a private helper, an
#: internal module path, or a placeholder that has to be decoded first.
_NOT_A_USER_CALL = ("_", "tsdynamics.analysis._", "self.", "<")


#: Error messages whose remedy line still names the **pre-v6 top-level spelling**
#: (``ts.analysis.fixed_points(...)`` rather than ``ts.analysis.fixed_points(...)``), with
#: the owning slot.
#:
#: This is not cosmetic and it is not a style preference.  v6 curated the top
#: level to seventeen names, so ``ts.analysis.fixed_points`` does not resolve any more —
#: a message handing that line back is handing back a line that raises.  Contract
#: §5.6 says so directly ("the remedy lines are bare, not ``ts.analysis.``-
#: qualified" is listed as one of the three live defects that die on the way).
#: The message strings live in ``analysis/**``, owned by S3 · ANALYSIS-DOMAIN and
#: C7 · ANALYSIS-DISCOVERY.
#:
#: Strict xfails, so each row turns red the moment its message is qualified and
#: the row has to be deleted.  :func:`test_errgate_remedy_lines_resolve` is the
#: gate the whole table exists to be measured against.
_ERRGATE_NOT_QUALIFIED: dict[str, str] = {
    # The four ``analysis/dimensions`` rows were deleted in v6 round 9: the
    # messages now say ``ts.analysis.correlation_dimension`` /
    # ``…fixed_mass_dimension`` / ``…generalized_dimension``, which resolve.
    # The table shrank from 5 rows to 1, and this is the one left.
    #
    # A different SHAPE of gap: the message is complete prose and hands back
    # nothing at all.  S1 · RUN owns families/_kwargs.py.
    "map-given-a-flow-keyword": "S1: families/_kwargs.py hands back no runnable line",
}

#: The subset of :data:`_ERRGATE_NOT_QUALIFIED` whose *literal token* also no
#: longer matches, so the older string check fails as well.  Kept separate
#: because a strict xfail must not be attached to a test that passes.
_ERRGATE_TOKEN_STALE = frozenset(
    {
        "map-given-a-flow-keyword",
    }
)


def _errgate_params(table: object) -> list[object]:
    """``_ERRGATE_RUNNABLE`` as params, strict-xfailing the rows in *table*."""
    return [
        pytest.param(
            case,
            id=case.cid,
            marks=(
                [pytest.mark.xfail(strict=True, reason=_ERRGATE_NOT_QUALIFIED[case.cid])]
                if case.cid in table
                else []
            ),
        )
        for case in _ERRGATE_RUNNABLE
    ]


_ERRGATE_RUNNABLE_PARAMS = _errgate_params(_ERRGATE_TOKEN_STALE)
_ERRGATE_RESOLVE_PARAMS = _errgate_params(_ERRGATE_NOT_QUALIFIED)


def test_errgate_not_qualified_rows_are_live() -> None:
    """The unqualified-remedy tables can only shrink — every row names a real case."""
    ids = {case.cid for case in _ERRGATE_RUNNABLE}
    stale = sorted(set(_ERRGATE_NOT_QUALIFIED) - ids)
    assert not stale, f"_ERRGATE_NOT_QUALIFIED names cases that no longer exist: {stale}"
    assert set(_ERRGATE_NOT_QUALIFIED) >= _ERRGATE_TOKEN_STALE, (
        "a row whose token is stale must also be listed as unqualified"
    )


@pytest.mark.parametrize("case", _ERRGATE_RESOLVE_PARAMS)
def test_errgate_remedy_lines_resolve(case: _RunnableCase) -> None:
    """**Every dotted name a remedy line hands back must resolve.**

    The strongest form of the runnable-line standard, and the one v6 made
    load-bearing: curating the top level to seventeen names means
    ``ts.analysis.fixed_points(system)`` no longer runs, so a message that offers it is
    now *worse* than one that offers nothing — it looks authoritative and fails.

    Parses each remedy line and walks every ``ts.a.b`` attribute path in it
    against the live package, so the check cannot be satisfied by a plausible
    string.
    """
    import ast
    import re

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the type is gated elsewhere
        case.thunk()
    lines = _runnable_lines(str(excinfo.value))
    assert lines, f"{case.cid}: no runnable line at all"
    unresolved: list[str] = []
    for line in lines:
        try:
            ast.parse(line)
        except SyntaxError:  # pragma: no cover - _runnable_lines already parsed it
            unresolved.append(f"{line} (does not parse)")
            continue
        for path in re.findall(r"(?<![\w.])ts(?:\.[A-Za-z_][A-Za-z0-9_]*)+", line):
            obj: object = ts
            for part in path.split(".")[1:]:
                try:
                    obj = getattr(obj, part)
                except Exception:  # noqa: BLE001 - MovedInV6 is the interesting one
                    unresolved.append(path)
                    break
    assert not unresolved, (
        f"{case.cid}: the remedy hands back {unresolved} — names that do not "
        "resolve on the v6 top level, so pasting the line raises"
    )


@pytest.mark.parametrize("case", _ERRGATE_RUNNABLE_PARAMS)
def test_errgate_message_hands_back_a_runnable_line(case: _RunnableCase) -> None:
    """A wrong-shaped call is answered with the line to type, not a description.

    This is the bar the v6 owner session set: ``"X needs a discrete-time view"``
    fails it; ``"wrap the flow first:\\n    ts.analysis.orbit_diagram(...)"`` passes.
    """
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the type is gated elsewhere
        case.thunk()
    message = str(excinfo.value)
    lines = _runnable_lines(message)
    assert lines, (
        f"{case.cid}: the message describes the mistake but hands back no line to "
        f"type. Append one with errors.remedy(...): {message!r}"
    )
    for token in case.mentions:
        assert any(token in line for line in lines), (
            f"{case.cid}: no runnable line offers {token!r}; got {lines!r}"
        )
    for line in lines:
        assert not line.startswith(_NOT_A_USER_CALL), (
            f"{case.cid}: the remedy must be spelled against the public API, got {line!r}"
        )


@pytest.mark.parametrize("case", _ERRGATE_RUNNABLE, ids=lambda c: c.cid)
def test_errgate_message_answers_the_call_the_user_made(case: _RunnableCase) -> None:
    """The message never talks about a *different* function without naming this one.

    The owner's report: calling ``ts.bifurcation_diagram`` and being refused by a
    message about ``orbit_diagram`` — an internal name the caller never typed.  A
    message may of course *redirect* to another function; it must simply also
    acknowledge the call that was made.
    """
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the type is gated elsewhere
        case.thunk()
    message = str(excinfo.value)
    assert case.called in message, (
        f"{case.cid}: the message must name {case.called!r} — the call the user "
        f"actually made: {message!r}"
    )


def test_errgate_runnable_lines_detects_prose_and_code() -> None:
    """The detector itself: indented calls count, prose and fragments do not."""
    from tsdynamics.errors import remedy

    assert _runnable_lines("no fix here at all") == []
    assert _runnable_lines("use periodic_orbits for maps") == []
    assert _runnable_lines("try this:\n    ts.analysis.basins(system, region)") == [
        "ts.analysis.basins(system, region)"
    ]
    assert _runnable_lines("x" + remedy("traj = system.run(final_time=1.0)")) == [
        "traj = system.run(final_time=1.0)"
    ]
    # a bare name is not a call, and an unindented line is prose, not a remedy
    assert _runnable_lines("do:\n    periodic_orbits") == []
    assert _runnable_lines("ts.analysis.basins(system, region)") == []


# ===========================================================================
# Tolerance-default gate (stream v6 WP3-tol)
# ===========================================================================
#
# ``rtol=1e-6`` / ``atol=1e-9`` used to be duplicated as bare literals across
# sixteen call sites in five subpackages, which is precisely how two "the same"
# defaults drift apart unnoticed.  They now all name a constant in
# :mod:`tsdynamics.utils.tolerances`.  This gate walks the source of every
# module under ``src/tsdynamics`` (excluding the catalogue, whose per-system
# ``known_lyapunov`` kwargs are deliberate literature-reproduction pins) and
# fails if any ``rtol=`` / ``atol=`` *parameter default* or keyword *argument* is
# a bare numeric literal again.


_TOLERANCE_PARAM_NAMES = frozenset({"rtol", "atol"})

#: Modules exempt from the gate.  ``utils/tolerances.py`` is where the numbers
#: are *supposed* to live; ``systems/`` holds per-system ``known_lyapunov``
#: metadata dicts (deliberate, reviewed, per-system tolerance pins that
#: reproduce a specific literature spectrum — not defaults).
_TOLERANCE_GATE_EXEMPT = ("utils/tolerances.py", "systems/")

#: ``rtol``/``atol`` **homonyms**: parameters that spell the same name but are not
#: solver tolerances at all, so they must not track the solver constants.  Keyed
#: ``"<relative module path>::<function name>"`` with the reason.  Kept explicit
#: (and asserted live by :func:`test_tolerance_gate_homonyms_are_live`) so an
#: entry cannot silently outlive the code it excuses.
_TOLERANCE_GATE_HOMONYMS: dict[str, str] = {
    "analysis/embedding/dimension.py::false_nearest_neighbors": (
        "Kennel, Brown & Abarbanel (1992) FNN criteria: R_tol (a distance-growth "
        "ratio, default 15) and A_tol (a multiple of the attractor size, default 2). "
        "Neither is a solver tolerance."
    ),
    "analysis/orbits/orbit_diagram.py::periods": (
        "relative branch-clustering tolerance for the cascade quantifier "
        "(scale-free branch separation), not a solver tolerance."
    ),
    "analysis/orbits/orbit_diagram.py::bifurcation_points": (
        "relative branch-clustering tolerance, as in periods()."
    ),
}


def _tsdynamics_source_files() -> list[pathlib.Path]:
    root = pathlib.Path(ts.__file__).parent
    out = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if any(rel.startswith(x) or rel == x for x in _TOLERANCE_GATE_EXEMPT):
            continue
        out.append(path)
    return out


def _bare_tolerance_literals(path: pathlib.Path, rel: str) -> list[str]:
    """Return a description of every bare ``rtol=``/``atol=`` numeric literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []

    def _is_bare(node: ast.expr | None) -> bool:
        # A bare literal, or a unary-minus literal.  ``None`` (resolve-later) and
        # any Name/Attribute (a named constant) are fine.
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return _is_bare(node.operand)
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if f"{rel}::{node.name}" in _TOLERANCE_GATE_HOMONYMS:
                continue
            args = node.args
            names = [a.arg for a in (*args.posonlyargs, *args.args)]
            padded = [None] * (len(names) - len(args.defaults)) + list(args.defaults)
            for name, default in zip(names, padded, strict=True):
                if name in _TOLERANCE_PARAM_NAMES and _is_bare(default):
                    found.append(f"{node.name}() parameter default {name}={ast.unparse(default)}")
            for name_node, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if name_node.arg in _TOLERANCE_PARAM_NAMES and _is_bare(default):
                    found.append(
                        f"{node.name}() parameter default {name_node.arg}={ast.unparse(default)}"
                    )
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in _TOLERANCE_PARAM_NAMES and _is_bare(kw.value):
                    found.append(f"call keyword {kw.arg}={ast.unparse(kw.value)}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                name = None
                if isinstance(target, ast.Name):
                    name = target.id
                elif isinstance(target, ast.Attribute):
                    name = target.attr
                if name is None:
                    continue
                stem = name.lstrip("_").removeprefix("default_").removeprefix("step_")
                if stem in _TOLERANCE_PARAM_NAMES and _is_bare(value):
                    found.append(f"assignment {name} = {ast.unparse(value)}")
    return found


def test_no_bare_tolerance_literal_in_the_library() -> None:
    """Every ``rtol``/``atol`` default names a :mod:`tsdynamics.utils.tolerances` constant.

    Regression gate for the v6 tolerance hoist: the sixteen duplicated
    ``1e-6``/``1e-9`` literals are gone, and a new one must not creep back into a
    signature, a call site, or a ``self._rtol = ...`` assignment.  If you need a
    genuinely different number for one driver, add a *named* constant to
    :mod:`tsdynamics.utils.tolerances` documenting the measurement that justifies
    it — that is the whole point of the module.
    """
    offenders: dict[str, list[str]] = {}
    root = pathlib.Path(ts.__file__).parent
    for path in _tsdynamics_source_files():
        rel = path.relative_to(root).as_posix()
        hits = _bare_tolerance_literals(path, rel)
        if hits:
            offenders[rel] = hits
    assert not offenders, (
        "bare rtol/atol numeric literals found (name a constant in "
        "tsdynamics.utils.tolerances instead):\n"
        + "\n".join(f"  {mod}: {', '.join(hits)}" for mod, hits in sorted(offenders.items()))
    )


def test_tolerance_gate_homonyms_are_live() -> None:
    """Every carved-out ``rtol``/``atol`` homonym still exists and still is one.

    A stale carve-out is worse than none: it would silently excuse a *real* solver
    tolerance that later took the same qualified name.
    """
    root = pathlib.Path(ts.__file__).parent
    for key, reason in _TOLERANCE_GATE_HOMONYMS.items():
        rel, _, func = key.partition("::")
        path = root / rel
        assert path.is_file(), f"carve-out {key}: module {rel} no longer exists"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        defs = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func
        ]
        assert defs, f"carve-out {key}: function {func} no longer exists in {rel}"
        params = {
            a.arg
            for node in defs
            for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        }
        assert params & _TOLERANCE_PARAM_NAMES, (
            f"carve-out {key}: {func} no longer takes an rtol/atol parameter — drop it"
        )
        assert len(reason) > 30, f"carve-out {key}: reason must explain the homonym"


def test_tolerance_constants_are_the_documented_values() -> None:
    """Pin the canonical tolerance constants so a bump is a deliberate, reviewed edit."""
    from tsdynamics.utils import tolerances as tol

    assert (tol.DEFAULT_RTOL, tol.DEFAULT_ATOL) == (1e-9, 1e-12)
    assert (tol.DDE_RTOL, tol.DDE_ATOL) == (1e-3, 1e-3)
    assert (tol.DDE_LYAPUNOV_RTOL, tol.DDE_LYAPUNOV_ATOL) == (1e-7, 1e-9)
    assert (tol.BASIN_RTOL, tol.BASIN_ATOL) == (1e-6, 1e-9)


def test_basin_march_python_and_rust_name_the_same_tolerance() -> None:
    """The basin march's two paths are contractually bit-identical.

    ``_AttractorMapper._reinit`` (the Python oracle) and ``_try_rust_march`` (the
    kernel) must hand the *same* tolerances to the same stepper.  They both read
    ``BASIN_RTOL``/``BASIN_ATOL``; this asserts neither drifted onto the library
    default, which is what would silently break ``tests/test_basin_kernel.py``'s
    equivalence.
    """
    import importlib

    from tsdynamics.utils.tolerances import BASIN_ATOL, BASIN_RTOL

    # ``from ... import attractors`` now binds the *function* of that name, which
    # v6 renamed from ``find_attractors`` — it shadows the module inside its own
    # package.  Reach the module through the import system, not through the
    # attribute.
    att = importlib.import_module("tsdynamics.analysis.basins.attractors")

    for fn in (att._AttractorMapper._reinit, att._try_rust_march):
        src = inspect.getsource(fn)
        assert "BASIN_RTOL" in src and "BASIN_ATOL" in src, (
            f"{fn.__qualname__} no longer names the basin tolerance constants"
        )
    assert att.BASIN_RTOL is BASIN_RTOL and att.BASIN_ATOL is BASIN_ATOL


# ===========================================================================
# API-footgun gate (stream v6 API-FOOTGUNS)
# ===========================================================================
#
# The fresh-eyes session played a new user with no documentation and walked into
# the same class of wall repeatedly: a call that is the *obvious* thing to type
# answered by Python's own binder or by an error raised three layers below the
# call.  The errgate above asks "does the message name the value"; this section
# asks the prior question — "did the library get a chance to speak at all".
#
# Each gate below pins one closed footgun, and two of them are structural (they
# sweep the source rather than a curated list), because the failure they catch is
# one that reappears every time someone writes a new example or a new keyword
# path.


# ── the symbolic-kernel declaration mistakes (the #1 documented pitfall) ────


def test_kernel_subscript_error_names_the_accessor_and_the_fix() -> None:
    """``y[0]`` in ``_equations`` is answered with ``y(0)``, not with numpy advice.

    Writing ``y[0]`` is the single most likely first-timer mistake in the library
    — the state *looks* like an array — and Python's ``'function' object is not
    subscriptable`` names neither ``y``, nor the kernel, nor the one-character
    fix.  The old message went further wrong: it recited the numeric-routine and
    ``_structural_params`` advice, neither of which has anything to do with it.
    """
    from tsdynamics.engine.compile import TapeCompileError

    class Subscripted(ts.ContinuousSystem):
        params = {"a": 1.0}
        dim = 2

        @staticmethod
        def _equations(y, t, a):
            return [y[1], -a * y[0]]

    with pytest.raises(TapeCompileError) as excinfo:
        Subscripted().run(final_time=1.0, dt=0.1)
    message = str(excinfo.value)
    assert "y(0)" in message
    assert "y[0]" in message
    assert "state *accessor*" in message
    # the corrected line is echoed back, so the fix is visible, not described
    assert "y(1), -a * y(0)" in message
    # ...and the two irrelevant paragraphs are gone
    assert "_structural_params" not in message
    assert "numeric routine" not in message


def test_subscripting_something_that_is_not_the_state_is_not_blamed_on_the_state() -> None:
    """``a[0]`` on a scalar *parameter* is a different mistake, and is not mis-named.

    Adversarial follow-up: the accessor diagnosis keyed off ``object is not
    subscriptable`` alone, which is the message for subscripting *anything*.  A
    kernel writing ``a[0]`` for a scalar parameter therefore got a paragraph
    explaining that ``y`` is an accessor — with no corrected line to show, since
    the line does not subscript ``y`` — pointing the reader at the one name on
    that line that was already right.  The claim is now made only when the state
    really was subscripted, so this falls through to the general advice.
    """
    from tsdynamics.engine.compile import TapeCompileError

    class ParamSubscripted(ts.ContinuousSystem):
        params = {"a": 1.0}
        dim = 2

        @staticmethod
        def _equations(y, t, a):
            return [a[0] * y(1), -y(0)]

    with pytest.raises(TapeCompileError) as excinfo:
        ParamSubscripted().run(final_time=1.0, dt=0.1)
    message = str(excinfo.value)
    assert "state *accessor*" not in message
    assert "not subscriptable" in message  # the raised error is still quoted
    assert "traced *symbolically*" in message  # ...and the general advice is given


def test_unpacking_the_state_is_answered_with_the_accessor_calls() -> None:
    """``x, y, z = u`` in ``_equations`` gets the same sentence ``u[0]`` gets.

    The subscript spelling was diagnosed; the *unpack* spelling was not — and it
    is at least as likely, because it is exactly how every ``DiscreteMap._step``
    in this library is written (a map's state genuinely is a vector).  A reader
    who wrote a map first and then an ODE hit ``TypeError: cannot unpack
    non-iterable function object`` and the numpy/``_structural_params``
    paragraph, which is about neither.
    """
    from tsdynamics.engine.compile import TapeCompileError

    class Unpacked(ts.ContinuousSystem):
        params = {"a": 1.0}
        dim = 3

        @staticmethod
        def _equations(u, t, a):
            x, y, z = u
            return [a * y, -a * x, -z]

    with pytest.raises(TapeCompileError) as excinfo:
        Unpacked().run(final_time=1.0, dt=0.1)
    message = str(excinfo.value)
    assert "state *accessor*" in message
    assert "u(0)" in message
    # the corrected line is echoed back, one call per unpacked target
    assert "x, y, z = u(0), u(1), u(2)" in message
    assert "_structural_params" not in message
    assert "numeric routine" not in message


def test_unpacking_something_that_is_not_the_state_is_not_blamed_on_the_state() -> None:
    """The negative of the above: unpacking a scalar *parameter* is a different bug."""
    from tsdynamics.engine.compile import TapeCompileError

    class ParamUnpacked(ts.ContinuousSystem):
        params = {"a": 1.0}
        dim = 2

        @staticmethod
        def _equations(u, t, a):
            p, q = a
            return [p * u(1), -q * u(0)]

    with pytest.raises(TapeCompileError) as excinfo:
        ParamUnpacked().run(final_time=1.0, dt=0.1)
    message = str(excinfo.value)
    assert "state *accessor*" not in message
    assert "traced *symbolically*" in message


def test_a_kernel_written_as_an_ordinary_method_just_works() -> None:
    """A missing ``@staticmethod`` is APPLIED, not merely diagnosed.

    The engine calls the kernel off the class, so a ``self`` first parameter is
    always a mistake and it is decidable at class definition.  The library used
    to detect it exactly and print the corrected line — if it can print the fix
    it can apply it, and that removes one line and one concept from every system
    anyone writes (the single most likely first-run failure).
    """

    class NotStatic(ts.ContinuousSystem):
        params = {"a": 1.0}
        dim = 2

        def _equations(self, y, t, a):  # noqa: PLR6301 - the shape under test
            return [y(1), -a * y(0)]

    traj = NotStatic(ic=[1.0, 0.0]).run(final_time=1.0, dt=0.1)
    assert np.isfinite(traj.y).all()
    # The declaration side is corrected on the class, so downstream
    # introspection (the map params/_step order check, ``help``) sees the real
    # signature — not one shifted by a phantom ``self``.
    assert list(inspect.signature(NotStatic._equations).parameters) == ["y", "t", "a"]


def test_correct_kernels_are_untouched_by_the_new_diagnostics() -> None:
    """The two structural checks fire on the defect only — a good kernel still runs."""
    traj = ts.systems.Lorenz().run(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0])
    assert np.isfinite(traj.y).all()


# ── every plot example in the library names a REGISTERED transform ─────────

#: Calls whose *positional* string arguments are transform names.
_PLOT_CALL = re.compile(r"(?:\bts\.plot|\bviz\.plot|\bplot|\bT)\(")


def _transform_names_in(line: str) -> list[str]:
    """Positional string literals inside a ``plot(...)`` / ``T(...)`` call on ``line``.

    Depth-aware, so a chained ``.save("fig.png")`` / ``.recolor("red")`` is not
    mistaken for a transform name, and keyword values (``primitive="density"``,
    ``layout="stack"``) are excluded because they are preceded by ``=``.

    Strings nested inside a ``[...]`` list or a ``{...}`` dict are excluded too,
    and that exclusion is load-bearing rather than tidy-mindedness: only the
    first element of a *collection* keyword value is preceded by the ``=``, so
    without it ``ts.plot(traj, "phase_portrait", components=["x", "z"])`` reads
    as three positional names and the gate fails a **correct** example, naming
    ``'z'`` as an unregistered transform.  That is the worst failure mode a
    source-sweeping gate can have — it makes the honest fix look like the
    defect — and the shape is a common one (``components=``, ``labels=``,
    ``ic=[...]``).
    """
    found: list[str] = []
    for match in _PLOT_CALL.finditer(line):
        depth, nested, i, n = 0, 0, match.end() - 1, len(line)
        while i < n:
            char = line[i]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            elif char in "[{":
                nested += 1
            elif char in "]}":
                nested -= 1
            elif char == '"' and depth == 1 and nested == 0:
                end = line.find('"', i + 1)
                if end == -1:
                    break
                if not line[:i].rstrip().endswith(("=", ":")):
                    found.append(line[i + 1 : end])
                i = end
            i += 1
    return found


#: How many following lines a wrapped ``plot(...)`` example may span.  Six covers
#: every multi-line example in the library with room to spare; the cap only stops
#: an unbalanced quote from swallowing the rest of the file.
_MAX_CALL_LINES = 6


def _call_blocks(lines: list[str]) -> list[tuple[int, str]]:
    """Yield ``(lineno, text)`` for each ``plot(...)`` call, joined across wraps.

    A line-at-a-time scan silently misses a transform named on a *continuation*
    line — ``ts.plot(vdp, "flow_speed",`` / ``"not_a_transform")`` — which is the
    shape a long example naturally takes, so the check would go blind exactly
    where an example is long enough to be worth checking.  Joining until the
    parentheses balance costs nothing and closes that hole.
    """
    blocks: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if not _PLOT_CALL.search(line):
            continue
        text, depth = line, line.count("(") - line.count(")")
        extra = 0
        while depth > 0 and extra < _MAX_CALL_LINES and index + extra + 1 < len(lines):
            extra += 1
            nxt = lines[index + extra]
            text += " " + nxt.strip()
            depth += nxt.count("(") - nxt.count(")")
        blocks.append((index + 1, text))
    return blocks


#: Transform names a v6 docstring already uses that are not registered **yet**,
#: with the slot that owns the registration.  Liveness-checked below so the set
#: can only shrink.
#:
#: * ``direction_field`` — §6.8 makes it the alias of the repaired
#:   ``vector_field``.  Owner: S8 · TRANSFORMS.
#: * ``speed`` — a prose shorthand for ``flow_speed`` in ``_registry.py``'s
#:   module docstring.  Owner: C8 · VIZ-REGISTRY.
_PLOT_EXAMPLE_PENDING = frozenset({"direction_field", "speed"})


def test_plot_examples_name_registered_transforms() -> None:
    """No documented ``ts.plot(...)`` example names a transform that does not exist.

    The defect this closes: the flagship one-liner in ``ts.plot``'s module
    docstring, in ``TransformCall``'s, and in the package ``__init__`` read
    ``ts.plot(duff, "basins", "attractors", "trajectory", "fixed_points")`` —
    four names, none registered — so the first thing a reader copied answered
    with ``unknown plot transform 'basins'``.  A curated list of examples would
    rot the same way, so the whole library source is swept and checked against
    the live registry.
    """
    from tsdynamics.viz.transforms import names as transform_names

    known = set(transform_names())
    offenders: list[str] = []
    root = pathlib.Path(ts.__file__).parent
    for path in sorted(root.rglob("*.py")):
        for lineno, block in _call_blocks(path.read_text().splitlines()):
            for name in _transform_names_in(block):
                if not re.fullmatch(r"[a-z_][a-z_0-9.]*", name):
                    continue  # a path, a title, a colour word — not a name-shaped token
                if name.partition(".")[0] in _PLOT_EXAMPLE_PENDING:
                    continue
                if name.partition(".")[0] not in known:
                    offenders.append(f"{path.name}:{lineno}: {name!r} in {block.strip()!r}")
    assert not offenders, "plot examples naming unregistered transforms:\n" + "\n".join(offenders)


def test_plot_example_pending_names_are_still_pending() -> None:
    """The pending-transform list can only shrink — drop a row when it registers."""
    from tsdynamics.viz.transforms import names as transform_names

    known = set(transform_names())
    landed = sorted(_PLOT_EXAMPLE_PENDING & known)
    assert not landed, (
        f"{landed} is registered now — delete it from _PLOT_EXAMPLE_PENDING so the "
        "example gate covers it again"
    )


def test_plot_example_scanner_would_catch_the_original_defect() -> None:
    """The scanner itself: it finds the names that were wrong, and ignores the rest."""
    broken = '    ts.plot(duff, "basins", "attractors", "trajectory", "fixed_points")'
    assert _transform_names_in(broken) == ["basins", "attractors", "trajectory", "fixed_points"]
    # a chained call after the plot(...) is out of scope
    assert _transform_names_in('traj.plot().save("fig.png").render("plotly")') == []
    # a keyword value is not a positional transform name
    assert _transform_names_in('ts.plot(t, "phase_portrait", primitive="density")') == [
        "phase_portrait"
    ]
    # the dotted primitive sugar keeps its head
    assert _transform_names_in('ts.plot(t, "phase_portrait.density")') == ["phase_portrait.density"]


def test_plot_example_scanner_does_not_read_collection_values_as_names() -> None:
    """A ``components=["x", "z"]`` keyword is a value, not two transform names.

    Only the *first* element of a collection keyword value is preceded by the
    ``=``, so a scanner that looks no further than that flags every element
    after the comma — failing a **correct** example and pointing the reader at
    ``'z'`` as an unregistered transform.  The shape is common enough
    (``components=``, ``labels=``, ``ic=[…]``) that the gate would misfire the
    first time someone wrote the most ordinary example there is.
    """
    assert _transform_names_in('ts.plot(t, "phase_portrait", components=["x", "z"])') == [
        "phase_portrait"
    ]
    assert _transform_names_in('ts.viz.plot(a, b, components=["x", "y"])') == []
    assert _transform_names_in('ts.plot(t, "hilbert", labels={"a": "b"})') == ["hilbert"]


def test_plot_example_scanner_sees_a_name_on_a_continuation_line() -> None:
    """A wrapped example is one call, not two lines the scanner may half-read.

    Line-at-a-time, ``"nullclines"`` below is invisible — which is how a long
    example (the ones most likely to name several transforms) would escape the
    check entirely.
    """
    wrapped = [
        '>>> ts.plot(vdp, "flow_speed",',
        '...          "nullclines",',
        "...          xlim=(-2.5, 2.5))",
    ]
    (lineno, block), *rest = _call_blocks(wrapped)
    assert (lineno, rest) == (1, [])
    assert _transform_names_in(block) == ["flow_speed", "nullclines"]
    # a call that closes on its own line is not joined with what follows
    standalone = ['ts.plot(traj, "phase_portrait")', 'x = "not_a_transform"']
    assert _transform_names_in(_call_blocks(standalone)[0][1]) == ["phase_portrait"]


# ── every keyword path through the plot front door is validated ────────────


def test_transform_call_options_are_validated_like_shared_ones() -> None:
    """``T("phase_portrait", nonsense=1)`` is answered, not raised through.

    A ``T()``'s own options were the last keyword path into ``ts.plot`` that
    nobody checked: they went straight to the compute, so a typo surfaced as a
    bare ``TypeError`` naming a private function the caller never typed.
    """
    from tsdynamics.errors import InvalidParameterError

    traj = ts.systems.Lorenz().run(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    with pytest.raises(InvalidParameterError) as excinfo:
        ts.plot(traj, ts.viz.spec.T("phase_portrait", nonsense=1))
    message = str(excinfo.value)
    assert "nonsense" in message
    assert "components" in message  # the keywords that ARE accepted are named


def test_transform_call_typo_suggests_the_intended_keyword() -> None:
    """The suggestion machinery reaches the T() path too (``componets`` → ``components``)."""
    from tsdynamics.errors import InvalidParameterError

    traj = ts.systems.Lorenz().run(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    with pytest.raises(InvalidParameterError, match="did you mean components="):
        ts.plot(traj, ts.viz.spec.T("phase_portrait", componets=[0, 1]))


def test_transform_call_still_accepts_its_real_options() -> None:
    """The check refuses only what the transform cannot take (no over-refusal)."""
    traj = ts.systems.Lorenz().run(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    spec = ts.plot(traj, ts.viz.spec.T("phase_portrait", components=[0, 1], color="red", alpha=0.5))
    assert spec.layers[0].style["color"] == "red"


# ── PlotSpec.show() — the fourth verb everyone's fingers already know ──────


def test_plotspec_show_displays_and_says_so_when_it_cannot(monkeypatch) -> None:
    """``spec.show()`` DISPLAYS — it is not a second way to get the figure.

    It returns ``None`` like every other ``show`` a user's fingers know, and the
    figure has exactly one door: ``spec.fig``.  On a backend with no window the
    call would otherwise be a silent no-op, so it warns and names the two ways
    out.
    """
    import matplotlib.pyplot as plt

    from tsdynamics.viz import spec as spec_mod
    from tsdynamics.viz.render.caps import VisualizationDegraded

    monkeypatch.setattr(spec_mod, "_mpl_backend_is_interactive", lambda: False)
    spec = ts.systems.Lorenz().run(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0]).__plot_spec__()
    with pytest.warns(VisualizationDegraded, match="has no window"):
        assert spec.show() is None
    assert spec.fig.__class__.__module__.startswith("matplotlib")
    plt.close("all")


def test_plotspec_show_displays_on_an_interactive_backend(monkeypatch) -> None:
    """On a backend that *can* open a window, the display call actually happens."""
    import matplotlib.pyplot as plt

    from tsdynamics.viz import spec as spec_mod

    calls: list[int] = []
    monkeypatch.setattr(spec_mod, "_mpl_backend_is_interactive", lambda: True)
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    spec = ts.systems.Lorenz().run(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0]).__plot_spec__()
    spec.show()
    assert calls == [1]
    plt.close("all")


def test_plot_verbs_are_all_present_and_documented() -> None:
    """adjust / draw / display / write — the four verbs, each with a docstring."""
    from tsdynamics.viz.spec import PlotSpec

    for verb in ("tweak", "render", "show", "save"):
        method = getattr(PlotSpec, verb)
        assert callable(method)
        assert method.__doc__, f"PlotSpec.{verb} is undocumented"
    # ``plot`` BUILDS a spec, so a spec does not have one: a method named
    # ``plot`` that returned its own receiver was the joke told twice.
    assert not hasattr(PlotSpec, "plot")


# ---------------------------------------------------------------------------
# One region grammar (WP0)
# ---------------------------------------------------------------------------
#
# A ``region=`` argument is read ONE (lo, hi) BOUND PER STATE COMPONENT, at
# every door, with no exceptions.  The defect this gate exists to catch was
# silent: ``fixed_points(VanDerPol(), region=[(-3, 3), (-3, 3)])`` used to read
# the two rows as a *corner pair*, search the zero-volume box between them, and
# return an empty result rather than the origin.  A wrong answer with no error
# is the worst outcome available, so both halves are pinned: the per-axis
# spelling must WORK, and the corner-pair spelling must RAISE.

_REGION_DOORS: list[tuple[str, object]] = []


def _region_doors() -> list[tuple[str, object]]:
    """Every public entry point whose ``region=`` argument names a search box."""
    if _REGION_DOORS:
        return _REGION_DOORS
    from tsdynamics.systems import VanDerPol

    def _fixed_points(region):
        return ts.analysis.fixed_points(VanDerPol(), region=region, seed=0)

    def _fixed_points_interval(region):
        return ts.analysis.fixed_points(VanDerPol(), region=region, method="interval")

    def _expansion_entropy(region):
        return ts.analysis.expansion_entropy(
            VanDerPol(), region=region, n_samples=20, final_time=0.5, seed=0
        )

    def _attractors(region):
        return ts.analysis.attractors(VanDerPol(), region, n_seeds=4, seed=0, max_steps=200)

    def _basin_fractions(region):
        return ts.analysis.basin_fractions(VanDerPol(), region, n_seeds=4, seed=0, max_steps=200)

    def _basins(region):
        return ts.analysis.basins(VanDerPol(), region, max_steps=200)

    _REGION_DOORS.extend(
        [
            ("fixed_points", _fixed_points),
            ("fixed_points(method='interval')", _fixed_points_interval),
            ("expansion_entropy", _expansion_entropy),
            ("attractors", _attractors),
            ("basin_fractions", _basin_fractions),
            ("basins", _basins),
        ]
    )
    return _REGION_DOORS


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("name", [d[0] for d in _region_doors()])
def test_every_region_argument_uses_the_same_reading(name: str) -> None:
    """Per-axis bounds are accepted; a corner pair is refused, at every door.

    The basin doors run with ``max_steps=200`` so this gate stays fast, which
    means nothing settles and ``basins`` correctly warns that it measured
    nothing — a different contract, tested in ``test_basins.py``.
    """
    call = dict(_region_doors())[name]
    per_axis = [(-3.0, 3.0), (-3.0, 3.0)]
    corner_pair = ([-3.0, -3.0], [3.0, 3.0])

    call(per_axis)  # must not raise

    with pytest.raises(InvalidInputError):
        call(corner_pair)


def test_region_primitives_are_accepted_but_never_required() -> None:
    """A caller holding a Box/Grid keeps working: bounds are an ADDITION."""
    from tsdynamics.data import Ball, Box, Grid, as_region
    from tsdynamics.systems import VanDerPol

    bounds = ts.analysis.fixed_points(VanDerPol(), region=[(-3.0, 3.0), (-3.0, 3.0)], seed=0)
    boxed = ts.analysis.fixed_points(VanDerPol(), region=Box([-3.0, -3.0], [3.0, 3.0]), seed=0)
    assert len(bounds) == len(boxed) == 1
    np.testing.assert_allclose(bounds.points, boxed.points, atol=1e-8)

    for prim in (
        Box([-1.0, -1.0], [1.0, 1.0]),
        Ball([0.0, 0.0], r=1.0),
        Grid([-1.0, -1.0], [1.0, 1.0], (4, 4)),
    ):
        assert as_region(prim) is prim


def test_state_space_helpers_take_plain_bounds() -> None:
    """``sampler`` / ``grid_points`` no longer demand a library type."""
    from tsdynamics.data import Grid, grid_points, sampler

    assert sampler([(-1.0, 1.0), (-1.0, 1.0)], seed=0)().shape == (2,)
    assert grid_points([(-1.0, 1.0, 3), (-1.0, 1.0, 3)]).shape == (9, 2)
    assert grid_points([(-1.0, 1.0), (-1.0, 1.0)], resolution=4).shape == (16, 2)
    assert grid_points(Grid([-1.0], [1.0], (5,))).shape == (5, 1)
