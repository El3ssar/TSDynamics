"""Registry-driven meta-QA over the generic analysis/transform registries.

Stream I-QA: these tests sweep the D4 plugin registries
(:data:`tsdynamics.registry.analyses` / :data:`~tsdynamics.registry.transforms`)
once per registered entry, asserting the invariants every public quantifier must
satisfy — callable, documented, round-trips through its own registry, and (when
re-exported) agrees with the top-level package attribute.  A set of curated
headline-name guards then catches a stream's self-registration silently breaking
or a public name disappearing.

These are pure structural/contract checks (no Hypothesis needed): the
parametrized fixtures ``analysis_entry`` / ``transform_entry`` (provided by
``conftest.py``) yield one :class:`~tsdynamics.registry.RegistryEntry` per
registered analysis/transform.
"""

from __future__ import annotations

import types as _types
import typing
from collections.abc import Mapping

import numpy as np
import pytest

import tsdynamics as ts
import tsdynamics.transforms as tx  # noqa: F401  (import populates registry.transforms)
from tsdynamics import registry
from tsdynamics.analysis._result import AnalysisResult
from tsdynamics.data import Trajectory

# ---------------------------------------------------------------------------
# Parametrized contract: analyses (one run per registered analysis)
# ---------------------------------------------------------------------------


def test_analysis_entry_is_callable(analysis_entry):
    """Every registered analysis must be a callable (a quantifier you can invoke)."""
    assert callable(analysis_entry.obj)


def test_analysis_entry_documented(analysis_entry):
    """Every registered public analysis carries a non-empty docstring."""
    doc = analysis_entry.obj.__doc__
    assert isinstance(doc, str)
    assert doc.strip(), f"analysis {analysis_entry.name!r} has an empty docstring"


def test_analysis_entry_metadata_is_mapping(analysis_entry):
    """Entry name is a non-empty str and metadata behaves as a mapping."""
    assert isinstance(analysis_entry.name, str)
    assert analysis_entry.name.strip()
    assert isinstance(analysis_entry.metadata, Mapping)
    # ``dict(...)`` must succeed (and round-trip the same keys) — the mapping
    # contract the plugin layer relies on.
    as_dict = dict(analysis_entry.metadata)
    assert set(as_dict) == set(analysis_entry.metadata)


def test_analysis_entry_roundtrips(analysis_entry):
    """The entry is reachable under its own name and resolves to the same object."""
    assert analysis_entry.name in registry.analyses
    assert registry.analyses.get(analysis_entry.name) is analysis_entry.obj
    # ``.entry(name)`` returns the very same RegistryEntry the sweep iterated.
    assert registry.analyses.entry(analysis_entry.name).obj is analysis_entry.obj


def test_analysis_entry_top_level_export(analysis_entry):
    """If re-exported at top level, ``ts.<name>`` is the *same* object (no shadowing)."""
    # Lenient: not every analysis is advertised at the top level; only the ones
    # that ARE must agree with the registered object.
    if hasattr(ts, analysis_entry.name):
        assert getattr(ts, analysis_entry.name) is analysis_entry.obj


# ---------------------------------------------------------------------------
# Parametrized contract: transforms (one run per registered transform)
# ---------------------------------------------------------------------------


def test_transform_entry_is_callable(transform_entry):
    """Every registered transform must be callable."""
    assert callable(transform_entry.obj)


def test_transform_entry_documented(transform_entry):
    """Every registered public transform carries a non-empty docstring."""
    doc = transform_entry.obj.__doc__
    assert isinstance(doc, str)
    assert doc.strip(), f"transform {transform_entry.name!r} has an empty docstring"


def test_transform_entry_metadata_is_mapping(transform_entry):
    """Entry name is a non-empty str and metadata behaves as a mapping."""
    assert isinstance(transform_entry.name, str)
    assert transform_entry.name.strip()
    assert isinstance(transform_entry.metadata, Mapping)
    as_dict = dict(transform_entry.metadata)
    assert set(as_dict) == set(transform_entry.metadata)


def test_transform_entry_roundtrips(transform_entry):
    """The entry is reachable under its own name and resolves to the same object."""
    assert transform_entry.name in registry.transforms
    assert registry.transforms.get(transform_entry.name) is transform_entry.obj
    assert registry.transforms.entry(transform_entry.name).obj is transform_entry.obj


def test_transform_entry_top_level_export(transform_entry):
    """If re-exported on ``tsdynamics.transforms``, the attribute is the same object."""
    if hasattr(tx, transform_entry.name):
        assert getattr(tx, transform_entry.name) is transform_entry.obj


# ---------------------------------------------------------------------------
# Non-parametrized guards: headline membership + sane sizes
#
# These freeze the public surface: if a stream's self-registration regresses
# (a name vanishes, or a whole subpackage stops importing), one of these fails
# loudly instead of the sweep above simply running over fewer entries.
# ---------------------------------------------------------------------------

#: Headline analyses spanning every Tier-3 A-* stream — Lyapunov, chaos
#: indicators, fixed points / orbits, dimensions, embedding, entropy,
#: recurrence, surrogates and basins.  A subset (the registry may carry more).
_EXPECTED_ANALYSES = frozenset(
    {
        # A-LYAP
        "lyapunov_spectrum",
        "max_lyapunov",
        "kaplan_yorke_dimension",
        "lyapunov_from_data",
        # A-CHAOS
        "gali",
        "zero_one_test",
        "expansion_entropy",
        # A-FP
        "fixed_points",
        "periodic_orbits",
        # A-ORBIT
        "orbit_diagram",
        "poincare_section",
        "return_map",
        # A-DIM
        "correlation_dimension",
        "generalized_dimension",
        # A-EMBED
        "embed",
        "optimal_delay",
        "embedding_dimension",
        # A-ENT
        "permutation_entropy",
        "sample_entropy",
        "lz76_complexity",
        # A-RQA
        "recurrence_matrix",
        "rqa",
        "windowed_rqa",
        # A-SURR
        "surrogates",
        "surrogate_test",
        # A-BASIN
        "find_attractors",
        "basins_of_attraction",
    }
)

#: Headline transforms — spectral measures + Butterworth filter family +
#: feature extraction (stream T-XFORM).
_EXPECTED_TRANSFORMS = frozenset(
    {
        "detrend",
        "normalize",
        "power_spectral_density",
        "spectral_entropy",
        "spectral_centroid",
        "dominant_frequency",
        "lowpass",
        "highpass",
        "bandpass",
        "bandstop",
        "extract_features",
    }
)


def test_analyses_registry_has_expected_members():
    """Every headline analysis is registered, and the registry is non-trivially full."""
    names = set(registry.analyses.names())
    missing = _EXPECTED_ANALYSES - names
    assert not missing, f"analyses registry is missing headline members: {sorted(missing)}"
    # The full A-* fan-out registers well over forty quantifiers; a smaller
    # count means a whole subpackage failed to self-register.
    assert len(registry.analyses) >= 40


def test_transforms_registry_has_expected_members():
    """Every headline transform is registered (after importing tsdynamics.transforms)."""
    names = set(registry.transforms.names())
    missing = _EXPECTED_TRANSFORMS - names
    assert not missing, f"transforms registry is missing headline members: {sorted(missing)}"
    assert len(registry.transforms) >= 10


def test_registries_are_distinct_kinds():
    """The two generic registries are tagged with their distinct kind labels."""
    assert registry.analyses.kind == "analysis"
    assert registry.transforms.kind == "transform"
    # Distinct container instances — they must not be the same object.
    assert registry.analyses is not registry.transforms


def test_registry_names_match_entry_names():
    """``names()`` and ``all()`` agree element-for-element (no stale/aliased keys)."""
    for reg in (registry.analyses, registry.transforms):
        assert reg.names() == [e.name for e in reg.all()]


# ---------------------------------------------------------------------------
# Result-object contract (stream WS-WRAP)
#
# Every registered analysis returns a self-describing AnalysisResult (carrying
# .meta), never a bare float/ndarray/list — the v4 result-model invariant.  The
# sweep below is registry-driven, so a new analysis joins it with zero edits; a
# function that forgets to wrap its return fails loudly here.
# ---------------------------------------------------------------------------

#: Registered analyses that legitimately return something other than an
#: ``AnalysisResult``, with the type they DO return.  ``poincare_section`` returns
#: a :class:`~tsdynamics.data.Trajectory` carrying provenance; the named,
#: viz-ready ``PoincareSection`` result is delivered by stream WS-POINCARE-API
#: (issue #209), which owns ``analysis/orbits/poincare.py``.
_RESULT_CARVE_OUTS: dict[str, type] = {
    "poincare_section": Trajectory,
}


def _return_annotation_types(fn: object) -> tuple[object, ...]:
    """Flatten a callable's resolved return annotation into its component types.

    Unwraps ``Optional`` / ``X | Y`` unions so an annotation like
    ``ScalarResult | tuple[ScalarResult, ndarray]`` yields the ``ScalarResult``
    member.  Returns an empty tuple when there is no return annotation.
    """
    hints = typing.get_type_hints(fn)
    annotation = hints.get("return")
    if annotation is None:
        return ()
    flat: list[object] = []

    def _walk(node: object) -> None:
        origin = typing.get_origin(node)
        if origin in (typing.Union, _types.UnionType):
            for arg in typing.get_args(node):
                _walk(arg)
        else:
            flat.append(node)

    _walk(annotation)
    return tuple(flat)


def test_analysis_returns_analysis_result(analysis_entry):
    """Every registered analysis declares an ``AnalysisResult`` return (or a carve-out).

    Freezes the v4 result-model invariant: bare ``float``/``ndarray``/``list``
    returns are gone.  The check reads the (resolved) return annotation, so it
    runs without constructing inputs for all 48 analyses — the per-area test
    modules verify the *runtime* objects.
    """
    name = analysis_entry.name
    types = _return_annotation_types(analysis_entry.obj)

    if name in _RESULT_CARVE_OUTS:
        expected = _RESULT_CARVE_OUTS[name]
        assert expected in types, (
            f"carve-out {name!r} should return {expected.__name__}, got annotation {types}"
        )
        return

    assert any(isinstance(t, type) and issubclass(t, AnalysisResult) for t in types), (
        f"analysis {name!r} must return an AnalysisResult subclass "
        f"(got return annotation {types or 'none'})"
    )


def _henon():
    """A small Hénon map for runtime result-contract smoke checks."""
    return ts.systems.Henon()


def _logistic_series() -> np.ndarray:
    """A deterministic chaotic series (logistic, r=3.9) for data-consuming analyses."""
    x = np.empty(600)
    x[0] = 0.4
    for i in range(1, x.size):
        x[i] = 3.9 * x[i - 1] * (1.0 - x[i - 1])
    return x


# (name, thunk) covering every result-wrapper KIND at runtime: scalar, count,
# array, scaling, collection, and the rich per-stream result dataclasses.  Proves
# the wrapping actually fires (isinstance + a populated .meta), complementing the
# annotation sweep above.
def _runtime_cases() -> list[tuple[str, object]]:
    series = _logistic_series()
    traj = _henon().iterate(steps=600, ic=[0.1, 0.1])
    spectrum = [0.42, -1.62]
    return [
        ("lyapunov_spectrum", lambda: ts.lyapunov_spectrum(_henon(), k=2, n=1500, ic=[0.1, 0.1])),
        ("max_lyapunov", lambda: ts.max_lyapunov(_henon(), n=150, ic=[0.1, 0.1])),
        ("kaplan_yorke_dimension", lambda: ts.kaplan_yorke_dimension(spectrum)),
        ("zero_one_test", lambda: ts.zero_one_test(series)),
        ("permutation_entropy", lambda: ts.permutation_entropy(series)),
        ("lz76_complexity", lambda: ts.lz76_complexity(series)),
        ("correlation_dimension", lambda: ts.correlation_dimension(traj)),
        ("embed", lambda: ts.embed(series, 3, 1)),
        ("optimal_delay", lambda: ts.optimal_delay(series, max_delay=20)),
        ("mutual_information", lambda: ts.mutual_information(series, max_delay=20)),
        ("surrogates", lambda: ts.surrogates(series, "shuffle", 4, seed=0)),
        ("surrogate_test", lambda: ts.surrogate_test(series, n=19, seed=0)),
        ("recurrence_matrix", lambda: ts.recurrence_matrix(traj, recurrence_rate=0.05)),
        ("fixed_points", lambda: ts.fixed_points(_henon(), seed=0)),
    ]


_RUNTIME_CASES = _runtime_cases()


@pytest.mark.parametrize("name,thunk", _RUNTIME_CASES, ids=[c[0] for c in _RUNTIME_CASES])
def test_analysis_runtime_result_contract(name, thunk):
    """A representative analysis of each result kind returns a live AnalysisResult.

    Asserts the wrapping fires at runtime: the value is an ``AnalysisResult`` and
    carries a mapping ``.meta``.  Complements the annotation sweep, which is static.
    """
    result = thunk()
    assert isinstance(result, AnalysisResult), f"{name} returned {type(result).__name__}"
    assert isinstance(result.meta, Mapping) and result.meta, f"{name} has no provenance .meta"
