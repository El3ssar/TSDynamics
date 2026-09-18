"""Registry-driven meta-QA over the generic analysis registry.

Stream I-QA: these tests sweep the D4 plugin registry
(:data:`tsdynamics.registry.analyses`)
once per registered entry, asserting the invariants every public quantifier must
satisfy — callable, documented, round-trips through its own registry, and (when
re-exported) agrees with the top-level package attribute.  A set of curated
headline-name guards then catches a stream's self-registration silently breaking
or a public name disappearing.

These are pure structural/contract checks (no Hypothesis needed): the
parametrized fixture ``analysis_entry`` (provided by ``conftest.py``) yields one
:class:`~tsdynamics.registry.RegistryEntry` per registered analysis.
"""

from __future__ import annotations

import types as _types
import typing
from collections.abc import Mapping

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.analysis._result import AnalysisResult
from tsdynamics.derived import PoincareSection

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


def test_analysis_entry_is_reachable_at_its_one_public_address(analysis_entry):
    """``ts.analysis.<name>`` IS the registered object, and it is the only address.

    v6 took the analyses off the 17-name top level, so ``ts.<name>`` raises a
    redirect naming ``ts.analysis.<name>``.  This is the gate for that: one
    concept, one spelling, and the spelling resolves to the registered object.
    """
    name = analysis_entry.name
    assert getattr(ts.analysis, name) is analysis_entry.obj
    assert name in ts.analysis.__all__
    with pytest.raises((AttributeError, ImportError)) as err:
        getattr(ts, name)
    assert f"ts.analysis.{name}" in str(err.value)


# ---------------------------------------------------------------------------
# Non-parametrized guards: headline membership + sane sizes
#
# These freeze the public surface: if a stream's self-registration regresses
# (a name vanishes, or a whole subpackage stops importing), one of these fails
# loudly instead of the sweep above simply running over fewer entries.
# ---------------------------------------------------------------------------

#: Headline analyses spanning every surviving A-* stream — Lyapunov, chaos
#: indicators, fixed points / orbits, dimensions, embedding, recurrence and
#: basins.  A subset (the registry may carry more).
_EXPECTED_ANALYSES = frozenset(
    {
        # A-LYAP
        "lyapunov_spectrum",
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
        # A-RQA
        "recurrence_matrix",
        "rqa",
        "windowed_rqa",
        # A-BASIN
        "attractors",
        "basins",
        # A-FIELDS (promoted public in v6)
        "flow_field",
        "ftle_field",
        "nullclines",
    }
)


def test_analyses_registry_has_expected_members():
    """Every headline analysis is registered, and the registry is non-trivially full."""
    names = set(registry.analyses.names())
    missing = _EXPECTED_ANALYSES - names
    assert not missing, f"analyses registry is missing headline members: {sorted(missing)}"
    # The surviving A-* fan-out registers well over thirty quantifiers; a smaller
    # count means a whole subpackage failed to self-register.
    assert len(registry.analyses) >= 30


def test_registries_are_distinct_kinds():
    """The two generic registries are tagged with their distinct kind labels."""
    assert registry.analyses.kind == "analysis"
    assert registry.renderers.kind == "renderer"
    # Distinct container instances — they must not be the same object.
    assert registry.analyses is not registry.renderers


def test_registry_names_match_entry_names():
    """``names()`` and ``all()`` agree element-for-element (no stale/aliased keys)."""
    assert registry.analyses.names() == [e.name for e in registry.analyses.all()]


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
#: a :class:`~tsdynamics.derived.PoincareSection` — a :class:`~tsdynamics.data.Trajectory`
#: subclass carrying section intent + the ``.summary()`` / ``.to_dict()`` / ``.plot``
#: result surface (stream WS-POINCARE-API, issue #209) — rather than an
#: ``AnalysisResult`` proper, so the section keeps all the trajectory affordances.
_RESULT_CARVE_OUTS: dict[str, type] = {
    "poincare_section": PoincareSection,
}

#: Registered analyses that return a **plain value** — an array, a float, a
#: tuple, a list of small records.  v6 promoted these onto the public surface
#: (CONTRACT §5.7: the eight ``planar`` field analyses had no door at all, and a
#: user who wanted FTLE *numbers* rather than a picture could not reach them), and
#: promoting them collided with §4.2 r1 ("every registered analysis returns an
#: ``AnalysisResult``").  §2.4's 53-name listing is the harder contract, so they
#: are registered, and the honest record of the exception is THIS table plus the
#: registry's own ``returns=`` field: an analysis either DECLARES the result class
#: it returns, or it is listed here.  Giving the 13 result wrappers is a v6.1
#: item (it moves the returned *type*, in two files this slot does not own).
_PLAIN_VALUE_ANALYSES: frozenset[str] = frozenset(
    {
        "autocorrelation",
        "correlation_sum",
        "dimension_spectrum",
        "escape_time_field",
        "estimate_dt_from_sagitta",
        "flow_field",
        "ftle_field",
        "invariant_density",
        "nullclines",
        "sagitta_profile",
        "set_distance",
        "streamlines",
        "trace_determinant",
        "transient_time_field",
    }
)


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

    if name in _PLAIN_VALUE_ANALYSES:
        assert analysis_entry.metadata.get("returns") is None, (
            f"{name!r} is listed as a plain-value analysis but DECLARES returns="
            f"{analysis_entry.metadata['returns']!r}; drop it from _PLAIN_VALUE_ANALYSES"
        )
        return

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
    declared = analysis_entry.metadata.get("returns")
    assert declared is not None and issubclass(declared, AnalysisResult), (
        f"analysis {name!r} must DECLARE its result class on the decorator "
        f"(returns=...), got {declared!r}"
    )
    assert declared in types, (
        f"analysis {name!r} declares returns={declared.__name__} but is annotated {types}"
    )


def _henon():
    """A small Hénon map for runtime result-contract smoke checks."""
    return ts.systems.Henon()


def _logistic_series() -> np.ndarray:
    """A deterministic chaotic series (logistic, r=3.9) for data-consuming analyses."""
    from _strategies import logistic_series

    return logistic_series(600, r=3.9, x0=0.4, burn=0)


# (name, thunk) covering every result-wrapper KIND at runtime: scalar, count,
# array, scaling, collection, and the rich per-stream result dataclasses.  Proves
# the wrapping actually fires (isinstance + a populated .meta), complementing the
# annotation sweep above.
def _runtime_cases() -> list[tuple[str, object]]:
    series = _logistic_series()
    traj = _henon().run(steps=600, ic=[0.1, 0.1])
    spectrum = [0.42, -1.62]
    return [
        (
            "lyapunov_spectrum",
            lambda: ts.analysis.lyapunov_spectrum(_henon(), k=2, n=1500, ic=[0.1, 0.1]),
        ),
        ("kaplan_yorke_dimension", lambda: ts.analysis.kaplan_yorke_dimension(spectrum)),
        ("zero_one_test", lambda: ts.analysis.zero_one_test(series)),
        ("correlation_dimension", lambda: ts.analysis.correlation_dimension(traj)),
        ("embed", lambda: ts.analysis.embed(series, 3, 1)),
        ("optimal_delay", lambda: ts.analysis.optimal_delay(series, max_delay=20)),
        ("mutual_information", lambda: ts.analysis.mutual_information(series, max_delay=20)),
        ("recurrence_matrix", lambda: ts.analysis.recurrence_matrix(traj, recurrence_rate=0.05)),
        ("rqa", lambda: ts.analysis.rqa(traj, recurrence_rate=0.05)),
        ("fixed_points", lambda: ts.analysis.fixed_points(_henon(), seed=0)),
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
