"""``ts.analysis.results`` — the 32 result types, behind one dot (contract §2.7).

The rule (contract §1, corollary C2): *a type you get back is not one you type*.
Every one of the 32 result classes is something an analysis **returns**; a user
constructs none of them.  Before v6 they were 32 of the 84 names on
``ts.analysis`` — 38 % of the tab surface, none of it an analysis — so the answer
to "what analyses are there?" was mostly a list of things that are not analyses.

They move to ``ts.analysis.results``: off the flat listing, still importable,
still exactly the classes ``isinstance`` sees.  Nothing about the objects changed.
"""

from __future__ import annotations

import importlib

import pytest

from tsdynamics.analysis._result import AnalysisResult

results = importlib.import_module("tsdynamics.analysis.results")


def _all_subclasses(cls: type) -> set[type]:
    """Every (transitive) **library** subclass of ``cls``.

    Restricted to ``tsdynamics.*`` because other test modules define throwaway
    subclasses to probe the machinery, and ``__subclasses__()`` sees them all as
    soon as their module has been imported into the same worker.
    """
    found: set[type] = set()
    for sub in cls.__subclasses__():
        if sub.__module__.startswith("tsdynamics."):
            found.add(sub)
        found |= _all_subclasses(sub)
    return found


def test_the_namespace_is_exactly_the_result_hierarchy():
    """Every result class is here, and nothing else is."""
    live = {c.__name__ for c in _all_subclasses(AnalysisResult)} | {"AnalysisResult"}
    assert set(results.__all__) == live


def test_the_namespace_is_sorted_and_dir_mirrors_it():
    """``dir()`` is the listing; a hand-ordered ``__all__`` would drift."""
    assert results.__all__ == sorted(results.__all__)
    assert dir(results) == sorted(results.__all__)


@pytest.mark.parametrize("name", sorted(results.__all__))
def test_every_name_resolves_to_a_result_class(name):
    """Each entry is a real, importable ``AnalysisResult`` subclass."""
    obj = getattr(results, name)
    assert isinstance(obj, type) and issubclass(obj, AnalysisResult)


@pytest.mark.parametrize("name", sorted(results.__all__))
def test_the_namespace_re_exports_and_never_redefines(name):
    """It is a *namespace*, not a second home: same object as the owning module."""
    obj = getattr(results, name)
    owner = importlib.import_module(obj.__module__)
    assert getattr(owner, name) is obj


def test_isinstance_still_sees_the_same_classes():
    """A result produced by an analysis is an instance of the namespaced class."""
    import numpy as np

    from tsdynamics.analysis.lyapunov import LyapunovSpectrum

    spectrum = LyapunovSpectrum(values=np.array([0.9, 0.0, -14.0]))
    assert isinstance(spectrum, results.LyapunovSpectrum)
    assert isinstance(spectrum, results.ArrayResult)
    assert isinstance(spectrum, results.AnalysisResult)


def test_the_old_import_paths_still_work():
    """Demotion is not removal: every historical import keeps resolving."""
    import tsdynamics.analysis as analysis
    import tsdynamics.analysis._result as result_module
    import tsdynamics.analysis.lyapunov as lyapunov

    assert analysis.LyapunovSpectrum is lyapunov.LyapunovSpectrum is results.LyapunovSpectrum
    assert result_module.ArrayResult is results.ArrayResult
