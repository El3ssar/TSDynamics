"""The object-side accessor layer is GONE — this file is its headstone.

Stream WS-ACCESSORS shipped ``sys.lyap`` / ``.chaos`` / ``.dims`` / ``.recurrence``
plus the derived-builder verbs ``stroboscope`` / ``project`` / ``tangent`` /
``copies``.  Owner ruling **A2** (v6 CONTRACT §5.1, §8.3) removes every one of
them: an analysis is a free function whose first argument is its subject, and the
convenience that hid a sampling choice was a silent-wrong-answer generator — the
module's own source recorded ``sys.chaos.zero_one()`` reporting **K = -0.026** for
Lorenz against the free function's **K = 0.999**.

``families/_accessors.py`` is deleted.  The 590 lines of tests that exercised it
are replaced by the inverse gate below: the surface must stay gone, every
``AttributeError`` must teach the replacement, and the *capability* must remain
reachable at its one canonical address.
"""

from __future__ import annotations

import importlib

import pytest

import tsdynamics as ts
from tsdynamics.systems import Henon, Lorenz

#: The four topical namespaces ruling A2 deleted.
TOPICAL = ("lyap", "dims", "recurrence", "chaos")

#: The object verbs ruling A3 / CONTRACT §3.2 took off the system.
REMOVED_VERBS = ("stroboscope", "project", "tangent", "copies")


def test_the_accessor_module_itself_is_gone() -> None:
    """``families/_accessors.py`` is deleted outright (CONTRACT §8.3)."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tsdynamics.families._accessors")


@pytest.mark.parametrize("name", TOPICAL)
def test_a_topical_namespace_does_not_resolve_and_teaches(name: str) -> None:
    """``lor.lyap`` is absent, and its error hands back a free-function line."""
    lor = Lorenz()
    assert not hasattr(lor, name)
    with pytest.raises(AttributeError) as excinfo:
        getattr(lor, name)
    message = str(excinfo.value)
    assert name in message
    assert "free function" in message
    assert "ts.analysis." in message


@pytest.mark.parametrize("name", REMOVED_VERBS)
def test_a_removed_derived_verb_does_not_resolve_and_names_its_home(name: str) -> None:
    """Each removed verb's error names what to type instead."""
    lor = Lorenz()
    assert not hasattr(lor, name)
    with pytest.raises(AttributeError) as excinfo:
        getattr(lor, name)
    assert name in str(excinfo.value)


def test_the_capability_survives_at_its_canonical_address() -> None:
    """Nothing lost capability — the four wrappers still build by hand."""
    lor = Lorenz()
    assert isinstance(ts.derived.TangentSystem(lor, k=2), ts.derived.TangentSystem)
    # NB the components are ONE argument, a sequence -- CONTRACT §3.2 wrote
    # ``ProjectedSystem(sys, 0, 2)``, which raises; §3.2 is corrected to match.
    assert isinstance(ts.derived.ProjectedSystem(lor, (0, 2)), ts.derived.ProjectedSystem)
    assert isinstance(ts.derived.StroboscopicMap(lor, 1.0), ts.derived.StroboscopicMap)


def test_infer_forcing_period_moved_and_did_not_die_with_the_module() -> None:
    """CONTRACT §5.1: it moves to ``derived/stroboscopic.py`` before the delete."""
    from tsdynamics.derived.stroboscopic import infer_forcing_period

    assert callable(infer_forcing_period)


def test_the_analyses_are_reachable_as_free_functions() -> None:
    """The one door ruling A2 leaves open actually answers."""
    hen = Henon()
    assert float(ts.analysis.max_lyapunov(hen, ic=[0.1, 0.1])) > 0.0
