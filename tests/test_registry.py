"""
Registry consistency: the registry, the ``__all__`` exports, and the curated
test samples must all agree.  These are the tests that make "just write the
class" safe — a system that is exported but unregistered (or vice versa)
fails here.
"""

from __future__ import annotations

import pytest
from _sampling import DDE_HISTORIES, INTEGRATION_SAMPLE

import tsdynamics as ts
from tsdynamics import registry

# ---------------------------------------------------------------------------
# Registry ⟷ __all__ agreement
# ---------------------------------------------------------------------------


def test_registry_matches_all_exports() -> None:
    """Every exported system is registered and every builtin is exported."""
    exported = set(ts.systems.continuous.__all__) | set(ts.systems.discrete.__all__)
    registered = {e.name for e in registry.all_systems()}
    missing_from_registry = exported - registered
    missing_from_exports = registered - exported
    assert not missing_from_registry, f"exported but not registered: {missing_from_registry}"
    assert not missing_from_exports, f"registered but not exported: {missing_from_exports}"


def test_registry_matches_top_level_namespace() -> None:
    for entry in registry.all_systems():
        assert getattr(ts, entry.name) is entry.cls


def test_family_counts() -> None:
    counts = registry.families()
    assert counts["ode"] >= 118
    assert counts["dde"] >= 5
    assert counts["map"] >= 26


def test_entries_have_consistent_metadata(system_entry) -> None:
    cls = system_entry.cls
    assert system_entry.name == cls.__name__
    assert system_entry.module == cls.__module__
    assert dict(system_entry.params) == dict(cls.params)
    assert system_entry.family in ("ode", "dde", "map")
    assert system_entry.is_builtin


def test_get_prefers_builtin_and_suggests() -> None:
    assert registry.get("Lorenz").cls is ts.Lorenz
    with pytest.raises(KeyError, match="Did you mean 'Lorenz'"):
        registry.get("lorenz")
    with pytest.raises(KeyError):
        registry.get("DefinitelyNotASystem")


def test_user_subclass_registers_as_non_builtin() -> None:
    class _ProbeSystem(ts.ContinuousSystem):
        params = {"a": 1.0}
        dim = 1

        @staticmethod
        def _equations(y, t, *, a):
            return (a * y(0),)

    names_default = {e.name for e in registry.all_systems()}
    assert "_ProbeSystem" not in names_default
    names_all = {e.name for e in registry.all_systems(builtin=None)}
    assert "_ProbeSystem" in names_all
    assert registry.get("_ProbeSystem", builtin=False).family == "ode"


def test_variables_metadata_matches_dim() -> None:
    """Where ``variables`` is declared, its length must equal the class dim."""
    for entry in registry.all_systems():
        names = entry.cls.variables
        if names is not None and entry.dim is not None:
            assert len(names) == entry.dim, f"{entry.name}: variables/dim mismatch"


# ---------------------------------------------------------------------------
# Curated-sample guards — keep the slow tier representative
# ---------------------------------------------------------------------------


def test_integration_sample_names_exist() -> None:
    for name in INTEGRATION_SAMPLE:
        assert registry.get(name).family == "ode"


def test_integration_sample_covers_every_ode_category() -> None:
    """Every ODE category needs >= 2 representatives in the slow tier."""
    sample_categories: dict[str, int] = {}
    for name in INTEGRATION_SAMPLE:
        cat = registry.get(name).category
        sample_categories[cat] = sample_categories.get(cat, 0) + 1
    for category in registry.categories("ode"):
        assert sample_categories.get(category, 0) >= 2, (
            f"ODE category {category!r} has fewer than 2 representatives in "
            f"INTEGRATION_SAMPLE (tests/_sampling.py) — add some."
        )


def test_dde_histories_complete() -> None:
    """Every DDE system needs a non-equilibrium history in DDE_HISTORIES."""
    dde_names = {e.name for e in registry.all_systems(family="dde")}
    assert dde_names == set(DDE_HISTORIES), (
        f"DDE_HISTORIES (tests/_sampling.py) out of sync with registry: "
        f"missing {dde_names - set(DDE_HISTORIES)}, stale {set(DDE_HISTORIES) - dde_names}"
    )
