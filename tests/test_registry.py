"""
Registry consistency: the registry, the ``__all__`` exports, and the curated
test samples must all agree.  These are the tests that make "just write the
class" safe — a system that is exported but unregistered (or vice versa)
fails here.
"""

from __future__ import annotations

import pytest
from _sampling import DDE_HISTORIES, INTEGRATION_SAMPLE, SDE_SAMPLES

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
    assert counts["ode"] >= 120
    assert counts["dde"] >= 5
    assert counts["map"] >= 26


def test_entries_have_consistent_metadata(system_entry) -> None:
    cls = system_entry.cls
    assert system_entry.name == cls.__name__
    assert system_entry.module == cls.__module__
    assert dict(system_entry.params) == dict(cls.params)
    assert system_entry.family in ("ode", "dde", "map", "sde")
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
# `variables` completeness gate
#
# ``traj["x"]`` — the named-component access sold in the README, in the docs and
# on every per-system page — only works for a class that declares ``variables``.
# Before v6 only 23 of 171 catalogue systems did, so the documented front door
# raised ``KeyError`` for 86% of the catalogue.  This gate keeps it at 100%: a
# new system either names its components or is added to the exemption table
# below *with a reason*.
# ---------------------------------------------------------------------------

#: Systems deliberately shipped without ``variables``, and why.  Every entry is a
#: **variable-dimension** system whose state is a lattice / discretised field:
#: the components are indexed sites, not named quantities, and ``dim`` is not
#: known until an instance is built, so no fixed-length tuple could be correct.
#: Field-valued systems name their *blocks* with ``field_labels`` instead.
VARIABLES_EXEMPT: dict[str, str] = {
    "Lorenz96": "N ring sites; the state is a lattice, not named quantities",
    "MultiChua": "3 * n_circuits lattice of coupled Chua circuits",
    "KuramotoSivashinsky": "method-of-lines PDE: N grid samples of the field u(x)",
    "GrayScott": "method-of-lines PDE: 2 * N**2 grid samples (see field_labels)",
    "SwiftHohenberg": "method-of-lines PDE: N**2 grid samples of the field u(x, y)",
}


def test_every_system_declares_variables_or_is_exempt() -> None:
    """Every catalogue system names its components (or is a justified exemption).

    Registry-driven, so a newly added system joins the gate with zero test
    edits: give it a ``variables`` ClassVar of length ``dim``, or add it to
    :data:`VARIABLES_EXEMPT` with the reason it cannot have one.
    """
    undeclared = {
        entry.name for entry in registry.all_systems() if not getattr(entry.cls, "variables", None)
    }
    unexplained = sorted(undeclared - set(VARIABLES_EXEMPT))
    assert not unexplained, (
        "these systems declare no `variables`, so traj['x'] raises for them: "
        f"{unexplained}. Add a `variables = (...)` ClassVar naming each component "
        "(names taken from the system's own equations / docstring), or add the "
        "system to VARIABLES_EXEMPT with the reason."
    )


def test_variables_exemptions_are_live_and_justified() -> None:
    """Every exemption still exists, is still undeclared, and is variable-dim."""
    for name, reason in VARIABLES_EXEMPT.items():
        entry = registry.get(name)
        assert not getattr(entry.cls, "variables", None), (
            f"{name} now declares `variables` — drop it from VARIABLES_EXEMPT."
        )
        assert entry.dim is None, (
            f"{name} has a fixed dim={entry.dim}; a fixed-dim system can name its "
            "components, so it must not be exempt."
        )
        assert reason.strip(), f"{name}: exemption needs a reason."


def test_named_component_access_works_for_every_declaring_system(system_entry) -> None:
    """``traj[name]`` resolves for every declared name (the README's front door).

    Exercised on a synthetic trajectory so the gate stays in the fast tier and
    covers *all* 166 declaring systems rather than an integrable subset.
    """
    names = getattr(system_entry.cls, "variables", None)
    if not names:
        assert system_entry.name in VARIABLES_EXEMPT
        return

    import numpy as np

    from tsdynamics.data import Trajectory

    dim = len(names)
    y = np.arange(6 * dim, dtype=float).reshape(6, dim)
    traj = Trajectory(t=np.arange(6, dtype=float), y=y, system=system_entry.cls())
    assert traj.variables == tuple(names)
    for i, name in enumerate(names):
        np.testing.assert_array_equal(traj[name], y[:, i])
    assert len(set(names)) == dim, f"{system_entry.name}: duplicate component names {names}"


# ---------------------------------------------------------------------------
# Curated-sample guards — keep the slow tier representative
# ---------------------------------------------------------------------------


def test_integration_sample_names_exist() -> None:
    for name in INTEGRATION_SAMPLE:
        assert registry.get(name).family == "ode"


def test_integration_sample_covers_every_ode_category() -> None:
    """Every ODE category needs >= 2 representatives in the slow tier.

    Exception: categories in ``HEAVY_FIELD_CATEGORIES`` (high-dim method-of-lines
    PDE fields, too heavy for the integration / reference-xval sweeps) are
    deliberately not sampled — they are covered by the small-grid viz field tests.
    """
    from _sampling import HEAVY_FIELD_CATEGORIES

    sample_categories: dict[str, int] = {}
    for name in INTEGRATION_SAMPLE:
        cat = registry.get(name).category
        sample_categories[cat] = sample_categories.get(cat, 0) + 1
    for category in registry.categories("ode"):
        if category in HEAVY_FIELD_CATEGORIES:
            continue
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


def test_sde_samples_complete() -> None:
    """Every built-in SDE system needs a seed/ic entry in SDE_SAMPLES.

    The diagonal-Itô analogue of ``test_dde_histories_complete``.  Trivially
    satisfied today (no built-in ``sde`` systems), it fails loudly the moment a
    built-in :class:`~tsdynamics.StochasticSystem` is added without its sample —
    the same "just write the class" safety the DDE guard provides.
    """
    sde_names = {e.name for e in registry.all_systems(family="sde")}
    assert sde_names == set(SDE_SAMPLES), (
        f"SDE_SAMPLES (tests/_sampling.py) out of sync with registry: "
        f"missing {sde_names - set(SDE_SAMPLES)}, stale {set(SDE_SAMPLES) - sde_names}"
    )


# ---------------------------------------------------------------------------
# One solver registry, not two
# ---------------------------------------------------------------------------


def test_solver_registry_is_not_duplicated_in_registry_module() -> None:
    """Solvers register in ``tsdynamics.solvers``, never in ``registry``.

    ``registry`` keeps only the *reserved* generic ``analyses``/``renderers``
    seams; the solver registry is the richer ``SolverSpec`` table in
    ``tsdynamics.solvers``. A stray ``registry.solvers`` would resurrect the
    two-registries-for-one-thing split this guard exists to prevent.
    """
    from tsdynamics import solvers
    from tsdynamics.registry import Registry

    assert not hasattr(registry, "solvers"), (
        "registry.solvers is back — solvers belong in tsdynamics.solvers"
    )
    assert isinstance(registry.analyses, Registry)
    assert isinstance(registry.renderers, Registry)
    # The real solver registry exposes the SolverSpec-based API.
    assert hasattr(solvers, "register") and hasattr(solvers, "available")
