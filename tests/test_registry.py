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


def test_registry_matches_the_systems_namespace() -> None:
    """``ts.systems.<Name>`` is the canonical path, and it *is* the registered class.

    It used to be ``ts.<Name>``.  Since v6 the top level is seventeen names and a
    built-in system resolves at one address; the registry is how tooling finds
    them, so the two must not be able to disagree.
    """
    for entry in registry.all_systems():
        assert getattr(ts.systems, entry.name) is entry.cls


# ---------------------------------------------------------------------------
# Catalogue metadata reaches the entry (the 155-dropped-DOI headline bug)
# ---------------------------------------------------------------------------


def test_system_entry_carries_the_catalogue_metadata_fields() -> None:
    """``SystemEntry`` has somewhere to put a DOI, a field shape and field labels.

    It did not, which is why the docs tool read ``None`` for all 155 declared
    DOIs: the values were on the classes the whole time and the record they were
    copied into had no slot for them.
    """
    fields = {f.name for f in __import__("dataclasses").fields(registry.SystemEntry)}
    assert {"reference", "doi", "field_shape", "field_labels", "known_lyapunov"} <= fields


def test_catalogue_metadata_reaches_the_registry() -> None:
    """Every value a catalogue class declares must arrive on its ``SystemEntry``.

    This is the gate for the failure mode that has now bitten this repo twice:
    **renaming a ClassVar whose absence is legal orphans its readers silently.**
    ``None`` is a legal value for every field below, so a reader that looks under
    only one spelling records "no citation" for all 177 systems and *nothing
    raises* — the docs lose their bibliography, ``test_known_values.py`` quietly
    stops comparing against literature, and the suite stays green.

    So the assertion is not "the field exists" but "the declared value arrived",
    counted, with a floor that a silent regression cannot clear.
    """
    entries = registry.all_systems()
    counts = {
        field: sum(1 for e in entries if getattr(e, field) is not None)
        for field in ("reference", "doi", "known_lyapunov")
    }
    assert counts["reference"] >= 170, f"citations stopped reaching the registry: {counts}"
    assert counts["doi"] >= 150, f"DOIs stopped reaching the registry: {counts}"
    assert counts["known_lyapunov"] >= 20, f"known_lyapunov stopped arriving: {counts}"

    # ...and spot-check one system end to end, under whichever spelling it declares.
    lorenz = registry.get("Lorenz")
    assert lorenz.doi and lorenz.doi.startswith("10."), lorenz.doi
    assert lorenz.reference and "Lorenz" in lorenz.reference


def test_metadata_is_read_under_both_spellings() -> None:
    """The dual read is deliberate, and it prefers the underscored ClassVar.

    v6 moves these ClassVars behind an underscore so they stay off
    ``system.<TAB>``.  Reading both spellings is what makes the declaration side
    of that rename a separable change instead of a silent data loss, and the
    underscored one wins so the migration is a real migration.
    """

    class Both:
        _doi = "10.underscored"
        doi = "10.bare"

    class BareOnly:
        doi = "10.bare"

    assert registry._classvar(Both, "doi") == "10.underscored"
    assert registry._classvar(BareOnly, "doi") == "10.bare"
    assert registry._classvar(object, "doi") is None


def test_field_metadata_reaches_the_registry() -> None:
    """A spatially-extended system's grid and block names arrive on the entry."""
    gs = registry.get("GrayScott")
    assert gs.field_shape is not None and len(gs.field_shape) == 2
    assert gs.field_labels == ("u", "v")


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
    assert registry.get("Lorenz").cls is ts.systems.Lorenz
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
    """Solvers register in ``tsdynamics._solvers``, never in ``registry``.

    ``registry`` keeps only the *reserved* generic
    ``analyses``/``renderers``/``plot_transforms`` seams; the solver registry is
    the richer ``SolverSpec`` table in ``tsdynamics._solvers``. A stray
    ``registry.solvers`` would resurrect the two-registries-for-one-thing split
    this guard exists to prevent.
    """
    from tsdynamics import _solvers as solvers
    from tsdynamics.registry import Registry

    assert not hasattr(registry, "solvers"), (
        "registry.solvers is back — solvers belong in tsdynamics._solvers"
    )
    assert isinstance(registry.analyses, Registry)
    assert isinstance(registry.renderers, Registry)
    assert isinstance(registry.plot_transforms, Registry)
    # The real solver registry exposes the SolverSpec-based API.
    assert hasattr(solvers, "register") and hasattr(solvers, "available")


def test_plot_transform_registry_is_created_empty_and_filled_by_viz() -> None:
    """``registry.plot_transforms`` mirrors ``renderers``: empty until ``viz`` loads.

    The registry module is imported while ``tsdynamics`` itself is still
    initialising, so it may only depend on the standard library — which is the
    same reason the plot transforms cannot be registered there.  They arrive when
    ``tsdynamics.viz`` is first imported, so a plain ``import tsdynamics`` still
    pulls in no plotting machinery.
    """
    import subprocess
    import sys

    code = (
        "import tsdynamics as ts\n"
        "from tsdynamics import registry\n"
        "assert len(registry.plot_transforms) == 0, 'transforms registered too early'\n"
        "ts.viz\n"
        "assert 'phase_portrait' in registry.plot_transforms\n"
        "assert registry.plot_transforms.kind == 'plot transform'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


#: Every declared plugin group, and the module that must actually load it.
#:
#: A declared group is a promise.  ``SYSTEMS_GROUP`` was declared inside "the
#: frozen contract a plugin author declares against", listed in ``ALL_GROUPS``,
#: and loaded by **nothing** — so a third-party package publishing
#: ``tsdynamics.systems`` entry points was silently ignored, with no error to
#: debug, for as long as the group has existed.
_GROUP_CONSUMERS = {
    "SYSTEMS_GROUP": "tsdynamics.systems",
    "SOLVERS_GROUP": "tsdynamics._solvers",
    "ANALYSES_GROUP": "tsdynamics.analysis",
    "RENDERERS_GROUP": "tsdynamics.viz",
    "PLOT_TRANSFORMS_GROUP": "tsdynamics.viz",
    "PLOT_PRIMITIVES_GROUP": "tsdynamics.viz",
}


def test_every_declared_plugin_group_is_in_all_groups() -> None:
    """The six extension doors are declared, and ``ALL_GROUPS`` is the whole set."""
    from tsdynamics import plugins

    declared = {getattr(plugins, name) for name in _GROUP_CONSUMERS}
    assert set(plugins.ALL_GROUPS) == declared
    assert len(plugins.ALL_GROUPS) == 6, plugins.ALL_GROUPS
    assert plugins.PLOT_PRIMITIVES_GROUP == "tsdynamics.plot_primitives"


#: Groups whose loader belongs to another v6 slot and has not landed yet.
#: ``strict`` xfails, so each one **fails the moment it starts passing** and the
#: row has to be deleted — the repo's established cross-slot handoff.
_LOADER_NOT_LANDED = {
    # §7.1: "SYSTEMS_GROUP is advertised and dead ... systems/__init__.py gains
    # the loader."  Owner: C9 · CATALOGUE.
    "SYSTEMS_GROUP": "C9 - CATALOGUE owns systems/__init__.py (contract 7.1)",
}


@pytest.mark.parametrize(
    "group_name",
    [
        pytest.param(
            name,
            marks=(
                [pytest.mark.xfail(strict=True, reason=_LOADER_NOT_LANDED[name])]
                if name in _LOADER_NOT_LANDED
                else []
            ),
        )
        for name in sorted(_GROUP_CONSUMERS)
    ],
)
def test_every_declared_plugin_group_has_a_consumer(group_name) -> None:
    """Some module must actually *load* each declared group.

    Detected by looking for a **loader call** — ``load_plugins`` /
    ``register_entry_points`` — applied to the group, named either by its literal
    string or by any local alias of the group constant (the consumer imports it
    under an underscored name and re-exports it under a shorter one).  Matching
    the group *string* alone would be useless: ``"tsdynamics.systems"`` appears in
    every module path under ``tsdynamics/systems/``.

    .. versionchanged:: 6.0.1
        A bare **mention** of the constant used to satisfy this gate, so an
        ``import`` with no loader behind it read as healthy.  That is not why
        ``PLOT_PRIMITIVES_GROUP`` stayed dead — it was a tracked ``strict`` xfail
        — but it is why the fix could have been faked.
    """
    import importlib
    import pathlib
    import re

    from tsdynamics import plugins

    group = getattr(plugins, group_name)
    consumer = importlib.import_module(_GROUP_CONSUMERS[group_name])
    root = pathlib.Path(consumer.__file__).parent
    sources = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.py"))

    # Every local spelling of the group: the constant, its import aliases, and
    # anything assigned from one of those at module level.
    names = {group_name}
    for _ in range(3):  # resolve chains (import as _X; Y = _X)
        for pattern in (
            rf"\b({'|'.join(names)})\s+as\s+(\w+)",
            rf"(\w+)\s*=\s*({'|'.join(names)})\b",
        ):
            for match in re.finditer(pattern, sources):
                names.update({match.group(1), match.group(2)})
    loads = any(
        re.search(rf"(load_plugins|register_entry_points)\([^)]*\b{name}\b", sources)
        for name in names
    ) or re.search(rf"""(load_plugins|register_entry_points)\([^)]*{re.escape(group)}""", sources)
    assert loads, (
        f"{group_name} ({group!r}) is declared in plugins.ALL_GROUPS but "
        f"{_GROUP_CONSUMERS[group_name]} never loads it — a declared group that "
        "nothing reads silently ignores every plugin that declares against it"
    )


def test_plot_transform_entry_point_group_is_declared() -> None:
    """Out-of-tree transforms have a declared, discovered entry-point group.

    Third-party plots must be genuinely first-class — the same registration path,
    the same compatibility row, the same front door — which needs the group to be
    in ``ALL_GROUPS`` *and* actually loaded by ``viz.discover_plugins``.
    """
    import tsdynamics.viz as viz
    from tsdynamics import plugins

    assert plugins.PLOT_TRANSFORMS_GROUP == "tsdynamics.plot_transforms"
    assert plugins.PLOT_TRANSFORMS_GROUP in plugins.ALL_GROUPS
    assert viz.TRANSFORMS_GROUP == plugins.PLOT_TRANSFORMS_GROUP


def test_a_third_party_primitive_published_under_the_advertised_group_registers(
    tmp_path, monkeypatch
) -> None:
    """A primitive installed under ``tsdynamics.plot_primitives`` must arrive.

    The group has been advertised in ``ALL_GROUPS`` since v6 while
    ``viz.discover_plugins`` loaded only renderers and plot transforms, so a
    package declaring one was silently ignored — and with it the *only* route a
    custom primitive has into a shipped transform, ``ts.viz.transforms.allow``.

    The distribution is a hand-built ``.dist-info`` on ``sys.path``:
    :mod:`importlib.metadata` reads it exactly as an installed one, so this
    needs no wheel build and no environment mutation.
    """
    import sys

    import tsdynamics.viz as viz

    (tmp_path / "tsd_primitive_plugin.py").write_text(
        "import numpy as np\n"
        "def stem(part, **options):\n"
        "    '''A vertical drop to the baseline plus a marker at each point.'''\n"
        "    x, y = np.asarray(part['x']), np.asarray(part['y'])\n"
        "    return [{'mark': 'points', 'x': x, 'y': y}]\n"
        "stem.requires = ('x', 'y')\n"
        "stem.marks = ('points',)\n"
    )
    dist = tmp_path / "tsd_primitive_plugin-0.0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: tsd-primitive-plugin\nVersion: 0.0.1\n"
    )
    (dist / "entry_points.txt").write_text(
        "[tsdynamics.plot_primitives]\ntsd_stem = tsd_primitive_plugin:stem\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    from importlib.metadata import MetadataPathFinder

    MetadataPathFinder.invalidate_caches()

    from tsdynamics import registry as _registry

    # ``allow`` RE-REGISTERS the shipped transform with a widened row, and that
    # outlives this test.  Popping the primitive without restoring the row left
    # ``time_series.primitives`` naming a primitive no longer in ``PRIMITIVES``,
    # so every later test that read its option row died with ``KeyError:
    # 'tsd_stem'`` — 8 of them, in a different FILE, only when this one ran first.
    original = viz.transforms.get("time_series")

    def _restore_row() -> None:
        _registry.plot_transforms.register(
            original.name,
            original,
            replace=True,
            source=original.source,
            default_primitive=original.default_primitive,
            requires=original.requires,
        )

    try:
        assert "tsd_stem" in viz.discover_plugins()
        assert "tsd_stem" in viz.primitives.names()
        # The documented escape hatch: a custom primitive reaches a shipped
        # transform only through ``allow``, which needs it registered first.
        widened = viz.transforms.allow("time_series", "tsd_stem")
        assert "tsd_stem" in widened.primitives
        assert "tsd_stem" in viz.transforms.get("time_series").primitives
    finally:
        from tsdynamics.viz.transforms._primitives import PRIMITIVES

        _restore_row()
        PRIMITIVES.pop("tsd_stem", None)
        sys.modules.pop("tsd_primitive_plugin", None)

    assert "tsd_stem" not in viz.transforms.get("time_series").primitives
