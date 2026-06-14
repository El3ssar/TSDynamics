"""Tests for the Python solver registry + entry-point plugin discovery (F2).

Covers the F2 acceptance criteria on the Python side:

* a (dummy) solver registers and is discoverable **by name** from Python; and
* entry-point discovery loads an **out-of-tree** toy plugin.

The out-of-tree tests are hermetic: they synthesize a fake installed
distribution (`*.dist-info` + module) on a temporary ``sys.path`` entry and let
the real :func:`importlib.metadata.entry_points` machinery find it — no
``pip install`` and no network.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from tsdynamics import plugins, solvers
from tsdynamics.solvers import SolverCaps, SolverSpec


@pytest.fixture
def clean_registry():
    """Snapshot the global solver registry and restore it after the test."""
    before = solvers.all_specs()
    yield
    for name in list(solvers.all_specs()):
        if name not in before:
            solvers.unregister(name)
    for spec in before.values():
        solvers.register(spec, override=True)


# ── basic registry semantics ───────────────────────────────────────────────────


def test_import_is_clean():
    assert isinstance(solvers.available(), list)
    assert plugins.ALL_GROUPS == (
        "tsdynamics.systems",
        "tsdynamics.solvers",
        "tsdynamics.analyses",
        "tsdynamics.transforms",
    )


def test_register_and_discover_by_name(clean_registry):
    spec = SolverSpec(
        name="dummy",
        caps=SolverCaps(kind="explicit", supports={"ode"}),
        description="F2 dummy",
    )
    solvers.register(spec)

    assert "dummy" in solvers.available()
    got = solvers.get("dummy")
    assert got is spec
    assert got.kernel == "dummy"  # defaults to name
    assert got.caps.kind == "explicit"
    assert got.caps.supports_family("ode")
    assert not got.caps.supports_family("sde")


def test_duplicate_registration_raises(clean_registry):
    spec = SolverSpec(name="dup", caps=SolverCaps(kind="explicit"))
    solvers.register(spec)
    with pytest.raises(ValueError, match="already registered"):
        solvers.register(spec)
    # override replaces silently
    solvers.register(SolverSpec(name="dup", caps=SolverCaps(kind="implicit")), override=True)
    assert solvers.get("dup").caps.kind == "implicit"


def test_unknown_solver_lists_available(clean_registry):
    solvers.register(SolverSpec(name="known", caps=SolverCaps(kind="explicit")))
    with pytest.raises(KeyError, match="unknown solver 'nope'"):
        solvers.get("nope")


def test_unregister(clean_registry):
    solvers.register(SolverSpec(name="temp", caps=SolverCaps(kind="explicit")))
    assert solvers.unregister("temp") is True
    assert solvers.unregister("temp") is False
    assert "temp" not in solvers.available()


# ── spec / caps validation ─────────────────────────────────────────────────────


def test_caps_rejects_bad_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        SolverCaps(kind="bogus")


def test_caps_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown problem families"):
        SolverCaps(kind="explicit", supports={"quantum"})


def test_caps_normalises_supports_to_frozenset():
    caps = SolverCaps(kind="explicit", supports={"ode", "map"})
    assert isinstance(caps.supports, frozenset)
    assert caps.supports == frozenset({"ode", "map"})


def test_spec_kernel_defaults_to_name_and_rejects_empty():
    assert SolverSpec(name="rk4", caps=SolverCaps(kind="explicit")).kernel == "rk4"
    assert (
        SolverSpec(name="rk4", caps=SolverCaps(kind="explicit"), kernel="rk4_v2").kernel == "rk4_v2"
    )
    with pytest.raises(ValueError, match="non-empty"):
        SolverSpec(name="", caps=SolverCaps(kind="explicit"))


# ── directory-scan primitive ────────────────────────────────────────────────────


def test_import_submodules_scans_public_modules(tmp_path, monkeypatch):
    pkg = tmp_path / "scanpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "alpha.py").write_text("VALUE = 1\n")
    (pkg / "beta.py").write_text("VALUE = 2\n")
    (pkg / "_private.py").write_text("raise RuntimeError('should not be imported')\n")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        scanpkg = importlib.import_module("scanpkg")
        imported = plugins.import_submodules(scanpkg)
        assert set(imported) == {"alpha", "beta"}  # underscore module skipped
        assert imported["alpha"].VALUE == 1
        assert imported["beta"].VALUE == 2
    finally:
        for mod in ("scanpkg", "scanpkg.alpha", "scanpkg.beta"):
            sys.modules.pop(mod, None)


# ── out-of-tree entry-point plugin discovery (headline acceptance) ──────────────


def _write_fake_distribution(
    site: Path, *, dist: str, module: str, group: str, ep_name: str, target: str, body: str
) -> None:
    """Write a synthetic installed distribution (module + ``.dist-info``)."""
    site.mkdir(parents=True, exist_ok=True)
    (site / f"{module}.py").write_text(body)
    info = site / f"{dist}-0.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.0.0\n")
    (info / "entry_points.txt").write_text(f"[{group}]\n{ep_name} = {target}\n")


def test_out_of_tree_plugin_is_discovered(tmp_path, monkeypatch, clean_registry):
    site = tmp_path / "site"
    _write_fake_distribution(
        site,
        dist="toy-solver",
        module="toy_solver_pkg",
        group=plugins.SOLVERS_GROUP,
        ep_name="toy",
        target="toy_solver_pkg:TOY",
        body=(
            "from tsdynamics.solvers import SolverSpec, SolverCaps\n"
            "TOY = SolverSpec(name='toy',\n"
            "                 caps=SolverCaps(kind='explicit', supports={'ode'}),\n"
            "                 description='out-of-tree toy', origin='plugin')\n"
        ),
    )
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    try:
        # The generic loader sees the entry point...
        loaded = plugins.load_plugins(plugins.SOLVERS_GROUP, strict=True)
        assert "toy" in loaded
        assert isinstance(loaded["toy"], SolverSpec)

        # ...and the solver registry registers it by name.
        newly = solvers.discover_plugins(strict=True)
        assert "toy" in newly
        spec = solvers.get("toy")
        assert spec.origin == "plugin"
        assert spec.caps.supports_family("ode")
    finally:
        sys.modules.pop("toy_solver_pkg", None)
        importlib.invalidate_caches()


def test_broken_plugin_is_isolated(tmp_path, monkeypatch, clean_registry):
    site = tmp_path / "site"
    _write_fake_distribution(
        site,
        dist="broken-solver",
        module="broken_solver_pkg",
        group=plugins.SOLVERS_GROUP,
        ep_name="broken",
        target="broken_solver_pkg:SPEC",
        body="raise RuntimeError('boom on import')\n",
    )
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    try:
        # Non-strict: a broken plugin warns and is skipped, never breaking import.
        with pytest.warns(UserWarning, match="failed to load plugin 'broken'"):
            loaded = plugins.load_plugins(plugins.SOLVERS_GROUP, strict=False)
        assert "broken" not in loaded

        # Strict: the failure propagates (used by tests / debugging).
        with pytest.raises(RuntimeError, match="boom on import"):
            plugins.load_plugins(plugins.SOLVERS_GROUP, strict=True)
    finally:
        sys.modules.pop("broken_solver_pkg", None)
        importlib.invalidate_caches()
