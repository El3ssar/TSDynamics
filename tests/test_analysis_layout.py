"""Tests for the A-LAYOUT analysis subpackage restructure.

Covers the stream's acceptance:

* the public API is preserved through the move/rename (``from tsdynamics import
  lyapunov_spectrum, fixed_points, orbit_diagram, poincare_section`` and the
  ``tsdynamics.analysis`` re-exports are unchanged objects);
* the new per-stream subpackages exist and the old flat module paths are gone;
* the canonical (definition-site) paths the docs reference resolve; and
* the ``tsdynamics.analyses`` / ``tsdynamics.transforms`` plugin kinds now have a
  consumer — an out-of-tree plugin is discovered into the generic registries.

The out-of-tree tests are hermetic: they synthesize a fake installed
distribution on a temporary ``sys.path`` entry and let the real
``importlib.metadata`` machinery find it — no ``pip install``, no network
(mirrors ``tests/test_solver_registry.py``).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

import tsdynamics as ts
from tsdynamics import analysis, plugins, registry, transforms

# ── public API preservation ─────────────────────────────────────────────────────

_PUBLIC = [
    "lyapunov_spectrum",
    "max_lyapunov",
    "kaplan_yorke_dimension",
    "fixed_points",
    "FixedPoint",
    "orbit_diagram",
    "OrbitDiagram",
    "poincare_section",
]


@pytest.mark.parametrize("name", _PUBLIC)
def test_top_level_reexports_unchanged(name):
    """Every analysis symbol is importable from the top level and from analysis."""
    assert hasattr(ts, name), f"tsdynamics.{name} disappeared"
    assert getattr(ts, name) is getattr(analysis, name)


def test_analysis_all_is_stable():
    assert set(analysis.__all__) == set(_PUBLIC)


# ── new subpackage layout ───────────────────────────────────────────────────────

_SUBPACKAGES = [
    "lyapunov",
    "fixedpoints",
    "orbits",
    "chaos",
    "basins",
    "dimensions",
    "embedding",
    "entropy",
    "recurrence",
    "surrogate",
]


@pytest.mark.parametrize("pkg", _SUBPACKAGES)
def test_subpackage_importable(pkg):
    mod = importlib.import_module(f"tsdynamics.analysis.{pkg}")
    assert mod.__name__ == f"tsdynamics.analysis.{pkg}"
    # All A-* subpackages declare an __all__ (empty for the placeholders).
    assert isinstance(mod.__all__, list)


@pytest.mark.parametrize(
    "pkg", ["chaos", "basins", "dimensions", "embedding", "entropy", "recurrence", "surrogate"]
)
def test_placeholder_subpackages_are_empty(pkg):
    mod = importlib.import_module(f"tsdynamics.analysis.{pkg}")
    assert mod.__all__ == []


@pytest.mark.parametrize(
    "path",
    [
        "tsdynamics.analysis.lyapunov",
        "tsdynamics.analysis.fixedpoints",
        "tsdynamics.analysis.orbits",
        "tsdynamics.analysis.orbits.orbit_diagram",
        "tsdynamics.analysis.orbits.poincare",
    ],
)
def test_canonical_definition_sites_resolve(path):
    """The definition-site module paths the docs reference all import."""
    importlib.import_module(path)


def test_canonical_symbols_live_at_definition_sites():
    from tsdynamics.analysis.fixedpoints import FixedPoint, fixed_points
    from tsdynamics.analysis.lyapunov import (
        kaplan_yorke_dimension,
        lyapunov_spectrum,
        max_lyapunov,
    )
    from tsdynamics.analysis.orbits.orbit_diagram import OrbitDiagram, orbit_diagram
    from tsdynamics.analysis.orbits.poincare import poincare_section

    assert fixed_points is ts.fixed_points
    assert FixedPoint is ts.FixedPoint
    assert lyapunov_spectrum is ts.lyapunov_spectrum
    assert max_lyapunov is ts.max_lyapunov
    assert kaplan_yorke_dimension is ts.kaplan_yorke_dimension
    assert orbit_diagram is ts.orbit_diagram
    assert OrbitDiagram is ts.OrbitDiagram
    assert poincare_section is ts.poincare_section


@pytest.mark.parametrize(
    "old_path",
    [
        "tsdynamics.analysis.fixed_points",  # renamed → fixedpoints
        "tsdynamics.analysis.orbit_diagram",  # moved → orbits.orbit_diagram
        "tsdynamics.analysis.poincare",  # moved → orbits.poincare
    ],
)
def test_old_flat_module_paths_are_gone(old_path):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old_path)


# ── analyses / transforms plugin discovery (the new consumers) ──────────────────


@pytest.fixture
def clean_generic_registries():
    """Snapshot the generic analyses/transforms registries; restore afterwards."""
    before = {
        "analyses": set(registry.analyses.names()),
        "transforms": set(registry.transforms.names()),
    }
    yield
    for reg, kind in ((registry.analyses, "analyses"), (registry.transforms, "transforms")):
        for name in list(reg.names()):
            if name not in before[kind]:
                reg.unregister(name)


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


def test_analysis_discover_plugins_registers_out_of_tree(
    tmp_path, monkeypatch, clean_generic_registries
):
    site = tmp_path / "site"
    _write_fake_distribution(
        site,
        dist="toy-analysis",
        module="toy_analysis_pkg",
        group=plugins.ANALYSES_GROUP,
        ep_name="toy_count",
        target="toy_analysis_pkg:analyze",
        body="def analyze(traj):\n    return len(traj)\n",
    )
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    try:
        newly = analysis.discover_plugins(strict=True)
        assert "toy_count" in newly
        assert "toy_count" in registry.analyses
        assert registry.analyses.get("toy_count")(range(7)) == 7
        # Idempotent: a second pass registers nothing new.
        assert analysis.discover_plugins(strict=True) == []
    finally:
        sys.modules.pop("toy_analysis_pkg", None)
        importlib.invalidate_caches()


def test_transforms_discover_plugins_registers_out_of_tree(
    tmp_path, monkeypatch, clean_generic_registries
):
    site = tmp_path / "site"
    _write_fake_distribution(
        site,
        dist="toy-transform",
        module="toy_transform_pkg",
        group=plugins.TRANSFORMS_GROUP,
        ep_name="toy_double",
        target="toy_transform_pkg:transform",
        body="def transform(x):\n    return [2 * v for v in x]\n",
    )
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    try:
        newly = transforms.discover_plugins(strict=True)
        assert "toy_double" in newly
        assert "toy_double" in registry.transforms
        assert registry.transforms.get("toy_double")([1, 2, 3]) == [2, 4, 6]
    finally:
        sys.modules.pop("toy_transform_pkg", None)
        importlib.invalidate_caches()


def test_register_entry_points_skips_existing(clean_generic_registries):
    """A name already present is not overwritten and not reported as new."""

    def first(_):
        return "first"

    registry.analyses.register("dup_probe", first)
    newly = plugins.register_entry_points(registry.analyses, plugins.ANALYSES_GROUP)
    assert "dup_probe" not in newly  # discovery never re-touches an existing name
    assert registry.analyses.get("dup_probe") is first
