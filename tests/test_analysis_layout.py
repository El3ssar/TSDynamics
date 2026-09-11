"""Tests for the A-LAYOUT analysis subpackage restructure.

Covers the stream's acceptance:

* the public API is preserved through the move/rename (``from tsdynamics import
  lyapunov_spectrum, fixed_points, orbit_diagram, poincare_section`` and the
  ``tsdynamics.analysis`` re-exports are unchanged objects);
* the new per-stream subpackages exist and the old flat module paths are gone;
* the canonical (definition-site) paths the docs reference resolve; and
* the ``tsdynamics.analyses`` plugin kind now has a consumer — an out-of-tree
  plugin is discovered into the generic registry.

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
from tsdynamics import analysis, plugins, registry

# ── public API preservation ─────────────────────────────────────────────────────

#: The analyses A-LAYOUT moved.  In v6 they live at ONE address —
#: ``ts.analysis.<name>`` — because the top level is 17 names (CONTRACT §2.1).
_PUBLIC = [
    "lyapunov_spectrum",
    "max_lyapunov",
    "kaplan_yorke_dimension",
    "fixed_points",
    "orbit_diagram",
    "poincare_section",
    # chaos indicators (stream A-CHAOS)
    "gali",
    "zero_one_test",
    "expansion_entropy",
    # recurrence & RQA (stream A-RQA)
    "recurrence_matrix",
    "rqa",
    "windowed_rqa",
]

#: Result classes A-LAYOUT moved.  They are reachable on ``ts.analysis`` and
#: listed at ``ts.analysis.results`` — never on the tab surface (C2: a type you
#: only ever get *back*).
_PUBLIC_RESULTS = ["FixedPoint", "OrbitDiagram"]


@pytest.mark.parametrize("name", _PUBLIC)
def test_the_analysis_survived_the_move_at_its_one_address(name):
    """Every analysis symbol resolves on ``ts.analysis`` and is on its tab surface."""
    assert hasattr(analysis, name), f"tsdynamics.analysis.{name} disappeared"
    assert name in analysis.__all__


@pytest.mark.parametrize("name", _PUBLIC + _PUBLIC_RESULTS)
def test_the_demoted_name_redirects_from_the_top_level(name):
    """``ts.<name>`` is gone, and the error names the address that replaced it."""
    with pytest.raises((AttributeError, ImportError)) as err:
        getattr(ts, name)
    assert "ts.analysis" in str(err.value)


@pytest.mark.parametrize("name", _PUBLIC_RESULTS)
def test_a_result_class_is_reachable_but_off_the_tab_surface(name):
    assert getattr(analysis, name) is getattr(analysis.results, name)
    assert name not in analysis.__all__


def test_analysis_all_is_stable():
    # The A-LAYOUT public surface must remain exported; analysis streams (A-DIM,
    # A-CHAOS, …) append to __all__, so this is a subset check, not equality.
    assert set(_PUBLIC) <= set(analysis.__all__)


# ── new subpackage layout ───────────────────────────────────────────────────────

_SUBPACKAGES = [
    "lyapunov",
    "fixedpoints",
    "orbits",
    "chaos",
    "basins",
    "dimensions",
    "embedding",
    "recurrence",
    "sampling",
]


@pytest.mark.parametrize("pkg", _SUBPACKAGES)
def test_subpackage_importable(pkg):
    mod = importlib.import_module(f"tsdynamics.analysis.{pkg}")
    assert mod.__name__ == f"tsdynamics.analysis.{pkg}"
    # All A-* subpackages declare an __all__ (empty for the placeholders).
    assert isinstance(mod.__all__, list)


@pytest.mark.parametrize("pkg", _SUBPACKAGES)
def test_subpackages_are_filled(pkg):
    # Every A-* subpackage is now filled — ``basins`` (A-BASIN) was the last
    # placeholder.  A real guard (sweeping the actual subpackages) so a future
    # empty stub trips it, unlike an empty-parametrize no-op.
    mod = importlib.import_module(f"tsdynamics.analysis.{pkg}")
    assert mod.__all__, f"tsdynamics.analysis.{pkg} regressed to an empty placeholder"


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

    assert fixed_points is ts.analysis.fixed_points
    assert FixedPoint is ts.analysis.FixedPoint
    assert lyapunov_spectrum is ts.analysis.lyapunov_spectrum
    assert max_lyapunov is ts.analysis.max_lyapunov
    assert kaplan_yorke_dimension is ts.analysis.kaplan_yorke_dimension
    assert orbit_diagram is ts.analysis.orbit_diagram
    assert OrbitDiagram is ts.analysis.OrbitDiagram
    assert poincare_section is ts.analysis.poincare_section


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


# ── analyses plugin discovery (the new consumer) ───────────────────────────────


@pytest.fixture
def clean_generic_registries():
    """Snapshot the generic analyses registry; restore afterwards."""
    before = set(registry.analyses.names())
    yield
    for name in list(registry.analyses.names()):
        if name not in before:
            registry.analyses.unregister(name)


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


def test_register_entry_points_skips_existing(clean_generic_registries):
    """A name already present is not overwritten and not reported as new."""

    def first(_):
        return "first"

    registry.analyses.register("dup_probe", first)
    newly = plugins.register_entry_points(registry.analyses, plugins.ANALYSES_GROUP)
    assert "dup_probe" not in newly  # discovery never re-touches an existing name
    assert registry.analyses.get("dup_probe") is first
