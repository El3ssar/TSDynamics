"""Regenerate representative scientific illustrations through their public CLIs."""

from __future__ import annotations

import importlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLING = ROOT / "docs" / "_tooling"


@pytest.mark.parametrize(
    ("module", "slugs"),
    [
        ("make_analysis_figures", ["fixed-points", "basins", "poincare", "lyapunov"]),
        ("make_tutorial_figures", ["basins-image", "ou-ensemble", "lyapunov-readout"]),
        ("make_viz_figures", ["styling", "kind-phase-3d"]),
    ],
)
def test_static_generator_cli_draws_current_api_and_restores_caller_style(
    module, slugs, monkeypatch, tmp_path
):
    mpl = pytest.importorskip("matplotlib")
    import tsdynamics as ts

    monkeypatch.syspath_prepend(str(TOOLING))
    tool = importlib.import_module(module)
    monkeypatch.setattr(tool, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(TOOLING / f"{module}.py"), "--only", *slugs])
    mpl.get_backend()
    theme = ts.viz.get_theme()
    before = dict(mpl.rcParams)
    tool.main()
    assert dict(mpl.rcParams) == before
    assert ts.viz.get_theme() == theme
    for slug in slugs:
        output = tmp_path / f"{slug}.svg"
        root = ET.parse(output).getroot()
        assert root.tag.endswith("svg")
        assert any(node.tag.endswith("path") for node in root.iter())
        assert output.stat().st_size > 5000


def test_styled_spectrum_preserves_signed_estimates_as_discrete_points(monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(TOOLING))
    recipes = importlib.import_module("make_viz_recipes")
    namespace = {}
    exec(compile(recipes.source("recipe-analysis"), "analysis-recipe", "exec"), namespace)
    spectrum = namespace["figure"][0]
    values = namespace["rates"].values
    point_layer = next(layer for layer in spectrum.layers if layer.kind.value == "scatter")
    np.testing.assert_array_equal(point_layer.data["y"], np.sort(values)[::-1])
    assert values.min() < -10 < 0 < values.max()
    assert all(layer.kind.value != "bar" for layer in spectrum.layers)


def test_dark_system_portraits_remain_opaque_through_png_export(monkeypatch, tmp_path):
    mpl = pytest.importorskip("matplotlib")
    image = pytest.importorskip("PIL.Image")
    monkeypatch.syspath_prepend(str(TOOLING))
    figures = importlib.import_module("figures")
    catalog = importlib.import_module("catalog")
    monkeypatch.setattr(figures, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(figures, "OUT_DIR", tmp_path / "figures")
    monkeypatch.setitem(figures.FIG_OVERRIDES, "Henon", {})
    with mpl.rc_context({"savefig.facecolor": "white", "savefig.transparent": True}):
        output = figures.render(catalog.load_catalog().by_name("Henon"))
    assert output is not None
    with image.open(output) as actual:
        rgba = np.asarray(actual.convert("RGBA"))
    assert (rgba[..., 3] == 255).all()
    assert (rgba[0, 0, :3] < 30).all()
    # Ensure this is an actual bright point cloud, not an empty dark card.
    assert np.count_nonzero(rgba[..., 1] > 100) > 500


def test_shared_illustration_style_participates_in_gallery_and_model_cache(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLING))
    figures = importlib.import_module("figures")
    for whole_library in (False, True):
        labels = {row[0] for row in figures._render_source_signature(whole_library=whole_library)}
        assert "tooling/figure_style.py" in labels
