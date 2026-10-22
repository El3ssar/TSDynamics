"""The scientific recipes execute their published code and keep media current."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "docs" / "_tooling" / "make_viz_recipes.py"
spec = importlib.util.spec_from_file_location("tsd_viz_recipes", TOOL)
assert spec is not None and spec.loader is not None
recipes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipes)


def test_recipe_freshness_check_needs_no_plot_backend():
    code = (
        "import importlib.util, sys; "
        f"s=importlib.util.spec_from_file_location('recipes',{str(TOOL)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "assert m.validate_assets() == []; "
        "assert 'matplotlib' not in sys.modules; "
        "assert 'tsdynamics' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_recipe_media_is_complete_and_matches_published_source():
    assert recipes.validate_assets() == []
    manifest = json.loads(recipes.MANIFEST.read_text())
    assert set(manifest["recipes"]) == set(recipes.RECIPES)
    for slug in recipes.RECIPES:
        assert "import tsdynamics as ts" in recipes.source(slug)
        assert "__plot_spec__" not in recipes.source(slug)


def test_editing_a_recipe_requires_its_figure_to_be_regenerated(monkeypatch):
    original = recipes.source
    monkeypatch.setattr(recipes, "source", lambda slug: original(slug) + "\n# edited recipe\n")
    errors = recipes.validate_assets()
    assert errors and "recipe-data" in errors[0]
    assert "uv run --no-sync python docs/_tooling/make_viz_recipes.py" in errors[0]


def test_partial_regeneration_cannot_mark_older_renderer_output_current(monkeypatch, tmp_path):
    manifest = json.loads(recipes.MANIFEST.read_text())
    manifest["recipes"]["recipe-analysis"]["generator_sha256"] = "older renderer"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(recipes, "MANIFEST", path)
    errors = recipes.validate_assets()
    assert errors and "recipe-analysis" in errors[0]


@pytest.mark.parametrize("slug", recipes.RECIPES)
def test_each_actual_recipe_renders_visible_svg_without_leaking_settings(slug, tmp_path):
    mpl = pytest.importorskip("matplotlib")
    import tsdynamics as ts

    before_rc = dict(mpl.rcParams)
    before_theme = ts.viz.themes.get()
    output = recipes.render(slug, tmp_path)
    assert ET.parse(output).getroot().tag.endswith("svg")
    text = output.read_text()
    assert "<path" in text and "<text" in text
    assert output.stat().st_size > 5000
    assert dict(mpl.rcParams) == before_rc
    assert ts.viz.themes.get() == before_theme


def test_measured_signal_recipe_has_correct_frequency_and_power_units():
    pytest.importorskip("matplotlib")
    figure = recipes.build("recipe-data")
    spectrum = figure[1].layers[0].data
    frequency = spectrum["x"]
    power = spectrum["y"]
    assert frequency[np.argmax(power)] == pytest.approx(1.2)
    assert frequency[np.argmax(np.where(frequency > 2, power, 0))] == pytest.approx(3.5)
    # Independent two-tone variance: A1²/2 + A2²/2, in amplitude-squared units.
    assert np.trapezoid(power, frequency) == pytest.approx((1 + 0.3**2) / 2, abs=1e-10)


def test_analysis_recipe_displays_discrete_crossings_as_points():
    pytest.importorskip("matplotlib")
    figure = recipes.build("recipe-analysis")
    assert all(layer.kind.value == "scatter" for layer in figure[2].layers)


def test_recipe_svg_is_reproducible(tmp_path):
    pytest.importorskip("matplotlib")
    first = recipes.render("recipe-data", tmp_path).read_bytes()
    second = recipes.render("recipe-data", tmp_path).read_bytes()
    assert first == second
