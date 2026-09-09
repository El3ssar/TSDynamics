"""The plot gallery's governance gate — the page cannot drift from the registry.

``docs/visualization/gallery.md`` is generated from ``registry.plot_transforms``
at documentation-build time: every registered transform gets an entry, every
primitive in its declared row gets a tab, and every figure is produced by
executing the snippet printed beside it.  That design only pays off if three
things stay true, which is what this module checks:

1. **Coverage** — every registered transform and every declared cell is in the
   gallery.  A transform that nobody curated still appears, drawn on the
   ``example`` fixture its own registration ships, so P1's "adding a transform is
   one registration and nothing else" claim survives contact with this page.
2. **Honesty** — a snippet is valid Python, it evaluates to a ``PlotSpec``, and
   the figure it makes really does carry the primitive its tab is labelled with.
   The interesting failure is not a crash: it is a tab that says ``contour`` over
   a picture of the default ``image``, which nothing else in the suite would
   notice.
3. **The claims the snippets rest on** — chiefly that a style keyword survives
   ``ts.plot(subject, "transform", color=...)``, which the gallery uses and which
   silently dropped the keyword until this phase.

The expensive half (rendering all ~90 cells) is marked ``slow``; the fast tier
runs the structural checks plus a small real render, because a page of figures
that all draw blank would pass every structural check ever written.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "docs" / "_tooling"))

import gallery  # noqa: E402  (docs/_tooling, path-inserted above)

from tsdynamics.viz.render import register_builtin_renderers  # noqa: E402
from tsdynamics.viz.transforms import PRIMITIVES, get, transforms  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _renderers():
    register_builtin_renderers()


@pytest.fixture(scope="module")
def structure():
    """A figure-free build: every cell the page will have, with no rendering."""
    return gallery.render_all(figures=False)


# ---------------------------------------------------------------------------
# 1. coverage — the page is the registry
# ---------------------------------------------------------------------------


def test_the_gallery_covers_every_registered_transform(structure):
    covered = {cell.transform for cell in structure.cells}
    registered = {t.name for t in transforms()}
    assert covered == registered, (
        "the gallery is generated from the registry, so these sets cannot differ; "
        f"missing={sorted(registered - covered)} extra={sorted(covered - registered)}"
    )


def test_the_gallery_covers_every_declared_compatibility_cell(structure):
    """One tab per declared (transform, primitive) pair — the whole matrix, drawn."""
    covered = {(cell.transform, cell.primitive) for cell in structure.cells}
    declared = {(t.name, p) for t in transforms() for p in t.primitives}
    assert covered == declared, f"undrawn cells: {sorted(declared - covered)}"


def test_an_uncurated_transform_still_reaches_the_gallery():
    """Adding a transform must stay "one registration and nothing else".

    A transform with no :data:`gallery.SHOWCASE` entry falls back to the
    ``example`` factory its registration already ships for the compatibility
    gate, so it appears in the gallery the day it is registered.  Losing this
    would make the gallery a second place every new transform has to be
    mentioned, which is exactly the coupling the registry exists to remove.
    """
    record = get("time_series")
    show = gallery._fallback_showcase(record)
    assert show is not None
    variant = show.variant("line", default=record.default_primitive)
    assert "example" in variant.setup
    assert '"time_series"' in variant.call or "'time_series'" in variant.call


def test_every_in_tree_transform_ships_an_example_so_the_fallback_can_work():
    missing = [t.name for t in transforms() if t.example is None]
    assert not missing, f"no example factory (the gallery fallback needs one): {missing}"


# ---------------------------------------------------------------------------
# 2. honesty — the code beside the figure is the code that made it
# ---------------------------------------------------------------------------


def test_every_snippet_parses_and_is_one_expression(structure):
    for cell in structure.cells:
        lines = cell.snippet.split("\n\n")
        assert lines[0] == "import tsdynamics as ts"
        ast.parse(cell.snippet)  # the whole block is valid Python
        ast.parse(lines[-1], mode="eval")  # and the last statement is an expression


def test_every_non_default_tab_asks_for_its_primitive(structure):
    """A tab is a promise about *which* primitive drew the figure.

    The generator checks the produced spec's marks at render time; this is the
    cheap textual half — a non-default tab whose call never mentions the
    primitive would be showing the default picture under the wrong heading,
    unless the geometry itself defaults to that primitive.
    """
    geometry_defaulted = {("phase_portrait", "line3d")}  # a 3-D orbit picks line3d itself
    for cell in structure.cells:
        record = get(cell.transform)
        if cell.primitive == record.default_primitive:
            continue
        if (cell.transform, cell.primitive) in geometry_defaulted:
            continue
        assert f'primitive="{cell.primitive}"' in cell.snippet, (
            f"{cell.transform}.{cell.primitive}: the tab names a primitive the call "
            f"never asks for:\n{cell.snippet}"
        )


def test_the_primitive_check_catches_a_mislabelled_tab(tmp_path):
    """The guard that makes the point above enforceable at render time."""
    variant = gallery.Variant(
        setup="traj = ts.systems.Lorenz().integrate(final_time=2.0, dt=0.1, ic=[1.0, 1.0, 20.0])",
        call='ts.plot(traj, "spacetime")',  # an image, shown under a "contour" tab
    )
    with pytest.raises(ValueError, match="shown under the 'contour' tab but drew"):
        gallery._render_one(variant, tmp_path / "x.png", primitive="contour", ns_cache={})


def test_with_primitive_produces_a_parseable_call():
    assert (
        gallery._with_primitive('ts.plot(traj, "psd")', "points")
        == 'ts.plot(traj, "psd", primitive="points")'
    )
    ast.parse(gallery._with_primitive('ts.plot(t, "a", b=1)', "line"), mode="eval")
    with pytest.raises(ValueError, match="must end in"):
        gallery._with_primitive("ts.plot(traj)  # comment", "line")


# ---------------------------------------------------------------------------
# 3. the page
# ---------------------------------------------------------------------------


def test_the_page_names_every_transform_and_every_primitive(structure):
    body = gallery.page(structure)
    for record in transforms():
        assert f"### `{record.name}` {{ #{record.name} }}" in body, record.name
        for primitive in record.primitives:
            assert f'=== "`{primitive}`' in body, f"{record.name}.{primitive}"


def test_the_page_is_split_by_the_two_source_categories(structure):
    body = gallery.page(structure)
    assert "## Data-capable transforms" in body
    assert "## Model-only transforms" in body
    assert {t.source for t in transforms()} <= {"data", "model"}, "there is no third category"


def test_the_page_shell_carries_the_token_the_hook_replaces():
    page = (_ROOT / "docs" / "visualization" / "gallery.md").read_text()
    assert gallery.TOKEN in page, "the generated body has nowhere to go"
    nav = (_ROOT / "mkdocs.yml").read_text()
    assert "visualization/gallery.md" in nav, "the gallery is not in the nav"


def test_the_hook_wires_the_gallery_in():
    """The generator is only useful if the docs build actually calls it."""
    hook = (_ROOT / "hooks" / "docs_autogen.py").read_text()
    assert "import gallery as _gallery" in hook
    assert "_gallery.render_all" in hook
    assert "_gallery.TOKEN" in hook


# ---------------------------------------------------------------------------
# 4. the claims the snippets rest on
# ---------------------------------------------------------------------------


def test_a_style_keyword_survives_the_bare_string_selector():
    """``ts.plot(traj, "phase_portrait", color="red")`` must colour something.

    It used to be dropped: the shared keywords were filtered against the
    transform's *compute* signature before the style split ran, so every style
    key fell out unless it was wrapped in ``ts.T(...)``.  The gallery styles
    several of its figures this way, and so will every user who reads it.
    """
    import tsdynamics as ts

    traj = ts.systems.Lorenz().integrate(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 20.0])
    spec = ts.plot(traj, "phase_portrait", components=("x", "z"), color="red", lw=3)
    assert spec.layers[0].style == {"color": "red", "linewidth": 3.0}
    aliased = ts.plot(traj, "phase_portrait", components=("x", "z"), primitive="points", ms=2.0)
    assert aliased.layers[0].style == {"markersize": 2.0}


def test_a_shared_option_still_does_not_reach_a_transform_that_cannot_take_it():
    """The filter the style fix had to keep: compute options are still routed by signature.

    ``tau`` belongs to ``delay_embedding``; handing it to ``phase_portrait`` in a
    shared keyword must be a no-op, not a ``TypeError``.  Calling the transform
    directly with it still is one — which is what makes this a *routing* rule and
    not a silent swallow.
    """
    import tsdynamics as ts

    traj = ts.systems.Lorenz().integrate(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 20.0])
    spec = ts.plot(traj, "phase_portrait", components=("x", "z"), tau=7)
    assert len(spec.layers) == 1
    with pytest.raises(TypeError):
        ts.viz.geometry(traj, "phase_portrait", components=("x", "z"), tau=7)


# ---------------------------------------------------------------------------
# 5. the figures — a real render, because blankness passes every check above
# ---------------------------------------------------------------------------


def _ink(path: pathlib.Path) -> float:
    """Pixel standard deviation of a rendered figure, alpha flattened onto white."""
    import matplotlib.image as mpimg

    img = mpimg.imread(path)
    if img.shape[-1] == 4:
        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + (1.0 - alpha)
    return float((img * 255.0).std())


#: Below this, a figure is a blank rectangle.  Measured: the emptiest real cell
#: in the gallery scores ~16, and a figure with axes but no data scores ~7.
_INK_FLOOR = 10.0


@pytest.mark.parametrize("name", ["time_series", "phase_portrait", "cobweb", "nullclines"])
def test_a_sample_of_cells_renders_something_visible(name, tmp_path, monkeypatch):
    monkeypatch.setattr(gallery, "CACHE_DIR", tmp_path)
    build = gallery.render_all(only={name})
    assert not build.failures, build.failures
    assert build.cells
    for cell in build.cells:
        assert cell.cached_path is not None and cell.cached_path.exists()
        ink = _ink(cell.cached_path)
        assert ink > _INK_FLOOR, f"{cell.transform}.{cell.primitive} rendered nearly blank ({ink})"


@pytest.mark.slow
def test_every_declared_cell_renders_something_visible(tmp_path, monkeypatch):
    """The whole matrix, rendered fresh and measured — the gate the page rests on."""
    monkeypatch.setattr(gallery, "CACHE_DIR", tmp_path)
    build = gallery.render_all()
    assert not build.failures, build.failures

    declared = {(t.name, p) for t in transforms() for p in t.primitives}
    assert {(c.transform, c.primitive) for c in build.cells} == declared

    faint = {
        f"{c.transform}.{c.primitive}": round(_ink(c.cached_path), 1)
        for c in [*build.cells, *build.compositions]
        if c.cached_path is not None and _ink(c.cached_path) <= _INK_FLOOR
    }
    assert not faint, f"nearly-blank figures: {faint}"


@pytest.mark.slow
def test_every_composition_overlays_transforms_from_more_than_one_family(tmp_path, monkeypatch):
    """The composition claim, checked by building the overlays rather than asserting it."""
    monkeypatch.setattr(gallery, "CACHE_DIR", tmp_path)
    build = gallery.render_all()
    assert build.compositions, "the page claims everything composes; show it"
    for cell in build.compositions:
        assert cell.cached_path is not None and cell.cached_path.exists()
        assert _ink(cell.cached_path) > _INK_FLOOR


def test_a_primitive_name_in_a_tab_is_a_registered_primitive(structure):
    unknown = {c.primitive for c in structure.cells} - set(PRIMITIVES)
    assert not unknown, unknown


def test_the_figure_geometry_is_finite_and_square_where_it_claims_to_be():
    """A guard on the two size constants, which are otherwise silent."""
    for size in (gallery.FIGSIZE, gallery.FIGSIZE_SQUARE):
        assert len(size) == 2
        assert all(np.isfinite(v) and v > 0 for v in size)
    assert (
        gallery.FIGSIZE_SQUARE[0] / gallery.FIGSIZE_SQUARE[1]
        < gallery.FIGSIZE[0] / (gallery.FIGSIZE[1])
    )
