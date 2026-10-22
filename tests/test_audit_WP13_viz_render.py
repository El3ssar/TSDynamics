"""Regression tests for WP13 (viz render: plotly loop/pingpong + mpl cmap/extent).

Covers four audited defects:

- **X4b-1**: the plotly HTML real-time export silently dropped ``pingpong`` (a
  forward-only comet instead of forward-then-reverse).
- **X4b-2**: the plotly HTML export ignored ``loop=False`` (it looped forever).
- **X4b-3**: ``_make_discrete_cmap_norm`` assumed a 20-swatch (tab20) base, so a
  non-tab20 / shorter qualitative cmap got mis-spaced swatches.
- **X4b-6**: the imshow ``extent`` used pixel-*centre* coordinates as the *edge*
  extent, shifting a grid-backed image by half a cell relative to plotly.

These assert library-internal behavior (the generated JS schedule, the discrete
colour map, the extent tuple), so they are fast and need no rendered figure.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_viz_plotly_anim import plotly_clock_probe as plotly_clock_probe

from tsdynamics.viz.render.mpl._core import _image_extent, _make_discrete_cmap_norm, _preset_for
from tsdynamics.viz.spec import Animation, Colorbar, Layer, PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# X4b-1 / X4b-2: plotly HTML real-time export honors pingpong + loop
# ---------------------------------------------------------------------------


def _line3d_animated(anim: Animation) -> PlotSpec:
    """A 3-D LINE spec carrying ``anim`` — the real-time HTML export's shape."""
    t = np.linspace(0.0, 1.0, 64)
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        layers=[Layer(kind=PlotKind.LINE3D, data={"x": np.sin(t), "y": np.cos(t), "z": t})],
    )
    spec.animation = anim
    return spec


def _exported_clock(html: str) -> str:
    start = html.index("(function() {\n  var gd = document.getElementById(")
    return html[start:].split("\n})();", 1)[0] + "\n})();"


def test_plotly_html_threads_loop_and_pingpong_tokens(plotly_clock_probe) -> None:
    """The exported driver reverses once and stops when looping is disabled."""
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._anim import animated_html

    html = animated_html(
        _line3d_animated(Animation(pingpong=True, loop=False, n_frames=4, fps=4)),
        html=True,
        full_html=False,
    )
    assert isinstance(html, str)
    result = plotly_clock_probe(
        """
        const v=mount({lengths:[64]}), values=[];
        for(const t of [0,500,1000,1250,1500,2500]) {
          await v.frame(t);values.push(v.heads()[0]);
        }
        console.log(JSON.stringify([values,v.play.attributes['aria-label']]));
        """,
        clock=_exported_clock(html),
    )
    assert result == [[0, 32, 63, 32, 0, 0], "Play animation"]


def test_plotly_html_defaults_loop_true_pingpong_false(plotly_clock_probe) -> None:
    """The exported default playback repeats forwards rather than reversing or stopping."""
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._anim import animated_html

    html = animated_html(_line3d_animated(Animation(n_frames=4, fps=4)), html=True, full_html=False)
    assert isinstance(html, str)
    result = plotly_clock_probe(
        """
        const v=mount({lengths:[64]}), values=[];
        for(const t of [0,500,1000,1500,2000,2500]) {
          await v.frame(t);values.push(v.heads()[0]);
        }
        console.log(JSON.stringify([values,v.play.attributes['aria-label']]));
        """,
        clock=_exported_clock(html),
    )
    assert result == [[0, 32, 63, 0, 32, 63], "Pause animation"]


# ---------------------------------------------------------------------------
# X4b-3: discrete cmap adapts to the resolved colormap, not a hardcoded tab20
# ---------------------------------------------------------------------------


def _basins_spec(cmap: str | None) -> PlotSpec:
    """A BASINS_IMAGE spec with a palette_index and an optional override cmap."""
    cb = Colorbar(discrete=True, cmap=cmap) if cmap is not None else Colorbar(discrete=True)
    img = np.array([[0, 1], [1, 2]])
    return PlotSpec(
        kind=PlotKind.BASINS_IMAGE,
        layers=[Layer(kind=PlotKind.IMAGE, data={"z": img})],
        colorbar=cb,
        meta={"palette_index": {0: 0, 1: 1, 2: 2}},
    )


def test_discrete_cmap_uses_resolved_swatch_grid() -> None:
    """A short qualitative cmap (Set1, 9 swatches) gets its OWN swatch centres.

    Pre-fix the swatch-centre fraction was hard-coded ``(2s+1)/40`` (tab20's
    20-swatch grid) regardless of the resolved map, so a Set1 palette-index sampled
    the wrong fractions and did not land on Set1's actual swatches.  Post-fix the
    size is ``base_cmap.N`` (9 for Set1), so swatch ``s`` lands exactly on Set1's
    ``s``-th colour.
    """
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    spec = _basins_spec("Set1")
    layer = spec.layers[0]
    preset = _preset_for(PlotKind.BASINS_IMAGE)
    cmap, _norm = _make_discrete_cmap_norm(np.asarray(layer.data["z"]), spec, layer, preset)

    set1 = mpl.colormaps["Set1"]
    # Each label's colour must equal Set1's own swatch (the intended distinct colour).
    for swatch in (0, 1, 2):
        assert tuple(cmap.colors[swatch]) == pytest.approx(tuple(set1(swatch)))

    # The pre-fix tab20-fraction would have sampled Set1 at (2s+1)/40 — a DIFFERENT
    # colour for swatches >= 1 (the bug this test guards against).
    assert tuple(cmap.colors[1]) != pytest.approx(tuple(set1((2 * 1 + 1) / 40)))


def test_discrete_cmap_tab20_unchanged() -> None:
    """The default tab20 basin colours are answer-preserving (N == 20 ⇒ same fractions)."""
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    spec = _basins_spec(None)  # falls back to the tab20 preset
    layer = spec.layers[0]
    preset = _preset_for(PlotKind.BASINS_IMAGE)
    cmap, _norm = _make_discrete_cmap_norm(np.asarray(layer.data["z"]), spec, layer, preset)

    tab20 = mpl.colormaps["tab20"]
    for swatch in (0, 1, 2):
        # The old formula (2s+1)/40 and the new (2s+1)/(2*N=40) coincide for tab20.
        assert tuple(cmap.colors[swatch]) == pytest.approx(tuple(tab20((2 * swatch + 1) / 40)))


# ---------------------------------------------------------------------------
# X4b-6: imshow extent expands pixel centres to edges (half-cell each side)
# ---------------------------------------------------------------------------


def test_image_extent_expands_centres_to_edges() -> None:
    """Centre coordinates become edge extent (x0-dx/2 .. x1+dx/2), matching plotly.

    Pre-fix the extent was ``(x[0], x[-1], y[0], y[-1])`` — the pixel *centres*
    used as imshow *edges*, shifting the image half a cell.  Post-fix it is
    expanded by half a cell on each side.
    """
    # Centres at 0,1,2,3 (dx=1) and 10,12,14 (dy=2).
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([10.0, 12.0, 14.0])
    layer = Layer(kind=PlotKind.IMAGE, data={"x": x, "y": y, "z": np.zeros((3, 4))})

    extent = _image_extent(layer)
    assert extent is not None
    x0, x1, y0, y1 = extent
    # Half a cell on each side: dx=1 → -0.5..3.5 ; dy=2 → 9.0..15.0.
    assert (x0, x1, y0, y1) == pytest.approx((-0.5, 3.5, 9.0, 15.0))

    # The full extent width is the data span PLUS one cell (edge-to-edge),
    # not the centre-to-centre span (the pre-fix value).
    assert (x1 - x0) == pytest.approx(4.0)  # 4 columns * dx
    assert (x1 - x0) != pytest.approx(3.0)  # the centre-to-centre pre-fix span
