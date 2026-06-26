"""Tests for native plotly ``COMPOSITE`` (multi-panel) rendering (stream PLOTLY-COMPOSITE).

The composition layer (:func:`tsdynamics.viz.plot`) builds a ``COMPOSITE`` spec
for ``layout="stack"`` / ``"row"`` / ``"grid"``.  The plotly backend now tiles
these into one :func:`plotly.subplots.make_subplots` grid (rather than declining
and falling back to matplotlib).  These tests assert the structural contract:

- a stacked / row / grid composite renders to a ``plotly.graph_objects.Figure``
  with one subplot axis (or ``scene``) and one trace set per panel;
- a composite **mixing** a 2-D time-series panel and a 3-D portrait panel
  renders with the panels on the correct subplot types (``xy`` + ``scene``);
- two stacked colour-mapped (spacetime) panels get non-overlapping colorbars;
- ``.save("fig.html")`` writes a non-empty standalone interactive page;
- **no** :class:`~tsdynamics.viz.render.caps.VisualizationDegraded` fallback
  fires for a composite under ``backend="plotly"``.

Engine-free by design (no ``tsdynamics._rust`` import); plotly is required, so
each test ``importorskip("plotly")``.  Building specs imports no plotting
library — the import-light guarantee is covered in ``test_viz_compose.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tsdynamics as ts
import tsdynamics.viz as viz
from tsdynamics.viz.render.caps import VisualizationDegraded
from tsdynamics.viz.spec import Axis, Colorbar, Layer, Layout, PlotKind, PlotSpec

pytest.importorskip("plotly")

# ---------------------------------------------------------------------------
# Builders (fast tier — short integrations / synthetic specs)
# ---------------------------------------------------------------------------


def _lorenz(ic=(1.0, 1.0, 1.0)):
    return ts.Lorenz().integrate(final_time=20.0, dt=0.02, ic=list(ic)).after(5.0)


def _l96(ic=None):
    kw = {} if ic is None else {"ic": ic}
    return ts.Lorenz96(N=8).trajectory(final_time=8.0, dt=0.1, **kw)


def _image_panel(label: str) -> PlotSpec:
    """A standalone SPACETIME-like image panel carrying its own colorbar."""
    z = np.arange(20.0).reshape(4, 5)
    return PlotSpec(
        kind=PlotKind.SPACETIME,
        layers=[Layer(kind=PlotKind.IMAGE, data={"z": z})],
        clim=(0.0, 19.0),
        colorbar=Colorbar(label=label, cmap="viridis", show=True),
        title=label,
    )


def _ts_panel(label: str) -> PlotSpec:
    t = np.linspace(0.0, 10.0, 40)
    return PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(kind=PlotKind.LINE, data={"x": t, "y": np.sin(t)})],
        x=Axis(label="t"),
        y=Axis(label=label),
        title=label,
    )


def _composite(panels, mode="stack", **layout_kw) -> PlotSpec:
    return PlotSpec(
        kind=PlotKind.COMPOSITE,
        panels=list(panels),
        layout=Layout(mode=mode, **layout_kw),
    )


def _render(spec: PlotSpec):
    with warnings.catch_warnings():
        warnings.simplefilter("error", VisualizationDegraded)
        return spec.render(backend="plotly")


def _is_plotly_figure(fig) -> bool:
    return type(fig).__module__.startswith("plotly") and type(fig).__name__ == "Figure"


# ---------------------------------------------------------------------------
# Layout modes → subplot grids
# ---------------------------------------------------------------------------


def test_stacked_2d_composite_has_one_axis_per_panel():
    spec = _composite([_ts_panel("a"), _ts_panel("b")], mode="stack")
    fig = _render(spec)
    assert _is_plotly_figure(fig)
    xaxes = [k for k in fig.layout if k.startswith("xaxis")]
    yaxes = [k for k in fig.layout if k.startswith("yaxis")]
    assert len(xaxes) == 2  # one x-axis per stacked panel
    assert len(yaxes) == 2
    assert len(fig.data) == 2  # one line trace per panel


def test_row_2d_composite_has_one_axis_per_panel():
    spec = _composite([_ts_panel("a"), _ts_panel("b"), _ts_panel("c")], mode="row")
    fig = _render(spec)
    assert _is_plotly_figure(fig)
    assert len([k for k in fig.layout if k.startswith("xaxis")]) == 3
    assert len(fig.data) == 3


def test_grid_2d_composite_tiles_into_a_grid():
    spec = _composite([_ts_panel(c) for c in "abcd"], mode="grid")
    fig = _render(spec)
    assert _is_plotly_figure(fig)
    # 4 near-square → 2x2 → 4 xy cells, 4 traces
    assert len([k for k in fig.layout if k.startswith("xaxis")]) == 4
    assert len(fig.data) == 4


def test_grid_explicit_rows_cols_honoured():
    spec = _composite([_ts_panel(c) for c in "abc"], mode="grid", rows=1, cols=3)
    fig = _render(spec)
    assert len([k for k in fig.layout if k.startswith("xaxis")]) == 3
    assert len(fig.data) == 3


# ---------------------------------------------------------------------------
# 3-D and mixed panels (scene cells)
# ---------------------------------------------------------------------------


def test_row_of_3d_portraits_makes_a_scene_per_panel():
    spec = viz.plot(_lorenz(), _lorenz([1.1, 1.0, 1.0]), layout="row")
    assert spec.kind == PlotKind.COMPOSITE
    fig = _render(spec)
    assert _is_plotly_figure(fig)
    scenes = [k for k in fig.layout if k.startswith("scene")]
    assert len(scenes) == 2  # each 3-D portrait gets its own orbitable scene
    assert len(fig.data) == 2


def test_mixed_2d_and_3d_panels_render_with_correct_subplot_types():
    # a 2-D time-series panel + a 3-D portrait panel in one composite
    ts_panel = viz.plot(_lorenz(), components="x")  # 2-D
    portrait = viz.plot(_lorenz())  # 3-D
    spec = viz.plot(ts_panel, portrait, layout="stack")
    fig = _render(spec)
    assert _is_plotly_figure(fig)
    # exactly one xy cell (the time series) and one scene cell (the portrait)
    assert len([k for k in fig.layout if k.startswith("xaxis")]) == 1
    assert len([k for k in fig.layout if k.startswith("scene")]) == 1
    assert len(fig.data) == 2


def test_mixed_panels_route_traces_to_their_cell():
    ts_panel = _ts_panel("x")
    portrait = viz.plot(_lorenz())
    fig = _render(viz.plot(ts_panel, portrait, layout="row"))
    kinds = {type(t).__name__ for t in fig.data}
    # a 2-D Scatter and a 3-D Scatter3d coexist in the one figure
    assert "Scatter" in kinds
    assert "Scatter3d" in kinds


# ---------------------------------------------------------------------------
# Per-panel colorbars do not collide
# ---------------------------------------------------------------------------


def test_two_stacked_image_panels_get_non_overlapping_colorbars():
    spec = _composite([_image_panel("top"), _image_panel("bottom")], mode="stack")
    fig = _render(spec)
    assert _is_plotly_figure(fig)
    heatmaps = [t for t in fig.data if type(t).__name__ == "Heatmap"]
    assert len(heatmaps) == 2
    ys = sorted(float(t.colorbar.y) for t in heatmaps)
    # the two scales sit at distinct vertical centres (one per stacked panel)
    assert ys[0] < ys[1]
    assert ys[1] - ys[0] > 0.2
    # each scale is shrunk to its panel's extent, so they cannot overlap
    assert all(float(t.colorbar.len) < 0.6 for t in heatmaps)


def test_spacetime_trajectory_composite_colorbars(tmp_path):
    spec = viz.plot(_l96(), _l96(ic=np.full(8, 0.5)), layout="stack")
    fig = _render(spec)
    cbars = [t for t in fig.data if "colorbar" in t]
    assert len(cbars) == 2
    assert cbars[0].colorbar.y != cbars[1].colorbar.y


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------


def test_composite_saves_one_standalone_html_page(tmp_path):
    out = tmp_path / "composite.html"
    returned = _composite([_ts_panel("a"), _ts_panel("b")], mode="stack").save(str(out))
    assert returned == str(out)
    assert out.stat().st_size > 0
    text = out.read_text()
    assert "plotly" in text.lower()  # it is a plotly page, not a static image
    assert "<html" in text.lower()  # a standalone document


def test_mixed_composite_saves_html(tmp_path):
    out = tmp_path / "mixed.html"
    spec = viz.plot(viz.plot(_lorenz(), components="x"), viz.plot(_lorenz()), layout="stack")
    spec.save(str(out))
    assert out.stat().st_size > 0


def test_composite_html_fragment_via_render():
    spec = _composite([_ts_panel("a"), _ts_panel("b")], mode="stack")
    frag = spec.render(backend="plotly", html=True)
    assert isinstance(frag, str)
    assert len(frag) > 0


# ---------------------------------------------------------------------------
# No degraded fallback
# ---------------------------------------------------------------------------


def test_no_visualization_degraded_warning_for_composite():
    spec = viz.plot(
        viz.plot(_lorenz(), components="x"),
        viz.plot(_lorenz(), components="y"),
        layout="stack",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = spec.render(backend="plotly")
    assert _is_plotly_figure(fig)
    assert not [w for w in caught if issubclass(w.category, VisualizationDegraded)]


def test_empty_composite_renders_a_figure():
    spec = PlotSpec(kind=PlotKind.COMPOSITE, panels=[], layout=Layout(mode="stack"))
    fig = _render(spec)
    assert _is_plotly_figure(fig)


def test_animated_composite_declines_to_matplotlib():
    # an *animated* composite stays the matplotlib renderer's job (plotly's
    # animation core is single-panel), so plotly declines it and dispatch falls
    # back — the static composite path here is for non-animated composites only.
    # (Asserted on the capability decision, so no dangling FuncAnimation is built.)
    pytest.importorskip("matplotlib")
    from tsdynamics import registry
    from tsdynamics.viz.render import register_builtin_renderers, select_renderer

    register_builtin_renderers()
    plotly_caps = registry.renderers.get("plotly").capabilities
    static = viz.plot(
        viz.plot(_lorenz(), components="x"),
        viz.plot(_lorenz(), components="y"),
        layout="stack",
    )
    animated = PlotSpec.from_dict(static.to_dict()).animate(fps=10)
    # plotly draws the static composite but declines the animated one
    assert plotly_caps.can_render_spec(static) is True
    assert plotly_caps.can_render_spec(animated) is False
    # so dispatch falls back to matplotlib (with a VisualizationDegraded warning)
    with pytest.warns(VisualizationDegraded):
        name, _ = select_renderer(animated, backend="plotly")
    assert name == "matplotlib"
