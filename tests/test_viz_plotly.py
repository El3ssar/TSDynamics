"""Structural contract for the plotly interactive 2-D backend (stream PLOTLY-RENDER).

The plotly backend (:mod:`tsdynamics.viz.render.plotly`) turns a
:class:`~tsdynamics.viz.spec.PlotSpec` into an interactive
:class:`plotly.graph_objects.Figure`.  plotly is an *optional* dependency, so the
tests split into two groups:

- **Render tests** ``importorskip("plotly")`` and exercise the trace contract:
  representative 2-D specs render to a ``go.Figure`` with the expected trace
  types / counts, and the axes / annotations land in the layout.
- **Capability / wiring tests** run with or without plotly: they assert the
  declared :class:`~tsdynamics.viz.render.caps.RendererCapabilities` decline the
  3-D marks (so dispatch falls back to matplotlib, warning
  :class:`~tsdynamics.viz.render.caps.VisualizationDegraded`), and that importing
  ``tsdynamics`` (and registering the backends) pulls in **no** plotly.

Engine-free by design — no ``tsdynamics._rust`` import.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from tsdynamics import registry
from tsdynamics.viz.render import register_builtin_renderers, select_renderer
from tsdynamics.viz.render.caps import RendererCapabilities, VisualizationDegraded
from tsdynamics.viz.render.plotly import _SUPPORTED_KINDS
from tsdynamics.viz.spec import Annotation, Axis, Colorbar, Layer, Legend, PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# Representative 2-D specs (one per supported mark)
# ---------------------------------------------------------------------------


def _time_series() -> PlotSpec:
    t = np.linspace(0.0, 10.0, 50)
    return PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(kind=PlotKind.LINE, data={"x": t, "y": np.sin(t)}, label="x(t)")],
        x=Axis(label="t"),
        y=Axis(label="x", scale="linear"),
        legend=Legend(show=True),
        title="series",
    )


def _scatter_colored() -> PlotSpec:
    rng = np.random.default_rng(0)
    n = 30
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[
            Layer(
                kind=PlotKind.SCATTER,
                data={"x": rng.random(n), "y": rng.random(n), "c": rng.random(n)},
                style={"cmap": "viridis"},
            )
        ],
        clim=(0.0, 1.0),
        colorbar=Colorbar(label="speed", cmap="viridis"),
        aspect="equal",
    )


def _image() -> PlotSpec:
    z = np.arange(12.0).reshape(3, 4)
    return PlotSpec(
        kind=PlotKind.IMAGE,
        layers=[
            Layer(kind=PlotKind.IMAGE, data={"z": z, "x": np.arange(4.0), "y": np.arange(3.0)})
        ],
        clim=(0.0, 11.0),
        colorbar=Colorbar(label="value", cmap="magma"),
    )


def _histogram_prebinned() -> PlotSpec:
    centres = np.linspace(0.0, 1.0, 8)
    counts = np.array([1.0, 3.0, 5.0, 7.0, 6.0, 4.0, 2.0, 1.0])
    return PlotSpec(
        kind=PlotKind.HISTOGRAM_NULL,
        layers=[Layer(kind=PlotKind.HISTOGRAM, data={"x": centres, "y": counts})],
    )


def _histogram_raw() -> PlotSpec:
    samples = np.random.default_rng(1).standard_normal(40)
    return PlotSpec(
        kind=PlotKind.HISTOGRAM_NULL,
        layers=[Layer(kind=PlotKind.HISTOGRAM, data={"x": samples})],
    )


def _bar_categorical() -> PlotSpec:
    return PlotSpec(
        kind=PlotKind.FEATURE_BARS,
        layers=[
            Layer(kind=PlotKind.BAR, data={"cat": np.arange(3.0), "y": np.array([1.0, 2.0, 3.0])})
        ],
        x=Axis(scale="categorical", categories=["a", "b", "c"]),
    )


def _area_band() -> PlotSpec:
    x = np.linspace(0.0, 1.0, 10)
    return PlotSpec(
        kind=PlotKind.ENSEMBLE_FAN,
        layers=[Layer(kind=PlotKind.AREA, data={"x": x, "lo": x - 0.1, "hi": x + 0.1, "y": x})],
    )


def _errorbar() -> PlotSpec:
    x = np.linspace(0.0, 1.0, 6)
    return PlotSpec(
        kind=PlotKind.DIMENSION_SPECTRUM,
        layers=[
            Layer(kind=PlotKind.ERRORBAR, data={"x": x, "y": 2.0 - x, "err": 0.1 * np.ones(6)})
        ],
    )


def _quiver() -> PlotSpec:
    g = np.linspace(0.0, 1.0, 4)
    xx, yy = np.meshgrid(g, g)
    return PlotSpec(
        kind=PlotKind.VECTOR_FIELD,
        layers=[
            Layer(
                kind=PlotKind.QUIVER,
                data={"x": xx.ravel(), "y": yy.ravel(), "u": -yy.ravel(), "v": xx.ravel()},
            )
        ],
    )


def _line3d_spec() -> PlotSpec:
    t = np.linspace(0.0, 1.0, 10)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        layers=[Layer(kind=PlotKind.LINE3D, data={"x": t, "y": t, "z": t})],
        z=Axis(label="z"),
        ndim=3,
    )


# ---------------------------------------------------------------------------
# Render tests (need plotly)
# ---------------------------------------------------------------------------


@pytest.fixture
def go():
    """The ``plotly.graph_objects`` module (skip the whole test if plotly absent)."""
    return pytest.importorskip("plotly.graph_objects")


def test_register_adds_plotly_when_installed(go):
    """``register`` adds the plotly backend with its declared capabilities."""
    from tsdynamics.viz.render import plotly as plotly_backend

    reg = type(registry.renderers)(kind="renderer")
    added = plotly_backend.register(reg)
    assert added is True
    assert "plotly" in reg
    caps = reg.get("plotly").capabilities
    assert isinstance(caps, RendererCapabilities)
    assert caps.interactive is True
    assert caps.web_export is True
    # The styling overhaul gives plotly a real orbitable 3-D scene, so it now
    # advertises supports_3d=True (3-D specs render here instead of falling back).
    assert caps.supports_3d is True
    # Re-registering is a no-op (idempotent).
    assert plotly_backend.register(reg) is False


def test_line_spec_renders_scatter_trace(go):
    """A LINE time series becomes one ``go.Scatter`` line trace."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_time_series())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter)
    assert fig.data[0].mode == "lines"
    assert fig.layout.xaxis.title.text == "t"
    assert fig.layout.title.text == "series"
    assert fig.layout.showlegend is True


def test_colored_scatter_carries_colorscale(go):
    """A colour-by-``c`` SCATTER becomes a marker-coloured ``go.Scatter`` with a scale."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_scatter_colored())
    assert len(fig.data) == 1
    tr = fig.data[0]
    assert isinstance(tr, go.Scatter)
    assert tr.mode == "markers"
    assert tr.marker.showscale is True
    assert tr.marker.cmin == 0.0 and tr.marker.cmax == 1.0
    # Equal aspect maps to a y-axis anchored to x.
    assert fig.layout.yaxis.scaleanchor == "x"


def test_image_renders_heatmap(go):
    """An IMAGE mark becomes a ``go.Heatmap`` honouring clim as ``zmin``/``zmax``."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_image())
    assert len(fig.data) == 1
    hm = fig.data[0]
    assert isinstance(hm, go.Heatmap)
    assert hm.zmin == 0.0 and hm.zmax == 11.0


def test_prebinned_histogram_is_bar(go):
    """A pre-binned HISTOGRAM (x centres + y counts) becomes a ``go.Bar``."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_histogram_prebinned())
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)


def test_raw_histogram_is_histogram(go):
    """A raw-sample HISTOGRAM (only x) becomes a ``go.Histogram`` (plotly bins it)."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_histogram_raw())
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Histogram)


def test_categorical_bar_sets_category_axis(go):
    """A categorical BAR sets a plotly ``category`` x-axis with the category labels."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_bar_categorical())
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)
    assert fig.layout.xaxis.type == "category"
    assert tuple(fig.layout.xaxis.ticktext) == ("a", "b", "c")


def test_area_band_uses_tonexty_fill(go):
    """An AREA band emits an invisible ``lo`` trace and a ``hi`` trace with tonexty fill."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_area_band())
    # lo boundary + hi (filled) + central y line.
    assert len(fig.data) == 3
    assert fig.data[1].fill == "tonexty"


def test_errorbar_sets_error_y(go):
    """An ERRORBAR mark becomes a marker ``go.Scatter`` with ``error_y`` set."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_errorbar())
    assert len(fig.data) == 1
    tr = fig.data[0]
    assert tr.mode == "markers"
    assert tr.error_y.visible is True
    assert np.allclose(np.asarray(tr.error_y.array), 0.1)


def test_quiver_draws_segment_and_tip_traces(go):
    """A QUIVER mark becomes a segment line trace plus a marker tip trace."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_quiver())
    assert len(fig.data) == 2
    assert fig.data[0].mode == "lines"
    assert fig.data[1].mode == "markers"


def test_annotations_become_shapes_and_text(go):
    """vline / hline / span become layout shapes; text becomes a layout annotation."""
    from tsdynamics.viz.render.plotly._core import render

    spec = _time_series()
    spec.annotations = [
        Annotation(kind="vline", x=3.0, text="onset"),
        Annotation(kind="hline", y=0.0),
        Annotation(kind="span", span=(1.0, 2.0), axis="x"),
        Annotation(kind="text", x=5.0, y=0.5, text="peak"),
    ]
    fig = render(spec)
    assert len(fig.layout.shapes) == 3  # vline + hline + span
    # the vline 'onset' label + the standalone 'peak' text → 2 layout annotations.
    assert len(fig.layout.annotations) == 2


# ---------------------------------------------------------------------------
# Capability / dispatch tests (run with or without plotly installed)
# ---------------------------------------------------------------------------


def _plotly_caps() -> RendererCapabilities:
    """The capabilities the plotly backend declares (built without importing plotly)."""
    return RendererCapabilities.of_kinds(
        "plotly", _SUPPORTED_KINDS, supports_3d=False, interactive=True, web_export=True
    )


def test_caps_decline_3d_marks():
    """The declared capabilities decline a 3-D mark / kind (so dispatch falls back)."""
    caps = _plotly_caps()
    assert caps.can_render(PlotKind.LINE) is True
    assert caps.can_render(PlotKind.IMAGE) is True
    assert caps.can_render(PlotKind.LINE3D) is False
    assert caps.can_render(PlotKind.SURFACE3D) is False
    assert caps.can_render_spec(_line3d_spec()) is False
    assert caps.can_render_spec(_time_series()) is True


def test_caps_decline_animation_kinds():
    """The animation semantic kinds are declined (deferred everywhere)."""
    caps = _plotly_caps()
    assert PlotKind.TRAJECTORY_ANIMATION not in _SUPPORTED_KINDS
    assert caps.can_render(PlotKind.TRAJECTORY_ANIMATION) is False


def test_3d_spec_falls_back_to_matplotlib_with_warning():
    """A 3-D spec routed to plotly falls back to matplotlib, warning VisualizationDegraded.

    Register a stub plotly renderer carrying the *real* declared capabilities
    (so this runs even with plotly uninstalled), put matplotlib in too, then ask
    dispatch for the plotly backend on a 3-D spec: it must decline and select
    matplotlib, emitting :class:`VisualizationDegraded`.
    """
    pytest.importorskip("matplotlib")
    register_builtin_renderers()  # ensures matplotlib is registered (it is installed)

    stub_caps = _plotly_caps()

    def _stub(spec, /, **kw):  # pragma: no cover - never called (it is declined)
        raise AssertionError("the declined plotly stub must not be invoked")

    _stub.capabilities = stub_caps
    registry.renderers.register("plotly", _stub, replace=True)
    try:
        with pytest.warns(VisualizationDegraded):
            name, _renderer = select_renderer(_line3d_spec(), backend="plotly")
        assert name == "matplotlib"
        # A 2-D spec is served by plotly itself (no fallback / warning).
        name2, renderer2 = select_renderer(_time_series(), backend="plotly")
        assert name2 == "plotly"
        assert renderer2 is _stub
    finally:
        # Restore: drop the stub so other tests re-register the real (absent) one.
        if "plotly" in registry.renderers:
            registry.renderers.unregister("plotly")


def test_supported_kinds_partition():
    """The advertised kinds are the 2-D semantic kinds plus the 2-D marks, no 3-D."""
    assert PlotKind.PHASE_PORTRAIT_2D in _SUPPORTED_KINDS
    assert PlotKind.TIME_SERIES in _SUPPORTED_KINDS
    assert PlotKind.BAR in _SUPPORTED_KINDS
    assert PlotKind.PHASE_PORTRAIT_3D not in _SUPPORTED_KINDS
    assert PlotKind.LINE3D not in _SUPPORTED_KINDS
    assert PlotKind.SURFACE3D not in _SUPPORTED_KINDS


# ---------------------------------------------------------------------------
# The no-plot-import guarantee
# ---------------------------------------------------------------------------


def test_import_tsdynamics_does_not_import_plotly():
    """``import tsdynamics`` (and registering backends) pulls in no plotly.

    Run in a fresh subprocess: import tsdynamics, register the in-tree backends,
    and assert ``plotly`` never entered ``sys.modules`` (lazy / in-method import).
    """
    code = (
        "import sys; import tsdynamics; "
        "from tsdynamics.viz.render import register_builtin_renderers; "
        "register_builtin_renderers(); "
        "assert 'plotly' not in sys.modules, sorted(m for m in sys.modules if 'plotly' in m); "
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
