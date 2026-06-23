"""Self-contained HTML export for the plotly backend (stream PLOTLY-HTML).

The plotly backend can serialise a :class:`~tsdynamics.viz.spec.PlotSpec` to a
self-contained, kernel-free HTML fragment (:mod:`tsdynamics.viz.render.plotly._html`)
and expose it through the renderer's ``html=`` / ``path=`` kwargs (the
``result.plot(backend="plotly", html=True)`` seam).  plotly is an *optional*
dependency, so every test that actually exports HTML ``importorskip("plotly")``;
the structural import-shape check runs without it.

Engine-free by design — no ``tsdynamics._rust`` import.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.viz.spec import Axis, Layer, Legend, PlotKind, PlotSpec


def _time_series() -> PlotSpec:
    """A representative 2-D line spec the plotly backend renders."""
    t = np.linspace(0.0, 10.0, 50)
    return PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(kind=PlotKind.LINE, data={"x": t, "y": np.sin(t)}, label="x(t)")],
        x=Axis(label="t"),
        y=Axis(label="x", scale="linear"),
        legend=Legend(show=True),
        title="series",
    )


# ---------------------------------------------------------------------------
# Import-shape contract (runs with or without plotly)
# ---------------------------------------------------------------------------


def test_html_module_imports_no_plotly() -> None:
    """Importing the HTML helper must not drag plotly into ``sys.modules``."""
    import sys

    import tsdynamics.viz.render.plotly._html as html_mod  # noqa: F401

    assert "plotly" not in sys.modules


def test_html_helper_surface() -> None:
    """The HTML helper exposes the documented ``to_html`` / ``write_html`` API."""
    from tsdynamics.viz.render.plotly import _html

    assert callable(_html.to_html)
    assert callable(_html.write_html)
    assert set(_html.__all__) == {"to_html", "write_html"}


# ---------------------------------------------------------------------------
# HTML export (requires plotly)
# ---------------------------------------------------------------------------


def test_to_html_fragment_has_div_and_cdn() -> None:
    """A 2-D spec exports an embeddable HTML fragment with a div + CDN script."""
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._html import to_html

    text = to_html(_time_series())

    assert isinstance(text, str)
    # An interactive plotly div is present ...
    assert "plotly" in text.lower()
    assert "<div" in text
    # ... the CDN bundle is referenced (a <script src=...cdn...> tag) ...
    assert "cdn.plot" in text
    # ... and the default fragment is NOT a full standalone document.
    assert "<html" not in text.lower()
    assert "<body" not in text.lower()


def test_to_html_full_html_is_standalone_document() -> None:
    """``full_html=True`` wraps the fragment in a standalone ``<html>`` document."""
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._html import to_html

    text = to_html(_time_series(), full_html=True)

    assert "<html" in text.lower()
    assert "<body" in text.lower()
    assert "<div" in text


def test_to_html_accepts_a_plotly_figure() -> None:
    """``to_html`` serialises an already-built plotly figure unchanged."""
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    from tsdynamics.viz.render.plotly._core import render
    from tsdynamics.viz.render.plotly._html import to_html

    fig = render(_time_series())
    assert isinstance(fig, go.Figure)
    text = to_html(fig)
    assert "<div" in text
    assert "cdn.plot" in text


def test_write_html_writes_a_standalone_file(tmp_path) -> None:
    """``write_html`` writes a standalone HTML file and returns its path."""
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._html import write_html

    out = tmp_path / "figure.html"
    returned = write_html(_time_series(), out)

    assert returned == out
    assert out.exists()
    written = out.read_text(encoding="utf-8")
    assert "<html" in written.lower()
    assert "<div" in written
    assert "cdn.plot" in written


# ---------------------------------------------------------------------------
# The renderer hook: html= / path= kwargs (requires plotly)
# ---------------------------------------------------------------------------


def test_render_html_kwarg_returns_fragment() -> None:
    """``render(spec, html=True)`` returns the embeddable HTML fragment string."""
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._core import render

    text = render(_time_series(), html=True)

    assert isinstance(text, str)
    assert "<div" in text
    assert "cdn.plot" in text
    assert "<html" not in text.lower()


def test_render_path_kwarg_writes_file(tmp_path) -> None:
    """``render(spec, path=...)`` writes a standalone HTML file and returns the path."""
    pytest.importorskip("plotly")
    from pathlib import Path

    from tsdynamics.viz.render.plotly._core import render

    out = tmp_path / "rendered.html"
    returned = render(_time_series(), path=out)

    assert isinstance(returned, Path)
    assert returned == out
    written = out.read_text(encoding="utf-8")
    assert "<html" in written.lower()
    assert "<div" in written


def test_render_default_still_returns_figure() -> None:
    """With no html/path kwarg the renderer still returns a live plotly figure."""
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    from tsdynamics.viz.render.plotly._core import render

    assert isinstance(render(_time_series()), go.Figure)
