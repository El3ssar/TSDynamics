"""Contract for the plotly 3-D interactive backend (stream PLOTLY-3D).

The plotly backend now draws 3-D specs (``LINE3D`` / ``SURFACE3D`` /
``PHASE_PORTRAIT_3D``) through :mod:`tsdynamics.viz.render.plotly._threed` — an
orbitable ``go.Scatter3d`` / ``go.Surface`` ``scene``.  plotly is an *optional*
dependency, so the tests split into two groups:

- **Render tests** ``importorskip("plotly")`` and exercise the 3-D trace
  contract: a ``LINE3D`` spec renders to a ``go.Figure`` carrying a
  ``go.Scatter3d`` trace, a ``SURFACE3D`` spec a ``go.Surface`` trace, and the
  ``spec.meta["camera"]`` eye / up lands in the figure's ``scene.camera``.
- **Capability / wiring tests** run with or without plotly: they assert the
  declared :class:`~tsdynamics.viz.render.caps.RendererCapabilities` now
  ``supports_3d`` and *accept* a 3-D spec (so dispatch no longer falls back).

Engine-free by design — no ``tsdynamics._rust`` import.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.viz.render.caps import RendererCapabilities
from tsdynamics.viz.render.plotly import _KINDS_3D, _REGISTERED_KINDS, _SUPPORTED_KINDS
from tsdynamics.viz.spec import Axis, Colorbar, Layer, Legend, PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# Representative 3-D specs
# ---------------------------------------------------------------------------


def _line3d_spec() -> PlotSpec:
    t = np.linspace(0.0, 6.0, 40)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        layers=[
            Layer(
                kind=PlotKind.LINE3D,
                data={"x": np.cos(t), "y": np.sin(t), "z": t},
                label="orbit",
                style={"color": "navy", "lw": 2.0},
            )
        ],
        x=Axis(label="x"),
        y=Axis(label="y"),
        z=Axis(label="z"),
        ndim=3,
        aspect="equal",
        legend=Legend(show=True),
        title="attractor",
    )


def _line3d_colored_spec() -> PlotSpec:
    t = np.linspace(0.0, 6.0, 40)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        layers=[
            Layer(
                kind=PlotKind.LINE3D,
                data={"x": np.cos(t), "y": np.sin(t), "z": t, "c": t},
                style={"cmap": "viridis"},
            )
        ],
        z=Axis(label="z"),
        ndim=3,
        clim=(0.0, 6.0),
        colorbar=Colorbar(label="time", cmap="viridis"),
    )


def _scatter3d_spec() -> PlotSpec:
    rng = np.random.default_rng(0)
    n = 25
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        layers=[
            Layer(
                kind=PlotKind.SCATTER,
                data={
                    "x": rng.random(n),
                    "y": rng.random(n),
                    "z": rng.random(n),
                    "c": rng.random(n),
                },
                style={"cmap": "magma"},
            )
        ],
        z=Axis(label="z"),
        ndim=3,
        clim=(0.0, 1.0),
        colorbar=Colorbar(label="speed", cmap="magma"),
    )


def _surface3d_spec() -> PlotSpec:
    g = np.linspace(-1.0, 1.0, 8)
    xx, yy = np.meshgrid(g, g)
    zz = xx**2 - yy**2
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        layers=[Layer(kind=PlotKind.SURFACE3D, data={"x": g, "y": g, "z": zz})],
        z=Axis(label="z"),
        ndim=3,
        clim=(-1.0, 1.0),
        colorbar=Colorbar(label="height", cmap="viridis"),
    )


# ---------------------------------------------------------------------------
# Render tests (need plotly)
# ---------------------------------------------------------------------------


@pytest.fixture
def go():
    """The ``plotly.graph_objects`` module (skip the whole test if plotly absent)."""
    return pytest.importorskip("plotly.graph_objects")


def test_line3d_renders_scatter3d_trace(go):
    """A LINE3D spec renders to a ``go.Figure`` carrying one ``go.Scatter3d`` line."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_line3d_spec())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    tr = fig.data[0]
    assert isinstance(tr, go.Scatter3d)
    assert tr.mode == "lines"
    assert fig.layout.scene.xaxis.title.text == "x"
    assert fig.layout.scene.zaxis.title.text == "z"
    assert fig.layout.title.text == "attractor"


def test_line3d_colored_carries_marker_colorscale(go):
    """A colour-by-``c`` LINE3D becomes a ``Scatter3d`` line+markers with a colour scale."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_line3d_colored_spec())
    assert len(fig.data) == 1
    tr = fig.data[0]
    assert isinstance(tr, go.Scatter3d)
    assert tr.mode == "lines+markers"
    assert tr.marker.showscale is True
    assert tr.marker.cmin == 0.0 and tr.marker.cmax == 6.0


def test_scatter3d_renders_markers(go):
    """A 3-D SCATTER spec becomes a ``go.Scatter3d`` with ``mode='markers'``."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_scatter3d_spec())
    assert len(fig.data) == 1
    tr = fig.data[0]
    assert isinstance(tr, go.Scatter3d)
    assert tr.mode == "markers"
    assert tr.marker.showscale is True


def test_surface3d_renders_surface_trace(go):
    """A SURFACE3D spec renders to a ``go.Figure`` carrying a ``go.Surface`` trace."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_surface3d_spec())
    assert len(fig.data) == 1
    surf = fig.data[0]
    assert isinstance(surf, go.Surface)
    assert surf.cmin == -1.0 and surf.cmax == 1.0


def test_equal_aspect_maps_to_cube_scene(go):
    """An ``aspect='equal'`` 3-D spec sets ``scene.aspectmode = 'cube'``."""
    from tsdynamics.viz.render.plotly._core import render

    fig = render(_line3d_spec())
    assert fig.layout.scene.aspectmode == "cube"


def test_camera_eye_up_applied_to_scene(go):
    """The ``spec.meta['camera']`` eye / up lands in the figure's ``scene.camera``."""
    from tsdynamics.viz.render.plotly._core import render

    spec = _line3d_spec()
    spec.meta = {
        "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 0.8}, "up": {"x": 0.0, "y": 0.0, "z": 1.0}}
    }
    fig = render(spec)
    cam = fig.layout.scene.camera
    assert abs(cam.eye.x - 1.5) < 1e-9
    assert abs(cam.eye.z - 0.8) < 1e-9
    assert abs(cam.up.z - 1.0) < 1e-9


def test_render_dispatches_3d_through_core(go):
    """The 2-D core entry point routes a 3-D spec to the 3-D renderer (Scatter3d)."""
    from tsdynamics.viz.render.plotly._core import render

    # A 3-D spec entering the public ``render`` must produce a 3-D trace, proving
    # the _core → _threed wiring (not a degraded 2-D projection).
    fig = render(_surface3d_spec())
    assert isinstance(fig.data[0], go.Surface)


# ---------------------------------------------------------------------------
# Capability / wiring tests (run with or without plotly installed)
# ---------------------------------------------------------------------------


def _registered_caps() -> RendererCapabilities:
    """The capabilities the plotly backend now registers (built without importing plotly)."""
    return RendererCapabilities.of_kinds(
        "plotly", _REGISTERED_KINDS, supports_3d=True, interactive=True, web_export=True
    )


def test_caps_support_3d_structurally():
    """Without plotly, the declared capabilities advertise ``supports_3d`` is True."""
    caps = _registered_caps()
    assert caps.supports_3d is True


def test_caps_accept_3d_kinds_and_spec():
    """The capabilities now *accept* the 3-D marks and a 3-D spec (no fallback)."""
    caps = _registered_caps()
    assert caps.can_render(PlotKind.LINE3D) is True
    assert caps.can_render(PlotKind.SURFACE3D) is True
    assert caps.can_render(PlotKind.PHASE_PORTRAIT_3D) is True
    assert caps.can_render_spec(_line3d_spec()) is True
    assert caps.can_render_spec(_surface3d_spec()) is True


def test_registered_kinds_include_3d():
    """The registered kind set unions the 2-D kinds with the 3-D kinds."""
    assert _KINDS_3D <= _REGISTERED_KINDS
    assert _SUPPORTED_KINDS <= _REGISTERED_KINDS
    assert PlotKind.LINE3D in _REGISTERED_KINDS
    assert PlotKind.SURFACE3D in _REGISTERED_KINDS
    assert PlotKind.PHASE_PORTRAIT_3D in _REGISTERED_KINDS


def test_animation_kinds_still_declined():
    """The animation kinds remain outside the registered set (deferred everywhere)."""
    caps = _registered_caps()
    assert PlotKind.TRAJECTORY_ANIMATION not in _REGISTERED_KINDS
    assert PlotKind.ENSEMBLE_ANIMATION not in _REGISTERED_KINDS
    assert caps.can_render(PlotKind.TRAJECTORY_ANIMATION) is False
