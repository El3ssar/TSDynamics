"""matplotlib 3-D rendering (stream VIZ-MPL-3D).

Drives the matplotlib reference renderer on 3-D specs: a 3-D spec (``ndim == 3`` /
a ``z`` axis / a ``LINE3D`` / ``SURFACE3D`` mark) must render to a Figure with an
``mplot3d`` ``Axes3D``, with equal box aspect and the camera applied.  The
renderer is triple-agnostic — it draws whatever ``x`` / ``y`` / ``z`` channels the
spec carries, so a Lorenz-96 phase portrait over an arbitrary (non-first-three)
component triple renders just the same.

Skips cleanly where matplotlib (and, for the engine cases, the compiled
extension) is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics import registry
from tsdynamics.viz.render import register_builtin_renderers
from tsdynamics.viz.render.mpl._threed import is_three_d
from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

# Skip the whole module where matplotlib is absent (the mpl submodule imports
# matplotlib only lazily, so the imports above are matplotlib-free).
pytest.importorskip("matplotlib")


@pytest.fixture
def mpl_backend():
    """Register the matplotlib backend for the test, then restore the registry."""
    saved = list(registry.renderers.all())
    registry.renderers.clear()
    register_builtin_renderers()
    if "matplotlib" not in registry.renderers:  # pragma: no cover - matplotlib present here
        pytest.skip("matplotlib backend did not register")
    try:
        yield
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj)


def _line3d_spec(n: int = 80) -> PlotSpec:
    t = np.linspace(0.0, 6.0, n)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        ndim=3,
        x=Axis(label="x"),
        y=Axis(label="y"),
        z=Axis(label="z"),
        layers=[Layer(PlotKind.LINE3D, {"x": np.cos(t), "y": np.sin(t), "z": t})],
    )


def _axes3d_name(fig) -> str:
    return getattr(fig.axes[0], "name", "")


def test_is_three_d_detects_3d_specs():
    assert is_three_d(_line3d_spec())
    flat = PlotSpec(
        kind=PlotKind.TIME_SERIES, layers=[Layer(PlotKind.LINE, {"x": [0.0], "y": [0.0]})]
    )
    assert not is_three_d(flat)


def test_line3d_renders_on_a_3d_axes(mpl_backend):
    from matplotlib.figure import Figure

    fig = _line3d_spec().render("matplotlib")
    assert isinstance(fig, Figure)
    assert _axes3d_name(fig) == "3d", "a 3-D spec must render on an mplot3d Axes3D"


def test_colored_line3d_gets_a_colorbar(mpl_backend):
    from tsdynamics.viz.spec import Colorbar

    t = np.linspace(0.0, 6.0, 60)
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        ndim=3,
        x=Axis(),
        y=Axis(),
        z=Axis(),
        clim=(0.0, 6.0),
        colorbar=Colorbar(label="t", cmap="viridis"),
        layers=[Layer(PlotKind.LINE3D, {"x": np.cos(t), "y": np.sin(t), "z": t, "c": t})],
    )
    fig = spec.render("matplotlib")
    assert _axes3d_name(fig) == "3d"
    # a colour-mapped 3-D line attaches a colorbar (a second axes).
    assert len(fig.axes) >= 2


def test_surface3d_renders(mpl_backend):
    xs = np.linspace(-1.0, 1.0, 8)
    ys = np.linspace(-1.0, 1.0, 8)
    zz = np.add.outer(ys**2, xs**2)
    spec = PlotSpec(
        kind=PlotKind.SURFACE3D,
        ndim=3,
        x=Axis(),
        y=Axis(),
        z=Axis(),
        layers=[Layer(PlotKind.SURFACE3D, {"x": xs, "y": ys, "z": zz})],
    )
    fig = spec.render("matplotlib")
    assert _axes3d_name(fig) == "3d"


def test_scatter3d_renders(mpl_backend):
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((40, 3))
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        ndim=3,
        x=Axis(),
        y=Axis(),
        z=Axis(),
        layers=[Layer(PlotKind.SCATTER, {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]})],
    )
    fig = spec.render("matplotlib")
    assert _axes3d_name(fig) == "3d"


def test_camera_and_box_aspect_applied(mpl_backend):
    spec = _line3d_spec()
    spec.meta = {"camera": {"elev": 12.0, "azim": 45.0}}
    fig = spec.render("matplotlib")
    ax = fig.axes[0]
    assert abs(ax.elev - 12.0) < 1e-6
    assert abs(ax.azim - 45.0) < 1e-6


# ---------------------------------------------------------------------------
# Engine-backed: real attractors render in 3-D, incl. an arbitrary L96 triple
# ---------------------------------------------------------------------------


def test_lorenz_and_rossler_render_in_3d(mpl_backend):
    pytest.importorskip("tsdynamics._rust")
    import tsdynamics as ts

    for system in (ts.Lorenz(), ts.Rossler()):
        traj = system.trajectory(final_time=20.0, dt=0.02)
        y = np.asarray(traj.y, dtype=float)
        spec = PlotSpec(
            kind=PlotKind.PHASE_PORTRAIT_3D,
            ndim=3,
            x=Axis(label="x"),
            y=Axis(label="y"),
            z=Axis(label="z"),
            layers=[Layer(PlotKind.LINE3D, {"x": y[:, 0], "y": y[:, 1], "z": y[:, 2]})],
        )
        fig = spec.render("matplotlib")
        assert _axes3d_name(fig) == "3d"


def test_lorenz96_renders_a_non_first_three_triple(mpl_backend):
    pytest.importorskip("tsdynamics._rust")
    import tsdynamics as ts

    sys = ts.systems.Lorenz96(N=8)
    traj = sys.trajectory(final_time=20.0, dt=0.02)
    y = np.asarray(traj.y, dtype=float)
    triple = (3, 5, 7)  # a real, non-first-three component triple
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        ndim=3,
        x=Axis(label="x3"),
        y=Axis(label="x5"),
        z=Axis(label="x7"),
        layers=[
            Layer(
                PlotKind.LINE3D, {"x": y[:, triple[0]], "y": y[:, triple[1]], "z": y[:, triple[2]]}
            )
        ],
    )
    fig = spec.render("matplotlib")
    assert _axes3d_name(fig) == "3d"
