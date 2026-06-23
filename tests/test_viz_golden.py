"""Golden render-image regression for the matplotlib backend (stream VIZ-RENDER-GATE).

The showcase figures rendered by ``docs/_tooling/make_analysis_figures.py`` are the
visual reference for the library.  This module is the *render-image* regression
that keeps the matplotlib backend producing real, non-degenerate figures for the
showcase kinds.

**Why not pixel-exact baselines.**  A committed pixel-for-pixel baseline is
notoriously fragile across matplotlib versions, freetype builds, and platforms
(the project has already seen arch-specific figure flakiness on macOS-ARM).  A
pixel diff would therefore flake in CI without catching a real rendering bug any
better than a structural check.  So this gate asserts the *structural* image
contract instead — every showcase spec renders to a valid, non-empty PNG with the
expected axes geometry (a colorbar axes for an image, a 3-D axes for an attractor)
— which catches an empty / broken / mis-dispatched render without false positives.
Pixel-exact baselines remain available as a future, opt-in `pytest-mpl` job.

Skips where matplotlib (or, for the engine showcases, the compiled extension) is
absent.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from tsdynamics import registry
from tsdynamics.viz.spec import Axis, Colorbar, Layer, PlotKind, PlotSpec

pytest.importorskip("matplotlib")

from tsdynamics.viz.render import register_builtin_renderers  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _mpl_backend():
    register_builtin_renderers()
    if "matplotlib" not in registry.renderers:  # pragma: no cover - matplotlib present here
        pytest.skip("matplotlib backend did not register")
    yield


def _png(spec: PlotSpec) -> bytes:
    """Render ``spec`` and return its PNG bytes."""
    fig = spec.render("matplotlib")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    return buf.getvalue()


def _assert_valid_png(data: bytes, *, min_bytes: int = 1200) -> None:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    assert len(data) > min_bytes, f"suspiciously small PNG ({len(data)} bytes)"


# ---------------------------------------------------------------------------
# Showcase specs render to valid, non-empty PNGs
# ---------------------------------------------------------------------------


def test_time_series_showcase_renders_png():
    t = np.linspace(0.0, 10.0, 200)
    spec = PlotSpec(
        kind=PlotKind.TIME_SERIES,
        x=Axis(label="t"),
        y=Axis(label="x"),
        layers=[Layer(PlotKind.LINE, {"x": t, "y": np.sin(t)}, label="x")],
    )
    _assert_valid_png(_png(spec))


def test_recurrence_showcase_is_sparse_and_renders_png():
    # A sparse recurrence scatter (never a dense N x N image).
    rng = np.random.default_rng(0)
    i = rng.integers(0, 64, 200).astype(float)
    j = rng.integers(0, 64, 200).astype(float)
    spec = PlotSpec(
        kind=PlotKind.RECURRENCE_PLOT,
        aspect="equal",
        x=Axis(label="i"),
        y=Axis(label="j"),
        layers=[Layer(PlotKind.SCATTER, {"x": i, "y": j}, style={"s": 1, "marker": "s"})],
    )
    _assert_valid_png(_png(spec))


def test_image_showcase_has_a_colorbar_axes():
    g = np.add.outer(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
    spec = PlotSpec(
        kind=PlotKind.BASINS_IMAGE,
        x=Axis(),
        y=Axis(),
        clim=(0.0, 2.0),
        colorbar=Colorbar(label="basin", cmap="tab20", discrete=True),
        layers=[Layer(PlotKind.IMAGE, {"x": np.arange(32.0), "y": np.arange(32.0), "c": g})],
    )
    fig = spec.render("matplotlib")
    # image + colorbar -> at least two axes (the image and the colorbar).
    assert len(fig.axes) >= 2
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    _assert_valid_png(buf.getvalue())


def test_bar_showcase_renders_png():
    spec = PlotSpec(
        kind=PlotKind.LYAPUNOV_SPECTRUM,
        x=Axis(label="index"),
        y=Axis(label=r"$\lambda$"),
        annotations=[],
        layers=[Layer(PlotKind.BAR, {"x": np.arange(3.0), "y": np.array([0.9, 0.0, -14.6])})],
    )
    _assert_valid_png(_png(spec))


# ---------------------------------------------------------------------------
# Engine-backed showcase: a real attractor renders in 3-D
# ---------------------------------------------------------------------------


def test_lorenz_attractor_showcase_renders_in_3d():
    pytest.importorskip("tsdynamics._rust")
    import tsdynamics as ts

    y = np.asarray(ts.Lorenz().trajectory(final_time=40.0, dt=0.01).y, dtype=float)
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        ndim=3,
        x=Axis(label="x"),
        y=Axis(label="y"),
        z=Axis(label="z"),
        layers=[Layer(PlotKind.LINE3D, {"x": y[:, 0], "y": y[:, 1], "z": y[:, 2]})],
    )
    fig = spec.render("matplotlib")
    assert getattr(fig.axes[0], "name", "") == "3d"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    _assert_valid_png(buf.getvalue(), min_bytes=3000)
