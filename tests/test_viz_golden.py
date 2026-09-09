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
from tsdynamics.viz.spec import Axis, Colorbar, Layer, Legend, PlotKind, PlotSpec

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


# ---------------------------------------------------------------------------
# Overlay showcase (P0 composability) — the merged spec is a real figure, and
# the merge itself changes no pixels
# ---------------------------------------------------------------------------


def _field_spec() -> PlotSpec:
    """A basin-image-shaped field spec on the ``(x, v)`` plane."""
    g = np.add.outer(np.linspace(0, 1, 24), np.linspace(0, 1, 24))
    return PlotSpec(
        kind=PlotKind.BASINS_IMAGE,
        aspect="equal",
        x=Axis(label="x"),
        y=Axis(label="v"),
        clim=(0.0, 2.0),
        colorbar=Colorbar(label="basin", cmap="tab20", discrete=True),
        layers=[
            Layer(
                PlotKind.IMAGE,
                {"x": np.arange(24.0), "y": np.arange(24.0), "c": g},
                label="basins",
            )
        ],
        title="basins",
    )


def _orbit_spec(label: str = "orbit") -> PlotSpec:
    t = np.linspace(0.0, 6.0, 64)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        x=Axis(label="x"),
        y=Axis(label="v"),
        layers=[
            Layer(PlotKind.LINE, {"x": 8 + 6 * np.cos(t), "y": 12 + 6 * np.sin(t)}, label=label)
        ],
        title=label,
    )


def _equilibria_spec() -> PlotSpec:
    return PlotSpec(
        kind=PlotKind.FIXED_POINTS_OVERLAY,
        x=Axis(label="x"),
        y=Axis(label="v"),
        layers=[
            Layer(
                PlotKind.SCATTER,
                {"x": np.array([6.0, 12.0]), "y": np.array([12.0, 12.0])},
                label="equilibria",
            )
        ],
        title="equilibria",
    )


def test_overlay_showcase_draws_field_curve_and_markers_on_one_axes():
    """The flagship composition renders as one real, non-degenerate figure."""
    import tsdynamics.viz as viz

    spec = viz.plot(_field_spec(), _orbit_spec(), _equilibria_spec())
    fig = spec.render("matplotlib")
    ax = fig.axes[0]
    assert len(ax.images) == 1  # the field, underneath
    assert len(ax.lines) == 1  # the orbit, over it
    assert len(ax.collections) == 1  # the equilibria, on top
    assert len(fig.axes) >= 2  # the field kept its colorbar
    _assert_valid_png(_png(spec))


def test_overlay_render_is_invariant_under_argument_order():
    """Z-order is by role, so the picture does not depend on the call order.

    Byte-identical PNGs — the strongest form of the claim, and the one that
    would break the moment draw order started following argument order again.
    """
    import tsdynamics.viz as viz

    forward = _png(viz.plot(_field_spec(), _orbit_spec(), _equilibria_spec()))
    backward = _png(viz.plot(_equilibria_spec(), _orbit_spec(), _field_spec()))
    _assert_valid_png(forward)
    assert forward == backward


def test_merging_two_same_kind_specs_draws_exactly_the_hand_built_merge():
    """The regression bar: composing changes no pixels it did not have to.

    A same-kind overlay is the only shape that was legal before v6.  Rendering
    one is byte-identical to rendering a hand-built spec carrying the same
    layers, so the frame check, the role sort and the two new spec fields are
    provably inert on the paths that already worked.
    """
    import tsdynamics.viz as viz

    a, b = _orbit_spec("a"), _orbit_spec("b")
    composed = viz.plot(_orbit_spec("a"), _orbit_spec("b"))
    hand_built = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        x=Axis(label="x"),
        y=Axis(label="v"),
        layers=[
            Layer(a.layers[0].kind, dict(a.layers[0].data), label="a: a"),
            Layer(b.layers[0].kind, dict(b.layers[0].data), label="b: b"),
        ],
        legend=Legend(),
    )
    assert _png(composed) == _png(hand_built)
