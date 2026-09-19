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

    y = np.asarray(ts.systems.Lorenz().run(final_time=40.0, dt=0.01).y, dtype=float)
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
            # These specs are titled the same as their single layer, so the
            # overlay tag is not prefixed (``compose._relabel_for_overlay``
            # refuses to produce "a: a").
            Layer(a.layers[0].kind, dict(a.layers[0].data), label="a"),
            Layer(b.layers[0].kind, dict(b.layers[0].data), label="b"),
        ],
        legend=Legend(),
    )
    assert _png(composed) == _png(hand_built)


# ---------------------------------------------------------------------------
# The producer migration (P1) — the picture must not move
#
# The eight ``viz.producers`` builders became two-line shims over registered
# plot transforms.  A migration is the one change where "the output is the same"
# is the whole point: if a plot changes here, no later change to it can be
# attributed to the reason it was made.  So each producer is pinned twice —
#
#   1. a **structural fingerprint** committed as a literal below (the semantic
#      kind, the axes and their limits, the colorbar / legend, and every layer's
#      mark, label, provenance and channel set);
#   2. a **rendered figure** with the artist counts the fingerprint implies.
#
# Pixel-exact baselines stay out for the reason recorded at the top of this file
# (they flake across matplotlib / freetype / platform without catching more).
# Byte-identity of the rendered PNGs *was* verified across the migration itself,
# out of tree, on all 24 producer call shapes; the two deliberate spec-level
# differences are additive and invisible to every renderer: each layer now
# carries ``Layer.transform`` provenance, and each spec states the ``frame`` that
# was previously derived (``test_migrated_frames_match_the_derived_ones`` proves
# the stated one *is* the derived one, so overlay behaviour is unchanged).
# ---------------------------------------------------------------------------


class _GoldenSystem:
    """The minimal system stand-in the golden trajectories carry."""

    def __init__(self, discrete: bool = False, variables: tuple[str, ...] | None = None) -> None:
        self.is_discrete = discrete
        self.variables = variables


def _golden_flow(dim: int = 3, n: int = 120):
    from tsdynamics.data import Trajectory

    t = np.linspace(0.0, 10.0, n)
    y = np.column_stack([np.sin(t + k) * (k + 1) for k in range(dim)])
    names = ("x", "y", "z", "w", "v")[:dim]
    return Trajectory(t, y, _GoldenSystem(False, names), {"system": "demo", "dt": float(t[1])})


def _golden_orbit(n: int = 40):
    from tsdynamics.data import Trajectory

    t = np.arange(n, dtype=float)
    y = np.column_stack([np.cos(0.3 * t), np.sin(0.4 * t)])
    return Trajectory(t, y, _GoldenSystem(True, ("a", "b")), {"system": "demo map"})


def _golden_field(shape: tuple[int, ...] = (6, 8), n: int = 5):
    from tsdynamics.data import Trajectory

    t = np.linspace(0.0, 1.0, n)
    cells = int(np.prod(shape))
    y = np.stack([np.sin(np.arange(cells) * 0.3 + ti) for ti in t])
    return Trajectory(t, y, _GoldenSystem(), {"system": "demo field", "field_shape": shape})


def _golden_rhs(u):
    return np.array([-u[1] + 0.1 * u[0], u[0] + 0.1 * u[1]])


def _producer_specs() -> dict[str, PlotSpec]:
    """Build one spec per migrated producer (the golden call shapes)."""
    from tsdynamics.viz import producers

    return {
        "time_series": producers.time_series(_golden_flow()),
        "phase_portrait": producers.phase_portrait(_golden_flow()),
        "delay_embedding": producers.delay_embedding(_golden_flow(), delay=5, components="y"),
        "vector_field": producers.vector_field(_golden_rhs, xlim=(-1, 1), ylim=(-1, 1), grid=6),
        "phase_portrait_field": producers.phase_portrait_field(
            _golden_rhs, _golden_flow(dim=2), grid=5
        ),
        "cobweb": producers.cobweb(_golden_orbit(), components="a"),
        "spacetime": producers.spacetime(_golden_flow(dim=5)),
        "spatial_field": producers.spatial_field(_golden_field()),
    }


def _fingerprint(spec: PlotSpec) -> dict:
    """The committed structural signature of a spec (no float data, no pixels)."""

    def axis(a):
        if a is None:
            return None
        limits = None if a.limits is None else tuple(round(float(v), 6) for v in a.limits)
        return (a.label, limits)

    return {
        "kind": str(spec.kind),
        "ndim": spec.ndim,
        "aspect": spec.aspect,
        "axes": [axis(a) for a in (spec.x, spec.y, spec.z)],
        "colorbar": None if spec.colorbar is None else spec.colorbar.label,
        "legend": spec.legend is not None,
        "frame": spec.frame.describe(),
        "layers": [
            (str(lyr.kind), lyr.label, lyr.transform, sorted(lyr.data)) for lyr in spec.layers
        ],
    }


#: The golden structural fingerprints.  Regenerate only after a **reviewed**
#: change to what a producer draws — that is the point of committing them.
_GOLDEN_PRODUCERS: dict[str, dict] = {
    "time_series": {
        "kind": "time_series",
        "ndim": 1,
        "aspect": "auto",
        "axes": [("t", None), ("", None), None],
        "colorbar": None,
        "legend": True,
        "frame": "time(t)",
        "layers": [
            ("line", "x", "time_series", ["x", "y"]),
            ("line", "y", "time_series", ["x", "y"]),
            ("line", "z", "time_series", ["x", "y"]),
        ],
    },
    "phase_portrait": {
        "kind": "phase_portrait_3d",
        "ndim": 3,
        "aspect": "equal",
        "axes": [("x", None), ("y", None), ("z", None)],
        "colorbar": None,
        "legend": False,
        "frame": "state3(x, y, z)",
        "layers": [("line3d", None, "phase_portrait", ["x", "y", "z"])],
    },
    # v6: the axes name the CHANNEL that was embedded (``components="y"`` here)
    # instead of the ``label="x"`` default, and the lag is in the trajectory's own
    # time units instead of being a sample count typeset as a time.
    "delay_embedding": {
        "kind": "phase_portrait_2d",
        "ndim": 2,
        "aspect": "equal",
        "axes": [("y(t)", None), ("y(t - 0.420168)", None), None],
        "colorbar": None,
        "legend": False,
        "frame": "state2(y(t), y(t - 0.420168))",
        "layers": [("line", None, "delay_embedding", ["x", "y"])],
    },
    "vector_field": {
        "kind": "vector_field",
        "ndim": 2,
        "aspect": "equal",
        "axes": [("x", (-1.0, 1.0)), ("y", (-1.0, 1.0)), None],
        "colorbar": None,
        "legend": False,
        "frame": "state2(x, y)",
        "layers": [("quiver", None, "vector_field", ["u", "v", "x", "y"])],
    },
    "phase_portrait_field": {
        "kind": "phase_portrait_field",
        "ndim": 2,
        "aspect": "equal",
        "axes": [("x", (-1.099961, 1.099648)), ("y", (-2.199964, 2.19968)), None],
        "colorbar": None,
        "legend": True,
        "frame": "state2(x, y)",
        "layers": [
            ("quiver", None, "phase_portrait_field", ["u", "v", "x", "y"]),
            ("line", "trajectory", "phase_portrait_field", ["x", "y"]),
        ],
    },
    "cobweb": {
        "kind": "cobweb",
        "ndim": 2,
        "aspect": "equal",
        # A cobweb's axes ARE its domain — the box the map lives in, read off the
        # kernel when there is one and the orbit's own span when there is not (a
        # bare series, which is what the golden orbit is).  They used to be left
        # unset, so matplotlib autoscaled to the orbit and the ``f(x)`` curve's
        # own ends fell outside the frame it had been computed for.
        "axes": [("x_n", (-0.992225, 1.0)), ("x_(n+1)", (-0.992225, 1.0)), None],
        "colorbar": None,
        "legend": True,
        "frame": "state2(x_n, x_(n+1))",
        "layers": [
            ("line", "y = x", "cobweb", ["x", "y"]),
            ("line", "orbit", "cobweb", ["x", "y"]),
        ],
    },
    "spacetime": {
        "kind": "spacetime",
        "ndim": 2,
        "aspect": "auto",
        "axes": [("t", None), ("component", None), None],
        "colorbar": "state",
        "legend": False,
        "frame": "grid2(t, component)",
        "layers": [("image", None, "spacetime", ["c", "x", "y", "z"])],
    },
    "spatial_field": {
        "kind": "spatial_field",
        "ndim": 2,
        "aspect": "equal",
        "axes": [("x", None), ("y", None), None],
        "colorbar": "u",
        "legend": False,
        "frame": "grid2(x, y)",
        "layers": [("image", None, "spatial_field", ["frames", "x", "y", "z"])],
    },
}

#: Rendered-artist expectations per producer: ``(lines, collections, images)``.
#: ``quiver`` is a collection; a 3-D line lands in ``ax.lines`` like a 2-D one.
_GOLDEN_ARTISTS: dict[str, tuple[int, int, int]] = {
    "time_series": (3, 0, 0),
    "phase_portrait": (1, 0, 0),
    "delay_embedding": (1, 0, 0),
    "vector_field": (0, 1, 0),
    "phase_portrait_field": (1, 1, 0),
    "cobweb": (2, 0, 0),
    "spacetime": (0, 0, 1),
    "spatial_field": (0, 0, 1),
}


@pytest.mark.parametrize("name", sorted(_GOLDEN_PRODUCERS))
def test_migrated_producer_matches_its_golden_fingerprint(name):
    """Each migrated producer builds structurally exactly what it built before."""
    spec = _producer_specs()[name]
    assert _fingerprint(spec) == _GOLDEN_PRODUCERS[name]


@pytest.mark.parametrize("name", sorted(_GOLDEN_ARTISTS))
def test_migrated_producer_renders_the_expected_artists(name):
    """Each migrated producer draws real, non-degenerate artists — not just "no error"."""
    spec = _producer_specs()[name]
    fig = spec.render("matplotlib")
    ax = fig.axes[0]
    lines, collections, images = _GOLDEN_ARTISTS[name]
    assert len(ax.lines) == lines, f"{name}: lines"
    assert len(ax.collections) == collections, f"{name}: collections"
    assert len(ax.images) == images, f"{name}: images"
    for line in ax.lines:
        assert np.isfinite(np.asarray(line.get_xydata(), dtype=float)).any()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    _assert_valid_png(buf.getvalue())


@pytest.mark.parametrize("name", sorted(_GOLDEN_PRODUCERS))
def test_producer_shim_and_transform_agree(name):
    """The shim is the transform: same spec, byte-for-byte in every channel.

    If these two ever diverge the shim has grown logic of its own, which is
    exactly the drift the migration removed.
    """
    from tsdynamics.viz.transforms import build_spec, get

    calls = {
        "time_series": (_golden_flow(), {}),
        "phase_portrait": (_golden_flow(), {}),
        "delay_embedding": (_golden_flow(), {"delay": 5, "components": "y"}),
        "vector_field": (_golden_rhs, {"xlim": (-1, 1), "ylim": (-1, 1), "grid": 6}),
        "phase_portrait_field": (_golden_rhs, {"source": _golden_flow(dim=2), "grid": 5}),
        "cobweb": (_golden_orbit(), {"components": "a"}),
        "spacetime": (_golden_flow(dim=5), {}),
        "spatial_field": (_golden_field(), {}),
    }
    subject, options = calls[name]
    transform = get(_GOLDEN_PRODUCERS[name]["layers"][0][2])
    direct = build_spec(subject, transform.name, **options)
    shim = _producer_specs()[name]
    assert _fingerprint(direct) == _fingerprint(shim)
    for a, b in zip(direct.layers, shim.layers, strict=True):
        for channel, values in a.data.items():
            assert np.array_equal(values, b.data[channel], equal_nan=True), channel


@pytest.mark.parametrize("name", sorted(_GOLDEN_PRODUCERS))
def test_migrated_frames_match_the_derived_ones(name):
    """A stated frame equals the one P0 derived, so the overlay rule is untouched.

    Before P1 no producer set ``PlotSpec.frame`` and the composition front door
    derived it from the kind and the axis labels.  Stating it explicitly is only
    safe if the statement *is* the derivation — otherwise the migration would
    silently change which overlays are legal.
    """
    import dataclasses

    from tsdynamics.viz._frames import frame_of

    spec = _producer_specs()[name]
    assert spec.frame is not None
    assert spec.frame == frame_of(dataclasses.replace(spec, frame=None))
