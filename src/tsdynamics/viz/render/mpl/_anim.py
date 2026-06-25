"""matplotlib animation rendering (stream VIZ-ANIM).

An animated :class:`~tsdynamics.viz.spec.PlotSpec` (one carrying an
:class:`~tsdynamics.viz.spec.Animation`) renders here to a
:class:`matplotlib.animation.FuncAnimation`.  The model is **reveal**: the spec's
layers keep their full static data and each frame shows a moving slice — a comet
whose head is the current sample and whose tail reaches back
:attr:`~tsdynamics.viz.spec.Animation.trail_length` (``None`` ⇒ persistent).

Per kind the "current state" head is drawn appropriately: a point marker on a
curve (phase portraits / delay embeddings), a vertical sweep line on a time
series or spacetime image.  Axis limits are computed once from the **full** data
and held fixed so the view does not jump between frames; a 3-D camera optionally
spins; an optional clock prints the current time.

This module imports matplotlib only when called (never at ``import tsdynamics``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...spec import Animation, PlotKind, PlotSpec
from .. import normalize_kind

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["render_animation"]


def _spec_dt(spec: PlotSpec) -> float | None:
    """Best-effort sample spacing for time-unit trails / the clock."""
    dt = spec.meta.get("dt") if isinstance(spec.meta, dict) else None
    try:
        return float(dt) if dt is not None and float(dt) > 0 else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _times(spec: PlotSpec, layer_x: np.ndarray | None, n: int) -> np.ndarray:
    """Return a per-sample time vector for the clock: a time-like x channel, else t0 + i·dt."""
    if layer_x is not None and layer_x.shape[0] == n and np.all(np.diff(layer_x) >= 0):
        return np.asarray(layer_x, dtype=float)
    dt = _spec_dt(spec) or 1.0
    t0 = float(spec.meta.get("t0", 0.0)) if isinstance(spec.meta, dict) else 0.0
    return t0 + dt * np.arange(n, dtype=float)


def _animated_marks() -> frozenset[PlotKind]:
    """Layer marks the reveal animator drives (curves; others are drawn static)."""
    return frozenset(
        {PlotKind.LINE, PlotKind.LINE3D, PlotKind.SCATTER, PlotKind.MARKERS, PlotKind.IMAGE}
    )


def render_animation(
    spec: PlotSpec, *, figsize: tuple[float, float] | None = None, **_kw: Any
) -> FuncAnimation:
    """Render an animated :class:`PlotSpec` to a :class:`FuncAnimation`.

    Single-panel specs animate their curve / image layers in reveal mode;
    composite specs animate every panel in lockstep on one shared frame clock.
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if spec.is_composite:
        return _render_composite_animation(spec, figsize=figsize)

    from . import _threed

    anim = spec.animation
    assert anim is not None  # guaranteed by the dispatch (is_animated)

    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    three_d = _threed.is_three_d(spec)
    ax = fig.add_subplot(1, 1, 1, projection="3d" if three_d else None)

    updater = _build_panel_animation(fig, ax, spec, three_d=three_d)
    interval = 1000.0 / float(anim.fps) if anim.fps > 0 else 50.0
    return FuncAnimation(
        fig,
        updater.update,
        frames=updater.n_steps,
        interval=interval,
        blit=False,
        repeat=bool(anim.loop),
    )


class _PanelUpdater:
    """Per-frame update closure for one panel (curves + head + camera + clock)."""

    def __init__(self, frame_seq: list[int], update_fn: Any) -> None:
        self._frame_seq = frame_seq
        self._update_fn = update_fn
        self.n_steps = len(frame_seq)

    def update(self, frame: int) -> Any:
        """Advance to playback frame ``frame`` (mapped through the head schedule)."""
        return self._update_fn(self._frame_seq[frame])


def _build_panel_animation(fig: Figure, ax: Any, spec: PlotSpec, *, three_d: bool) -> _PanelUpdater:
    """Draw the static frame + empty animated artists; return the per-frame updater.

    Returns an updater whose ``update(playback_frame)`` mutates the artists.  Used
    both for a single-panel animation and for each panel of a lockstep composite.
    """
    anim = spec.animation
    assert anim is not None
    dt = _spec_dt(spec)

    n_samples = _layer_sample_count(spec)
    head_idx = anim.head_indices(n_samples)
    tail = anim.tail_samples(dt)

    _apply_fixed_limits(ax, spec, three_d=three_d)
    _apply_static_labels(ax, spec, three_d=three_d)

    drivers = [_make_layer_driver(ax, layer, spec, anim, three_d=three_d) for layer in spec.layers]
    drivers = [d for d in drivers if d is not None]

    base_azim = _base_azim(spec) if three_d else None
    clock_artist = _make_clock(ax, spec, three_d=three_d) if anim.clock else None
    clock_times = _times(spec, _first_x(spec), n_samples) if anim.clock else None

    def update(i: int) -> Any:
        for driver in drivers:
            driver(i, tail)
        if three_d and base_azim is not None and anim.spin:
            frac = i / max(1, n_samples - 1)
            ax.view_init(elev=_base_elev(spec), azim=base_azim + 360.0 * anim.spin * frac)
        if clock_artist is not None and clock_times is not None:
            clock_artist.set_text(anim.clock_format.format(t=float(clock_times[i])))
        return []

    return _PanelUpdater(head_idx, update)


def _layer_sample_count(spec: PlotSpec) -> int:
    """Return the number of samples to reveal (the longest animated curve's ``x``/``y``)."""
    n = 0
    for layer in spec.layers:
        if PlotKind(layer.kind) not in _animated_marks():
            continue
        arr = layer.data.get("x", layer.data.get("y"))
        if arr is not None:
            n = max(n, int(np.asarray(arr).shape[0]))
    return max(n, 2)


def _first_x(spec: PlotSpec) -> np.ndarray | None:
    """Return the first animated layer's ``x`` channel (for clock time inference)."""
    for layer in spec.layers:
        if PlotKind(layer.kind) in _animated_marks() and "x" in layer.data:
            return np.asarray(layer.data["x"], dtype=float)
    return None


def _make_layer_driver(
    ax: Any, layer: Any, spec: PlotSpec, anim: Animation, *, three_d: bool
) -> Any:
    """Build a `(head_index, tail) -> None` updater for one layer (or ``None`` to skip).

    Curve layers animate as a revealing trail + a head marker; an ``IMAGE`` layer
    (spacetime) is drawn statically with a moving vertical sweep line; any other
    mark is drawn fully and not animated.
    """
    mark = PlotKind(layer.kind)
    kind = normalize_kind(spec.kind)

    if mark == PlotKind.IMAGE:
        return _image_sweep_driver(ax, layer, spec)
    if mark not in _animated_marks():
        _draw_static_layer(ax, layer, spec, three_d=three_d)
        return None
    return _curve_driver(ax, layer, spec, anim, kind, three_d=three_d)


def _curve_driver(
    ax: Any, layer: Any, spec: PlotSpec, anim: Animation, kind: PlotKind, *, three_d: bool
) -> Any:
    """Reveal a curve as a trail + a head marker (point, or sweep line for series)."""
    x = np.asarray(layer.data["x"], dtype=float)
    y = np.asarray(layer.data["y"], dtype=float)
    z = np.asarray(layer.data["z"], dtype=float) if "z" in layer.data else None
    color = layer.style.get("color")
    lw = layer.style.get("lw", layer.style.get("linewidth", 2.0))

    # Head style: a point on a portrait; a vertical sweep line on a time series.
    series_like = kind in (PlotKind.TIME_SERIES, PlotKind.SPACETIME)

    # With a windowed trail, draw the full curve once, faintly — the static,
    # context "attractor backdrop" the comet sweeps over (matches the plotly look).
    if anim.trail_kind is not None and not series_like:
        if three_d:
            assert z is not None
            ax.plot(x, y, z, color=color, lw=1.0, alpha=0.18)
        else:
            ax.plot(x, y, color=color, lw=1.0, alpha=0.18)

    # Fading-comet (glowing-tail) trail is opt-in via ``.trail(fade=True)``.
    if anim.trail_fade and not series_like:
        return _fade_comet_driver(ax, x, y, z, color, lw, anim, three_d=three_d)

    if three_d:
        (line,) = ax.plot([], [], [], color=color, lw=lw, label=layer.label)
        head = ax.plot(
            [], [], [], anim.head_symbol, color=anim.head_color or color, ms=anim.head_size
        )[0]
    else:
        (line,) = ax.plot([], [], color=color, lw=lw, label=layer.label)
        head = ax.plot([], [], anim.head_symbol, color=anim.head_color or color, ms=anim.head_size)[
            0
        ]
    head.set_visible(anim.head and not series_like)
    vline = (
        ax.axvline(x[0], color=anim.head_color or color, lw=1.0)
        if (anim.head and series_like and not three_d)
        else None
    )

    def drive(i: int, tail: int | None) -> None:
        lo = 0 if tail is None else max(0, i - tail)
        if three_d:
            assert z is not None  # a 3-D spec always carries the z channel
            line.set_data(x[lo : i + 1], y[lo : i + 1])
            line.set_3d_properties(z[lo : i + 1])
            if anim.head and not series_like:
                head.set_data([x[i]], [y[i]])
                head.set_3d_properties([z[i]])
        else:
            line.set_data(x[lo : i + 1], y[lo : i + 1])
            if anim.head and not series_like:
                head.set_data([x[i]], [y[i]])
        if vline is not None:
            vline.set_xdata([x[i], x[i]])

    return drive


def _set_head(
    head: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray | None, i: int, three_d: bool
) -> None:
    """Move the head marker to sample ``i`` (2-D or 3-D)."""
    head.set_data([x[i]], [y[i]])
    if three_d:
        assert z is not None
        head.set_3d_properties([z[i]])


def _fade_comet_driver(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    color: Any,
    lw: float,
    anim: Animation,
    *,
    three_d: bool,
) -> Any:
    """Drive a glowing comet: a fading per-segment-alpha trail + a bright head.

    Opt-in via ``.trail(fade=True)``; returns the per-frame ``(head, tail)`` updater.
    """
    import matplotlib.colors as mcolors

    base = mcolors.to_rgb(color) if color else mcolors.to_rgb("C0")
    if three_d:
        assert z is not None
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        # Seed with a real segment: add_collection3d computes bounds from the
        # segments and raises on an empty collection.
        seed = [np.array([[x[0], y[0], z[0]], [x[1], y[1], z[1]]])]
        lc: Any = Line3DCollection(seed, linewidths=lw)
        ax.add_collection3d(lc)
        head = ax.plot(
            [], [], [], anim.head_symbol, color=anim.head_color or color, ms=anim.head_size
        )[0]
    else:
        from matplotlib.collections import LineCollection

        lc = LineCollection([], linewidths=lw)
        ax.add_collection(lc)
        head = ax.plot([], [], anim.head_symbol, color=anim.head_color or color, ms=anim.head_size)[
            0
        ]
    head.set_visible(anim.head)

    def drive(i: int, tail: int | None) -> None:
        lo = 0 if tail is None else max(0, i - tail)
        if i - lo >= 1:
            if three_d:
                assert z is not None
                pts = np.column_stack([x[lo : i + 1], y[lo : i + 1], z[lo : i + 1]])
            else:
                pts = np.column_stack([x[lo : i + 1], y[lo : i + 1]])
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            rgba = np.tile([*base, 1.0], (len(segs), 1))
            rgba[:, 3] = np.linspace(0.05, 1.0, len(segs))  # fade tail→head
            lc.set_segments(list(segs))
            lc.set_color(rgba)
        if anim.head:
            _set_head(head, x, y, z, i, three_d)

    return drive


def _image_sweep_driver(ax: Axes, layer: Any, spec: PlotSpec) -> Any:
    """Draw a static spacetime image with a moving vertical "now" sweep line."""
    from ._core import _draw_image, _preset_for

    _draw_image(ax, layer, spec, _preset_for(normalize_kind(spec.kind)))
    x = layer.data.get("x")
    xs = np.asarray(x, dtype=float) if x is not None else None
    n = xs.shape[0] if xs is not None else _layer_sample_count(spec)
    line = ax.axvline(xs[0] if xs is not None else 0.0, color="white", lw=1.2, alpha=0.8)

    def drive(i: int, tail: int | None) -> None:
        xi = float(xs[min(i, n - 1)]) if xs is not None else float(i)
        line.set_xdata([xi, xi])

    return drive


def _draw_static_layer(ax: Axes, layer: Any, spec: PlotSpec, *, three_d: bool) -> None:
    """Draw a non-animated layer in full (e.g. a y=x diagonal under a cobweb)."""
    from ._core import MARK_DISPATCH, _preset_for

    if three_d:
        return
    drawer = MARK_DISPATCH.get(PlotKind(layer.kind))
    if drawer is not None:
        drawer(ax, layer, spec, _preset_for(normalize_kind(spec.kind)))


# ---------------------------------------------------------------------------
# Axes framing (fixed limits + labels), camera, clock
# ---------------------------------------------------------------------------


def _data_range(spec: PlotSpec, channel: str) -> tuple[float, float] | None:
    """Min/max of ``channel`` across all layers (for fixed animation limits)."""
    lo, hi = np.inf, -np.inf
    for layer in spec.layers:
        arr = layer.data.get(channel)
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    if hi <= lo:
        return (lo - 1.0, hi + 1.0)
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)


def _apply_fixed_limits(ax: Any, spec: PlotSpec, *, three_d: bool) -> None:
    """Hold axis limits fixed across frames (spec override wins over data extent)."""
    xr = spec.x.limits or _data_range(spec, "x")
    yr = spec.y.limits or _data_range(spec, "y")
    if xr is not None:
        ax.set_xlim(*xr)
    if yr is not None:
        ax.set_ylim(*yr)
    if three_d:
        zr = (spec.z.limits if spec.z is not None else None) or _data_range(spec, "z")
        if zr is not None:
            ax.set_zlim(*zr)


def _apply_static_labels(ax: Any, spec: PlotSpec, *, three_d: bool) -> None:
    """Apply axis labels, title, aspect (the non-data framing)."""
    if spec.x.label:
        ax.set_xlabel(spec.x.label)
    if spec.y.label:
        ax.set_ylabel(spec.y.label)
    if three_d and spec.z is not None and spec.z.label:
        ax.set_zlabel(spec.z.label)
    if spec.title:
        ax.set_title(spec.title)
    if not three_d and spec.aspect == "equal":
        ax.set_aspect("equal", adjustable="box")
    if spec._axes_hidden():
        ax.set_axis_off()


def _base_elev(spec: PlotSpec) -> float:
    """Return the base camera elevation from ``meta["camera"]`` (matplotlib default else)."""
    cam = spec.meta.get("camera") if isinstance(spec.meta, dict) else None
    return float(cam["elev"]) if isinstance(cam, dict) and "elev" in cam else 30.0


def _base_azim(spec: PlotSpec) -> float:
    """Return the base camera azimuth from ``meta["camera"]`` (matplotlib default else)."""
    cam = spec.meta.get("camera") if isinstance(spec.meta, dict) else None
    return float(cam["azim"]) if isinstance(cam, dict) and "azim" in cam else -60.0


def _make_clock(ax: Any, spec: PlotSpec, *, three_d: bool) -> Any:
    """Create the per-frame time-readout text artist (top-left, axes coords)."""
    if three_d:
        return ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    return ax.text(0.02, 0.95, "", transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Composite (lockstep): one FuncAnimation drives every panel on a shared clock
# ---------------------------------------------------------------------------


def _render_composite_animation(
    spec: PlotSpec, *, figsize: tuple[float, float] | None
) -> FuncAnimation:
    """Animate a composite: tile the panels and advance them all on one clock."""
    from matplotlib.animation import FuncAnimation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from . import _threed
    from ._core import _composite_grid

    anim = spec.animation or Animation()
    panels = spec.panels
    rows, cols = _composite_grid(spec.layout, len(panels))
    if figsize is None:
        figsize = (cols * 5.0, rows * 3.2)
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)

    updaters: list[_PanelUpdater] = []
    for i, panel in enumerate(panels):
        # Each panel inherits the composite's clock if it has none of its own.
        if panel.animation is None:
            panel.animation = anim
        three_d = _threed.is_three_d(panel)
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d" if three_d else None)
        updaters.append(_build_panel_animation(fig, ax, panel, three_d=three_d))
    if spec.title:
        fig.suptitle(spec.title)

    n_steps = max((u.n_steps for u in updaters), default=2)

    def update(frame: int) -> Any:
        for u in updaters:
            u.update(min(frame, u.n_steps - 1))  # clamp shorter panels to their last frame
        return []

    interval = 1000.0 / float(anim.fps) if anim.fps > 0 else 50.0
    return FuncAnimation(
        fig, update, frames=n_steps, interval=interval, blit=False, repeat=bool(anim.loop)
    )
