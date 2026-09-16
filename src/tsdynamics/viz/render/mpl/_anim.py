"""matplotlib animation rendering (stream VIZ-ANIM).

An animated :class:`~tsdynamics.viz.spec.PlotSpec` (one carrying an
:class:`~tsdynamics.viz.spec.Animation`) renders here to a
:class:`matplotlib.animation.FuncAnimation`.  Two frame models are supported
(:attr:`~tsdynamics.viz.spec.Animation.mode`):

- ``"reveal"`` (the default): the spec's layers keep their full static data and
  each frame shows a moving slice — a comet whose head is the current sample and
  whose tail reaches back :attr:`~tsdynamics.viz.spec.Animation.trail_length`
  (``None`` ⇒ persistent).  Per kind the "current state" head is drawn
  appropriately: a point marker on a curve (phase portraits / delay embeddings),
  a vertical sweep line on a time series or spacetime image.
- ``"frames"``: a **spatial-field movie** — a
  :data:`~tsdynamics.viz.spec.PlotKind.SPATIAL_FIELD` spec whose layer carries the
  full per-time field stack on its ``"frames"`` channel (shape ``(T, *spatial)``).
  Each playback frame is the field's *spatial* state at that instant, so the plot
  genuinely evolves: a 2-D field plays as an ``imshow`` heatmap movie (Gray–Scott
  / Swift–Hohenberg), a 1-D field as a travelling-wave line (the
  Kuramoto–Sivashinsky profile).  Consecutive frames carry different data, not a
  sweep line over a static image.

Axis limits (and an image's colour range) are computed once from the **full**
data and held fixed so the view does not jump between frames; a 3-D camera
optionally spins; an optional clock prints the current time.

Orthogonally to :attr:`~tsdynamics.viz.spec.Animation.mode`, a **composite** spec
animates in one of two ways, selected by its :class:`~tsdynamics.viz.spec.Layout`:

- ``layout`` mode ``"stack"`` / ``"row"`` / ``"grid"`` — the panels are tiled and
  played in **lockstep** on one master clock
  (:func:`_render_composite_animation`); the panels are consecutive in *space*.
- ``layout`` mode ``"frames"`` — the panels are consecutive in *time*
  (:func:`_render_frames_movie`).  Each panel is one frame of the movie, so a
  parameter sweep (``[ts.plot(sys.with_params(r=r), "cobweb") for r in rs]``,
  ``layout="frames"``) plays as a cascade with **no new API**: panels are already
  the composition unit, so ``share_x`` / ``share_y`` / per-panel styling / nesting
  all apply unchanged.  This mode is matplotlib-only (mp4 / gif) in v6 — plotly
  and three.js decline an animated composite and say so.

This module imports matplotlib only when called (never at ``import tsdynamics``).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np

from ...spec import Animation, PlotKind, PlotSpec
from .. import normalize_kind
from ._core import (
    _apply_theme_color_cycle,
    _apply_theme_to_figure,
    _resolve_theme,
    figure_geometry,
    new_figure,
)

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
    composite specs animate every panel in lockstep on one shared frame clock —
    unless their layout mode is ``"frames"``, in which case the panels *are* the
    frames and are played one after another (:func:`_render_frames_movie`).
    """
    from matplotlib.animation import FuncAnimation

    if spec.is_composite:
        if _is_frames_layout(spec):
            return _render_frames_movie(spec, figsize=figsize)
        return _render_composite_animation(spec, figsize=figsize)

    from . import _threed

    anim = spec.animation
    assert anim is not None  # guaranteed by the dispatch (is_animated)

    theme = _resolve_theme(spec)
    figsize, dpi, layout = figure_geometry(spec, figsize, theme=theme)
    fig = new_figure(figsize, dpi, layout)
    three_d = _threed.is_three_d(spec)
    ax = fig.add_subplot(1, 1, 1, projection="3d" if three_d else None)
    _apply_theme_to_figure(fig, ax, theme)
    _apply_theme_color_cycle(ax, theme)

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

    Applies the spec's theme figure-locally (background, color cycle, font) before
    drawing any layer.
    """
    spec = _play_the_field_stack(spec)
    anim = spec.animation
    assert anim is not None
    if anim.mode == "frames":
        _warn_if_frames_without_field(spec)
    dt = _spec_dt(spec)

    # Apply theme to this panel's axes
    theme = _resolve_theme(spec)
    _apply_theme_to_figure(fig, ax, theme)
    _apply_theme_color_cycle(ax, theme)

    n_samples = _layer_sample_count(spec)
    head_idx = anim.head_indices(n_samples)
    tail = anim.tail_samples(dt)

    _apply_fixed_limits(ax, spec, three_d=three_d)
    _apply_static_labels(ax, spec, three_d=three_d)

    drivers = [
        _make_layer_driver(ax, layer, spec, anim, three_d=three_d, n_clock=n_samples)
        for layer in spec.layers
    ]
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
    """Return the number of samples to reveal (the longest animated curve / field).

    For a curve mark this is the length of its ``x`` / ``y`` channel; for a
    ``frames``-mode spatial-field layer it is the number of **time frames** on the
    ``"frames"`` channel (``frames.shape[0]``).
    """
    anim = spec.animation
    frames_mode = anim is not None and anim.mode == "frames"
    n = 0
    for layer in spec.layers:
        if frames_mode:
            stack = _field_stack(layer)
            if stack is not None:
                n = max(n, int(stack.shape[0]))
                continue
        if PlotKind(layer.kind) not in _animated_marks():
            continue
        arr = layer.data.get("x", layer.data.get("y"))
        if arr is not None:
            n = max(n, int(np.asarray(arr).shape[0]))
    return max(n, 2)


def _field_stack(layer: Any) -> np.ndarray | None:
    """Return a layer's per-time field stack from its ``"frames"`` channel, else ``None``.

    The :data:`~tsdynamics.viz.spec.PlotKind.SPATIAL_FIELD` producer stacks every
    per-time spatial snapshot on this channel — shape ``(T, Nx)`` for a 1-D profile
    or ``(T, Ny, Nx)`` for a 2-D field.  A spec with no such channel has no field
    movie to play (the ``frames``-mode warning fires).
    """
    arr = layer.data.get("frames")
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    return a if a.ndim in (2, 3) else None


def _play_the_field_stack(spec: PlotSpec) -> PlotSpec:
    """Play a field stack that is present, whichever door asked for the animation.

    A layer's ``"frames"`` channel exists for exactly one reason: the
    :data:`~tsdynamics.viz.spec.PlotKind.SPATIAL_FIELD` producer stacked every
    per-time snapshot on it so the field could be *played*.  Animating such a spec
    in ``reveal`` mode sweeps a line across the **final** field instead — a picture
    that is not wrong-looking, just wrong.

    Two doors build that spec and only one of them said so.  Measured on a
    Swift–Hohenberg lattice whose stack is ``(101, 8, 8)``::

        traj.to_plot_spec(kind="field", animate=True)   mode='frames'   101 frames
        ts.plot(traj, "spatial_field", animate=True)    mode='reveal'     8 frames

    — eight being the width of the lattice, because the reveal model read the
    *spatial* x-axis as its sample axis.  The second spelling is the one §6.6's
    own example uses, so the movie the contract advertises played 8 of 101 frames
    and swept a ruler over a frozen field.  The mode is a property of the data, not
    of the door, so it is resolved here, once, for every caller.

    Mirrors the field defaults the recipe door already forces (no head, no trail —
    a heatmap has no comet).  Returns ``spec`` unchanged when there is no stack or
    the mode is already ``"frames"``.
    """
    anim = spec.animation
    if anim is None or anim.mode == "frames":
        return spec
    if spec.kind is not PlotKind.SPATIAL_FIELD:
        return spec
    if not any(_field_stack(layer) is not None for layer in spec.layers):
        return spec
    return dataclasses.replace(
        spec,
        animation=dataclasses.replace(
            anim, mode="frames", head=False, trail_kind=None, trail_length=None
        ),
    )


def _warn_if_frames_without_field(spec: PlotSpec) -> None:
    """Warn (degrade) when ``mode="frames"`` is asked for a spec with no field stack.

    The spatial-field movie model needs a layer carrying the per-time field stack
    on its ``"frames"`` channel (a ``SPATIAL_FIELD`` spec).  On any other spec there
    is no such stack, so the renderer falls back to the reveal drivers; this emits a
    :class:`~tsdynamics.viz.render.caps.VisualizationDegraded` so the degrade is
    visible rather than silent.  A no-op when a field stack is present.
    """
    import warnings

    has_field = any(_field_stack(layer) is not None for layer in spec.layers)
    if has_field:
        return
    from ..caps import VisualizationDegraded

    warnings.warn(
        'animate mode="frames" plays a spatial-field movie, which needs a field '
        f'stack (a "field" kind); this spec ({spec.kind.value!r}) has none, so it '
        'animates with the reveal model instead. Use kind="field" for a field movie.',
        VisualizationDegraded,
        stacklevel=2,
    )


def _first_x(spec: PlotSpec) -> np.ndarray | None:
    """Return the first animated layer's ``x`` channel (for clock time inference)."""
    anim = spec.animation
    frames_mode = anim is not None and anim.mode == "frames"
    for layer in spec.layers:
        mark = PlotKind(layer.kind)
        animated = mark in _animated_marks() or (frames_mode and _field_stack(layer) is not None)
        if animated and "x" in layer.data:
            xa = np.asarray(layer.data["x"], dtype=float)
            if xa.ndim == 1:
                return xa
    return None


def _local_clock(n_local: int, n_clock: int) -> Any:
    """Map the figure's global sample index onto ONE layer's own sample axis.

    The animation clock is sized from the longest curve, and every layer used to
    be indexed with that same global number — so ``ts.plot(a, b,
    "phase_portrait", animate=True)`` on two orbits of unequal length raised
    ``IndexError: index 2006 is out of bounds for axis 0 with size 2001``.  That
    is the most ordinary animation in dynamics (compare two initial conditions),
    and it crashed on the first spelling.

    The mapping is by **progress**: every curve is revealed over the whole movie
    and every comet arrives at its own last sample on the last frame.  A tail
    length in global samples scales the same way, so a trail looks the same
    length on every curve.
    """
    if n_clock <= 1 or n_local <= 1 or n_local == n_clock:
        return lambda i, tail: (min(max(int(i), 0), max(n_local - 1, 0)), tail)
    ratio = (n_local - 1) / (n_clock - 1)

    def _map(i: int, tail: int | None) -> tuple[int, int | None]:
        j = int(round(min(max(int(i), 0), n_clock - 1) * ratio))
        scaled = None if tail is None else max(1, int(round(tail * ratio)))
        return min(j, n_local - 1), scaled

    return _map


def _make_layer_driver(
    ax: Any, layer: Any, spec: PlotSpec, anim: Animation, *, three_d: bool, n_clock: int = 0
) -> Any:
    """Build a `(frame_index, tail) -> None` updater for one layer (or ``None`` to skip).

    In ``frames`` mode a layer carrying a per-time field stack plays as a
    **spatial-field movie** — a 2-D heatmap or a 1-D profile materialised per frame;
    in ``reveal`` mode curve layers animate as a revealing trail + a head marker and
    an ``IMAGE`` layer (spacetime) is drawn statically with a moving vertical sweep
    line; any other mark is drawn fully and not animated.
    """
    mark = PlotKind(layer.kind)
    kind = normalize_kind(spec.kind)

    if anim.mode == "frames" and _field_stack(layer) is not None:
        return _field_movie_driver(ax, layer, spec)
    if mark == PlotKind.IMAGE:
        return _image_sweep_driver(ax, layer, spec)
    if mark not in _animated_marks():
        _draw_static_layer(ax, layer, spec, three_d=three_d)
        return None
    return _curve_driver(ax, layer, spec, anim, kind, three_d=three_d, n_clock=n_clock)


def _curve_driver(
    ax: Any,
    layer: Any,
    spec: PlotSpec,
    anim: Animation,
    kind: PlotKind,
    *,
    three_d: bool,
    n_clock: int = 0,
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
        return _fade_comet_driver(ax, x, y, z, color, lw, anim, three_d=three_d, n_clock=n_clock)

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

    clock = _local_clock(len(x), n_clock or len(x))

    def drive(i: int, tail: int | None) -> None:
        i, tail = clock(i, tail)
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
    n_clock: int = 0,
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
    clock = _local_clock(len(x), n_clock or len(x))

    def drive(i: int, tail: int | None) -> None:
        i, tail = clock(i, tail)
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

    theme = _resolve_theme(spec)
    _draw_image(ax, layer, spec, _preset_for(normalize_kind(spec.kind)), theme)
    x = layer.data.get("x")
    xs = np.asarray(x, dtype=float) if x is not None else None
    n = xs.shape[0] if xs is not None else _layer_sample_count(spec)
    line = ax.axvline(xs[0] if xs is not None else 0.0, color="white", lw=1.2, alpha=0.8)

    def drive(i: int, tail: int | None) -> None:
        xi = float(xs[min(i, n - 1)]) if xs is not None else float(i)
        line.set_xdata([xi, xi])

    return drive


def _field_movie_driver(ax: Axes, layer: Any, spec: PlotSpec) -> Any:
    """Drive a **spatial-field movie**: the per-time field stack played frame by frame.

    The :data:`~tsdynamics.viz.spec.PlotKind.SPATIAL_FIELD` producer stacks every
    per-time spatial snapshot on the layer's ``"frames"`` channel (shape
    ``(T, Ny, Nx)`` for a 2-D field, ``(T, Nx)`` for a 1-D profile).  Each playback
    frame ``i`` shows ``stack[i]`` — the field's genuine spatial state at that
    instant — so consecutive frames differ: a 2-D field plays as an ``imshow``
    heatmap movie, a 1-D field as a travelling-wave line.  The colour range (2-D) /
    the y-limits (1-D) are fixed from the **whole** stack so the view never jumps.
    """
    stack = _field_stack(layer)
    if stack is None:  # pragma: no cover - guarded by _make_layer_driver
        return _image_sweep_driver(ax, layer, spec)
    n_steps = int(stack.shape[0])
    finite = stack[np.isfinite(stack)]
    if spec.clim is not None:
        vmin, vmax = spec.clim
    elif finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
    else:  # pragma: no cover - degenerate empty field
        vmin, vmax = 0.0, 1.0

    if stack.ndim == 3:
        return _field_movie_2d(ax, layer, spec, stack, (vmin, vmax), n_steps)
    return _field_movie_1d(ax, layer, spec, stack, (vmin, vmax), n_steps)


def _field_movie_2d(
    ax: Axes,
    layer: Any,
    spec: PlotSpec,
    stack: np.ndarray,
    clim: tuple[float, float],
    n_steps: int,
) -> Any:
    """Play a 2-D field stack ``(T, Ny, Nx)`` as an ``imshow`` heatmap movie."""
    from ._core import _make_norm, _preset_for, _resolve_cmap, _resolve_norm

    preset = _preset_for(normalize_kind(spec.kind))
    cmap = _resolve_cmap(spec, layer, preset)
    interp = layer.style.get("interpolation", "nearest")
    norm = _make_norm(_resolve_norm(spec, preset), clim)
    ny, nx = stack.shape[1], stack.shape[2]
    im = ax.imshow(
        stack[0],
        origin="lower",
        aspect="equal" if spec.aspect == "equal" else "auto",
        extent=(0.0, float(nx), 0.0, float(ny)),
        cmap=cmap,
        norm=norm,
        interpolation=interp,
    )

    def drive(i: int, tail: int | None) -> None:
        im.set_data(stack[min(i, n_steps - 1)])

    return drive


def _field_movie_1d(
    ax: Axes,
    layer: Any,
    spec: PlotSpec,
    stack: np.ndarray,
    ylim: tuple[float, float],
    n_steps: int,
) -> Any:
    """Play a 1-D field stack ``(T, Nx)`` as a travelling-wave line movie."""
    x = layer.data.get("x")
    xs = np.asarray(x, dtype=float) if x is not None else np.arange(stack.shape[1], dtype=float)
    color = layer.style.get("color")
    lw = layer.style.get("lw", layer.style.get("linewidth", 2.0))
    (line,) = ax.plot(xs, stack[0], color=color, lw=lw, label=layer.label)
    # Fix the y-range from the whole stack (with a little pad) so the profile does
    # not rescale frame to frame.
    lo, hi = ylim
    if hi > lo:
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

    def drive(i: int, tail: int | None) -> None:
        line.set_ydata(stack[min(i, n_steps - 1)])

    return drive


def _draw_static_layer(ax: Axes, layer: Any, spec: PlotSpec, *, three_d: bool) -> None:
    """Draw a non-animated layer in full (e.g. a y=x diagonal under a cobweb)."""
    from ._core import MARK_DISPATCH, _preset_for

    if three_d:
        return
    drawer = MARK_DISPATCH.get(PlotKind(layer.kind))
    if drawer is not None:
        theme = _resolve_theme(spec)
        drawer(ax, layer, spec, _preset_for(normalize_kind(spec.kind)), theme)


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

    from . import _threed
    from ._core import _composite_grid

    anim = spec.animation or Animation()
    panels = spec.panels
    rows, cols = _composite_grid(spec.layout, len(panels))
    composite_theme = _resolve_theme(spec)
    figsize, dpi, layout_engine = figure_geometry(spec, figsize, theme=composite_theme)
    if figsize is None:
        figsize = (cols * 5.0, rows * 3.2)

    fig = new_figure(figsize, dpi, layout_engine)
    if composite_theme.background is not None:
        fig.patch.set_facecolor(composite_theme.background)

    updaters: list[_PanelUpdater] = []
    for i, panel in enumerate(panels):
        # Resolve the effective animation/theme LOCALLY — never mutate the caller's
        # panel specs. A panel inherits the composite's clock/animation and theme
        # when it carries none of its own; the resolved values are threaded into a
        # throwaway copy passed to the per-panel draw.
        effective = dataclasses.replace(
            panel,
            animation=panel.animation or anim,
            _theme=panel._theme or composite_theme,
        )
        three_d = _threed.is_three_d(effective)
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d" if three_d else None)
        updaters.append(_build_panel_animation(fig, ax, effective, three_d=three_d))
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


# ---------------------------------------------------------------------------
# layout="frames" — the panels ARE the frames (a movie of any N plots)
# ---------------------------------------------------------------------------


def _is_frames_layout(spec: PlotSpec) -> bool:
    """Whether ``spec`` is a composite whose panels are frames, not tiles.

    ``Layout.mode == "frames"`` is the fifth composition mode: the panels are
    consecutive **in time** rather than in space.  Read defensively (``getattr``)
    because a spec deserialized from an older envelope carries no ``layout`` at
    all, and a hard attribute read there would turn a missing key into a crash.
    """
    layout = getattr(spec, "layout", None)
    return bool(spec.is_composite and getattr(layout, "mode", None) == "frames")


def _frames_axis_ranges(
    panels: list[PlotSpec],
) -> dict[str, tuple[float, float]]:
    """Union each panel's data extent per axis, so the view does not jump per frame.

    A sweep movie whose axes rescale on every frame is unreadable — the motion you
    see is the *axes* moving, not the dynamics.  Every frame of a ``"frames"``
    composite is drawn on **one** axes, so the ranges are the union over all
    panels; ``share_x`` / ``share_y`` describe panel-to-panel sharing in a *tiled*
    layout and add nothing here (rather than shipping a knob that cannot be
    distinguished from its own default, this mode simply does the right thing).

    Honours a panel's explicit ``limits`` the same way :func:`_apply_fixed_limits`
    does — an author who pinned a range means it.
    """
    out: dict[str, tuple[float, float]] = {}
    for channel in ("x", "y", "z"):
        lo, hi = np.inf, -np.inf
        for panel in panels:
            axis = getattr(panel, channel, None)
            rng = (axis.limits if axis is not None else None) or _data_range(panel, channel)
            if rng is None:
                continue
            lo, hi = min(lo, float(rng[0])), max(hi, float(rng[1]))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out[channel] = (lo, hi)
    return out


def _frames_clim(panels: list[PlotSpec]) -> tuple[float, float] | None:
    """Union of the panels' colour ranges, for ``share_color=True``.

    The colour-scale half of the same rule: a heatmap whose colour meaning changes
    every frame cannot be compared with the frame before it.  ``None`` when no
    panel declares a ``clim`` and no panel carries a colour channel.
    """
    lo, hi = np.inf, -np.inf
    for panel in panels:
        rng = getattr(panel, "clim", None) or _data_range(panel, "c")
        if rng is None:
            continue
        lo, hi = min(lo, float(rng[0])), max(hi, float(rng[1]))
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else None


def _render_frames_movie(spec: PlotSpec, *, figsize: tuple[float, float] | None) -> FuncAnimation:
    """Play a composite's panels as consecutive frames of one movie.

    ``ts.plot(*[ts.plot(logistic.with_params(r=r), "cobweb") for r in rs],
    layout="frames", fps=15).save("cascade.mp4")`` — a parameter-sweep movie
    expressed entirely in the composition grammar.

    Each playback frame **re-draws one panel in full** through the same static
    panel bodies the tiled composite renderer uses (``_draw_2d_panel`` /
    ``_draw_3d_panel``), so a frame of the movie is byte-for-byte the picture that
    panel renders on its own — including its colorbar, legend and annotations, and
    including a 3-D panel next to a 2-D one (the figure is rebuilt per frame, so
    the projection may change between frames; a *tiled* composite cannot do that
    on one axes either).

    The frame sequence comes from :meth:`~tsdynamics.viz.spec.Animation.head_indices`
    over the panel count, so ``n_frames`` / ``duration`` / ``pingpong`` mean exactly
    what they mean everywhere else — with the panel list, not a sample axis, as the
    thing being indexed.

    Parameters
    ----------
    spec : PlotSpec
        A composite whose :class:`~tsdynamics.viz.spec.Layout` mode is ``"frames"``.
    figsize : tuple of float, optional
        Explicit figure size; otherwise the theme's (a frame shows **one** panel,
        so — unlike the tiled composite — there is no grid to scale up for).

    Returns
    -------
    FuncAnimation
        One frame per entry of the resolved schedule.

    Notes
    -----
    Axis ranges are the union over the panels so the view does not jump
    (:func:`_frames_axis_ranges`); ``share_color=True`` additionally unifies the
    colour range, since a heatmap whose colour *meaning* changes every frame
    cannot be compared with the frame before it.
    """
    from matplotlib.animation import FuncAnimation

    from . import _threed
    from ._core import _draw_2d_panel

    anim = spec.animation or Animation()
    panels = list(spec.panels)
    theme = _resolve_theme(spec)
    figsize, dpi, layout_engine = figure_geometry(spec, figsize, theme=theme)

    fig = new_figure(figsize, dpi, layout_engine)
    if theme.background is not None:
        fig.patch.set_facecolor(theme.background)

    layout = spec.layout
    ranges = _frames_axis_ranges(panels)
    shared_clim = _frames_clim(panels) if layout is not None and layout.share_color else None

    frame_seq = anim.head_indices(len(panels)) if panels else [0]

    def update(frame: int) -> Any:
        if not panels:
            return []
        panel = panels[frame_seq[min(frame, len(frame_seq) - 1)]]
        # Never mutate the caller's panel: a composite holds the SAME objects the
        # user built (documented), so the inherited theme (and the shared colour
        # scale) ride on a throwaway copy.
        effective = dataclasses.replace(panel, _theme=panel._theme or theme)
        if shared_clim is not None:
            effective = dataclasses.replace(effective, clim=shared_clim)
        fig.clear()
        three_d = _threed.is_three_d(effective)
        # ``Any`` because a 3-D axes carries ``set_zlim``, which the 2-D ``Axes``
        # stub does not declare (the same reason ``_apply_fixed_limits`` takes one).
        ax: Any = fig.add_subplot(1, 1, 1, projection="3d" if three_d else None)
        if three_d:
            _threed._draw_3d_panel(fig, ax, effective)
        else:
            _draw_2d_panel(fig, ax, effective)
        if "x" in ranges:
            ax.set_xlim(*ranges["x"])
        if "y" in ranges:
            ax.set_ylim(*ranges["y"])
        if three_d and "z" in ranges:
            ax.set_zlim(*ranges["z"])
        if spec.title:
            fig.suptitle(spec.title)
        return []

    # Draw the LAST frame eagerly.  A still save of an animation writes the
    # animation's underlying figure (``_core._write``), and ``Plot.fig`` hands that
    # same figure out — so a figure that stays empty until the frame loop runs
    # means ``movie.save("cascade.png")`` silently writes a blank page.  The final
    # frame is the right still for a movie, matching the spatial-field movie (whose
    # static layer data is the final field).
    update(len(frame_seq) - 1)

    interval = 1000.0 / float(anim.fps) if anim.fps > 0 else 50.0
    return FuncAnimation(
        fig,
        update,
        frames=len(frame_seq),
        interval=interval,
        blit=False,
        repeat=bool(anim.loop),
    )
