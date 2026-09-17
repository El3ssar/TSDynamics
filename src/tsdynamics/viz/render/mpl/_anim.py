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

**Writing a movie blits** (:class:`_FrameCompositor`).  The frame drivers have
always mutated artists rather than rebuilding them, but *writing* a frame still
cost a whole figure: matplotlib's :meth:`~matplotlib.animation.Animation.save`
draws once through ``draw_idle`` and throws that away, then draws again inside
``savefig`` — and each of those re-ran the constrained-layout solver and
re-measured every tick label.  Measured on a Lorenz comet at the defaults, that
was ~99 ms per frame, so a default 360-frame movie took ~36 s.  Three changes:
:meth:`_FastAnimation._post_draw` skips the discarded draw while saving,
:class:`_LayoutFreeze` converges the layout solver once and then drops it, and the
figure's ``draw`` is replaced for the duration of the save by a compositor that
restores a cached background and re-draws only the artists the drivers touch.
The static prefix — axes, spines, tick labels, colorbars, legends, annotations,
the faint full-curve backdrop — is rasterised **once**.  Counted rather than
timed: whole-figure draws went from **3.00 per frame**, at every movie length, to
a **constant 7** for the whole save.

The compositor is **validated, not assumed**: on its first call it renders probe
frames the unoptimised way and bit-compares them against the composited result,
and any difference disables it for that save (the movie is then drawn the slow,
faithful way).  It is not armed at all for a spinning 3-D camera or for a
``layout="frames"`` composite, both of which genuinely re-draw the axes — but a
spinning camera still gets the layout freeze, since an axes whose tick labels
change every frame is one constrained layout never converges on.

Every driver must be a **pure function of the frame index**.  The compositor
calibrates on scattered probe frames and then restores the frame the writer is
about to grab, so a driver that only moves forward would be calibrated against its
own stale state — and invisibly, since the reference draw sees the same stale
artists.  It is a user-facing property too: ``loop`` and ``pingpong`` replay frame
0 on every cycle.

The blitting is deliberately confined to *writing*.  ``FuncAnimation(blit=True)``
— matplotlib's own blitting, for on-screen playback — marks the animated artists
``set_animated(True)``, and an animated artist is skipped by any ordinary
``Figure.draw``; the same figure is what ``Plot.fig`` hands a caller and what a
still ``.png`` of a movie renders, so turning it on would empty the comet out of
both.  The compositor has no such side effect (it hides artists only inside its
own capture, and puts them back), and the frame cost it removes is the one a user
actually waits on.

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
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from ...spec import Animation, PlotKind, PlotSpec
from .. import normalize_kind
from ._core import (
    _apply_theme_color_cycle,
    _apply_theme_to_figure,
    _resolve_theme,
    apply_title,
    figure_geometry,
    new_figure,
)

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["render_animation"]

#: Truthy ⇒ never blit; every frame is drawn the unoptimised way.  The bypass that
#: proves WITH-compositor == WITHOUT-compositor, mirroring
#: ``TSDYNAMICS_NO_TAPE_CACHE`` / ``TSDYNAMICS_NO_JIT_CACHE``.
_NO_BLIT_ENV = "TSDYNAMICS_NO_BLIT"

#: How many frames the compositor bit-compares against an unoptimised draw before
#: it trusts itself.  One proves the artist partition; the extras are spread over
#: the movie so a comet that only overlaps a static artist late still fails fast.
_PROBE_FRAMES = 3

#: Cap on the draws :meth:`_FrameCompositor._settle_layout` spends converging the
#: layout engine.  Measured: a 2-D panel settles in 2, a 3-D one in 5.
_LAYOUT_SETTLE_DRAWS = 12


def _blitting_disabled() -> bool:
    """Whether :data:`_NO_BLIT_ENV` asks for the unoptimised frame path."""
    return os.environ.get(_NO_BLIT_ENV, "").strip().lower() not in ("", "0", "false", "no")


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


def _is_measuring(renderer: Any) -> bool:
    """Whether ``renderer`` is in matplotlib's *measure-only* mode.

    ``print_figure`` settles a layout engine by drawing the figure once through a
    renderer whose ``draw_*`` methods have been swapped for no-ops
    (:meth:`matplotlib.backend_bases.RendererBase._draw_disabled`, which installs
    them as **instance** attributes).  That call reaches ``Figure.draw`` — and so
    the compositor — with a renderer that paints nothing, which is neither a frame
    to composite nor a frame to calibrate against.  It must be handed straight to
    the real draw, or the "background" captured from it is empty and every frame
    of the movie comes out blank.
    """
    return "draw_path" in vars(renderer)


def _draw_foreground(renderer: Any, artists: list[Artist]) -> None:
    """Draw ``artists`` onto ``renderer`` in the order a full figure draw would.

    A 3-D collection (the fading comet's ``Line3DCollection``) carries only its
    *3-D* segments until :meth:`do_3d_projection` flattens them — a step
    :meth:`mpl_toolkits.mplot3d.axes3d.Axes3D.draw` performs and a direct
    ``artist.draw`` does not, so it is performed here for any artist that
    declares it.  ``Line3D`` projects inside its own ``draw`` and has no such
    method.

    The order is re-derived from the artists' **live** ``zorder`` on every call,
    not fixed when the plan was built, because a 3-D axes *rewrites* it:
    :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.draw` depth-sorts its collections
    and patches and assigns each a fresh ``zorder`` as it goes.  Measured on a
    ``.trail(fade=True)`` comet, whose tail is a ``Line3DCollection``: the plan
    was built with the collection at ``zorder`` 1 (below the head marker's 2), the
    first full draw moved it to 2.5 (above), and from then on the composite drew
    the head over the tail where a full draw drew the tail over the head — 9
    pixels, enough to fail calibration and lose the optimisation for every
    fading-comet movie.  Sorting here keeps the composite in step with whatever
    the last full draw decided; calibration still proves it.

    The sort is **within one axes**: a plan is grouped by axes (in figure order)
    and a composite's panels are drawn panel by panel, so re-ordering across that
    boundary would interleave two panels' artists — a difference no full draw
    makes.
    """
    groups: dict[int, int] = {}
    for artist in artists:
        groups.setdefault(id(artist.axes), len(groups))
    order = sorted(artists, key=lambda a: (groups[id(a.axes)], a.get_zorder()))
    for artist in order:
        project = getattr(artist, "do_3d_projection", None)
        if project is not None:
            project()
        artist.draw(renderer)


def _foreground_plans(fig: Figure, dynamic: list[Artist]) -> list[list[Artist]]:
    """Return the candidate per-frame redraw sets, cheapest first.

    Two plans, and the compositor takes the first whose frames come out
    byte-identical (and no blitting at all if neither does):

    **minimal** — only the artists the drivers mutate.  This is what makes a field
    movie cheap: its ``AxesImage`` sits at ``zorder=0``, *below* the tick labels,
    so anything more conservative drags the whole (text-heavy) axis into every
    frame — while the image is confined to the axes interior and the ticks are
    drawn outside it, so nothing above it ever covers a pixel it paints.

    **over-drawn tail** — the minimal set plus every artist a full draw paints
    *after* it, in that order, so an artist that genuinely overlaps a comet is
    composited in the right order.  This is what a small figure needs: measured at
    3x2 in, a Lorenz comet reaches within a pixel of the bottom spine (``zorder``
    2.5, i.e. above the curve) and the minimal plan differs on **3** pixels.

    The **titles** are excluded from the second plan, and that exclusion is the
    reason there is no third, fully-conservative one:
    :meth:`matplotlib.axes.Axes.draw` does more than draw its children in order —
    it calls ``_update_title_position`` first, so a title re-drawn outside that
    call sits where the *previous* draw left it.  Measured, a plan including the
    three title artists differs from a full draw on 306 pixels of the title row.
    The three are the only artists ``Axes.draw`` repositions that can land in the
    tail (the axis labels belong to the ``XAxis`` / ``YAxis``, at ``zorder`` 1.5,
    always in the cached background), so they are excluded **by identity** —
    :func:`_title_artists` — and not by type.  Excluding every ``Text`` instead, as
    this did, also threw out the one kind of text that genuinely lands in the tail
    and genuinely needs compositing: an **annotation**.  A comet crossing under a
    ``.vline(..., label=...)`` then had no plan that matched, so the whole movie
    fell back to full draws — silently, since the fallback is correct.

    Both claims are about geometry, which is why the compositor **measures** them
    on probe frames rather than trusting this docstring.

    Tie-broken exactly as :meth:`matplotlib.axes.Axes.draw` does: a stable sort by
    ``zorder`` over ``get_children()`` with the background patch removed.
    """
    dynamic_ids = {id(a) for a in dynamic}
    minimal: list[Artist] = []
    over_drawn: list[Artist] = []
    for ax in fig.axes:
        children = [c for c in ax.get_children() if c is not ax.patch]
        order = sorted(children, key=lambda a: a.get_zorder())
        first = next((k for k, a in enumerate(order) if id(a) in dynamic_ids), None)
        if first is None:
            continue
        titles = _title_artists(ax)
        tail = order[first:]
        minimal.extend(a for a in tail if id(a) in dynamic_ids)
        over_drawn.extend(a for a in tail if id(a) in dynamic_ids or id(a) not in titles)
    if not minimal:
        return []
    return [minimal, over_drawn] if len(over_drawn) > len(minimal) else [minimal]


def _title_artists(ax: Any) -> frozenset[int]:
    """Identities of the title texts :meth:`matplotlib.axes.Axes.draw` repositions.

    ``Axes.draw`` calls ``_update_title_position`` before drawing its children, so
    these three cannot be composited out of order.  Read defensively: the centre
    title is public, the left/right ones are not.
    """
    found = (getattr(ax, name, None) for name in ("title", "_left_title", "_right_title"))
    return frozenset(id(t) for t in found if t is not None)


def _agg_renderer(fig: Figure) -> Any:
    """Return ``fig``'s canvas renderer if it supports region capture / restore (Agg)."""
    get_renderer = getattr(fig.canvas, "get_renderer", None)
    if get_renderer is None:  # pragma: no cover - non-Agg canvas
        return None
    renderer = get_renderer()
    needed = ("copy_from_bbox", "restore_region", "buffer_rgba", "clear")
    if not all(hasattr(renderer, name) for name in needed):  # pragma: no cover - non-Agg
        return None
    return renderer


def _axes_geometry(fig: Figure) -> tuple[tuple[float, ...], ...]:
    """Return the axes rectangles — what a layout engine moves as it converges."""
    return tuple(tuple(ax.get_position().bounds) for ax in fig.axes)


class _LayoutFreeze:
    """Settle the figure's layout engine once, then drop it for the duration of a save.

    Constrained layout is **iterative and does not converge in one draw** — on a
    3-D axes it takes about five.  Measured on a Lorenz portrait, the axes
    rectangle walked ``(0.1479, 0.0087, 0.7042, 0.9389)`` →
    ``(0.1526, 0.0212, 0.6948, 0.9264)`` over the first five draws, so the opening
    frames of a 3-D movie were drawn at a *different* framing from the rest: the
    view visibly settles while the attractor is being revealed.

    A **spinning** camera never settles at all, because its tick-label extents
    change on every frame.  Measured on a 20-frame ``.camera(spin=90.0)`` movie
    before this class existed: **11 distinct axes rectangles**, moving on 14 of the
    19 frame transitions.  That is the library's own stated failure mode — *a movie
    whose axes rescale every frame shows you the axes moving, not the dynamics* —
    and it is also what made the spin path's output depend on how many times each
    frame happened to be drawn.  Iterating to a fixed point and then dropping the
    engine fixes both, so the freeze is applied to **every** single-figure
    animation, blitted or not.

    The exception is a ``layout="frames"`` composite, which calls ``fig.clear()``
    and re-adds an axes per frame: there is no one framing to settle on, and the
    per-frame panel genuinely needs its layout solved.

    Dropping the engine outright (rather than parking a ``PlaceHolderLayoutEngine``)
    is also what makes ``print_figure`` skip its own settling pre-pass, which
    re-measures every tick label once per frame.  ``set_layout_engine(None)`` can
    still resolve to ``"tight"`` when ``figure.autolayout`` is set in ``rcParams``,
    so the result is checked rather than assumed.
    """

    def __init__(self, fig: Figure) -> None:
        self._fig = fig
        self._engine = fig.get_layout_engine()
        self._done = False

    def prepare(self, draw: Any) -> None:
        """Converge the layout with ``draw``, then drop the engine (once per save).

        This has to happen **outside** ``print_figure``: that method masks the
        figure's layout engine with ``'none'`` for the duration of the write and
        restores it afterwards, so a freeze applied inside it is silently undone —
        and, worse, the per-frame settling pre-pass then keeps *moving the axes*
        under a background cached at the first frame's geometry.
        """
        if self._done:
            return
        self._done = True
        if self._fig.get_layout_engine() is None:
            return
        renderer = _agg_renderer(self._fig)
        previous = _axes_geometry(self._fig)
        for _ in range(_LAYOUT_SETTLE_DRAWS):
            if renderer is not None:
                renderer.clear()
                draw(renderer)
            else:  # pragma: no cover - non-Agg canvas
                self._fig.canvas.draw()
            current = _axes_geometry(self._fig)
            if current == previous:
                break
            previous = current
        self._fig.set_layout_engine(None)
        if self._fig.get_layout_engine() is not None:  # pragma: no cover - rcParams-driven
            self._fig.set_layout_engine("none")

    def restore(self) -> None:
        """Give the figure its layout engine back (the save is over)."""
        if self._engine is not None and self._fig.get_layout_engine() is not self._engine:
            self._fig.set_layout_engine(self._engine)
        self._done = False


class _FrameCompositor:
    """Draw a movie frame by restoring a cached background and blitting the rest.

    Installed as the figure's ``draw`` for the duration of a
    :meth:`_FastAnimation.save`.  ``FigureCanvasAgg.draw`` clears the renderer and
    then calls ``figure.draw(renderer)`` — looked up on the *instance* — so
    replacing that one attribute is enough to take over what every ``savefig``
    inside the writer loop rasterises, without touching matplotlib's writer
    negotiation.

    Correctness is **measured, not argued**: :meth:`_calibrate` renders probe
    frames through the untouched ``Figure.draw`` and bit-compares them with the
    composited result.  A single mismatched byte disables the compositor for the
    rest of the save and leaves a faithful full draw in the buffer.
    """

    def __init__(
        self, fig: Figure, dynamic: list[Artist], draw_frame: Any, probes: list[int]
    ) -> None:
        self._fig = fig
        self._draw_frame = draw_frame
        self._probes = probes
        #: The frame the writer is about to grab; restored after calibration.
        self.current_frame: Any = None
        self._plans = _foreground_plans(fig, dynamic)
        self._foreground: list[Artist] = self._plans[0] if self._plans else []
        self._real_draw = fig.draw
        self._freeze = _LayoutFreeze(fig)
        self._background: Any = None
        self._state: tuple[Any, ...] | None = None
        self._armed = False
        self._prepared = False
        self.enabled = bool(self._plans) and not _blitting_disabled()
        #: Set once a calibration has run; ``False`` afterwards means the probe
        #: frames did not match and every frame is being drawn in full.
        self.blitting = False

    # -- arming ------------------------------------------------------------
    def arm(self) -> None:
        """Take over the figure's ``draw`` (no-op when the compositor is off)."""
        if self.enabled and not self._armed:
            self._fig.draw = self._draw  # type: ignore[method-assign]
            self._armed = True

    def prepare(self) -> None:
        """Settle and drop the layout engine, once, before the first frame is written.

        Delegates to :class:`_LayoutFreeze` — the same freeze an *unblitted*
        animation gets — and additionally requires an Agg renderer, since without
        one there is no region to capture or restore.  Measured on a 3-D Lorenz
        comet, a freeze applied inside ``print_figure`` instead of here showed up as
        every one of 20 frames differing from the unoptimised render; hoisted out,
        all 20 are byte-identical.

        The writer has finished ``setup`` by the time this runs, so the figure
        size, dpi and canvas are the ones the movie will actually use.
        """
        if not self.enabled or self._prepared:
            return
        self._prepared = True
        if _agg_renderer(self._fig) is None:
            self.enabled = False  # not an Agg canvas: no region blitting available
            return
        self._freeze.prepare(self._real_draw)

    def disarm(self) -> None:
        """Restore the real ``Figure.draw`` and drop the cached background."""
        if self._armed:
            self._fig.__dict__.pop("draw", None)
            self._armed = False
        self._restore_layout()
        self._background = None
        self._state = None
        self._prepared = False

    # -- the hook ----------------------------------------------------------
    def _draw(self, renderer: Any) -> None:
        """Stand in for :meth:`matplotlib.figure.Figure.draw` during a save."""
        if not self.enabled or _is_measuring(renderer):
            self._real_draw(renderer)
            return
        state = (renderer.width, renderer.height, tuple(self._fig.get_facecolor()))
        if self._background is None or state != self._state:
            # The writer settles size / dpi / facecolor in ``setup`` and in each
            # ``savefig``; calibrate against whatever it has settled on, and
            # re-calibrate if it ever moves.
            if not self._calibrate(renderer, state):
                self.enabled = False
                self.blitting = False
                return  # _calibrate left an honest full draw in the buffer
            self.blitting = True
        renderer.restore_region(self._background)
        _draw_foreground(renderer, self._foreground)
        self._fig.stale = False

    # -- calibration -------------------------------------------------------
    def _reference(self, renderer: Any) -> bytes:
        """Full-draw the current frame the unoptimised way and return its pixels."""
        renderer.clear()
        self._real_draw(renderer)
        return bytes(renderer.buffer_rgba())

    def _capture(self, renderer: Any) -> Any:
        """Rasterise the static prefix (the foreground hidden) into a region."""
        visible = [a.get_visible() for a in self._foreground]
        for artist in self._foreground:
            artist.set_visible(False)
        try:
            renderer.clear()
            self._real_draw(renderer)
            return renderer.copy_from_bbox(self._fig.bbox)
        finally:
            for artist, was in zip(self._foreground, visible, strict=True):
                artist.set_visible(was)

    def _compose(self, renderer: Any, background: Any) -> bytes:
        """Restore ``background``, draw the foreground, return the pixels."""
        renderer.clear()
        renderer.restore_region(background)
        _draw_foreground(renderer, self._foreground)
        return bytes(renderer.buffer_rgba())

    def _restore_layout(self) -> None:
        """Give the figure its layout engine back (the save is over)."""
        self._freeze.restore()

    def _matches_on_probes(self, renderer: Any, background: Any) -> bool:
        """Whether the composite equals a full draw on the current + probe frames."""
        if self._compose(renderer, background) != self._reference(renderer):
            return False
        for frame in self._probes:
            self._draw_frame(frame)
            if self._compose(renderer, background) != self._reference(renderer):
                return False
        return True

    def _calibrate(self, renderer: Any, state: tuple[Any, ...]) -> bool:
        """Prove the composite is byte-identical on probe frames, or give up.

        Returns ``True`` with :attr:`_background` cached and the current frame
        composited into ``renderer``; ``False`` with a faithful full draw in
        ``renderer`` and the compositor to be switched off.
        """
        for plan in self._plans:
            self._foreground = plan
            self._restore_current_frame()
            background = self._capture(renderer)
            if self._matches_on_probes(renderer, background):
                self._restore_current_frame()
                self._background = background
                self._state = state
                renderer.clear()
                renderer.restore_region(background)
                _draw_foreground(renderer, self._foreground)
                return True
            del background
        self._restore_layout()
        self._restore_current_frame()
        renderer.clear()
        self._real_draw(renderer)
        return False

    def _restore_current_frame(self) -> None:
        """Put the artists back on the frame the writer is about to grab."""
        if self.current_frame is not None:
            self._draw_frame(self.current_frame)


def _spin_anywhere(spec: PlotSpec) -> bool:
    """Whether any panel spins its 3-D camera (which re-draws the axes itself)."""
    from . import _threed

    panels = spec.panels if spec.is_composite else [spec]
    for panel in panels:
        anim = panel.animation or spec.animation
        if anim is not None and anim.spin and _threed.is_three_d(panel):
            return True
    return False


class _FastAnimation:
    """Mixin for :class:`~matplotlib.animation.FuncAnimation` that makes saving cheap.

    Three savings, all confined to :meth:`save`:

    ``_post_draw``
        matplotlib's save loop draws each frame through ``draw_idle`` and then
        *throws it away*, because ``writer.grab_frame`` immediately re-draws
        inside ``savefig``.  While saving, the first of the pair is skipped.

    :class:`_LayoutFreeze`
        the constrained-layout solver is converged once and then dropped, so
        ``print_figure`` stops re-measuring every tick label per frame — and the
        movie's framing stops moving under the dynamics.

    :class:`_FrameCompositor`
        the second draw is served from a cached background plus the handful of
        artists the frame drivers actually mutate.

    The freeze and the compositor are wired **separately**, because they are not
    eligible together: a spinning 3-D camera cannot be blitted (it re-draws the
    axes) but is exactly the case that most needs a fixed framing.  Only a
    ``layout="frames"`` composite gets neither, since it rebuilds the figure per
    frame and so has no single layout to settle on.

    ``blit=`` on the :class:`FuncAnimation` constructor stays ``False`` — see the
    module docstring: matplotlib's own blitting marks the moving artists
    ``set_animated(True)``, which removes them from every *static* render of the
    same figure (``Plot.fig``, a still ``.png`` of a movie).

    The class is built lazily by :func:`_fast_animation_class` so importing this
    module never imports matplotlib.
    """

    _tsd_saving: bool = False
    _tsd_compositor: _FrameCompositor | None = None
    _tsd_freeze: _LayoutFreeze | None = None

    def _post_draw(self, framedata: Any, blit: bool) -> None:
        if self._tsd_saving and not blit:
            return  # the writer's own savefig is the draw that reaches the file
        super()._post_draw(framedata, blit)  # type: ignore[misc]

    def _draw_next_frame(self, framedata: Any, blit: bool) -> None:
        compositor = self._tsd_compositor
        if self._tsd_saving:
            # Runs once, between ``writer.setup`` (so the figure size is final)
            # and the first ``savefig`` (so the layout freeze is not masked).
            if compositor is not None:
                compositor.prepare()
                compositor.current_frame = framedata
            elif self._tsd_freeze is not None:
                self._tsd_freeze.prepare(self._fig.draw)  # type: ignore[attr-defined]
        super()._draw_next_frame(framedata, blit)  # type: ignore[misc]

    def save(self, *args: Any, **kwargs: Any) -> Any:
        """Write the movie, blitting the frames when that is provably identical."""
        compositor = self._tsd_compositor
        self._tsd_saving = True
        if compositor is not None:
            compositor.arm()
        try:
            return super().save(*args, **kwargs)  # type: ignore[misc]
        finally:
            self._tsd_saving = False
            if compositor is not None:
                compositor.disarm()
            elif self._tsd_freeze is not None:
                self._tsd_freeze.restore()


def _fast_animation_class() -> type:
    """Return ``_FastAnimation`` mixed into :class:`FuncAnimation` (built once)."""
    from matplotlib.animation import FuncAnimation

    cached = getattr(_fast_animation_class, "_cls", None)
    if cached is None:
        cached = type("FastFuncAnimation", (_FastAnimation, FuncAnimation), {})
        _fast_animation_class._cls = cached  # type: ignore[attr-defined]
    return cached


def _make_animation(
    fig: Figure,
    update: Any,
    *,
    n_steps: int,
    anim: Animation,
    dynamic: list[Artist] | None,
    draw_frame: Any = None,
    freeze_layout: bool = True,
) -> FuncAnimation:
    """Build the :class:`FuncAnimation`, wiring the frame compositor when eligible.

    ``dynamic`` is ``None`` for a frame model that genuinely re-draws the axes (a
    spinning 3-D camera, a ``layout="frames"`` composite): those keep the plain
    full-draw frame path, and only the discarded ``draw_idle`` is skipped.

    ``freeze_layout`` is the other axis, and the two are independent: a spinning
    camera is un-blittable but still wants its framing pinned once
    (:class:`_LayoutFreeze`).  It is ``False`` only for a ``layout="frames"``
    composite, which clears the figure and re-adds an axes per frame.
    """
    interval = 1000.0 / float(anim.fps) if anim.fps > 0 else 50.0
    animation = _fast_animation_class()(
        fig,
        update,
        frames=n_steps,
        interval=interval,
        blit=False,
        repeat=bool(anim.loop),
    )
    if dynamic:
        probe = draw_frame if draw_frame is not None else animation._draw_frame
        animation._tsd_compositor = _FrameCompositor(fig, dynamic, probe, _probe_schedule(n_steps))
    elif freeze_layout:
        animation._tsd_freeze = _LayoutFreeze(fig)
    return animation  # type: ignore[no-any-return]


def _probe_schedule(n_steps: int) -> list[int]:
    """Frames (besides the current one) the compositor bit-compares.

    The **last** frame always, since a reveal comet is longest there and so most
    likely to reach an artist it does not overlap early on; the rest spread
    evenly, so a disagreement anywhere in the movie has a chance to be caught
    before a single frame is written.
    """
    if n_steps <= 1:
        return []
    last = n_steps - 1
    count = min(_PROBE_FRAMES, n_steps) - 1
    spread = [int(round((k + 1) * last / (count + 1))) for k in range(count)]
    return sorted({*spread, last})


def render_animation(
    spec: PlotSpec, *, figsize: tuple[float, float] | None = None, **_kw: Any
) -> FuncAnimation:
    """Render an animated :class:`PlotSpec` to a :class:`FuncAnimation`.

    Single-panel specs animate their curve / image layers in reveal mode;
    composite specs animate every panel in lockstep on one shared frame clock —
    unless their layout mode is ``"frames"``, in which case the panels *are* the
    frames and are played one after another (:func:`_render_frames_movie`).
    """
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
    return _make_animation(
        fig,
        updater.update,
        n_steps=updater.n_steps,
        anim=anim,
        dynamic=None if _spin_anywhere(spec) else updater.artists,
    )


class _PanelUpdater:
    """Per-frame update closure for one panel (curves + head + camera + clock)."""

    def __init__(self, frame_seq: list[int], update_fn: Any, artists: list[Artist]) -> None:
        self._frame_seq = frame_seq
        self._update_fn = update_fn
        self.n_steps = len(frame_seq)
        #: The artists this panel's drivers mutate — the frame compositor's
        #: foreground.  Empty means nothing on this panel moves.
        self.artists = artists

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

    built = [
        _make_layer_driver(ax, layer, spec, anim, three_d=three_d, n_clock=n_samples)
        for layer in spec.layers
    ]
    drivers = [d.drive for d in built if d is not None]
    artists: list[Artist] = [a for d in built if d is not None for a in d.artists]

    _apply_animated_annotations(ax, spec, three_d=three_d)

    base_azim = _base_azim(spec) if three_d else None
    clock_artist = _make_clock(ax, spec, three_d=three_d) if anim.clock else None
    clock_times = _times(spec, _first_x(spec), n_samples) if anim.clock else None
    if clock_artist is not None:
        artists.append(clock_artist)

    def update(i: int) -> Any:
        for driver in drivers:
            driver(i, tail)
        if three_d and base_azim is not None and anim.spin:
            frac = i / max(1, n_samples - 1)
            ax.view_init(elev=_base_elev(spec), azim=base_azim + 360.0 * anim.spin * frac)
        if clock_artist is not None and clock_times is not None:
            clock_artist.set_text(anim.clock_format.format(t=float(clock_times[i])))
        return []

    return _PanelUpdater(head_idx, update, artists)


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

        ts.plot(traj, "spatial_field", animate=True)    mode='frames'   101 frames
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


@dataclasses.dataclass(frozen=True)
class _LayerDriver:
    """One layer's per-frame updater, **and the artists it mutates**.

    The artists are what makes the frame compositor possible: a driver that
    reports them lets the renderer cache everything else as one raster instead of
    re-rasterising the whole figure per frame.  A driver that mutates an artist
    and does not list it would have that artist frozen at its first frame, so the
    two are declared together, at the same site.
    """

    drive: Any
    artists: tuple[Artist, ...]


def _make_layer_driver(
    ax: Any, layer: Any, spec: PlotSpec, anim: Animation, *, three_d: bool, n_clock: int = 0
) -> _LayerDriver | None:
    """Build one layer's :class:`_LayerDriver` (or ``None`` when nothing animates).

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
) -> _LayerDriver:
    """Reveal a curve as a trail + a head marker (point, or sweep line for series)."""
    x = np.asarray(layer.data["x"], dtype=float)
    y = np.asarray(layer.data["y"], dtype=float)
    z = np.asarray(layer.data["z"], dtype=float) if "z" in layer.data else None
    color = layer.style.get("color")
    lw = layer.style.get("lw", layer.style.get("linewidth", 2.0))

    # Head style: a point on a portrait; a vertical sweep line on a time series.
    series_like = kind in (PlotKind.TIME_SERIES, PlotKind.SPACETIME)

    # With a windowed trail, draw the full curve once, faintly — the static,
    # context "attractor backdrop" the comet sweeps over (matches the plotly
    # look).  Switchable since v6 (``.trail(backdrop=False)``): it shows the
    # ending in frame 1, which is exactly wrong for a talk that reveals an
    # attractor, and it used to have no knob and no mention in ``.trail``'s help.
    if anim.trail_kind is not None and not series_like and anim.backdrop:
        alpha = float(anim.backdrop_alpha)
        if alpha > 0.0:
            if three_d:
                assert z is not None
                ax.plot(x, y, z, color=color, lw=1.0, alpha=alpha)
            else:
                ax.plot(x, y, color=color, lw=1.0, alpha=alpha)

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

    moving: tuple[Artist, ...] = (line, head) if vline is None else (line, head, vline)
    return _LayerDriver(drive, moving)


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
) -> _LayerDriver:
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
        else:
            # A frame with no segment yet (frame 0) must CLEAR the collection, not
            # leave it holding whatever the previous frame drew.  Every other driver
            # is a pure function of the frame index; this one was not, so frame 0
            # replayed after frame N showed frame N's comet — which is what a
            # ``pingpong`` loop does on every cycle, and what the frame compositor
            # does when it restores the frame it calibrated on.
            lc.set_segments([])
        if anim.head:
            _set_head(head, x, y, z, i, three_d)

    return _LayerDriver(drive, (lc, head))


def _image_sweep_driver(ax: Axes, layer: Any, spec: PlotSpec) -> _LayerDriver:
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

    return _LayerDriver(drive, (line,))


def _field_movie_driver(ax: Axes, layer: Any, spec: PlotSpec) -> _LayerDriver:
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
) -> _LayerDriver:
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

    return _LayerDriver(drive, (im,))


def _field_movie_1d(
    ax: Axes,
    layer: Any,
    spec: PlotSpec,
    stack: np.ndarray,
    ylim: tuple[float, float],
    n_steps: int,
) -> _LayerDriver:
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

    return _LayerDriver(drive, (line,))


def _apply_animated_annotations(ax: Any, spec: PlotSpec, *, three_d: bool) -> None:
    """Draw ``spec.annotations`` onto an animated panel.

    The reveal renderer built its own axes from scratch and **never called either
    annotation applier**, so every ``.vline()`` / ``.hline()`` / ``.span()`` /
    ``.text()`` on an animated plot was silently dropped — measured, a spec
    carrying three annotations rendered a figure with ``ax.texts == []``, while the
    *static* render of the same spec drew all three.  A reference line is the one
    thing a movie most needs (the threshold the orbit is about to cross), and
    marking it did nothing and said nothing.

    The annotations are created once, here, and never mutated afterwards, so they
    land in the frame compositor's cached background and cost nothing per frame.
    They are applied **after** the layers and after
    :func:`_apply_fixed_limits`, which is what stops an annotation from rescaling
    the view (and is what the 3-D applier needs, since it reads the limits to place
    a reference line).
    """
    if not spec.annotations:
        return
    if three_d:
        from ._threed import _apply_3d_annotations

        _apply_3d_annotations(ax, spec, _resolve_theme(spec))
        return
    from ._core import _apply_annotations

    _apply_annotations(ax, spec.annotations)


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


def _apply_suptitle(fig: Any, spec: PlotSpec, theme: Any) -> None:
    """Set a composite's figure title **in the theme's ink** (the still path does).

    ``fig.suptitle(spec.title)`` alone drew near-black text on a dark movie while
    the same spec's ``.png`` drew it light: a composite's title is figure-level,
    so ``_apply_theme_to_figure`` (which colours the *axes* title) never saw it.
    """
    if not spec.title:
        return
    kw: dict[str, Any] = {}
    if theme is not None and theme.foreground is not None:
        kw["color"] = theme.foreground
    if theme is not None and theme.font_family is not None:
        kw["fontfamily"] = theme.font_family
    fig.suptitle(spec.title, **kw)


def _apply_static_labels(ax: Any, spec: PlotSpec, *, three_d: bool) -> None:
    """Apply axis labels, title, aspect (the non-data framing).

    The title goes through the shared :func:`~._core.apply_title`, so an animated
    frame is themed exactly like the still of the same spec.
    """
    if spec.x.label:
        ax.set_xlabel(spec.x.label)
    if spec.y.label:
        ax.set_ylabel(spec.y.label)
    if three_d and spec.z is not None and spec.z.label:
        ax.set_zlabel(spec.z.label)
    theme = _resolve_theme(spec)
    apply_title(ax, spec, theme)
    if three_d:
        # ``mplot3d`` draws its own pane quads, which keep matplotlib's light
        # default whatever the figure facecolor is.  The still renderer themes
        # them; the animator did not, so a dark 3-D movie had **grey panes**
        # where its own ``.png`` had dark navy ones.
        from ._threed import _apply_3d_panes

        _apply_3d_panes(ax, theme)
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


#: The knobs that make up a composite's **master clock** — the timeline every panel
#: is played on.  Everything else on an :class:`~tsdynamics.viz.spec.Animation` is
#: per-panel *look* (head, trail, spin, clock) and stays the panel's own.
_MASTER_CLOCK_KNOBS = ("fps", "duration", "n_frames", "loop", "pingpong")


def _lockstep(panel_anim: Animation | None, master: Animation) -> Animation:
    """Put one panel on the composite's timeline, keeping its own look.

    A composite plays its panels "in lockstep on one master clock", and the timing
    half of that has to be taken from the composite — because composing with
    ``animate=`` stamps a default :class:`~tsdynamics.viz.spec.Animation` onto every
    panel that carries none, and a plain ``panel.animation or master`` then lets that
    stamp outrank the composite's own.  Measured::

        c = ts.viz.plot(a, b, layout="row", animate=True).animate(fps=30, duration=2.0)
        c.animation.frame_count(2001)   # 60 — what was asked for
        len(list(render_animation(c).new_frame_seq()))   # 360 — what was written

    Six seconds of movie for a two-second request, silently.  The panels' stamped
    defaults still own ``head`` / ``trail`` / ``spin`` / ``clock``, which is the
    whole reason they exist (a time-series panel gets no head marker, its portrait
    neighbour does).
    """
    if panel_anim is None:
        return master
    return dataclasses.replace(
        panel_anim, **{knob: getattr(master, knob) for knob in _MASTER_CLOCK_KNOBS}
    )


def _render_composite_animation(
    spec: PlotSpec, *, figsize: tuple[float, float] | None
) -> FuncAnimation:
    """Animate a composite: tile the panels and advance them all on one clock."""
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
            animation=_lockstep(panel.animation, anim),
            _theme=panel._theme or composite_theme,
        )
        three_d = _threed.is_three_d(effective)
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d" if three_d else None)
        updaters.append(_build_panel_animation(fig, ax, effective, three_d=three_d))
    _apply_suptitle(fig, spec, composite_theme)

    n_steps = max((u.n_steps for u in updaters), default=2)

    def update(frame: int) -> Any:
        for u in updaters:
            u.update(min(frame, u.n_steps - 1))  # clamp shorter panels to their last frame
        return []

    dynamic = [a for u in updaters for a in u.artists]
    return _make_animation(
        fig,
        update,
        n_steps=n_steps,
        anim=anim,
        dynamic=None if _spin_anywhere(spec) else dynamic,
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
        _apply_suptitle(fig, spec, _resolve_theme(spec))
        return []

    # Draw the LAST frame eagerly.  A still save of an animation writes the
    # animation's underlying figure (``_core._write``), and ``Plot.fig`` hands that
    # same figure out — so a figure that stays empty until the frame loop runs
    # means ``movie.save("cascade.png")`` silently writes a blank page.  The final
    # frame is the right still for a movie, matching the spatial-field movie (whose
    # static layer data is the final field).
    update(len(frame_seq) - 1)

    # ``dynamic=None`` and ``freeze_layout=False``: this mode genuinely re-draws,
    # because each frame IS a different panel (``fig.clear()`` above).  There is no
    # static prefix to cache, and no single framing to settle on either — the axes
    # is created afresh per frame and needs the layout solver.  So it keeps the
    # plain frame path, gaining only the skipped ``draw_idle``.
    return _make_animation(
        fig, update, n_steps=len(frame_seq), anim=anim, dynamic=None, freeze_layout=False
    )
