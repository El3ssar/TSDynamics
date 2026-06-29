"""Backend-agnostic plot intermediate representation (the viz seam).

This module defines the **declarative, JSON-serializable** intermediate
representation (IR) that lets a future multi-backend visualization suite
(matplotlib / Plotly / GPU / web) be added *by construction*, with zero churn to
the result types.  It ships **no rendering backend** — there is no matplotlib,
Plotly, or any plot import anywhere in this file, and ``import tsdynamics`` pulls
in no plot library because of it.

The pieces
----------
- :class:`PlotKind` — the *closed* enum of semantic plot kinds the contract owns
  up front (time series, phase portrait, bifurcation, recurrence, …) and the
  low-level layer marks (line, scatter, image, …).  Adding a *renderer* never
  needs a new kind; adding a kind is a deliberate, reviewed contract change.
- :class:`Axis` — a typed per-dimension axis (label / scale / limits / ticks /
  tickformat).
- :class:`Annotation` — a reference line or text overlay (e.g. the logistic
  onset ``r1 = 3``).
- :class:`Colorbar` — a typed description of the color legend for a scalar /
  image color channel (label / location / ticks / tickformat / visibility).
- :class:`Legend` — a typed description of the per-layer legend
  (visibility / location / title).
- :class:`Layer` — one drawable layer: a :class:`PlotKind` *mark* + a
  channel-name → :class:`numpy.ndarray` data mapping + neutral style keys.
- :class:`PlotSpec` — the top-level spec: a semantic :class:`PlotKind`, a list of
  :class:`Layer`, typed ``x`` / ``y`` / optional ``z`` axes, an optional
  ``clim`` color range, an optional :class:`Colorbar` and :class:`Legend`,
  title, ndim, aspect, annotations, and provenance ``meta``.  Tweak methods
  (:meth:`~PlotSpec.relabel`, :meth:`~PlotSpec.rescale`, :meth:`~PlotSpec.limits`,
  :meth:`~PlotSpec.ticks`, :meth:`~PlotSpec.style`) **mutate the spec and return
  it** so they chain, and because they touch the spec — not a renderer — a tweak
  like ``rescale(x="log")`` is identical across every backend.
  :meth:`~PlotSpec.to_dict` / :meth:`~PlotSpec.from_dict` round-trip the whole
  spec (NumPy arrays ↔ nested lists) so a computed spec can be cached, shipped to
  a web frontend, or replotted without recomputation.
- :class:`Plottable` — a tiny mixin that gives any object defining
  ``to_plot_spec()`` a ``.plot(...)`` convenience and a notebook display hook.

Design notes
------------
The grammar is synthesized from the data + encoding + mark model of declarative
visualization grammars [1]_, made numeric-array-native (TSDynamics ships NumPy
arrays, not tidy frames).  An *animation* is **not** a separate type: it is a
:class:`PlotSpec` whose layer data carries a leading ``frame`` axis plus
``meta["animate"]``; a backend that cannot animate renders the final frame.

References
----------
.. [1] Satyanarayan, A., Moritz, D., Wongsuphasawat, K. & Heer, J. (2017).
   "Vega-Lite: A Grammar of Interactive Graphics." *IEEE Transactions on
   Visualization and Computer Graphics*, 23(1), 341-350.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Literal

import numpy as np

from .style import Theme, get_theme, normalize_style

__all__ = [
    "Animation",
    "Annotation",
    "Axis",
    "Colorbar",
    "Layer",
    "Layout",
    "Legend",
    "PlotKind",
    "PlotSpec",
    "Plottable",
]


# ---------------------------------------------------------------------------
# Plot kinds (closed enum)
# ---------------------------------------------------------------------------


class PlotKind(StrEnum):
    """The closed vocabulary of plot kinds — both semantic kinds and layer marks.

    ``PlotKind`` is a :class:`~enum.StrEnum`, so a member compares equal to its value
    (``PlotKind.LINE == "line"``) and serializes to a plain string in
    :meth:`PlotSpec.to_dict`.  Two roles share the one enum:

    - **Semantic kinds** name what a whole :class:`PlotSpec` *means*
      (``TIME_SERIES``, ``PHASE_PORTRAIT_2D``, ``BIFURCATION``, …).  A renderer
      dispatches on the spec's :attr:`PlotSpec.kind`.
    - **Marks** name how a single :class:`Layer` is drawn (``LINE``,
      ``SCATTER``, ``IMAGE``, …).  A renderer dispatches on each
      :attr:`Layer.kind`.

    Keeping both in one closed enum means adding a *renderer* never needs a new
    kind, while adding a kind is a deliberate, reviewed contract change.
    """

    # ── semantic spec kinds ───────────────────────────────────────────────
    TIME_SERIES = "time_series"
    PHASE_PORTRAIT_2D = "phase_portrait_2d"
    PHASE_PORTRAIT_3D = "phase_portrait_3d"
    SPACETIME = "spacetime"
    # a spatial **field** at one instant: the state of a spatially-extended
    # system (a method-of-lines PDE) reshaped to its grid.  One kind covers both
    # spatial dimensionalities — a 1-D field is a line (a profile), a 2-D field
    # an ``IMAGE`` heatmap; an animation plays the field over time (a travelling
    # wave / an evolving 2-D field movie).  See stream VIZ-SPATIAL-FIELD.
    SPATIAL_FIELD = "spatial_field"
    # a multi-panel figure: an arrangement of sub-:class:`PlotSpec` ``panels``
    # under a :class:`Layout` (the composition seam — ``tsdynamics.viz.plot``)
    COMPOSITE = "composite"
    BIFURCATION = "bifurcation"
    ORBIT_DIAGRAM = "orbit_diagram"
    COBWEB = "cobweb"
    RETURN_MAP = "return_map"
    POINCARE_SECTION = "poincare_section"
    BASINS_IMAGE = "basins_image"
    RECURRENCE_PLOT = "recurrence_plot"
    POWER_SPECTRUM = "power_spectrum"
    SPECTROGRAM = "spectrogram"
    SCALING_FIT = "scaling_fit"
    DIMENSION_SPECTRUM = "dimension_spectrum"
    DIAGNOSTIC_CURVE = "diagnostic_curve"
    COMPLEXITY_CURVE = "complexity_curve"
    LINE_FAMILY = "line_family"
    ENSEMBLE_FAN = "ensemble_fan"
    HISTOGRAM_NULL = "histogram_null"
    LYAPUNOV_SPECTRUM = "lyapunov_spectrum"
    EIGENVALUE_PLANE = "eigenvalue_plane"
    FIXED_POINTS_OVERLAY = "fixed_points_overlay"
    VECTOR_FIELD = "vector_field"
    PHASE_PORTRAIT_FIELD = "phase_portrait_field"
    CONTINUATION = "continuation"
    CATEGORICAL_BAR = "categorical_bar"
    FEATURE_BARS = "feature_bars"
    # animation kinds are the same spec + a leading ``frame`` axis on the data
    # (animation rendering is deferred; a non-animating backend draws the final
    # frame — these members stay in the frozen vocabulary so such a spec still
    # round-trips, even though no shipped backend animates):
    TRAJECTORY_ANIMATION = "trajectory_animation"
    ENSEMBLE_ANIMATION = "ensemble_animation"

    # ── layer marks ───────────────────────────────────────────────────────
    LINE = "line"
    LINE3D = "line3d"
    SCATTER = "scatter"
    MARKERS = "markers"
    IMAGE = "image"
    QUIVER = "quiver"
    SURFACE3D = "surface3d"
    HISTOGRAM = "histogram"
    BAR = "bar"
    AREA = "area"
    ERRORBAR = "errorbar"

    # ── governance: the closed vocabulary partitions into semantic kinds +
    #    layer marks (frozen — adding/removing a member is a reviewed contract
    #    change gated by tests/test_viz_vocab.py). ──────────────────────────
    @classmethod
    def semantic_kinds(cls) -> frozenset[PlotKind]:
        """Return the closed set of *semantic* kinds a :class:`PlotSpec` can be.

        These name what a plot *means* (a renderer dispatches on
        :attr:`PlotSpec.kind`).  Disjoint from :meth:`layer_marks`; together they
        exhaust the enum (frozen by ``tests/test_viz_vocab.py``).
        """
        return _SEMANTIC_KINDS

    @classmethod
    def layer_marks(cls) -> frozenset[PlotKind]:
        """Return the closed set of layer *marks* a single :class:`Layer` draws.

        These name *how* one layer is drawn (a renderer dispatches on
        :attr:`Layer.kind`).  Disjoint from :meth:`semantic_kinds`.
        """
        return _LAYER_MARKS

    @classmethod
    def is_semantic(cls, kind: PlotKind | str) -> bool:
        """Whether ``kind`` is a semantic spec kind (vs. a layer mark)."""
        return PlotKind(kind) in _SEMANTIC_KINDS

    @classmethod
    def is_mark(cls, kind: PlotKind | str) -> bool:
        """Whether ``kind`` is a layer mark (vs. a semantic spec kind)."""
        return PlotKind(kind) in _LAYER_MARKS


#: The frozen set of **layer marks** — how a single :class:`Layer` is drawn.
#: A renderer maps each of these to a drawing primitive.  Closed: extending it is
#: a reviewed contract change (the membership guard in ``tests/test_viz_vocab.py``
#: pins the exact set).
_LAYER_MARKS: frozenset[PlotKind] = frozenset(
    {
        PlotKind.LINE,
        PlotKind.LINE3D,
        PlotKind.SCATTER,
        PlotKind.MARKERS,
        PlotKind.IMAGE,
        PlotKind.QUIVER,
        PlotKind.SURFACE3D,
        PlotKind.HISTOGRAM,
        PlotKind.BAR,
        PlotKind.AREA,
        PlotKind.ERRORBAR,
    }
)

#: The frozen set of **semantic kinds** — what a whole :class:`PlotSpec` means.
#: Every enum member that is not a layer mark; the two sets partition the enum.
_SEMANTIC_KINDS: frozenset[PlotKind] = frozenset(set(PlotKind) - _LAYER_MARKS)


# Type aliases for the public tweak API (one spelling each).  ``"categorical"``
# joins the numeric scales for a categorical axis (a CATEGORICAL_BAR / FEATURE_BARS
# x-axis whose tick positions index :attr:`Axis.categories`).
_Scale = Literal["linear", "log", "symlog", "categorical"]
# A colour *norm* is numeric only (categorical colour is :attr:`Colorbar.discrete`).
_Norm = Literal["linear", "log", "symlog"]
_Aspect = Literal["auto", "equal"]
_Ndim = Literal[1, 2, 3]
_CbarLoc = Literal["right", "left", "top", "bottom"]
_LegendLoc = Literal[
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]


def _as_pair(value: Any) -> tuple[float, float] | None:
    """Coerce a 2-sequence to a ``(float, float)`` tuple, or pass ``None`` through."""
    if value is None:
        return None
    lo, hi = value
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Axis
# ---------------------------------------------------------------------------


@dataclass
class Axis:
    r"""A typed description of one plotting axis.

    Parameters
    ----------
    label : str, optional
        Axis label.  May carry LaTeX (e.g. ``r"$\log r$"``); renderers decide
        how to typeset it.
    scale : {"linear", "log", "symlog", "categorical"}, optional
        The axis scale.  Default ``"linear"``.  ``"categorical"`` marks an axis
        whose integer tick positions index :attr:`categories` (a
        ``CATEGORICAL_BAR`` / ``FEATURE_BARS`` category axis).
    limits : tuple of float, optional
        ``(lo, hi)`` view limits, or ``None`` to auto-scale.
    ticks : sequence of float, optional
        Explicit tick locations, or ``None`` to auto-tick.
    tickformat : str, optional
        A backend-neutral format string for tick labels, or ``None``.
    categories : sequence of str, optional
        Category labels for a ``"categorical"`` axis — the tick label at integer
        position ``i`` is ``categories[i]`` (basin ids, feature names, …).
        ``None`` for a numeric axis.
    grid : bool, optional
        Whether to draw gridlines along this axis.  ``None`` (default) defers to
        the theme's ``grid`` default; ``True`` / ``False`` force it.
    color : str, optional
        The axis ink color (spine / ticks / label), or ``None`` to defer to the
        theme's ``foreground``.
    label_size : float, optional
        Font size for the axis label, or ``None`` to defer to the theme.
    tick_size : float, optional
        Font size for the tick labels, or ``None`` to defer to the theme.
    tick_rotation : float, optional
        Rotation (degrees) of the tick labels, or ``None`` for no rotation.
    """

    label: str = ""
    scale: _Scale = "linear"
    limits: tuple[float, float] | None = None
    ticks: Sequence[float] | None = None
    tickformat: str | None = None
    categories: Sequence[str] | None = None
    grid: bool | None = None
    color: str | None = None
    label_size: float | None = None
    tick_size: float | None = None
    tick_rotation: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this axis."""
        return {
            "label": self.label,
            "scale": self.scale,
            "limits": list(self.limits) if self.limits is not None else None,
            "ticks": [float(t) for t in self.ticks] if self.ticks is not None else None,
            "tickformat": self.tickformat,
            "categories": list(self.categories) if self.categories is not None else None,
            "grid": self.grid,
            "color": self.color,
            "label_size": None if self.label_size is None else float(self.label_size),
            "tick_size": None if self.tick_size is None else float(self.tick_size),
            "tick_rotation": None if self.tick_rotation is None else float(self.tick_rotation),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Axis:
        """Rebuild an :class:`Axis` from :meth:`to_dict` output."""
        limits = d.get("limits")
        ticks = d.get("ticks")
        categories = d.get("categories")
        return cls(
            label=d.get("label", ""),
            scale=d.get("scale", "linear"),
            limits=tuple(limits) if limits is not None else None,
            ticks=list(ticks) if ticks is not None else None,
            tickformat=d.get("tickformat"),
            categories=list(categories) if categories is not None else None,
            grid=d.get("grid"),
            color=d.get("color"),
            label_size=d.get("label_size"),
            tick_size=d.get("tick_size"),
            tick_rotation=d.get("tick_rotation"),
        )


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


@dataclass
class Annotation:
    """A reference line or text overlay on a spec.

    Annotations carry semantic markers that are part of the *result*, not the
    styling — e.g. the logistic period-doubling onsets ``r1 = 3`` /
    ``r2 = 1 + sqrt(6)`` on a bifurcation diagram, or a fit-region shading.

    Parameters
    ----------
    kind : {"vline", "hline", "text", "span"}
        The annotation primitive.  ``"vline"`` / ``"hline"`` are reference lines
        at a constant ``x`` / ``y``; ``"text"`` places a label at ``(x, y)``;
        ``"span"`` shades the band between two values along one axis.
    text : str, optional
        Display text (the label, or the line's legend entry).
    x, y : float, optional
        Position (interpretation depends on ``kind``).
    span : tuple of float, optional
        ``(lo, hi)`` for a ``"span"`` annotation.
    axis : {"x", "y"}, optional
        Which axis a ``"span"`` runs along.  Default ``"x"``.
    style : dict, optional
        Backend-neutral style keys (color / alpha / linestyle).
    """

    kind: Literal["vline", "hline", "text", "span"]
    text: str = ""
    x: float | None = None
    y: float | None = None
    span: tuple[float, float] | None = None
    axis: Literal["x", "y"] = "x"
    style: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this annotation."""
        return {
            "kind": self.kind,
            "text": self.text,
            "x": None if self.x is None else float(self.x),
            "y": None if self.y is None else float(self.y),
            "span": list(self.span) if self.span is not None else None,
            "axis": self.axis,
            "style": dict(self.style),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Annotation:
        """Rebuild an :class:`Annotation` from :meth:`to_dict` output."""
        span = d.get("span")
        return cls(
            kind=d["kind"],
            text=d.get("text", ""),
            x=d.get("x"),
            y=d.get("y"),
            span=tuple(span) if span is not None else None,
            axis=d.get("axis", "x"),
            style=dict(d.get("style", {})),
        )


# ---------------------------------------------------------------------------
# Colorbar
# ---------------------------------------------------------------------------


@dataclass
class Colorbar:
    r"""A typed description of the color legend for a scalar / image color channel.

    A :class:`Colorbar` is the presentation of the *color* dimension — the
    counterpart of an :class:`Axis` for a layer's ``"c"`` channel or an
    ``IMAGE`` mark.  It is backend-neutral: it carries *what* to show, never
    *how* a particular renderer draws it.  The numeric color range it maps lives
    on the owning :class:`PlotSpec` as :attr:`PlotSpec.clim` (a single source of
    truth shared by every layer), so this dataclass holds only label / location /
    ticks / format / visibility.

    Parameters
    ----------
    label : str, optional
        Colorbar label (e.g. ``r"$|\nabla|$"`` or a basin index).  May carry
        LaTeX; renderers decide how to typeset it.
    location : {"right", "left", "top", "bottom"}, optional
        Where the colorbar sits relative to the plot.  Default ``"right"``.
    ticks : sequence of float, optional
        Explicit colorbar tick locations, or ``None`` to auto-tick.
    tickformat : str, optional
        A backend-neutral format string for the colorbar tick labels, or
        ``None``.
    show : bool, optional
        Whether to draw the colorbar at all.  Default ``True`` — a
        :class:`Colorbar` only exists on a spec when there *is* a color channel
        to legend, so its presence is the "draw a colorbar" signal; ``show`` lets
        a caller suppress it without dropping the (label-carrying) object.
    cmap : str, optional
        A backend-neutral colormap name for the color channel (e.g.
        ``"viridis"``, ``"tab20"`` for a categorical basin image), or ``None``
        to let the backend pick its default.
    norm : {"linear", "log", "symlog"}, optional
        How the color *data* maps onto the colormap (a log norm for a power
        spectrogram, say).  ``None`` is a linear norm.
    discrete : bool, optional
        Whether the color channel is categorical (discrete swatches, one per
        label — a basin / attractor index image) rather than a continuous ramp.
        Default ``False``.
    label_size : float, optional
        Font size for the colorbar label, or ``None`` to defer to the theme.
    """

    label: str = ""
    location: _CbarLoc = "right"
    ticks: Sequence[float] | None = None
    tickformat: str | None = None
    show: bool = True
    cmap: str | None = None
    norm: _Norm | None = None
    discrete: bool = False
    label_size: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this colorbar."""
        return {
            "label": self.label,
            "location": self.location,
            "ticks": [float(t) for t in self.ticks] if self.ticks is not None else None,
            "tickformat": self.tickformat,
            "show": bool(self.show),
            "cmap": self.cmap,
            "norm": self.norm,
            "discrete": bool(self.discrete),
            "label_size": None if self.label_size is None else float(self.label_size),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Colorbar:
        """Rebuild a :class:`Colorbar` from :meth:`to_dict` output."""
        ticks = d.get("ticks")
        return cls(
            label=d.get("label", ""),
            location=d.get("location", "right"),
            ticks=list(ticks) if ticks is not None else None,
            tickformat=d.get("tickformat"),
            show=bool(d.get("show", True)),
            cmap=d.get("cmap"),
            norm=d.get("norm"),
            discrete=bool(d.get("discrete", False)),
            label_size=d.get("label_size"),
        )


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


@dataclass
class Legend:
    """A typed description of the per-layer legend.

    A :class:`Legend` keys off each :attr:`Layer.label`; it carries only the
    legend's presentation (visibility / placement / title), never the entries
    themselves (those *are* the layer labels).

    Parameters
    ----------
    show : bool, optional
        Whether to draw the legend.  Default ``True`` — a :class:`Legend` only
        exists on a spec when labelled layers warrant one.
    location : str, optional
        Legend placement.  One of the backend-neutral spellings
        (``"best"``, ``"upper right"``, …).  Default ``"best"``.
    title : str, optional
        Legend title, or ``""`` for none.
    font_size : float, optional
        Font size for the legend entries, or ``None`` to defer to the theme.
    ncol : int, optional
        Number of columns to lay the entries out in.  Default ``1``.
    frame : bool, optional
        Whether to draw the legend's bounding frame / box.  Default ``True``.
    """

    show: bool = True
    location: _LegendLoc = "best"
    title: str = ""
    font_size: float | None = None
    ncol: int = 1
    frame: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this legend."""
        return {
            "show": bool(self.show),
            "location": self.location,
            "title": self.title,
            "font_size": None if self.font_size is None else float(self.font_size),
            "ncol": int(self.ncol),
            "frame": bool(self.frame),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Legend:
        """Rebuild a :class:`Legend` from :meth:`to_dict` output."""
        return cls(
            show=bool(d.get("show", True)),
            location=d.get("location", "best"),
            title=d.get("title", ""),
            font_size=d.get("font_size"),
            ncol=int(d.get("ncol", 1)),
            frame=bool(d.get("frame", True)),
        )


# ---------------------------------------------------------------------------
# Layout (composite figures)
# ---------------------------------------------------------------------------


@dataclass
class Layout:
    """How a :data:`PlotKind.COMPOSITE` spec arranges its ``panels`` into a figure.

    A composite :class:`PlotSpec` carries a list of sub-spec :attr:`PlotSpec.panels`
    and one :class:`Layout` saying how to tile them.  Like every other piece of the
    IR it is backend-neutral data: a renderer maps ``mode`` to its own subplot grid.

    Parameters
    ----------
    mode : {"stack", "row", "grid"}, optional
        The arrangement.  ``"stack"`` is one column of stacked panels (the
        default), ``"row"`` one row side-by-side, ``"grid"`` a 2-D grid sized from
        :attr:`rows` / :attr:`cols` (or made near-square when both are ``None``).
        New arrangements (picture-in-picture, …) are added here as new modes —
        the structure (``panels`` + ``Layout``) does not change.
    rows, cols : int, optional
        Explicit grid shape for ``mode="grid"``; ``None`` lets the renderer pick.
    share_x, share_y : bool, optional
        Whether the panels share an x / y axis (a stacked time-series column
        typically shares x).  Defaults: ``share_x`` follows the mode (stack → True),
        ``share_y`` ``False``.
    """

    mode: Literal["stack", "row", "grid"] = "stack"
    rows: int | None = None
    cols: int | None = None
    share_x: bool = False
    share_y: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this layout."""
        return {
            "mode": self.mode,
            "rows": self.rows,
            "cols": self.cols,
            "share_x": bool(self.share_x),
            "share_y": bool(self.share_y),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Layout:
        """Rebuild a :class:`Layout` from :meth:`to_dict` output."""
        return cls(
            mode=d.get("mode", "stack"),
            rows=d.get("rows"),
            cols=d.get("cols"),
            share_x=bool(d.get("share_x", False)),
            share_y=bool(d.get("share_y", False)),
        )


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


@dataclass
class Animation:
    """How an animated :class:`PlotSpec` plays — a backend-neutral directive.

    Animation is an **orthogonal modifier**: any spec of any :class:`PlotKind`
    becomes a movie by carrying an :class:`Animation` (``PlotSpec.animation``);
    the semantic kind is unchanged, and a backend that cannot animate draws the
    final frame.  The directive is plain data — it round-trips through
    :meth:`PlotSpec.to_dict` and ships to a web frontend untouched.

    Frame model (``mode``)
    ----------------------
    - ``"reveal"`` (the default): the layer keeps its full static data and each
      frame shows a *slice* of it — a comet whose head is the current sample and
      whose tail reaches back :attr:`trail_length` (``None`` ⇒ the whole curve
      persists).  Covers trajectories, phase portraits, delay embeddings, time
      series, spacetime.  Memory ``O(data)``.
    - ``"frames"``: a **spatial-field movie** — the field of a spatially-extended
      system (a method-of-lines PDE) played over time.  Each frame is the field's
      *spatial* state at that instant: a 1-D field plays as a travelling-wave line,
      a 2-D field as an ``imshow`` heatmap (Gray–Scott / Swift–Hohenberg), so
      consecutive frames carry genuinely different data.  Built via
      ``to_plot_spec(kind="field", animate=...)`` (a
      :data:`~tsdynamics.viz.spec.PlotKind.SPATIAL_FIELD` spec carrying the
      per-time field stack on its layer's ``"frames"`` channel); the matplotlib
      renderer plays the stack frame by frame, and a backend that cannot animate
      draws the final field.  **matplotlib-only** (mp4 / gif).

    Parameters
    ----------
    fps : float, optional
        Playback frames per second.  Default ``30``.
    duration : float, optional
        Total wall-clock seconds; with :attr:`fps` this fixes the frame count
        (``n_frames = round(duration * fps)``).  ``None`` lets the renderer pick a
        frame count from the data length.
    n_frames : int, optional
        Explicit frame count; overrides the :attr:`duration`/:attr:`fps` estimate.
    loop : bool, optional
        Whether the animation repeats.  Default ``True``.
    pingpong : bool, optional
        Play forward then in reverse each loop (implies looping).  Default
        ``False``.
    mode : {"reveal", "frames"}, optional
        The frame model (see above).  Default ``"reveal"``.
    trail_kind : {"time", "steps"}, optional
        Units of the comet tail length: physical ``"time"`` or sample ``"steps"``.
        ``None`` (with ``trail_length=None``) means a **persistent** trail (the
        curve never erases — the classic "orbit draws itself in").
    trail_length : float, optional
        Tail length in the chosen units; ``None`` ⇒ persistent.
    trail_fade : bool, optional
        Fade the tail's opacity from head to tail.  Default ``False``.
    head : bool, optional
        Draw the moving "current state" marker (a point on a curve, a sweep line
        on a spacetime image).  Default ``True``.
    head_size : float, optional
        Head marker size.  Default ``6``.
    head_color : str, optional
        Head marker color; ``None`` inherits the layer color.
    head_symbol : str, optional
        Head marker symbol (backend-neutral, e.g. ``"o"``).  Default ``"o"``.
    spin : float, optional
        Camera revolutions over the whole animation for a 3-D spec (the azimuth
        sweeps ``spin`` full turns).  ``0`` holds the camera still.  Default ``0``.
    clock : bool, optional
        Draw a live time readout that updates each frame.  Default ``False``.
    clock_format : str, optional
        Format for the clock label; ``{t}`` is the current time.  Default
        ``"t = {t:.2f}"``.
    """

    fps: float = 30.0
    duration: float | None = None
    n_frames: int | None = None
    loop: bool = True
    pingpong: bool = False
    mode: Literal["reveal", "frames"] = "reveal"
    trail_kind: Literal["time", "steps"] | None = None
    trail_length: float | None = None
    trail_fade: bool = False
    head: bool = True
    head_size: float = 6.0
    head_color: str | None = None
    head_symbol: str = "o"
    spin: float = 0.0
    clock: bool = False
    clock_format: str = "t = {t:.2f}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this animation directive."""
        return {
            "fps": float(self.fps),
            "duration": None if self.duration is None else float(self.duration),
            "n_frames": None if self.n_frames is None else int(self.n_frames),
            "loop": bool(self.loop),
            "pingpong": bool(self.pingpong),
            "mode": self.mode,
            "trail_kind": self.trail_kind,
            "trail_length": None if self.trail_length is None else float(self.trail_length),
            "trail_fade": bool(self.trail_fade),
            "head": bool(self.head),
            "head_size": float(self.head_size),
            "head_color": self.head_color,
            "head_symbol": self.head_symbol,
            "spin": float(self.spin),
            "clock": bool(self.clock),
            "clock_format": self.clock_format,
        }

    #: Default frame count when neither ``n_frames`` nor ``duration`` is set —
    #: high enough to read as continuous motion, capped so the artifact stays light
    #: (the comet renderers keep per-frame data tiny, so this can be generous).
    DEFAULT_FRAMES: ClassVar[int] = 360

    def frame_count(self, n_samples: int) -> int:
        """Resolve the number of playback frames from the directive + data length.

        ``n_frames`` wins; else ``round(duration * fps)``; else a capped default.
        At most ``n_samples`` (you cannot reveal more distinct samples than exist),
        and at least 2 — *except* a single-sample layer is a still, so a degenerate
        ``n_samples <= 1`` yields exactly one frame (the ``max(2, …)`` floor never
        overrides the data cap into a duplicate frame).
        """
        if n_samples <= 1:
            return 1
        if self.n_frames is not None:
            n = int(self.n_frames)
        elif self.duration is not None:
            n = int(round(self.duration * self.fps))
        else:
            n = min(n_samples, self.DEFAULT_FRAMES)
        return max(2, min(n, n_samples))

    def head_indices(self, n_samples: int) -> list[int]:
        """Map each playback frame to a sample index (the comet head).

        Frame 0 → sample 0, the last forward frame → ``n_samples - 1``; with
        :attr:`pingpong` the sequence then mirrors back (forward, then reverse).
        """
        n_frames = self.frame_count(n_samples)
        if n_frames <= 1:  # a single-sample still — one frame on the only sample
            return [0]
        fwd = [round(k * (n_samples - 1) / (n_frames - 1)) for k in range(n_frames)]
        if self.pingpong and n_frames > 2:
            fwd = fwd + fwd[-2:0:-1]
        return fwd

    def tail_samples(self, dt: float | None) -> int | None:
        """Convert the trail length to a sample count (``None`` ⇒ persistent trail).

        ``"steps"`` is used directly; ``"time"`` divides by ``dt`` (falling back to
        treating the value as steps when no ``dt`` is available).
        """
        if self.trail_kind is None or self.trail_length is None:
            return None
        if self.trail_kind == "steps":
            return max(1, int(round(self.trail_length)))
        if dt and dt > 0:  # "time" → samples
            return max(1, int(round(self.trail_length / dt)))
        return max(1, int(round(self.trail_length)))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Animation:
        """Rebuild an :class:`Animation` from :meth:`to_dict` output."""
        dur = d.get("duration")
        n = d.get("n_frames")
        tl = d.get("trail_length")
        return cls(
            fps=float(d.get("fps", 30.0)),
            duration=None if dur is None else float(dur),
            n_frames=None if n is None else int(n),
            loop=bool(d.get("loop", True)),
            pingpong=bool(d.get("pingpong", False)),
            mode=d.get("mode", "reveal"),
            trail_kind=d.get("trail_kind"),
            trail_length=None if tl is None else float(tl),
            trail_fade=bool(d.get("trail_fade", False)),
            head=bool(d.get("head", True)),
            head_size=float(d.get("head_size", 6.0)),
            head_color=d.get("head_color"),
            head_symbol=d.get("head_symbol", "o"),
            spin=float(d.get("spin", 0.0)),
            clock=bool(d.get("clock", False)),
            clock_format=d.get("clock_format", "t = {t:.2f}"),
        )


#: Sentinel for animation tweaks where ``None`` is itself a meaningful value
#: (e.g. ``trail(length=None)`` = a persistent trail vs. omitting ``length``).
_UNSET: Any = object()


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------


@dataclass
class Layer:
    """One drawable layer: a mark + its channel data + neutral style.

    Parameters
    ----------
    kind : PlotKind
        The layer *mark* — how this layer is drawn (``LINE``, ``SCATTER``,
        ``IMAGE``, ``QUIVER``, ``MARKERS``, ``LINE3D``, ``SURFACE3D``,
        ``HISTOGRAM``).
    data : dict of str to ndarray
        Channel name → array.  Inputs are coerced to :class:`numpy.ndarray` on
        construction.  The closed channel vocabulary (a renderer ignores a
        channel it does not consume):

        - ``"x"`` / ``"y"`` / ``"z"`` — coordinates (``"z"`` for 3-D marks).
        - ``"c"`` — a per-vertex color / scalar field; valid on a ``LINE`` or
          ``SCATTER`` too (color-by-time / color-by-speed), not only ``IMAGE``.
        - ``"u"`` / ``"v"`` — vector components for a ``QUIVER`` / ``VECTOR_FIELD``.
        - ``"lo"`` / ``"hi"`` — lower / upper band edges for an ``AREA`` /
          ``ENSEMBLE_FAN`` (a shaded ``lo <= hi`` envelope around ``y``).
        - ``"err"`` — symmetric error magnitudes for an ``ERRORBAR`` (the error
          bars on a ``DIMENSION_SPECTRUM`` ``D(q)``).
        - ``"cat"`` — integer category indices pairing with the categorical
          :attr:`Axis.categories` (a ``BAR`` / ``CATEGORICAL_BAR`` / ``FEATURE_BARS``).
        - ``"size"`` — per-point marker size for a ``SCATTER``.
        - ``"frames"`` — the per-time field stack of a ``SPATIAL_FIELD`` layer
          (shape ``(T, *spatial)`` — ``(T, Nx)`` for a 1-D profile, ``(T, Ny, Nx)``
          for a 2-D field), played frame by frame by the ``"frames"``-mode
          animator.  The layer's static channels hold the *final* field.
    label : str, optional
        Legend entry for this layer, or ``None``.
    style : dict, optional
        Backend-neutral style keys — ``color``, ``cmap``, ``lw``, ``alpha``,
        ``marker``, ``s``.  Renderers map these to their own idioms; unknown
        keys are ignored by a renderer rather than erroring.
    """

    kind: PlotKind
    data: dict[str, np.ndarray] = field(default_factory=dict)
    label: str | None = None
    style: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize ``kind`` to :class:`PlotKind` and coerce data to arrays."""
        self.kind = PlotKind(self.kind)
        self.data = {k: np.asarray(v) for k, v in self.data.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping (arrays become nested lists)."""
        return {
            "kind": self.kind.value,
            "data": {k: np.asarray(v).tolist() for k, v in self.data.items()},
            "label": self.label,
            "style": dict(self.style),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Layer:
        """Rebuild a :class:`Layer` from :meth:`to_dict` output (lists → arrays)."""
        return cls(
            kind=PlotKind(d["kind"]),
            data={k: np.asarray(v) for k, v in d.get("data", {}).items()},
            label=d.get("label"),
            style=dict(d.get("style", {})),
        )


# ---------------------------------------------------------------------------
# PlotSpec
# ---------------------------------------------------------------------------


@dataclass
class PlotSpec:
    """A backend-agnostic, serializable description of a plot.

    A :class:`PlotSpec` carries everything a renderer needs and nothing it does
    not: the semantic :attr:`kind`, the drawable :attr:`layers`, the typed axes,
    and presentation metadata.  It holds **no** rendering state and imports
    **no** plotting library.

    Parameters
    ----------
    kind : PlotKind
        The *semantic* kind of the whole plot (``TIME_SERIES``,
        ``PHASE_PORTRAIT_3D``, ``BIFURCATION``, …).  A renderer dispatches on it.
    layers : list of Layer
        The drawable layers, in draw order.
    x, y : Axis, optional
        The horizontal / vertical axes.  Default empty :class:`Axis`.
    z : Axis, optional
        The depth axis; present iff the plot is 3D, else ``None``.
    clim : tuple of float, optional
        ``(vmin, vmax)`` range for the color channel (a layer's ``"c"`` field or
        an ``IMAGE`` mark).  ``None`` auto-scales the color mapping.  This is the
        single source of truth for the color range; a :class:`Colorbar` legends
        it.
    colorbar : Colorbar, optional
        The color legend, present iff the plot has a color dimension to legend
        (a scalar ``"c"`` channel, an ``IMAGE`` / ``SURFACE3D`` mark, or a
        semantic image kind), else ``None``.
    legend : Legend, optional
        The per-layer legend, present iff a legend is wanted (typically when
        ≥ 2 layers carry labels), else ``None``.
    title : str, optional
        Plot title.
    ndim : {1, 2, 3}, optional
        Spatial dimensionality of the plot.  Default ``2``.
    aspect : {"auto", "equal"}, optional
        Aspect-ratio hint.  ``"equal"`` for phase portraits / sections /
        recurrence images; ``"auto"`` otherwise.  Default ``"auto"``.
    annotations : list of Annotation, optional
        Reference lines / text overlays carried by the result.
    meta : dict, optional
        Provenance and rendering hints passed through untouched (e.g.
        ``meta["animate"] = {"fps": 30}`` for an animated spec, or
        ``meta["figsize"]`` / ``meta["dpi"]`` set by :meth:`size`).
    theme : Theme, optional
        The figure-level look (palette / font / background / grid / line
        defaults).  ``None`` (default) defers to the active global default theme
        (renderers call :func:`~tsdynamics.viz.style.get_theme` when
        :attr:`theme` is ``None``).

    Notes
    -----
    The tweak methods (:meth:`relabel`, :meth:`rescale`, :meth:`limits`,
    :meth:`ticks`, :meth:`style`, :meth:`recolor`, :meth:`theme`,
    :meth:`palette`, :meth:`grid`, :meth:`font`, :meth:`background`,
    :meth:`size`) mutate the spec in place and return ``self``,
    so they chain::

        spec.rescale(x="log").limits(y=(1e-17, 5)).ticks(x=[1, 10, 100])

    Because they touch the spec rather than a renderer, the same tweak renders
    identically on every backend.
    """

    kind: PlotKind
    layers: list[Layer] = field(default_factory=list)
    x: Axis = field(default_factory=Axis)
    y: Axis = field(default_factory=Axis)
    z: Axis | None = None
    clim: tuple[float, float] | None = None
    colorbar: Colorbar | None = None
    legend: Legend | None = None
    title: str = ""
    ndim: _Ndim = 2
    aspect: _Aspect = "auto"
    annotations: list[Annotation] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    # Composition: a :data:`PlotKind.COMPOSITE` spec carries child ``panels`` (each
    # a single-panel :class:`PlotSpec`) and a :class:`Layout`; a single-panel spec
    # leaves these empty.  ``tsdynamics.viz.plot`` builds composites; the renderers
    # tile ``panels`` per ``layout``.
    panels: list[PlotSpec] = field(default_factory=list)
    layout: Layout | None = None
    # Animation: an orthogonal modifier — any spec (any kind, single-panel or a
    # composite) becomes a movie by carrying an :class:`Animation`.  The semantic
    # ``kind`` is unchanged; a backend that cannot animate draws the final frame.
    animation: Animation | None = None
    # Theme: the figure-level look (palette / font / background / grid / line
    # defaults).  ``None`` ⇒ renderers resolve the active global default at draw
    # time (``get_theme(None)``); a panel's own theme overrides a composite's.
    #
    # Stored under the private ``_theme`` field because the public fluent tweak is
    # the ``theme(...)`` **method** (frozen public name) — a dataclass field and a
    # method cannot share one name.  Read the resolved value via the
    # :attr:`resolved_theme` property or :meth:`theme_or_default`; renderers read
    # the raw ``Theme | None`` via the :attr:`_theme` field.
    _theme: Theme | None = None

    def __post_init__(self) -> None:
        """Normalize ``kind`` and the optional ``clim`` color range."""
        self.kind = PlotKind(self.kind)
        self.clim = _as_pair(self.clim)

    @property
    def is_composite(self) -> bool:
        """Whether this spec is a multi-panel composite (has child ``panels``)."""
        return bool(self.panels) or self.kind == PlotKind.COMPOSITE

    @property
    def is_animated(self) -> bool:
        """Whether this spec carries an :class:`Animation` directive."""
        return self.animation is not None

    # -- uniform, backend-independent tweaks (mutate + return self) ---------

    def relabel(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        title: str | None = None,
    ) -> PlotSpec:
        """Set axis labels and/or the title (only the arguments you pass).

        Parameters
        ----------
        x, y, z : str, optional
            New axis labels.  ``z`` is ignored if the spec has no ``z`` axis.
        title : str, optional
            New plot title.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        if x is not None:
            self.x.label = x
        if y is not None:
            self.y.label = y
        if z is not None and self.z is not None:
            self.z.label = z
        if title is not None:
            self.title = title
        return self

    def rescale(
        self,
        *,
        x: _Scale | None = None,
        y: _Scale | None = None,
        z: _Scale | None = None,
    ) -> PlotSpec:
        """Set axis scales to ``"linear"`` / ``"log"`` / ``"symlog"``.

        Parameters
        ----------
        x, y, z : {"linear", "log", "symlog"}, optional
            New scales.  ``z`` is ignored if the spec has no ``z`` axis.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        if x is not None:
            self.x.scale = x
        if y is not None:
            self.y.scale = y
        if z is not None and self.z is not None:
            self.z.scale = z
        return self

    def limits(
        self,
        *,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
    ) -> PlotSpec:
        """Set ``(lo, hi)`` view limits per axis.

        Parameters
        ----------
        x, y, z : tuple of float, optional
            ``(lo, hi)`` limits.  ``z`` is ignored if the spec has no ``z`` axis.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        if x is not None:
            self.x.limits = x
        if y is not None:
            self.y.limits = y
        if z is not None and self.z is not None:
            self.z.limits = z
        return self

    def ticks(
        self,
        *,
        x: Sequence[float] | None = None,
        y: Sequence[float] | None = None,
        z: Sequence[float] | None = None,
    ) -> PlotSpec:
        """Set explicit tick locations per axis.

        Parameters
        ----------
        x, y, z : sequence of float, optional
            Tick locations.  ``z`` is ignored if the spec has no ``z`` axis.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        if x is not None:
            self.x.ticks = list(x)
        if y is not None:
            self.y.ticks = list(y)
        if z is not None and self.z is not None:
            self.z.ticks = list(z)
        return self

    def style(self, *, layer: int | None = None, axes: bool | None = None, **kw: Any) -> PlotSpec:
        """Merge backend-neutral style keys into one layer or every layer.

        Parameters
        ----------
        layer : int, optional
            Index of the layer to style.  If ``None`` (default), the style is
            merged into *every* layer.
        axes : bool, optional
            Figure-level (not per-layer): set ``False`` to hide the axes entirely
            — ticks, labels, gridlines, and (in 3-D) the grey background panes —
            for a clean "object floating in space" look (e.g. an attractor or its
            animation).  ``None`` leaves axis visibility unchanged.
        **kw
            Per-layer style keys to set (``color``, ``cmap``, ``lw``, ``alpha``,
            ``marker``, …).  These are routed through
            :func:`~tsdynamics.viz.style.normalize_style`: aliases (``lw`` →
            ``linewidth``, ``c`` → ``color``, ``s`` → ``markersize``) are
            canonicalized, values validated, and an unknown key is dropped with a
            single ``VisualizationDegraded`` warning.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        if axes is not None:
            self.meta["axes_visible"] = bool(axes)
        canon = normalize_style(kw)
        targets = self.layers if layer is None else [self.layers[layer]]
        for lyr in targets:
            lyr.style.update(canon)
        return self

    def recolor(self, *colors: str, layer: int | None = None) -> PlotSpec:
        """Assign explicit colors to layers (the per-layer color shorthand).

        Parameters
        ----------
        *colors : str
            One or more colors (CSS name / hex / …).
        layer : int, optional
            When ``None`` (default), color ``i`` is assigned to layer ``i``,
            cycling (``colors[i % len(colors)]``).  When an ``int``, that one
            layer's color is set to ``colors[0]``.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        if not colors:
            return self
        if layer is not None:
            self.layers[layer].style["color"] = colors[0]
            return self
        n = len(colors)
        for i, lyr in enumerate(self.layers):
            lyr.style["color"] = colors[i % n]
        return self

    @property
    def resolved_theme(self) -> Theme:
        """The effective :class:`~tsdynamics.viz.style.Theme` for this spec.

        This is the **READ** half of the three-part theme pattern (see
        :meth:`theme` for the full picture): ``spec.theme(...)`` **SETS**,
        ``spec.resolved_theme`` **READS** the effective theme, and ``spec._theme``
        is the **raw private field**.

        Returns this spec's own theme if it set one (``spec._theme is not None``),
        otherwise the active global default
        (:func:`~tsdynamics.viz.style.get_theme`).  Renderers that need a concrete
        theme read this; the unresolved ``Theme | None`` lives on the private
        :attr:`_theme` field.
        """
        return self._theme if self._theme is not None else get_theme(None)

    def theme(self, theme: str | Theme | None = None, /, **overrides: Any) -> PlotSpec:
        """Set this spec's figure-level :class:`~tsdynamics.viz.style.Theme`.

        The figure-level look (palette / font / background / grid / line
        defaults).  This is the **SET** half of a deliberate three-part pattern:

        - ``spec.theme(...)`` **SETS** (mutate this spec's theme + return ``self``
          for chaining) — *this method*.
        - :attr:`spec.resolved_theme <resolved_theme>` **READS** the effective
          theme, falling back to the active global default when this spec set
          none.
        - ``spec._theme`` is the **raw private field** (``Theme | None``) — the
          unresolved attribute renderers may read directly; prefer
          :attr:`resolved_theme` everywhere else.

        ``theme`` is **positional-only** — call ``spec.theme("dark")``, not
        ``spec.theme(theme="dark")`` (the keyword name is reserved so a theme
        field literally named ``theme`` could be passed in ``**overrides``).

        Eager-materialisation semantics:

        - ``spec.theme("dark")`` / ``spec.theme(some_theme)`` pins that named /
          explicit theme.
        - ``spec.theme(None)`` (or ``spec.theme()``) with **no** overrides stores
          the active global default *by snapshot* — and because that snapshot is
          taken now, the spec is **detached from any later**
          :func:`~tsdynamics.viz.style.set_theme` (it will not track a future
          global-default change).
        - ``spec.theme(None, **overrides)`` **eagerly materialises** the active
          global default and applies ``overrides`` on top via
          :meth:`~tsdynamics.viz.style.Theme.merged` — likewise detaching from a
          future ``set_theme``.

        Parameters
        ----------
        theme : str or Theme or None, positional-only
            A registered theme name, a :class:`~tsdynamics.viz.style.Theme`
            instance, or ``None`` (snapshot the active global default *now*).
            Positional-only: pass it by position, never as ``theme=``.
        **overrides
            :class:`~tsdynamics.viz.style.Theme` fields to override on top of the
            resolved theme (``palette``, ``background``, ``font_family``, …),
            applied via :meth:`~tsdynamics.viz.style.Theme.merged`.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        base = theme if isinstance(theme, Theme) else get_theme(theme)
        self._theme = base.merged(**overrides) if overrides else base
        return self

    def palette(self, colors: str | Sequence[str]) -> PlotSpec:
        """Set the theme's color cycle (a named palette or an explicit list).

        Parameters
        ----------
        colors : str or sequence of str
            A registered theme name (its palette) or an explicit sequence of
            colors.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        from .style import resolve_palette

        base = self._theme if self._theme is not None else get_theme()
        self._theme = base.merged(palette=resolve_palette(colors))
        return self

    def grid(
        self,
        show: bool = True,
        *,
        axis: Literal["x", "y", "both"] = "both",
        color: str | None = None,
        alpha: float | None = None,
    ) -> PlotSpec:
        """Toggle / style gridlines on the chosen axis (or both).

        Parameters
        ----------
        show : bool, optional
            Whether to draw gridlines.  Default ``True``.
        axis : {"x", "y", "both"}, optional
            Which axis the grid applies to.  Default ``"both"``.
        color : str, optional
            Gridline color — applied as a theme override (``grid_color``) so it is
            shared across the figure.
        alpha : float, optional
            Gridline opacity — applied as a theme override (``grid_alpha``).

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        targets: list[Axis] = []
        if axis in ("x", "both"):
            targets.append(self.x)
        if axis in ("y", "both"):
            targets.append(self.y)
        for ax in targets:
            ax.grid = bool(show)
        if color is not None or alpha is not None:
            base = self._theme if self._theme is not None else get_theme()
            overrides: dict[str, Any] = {}
            if color is not None:
                overrides["grid_color"] = color
            if alpha is not None:
                overrides["grid_alpha"] = float(alpha)
            self._theme = base.merged(**overrides)
        return self

    def font(self, family: str | None = None, size: float | None = None) -> PlotSpec:
        """Set the theme font family and/or size.

        Parameters
        ----------
        family : str, optional
            Font family.
        size : float, optional
            Base font size.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        base = self._theme if self._theme is not None else get_theme()
        overrides: dict[str, Any] = {}
        if family is not None:
            overrides["font_family"] = family
        if size is not None:
            overrides["font_size"] = float(size)
        if overrides:
            self._theme = base.merged(**overrides)
        return self

    def background(self, color: str) -> PlotSpec:
        """Set the theme background (figure / axes facecolor).

        Parameters
        ----------
        color : str
            Background color (CSS name / hex / …).

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        base = self._theme if self._theme is not None else get_theme()
        self._theme = base.merged(background=color)
        return self

    def size(
        self,
        width: float | None = None,
        height: float | None = None,
        dpi: float | None = None,
    ) -> PlotSpec:
        """Set the figure size (pixels-as-inches via ``meta``) and/or resolution.

        ``figsize`` / ``dpi`` are *not* theme fields — they live in ``meta``
        (``meta["figsize"] = (width, height)``, ``meta["dpi"]``), only the
        dimensions you pass being updated.

        Parameters
        ----------
        width, height : float, optional
            Figure width / height; an omitted dimension keeps its current value.
        dpi : float, optional
            Output resolution (dots per inch).

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        if width is not None or height is not None:
            cur = self.meta.get("figsize")
            cw, ch = cur if isinstance(cur, (tuple, list)) and len(cur) == 2 else (None, None)
            new_w = float(width) if width is not None else cw
            new_h = float(height) if height is not None else ch
            self.meta["figsize"] = (new_w, new_h)
        if dpi is not None:
            self.meta["dpi"] = float(dpi)
        return self

    def _axes_hidden(self) -> bool:
        """Whether ``style(axes=False)`` asked to hide the axes (renderer helper)."""
        return isinstance(self.meta, dict) and self.meta.get("axes_visible") is False

    def colorize(
        self,
        *,
        clim: tuple[float, float] | None = None,
        colorbar: Colorbar | bool | None = None,
        legend: Legend | bool | None = None,
    ) -> PlotSpec:
        """Set the color range, colorbar, and/or legend (only the args you pass).

        Like the other tweaks this mutates the spec and returns ``self`` so it
        chains.  Each keyword defaults to the sentinel "leave unchanged".

        Parameters
        ----------
        clim : tuple of float, optional
            ``(vmin, vmax)`` color range, coerced to floats.  Leaves
            :attr:`clim` unchanged when omitted.
        colorbar : Colorbar or bool, optional
            A :class:`Colorbar` to attach, or ``True`` to attach a default one /
            ``False`` to drop it.  Omitting it leaves :attr:`colorbar`
            unchanged.
        legend : Legend or bool, optional
            A :class:`Legend` to attach, or ``True`` to attach a default one /
            ``False`` to drop it.  Omitting it leaves :attr:`legend` unchanged.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        if clim is not None:
            self.clim = _as_pair(clim)
        if colorbar is not None:
            self.colorbar = Colorbar() if colorbar is True else (colorbar or None)
        if legend is not None:
            self.legend = Legend() if legend is True else (legend or None)
        return self

    # -- animation tweaks (mutate + return self; also turn animation on) ----

    def _ensure_animation(self) -> Animation:
        """Return this spec's :class:`Animation`, creating a default if absent."""
        if self.animation is None:
            self.animation = Animation()
        return self.animation

    def animate(
        self,
        *,
        fps: float | None = None,
        duration: float | None = None,
        n_frames: int | None = None,
        loop: bool | None = None,
        pingpong: bool | None = None,
        mode: Literal["reveal", "frames"] | None = None,
    ) -> PlotSpec:
        """Turn this spec into an animation and/or set its timeline.

        Calling ``animate()`` on a static spec makes it animated (with defaults);
        the keyword arguments set the playback timeline.  Like every tweak it
        mutates the spec and returns ``self`` so it chains.

        Parameters
        ----------
        fps : float, optional
            Playback frames per second.
        duration : float, optional
            Total seconds (with ``fps`` this fixes the frame count).
        n_frames : int, optional
            Explicit frame count (overrides ``duration``/``fps``).
        loop : bool, optional
            Whether the animation repeats.
        pingpong : bool, optional
            Play forward then reverse each loop.
        mode : {"reveal", "frames"}, optional
            The frame model.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        a = self._ensure_animation()
        if fps is not None:
            a.fps = float(fps)
        if duration is not None:
            a.duration = float(duration)
        if n_frames is not None:
            a.n_frames = int(n_frames)
        if loop is not None:
            a.loop = bool(loop)
        if pingpong is not None:
            a.pingpong = bool(pingpong)
        if mode is not None:
            a.mode = mode
        return self

    def trail(
        self,
        length: tuple[Literal["time", "steps"], float] | None = _UNSET,
        *,
        fade: bool | None = None,
    ) -> PlotSpec:
        """Set the comet tail behind the animation's moving head.

        Parameters
        ----------
        length : ``("time", t)`` or ``("steps", n)`` or ``None``, optional
            The tail length — in physical ``"time"`` units or sample ``"steps"``,
            or ``None`` for a **persistent** trail (the curve never erases).
            Omitting the argument leaves the current trail unchanged.
        fade : bool, optional
            Fade the tail opacity from head to tail.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        a = self._ensure_animation()
        if length is not _UNSET:
            if length is None:
                a.trail_kind, a.trail_length = None, None
            else:
                kind, value = length
                a.trail_kind, a.trail_length = kind, float(value)
        if fade is not None:
            a.trail_fade = bool(fade)
        return self

    def head(
        self,
        show: bool | None = None,
        *,
        size: float | None = None,
        color: str | None = None,
        symbol: str | None = None,
    ) -> PlotSpec:
        """Configure the moving "current state" marker.

        Parameters
        ----------
        show : bool, optional
            Whether to draw the head marker.
        size : float, optional
            Marker size.
        color : str, optional
            Marker color (``None`` inherits the layer color).
        symbol : str, optional
            Marker symbol.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        a = self._ensure_animation()
        if show is not None:
            a.head = bool(show)
        if size is not None:
            a.head_size = float(size)
        if color is not None:
            a.head_color = color
        if symbol is not None:
            a.head_symbol = symbol
        return self

    def camera(
        self,
        *,
        elev: float | None = None,
        azim: float | None = None,
        spin: float | None = None,
    ) -> PlotSpec:
        """Set the 3-D camera angle and/or its animated spin.

        ``elev`` / ``azim`` set a fixed viewing angle (a static tweak, recorded in
        ``meta["camera"]``); ``spin`` makes the camera revolve ``spin`` full turns
        over an animation (and turns animation on).

        Parameters
        ----------
        elev, azim : float, optional
            Fixed elevation / azimuth in degrees.
        spin : float, optional
            Camera revolutions over the whole animation (0 holds it still).

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        if elev is not None or azim is not None:
            camera = dict(self.meta.get("camera", {}))
            if elev is not None:
                camera["elev"] = float(elev)
            if azim is not None:
                camera["azim"] = float(azim)
            self.meta["camera"] = camera
        if spin is not None:
            self._ensure_animation().spin = float(spin)
        return self

    def clock(self, show: bool = True, *, fmt: str | None = None) -> PlotSpec:
        """Show (or hide) a live time readout that updates each frame.

        Parameters
        ----------
        show : bool, optional
            Whether to draw the clock.  Default ``True``.
        fmt : str, optional
            Label format; ``{t}`` is the current time (e.g. ``"t = {t:.2f}"``).

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        a = self._ensure_animation()
        a.clock = bool(show)
        if fmt is not None:
            a.clock_format = fmt
        return self

    # -- color / legend completeness ---------------------------------------

    _COLOR_KINDS: ClassVar[frozenset[PlotKind]] = frozenset(
        {
            PlotKind.IMAGE,
            PlotKind.SURFACE3D,
            PlotKind.BASINS_IMAGE,
            PlotKind.RECURRENCE_PLOT,
            PlotKind.SPACETIME,
            PlotKind.SPECTROGRAM,
        }
    )

    def has_color_channel(self) -> bool:
        """Whether this spec has a color dimension to legend with a colorbar.

        ``True`` when the semantic :attr:`kind` is an image-like kind
        (``IMAGE`` / ``BASINS_IMAGE`` / ``RECURRENCE_PLOT`` / ``SPACETIME`` /
        ``SURFACE3D``) or any :class:`Layer` is a color-mapped mark
        (``IMAGE`` / ``SURFACE3D``) or carries a scalar ``"c"`` channel.
        """
        if self.kind in self._COLOR_KINDS:
            return True
        for lyr in self.layers:
            if lyr.kind in (PlotKind.IMAGE, PlotKind.SURFACE3D) or "c" in lyr.data:
                return True
        return False

    def autocolor(self) -> PlotSpec:
        """Attach a :class:`Colorbar` and infer :attr:`clim` for a colored spec.

        This is the "image / colored kinds express a colorbar + range" contract:
        a :class:`PlotSpec` that :meth:`has_color_channel` gains a default
        :class:`Colorbar` (if it has none) and, when its color range is not
        already set, a :attr:`clim` computed from the finite extent of the color
        data — the ``"c"`` channel of each layer, or the layer ``"z"`` /
        spec-level ``z`` data for an ``IMAGE`` mark.

        The method is a no-op on a spec with no color channel, never overwrites a
        :attr:`clim` / :attr:`colorbar` the caller already set, and (like the
        other tweaks) mutates and returns ``self`` so it chains.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        if not self.has_color_channel():
            return self
        if self.colorbar is None:
            self.colorbar = Colorbar()
        if self.clim is None:
            self.clim = self._infer_clim()
        return self

    def _infer_clim(self) -> tuple[float, float] | None:
        """Compute ``(vmin, vmax)`` from the finite color data, or ``None``."""
        lo = np.inf
        hi = -np.inf
        for lyr in self.layers:
            for chan in ("c", "z"):
                arr = lyr.data.get(chan)
                if arr is None:
                    continue
                finite = np.asarray(arr, dtype=float)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    lo = min(lo, float(finite.min()))
                    hi = max(hi, float(finite.max()))
        if np.isfinite(lo) and np.isfinite(hi):
            return (lo, hi)
        return None

    # -- rendering ---------------------------------------------------------

    def render(self, backend: str | None = None, **backend_kw: Any) -> Any:
        """Render this spec through a registered backend.

        Delegates to :func:`tsdynamics.viz.render.render_spec` (stream
        VIZ-DISPATCH), which registers the installed in-tree backends on first
        use, selects one by name or by capability — falling back to the
        matplotlib reference renderer when the requested backend declines this
        spec's kind — and calls it.  Until any backend is registered this raises
        :class:`VisualizationNotInstalled`.

        Parameters
        ----------
        backend : str, optional
            The renderer name (e.g. ``"matplotlib"``).  If ``None``, the default
            capable backend is used.
        **backend_kw
            Forwarded to the renderer callable.

        Returns
        -------
        Any
            Whatever the backend returns (e.g. a figure handle or export payload).

        Raises
        ------
        VisualizationNotInstalled
            If no rendering backend is registered.
        """
        from tsdynamics.viz.render import render_spec

        return render_spec(self, backend, **backend_kw)

    def plot(self, backend: str | None = None, **tweaks: Any) -> Any:
        """Render this spec, applying inline tweaks first (the ``.plot`` sugar).

        A :class:`PlotSpec` *is* the thing :func:`tsdynamics.viz.plot` returns, so
        it carries the same ``.plot`` convenience as the data types: recognised
        inline tweaks (``xlabel`` / ``yscale`` / ``title`` / …) are applied to the
        spec, then it is rendered through the backend dispatch.
        """
        backend_kw = _apply_inline_tweaks(self, tweaks)
        return self.render(backend, **backend_kw)

    def save(
        self,
        path: str,
        *,
        backend: str | None = None,
        fps: float | None = None,
        dpi: float | None = None,
        size: tuple[float, float] | None = None,
        **backend_kw: Any,
    ) -> str:
        """Render and write the figure (or animation) to ``path``; return the path.

        When ``backend`` is not given, one is chosen from the file extension. For a
        **static** spec: a raster / vector image (``.png`` / ``.pdf`` / ``.svg`` /
        ``.jpg``) prefers the always-present matplotlib reference renderer,
        ``.html`` prefers plotly (a self-contained interactive page), and ``.json``
        prefers the json data exporter.  For an **animated** spec (``is_animated``):
        ``.html`` prefers plotly (an interactive scrubber), while ``.mp4`` / ``.gif``
        go to matplotlib (its :class:`~matplotlib.animation.FuncAnimation` writes
        them via ffmpeg / pillow); a still image renders the final frame.

        Parameters
        ----------
        path : str
            Output path; its extension selects the format / default backend.
        backend : str, optional
            Force a backend instead of choosing by extension.
        fps : float, optional
            Frames per second for a video / gif export (overrides the spec's
            :attr:`Animation.fps` for this write only).
        dpi : float, optional
            Output resolution in dots per inch (video, image).
        size : tuple of float, optional
            Output size ``(width, height)`` in **pixels** (converted to a
            matplotlib figure size via ``dpi``).
        **backend_kw
            Forwarded to the renderer.

        Raises
        ------
        TypeError
            If the produced result is neither a savable figure / animation nor a
            data-export write.
        """
        if backend is None:
            backend = self._preferred_save_backend(path)
        if backend in ("json", "threejs"):
            # The data-export backends write the file themselves (and return the path).
            self.render(backend, path=path, **backend_kw)
            return path
        _lower = str(path).lower()
        _movie = (".mp4", ".gif", ".webm", ".mov", ".m4v", ".apng")
        if self.is_animated and backend == "plotly" and _lower.endswith((".html", ".htm")):
            # Animated HTML → the plotly backend's real-time (rAF + restyle) export,
            # which writes the file itself (a smooth, rotatable-while-playing page).
            self.render("plotly", path=path, **backend_kw)
            return path
        if (
            self.is_animated
            and not _lower.endswith(_movie)
            and not _lower.endswith((".html", ".htm"))
        ):
            # A still image (.png / .pdf / .svg / …) of an animated spec is its
            # final, fully-revealed frame: render the static spec and write that.
            static = PlotSpec.from_dict({**self.to_dict(), "animation": None})
            return static.save(path, backend=backend, dpi=dpi, size=size, **backend_kw)
        if size is not None and "figsize" not in backend_kw:
            w, h = size
            scale = float(dpi) if dpi else 100.0
            backend_kw["figsize"] = (float(w) / scale, float(h) / scale)
        result = self.render(backend, **backend_kw)
        # A matplotlib animation (FuncAnimation — uniquely carries ``to_jshtml``)
        # writes mp4 / gif via its own ``.save`` (writer inferred from the extension).
        if hasattr(result, "to_jshtml") and callable(getattr(result, "save", None)):
            anim_kw: dict[str, Any] = {}
            if fps is not None:
                anim_kw["fps"] = float(fps)
            if dpi is not None:
                anim_kw["dpi"] = float(dpi)
            result.save(path, **anim_kw)
            return path
        figure = getattr(result, "figure", result)
        savefig = getattr(figure, "savefig", None)
        if callable(savefig):
            savefig(path, **({"dpi": float(dpi)} if dpi is not None else {}))
            return path
        if str(path).endswith((".html", ".htm")):
            write_html = getattr(figure, "write_html", None)
            if callable(write_html):
                write_html(path)
                return path
        write_image = getattr(figure, "write_image", None)
        if callable(write_image):
            write_image(path)
            return path
        raise TypeError(
            f"the {backend or 'selected'} backend produced a non-savable result "
            f"({type(figure).__name__}); use .to_dict() to serialize it instead."
        )

    def _preferred_save_backend(self, path: str) -> str | None:
        """Pick a save backend from ``path``'s extension (``None`` = dispatch default).

        Static specs: ``.json`` → json exporter, ``.html`` → plotly, everything else
        → the matplotlib reference renderer.  Animated specs: ``.html`` → plotly (an
        interactive scrubber), everything else (``.mp4`` / ``.gif`` / a still frame)
        → matplotlib.  Falls back to the dispatch default when the preferred backend
        is not registered.
        """
        try:
            from tsdynamics import registry
            from tsdynamics.viz.render import register_builtin_renderers

            register_builtin_renderers()
            names = set(registry.renderers.names())
        except Exception:  # pragma: no cover - defensive
            return None
        p = str(path).lower()
        if self.is_animated:
            if p.endswith((".html", ".htm")) and "plotly" in names:
                return "plotly"
            return "matplotlib" if "matplotlib" in names else None
        if p.endswith(".json") and "json" in names:
            return "json"
        if p.endswith((".html", ".htm")) and "plotly" in names:
            return "plotly"
        if "matplotlib" in names:
            return "matplotlib"
        return None

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Notebook display hook — render inline once a backend is installed.

        Mirrors :meth:`Plottable._repr_mimebundle_`: returns ``None`` (so the
        console falls back to ``repr``) until a rendering backend is registered,
        keeping a plain ``import`` plot-library-free.
        """
        if _resolve_renderers() is None:
            return None
        try:  # pragma: no cover - exercised only once a backend is installed
            return self.plot()
        except Exception:  # pragma: no cover - never break repr on a render error
            return None

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping (every NumPy array becomes a list).

        The result round-trips through :meth:`from_dict`, so a computed spec can
        be JSON-serialized, cached, or shipped to a web frontend and rebuilt
        without recomputing the underlying analysis.

        Returns
        -------
        dict
            A nested mapping of plain ``str`` / ``float`` / ``list`` values.
        """
        return {
            "kind": self.kind.value,
            "layers": [lyr.to_dict() for lyr in self.layers],
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
            "z": self.z.to_dict() if self.z is not None else None,
            "clim": list(self.clim) if self.clim is not None else None,
            "colorbar": self.colorbar.to_dict() if self.colorbar is not None else None,
            "legend": self.legend.to_dict() if self.legend is not None else None,
            "title": self.title,
            "ndim": self.ndim,
            "aspect": self.aspect,
            "annotations": [a.to_dict() for a in self.annotations],
            "meta": _jsonify(self.meta),
            "panels": [p.to_dict() for p in self.panels],
            "layout": self.layout.to_dict() if self.layout is not None else None,
            "animation": self.animation.to_dict() if self.animation is not None else None,
            "theme": self._theme.to_dict() if self._theme is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PlotSpec:
        """Rebuild a :class:`PlotSpec` from :meth:`to_dict` output.

        Layer / annotation data lists are coerced back to
        :class:`numpy.ndarray`.  ``meta`` is restored verbatim (it is left as
        plain JSON types — it is provenance, not plot data).

        Parameters
        ----------
        d : Mapping
            The mapping produced by :meth:`to_dict`.

        Returns
        -------
        PlotSpec
        """
        z = d.get("z")
        clim = d.get("clim")
        colorbar = d.get("colorbar")
        legend = d.get("legend")
        return cls(
            kind=PlotKind(d["kind"]),
            layers=[Layer.from_dict(lyr) for lyr in d.get("layers", [])],
            x=Axis.from_dict(d["x"]) if d.get("x") is not None else Axis(),
            y=Axis.from_dict(d["y"]) if d.get("y") is not None else Axis(),
            z=Axis.from_dict(z) if z is not None else None,
            clim=tuple(clim) if clim is not None else None,
            colorbar=Colorbar.from_dict(colorbar) if colorbar is not None else None,
            legend=Legend.from_dict(legend) if legend is not None else None,
            title=d.get("title", ""),
            ndim=d.get("ndim", 2),
            aspect=d.get("aspect", "auto"),
            annotations=[Annotation.from_dict(a) for a in d.get("annotations", [])],
            meta=dict(d.get("meta", {})),
            panels=[cls.from_dict(p) for p in d.get("panels", [])],
            layout=Layout.from_dict(d["layout"]) if d.get("layout") is not None else None,
            animation=Animation.from_dict(d["animation"])
            if d.get("animation") is not None
            else None,
            _theme=Theme.from_dict(d["theme"]) if d.get("theme") is not None else None,
        )


# ---------------------------------------------------------------------------
# Plottable mixin
# ---------------------------------------------------------------------------


class Plottable:
    """Mixin giving any ``to_plot_spec()`` provider a ``.plot()`` and notebook hook.

    A class that produces a :class:`PlotSpec` only has to implement
    ``to_plot_spec(self) -> PlotSpec``; this mixin layers the rendering sugar on
    top:

    - :meth:`plot` — ``self.to_plot_spec()`` plus optional inline tweaks, sent
      to a backend.
    - ``_repr_mimebundle_`` — a notebook display hook that renders inline once a
      backend is installed, and **no-ops** until then (so importing core never
      pulls a plot library, and a result still reprs as text in plain consoles).

    Result types in :mod:`tsdynamics.analysis` inherit their ``.plot`` accessor
    from :class:`~tsdynamics.analysis._result.AnalysisResult` instead; this mixin
    is for the plain data types (e.g. :class:`~tsdynamics.data.Trajectory`) that
    are not analysis results.
    """

    def to_plot_spec(self, *args: Any, **kwargs: Any) -> PlotSpec:
        """Return the :class:`PlotSpec` describing this object.

        Subclasses must override this.  The base raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement to_plot_spec() to be Plottable."
        )

    def plot(self, backend: str | None = None, **tweaks: Any) -> Any:
        """Render this object via a backend, applying inline tweaks first.

        Tweak keywords matching a :class:`PlotSpec` tweak method
        (``xscale`` / ``yscale`` / ``zscale``, ``xlabel`` / ``ylabel`` /
        ``zlabel`` / ``title``, ``xlim`` / ``ylim`` / ``zlim``,
        ``clim`` / ``colorbar`` / ``legend``) are applied to the spec before
        rendering; any remaining keywords are forwarded to the backend.

        Parameters
        ----------
        backend : str, optional
            Renderer name; ``None`` uses the first registered backend.
        **tweaks
            Inline spec tweaks and/or backend keyword arguments.

        Returns
        -------
        Any
            Whatever the backend returns.

        Raises
        ------
        VisualizationNotInstalled
            If no rendering backend is registered.
        """
        spec = self.to_plot_spec()
        backend_kw = _apply_inline_tweaks(spec, tweaks)
        return spec.render(backend, **backend_kw)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Rich notebook display — renders inline once a backend is installed.

        Returns ``None`` (a no-op, so IPython falls back to ``__repr__``) when
        no rendering backend is registered.  This keeps notebook import of core
        plot-library-free until a viz backend ships.
        """
        if _resolve_renderers() is None:
            return None
        try:  # pragma: no cover - exercised only once a backend is installed
            return self.plot()
        except Exception:  # pragma: no cover - never break repr on a render error
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maps inline ``.plot(...)`` keywords to (tweak-method-name, axis-or-None).
_INLINE_TWEAKS: dict[str, tuple[str, str | None]] = {
    "xscale": ("rescale", "x"),
    "yscale": ("rescale", "y"),
    "zscale": ("rescale", "z"),
    "xlabel": ("relabel", "x"),
    "ylabel": ("relabel", "y"),
    "zlabel": ("relabel", "z"),
    "title": ("relabel", None),
    "xlim": ("limits", "x"),
    "ylim": ("limits", "y"),
    "zlim": ("limits", "z"),
    "xticks": ("ticks", "x"),
    "yticks": ("ticks", "y"),
    "zticks": ("ticks", "z"),
}

# Inline ``.plot(...)`` keywords routed through :meth:`PlotSpec.colorize`.
_COLORIZE_TWEAKS = frozenset({"clim", "colorbar", "legend"})


def _apply_inline_tweaks(spec: PlotSpec, tweaks: dict[str, Any]) -> dict[str, Any]:
    """Apply recognized inline tweaks to ``spec``; return the leftover kwargs.

    Mutates ``spec`` in place and returns the keyword arguments that were *not*
    consumed (to forward to the backend renderer).
    """
    backend_kw: dict[str, Any] = {}
    for key, value in tweaks.items():
        if key in _COLORIZE_TWEAKS:
            spec.colorize(**{key: value})
            continue
        spec_key = _INLINE_TWEAKS.get(key)
        if spec_key is None:
            backend_kw[key] = value
            continue
        method, axis = spec_key
        if axis is None:  # title=
            spec.relabel(title=value)
        else:
            getattr(spec, method)(**{axis: value})
    return backend_kw


def _jsonify(value: Any) -> Any:
    """Recursively coerce a value to JSON-friendly types (arrays → lists).

    Deliberately not shared with :func:`tsdynamics.analysis._result._jsonify`
    (a strict superset that also handles sets, nested ``AnalysisResult`` /
    dataclasses, and SciPy sparse): viz is the lower layer in the IR seam and
    must not import the analysis layer, so this minimal copy stays here.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _resolve_renderers() -> Any | None:
    """Return the renderer registry if it exists and is non-empty, else ``None``.

    The renderer registry (``tsdynamics.registry.renderers``) and the rendering
    backends are added by later visualization streams.  Resolving it lazily and
    defensively keeps this module self-contained today while wiring itself up
    automatically the moment a backend lands.
    """
    try:
        from tsdynamics.registry import renderers
    except Exception:  # pragma: no cover - registry has no renderers yet
        return None
    try:
        return renderers if len(renderers) else None
    except Exception:  # pragma: no cover - defensive
        return None


def _visualization_not_installed() -> Exception:
    """Build the no-backend error, reusing the canonical type if it exists.

    The canonical :class:`VisualizationNotInstalled` lives in
    :mod:`tsdynamics.analysis._result` (stream WS-RESULT).  Import it lazily so
    this module has no hard dependency on the analysis layer; fall back to a
    plain :class:`ImportError` with the same message if it is unavailable.
    """
    msg = (
        "No visualization backend is registered. Visualization is deferred in "
        "this release: export the spec with .to_dict() and render it yourself, "
        "or install a backend once one is available."
    )
    try:
        from tsdynamics.analysis._result import VisualizationNotInstalled
    except Exception:  # pragma: no cover - analysis layer unavailable
        return ImportError(msg)
    return VisualizationNotInstalled(msg)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
