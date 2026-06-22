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

__all__ = [
    "Annotation",
    "Axis",
    "Colorbar",
    "Layer",
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
    BIFURCATION = "bifurcation"
    ORBIT_DIAGRAM = "orbit_diagram"
    COBWEB = "cobweb"
    RETURN_MAP = "return_map"
    POINCARE_SECTION = "poincare_section"
    BASINS_IMAGE = "basins_image"
    RECURRENCE_PLOT = "recurrence_plot"
    POWER_SPECTRUM = "power_spectrum"
    SCALING_FIT = "scaling_fit"
    DIAGNOSTIC_CURVE = "diagnostic_curve"
    LINE_FAMILY = "line_family"
    HISTOGRAM_NULL = "histogram_null"
    # animation kinds are the same spec + a leading ``frame`` axis on the data:
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


# Type aliases for the public tweak API (one spelling each).
_Scale = Literal["linear", "log", "symlog"]
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
    scale : {"linear", "log", "symlog"}, optional
        The axis scale.  Default ``"linear"``.
    limits : tuple of float, optional
        ``(lo, hi)`` view limits, or ``None`` to auto-scale.
    ticks : sequence of float, optional
        Explicit tick locations, or ``None`` to auto-tick.
    tickformat : str, optional
        A backend-neutral format string for tick labels, or ``None``.
    """

    label: str = ""
    scale: _Scale = "linear"
    limits: tuple[float, float] | None = None
    ticks: Sequence[float] | None = None
    tickformat: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this axis."""
        return {
            "label": self.label,
            "scale": self.scale,
            "limits": list(self.limits) if self.limits is not None else None,
            "ticks": [float(t) for t in self.ticks] if self.ticks is not None else None,
            "tickformat": self.tickformat,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Axis:
        """Rebuild an :class:`Axis` from :meth:`to_dict` output."""
        limits = d.get("limits")
        ticks = d.get("ticks")
        return cls(
            label=d.get("label", ""),
            scale=d.get("scale", "linear"),
            limits=tuple(limits) if limits is not None else None,
            ticks=list(ticks) if ticks is not None else None,
            tickformat=d.get("tickformat"),
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
    """

    label: str = ""
    location: _CbarLoc = "right"
    ticks: Sequence[float] | None = None
    tickformat: str | None = None
    show: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this colorbar."""
        return {
            "label": self.label,
            "location": self.location,
            "ticks": [float(t) for t in self.ticks] if self.ticks is not None else None,
            "tickformat": self.tickformat,
            "show": bool(self.show),
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
    """

    show: bool = True
    location: _LegendLoc = "best"
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this legend."""
        return {
            "show": bool(self.show),
            "location": self.location,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Legend:
        """Rebuild a :class:`Legend` from :meth:`to_dict` output."""
        return cls(
            show=bool(d.get("show", True)),
            location=d.get("location", "best"),
            title=d.get("title", ""),
        )


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
        Channel name → array.  Conventional channels are ``"x"``, ``"y"``,
        ``"z"`` (coordinates), ``"c"`` (a color/scalar field), ``"u"`` / ``"v"``
        (vector components for ``QUIVER``), and ``"frame"`` (a leading animation
        axis).  Inputs are coerced to :class:`numpy.ndarray` on construction.
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
        ``meta["animate"] = {"fps": 30}`` for an animated spec).

    Notes
    -----
    The tweak methods (:meth:`relabel`, :meth:`rescale`, :meth:`limits`,
    :meth:`ticks`, :meth:`style`) mutate the spec in place and return ``self``,
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

    def __post_init__(self) -> None:
        """Normalize ``kind`` and the optional ``clim`` color range."""
        self.kind = PlotKind(self.kind)
        self.clim = _as_pair(self.clim)

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

    def style(self, *, layer: int | None = None, **kw: Any) -> PlotSpec:
        """Merge backend-neutral style keys into one layer or every layer.

        Parameters
        ----------
        layer : int, optional
            Index of the layer to style.  If ``None`` (default), the style is
            merged into *every* layer.
        **kw
            Style keys to set (``color``, ``cmap``, ``lw``, ``alpha``,
            ``marker``, …).

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        targets = self.layers if layer is None else [self.layers[layer]]
        for lyr in targets:
            lyr.style.update(kw)
        return self

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

    # -- color / legend completeness ---------------------------------------

    _COLOR_KINDS: ClassVar[frozenset[PlotKind]] = frozenset(
        {
            PlotKind.IMAGE,
            PlotKind.SURFACE3D,
            PlotKind.BASINS_IMAGE,
            PlotKind.RECURRENCE_PLOT,
            PlotKind.SPACETIME,
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

        Backends are renderers registered in ``tsdynamics.registry.renderers``
        (added by a later visualization stream) and discovered the same way as
        analyses and transforms.  Until a backend registers, this raises
        :class:`VisualizationNotInstalled`.

        Parameters
        ----------
        backend : str, optional
            The renderer name (e.g. ``"matplotlib"``).  If ``None``, the first
            registered backend is used.
        **backend_kw
            Forwarded to the renderer callable.

        Returns
        -------
        Any
            Whatever the backend returns (e.g. a figure handle).

        Raises
        ------
        VisualizationNotInstalled
            If no rendering backend is registered.
        """
        renderers = _resolve_renderers()
        if renderers is None:
            raise _visualization_not_installed()
        if backend is None:
            backend = renderers.names()[0]
        renderer = renderers.get(backend)
        return renderer(self, **backend_kw)

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
