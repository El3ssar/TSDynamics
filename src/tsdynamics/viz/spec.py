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
- :class:`Layer` — one drawable layer: a :class:`PlotKind` *mark* + a
  channel-name → :class:`numpy.ndarray` data mapping + neutral style keys.
- :class:`PlotSpec` — the top-level spec: a semantic :class:`PlotKind`, a list of
  :class:`Layer`, typed ``x`` / ``y`` / optional ``z`` axes, title, ndim,
  aspect, annotations, and provenance ``meta``.  Tweak methods
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
from typing import Any, Literal

import numpy as np

__all__ = [
    "Annotation",
    "Axis",
    "Layer",
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
            limits=tuple(limits) if limits is not None else None,  # type: ignore[arg-type]
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
            span=tuple(span) if span is not None else None,  # type: ignore[arg-type]
            axis=d.get("axis", "x"),
            style=dict(d.get("style", {})),
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
    title: str = ""
    ndim: _Ndim = 2
    aspect: _Aspect = "auto"
    annotations: list[Annotation] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize ``kind`` to a :class:`PlotKind` member."""
        self.kind = PlotKind(self.kind)

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
        return cls(
            kind=PlotKind(d["kind"]),
            layers=[Layer.from_dict(lyr) for lyr in d.get("layers", [])],
            x=Axis.from_dict(d["x"]) if d.get("x") is not None else Axis(),
            y=Axis.from_dict(d["y"]) if d.get("y") is not None else Axis(),
            z=Axis.from_dict(z) if z is not None else None,
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
        ``zlabel`` / ``title``, ``xlim`` / ``ylim`` / ``zlim``) are applied to
        the spec before rendering; any remaining keywords are forwarded to the
        backend.

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


def _apply_inline_tweaks(spec: PlotSpec, tweaks: dict[str, Any]) -> dict[str, Any]:
    """Apply recognized inline tweaks to ``spec``; return the leftover kwargs.

    Mutates ``spec`` in place and returns the keyword arguments that were *not*
    consumed (to forward to the backend renderer).
    """
    backend_kw: dict[str, Any] = {}
    for key, value in tweaks.items():
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
    """Recursively coerce a value to JSON-friendly types (arrays → lists)."""
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
