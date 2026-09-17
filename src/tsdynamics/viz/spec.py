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
  ``__plot_spec__()`` a ``.plot(...)`` convenience and a notebook display hook.

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

import functools
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

import numpy as np

from ._frames import Frame, FrameSpace, frame_of
from ._tweaks import figure_scoped, panel_scoped, panel_scoped_custom
from .style import Theme, get_theme, normalize_style

if TYPE_CHECKING:  # pragma: no cover - typing only; resolved by ``__getattr__``
    from .export import (
        SCHEMA_VERSION as SCHEMA_VERSION,
    )
    from .export import (
        from_dict_envelope as from_dict_envelope,
    )
    from .export import (
        to_dict_envelope as to_dict_envelope,
    )
    from .transforms import (
        Geometry as Geometry,
    )
    from .transforms import (
        Part as Part,
    )
    from .transforms import (
        PlotTransform as PlotTransform,
    )
    from .transforms import (
        Presentation as Presentation,
    )
    from .transforms import (
        T as T,
    )
    from .transforms import (
        make_frame as make_frame,
    )

#: ``ts.viz.spec`` — the IR sub-namespace (contract §2.7).  Nineteen nouns you
#: only ever *receive*: a renderer author reads them, a transform author never
#: needs one (both extension doors take plain mappings).  ``Plot`` itself is
#: deliberately **absent** — it is the one type you annotate, and it lives one
#: level up at :data:`tsdynamics.viz.Plot`.
#:
#: Ten of the nineteen are owned by sibling modules (``viz._frames``,
#: ``viz.export``, ``viz.transforms``) which import *this* module; they are
#: re-exported through the module :func:`__getattr__` below so the cycle never
#: forms and ``import tsdynamics.viz.spec`` still costs nothing.
__all__ = [
    "Animation",
    "Annotation",
    "Axis",
    "Colorbar",
    "Frame",
    "FrameSpace",
    "Geometry",
    "Layer",
    "Layout",
    "Legend",
    "Part",
    "PlotKind",
    "PlotTransform",
    "Presentation",
    "SCHEMA_VERSION",
    "T",
    "from_dict_envelope",
    "make_frame",
    "to_dict_envelope",
]

#: ``name -> module`` for the ten IR nouns this module re-exports lazily.  Each
#: owner imports ``viz.spec``, so an eager import here would be a cycle; the
#: module :func:`__getattr__` resolves and caches them on first touch.
_LAZY_IR_NAMES: dict[str, str] = {
    "Geometry": "tsdynamics.viz.transforms",
    "Part": "tsdynamics.viz.transforms",
    "PlotTransform": "tsdynamics.viz.transforms",
    "Presentation": "tsdynamics.viz.transforms",
    "T": "tsdynamics.viz.transforms",
    "make_frame": "tsdynamics.viz.transforms",
    "SCHEMA_VERSION": "tsdynamics.viz.export",
    "to_dict_envelope": "tsdynamics.viz.export",
    "from_dict_envelope": "tsdynamics.viz.export",
}


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
    SCALING_FIT = "scaling_fit"
    DIMENSION_SPECTRUM = "dimension_spectrum"
    DIAGNOSTIC_CURVE = "diagnostic_curve"
    LINE_FAMILY = "line_family"
    ENSEMBLE_FAN = "ensemble_fan"
    LYAPUNOV_SPECTRUM = "lyapunov_spectrum"
    EIGENVALUE_PLANE = "eigenvalue_plane"
    FIXED_POINTS_OVERLAY = "fixed_points_overlay"
    VECTOR_FIELD = "vector_field"
    PHASE_PORTRAIT_FIELD = "phase_portrait_field"
    CONTINUATION = "continuation"
    CATEGORICAL_BAR = "categorical_bar"
    # ── removed in v6 (see tests/test_viz_vocab.py for the reviewed rationale) ──
    # POWER_SPECTRUM / SPECTROGRAM / HISTOGRAM_NULL / FEATURE_BARS / COMPLEXITY_CURVE
    # named analyses the v6 scope surgery deleted (transforms / entropy /
    # surrogate); TRAJECTORY_ANIMATION / ENSEMBLE_ANIMATION were superseded by the
    # orthogonal ``Animation`` modifier (PR #463).  None of the seven was ever
    # produced by any code path.

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

#: The layer marks that require 3-D drawing support (see :attr:`PlotSpec.is_three_d`).
_THREE_D_MARKS: frozenset[PlotKind] = frozenset({PlotKind.LINE3D, PlotKind.SURFACE3D})


# ---------------------------------------------------------------------------
# The save() extension contract (see :meth:`PlotSpec.save`)
# ---------------------------------------------------------------------------

#: Web-page extensions — written by a backend's own file writer (plotly / three.js),
#: which emits a CDN-referencing page instead of a bundle-inlining figure dump.
#: Kept because :meth:`Plot._write` *routes* on it, not because it decides what is
#: writable: that is each backend's own ``writes_static`` / ``writes_animated``.
_HTML_EXT: frozenset[str] = frozenset({".html", ".htm"})

#: Which backend gets first refusal for an ambiguous extension.  ``.json`` is
#: written by both ``json`` (the IR envelope, round-trippable) and ``threejs`` (a
#: BufferGeometry payload, which is not a Plot), and ``.html`` by both ``plotly``
#: (an interactive figure) and ``threejs`` (an embeddable 3-D viewer).  Anything
#: not listed prefers whatever the dispatch seats first — matplotlib.
_SAVE_PREFERENCE: dict[str, tuple[str, ...]] = {
    ".json": ("json", "threejs"),
    ".html": ("plotly", "threejs"),
    ".htm": ("plotly", "threejs"),
}


def _extension_of(path: Any) -> str:
    """Return ``path``'s lower-cased extension including the dot (``""`` if none)."""
    text = str(path)
    dot = text.rfind(".")
    slash = max(text.rfind("/"), text.rfind("\\"))
    return text[dot:].lower() if dot > slash else ""


def _registered_renderer_names() -> set[str] | None:
    """Return the registered renderer names, or ``None`` if the registry is unusable."""
    names = list(_renderer_capabilities())
    return set(names) if names else None


def _renderer_capabilities() -> dict[str, Any]:
    """``name -> capabilities`` for every registered renderer, in preference order.

    Registers the in-tree backends first, so introspection before the first
    render tells the truth (measured, it used to answer ``[]`` in a fresh session
    and the full list afterwards).  matplotlib is seated first by the dispatch
    layer, so plain iteration order **is** the default preference.
    """
    try:
        from tsdynamics import registry
        from tsdynamics.viz.render import register_builtin_renderers

        register_builtin_renderers()
        out: dict[str, Any] = {}
        for name in registry.renderers.names():
            renderer = registry.renderers.get(name)
            caps = getattr(renderer, "capabilities", None)
            out[name] = caps if caps is not None else renderer
        return out
    except Exception:  # pragma: no cover - defensive
        return {}


def _declares_save(caps: Any, ext: str, *, animated: bool) -> bool:
    """Whether ``caps`` claims it can write ``ext`` (``False`` when it declares nothing)."""
    can_save = getattr(caps, "can_save", None)
    if not callable(can_save):
        return False
    try:
        return bool(can_save(ext, animated=animated))
    except TypeError:  # a backend whose predicate predates the animated= split
        try:
            return bool(can_save(ext))
        except Exception:  # pragma: no cover - a backend's own predicate failed
            return False
    except Exception:  # pragma: no cover - a backend's own predicate failed
        return False


def _writers_for(ext: str, *, animated: bool) -> list[str]:
    """Backends declaring they write ``ext``, most-preferred first.

    **This is the whole save contract.** ``Plot.save`` asks the installed
    backends what they can write and believes them; it keeps no table of its own.
    Before v6 it kept both, and they disagreed in both directions — ``.webp`` was
    declared by matplotlib and refused, ``.pgf`` was undeclared and accepted, and
    a registered third-party backend could be rendered by name but its declared
    extension could never be saved.
    """
    caps = _renderer_capabilities()
    preferred = _SAVE_PREFERENCE.get(ext, ())
    order = [n for n in preferred if n in caps] + [n for n in caps if n not in preferred]
    return [n for n in order if _declares_save(caps[n], ext, animated=animated)]


def _writes_its_own_file(backend: str, ext: str) -> bool:
    """Whether ``backend`` should be handed ``path=`` rather than asked for a figure.

    True when the backend declares ``data_export`` or ``web_export`` — it emits a
    document, not a figure — **and** its renderer accepts a ``path`` keyword.
    Both halves come from the backend's own declaration; there is no list of
    names here, which is what lets a third-party exporter work.

    matplotlib declares neither, so it takes the figure route: ``savefig`` (with
    ``dpi``) for a still and ``FuncAnimation.save`` (with ``fps``) for a movie.
    """
    del ext
    try:
        from tsdynamics import registry
        from tsdynamics.viz.render import accepted_render_kwargs, register_builtin_renderers

        register_builtin_renderers()
        renderer = registry.renderers.get(backend)
    except Exception:  # pragma: no cover - defensive
        return False
    caps = getattr(renderer, "capabilities", None)
    if not (getattr(caps, "data_export", False) or getattr(caps, "web_export", False)):
        return False
    accepted = accepted_render_kwargs(backend, renderer)
    return accepted is None or "path" in accepted


def _accepts_path(backend: str | None) -> bool:
    """Whether ``backend``'s renderer takes a ``path=`` keyword (its own writer)."""
    if backend is None:
        return False
    try:
        from tsdynamics import registry
        from tsdynamics.viz.render import accepted_render_kwargs, register_builtin_renderers

        register_builtin_renderers()
        renderer = registry.renderers.get(backend)
    except Exception:  # pragma: no cover - defensive
        return False
    accepted = accepted_render_kwargs(backend, renderer)
    return accepted is None or "path" in accepted


def _writable_extensions(*, animated: bool) -> list[str]:
    """Every extension some installed backend declares it can write."""
    out: set[str] = set()
    for caps in _renderer_capabilities().values():
        field = "writes_animated" if animated else "writes_static"
        out |= set(getattr(caps, field, None) or getattr(caps, "writes", None) or ())
    return sorted(out)


def _writes_table(*, animated: bool) -> str:
    """``matplotlib: .png .pdf …  plotly: .html`` — who writes what, for an error."""
    parts = []
    for name, caps in _renderer_capabilities().items():
        field = "writes_animated" if animated else "writes_static"
        exts = sorted(getattr(caps, field, None) or getattr(caps, "writes", None) or ())
        if exts:
            parts.append(f"{name}: {' '.join(exts)}")
    return "; ".join(parts)


def _first_char(path: str) -> str:
    """Return the first non-whitespace character of the file at ``path`` (``""`` if none).

    The shared primitive behind the two format sniffs below.  Reading a fixed
    512-byte prefix is enough to classify every format :meth:`PlotSpec.save`
    writes, and costs nothing next to producing the file.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = fh.read(512).lstrip()
    except OSError:  # pragma: no cover - defensive
        return ""
    return head[:1]


def _looks_like_markup(path: str) -> bool:
    """Whether the file at ``path`` opens like an HTML document (``<!doctype`` / ``<``).

    A deliberately dumb sniff on the first non-whitespace character: it is enough
    to separate a real page from the JSON payload a data-export backend produces,
    and it stays true for whatever page a backend emits next (it asserts a
    *format*, not a fixed template).
    """
    return _first_char(path) == "<"


def _looks_like_json(path: str) -> bool:
    """Whether the file at ``path`` opens like a JSON document (``{`` / ``[``).

    Used to catch a *data-export* payload written under an image extension.  The
    sniff cannot false-positive on any image format this library writes: the text
    ones open with ``<`` (SVG) or ``%`` (EPS / PS / PGF), and the binary ones with
    their own magic bytes.
    """
    return _first_char(path) in ("{", "[")


# Type aliases for the public tweak API (one spelling each).  ``"categorical"``
# joins the numeric scales for a categorical axis (a CATEGORICAL_BAR x-axis whose
# tick positions index :attr:`Axis.categories`).
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
        ``CATEGORICAL_BAR`` category axis).
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

    @classmethod
    def from_mapping(cls, value: Any) -> Annotation:
        """Coerce ``value`` to an :class:`Annotation`, passing one through unchanged.

        ``Plot.annotations`` is a plain list and a plain ``dict`` appended to it
        used to reach a renderer and die there with
        ``AttributeError: 'dict' object has no attribute 'style'``.  Every
        renderer normalises through this instead, so plain Python works at that
        door too (corollary C1) — without changing the serialized schema.
        """
        if isinstance(value, Annotation):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        from tsdynamics.errors import InvalidInputError

        raise InvalidInputError(
            f"an annotation must be a mapping with a 'kind' key, not "
            f"{type(value).__name__}; build one with p.vline(x) / p.hline(y) / "
            "p.span(lo, hi) / p.text(x, y, s)."
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
        Explicit grid shape for ``mode="grid"``; ``None`` lets :meth:`grid` pick.
    share_x, share_y : bool, optional
        Whether the panels share an x / y axis (a stacked time-series column
        typically shares x).  Defaults: ``share_x`` follows the mode (stack → True),
        ``share_y`` ``False``.
    share_color : bool, optional
        Whether the panels share **one** figure-level colorbar instead of drawing
        one apiece — the right presentation for a row of basin images across a
        parameter, where per-panel colorbars repeat the same categorical scale.
        Default ``False``.
    """

    mode: Literal["stack", "row", "grid", "frames"] = "stack"
    rows: int | None = None
    cols: int | None = None
    share_x: bool = False
    share_y: bool = False
    share_color: bool = False

    def grid(self, n_panels: int) -> tuple[int, int]:
        """Resolve the ``(rows, cols)`` subplot grid for ``n_panels`` panels.

        The **one** implementation of the panel-tiling arithmetic every renderer
        needs (it lived, byte-identical, in the matplotlib, plotly and three.js
        backends).  ``"stack"`` is one column, ``"row"`` one row, and ``"grid"``
        honours whichever of :attr:`rows` / :attr:`cols` is set — filling in the
        other by ceiling division — or, with neither set, a near-square grid
        ``cols = ceil(sqrt(n))``.

        Parameters
        ----------
        n_panels : int
            How many panels to tile (at least 1).

        Returns
        -------
        tuple of int
            ``(rows, cols)``, always large enough that ``rows * cols >= n_panels``.
        """
        n = max(1, int(n_panels))
        if self.mode == "stack":
            return (n, 1)
        if self.mode == "row":
            return (1, n)
        if self.rows is not None and self.cols is not None:
            return (max(1, int(self.rows)), max(1, int(self.cols)))
        if self.cols is not None:
            cols = max(1, int(self.cols))
            return (-(-n // cols), cols)
        if self.rows is not None:
            rows = max(1, int(self.rows))
            return (rows, -(-n // rows))
        cols = int(np.ceil(np.sqrt(n)))
        return (-(-n // cols), cols)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this layout."""
        return {
            "mode": self.mode,
            "rows": self.rows,
            "cols": self.cols,
            "share_x": bool(self.share_x),
            "share_y": bool(self.share_y),
            "share_color": bool(self.share_color),
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
            share_color=bool(d.get("share_color", False)),
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
      ``__plot_spec__(kind="field", animate=...)`` (a
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

    def playback_seconds(self, n_samples: int) -> float:
        """How long a browser should take to traverse the whole series, in seconds.

        [M41] The single home of the algebra both HTML exports need.  Neither
        browser loop has a frame clock — each traverses the series in
        ``duration`` seconds at the browser's ~60 Hz — and both used to fall back
        to a hard-coded ``12.0`` whenever ``duration`` was unset, so
        ``.animate(fps=60)`` reached matplotlib and was **dropped in silence** by
        the web exports.  :meth:`frame_count` already relates the two
        (``frame_count = round(duration * fps)``), so inverting it is the
        definition: play ``frame_count(n_samples)`` frames at ``fps`` per second.
        At the defaults that is 12.0 s exactly, so no export changed speed.

        An explicit ``duration`` always wins: it is the same quantity stated
        directly.
        """
        if self.duration is not None:
            return float(self.duration)
        fps = float(self.fps)
        if fps <= 0:  # pragma: no cover - Animation validates fps > 0
            return float(self.DEFAULT_FRAMES) / 30.0
        return float(self.frame_count(int(n_samples))) / fps

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
          :attr:`Axis.categories` (a ``BAR`` / ``CATEGORICAL_BAR``).
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
    transform : str, optional
        **Provenance**: the name of the thing that produced this layer
        (``"escape_time"``, ``"streamlines"``, ``"nullclines"``, …), or ``None``
        when it is unknown.  A layer that cannot say where it came from cannot be
        addressed after the fact, which is what blocks per-source restyling
        inside an overlay (``spec.style("escape_time", cmap=...)`` rather than
        today's all-or-nothing ``.style()``), legend grouping by source, and
        telling a renderer that *this* ``IMAGE`` is a categorical basin image
        rather than a continuous spacetime one.  Purely additive: it defaults to
        ``None``, is the **last** field (so positional construction is
        unchanged), and :meth:`from_dict` accepts a payload written before it
        existed.

        .. versionadded:: 6.0

    Notes
    -----
    ``kind`` / ``data`` / ``label`` / ``style`` are positional-compatible with
    every earlier release — ``transform`` was appended, never inserted.
    """

    kind: PlotKind
    data: dict[str, np.ndarray] = field(default_factory=dict)
    label: str | None = None
    style: dict[str, Any] = field(default_factory=dict)
    transform: str | None = None

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
            "transform": self.transform,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Layer:
        """Rebuild a :class:`Layer` from :meth:`to_dict` output (lists → arrays).

        Tolerant of a payload written before ``transform`` existed (the key is
        read with a default), so old serialized specs load unchanged.
        """
        return cls(
            kind=PlotKind(d["kind"]),
            data={k: np.asarray(v) for k, v in d.get("data", {}).items()},
            label=d.get("label"),
            style=dict(d.get("style", {})),
            transform=d.get("transform"),
        )


# ---------------------------------------------------------------------------
# PlotSpec
# ---------------------------------------------------------------------------


#: The methods on :class:`Plot` that **mutate it and hand it back**.  Every one is
#: wrapped by :func:`_mutates`, which drops the cached matplotlib figure first, so
#: ``p.fig`` can never disagree with the plot it came from.  The gate
#: ``tests/test_viz_spec.py::test_every_mutating_plot_method_drops_the_figure_cache``
#: derives this set from the source and fails when a new tweak forgets.
_MUTATING_RETURNS: frozenset[str] = frozenset({"Plot", "PlotSpec", "Self"})


def _mutates[F: Callable[..., Any]](func: F) -> F:
    """Wrap a fluent tweak so it invalidates the cached figure before running.

    The rendered :class:`matplotlib.figure.Figure` behind :attr:`Plot.fig` is a
    *derived* artifact.  A cached figure that survives a later ``.style(...)`` is
    a silent-wrong-answer generator — you look at a picture that no longer
    matches the object you are holding — so the invalidation is attached at the
    definition site rather than listed in a table that drifts.

    Deliberately applied **outermost**, above ``panel_scoped`` / ``figure_scoped``,
    so the scope markers those decorators stamp survive (``functools.wraps``
    copies ``__dict__``, which is where they live).
    """

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._invalidate()
        return func(self, *args, **kwargs)

    wrapper.__tsd_mutates__ = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


@dataclass
class Plot:
    """A plot: what to draw, how it looks, and every verb that gets it on screen.

    ``ts.plot(...)``, ``traj.plot(...)``, ``system.plot(...)`` and
    ``result.plot(...)`` all return one of these, and so does every tweak on it —
    which is what makes a plot compose with itself
    (``ts.plot(ts.plot(a), ts.plot(b), layout="row")``).

    **There is no second type.** Going from the easy tier to the expert tier is
    one dot, never a rewrite::

        p = ts.plot(traj, color="crimson")   # easy
        p.ax.axvline(3.0, ls="--")           # expert: raw matplotlib Axes
        p.save("f.png")                      # ...and the library still works

    A :class:`Plot` carries everything a renderer needs and nothing it does not:
    the semantic :attr:`kind`, the drawable :attr:`layers`, the typed axes,
    and presentation metadata.  It holds **no** rendering state and imports
    **no** plotting library.

    .. versionchanged:: 6.0
        Renamed from ``PlotSpec``.  The class is unchanged — ``PlotSpec`` remains
        bound in this module as an alias so existing annotations and
        ``isinstance`` checks keep working — but the name a user sees, types and
        reads in a repr is ``Plot``.

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
    frame : Frame, optional
        What these axes *mean* — the coordinate space, its dimension, and one
        normalized name per coordinate axis.  ``None`` (default) derives it from
        :attr:`kind` plus the axis labels; read the resolved value via
        :attr:`resolved_frame`.  Overlay legality is frame compatibility, not
        kind identity, which is what lets a basin image, its attractors, a
        trajectory and the equilibria share one set of axes while an ``(x, y)``
        portrait and an ``(x, z)`` overlay are refused.

        .. versionadded:: 6.0

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

    **Composite scoping.** Every tweak is classified *panel-scoped* or
    *figure-scoped* (see :mod:`tsdynamics.viz._tweaks`), and on a
    :data:`PlotKind.COMPOSITE` a panel-scoped tweak **recurses into every
    panel** — a composite owns no axes and no layers of its own, so before v6
    all twelve of them were silent no-ops::

        ts.viz.plot(a, b, layout="stack").recolor("red", "blue").limits(x=(0, 10))

    - panel-scoped (forwarded): :meth:`relabel` (axis labels), :meth:`rescale`,
      :meth:`limits`, :meth:`ticks`, :meth:`style`, :meth:`recolor`,
      :meth:`palette`, :meth:`grid`, :meth:`font`, :meth:`colorize`,
      :meth:`autocolor`, :meth:`camera` (``elev`` / ``azim``).
    - figure-scoped (not forwarded — panels inherit or must not carry them):
      ``relabel(title=)``, :meth:`theme`, :meth:`background`, :meth:`size`,
      :meth:`animate`, :meth:`trail`, :meth:`head`, :meth:`clock`,
      ``camera(spin=)``.

    Because :meth:`palette` / :meth:`font` / ``grid(color=, alpha=)`` write a
    *theme* override onto each panel, and a panel's own theme wins over the
    composite's, set the composite theme **first**:
    ``spec.theme("dark").palette(...)``, not the other way round.  Taken in that
    order the panels inherit the composite's theme before overriding it (the
    forwarding wrapper seeds them — see ``_tweaks.panel_scoped(writes_theme=)``),
    so a dark composite stays dark.  The reverse order —
    ``spec.palette(...).theme("dark")`` — still pins a *default*-based theme on
    each panel that the later composite ``theme()`` cannot rebase, and renders
    dark figure chrome around light panel axes.  Removing that last trap needs
    panels to record theme *deltas* rather than materialised themes.
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
    # Frame: what this spec's axes *mean* (a coordinate space, its dimension, its
    # axis names).  ``None`` (the default, and what every in-tree producer leaves
    # it as today) means "derive it from the kind and the axis labels" — see
    # :attr:`resolved_frame`.  Overlay legality is frame compatibility, so a
    # producer that states its frame explicitly gets an exact answer instead of a
    # derived one.  Appended last, so positional construction is unchanged.
    frame: Frame | None = None

    def __post_init__(self) -> None:
        """Normalize ``kind`` / ``clim`` and enforce the ``COMPOSITE`` ⟺ ``panels`` invariant.

        Raises
        ------
        InvalidParameterError
            If ``kind`` is :data:`PlotKind.COMPOSITE` but ``panels`` is empty, or
            ``panels`` is non-empty but ``kind`` is not ``COMPOSITE``.
        """
        self.kind = PlotKind(self.kind)
        self.clim = _as_pair(self.clim)
        # Rendering state, deliberately NOT dataclass fields: it must not
        # serialize, must not take part in ``==`` / ``replace()``, and must not
        # survive a round trip through ``to_dict``.
        self._figure_cache: tuple[Any, list[Any]] | None = None
        self._figure_handed_out = False
        self._handout_warned = False
        self._check_composite_invariant()

    # -- the matplotlib escape hatch ---------------------------------------

    def _invalidate(self) -> None:
        """Drop the cached figure, warning once if the caller is holding it.

        Two rules meet here and both matter:

        *Never show a stale picture.* Any tweak makes the rendered figure wrong,
        so the cache goes.

        *Never silently destroy your work.* If :attr:`fig` / :attr:`ax` /
        :attr:`axes` already handed the figure out, the caller may have drawn on
        it by hand — and re-rendering throws those edits away.  That gets exactly
        one :class:`~tsdynamics.viz.render.caps.VisualizationDegraded` warning,
        naming the rule: **library tweaks first, matplotlib last.**
        """
        if getattr(self, "_figure_cache", None) is None:
            return
        if self._figure_handed_out and not self._handout_warned:
            self._handout_warned = True
            warnings.warn(
                "this tweak re-renders the Plot; edits you made on the Figure returned "
                "by .fig / .ax / .axes are not part of the Plot and will be lost. Do the "
                "library tweaks first and matplotlib last -- or keep the artifact: "
                "fig = p.fig  (after the tweaks).",
                _degraded_warning(),
                stacklevel=3,
            )
        self._figure_cache = None
        self._figure_handed_out = False

    def _rendered(self) -> tuple[Any, list[Any]]:
        """Render once through matplotlib and cache ``(figure, axes-in-panel-order)``."""
        cached: tuple[Any, list[Any]] | None = getattr(self, "_figure_cache", None)
        if cached is not None:
            return cached
        target: Plot = self
        if self.is_animated:
            warnings.warn(
                ".fig is a still of the final frame; the animation is written by "
                ".save('f.gif' / 'f.mp4' / 'f.html').",
                _degraded_warning(),
                stacklevel=3,
            )
            target = Plot.from_dict({**self.to_dict(), "animation": None})
        result = target.render("matplotlib")
        figure = getattr(result, "figure", result)
        axes = list(getattr(figure, "axes", []) or [])
        self._figure_cache = (figure, axes)
        return self._figure_cache

    @property
    def fig(self) -> Any:
        """The matplotlib :class:`~matplotlib.figure.Figure` for this plot.

        **The escape hatch.** Everything matplotlib can do is one dot away, with
        no rewrite and no change of type — the :class:`Plot` you were holding is
        the :class:`Plot` you are still holding::

            p = ts.plot(traj, color="crimson")
            p.fig.suptitle("run 4")
            p.save("f.png")

        Rendering is lazy (nothing is drawn until you ask) and cached, so
        ``p.fig is p.fig``.  Any later tweak drops the cache — and warns once if
        you are holding the figure, because re-rendering would discard your hand
        edits.  **Library tweaks first, matplotlib last.**

        Raises
        ------
        VisualizationNotInstalled
            If no rendering backend is installed.
        """
        figure = self._rendered()[0]
        # Arm the guard, exactly as ``.ax`` and ``.axes`` do.  Without this the
        # whole ``_invalidate`` mechanism was DEAD for the spelling this
        # docstring puts first: hand-editing ``p.fig`` and then calling
        # ``p.style(...)`` discarded the edit in silence.
        self._figure_handed_out = True
        return figure

    @property
    def ax(self) -> Any:
        """The single matplotlib :class:`~matplotlib.axes.Axes` this plot draws on.

        Raises
        ------
        tsdynamics.errors.InvalidParameterError
            On a composite — a multi-panel figure has no single axes.  Use
            :attr:`axes` for the list, or ``p.panels[i].ax``.
        """
        from tsdynamics.errors import InvalidParameterError

        if self.is_composite:
            n = len(self.panels)
            raise InvalidParameterError(
                f"this Plot has {n} panel{'s' if n != 1 else ''}; use .axes for the "
                "list, or .panels[i].ax."
            )
        axes = self._rendered()[1]
        if not axes:  # pragma: no cover - a backend that drew no axes
            raise InvalidParameterError(
                "the matplotlib backend produced no axes for this Plot; use .fig."
            )
        self._figure_handed_out = True
        return axes[0]

    @property
    def axes(self) -> list[Any]:
        """Every matplotlib Axes, in panel order (a single-panel plot gives a 1-list).

        ``p.axes[3].set_yscale("log")`` is the per-panel escape hatch::

            p = ts.plot(a, b, c, d, layout="grid", cols=2)
            p.axes[3].set_yscale("log")
        """
        axes = self._rendered()[1]
        self._figure_handed_out = True
        return list(axes)

    def _check_composite_invariant(self) -> None:
        """Raise unless ``kind is COMPOSITE`` and ``panels`` non-empty agree.

        A composite *is* its panels.  Letting the two drift apart produced the
        two failure modes this guard closes: a ``COMPOSITE`` with no panels
        rendered as a blank figure (which is how ``__plot_spec__(kind="composite")``
        silently threw a trajectory away and saved an empty PNG), and a
        panel-bearing spec labelled something else had its panels ignored by every
        renderer.  Both are silent — hence a construction-time error, not a
        render-time one.
        """
        from tsdynamics.errors import InvalidParameterError

        if self.kind is PlotKind.COMPOSITE and not self.panels:
            raise InvalidParameterError(
                "a COMPOSITE PlotSpec must carry at least one panel; build one with "
                "tsdynamics.viz.plot(..., layout='stack'/'row'/'grid'). "
                "(A composite is its panels — an empty one renders as a blank figure.)"
            )
        if self.panels and self.kind is not PlotKind.COMPOSITE:
            raise InvalidParameterError(
                f"a PlotSpec with panels must have kind=COMPOSITE, got {self.kind.value!r}; "
                "every renderer dispatches on the kind, so these panels would be ignored."
            )

    @property
    def is_composite(self) -> bool:
        """Whether this spec is a multi-panel composite (has child ``panels``).

        By the construction invariant the two halves are equivalent — a
        ``COMPOSITE`` kind always has panels and vice versa.
        """
        return bool(self.panels) or self.kind == PlotKind.COMPOSITE

    @property
    def is_animated(self) -> bool:
        """Whether this spec carries an :class:`Animation` directive."""
        return self.animation is not None

    @property
    def is_three_d(self) -> bool:
        """Whether this spec draws in 3-D (the one definition every backend shares).

        True when the spec declares ``ndim == 3``, carries a ``z`` axis, or holds
        any 3-D mark (``LINE3D`` / ``SURFACE3D``) — and, for a composite, when
        **any** panel is 3-D.

        This predicate lived in three byte-identical copies (the capability
        check, the matplotlib renderer and the plotly renderer).  It is a property
        of the *spec*, so it belongs here; a renderer that disagreed with the
        capability check about what "3-D" means is a dispatch bug waiting to
        happen.  The single-panel answer is deliberately **identical** to those
        copies (the panel recursion is the only addition — it is what the copies
        got by being called once per panel).
        """
        if self.panels:
            return any(p.is_three_d for p in self.panels)
        if self.ndim == 3 or self.z is not None:
            return True
        return any(lyr.kind in _THREE_D_MARKS for lyr in self.layers)

    @property
    def resolved_frame(self) -> Frame:
        """The :class:`~tsdynamics.viz._frames.Frame` this spec draws in.

        The spec's own :attr:`frame` when it declares one, else derived from the
        semantic :attr:`kind` and the axis labels (see
        :func:`tsdynamics.viz._frames.frame_of`) — so every spec, including a
        hand-built one and every spec the library builds today, has a frame.

        Two specs may share one set of axes iff their frames are *compatible*
        (:meth:`~tsdynamics.viz._frames.Frame.compatible_with`); that is the one
        overlay rule, used by :func:`tsdynamics.viz.plot`, :meth:`add`, and
        :meth:`tsdynamics.analysis.AnalysisResult.overlay_on` alike.

        Raises
        ------
        tsdynamics.errors.InvalidParameterError
            If this is a composite (a multi-panel figure owns no single axes).
        """
        return frame_of(self)

    # NOTE: ``add`` is deliberately **outside** the panel/figure tweak partition
    # (:mod:`tsdynamics.viz._tweaks`) and returns ``Self`` rather than
    # ``PlotSpec``.  That partition exists to force a forwarding decision for
    # tweaks that would otherwise be *silent* no-ops on a composite; ``add`` is
    # not a tweak — it is the incremental form of ``tsdynamics.viz.plot``, it
    # describes neither a panel's presentation nor the figure's, and on a
    # composite it **raises** (adding one thing to every panel is not what anyone
    # means).  ``Self`` is also simply the truer annotation: it returns *this*
    # object, not some ``PlotSpec``.
    @_mutates
    def add(self, *things: Any, on: str | None = None, **build_kw: Any) -> Self:
        """Overlay more things onto this spec **in place**, and return it.

        The incremental half of the composition API (Makie's ``plot!`` in a
        chainable spelling): where :func:`tsdynamics.viz.plot` composes in one
        call, :meth:`add` grows a figure a piece at a time, and — like every
        other tweak — it mutates and returns ``self``, so it chains::

            (ts.viz.plot(basins)
                .add(attractors)
                .add(traj)
                .add(fps)
                .theme("publication")
                .save("fig.pdf"))

        The result is **identical** to passing everything to
        :func:`~tsdynamics.viz.plot` at once, because both go through the same
        merge: the same frame check, and the same z-ordering *by role* (fields
        under curves under markers), so moving a thing **between** roles — adding
        the basin image first or last — does not change the picture.  The sort is
        stable, so things of the *same* role (two curves, say) keep the order they
        were added in, exactly as they keep their argument order in
        :func:`~tsdynamics.viz.plot`.

        Parameters
        ----------
        *things
            Anything :func:`tsdynamics.viz.plot` accepts — a
            :class:`~tsdynamics.data.Trajectory`, a system, an analysis result,
            or an already-built :class:`PlotSpec`.
        on : {"force"}, optional
            ``"force"`` overlays a deliberate frame mismatch with a one-time
            :class:`~tsdynamics.viz.render.caps.VisualizationDegraded` warning
            instead of raising.
        **build_kw
            Forwarded to each non-spec thing's ``__plot_spec__`` (``components``
            / ``kind`` / per-kind options), exactly as in
            :func:`tsdynamics.viz.plot`.

        Returns
        -------
        PlotSpec
            ``self``, with the new layers merged in.

        Raises
        ------
        tsdynamics.errors.InvalidParameterError
            If ``self`` is a composite (add to one of its ``panels`` instead), if
            a thing's frame is incompatible and ``on`` is not ``"force"``, or if
            ``on`` is neither ``None`` nor ``"force"``.
        """
        from tsdynamics.errors import InvalidParameterError

        from .compose import _merge_into, _overlay, _to_spec

        if self.is_composite:
            raise InvalidParameterError(
                "cannot add() to a COMPOSITE figure: it owns no axes of its own. "
                "Add to one of its panels (spec.panels[i].add(...)), or compose the "
                "panel first and arrange with tsdynamics.viz.plot(..., layout=...)."
            )
        if not things:
            raise InvalidParameterError("add() needs at least one thing to add.")
        specs = [_to_spec(thing, build_kw) for thing in things]
        _merge_into(self, _overlay([self, *specs], on=on))
        return self

    def resolved_panels(self) -> list[PlotSpec]:
        """Return this composite's panels with the figure-level context pushed down.

        The composite's :attr:`_theme` and :attr:`animation` are *inherited* by a
        panel that declares none of its own — the documented resolution order
        (``panel.theme or composite.theme or get_theme(None)``, and one lockstep
        master clock).  Every renderer used to hand-copy that inheritance at its
        own composite site; this is the single implementation they iterate.

        The panels are **copies** (``dataclasses.replace``), so pushing the
        context down never mutates the spec the caller holds.  A non-composite
        spec returns ``[]``.

        Returns
        -------
        list of PlotSpec
            One resolved panel per :attr:`panels` entry, in order.
        """
        from dataclasses import replace as _dc_replace

        out: list[PlotSpec] = []
        for panel in self.panels:
            theme = panel._theme if panel._theme is not None else self._theme
            animation = panel.animation if panel.animation is not None else self.animation
            out.append(_dc_replace(panel, _theme=theme, animation=animation))
        return out

    # -- uniform, backend-independent tweaks (mutate + return self) ---------

    @_mutates
    @panel_scoped(figure_only=("title",))
    def relabel(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        title: str | None = None,
    ) -> Plot:
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

    @_mutates
    @panel_scoped()
    def rescale(
        self,
        *,
        x: _Scale | None = None,
        y: _Scale | None = None,
        z: _Scale | None = None,
    ) -> Plot:
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

    @_mutates
    @panel_scoped()
    def limits(
        self,
        *,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
    ) -> Plot:
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

    @_mutates
    @panel_scoped()
    def ticks(
        self,
        *,
        x: Sequence[float] | None = None,
        y: Sequence[float] | None = None,
        z: Sequence[float] | None = None,
    ) -> Plot:
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

    @_mutates
    @panel_scoped()
    def style(
        self,
        *which: str,
        layer: int | None = None,
        axes: bool | None = None,
        **kw: Any,
    ) -> Plot:
        """Merge backend-neutral style keys into the layers you name (or all of them).

        Parameters
        ----------
        *which : str
            Name the layers to restyle.  A name matches a layer's producing
            **transform** (the provenance every primitive stamps) or its legend
            **label** — which is how you address one source inside an overlay::

                p = ts.plot(vdp, "flow_speed", "streamlines", "nullclines")
                p.style("flow_speed", alpha=0.35)
                p.style("nullclines", color="white", linewidth=1.2)
                p.style("VanDerPol (2)", color="crimson")     # by legend label

            With no names the style lands on every layer, exactly as before.
            A name that matches nothing raises, naming what this plot does have —
            silently styling nothing is the failure mode this argument exists to
            end.
        layer : int, optional
            Index of the layer to style.  If ``None`` (default), the style is
            merged into every *matched* layer.
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
        Plot
            ``self``, for chaining.

        Raises
        ------
        tsdynamics.errors.InvalidParameterError
            If a name in ``which`` matches no layer of this plot.
        """
        if axes is not None:
            self.meta["axes_visible"] = bool(axes)
        canon = normalize_style(kw)
        if layer is not None:
            targets = [self.layers[layer]]
        elif which:
            targets = [lyr for lyr in self.layers if lyr.transform in which or lyr.label in which]
            if not targets and not self.panels:
                self._raise_unknown_layer_names(which)
        else:
            targets = self.layers
        for lyr in targets:
            lyr.style.update(canon)
        return self

    def _raise_unknown_layer_names(self, which: tuple[str, ...]) -> None:
        """Report ``style(*which)`` names that address no layer, listing the real ones."""
        from tsdynamics.errors import InvalidParameterError

        known = sorted(
            {lyr.transform for lyr in self.layers if lyr.transform}
            | {lyr.label for lyr in self.layers if lyr.label}
        )
        have = ", ".join(repr(k) for k in known) if known else "no named layers"
        names = ", ".join(repr(w) for w in which)
        raise InvalidParameterError(
            f"style({names}) matched no layer; this Plot has {have}. "
            "A name matches a layer's producing transform or its legend label."
        )

    @_mutates
    @panel_scoped_custom
    def recolor(self, *colors: str, layer: int | None = None) -> Plot:
        """Assign explicit colors to layers (the per-layer color shorthand).

        Parameters
        ----------
        On a **composite** the unit of colour is a *panel*, not a layer: colour
        ``i`` goes to panel ``i`` (cycling), so
        ``ts.viz.plot(a, b, layout="stack").recolor("red", "blue")`` gives a red
        figure on top of a blue one — the obvious reading, and the only one that
        can be expressed (the composite owns no layers of its own).  Pass
        ``layer=`` to address a layer *within* every panel instead.

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
        if self.panels:
            n_c = len(colors)
            for i, panel in enumerate(self.panels):
                if layer is not None:
                    panel.recolor(colors[i % n_c], layer=layer)
                else:
                    panel.recolor(colors[i % n_c])
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

    @_mutates
    @figure_scoped
    def theme(self, theme: str | Theme | None = None, /, **overrides: Any) -> Plot:
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

    @_mutates
    @panel_scoped(writes_theme=True)
    def palette(self, *colors: str | Sequence[str]) -> Plot:
        """Set the theme's color cycle (a named palette or explicit colours).

        Both spellings work, because its sibling ``recolor(*colors)`` already
        took loose colours and two spellings for "just the colours" that
        disagreed is the defect, not the flexibility::

            p.palette("#111", "#e63946")      # loose — the documented line
            p.palette(["#111", "#e63946"])    # one sequence
            p.palette("dark")                 # a registered theme's palette

        Parameters
        ----------
        *colors : str or sequence of str
            A registered theme name (its palette), an explicit sequence of
            colors, or the colors themselves.

        Returns
        -------
        PlotSpec
            ``self``, for chaining.
        """
        from tsdynamics.errors import InvalidParameterError

        from .style import resolve_palette

        if not colors:
            raise InvalidParameterError(
                "palette() needs at least one colour (or one theme name): "
                'p.palette("#111", "#e63946")  /  p.palette("dark")'
            )
        spec_arg: str | Sequence[str]
        if len(colors) == 1:
            spec_arg = colors[0]
        else:
            flat: list[str] = []
            for c in colors:
                flat.extend([c] if isinstance(c, str) else list(c))
            spec_arg = flat
        base = self._theme if self._theme is not None else get_theme()
        self._theme = base.merged(palette=resolve_palette(spec_arg))
        return self

    # -- composition operators (§6.3) --------------------------------------

    def __add__(self, other: Any) -> Plot:
        """Overlay ``b`` onto a COPY of ``a`` — the ``a + b`` spelling.

        The operators compose without mutating either operand, so the closure
        property holds for expressions as well as for calls::

            ts.plot(traj, "phase_portrait") + ts.viz.draw({"x": xs, "y": ys}, "line")
        """
        return self._compose(other, layout="overlay")

    def __or__(self, other: Any) -> Plot:
        """Put ``b`` beside ``a`` (a one-row grid) — the ``a | b`` spelling."""
        return self._compose(other, layout="row")

    def __truediv__(self, other: Any) -> Plot:
        """Put ``b`` below ``a`` (a one-column stack) — the ``a / b`` spelling."""
        return self._compose(other, layout="stack")

    def _compose(self, other: Any, *, layout: str) -> Plot:
        """Build ``plot(self, other, layout=...)`` without mutating either side."""
        if not isinstance(other, Plot):
            return cast("Plot", NotImplemented)
        import copy as _copy

        from .compose import plot as _plot

        left, right = _copy.deepcopy(self), _copy.deepcopy(other)
        left._figure_cache = right._figure_cache = None
        left._figure_handed_out = right._figure_handed_out = False
        composed: Plot = _plot(left, right, layout=layout)
        return composed

    @_mutates
    @panel_scoped(writes_theme=True)
    def gridlines(
        self,
        show: bool = True,
        *,
        axis: Literal["x", "y", "both"] = "both",
        color: str | None = None,
        alpha: float | None = None,
    ) -> Plot:
        """Toggle / style gridlines on the chosen axis (or both).

        .. versionchanged:: 6.0
            Renamed from ``grid``.  One attribute cannot mean two things: the
            *panel arranger* is the module-level :func:`tsdynamics.viz.grid`, and
            a ``Plot.grid`` that toggled gridlines next to a ``ts.viz.grid`` that
            tiles panels is the silent-wrong-answer defect this release is
            closing everywhere else.  ``p.grid`` now names ``gridlines``.

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

    @_mutates
    @panel_scoped(writes_theme=True)
    def font(self, family: str | None = None, size: float | None = None) -> Plot:
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

    @_mutates
    @figure_scoped
    def background(self, color: str) -> Plot:
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

    @_mutates
    @figure_scoped
    def size(
        self,
        width: float | None = None,
        height: float | None = None,
        dpi: float | None = None,
    ) -> Plot:
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

    # -- annotate ----------------------------------------------------------

    def _annotate(self, kind: str, style: dict[str, Any], **fields: Any) -> Plot:
        """Append one normalised :class:`Annotation` and return ``self``."""
        self.annotations.append(
            Annotation(kind=kind, style=normalize_style(style), **fields)  # type: ignore[arg-type]
        )
        return self

    @_mutates
    @panel_scoped()
    def vline(self, x: float | Sequence[float], *, label: str = "", **style: Any) -> Plot:
        """Draw a vertical reference line at ``x`` (or one at each ``x``).

        ``Annotation`` used to be exported *because a signature demanded it* — a
        C1 signature bug by the project's own rule.  These four verbs take plain
        Python instead::

            p.vline(3.0, color="crimson", linestyle="dashed")
            p.vline([3.0, 3.449, 3.544], label="onsets")     # a whole cascade

        Parameters
        ----------
        x : float or sequence of float
            Where to draw.  A sequence draws one line per value.
        label : str, optional
            Legend / annotation text; only the first line of a sequence carries it.
        **style
            Style keys, canonicalised by
            :func:`~tsdynamics.viz.style.normalize_style` exactly as ``.style()``
            does — so ``ls="--"`` and ``linestyle="dashed"`` mean the same thing
            here too.
        """
        values = [x] if isinstance(x, (int, float, np.number)) else list(x)
        for i, value in enumerate(values):
            self._annotate("vline", style, x=float(value), text=label if i == 0 else "")
        return self

    @_mutates
    @panel_scoped()
    def hline(self, y: float | Sequence[float], *, label: str = "", **style: Any) -> Plot:
        """Draw a horizontal reference line at ``y`` (or one at each ``y``).

        See :meth:`vline` — the same shape, the other axis.
        """
        values = [y] if isinstance(y, (int, float, np.number)) else list(y)
        for i, value in enumerate(values):
            self._annotate("hline", style, y=float(value), text=label if i == 0 else "")
        return self

    @_mutates
    @panel_scoped()
    def span(
        self,
        lo: float,
        hi: float,
        *,
        axis: Literal["x", "y"] = "x",
        label: str = "",
        **style: Any,
    ) -> Plot:
        """Shade the band between ``lo`` and ``hi`` along ``axis``.

        The fit-region shading on a scaling plot::

            p.span(1.2, 3.4, alpha=0.15, color="0.4", label="fit region")
        """
        return self._annotate("span", style, span=(float(lo), float(hi)), axis=axis, text=label)

    @_mutates
    @panel_scoped()
    def text(self, x: float, y: float, s: str, **style: Any) -> Plot:
        """Place the label ``s`` at data coordinates ``(x, y)``."""
        return self._annotate("text", style, x=float(x), y=float(y), text=str(s))

    @_mutates
    @panel_scoped()
    def colorize(
        self,
        *,
        clim: tuple[float, float] | None = None,
        cmap: str | None = None,
        norm: _Norm | None = None,
        discrete: bool | None = None,
        colorbar: Colorbar | bool | None = None,
        legend: Legend | bool | None = None,
    ) -> Plot:
        """Set the color range, colorbar, and/or legend (only the args you pass).

        Like the other tweaks this mutates the spec and returns ``self`` so it
        chains.  Each keyword defaults to the sentinel "leave unchanged".

        Parameters
        ----------
        clim : tuple of float, optional
            ``(vmin, vmax)`` color range, coerced to floats.  Leaves
            :attr:`clim` unchanged when omitted.
        cmap : str, optional
            Colormap name for the color channel (``"viridis"``, ``"tab20"``, …).
            Written onto the spec's :class:`Colorbar` — **creating** one if the
            spec has none, because a colormap is meaningless without the color
            legend that documents it.  This is the spelling every caller reaches
            for first; before v6 it raised ``TypeError``.
        norm : {"linear", "log", "symlog"}, optional
            How the color *data* maps onto the colormap.
        discrete : bool, optional
            Whether the color channel is categorical (one swatch per label — a
            basin / attractor index image) rather than a continuous ramp.
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
        if cmap is not None or norm is not None or discrete is not None:
            if self.colorbar is None:
                self.colorbar = Colorbar()
            if cmap is not None:
                self.colorbar.cmap = cmap
            if norm is not None:
                self.colorbar.norm = norm
            if discrete is not None:
                self.colorbar.discrete = bool(discrete)
        return self

    # -- animation tweaks (mutate + return self; also turn animation on) ----

    def _ensure_animation(self) -> Animation:
        """Return this spec's :class:`Animation`, creating a default if absent."""
        if self.animation is None:
            self.animation = Animation()
        return self.animation

    @_mutates
    @figure_scoped
    def animate(
        self,
        *,
        fps: float | None = None,
        duration: float | None = None,
        n_frames: int | None = None,
        loop: bool | None = None,
        pingpong: bool | None = None,
        mode: Literal["reveal", "frames"] | None = None,
    ) -> Plot:
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

    @_mutates
    @figure_scoped
    def trail(
        self,
        length: tuple[Literal["time", "steps"], float] | None = _UNSET,
        *,
        fade: bool | None = None,
    ) -> Plot:
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

    @_mutates
    @figure_scoped
    def head(
        self,
        show: bool | None = None,
        *,
        size: float | None = None,
        color: str | None = None,
        symbol: str | None = None,
    ) -> Plot:
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

    # ``elev`` / ``azim`` are a *panel*'s viewing angle (each 3-D axes has its
    # own camera) and forward; ``spin`` is the figure's animation timeline and
    # must not become one desynchronised per-panel Animation.
    @_mutates
    @panel_scoped(figure_only=("spin",))
    def camera(
        self,
        *,
        elev: float | None = None,
        azim: float | None = None,
        spin: float | None = None,
    ) -> Plot:
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

    @_mutates
    @figure_scoped
    def clock(self, show: bool | str = True, *, fmt: str | None = None) -> Plot:
        """Show (or hide) a live time readout that updates each frame.

        Parameters
        ----------
        show : bool or str, optional
            Whether to draw the clock.  **A string is read as ``fmt``** —
            ``p.clock("t = {t:.1f}")`` is the spelling the docs and the contract
            print, and it used to bind the format string to this boolean and
            throw it away in silence (three different formats produced
            byte-identical movies).
        fmt : str, optional
            Label format.  The available fields are ``{t}`` (the current time)
            and ``{i}`` (the frame index) — e.g. ``"t = {t:.2f}"``.  A bare
            ``{}`` is accepted and normalised to ``{t}``.

            **Validated here, at the call.** A positional field such as
            ``"t={:.1f}"`` used to be accepted and then raise a raw
            ``IndexError`` inside ``.save()``, hundreds of frames later — the
            worst place to learn about a typo.

        Returns
        -------
        Plot
            ``self``, for chaining.

        Raises
        ------
        tsdynamics.errors.InvalidParameterError
            If ``fmt`` is not a format string over ``{t}`` / ``{i}``.
        """
        if isinstance(show, str):
            from tsdynamics.errors import InvalidParameterError

            if fmt is not None:
                raise InvalidParameterError(
                    "clock() was given a format twice — positionally and as fmt=. "
                    'Pass it once: p.clock("t = {t:.1f}").'
                )
            fmt, show = show, True
        a = self._ensure_animation()
        a.clock = bool(show)
        if fmt is not None:
            a.clock_format = _validated_clock_format(fmt)
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

    @_mutates
    @panel_scoped()
    def autocolor(self) -> Plot:
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

    def render(self, backend: str | None = None, *, ax: Any = None, **backend_kw: Any) -> Any:
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
        ax : matplotlib.axes.Axes, optional
            Draw into an existing axes instead of creating a figure — the way to
            put a tsdynamics plot inside a figure you are laying out yourself.
            Matplotlib only; ignored by the other backends.

            .. versionadded:: 6.0
                It worked before, through ``**backend_kw``, but appeared in no
                signature and no ``help()`` — so the one keyword users most need
                for integration was undiscoverable.  Naming it changes no
                behaviour.
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

        if ax is not None:
            backend_kw["ax"] = ax
        return render_spec(self, backend, **backend_kw)

    def show(self, backend: str | None = None, **backend_kw: Any) -> Any:
        """Render this spec and display it — the verb every plotting user types.

        The library's three verbs are :meth:`plot` (build), :meth:`render` (draw)
        and :meth:`save` (write), which is a coherent vocabulary — but ``.show()``
        is what a decade of ``plt.show()`` and ``fig.show()`` has wired into the
        fingers of every matplotlib and plotly user, and an ``AttributeError`` is
        a poor answer to a reasonable guess.  So it exists, and it means exactly
        what it says: :meth:`render`, then hand the figure to its own backend's
        display.

        On a **non-interactive** matplotlib backend (``Agg`` — a headless script,
        a CI job, this library's own test suite) there is no window to open, so
        the figure is rendered and returned and nothing else happens; use
        :meth:`save` to get a file.  This is deliberately quieter than
        :func:`matplotlib.pyplot.show`, which warns in that situation: the caller
        of *this* method is told the same thing by the docstring and by getting a
        figure back, and a warning that fires on every headless call is noise.

        Parameters
        ----------
        backend : str, optional
            The renderer name; ``None`` selects the default capable backend.
        **backend_kw
            Forwarded to the renderer.

        Returns
        -------
        Any
            The backend's figure — the same object :meth:`render` returns, so
            ``fig = spec.show()`` still gives you the handle to poke at.

        See Also
        --------
        tweak : apply inline tweaks, return the spec.
        render : draw and return the backend's figure, displaying nothing.
        save : write the figure (or animation) to a file.
        """
        figure = self.render(backend, **backend_kw)
        _display_figure(figure)
        return figure

    # Figure-scoped: ``plot`` forwards to the individual tweak methods, each of
    # which already declares (and performs) its own panel recursion — so ``plot``
    # itself must NOT recurse, or a panel-scoped tweak would be applied twice.
    @_mutates
    @figure_scoped
    def tweak(self, **tweaks: Any) -> Plot:
        """Apply inline tweaks and return **this spec**, so calls chain.

        The library's four plotting verbs, one return type each:

        =============== =====================================================
        ``.tweak(...)`` adjusts this spec and hands it back — chainable
        ``.render()``   gives you the backend's figure
        ``.show()``     draws it and displays it (the matplotlib/plotly reflex)
        ``.save(path)`` writes the file
        =============== =====================================================

        This method used to be called ``plot``, which was the joke told twice: a
        method named ``plot`` that does not plot, sitting next to a ``ts.plot``
        front door that does not draw either.  ``.style()`` / ``.relabel()`` /
        ``.theme()`` are the *named* ways to do the same thing and read better at
        a call site; ``tweak`` remains for the mixed bag (``xlabel`` / ``yscale``
        / ``title`` / ``xlim`` / …) in one call.

        .. versionchanged:: 6.0
            Renamed from ``plot``.  ``traj.plot()``, ``system.plot()``,
            ``result.plot()`` and ``ts.plot(...)`` all still return a
            :class:`PlotSpec` — the *building* verb is unchanged; it is only this
            no-op-on-a-spec spelling that is gone.

        Raises
        ------
        tsdynamics.errors.InvalidParameterError
            For a keyword that is neither an inline tweak nor a spec option — in
            particular a *renderer* option such as ``ax=`` / ``figsize=``, which
            belongs to :meth:`render`.
        """
        leftover = _apply_inline_tweaks(self, tweaks)
        if leftover:
            from tsdynamics.errors import InvalidParameterError

            names = ", ".join(repr(k) for k in sorted(leftover))
            raise InvalidParameterError(
                f"tweak() applies spec tweaks and returns the spec; {names} is not one. "
                "A renderer option (ax=, figsize=, dpi=, html=, …) or a backend name goes "
                "to render(), which returns the figure:\n"
                f"    spec.render('matplotlib', {next(iter(sorted(leftover)))}=...)"
            )
        return self

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

        **The contract (v6): this method never returns a path it did not write.**
        It resolves ``(extension, backend)`` to a writer and either writes the
        file or raises :class:`~tsdynamics.errors.InvalidParameterError` naming a
        combination that works.  Previously an unsupported pair — an animated
        composite to ``.html``, say — returned the path having written nothing,
        which is the worst failure mode in this layer: the caller believes a
        figure exists.  The guarantee is enforced twice: unsupported pairs are
        rejected up front, and the written file is **verified to exist and be
        non-empty** afterwards, so a backend that silently declines is caught even
        if this table does not know about it.

        When ``backend`` is not given, one is chosen from the file extension. For a
        **static** spec: a raster / vector image (``.png`` / ``.pdf`` / ``.svg`` /
        ``.jpg``) prefers the always-present matplotlib reference renderer,
        ``.html`` prefers plotly (a self-contained interactive page), and ``.json``
        prefers the json data exporter.  For an **animated** spec (``is_animated``):
        ``.html`` prefers plotly (an interactive scrubber), while ``.mp4`` / ``.gif``
        go to matplotlib (its :class:`~matplotlib.animation.FuncAnimation` writes
        them via ffmpeg / pillow); a still image renders the final frame.

        ``.html`` is written through the renderer's own file writer
        (``render(path=...)``), **not** by materialising a figure and inlining a
        JavaScript bundle into it: the same 2-D Lorenz spec is 0.05 MB the first
        way and 4.90 MB the second (**98x**), because plotly's default embeds its
        whole library in every page.

        .. note::
           ``.json`` is **two** formats behind one extension.  The default
           (``backend="json"``) writes the :class:`PlotSpec` **IR envelope** — the
           thing :meth:`from_dict` reads back.  ``backend="threejs"`` writes the
           three.js **geometry payload** (buffers + metadata), which is not a
           PlotSpec.  Pass ``backend=`` to disambiguate.

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

        Returns
        -------
        str
            ``path`` — and the file at ``path`` exists.

        Raises
        ------
        InvalidParameterError
            If the extension is not a format this library writes, if the
            ``(extension, backend)`` pair has no writer, or if the chosen backend
            returned without writing the file.
        TypeError
            If the produced result is neither a savable figure / animation nor a
            data-export write.
        """
        ext = _extension_of(path)
        # Check against the backend the *caller* named (possibly none), so an
        # extension nobody writes is reported as "no installed backend writes
        # '.tikz'" rather than blamed on whichever backend the fallback picked.
        self._check_save_supported(ext, backend)
        if backend is None:
            backend = self._preferred_save_backend(path)
        self._write(path, ext, backend, fps=fps, dpi=dpi, size=size, **backend_kw)
        self._verify_written(path, ext, backend)
        return path

    def _write(
        self,
        path: str,
        ext: str,
        backend: str | None,
        *,
        fps: float | None,
        dpi: float | None,
        size: tuple[float, float] | None,
        **backend_kw: Any,
    ) -> None:
        """Perform the write for a resolved ``(extension, backend)`` pair."""
        if backend is not None and _writes_its_own_file(backend, ext):
            # The backend owns a file writer: hand it the path.  Resolved from the
            # backend's *declaration*, not from a hardcoded ``("json", "threejs")``
            # list — that list is why a registered third-party backend could be
            # rendered by name but its declared extension could never be saved.
            self.render(backend, path=path, **backend_kw)
            return
        if ext in _HTML_EXT and backend == "plotly":
            # Route through the renderer's own writer: it emits a CDN-referencing
            # page (0.05 MB) instead of plotly's bundle-inlining default (4.90 MB),
            # and it is the same code path for a static figure and the animated
            # real-time (rAF + extendTraces) export.
            self.render("plotly", path=path, **backend_kw)
            return
        if self.is_animated and not _writers_for(ext, animated=True) and ext not in _HTML_EXT:  # noqa: E501
            # A still image (.png / .pdf / .svg / ...) of an animated spec is its
            # final, fully-revealed frame: render the static spec and write that.
            static = Plot.from_dict({**self.to_dict(), "animation": None})
            static.save(path, backend=backend, dpi=dpi, size=size, **backend_kw)
            return
        if size is not None and "figsize" not in backend_kw:
            w, h = size
            scale = float(dpi) if dpi else 100.0
            backend_kw["figsize"] = (float(w) / scale, float(h) / scale)
        if (
            backend in (None, "matplotlib", "mpl")
            and not backend_kw
            and self._figure_cache is not None
            and not self.is_animated
        ):
            # Reuse the figure the caller may already be holding.  ``save`` used
            # to call ``render`` unconditionally, so a hand edit made through
            # ``p.ax`` / ``p.fig`` was silently absent from ``p.save(...)`` while
            # ``p.fig.savefig(...)`` kept it — two different pictures from one
            # object, no warning.
            figure = self._figure_cache[0]
            savefig = getattr(figure, "savefig", None)
            if callable(savefig):
                savefig(path, **({"dpi": float(dpi)} if dpi is not None else {}))
                return
        result = self.render(backend, **backend_kw)
        # A matplotlib animation (FuncAnimation - uniquely carries ``to_jshtml``)
        # writes mp4 / gif via its own ``.save`` (writer inferred from the extension).
        if hasattr(result, "to_jshtml") and callable(getattr(result, "save", None)):
            anim_kw: dict[str, Any] = {}
            if fps is not None:
                anim_kw["fps"] = float(fps)
            if dpi is not None:
                anim_kw["dpi"] = float(dpi)
            result.save(path, **anim_kw)
            return
        figure = getattr(result, "figure", result)
        savefig = getattr(figure, "savefig", None)
        if callable(savefig):
            savefig(path, **({"dpi": float(dpi)} if dpi is not None else {}))
            return
        if ext in _HTML_EXT:
            write_html = getattr(figure, "write_html", None)
            if callable(write_html):
                write_html(path)
                return
        write_image = getattr(figure, "write_image", None)
        if callable(write_image):
            write_image(path)
            return
        if _accepts_path(backend):
            # Last resort before giving up: a backend that produced no figure but
            # takes ``path=`` is a file writer we did not recognise up front.
            # This is the branch that makes a *third-party* backend's declared
            # extension saveable — before v6 it could be rendered by name and
            # never saved.
            self.render(backend, path=path, **backend_kw)
            return
        raise TypeError(
            f"the {backend or 'selected'} backend produced a non-savable result "
            f"({type(figure).__name__}); use .to_dict() to serialize it instead."
        )

    def _check_save_supported(self, ext: str, backend: str | None) -> None:
        """Reject an ``(extension, backend)`` pair no installed backend can write.

        The *up-front* half of the save contract (:meth:`_verify_written` is the
        after-the-fact half).  It asks the **backends** — each declares
        ``writes_static`` / ``writes_animated`` on its capabilities — and keeps no
        table of its own, so a third-party renderer's declared extension is
        saveable the moment it registers, and this layer can never contradict a
        backend about what that backend can do.
        """
        from tsdynamics.errors import InvalidParameterError

        writers = self._save_writers(ext)
        if not writers and not self.is_animated and ext in _writable_extensions(animated=True):
            # The one wrong-pair worth its own sentence: asking a still for a
            # movie. Checked before the backend-specific branch, because the
            # answer is about the *plot*, not about which renderer you named.
            raise InvalidParameterError(
                f"{ext!r} is a movie format and this Plot is not animated. "
                "Add animate=True at the door, or call .animate() here."
            )
        if self.is_animated and self.is_composite and ext in _HTML_EXT:
            raise InvalidParameterError(
                "plotly cannot animate a composite (its animation export is "
                "single-panel); save to .mp4 / .gif (matplotlib writes them), or "
                "render the panels separately and save each to .html."
            )
        if backend is None:
            if not writers:
                raise InvalidParameterError(self._no_writer_message(ext))
            return
        if backend in writers:
            return
        raise InvalidParameterError(self._backend_cannot_write_message(ext, backend))

    def _save_writers(self, ext: str) -> list[str]:
        """Return the backends that can write ``ext`` for this plot, most-preferred first.

        An animated plot prefers a backend that writes ``ext`` as a movie, and
        falls back to one that writes it as a still — that fallback is the
        documented "a ``.png`` of a movie is its final frame" behaviour.  A static
        plot only ever gets the still writers, which is what turns ``.mp4`` on a
        still into a typed error instead of matplotlib's raw ``ValueError``.
        """
        if not self.is_animated:
            return _writers_for(ext, animated=False)
        if self.is_composite and ext in _HTML_EXT:
            # A spec-level decline no extension table can express: both .html
            # writers animate a *single panel* only (plotly's animation export is
            # single-panel; three.js reveals one draw-range), so an animated
            # composite has no HTML writer at all. Left to the table it wrote a
            # file that silently dropped every panel but one.
            return []
        return _writers_for(ext, animated=True) or _writers_for(ext, animated=False)

    def _no_writer_message(self, ext: str) -> str:
        """Explain that nothing installed writes ``ext`` — and name what does exist."""
        animated = self.is_animated
        if not animated and ext in _writable_extensions(animated=True):
            return (
                f"{ext!r} is a movie format and this Plot is not animated. "
                "Add animate=True at the door, or call .animate() here."
            )
        writable = sorted(
            set(_writable_extensions(animated=animated))
            | set(_writable_extensions(animated=False) if animated else [])
        )
        return (
            f"no installed backend writes {ext or '(no extension)'!r}.\n"
            f"    Writable extensions: {' '.join(writable)}\n"
            f"    ({_writes_table(animated=animated)})"
        )

    def _backend_cannot_write_message(self, ext: str, backend: str) -> str:
        """Explain that ``backend`` does not write ``ext`` — and name the ones that do."""
        animated = self.is_animated
        writers = self._save_writers(ext)
        caps = _renderer_capabilities().get(backend)
        if caps is None:
            known = ", ".join(_renderer_capabilities()) or "none"
            return f"no backend named {backend!r} is registered; installed backends are {known}."
        field = "writes_animated" if animated else "writes_static"
        mine = sorted(getattr(caps, field, None) or getattr(caps, "writes", None) or ())
        if getattr(caps, "data_export", False):
            reason = (
                f"the {backend!r} backend does not draw, so it cannot write {ext} "
                "(it serializes the figure)"
            )
        else:
            reason = f"the {backend!r} backend does not write {ext}"
        tail = f" It writes {' '.join(mine)}." if mine else " It writes no file itself."
        if writers:
            return f"{reason}; use backend={writers[0]!r} ({', '.join(writers)} can).{tail}"
        return f"{reason}, and nor does any other installed backend.{tail}"

    @staticmethod
    def _verify_written(path: str, ext: str, backend: str | None) -> None:
        """Raise unless ``path`` now holds a non-empty file of the right *format*.

        The backstop that makes "``save`` never returns a path it did not write"
        true *by construction* rather than by the completeness of a table: a
        backend that declines a spec and returns quietly is caught here, and so is
        a backend that writes the **wrong kind of document** under the requested
        extension — which is what ``save("x.html", backend="threejs")`` did (a
        142 KB JSON payload named ``.html``, which no browser can open).  A
        mis-formatted file is removed before raising, so a failed ``save`` never
        leaves a misleading artifact behind.
        """
        import os

        from tsdynamics.errors import InvalidParameterError

        try:
            written = os.path.getsize(path) > 0
        except OSError:
            written = False
        if not written:
            raise InvalidParameterError(
                f"the {backend or 'selected'} backend did not write {path!r} "
                f"(no file, or an empty one) - it cannot produce {ext}. "
                "Choose another backend= or another extension; save() never "
                "reports success for a file it did not write."
            )
        if ext in _HTML_EXT and not _looks_like_markup(path):
            os.unlink(path)
            raise InvalidParameterError(
                f"the {backend or 'selected'} backend wrote {path!r}, but it is not an "
                "HTML document (it does not begin with markup) - a browser cannot open "
                "it. Use backend='plotly' for an interactive page, or save the payload "
                "to .json. The file has been removed."
            )
        if ext not in _HTML_EXT and ext != ".json" and _looks_like_json(path):
            os.unlink(path)
            raise InvalidParameterError(
                f"the {backend or 'selected'} backend wrote {path!r}, but it is a JSON "
                f"document, not {ext} - it serialized the figure instead of drawing it. "
                "Use backend='matplotlib' for an image, or save to .json. The file has "
                "been removed."
            )

    def _preferred_save_backend(self, path: str) -> str | None:
        """Pick a save backend from ``path``'s extension (``None`` = dispatch default).

        Delegates to :func:`_writers_for`, which asks each installed backend what
        it declares — so the preference here is *only* the tie-break between two
        backends that both claim the extension (``.json`` prefers the IR envelope
        over the three.js payload, ``.html`` prefers plotly over three.js).
        """
        writers = self._save_writers(_extension_of(path))
        if writers:
            return writers[0]
        names = _registered_renderer_names()
        if names is None:
            return None
        if "matplotlib" in names:
            return "matplotlib"
        return None

    def __repr__(self) -> str:
        """Describe the spec **and name the next verb**.

        ``ts.plot(traj)`` in a script builds a spec and draws nothing — which is
        the right semantics (spec-in-spec-out is what makes
        ``ts.plot(ts.plot(a), ts.plot(b), layout="row")`` compose) but a baffling
        first experience if the object says nothing about how to see it.  So the
        repr carries the two verbs that do::

            PlotSpec(phase_portrait_3d, 1 layer) — .show() to display, .save('f.png') to write

        An **animated** spec names a filename it can actually write: ``.save``
        picks the backend by extension, and a ``.png`` of a movie is a still, so
        offering one would send the reader to the wrong verb::

            Plot(phase_portrait_3d, 1 layer, animated 30 fps) — .show() to display, .save('f.gif') to write

        In a notebook the plot draws itself and this is never seen; in a console
        it is the whole answer.
        """
        kind = getattr(self.kind, "value", self.kind)
        animated = getattr(self, "animation", None) is not None
        if self.panels:
            n_panels = len(self.panels)
            layout = self.layout or Layout()
            rows, cols = layout.grid(n_panels)
            arrangement = f"a {rows}x{cols} {layout.mode}"
            body = f"{n_panels} panel{'s' if n_panels != 1 else ''} in {arrangement}"
        else:
            n = len(self.layers)
            body = f"{n} layer{'s' if n != 1 else ''}"
            named = [t for t in (lyr.transform for lyr in self.layers) if t]
            distinct = list(dict.fromkeys(named))
            if named and len(named) == n and 1 < len(distinct) <= 4:
                body += ": " + ", ".join(distinct)
        anim = ""
        if animated:
            fps = getattr(self.animation, "fps", None)
            anim = f", animated {fps:g} fps" if fps else ", animated"
        target = "f.gif" if animated else "f.png"
        return f"Plot({kind}, {body}{anim}) — .show() to display, .save({target!r}) to write"

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Notebook display hook — render inline once a backend is installed.

        Mirrors :meth:`Plottable._repr_mimebundle_`: returns ``None`` (so the
        console falls back to ``repr``) outside a notebook or when no rendering
        backend is installed, keeping a plain ``import`` plot-library-free.
        """
        return _notebook_mimebundle(self.render, include, exclude)

    def _repr_html_(self) -> str | None:
        """Notebook fallback when **no** drawing backend is installed.

        With a backend, :meth:`_repr_mimebundle_` draws and this is never
        consulted.  Without one, a notebook cell used to show the 40-character
        repr; it now shows a small table of what the plot is, which is the whole
        answer available in that situation.
        """
        if _resolve_renderers() is not None:
            return None  # the mimebundle hook will draw it
        rows = [("kind", str(self.kind))]
        if self.panels:
            rows.append(("panels", str(len(self.panels))))
        else:
            rows.append(("layers", ", ".join(str(lyr.kind) for lyr in self.layers) or "none"))
        rows.append(("axes", ", ".join(a.label or "?" for a in (self.x, self.y) if a is not None)))
        rows.append(("theme", self.resolved_theme.name))
        if self.title:
            rows.insert(0, ("title", self.title))
        body = "".join(f"<tr><th align='left'>{k}</th><td>{v}</td></tr>" for k, v in rows)
        return (
            "<div><b>Plot</b> (no drawing backend installed — "
            "<code>pip install tsdynamics[viz]</code>)"
            f"<table>{body}</table></div>"
        )

    def __getitem__(self, key: int | str) -> Plot:
        """Select a panel: ``p[0]`` by position, ``p["psd"]`` by name.

        Returns a :class:`Plot`, so selection chains straight into a tweak::

            p = ts.plot(a, b, c, layout="row")
            p["psd"].rescale(x="log", y="log")
            p[0].style(color="crimson")

        A string matches a panel's ``title``, the transform that produced any of
        its layers, or its ``kind`` — in that order.  On a **single-panel** plot
        ``p[0] is p``, so code written for a grid also works on one panel.

        :attr:`panels` stays the *list* (iterate it, take its ``len``); ``p[...]``
        is *selection* — the ``df.columns`` / ``df["x"]`` relation, not a second
        spelling of the same thing.
        """
        from tsdynamics.errors import InvalidInputError, InvalidParameterError

        panels = self.panels or [self]
        if isinstance(key, (int, np.integer)):
            index = int(key)
            try:
                return panels[index]
            except IndexError:
                raise InvalidParameterError(
                    f"panel {index} does not exist; this Plot has {len(panels)} "
                    f"panel{'s' if len(panels) != 1 else ''} (0..{len(panels) - 1})."
                ) from None
        if isinstance(key, str):
            for panel in panels:
                if panel.title == key:
                    return panel
            for panel in panels:
                if any(lyr.transform == key for lyr in panel.layers):
                    return panel
            for panel in panels:
                if panel.kind == key:
                    return panel
            known = sorted(
                {p.title for p in panels if p.title}
                | {lyr.transform for p in panels for lyr in p.layers if lyr.transform}
                | {str(p.kind) for p in panels}
            )
            raise InvalidParameterError(
                f"no panel named {key!r}; this Plot's panels answer to "
                f"{', '.join(repr(k) for k in known)}."
            )
        raise InvalidInputError(
            f"a Plot is indexed by panel position (int) or panel name (str), not "
            f"{type(key).__name__}."
        )

    def __getattr__(self, name: str) -> Any:
        """Answer a plausible-but-wrong attribute with the spelling that works.

        Reached only when normal lookup fails, so it costs nothing on the hot
        path.  ``dunder`` probes short-circuit, or ``copy`` / ``pickle`` /
        ``inspect`` would be answered with prose.
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        moved = _PLOT_MOVED.get(name)
        if moved is not None:
            raise AttributeError(f"Plot has no {name!r}. {moved}")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __dir__(self) -> list[str]:
        """Expose the curated user surface (:data:`_PLOT_PUBLIC`) plus the dunders.

        The renderer internals (``autocolor`` / ``has_color_channel`` /
        ``resolved_panels`` / ``resolved_frame`` / ``is_three_d`` / ``tweak``) and
        the nine dataclass fields a caller reads but never types (``aspect``
        ``clim`` ``colorbar`` ``legend`` ``frame`` ``ndim`` ``x`` ``y`` ``z``)
        stay fully readable and callable — they leave the *tab surface*, not the
        object.  ``panels`` and ``layers`` stay listed: they appear on eleven
        documentation lines, seven of them runnable.
        """
        return sorted(set(_PLOT_PUBLIC) | {n for n in type(self).__dict__ if n.startswith("__")})

    # -- serialization -----------------------------------------------------

    @property
    def spec(self) -> Plot:
        """Return this plot — a :class:`Plot` **is** the IR, so there is nothing to unwrap.

        Bound deliberately, so ``p.spec`` reads naturally when you mean "the
        description, not the picture", and so a reader who expects a wrapper
        discovers by identity (``p.spec is p``) that there is none.
        """
        return self

    def to_json(self, **kwargs: Any) -> str:
        """Serialize this plot to the versioned JSON envelope.

        The write half of the round trip whose read half is
        :func:`tsdynamics.viz.load`::

            open("f.json", "w").write(p.to_json())
            p2 = ts.viz.load("f.json")

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`json.dumps` (e.g. ``indent=2``).
        """
        from .export import to_json

        return to_json(self, **kwargs)

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
            "frame": self.frame.to_dict() if self.frame is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Plot:
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
            frame=Frame.from_dict(d["frame"]) if d.get("frame") is not None else None,
        )


#: The pre-v6 name of :class:`Plot`.  The **same class object**, not a subclass
#: and not a wrapper — kept bound so the 557 in-tree annotations and the
#: ``isinstance(x, PlotSpec)`` checks in the renderers keep working while the
#: user-facing name is ``Plot``.  It is in no ``__all__`` and in no ``dir()``;
#: ``ts.viz.PlotSpec`` answers with the new spelling.
PlotSpec = Plot

#: The curated tab surface of a :class:`Plot` (contract §6.3 — 34 names).
#: Everything omitted stays readable and callable; it just stops shouting.
_PLOT_PUBLIC: tuple[str, ...] = (
    # draw it
    "show",
    "save",
    "render",
    "fig",
    "ax",
    "axes",
    # compose
    "add",
    "panels",
    "layers",
    # style
    "style",
    "recolor",
    "palette",
    "theme",
    "font",
    "background",
    "size",
    "gridlines",
    # label
    "relabel",
    "rescale",
    "limits",
    "ticks",
    "colorize",
    "title",
    # annotate
    "vline",
    "hline",
    "span",
    "text",
    # animate
    "animate",
    "trail",
    "head",
    "camera",
    "clock",
    # serialize / introspect
    "to_json",
    "to_dict",
    "from_dict",
    "kind",
    "meta",
    "is_animated",
    "is_composite",
)

#: ``old attribute -> the sentence that names the working spelling``.  Read by
#: :meth:`Plot.__getattr__`: a rename's error message *is* its migration guide.
_PLOT_MOVED: dict[str, str] = {
    "annotate": "Use .vline(x) / .hline(y) / .span(lo, hi) / .text(x, y, s).",
    "grid": (
        "Gridlines are .gridlines(...); the panel arranger is ts.viz.grid(...). "
        "One attribute cannot mean both."
    ),
    "plot": "A Plot is already a plot; .show() displays it and .save(path) writes it.",
    "to_plot_spec": "A Plot is already a Plot; .to_dict() gives the serializable mapping.",
}


# ---------------------------------------------------------------------------
# Plottable mixin
# ---------------------------------------------------------------------------


class Plottable:
    """Mixin giving any ``__plot_spec__()`` provider a ``.plot()`` and notebook hook.

    A class that produces a :class:`PlotSpec` only has to implement
    ``__plot_spec__(self) -> PlotSpec``; this mixin layers the rendering sugar on
    top:

    - :meth:`plot` — ``self.__plot_spec__()`` plus optional inline tweaks, sent
      to a backend.
    - ``_repr_mimebundle_`` — a notebook display hook that renders inline once a
      backend is installed, and **no-ops** until then (so importing core never
      pulls a plot library, and a result still reprs as text in plain consoles).

    Result types in :mod:`tsdynamics.analysis` inherit their ``.plot`` accessor
    from :class:`~tsdynamics.analysis._result.AnalysisResult` instead; this mixin
    is for the plain data types (e.g. :class:`~tsdynamics.data.Trajectory`) that
    are not analysis results.
    """

    def __plot_spec__(self, *args: Any, **kwargs: Any) -> Plot:
        """Return the :class:`PlotSpec` describing this object.

        Subclasses must override this.  The base raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement __plot_spec__() to be Plottable."
        )

    def plot(self, *transforms: Any, **tweaks: Any) -> Plot:
        """Build this object's :class:`PlotSpec`, applying inline tweaks first.

        ``plot`` **builds**, ``render`` **draws**, ``save`` **writes** — the same
        three verbs everywhere in the library (see :meth:`PlotSpec.plot`).  Tweak
        keywords matching a :class:`PlotSpec` tweak method (``xscale`` /
        ``yscale`` / ``zscale``, ``xlabel`` / ``ylabel`` / ``zlabel`` / ``title``,
        ``xlim`` / ``ylim`` / ``zlim``, ``clim`` / ``colorbar`` / ``legend``) are
        applied to the spec.

        Parameters
        ----------
        **tweaks
            Inline spec tweaks.

        Returns
        -------
        PlotSpec
            Chainable, saveable (``.save("fig.png")``), renderable
            (``.render("plotly")``), and self-drawing in a notebook.
        """
        reject_positional_transform(transforms, "obj")
        return self.__plot_spec__().tweak(**tweaks)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Rich notebook display — renders inline once a backend is installed.

        Returns ``None`` (a no-op, so IPython falls back to ``__repr__``) outside
        a notebook or when no rendering backend is installed.  This keeps
        notebook import of core plot-library-free until a viz backend ships.
        """
        return _notebook_mimebundle(lambda: self.__plot_spec__().render(), include, exclude)


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

# Inline ``.plot(...)`` keywords routed through :meth:`Plot.colorize`.
_COLORIZE_TWEAKS = frozenset({"clim", "colorbar", "legend"})

#: **The figure vocabulary — one definition, used at every plotting door.**
#:
#: Seventeen keywords that *name* a figure rather than compute it::
#:
#:     clim colorbar legend theme title
#:     xlabel  ylabel  zlabel
#:     xlim    ylim    zlim
#:     xscale  yscale  zscale
#:     xticks  yticks  zticks
#:
#: Derived, never hand-listed, so it cannot drift from the tweaks that implement
#: it.  Measured before v6: **12 of these 17 raised at ``ts.plot(...)`` and all 17
#: worked at ``traj.plot(...)``** — ``ts.plot(traj, xlim=(0, 1))`` answered
#: ``kind='phase_portrait_3d' does not accept keyword(s) ['xlim']`` — because the
#: front door carried its own five-name copy of this set and the leftovers were
#: validated against a per-*kind* allow-list one layer down.
#:
#: Every door peels :data:`FIGURE_KEYS` (and the style vocabulary,
#: :func:`~tsdynamics.viz.style.style_names`) **before** the remainder is treated
#: as something to compute, so an integration typo is still reported as an
#: integration typo.
FIGURE_KEYS: frozenset[str] = frozenset(_INLINE_TWEAKS) | _COLORIZE_TWEAKS | {"theme"}


def split_figure_keywords(kw: dict[str, Any]) -> dict[str, Any]:
    """Peel the :data:`FIGURE_KEYS` out of ``kw`` **in place** and return them.

    ``kw`` keeps only the keywords that describe *what to compute*.
    """
    return {k: kw.pop(k) for k in list(kw) if k in FIGURE_KEYS}


def apply_figure_keywords(plot: Plot, figure: Mapping[str, Any]) -> Plot:
    """Apply peeled :data:`FIGURE_KEYS` to a finished plot, and return it.

    The single applier behind every door.  On a composite these land on **every**
    panel, because that is what the underlying tweaks already do (``title`` stays
    figure-level) — an undocumented "first panel only" would be the same
    two-meanings-for-one-spelling defect this vocabulary exists to end.
    """
    theme = figure.get("theme")
    if theme is not None:
        plot.theme(theme)
    rest = {k: v for k, v in figure.items() if k != "theme"}
    if rest:
        leftover = _apply_inline_tweaks(plot, rest)
        assert not leftover, leftover  # FIGURE_KEYS is derived from what this applies
    return plot


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

    Note it reports the registry's state *now*: the in-tree backends register on
    first render, so this is empty in a session that has not drawn anything yet.
    Gating the notebook hook on it therefore made the very first cell no-op —
    which is why :func:`_notebook_mimebundle` gates on the IPython shell and a
    real render attempt instead.
    """
    try:
        from tsdynamics.registry import renderers
    except Exception:  # pragma: no cover - registry has no renderers yet
        return None
    try:
        return renderers if len(renderers) else None
    except Exception:  # pragma: no cover - defensive
        return None


def reject_positional_transform(positional: tuple[Any, ...], subject: str) -> None:
    """Refuse ``obj.plot("delay_embedding")`` with the line that does work.

    ``ts.plot(traj, "delay_embedding", delay=7)`` reads a positional string as a
    *transform name*, so a reader who has seen that call naturally tries the same
    thing on the method — and got a bare
    ``TypeError: plot() takes 1 positional argument but 2 were given``, which
    names neither the concept nor the spelling that works.  The method takes only
    keywords (its positional slot is the object itself), so the honest answer is
    the front door.

    Parameters
    ----------
    positional : tuple
        Whatever was captured by the method's ``*args``.
    subject : str
        How to spell the receiver in the example (``"traj"``, ``"system"``).

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        Whenever ``positional`` is non-empty.
    """
    if not positional:
        return
    from tsdynamics.errors import InvalidParameterError, remedy

    first = positional[0]
    if isinstance(first, str):
        lead = f"{subject}.plot() takes no positional arguments; a transform is named on the "
        lines = [
            f"ts.plot({subject}, {first!r})",
            f"{subject}.plot(kind={first!r})   # if {first!r} is a plot KIND, not a transform",
        ]
        raise InvalidParameterError(
            lead + "front door (and a plot kind is a keyword):" + remedy(*lines)
        )
    raise InvalidParameterError(
        f"{subject}.plot() takes no positional arguments, got {type(first).__name__}. "
        "Compose several things with the front door instead:" + remedy(f"ts.plot({subject}, other)")
    )


#: Matplotlib backends that draw into a file rather than onto a screen.  Used only
#: as the fallback when the installed matplotlib does not expose its backend
#: registry; the registry is asked first, so a new GUI backend needs no edit here.
_NON_INTERACTIVE_MPL: frozenset[str] = frozenset(
    {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
)


def _mpl_backend_is_interactive() -> bool:
    """Whether the active matplotlib backend can actually open a window."""
    import matplotlib

    name = matplotlib.get_backend().lower().removeprefix("module://")
    try:
        from matplotlib.backends.registry import BackendFilter, backend_registry

        gui: Any = backend_registry.list_builtin(BackendFilter.INTERACTIVE)  # type: ignore[no-untyped-call]
        return name in {str(b).lower() for b in gui}
    except Exception:  # pragma: no cover - older/odd matplotlib: fall back to the set
        return name not in _NON_INTERACTIVE_MPL


def _display_figure(figure: Any) -> None:
    """Hand a rendered figure to its own backend's display, if it has one.

    Dispatches on where the figure came from rather than on which backend was
    asked for, so it stays right when the dispatcher falls back to matplotlib.
    A payload that is not a figure at all (the json / three.js exporters return
    data) has nothing to display, and that is not an error — ``show()`` on an
    export is simply the export.
    """
    module = type(figure).__module__.split(".")[0]
    if module == "matplotlib":
        if not _mpl_backend_is_interactive():
            return  # headless: the figure is the result; save() writes the file
        import matplotlib.pyplot as plt

        plt.show()
        return
    show = getattr(figure, "show", None)
    if callable(show):  # plotly: opens the figure in a browser / notebook cell
        show()


def _notebook_mimebundle(draw: Any, include: Any, exclude: Any) -> Any:
    """Shared ``_repr_mimebundle_`` body: draw, then return a **real** mime bundle.

    ``_repr_mimebundle_`` must return a *mapping* of mime type to payload (or a
    ``(data, metadata)`` pair).  Returning the backend's figure object instead
    makes IPython emit ``FormatterWarning: ... returned invalid type`` and fall
    back to ``__repr__`` — which for a :class:`PlotSpec` is a multi-thousand-line
    dump of its data arrays.  That is exactly what happened when ``.plot()``
    stopped returning a figure: the hook still handed a figure straight back, so
    a notebook cell printed the dump instead of drawing the picture.

    Every plotting backend already registers its own IPython display hooks
    (matplotlib a PNG, plotly HTML+JS), so the honest bundle is whatever IPython
    itself makes of the drawn figure.  Asking IPython keeps this
    backend-agnostic and keeps the notebook stack out of the import path.

    Parameters
    ----------
    draw : callable
        Zero-argument callable returning the drawn figure (``spec.render()``).
        Called only inside a live IPython shell, so a plain console echo never
        imports a plotting library.
    include, exclude
        Forwarded to IPython's formatter, as the display protocol requires.

    Returns
    -------
    tuple of dict or None
        ``(data, metadata)`` when something was drawn and IPython could format
        it; ``None`` (fall back to ``repr``) otherwise — no backend installed,
        no IPython, or a render error.
    """
    try:
        from IPython.core.getipython import get_ipython
    except Exception:  # pragma: no cover - IPython not installed
        return None
    shell: Any = get_ipython()  # type: ignore[no-untyped-call]
    if shell is None:  # a plain console / script: nothing to display into
        return None
    try:
        drawn = draw()
    except Exception:  # pragma: no cover - never break repr on a render error
        return None
    if drawn is None:
        return None
    try:
        data, metadata = shell.display_formatter.format(drawn, include=include, exclude=exclude)
    except Exception:  # pragma: no cover - defensive
        return None
    return (data, metadata) if data else None


def _validated_clock_format(fmt: str) -> str:
    """Return ``fmt`` normalised for the clock, or raise naming the fields that exist.

    Accepts ``{t}`` (time), ``{i}`` (frame index) and a bare ``{}`` (normalised to
    ``{t}``), each with any format spec.  Anything else — a positional field, an
    unknown name, unbalanced braces — is refused **here**, at the call site,
    instead of raising inside a per-frame callback during ``.save()``.
    """
    import string

    from tsdynamics.errors import InvalidParameterError

    def refuse(problem: str) -> InvalidParameterError:
        return InvalidParameterError(
            f"clock format {fmt!r} {problem}; the available fields are {{t}} (time) "
            'and {i} (frame index). Try "t = {t:.1f}".'
        )

    try:
        parsed = list(string.Formatter().parse(fmt))
    except ValueError as exc:
        raise refuse(f"is not a valid format string ({exc})") from None
    if not any(name is not None for _, name, _, _ in parsed):
        raise refuse("names no field, so the readout would never change")
    out: list[str] = []
    for literal, name, spec, conversion in parsed:
        out.append(literal)
        if name is None:
            continue
        # An empty field name is Python's *auto-numbered positional* slot, and
        # both "{}" and "{:.1f}" parse to name == "". A bare "{}" is the obvious
        # shorthand for the time, so it is normalised; anything carrying a format
        # spec or a conversion is a genuine positional field, which is exactly
        # what blows up at save time (the frame callback formats by keyword).
        if name == "" and (spec or conversion):
            raise refuse("refers to a positional field")
        if name not in ("", "t", "i"):
            what = "refers to a positional field" if name.isdigit() else f"names {name!r}"
            raise refuse(what)
        field_name = name or "t"
        out.append(
            "{"
            + field_name
            + (f"!{conversion}" if conversion else "")
            + (f":{spec}" if spec else "")
            + "}"
        )
    return "".join(out)


def _degraded_warning() -> type[Warning]:
    """Return :class:`~tsdynamics.viz.render.caps.VisualizationDegraded`.

    Imported lazily: ``caps`` imports this module, and ``import tsdynamics`` must
    not pull the render subpackage in.  Falls back to :class:`UserWarning` (the
    class ``VisualizationDegraded`` itself subclasses) if the render layer is
    unavailable, so a warning is never swallowed.
    """
    try:
        from tsdynamics.viz.render.caps import VisualizationDegraded
    except Exception:  # pragma: no cover - render layer unavailable
        return UserWarning
    return VisualizationDegraded


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


def __getattr__(name: str) -> Any:
    """Resolve the ten IR nouns owned by sibling modules (:data:`_LAZY_IR_NAMES`).

    ``viz.transforms``, ``viz._frames`` and ``viz.export`` all import *this*
    module, so re-exporting their nouns eagerly would be an import cycle.
    Resolving them here makes ``ts.viz.spec.Geometry`` and
    ``ts.viz.spec.SCHEMA_VERSION`` work without one, and without adding a
    plotting import to ``import tsdynamics``.
    """
    target = _LAZY_IR_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target), name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
