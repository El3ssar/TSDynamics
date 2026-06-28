"""Renderer capability protocol + result types (stream VIZ-CAP).

The visualization seam dispatches a :class:`~tsdynamics.viz.spec.PlotSpec` to a
*renderer* (matplotlib, plotly, json, three.js).  Not every backend can draw
every :class:`~tsdynamics.viz.spec.PlotKind` — plotly has no native
``vector_field`` quiver-on-stream, the ``json`` / ``threejs`` exporters return a
payload rather than a figure, and only some backends do 3-D.  This module is the
*capability* layer the dispatch (stream VIZ-DISPATCH) consults to pick a backend
and **fall back** to the matplotlib reference renderer when the chosen one
declines a kind.

The pieces
----------
- :class:`RendererCapabilities` — what a backend can draw: the set of
  :class:`~tsdynamics.viz.spec.PlotKind` marks/semantic kinds it handles (or
  *all*), plus the four orthogonal capability flags ``supports_3d`` /
  ``interactive`` / ``web_export`` / ``data_export``.  :meth:`can_render` /
  :meth:`can_render_spec` answer the dispatch's "can this backend draw it?".
- :class:`RenderResult` — an *optional* typed return value a backend may use to
  describe what it produced: a native ``figure`` handle (matplotlib / plotly), a
  data ``payload`` (json / three.js), or both, tagged with the ``backend`` name
  and an optional ``mimetype``.  Renderers are free to return a bare figure
  instead; the dispatch forwards whatever they return.
- :class:`Renderer` — the ``runtime_checkable`` :class:`~typing.Protocol` a real
  backend satisfies: a callable ``(spec, **kw) -> Any`` carrying a
  :attr:`~RendererCapabilities` descriptor.
- :class:`VisualizationDegraded` — the warning emitted when the requested backend
  cannot draw a spec and the dispatch silently falls back to another, **or** when
  a rendered spec carries knobs (style keys, animation directives, theme fields)
  that the chosen backend does not honor.
- :func:`style_honoring_gaps` — collect every per-layer canonical style key,
  :class:`~tsdynamics.viz.spec.Animation` knob, and theme/axis presentation field
  the chosen backend does **not** honor for the given spec.  The dispatcher emits
  one consolidated :class:`VisualizationDegraded` per render naming the dropped
  knobs; renderers then run with ``warn=False``.

This module is **import-light**: it pulls in no plotting backend (only the
backend-agnostic spec IR), so importing it never drags matplotlib/plotly into
``sys.modules`` and the ``import tsdynamics`` no-plot-library guarantee holds.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..spec import PlotKind, PlotSpec

__all__ = [
    "Renderer",
    "RendererCapabilities",
    "RenderResult",
    "VisualizationDegraded",
    "style_honoring_gaps",
]


class VisualizationDegraded(UserWarning):
    """Warned when a backend cannot fully honor a spec.

    There are two situations that trigger this warning:

    1. **Backend fallback**: the requested backend declines a spec's
       :class:`~tsdynamics.viz.spec.PlotKind` (or its 3-D-ness) and dispatch
       picks a capable backend — the matplotlib reference renderer is the
       universal fallback — so the caller knows the figure did not come from the
       backend they named.
    2. **Knob degradation**: the chosen backend does not honor one or more style
       keys, :class:`~tsdynamics.viz.spec.Animation` directives, or theme/axis
       fields carried by the spec.  The dispatcher emits **one** consolidated
       warning naming all the ignored knobs before calling the renderer;
       renderers then run with ``warn=False`` (the user has already been told).

    It is a :class:`UserWarning` (not an error) so the call still returns a
    figure; a *hard* failure (no capable backend at all) raises instead.
    """


# ---------------------------------------------------------------------------
# Per-backend knob-honor declarations (Animation + theme/axis fields)
# ---------------------------------------------------------------------------

#: Animation directive fields that the **plotly** backend does *not* honor.
#: (plotly cannot do a camera spin, a clock overlay, or trail fade; elev/azim are
#: 3-D positioning knobs that require plotly's ``eye`` conversion — not implemented;
#: head_symbol is a marker shape plotly's rAF loop does not support per-frame.)
_PLOTLY_ANIMATION_GAPS: frozenset[str] = frozenset(
    {"spin", "clock", "clock_format", "trail_fade", "elev", "azim", "head_symbol"}
)

#: Animation directive fields that the **threejs** backend does *not* honor.
#: (three.js has no camera-spin API tied to the reveal comet; no clock overlay;
#: no trail fade; no per-frame head symbol; elev/azim camera are not wired.)
_THREEJS_ANIMATION_GAPS: frozenset[str] = frozenset(
    {"spin", "clock", "clock_format", "trail_fade", "head_symbol", "elev", "azim"}
)

#: Animation directive fields (as they appear on the :class:`~tsdynamics.viz.spec.Animation`
#: object) that are sourced from the ``camera`` meta key rather than directly from the
#: ``Animation`` dataclass — needed for the ``elev`` / ``azim`` gap check.
_CAMERA_META_KEYS: frozenset[str] = frozenset({"elev", "azim"})

#: Per-backend animation gaps table: backend name → frozenset of unhonored animation knobs.
_BACKEND_ANIMATION_GAPS: dict[str, frozenset[str]] = {
    "plotly": _PLOTLY_ANIMATION_GAPS,
    "threejs": _THREEJS_ANIMATION_GAPS,
}

#: Theme fields that the **threejs** backend does *not* honor.  The three.js loader
#: honors **only** ``background`` and ``palette`` (the exporter serializes those into a
#: ``metadata.theme`` block); ``foreground`` is dead (the loader marks it "unused here"
#: and the exporter no longer emits it) and font / grid / title presentation is a browser
#: concern that is not wired into the three.js scene — every one of these is a gap.
_THREEJS_THEME_GAPS: frozenset[str] = frozenset(
    {"foreground", "font_family", "font_size", "title_size", "grid", "grid_color", "grid_alpha"}
)

#: Per-backend theme presentation gaps.
_BACKEND_THEME_GAPS: dict[str, frozenset[str]] = {
    "threejs": _THREEJS_THEME_GAPS,
}

#: Axis/Legend/Colorbar fields that the **threejs** backend does *not* honor.
#: (the exporter writes position data only; label formatting is not wired into
#: the loader's three.js scene.)
_THREEJS_AXIS_GAPS: frozenset[str] = frozenset(
    {"label_size", "tick_size", "tick_rotation", "grid_color", "font_size", "ncol", "frame"}
)

#: Per-backend axis/legend/colorbar gaps.
_BACKEND_AXIS_GAPS: dict[str, frozenset[str]] = {
    "threejs": _THREEJS_AXIS_GAPS,
}

#: Backends that faithfully **serialize** the entire spec rather than draw it.
#: They round-trip every field, so by construction they have no honoring gaps —
#: :func:`style_honoring_gaps` returns ``[]`` for them (design contract §3).
_SERIALIZING_BACKENDS: frozenset[str] = frozenset({"json"})


def style_honoring_gaps(spec: PlotSpec, backend_name: str) -> list[str]:
    """Collect every knob the chosen backend will *not* honor for ``spec``.

    Called by the dispatcher **before** calling the renderer so it can emit one
    consolidated :class:`VisualizationDegraded` warning naming all the dropped
    knobs.  Renderers then pass ``warn=False`` to
    :func:`~tsdynamics.viz.style.normalize_style` (the user has already been
    told).

    The check covers three categories:

    1. **Per-layer canonical style keys**: any key in a layer's ``style`` dict
       that is known to :data:`~tsdynamics.viz.style.STYLE_KEYS` but whose
       :attr:`~tsdynamics.viz.style.StyleKey.honored_by` does **not** include the
       backend.
    2. **Animation directive knobs**: fields of the spec's
       :class:`~tsdynamics.viz.spec.Animation` (``spin``, ``clock``,
       ``trail_fade``, ``elev`` / ``azim`` from ``meta["camera"]``, …) that are
       set to a non-default value but the backend cannot honor.
    3. **Theme / axis presentation fields**: theme fields (``font_family``,
       ``grid``, …) and axis/legend/colorbar fields (``label_size``,
       ``tick_rotation``, …) the backend does not serialize or apply.

    Only knobs that are **actually set** (non-default / non-None / non-zero for
    optional fields) are reported — a backend ignoring ``spin=0`` (the "hold the
    camera still" default) is not a degradation.

    Parameters
    ----------
    spec : PlotSpec
        The spec about to be rendered.
    backend_name : str
        The resolved backend name (e.g. ``"matplotlib"``, ``"plotly"``,
        ``"threejs"``).

    Returns
    -------
    list of str
        Sorted list of knob names that will be silently ignored, ready for
        inclusion in a :class:`VisualizationDegraded` warning message.
    """
    from tsdynamics.viz.style import STYLE_KEYS

    gaps: set[str] = set()

    canonical_name = _normalize_backend_name(backend_name)

    # The json backend is a faithful data exporter — it serializes the *whole*
    # spec (every style key, the resolved theme, all axis / legend / colorbar
    # fields) and round-trips it byte-for-byte.  It draws nothing, so it drops
    # nothing: it has no honoring gaps by construction (design contract §3).
    if canonical_name in _SERIALIZING_BACKENDS:
        return []

    # ── 1. Per-layer style key gaps ─────────────────────────────────────────
    for layer in _all_layers(spec):
        for key in layer.style:
            sk = STYLE_KEYS.get(key)
            if sk is None:
                continue  # unknown key — normalize_style already warned
            if canonical_name not in sk.honored_by:
                gaps.add(key)

    # ── 2. Animation directive gaps ─────────────────────────────────────────
    anim_gaps = _BACKEND_ANIMATION_GAPS.get(canonical_name, frozenset())
    if spec.animation is not None and anim_gaps:
        anim = spec.animation
        # Fields on the Animation dataclass (non-default / active values only):
        if "spin" in anim_gaps and anim.spin != 0.0:
            gaps.add("spin")
        if "clock" in anim_gaps and anim.clock:
            gaps.add("clock")
        if "clock_format" in anim_gaps and anim.clock and anim.clock_format != "t = {t:.2f}":
            gaps.add("clock_format")
        if "trail_fade" in anim_gaps and anim.trail_fade:
            gaps.add("trail_fade")
        if "head_symbol" in anim_gaps and anim.head_symbol not in ("o", ""):
            gaps.add("head_symbol")
        # elev / azim live in meta["camera"], not on Animation:
        if "elev" in anim_gaps or "azim" in anim_gaps:
            camera = spec.meta.get("camera") if isinstance(spec.meta, dict) else None
            if isinstance(camera, dict):
                if "elev" in anim_gaps and "elev" in camera:
                    gaps.add("elev")
                if "azim" in anim_gaps and "azim" in camera:
                    gaps.add("azim")

    # ── 3. Theme / axis / legend / colorbar presentation gaps ───────────────
    theme_gaps = _BACKEND_THEME_GAPS.get(canonical_name, frozenset())
    if theme_gaps and spec._theme is not None:
        t = spec._theme
        # Only report when the theme actually sets the field (not None / default).
        if "foreground" in theme_gaps and t.foreground is not None:
            gaps.add("theme.foreground")
        if "font_family" in theme_gaps and t.font_family is not None:
            gaps.add("theme.font_family")
        if "font_size" in theme_gaps and t.font_size is not None:
            gaps.add("theme.font_size")
        if "title_size" in theme_gaps and t.title_size is not None:
            gaps.add("theme.title_size")
        if "grid" in theme_gaps and t.grid:
            gaps.add("theme.grid")
        if "grid_color" in theme_gaps and t.grid_color is not None:
            gaps.add("theme.grid_color")
        if "grid_alpha" in theme_gaps and t.grid_alpha is not None:
            gaps.add("theme.grid_alpha")

    axis_gaps = _BACKEND_AXIS_GAPS.get(canonical_name, frozenset())
    if axis_gaps:
        for ax in [spec.x, spec.y, spec.z]:
            if ax is None:
                continue
            if "label_size" in axis_gaps and ax.label_size is not None:
                gaps.add("axis.label_size")
            if "tick_size" in axis_gaps and ax.tick_size is not None:
                gaps.add("axis.tick_size")
            if "tick_rotation" in axis_gaps and ax.tick_rotation is not None:
                gaps.add("axis.tick_rotation")
        if spec.legend is not None:
            leg = spec.legend
            if "ncol" in axis_gaps and leg.ncol != 1:
                gaps.add("legend.ncol")
            if "frame" in axis_gaps and not leg.frame:
                gaps.add("legend.frame")
            if "font_size" in axis_gaps and leg.font_size is not None:
                gaps.add("legend.font_size")
        if spec.colorbar is not None:
            cb = spec.colorbar
            if "label_size" in axis_gaps and cb.label_size is not None:
                gaps.add("colorbar.label_size")

    # Also recurse into panels of a composite spec.
    for panel in spec.panels:
        gaps.update(style_honoring_gaps(panel, backend_name))

    return sorted(gaps)


def _normalize_backend_name(name: str) -> str:
    """Map registry names to the canonical names used in ``honored_by`` sets.

    The registry uses short names (``"mpl"``, ``"plotly"``, ``"threejs"``);
    :data:`~tsdynamics.viz.style.StyleKey.honored_by` uses the long form
    (``"matplotlib"``).  Normalise so the lookup works regardless.
    """
    _MAP = {
        "mpl": "matplotlib",
        "matplotlib": "matplotlib",
        "plotly": "plotly",
        "json": "json",
        "threejs": "threejs",
    }
    return _MAP.get(name, name)


def _all_layers(spec: PlotSpec) -> list[Any]:
    """Return all layers in ``spec``, including those of any child panels."""
    layers: list[Any] = list(spec.layers)
    for panel in spec.panels:
        layers.extend(_all_layers(panel))
    return layers


# ---------------------------------------------------------------------------
# RendererCapabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RendererCapabilities:
    """Declared drawing capabilities of a rendering backend.

    A backend advertises which :class:`~tsdynamics.viz.spec.PlotKind` values it
    can draw and four orthogonal feature flags.  The dispatch uses
    :meth:`can_render_spec` to decide whether to route a spec here or fall back.

    Parameters
    ----------
    name : str
        The backend's registry name (e.g. ``"matplotlib"``, ``"plotly"``).
    kinds : frozenset of PlotKind, optional
        The semantic kinds **and** layer marks this backend can draw.  ``None``
        (the default) means *all* kinds — the matplotlib reference renderer, the
        conformance oracle, declares ``None``; a partial backend (plotly) lists
        exactly what it supports so the rest fall back.
    supports_3d : bool, optional
        Whether the backend can render 3-D specs (``ndim == 3`` / a ``z`` axis /
        a ``LINE3D`` / ``SURFACE3D`` mark).  Default ``False``.
    interactive : bool, optional
        Whether the backend produces an interactive figure (pan/zoom/hover).
        Default ``False``.
    web_export : bool, optional
        Whether the backend can emit a self-contained web artifact (e.g. an HTML
        fragment).  Default ``False``.
    data_export : bool, optional
        Whether the backend returns a serializable *payload* (json / three.js)
        rather than a live figure handle.  Default ``False``.
    """

    name: str
    kinds: frozenset[PlotKind] | None = None
    supports_3d: bool = False
    interactive: bool = False
    web_export: bool = False
    data_export: bool = False

    @classmethod
    def all_kinds(
        cls,
        name: str,
        *,
        supports_3d: bool = True,
        interactive: bool = False,
        web_export: bool = False,
        data_export: bool = False,
    ) -> RendererCapabilities:
        """Build capabilities for a backend that draws **every** kind.

        The shorthand the reference renderer (and any other "draws anything"
        backend) uses: ``kinds=None`` means no kind is ever declined.
        """
        return cls(
            name=name,
            kinds=None,
            supports_3d=supports_3d,
            interactive=interactive,
            web_export=web_export,
            data_export=data_export,
        )

    @classmethod
    def of_kinds(
        cls,
        name: str,
        kinds: Iterable[PlotKind | str],
        *,
        supports_3d: bool = False,
        interactive: bool = False,
        web_export: bool = False,
        data_export: bool = False,
    ) -> RendererCapabilities:
        """Build capabilities for a backend that draws only ``kinds``.

        Coerces each entry to a :class:`~tsdynamics.viz.spec.PlotKind`, so a
        backend can list either enum members or their string spellings.
        """
        return cls(
            name=name,
            kinds=frozenset(PlotKind(k) for k in kinds),
            supports_3d=supports_3d,
            interactive=interactive,
            web_export=web_export,
            data_export=data_export,
        )

    def can_render(self, kind: PlotKind | str) -> bool:
        """Whether this backend can draw the semantic/​mark ``kind``.

        ``True`` when :attr:`kinds` is ``None`` (draws everything) or ``kind`` is
        in the declared set.  An unrecognized string (not a real
        :class:`~tsdynamics.viz.spec.PlotKind`) is declined rather than raising,
        so a stray kind degrades to the fallback instead of crashing dispatch.
        """
        if self.kinds is None:
            return True
        try:
            resolved = PlotKind(kind)
        except ValueError:
            return False
        return resolved in self.kinds

    def can_render_spec(self, spec: PlotSpec) -> bool:
        """Whether this backend can draw the whole ``spec``.

        Combines the kind check (the spec's semantic :attr:`~PlotSpec.kind` and
        every layer's mark must be drawable) with the 3-D check: a spec that is
        3-D (``ndim == 3``, a ``z`` axis, or any ``LINE3D`` / ``SURFACE3D`` mark)
        needs :attr:`supports_3d`.

        A **composite** spec (one carrying child :attr:`~PlotSpec.panels`) is
        renderable only when the backend can draw its ``COMPOSITE`` kind **and**
        every panel — so a backend whose composite path tiles panels (plotly)
        still falls back when a panel uses a kind it declines.
        """
        if self._is_three_d(spec) and not self.supports_3d:
            return False
        if not self.can_render(spec.kind):
            return False
        if not all(self.can_render(layer.kind) for layer in spec.layers):
            return False
        return all(self.can_render_spec(panel) for panel in spec.panels)

    @staticmethod
    def _is_three_d(spec: PlotSpec) -> bool:
        """Whether ``spec`` needs 3-D drawing support."""
        if spec.ndim == 3 or spec.z is not None:
            return True
        return any(layer.kind in (PlotKind.LINE3D, PlotKind.SURFACE3D) for layer in spec.layers)


# ---------------------------------------------------------------------------
# RenderResult
# ---------------------------------------------------------------------------


@dataclass
class RenderResult:
    """An optional typed description of what a renderer produced.

    Backends may return a bare native figure handle (the dispatch forwards it
    untouched), but the *data-export* backends (``json`` / ``threejs``) have no
    figure — they produce a serializable payload.  :class:`RenderResult` is the
    uniform envelope a backend can return so a caller can tell a figure from a
    payload without backend-specific knowledge.

    Parameters
    ----------
    backend : str
        The backend that produced this result.
    figure : Any, optional
        A live figure handle (a matplotlib ``Figure``, a plotly ``Figure``), or
        ``None`` for a pure data-export backend.
    payload : Any, optional
        A serializable export payload (a JSON-ready dict, a three.js
        BufferGeometry mapping), or ``None`` for a figure-only backend.
    mimetype : str, optional
        The payload's MIME type when relevant (e.g. ``"application/json"``,
        ``"text/html"``), else ``None``.
    kind : PlotKind, optional
        The semantic kind that was rendered, for provenance.
    meta : dict, optional
        Backend-specific extras (figure size, the axes handle, …).
    """

    backend: str
    figure: Any = None
    payload: Any = None
    mimetype: str | None = None
    kind: PlotKind | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize ``kind`` to :class:`~tsdynamics.viz.spec.PlotKind` if set."""
        if self.kind is not None:
            self.kind = PlotKind(self.kind)


# ---------------------------------------------------------------------------
# Renderer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Renderer(Protocol):
    """The contract a rendering backend satisfies (``runtime_checkable``).

    A renderer is a **callable** ``(spec, **kw) -> Any`` that consumes a
    :class:`~tsdynamics.viz.spec.PlotSpec` and returns a figure handle, a
    :class:`RenderResult`, or an export payload.  It carries a
    :attr:`capabilities` descriptor so the dispatch can ask, before calling,
    whether it can draw a given spec and otherwise fall back.

    Because :class:`~typing.Protocol` is ``runtime_checkable``, the dispatch can
    ``isinstance(obj, Renderer)`` to tell a capability-carrying backend from a
    plain callable (the latter is treated as a universal fallback that draws
    anything).
    """

    capabilities: RendererCapabilities

    def __call__(self, spec: PlotSpec, /, **kwargs: Any) -> Any:
        """Render ``spec``, returning a figure / :class:`RenderResult` / payload."""
        ...
