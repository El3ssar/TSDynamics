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
- :func:`accepted_render_kwargs` / :func:`check_render_kwargs` — the *keyword*
  half of the same honesty principle.  Every in-tree renderer is registered as a
  ``**kw`` wrapper around a core that ends in a catch-all, so an unknown render
  keyword used to be accepted and dropped in silence by all four backends.  These
  resolve what each backend genuinely reads (its own declaration, the in-tree
  table, or introspection of its callable) and raise
  :class:`~tsdynamics.errors.InvalidParameterError` for anything else — while
  leaving an undeclared out-of-tree backend's documented ``**kwargs``
  pass-through alone.

This module is **import-light**: it pulls in no plotting backend (only the
backend-agnostic spec IR), so importing it never drags matplotlib/plotly into
``sys.modules`` and the ``import tsdynamics`` no-plot-library guarantee holds.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .._visibility import dir_without, listing_dir
from ..spec import PlotKind, PlotSpec

__all__ = [
    "Renderer",
    "RendererCapabilities",
    "RenderResult",
    "VisualizationDegraded",
    "accepted_render_kwargs",
    "check_render_kwargs",
    "style_honoring_gaps",
]

__dir__ = listing_dir(__all__)


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
    3. **A forced frame mismatch**: ``on="force"`` overlaid two specs that are
       drawings of *different* spaces or planes (see
       :func:`tsdynamics.viz._frames.check_overlay`).  The figure is drawn as
       asked; the warning records that the axes mean two things at once, so the
       picture may not mean what it looks like.

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

#: Figure-geometry / presentation :class:`~tsdynamics.viz.style.Theme` fields that
#: the **plotly** backend does not honor.  ``figsize`` / ``dpi`` /
#: ``layout_engine`` are matplotlib output-geometry concepts: plotly sizes a
#: figure from ``layout.width`` / ``layout.height`` (in *pixels*, and only when
#: explicitly set) and lays it out in the browser, and the backend wires neither —
#: a ``publication`` theme's ``figsize=(5, 3.5)`` at 300 dpi reaches a plotly
#: figure as nothing at all.  ``autostyle`` **is** honored (``_core`` /
#: ``_threed`` both resolve it through
#: :func:`~tsdynamics.viz.producers.autostyle_enabled`), so it is absent here.
_PLOTLY_THEME_GEOMETRY_GAPS: frozenset[str] = frozenset({"figsize", "dpi", "layout_engine"})

#: The same four fields for **threejs**, which honors *none* of them: the exporter
#: emits geometry buffers plus a ``metadata.theme`` block carrying only
#: ``background`` / ``palette``, and the reference loader sizes its canvas from the
#: host page.  ``autostyle`` (density-aware line width / opacity) is a raster
#: stroke concept the WebGL line material does not take.
_THREEJS_THEME_GEOMETRY_GAPS: frozenset[str] = frozenset(
    {"figsize", "dpi", "layout_engine", "autostyle"}
)


#: Per-backend **figure-geometry** gaps — the four :class:`~tsdynamics.viz.style.Theme`
#: fields this phase added (``figsize`` / ``dpi`` / ``layout_engine`` /
#: ``autostyle``).  They were outside the honoring contract when they landed:
#: :func:`style_honoring_gaps` returned ``[]`` for matplotlib *and* plotly and, for
#: threejs, named only the six pre-existing theme fields — so three of the four
#: were accepted and dropped in silence by two backends.  matplotlib honors all
#: four (``mpl._core.figure_geometry`` resolves the first three,
#: :func:`~tsdynamics.viz.producers.autostyle_enabled` the fourth), so it has no
#: entry.  Sourced from either the resolved theme **or** ``spec.meta`` (what
#: ``spec.size(...)`` / ``save(size=, dpi=)`` write), because both reach the same
#: renderer field.
_BACKEND_GEOMETRY_GAPS: dict[str, frozenset[str]] = {
    "plotly": _PLOTLY_THEME_GEOMETRY_GAPS,
    "threejs": _THREEJS_THEME_GEOMETRY_GAPS,
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

# ---------------------------------------------------------------------------
# Per-backend render-keyword declarations (the "no silently swallowed typo" gate)
# ---------------------------------------------------------------------------

#: Keywords the dispatcher itself may inject into any renderer call, so every
#: backend accepts them regardless of what it declares.  ``warn`` is passed by
#: :func:`~tsdynamics.viz.render.render_spec` after it has emitted the one
#: consolidated :class:`VisualizationDegraded` warning.
_DISPATCHER_INJECTED_KWARGS: frozenset[str] = frozenset({"warn"})

#: Canonical backend name → the render keywords that backend actually reads.
#:
#: Every in-tree renderer is registered as a ``def _render(spec, /, **kw)``
#: wrapper around a core function, so the registered callable's signature says
#: nothing and ``inspect`` cannot recover the truth.  Worse, three of the four
#: cores end in a ``**_kw`` / ``**_ignored`` catch-all, so a misspelled option was
#: accepted and dropped in silence by **all four** backends —
#: ``spec.render(backend=b, totally_bogus_kwarg=42)`` returned a figure, no
#: warning, on matplotlib, plotly, json and three.js alike.  This table is the
#: declaration that makes the check possible; it is kept honest by
#: ``tests/test_viz_dispatch.py::test_declared_render_kwargs_match_each_backend_core``,
#: which introspects each backend's real core function and fails if the two
#: disagree.  A backend outside this table (an out-of-tree plugin) is validated by
#: introspecting its own callable instead — and a plugin whose signature ends in
#: ``**kwargs`` keeps its documented free pass-through.
_BUILTIN_RENDER_KWARGS: dict[str, frozenset[str]] = {
    # tsdynamics.viz.render.mpl._core.render
    # ``ax`` is matplotlib's alone by construction: it *is* a matplotlib Axes, so
    # ``spec.render(backend="plotly", ax=...)`` raising here (naming plotly's
    # accepted set) is the correct answer rather than an oversight.
    "matplotlib": frozenset({"figsize", "path", "ax"}),
    # tsdynamics.viz.render.plotly._core.render
    "plotly": frozenset({"html", "path", "full_html", "include_plotlyjs"}),
    # tsdynamics.viz.render.json (the registered closure)
    "json": frozenset({"path", "indent", "raw"}),
    # tsdynamics.viz.render.threejs (the registered closure)
    "threejs": frozenset(
        {
            "path",
            "html",
            "indent",
            "raw",
            "max_points",
            "decimals",
            "assets",
            "loader_url",
            "poster",
            "axes",
            "background",
            "title",
        }
    ),
}


def accepted_render_kwargs(backend_name: str, renderer: Any = None) -> frozenset[str] | None:
    """Return the render keywords ``backend_name`` accepts, or ``None`` for "any".

    Resolution order, most authoritative first:

    1. the backend's own :attr:`RendererCapabilities.render_kwargs` declaration
       (an out-of-tree backend can be explicit and get the same protection);
    2. :data:`_BUILTIN_RENDER_KWARGS` for the four in-tree backends;
    3. introspection of the renderer callable — its named keyword parameters,
       unless it declares a ``**kwargs`` catch-all, in which case the answer is
       ``None`` ("accepts anything", the documented pass-through).

    Parameters
    ----------
    backend_name : str
        The resolved backend name (aliases are normalised internally).
    renderer : callable, optional
        The renderer callable, used for step 3.

    Returns
    -------
    frozenset of str or None
        The accepted keyword names, or ``None`` when the backend accepts any.
    """
    import inspect

    canonical = _normalize_backend_name(backend_name)

    caps = getattr(renderer, "capabilities", None)
    declared = getattr(caps, "render_kwargs", None)
    if declared is not None:
        return frozenset(declared) | _DISPATCHER_INJECTED_KWARGS

    builtin = _BUILTIN_RENDER_KWARGS.get(canonical)
    if builtin is not None:
        return builtin | _DISPATCHER_INJECTED_KWARGS

    if renderer is None:
        return None
    try:
        sig = inspect.signature(renderer)
    except (TypeError, ValueError):  # pragma: no cover - builtin / C callable
        return None
    params = list(sig.parameters.values())
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None  # documented free pass-through — cannot know, do not guess
    # Drop the leading ``spec`` parameter: it is the positional the dispatcher
    # supplies, not an option, and naming it in "this backend accepts …" would
    # invite a caller to pass it twice.
    options = params[1:] if params and params[0].kind is not params[0].KEYWORD_ONLY else params
    return (
        frozenset(p.name for p in options if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
        | _DISPATCHER_INJECTED_KWARGS
    )


def check_render_kwargs(backend_name: str, renderer: Any, kwargs: Mapping[str, Any]) -> None:
    """Raise if ``kwargs`` carries a keyword ``backend_name`` does not read.

    The gate that turns a silently swallowed typo into an error naming both the
    offending key and what the chosen backend does accept.  A backend whose
    accepted set is unknown (:func:`accepted_render_kwargs` returns ``None`` — an
    out-of-tree renderer with a ``**kwargs`` catch-all and no declaration) is left
    alone: its pass-through is the documented contract.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        Naming each unaccepted keyword and listing the backend's accepted set.
    """
    accepted = accepted_render_kwargs(backend_name, renderer)
    if accepted is None:
        return
    unknown = sorted(k for k in kwargs if k not in accepted)
    if not unknown:
        return
    from tsdynamics.errors import InvalidParameterError

    public = sorted(accepted - _DISPATCHER_INJECTED_KWARGS)
    raise InvalidParameterError(
        f"backend {backend_name!r} got unexpected render keyword argument(s) "
        f"{', '.join(repr(u) for u in unknown)}. "
        f"{backend_name!r} accepts: {', '.join(public) or '(none)'}. "
        "Backends differ in what they read — pass the keyword to the backend that "
        "owns it (e.g. figsize=  to matplotlib, include_plotlyjs=  to plotly, "
        "max_points=  to threejs)."
    )


def _baseline_theme() -> Any:
    """Return the library's shipped ``default`` theme (the "untouched" reference)."""
    from tsdynamics.viz.style import THEMES

    return THEMES["default"]


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
    4. **Figure geometry**: the ``figsize`` / ``dpi`` / ``layout_engine`` /
       ``autostyle`` knobs, whether they arrive from the resolved
       :class:`~tsdynamics.viz.style.Theme` or from ``spec.meta`` (``spec.size(...)``).
       matplotlib honors all four; plotly honors only ``autostyle``; three.js
       honors none.

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
    # [M42] ``resolved_theme``, not ``_theme``: a SESSION-DEFAULT theme
    # (``ts.viz.themes.use("lab")``) is what the renderer actually applies, and
    # reading the per-plot override alone meant the same theme on the same
    # backend reported seven dropped fields one way and none the other.
    #
    # ``resolved_theme`` is always a full Theme, so "did the caller set this?"
    # is answered by comparing against the BASELINE default rather than against
    # ``None`` — otherwise every render of an unthemed plot warns about the
    # library's own defaults, which is noise, not honesty.
    if theme_gaps:
        t = spec.resolved_theme
        base = _baseline_theme()
        for field in ("foreground", "font_family", "font_size", "title_size", "grid_color"):
            if field in theme_gaps:
                value = getattr(t, field)
                if value is not None and value != getattr(base, field):
                    gaps.add(f"theme.{field}")
        if "grid" in theme_gaps and t.grid and t.grid != base.grid:
            gaps.add("theme.grid")
        if (
            "grid_alpha" in theme_gaps
            and t.grid_alpha is not None
            and t.grid_alpha != base.grid_alpha
        ):  # noqa: E501
            gaps.add("theme.grid_alpha")

    # ── 3b. Figure-geometry gaps (figsize / dpi / layout_engine / autostyle) ──
    # These reach a renderer from *two* places — the resolved theme and
    # ``spec.meta`` (what ``spec.size(...)`` writes) — so both are consulted.  Only
    # a value that is actually set is reported: ``autostyle`` defaults to ``True``
    # and ``layout_engine`` to ``None``, and warning about an untouched default
    # every render would be noise, not honesty.
    geometry_gaps = _BACKEND_GEOMETRY_GAPS.get(canonical_name, frozenset())
    if geometry_gaps:
        gt = spec.resolved_theme
        gmeta = spec.meta if isinstance(spec.meta, dict) else {}
        if "figsize" in geometry_gaps and (
            gt.figsize is not None or gmeta.get("figsize") is not None
        ):
            gaps.add("figsize")
        if "dpi" in geometry_gaps and (gt.dpi is not None or gmeta.get("dpi") is not None):
            gaps.add("dpi")
        if "layout_engine" in geometry_gaps and gt.layout_engine is not None:
            gaps.add("layout_engine")
        if "autostyle" in geometry_gaps and (not gt.autostyle or "autostyle" in gmeta):
            gaps.add("autostyle")

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


def _normalize_extensions(exts: Iterable[str]) -> frozenset[str]:
    """Normalize an iterable of file extensions to lowercase, dot-prefixed form."""
    out: set[str] = set()
    for e in exts:
        s = str(e).lower()
        out.add(s if s.startswith(".") else f".{s}")
    return frozenset(out)


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
    render_kwargs : frozenset of str, optional
        The render keywords this backend actually reads.  ``None`` (the default)
        means "undeclared": the in-tree backends resolve through
        :data:`_BUILTIN_RENDER_KWARGS` and an out-of-tree one through
        introspection of its callable (a ``**kwargs`` catch-all keeps its
        documented free pass-through).  Declaring the set opts a plugin into the
        same "an unknown keyword raises instead of being swallowed" protection the
        in-tree backends get — see :func:`check_render_kwargs`.
    """

    name: str
    kinds: frozenset[PlotKind] | None = None
    supports_3d: bool = False
    interactive: bool = False
    web_export: bool = False
    data_export: bool = False
    #: File extensions (lowercase, **with** the leading dot) this backend can
    #: write for a **static** plot.  Empty means "writes no still image".
    writes_static: frozenset[str] = frozenset()
    #: File extensions this backend can write for an **animated** plot.  A movie
    #: format lives here and *not* in :attr:`writes_static`: measured,
    #: ``savefig`` refuses ``.mp4`` / ``.webm`` / ``.mov`` / ``.m4v`` / ``.apng``
    #: outright, so declaring them in one undivided ``writes`` made ``save`` promise
    #: four formats it could never produce and reject ``.webp``, which it can.
    writes_animated: frozenset[str] = frozenset()
    #: The render keywords this backend reads, or ``None`` (undeclared).  See
    #: :func:`accepted_render_kwargs`.
    render_kwargs: frozenset[str] | None = None

    def __dir__(self) -> list[str]:
        """Expose the **declaration** a backend author writes and a user reads.

        This record is one of the six plugin doors' payloads: a renderer says what
        it draws (``name`` ``kinds`` ``supports_3d`` ``interactive``
        ``web_export`` ``data_export``), what it writes (``writes_static``
        ``writes_animated``, and the undivided ``writes`` that
        ``ts.viz.renderers.find(writes=".svg")`` reads), and which render keywords
        it honours (``render_kwargs``).

        :meth:`can_render`, :meth:`can_render_spec` and :meth:`can_save` are the
        **dispatcher's** questions of that declaration — they are how
        ``ts.plot(..., backend=…)`` and ``Plot.save`` pick a backend and decide
        whether to fall back.  Answering them by hand re-decides something the
        library has already decided.  All three stay public and tested.
        """
        return dir_without(self, {"can_render", "can_render_spec", "can_save"})

    @property
    def writes(self) -> frozenset[str]:
        """Every extension this backend can write, static **or** animated.

        The undivided view, kept because "which formats does this backend know?"
        is a real question (``ts.viz.renderers.find(writes=".svg")`` asks it).
        :meth:`can_save` is the one that decides a *particular* save, because that
        question always has an animated-or-not half.
        """
        return self.writes_static | self.writes_animated

    def can_save(self, ext: str, *, animated: bool = False) -> bool:
        """Whether this backend can **write** a file with extension ``ext``.

        The other half of the save contract (:meth:`can_render_spec` answers "can
        you draw it?"; this answers "can you write it?").  :meth:`Plot.save`
        resolves ``(extension, backend)`` through this predicate and **raises**
        when no capable writer exists — rather than returning a path it never
        wrote, which is what it used to do for an animated composite ``.html`` and
        for ``save(..., backend="threejs")`` with an ``.html`` name.

        Parameters
        ----------
        ext : str
            The output extension, with or without a leading dot; case-insensitive.
        animated : bool, optional
            Whether the spec carries an :class:`~tsdynamics.viz.spec.Animation`.
            The two halves genuinely differ: matplotlib writes ``.png`` only
            statically and ``.mp4`` only animated, and ``.gif`` both ways.

        Returns
        -------
        bool
        """
        e = ext.lower()
        if not e.startswith("."):
            e = f".{e}"
        return e in (self.writes_animated if animated else self.writes_static)

    @classmethod
    def all_kinds(
        cls,
        name: str,
        *,
        supports_3d: bool = True,
        interactive: bool = False,
        web_export: bool = False,
        data_export: bool = False,
        writes: Iterable[str] = (),
        writes_animated: Iterable[str] | None = None,
        render_kwargs: Iterable[str] | None = None,
    ) -> RendererCapabilities:
        """Build capabilities for a backend that draws **every** kind.

        The shorthand the reference renderer (and any other "draws anything"
        backend) uses: ``kinds=None`` means no kind is ever declined.

        ``writes`` is the *static* extension set; ``writes_animated`` defaults to
        it, so a backend with one file writer declares once and a backend whose
        movie formats differ (matplotlib) declares both.
        """
        return cls(
            name=name,
            kinds=None,
            supports_3d=supports_3d,
            interactive=interactive,
            web_export=web_export,
            data_export=data_export,
            writes_static=_normalize_extensions(writes),
            writes_animated=_normalize_extensions(
                writes if writes_animated is None else writes_animated
            ),
            render_kwargs=None if render_kwargs is None else frozenset(render_kwargs),
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
        writes: Iterable[str] = (),
        writes_animated: Iterable[str] | None = None,
        render_kwargs: Iterable[str] | None = None,
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
            writes_static=_normalize_extensions(writes),
            writes_animated=_normalize_extensions(
                writes if writes_animated is None else writes_animated
            ),
            render_kwargs=None if render_kwargs is None else frozenset(render_kwargs),
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

        The 3-D test is :attr:`PlotSpec.is_three_d` — the **one** definition, on
        the spec.  It used to be a private copy here (and two more in the
        renderers); a capability check that disagreed with a renderer about what
        "3-D" means is a dispatch bug waiting to happen.
        """
        if spec.is_three_d and not self.supports_3d:
            return False
        if not self.can_render(spec.kind):
            return False
        if not all(self.can_render(layer.kind) for layer in spec.layers):
            return False
        return all(self.can_render_spec(panel) for panel in spec.panels)


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
