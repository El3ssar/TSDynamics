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
  cannot draw a spec and the dispatch silently falls back to another.

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
]


class VisualizationDegraded(UserWarning):
    """Warned when a backend cannot draw a kind and dispatch falls back.

    Visualization *degrades gracefully*: if the requested backend declines a
    spec's :class:`~tsdynamics.viz.spec.PlotKind` (or its 3-D-ness), the dispatch
    picks a capable backend — the matplotlib reference renderer is the universal
    fallback — and emits this warning so the caller knows the figure did not come
    from the backend they named.  It is a :class:`UserWarning` (not an error) so
    the call still returns a figure; a *hard* failure (no capable backend at all)
    raises instead.
    """


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
        """
        if self._is_three_d(spec) and not self.supports_3d:
            return False
        if not self.can_render(spec.kind):
            return False
        return all(self.can_render(layer.kind) for layer in spec.layers)

    @staticmethod
    def _is_three_d(spec: PlotSpec) -> bool:
        """Whether ``spec`` needs 3-D drawing support."""
        if spec.ndim == 3 or spec.z is not None:
            return True
        return any(layer.kind in (PlotKind.LINE3D, PlotKind.SURFACE3D) for layer in spec.layers)


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
