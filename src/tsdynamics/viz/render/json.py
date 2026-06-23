"""The ``json`` data-export backend (stream VIZ-JSON-EXPORT).

This in-tree backend is a *data exporter*, not a drawer: it consumes a
:class:`~tsdynamics.viz.spec.PlotSpec` and serializes it to the versioned JSON
text produced by :func:`tsdynamics.viz.export.to_json`.  It draws nothing, so its
:class:`~tsdynamics.viz.render.caps.RendererCapabilities` declare ``kinds=None``
(it accepts *every* kind — there is no kind it cannot serialize) and
``data_export=True``.

Because it is pure standard-library :mod:`json` — no optional dependency — the
:func:`register` hook registers **unconditionally**, so
:func:`tsdynamics.viz.render.register_builtin_renderers` always wires it: a
caller can always reach ``spec.render("json")`` / ``result.plot(backend="json")``
regardless of which plotting libraries are installed.

The package exposes one hook, :func:`register`, called by
:func:`tsdynamics.viz.render.register_builtin_renderers`.  Importing this module
imports no plotting library (only :mod:`json` and the backend-agnostic spec IR),
so the ``import tsdynamics`` no-plot-library guarantee holds.

Behaviour
---------
The renderer ``(spec, *, path=None, indent=None) -> str | RenderResult``:

- with no ``path`` it **returns** the JSON payload — by default a
  :class:`~tsdynamics.viz.render.caps.RenderResult` carrying the JSON string as
  its ``payload`` and ``mimetype="application/json"`` (pass ``raw=True`` to get
  the bare ``str`` instead);
- with a ``path`` it **writes** the JSON to that file (UTF-8) and returns the
  path (a :class:`pathlib.Path`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .caps import RendererCapabilities, RenderResult

if TYPE_CHECKING:
    from tsdynamics.registry import Registry

    from ..spec import PlotSpec

__all__ = ["register"]

#: The registry name the json exporter registers under.
_BACKEND_NAME = "json"

#: The MIME type the exporter tags its payload with.
_MIMETYPE = "application/json"


def register(registry: Registry) -> bool:
    """Register the ``json`` data-export backend into ``registry``.

    Builds the exporter callable, attaches an all-kinds
    :class:`~tsdynamics.viz.render.caps.RendererCapabilities` with
    ``data_export=True`` (``kinds=None`` — it serializes any spec rather than
    declining), and adds it under ``"json"``.

    Registers **unconditionally** (the exporter is pure standard-library
    :mod:`json`, no optional dependency) — so
    :func:`tsdynamics.viz.render.register_builtin_renderers` always wires it.  The
    hook is a no-op (bar re-seating, below) when the backend is already
    registered, so a second registration pass does not re-register.

    The exporter accepts every kind (``kinds=None``) but is a *data exporter*
    (``data_export=True``), not a drawer, so the dispatch's default selection
    (:func:`tsdynamics.viz.render.select_renderer` with no backend) **skips** it
    in favour of a real drawing backend — registry position is irrelevant, and a
    repeat ``register_builtin_renderers`` pass is a plain no-op.  Selecting it by
    name (``spec.render("json")``) always reaches it.

    Parameters
    ----------
    registry : Registry
        The :data:`tsdynamics.registry.renderers` container to add the backend to.

    Returns
    -------
    bool
        ``True`` if the backend was newly registered, ``False`` if it was already
        present.
    """
    if _BACKEND_NAME in registry:
        return False

    capabilities = RendererCapabilities.all_kinds(
        _BACKEND_NAME,
        supports_3d=True,
        web_export=False,
        data_export=True,
    )

    def _render(
        spec: PlotSpec,
        /,
        *,
        path: str | os.PathLike[str] | None = None,
        indent: int | None = None,
        raw: bool = False,
        **_ignored: Any,
    ) -> Any:
        # Import the serializer lazily (it is light — stdlib json + the spec IR —
        # but keeping it in-method matches the other backends' shape).
        from ..export import to_json

        text = to_json(spec, indent=indent)
        if path is not None:
            out = Path(path)
            out.write_text(text, encoding="utf-8")
            return out
        if raw:
            return text
        return RenderResult(
            backend=_BACKEND_NAME,
            payload=text,
            mimetype=_MIMETYPE,
            kind=spec.kind,
        )

    _render.capabilities = capabilities  # type: ignore[attr-defined]
    registry.register(_BACKEND_NAME, _render)
    return True
