"""The ``threejs`` data-export backend (stream VIZ-THREEJS-EXPORT).

This in-tree backend is a *data exporter*, not a drawer: it consumes a
:class:`~tsdynamics.viz.spec.PlotSpec` and lowers its drawable layers into a
**three.js BufferGeometry-ready** JSON-able payload — flat ``Float32``-style
positions / colors / indices plus a top-level ``metadata`` block (labels / units /
bounds / camera).  A web frontend reads the payload and builds
``THREE.BufferGeometry`` objects directly, with no re-running of the analysis.

The geometry lowering — and the documented payload schema — live in
:mod:`._lower` (:func:`tsdynamics.viz.render.threejs._lower.lower_spec`).  This
module is the registry wiring.

Because the lowering is **pure Python** (only the standard library, NumPy, and
the backend-agnostic spec IR — no plotting library, no optional dependency, in
particular no three.js / WebGL / matplotlib import), :func:`register` registers
**unconditionally**, so :func:`tsdynamics.viz.render.register_builtin_renderers`
always wires it: a caller can always reach ``spec.render("threejs")``.  Importing
this module imports no plotting library, so the ``import tsdynamics`` no-plot
guarantee holds.

Behaviour
---------
The renderer ``(spec, *, path=None, indent=None, raw=False) -> dict | RenderResult``:

- with no ``path`` it **returns** the payload — by default a
  :class:`~tsdynamics.viz.render.caps.RenderResult` carrying the payload ``dict``
  with ``mimetype="application/json"`` (pass ``raw=True`` to get the bare
  ``dict``);
- with a ``path`` it **writes** the payload as JSON to that file (UTF-8) and
  returns the path (a :class:`pathlib.Path`).

Its :class:`~tsdynamics.viz.render.caps.RendererCapabilities` declare
``kinds=None`` (it lowers *any* spec — the geometry walk simply skips a layer
mark with no BufferGeometry analogue), ``data_export=True`` (so default
``spec.render()`` selection prefers a real drawing backend and never returns this
payload by surprise), ``web_export=True`` (the payload is a web artifact), and
``supports_3d=True``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..caps import RendererCapabilities, RenderResult

if TYPE_CHECKING:
    from tsdynamics.registry import Registry

    from ...spec import PlotSpec

__all__ = ["register"]

#: The registry name the threejs exporter registers under.
_BACKEND_NAME = "threejs"

#: The MIME type the exporter tags its payload with (a JSON document).
_MIMETYPE = "application/json"


def register(registry: Registry) -> bool:
    """Register the ``threejs`` data-export backend into ``registry``.

    Builds the exporter callable, attaches an all-kinds
    :class:`~tsdynamics.viz.render.caps.RendererCapabilities` with
    ``data_export=True`` / ``web_export=True`` / ``supports_3d=True`` (``kinds=None``
    — it lowers any spec rather than declining), and adds it under ``"threejs"``.

    Registers **unconditionally** (the lowering is pure Python — no optional
    dependency), so :func:`tsdynamics.viz.render.register_builtin_renderers` always
    wires it.  The hook is a no-op when the backend is already registered, so a
    second registration pass does not re-register.

    The exporter accepts every kind (``kinds=None``) but is a *data exporter*
    (``data_export=True``), not a drawer, so the dispatch's default selection
    (:func:`tsdynamics.viz.render.select_renderer` with no backend) **skips** it in
    favour of a real drawing backend — it never shadows the matplotlib drawing
    default.  Selecting it by name (``spec.render("threejs")``) always reaches it.

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
        web_export=True,
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
        # Import the geometry lowering lazily (it is light — NumPy + the spec IR —
        # but keeping it in-method matches the other backends' shape).
        from ._lower import lower_spec

        payload = lower_spec(spec)
        if path is not None:
            out = Path(path)
            out.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
            return out
        if raw:
            return payload
        return RenderResult(
            backend=_BACKEND_NAME,
            payload=payload,
            mimetype=_MIMETYPE,
            kind=spec.kind,
        )

    _render.capabilities = capabilities  # type: ignore[attr-defined]
    registry.register(_BACKEND_NAME, _render)
    return True
