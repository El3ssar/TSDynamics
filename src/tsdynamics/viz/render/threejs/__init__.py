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
The renderer is
``(spec, *, path=None, html=None, indent=None, raw=False, max_points=…, decimals=…,
assets="inline", axes=True, …)``:

- with no ``path`` it **returns** the payload — by default a
  :class:`~tsdynamics.viz.render.caps.RenderResult` carrying the payload ``dict``
  with ``mimetype="application/json"`` (pass ``raw=True`` to get the bare
  ``dict``, or ``html=True`` to get the viewer page as a ``str``);
- with a ``path`` it **writes**: an ``.html`` / ``.htm`` path gets a complete,
  self-contained viewer page (see :mod:`._page`), any other path gets the payload
  as JSON.  Either way it returns the path (a :class:`pathlib.Path`).

**Two different documents share the ``.json`` extension**, so this is worth stating
plainly: ``spec.save("x.json")`` writes the *PlotSpec IR envelope* (the json
backend — ``{"schema_version", "spec"}``, reloadable with
:func:`tsdynamics.viz.export.from_json`), while
``spec.save("x.json", backend="threejs")`` writes the *BufferGeometry payload*
(``{"schema_version", "kind", "geometries", "metadata"}``).  They are not
interchangeable; ``backend=`` is the disambiguator.

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

from tsdynamics.errors import InvalidParameterError

from ..caps import RendererCapabilities, RenderResult
from ._lower import DEFAULT_DECIMALS as _DEFAULT_DECIMALS
from ._lower import DEFAULT_MAX_POINTS as _DEFAULT_MAX_POINTS
from ._page import (
    ASSET_MODES,
    DEFAULT_LOADER_URL,
    LOADER_FILENAME,
    THREE_VERSION,
    loader_path,
    loader_source,
    render_page,
    write_loader_asset,
)

if TYPE_CHECKING:
    from tsdynamics.registry import Registry

    from ...spec import PlotSpec

__all__ = [
    "ASSET_MODES",
    "DEFAULT_LOADER_URL",
    "LOADER_FILENAME",
    "SAVE_EXTENSIONS",
    "THREE_VERSION",
    "loader_path",
    "loader_source",
    "register",
    "render_page",
    "write_loader_asset",
]

#: File extensions this backend can genuinely **write**.  ``PlotSpec.save`` uses it
#: to refuse an ``(extension, backend)`` pair loudly instead of returning a path it
#: never wrote — the failure mode this backend used to embody by writing a JSON
#: document into a ``.html`` file.
SAVE_EXTENSIONS = frozenset({".html", ".htm", ".json"})

#: The HTML extensions that get a viewer page rather than a raw JSON payload.
_HTML_EXTENSIONS = (".html", ".htm")

#: The registry name the threejs exporter registers under.
_BACKEND_NAME = "threejs"

#: The MIME type the exporter tags its payload with (a JSON document).
_MIMETYPE = "application/json"


def _check_page_extension(out: Path) -> None:
    """Refuse to write an HTML *document* into a path that promises another format.

    The mirror of :func:`_check_writable_extension` for the page branch, and the
    hole that guard left open: ``html=True`` short-circuits to the page writer
    *before* any extension check, so ``render(path="fig.png", html=True)`` wrote a
    ``<!doctype html>`` document into a ``.png`` and returned the path.  That is the
    same "returns a path it did not honestly write" failure the JSON branch guard
    exists to kill, one branch over — an existence check passes, a size check
    passes, and every image viewer rejects the file.
    """
    suffix = out.suffix.lower()
    if suffix in _HTML_EXTENSIONS:
        return
    raise InvalidParameterError(
        f"the 'threejs' viewer page is an HTML document, so it cannot be written to "
        f"{suffix or 'a file with no extension'}; use "
        f"'{out.with_suffix('.html').name}'. To write the geometry payload instead, "
        f"drop html=True and save to '{out.with_suffix('.json').name}'."
    )


def _check_writable_extension(out: Path) -> None:
    """Refuse to write the payload into a file whose extension promises something else.

    :data:`SAVE_EXTENSIONS` is a *declared* contract, so it has to be enforced
    somewhere or it is decoration.  Without this guard the JSON fallback accepted
    **any** path, so ``spec.save("fig.png", backend="threejs")`` wrote a JSON
    document into a ``.png`` and returned the path — a caller then has a file that
    every image viewer rejects, which is exactly the "returns a path it did not
    honestly write" failure mode the v6 save contract exists to kill.  (It is the
    same bug the ``.html`` path used to have, merely moved to another extension.)

    Raising names a spelling that works, because the two things a caller reaching
    for ``.png`` actually wants are a real image (matplotlib) or the interactive
    page this backend does produce.
    """
    suffix = out.suffix.lower()
    if suffix in SAVE_EXTENSIONS:
        return
    raise InvalidParameterError(
        f"the 'threejs' backend cannot write {suffix or 'a file with no extension'}; "
        f"it writes {sorted(SAVE_EXTENSIONS)} only. For a self-contained interactive "
        f"page use '{out.with_suffix('.html').name}'; for a static image drop "
        "backend='threejs' and let matplotlib render it."
    )


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

    # ``writes=`` is what makes ``RendererCapabilities.can_save`` — the *published*
    # extension point ``PlotSpec.save`` consults before its own table — tell the
    # truth about this backend.  Declaring the formats only on the module constant
    # left ``caps.can_save(".html")`` answering False for the one backend whose
    # whole purpose is writing an embeddable ``.html``.
    capabilities = RendererCapabilities.all_kinds(
        _BACKEND_NAME,
        supports_3d=True,
        web_export=True,
        data_export=True,
        writes=SAVE_EXTENSIONS,
    )

    def _render(
        spec: PlotSpec,
        /,
        *,
        path: str | os.PathLike[str] | None = None,
        html: bool | None = None,
        indent: int | None = None,
        raw: bool = False,
        max_points: int | None = _DEFAULT_MAX_POINTS,
        decimals: int | None = _DEFAULT_DECIMALS,
        assets: str = "inline",
        loader_url: str = DEFAULT_LOADER_URL,
        poster: bool = True,
        axes: bool = True,
        background: str | None = None,
        title: str | None = None,
        warn: bool = True,
        **_ignored: Any,
    ) -> Any:
        # ``warn`` is injected by the dispatcher (which has already emitted the one
        # consolidated honoring warning); this backend emits no per-key style
        # warnings of its own, so it simply accepts and ignores it.  ``**_ignored``
        # exists only to absorb further dispatcher-injected kwargs — it is NOT an
        # invitation to pass this backend options it does not implement.  Closing
        # that swallow belongs at the dispatcher, once, not in each backend.
        del warn

        # Import the geometry lowering lazily (it is light — NumPy + the spec IR —
        # but keeping it in-method matches the other backends' shape).
        from ._lower import lower_spec

        payload = lower_spec(spec, max_points=max_points, decimals=decimals)

        wants_page = bool(html) or (
            path is not None and str(path).lower().endswith(_HTML_EXTENSIONS)
        )
        if wants_page:
            if path is not None:
                _check_page_extension(Path(path))
            page = render_page(
                payload,
                spec=spec,
                title=title,
                background=background,
                assets=assets,
                loader_url=loader_url,
                poster=poster,
                axes=axes,
            )
            if path is None:
                return page
            out = Path(path)
            out.write_text(page, encoding="utf-8")
            return out

        if path is not None:
            out = Path(path)
            _check_writable_extension(out)
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
    _render.save_extensions = SAVE_EXTENSIONS  # type: ignore[attr-defined]
    registry.register(_BACKEND_NAME, _render)
    return True
