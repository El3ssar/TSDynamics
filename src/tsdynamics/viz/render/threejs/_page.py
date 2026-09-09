"""The self-contained HTML page the ``threejs`` backend writes (stream VIZ-WEB-EXPORT).

``spec.save("attractor.html", backend="threejs")`` used to write a **JSON document
with an ``.html`` extension** — a file no browser would render, produced by a
backend whose whole reason to exist is web embedding.  This module is the fix: it
emits a real page that opens by double-clicking it.

What "self-contained" means here, precisely
-------------------------------------------
The emitted document makes **zero same-origin requests**.  The geometry payload and
the reference loader are both *inlined* (under the default ``assets="inline"``), so
the artifact works from a ``file://`` URL, from inside a zip, or pasted into a CMS —
none of which can serve a sibling ``.json`` for a ``fetch()`` to find.  That
``fetch``-a-sibling shape was the previous design's fatal flaw: it required a web
server to view a file whose purpose was to be portable.

The one external reference is the **pinned three.js build**, declared through an ES
module import map.  three.js is ~600 KB of library that is not ours to vendor, and
pinning an exact version in an import map is the standard way to depend on it.  When
that CDN is unreachable the page degrades to a **poster image** rendered through the
matplotlib backend and inlined as a ``data:`` URI — so an offline reader still sees
the attractor, and a reader with no JavaScript at all sees it through ``<noscript>``.

``assets``
----------
``"inline"`` (default)
    The loader source is inlined into the module script.  Fully portable.
``"link"``
    The page imports the loader from ``loader_url`` instead of inlining it — for a
    docs site or gallery that ships **many** viewers and wants the browser to cache
    one shared copy.  Write that copy with :func:`write_loader_asset`.
"""

from __future__ import annotations

import base64
import contextlib
import html as _html
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tsdynamics.errors import InvalidParameterError

if TYPE_CHECKING:
    from ...spec import PlotSpec

__all__ = [
    "ASSET_MODES",
    "DEFAULT_LOADER_URL",
    "LOADER_FILENAME",
    "THREE_VERSION",
    "loader_path",
    "loader_source",
    "render_page",
    "write_loader_asset",
]

#: The reference loader's filename, both in the package and when written out.
LOADER_FILENAME = "tsdyn-threejs-loader.js"

#: Default URL an ``assets="link"`` page imports the loader from (a sibling file).
DEFAULT_LOADER_URL = f"./{LOADER_FILENAME}"

#: The pinned three.js build the page's import map resolves ``"three"`` to.  Pinned
#: (never ``@latest``) so an emitted page cannot break when three.js ships a
#: breaking release: an artifact saved today must still render in five years.
THREE_VERSION = "0.160.0"

#: How the reference loader reaches the page.  See the module docstring.
ASSET_MODES = ("inline", "link")

_THREE_CDN = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}"

#: The dark stage the viewer paints when the payload's theme names no background.
_DEFAULT_BACKGROUND = "#0b0f14"


def loader_path() -> Path:
    """Absolute path to the reference loader **shipped inside the package**.

    This — not the ``docs/_static`` copy — is the source of truth.  Before v6 the
    loader lived only in ``docs/``, so it was in no wheel at all: the documented
    "copy this loader" story pointed at a file an installed user did not have.
    """
    return Path(__file__).resolve().parent / "_assets" / LOADER_FILENAME


def loader_source() -> str:
    """Read the reference loader's JavaScript source and return it as text."""
    return loader_path().read_text(encoding="utf-8")


def write_loader_asset(directory: str | Path) -> Path:
    """Write the reference loader into ``directory`` and return the written path.

    The companion to ``assets="link"``: a site that embeds many viewers writes the
    loader **once** here and points every page's ``loader_url`` at it, so the
    browser caches one copy instead of re-parsing it per page.
    """
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / LOADER_FILENAME
    out.write_text(loader_source(), encoding="utf-8")
    return out


def render_page(
    payload: dict[str, Any],
    *,
    spec: PlotSpec | None = None,
    title: str | None = None,
    background: str | None = None,
    assets: str = "inline",
    loader_url: str = DEFAULT_LOADER_URL,
    poster: bool = True,
    axes: bool = True,
) -> str:
    """Build the self-contained viewer page for a lowered ``payload``.

    Parameters
    ----------
    payload : dict
        A lowered threejs payload (:func:`tsdynamics.viz.render.threejs._lower.lower_spec`).
    spec : PlotSpec, optional
        The spec the payload came from.  Used **only** to render the poster
        fallback through the matplotlib backend; omit it (or pass ``poster=False``)
        to skip that.
    title : str, optional
        The document title; defaults to the payload's own title.
    background : str, optional
        Scene / page background CSS colour.  Defaults to the payload theme's
        background, else a dark stage.
    assets : {"inline", "link"}, optional
        How the reference loader reaches the page — see the module docstring.
    loader_url : str, optional
        The URL an ``assets="link"`` page imports the loader from.
    poster : bool, optional
        Whether to inline a matplotlib-rendered PNG as the no-WebGL / no-JS
        fallback (default ``True``).  Silently skipped when matplotlib is
        unavailable or declines the spec.
    axes : bool, optional
        Whether the viewer draws its labelled axes frame — box edges, nice-numbered
        ticks and the axis names from the payload's own ``metadata`` (default
        ``True``).  A saved page is a *plot*, so it carries a scale by default; pass
        ``False`` for a decorative, frameless attractor.

    Returns
    -------
    str
        A complete HTML document beginning with ``<!doctype html>``.

    Raises
    ------
    InvalidParameterError
        If ``assets`` is not one of :data:`ASSET_MODES`.  A page cannot be
        *partly* self-contained, so an unrecognised mode is a hard error rather
        than a silent fallback to ``"inline"``.
    """
    if assets not in ASSET_MODES:
        raise InvalidParameterError(
            f"threejs page assets={assets!r} is not supported; expected one of "
            f"{list(ASSET_MODES)}. 'inline' embeds the loader (a portable file:// "
            "artifact); 'link' imports it from loader_url (write it with "
            "tsdynamics.viz.render.threejs.write_loader_asset)."
        )

    meta = payload.get("metadata") or {}
    theme = meta.get("theme") or {}
    bg = background or theme.get("background") or _DEFAULT_BACKGROUND
    doc_title = title or payload.get("title") or "TSDynamics attractor"

    payload_json = _json_for_html(payload)
    poster_uri = _poster_data_uri(spec, dark=_is_dark(bg)) if poster else None
    loader_block = _loader_block(assets)
    boot = _boot_script(assets=assets, loader_url=loader_url, background=bg, axes=axes)

    poster_img = (
        f'<img id="tsdyn-poster" src="{poster_uri}" alt="{_html.escape(doc_title)}" />'
        if poster_uri
        else ""
    )
    noscript = (
        f'<noscript><img src="{poster_uri}" alt="{_html.escape(doc_title)}" '
        'style="width:100%;height:100%;object-fit:contain" /></noscript>'
        if poster_uri
        else '<noscript><p style="color:#9aa4c0;font:14px system-ui,sans-serif;padding:1rem">'
        "This interactive attractor needs JavaScript and WebGL.</p></noscript>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{_html.escape(doc_title)}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: {bg}; overflow: hidden; }}
  #tsdyn-viewer {{ position: absolute; inset: 0; }}
  #tsdyn-poster {{
    position: absolute; inset: 0; display: none; object-fit: contain;
    width: 100%; height: 100%; background: {bg};
  }}
</style>
<script type="importmap">
{{
  "imports": {{
    "three": "{_THREE_CDN}/build/three.module.js",
    "three/addons/": "{_THREE_CDN}/examples/jsm/"
  }}
}}
</script>
</head>
<body>
<div id="tsdyn-viewer"></div>
{poster_img}
{noscript}
<script type="application/json" id="tsdyn-payload">{payload_json}</script>
{loader_block}
{boot}
</body>
</html>
"""


def _is_dark(color: str) -> bool:
    """Whether a CSS colour reads as dark, so the poster can be themed to match.

    Only ``#rgb`` / ``#rrggbb`` are decided by luminance; anything else (a named
    colour, ``rgb()``, a gradient) is treated as light — the conservative answer,
    since a light poster on a light page is merely plain, while a dark poster on a
    light page is unreadable.
    """
    text = color.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        return False
    try:
        r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return False
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 110.0


def _json_for_html(payload: dict[str, Any]) -> str:
    r"""Serialize to compact JSON that is safe inside a ``<script>`` element.

    ``</script>`` anywhere inside the document — in a user's plot title, an axis
    label, a layer label — would terminate the block early and turn the rest of the
    payload into markup.  Escaping ``<`` as ``\\u003c`` is valid JSON that
    ``JSON.parse`` restores exactly, and makes that impossible.
    """
    text = json.dumps(payload, separators=(",", ":"))
    return text.replace("<", "\\u003c")


def _loader_block(assets: str) -> str:
    """Emit the inlined loader source as an inert ``text/plain`` block (or nothing).

    The loader is carried as **data**, not as a module script, so the boot code can
    import it *dynamically* and catch the failure — see :func:`_boot_script`.

    The block is inert (``type="text/plain"``), so the browser does not execute or
    even parse it; the boot code reads its ``textContent``.  Its one hazard is a
    literal ``</script`` inside the JavaScript, which would close the element early,
    so that is checked rather than assumed.  (The payload block escapes ``<``
    instead; JavaScript cannot, because ``<`` is an operator.)
    """
    if assets == "link":
        return ""
    src = loader_source()
    if "</script" in src.lower():
        raise InvalidParameterError(
            "the reference loader contains a literal '</script' sequence, which "
            "cannot be inlined into an HTML page. Use assets='link' (and "
            "write_loader_asset) until the loader is fixed."
        )
    return f'<script type="text/plain" id="tsdyn-loader">{src}</script>'


def _boot_script(*, assets: str, loader_url: str, background: str, axes: bool) -> str:
    """Build the module script that loads three.js, the loader, and the payload.

    **Every import here is dynamic, and that is the whole point.**  A *static*
    ``import`` that fails — an unreachable three.js CDN, a missing loader — aborts
    the entire module before any statement runs, so a ``try``/``catch`` around the
    render call would never execute and the poster fallback would never appear:
    the reader would get a blank page, which is the exact failure the poster exists
    to prevent.  (This is not hypothetical; it is what the first cut of this page
    did, caught by booting it in a browser with the CDN host broken.)

    So three.js is *probed* first, then the loader is imported — inline, from a
    ``blob:`` URL built out of the inert ``text/plain`` block; linked, from
    ``loader_url`` — and any failure in that chain lands in one ``catch`` that
    reveals the poster.
    """
    if assets == "link":
        import_loader = f'await import("{loader_url}")'
    else:
        import_loader = (
            "await import(URL.createObjectURL(new Blob("
            '[document.getElementById("tsdyn-loader").textContent], '
            '{ type: "text/javascript" })))'
        )

    return f"""<script type="module">
const container = document.getElementById("tsdyn-viewer");
const poster = document.getElementById("tsdyn-poster");
function degrade(err) {{
  if (container) container.style.display = "none";
  if (poster) poster.style.display = "block";
  console.warn("tsdyn-threejs: interactive viewer unavailable, showing the poster:", err);
}}
try {{
  await import("three");            // probe the pinned CDN build before anything else
  const mod = {import_loader};
  const payload = JSON.parse(document.getElementById("tsdyn-payload").textContent);
  mod.renderThreejsPayload(container, payload, {{
    background: "{background}",
    axes: {"true" if axes else "false"},
  }});
}} catch (err) {{
  degrade(err);
}}
</script>"""


def _poster_data_uri(spec: PlotSpec | None, *, dark: bool = False) -> str | None:
    """Render ``spec`` through matplotlib and return a base64 ``data:`` PNG URI.

    The poster is the honest degradation path: a reader with no WebGL, no network
    for the three.js CDN, or no JavaScript at all still sees the attractor.  It is
    rendered through the **always-present** matplotlib reference renderer, and any
    failure at all (matplotlib absent, the kind declined, a headless backend
    problem) simply yields ``None`` — a missing poster degrades the page, a raised
    exception would destroy it.

    ``dark`` themes the poster to match a dark page.  Without it the fallback is a
    white matplotlib figure dropped into a near-black page — which reads as a
    broken layout rather than a considered fallback.  It is applied only when the
    spec set **no** theme of its own: an explicit theme is the author's choice and
    is never overridden.
    """
    if spec is None:
        return None
    try:
        from ...spec import PlotSpec as _PlotSpec

        static = spec
        if getattr(spec, "animation", None) is not None:
            # A still of an animated spec is its final, fully-revealed frame.
            static = _PlotSpec.from_dict({**spec.to_dict(), "animation": None})
        if dark and getattr(spec, "_theme", None) is None:
            if static is spec:  # do not mutate the caller's spec
                static = _PlotSpec.from_dict(spec.to_dict())
            with contextlib.suppress(Exception):
                static.theme("dark")

        import io

        with warnings.catch_warnings():
            # The poster is an internal fallback render; a backend's honoring
            # warnings are about the *interactive* export the caller asked for.
            warnings.simplefilter("ignore")
            result = static.render("matplotlib")
        figure = getattr(result, "figure", result)
        savefig = getattr(figure, "savefig", None)
        if not callable(savefig):
            return None
        buf = io.BytesIO()
        savefig(buf, format="png", dpi=110)
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except Exception:  # noqa: BLE001 - closing is best-effort cleanup
            pass
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:  # noqa: BLE001 - a missing poster must never break the page
        return None
