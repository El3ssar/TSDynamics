"""Self-contained HTML export for the plotly backend (stream PLOTLY-HTML).

A plotly figure serialises to a self-contained, **kernel-free** HTML fragment:
an interactive ``<div>`` (pan / zoom / hover) plus the JavaScript that wires it
up, with the plotly bundle pulled from a CDN.  This module is the seam that turns
a :class:`~tsdynamics.viz.spec.PlotSpec` (or an already-built plotly figure) into
that HTML string, so a result can be embedded straight into an mkdocs page or a
web frontend without a running Python kernel.

It uses plotly's :meth:`plotly.graph_objects.Figure.to_html` /
:func:`plotly.io.to_html` only — no ``kaleido`` (that is the static-image
exporter's concern, not this live-fragment one).  Two defaults make the output a
small, embeddable fragment rather than a standalone page:

- ``include_plotlyjs="cdn"`` — reference the plotly bundle from a CDN
  (``<script src="https://cdn.plot.ly/...">``) instead of inlining ~3 MB of
  JavaScript, so the fragment is tiny and shareable;
- ``full_html=False`` — emit the bare ``<div>`` + wiring ``<script>`` (no
  ``<html>`` / ``<head>`` / ``<body>`` wrapper), so it drops into an existing
  document.

Importing this module imports plotly only **lazily** (in
:func:`to_html`), never at ``import tsdynamics`` — matching the rest of the
plotly backend's no-plot-library-at-import contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ...spec import PlotSpec

__all__ = ["to_html", "write_html"]


def _as_figure(figure_or_spec: Any) -> go.Figure:
    """Coerce a :class:`PlotSpec` (or a plotly figure) to a plotly ``Figure``.

    A :class:`plotly.graph_objects.Figure` is returned unchanged; anything else
    is treated as a :class:`~tsdynamics.viz.spec.PlotSpec` and rendered through
    the plotly drawing core (:func:`tsdynamics.viz.render.plotly._core.render`).
    """
    import plotly.graph_objects as go

    if isinstance(figure_or_spec, go.Figure):
        return figure_or_spec
    from ._core import render as _render_spec

    return _render_spec(figure_or_spec)


def to_html(
    figure_or_spec: PlotSpec | go.Figure,
    *,
    full_html: bool = False,
    include_plotlyjs: str | bool = "cdn",
) -> str:
    """Export a spec / plotly figure to a self-contained interactive HTML string.

    The returned text embeds an interactive plotly ``<div>`` (pan / zoom / hover)
    plus the JavaScript that wires it up — it needs **no** Python kernel to
    render in a browser.

    Parameters
    ----------
    figure_or_spec : PlotSpec or plotly.graph_objects.Figure
        The spec to draw (rendered through the plotly core) or an already-built
        plotly figure to serialise directly.
    full_html : bool, optional
        When ``True`` emit a standalone document (``<html>`` / ``<head>`` /
        ``<body>``); when ``False`` (the default) emit only the embeddable
        ``<div>`` + wiring ``<script>`` fragment for dropping into an existing
        page (e.g. an mkdocs document).
    include_plotlyjs : str or bool, optional
        How to provide the plotly JavaScript bundle.  ``"cdn"`` (the default)
        references it from a CDN ``<script>`` tag so the fragment stays small;
        forwarded verbatim to plotly (``True`` inlines the full bundle,
        ``"directory"`` / a URL are also accepted).

    Returns
    -------
    str
        The HTML text — an embeddable fragment by default, a full document when
        ``full_html`` is set.

    Notes
    -----
    Uses :meth:`plotly.graph_objects.Figure.to_html` only; no ``kaleido``
    static-image rendering is involved.
    """
    figure = _as_figure(figure_or_spec)
    html = figure.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs)
    return str(html)


def write_html(
    figure_or_spec: PlotSpec | go.Figure,
    path: str | os.PathLike[str],
    *,
    full_html: bool = True,
    include_plotlyjs: str | bool = "cdn",
) -> Path:
    """Write the self-contained interactive HTML for a spec / figure to ``path``.

    A convenience around :func:`to_html` that writes the result (UTF-8) and
    returns the path.  Writing a file defaults to ``full_html=True`` so the
    artifact opens standalone in a browser; pass ``full_html=False`` to write a
    bare embeddable fragment instead.

    Parameters
    ----------
    figure_or_spec : PlotSpec or plotly.graph_objects.Figure
        The spec to draw or an already-built plotly figure to serialise.
    path : str or os.PathLike
        Destination file path.
    full_html : bool, optional
        Whether to write a standalone document (default ``True``) or a bare
        fragment.
    include_plotlyjs : str or bool, optional
        How to provide the plotly bundle; ``"cdn"`` by default.  See
        :func:`to_html`.

    Returns
    -------
    pathlib.Path
        The path written to.
    """
    text = to_html(figure_or_spec, full_html=full_html, include_plotlyjs=include_plotlyjs)
    out = Path(path)
    out.write_text(text, encoding="utf-8")
    return out
