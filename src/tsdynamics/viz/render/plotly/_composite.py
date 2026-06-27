"""Native plotly rendering for ``COMPOSITE`` (multi-panel) figures (stream PLOTLY-COMPOSITE).

The composition layer (:func:`tsdynamics.viz.plot`) builds a
:data:`~tsdynamics.viz.spec.PlotKind.COMPOSITE` spec for ``layout="stack"`` /
``"row"`` / ``"grid"`` — a figure carrying child :attr:`~tsdynamics.viz.spec.PlotSpec.panels`
and a :class:`~tsdynamics.viz.spec.Layout`.  The matplotlib reference renderer
tiles these into a subplot grid; this module is the **plotly** equivalent, so
``ts.viz.plot(a, b, layout="stack").render(backend="plotly")`` (and
``.save("fig.html")``) yields *one* interactive multi-panel
:class:`plotly.graph_objects.Figure` instead of falling back to matplotlib.

How it works
------------
- :func:`_composite_grid` maps the :class:`~tsdynamics.viz.spec.Layout` ``mode``
  (``"stack"`` → one column, ``"row"`` → one row, ``"grid"`` → an explicit /
  near-square grid) to a ``(rows, cols)`` shape — the same arithmetic the
  matplotlib renderer uses, so both backends tile a composite identically.
- :func:`render_composite` builds a :func:`plotly.subplots.make_subplots` grid
  whose ``specs`` mark each panel's cell ``"scene"`` (a 3-D panel) or ``"xy"``
  (a 2-D panel) — so a composite **mixing** a time-series panel and a 3-D
  portrait renders with each panel on the correct subplot type.  Each panel's
  traces are built through the single-panel cores (:func:`._core.build_2d_traces`
  / :func:`._threed.build_3d_traces`) and added into its cell with
  ``add_trace(..., row, col)``; the panel's axes / scene layout is then applied
  to the cell's ``xaxisN`` / ``yaxisN`` / ``sceneN`` layout slot.
- **Per-panel colorbars** (e.g. two stacked spacetime images) are repositioned
  into each panel's own domain (:func:`_place_colorbars`) so they sit beside
  their panel rather than stacking at the figure's right edge and overlapping.

plotly is imported only **lazily** (inside the functions), never at
``import tsdynamics`` — matching the rest of the plotly backend's
no-plot-library-at-import contract.  HTML export rides on the same
:mod:`._html` seam the single-panel path uses.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from ...spec import PlotSpec
from . import _threed
from ._core import _build_2d_layout, _theme_layout, build_2d_traces

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ...spec import Layout

__all__ = ["render_composite"]


def _composite_grid(layout: Layout | None, n: int) -> tuple[int, int]:
    """Return the ``(rows, cols)`` subplot grid for a composite's :class:`Layout`.

    Mirrors the matplotlib renderer's grid arithmetic so the two backends tile a
    composite the same way: ``"row"`` → one row, ``"grid"`` → the explicit
    ``rows`` / ``cols`` (or a near-square fill), ``"stack"`` (default) → one
    column.
    """
    mode = getattr(layout, "mode", "stack") if layout is not None else "stack"
    if mode == "row":
        return 1, n
    if mode == "grid":
        rows = getattr(layout, "rows", None)
        cols = getattr(layout, "cols", None)
        if rows and cols:
            return int(rows), int(cols)
        c = int(np.ceil(np.sqrt(n)))
        r = int(np.ceil(n / c))
        return r, c
    return n, 1  # "stack" (default): one column


def _cell_indices(n: int, rows: int, cols: int) -> list[tuple[int, int]]:
    """Return the row-major ``(row, col)`` (1-based) cell for each of ``n`` panels."""
    return [(i // cols + 1, i % cols + 1) for i in range(n)]


def _panel_is_3d(panel: PlotSpec) -> bool:
    """Whether a panel needs a 3-D ``scene`` cell (vs a 2-D ``xy`` cell)."""
    return _threed.is_three_d(panel)


def _panel_has_colorbar(panel: PlotSpec) -> bool:
    """Whether a 2-D panel draws a colour-mapped image / colorbar of its own.

    Reads the panel's :class:`~tsdynamics.viz.spec.Colorbar`: a shown colorbar
    means the panel's image trace carries a scale that needs repositioning into
    the panel's domain (so stacked images do not collide at the figure edge).
    """
    return panel.colorbar is not None and panel.colorbar.show


def render_composite(
    spec: PlotSpec,
    *,
    html: bool = False,
    path: str | os.PathLike[str] | None = None,
    full_html: bool | None = None,
    include_plotlyjs: str | bool = "cdn",
    **_kw: Any,
) -> Any:
    """Render a ``COMPOSITE`` :class:`~tsdynamics.viz.spec.PlotSpec` to a plotly Figure.

    Tiles ``spec.panels`` into one :func:`plotly.subplots.make_subplots` grid per
    ``spec.layout`` — a 3-D panel gets a ``scene`` cell, a 2-D panel an ``xy``
    cell — drawing each panel through the single-panel cores and positioning
    per-panel colorbars within their own domain so they do not overlap.

    Parameters
    ----------
    spec : PlotSpec
        The composite spec (``is_composite`` is ``True``).
    html, path, full_html, include_plotlyjs
        HTML-export controls, forwarded to the shared :mod:`._html` seam exactly
        as the single-panel :func:`._core.render` does: ``html=True`` returns the
        embeddable fragment, ``path=`` writes a standalone file, else the live
        figure is returned.
    **_kw
        Forwarded but unused backend keywords (kept for a uniform signature).

    Returns
    -------
    plotly.graph_objects.Figure or str or pathlib.Path
        The interactive multi-panel figure by default; the HTML **string** when
        ``html`` is set; the written :class:`pathlib.Path` when ``path`` is given.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = list(spec.panels)
    layout = spec.layout
    if not panels:
        # A composite with no panels is degenerate; emit a valid (empty) figure.
        fig = go.Figure()
        if spec.title:
            fig.update_layout(title={"text": spec.title})
        return _finish(
            fig, html=html, path=path, full_html=full_html, include_plotlyjs=include_plotlyjs
        )

    rows, cols = _composite_grid(layout, len(panels))
    cells = _cell_indices(len(panels), rows, cols)
    specs = [[{"type": "xy"} for _ in range(cols)] for _ in range(rows)]
    for panel, (r, c) in zip(panels, cells, strict=True):
        if _panel_is_3d(panel):
            specs[r - 1][c - 1] = {"type": "scene"}

    share_x = bool(getattr(layout, "share_x", False)) if layout is not None else False
    share_y = bool(getattr(layout, "share_y", False)) if layout is not None else False

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        shared_xaxes=share_x,
        shared_yaxes=share_y,
    )

    # Track which 2-D cells carry a colorbar so we can space the scales out after
    # the traces are laid down (the domains are final only once make_subplots ran).
    colorbar_cells: list[tuple[int, int]] = []
    for panel, (r, c) in zip(panels, cells, strict=True):
        if _panel_is_3d(panel):
            for trace in _threed.build_3d_traces(panel):
                fig.add_trace(trace, row=r, col=c)
            _apply_scene(fig, panel, r, c)
        else:
            for trace in build_2d_traces(panel):
                fig.add_trace(trace, row=r, col=c)
            _apply_cell_axes(fig, panel, r, c)
            if _panel_has_colorbar(panel):
                colorbar_cells.append((r, c))

    _place_colorbars(fig, colorbar_cells)

    # Composites carry many per-panel legends/labels; a single shared legend would
    # be ambiguous, so leave plotly's per-trace legend entries off by default
    # (each panel's title carries its identity).
    fig.update_layout(showlegend=False)

    # Apply theme-level presentation to the composite figure (background, font).
    # The composite spec may carry its own theme; panels may override per-panel.
    composite_theme = spec.resolved_theme
    theme_layout = _theme_layout(composite_theme)
    if theme_layout:
        fig.update_layout(**theme_layout)

    if spec.title:
        fig.update_layout(title={"text": spec.title})

    return _finish(
        fig, html=html, path=path, full_html=full_html, include_plotlyjs=include_plotlyjs
    )


def _apply_cell_axes(fig: go.Figure, panel: PlotSpec, row: int, col: int) -> None:
    """Apply a 2-D panel's axes / aspect / title to its ``(row, col)`` subplot cell.

    Reuses the single-panel 2-D layout builder (:func:`._core._build_2d_layout`)
    and retargets its ``xaxis`` / ``yaxis`` dicts onto the cell's actual layout
    keys (``xaxisN`` / ``yaxisN``) that :func:`make_subplots` assigned, so each
    panel keeps its own labels / ranges / equal-aspect constraint and per-panel
    annotations.
    """
    layout = _build_2d_layout(panel)
    subplot = fig.get_subplot(row, col)
    x_name = subplot.xaxis.plotly_name  # e.g. "xaxis" / "xaxis3"
    y_name = subplot.yaxis.plotly_name
    x_dict = dict(layout.get("xaxis", {}))
    y_dict = dict(layout.get("yaxis", {}))
    # An "equal" aspect anchors y to the panel's *own* x-axis, not the figure's.
    if y_dict.get("scaleanchor") == "x":
        y_dict["scaleanchor"] = subplot.xaxis.anchor or _axis_ref(x_name)
    update: dict[str, Any] = {x_name: x_dict, y_name: y_dict}
    fig.update_layout(**update)
    # Per-panel annotations (vline / hline / text / span) are paper-referenced in
    # the single-panel builder; re-anchor them to this cell so they land in-panel.
    _add_cell_annotations(fig, layout, subplot, row, col)
    if panel.title:
        _add_panel_title(fig, panel.title, subplot)


def _axis_ref(plotly_name: str) -> str:
    """Map an axis layout key (``"xaxis3"``) to its trace-ref form (``"x3"``)."""
    if plotly_name.startswith("xaxis"):
        return "x" + plotly_name[len("xaxis") :]
    if plotly_name.startswith("yaxis"):
        return "y" + plotly_name[len("yaxis") :]
    return plotly_name


def _add_cell_annotations(
    fig: go.Figure, layout: dict[str, Any], subplot: Any, row: int, col: int
) -> None:
    """Re-anchor a panel's reference-line / span shapes + text to its cell.

    The single-panel builder emits ``shapes`` / ``annotations`` referenced to the
    figure ``paper`` (full-extent lines, corner text).  In a composite that would
    span every panel, so each shape / annotation is re-referenced to the cell's
    own axes (or its domain) before being added.
    """
    x_ref = _axis_ref(subplot.xaxis.plotly_name)
    y_ref = _axis_ref(subplot.yaxis.plotly_name)
    x_dom = subplot.xaxis.domain
    y_dom = subplot.yaxis.domain
    for shape in layout.get("shapes", []):
        s = dict(shape)
        # A full-extent reference line uses one data axis + one "paper" axis; swap
        # the data axis to the cell's axis and the paper extent to the cell domain.
        if s.get("yref") == "paper":
            s["yref"] = "paper"
            s["y0"], s["y1"] = y_dom[0], y_dom[1]
            s["xref"] = x_ref
        elif s.get("xref") == "paper":
            s["xref"] = "paper"
            s["x0"], s["x1"] = x_dom[0], x_dom[1]
            s["yref"] = y_ref
        fig.add_shape(s, row=row, col=col)
    for ann in layout.get("annotations", []):
        a = dict(ann)
        # Corner text uses one data coord + one "paper" coord; map "paper" → the
        # cell domain edge and the data coord → the cell axis.
        if a.get("yref") == "paper":
            a["yref"] = "paper"
            a["y"] = y_dom[1]
            a["xref"] = x_ref
        elif a.get("xref") == "paper":
            a["xref"] = "paper"
            a["x"] = x_dom[1]
            a["yref"] = y_ref
        else:
            a["xref"] = x_ref
            a["yref"] = y_ref
        fig.add_annotation(a)


def _add_panel_title(fig: go.Figure, title: str, subplot: Any) -> None:
    """Add a small panel title centred above a 2-D cell (a paper annotation)."""
    x_dom = subplot.xaxis.domain
    y_dom = subplot.yaxis.domain
    fig.add_annotation(
        x=0.5 * (x_dom[0] + x_dom[1]),
        y=y_dom[1],
        xref="paper",
        yref="paper",
        text=title,
        showarrow=False,
        yanchor="bottom",
        font={"size": 13},
    )


def _apply_scene(fig: go.Figure, panel: PlotSpec, row: int, col: int) -> None:
    """Apply a 3-D panel's ``scene`` (axes / camera / aspect) to its cell.

    Reuses the single-panel scene builder (:func:`._threed.scene_layout`) and
    attaches it to the cell's ``sceneN`` layout slot that :func:`make_subplots`
    assigned, so a 3-D portrait panel stays orbitable with its own axes / camera.
    """
    subplot = fig.get_subplot(row, col)  # a Scene layout object for a scene cell
    scene_name = subplot.plotly_name  # "scene" / "scene2" / ...
    scene = _threed.scene_layout(panel)
    fig.update_layout(**{scene_name: scene})
    if panel.title:
        domain = subplot.domain
        fig.add_annotation(
            x=0.5 * (domain.x[0] + domain.x[1]),
            y=domain.y[1],
            xref="paper",
            yref="paper",
            text=panel.title,
            showarrow=False,
            yanchor="bottom",
            font={"size": 13},
        )


def _place_colorbars(fig: go.Figure, cells: list[tuple[int, int]]) -> None:
    """Position each colour-mapped panel's colorbar inside its own subplot domain.

    plotly defaults every colorbar to the figure's right edge, so stacked /
    tiled images pile their scales on top of one another.  For each cell that
    carries a colorbar, the trace's colorbar is moved to just right of that
    cell and shrunk to the cell's vertical extent, so each image keeps its own
    legible, non-overlapping scale.
    """
    for r, c in cells:
        subplot = fig.get_subplot(r, c)
        x_dom = subplot.xaxis.domain
        y_dom = subplot.yaxis.domain
        x_pos = min(1.0, x_dom[1] + 0.02)
        length = max(0.05, y_dom[1] - y_dom[0])
        y_centre = 0.5 * (y_dom[0] + y_dom[1])
        x_ref = _axis_ref(subplot.xaxis.plotly_name)
        y_ref = _axis_ref(subplot.yaxis.plotly_name)
        cbar_pos = {
            "x": x_pos,
            "xanchor": "left",
            "y": y_centre,
            "yanchor": "middle",
            "len": length,
        }
        for trace in fig.data:
            # Match the colour-mapped trace(s) belonging to this cell (a Heatmap /
            # Surface / colour-scaled scatter sits on the cell's axes).
            if getattr(trace, "xaxis", None) != x_ref or getattr(trace, "yaxis", None) != y_ref:
                continue
            if "colorbar" in trace:
                # Heatmap / Surface carry a top-level colorbar.
                trace.update(colorbar=cbar_pos)
            elif "marker" in trace and "colorbar" in trace.marker:
                # A colour-scaled scatter / line keeps its scale on the marker.
                trace.update(marker={"colorbar": cbar_pos})


def _finish(
    fig: go.Figure,
    *,
    html: bool,
    path: str | os.PathLike[str] | None,
    full_html: bool | None,
    include_plotlyjs: str | bool,
) -> Any:
    """Return the figure, or export it to HTML, mirroring :func:`._core.render`.

    A composite serialises through the same :mod:`._html` seam as a single-panel
    spec, so ``.save("fig.html")`` writes one interactive multi-panel page and
    ``html=True`` returns the embeddable fragment.
    """
    if path is not None:
        from ._html import write_html

        return write_html(
            fig,
            path,
            full_html=True if full_html is None else full_html,
            include_plotlyjs=include_plotlyjs,
        )
    if html:
        from ._html import to_html

        return to_html(
            fig,
            full_html=False if full_html is None else full_html,
            include_plotlyjs=include_plotlyjs,
        )
    return fig
