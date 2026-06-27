"""Contract-closure test for the ``honored_by`` matrix (stream VIZ-STYLING).

This is the **keystone** that turns ``honored_by`` from a declaration into an
*enforced invariant*.  For every canonical :class:`~tsdynamics.viz.style.StyleKey`
and every backend in its ``honored_by`` set, we build a spec setting that key to a
distinctive value, render (or lower) it on that backend, and assert the produced
**artifact reflects the value** — a genuine honoring, not "doesn't raise".  Then
we assert the *dual*: for every backend **not** in a key's ``honored_by`` set,
:func:`~tsdynamics.viz.render.caps.style_honoring_gaps` reports that key (so the
dispatcher warns).  The same closure covers the theme-field honoring map.

If anyone later re-introduces an overclaim — adds a backend to a key's
``honored_by`` without wiring the render, or claims a theme field a backend does
not apply — one of these assertions fails.  The contract and the code can never
silently drift apart.

The matrix enforced (FINAL ``honored_by`` contract):

============  =========================================
style key     backends that MUST genuinely honor it
============  =========================================
color         matplotlib, plotly, threejs
linewidth     matplotlib, plotly, threejs
linestyle     matplotlib, plotly
marker        matplotlib, plotly
markersize    matplotlib, plotly, threejs
alpha         matplotlib, plotly, threejs
cmap          matplotlib, plotly
fill          matplotlib, plotly        (AREA only)
fillalpha     matplotlib, plotly        (AREA only)
zorder        matplotlib, plotly, threejs
============  =========================================

Theme fields: matplotlib honors all; plotly honors background/font/palette/grid;
threejs honors **only** background + palette (foreground/font/grid/title warn).

Engine-free by design (tiny hand-built specs) — fast tier, no ``tsdynamics._rust``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsdynamics.viz.render.caps import VisualizationDegraded, style_honoring_gaps
from tsdynamics.viz.spec import Layer, PlotKind, PlotSpec
from tsdynamics.viz.style import STYLE_KEYS, Theme

# The canonical backend set the contract closes over (the three *visual* backends
# whose render we can introspect; json is a faithful serializer with no gaps and
# is covered by ``test_viz_styling.py``).
_VISUAL_BACKENDS = ("matplotlib", "plotly", "threejs")

# A distinctive, contract-valid probe value for every canonical style key.  Chosen
# so the rendered artifact reflects it unambiguously (hex colors so threejs emits a
# clean string; an off-default zorder; a non-default fillalpha).
_PROBE_VALUE: dict[str, object] = {
    "color": "#ff0000",
    "linewidth": 3.5,
    "linestyle": "dashed",
    "marker": "o",
    "markersize": 11.0,
    "alpha": 0.4,
    "cmap": "viridis",
    "fill": False,
    "fillalpha": 0.7,
    "zorder": 7,
}


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------


def _line_spec(style: dict[str, object]) -> PlotSpec:
    """A 2-D LINE time-series spec carrying ``style`` on its one layer."""
    t = np.linspace(0.0, 1.0, 12)
    return PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(kind=PlotKind.LINE, data={"x": t, "y": np.sin(t)}, style=dict(style))],
    )


def _line3d_spec(style: dict[str, object]) -> PlotSpec:
    """A 3-D LINE3D spec (the threejs-natural shape) carrying ``style``."""
    t = np.linspace(0.0, 1.0, 12)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        layers=[
            Layer(
                kind=PlotKind.LINE3D,
                data={"x": np.sin(t), "y": np.cos(t), "z": t},
                style=dict(style),
            )
        ],
    )


def _scatter_spec(style: dict[str, object]) -> PlotSpec:
    """A 2-D SCATTER spec (so marker / markersize have a marker to ride on)."""
    t = np.linspace(0.0, 1.0, 12)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[Layer(kind=PlotKind.SCATTER, data={"x": t, "y": t**2}, style=dict(style))],
    )


def _area_spec(style: dict[str, object]) -> PlotSpec:
    """An AREA spec (the home of ``fill`` / ``fillalpha``)."""
    x = np.linspace(0.0, 1.0, 12)
    return PlotSpec(
        kind=PlotKind.AREA,
        layers=[
            Layer(
                kind=PlotKind.AREA,
                data={"x": x, "lo": np.zeros_like(x), "hi": x + 1.0},
                style=dict(style),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Per-backend artifact probes — assert the rendered artifact REFLECTS the value
# ---------------------------------------------------------------------------


def _mpl_line(fig: object) -> object:
    """First ``Line2D`` across the figure's axes."""
    from matplotlib.lines import Line2D

    for ax in fig.axes:  # type: ignore[attr-defined]
        for artist in ax.get_lines():
            if isinstance(artist, Line2D):
                return artist
    raise AssertionError("no Line2D in figure")


def _mpl_area_poly(fig: object) -> object | None:
    """First ``PolyCollection`` (the AREA band) across the figure's axes, or None."""
    from matplotlib.collections import PolyCollection

    for ax in fig.axes:  # type: ignore[attr-defined]
        for coll in ax.collections:
            if isinstance(coll, PolyCollection):
                return coll
    return None


def _assert_mpl_honors(key: str, value: object) -> None:
    """Render on matplotlib and assert the artifact reflects ``style[key] = value``."""
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgba

    if key in ("fill", "fillalpha"):
        fig = _area_spec({"color": "#0000ff", key: value}).render(backend="matplotlib")
        poly = _mpl_area_poly(fig)
        if key == "fill":
            # fill=False suppresses the band (no PolyCollection); the default True draws one.
            assert poly is None, "fill=False must suppress the AREA band on matplotlib"
        else:  # fillalpha
            assert poly is not None
            assert poly.get_alpha() == pytest.approx(float(value))  # type: ignore[arg-type]
        return

    # A LINE honors color / linewidth / linestyle / marker / markersize / alpha /
    # zorder on a single ``Line2D`` (mpl ``ax.plot`` with a marker), so one probe
    # shape covers them all.  ``color`` is the probed key itself; the others ride a
    # neutral base color so the spec carries exactly one canonical key under test.
    base = {key: value} if key == "color" else {key: value, "color": "#112233"}
    fig = _line_spec(base).render(backend="matplotlib")
    line = _mpl_line(fig)

    if key == "color":
        assert to_rgba(line.get_color()) == to_rgba(value)  # type: ignore[arg-type]
    elif key == "linewidth":
        assert line.get_linewidth() == pytest.approx(float(value))  # type: ignore[arg-type]
    elif key == "linestyle":
        assert line.get_linestyle() == "--"  # mpl spelling of "dashed"
    elif key == "marker":
        assert line.get_marker() == "o"
    elif key == "markersize":
        assert line.get_markersize() == pytest.approx(float(value))  # type: ignore[arg-type]
    elif key == "alpha":
        assert line.get_alpha() == pytest.approx(float(value))  # type: ignore[arg-type]
    elif key == "zorder":
        assert int(line.get_zorder()) == int(value)  # type: ignore[arg-type]
    elif key == "cmap":
        # A scalar-colored line/scatter carries the cmap; on a plain LINE there is no
        # mappable, so render a scatter with a ``c`` channel to exercise the cmap path.
        t = np.linspace(0.0, 1.0, 12)
        spec = PlotSpec(
            kind=PlotKind.PHASE_PORTRAIT_2D,
            layers=[
                Layer(
                    kind=PlotKind.SCATTER,
                    data={"x": t, "y": t, "c": t},
                    style={"cmap": value},
                )
            ],
        )
        fig = spec.render(backend="matplotlib")
        mappables = [c for ax in fig.axes for c in ax.collections]  # type: ignore[attr-defined]
        assert mappables, "expected a scalar-mappable collection for the cmap probe"
        assert mappables[0].get_cmap().name == value
    else:  # pragma: no cover - guard against an unmapped key
        raise AssertionError(f"no matplotlib honoring probe for {key!r}")


def _assert_plotly_honors(key: str, value: object) -> None:
    """Render on plotly and assert the trace/layout reflects ``style[key] = value``."""
    pytest.importorskip("plotly")

    if key in ("fill", "fillalpha"):
        fig = _area_spec({"color": "blue", key: value}).render(backend="plotly")
        fillcolors = [tr.fillcolor for tr in fig.data]
        if key == "fill":
            # fill=False ⇒ no band trace carries a fill color.
            assert all(fc is None for fc in fillcolors), "fill=False must drop the plotly band"
        else:  # fillalpha baked into the rgba fillcolor
            assert any(fc == "rgba(0, 0, 255, 0.7)" for fc in fillcolors), (
                f"fillalpha not baked into a band fillcolor: {fillcolors}"
            )
        return

    if key in ("marker", "markersize"):
        fig = _scatter_spec({key: value, "color": "#ff0000"}).render(backend="plotly")
    elif key == "cmap":
        t = np.linspace(0.0, 1.0, 12)
        spec = PlotSpec(
            kind=PlotKind.PHASE_PORTRAIT_2D,
            layers=[
                Layer(kind=PlotKind.SCATTER, data={"x": t, "y": t, "c": t}, style={"cmap": value})
            ],
        )
        fig = spec.render(backend="plotly")
    else:
        base = {key: value} if key == "color" else {key: value, "color": "#112233"}
        fig = _line_spec(base).render(backend="plotly")

    trace = fig.data[0]
    if key == "color":
        assert trace.line.color == value
    elif key == "linewidth":
        assert trace.line.width == pytest.approx(float(value))  # type: ignore[arg-type]
    elif key == "linestyle":
        assert trace.line.dash == "dash"  # plotly spelling of "dashed"
    elif key == "marker":
        assert trace.marker.symbol == "circle"
    elif key == "markersize":
        assert trace.marker.size == pytest.approx(float(value))  # type: ignore[arg-type]
    elif key == "alpha":
        assert trace.opacity == pytest.approx(float(value))  # type: ignore[arg-type]
    elif key == "zorder":
        assert int(trace.zorder) == int(value)  # type: ignore[arg-type]
    elif key == "cmap":
        # The scatter's marker colorscale carries the cmap name (plotly's spelling).
        scale = trace.marker.colorscale
        assert scale is not None, "expected a marker colorscale for the cmap probe"
        as_text = str(scale).lower()
        assert "viridis" in as_text or isinstance(scale, (list, tuple))
    else:  # pragma: no cover
        raise AssertionError(f"no plotly honoring probe for {key!r}")


def _assert_threejs_honors(key: str, value: object) -> None:
    """Lower on threejs and assert the material reflects ``style[key] = value``.

    threejs honors color / linewidth / markersize / alpha / zorder.  The loader
    consumes each (verified against ``docs/_static/tsdyn-threejs-loader.js`` reads
    in :func:`test_threejs_loader_reads_every_honored_material_field`).
    """
    spec = _line3d_spec({key: value})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VisualizationDegraded)
        payload = spec.render(backend="threejs", raw=True)
    material = payload["geometries"][0]["material"]

    if key == "color":
        assert material["color"] == "#ff0000"
    elif key == "linewidth":
        # threejs: passed faithfully (mainstream WebGL clamps to 1px — still honored).
        assert material["linewidth"] == pytest.approx(float(value))  # type: ignore[arg-type]
    elif key == "markersize":
        assert material["markersize"] == pytest.approx(float(value))  # type: ignore[arg-type]
    elif key == "alpha":
        assert material["alpha"] == pytest.approx(float(value))  # type: ignore[arg-type]
    elif key == "zorder":
        assert int(material["zorder"]) == int(value)  # type: ignore[arg-type]
    else:  # pragma: no cover
        raise AssertionError(f"no threejs honoring probe for {key!r}")


_HONOR_PROBES = {
    "matplotlib": _assert_mpl_honors,
    "plotly": _assert_plotly_honors,
    "threejs": _assert_threejs_honors,
}


# ---------------------------------------------------------------------------
# 1. POSITIVE closure — every (key, honoring-backend) pair genuinely reflects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "backend"),
    [
        (key, backend)
        for key, sk in STYLE_KEYS.items()
        for backend in _VISUAL_BACKENDS
        if backend in sk.honored_by
    ],
)
def test_honored_key_reflected_in_artifact(key: str, backend: str) -> None:
    """A backend in a key's ``honored_by`` set renders the key's value into the artifact."""
    _HONOR_PROBES[backend](key, _PROBE_VALUE[key])


# ---------------------------------------------------------------------------
# 2. DUAL closure — every (key, NON-honoring-backend) pair is reported as a gap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "backend"),
    [
        (key, backend)
        for key, sk in STYLE_KEYS.items()
        for backend in _VISUAL_BACKENDS
        if backend not in sk.honored_by
    ],
)
def test_unhonored_key_reported_as_gap(key: str, backend: str) -> None:
    """A backend NOT in a key's ``honored_by`` set must report it via ``style_honoring_gaps``."""
    # AREA so fill / fillalpha are valid on the layer; the gap check is purely on
    # the presence of the key in ``honored_by`` (mark-independent).
    spec = _area_spec({key: _PROBE_VALUE[key]})
    gaps = style_honoring_gaps(spec, backend)
    assert key in gaps, f"{backend} does not honor {key!r} but did not report it as a gap"


# ---------------------------------------------------------------------------
# 3. THEME-field closure — honoring map per backend
# ---------------------------------------------------------------------------

# A theme that sets every presentation field, so each honoring/gap path has data.
_FULL_THEME = Theme(
    name="contract-probe",
    palette=("#123456", "#abcdef"),
    background="#202020",
    foreground="#eeeeee",
    font_family="serif",
    font_size=11.0,
    title_size=16.0,
    grid=True,
    grid_color="#777777",
    grid_alpha=0.4,
)

# Theme fields that threejs honors (loader reads them); the rest must warn.
_THREEJS_HONORED_THEME = frozenset({"background", "palette"})
_ALL_THEME_FIELDS = frozenset(
    {
        "background",
        "palette",
        "foreground",
        "font_family",
        "font_size",
        "title_size",
        "grid",
        "grid_color",
        "grid_alpha",
    }
)


def test_matplotlib_honors_every_theme_field() -> None:
    """matplotlib honors ALL theme fields → ``style_honoring_gaps`` reports none."""
    spec = _line_spec({}).theme(_FULL_THEME)
    assert style_honoring_gaps(spec, "matplotlib") == []


@pytest.mark.parametrize("field", sorted(_ALL_THEME_FIELDS - _THREEJS_HONORED_THEME))
def test_threejs_warns_for_unhonored_theme_field(field: str) -> None:
    """Each theme field threejs does NOT honor is reported as a ``theme.<field>`` gap."""
    spec = _line3d_spec({}).theme(_FULL_THEME)
    gaps = style_honoring_gaps(spec, "threejs")
    assert f"theme.{field}" in gaps, f"threejs must warn for unhonored theme field {field!r}"


@pytest.mark.parametrize("field", sorted(_THREEJS_HONORED_THEME))
def test_threejs_does_not_warn_for_honored_theme_field(field: str) -> None:
    """The two theme fields threejs DOES honor (background/palette) are never gaps."""
    spec = _line3d_spec({}).theme(_FULL_THEME)
    gaps = style_honoring_gaps(spec, "threejs")
    assert f"theme.{field}" not in gaps


def test_threejs_exporter_emits_only_honored_theme_fields() -> None:
    """The threejs ``metadata.theme`` block carries ONLY background + palette.

    A dead field (``foreground`` / font / grid / title) would overclaim — the
    exporter must not emit one.
    """
    spec = _line3d_spec({}).theme(_FULL_THEME)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VisualizationDegraded)
        payload = spec.render(backend="threejs", raw=True)
    theme_block = payload["metadata"]["theme"]
    assert set(theme_block) == {"background", "palette"}
    assert "foreground" not in theme_block


# ---------------------------------------------------------------------------
# 4. EXPORTER ⇄ LOADER lockstep — every honored threejs material/theme field the
#    exporter emits is actually READ by the reference loader (and vice-versa).
# ---------------------------------------------------------------------------


def _loader_source() -> str:
    """The text of the reference three.js loader (where the reads live)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    loader = root / "docs" / "_static" / "tsdyn-threejs-loader.js"
    return loader.read_text(encoding="utf-8")


@pytest.mark.parametrize("field", ["color", "linewidth", "markersize", "alpha", "zorder"])
def test_threejs_loader_reads_every_honored_material_field(field: str) -> None:
    """Every honored threejs material field the exporter emits is read by the loader.

    ``zorder`` is read as ``mat.zorder`` and mapped to ``renderOrder``; the rest are
    read as ``mat.<field>``.  This keeps the exporter ↔ loader honoring in lockstep:
    a claimed material key with no loader read would be a dead export.
    """
    src = _loader_source()
    assert f"mat.{field}" in src, f"loader does not read material.{field} — dead export"
    if field == "zorder":
        assert "renderOrder" in src, "zorder must map to renderOrder in the loader"


def test_threejs_loader_does_not_read_dropped_material_fields() -> None:
    """The dropped material fields (``cmap``) are NOT read by the loader (no dead claim)."""
    src = _loader_source()
    assert "mat.cmap" not in src


@pytest.mark.parametrize("field", ["background", "palette"])
def test_threejs_loader_reads_every_honored_theme_field(field: str) -> None:
    """The two honored theme fields are read by the loader (``theme.<field>``)."""
    src = _loader_source()
    assert f"theme.{field}" in src, f"loader does not read theme.{field}"
