"""Tests for the styling / theming overhaul (stream VIZ-STYLING).

New coverage for the public ``tsdynamics.viz`` styling surface (contract §6):

- :func:`~tsdynamics.viz.style.normalize_style` — alias resolution, value
  validation (a bad value raises ``ValueError``), and the single
  ``VisualizationDegraded`` warning emitted when an unknown key is dropped.
- the theme registry (:func:`themes` / :func:`set_theme` / :func:`get_theme` /
  :func:`register_theme`) and its single mutable global default.
- :meth:`~tsdynamics.viz.spec.PlotSpec.to_dict` / ``from_dict`` round-trip of a
  fully-themed + styled spec (deep equality on theme / axes / legend / colorbar /
  per-layer styles).
- the fluent tweak methods (``style`` / ``recolor`` / ``theme`` / ``palette`` /
  ``grid`` / ``font`` / ``background`` / ``size``) — mutate + return self + chain.
- **per-backend honoring** (structural, not "doesn't raise"): matplotlib draws
  the requested ``Line2D`` color / width / dash, sets the axes facecolor + grid;
  plotly carries ``line.dash`` / ``line.width`` / ``paper_bgcolor`` / ``font``;
  threejs serializes the per-layer material + ``metadata.theme``; json deep-equals
  the styled + themed spec.  Where a backend cannot honor a knob it emits exactly
  one ``VisualizationDegraded`` naming it (and a clean spec emits none).
- backend-flip regression: rendering the same spec twice with no explicit backend
  yields the identical (matplotlib) backend type.

Engine-free by design (tiny hand-built specs / a synthetic ``Trajectory``) — no
``tsdynamics._rust`` import, fast tier.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsdynamics.viz.render.caps import VisualizationDegraded
from tsdynamics.viz.spec import (
    Axis,
    Colorbar,
    Layer,
    Legend,
    PlotKind,
    PlotSpec,
)
from tsdynamics.viz.style import (
    DEFAULT_PALETTE,
    STYLE_KEYS,
    Theme,
    get_theme,
    normalize_style,
    register_theme,
    set_theme,
    themes,
)

# ---------------------------------------------------------------------------
# Global-state isolation
#
# The autouse ``_reset_global_theme`` fixture that snapshots + restores the
# single mutable global theme state (``THEMES`` / ``_ACTIVE``) lives in
# ``tests/conftest.py`` so it wraps **every** test in the suite (not just this
# module), eliminating the cross-module pollution landmine: a ``set_theme(...)``
# here can no longer leak into a registry-driven sweep elsewhere.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Builders (tiny engine-free data)
# ---------------------------------------------------------------------------


def _line_spec() -> PlotSpec:
    """A 2-D time-series spec with one labelled LINE layer (no explicit color)."""
    t = np.linspace(0.0, 1.0, 16)
    return PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(kind=PlotKind.LINE, data={"x": t, "y": np.sin(t)}, label="x(t)")],
        x=Axis(label="t"),
        y=Axis(label="x"),
    )


def _styled_themed_spec() -> PlotSpec:
    """A LINE spec carrying explicit color/linewidth/linestyle/marker/alpha + a theme.

    The theme sets background + font + grid, so every honoring/degradation path
    has something to act on.
    """
    spec = _line_spec()
    spec.style(color="#ff0000", linewidth=3.0, linestyle="dashed", marker="o", alpha=0.5)
    spec.theme(
        Theme(
            name="probe",
            palette=("#123456", "#abcdef"),
            background="#202020",
            foreground="#eeeeee",
            font_family="serif",
            font_size=11.0,
            grid=True,
            grid_color="#777777",
            grid_alpha=0.4,
            line_width=2.0,
        )
    )
    return spec


# ---------------------------------------------------------------------------
# normalize_style
# ---------------------------------------------------------------------------


def test_normalize_style_resolves_aliases():
    """Every documented alias canonicalizes to its real key + canonical value."""
    out = normalize_style(
        {
            "lw": 2.0,
            "c": "red",
            "s": 8.0,
            "ms": 9.0,
            "opacity": 0.25,
            "ls": "--",
            "marker": "o",
            "colormap": "viridis",
        }
    )
    assert out == {
        "linewidth": 2.0,
        "color": "red",
        # both ``s`` and ``ms`` alias markersize — last write wins on dict build
        "markersize": 9.0,
        "alpha": 0.25,
        "linestyle": "dashed",
        "marker": "circle",
        "cmap": "viridis",
    }


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ({"marker": "s"}, ("marker", "square")),
        ({"marker": "^"}, ("marker", "triangle")),
        ({"marker": "D"}, ("marker", "diamond")),
        ({"marker": "+"}, ("marker", "cross")),
        ({"marker": "*"}, ("marker", "star")),
        ({"ls": ":"}, ("linestyle", "dotted")),
        ({"ls": "-."}, ("linestyle", "dashdot")),
        ({"ls": "-"}, ("linestyle", "solid")),
        ({"colorscale": "magma"}, ("cmap", "magma")),
    ],
)
def test_normalize_style_canonical_forms(raw, canonical):
    """Marker / linestyle / cmap short spellings canonicalize to the named form."""
    key, value = canonical
    assert normalize_style(raw) == {key: value}


@pytest.mark.parametrize(
    "bad",
    [
        {"alpha": 5.0},  # out of [0, 1]
        {"alpha": -0.1},
        {"linewidth": -1.0},  # negative width
        {"markersize": -2.0},
        {"linestyle": "wiggly"},  # not a real linestyle
        {"marker": "heart"},  # not a real marker
    ],
)
def test_normalize_style_rejects_bad_value(bad):
    """A recognized key with an out-of-contract value raises ``ValueError``."""
    with pytest.raises(ValueError):
        normalize_style(bad)


def test_normalize_style_drops_unknown_key_with_one_warning():
    """An unknown key is dropped, emitting exactly ONE ``VisualizationDegraded``."""
    with pytest.warns(VisualizationDegraded) as record:
        out = normalize_style({"color": "blue", "bogus": 1, "alsobad": 2})
    assert out == {"color": "blue"}
    degraded = [w for w in record if issubclass(w.category, VisualizationDegraded)]
    assert len(degraded) == 1
    # both unknown keys named in the single message
    assert "bogus" in str(degraded[0].message)
    assert "alsobad" in str(degraded[0].message)


def test_normalize_style_warn_false_is_silent():
    """``warn=False`` (the renderer path) drops unknown keys without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", VisualizationDegraded)  # any warning would raise here
        out = normalize_style({"color": "blue", "bogus": 1}, warn=False)
    assert out == {"color": "blue"}


def test_style_keys_introspection_surface():
    """``STYLE_KEYS`` is the public name→StyleKey table covering the canonical keys."""
    for key in ("color", "linewidth", "linestyle", "marker", "markersize", "alpha", "cmap"):
        assert key in STYLE_KEYS
        assert STYLE_KEYS[key].name == key


# ---------------------------------------------------------------------------
# Theme registry + global default
# ---------------------------------------------------------------------------


def test_themes_lists_four_builtins():
    """``themes()`` lists the four built-in themes (default/dark/minimal/publication)."""
    names = themes()
    assert {"default", "dark", "minimal", "publication"} <= set(names)


def test_set_theme_by_name_activates():
    """``set_theme("dark")`` makes ``get_theme()`` return the dark theme."""
    set_theme("dark")
    active = get_theme()
    assert active.name == "dark"
    # ``get_theme("default")`` still reaches a specific theme by name
    assert get_theme("default").name == "default"


def test_set_theme_instance_registers_and_activates():
    """``set_theme(Theme(...))`` registers the theme under its name and activates it."""
    custom = Theme(name="midnight", background="#000010", palette=("#fff",))
    set_theme(custom)
    assert get_theme().name == "midnight"
    # now reachable by name
    assert get_theme("midnight").background == "#000010"
    assert "midnight" in themes()


def test_set_theme_default_resets():
    """``set_theme("default")`` returns the global default to the baseline theme."""
    set_theme("dark")
    assert get_theme().name == "dark"
    set_theme("default")
    assert get_theme().name == "default"
    assert get_theme().palette == DEFAULT_PALETTE


def test_register_theme_reachable_by_name_without_activating():
    """``register_theme`` adds a custom theme reachable by name, leaving default active."""
    set_theme("default")
    register_theme(Theme(name="sepia", background="#f4ecd8"))
    # registered + reachable
    assert "sepia" in themes()
    assert get_theme("sepia").background == "#f4ecd8"
    # but NOT activated (register != set)
    assert get_theme().name == "default"


def test_get_theme_unknown_raises():
    """``get_theme`` on an unregistered name raises ``KeyError``."""
    with pytest.raises(KeyError):
        get_theme("no-such-theme")


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip (themed + styled, deep equality)
# ---------------------------------------------------------------------------


def test_themed_styled_spec_roundtrips_deep():
    """A fully-themed + enriched spec survives ``to_dict`` → ``from_dict`` deeply."""
    t = np.linspace(0.0, 1.0, 12)
    spec = PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[
            Layer(
                kind=PlotKind.LINE,
                data={"x": t, "y": np.cos(t)},
                label="x(t)",
                style={"color": "#ff8800", "linewidth": 2.5, "linestyle": "dashed"},
            )
        ],
        x=Axis(label="t", grid=True, color="#333333", label_size=14.0, tick_size=9.0),
        y=Axis(label="x", grid=False, tick_rotation=45.0),
        legend=Legend(show=True, title="state", ncol=2, frame=False, font_size=8.0),
        colorbar=Colorbar(label="speed", label_size=10.0),
        title="probe",
    )
    spec.theme(
        Theme(
            name="probe",
            palette=("#101010", "#202020", "#303030"),
            background="#fafafa",
            foreground="#111111",
            font_family="serif",
            font_size=12.0,
            title_size=16.0,
            grid=True,
            grid_color="#999999",
            grid_alpha=0.3,
            line_width=1.75,
            marker_size=7.0,
        )
    )

    rebuilt = PlotSpec.from_dict(spec.to_dict())

    # theme (deep)
    assert rebuilt._theme is not None
    assert rebuilt._theme.to_dict() == spec._theme.to_dict()
    assert rebuilt.resolved_theme.palette == ("#101010", "#202020", "#303030")

    # enriched axes / legend / colorbar (deep)
    assert rebuilt.x.to_dict() == spec.x.to_dict()
    assert rebuilt.y.to_dict() == spec.y.to_dict()
    assert rebuilt.legend.to_dict() == spec.legend.to_dict()
    assert rebuilt.colorbar.to_dict() == spec.colorbar.to_dict()

    # per-layer style (deep; canonical keys preserved)
    assert rebuilt.layers[0].style == spec.layers[0].style


def test_theme_wire_key_is_theme():
    """The serialized wire key for the theme is ``"theme"`` (not ``_theme``)."""
    spec = _line_spec().theme("dark")
    d = spec.to_dict()
    assert "theme" in d
    assert d["theme"]["name"] == "dark"


def test_no_theme_serializes_as_none():
    """A spec that set no theme serializes ``theme=None`` and round-trips to ``None``."""
    spec = _line_spec()
    assert spec._theme is None
    d = spec.to_dict()
    assert d["theme"] is None
    assert PlotSpec.from_dict(d)._theme is None


# ---------------------------------------------------------------------------
# Fluent tweak methods: mutate + return self + chain
# ---------------------------------------------------------------------------


def test_style_mutates_returns_self_canonicalizing():
    """``.style`` returns self and STORES canonical keys (lw→linewidth, marker o→circle)."""
    spec = _line_spec()
    returned = spec.style(lw=4.0, marker="o")
    assert returned is spec
    assert spec.layers[0].style["linewidth"] == 4.0
    assert spec.layers[0].style["marker"] == "circle"


def test_recolor_mutates_returns_self():
    """``.recolor`` assigns explicit per-layer colors and returns self."""
    spec = _line_spec()
    returned = spec.recolor("teal")
    assert returned is spec
    assert spec.layers[0].style["color"] == "teal"


def test_theme_palette_grid_font_background_size_chain():
    """Every figure-level tweak mutates + returns self, composing in one chain."""
    spec = _line_spec()
    returned = (
        spec.theme("dark")
        .palette(["#111", "#222"])
        .grid(True, color="#555", alpha=0.6)
        .font(family="monospace", size=13.0)
        .background("#0a0a0a")
        .size(width=6.0, height=4.0, dpi=120.0)
    )
    assert returned is spec
    th = spec.resolved_theme
    # palette override
    assert th.palette == ("#111", "#222")
    # grid override (theme grid color/alpha + per-axis grid flag)
    assert th.grid_color == "#555"
    assert th.grid_alpha == pytest.approx(0.6)
    assert spec.x.grid is True
    assert spec.y.grid is True
    # font override
    assert th.font_family == "monospace"
    assert th.font_size == pytest.approx(13.0)
    # background override (background() wins over the theme's dark bg)
    assert th.background == "#0a0a0a"
    # size → meta (NOT a theme field)
    assert spec.meta["figsize"] == (6.0, 4.0)
    assert spec.meta["dpi"] == pytest.approx(120.0)


def test_theme_overrides_kwargs_merge_onto_base():
    """``theme(name, **overrides)`` merges overrides onto the named base theme."""
    spec = _line_spec().theme("dark", font_family="serif")
    th = spec.resolved_theme
    assert th.name == "dark"  # base preserved
    assert th.background == "#11131a"  # dark's background kept
    assert th.font_family == "serif"  # override applied


def test_resolved_theme_falls_back_to_global_default():
    """``resolved_theme`` is never None — it falls back to the active global default."""
    set_theme("publication")
    spec = _line_spec()  # no own theme
    assert spec._theme is None
    assert spec.resolved_theme.name == "publication"


# ---------------------------------------------------------------------------
# Per-backend honoring — matplotlib
# ---------------------------------------------------------------------------


def _first_line2d(fig):
    """Return the first ``Line2D`` artist found across the figure's axes."""
    from matplotlib.lines import Line2D

    for ax in fig.axes:
        for artist in ax.get_lines():
            if isinstance(artist, Line2D):
                return artist
    return None


def test_matplotlib_honors_style_and_theme():
    """mpl draws the requested color/width/dash and applies background + grid."""
    pytest.importorskip("matplotlib")
    spec = _styled_themed_spec()
    fig = spec.render(backend="matplotlib")

    line = _first_line2d(fig)
    assert line is not None
    # color: matplotlib normalizes "#ff0000" → an (r,g,b,a) tuple
    from matplotlib.colors import to_rgba

    assert to_rgba(line.get_color()) == to_rgba("#ff0000")
    assert line.get_linewidth() == pytest.approx(3.0)
    assert line.get_linestyle() == "--"  # mpl spelling of "dashed"

    ax = fig.axes[0]
    # background (theme.background) on the axes facecolor
    assert to_rgba(ax.get_facecolor()) == to_rgba("#202020")
    # grid visible (theme.grid True / per-axis)
    assert any(gl.get_visible() for gl in ax.get_xgridlines() + ax.get_ygridlines())


def test_matplotlib_uncolored_layer_gets_palette_color():
    """An uncolored layer is colored from the theme palette (mpl color cycle)."""
    pytest.importorskip("matplotlib")
    spec = _line_spec().theme(Theme(name="p", palette=("#0000ff",)))
    fig = spec.render(backend="matplotlib")
    line = _first_line2d(fig)
    from matplotlib.colors import to_rgba

    assert to_rgba(line.get_color()) == to_rgba("#0000ff")


# ---------------------------------------------------------------------------
# Per-backend honoring — plotly
# ---------------------------------------------------------------------------


def test_plotly_honors_style_and_theme():
    """plotly carries line.dash / line.width / marker, paper_bgcolor + font.family."""
    pytest.importorskip("plotly")
    spec = _styled_themed_spec()
    fig = spec.render(backend="plotly")

    trace = fig.data[0]
    assert trace.line.dash == "dash"  # plotly spelling of "dashed"
    assert trace.line.width == pytest.approx(3.0)
    assert trace.line.color == "#ff0000"
    assert trace.opacity == pytest.approx(0.5)

    layout = fig.layout
    assert layout.paper_bgcolor == "#202020"
    assert layout.font.family == "serif"


# ---------------------------------------------------------------------------
# Per-backend honoring — threejs
# ---------------------------------------------------------------------------


def test_threejs_material_and_theme_metadata():
    """threejs serializes per-layer material (color/linewidth/alpha) + metadata.theme."""
    spec = _styled_themed_spec()
    # threejs declines linestyle / theme.font / theme.grid for this spec → expect
    # exactly one consolidated VisualizationDegraded naming those, so wrap render.
    with pytest.warns(VisualizationDegraded):
        payload = spec.render(backend="threejs", raw=True)

    geom = payload["geometries"][0]
    material = geom["material"]
    assert material["color"] == "#ff0000"
    assert material["linewidth"] == pytest.approx(3.0)
    assert material["alpha"] == pytest.approx(0.5)

    theme_block = payload["metadata"]["theme"]
    assert theme_block["background"] == "#202020"
    assert theme_block["palette"] == ["#123456", "#abcdef"]
    # ``foreground`` is NOT honored by the three.js loader, so the exporter no
    # longer emits it (a dead field would overclaim — contract: threejs theme set
    # is ``{background, palette}`` only).
    assert "foreground" not in theme_block


def test_threejs_uncolored_layer_gets_palette_color():
    """An uncolored line bakes the theme palette color into its material."""
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[Layer(kind=PlotKind.LINE, data={"x": [0.0, 1.0], "y": [0.0, 1.0]})],
    ).theme(Theme(name="p", palette=("#0a0b0c",)))
    payload = spec.render(backend="threejs", raw=True)
    assert payload["geometries"][0]["material"]["color"] == "#0a0b0c"


# ---------------------------------------------------------------------------
# Per-backend honoring — json (serializer: deep round-trip, no honoring gaps)
# ---------------------------------------------------------------------------


def test_json_backend_deep_roundtrips_styled_themed_spec():
    """The json exporter serializes the styled + themed spec; from_json deep-equals it."""
    from tsdynamics.viz.export import from_json

    spec = _styled_themed_spec()
    # json is a faithful serializer → NO honoring gaps → no warning at all.
    with warnings.catch_warnings():
        warnings.simplefilter("error", VisualizationDegraded)
        text = spec.render(backend="json", raw=True)
    rebuilt = from_json(text)

    assert rebuilt._theme.to_dict() == spec._theme.to_dict()
    assert rebuilt.layers[0].style == spec.layers[0].style
    assert rebuilt.x.to_dict() == spec.x.to_dict()


# ---------------------------------------------------------------------------
# Honest degradation: exactly one VisualizationDegraded, named; clean spec = none
# ---------------------------------------------------------------------------


def test_threejs_warns_once_for_linestyle_gap():
    """threejs cannot honor ``linestyle`` → exactly one VisualizationDegraded naming it."""
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[
            Layer(
                kind=PlotKind.LINE,
                data={"x": [0.0, 1.0], "y": [0.0, 1.0]},
                style={"linestyle": "dashed"},
            )
        ],
    )
    with pytest.warns(VisualizationDegraded) as record:
        spec.render(backend="threejs", raw=True)
    degraded = [w for w in record if issubclass(w.category, VisualizationDegraded)]
    assert len(degraded) == 1
    assert "linestyle" in str(degraded[0].message)


def test_threejs_warns_once_for_theme_font_and_grid():
    """A theme.font + theme.grid the threejs loader cannot apply → one named warning."""
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[Layer(kind=PlotKind.LINE, data={"x": [0.0, 1.0], "y": [0.0, 1.0]})],
    ).theme(Theme(name="p", font_family="serif", grid=True))
    with pytest.warns(VisualizationDegraded) as record:
        spec.render(backend="threejs", raw=True)
    degraded = [w for w in record if issubclass(w.category, VisualizationDegraded)]
    assert len(degraded) == 1
    msg = str(degraded[0].message)
    assert "theme.font_family" in msg
    assert "theme.grid" in msg


def test_plotly_warns_once_for_camera_spin():
    """plotly cannot honor an animated camera spin → exactly one named warning."""
    pytest.importorskip("plotly")
    t = np.linspace(0.0, 1.0, 8)
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[Layer(kind=PlotKind.LINE, data={"x": np.sin(t), "y": np.cos(t)})],
    ).camera(spin=2.0)  # spin turns animation on + sets an unhonorable knob
    with pytest.warns(VisualizationDegraded) as record:
        spec.render(backend="plotly")
    degraded = [w for w in record if issubclass(w.category, VisualizationDegraded)]
    assert len(degraded) == 1
    assert "spin" in str(degraded[0].message)


def test_clean_spec_emits_no_degradation_warning_mpl():
    """A spec carrying only knobs mpl honors emits NO VisualizationDegraded."""
    pytest.importorskip("matplotlib")
    spec = _line_spec().style(color="#00ff00", linewidth=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", VisualizationDegraded)  # any VisualizationDegraded would raise
        spec.render(backend="matplotlib")


def test_clean_spec_emits_no_degradation_warning_threejs():
    """A threejs spec using only honored keys (color/linewidth/alpha) emits none."""
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[
            Layer(
                kind=PlotKind.LINE,
                data={"x": [0.0, 1.0], "y": [0.0, 1.0]},
                style={"color": "#abcdef", "linewidth": 2.0, "alpha": 0.8},
            )
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", VisualizationDegraded)
        spec.render(backend="threejs", raw=True)


# ---------------------------------------------------------------------------
# Backend-flip regression
# ---------------------------------------------------------------------------


def test_default_backend_is_stable_matplotlib():
    """Rendering the same spec twice with no backend yields the identical (mpl) type."""
    pytest.importorskip("matplotlib")
    spec = _line_spec()
    fig1 = spec.render()
    fig2 = spec.render()
    assert type(fig1) is type(fig2)
    # matplotlib is the deterministic default → a real Figure, not a PlotSpec
    from matplotlib.figure import Figure

    assert isinstance(fig1, Figure)
    assert not isinstance(fig1, PlotSpec)
