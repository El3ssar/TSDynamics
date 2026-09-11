"""House styles — ``ts.viz.themes`` (contract §6.10).

*A user defines a house style once and it applies everywhere.*  "Everywhere" is
the load-bearing word, and it is what this file measures: the **same** registered
theme is followed by matplotlib, by plotly and by the three.js export, and where a
backend genuinely cannot follow a field it says so in **one** warning naming
exactly what it dropped.

The four-verb registry shape (``register`` / ``names`` / ``find`` / ``get``, plus
``use``) is shared with ``ts.viz.transforms`` / ``primitives`` / ``renderers``;
``register`` takes **keywords**, which is the whole reason ``Theme`` is not an
exported name (C1: a type you need to construct to make a call is a signature bug).

``tests/conftest.py`` snapshots and restores the theme globals around every test,
so registering and activating here cannot leak.
"""

from __future__ import annotations

import warnings

import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidParameterError
from tsdynamics.viz.render.caps import VisualizationDegraded

#: One house style, declared once, used by every test below.
HOUSE = dict(
    palette=("#264653", "#e76f51", "#2a9d8f", "#e9c46a"),
    background="#fffdf7",
    foreground="#1d3557",
    font_family="DejaVu Sans",
    font_size=9.0,
    grid=True,
    grid_alpha=0.6,
    line_width=1.4,
)


@pytest.fixture
def house():
    """Register the house style and make it the session default."""
    theme = ts.viz.themes.register("lab", **HOUSE)
    ts.viz.themes.use("lab")
    return theme


@pytest.fixture
def traj():
    return ts.systems.Lorenz().run(final_time=3.0, dt=0.05, ic=[1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# the registry: register / names / get / use / find
# ---------------------------------------------------------------------------


def test_register_takes_keywords_and_returns_the_built_theme():
    """``themes.register(name, **fields)`` — no library type to construct (C1)."""
    theme = ts.viz.themes.register("lab", **HOUSE)
    assert theme.name == "lab"
    assert theme.palette == HOUSE["palette"]
    assert theme.background == "#fffdf7"
    assert ts.viz.themes.get("lab") is theme


def test_themes_lists_what_exists():
    """``names()`` enumerates the built-ins, and a new one joins the list."""
    builtins = ts.viz.themes.names()
    assert builtins == ["dark", "default", "minimal", "publication"]
    ts.viz.themes.register("lab", **HOUSE)
    assert ts.viz.themes.names() == ["dark", "default", "lab", "minimal", "publication"]
    assert {t.name for t in ts.viz.themes.find()} == set(ts.viz.themes.names())


def test_register_derives_from_a_built_in():
    """``register(name, base, **overrides)`` — the derived form §6.10 documents.

    (``themes.get("publication").replace(dpi=600)``, which `04` taught, does not
    run: ``Theme`` has no ``.replace``.)
    """
    paper = ts.viz.themes.register(
        "paper", ts.viz.themes.get("publication"), font_family="Charter", dpi=600
    )
    pub = ts.viz.themes.get("publication")
    assert paper.font_family == "Charter" and paper.dpi == 600
    assert paper.palette == pub.palette  # everything not overridden is inherited
    assert pub.font_family == "serif" and pub.dpi == 300  # ...and the base is untouched
    assert not hasattr(pub, "replace")


def test_use_activates_it_for_the_session_and_an_unknown_name_names_the_list():
    ts.viz.themes.register("lab", **HOUSE)
    assert ts.viz.themes.use("lab").name == "lab"
    assert ts.viz.themes.get().name == "lab"

    with pytest.raises(InvalidParameterError, match="registered themes are"):
        ts.viz.themes.use("nope")


def test_the_repr_says_which_one_is_active():
    """``ts.viz.themes`` printed is the answer to "what exists, and which am I on?"."""
    ts.viz.themes.register("lab", **HOUSE)
    ts.viz.themes.use("lab")
    text = repr(ts.viz.themes)
    assert "*lab" in text  # the active marker
    for name in ts.viz.themes.names():
        assert name in text


# ---------------------------------------------------------------------------
# THE HEADLINE: one style, three backends
# ---------------------------------------------------------------------------


def test_the_house_style_lands_on_matplotlib(house, traj):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    fig = ts.plot(traj, "time_series", components="x").render(backend="matplotlib")
    ax = fig.axes[0]
    line = ax.get_lines()[0]
    try:
        assert line.get_color() == HOUSE["palette"][0]  # the palette colours the curve
        assert line.get_linewidth() == HOUSE["line_width"]
        assert _rgb(ax.patch.get_facecolor()) == _rgb(HOUSE["background"])
        assert any(g.get_visible() for g in ax.get_xgridlines())  # grid=True
    finally:
        plt.close(fig)


def test_the_house_style_lands_on_plotly(house, traj):
    """plotly follows the palette through ``layout.colorway`` — its own idiom for it."""
    pytest.importorskip("plotly")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VisualizationDegraded)
        fig = ts.plot(traj, "time_series", components="x").render(backend="plotly")
    assert tuple(fig.layout.colorway) == HOUSE["palette"]
    assert fig.layout.paper_bgcolor == HOUSE["background"]
    assert fig.layout.plot_bgcolor == HOUSE["background"]
    assert fig.layout.font.family == HOUSE["font_family"]
    assert fig.layout.font.size == HOUSE["font_size"]
    assert fig.layout.xaxis.showgrid is True
    assert fig.data[0].line.width == HOUSE["line_width"]


def test_the_house_style_lands_on_the_threejs_export(house, traj):
    """The web export carries the same palette and stage colour into ``metadata.theme``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VisualizationDegraded)
        payload = ts.plot(traj, "phase_portrait").render(backend="threejs").payload
    theme_block = payload["metadata"]["theme"]
    assert theme_block["background"] == HOUSE["background"]
    assert tuple(theme_block["palette"]) == HOUSE["palette"]
    assert payload["geometries"][0]["material"]["color"] == HOUSE["palette"][0]


def test_the_same_colour_reaches_all_three(house, traj):
    """The one assertion the promise reduces to: one style, one colour, three backends."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("plotly")
    import matplotlib.pyplot as plt

    want = HOUSE["palette"][0]
    line_spec = ts.plot(traj, "time_series", components="x")

    fig = line_spec.render(backend="matplotlib")
    try:
        mpl_color = fig.axes[0].get_lines()[0].get_color()
    finally:
        plt.close(fig)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VisualizationDegraded)
        plotly_color = tuple(line_spec.render(backend="plotly").layout.colorway)[0]
        threejs_color = (
            ts.plot(traj, "phase_portrait")
            .render(backend="threejs")
            .payload["geometries"][0]["material"]["color"]
        )

    assert mpl_color == plotly_color == threejs_color == want


# ---------------------------------------------------------------------------
# where a backend cannot follow: ONE warning, naming what was dropped
# ---------------------------------------------------------------------------


def test_a_backend_that_cannot_follow_a_field_says_which_one_once(traj):
    """One consolidated ``VisualizationDegraded`` per render — never silence, never a wall."""
    pytest.importorskip("plotly")
    ts.viz.themes.register("printy", palette=("#264653",), figsize=(5.0, 3.5), dpi=600)
    ts.viz.themes.use("printy")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ts.plot(traj, "time_series", components="x").render(backend="plotly")
    degraded = [w for w in caught if issubclass(w.category, VisualizationDegraded)]

    assert len(degraded) == 1, [str(w.message) for w in degraded]
    text = str(degraded[0].message)
    assert "plotly" in text and "dpi" in text and "figsize" in text


def test_a_field_every_backend_honors_warns_nowhere(traj):
    """A house style inside the honored vocabulary renders silently on every backend."""
    pytest.importorskip("plotly")
    ts.viz.themes.register("quiet", palette=("#264653", "#e76f51"), background="#ffffff")
    ts.viz.themes.use("quiet")

    for backend in ("matplotlib", "plotly", "threejs"):
        subject = "phase_portrait" if backend == "threejs" else "time_series"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = ts.plot(traj, subject).render(backend=backend)
            _close(result)
        degraded = [w for w in caught if issubclass(w.category, VisualizationDegraded)]
        assert degraded == [], (backend, [str(w.message) for w in degraded])


def test_the_gaps_a_backend_declares_are_the_gaps_it_warns_about(house, traj):
    """``caps`` is the single source of truth; the warning is generated from it."""
    from tsdynamics.viz.render.caps import style_honoring_gaps

    spec = ts.plot(traj, "phase_portrait").theme("lab")
    assert style_honoring_gaps(spec, "matplotlib") == []  # the reference renderer
    gaps = style_honoring_gaps(spec, "threejs")
    assert "theme.foreground" in gaps  # declared dead in the three.js scene
    assert "background" not in gaps and "palette" not in gaps  # ...these it does honor


@pytest.mark.xfail(
    reason="contract §6.11 / the never-silence rule: caps.style_honoring_gaps gates the "
    "THEME-presentation block on `spec._theme is not None` (caps.py:432), so a house "
    "style installed with `themes.use(...)` — the documented 'every plot, this session' "
    "spelling — drops 7 fields on three.js with NO warning, while the identical theme "
    "set per-plot warns about all 7. The adjacent geometry block already reads "
    "`spec.resolved_theme`; one word. viz/render/caps.py is S7's file "
    "(RENDER-OTHER) — filed in needs_from_others.",
    strict=True,
)
def test_a_session_default_theme_is_checked_like_a_per_plot_one(traj):
    """The same house style must warn the same way however it was installed.

    Measured today::

        themes.use("lab")   -> style_honoring_gaps(spec, "threejs") == []
        spec.theme("lab")   -> [... 7 fields ...]

    A user who sets their house style once — the way §6.10 teaches — is the one who
    gets no warning at all.
    """
    from tsdynamics.viz.render.caps import style_honoring_gaps

    ts.viz.themes.register("lab", **HOUSE)

    ts.viz.themes.use("lab")
    as_default = style_honoring_gaps(ts.plot(traj, "phase_portrait"), "threejs")

    ts.viz.themes.use("default")
    per_plot = style_honoring_gaps(ts.plot(traj, "phase_portrait").theme("lab"), "threejs")

    assert as_default == per_plot != []


# ---------------------------------------------------------------------------
# per-plot theming does not leak
# ---------------------------------------------------------------------------


def test_one_plot_can_override_the_house_style_without_changing_it(house, traj):
    """``p.theme("lab", grid=False)`` is one plot only — the session default is untouched."""
    ts.viz.themes.use("default")
    p = ts.plot(traj, "time_series", components="x")
    assert p.theme("lab", grid=False) is p  # chainable, mutate-and-return-self
    assert p.resolved_theme.name == "lab"
    assert p.resolved_theme.grid is False
    assert p.resolved_theme.palette == HOUSE["palette"]

    assert ts.viz.themes.get().name == "default"
    assert ts.plot(traj, "time_series", components="x").resolved_theme.name == "default"
    assert ts.viz.themes.get("lab").grid is True  # the registered theme is not mutated


def test_palette_overrides_just_the_colours(house, traj):
    """``p.palette([...])`` keeps every other house field."""
    p = ts.plot(traj, "time_series").palette(["#111111", "#e63946"])
    assert p.resolved_theme.palette[:2] == ("#111111", "#e63946")
    assert p.resolved_theme.background == HOUSE["background"]
    assert p.resolved_theme.line_width == HOUSE["line_width"]


@pytest.mark.xfail(
    reason="contract §6.10 spells `p.palette('#111', '#e63946')` (varargs); the shipped "
    "signature is `palette(colors)` and the documented line raises TypeError. "
    "viz/spec.py + viz/_tweaks.py are another slot's files — filed in needs_from_others",
    strict=False,
)
def test_palette_accepts_varargs_as_the_contract_spells_it(house, traj):
    ts.plot(traj, "time_series").palette("#111111", "#e63946")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rgb(color):
    """Normalize a matplotlib colour (name / hex / RGBA tuple) to an ``(r, g, b)`` triple."""
    from matplotlib.colors import to_rgb

    return tuple(round(c, 6) for c in to_rgb(color))


def _close(result):
    """Release whatever a backend handed back (a Figure, an animation, a payload)."""
    figure = getattr(result, "_fig", None) or getattr(result, "figure", None)
    if figure is not None and hasattr(figure, "clf"):
        import matplotlib.pyplot as plt

        plt.close(figure)
