"""Conformance gate for the matplotlib reference renderer (stream VIZ-RENDER-GATE).

The matplotlib backend is the **conformance oracle**: it must draw *every* mark
and *every* semantic :class:`~tsdynamics.viz.spec.PlotKind` in the frozen
vocabulary on the Agg canvas without error.  This gate freezes that:

1. **Every layer mark renders** — a minimal spec carrying each
   :meth:`PlotKind.layer_marks` mark renders to a :class:`matplotlib.figure.Figure`.
2. **Every semantic kind renders** — a minimal spec of each
   :meth:`PlotKind.semantic_kinds` kind (2-D, 3-D, image, bar, …) renders.
3. **Registry conformance** — every registered analysis whose result carries a
   ``to_plot_spec`` produces a spec whose semantic kind and layer marks are real
   :class:`~tsdynamics.viz.spec.PlotKind` members that round-trip through
   ``to_dict`` / ``from_dict`` (engine-free; the synthetic builders live in
   :mod:`tests.test_viz_fake_renderer`).
4. **No-plot-import guard** — a fresh ``import tsdynamics`` pulls in **no**
   matplotlib (registration is lazy, only on first render).
5. **Negative control** — a deliberately broken renderer / spec fails, so the
   gate is not a tautology.

Engine-free.  Skips the render assertions where matplotlib is absent.
"""

from __future__ import annotations

import importlib
import io
import subprocess
import sys

import numpy as np
import pytest

from tsdynamics import registry
from tsdynamics.viz.spec import Axis, Colorbar, Layer, PlotKind, PlotSpec

pytest.importorskip("matplotlib")

from tsdynamics.viz.render import register_builtin_renderers  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _mpl_backend():
    """Ensure the matplotlib backend is registered for the whole module."""
    register_builtin_renderers()
    if "matplotlib" not in registry.renderers:  # pragma: no cover - matplotlib present here
        pytest.skip("matplotlib backend did not register")
    yield


# ---------------------------------------------------------------------------
# Minimal renderable specs for every mark / kind
# ---------------------------------------------------------------------------

_X = np.linspace(0.0, 1.0, 6)
_GRID = np.add.outer(_X, _X)


def _layer_for_mark(mark: PlotKind) -> Layer:
    """A minimal valid layer carrying the channels ``mark`` consumes."""
    if mark in (PlotKind.LINE3D,):
        return Layer(mark, {"x": _X, "y": _X[::-1], "z": _X})
    if mark == PlotKind.SURFACE3D:
        return Layer(mark, {"x": _X, "y": _X, "z": _GRID})
    if mark == PlotKind.IMAGE:
        return Layer(mark, {"x": _X, "y": _X, "c": _GRID})
    if mark == PlotKind.QUIVER:
        g = np.zeros((3, 3))
        return Layer(mark, {"x": g, "y": g, "u": g + 1.0, "v": g + 1.0})
    if mark == PlotKind.HISTOGRAM:
        return Layer(mark, {"x": _X})
    if mark == PlotKind.AREA:
        return Layer(mark, {"x": _X, "lo": _X - 0.1, "hi": _X + 0.1})
    if mark == PlotKind.ERRORBAR:
        return Layer(mark, {"x": _X, "y": _X, "err": _X * 0.1})
    if mark == PlotKind.BAR:
        return Layer(mark, {"x": _X, "y": _X})
    # LINE / SCATTER / MARKERS
    return Layer(mark, {"x": _X, "y": _X[::-1]})


_IMAGE_KINDS = frozenset(
    {
        PlotKind.IMAGE,
        PlotKind.BASINS_IMAGE,
        PlotKind.SPACETIME,
    }
)


def _spec_for_kind(kind: PlotKind) -> PlotSpec:
    """A minimal renderable spec for a *semantic* kind (2-D / 3-D / image)."""
    if kind in (PlotKind.PHASE_PORTRAIT_3D,):
        return PlotSpec(
            kind=kind,
            ndim=3,
            x=Axis(),
            y=Axis(),
            z=Axis(),
            layers=[_layer_for_mark(PlotKind.LINE3D)],
        )
    if kind in _IMAGE_KINDS:
        return PlotSpec(kind=kind, layers=[_layer_for_mark(PlotKind.IMAGE)], colorbar=Colorbar())
    if kind == PlotKind.RECURRENCE_PLOT:
        # Sparse recurrence plot — a scatter, never a dense image (anti-OOM).
        return PlotSpec(kind=kind, aspect="equal", layers=[_layer_for_mark(PlotKind.SCATTER)])
    if kind in (PlotKind.CATEGORICAL_BAR, PlotKind.LYAPUNOV_SPECTRUM):
        return PlotSpec(kind=kind, layers=[_layer_for_mark(PlotKind.BAR)])
    if kind == PlotKind.COMPOSITE:
        # A composite is a multi-panel figure: it tiles ``panels``, not ``layers``.
        from tsdynamics.viz.spec import Layout

        panel = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[_layer_for_mark(PlotKind.LINE)])
        return PlotSpec(kind=kind, panels=[panel, panel], layout=Layout(mode="stack"))
    return PlotSpec(kind=kind, layers=[_layer_for_mark(PlotKind.LINE)])


# ---------------------------------------------------------------------------
# 1 — every layer mark renders on Agg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mark", sorted(PlotKind.layer_marks(), key=lambda k: k.value), ids=str)
def test_every_mark_renders(mark: PlotKind):
    from matplotlib.figure import Figure

    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D
        if mark in (PlotKind.LINE3D, PlotKind.SURFACE3D)
        else PlotKind.TIME_SERIES,
        ndim=3 if mark in (PlotKind.LINE3D, PlotKind.SURFACE3D) else 2,
        z=Axis() if mark in (PlotKind.LINE3D, PlotKind.SURFACE3D) else None,
        layers=[_layer_for_mark(mark)],
    )
    fig = spec.render("matplotlib")
    assert isinstance(fig, Figure), mark
    assert fig.axes, f"{mark} produced no axes"


# ---------------------------------------------------------------------------
# 2 — every semantic kind renders on Agg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(PlotKind.semantic_kinds(), key=lambda k: k.value), ids=str)
def test_every_semantic_kind_renders(kind: PlotKind):
    from matplotlib.figure import Figure

    fig = _spec_for_kind(kind).render("matplotlib")
    assert isinstance(fig, Figure), kind
    assert fig.axes, f"{kind} produced no axes"


# ---------------------------------------------------------------------------
# 3 — registry conformance: every result's spec is coercible-kind + round-trips
# ---------------------------------------------------------------------------


def _result_builders() -> dict[type, object]:
    """The synthetic result builders from the fake-renderer gate (engine-free)."""
    module = importlib.import_module("test_viz_fake_renderer")
    return dict(module._BUILDERS)  # type: ignore[attr-defined]


def test_every_result_spec_is_coercible_and_round_trips():
    """Every synthetic result either yields a valid PlotSpec that round-trips, or raises."""
    from tsdynamics.analysis._result import VisualizationNotInstalled

    builders = _result_builders()
    assert builders, "the fake-renderer gate's builders must be importable"
    for cls, build in builders.items():
        result = build()
        try:
            spec = result.to_plot_spec()
        except VisualizationNotInstalled:
            continue  # the documented "nothing to draw" path
        assert isinstance(spec, PlotSpec), cls.__name__
        assert isinstance(spec.kind, PlotKind), cls.__name__
        for layer in spec.layers:
            assert isinstance(layer.kind, PlotKind), cls.__name__
        rebuilt = PlotSpec.from_dict(spec.to_dict())
        assert rebuilt.kind == spec.kind, cls.__name__
        assert len(rebuilt.layers) == len(spec.layers), cls.__name__


def test_every_registered_analysis_result_renders_or_is_documented():
    """Every synthetic result's spec renders on matplotlib (or documents no spec)."""
    from matplotlib.figure import Figure

    from tsdynamics.analysis._result import VisualizationNotInstalled

    for cls, build in _result_builders().items():
        result = build()
        try:
            spec = result.to_plot_spec()
        except VisualizationNotInstalled:
            continue
        fig = spec.render("matplotlib")
        assert isinstance(fig, Figure), cls.__name__


# ---------------------------------------------------------------------------
# 4 — no-plot-import guard
# ---------------------------------------------------------------------------


def test_import_tsdynamics_pulls_no_matplotlib():
    """A fresh ``import tsdynamics`` must not import matplotlib (lazy registration)."""
    code = (
        "import sys\n"
        "import tsdynamics\n"
        "import tsdynamics.viz\n"
        "for banned in ('matplotlib', 'matplotlib.pyplot', 'plotly'):\n"
        "    assert banned not in sys.modules, banned\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# 5 — negative control (the gate is not a tautology)
# ---------------------------------------------------------------------------


def test_negative_control_broken_layer_is_caught():
    """A spec carrying a non-PlotKind layer kind cannot be constructed/validated.

    Proves the kind assertions above have teeth: a stray string is rejected at
    Layer construction, so a broken spec can never slip the conformance check.
    """
    with pytest.raises(ValueError):
        Layer("definitely_not_a_mark", {"x": _X, "y": _X})


def test_negative_control_unrenderable_spec_surfaces():
    """A spec whose only layer carries no usable channels still returns a Figure or errors.

    The renderer must not silently swallow a structurally-empty spec into a
    non-figure; it returns a Figure (possibly empty axes) — never ``None``.
    """
    from matplotlib.figure import Figure

    spec = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[])
    fig = spec.render("matplotlib")
    assert isinstance(fig, Figure)


def test_render_buffer_is_a_nonempty_png():
    """A rendered figure saves to a real, non-empty PNG (a render-to-image smoke)."""
    spec = _spec_for_kind(PlotKind.PHASE_PORTRAIT_2D)
    fig = spec.render("matplotlib")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = buf.getvalue()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    assert len(data) > 1000, "suspiciously small PNG (empty render?)"


# ---------------------------------------------------------------------------
# 6 — default backend selection is order-independent (matplotlib by name)
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_renderers():
    """Snapshot + restore the global renderers registry around a test.

    The renderers registry is a process global; these tests rewrite it to a
    controlled order, so restore it afterwards to avoid leaking into other tests.
    """
    saved = list(registry.renderers.all())
    registry.renderers.clear()
    try:
        yield registry.renderers
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj)


def _fake_renderer(caps):
    """A minimal renderer callable carrying a ``.capabilities`` descriptor."""

    def _render(spec, **kw):  # pragma: no cover - never actually invoked here
        return spec

    _render.capabilities = caps  # type: ignore[attr-defined]
    return _render


def test_matplotlib_is_default_when_registered_last(_isolated_renderers):
    """matplotlib is the no-backend default even when registered *after* others.

    Regression guard: the default is chosen by name, not by registration order,
    so seating it first in the registry is unnecessary (the removed
    ``_seat_preferred_first`` reshuffle had no observable effect).
    """
    from tsdynamics.viz.render import select_renderer
    from tsdynamics.viz.render.caps import RendererCapabilities

    renderers = _isolated_renderers
    # Register a capable *drawing* backend first, then matplotlib last.
    renderers.register("plotly", _fake_renderer(RendererCapabilities.all_kinds("plotly")))
    renderers.register("matplotlib", _fake_renderer(RendererCapabilities.all_kinds("matplotlib")))

    spec = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[_layer_for_mark(PlotKind.LINE)])
    name, _ = select_renderer(spec, backend=None)
    assert name == "matplotlib"


def test_matplotlib_default_independent_of_order(_isolated_renderers):
    """The no-backend default is matplotlib for either registration order."""
    from tsdynamics.viz.render import select_renderer
    from tsdynamics.viz.render.caps import RendererCapabilities

    spec = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[_layer_for_mark(PlotKind.LINE)])
    for order in (("matplotlib", "plotly"), ("plotly", "matplotlib")):
        _isolated_renderers.clear()
        for nm in order:
            _isolated_renderers.register(nm, _fake_renderer(RendererCapabilities.all_kinds(nm)))
        name, _ = select_renderer(spec, backend=None)
        assert name == "matplotlib", order


def test_last_resort_prefers_drawing_backend_over_exporter(_isolated_renderers):
    """When no backend *declares* it can draw, the first drawing backend wins.

    matplotlib is absent and neither remaining backend declares the spec's kind,
    so selection falls to step 3.  A data-export backend registered first must
    not win over a later drawing backend — the figure-producing one is preferred.
    """
    from tsdynamics.viz.render import select_renderer
    from tsdynamics.viz.render.caps import RendererCapabilities

    renderers = _isolated_renderers
    # An exporter registered first, then a (partial) drawing backend that does
    # not declare this kind — both decline, so step 3 decides.
    renderers.register(
        "json",
        _fake_renderer(RendererCapabilities.of_kinds("json", [], data_export=True)),
    )
    renderers.register(
        "plotly",
        _fake_renderer(RendererCapabilities.of_kinds("plotly", [PlotKind.IMAGE])),
    )

    spec = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[_layer_for_mark(PlotKind.LINE)])
    name, _ = select_renderer(spec, backend=None)
    assert name == "plotly"


# ---------------------------------------------------------------------------
# 6 — the honesty contract covers the figure-geometry theme fields
# ---------------------------------------------------------------------------
#
# ``Theme`` grew ``figsize`` / ``dpi`` / ``layout_engine`` / ``autostyle`` in this
# phase, but ``style_honoring_gaps`` did not learn about them: it returned ``[]``
# for matplotlib *and* plotly, and for threejs named only the six pre-existing
# theme fields.  Three of the four were therefore accepted and dropped in silence
# by two backends — a documented knob that does nothing, which is the precise
# failure the honoring contract exists to make loud.
#
# Each assertion below is checked against the *rendered artifact*, not just the
# gap table, so an overclaim ("we honor it") cannot ship green.


def _geometry_spec(**theme_kw):
    spec = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[_layer_for_mark(PlotKind.LINE)])
    return spec.theme("default", **theme_kw)


def test_matplotlib_honors_all_four_geometry_fields_so_reports_no_gap():
    """matplotlib applies figsize / dpi / layout_engine / autostyle — no gap claimed."""
    from tsdynamics.viz.render.caps import style_honoring_gaps

    spec = _geometry_spec(
        figsize=(5.0, 3.5), dpi=300.0, layout_engine="constrained", autostyle=False
    )
    assert style_honoring_gaps(spec, "matplotlib") == []
    # ... and the artifact proves the claim.
    fig = spec.render("matplotlib")
    assert tuple(fig.get_size_inches()) == (5.0, 3.5)
    assert fig.get_dpi() == 300.0
    assert type(fig.get_layout_engine()).__name__ == "ConstrainedLayoutEngine"
    assert fig.axes[0].lines[0].get_linewidth() == 1.5  # autostyle off -> theme width


def test_plotly_reports_the_three_geometry_fields_it_ignores():
    """plotly honors ``autostyle`` only; the other three are declared gaps."""
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.caps import style_honoring_gaps

    spec = _geometry_spec(figsize=(5.0, 3.5), dpi=300.0, layout_engine="constrained")
    from tsdynamics.viz.render.caps import VisualizationDegraded

    assert style_honoring_gaps(spec, "plotly") == ["dpi", "figsize", "layout_engine"]
    # The artifact confirms the gap is real (plotly sizes from layout.width/height),
    # and the user is told about it exactly once.
    with pytest.warns(VisualizationDegraded, match="dpi, figsize, layout_engine"):
        layout = spec.render("plotly").to_plotly_json()["layout"]
    assert layout.get("width") is None and layout.get("height") is None

    # autostyle IS honored, so it must NOT be claimed as a gap...
    auto_off = _geometry_spec(autostyle=False)
    assert "autostyle" not in style_honoring_gaps(auto_off, "plotly")
    # ... and the trace proves it (density-aware width is switched off).
    trace = auto_off.render("plotly").to_plotly_json()["data"][0]
    assert trace["line"]["width"] == 1.5


def test_threejs_reports_all_four_geometry_fields():
    """three.js honors none of the four; ``autostyle`` only when actually set off."""
    from tsdynamics.viz.render.caps import style_honoring_gaps

    spec = _geometry_spec(
        figsize=(5.0, 3.5), dpi=300.0, layout_engine="constrained", autostyle=False
    )
    from tsdynamics.viz.render.caps import VisualizationDegraded

    gaps = style_honoring_gaps(spec, "threejs")
    assert {"figsize", "dpi", "layout_engine", "autostyle"} <= set(gaps)
    # The payload confirms it: the exported theme block carries background/palette
    # only, so no geometry field can have reached the loader.
    with pytest.warns(VisualizationDegraded):
        payload = spec.render("threejs").payload
    theme_block = payload["metadata"]["theme"]
    assert set(theme_block) <= {"background", "palette"}


def test_untouched_geometry_defaults_are_not_reported_as_gaps():
    """Only knobs actually *set* are reported — an untouched default is not noise."""
    from tsdynamics.viz.render.caps import style_honoring_gaps

    plain = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[_layer_for_mark(PlotKind.LINE)])
    for backend in ("matplotlib", "plotly", "threejs"):
        gaps = style_honoring_gaps(plain, backend)
        assert not ({"figsize", "dpi", "layout_engine", "autostyle"} & set(gaps)), backend


def test_geometry_gap_is_also_seen_via_spec_size():
    """``spec.size(...)`` writes ``meta``, not the theme — both must be checked."""
    from tsdynamics.viz.render.caps import style_honoring_gaps

    spec = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[_layer_for_mark(PlotKind.LINE)])
    spec.size(5.0, 3.5, dpi=200.0)
    gaps = style_honoring_gaps(spec, "threejs")
    assert "figsize" in gaps and "dpi" in gaps


def test_the_geometry_gaps_reach_the_user_as_one_warning():
    """The dispatcher's consolidated VisualizationDegraded names the ignored knobs."""
    from tsdynamics.viz.render.caps import VisualizationDegraded

    spec = _geometry_spec(figsize=(5.0, 3.5), dpi=300.0)
    with pytest.warns(VisualizationDegraded, match="figsize"):
        spec.render("threejs")


# ---------------------------------------------------------------------------
# 7 — one definition of "is this spec 3-D"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec_factory",
    [
        lambda: PlotSpec(kind=PlotKind.TIME_SERIES, layers=[_layer_for_mark(PlotKind.LINE)]),
        lambda: _spec_for_kind(PlotKind.PHASE_PORTRAIT_3D),
        lambda: PlotSpec(
            kind=PlotKind.TIME_SERIES,
            layers=[_layer_for_mark(PlotKind.SURFACE3D)],
            ndim=2,
        ),
    ],
    ids=["2d", "3d", "surface3d_with_ndim2"],
)
def test_every_backend_shares_one_three_d_predicate(spec_factory):
    """The renderers' ``is_three_d`` is the spec's property, not a private copy.

    It lived in three byte-identical copies (the capability check, the matplotlib
    renderer, the plotly renderer).  A renderer that disagreed with the dispatcher
    about what "3-D" means routes a spec onto axes it cannot draw on, so the
    predicate belongs to the spec and the copies are gone.
    """
    from tsdynamics.viz.render.mpl import _threed as mpl_threed

    spec = spec_factory()
    assert mpl_threed.is_three_d(spec) is spec.is_three_d
    plotly = pytest.importorskip("plotly")  # noqa: F841
    from tsdynamics.viz.render.plotly import _threed as plotly_threed

    assert plotly_threed.is_three_d(spec) is spec.is_three_d


def test_capability_check_uses_the_spec_three_d_property():
    """A 2-D-only backend declines a 3-D spec through ``PlotSpec.is_three_d``."""
    from tsdynamics.viz.render.caps import RendererCapabilities

    caps = RendererCapabilities.all_kinds("flat", supports_3d=False)
    assert not hasattr(RendererCapabilities, "_is_three_d")  # the copy is gone
    assert caps.can_render_spec(_spec_for_kind(PlotKind.PHASE_PORTRAIT_3D)) is False
    assert caps.can_render_spec(_spec_for_kind(PlotKind.TIME_SERIES)) is True
