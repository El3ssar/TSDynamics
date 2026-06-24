"""Tests for ``tsdynamics.viz.plot`` — the composition front door (Phase 2).

Covers the agreed contract:

1. ``plot(thing)`` returns that thing's single-panel spec; ``plot(a, b)`` overlays
   compatible things on one set of axes; ``plot(a, b, layout="stack"/"row"/"grid")``
   builds a ``COMPOSITE`` spec with one panel each.
2. The return type is always a :class:`PlotSpec`, so a ``plot`` result feeds back
   into ``plot`` (recursion), and composite inputs are flattened into panels.
3. ``build_kw`` (``components`` / per-kind options) is forwarded to each thing.
4. Composite specs round-trip through ``to_dict`` / ``from_dict`` byte-identical
   and render to a multi-panel matplotlib figure; the spec saves itself.
5. Building a composite spec imports no plotting library.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import tsdynamics as ts
import tsdynamics.viz as viz
from tsdynamics.data import Trajectory
from tsdynamics.errors import InvalidInputError, InvalidParameterError
from tsdynamics.viz.spec import PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# Builders (fast tier — short integrations)
# ---------------------------------------------------------------------------


def _lorenz(ic=(1.0, 1.0, 1.0)):
    return ts.Lorenz().integrate(final_time=20.0, dt=0.02, ic=list(ic)).after(5.0)


def _l96(ic=None):
    kw = {} if ic is None else {"ic": ic}
    return ts.Lorenz96(N=8).trajectory(final_time=8.0, dt=0.1, **kw)


def _named_3d(title: str) -> Trajectory:
    """A 3-D trajectory titled ``title`` (→ a PHASE_PORTRAIT_3D, single layer)."""
    t = np.linspace(0.0, 1.0, 20)
    y = np.random.default_rng(0).standard_normal((20, 3))
    return Trajectory(t, y, system=None, meta={"system": title})


def _roundtrips(spec: PlotSpec) -> None:
    assert PlotSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


# ---------------------------------------------------------------------------
# Single + overlay (one panel)
# ---------------------------------------------------------------------------


def test_single_thing_returns_its_spec():
    tr = _lorenz()
    spec = viz.plot(tr)
    assert isinstance(spec, PlotSpec)
    assert spec.kind == PlotKind.PHASE_PORTRAIT_3D
    assert not spec.is_composite


def test_overlay_two_trajectories_one_panel():
    spec = viz.plot(_lorenz([1.0, 1.0, 1.0]), _lorenz([1.1, 1.0, 1.0]))
    assert spec.kind == PlotKind.PHASE_PORTRAIT_3D
    assert not spec.is_composite
    assert len(spec.layers) == 2  # one curve per source
    assert spec.legend is not None
    # labels are disambiguated so the legend can tell the two apart
    assert len({lyr.label for lyr in spec.layers}) == 2
    _roundtrips(spec)


def test_overlay_forwards_build_kwargs_to_each_thing():
    spec = viz.plot(_lorenz(), _lorenz([1.1, 1.0, 1.0]), components="x")
    assert spec.kind == PlotKind.TIME_SERIES
    assert len(spec.layers) == 2


def test_overlay_of_incompatible_kinds_raises():
    # a spacetime image cannot share axes with a 3-D portrait
    with pytest.raises(InvalidParameterError):
        viz.plot(_l96(), _lorenz())


def test_overlay_mixing_2d_and_3d_portrait_raises():
    two_d = _lorenz().to_plot_spec(components=["x", "y"])  # PHASE_PORTRAIT_2D
    three_d = _lorenz().to_plot_spec()  # PHASE_PORTRAIT_3D
    with pytest.raises(InvalidParameterError):
        viz.plot(two_d, three_d)


# ---------------------------------------------------------------------------
# Composite (panels)
# ---------------------------------------------------------------------------


def test_stack_builds_composite_with_one_panel_each():
    spec = viz.plot(_lorenz(), _lorenz([1.1, 1.0, 1.0]), layout="stack")
    assert spec.kind == PlotKind.COMPOSITE
    assert spec.is_composite
    assert len(spec.panels) == 2
    assert spec.layout is not None and spec.layout.mode == "stack"
    _roundtrips(spec)


@pytest.mark.parametrize("mode", ["stack", "row", "grid"])
def test_layout_modes_set_the_layout(mode):
    spec = viz.plot(_lorenz(), _lorenz(), _lorenz(), layout=mode)
    assert spec.kind == PlotKind.COMPOSITE
    assert spec.layout is not None and spec.layout.mode == mode
    assert len(spec.panels) == 3


def test_stack_of_same_x_shares_x():
    px = viz.plot(_lorenz(), components="x")
    py = viz.plot(_lorenz(), components="y")
    spec = viz.plot(px, py, layout="stack")
    assert spec.layout is not None and spec.layout.share_x is True


def test_recursion_flattens_composite_inputs():
    px = viz.plot(_lorenz(), _lorenz([1.1, 1.0, 1.0]), components="x")  # overlay panel
    py = viz.plot(_lorenz(), _lorenz([1.1, 1.0, 1.0]), components="y")
    pz = viz.plot(_lorenz(), _lorenz([1.1, 1.0, 1.0]), components="z")
    fig_spec = viz.plot(px, py, pz, layout="stack")
    assert fig_spec.kind == PlotKind.COMPOSITE
    assert len(fig_spec.panels) == 3  # each overlay is one panel
    # feeding a composite back in flattens its panels (no nested composite)
    bigger = viz.plot(fig_spec, viz.plot(_lorenz(), components="x"), layout="stack")
    assert len(bigger.panels) == 4
    assert all(not p.is_composite for p in bigger.panels)


def test_two_spacetime_images_stack():
    spec = viz.plot(_l96(), _l96(ic=np.full(8, 0.5)), layout="stack")
    assert spec.kind == PlotKind.COMPOSITE
    assert [p.kind for p in spec.panels] == [PlotKind.SPACETIME, PlotKind.SPACETIME]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_plot_raises():
    with pytest.raises(InvalidParameterError):
        viz.plot()


def test_unknown_layout_raises():
    with pytest.raises(InvalidParameterError):
        viz.plot(_lorenz(), _lorenz(), layout="diagonal")


def test_build_kwargs_with_prebuilt_spec_raises():
    spec = _lorenz().to_plot_spec()
    with pytest.raises(InvalidParameterError):
        viz.plot(spec, components="x")


def test_non_plottable_thing_raises():
    with pytest.raises(InvalidInputError):
        viz.plot(object())


def test_list_argument_is_unwrapped():
    spec = viz.plot([_lorenz(), _lorenz()], layout="row")
    assert spec.kind == PlotKind.COMPOSITE
    assert len(spec.panels) == 2


# ---------------------------------------------------------------------------
# Rendering (matplotlib tiles composites; plotly declines → falls back)
# ---------------------------------------------------------------------------


def test_render_stacked_time_series_has_one_axes_per_panel():
    pytest.importorskip("matplotlib")
    px = viz.plot(_lorenz(), components="x")
    py = viz.plot(_lorenz(), components="y")
    fig = viz.plot(px, py, layout="stack").render(backend="matplotlib")
    assert type(fig).__name__ == "Figure"
    assert len(fig.axes) == 2  # time-series panels carry no colorbar


def test_render_stacked_spacetime_has_a_panel_axes_each():
    pytest.importorskip("matplotlib")
    fig = viz.plot(_l96(), _l96(ic=np.full(8, 0.5)), layout="stack").render(backend="matplotlib")
    # each spacetime panel adds its own colorbar axes, so >= 2 panel axes
    assert len(fig.axes) >= 2


def test_render_composite_with_3d_panels():
    pytest.importorskip("matplotlib")
    fig = viz.plot(_lorenz(), _lorenz([1.1, 1.0, 1.0]), layout="row").render(backend="matplotlib")
    assert type(fig).__name__ == "Figure"
    assert len(fig.axes) == 2


def test_render_composite_mixed_2d_and_3d_panels():
    pytest.importorskip("matplotlib")
    ts_panel = viz.plot(_lorenz(), components="x")  # 2-D
    portrait = viz.plot(_lorenz())  # 3-D
    fig = viz.plot(ts_panel, portrait, layout="stack").render(backend="matplotlib")
    assert len(fig.axes) == 2


def test_spec_saves_itself_to_png(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "composite.png"
    returned = viz.plot(
        viz.plot(_lorenz(), components="x"),
        viz.plot(_lorenz(), components="y"),
        layout="stack",
    ).save(str(out))
    assert returned == str(out)
    assert out.stat().st_size > 0


def test_spec_plot_returns_a_figure():
    pytest.importorskip("matplotlib")
    fig = viz.plot(_lorenz(), _lorenz([1.1, 1.0, 1.0]), layout="stack").plot(backend="matplotlib")
    assert type(fig).__name__ == "Figure"


# ---------------------------------------------------------------------------
# Import-light: composing builds no plotting library
# ---------------------------------------------------------------------------


def test_overlay_label_disambiguation_numbers_colliding_titles():
    spec = viz.plot(_named_3d("S"), _named_3d("S"), _named_3d("S"))
    assert [lyr.label for lyr in spec.layers] == ["S (1)", "S (2)", "S (3)"]


def test_overlay_distinct_titles_are_not_numbered():
    spec = viz.plot(_named_3d("A"), _named_3d("B"))
    assert sorted(lyr.label or "" for lyr in spec.layers) == ["A", "B"]


def test_single_element_list_returns_the_spec_not_a_one_panel_composite():
    spec = viz.plot([_lorenz()])
    assert not spec.is_composite
    assert spec.kind == PlotKind.PHASE_PORTRAIT_3D


def test_share_x_only_for_stacked_time_series_with_a_common_x():
    tsx = viz.plot(_lorenz(), components="x")
    tsy = viz.plot(_lorenz(), components="y")
    portrait = viz.plot(_lorenz())  # not a time series
    # stacked time series with the same x label → shared
    assert viz.plot(tsx, tsy, layout="stack").layout.share_x is True
    # a non-time-series panel in the stack → not shared
    assert viz.plot(tsx, portrait, layout="stack").layout.share_x is False
    # row / grid never auto-share
    assert viz.plot(tsx, tsy, layout="row").layout.share_x is False


def test_composite_of_3d_panels_round_trips_byte_identical():
    spec = viz.plot(_lorenz(), _lorenz([1.1, 1.0, 1.0]), layout="row")
    assert spec.kind == PlotKind.COMPOSITE
    assert all(p.kind == PlotKind.PHASE_PORTRAIT_3D for p in spec.panels)
    _roundtrips(spec)


def test_save_json_writes_the_ir(tmp_path):
    import json

    out = tmp_path / "spec.json"
    returned = viz.plot(_lorenz(), _lorenz([1.1, 1.0, 1.0]), layout="stack").save(str(out))
    assert returned == str(out)
    data = json.loads(out.read_text())["spec"]  # json export wraps in a versioned envelope
    assert data["kind"] == "composite"
    assert len(data["panels"]) == 2


def test_composite_render_with_plotly_backend_falls_back_to_matplotlib():
    pytest.importorskip("plotly")
    pytest.importorskip("matplotlib")  # the fallback target tiles the composite
    from tsdynamics.viz.render.caps import VisualizationDegraded

    comp = viz.plot(
        viz.plot(_lorenz(), components="x"),
        viz.plot(_lorenz(), components="y"),
        layout="stack",
    )
    with pytest.warns(VisualizationDegraded):
        fig = comp.render(backend="plotly")
    assert type(fig).__name__ == "Figure"  # matplotlib tiled it after the decline


def test_composing_imports_no_plot_library():
    code = (
        "import sys, numpy as np, tsdynamics as ts, tsdynamics.viz as viz;"
        "a = ts.Lorenz().integrate(final_time=10.0, dt=0.05).after(2.0);"
        "b = ts.Lorenz().integrate(final_time=10.0, dt=0.05, ic=[1.1,1,1]).after(2.0);"
        "s = viz.plot(viz.plot(a, b, components='x'), viz.plot(a, b, components='y'), layout='stack');"
        "s.to_dict();"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'plotly')];"
        "assert not bad, bad; print('NO_PLOT_LIBS')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "NO_PLOT_LIBS" in out.stdout
