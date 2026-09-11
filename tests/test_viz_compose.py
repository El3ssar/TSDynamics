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
    return ts.systems.Lorenz().run(final_time=20.0, dt=0.02, ic=list(ic)).after(5.0)


def _l96(ic=None):
    kw = {} if ic is None else {"ic": ic}
    return ts.systems.Lorenz96(N=8).run(final_time=8.0, dt=0.1, **kw)


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


def test_overlay_of_incompatible_frames_raises():
    """A spacetime lattice and a 3-D phase portrait are different spaces.

    Still refused after the v6 widening — but now for the *reason* rather than
    because a three-member kind whitelist happened to exclude both.
    """
    with pytest.raises(InvalidParameterError, match="different spaces"):
        viz.plot(_l96(), _lorenz())


def test_overlay_mixing_2d_and_3d_portrait_raises():
    """Same space family, different dimension — ``state2`` is not ``state3``."""
    two_d = _lorenz().to_plot_spec(components=["x", "y"])  # PHASE_PORTRAIT_2D
    three_d = _lorenz().to_plot_spec()  # PHASE_PORTRAIT_3D
    with pytest.raises(InvalidParameterError, match="different spaces"):
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


def test_spec_plot_returns_the_spec_and_render_returns_the_figure():
    """v6: ``plot`` builds, ``render`` draws — one return type per verb."""
    pytest.importorskip("matplotlib")
    spec = viz.plot(_lorenz(), _lorenz([1.1, 1.0, 1.0]), layout="stack")
    assert spec.tweak(title="mine") is spec
    assert type(spec.render("matplotlib")).__name__ == "Figure"


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


def test_composite_render_with_plotly_backend_is_native_no_fallback():
    # plotly now tiles a composite natively (make_subplots), so no fallback /
    # VisualizationDegraded warning fires for a composite of supported panels.
    pytest.importorskip("plotly")
    import warnings

    from tsdynamics.viz.render.caps import VisualizationDegraded

    comp = viz.plot(
        viz.plot(_lorenz(), components="x"),
        viz.plot(_lorenz(), components="y"),
        layout="stack",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", VisualizationDegraded)
        fig = comp.render(backend="plotly")
    # a plotly figure, not a matplotlib one — the module is plotly's
    assert type(fig).__module__.startswith("plotly")
    assert type(fig).__name__ == "Figure"


def test_composing_imports_no_plot_library():
    code = (
        "import sys, numpy as np, tsdynamics as ts, tsdynamics.viz as viz;"
        "a = ts.systems.Lorenz().run(final_time=10.0, dt=0.05).after(2.0);"
        "b = ts.systems.Lorenz().run(final_time=10.0, dt=0.05, ic=[1.1,1,1]).after(2.0);"
        "s = viz.plot(viz.plot(a, b, components='x'), viz.plot(a, b, components='y'), layout='stack');"
        "s.to_dict();"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'plotly')];"
        "assert not bad, bad; print('NO_PLOT_LIBS')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "NO_PLOT_LIBS" in out.stdout


# ---------------------------------------------------------------------------
# H3 — composite as a first-class figure (v6 Phase 5, Owner A)
# ---------------------------------------------------------------------------


def test_grid_layout_accepts_an_explicit_shape():
    """``rows=`` / ``cols=`` reach the Layout.

    ``Layout`` always carried the fields, but ``plot()`` had no way to set them —
    so a 4-panel grid was stuck on the auto-derived 2x2 and a 1x4 could not be
    asked for at all.
    """
    panels = [viz.plot(_lorenz(), components=c) for c in ("x", "y", "z")]
    spec = viz.plot(*panels, layout="grid", rows=1, cols=3)
    assert (spec.layout.rows, spec.layout.cols) == (1, 3)
    assert spec.layout.grid(3) == (1, 3)


def test_grid_without_a_shape_stays_near_square():
    panels = [viz.plot(_lorenz(), components=c) for c in ("x", "y", "z")]
    spec = viz.plot(*panels, layout="grid")
    assert spec.layout.grid(3) == (2, 2)


def test_explicit_share_flags_override_the_auto_default():
    tsx = viz.plot(_lorenz(), components="x")
    tsy = viz.plot(_lorenz(), components="y")
    # auto would say True for a same-x time-series stack; the explicit flag wins
    assert viz.plot(tsx, tsy, layout="stack", share_x=False).layout.share_x is False
    assert viz.plot(tsx, tsy, layout="row", share_y=True).layout.share_y is True
    assert viz.plot(tsx, tsy, layout="row", share_color=True).layout.share_color is True


def test_layout_keywords_are_rejected_for_an_overlay():
    """An overlay is one set of axes — a grid shape there has no meaning."""
    with pytest.raises(InvalidParameterError, match="layout='overlay'"):
        viz.plot(_lorenz(), _lorenz(), rows=2)


def test_flattening_a_child_composite_keeps_its_theme_and_animation():
    """A flattened child's figure-level context is pushed down, not discarded.

    Before v6 ``plot(plot(a, b).theme("dark"), c, layout="stack")`` silently lost
    the dark theme: ``_composite`` extended with ``spec.panels`` directly, so the
    child's ``_theme`` / ``animation`` (which live on the child, not its panels)
    went nowhere.
    """
    inner = viz.plot(
        viz.plot(_lorenz(), components="x"),
        viz.plot(_lorenz(), components="y"),
        layout="row",
    ).theme("dark")
    inner.animate(fps=15.0)
    outer = viz.plot(inner, viz.plot(_lorenz(), components="z"), layout="stack")
    assert len(outer.panels) == 3
    inherited = outer.panels[:2]
    assert all(p.resolved_theme.name == "dark" for p in inherited)
    assert all(p.animation is not None and p.animation.fps == 15.0 for p in inherited)
    # ... and the arrangement it could not keep is recorded, not vanished.
    assert outer.meta["flattened_layouts"] == ["row"]


def test_composite_tweaks_reach_the_panels_end_to_end():
    """The user-facing spelling of F1: a tweak chained on ``ts.viz.plot(...)`` lands."""
    spec = viz.plot(
        viz.plot(_lorenz(), components="x"),
        viz.plot(_lorenz(), components="y"),
        layout="stack",
    )
    before = [p.to_dict() for p in spec.panels]
    spec.recolor("red", "blue").limits(x=(0.0, 5.0)).relabel(x="t", title="two views")
    after = [p.to_dict() for p in spec.panels]
    assert all(b != a for b, a in zip(before, after, strict=True))
    assert [p.layers[0].style["color"] for p in spec.panels] == ["red", "blue"]
    assert all(p.x.limits == (0.0, 5.0) and p.x.label == "t" for p in spec.panels)
    assert spec.title == "two views" and all(p.title != "two views" for p in spec.panels)


# ---------------------------------------------------------------------------
# P0 — COMPOSABILITY: overlay legality is FRAME identity, not kind identity
# ---------------------------------------------------------------------------


class _DuffingTwoWell(ts.ContinuousSystem):
    """Damped two-well Duffing ``x'' = x - x**3 - delta x'`` — wells at ``x = ±1``.

    Test-local (like the copy in ``tests/test_basins.py``): the catalogue has
    exactly one unforced planar ODE, and the flagship overlay needs a 2-D flow
    with two basins, equilibria and an attractor set.
    """

    params = {"delta": 0.3}
    dim = 2
    variables = ("x", "v")

    @staticmethod
    def _equations(Y, t, *, delta):
        x, v = Y(0), Y(1)
        return (v, x - x**3 - delta * v)


def _image_spec(label="basins", kind=PlotKind.BASINS_IMAGE, xlabel="x", ylabel="v"):
    """A minimal field spec on the ``(x, v)`` plane (a basin image, in miniature)."""
    from tsdynamics.viz.spec import Axis, Layer

    g = np.add.outer(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))
    return PlotSpec(
        kind=kind,
        aspect="equal",
        x=Axis(label=xlabel),
        y=Axis(label=ylabel),
        layers=[
            Layer(PlotKind.IMAGE, {"x": np.arange(8.0), "y": np.arange(8.0), "c": g}, label=label)
        ],
        title=label,
    )


def _curve_spec(label="trajectory", xlabel="x", ylabel="v"):
    """A minimal ``PHASE_PORTRAIT_2D`` curve on the ``(x, v)`` plane."""
    from tsdynamics.viz.spec import Axis, Layer

    t = np.linspace(0.0, 6.0, 32)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        x=Axis(label=xlabel),
        y=Axis(label=ylabel),
        layers=[Layer(PlotKind.LINE, {"x": np.cos(t), "y": np.sin(t)}, label=label)],
        title=label,
    )


def _marker_spec(label="equilibria", xlabel="x", ylabel="v"):
    """A minimal ``FIXED_POINTS_OVERLAY`` on the ``(x, v)`` plane."""
    from tsdynamics.viz.spec import Annotation, Axis, Layer

    return PlotSpec(
        kind=PlotKind.FIXED_POINTS_OVERLAY,
        x=Axis(label=xlabel),
        y=Axis(label=ylabel),
        layers=[Layer(PlotKind.SCATTER, {"x": np.zeros(1), "y": np.zeros(1)}, label=label)],
        annotations=[Annotation(kind="text", text="lambda=+0.86", x=0.0, y=0.0)],
        title=label,
    )


def _png(spec: PlotSpec) -> bytes:
    import io

    fig = spec.render("matplotlib")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    return buf.getvalue()


def test_the_flagship_overlay_of_four_different_kinds_on_one_axes():
    """Four kinds, one plane, one set of axes — the defect this phase closes.

    Before v6 this raised ``InvalidParameterError: cannot overlay specs of kinds
    ['fixed_points_overlay', 'phase_portrait_2d']`` — a *policy* refusal, since
    the renderers already drew image + line + scatter on one axes correctly.
    """
    spec = viz.plot(_image_spec(), _curve_spec(), _marker_spec())
    assert not spec.is_composite
    assert [lyr.kind for lyr in spec.layers] == [
        PlotKind.IMAGE,
        PlotKind.LINE,
        PlotKind.SCATTER,
    ]
    # the field owns the merged identity: a basin image keeps its own kind (and
    # therefore its categorical presentation) with curves drawn over it
    assert spec.kind == PlotKind.BASINS_IMAGE
    assert str(spec.resolved_frame) == "state2(x, v)"
    _roundtrips(spec)


def test_overlay_z_order_is_by_role_not_argument_order():
    """``plot(basins, traj)`` and ``plot(traj, basins)`` are the same picture.

    Byte-identical PNGs, not merely the same layer set: the field draws under
    the curve because of what those things *are*, so the call is order-free.
    """
    pytest.importorskip("matplotlib")
    forward = viz.plot(_image_spec(), _curve_spec())
    reversed_ = viz.plot(_curve_spec(), _image_spec())
    assert [lyr.kind for lyr in forward.layers] == [PlotKind.IMAGE, PlotKind.LINE]
    assert [lyr.kind for lyr in reversed_.layers] == [PlotKind.IMAGE, PlotKind.LINE]
    assert _png(forward) == _png(reversed_)


def test_equal_role_specs_keep_their_argument_order():
    """Role sorting is *stable*: it orders roles, never same-role siblings.

    This is what keeps every overlay that was legal before v6 byte-identical —
    those were all same-kind (and therefore same-role) merges.
    """
    a, b = _curve_spec("a"), _curve_spec("b")
    # The labels are identity markers here, not the subject: these specs are
    # titled the same as their single layer, so the overlay tag adds nothing and
    # is not prefixed (see ``_relabel_for_overlay``).  What is asserted is the
    # *order*.
    assert [lyr.label for lyr in viz.plot(a, b).layers] == ["a", "b"]
    assert [lyr.label for lyr in viz.plot(b, a).layers] == ["b", "a"]


def test_overlay_onto_a_different_plane_raises_naming_both():
    """The wrong-plane bug, now structurally impossible.

    ``fixedpoints/fixed.py`` used to patch this by hand for its own overlay path
    (forwarding ``components=``); the frame check subsumes it and covers every
    producer, not just that one.
    """
    with pytest.raises(InvalidParameterError, match="axes mismatch"):
        viz.plot(_curve_spec(ylabel="v"), _marker_spec(ylabel="z"))


def test_overlay_of_different_spaces_raises_naming_both_frames():
    """A recurrence plot is not a drawing of a phase plane."""
    from tsdynamics.viz.spec import Axis, Layer

    recurrence = PlotSpec(
        kind=PlotKind.RECURRENCE_PLOT,
        x=Axis(label="i"),
        y=Axis(label="j"),
        layers=[Layer(PlotKind.SCATTER, {"x": np.zeros(3), "y": np.zeros(3)})],
    )
    with pytest.raises(InvalidParameterError, match="different spaces"):
        viz.plot(_curve_spec(), recurrence)


def test_time_series_of_different_components_still_overlay():
    """A ``time`` frame is ONE coordinate: the y axis is free, so x(t) + y(t) is legal."""
    spec = viz.plot(_lorenz(), _lorenz(), components="x")
    assert spec.kind == PlotKind.TIME_SERIES
    px = viz.plot(_lorenz(), components="x")
    py = viz.plot(_lorenz(), components="y")
    merged = viz.plot(px, py)
    assert len(merged.layers) == 2


def test_on_force_overlays_a_deliberate_mismatch_with_one_warning():
    from tsdynamics.viz.render.caps import VisualizationDegraded

    with pytest.warns(VisualizationDegraded, match="on='force'"):
        spec = viz.plot(_curve_spec(ylabel="v"), _marker_spec(ylabel="z"), on="force")
    assert len(spec.layers) == 2


def test_unknown_on_value_raises():
    with pytest.raises(InvalidParameterError, match="unknown on="):
        viz.plot(_curve_spec(), _marker_spec(), on="forse")


def test_on_is_rejected_for_a_panelled_layout():
    with pytest.raises(InvalidParameterError, match="nothing to force"):
        viz.plot(_curve_spec(), _marker_spec(), layout="stack", on="force")


def test_overlay_keeps_the_annotations_of_every_source():
    """Annotations used to be dropped by ``plot`` (``overlay_on`` kept them)."""
    spec = viz.plot(_curve_spec(), _marker_spec())
    assert [a.text for a in spec.annotations] == ["lambda=+0.86"]


def test_overlay_keeps_the_first_argument_s_theme_and_animation():
    """Presentation context is the caller's; z-order is the data's.

    ``_theme`` / ``animation`` follow *argument* order (the first thing you named
    is the figure you are building), while layers follow *role* order.
    """
    base = _curve_spec().theme("dark")
    base.animate(fps=15.0)
    spec = viz.plot(base, _image_spec())
    assert spec.resolved_theme.name == "dark"
    assert spec.animation is not None and spec.animation.fps == 15.0


# ---------------------------------------------------------------------------
# PlotSpec.add — the incremental (Makie-style) build
# ---------------------------------------------------------------------------


def test_add_chains_and_equals_the_one_shot_call():
    """``plot(a).add(b).add(c)`` is the same spec as ``plot(a, b, c)``.

    Same merge, same frame check, same role ordering — so the incremental and
    the one-shot spellings cannot drift apart (including the legend labels,
    which an earlier draft double-prefixed on the second ``add``).
    """
    one_shot = viz.plot(_image_spec(), _curve_spec(), _marker_spec())
    built = viz.plot(_image_spec()).add(_curve_spec()).add(_marker_spec())
    assert built.to_dict() == one_shot.to_dict()


def test_add_mutates_in_place_and_returns_self():
    spec = viz.plot(_image_spec())
    assert spec.add(_curve_spec()) is spec
    assert len(spec.layers) == 2


def test_add_accepts_plottables_and_build_kwargs():
    spec = viz.plot(_lorenz(), components="x").add(_lorenz([1.1, 1.0, 1.0]), components="x")
    assert spec.kind == PlotKind.TIME_SERIES
    assert len(spec.layers) == 2


def test_add_frame_check_and_force():
    from tsdynamics.viz.render.caps import VisualizationDegraded

    with pytest.raises(InvalidParameterError, match="axes mismatch"):
        viz.plot(_curve_spec(ylabel="v")).add(_marker_spec(ylabel="z"))
    with pytest.warns(VisualizationDegraded):
        forced = viz.plot(_curve_spec(ylabel="v")).add(_marker_spec(ylabel="z"), on="force")
    assert len(forced.layers) == 2


def test_add_to_a_composite_raises_pointing_at_the_panels():
    composite = viz.plot(_curve_spec("a"), _curve_spec("b"), layout="stack")
    with pytest.raises(InvalidParameterError, match="panels"):
        composite.add(_marker_spec())


def test_add_with_nothing_raises():
    with pytest.raises(InvalidParameterError, match="at least one"):
        viz.plot(_curve_spec()).add()


# ---------------------------------------------------------------------------
# End-to-end acceptance: basins + attractors + trajectory + equilibria
# ---------------------------------------------------------------------------


def test_acceptance_basins_attractors_trajectory_and_equilibria_on_one_axes(tmp_path):
    """The owner's flagship call, end to end, on real computed results.

    Asserts the *picture*, not just the spec: the basin image underneath, the
    trajectory over it, the equilibria on top — and the equilibria in the right
    place (``x = 0, ±1`` for the two-well Duffing, ``v = 0``).
    """
    pytest.importorskip("matplotlib")
    pytest.importorskip("tsdynamics._rust")

    duffing = _DuffingTwoWell()
    grid = ts.data.Grid([-2.0, -2.0], [2.0, 2.0], (40, 40))
    basins = ts.analysis.basins(duffing, grid, dt=0.5)
    traj = duffing.run(final_time=30.0, dt=0.02, ic=[1.6, 1.2])
    fps = ts.analysis.fixed_points(duffing, region=ts.data.Box([-2.0, -2.0], [2.0, 2.0]), seed=0)
    assert len(fps) == 3

    spec = viz.plot(basins, basins.attractors, traj, fps)
    assert spec.kind == PlotKind.BASINS_IMAGE
    assert str(spec.resolved_frame) == "state2(x, v)"

    marks = [lyr.kind for lyr in spec.layers]
    assert marks.index(PlotKind.IMAGE) == 0  # the field is the backdrop
    assert marks.index(PlotKind.LINE) < len(marks) - 1  # the orbit is over it
    assert marks[-1] == PlotKind.SCATTER  # the equilibria are on top

    fig = spec.render("matplotlib")
    ax = fig.axes[0]
    assert len(ax.images) == 1 and len(ax.lines) == 1
    equilibria = np.concatenate(
        [
            c.get_offsets()
            for c in ax.collections
            if len(c.get_offsets()) and c.get_offsets()[0][1] == 0.0
        ]
    )
    xs = sorted({round(float(p[0]), 6) for p in equilibria})
    assert xs == pytest.approx([-1.0, 0.0, 1.0], abs=1e-6)

    out = tmp_path / "fig.png"
    assert spec.save(str(out)) == str(out)
    assert out.stat().st_size > 5000


# ---------------------------------------------------------------------------
# One overlay policy, not two — ``plot`` and ``AnalysisResult.overlay_on``
# ---------------------------------------------------------------------------


def test_plot_and_overlay_on_agree_on_what_is_legal():
    """The two doors used to disagree; they now share one check.

    Before v6 ``fps.overlay_on(portrait)`` *succeeded* on the exact pair
    ``viz.plot(portrait, fps)`` refused.  Both now go through
    ``_frames.check_overlay``, so a pair is legal at both doors or neither.
    """
    lorenz = ts.systems.Lorenz()
    traj = lorenz.run(final_time=10.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    fps = ts.analysis.fixed_points(lorenz, seed=0)

    same_plane = traj.to_plot_spec(components=("x", "z"))
    n_host = len(same_plane.layers)
    merged = fps.overlay_on(same_plane, components=("x", "z"))
    assert merged is same_plane  # host-first, mutate-and-return
    assert len(merged.layers) > n_host
    assert viz.plot(traj, fps, components=("x", "z")) is not None

    wrong_plane = traj.to_plot_spec(components=("x", "y"))
    with pytest.raises(InvalidParameterError, match="axes mismatch"):
        fps.overlay_on(wrong_plane, components=("x", "z"))
    with pytest.raises(InvalidParameterError, match="axes mismatch"):
        viz.plot(wrong_plane, fps.to_plot_spec(components=("x", "z")))


def test_the_generic_overlay_on_forwards_build_keywords():
    """The base method carries ``components=``, so no result re-implements it.

    ``FixedPoint`` / ``FixedPointSet`` each carried a hand-written
    ``overlay_on`` override whose only job was forwarding ``components`` past a
    base that dropped it.  The base forwards ``**build_kw`` now, and the frame
    check catches the mistake those overrides were patching around.
    """
    from tsdynamics.analysis.fixedpoints.fixed import FixedPoint, FixedPointSet

    assert "overlay_on" not in vars(FixedPoint)
    assert "overlay_on" not in vars(FixedPointSet)

    lorenz = ts.systems.Lorenz()
    traj = lorenz.run(final_time=10.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    fps = ts.analysis.fixed_points(lorenz, seed=0)
    host = traj.to_plot_spec(components=("x", "z"))
    n_host = len(host.layers)
    merged = fps.overlay_on(host, components=("x", "z"))
    zs = [float(v) for layer in merged.layers[n_host:] for v in layer.data["y"]]
    assert sorted(zs)[-2:] == pytest.approx([27.0, 27.0], abs=1e-6)


def test_a_recurrence_scatter_can_no_longer_be_spliced_onto_a_time_series():
    """The one deliberate behaviour break of this phase (release-note worthy).

    ``recurrence_matrix(...).overlay_on(time_series_spec)`` used to be accepted
    and produced a spec *labelled* ``time_series`` containing a recurrence
    scatter — a plot of one thing presented as another.  The frames genuinely
    differ (``index`` vs ``time``), so it now raises.
    """
    traj = ts.systems.Lorenz().run(final_time=10.0, dt=0.02, ic=[1.0, 1.0, 1.0])
    rm = ts.analysis.recurrence_matrix(np.asarray(traj["x"])[:150], threshold=1.0)
    with pytest.raises(InvalidParameterError, match="different spaces"):
        rm.overlay_on(traj.to_plot_spec(components="x"))


def test_a_merged_overlay_never_aliases_its_inputs():
    """Styling the composition must not reach back into the specs you passed in."""
    field, curve = _image_spec(), _curve_spec()
    spec = viz.plot(field, curve)
    spec.recolor("magenta", "cyan")
    spec.relabel(x="X", title="merged")
    assert field.layers[0].style.get("color") is None
    assert curve.layers[0].style.get("color") is None
    assert field.x.label == "x" and field.title == "basins"


# ===========================================================================
# One style vocabulary, every plotting door (verifier pass)
# ===========================================================================
#
# ``ts.plot(traj, color="red", title="Lorenz", theme="dark")`` was made to work
# by WP2, but its two SIBLING doors were left behind, and each failed
# differently:
#
#   traj.plot(color="red")  -> InvalidParameterError from ``tweak()``  ("color
#                              is not a spec tweak") — while ``title=`` worked,
#                              so half the same sentence landed and half did not
#   lor.plot(color="red")   -> the style word fell all the way through to the
#                              INTEGRATION and was reported as
#                              "color is not a valid integrate()/run() keyword",
#                              naming a vocabulary the caller was not speaking
#
# All three now peel the same ``STYLE_KEYS`` set (``viz.style.style_names()``),
# so "make it red and give it a title" is one sentence at every door.


def _style_call(subject, **kw):
    return subject.plot(**kw)


def test_style_keywords_land_identically_at_all_three_plot_doors():
    """``color`` / ``lw`` / ``title`` / ``theme`` mean the same thing everywhere."""
    lor = ts.systems.Lorenz()
    traj = lor.run(final_time=10.0, dt=0.02, ic=[1.0, 1.0, 1.0])

    front = viz.plot(traj, color="crimson", linewidth=0.6, title="Lorenz", theme="dark")
    method = traj.plot(color="crimson", linewidth=0.6, title="Lorenz", theme="dark")
    system = lor.plot(
        final_time=10.0,
        dt=0.02,
        ic=[1.0, 1.0, 1.0],
        color="crimson",
        lw=0.6,
        title="Lorenz",
        theme="dark",
    )

    for spec in (front, method, system):
        assert spec.layers[0].style["color"] == "crimson"
        assert spec.layers[0].style["linewidth"] == pytest.approx(0.6)
        assert spec.title == "Lorenz"
        assert spec.resolved_theme.name == "dark"


def test_an_integration_typo_is_still_reported_as_an_integration_typo():
    """Peeling style must not swallow a misspelt *integration* keyword."""
    with pytest.raises(InvalidParameterError, match="not a valid Lorenz.run\\(\\) keyword"):
        ts.systems.Lorenz().plot(final_tim=2.0)


# ---------------------------------------------------------------------------
# v6: the figure vocabulary, the grid, and the legend
# ---------------------------------------------------------------------------


#: ``keyword -> (value, how to read it back off the finished Plot)``.  One entry
#: per member of :data:`~tsdynamics.viz.spec.FIGURE_KEYS`, so the table cannot
#: drift from the vocabulary (a gate below asserts it covers all 17).
_FIGURE_CHECKS = {
    "title": ("T", lambda p: p.title),
    "xlabel": ("XL", lambda p: p.x.label),
    "ylabel": ("YL", lambda p: p.y.label),
    "zlabel": ("ZL", lambda p: p.z.label if p.z is not None else "ZL"),
    "xscale": ("log", lambda p: p.x.scale),
    "yscale": ("log", lambda p: p.y.scale),
    "zscale": ("log", lambda p: p.z.scale if p.z is not None else "log"),
    "xlim": ((-1.0, 1.0), lambda p: p.x.limits),
    "ylim": ((-1.0, 1.0), lambda p: p.y.limits),
    "zlim": ((-1.0, 1.0), lambda p: p.z.limits if p.z is not None else (-1.0, 1.0)),
    "xticks": ([0.0, 1.0], lambda p: p.x.ticks),
    "yticks": ([0.0, 1.0], lambda p: p.y.ticks),
    "zticks": ([0.0, 1.0], lambda p: p.z.ticks if p.z is not None else [0.0, 1.0]),
    "clim": ((0.0, 1.0), lambda p: p.clim),
    "colorbar": (True, lambda p: p.colorbar is not None),
    "legend": (True, lambda p: p.legend is not None),
    "theme": ("dark", lambda p: p.resolved_theme.name),
}


def test_the_figure_vocabulary_is_covered_by_this_gate():
    """The 17 names are derived; this table must not fall behind them."""
    from tsdynamics.viz.spec import FIGURE_KEYS

    assert set(_FIGURE_CHECKS) == set(FIGURE_KEYS)


@pytest.mark.parametrize("keyword", sorted(_FIGURE_CHECKS))
def test_every_figure_keyword_lands_at_every_plot_door(keyword):
    """**17 x 4 = 68 cells, 12 of which failed before v6** — all at ``ts.plot``.

    ``ts.plot(traj, xlim=(0, 1))`` answered ``kind='phase_portrait_3d' does not
    accept keyword(s) ['xlim']`` while ``traj.plot(xlim=(0, 1))`` worked, because
    the front door carried its own five-name copy of the vocabulary.  The value
    is read back **off the spec**, so "it did not raise" is not enough.
    """
    value, read = _FIGURE_CHECKS[keyword]
    lor = ts.systems.Lorenz()
    traj = lor.run(final_time=6.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    doors = {
        "ts.plot": lambda: ts.plot(traj, **{keyword: value}),
        "ts.plot+transform": lambda: ts.plot(traj, "phase_portrait", **{keyword: value}),
        "traj.plot": lambda: traj.plot(**{keyword: value}),
        "system.plot": lambda: lor.plot(
            final_time=6.0, dt=0.05, ic=[1.0, 1.0, 1.0], **{keyword: value}
        ),
    }
    for name, build in doors.items():
        spec = build()
        assert read(spec) == value, f"{keyword} did not land at {name}"


def test_a_grid_of_six_different_plots_is_one_call():
    """The owner's ask by name: a 2x3 of six *different* plots, styled per panel."""
    tr = _lorenz()
    vdp = ts.systems.VanDerPol()
    g = viz.grid(
        ts.plot(tr, "phase_portrait", components=("x", "y"), color="crimson", title="orbit"),
        ts.plot(tr, "time_series", components="x", title="x(t)"),
        ts.plot(tr, "psd", xscale="log", yscale="log", title="spectrum"),
        ts.plot(vdp.with_params(mu=0.5), "flow_speed", grid=24, title="mu=0.5"),
        ts.plot(vdp.with_params(mu=1.0), "flow_speed", grid=24, title="mu=1.0"),
        ts.plot(vdp.with_params(mu=2.0), "flow_speed", grid=24, title="mu=2.0"),
        rows=2,
        cols=3,
        title="six views",
    )
    assert g.kind is PlotKind.COMPOSITE
    assert (g.layout.rows, g.layout.cols) == (2, 3)
    assert len(g.panels) == 6
    assert [p.title for p in g.panels][:2] == ["orbit", "x(t)"]
    # per-panel styling lands in the render, because panels hold the same objects
    g["mu=2.0"].style(cmap="magma")
    assert g.panels[5].layers[0].style["cmap"] == "magma"
    assert g.panels[0].layers[0].style["color"] == "crimson"
    # ...and a grid is a Plot, so it nests
    assert viz.plot(g, ts.plot(tr, "psd"), layout="row").kind is PlotKind.COMPOSITE


def test_share_color_unifies_the_scale_and_draws_one_bar():
    """Measured as a **complete no-op** before v6: three incomparable scales, three bars."""
    vdp = ts.systems.VanDerPol()
    panels = [
        ts.plot(vdp.with_params(mu=mu), "flow_speed", grid=20, title=f"mu={mu}")
        for mu in (0.5, 1.0, 2.0)
    ]
    apart = viz.plot(*[p for p in panels], layout="row")
    assert len({p.clim for p in apart.panels}) == 3  # ...the defect, still there without it

    panels = [
        ts.plot(vdp.with_params(mu=mu), "flow_speed", grid=20, title=f"mu={mu}")
        for mu in (0.5, 1.0, 2.0)
    ]
    shared = viz.plot(*panels, layout="row", share_color=True)
    assert len({p.clim for p in shared.panels}) == 1
    assert sum(p.colorbar is not None for p in shared.panels) == 1
    fig = shared.fig
    assert sum(1 for ax in fig.axes if ax.get_label() == "<colorbar>") == 1
    import matplotlib.pyplot as plt

    plt.close("all")


def test_share_color_refuses_two_colour_meanings():
    """One colour scale needs one colour meaning; unifying two would be worse."""
    vdp = ts.systems.VanDerPol()
    speed = ts.plot(vdp, "flow_speed", grid=20)
    time_coloured = ts.plot(_lorenz(), "phase_portrait", components=("x", "y"), color_by="time")
    with pytest.raises(InvalidParameterError, match="one colour meaning"):
        viz.plot(speed, time_coloured, layout="row", share_color=True)


def test_every_orbit_in_an_overlay_gets_its_own_legend_entry():
    """A legend naming two different orbits identically is a **wrong answer**.

    Measured before v6: ``plot(t1, t2, t3)`` gave ``(1)/(2)/(3)`` but
    ``plot(t1).add(t2).add(t3)`` gave ``(1)/(2)/(2)`` — a duplicate.
    """
    a, b, c = _named_3d("run"), _named_3d("run"), _named_3d("run")
    one_shot = viz.plot(a, b, c)
    chained = viz.plot(a).add(b).add(c)
    for spec in (one_shot, chained):
        labels = [layer.label for layer in spec.layers]
        assert len(set(labels)) == len(labels) == 3, labels
