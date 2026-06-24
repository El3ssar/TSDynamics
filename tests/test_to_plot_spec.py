"""Tests for ``to_plot_spec()`` on Trajectory + every result type (stream WS-TOSPEC).

Covers the acceptance pillars:

1. :class:`~tsdynamics.data.Trajectory` auto-dispatches its
   :class:`~tsdynamics.viz.spec.PlotKind` on dimensionality, and honours an
   explicit ``kind`` override.
2. ``poincare_section`` / ``PoincareMap.trajectory`` carry
   ``POINCARE_SECTION`` intent (a ``meta["plot_kind"]`` the trajectory's spec
   reads), so a section is never mistaken for a flow.
3. ``OrbitDiagram / DimensionResult / RecurrenceMatrix / RQAResult /
   GALIResult / BasinsResult / ReturnMap / LyapunovFromData / SurrogateTest``
   each emit a correct :class:`~tsdynamics.viz.spec.PlotSpec`.
4. Every emitted spec round-trips through ``to_dict`` / ``from_dict``
   (the property the dossier asks a property test to assert).
5. Building any spec — and importing :mod:`tsdynamics` — pulls in no plotting
   library.

All builders are fast-tier: short integrations or synthetic constructions, no
slow compiles.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.analysis.basins.attractors import Attractor, AttractorSet
from tsdynamics.analysis.basins.basins import BasinsResult
from tsdynamics.data import Grid, Trajectory
from tsdynamics.viz.spec import PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# Shared builders (fast tier)
# ---------------------------------------------------------------------------


def _lorenz_traj(final_time: float = 30.0, dt: float = 0.02) -> Trajectory:
    """A short Lorenz trajectory with its transient dropped."""
    return ts.Lorenz().integrate(final_time=final_time, dt=dt).after(5.0)


def _result_builders() -> dict[str, object]:
    """One instance of each result type that must emit a spec."""
    traj = _lorenz_traj()
    rm = ts.recurrence_matrix(traj.y[:200], recurrence_rate=0.05)
    return {
        "OrbitDiagram": ts.orbit_diagram(
            ts.Logistic(), "r", np.linspace(2.8, 4.0, 50), n=40, transient=100
        ),
        "DimensionResult": ts.correlation_dimension(traj),
        "RecurrenceMatrix": rm,
        "RQAResult": ts.rqa(rm),
        "GALIResult": ts.gali(ts.Lorenz(), k=2, final_time=20.0, dt=0.05),
        "ReturnMap": ts.return_map(traj, component=2, method="max"),
        "LyapunovFromData": ts.lyapunov_from_data(traj.y[:1000, 0], dt=0.02),
        "SurrogateTest": ts.surrogate_test(traj.y[:500, 2], n=9, seed=0),
        "BasinsResult": _synthetic_basins(),
    }


def _synthetic_basins() -> BasinsResult:
    """A two-attractor 4x4 basin image built directly (no slow integration)."""
    a1 = Attractor(id=1, points=np.array([[-1.0, 0.0]]), cells=1)
    a2 = Attractor(id=2, points=np.array([[1.0, 0.0]]), cells=1)
    aset = AttractorSet(attractors={1: a1, 2: a2}, diverged=0, seeds=16)
    grid = Grid(lo=[-2.0, -2.0], hi=[2.0, 2.0], counts=(4, 4))
    labels = np.array([[1, 1, 2, 2]] * 4)
    return BasinsResult(labels=labels, grid=grid, attractors=aset)


# The set of result types the acceptance criteria enumerate.
_RESULT_NAMES = (
    "OrbitDiagram",
    "DimensionResult",
    "RecurrenceMatrix",
    "RQAResult",
    "GALIResult",
    "ReturnMap",
    "LyapunovFromData",
    "SurrogateTest",
    "BasinsResult",
)


def _assert_roundtrips(spec: PlotSpec) -> None:
    """A spec must survive ``to_dict`` / ``from_dict`` (kind, axes, layer data)."""
    assert isinstance(spec, PlotSpec)
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert rebuilt.ndim == spec.ndim
    assert rebuilt.aspect == spec.aspect
    assert len(rebuilt.layers) == len(spec.layers)
    for before, after in zip(spec.layers, rebuilt.layers, strict=True):
        assert after.kind == before.kind
        assert set(after.data) == set(before.data)
        for channel, arr in before.data.items():
            np.testing.assert_allclose(np.asarray(after.data[channel]), np.asarray(arr))


# ---------------------------------------------------------------------------
# Trajectory: dimensionality dispatch + override
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        (1, PlotKind.TIME_SERIES),
        (2, PlotKind.PHASE_PORTRAIT_2D),
        (3, PlotKind.PHASE_PORTRAIT_3D),
        # dim > 3 auto-dispatches to a spacetime image, NOT a misleading 3-D
        # portrait of the first three coordinates (front-door dispatch).
        (4, PlotKind.SPACETIME),
        (8, PlotKind.SPACETIME),
    ],
)
def test_trajectory_kind_dispatches_on_dimensionality(dim, expected):
    t = np.linspace(0.0, 1.0, 32)
    y = np.random.default_rng(0).standard_normal((32, dim))
    traj = Trajectory(t=t, y=y, system=None)
    spec = traj.to_plot_spec()
    assert spec.kind == expected
    # A 3-D phase portrait draws three coordinate channels.
    if expected == PlotKind.PHASE_PORTRAIT_3D:
        assert set(spec.layers[0].data) == {"x", "y", "z"}
        assert spec.z is not None
    # A spacetime image draws a single IMAGE layer of the whole field.
    if expected == PlotKind.SPACETIME:
        assert spec.layers[0].kind == PlotKind.IMAGE
    _assert_roundtrips(spec)


def test_trajectory_kind_override():
    traj = _lorenz_traj()
    assert traj.to_plot_spec().kind == PlotKind.PHASE_PORTRAIT_3D
    forced = traj.to_plot_spec(kind="time_series")
    assert forced.kind == PlotKind.TIME_SERIES
    assert forced.ndim == 1
    # x channel is time for a time series.
    np.testing.assert_allclose(forced.layers[0].data["x"], traj.t)
    _assert_roundtrips(forced)


def test_trajectory_time_series_uses_named_variable():
    traj = _lorenz_traj()
    spec = traj.to_plot_spec(kind="time_series")
    assert spec.x.label == "t"
    assert spec.y.label == "x"  # Lorenz declares variables = ("x", "y", "z")


# ---------------------------------------------------------------------------
# Front door: components= selection + per-kind options + plot() forwarding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("components", "expected"),
    [
        ("x", PlotKind.TIME_SERIES),
        (["x", "y"], PlotKind.PHASE_PORTRAIT_2D),
        (["x", "y", "z"], PlotKind.PHASE_PORTRAIT_3D),
        ([0, 2], PlotKind.PHASE_PORTRAIT_2D),
    ],
)
def test_components_selection_drives_dispatch(components, expected):
    """``components=`` selects channels and the auto kind keys off how many."""
    spec = _lorenz_traj().to_plot_spec(components=components)
    assert spec.kind == expected
    _assert_roundtrips(spec)


def test_components_single_name_is_time_series_of_that_channel():
    traj = _lorenz_traj()
    spec = traj.to_plot_spec(components="z")
    assert spec.kind == PlotKind.TIME_SERIES
    assert spec.y.label == "z"
    np.testing.assert_allclose(spec.layers[0].data["y"], traj.component("z"))


def test_unknown_component_raises():
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        _lorenz_traj().to_plot_spec(components="nope")


def test_high_dim_components_triple_is_3d_portrait():
    """Selecting three channels of a high-dim flow yields a 3-D portrait."""
    tr = ts.Lorenz96(N=8).trajectory(final_time=10.0, dt=0.1)
    spec = tr.to_plot_spec(components=["y0", "y1", "y2"])
    assert spec.kind == PlotKind.PHASE_PORTRAIT_3D
    _assert_roundtrips(spec)


def test_delay_kind_builds_2d_embedding_with_time_units_tau():
    """kind='delay' builds an x(t) vs x(t - tau) embedding; tau is in time units."""
    tr = ts.MackeyGlass().integrate(
        final_time=200.0, dt=0.2, history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)]
    )
    spec = tr.to_plot_spec(kind="delay", tau=17.0)  # 17 time units → 85 samples at dt=0.2
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D
    # 85 samples dropped off each end of the embedded series.
    assert spec.layers[0].data["x"].shape[0] == tr.n_steps - 85
    _assert_roundtrips(spec)


def test_delay_requires_tau():
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        _lorenz_traj().to_plot_spec(kind="delay")


def test_kind_kw_rejected_for_wrong_kind():
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        _lorenz_traj().to_plot_spec(kind="spacetime", tau=1.0)
    with pytest.raises(InvalidParameterError):
        _lorenz_traj().to_plot_spec(kind="phase_portrait_3d", transpose=True)


def test_spacetime_transpose_swaps_axes():
    tr = ts.Lorenz96(N=6).trajectory(final_time=8.0, dt=0.1)
    normal = tr.to_plot_spec(kind="spacetime")
    swapped = tr.to_plot_spec(kind="spacetime", transpose=True)
    assert (normal.x.label, normal.y.label) == ("t", "component")
    assert (swapped.x.label, swapped.y.label) == ("component", "t")


def test_plot_forwards_spec_shaping_kwargs(monkeypatch):
    """``plot()`` peels spec-shaping kwargs to to_plot_spec; the rest go to the backend."""
    captured: dict[str, object] = {}

    def fake_render(self, backend=None, **backend_kw):
        captured["kind"] = self.kind
        captured["backend_kw"] = backend_kw
        return self

    monkeypatch.setattr(PlotSpec, "render", fake_render)
    _lorenz_traj().plot(kind="time_series", components="x", figsize=(4, 3))
    assert captured["kind"] == PlotKind.TIME_SERIES
    assert captured["backend_kw"] == {"figsize": (4, 3)}


def test_system_to_plot_spec_splits_plot_and_integration_kwargs():
    """A system splits plot kwargs (components) from integration kwargs (final_time/dt)."""
    spec = ts.Lorenz().to_plot_spec(components="x", final_time=10.0, dt=0.05)
    assert spec.kind == PlotKind.TIME_SERIES
    assert spec.y.label == "x"


# ---------------------------------------------------------------------------
# Front door: input-validation guards (no silently-wrong specs)
# ---------------------------------------------------------------------------


def test_empty_components_selection_raises():
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        _lorenz_traj().to_plot_spec(components=[])


def test_delay_default_embeds_first_component_of_multidim():
    """With no components=, kind='delay' embeds the first component (not an error)."""
    spec = _lorenz_traj().to_plot_spec(kind="delay", tau=0.5)
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D


def test_delay_rejects_explicit_multiple_components():
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        _lorenz_traj().to_plot_spec(kind="delay", tau=0.5, components=["x", "y"])


@pytest.mark.parametrize("bad_tau", [0.0, -1.0, float("inf"), float("nan")])
def test_delay_rejects_nonpositive_or_nonfinite_tau(bad_tau):
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        _lorenz_traj().to_plot_spec(kind="delay", tau=bad_tau)


def test_delay_tau_uses_time_grid_when_meta_dt_absent():
    """Without meta['dt'], the time-unit→sample conversion falls back to the grid."""
    t = np.linspace(0.0, 10.0, 501)  # dt = 0.02
    y = np.sin(t)[:, None]
    traj = Trajectory(t=t, y=y, system=None)  # no meta["dt"]
    spec = traj.to_plot_spec(kind="delay", tau=0.2)  # 0.2 / 0.02 = 10 samples
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D
    assert spec.layers[0].data["x"].shape[0] == t.size - 10


# The full wrong-kind / wrong-option rejection matrix (drives the _KIND_KW table).
@pytest.mark.parametrize(
    ("kind", "bad_kw"),
    [
        ("time_series", {"tau": 1.0}),
        ("time_series", {"transpose": True}),
        ("phase_portrait_2d", {"tau": 1.0}),
        ("phase_portrait_2d", {"transpose": True}),
        ("phase_portrait_3d", {"transpose": True}),
        ("spacetime", {"tau": 1.0}),
        ("spacetime", {"color_by": "time"}),
        ("delay", {"color_by": "time"}),
        ("delay", {"transpose": True}),
    ],
)
def test_kind_kw_rejection_matrix(kind, bad_kw):
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        _lorenz_traj().to_plot_spec(kind=kind, **bad_kw)


# ---------------------------------------------------------------------------
# Front door: component selectors (indices, negatives, numpy ints, gen names)
# ---------------------------------------------------------------------------


def test_negative_and_numpy_int_component_selectors():
    traj = _lorenz_traj()
    spec = traj.to_plot_spec(components=[-1, np.int64(0)])  # z, x
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D
    assert (spec.x.label, spec.y.label) == ("z", "x")


def test_out_of_range_component_index_raises():
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        _lorenz_traj().to_plot_spec(components=[0, 7])


def test_generated_y_names_resolve_for_unnamed_high_dim():
    """An unnamed high-dim trajectory exposes y0… names usable in components=."""
    t = np.linspace(0.0, 1.0, 40)
    y = np.random.default_rng(0).standard_normal((40, 6))
    traj = Trajectory(t=t, y=y, system=None)  # no variables
    spec = traj.to_plot_spec(components=["y0", "y2"])
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D
    assert (spec.x.label, spec.y.label) == ("y0", "y2")


# ---------------------------------------------------------------------------
# Front door: color_by delegation + Poincaré short-circuit override
# ---------------------------------------------------------------------------


def test_color_by_time_attaches_colorbar_channel():
    spec = _lorenz_traj().to_plot_spec(kind="time_series", components="x", color_by="time")
    assert spec.kind == PlotKind.TIME_SERIES
    assert "c" in spec.layers[0].data
    assert spec.colorbar is not None


def test_poincare_short_circuit_is_overridden_by_components_or_kind():
    section = ts.poincare_section(ts.Rossler(), plane=(1, 0.0), n=80)
    # Default view honours the section intent…
    assert section.to_plot_spec().kind == PlotKind.POINCARE_SECTION
    # …but selecting components or forcing a kind opts out of the short-circuit.
    assert section.to_plot_spec(components=["x", "z"]).kind == PlotKind.PHASE_PORTRAIT_2D
    assert section.to_plot_spec(kind="time_series").kind == PlotKind.TIME_SERIES


def test_system_plot_forwards_delay_recipe(monkeypatch):
    """A system's plot()/to_plot_spec route the delay recipe + tau through the split."""
    spec = ts.Lorenz().to_plot_spec(kind="delay", tau=0.5, final_time=10.0, dt=0.05)
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D

    captured: dict[str, object] = {}

    def fake_render(self, backend=None, **backend_kw):
        captured["kind"] = self.kind
        return self

    monkeypatch.setattr(PlotSpec, "render", fake_render)
    ts.Lorenz().plot(kind="delay", tau=0.5, final_time=10.0, dt=0.05)
    assert captured["kind"] == PlotKind.PHASE_PORTRAIT_2D


def test_trajectory_plot_raises_without_backend(monkeypatch):
    # The matplotlib backend auto-registers on render as of stream VIZ-MPL-CORE;
    # force an empty registry to keep testing the genuine no-backend path.
    from tsdynamics import registry
    from tsdynamics.analysis._result import VisualizationNotInstalled
    from tsdynamics.viz import render as render_mod

    saved = registry.renderers.all()
    registry.renderers.clear()
    monkeypatch.setattr(render_mod, "register_builtin_renderers", lambda *a, **k: [])
    try:
        with pytest.raises(VisualizationNotInstalled):
            _lorenz_traj().plot()
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj, replace=True)


# ---------------------------------------------------------------------------
# Poincaré section intent
# ---------------------------------------------------------------------------


def test_poincare_section_carries_intent_from_system():
    section = ts.poincare_section(ts.Rossler(), plane=(1, 0.0), n=80)
    assert section.meta.get("plot_kind") == "poincare_section"
    spec = section.to_plot_spec()
    assert spec.kind == PlotKind.POINCARE_SECTION
    assert spec.ndim == 2
    assert spec.aspect == "equal"
    assert spec.layers[0].kind == PlotKind.SCATTER
    _assert_roundtrips(spec)


def test_poincare_map_trajectory_carries_intent():
    pmap = ts.PoincareMap(ts.Rossler(), plane=(1, 0.0))
    section = pmap.trajectory(80)
    assert section.meta.get("plot_kind") == "poincare_section"
    assert section.to_plot_spec().kind == PlotKind.POINCARE_SECTION


def test_poincare_section_from_data_carries_intent():
    traj = ts.Rossler().integrate(final_time=80.0, dt=0.02)
    section = ts.poincare_section(traj, plane=(1, 0.0))
    assert section.meta.get("plot_kind") == "poincare_section"
    assert section.to_plot_spec().kind == PlotKind.POINCARE_SECTION


def test_poincare_section_drops_the_normal_coordinate():
    # plane (1, 0.0) fixes component 1; the in-plane axes must be the other two.
    section = ts.poincare_section(ts.Rossler(), plane=(1, 0.0), n=80)
    i, j = section._section_axes()
    assert 1 not in (i, j)
    assert i != j


def test_ordinary_trajectory_has_no_section_intent():
    # A plain flow must not accidentally carry section intent.
    traj = ts.Lorenz().integrate(final_time=10.0, dt=0.05)
    assert "plot_kind" not in traj.meta
    assert traj.to_plot_spec().kind == PlotKind.PHASE_PORTRAIT_3D


# ---------------------------------------------------------------------------
# Every result type emits a spec that round-trips
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_results() -> dict[str, object]:
    return _result_builders()


@pytest.mark.parametrize("name", _RESULT_NAMES)
def test_result_emits_roundtrippable_spec(built_results, name):
    result = built_results[name]
    spec = result.to_plot_spec()
    assert isinstance(spec, PlotSpec)
    # The spec carries at least one drawable layer with real array data.
    assert spec.layers
    assert any(layer.data for layer in spec.layers)
    _assert_roundtrips(spec)


def test_result_specs_have_expected_kinds(built_results):
    expected = {
        "OrbitDiagram": PlotKind.ORBIT_DIAGRAM,
        "DimensionResult": PlotKind.SCALING_FIT,
        "RecurrenceMatrix": PlotKind.RECURRENCE_PLOT,
        # GAPFILL-F: the RQA scalar measures now read out as a categorical bar.
        "RQAResult": PlotKind.CATEGORICAL_BAR,
        "GALIResult": PlotKind.DIAGNOSTIC_CURVE,
        "ReturnMap": PlotKind.RETURN_MAP,
        "LyapunovFromData": PlotKind.SCALING_FIT,
        "SurrogateTest": PlotKind.HISTOGRAM_NULL,
        "BasinsResult": PlotKind.BASINS_IMAGE,
    }
    for name, kind in expected.items():
        assert built_results[name].to_plot_spec().kind == kind, name


def test_result_kind_override(built_results):
    # Every result accepts a kind override (used by the .plot.<kind>() seam).
    forced = built_results["DimensionResult"].to_plot_spec(kind="diagnostic_curve")
    assert forced.kind == PlotKind.DIAGNOSTIC_CURVE


# -- channel-data correctness on a few representative results ----------------


def test_dimension_spec_carries_the_loglog_curve(built_results):
    res = built_results["DimensionResult"]
    spec = res.to_plot_spec()
    scatter = spec.layers[0]
    np.testing.assert_allclose(scatter.data["x"], np.asarray(res.x, dtype=float))
    np.testing.assert_allclose(scatter.data["y"], np.asarray(res.y, dtype=float))
    # A fit line is drawn from intercept + slope * x over the fit endpoints.
    assert spec.layers[-1].kind == PlotKind.LINE


def test_recurrence_spec_is_a_sparse_recurrence_plot(built_results):
    # GAPFILL-F: the recurrence plot is a SPARSE (i, j) scatter of the recurrent
    # pairs — it is NEVER densified to an (N, N) image (anti-OOM at large N).
    res = built_results["RecurrenceMatrix"]
    spec = res.to_plot_spec()
    assert spec.kind == PlotKind.RECURRENCE_PLOT
    assert spec.aspect == "equal"
    layer = spec.layers[0]
    assert layer.kind == PlotKind.SCATTER
    nnz = res.matrix.tocoo().nnz
    assert layer.data["x"].shape == (nnz,)
    assert layer.data["y"].shape == (nnz,)
    # No layer carries a dense 2-D (densified) array.
    for lyr in spec.layers:
        for arr in lyr.data.values():
            assert arr.ndim == 1, "recurrence spec must not densify the matrix"


def test_gali_spec_uses_log_y(built_results):
    spec = built_results["GALIResult"].to_plot_spec()
    assert spec.y.scale == "log"


def test_return_map_spec_has_diagonal_reference(built_results):
    spec = built_results["ReturnMap"].to_plot_spec()
    kinds = [layer.kind for layer in spec.layers]
    assert PlotKind.SCATTER in kinds
    assert PlotKind.LINE in kinds  # the v_{n+1} = v_n diagonal


def test_surrogate_spec_marks_the_data_statistic(built_results):
    res = built_results["SurrogateTest"]
    spec = res.to_plot_spec()
    assert spec.layers[0].kind == PlotKind.HISTOGRAM
    assert spec.annotations
    annotation = spec.annotations[0]
    assert annotation.kind == "vline"
    assert annotation.x == pytest.approx(float(res.data_statistic))


def test_basins_spec_marks_attractor_centres(built_results):
    spec = built_results["BasinsResult"].to_plot_spec()
    assert spec.layers[0].kind == PlotKind.IMAGE
    assert any(layer.kind == PlotKind.MARKERS for layer in spec.layers)


# ---------------------------------------------------------------------------
# No plotting backend is ever imported
# ---------------------------------------------------------------------------


def test_building_specs_imports_no_plot_library():
    """Building every spec must not pull matplotlib / plotly into ``sys.modules``."""
    code = (
        "import sys; import numpy as np; import tsdynamics as ts;"
        "from tsdynamics.analysis.basins.attractors import Attractor, AttractorSet;"
        "from tsdynamics.analysis.basins.basins import BasinsResult;"
        "from tsdynamics.data import Grid;"
        "traj = ts.Lorenz().integrate(final_time=20.0, dt=0.05).after(5.0);"
        "traj.to_plot_spec(); traj.to_plot_spec(kind='time_series');"
        "ts.poincare_section(ts.Rossler(), plane=(1, 0.0), n=40).to_plot_spec();"
        "rm = ts.recurrence_matrix(traj.y[:150], recurrence_rate=0.05); rm.to_plot_spec();"
        "ts.rqa(rm).to_plot_spec();"
        "ts.correlation_dimension(traj).to_plot_spec();"
        "ts.gali(ts.Lorenz(), k=2, final_time=15.0, dt=0.05).to_plot_spec();"
        "ts.return_map(traj, component=2, method='max').to_plot_spec();"
        "ts.lyapunov_from_data(traj.y[:800, 0], dt=0.02).to_plot_spec();"
        "ts.surrogate_test(traj.y[:400, 2], n=9, seed=0).to_plot_spec();"
        "a = AttractorSet({1: Attractor(1, np.array([[0.0, 0.0]]), 1)}, 0, 1);"
        "BasinsResult(np.ones((4, 4), int), Grid([-1, -1], [1, 1], (4, 4)), a).to_plot_spec();"
        "bad = [m for m in sys.modules if m == 'matplotlib' or m.startswith('matplotlib.')"
        " or m == 'plotly' or m.startswith('plotly.')];"
        "assert not bad, bad; print('NO_PLOT_LIBS')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "NO_PLOT_LIBS" in out.stdout


def test_import_does_not_load_viz_package():
    """``import tsdynamics`` must not import the viz package at all.

    ``Trajectory`` provides ``to_plot_spec``/``.plot`` but imports
    :mod:`tsdynamics.viz` lazily, so plain ``import tsdynamics`` never runs the
    package's renderer-backend discovery (which would eagerly load an installed
    matplotlib/plotly backend and break the no-backend-on-import contract).
    """
    code = (
        "import sys, tsdynamics;"
        "assert 'tsdynamics.viz' not in sys.modules, "
        "sorted(m for m in sys.modules if m.startswith('tsdynamics.viz'));"
        "print('VIZ_NOT_LOADED')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "VIZ_NOT_LOADED" in out.stdout


# ---------------------------------------------------------------------------
# Phase-portrait kind override is schema-consistent
# ---------------------------------------------------------------------------


def test_phase_portrait_2d_override_on_3d_trajectory():
    """Forcing 2-D on a 3-D trajectory yields a consistent 2-D schema (no z)."""
    traj = _lorenz_traj()
    spec = traj.to_plot_spec(kind="phase_portrait_2d")
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D
    assert spec.ndim == 2
    assert spec.z is None
    assert set(spec.layers[0].data) == {"x", "y"}  # no stray z channel
    assert spec.layers[0].kind == PlotKind.LINE  # not LINE3D
    _assert_roundtrips(spec)


def test_phase_portrait_3d_override_on_2d_trajectory_raises():
    """Forcing a 3-D portrait on a 2-D trajectory is rejected, not mis-built."""
    from tsdynamics.errors import InvalidParameterError

    t = np.linspace(0.0, 1.0, 16)
    traj = Trajectory(t=t, y=np.random.default_rng(0).standard_normal((16, 2)), system=None)
    with pytest.raises(InvalidParameterError):
        traj.to_plot_spec(kind="phase_portrait_3d")


# ---------------------------------------------------------------------------
# An empty Poincaré section still builds a (degenerate) spec
# ---------------------------------------------------------------------------


def test_empty_poincare_section_builds_spec():
    """A section with no crossings emits an empty POINCARE_SECTION spec, not a crash.

    The plane may miss the sampled trajectory entirely; the resulting ``(0, dim)``
    state array must not raise in the spread reduction that picks display axes.
    """
    traj = Trajectory(
        t=np.empty(0),
        y=np.empty((0, 3)),
        system=None,
        meta={"plot_kind": "poincare_section", "plane": (1, 0.0)},
    )
    spec = traj.to_plot_spec()
    assert spec.kind == PlotKind.POINCARE_SECTION
    assert spec.ndim == 2
    assert spec.layers[0].data["x"].shape == (0,)
    _assert_roundtrips(spec)
