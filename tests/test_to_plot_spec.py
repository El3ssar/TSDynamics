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
        "ReturnMap": ts.return_map(traj, observable=2, kind="max"),
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
        (5, PlotKind.PHASE_PORTRAIT_3D),
    ],
)
def test_trajectory_kind_dispatches_on_dimensionality(dim, expected):
    t = np.linspace(0.0, 1.0, 32)
    y = np.random.default_rng(0).standard_normal((32, dim))
    traj = Trajectory(t=t, y=y, system=None)
    spec = traj.to_plot_spec()
    assert spec.kind == expected
    # A 3-D (or higher) phase portrait draws three coordinate channels.
    if expected == PlotKind.PHASE_PORTRAIT_3D:
        assert set(spec.layers[0].data) == {"x", "y", "z"}
        assert spec.z is not None
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


def test_trajectory_plot_raises_without_backend():
    from tsdynamics.analysis._result import VisualizationNotInstalled

    with pytest.raises(VisualizationNotInstalled):
        _lorenz_traj().plot()


# ---------------------------------------------------------------------------
# Poincaré section intent
# ---------------------------------------------------------------------------


def test_poincare_section_carries_intent_from_system():
    section = ts.poincare_section(ts.Rossler(), plane=(1, 0.0), steps=80)
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
    section = ts.poincare_section(ts.Rossler(), plane=(1, 0.0), steps=80)
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
        "RQAResult": PlotKind.DIAGNOSTIC_CURVE,
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


def test_recurrence_spec_is_a_square_image(built_results):
    res = built_results["RecurrenceMatrix"]
    spec = res.to_plot_spec()
    assert spec.aspect == "equal"
    image = spec.layers[0]
    assert image.kind == PlotKind.IMAGE
    assert image.data["c"].shape == (res.size, res.size)


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
        "ts.poincare_section(ts.Rossler(), plane=(1, 0.0), steps=40).to_plot_spec();"
        "rm = ts.recurrence_matrix(traj.y[:150], recurrence_rate=0.05); rm.to_plot_spec();"
        "ts.rqa(rm).to_plot_spec();"
        "ts.correlation_dimension(traj).to_plot_spec();"
        "ts.gali(ts.Lorenz(), k=2, final_time=15.0, dt=0.05).to_plot_spec();"
        "ts.return_map(traj, observable=2, kind='max').to_plot_spec();"
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
