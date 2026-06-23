"""Engine-free spec-shape tests for the GAPFILL-A draw-views.

Stream GAPFILL-A: the enriched :meth:`tsdynamics.data.Trajectory.to_plot_spec`
(discrete-map scatter, multi-component overlay time series, spacetime image) plus
the parameterised pure spec builders in :mod:`tsdynamics.viz.producers`
(arbitrary component triple, colour-by-time / -speed, DDE delay-embedding, vector
fields, cobweb).  Every test builds tiny synthetic arrays / a tiny synthetic
Trajectory — no engine, no integration — and asserts the semantic kind, the layer
marks, the channels carried, and a ``to_dict`` / ``from_dict`` round-trip.

These are additive to the whole-result fake-renderer contract gate
(``tests/test_viz_fake_renderer.py``), which still passes unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.data import Trajectory
from tsdynamics.viz import producers
from tsdynamics.viz.spec import PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# Tiny synthetic trajectory builders (no system needed)
# ---------------------------------------------------------------------------


class _FakeSystem:
    """A minimal stand-in carrying ``is_discrete`` / ``variables``."""

    def __init__(self, *, is_discrete: bool, variables: tuple[str, ...] | None) -> None:
        self.is_discrete = is_discrete
        self.variables = variables


def _flow_traj(dim: int = 3, n: int = 40, variables: tuple[str, ...] | None = None) -> Trajectory:
    t = np.linspace(0.0, 1.0, n)
    y = np.stack([np.sin((k + 1) * t) for k in range(dim)], axis=1)
    sysv = variables if variables is not None else tuple("xyzuvw"[:dim])
    return Trajectory(t, y, _FakeSystem(is_discrete=False, variables=sysv), meta={"system": "Flow"})


def _map_traj(dim: int = 2, n: int = 30) -> Trajectory:
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(0)
    y = rng.standard_normal((n, dim))
    sysv = tuple("xyzuvw"[:dim])
    return Trajectory(t, y, _FakeSystem(is_discrete=True, variables=sysv), meta={"system": "Map"})


def _roundtrips(spec: PlotSpec) -> None:
    """Assert the spec is a real PlotSpec and survives a to_dict/from_dict cycle."""
    assert isinstance(spec, PlotSpec)
    assert PlotKind.is_semantic(spec.kind)
    for lyr in spec.layers:
        assert PlotKind.is_mark(lyr.kind)
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert len(rebuilt.layers) == len(spec.layers)
    for a, b in zip(spec.layers, rebuilt.layers, strict=True):
        assert a.kind == b.kind
        assert set(a.data) == set(b.data)


# ---------------------------------------------------------------------------
# to_plot_spec enrichments
# ---------------------------------------------------------------------------


def test_map_orbit_is_scatter_not_line():
    """A discrete-map orbit auto-dispatches to a SCATTER mark, not a LINE."""
    spec = _map_traj(dim=2).to_plot_spec()
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D
    assert [lyr.kind for lyr in spec.layers] == [PlotKind.SCATTER]
    _roundtrips(spec)


def test_flow_phase_portrait_is_line():
    """A flow phase portrait stays a connected LINE / LINE3D."""
    spec2 = _flow_traj(dim=2).to_plot_spec()
    assert spec2.layers[0].kind == PlotKind.LINE
    spec3 = _flow_traj(dim=3).to_plot_spec()
    assert spec3.kind == PlotKind.PHASE_PORTRAIT_3D
    assert spec3.layers[0].kind == PlotKind.LINE3D
    _roundtrips(spec2)
    _roundtrips(spec3)


def test_map_1d_orbit_time_series_scatter():
    """A 1-D discrete orbit is a TIME_SERIES of SCATTER points."""
    t = np.arange(20, dtype=float)
    y = np.cos(t)[:, None]
    traj = Trajectory(t, y, _FakeSystem(is_discrete=True, variables=("x",)))
    spec = traj.to_plot_spec()
    assert spec.kind == PlotKind.TIME_SERIES
    assert spec.layers[0].kind == PlotKind.SCATTER
    _roundtrips(spec)


def test_forced_time_series_overlays_components_with_legend():
    """Forcing kind='time_series' on a 3-D flow overlays one LINE per component."""
    spec = _flow_traj(dim=3, variables=("x", "y", "z")).to_plot_spec(kind="time_series")
    assert spec.kind == PlotKind.TIME_SERIES
    assert len(spec.layers) == 3
    assert all(lyr.kind == PlotKind.LINE for lyr in spec.layers)
    assert [lyr.label for lyr in spec.layers] == ["x", "y", "z"]
    assert spec.legend is not None and spec.legend.show
    _roundtrips(spec)


def test_spacetime_branch_is_image():
    """kind='spacetime' images component index vs time as a single IMAGE."""
    spec = _flow_traj(dim=6).to_plot_spec(kind="spacetime")
    assert spec.kind == PlotKind.SPACETIME
    assert spec.layers[0].kind == PlotKind.IMAGE
    assert spec.colorbar is not None
    assert spec.clim is not None
    _roundtrips(spec)


# ---------------------------------------------------------------------------
# producers: time_series
# ---------------------------------------------------------------------------


def test_producer_time_series_color_by_time():
    """time_series(color_by='time') attaches a 'c' channel + colorbar."""
    spec = producers.time_series(_flow_traj(dim=2), color_by="time")
    assert spec.kind == PlotKind.TIME_SERIES
    assert all("c" in lyr.data for lyr in spec.layers)
    assert spec.colorbar is not None
    _roundtrips(spec)


def test_producer_time_series_selected_components():
    """time_series(components=...) selects + names the chosen components."""
    spec = producers.time_series(_flow_traj(dim=3, variables=("x", "y", "z")), components=["z", 0])
    assert [lyr.label for lyr in spec.layers] == ["z", "x"]
    _roundtrips(spec)


def test_producer_time_series_color_by_speed():
    """color_by='speed' yields a finite per-point speed channel."""
    spec = producers.time_series(_flow_traj(dim=2), color_by="speed")
    c = spec.layers[0].data["c"]
    assert c.shape[0] == _flow_traj(dim=2).n_steps
    assert np.all(np.isfinite(c))
    _roundtrips(spec)


# ---------------------------------------------------------------------------
# producers: phase_portrait (arbitrary triple, not first three)
# ---------------------------------------------------------------------------


def test_producer_phase_portrait_arbitrary_triple():
    """phase_portrait selects an arbitrary triple — not the first three axes."""
    traj = _flow_traj(dim=5, variables=("a", "b", "c", "d", "e"))
    spec = producers.phase_portrait(traj, components=["e", "c", "a"])
    assert spec.kind == PlotKind.PHASE_PORTRAIT_3D
    assert spec.ndim == 3
    assert (spec.x.label, spec.y.label, spec.z.label) == ("e", "c", "a")
    assert spec.layers[0].kind == PlotKind.LINE3D
    np.testing.assert_allclose(spec.layers[0].data["x"], traj.y[:, 4])
    np.testing.assert_allclose(spec.layers[0].data["z"], traj.y[:, 0])
    _roundtrips(spec)


def test_producer_phase_portrait_pair_2d():
    """A two-component selection yields a PHASE_PORTRAIT_2D."""
    spec = producers.phase_portrait(_flow_traj(dim=4), components=[1, 3])
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D
    assert "z" not in spec.layers[0].data
    _roundtrips(spec)


def test_producer_phase_portrait_discrete_scatters():
    """A discrete-map phase portrait scatters its points."""
    spec = producers.phase_portrait(_map_traj(dim=2))
    assert spec.layers[0].kind == PlotKind.SCATTER
    _roundtrips(spec)


def test_producer_phase_portrait_bad_component_count():
    """Selecting one or four components is rejected."""
    with pytest.raises(ValueError):
        producers.phase_portrait(_flow_traj(dim=4), components=[0])
    with pytest.raises(ValueError):
        producers.phase_portrait(_flow_traj(dim=4), components=[0, 1, 2, 3])


# ---------------------------------------------------------------------------
# producers: delay_embedding
# ---------------------------------------------------------------------------


def test_producer_delay_embedding_from_series():
    """delay_embedding builds x(t) vs x(t-tau) from a scalar series."""
    x = np.sin(np.linspace(0, 10, 200))
    spec = producers.delay_embedding(x, tau=5, label="m")
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D
    lyr = spec.layers[0]
    assert lyr.data["x"].shape[0] == x.shape[0] - 5
    np.testing.assert_allclose(lyr.data["x"], x[:-5])
    np.testing.assert_allclose(lyr.data["y"], x[5:])
    assert spec.x.label == "m(t)" and spec.y.label == "m(t - 5)"
    _roundtrips(spec)


def test_producer_delay_embedding_from_trajectory_component():
    """delay_embedding can read a named component of a trajectory."""
    traj = _flow_traj(dim=3, variables=("x", "y", "z"))
    spec = producers.delay_embedding(traj, tau=3, component="y")
    np.testing.assert_allclose(spec.layers[0].data["x"], traj.y[:-3, 1])
    _roundtrips(spec)


def test_producer_delay_embedding_validates_tau():
    """tau must be >= 1 and shorter than the series."""
    x = np.arange(10.0)
    with pytest.raises(ValueError):
        producers.delay_embedding(x, tau=0)
    with pytest.raises(ValueError):
        producers.delay_embedding(x, tau=10)


# ---------------------------------------------------------------------------
# producers: vector_field / phase_portrait_field
# ---------------------------------------------------------------------------


def _spiral(u: np.ndarray) -> np.ndarray:
    return np.array([-u[1], u[0]])


def test_producer_vector_field_quiver():
    """vector_field samples a QUIVER grid carrying x/y/u/v channels."""
    spec = producers.vector_field(_spiral, xlim=(-1, 1), ylim=(-1, 1), grid=8)
    assert spec.kind == PlotKind.VECTOR_FIELD
    lyr = spec.layers[0]
    assert lyr.kind == PlotKind.QUIVER
    assert set(("x", "y", "u", "v")) <= set(lyr.data)
    assert lyr.data["x"].shape[0] == 8 * 8
    _roundtrips(spec)


def test_producer_vector_field_normalize():
    """normalize=True yields unit-magnitude arrows."""
    spec = producers.vector_field(_spiral, xlim=(-1, 1), ylim=(-1, 1), grid=6, normalize=True)
    u = spec.layers[0].data["u"]
    v = spec.layers[0].data["v"]
    np.testing.assert_allclose(np.hypot(u, v), 1.0, atol=1e-9)


def test_producer_phase_portrait_field_over_trajectory():
    """phase_portrait_field draws the field first, the trajectory on top."""
    traj = _flow_traj(dim=2, variables=("x", "y"))
    spec = producers.phase_portrait_field(_spiral, traj, grid=6)
    assert spec.kind == PlotKind.PHASE_PORTRAIT_FIELD
    assert [lyr.kind for lyr in spec.layers] == [PlotKind.QUIVER, PlotKind.LINE]
    # limits taken from the trajectory extent
    assert spec.x.limits is not None
    _roundtrips(spec)


def test_producer_phase_portrait_field_no_trajectory():
    """Without a trajectory the field stands alone."""
    spec = producers.phase_portrait_field(_spiral, grid=5, xlim=(-2, 2), ylim=(-2, 2))
    assert [lyr.kind for lyr in spec.layers] == [PlotKind.QUIVER]
    _roundtrips(spec)


# ---------------------------------------------------------------------------
# producers: cobweb
# ---------------------------------------------------------------------------


def test_producer_cobweb_staircase_and_diagonal():
    """cobweb emits the staircase plus the y=x diagonal."""
    x = np.array([0.1, 0.36, 0.92, 0.29, 0.82])
    spec = producers.cobweb(x)
    assert spec.kind == PlotKind.COBWEB
    assert len(spec.layers) == 2
    diag = spec.layers[0]
    assert diag.label == "y = x"
    np.testing.assert_allclose(diag.data["x"], diag.data["y"])
    stair = spec.layers[1]
    # first staircase vertex sits on the diagonal at x0
    np.testing.assert_allclose(stair.data["x"][0], 0.1)
    np.testing.assert_allclose(stair.data["y"][0], 0.1)
    # second vertex climbs to x1
    np.testing.assert_allclose(stair.data["y"][1], 0.36)
    _roundtrips(spec)


def test_producer_cobweb_needs_two_points():
    """A single-point orbit cannot make a staircase."""
    with pytest.raises(ValueError):
        producers.cobweb(np.array([0.5]))


# ---------------------------------------------------------------------------
# producers: spacetime
# ---------------------------------------------------------------------------


def test_producer_spacetime_image_and_transpose():
    """spacetime images the field; transpose swaps the axes."""
    traj = _flow_traj(dim=8, n=25)
    spec = producers.spacetime(traj)
    assert spec.kind == PlotKind.SPACETIME
    assert spec.layers[0].kind == PlotKind.IMAGE
    assert spec.layers[0].data["z"].shape == (8, 25)
    spec_t = producers.spacetime(traj, transpose=True)
    assert spec_t.layers[0].data["z"].shape == (25, 8)
    _roundtrips(spec)
    _roundtrips(spec_t)
