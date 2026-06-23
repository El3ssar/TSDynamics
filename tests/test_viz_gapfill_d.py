"""GAPFILL-D: fixed points / orbits / sections ``to_plot_spec`` coverage.

Engine-free, synthetic-data tests for the visualization specs added by stream
GAPFILL-D:

- :class:`FixedPoint` / :class:`FixedPointSet` → ``FIXED_POINTS_OVERLAY`` (stable
  vs unstable styled apart) plus an ``EIGENVALUE_PLANE`` (unit circle for maps /
  imaginary axis for flows), overlaying host-first via ``overlay_on``.
- :class:`PeriodicOrbit` → a phase-portrait spec plus a Floquet
  ``EIGENVALUE_PLANE`` marking the trivial ``μ ≈ 1`` multiplier distinctly.
- :class:`OrbitDiagram` → bifurcation onsets carried as ``"vline"`` annotations.
- :class:`ReturnMap` → a ``COBWEB`` staircase in addition to its scatter.
- ``period_diagnostic`` → a ``DIAGNOSTIC_CURVE`` of the autocorrelation / FFT.

Every spec is asserted to carry a real :class:`PlotKind`, real layer marks, and
to round-trip losslessly through ``to_dict`` / ``from_dict`` — the same contract
the whole-layer gate (``tests/test_viz_fake_renderer.py``) enforces.
"""

from __future__ import annotations

import numpy as np

from tsdynamics.analysis.fixedpoints.fixed import FixedPoint, FixedPointSet
from tsdynamics.analysis.fixedpoints.periodic import (
    PeriodicOrbit,
    estimate_period,
    period_diagnostic,
)
from tsdynamics.analysis.orbits.orbit_diagram import OrbitDiagram
from tsdynamics.analysis.orbits.return_map import ReturnMap
from tsdynamics.viz.spec import PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _assert_valid_spec(spec: object) -> PlotSpec:
    """Assert ``spec`` is a valid, round-tripping :class:`PlotSpec`."""
    assert isinstance(spec, PlotSpec)
    assert isinstance(spec.kind, PlotKind)
    for layer in spec.layers:
        assert isinstance(layer.kind, PlotKind)
        for arr in layer.data.values():
            assert isinstance(arr, np.ndarray)
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert len(rebuilt.layers) == len(spec.layers)
    assert len(rebuilt.annotations) == len(spec.annotations)
    return spec


def _fixed_point(stable: bool = True, continuous: bool = False) -> FixedPoint:
    eig = np.array([0.4 + 0.0j, 0.9 + 0.0j]) if not continuous else np.array([-1.0 + 2j, -1.0 - 2j])
    return FixedPoint(
        x=np.array([0.5, -0.3]), eigenvalues=eig, stable=stable, continuous=continuous
    )


def _periodic_orbit(dim: int = 3, continuous: bool = True) -> PeriodicOrbit:
    pts = np.array([[0.1 * k, 0.2 * k, 0.3 * k][:dim] for k in range(5)], dtype=float)
    mult = np.array([1.0 + 0.0j, 0.3 + 0.1j, 0.3 - 0.1j]) if continuous else np.array([0.5, 1.5])
    return PeriodicOrbit(
        points=pts,
        period=6.6 if continuous else 4,
        multipliers=mult,
        stable=False,
        continuous=continuous,
        residual=1e-9,
    )


# ---------------------------------------------------------------------------
# FixedPoint / FixedPointSet
# ---------------------------------------------------------------------------


def test_fixed_point_overlay_kind_and_style() -> None:
    """A FixedPoint plots as a FIXED_POINTS_OVERLAY scatter, styled by stability."""
    stable_spec = _assert_valid_spec(_fixed_point(stable=True).to_plot_spec())
    unstable_spec = _assert_valid_spec(_fixed_point(stable=False).to_plot_spec())
    assert stable_spec.kind == PlotKind.FIXED_POINTS_OVERLAY
    assert stable_spec.layers[0].kind == PlotKind.SCATTER
    # stable vs unstable are styled differently (filled vs open marker)
    assert stable_spec.layers[0].style.get("filled") is True
    assert unstable_spec.layers[0].style.get("filled") is False


def test_fixed_point_eigenvalue_plane_map_has_unit_circle() -> None:
    """A map fixed point's EIGENVALUE_PLANE draws the unit circle reference."""
    spec = _assert_valid_spec(_fixed_point(continuous=False).eigenvalue_plane())
    assert spec.kind == PlotKind.EIGENVALUE_PLANE
    # the unit circle is a closed LINE of radius 1
    circle = next(lyr for lyr in spec.layers if lyr.kind == PlotKind.LINE)
    r = np.hypot(circle.data["x"], circle.data["y"])
    assert np.allclose(r, 1.0)
    # no imaginary-axis vline for a map
    assert not any(a.kind == "vline" for a in spec.annotations)


def test_fixed_point_eigenvalue_plane_flow_has_imag_axis() -> None:
    """A flow equilibrium's EIGENVALUE_PLANE marks the imaginary axis (Re λ = 0)."""
    spec = _assert_valid_spec(_fixed_point(continuous=True).eigenvalue_plane())
    assert spec.kind == PlotKind.EIGENVALUE_PLANE
    vlines = [a for a in spec.annotations if a.kind == "vline"]
    assert vlines and vlines[0].x == 0.0
    # no unit circle for a flow
    assert not any(
        lyr.kind == PlotKind.LINE and np.allclose(np.hypot(lyr.data["x"], lyr.data["y"]), 1.0)
        for lyr in spec.layers
    )


def test_fixed_point_set_overlay_splits_stable_unstable() -> None:
    """A FixedPointSet draws stable and unstable points as separate layers."""
    fps = FixedPointSet(items=(_fixed_point(stable=True), _fixed_point(stable=False)))
    spec = _assert_valid_spec(fps.to_plot_spec())
    assert spec.kind == PlotKind.FIXED_POINTS_OVERLAY
    labels = {lyr.label for lyr in spec.layers}
    assert labels == {"stable", "unstable"}
    assert spec.legend is not None  # two layers → a legend


def test_fixed_point_set_eigenvalue_plane_pools_spectra() -> None:
    """The set's EIGENVALUE_PLANE pools every member's eigenvalues."""
    fps = FixedPointSet(items=(_fixed_point(stable=True), _fixed_point(stable=False)))
    spec = _assert_valid_spec(fps.eigenvalue_plane())
    assert spec.kind == PlotKind.EIGENVALUE_PLANE
    # 4 eigenvalues total (2 points × 2 each) land in a scatter layer
    scat = next(lyr for lyr in spec.layers if lyr.kind == PlotKind.SCATTER)
    assert scat.data["x"].size == 4


def test_fixed_point_overlay_on_host_keeps_host_first() -> None:
    """overlay_on draws the host phase portrait first, then the fixed point."""
    from tsdynamics.viz.spec import Axis, Layer

    host = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        ndim=2,
        x=Axis(label="x"),
        y=Axis(label="y"),
        layers=[Layer(PlotKind.LINE, {"x": np.zeros(3), "y": np.zeros(3)}, label="trajectory")],
    )
    merged = _fixed_point().overlay_on(host)
    assert merged is host
    assert merged.layers[0].label == "trajectory"  # host layer is first
    assert merged.layers[-1].kind == PlotKind.SCATTER  # fixed point on top


def test_fixed_point_set_empty_is_valid() -> None:
    """An empty FixedPointSet still yields a valid (layer-less) overlay + plane."""
    fps = FixedPointSet(items=())
    overlay = _assert_valid_spec(fps.to_plot_spec())
    plane = _assert_valid_spec(fps.eigenvalue_plane())
    assert overlay.kind == PlotKind.FIXED_POINTS_OVERLAY
    assert plane.kind == PlotKind.EIGENVALUE_PLANE


# ---------------------------------------------------------------------------
# PeriodicOrbit
# ---------------------------------------------------------------------------


def test_periodic_orbit_phase_portrait_3d() -> None:
    """A 3-D flow cycle plots as a PHASE_PORTRAIT_3D loop."""
    spec = _assert_valid_spec(_periodic_orbit(dim=3, continuous=True).to_plot_spec())
    assert spec.kind == PlotKind.PHASE_PORTRAIT_3D
    assert spec.ndim == 3
    assert any(lyr.kind == PlotKind.LINE3D for lyr in spec.layers)


def test_periodic_orbit_floquet_plane_marks_trivial_multiplier() -> None:
    """A flow cycle's Floquet EIGENVALUE_PLANE splits the trivial μ ≈ 1 marker."""
    spec = _assert_valid_spec(_periodic_orbit(continuous=True).eigenvalue_plane())
    assert spec.kind == PlotKind.EIGENVALUE_PLANE
    trivial = [lyr for lyr in spec.layers if lyr.label and "trivial" in lyr.label]
    assert trivial, "the trivial multiplier should be its own labelled layer"
    # the trivial marker sits at μ ≈ 1
    assert np.isclose(trivial[0].data["x"][0], 1.0, atol=1e-9)
    # the multiplier convention draws the unit circle
    assert any(
        lyr.kind == PlotKind.LINE and np.allclose(np.hypot(lyr.data["x"], lyr.data["y"]), 1.0)
        for lyr in spec.layers
    )


def test_periodic_orbit_map_plane_no_trivial_marker() -> None:
    """A map cycle's multiplier plane has no trivial-multiplier carve-out."""
    spec = _assert_valid_spec(_periodic_orbit(dim=2, continuous=False).eigenvalue_plane())
    assert spec.kind == PlotKind.EIGENVALUE_PLANE
    assert not any(lyr.label and "trivial" in lyr.label for lyr in spec.layers)


# ---------------------------------------------------------------------------
# OrbitDiagram bifurcation annotations
# ---------------------------------------------------------------------------


def _orbit_diagram_with_doubling() -> OrbitDiagram:
    """Synthetic orbit diagram: period 1 then period 2 (one bifurcation)."""
    values = np.linspace(2.8, 3.4, 7)
    points: list[np.ndarray] = []
    for v in values:
        if v < 3.1:
            points.append(np.array([[0.5], [0.5], [0.5]]))  # period 1
        else:
            points.append(np.array([[0.3], [0.7], [0.3], [0.7]]))  # period 2
    return OrbitDiagram(param="r", values=values, points=points, components=(0,))


def test_orbit_diagram_carries_bifurcation_vlines() -> None:
    """OrbitDiagram.to_plot_spec annotates detected bifurcation onsets as vlines."""
    od = _orbit_diagram_with_doubling()
    spec = _assert_valid_spec(od.to_plot_spec())
    assert spec.kind == PlotKind.ORBIT_DIAGRAM
    vlines = [a for a in spec.annotations if a.kind == "vline"]
    assert vlines, "the period-1 → period-2 onset should be annotated"
    onsets = od.bifurcation_points()
    assert np.isclose(vlines[0].x, float(onsets[0]))


def test_orbit_diagram_single_value_has_no_annotations() -> None:
    """A one-value sweep yields a valid spec with no bifurcation annotations."""
    od = OrbitDiagram(
        param="r", values=np.array([3.2]), points=[np.array([[0.5]])], components=(0,)
    )
    spec = _assert_valid_spec(od.to_plot_spec())
    assert spec.annotations == []


# ---------------------------------------------------------------------------
# ReturnMap cobweb
# ---------------------------------------------------------------------------


def _return_map() -> ReturnMap:
    return ReturnMap(
        current=np.array([0.1, 0.2, 0.3]),
        successor=np.array([0.2, 0.3, 0.1]),
        values=np.array([0.1, 0.2, 0.3, 0.1]),
        times=np.arange(4.0),
    )


def test_return_map_scatter_still_works() -> None:
    """The default return-map spec stays a RETURN_MAP scatter + diagonal."""
    spec = _assert_valid_spec(_return_map().to_plot_spec())
    assert spec.kind == PlotKind.RETURN_MAP
    assert spec.layers[0].kind == PlotKind.SCATTER


def test_return_map_cobweb_is_a_staircase() -> None:
    """ReturnMap.cobweb adds the COBWEB staircase alongside the scatter + diagonal."""
    rm = _return_map()
    spec = _assert_valid_spec(rm.cobweb())
    assert spec.kind == PlotKind.COBWEB
    kinds = [lyr.kind for lyr in spec.layers]
    assert PlotKind.SCATTER in kinds  # the return points
    assert kinds.count(PlotKind.LINE) >= 2  # diagonal + the staircase
    stair = next(lyr for lyr in spec.layers if lyr.label == "cobweb")
    # the staircase alternates vertical then horizontal segments: 3 vertices/pair
    assert stair.data["x"].size == 3 * rm.current.size


def test_return_map_empty_cobweb_is_valid() -> None:
    """An empty return map yields a valid (scatter-only) cobweb spec."""
    rm = ReturnMap(
        current=np.empty(0),
        successor=np.empty(0),
        values=np.empty(0),
        times=np.empty(0),
    )
    spec = _assert_valid_spec(rm.cobweb())
    assert spec.kind == PlotKind.COBWEB


# ---------------------------------------------------------------------------
# estimate_period diagnostic curve
# ---------------------------------------------------------------------------


def _sine(period: float = 20.0, n: int = 400) -> np.ndarray:
    t = np.arange(n, dtype=float)
    return np.sin(2.0 * np.pi * t / period)


def test_estimate_period_carries_curve_in_meta() -> None:
    """estimate_period stays a ScalarResult but stashes the diagnostic curve."""
    res = estimate_period(_sine(period=20.0))
    assert np.isclose(float(res), 20.0, atol=1.0)
    assert "curve_abscissa" in res.meta
    assert "curve_ordinate" in res.meta
    assert np.asarray(res.meta["curve_abscissa"]).size > 0


def test_period_diagnostic_autocorrelation_curve() -> None:
    """period_diagnostic renders a DIAGNOSTIC_CURVE of the autocorrelation."""
    spec = _assert_valid_spec(period_diagnostic(_sine(period=20.0)))
    assert spec.kind == PlotKind.DIAGNOSTIC_CURVE
    assert spec.layers[0].kind == PlotKind.LINE
    vlines = [a for a in spec.annotations if a.kind == "vline"]
    assert vlines  # the detected period is marked
    assert np.isclose(vlines[0].x, float(estimate_period(_sine(period=20.0))), atol=1.0)


def test_period_diagnostic_fft_marks_frequency() -> None:
    """The FFT diagnostic marks the period as a frequency 1/T on the spectrum."""
    spec = _assert_valid_spec(period_diagnostic(_sine(period=20.0), method="fft"))
    assert spec.kind == PlotKind.DIAGNOSTIC_CURVE
    assert spec.x.label == "frequency"
    vlines = [a for a in spec.annotations if a.kind == "vline"]
    assert vlines and np.isclose(vlines[0].x, 1.0 / 20.0, atol=0.01)
