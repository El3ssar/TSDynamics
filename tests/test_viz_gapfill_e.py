"""Derived-wrapper ``to_plot_spec`` contract (stream GAPFILL-E).

Every derived wrapper that can describe itself produces a valid, JSON-round-trip
:class:`~tsdynamics.viz.spec.PlotSpec`:

- :class:`~tsdynamics.derived.StroboscopicMap` — a **scatter** of sampled
  states (a discrete strobing, not a connected flow line).
- :class:`~tsdynamics.derived.EnsembleSystem` — a *static* ``ENSEMBLE_FAN``
  (median line + a shaded percentile band via the ``"lo"`` / ``"hi"`` channels of
  an ``AREA`` layer, with ``lo <= hi``).  No animation.
- :class:`~tsdynamics.derived.TangentSystem` — a ``DIAGNOSTIC_CURVE`` of each
  exponent's running Lyapunov estimate against time (a labelled line family).
- :class:`~tsdynamics.derived.PoincareMap` / :class:`~tsdynamics.derived.ProjectedSystem`
  — delegate to the lens-specific trajectory's spec.

These wrappers run the compiled engine to collect their data, so the module
imports it (``importorskip``) and is auto-tagged ``engine`` by the suite's marker
hook; it skips cleanly where the extension is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tsdynamics._rust")

import tsdynamics as ts
import tsdynamics.systems as systems
from tsdynamics.viz.spec import PlotKind, PlotSpec


def _round_trips(spec: PlotSpec) -> None:
    """Assert ``spec`` is a real spec that round-trips through ``to_dict``/``from_dict``."""
    assert isinstance(spec, PlotSpec)
    assert isinstance(spec.kind, PlotKind)
    for layer in spec.layers:
        assert isinstance(layer.kind, PlotKind)
        for channel, arr in layer.data.items():
            assert isinstance(arr, np.ndarray), f"channel {channel!r} not an array"
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert len(rebuilt.layers) == len(spec.layers)


# ---------------------------------------------------------------------------
# StroboscopicMap — discrete sampling is a SCATTER, not a LINE
# ---------------------------------------------------------------------------


def test_stroboscopic_spec_is_a_scatter() -> None:
    """A strobe sampling renders as scattered sampled points, never a flow line."""
    smap = ts.derived.StroboscopicMap(systems.ForcedVanDerPol(), period=2 * np.pi / 0.63)
    spec = smap.__plot_spec__(steps=40)
    assert spec.layers, "expected at least one layer"
    for layer in spec.layers:
        assert layer.kind is PlotKind.SCATTER, "strobe samples must be a SCATTER mark"
    # No connected LINE / LINE3D anywhere — the whole point of the strobe view.
    assert not any(layer.kind in (PlotKind.LINE, PlotKind.LINE3D) for layer in spec.layers)
    _round_trips(spec)


def test_stroboscopic_kind_override() -> None:
    """An explicit ``kind`` overrides the dimensionality dispatch but stays a scatter."""
    smap = ts.derived.StroboscopicMap(systems.ForcedVanDerPol(), period=2 * np.pi / 0.63)
    spec = smap.__plot_spec__(kind="phase_portrait_2d", steps=30)
    assert spec.kind is PlotKind.PHASE_PORTRAIT_2D
    assert all(layer.kind is PlotKind.SCATTER for layer in spec.layers)
    _round_trips(spec)


# ---------------------------------------------------------------------------
# EnsembleSystem — a STATIC fan (median + percentile band, lo <= hi)
# ---------------------------------------------------------------------------


def _lorenz_ensemble(m: int = 8) -> ts.derived.EnsembleSystem:
    rng = np.random.default_rng(0)
    states = np.array([1.0, 1.0, 20.0]) + 0.5 * rng.standard_normal((m, 3))
    return ts.derived.EnsembleSystem(systems.Lorenz(), states)


def test_ensemble_fan_is_static_with_band() -> None:
    """The ensemble spec is an ENSEMBLE_FAN: median line + a shaded ``lo<=hi`` band."""
    ens = _lorenz_ensemble()
    spec = ens.__plot_spec__(steps=60, components=0, band=90.0)
    assert spec.kind is PlotKind.ENSEMBLE_FAN

    marks = [layer.kind for layer in spec.layers]
    assert PlotKind.AREA in marks, "the band must be an AREA layer"
    assert PlotKind.LINE in marks, "the median must be a LINE layer"

    # The band is carried by the lo/hi channels and lo <= hi must hold everywhere.
    area = next(layer for layer in spec.layers if layer.kind is PlotKind.AREA)
    assert "lo" in area.data and "hi" in area.data
    lo, hi = area.data["lo"], area.data["hi"]
    assert lo.shape == hi.shape
    assert np.all(lo <= hi), "fan band requires lo <= hi"

    # Static — no animation frame axis anywhere.
    assert "animate" not in spec.meta
    for layer in spec.layers:
        assert "frame" not in layer.data
    _round_trips(spec)


def test_ensemble_band_widens_with_more_mass() -> None:
    """A wider percentile mass shades a wider band (sanity that lo/hi track ``band``)."""
    ens = _lorenz_ensemble(m=16)
    narrow = ens.__plot_spec__(steps=50, band=50.0)
    ens2 = _lorenz_ensemble(m=16)
    wide = ens2.__plot_spec__(steps=50, band=98.0)

    def _mean_width(spec: PlotSpec) -> float:
        area = next(layer for layer in spec.layers if layer.kind is PlotKind.AREA)
        return float(np.mean(area.data["hi"] - area.data["lo"]))

    assert _mean_width(wide) >= _mean_width(narrow)


def test_ensemble_rejects_bad_component_and_band() -> None:
    """Out-of-range ``component`` / ``band`` raise rather than draw nonsense."""
    ens = _lorenz_ensemble()
    with pytest.raises(ValueError):
        ens.__plot_spec__(components=99)
    with pytest.raises(ValueError):
        ens.__plot_spec__(band=0.0)
    with pytest.raises(ValueError):
        ens.__plot_spec__(band=150.0)


def test_ensemble_collect_shapes() -> None:
    """The collector stacks (steps, members, dim) samples."""
    ens = _lorenz_ensemble(m=6)
    times, states = ens.collect(20)
    assert times.shape == (20,)
    assert states.shape == (20, 6, 3)


# ---------------------------------------------------------------------------
# TangentSystem — running-Lyapunov convergence is a DIAGNOSTIC_CURVE
# ---------------------------------------------------------------------------


def test_tangent_convergence_is_a_diagnostic_curve() -> None:
    """The tangent spec is a DIAGNOSTIC_CURVE with one labelled line per exponent."""
    tang = ts.derived.TangentSystem(systems.Henon(), k=2)
    spec = tang.__plot_spec__(steps=400)
    assert spec.kind is PlotKind.DIAGNOSTIC_CURVE
    # One line per exponent, each legended (the line family).
    assert len(spec.layers) == 2
    for layer in spec.layers:
        assert layer.kind is PlotKind.LINE
        assert layer.label
        assert layer.data["x"].shape == layer.data["y"].shape
    assert spec.legend is not None
    _round_trips(spec)


def test_tangent_convergence_records_running_estimate() -> None:
    """The recorded estimates settle toward the spectrum (last row ≈ final exponents)."""
    tang = ts.derived.TangentSystem(systems.Henon(), k=2)
    times, estimates = tang.convergence(steps=500, ic=[0.1, 0.1])
    assert times.shape == (500,)
    assert estimates.shape == (500, 2)
    # Largest Hénon exponent is positive (≈ 0.42); the running estimate's leading
    # column is positive by the end of a converged run.
    assert estimates[-1, 0] > 0.0
    # Descending QR order: λ1 >= λ2 in the converged estimate.
    assert estimates[-1, 0] >= estimates[-1, 1]


def test_tangent_ode_convergence_curve() -> None:
    """An ODE tangent system also produces a valid convergence curve (engine path)."""
    tang = ts.derived.TangentSystem(systems.Lorenz(), k=2)
    spec = tang.__plot_spec__(steps=40, n_or_dt=0.1)
    assert spec.kind is PlotKind.DIAGNOSTIC_CURVE
    assert len(spec.layers) == 2
    _round_trips(spec)


# ---------------------------------------------------------------------------
# PoincareMap / ProjectedSystem — delegate to the lens trajectory
# ---------------------------------------------------------------------------


def test_poincare_spec_delegates_to_section() -> None:
    """A PoincareMap describes itself as its section scatter (POINCARE_SECTION)."""
    pmap = ts.derived.PoincareMap(systems.Rossler(), plane=("y", 0.0, "up"))
    spec = pmap.__plot_spec__()
    assert spec.kind is PlotKind.POINCARE_SECTION
    assert all(layer.kind is PlotKind.SCATTER for layer in spec.layers)
    _round_trips(spec)


def test_projected_spec_uses_projected_columns() -> None:
    """A ProjectedSystem describes the projected view (2-D phase portrait here)."""
    proj = ts.derived.ProjectedSystem(systems.Lorenz(), ["x", "z"])
    spec = proj.__plot_spec__()
    assert spec.kind is PlotKind.PHASE_PORTRAIT_2D
    _round_trips(spec)
