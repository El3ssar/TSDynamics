"""GAPFILL-C: fractal-dimension ``to_plot_spec`` viz adapters.

Covers the dimension half of the GAPFILL-C ticket (the entropy bullets left with
the estimators themselves when the library narrowed to phase-space methods):

1. a :math:`D_q` spectrum (``dimension_spectrum`` → ``dict[float, DimensionResult]``)
   renders as a ``DIMENSION_SPECTRUM`` with a ``LINE`` of :math:`D_q` vs :math:`q`
   plus an ``ERRORBAR`` whose ``"err"`` channel carries the per-order standard
   error;
2. the scalar fractal-dimension estimators keep their ``SCALING_FIT`` wrapper
   spec (via :class:`DimensionResult` / :class:`ScalingResult`).

Every produced spec must carry a real :class:`PlotKind`, real layer marks, and
round-trip losslessly through ``to_dict`` / ``from_dict`` (the same contract the
fake-renderer gate enforces).  Engine-free, fast tier.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsdynamics.analysis.dimensions import (
    correlation_dimension,
    dimension_spectrum,
    dimension_spectrum_plot_spec,
)
from tsdynamics.analysis.dimensions._common import DimensionResult
from tsdynamics.analysis.dimensions.generalized import NonMonotoneSpectrumWarning
from tsdynamics.viz.spec import PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_roundtrips(spec: PlotSpec) -> None:
    """Spec is a valid PlotSpec with real marks and round-trips through dict form."""
    assert isinstance(spec, PlotSpec)
    assert isinstance(spec.kind, PlotKind)
    for layer in spec.layers:
        assert isinstance(layer.kind, PlotKind)
        for channel, arr in layer.data.items():
            assert isinstance(arr, np.ndarray), f"channel {channel!r} not an array"
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert len(rebuilt.layers) == len(spec.layers)


def _cantor_like_points(n: int = 600, seed: int = 0) -> np.ndarray:
    """A small 2-D point cloud (a noisy line) for a quick, real dimension spectrum."""
    rng = np.random.default_rng(seed)
    t = rng.random(n)
    return np.column_stack([t, 0.5 * t + 0.02 * rng.standard_normal(n)])


# ---------------------------------------------------------------------------
# 1. DimensionSpectrum -> DIMENSION_SPECTRUM (LINE + ERRORBAR via "err")
# ---------------------------------------------------------------------------


def _synthetic_spectrum() -> dict[float, DimensionResult]:
    """Hand-built ``{q: DimensionResult}`` (no estimator run) for the viz unit."""
    x = np.linspace(0.0, 1.0, 8)
    spectrum: dict[float, DimensionResult] = {}
    for q, d in [(0.0, 2.1), (1.0, 2.0), (2.0, 1.9)]:
        spectrum[q] = DimensionResult(
            estimate=d,
            stderr=0.01 * (q + 1),
            kind="generalized",
            abscissa=x,
            ordinate=d * x,
            fit_region=(1, 6),
            intercept=0.0,
            q=q,
        )
    return spectrum


def test_dimension_spectrum_spec_kind_and_layers() -> None:
    spec = dimension_spectrum_plot_spec(_synthetic_spectrum())
    assert spec.kind is PlotKind.DIMENSION_SPECTRUM
    marks = [layer.kind for layer in spec.layers]
    assert PlotKind.LINE in marks
    assert PlotKind.ERRORBAR in marks
    _assert_roundtrips(spec)


def test_dimension_spectrum_spec_errorbar_carries_err_channel() -> None:
    spectrum = _synthetic_spectrum()
    spec = dimension_spectrum_plot_spec(spectrum)
    (errorbar,) = [layer for layer in spec.layers if layer.kind is PlotKind.ERRORBAR]
    assert "err" in errorbar.data
    # q ascending: 0, 1, 2 → D = 2.1, 2.0, 1.9 ; err = 0.01, 0.02, 0.03
    np.testing.assert_allclose(errorbar.data["x"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(errorbar.data["y"], [2.1, 2.0, 1.9])
    np.testing.assert_allclose(errorbar.data["err"], [0.01, 0.02, 0.03])


def test_dimension_spectrum_spec_line_is_dq_vs_q() -> None:
    spec = dimension_spectrum_plot_spec(_synthetic_spectrum())
    (line,) = [layer for layer in spec.layers if layer.kind is PlotKind.LINE]
    np.testing.assert_allclose(line.data["x"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(line.data["y"], [2.1, 2.0, 1.9])


def test_dimension_spectrum_spec_kind_override() -> None:
    spec = dimension_spectrum_plot_spec(_synthetic_spectrum(), kind="scaling_fit")
    assert spec.kind is PlotKind.SCALING_FIT


def test_dimension_spectrum_spec_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        dimension_spectrum_plot_spec({})


def test_dimension_spectrum_spec_from_real_estimator() -> None:
    # Structural test: the point cloud is small, so box counting need not resolve
    # D_0 and may warn that the spectrum is unresolved.  What is under test is the
    # plot spec built from the result, not the dimension values.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NonMonotoneSpectrumWarning)
        spectrum = dimension_spectrum(_cantor_like_points(), qs=[0.0, 1.0, 2.0])
    assert set(spectrum) == {0.0, 1.0, 2.0}
    spec = dimension_spectrum_plot_spec(spectrum)
    assert spec.kind is PlotKind.DIMENSION_SPECTRUM
    _assert_roundtrips(spec)


# ---------------------------------------------------------------------------
# 2. scalar fractal-dimension estimators keep their SCALING_FIT wrapper spec
# ---------------------------------------------------------------------------


def test_dimension_result_to_plot_spec_is_scaling_fit() -> None:
    x = np.linspace(0.0, 1.0, 12)
    result = DimensionResult(
        estimate=2.05,
        stderr=0.03,
        kind="correlation",
        abscissa=x,
        ordinate=-2.0 * x + 0.3,
        fit_region=(2, 9),
        intercept=0.3,
        q=2.0,
    )
    spec = result.to_plot_spec()
    assert spec.kind is PlotKind.SCALING_FIT
    _assert_roundtrips(spec)


def test_correlation_dimension_result_keeps_scaling_fit() -> None:
    result = correlation_dimension(_cantor_like_points())
    spec = result.to_plot_spec()
    assert spec.kind is PlotKind.SCALING_FIT
    _assert_roundtrips(spec)
