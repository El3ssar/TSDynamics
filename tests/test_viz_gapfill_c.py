"""GAPFILL-C: dimensions + entropy ``to_plot_spec`` viz adapters.

Covers the four acceptance bullets of the GAPFILL-C ticket:

1. a :math:`D_q` spectrum (``dimension_spectrum`` → ``dict[float, DimensionResult]``)
   renders as a ``DIMENSION_SPECTRUM`` with a ``LINE`` of :math:`D_q` vs :math:`q`
   plus an ``ERRORBAR`` whose ``"err"`` channel carries the per-order standard
   error;
2. a multiscale-entropy profile renders as a ``COMPLEXITY_CURVE`` (entropy vs
   scale factor);
3. an entropy outcome distribution (e.g. ordinal-pattern probabilities) renders
   as a ``CATEGORICAL_BAR`` — a ``BAR`` layer over a categorical axis whose
   ``categories`` are the pattern labels;
4. the scalar fractal-dimension estimators keep their ``SCALING_FIT`` wrapper
   spec (via :class:`DimensionResult` / :class:`ScalingResult`).

Every produced spec must carry a real :class:`PlotKind`, real layer marks, and
round-trip losslessly through ``to_dict`` / ``from_dict`` (the same contract the
fake-renderer gate enforces).  Engine-free, fast tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.analysis.dimensions import (
    correlation_dimension,
    dimension_spectrum,
    dimension_spectrum_plot_spec,
)
from tsdynamics.analysis.dimensions._common import DimensionResult
from tsdynamics.analysis.entropy import (
    OrdinalPatterns,
    multiscale_entropy,
    multiscale_entropy_plot_spec,
    outcome_distribution_plot_spec,
    probabilities,
)
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
    spectrum = dimension_spectrum(_cantor_like_points(), qs=[0.0, 1.0, 2.0])
    assert set(spectrum) == {0.0, 1.0, 2.0}
    spec = dimension_spectrum_plot_spec(spectrum)
    assert spec.kind is PlotKind.DIMENSION_SPECTRUM
    _assert_roundtrips(spec)


# ---------------------------------------------------------------------------
# 2. MultiscaleEntropy -> COMPLEXITY_CURVE (entropy vs scale)
# ---------------------------------------------------------------------------


def test_multiscale_entropy_spec_kind_and_axes() -> None:
    rng = np.random.default_rng(0)
    mse = multiscale_entropy(rng.standard_normal(2000), scales=5)
    spec = multiscale_entropy_plot_spec(mse)
    assert spec.kind is PlotKind.COMPLEXITY_CURVE
    assert PlotKind.LINE in [layer.kind for layer in spec.layers]
    _assert_roundtrips(spec)


def test_multiscale_entropy_spec_uses_scale_factors_from_meta() -> None:
    rng = np.random.default_rng(1)
    mse = multiscale_entropy(rng.standard_normal(2000), scales=4)
    spec = multiscale_entropy_plot_spec(mse)
    line = next(layer for layer in spec.layers if layer.kind is PlotKind.LINE)
    # scales= 4 → factors 1, 2, 3, 4 read from meta["scales"]
    np.testing.assert_allclose(line.data["x"], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(line.data["y"], np.asarray(mse.values, dtype=float))


def test_multiscale_entropy_spec_kind_override() -> None:
    rng = np.random.default_rng(2)
    mse = multiscale_entropy(rng.standard_normal(1500), scales=3)
    spec = multiscale_entropy_plot_spec(mse, kind="diagnostic_curve")
    assert spec.kind is PlotKind.DIAGNOSTIC_CURVE


def test_multiscale_entropy_spec_falls_back_when_no_scales_meta() -> None:
    from tsdynamics.analysis._result import ArrayResult

    bare = ArrayResult(values=np.array([1.0, 0.8, 0.6]))
    spec = multiscale_entropy_plot_spec(bare)
    line = next(layer for layer in spec.layers if layer.kind is PlotKind.LINE)
    np.testing.assert_allclose(line.data["x"], [1.0, 2.0, 3.0])


def test_multiscale_entropy_spec_empty_raises() -> None:
    from tsdynamics.analysis._result import ArrayResult

    with pytest.raises(ValueError, match="empty"):
        multiscale_entropy_plot_spec(ArrayResult(values=np.empty(0)))


# ---------------------------------------------------------------------------
# 3. outcome distribution -> CATEGORICAL_BAR (BAR + categorical Axis.categories)
# ---------------------------------------------------------------------------


def test_outcome_distribution_spec_kind_and_bar_layer() -> None:
    p = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    spec = outcome_distribution_plot_spec(p, outcomes=OrdinalPatterns(3))
    assert spec.kind is PlotKind.CATEGORICAL_BAR
    (bar,) = spec.layers
    assert bar.kind is PlotKind.BAR
    assert "y" in bar.data and "cat" in bar.data
    np.testing.assert_allclose(bar.data["y"], p)
    np.testing.assert_allclose(bar.data["cat"], np.arange(p.size))


def test_outcome_distribution_spec_has_categorical_axis_with_labels() -> None:
    space = OrdinalPatterns(3)
    p = np.full(space.cardinality, 1.0 / space.cardinality)
    spec = outcome_distribution_plot_spec(p, outcomes=space)
    assert spec.x.scale == "categorical"
    assert spec.x.categories is not None
    assert len(spec.x.categories) == space.cardinality
    # Ordinal labels are rank tuples rendered compactly; m=3 → 3! = 6 patterns.
    assert set(spec.x.categories) == {"012", "021", "102", "120", "201", "210"}
    _assert_roundtrips(spec)


def test_outcome_distribution_spec_explicit_labels() -> None:
    p = np.array([0.7, 0.3])
    spec = outcome_distribution_plot_spec(p, labels=["heads", "tails"])
    assert list(spec.x.categories) == ["heads", "tails"]


def test_outcome_distribution_spec_default_index_labels() -> None:
    p = np.array([0.25, 0.25, 0.5])
    spec = outcome_distribution_plot_spec(p)
    assert list(spec.x.categories) == ["0", "1", "2"]


def test_outcome_distribution_spec_label_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="labels length"):
        outcome_distribution_plot_spec(np.array([0.5, 0.5]), labels=["only-one"])


def test_outcome_distribution_spec_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        outcome_distribution_plot_spec(np.empty(0))


def test_outcome_distribution_spec_from_real_probabilities() -> None:
    rng = np.random.default_rng(0)
    space = OrdinalPatterns(3)
    p = probabilities(rng.random(3000), space)
    np.testing.assert_allclose(p.sum(), 1.0)
    spec = outcome_distribution_plot_spec(p, outcomes=space)
    assert spec.kind is PlotKind.CATEGORICAL_BAR
    _assert_roundtrips(spec)


def test_ordinal_pattern_labels_dense_index_order() -> None:
    space = OrdinalPatterns(3)
    labels = space.labels()
    assert len(labels) == space.cardinality
    # The increasing pattern (rank tuple (0, 1, 2)) sits at dense index 0.
    assert labels[0] == "012"


# ---------------------------------------------------------------------------
# 4. scalar fractal-dimension estimators keep their SCALING_FIT wrapper spec
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
