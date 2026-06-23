"""Spec-shape tests for the GAPFILL-B viz seam (lyapunov / chaos / embedding).

Engine-free: every result is built synthetically (tiny dummy arrays, no
``tsdynamics._rust`` import) and only the :class:`~tsdynamics.viz.spec.PlotSpec`
*shape* is asserted — the semantic kind, the layer marks, the annotations, and a
JSON round-trip.  These are the bespoke ``to_plot_spec`` figures the ticket asks
for; the whole-layer contract stays guarded by ``tests/test_viz_fake_renderer.py``.
"""

from __future__ import annotations

import numpy as np

from tsdynamics.analysis.chaos.zero_one import ZeroOneResult
from tsdynamics.analysis.embedding.delay import MutualInformation
from tsdynamics.analysis.embedding.dimension import EmbeddingDimension
from tsdynamics.analysis.embedding.embed import Embedding
from tsdynamics.analysis.lyapunov import LyapunovSpectrum
from tsdynamics.viz.spec import PlotKind, PlotSpec


def _roundtrips(spec: PlotSpec) -> None:
    """Assert the spec survives a JSON-friendly ``to_dict`` / ``from_dict`` cycle."""
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert len(rebuilt.layers) == len(spec.layers)
    for layer in spec.layers:
        assert isinstance(layer.kind, PlotKind)


# ---------------------------------------------------------------------------
# Lyapunov spectrum -> LYAPUNOV_SPECTRUM as BAR layers + zero line
# ---------------------------------------------------------------------------


def test_lyapunov_spectrum_is_bars_with_zero_line() -> None:
    """The spectrum plots one BAR per exponent, with a lambda = 0 reference line."""
    spec = LyapunovSpectrum(values=np.array([0.91, 0.0, -14.57])).to_plot_spec()
    assert spec.kind is PlotKind.LYAPUNOV_SPECTRUM
    assert len(spec.layers) == 1
    bar = spec.layers[0]
    assert bar.kind is PlotKind.BAR
    # one bar per exponent
    assert bar.data["y"].shape == (3,)
    assert bar.data["x"].shape == (3,)
    np.testing.assert_array_equal(bar.data["y"], np.array([0.91, 0.0, -14.57]))
    # the zero line is the sign separator between expanding and contracting dirs
    hlines = [a for a in spec.annotations if a.kind == "hline"]
    assert any(a.y == 0.0 for a in hlines)
    _roundtrips(spec)


def test_lyapunov_spectrum_kind_override() -> None:
    """A ``kind`` override is honoured (uniform signature, no new parameter)."""
    spec = LyapunovSpectrum(values=np.array([0.1, -0.2])).to_plot_spec(kind="diagnostic_curve")
    assert spec.kind is PlotKind.DIAGNOSTIC_CURVE
    assert spec.layers[0].kind is PlotKind.BAR


# ---------------------------------------------------------------------------
# Mutual information -> DIAGNOSTIC_CURVE with the chosen lag annotated (vline)
# ---------------------------------------------------------------------------


def test_mutual_information_diagnostic_marks_first_minimum() -> None:
    """The MI curve plots as a DIAGNOSTIC_CURVE with the first-minimum lag vline."""
    # first local minimum is at lag 3 (0.4 < 0.5, <= 0.6)
    curve = np.array([1.5, 0.9, 0.6, 0.4, 0.5, 0.7])
    mi = MutualInformation(values=curve)
    assert mi.optimal_lag == 3
    spec = mi.to_plot_spec()
    assert spec.kind is PlotKind.DIAGNOSTIC_CURVE
    assert len(spec.layers) == 1
    assert spec.layers[0].kind is PlotKind.LINE
    # the curve is drawn against the lag axis
    np.testing.assert_array_equal(spec.layers[0].data["y"], curve)
    np.testing.assert_array_equal(spec.layers[0].data["x"], np.arange(curve.size))
    vlines = [a for a in spec.annotations if a.kind == "vline"]
    assert len(vlines) == 1
    assert vlines[0].x == 3.0
    _roundtrips(spec)


def test_mutual_information_is_array_drop_in() -> None:
    """The result is still an ``ndarray`` drop-in (additive return-type refinement)."""
    curve = np.array([2.0, 1.0, 0.7, 0.8])
    mi = MutualInformation(values=curve)
    np.testing.assert_array_equal(np.asarray(mi), curve)
    assert mi[1] == 1.0
    # global-min fallback when there is no interior dip (monotone non-increasing)
    mono = MutualInformation(values=np.array([2.0, 1.0, 0.5, 0.2]))
    assert mono.optimal_lag == 3


# ---------------------------------------------------------------------------
# Embedding dimension -> DIAGNOSTIC_CURVE, selected dimension annotated (vline)
# ---------------------------------------------------------------------------


def test_embedding_dimension_fnn_curve_marks_selected_dim() -> None:
    """FNN: the false-neighbour fraction is a LINE with the selected-m vline."""
    spec = EmbeddingDimension(
        dimension=3,
        dims=np.array([1, 2, 3, 4]),
        method="fnn",
        delay=1,
        fnn_fraction=np.array([0.8, 0.4, 0.05, 0.04]),
    ).to_plot_spec()
    assert spec.kind is PlotKind.DIAGNOSTIC_CURVE
    assert len(spec.layers) == 1
    assert spec.layers[0].kind is PlotKind.LINE
    vlines = [a for a in spec.annotations if a.kind == "vline"]
    assert len(vlines) == 1 and vlines[0].x == 3.0
    _roundtrips(spec)


def test_embedding_dimension_cao_curves_e1_e2() -> None:
    """Cao: both E1 and E2 are drawn, with a saturation hline and the m vline."""
    spec = EmbeddingDimension(
        dimension=2,
        dims=np.array([1, 2, 3]),
        method="cao",
        delay=2,
        afn_e1=np.array([0.7, 0.95, 0.99]),
        afn_e2=np.array([0.9, 0.92, 0.95]),
    ).to_plot_spec()
    assert spec.kind is PlotKind.DIAGNOSTIC_CURVE
    assert len(spec.layers) == 2
    assert all(layer.kind is PlotKind.LINE for layer in spec.layers)
    assert spec.legend is not None  # two labelled curves warrant a legend
    assert any(a.kind == "hline" and a.y == 1.0 for a in spec.annotations)
    assert any(a.kind == "vline" and a.x == 2.0 for a in spec.annotations)
    _roundtrips(spec)


# ---------------------------------------------------------------------------
# Embedding point cloud -> phase portrait (2-D SCATTER / 3-D LINE3D)
# ---------------------------------------------------------------------------


def test_embedding_point_cloud_is_phase_portrait() -> None:
    """A 3-D embedding is a 3-D phase portrait; a 2-D embedding a 2-D one."""
    spec3 = Embedding(values=np.random.default_rng(0).random((20, 3))).to_plot_spec()
    assert spec3.kind is PlotKind.PHASE_PORTRAIT_3D
    assert spec3.ndim == 3
    assert spec3.layers[0].kind is PlotKind.LINE3D
    _roundtrips(spec3)

    spec2 = Embedding(values=np.random.default_rng(1).random((20, 2))).to_plot_spec()
    assert spec2.kind is PlotKind.PHASE_PORTRAIT_2D
    assert spec2.ndim == 2
    assert spec2.aspect == "equal"
    _roundtrips(spec2)


# ---------------------------------------------------------------------------
# 0-1 test translation plane (p, q) -> PHASE_PORTRAIT_2D
# ---------------------------------------------------------------------------


def test_zero_one_translation_plane_is_phase_portrait() -> None:
    """The (p, q) skew-translation plane plots as a 2-D phase portrait."""
    rng = np.random.default_rng(0)
    p = np.cumsum(rng.standard_normal(60))
    q = np.cumsum(rng.standard_normal(60))
    result = ZeroOneResult(value=0.96, p=p, q=q)
    # still a drop-in for K
    assert float(result) == 0.96
    assert result > 0.9
    spec = result.to_plot_spec()
    assert spec.kind is PlotKind.PHASE_PORTRAIT_2D
    assert spec.aspect == "equal"
    assert len(spec.layers) == 1
    layer = spec.layers[0]
    assert layer.kind is PlotKind.LINE
    np.testing.assert_array_equal(layer.data["x"], p)
    np.testing.assert_array_equal(layer.data["y"], q)
    _roundtrips(spec)


def test_zero_one_without_plane_falls_back_to_scalar_spec() -> None:
    """With no captured plane, the scalar fallback still yields a valid spec."""
    spec = ZeroOneResult(value=0.5).to_plot_spec()
    assert isinstance(spec, PlotSpec)
    assert isinstance(spec.kind, PlotKind)
    _roundtrips(spec)
