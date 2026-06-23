"""Spec contract for the attractors / basins result types (stream GAPFILL-G).

Each basin-layer result builds a backend-agnostic
:class:`~tsdynamics.viz.spec.PlotSpec` of the *right semantic kind*, with real
layer marks and a lossless JSON round-trip:

- :class:`AttractorSet` → a ``PHASE_PORTRAIT_2D`` scatter of the attractor points,
  coloured from the shared ``tab20`` palette (a fixed colour for the diverged set
  recorded in ``meta``);
- :class:`BasinFractions` → a ``CATEGORICAL_BAR`` (one bar per attractor id over a
  categorical x-axis);
- :class:`ContinuationResult` → a ``CONTINUATION`` of stacked basin-fraction bands
  with vertical tipping annotations where a basin annihilates;
- :class:`UncertaintyExponent` → a ``SCALING_FIT`` of ``log f`` vs ``log eps``;
- :class:`BasinsResult` → a ``BASINS_IMAGE`` (verified), including a 3-D slice.

The keystone invariant is **intra-result palette consistency**: the *same*
attractor id maps to the *same* swatch in the :class:`AttractorSet` scatter and
the :class:`BasinsResult` image (and the other basin views).  Engine-free by
design — every result is built synthetically.
"""

from __future__ import annotations

import numpy as np

from tsdynamics.analysis.basins.attractors import Attractor, AttractorSet
from tsdynamics.analysis.basins.basins import BasinFractions, BasinsResult
from tsdynamics.analysis.basins.continuation import ContinuationResult
from tsdynamics.analysis.basins.metrics import UncertaintyExponent
from tsdynamics.data import Grid
from tsdynamics.viz.spec import PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# Synthetic builders (tiny dummy data — no engine)
# ---------------------------------------------------------------------------


def _attractor(aid: int) -> Attractor:
    pts = np.array([[0.1 * aid, 0.2 * aid], [0.11 * aid, 0.19 * aid], [0.1 * aid, 0.2 * aid]])
    return Attractor(id=aid, points=pts, cells=3)


def _attractor_set(ids: tuple[int, ...] = (1, 2, 3)) -> AttractorSet:
    return AttractorSet(attractors={aid: _attractor(aid) for aid in ids}, diverged=1, seeds=10)


def _basins(slice3d: bool = False) -> BasinsResult:
    labels = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1, -1, 2, 2], [3, 3, 2, 2]], dtype=int)
    if slice3d:
        labels = labels.reshape(4, 1, 4)
        grid = Grid(lo=np.array([-1.0, -1.0, -1.0]), hi=np.array([1.0, 1.0, 1.0]), counts=(4, 1, 4))
    else:
        grid = Grid(lo=np.array([-1.0, -1.0]), hi=np.array([1.0, 1.0]), counts=(4, 4))
    return BasinsResult(labels=labels, attractors=_attractor_set(), grid=grid)


def _continuation() -> ContinuationResult:
    values = np.linspace(0.0, 1.0, 5)
    return ContinuationResult(
        param="r",
        values=values,
        fractions={
            1: np.array([0.6, 0.6, 0.4, 0.2, 0.0]),  # annihilates at the last value
            2: np.array([0.4, 0.4, 0.6, 0.8, 1.0]),
        },
        attractors=[{} for _ in values],
        diverged=np.zeros(values.size),
    )


def _uncertainty() -> UncertaintyExponent:
    return UncertaintyExponent(
        alpha=0.4,
        boundary_dimension=1.6,
        state_dimension=2,
        epsilons=np.array([0.1, 0.05, 0.025]),
        f=np.array([0.4, 0.3, 0.2]),
        r_squared=0.99,
    )


def _roundtrips(spec: PlotSpec) -> None:
    """Assert ``spec`` round-trips losslessly through the JSON dict form."""
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert len(rebuilt.layers) == len(spec.layers)
    for layer in spec.layers:
        assert isinstance(layer.kind, PlotKind)
        for channel, arr in layer.data.items():
            assert isinstance(arr, np.ndarray), f"channel {channel!r} not an array"


# ---------------------------------------------------------------------------
# AttractorSet -> SCATTER
# ---------------------------------------------------------------------------


def test_attractor_set_is_a_scatter_with_palette() -> None:
    spec = _attractor_set().to_plot_spec()
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D
    assert [layer.kind for layer in spec.layers] == [PlotKind.SCATTER]
    layer = spec.layers[0]
    # one point cloud per attractor, coloured by the per-point category swatch.
    assert "cat" in layer.data
    assert layer.data["x"].size == layer.data["cat"].size > 0
    assert spec.meta["palette"] == "tab20"
    assert spec.meta["diverged_color"]  # a fixed, non-empty colour for the diverged set
    # the swatch index is deterministic and 1-based: id k -> (k-1) % 20.
    assert spec.meta["palette_index"] == {1: 0, 2: 1, 3: 2}
    _roundtrips(spec)


def test_attractor_set_kind_override() -> None:
    spec = _attractor_set().to_plot_spec(kind="phase_portrait_2d")
    assert spec.kind == PlotKind.PHASE_PORTRAIT_2D


# ---------------------------------------------------------------------------
# BasinFractions -> CATEGORICAL_BAR
# ---------------------------------------------------------------------------


def test_basin_fractions_is_a_categorical_bar() -> None:
    bf = BasinFractions(
        fractions={1: 0.6, 2: 0.4}, diverged=0.0, n=100, attractors=_attractor_set()
    )
    spec = bf.to_plot_spec()
    assert spec.kind == PlotKind.CATEGORICAL_BAR
    assert [layer.kind for layer in spec.layers] == [PlotKind.BAR]
    layer = spec.layers[0]
    assert "cat" in layer.data and "y" in layer.data
    # one bar per attractor id (no diverged share here).
    assert layer.data["y"].size == 2
    assert spec.x.scale == "categorical"
    assert list(spec.x.categories) == ["attractor 1", "attractor 2"]
    _roundtrips(spec)


def test_basin_fractions_appends_a_diverged_bar() -> None:
    bf = BasinFractions(
        fractions={1: 0.5, 2: 0.3}, diverged=0.2, n=100, attractors=_attractor_set()
    )
    spec = bf.to_plot_spec()
    # the diverged share becomes a final bar with its own category label.
    assert spec.layers[0].data["y"].size == 3
    assert list(spec.x.categories)[-1] == "diverged"


# ---------------------------------------------------------------------------
# ContinuationResult -> CONTINUATION
# ---------------------------------------------------------------------------


def test_continuation_is_stacked_bands_with_tipping_vlines() -> None:
    spec = _continuation().to_plot_spec()
    assert spec.kind == PlotKind.CONTINUATION
    # one stacked AREA band per attractor id.
    assert [layer.kind for layer in spec.layers] == [PlotKind.AREA, PlotKind.AREA]
    for layer in spec.layers:
        assert "lo" in layer.data and "hi" in layer.data
    # the bands tile [0, 1] at each value (cumulative fractions sum to 1 here).
    top = spec.layers[-1].data["hi"]
    np.testing.assert_allclose(top, 1.0)
    # the annihilation of attractor 1 at the final value is a vertical tipping line.
    vlines = [a for a in spec.annotations if a.kind == "vline"]
    assert vlines, "expected a tipping vline where a basin annihilates"
    assert any(a.x == 1.0 and "disappear" in a.text for a in vlines)
    _roundtrips(spec)


# ---------------------------------------------------------------------------
# UncertaintyExponent -> SCALING_FIT
# ---------------------------------------------------------------------------


def test_uncertainty_exponent_is_a_scaling_fit() -> None:
    spec = _uncertainty().to_plot_spec()
    assert spec.kind == PlotKind.SCALING_FIT
    kinds = [layer.kind for layer in spec.layers]
    assert PlotKind.SCATTER in kinds and PlotKind.LINE in kinds
    # the fit LINE has slope alpha: (y1 - y0) / (x1 - x0) == alpha.
    line = next(layer for layer in spec.layers if layer.kind == PlotKind.LINE)
    x, y = line.data["x"], line.data["y"]
    slope = (y[1] - y[0]) / (x[1] - x[0])
    np.testing.assert_allclose(slope, 0.4)
    _roundtrips(spec)


def test_uncertainty_exponent_scaling_kind_override() -> None:
    # the .plot.scaling() seam passes kind="scaling_fit" explicitly.
    spec = _uncertainty().to_plot_spec(kind="scaling_fit")
    assert spec.kind == PlotKind.SCALING_FIT


# ---------------------------------------------------------------------------
# BasinsResult stays BASINS_IMAGE (+ 3-D slice support)
# ---------------------------------------------------------------------------


def test_basins_result_is_a_basins_image() -> None:
    spec = _basins().to_plot_spec()
    assert spec.kind == PlotKind.BASINS_IMAGE
    assert spec.layers[0].kind == PlotKind.IMAGE
    assert spec.layers[0].data["c"].ndim == 2
    assert spec.aspect == "equal"
    _roundtrips(spec)


def test_basins_result_3d_slice_squeezes_to_a_2d_image() -> None:
    spec = _basins(slice3d=True).to_plot_spec()
    assert spec.kind == PlotKind.BASINS_IMAGE
    image = spec.layers[0].data["c"]
    # the degenerate (counts == 1) axis is dropped: a 3-D slice paints 2-D.
    assert image.ndim == 2
    assert image.shape == (4, 4)
    # the axes label the two *free* coordinates (x1 and x3), not the pinned one.
    assert spec.x.label == "x1"
    assert spec.y.label == "x3"
    _roundtrips(spec)


# ---------------------------------------------------------------------------
# Intra-result palette consistency (the keystone invariant)
# ---------------------------------------------------------------------------


def test_same_id_same_colour_across_scatter_and_image() -> None:
    """The same attractor id maps to the same palette swatch in both views."""
    aset = _attractor_set(ids=(1, 2, 3))
    basins = BasinsResult(
        labels=np.array([[1, 1, 2], [3, 3, 2], [1, 2, 3]], dtype=int),
        attractors=aset,
        grid=Grid(lo=np.array([-1.0, -1.0]), hi=np.array([1.0, 1.0]), counts=(3, 3)),
    )
    scatter_spec = aset.to_plot_spec()
    image_spec = basins.to_plot_spec()

    # both views carry the same palette name and the same {id: swatch} mapping.
    assert scatter_spec.meta["palette"] == image_spec.meta["palette"] == "tab20"
    assert scatter_spec.meta["palette_index"] == image_spec.meta["palette_index"]
    assert image_spec.meta["palette_index"] == {1: 0, 2: 1, 3: 2}


def test_palette_index_is_consistent_with_basin_fractions_and_continuation() -> None:
    """Every basin view keys off the one {id: swatch} palette mapping."""
    aset = _attractor_set(ids=(1, 2, 3))
    bf = BasinFractions(fractions={1: 0.5, 2: 0.3, 3: 0.2}, diverged=0.0, n=100, attractors=aset)
    cont = ContinuationResult(
        param="r",
        values=np.linspace(0.0, 1.0, 3),
        fractions={1: np.array([0.5, 0.4, 0.3]), 2: np.array([0.5, 0.6, 0.7])},
        attractors=[{} for _ in range(3)],
        diverged=np.zeros(3),
    )
    expected = {1: 0, 2: 1, 3: 2}
    assert aset.to_plot_spec().meta["palette_index"] == expected
    assert bf.to_plot_spec().meta["palette_index"] == expected
    assert cont.to_plot_spec().meta["palette_index"] == {1: 0, 2: 1}


def test_palette_wraps_past_twenty_ids() -> None:
    """The 20-swatch palette is cyclic: id 21 reuses swatch 0."""
    aset = _attractor_set(ids=(1, 21))
    idx = aset.to_plot_spec().meta["palette_index"]
    assert idx[1] == idx[21] == 0
