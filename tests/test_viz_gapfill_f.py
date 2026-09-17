"""Viz specs for the recurrence result types (stream GAPFILL-F).

These tests pin the *sparse, non-densifying* contract of the recurrence
``__plot_spec__`` builders (the surrogate half left with the surrogate estimators
when the library narrowed to phase-space methods):

- ``RecurrenceMatrix.__plot_spec__`` emits the recurrence plot as a **sparse**
  ``(i, j)`` ``SCATTER`` read straight off the matrix COO — it must **not**
  densify (``toarray``) the matrix.  The OOM regression guard builds a sparse
  matrix at ``N = 50_000`` with only a few thousand stored recurrences and
  asserts the produced coordinate arrays are ``~nnz`` long (not ``N**2``) and the
  whole call stays under a fixed byte budget.
- ``WindowedRQA`` is a measure-vs-window ``DIAGNOSTIC_CURVE`` built from the
  already-computed per-window scalars (no nested dense-matrix walk).
- ``RQAResult`` is a small ``CATEGORICAL_BAR`` of the scalar RQA measures.

Engine-free by design (synthetic sparse matrices / arrays — no ``tsdynamics._rust``).
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import sparse

from tsdynamics.analysis.recurrence.matrix import RecurrenceMatrix
from tsdynamics.analysis.recurrence.rqa import RQAResult
from tsdynamics.analysis.recurrence.windowed import WindowedRQA
from tsdynamics.viz.spec import PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------


def _sparse_recurrence(n: int, nnz_pairs: int, *, seed: int = 0) -> sparse.csr_matrix:
    """A symmetric sparse boolean recurrence matrix with ``~2 * nnz_pairs`` entries.

    Off-diagonal ``i < j`` pairs are drawn at random and symmetrised, so the
    stored count is the number of recurrent pairs (the diagonal is never set) —
    exactly the shape ``recurrence_matrix`` produces.
    """
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n - 1, size=nnz_pairs)
    j = i + 1 + rng.integers(0, n - 1 - i)  # guarantees i < j < n
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.ones(rows.size, dtype=bool)
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    mat.sum_duplicates()
    return mat


def _rqa_result() -> RQAResult:
    return RQAResult(
        recurrence_rate=0.1,
        determinism=0.8,
        laminarity=0.5,
        avg_diagonal_length=3.0,
        max_diagonal_length=7,
        divergence=1.0 / 7.0,
        diagonal_entropy=0.9,
        trapping_time=2.0,
        max_vertical_length=4,
        size=6,
        epsilon=0.5,
        theiler_window=0,
        min_diagonal=2,
        min_vertical=2,
        diagonal_lengths=np.array([2, 3, 5], dtype=float),
        vertical_lengths=np.array([2, 4], dtype=float),
    )


# ---------------------------------------------------------------------------
# RecurrenceMatrix: sparse SCATTER, no densification
# ---------------------------------------------------------------------------


def test_recurrence_matrix_spec_is_sparse_scatter() -> None:
    """The recurrence plot is a sparse ``(i, j)`` SCATTER, not a dense IMAGE."""
    n, pairs = 200, 50
    rm = RecurrenceMatrix(matrix=_sparse_recurrence(n, pairs), epsilon=0.5)
    spec = rm.__plot_spec__()

    assert isinstance(spec, PlotSpec)
    assert spec.kind == PlotKind.RECURRENCE_PLOT
    assert spec.aspect == "equal"
    assert len(spec.layers) == 1
    layer = spec.layers[0]
    assert layer.kind == PlotKind.SCATTER  # sparse points, not an IMAGE
    assert set(layer.data) == {"x", "y"}
    # One scatter point per stored recurrence (nnz), never N**2.
    nnz = rm.matrix.nnz
    assert layer.data["x"].size == nnz
    assert layer.data["y"].size == nnz
    assert nnz < n * n
    # No layer carries the dense (N, N) image channel.
    for lyr in spec.layers:
        assert lyr.kind != PlotKind.IMAGE
        assert "c" not in lyr.data


def test_recurrence_matrix_spec_does_not_call_toarray(monkeypatch) -> None:
    """Building the spec must never densify the matrix via ``toarray``."""
    rm = RecurrenceMatrix(matrix=_sparse_recurrence(300, 40), epsilon=0.5)

    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("__plot_spec__ densified the recurrence matrix (toarray called)")

    # Forbid densification on both the wrapper and the underlying sparse matrix.
    monkeypatch.setattr(RecurrenceMatrix, "toarray", _boom)
    monkeypatch.setattr(type(rm.matrix), "toarray", _boom)
    monkeypatch.setattr(type(rm.matrix), "todense", _boom, raising=False)

    spec = rm.__plot_spec__()
    assert spec.layers[0].kind == PlotKind.SCATTER
    # to_dict must also stay sparse (no densification on serialization).
    rm.__plot_spec__().to_dict()


def test_recurrence_matrix_oom_regression_large_n() -> None:
    """OOM guard: N=50_000 with a few thousand nnz stays ~nnz, well under budget.

    A dense ``(50_000, 50_000)`` bool image is 2.5e9 bytes; the sparse SCATTER
    spec must instead produce coordinate arrays of length ``nnz`` (a few thousand)
    and round-trip far under a fixed byte budget.
    """
    n, pairs = 50_000, 3_000
    rm = RecurrenceMatrix(matrix=_sparse_recurrence(n, pairs), epsilon=0.5)
    nnz = rm.matrix.nnz
    assert nnz <= 2 * pairs  # symmetrised pair count, no densification

    spec = rm.__plot_spec__()
    x = spec.layers[0].data["x"]
    y = spec.layers[0].data["y"]

    # Coordinate arrays are ~nnz long, nowhere near N**2.
    assert x.size == nnz
    assert y.size == nnz
    assert x.size < 10 * pairs
    dense_bytes = n * n  # bytes a dense bool image would need
    budget = 1_000_000  # 1 MB — generous for ~3k float64 pairs, tiny vs dense
    used = x.nbytes + y.nbytes
    assert used < budget < dense_bytes

    # The serialized dict is bounded too (lists of ~nnz numbers, not N**2).
    payload = spec.to_dict()
    n_coords = len(payload["layers"][0]["data"]["x"]) + len(payload["layers"][0]["data"]["y"])
    assert n_coords == 2 * nnz
    assert sys.getsizeof(payload) < budget


def test_recurrence_matrix_spec_round_trips() -> None:
    """The sparse recurrence spec round-trips through ``to_dict`` / ``from_dict``."""
    rm = RecurrenceMatrix(matrix=_sparse_recurrence(120, 30), epsilon=0.5)
    spec = rm.__plot_spec__()
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert len(rebuilt.layers) == len(spec.layers)
    np.testing.assert_array_equal(rebuilt.layers[0].data["x"], spec.layers[0].data["x"])
    np.testing.assert_array_equal(rebuilt.layers[0].data["y"], spec.layers[0].data["y"])


def test_recurrence_matrix_empty_spec() -> None:
    """An all-zero (no recurrence) matrix yields an empty but valid scatter spec."""
    rm = RecurrenceMatrix(matrix=sparse.csr_matrix((50, 50), dtype=bool), epsilon=0.5)
    spec = rm.__plot_spec__()
    assert spec.layers[0].kind == PlotKind.SCATTER
    assert spec.layers[0].data["x"].size == 0
    PlotSpec.from_dict(spec.to_dict())  # still round-trips


# ---------------------------------------------------------------------------
# RQAResult: CATEGORICAL_BAR of the scalar measures
# ---------------------------------------------------------------------------


def test_rqa_result_spec_is_categorical_bar() -> None:
    """RQA's spec is a CATEGORICAL_BAR of RR / DET / LAM / ENTR (no histogram walk)."""
    result = _rqa_result()
    spec = result.__plot_spec__()
    assert spec.kind == PlotKind.CATEGORICAL_BAR
    assert len(spec.layers) == 1
    layer = spec.layers[0]
    assert layer.kind == PlotKind.BAR
    assert "cat" in layer.data and "y" in layer.data
    # The category axis carries the measure labels, paired with the integer cats.
    assert spec.x.scale == "categorical"
    assert list(spec.x.categories) == ["RR", "DET", "LAM", "ENTR"]
    assert layer.data["cat"].size == 4
    np.testing.assert_allclose(
        layer.data["y"],
        [result.recurrence_rate, result.determinism, result.laminarity, result.diagonal_entropy],
    )
    PlotSpec.from_dict(spec.to_dict())


# ---------------------------------------------------------------------------
# WindowedRQA: measure-vs-window DIAGNOSTIC_CURVE (no nested dense walk)
# ---------------------------------------------------------------------------


def test_windowed_rqa_spec_is_measure_vs_window_curve() -> None:
    """WindowedRQA is a determinism-vs-window-centre curve off the per-window scalars."""
    wr = WindowedRQA(
        centers=np.array([2.5, 7.5, 12.5]),
        results=(_rqa_result(), _rqa_result(), _rqa_result()),
        window=6,
        step=5,
    )
    spec = wr.__plot_spec__()
    assert spec.kind == PlotKind.DIAGNOSTIC_CURVE
    assert len(spec.layers) == 1
    layer = spec.layers[0]
    assert layer.kind == PlotKind.LINE
    # x is the window-centre axis, y the per-window determinism — both length n.
    np.testing.assert_array_equal(layer.data["x"], wr.centers)
    assert layer.data["y"].size == len(wr)
    np.testing.assert_allclose(layer.data["y"], [r.determinism for r in wr.results])
    PlotSpec.from_dict(spec.to_dict())
