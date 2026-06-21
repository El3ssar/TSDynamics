r"""
Regression tests for the Theiler-window neighbour query in :mod:`embedding`.

Ticket **FIX-EMBED-THEILER**.  The neighbour search underlying Cao's and
Kennel's minimum-embedding-dimension estimators excludes temporally-close
neighbours (the Theiler window, :math:`|i - j| > w`; Theiler, 1986).  The
exclusion must *select against* in-window neighbours, never *drop the point*: a
point is only invalid when no out-of-window neighbour exists at all.

The original implementation queried a fixed ``k = theiler + 2`` neighbours and
declared a point invalid whenever those ``k`` spatial near-neighbours all fell
inside the window.  For densely-sampled / quasi-periodic series a long run of
temporally-adjacent samples is spatially nearly coincident, so the in-window
cluster crowds the query and the genuine (further-ranked) out-of-window
neighbour is never seen — the point is *spuriously dropped*.  The fix grows
``k`` until every point either has a valid out-of-window neighbour or has
exhausted the whole set.

These tests assert (a) the neighbour returned is the *true* nearest neighbour
outside the window (brute-force oracle), (b) no point with a valid neighbour is
dropped on a constructed crowding case **and** on a quasi-periodic torus, and
(c) the genuine no-neighbour case is still flagged invalid.  The crowding test
fails on the pre-fix fixed-``k`` query.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.analysis import embedding as emb
from tsdynamics.analysis.embedding.dimension import _nearest_neighbor


def _brute_force_nn(points: np.ndarray, *, p: float, theiler: int):
    """Reference out-of-window nearest neighbour of every point (O(m^2))."""
    m = points.shape[0]
    nn = np.full(m, -1)
    dist = np.full(m, np.nan)
    valid = np.zeros(m, dtype=bool)
    idx = np.arange(m)
    for i in range(m):
        d = np.linalg.norm(points - points[i], ord=p, axis=1)
        d[np.abs(idx - i) <= theiler] = np.inf
        j = int(np.argmin(d))
        if np.isfinite(d[j]):
            nn[i], dist[i], valid[i] = j, d[j], True
    return nn, dist, valid


# ── (a) the neighbour returned is the true out-of-window nearest ────────────────


@pytest.mark.parametrize("theiler", [0, 1, 5, 23, 60])
def test_matches_brute_force_oracle(theiler: int) -> None:
    """For every Theiler window the kd-tree query agrees with the O(m^2) oracle."""
    rng = np.random.default_rng(7)
    points = rng.standard_normal((400, 3))

    nn, dist, valid = _nearest_neighbor(points, p=2.0, theiler=theiler)
    bnn, bdist, bvalid = _brute_force_nn(points, p=2.0, theiler=theiler)

    np.testing.assert_array_equal(valid, bvalid)
    # Distances of the chosen neighbour must match (ties may pick a different index,
    # but the *distance* is unique for the nearest out-of-window neighbour).
    np.testing.assert_allclose(dist[valid], bdist[valid], rtol=0, atol=1e-12)
    # Every reported neighbour is genuinely outside the window.
    assert np.all(np.abs(nn[valid] - np.flatnonzero(valid)) > theiler)


# ── (b1) the constructed crowding case: no spurious drops (fails pre-fix) ───────


def _far_revisit_series(plateau: int = 80, step_in: float = 1e-5, gap: float = 10.0):
    """Two identical tight ramps separated by ``gap``.

    Each ramp's points are strictly nearest to their own temporal neighbours, so
    a mid-ramp point's ``plateau`` closest spatial neighbours are all in-window;
    its only out-of-window match is the *other* ramp, ranked beyond the local
    run.  A fixed ``k = theiler + 2 < plateau`` query never reaches it.
    """
    n = 2 * plateau
    x = np.empty(n)
    for k in range(plateau):
        x[k] = k * step_in
        x[plateau + k] = k * step_in + gap
    return x.reshape(-1, 1)


@pytest.mark.parametrize("theiler", [20, 40, 60])
def test_crowding_does_not_drop_points(theiler: int) -> None:
    """Every point has a valid out-of-window neighbour, so none may be dropped.

    Regression guard: the pre-fix ``k = theiler + 2`` query dropped the bulk of
    the points here (the in-window ramp crowds the query).
    """
    points = _far_revisit_series()
    _, _, oracle_valid = _brute_force_nn(points, p=2.0, theiler=theiler)
    # The construction guarantees every point genuinely has an out-of-window match.
    assert oracle_valid.all()

    nn, dist, valid = _nearest_neighbor(points, p=2.0, theiler=theiler)
    assert valid.all(), f"{(~valid).sum()} points spuriously dropped at theiler={theiler}"

    # And the recovered neighbour is the true one (the far ramp).
    bnn, bdist, _ = _brute_force_nn(points, p=2.0, theiler=theiler)
    np.testing.assert_allclose(dist, bdist, rtol=0, atol=1e-12)


# ── (b2) a quasi-periodic torus: no drops at any reasonable window ──────────────


def test_quasiperiodic_no_drops() -> None:
    """A densely-sampled two-frequency torus drops no points under a Theiler window."""
    t = np.linspace(0.0, 600.0 * np.pi, 12000)
    x = np.sin(t) + np.sin(np.sqrt(2.0) * t)
    tau = 40
    # Build a 4-D delay reconstruction directly.
    rows = x.size - 4 * tau
    cols = np.empty((rows, 4))
    for j in range(4):
        cols[:, j] = x[j * tau : j * tau + rows]

    for theiler in (0, tau, 3 * tau):
        _, _, valid = _nearest_neighbor(cols, p=2.0, theiler=theiler)
        _, _, oracle = _brute_force_nn(cols, p=2.0, theiler=theiler)
        # No point that genuinely has a neighbour may be dropped.
        assert np.array_equal(valid, oracle)
        assert valid.all(), f"torus dropped points at theiler={theiler}"


# ── (c) the genuine no-neighbour case is still flagged invalid ──────────────────


def test_window_covers_everything_marks_invalid() -> None:
    """When the window spans the whole series no neighbour exists → all invalid."""
    rng = np.random.default_rng(0)
    points = rng.standard_normal((30, 2))
    nn, dist, valid = _nearest_neighbor(points, p=2.0, theiler=29)
    assert not valid.any()
    assert np.all(nn == -1)
    assert np.all(np.isnan(dist))


def test_single_point_has_no_neighbour() -> None:
    """A one-point set is degenerate: no neighbour, marked invalid."""
    nn, dist, valid = _nearest_neighbor(np.zeros((1, 2)), p=2.0, theiler=0)
    assert nn.tolist() == [-1]
    assert np.isnan(dist).all()
    assert not valid.any()


# ── (d) FNN / Cao stay green and behave on the crowding-prone torus ─────────────


def test_fnn_and_cao_green_on_torus() -> None:
    """FNN and Cao run cleanly on a dense quasi-periodic series with a Theiler window.

    A 2-frequency torus is genuinely 2-dimensional; both estimators must finish
    without spurious-drop errors and report a small minimum embedding dimension.
    """
    t = np.linspace(0.0, 600.0 * np.pi, 12000)
    x = np.sin(t) + np.sin(np.sqrt(2.0) * t)
    tau = 40

    fnn = emb.false_nearest_neighbors(x, delay=tau, max_dim=8, theiler=tau)
    cao = emb.cao_dimension(x, delay=tau, max_dim=8, theiler=tau)

    # A torus unfolds at a low dimension; the exact value is not the point here —
    # what matters is that neither estimator raised a "too few valid neighbours"
    # error from spurious drops, and both return a sane small dimension.
    assert 1 <= int(fnn) <= 6
    assert 1 <= int(cao) <= 6
    assert fnn.fnn_fraction is not None and np.all(np.isfinite(fnn.fnn_fraction))
    assert cao.afn_e1 is not None and np.all(np.isfinite(cao.afn_e1))
