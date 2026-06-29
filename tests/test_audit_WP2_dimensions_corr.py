"""Regression test for WP2_dimensions_corr (finding A2:A2-1).

``correlation_dimension`` accepts a public ``radii`` grid that need not be
monotone.  The internal correlation sum returns ``radii``/``C`` in the caller's
original order, while ``fit_scaling_region`` scans contiguous index windows and
requires inputs ordered by increasing scale.  Before the fix, a shuffled radii
grid fed scale-shuffled data into the fit and silently produced a wrong (and
order-dependent) :math:`D_2`.  The fix sorts the masked ``(log r, log C)`` pair
by ascending log-radius before fitting, so the answer is invariant to the order
of an otherwise-identical radii grid.
"""

from __future__ import annotations

import numpy as np

from tsdynamics.analysis import dimensions as dim


def _square_points(n: int = 4000, seed: int = 0) -> np.ndarray:
    """A uniform 2-D point cloud (correlation dimension ~= 2)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, 2))


def test_correlation_dimension_invariant_to_radii_order() -> None:
    """A shuffled radii grid must give the same D2 as the sorted grid."""
    points = _square_points()
    radii_sorted = np.logspace(-2.0, -0.3, 24)

    rng = np.random.default_rng(123)
    perm = rng.permutation(radii_sorted.size)
    radii_shuffled = radii_sorted[perm]
    # Guard the test premise: the grid is genuinely non-monotone.
    assert not np.all(np.diff(radii_shuffled) > 0)

    d_sorted = float(dim.correlation_dimension(points, radii=radii_sorted))
    d_shuffled = float(dim.correlation_dimension(points, radii=radii_shuffled))

    # Pre-fix: the contiguous-window fit ran on scale-shuffled data, so the two
    # answers diverged.  Post-fix they are bit-for-bit identical (same masked
    # set, re-sorted to the same ascending order).
    assert d_shuffled == d_sorted
