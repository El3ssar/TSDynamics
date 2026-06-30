"""Regression tests for WP7_embedding audit fixes.

Covers:

- A9:A9-3 — ``_first_local_min`` must return the *onset* of a flat-bottomed
  mutual-information valley, not its far edge, so the Fraser--Swinney delay is
  not over-estimated on quantised / flat curves.

References
----------
A. M. Fraser and H. L. Swinney, "Independent coordinates for strange attractors
from mutual information", *Phys. Rev. A* **33**, 1134 (1986).
"""

from __future__ import annotations

import numpy as np

from tsdynamics.analysis.embedding.delay import MutualInformation, _first_local_min


def test_first_local_min_returns_valley_onset_not_right_edge() -> None:
    """A flat-bottomed valley resolves to its first plateau lag (the onset).

    Pre-fix predicate ``curve[k] <= curve[k-1] and curve[k] < curve[k+1]`` only
    fired at the far edge of a flat plateau (where the strict right-rise finally
    holds), returning the larger lag.  The corrected predicate
    ``curve[k] < curve[k-1] and curve[k] <= curve[k+1]`` returns the onset.
    """
    # Plateau [.., 1, 1, 1, ..]: onset is index 2, right edge is index 4.
    flat = np.array([3.0, 2.0, 1.0, 1.0, 1.0, 2.0])
    assert _first_local_min(flat) == 2  # pre-fix would return 4

    # Longer flat valley: onset 3, right edge 5.
    flat2 = np.array([5.0, 4.0, 3.0, 2.0, 2.0, 2.0, 3.0, 4.0])
    assert _first_local_min(flat2) == 3  # pre-fix would return 5


def test_first_local_min_unchanged_on_strictly_shaped_curves() -> None:
    """The fix is answer-preserving on strictly descending-then-ascending curves."""
    strict = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0])
    assert _first_local_min(strict) == 4

    # Two minima: the *first* interior minimum still wins.
    two = np.array([3.0, 1.0, 2.0, 0.5, 2.0])
    assert _first_local_min(two) == 1

    # Monotone curves still have no interior minimum.
    assert _first_local_min(np.array([1.0, 2.0, 3.0, 4.0])) is None
    assert _first_local_min(np.array([4.0, 3.0, 2.0, 1.0])) is None


def test_optimal_lag_reads_onset_of_flat_valley() -> None:
    """``MutualInformation.optimal_lag`` reflects the onset convention."""
    flat = np.array([3.0, 2.0, 1.0, 1.0, 1.0, 2.0])
    mi = MutualInformation(values=flat, meta={"analysis": "mutual_information"})
    assert mi.optimal_lag == 2  # pre-fix would have been 4
