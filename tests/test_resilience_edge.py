r"""Regression: ``resilience`` must not overestimate the minimal fatal shock.

The resilience of an attractor (Halekotte & Feudel, *Sci. Rep.* **10**, 11374,
2020) is the minimal-fatal-shock distance — the smallest perturbation that pushes
the state out of the attractor's basin — estimated as the distance from the
attractor to the nearest cell of another basin via a Euclidean distance transform
(EDT) of the basin mask.

Two ways the naive estimate *overestimates* (which FIX-BASINS-DIVERGED corrects):

1. **Domain edge.** When the basin runs to the edge of the computed grid, the EDT
   sees no background there and reports the (large) distance to the far interior
   boundary instead of the (small) distance to the domain edge.  We cannot claim
   resilience beyond the computed domain, so the edge must act as a boundary.

2. **Single representative.** Reading the EDT only at the attractor's centroid
   misses an extended attractor (limit cycle / strange set) that grazes the
   boundary far from its centre; the minimal fatal shock is the *minimum* over the
   attractor's spatial extent.

Both removed overestimates are what this regression guards.  Tickets
FIX-RESILIENCE-EDGE / FIX-BASINS-DIVERGED.
"""

from __future__ import annotations

import numpy as np

import tsdynamics as ts
from tsdynamics.analysis.basins.attractors import Attractor, AttractorSet
from tsdynamics.analysis.basins.basins import BasinsResult
from tsdynamics.data import Grid


def _two_basin_split(boundary_x: float, n: int = 101) -> tuple[Grid, np.ndarray, np.ndarray]:
    """Unit-square grid split into basin 1 (``x < boundary_x``) and basin 2."""
    grid = Grid([0.0, 0.0], [1.0, 1.0], (n, n))
    xs = np.linspace(0.0, 1.0, n)
    labels = np.where(xs[:, None] < boundary_x, 1, 2) * np.ones((n, n), dtype=int)
    return grid, labels, xs


def test_resilience_does_not_overestimate_at_domain_edge() -> None:
    """An attractor near the grid edge: resilience is the (small) distance to the
    edge, not the (large) distance to the far interior boundary."""
    # basin 1 occupies x < 0.8; its attractor sits at x = 0.05, hard against the
    # left domain edge. Distance to the interior boundary (x=0.8) is 0.75; distance
    # to the left domain edge is 0.05. The minimal fatal shock is the latter.
    grid, labels, _ = _two_basin_split(boundary_x=0.8)
    att1 = Attractor(1, np.array([[0.05, 0.5]]), cells=1)
    att2 = Attractor(2, np.array([[0.9, 0.5]]), cells=1)
    aset = AttractorSet({1: att1, 2: att2}, diverged=0, seeds=labels.size)
    res = BasinsResult(labels=labels, grid=grid, attractors=aset)

    r1 = float(ts.resilience(res, 1))
    # The honest estimate is the distance to the computed domain edge (~0.05) ...
    assert abs(r1 - 0.05) < 0.02
    # ... and it must be far below the 0.75 the unpadded EDT would have reported.
    assert r1 < 0.5


def test_resilience_minimises_over_attractor_extent() -> None:
    """An extended attractor grazing the boundary: resilience reflects the closest
    point, not the centroid."""
    # basin 1 occupies x < 0.5. The attractor is a horizontal cloud centred at
    # x = 0.2 (centroid distance 0.3 to the x=0.5 boundary) but reaching out to
    # x = 0.45 (distance 0.05). The minimal fatal shock is ~0.05.
    grid, labels, _ = _two_basin_split(boundary_x=0.5)
    cloud = np.array([[0.2, 0.5], [0.25, 0.5], [0.45, 0.5], [0.1, 0.5]])
    att1 = Attractor(1, cloud, cells=4)
    att2 = Attractor(2, np.array([[0.8, 0.5]]), cells=1)
    aset = AttractorSet({1: att1, 2: att2}, diverged=0, seeds=labels.size)
    res = BasinsResult(labels=labels, grid=grid, attractors=aset)

    r1 = float(ts.resilience(res, 1))
    # Closest grazing point (x=0.45) is ~0.05 from the boundary; the centroid
    # (x=0.2) is 0.3 — taking the centroid would overestimate ~6x.
    assert r1 < 0.15
    assert abs(r1 - 0.05) < 0.03
