"""Regression tests for WP8_orbits — orbit_diagram branch/period heuristics.

Covers two audited defects in
:mod:`tsdynamics.analysis.orbits.orbit_diagram`:

A10-1
    The negligible-spread guard in :func:`_count_branches` scaled its collapse
    threshold by the orbit's DC offset (``max(|s[0]|, |s[-1]|)``), so a genuine
    multi-branch orbit living far from the origin (e.g. branches at
    ``{100.0, 100.5}``) was collapsed to a single branch and mis-reported as
    period 1.  The fix centers the dispersion scale about the orbit's mean.

A10-2
    A chaotic band whose finite-sample points happen to cluster into ``p <=
    max_period`` gap-separated bins was reported as a genuine period-``p``
    window.  The fix requires the iterate sequence to actually revisit its
    values cyclically (``v[i] ≈ v[i + p]``) before accepting a finite period.
"""

from __future__ import annotations

import numpy as np

import tsdynamics as ts
from tsdynamics.analysis.orbits.orbit_diagram import _count_branches, _is_cyclic


class TestBranchSpreadOffset:
    """A10-1: branch counting must not conflate the offset with the noise scale."""

    def test_offset_period_two_not_collapsed(self) -> None:
        # A genuine period-2 orbit at a large offset: branches 0.5 apart on a
        # mean of ~100.  Pre-fix scale = max(|100.0|, |100.5|, 1.0) = 100.5, so
        # span 0.5 <= rtol*scale = 1.005 collapsed it to ONE branch.
        col = np.tile([100.0, 100.5], 60)
        assert _count_branches(col, 0.01) == 2

    def test_offset_and_origin_agree(self) -> None:
        # The same branch *separation* must yield the same count regardless of
        # where the orbit sits — the count is offset-invariant after centering.
        offset = np.tile([100.0, 100.5], 60)
        origin = np.tile([1.0, 1.5], 60)
        assert _count_branches(offset, 0.01) == _count_branches(origin, 0.01) == 2

    def test_negligible_spread_still_collapses(self) -> None:
        # Answer-preserving: a single branch with integration-noise spread (tiny
        # absolute, near the origin) must still collapse to one branch.
        rng = np.random.default_rng(0)
        col = 0.7 + rng.normal(0.0, 1e-12, 300)
        assert _count_branches(col, 0.01) == 1


class TestSpuriousFinitePeriod:
    """A10-2: a chaotic band clustering into p bins is not a period-p window."""

    def test_chaotic_cluster_is_not_cyclic(self) -> None:
        # Logistic at r=3.7 is chaotic; with a coarse rtol its finite-sample
        # points cluster into a small number of gap-separated bins.  The cluster
        # count is real, but the sequence does NOT revisit its values cyclically.
        od = ts.orbit_diagram(ts.Logistic(), "r", [3.7], n=300, transient=1000, ic=[0.5])
        col = od.points[0][:, 0]
        rtol = 0.02
        nbranch = _count_branches(col, rtol)
        assert 2 <= nbranch <= 16  # clusters into a spuriously "finite" count
        # ... but the orbit is not cyclic at that period -> aperiodic.
        assert not _is_cyclic(col, nbranch, rtol)
        assert od.periods(rtol=rtol)[0] == 0

    def test_true_cycle_is_cyclic(self) -> None:
        # Control: a genuine period-2 logistic window stays period 2 (the cyclic
        # check must not reject real cycles).
        od = ts.orbit_diagram(
            ts.Logistic(), "r", [3.2], n=200, transient=2000, carry_state=False, ic=[0.5]
        )
        col = od.points[0][:, 0]
        assert _is_cyclic(col, 2, 0.01)
        assert od.periods()[0] == 2

    def test_period_doubling_sequence_unchanged(self) -> None:
        # Answer-preserving end-to-end: the clean period-doubling cascade still
        # reads 1, 2, 4, 8 (every step is a true, cyclically-repeating window).
        od = ts.orbit_diagram(
            ts.Logistic(),
            "r",
            [2.8, 3.2, 3.5, 3.56],
            n=120,
            transient=2000,
            carry_state=False,
            ic=[0.5],
        )
        np.testing.assert_array_equal(od.periods(), np.array([1, 2, 4, 8]))
