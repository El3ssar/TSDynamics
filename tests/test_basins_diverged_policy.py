r"""The diverged-cell (``-1``) policy of the basin metrics.

A basin diagram marks cells that diverged / never settled with ``-1``.  Escape is
*not* a basin: by default the metrics exclude ``-1`` so it cannot masquerade as an
extra colour (basin_entropy) or manufacture a spurious final-state boundary
(uncertainty_exponent, wada_property).  ``include_diverged=True`` restores the
legacy "treat ``-1`` as just another label" behaviour.

``resilience`` is the deliberate exception: perturbing *into* an escape region is a
fatal shock, so an adjacent ``-1`` region is correctly a boundary there.

Ticket FIX-BASINS-DIVERGED.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.analysis.basins.attractors import Attractor, AttractorSet
from tsdynamics.analysis.basins.basins import BasinsResult
from tsdynamics.data import Grid

# --- basin_entropy -----------------------------------------------------------


def test_basin_entropy_excludes_diverged_by_default() -> None:
    """A box of (basin 1 | escape) is single-colour once escape is dropped → S=0."""
    labels = np.where(np.arange(10)[None, :] < 5, 1, -1) * np.ones((10, 10), dtype=int)
    # One 10x10 box: left half basin 1, right half -1.
    be_default = ts.basin_entropy(labels, box_size=10)
    assert be_default.sb == pytest.approx(0.0, abs=1e-12)


def test_basin_entropy_include_diverged_counts_escape() -> None:
    """With include_diverged, the same 50/50 (1 | -1) box has entropy log 2."""
    labels = np.where(np.arange(10)[None, :] < 5, 1, -1) * np.ones((10, 10), dtype=int)
    be_incl = ts.basin_entropy(labels, box_size=10, include_diverged=True)
    assert be_incl.sb == pytest.approx(np.log(2.0), abs=1e-9)


# --- uncertainty_exponent ----------------------------------------------------


def test_uncertainty_escape_is_not_a_final_state_boundary() -> None:
    """A basin-vs-escape interface is not a boundary by default (no settled
    boundary → cannot fit), but is one when escape is included."""
    labels = np.where(np.arange(40)[None, :] < 20, 1, -1) * np.ones((40, 40), dtype=int)
    # Default: the only interface is 1|-1 → no settled boundary → nothing to fit.
    with pytest.raises(ValueError):
        ts.uncertainty_exponent(labels)
    # include_diverged: 1|-1 now counts → a fit succeeds.
    ue = ts.uncertainty_exponent(labels, include_diverged=True)
    assert np.isfinite(ue.alpha)


# --- wada_property -----------------------------------------------------------


def _three_basins_plus_escape(n: int = 60) -> np.ndarray:
    """Columns: basin 1 | basin 2 | escape (-1)."""
    col = np.arange(n)
    row = np.select([col < n // 3, col < 2 * n // 3], [1, 2], default=-1)
    return row[None, :] * np.ones((n, n), dtype=int)


def test_wada_boundary_excludes_escape_by_default() -> None:
    """The basin/escape interface is not counted as boundary unless requested."""
    labels = _three_basins_plus_escape()
    default = ts.wada_property(labels)
    incl = ts.wada_property(labels, include_diverged=True)
    # Including escape adds the basin-2|escape interface → strictly more boundary.
    assert incl.n_boundary_cells > default.n_boundary_cells


def test_wada_colours_always_exclude_escape() -> None:
    """``-1`` is never a Wada colour, regardless of include_diverged."""
    labels = _three_basins_plus_escape()
    assert ts.wada_property(labels).n_basins == 2
    assert ts.wada_property(labels, include_diverged=True).n_basins == 2


# --- resilience (the deliberate exception) -----------------------------------


def test_resilience_treats_escape_as_a_boundary() -> None:
    """Perturbing into an escape region is a fatal shock, so an adjacent ``-1``
    region IS a boundary for resilience."""
    grid = Grid([0.0, 0.0], [1.0, 1.0], (101, 101))
    xs = np.linspace(0.0, 1.0, 101)
    labels = np.where(xs[:, None] < 0.5, 1, -1) * np.ones((101, 101), dtype=int)
    att1 = Attractor(1, np.array([[0.3, 0.5]]), cells=1)
    aset = AttractorSet({1: att1}, diverged=int(np.sum(labels == -1)), seeds=labels.size)
    res = BasinsResult(labels=labels, grid=grid, attractors=aset)
    # Attractor at x=0.3, escape region begins at x=0.5 → minimal fatal shock ~0.2.
    assert float(ts.resilience(res, 1)) == pytest.approx(0.2, abs=0.03)
