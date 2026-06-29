r"""Regression: ``basin_entropy`` drops trailing partial boxes (Daza 2016).

Daza et al. (2016) partition a basin image into *non-overlapping* boxes of a
fixed number of cells per axis; a grid axis whose length is not a multiple of
``box_size`` leaves a trailing partial box, which samples fewer trajectories and
biases the per-box entropy.  The pristine convention discards those partial
edge boxes and averages :math:`S_b` / :math:`S_{bb}` over full boxes only.

This guards two properties of the fix:

1. A grid that divides ``box_size`` evenly is **byte-identical** to the
   reference (the partial-box path never fires there).
2. A grid that does *not* divide evenly now matches a full-box reference (the
   same grid cropped to a multiple of ``box_size``) and no longer counts the
   partial edge boxes.

A. Daza, A. Wagemakers, B. Georgeot, D. Guéry-Odelin and M. A. F. Sanjuán,
"Basin entropy: a new tool to analyze uncertainty in dynamical systems",
*Scientific Reports* **6**, 31416 (2016).
"""

from __future__ import annotations

import numpy as np

import tsdynamics.analysis.basins.metrics as bas


def test_divisible_grid_is_byte_identical() -> None:
    """A grid that divides box_size evenly is unchanged (no partial boxes)."""
    rng = np.random.default_rng(0)
    labels = rng.integers(1, 4, size=(20, 20))  # 20 / 5 == 4, divides evenly
    be = bas.basin_entropy(labels, box_size=5)
    # 4 x 4 == 16 full boxes, exactly as before the fix.
    assert be.n_boxes == 16
    # Reconstruct the full-box reference by hand to prove value identity.
    from itertools import product

    log = np.log(np.e)
    ref = []
    for oy, ox in product(range(0, 20, 5), range(0, 20, 5)):
        flat = labels[oy : oy + 5, ox : ox + 5].reshape(-1)
        _, counts = np.unique(flat, return_counts=True)
        p = counts / flat.size
        ref.append(float(-np.sum(p * np.log(p)) / log))
    assert be.sb == float(np.mean(ref))


def test_non_divisible_grid_drops_partial_boxes() -> None:
    """A 7x7 grid with box_size=5 must use ONE full box, not a 2x2 lattice.

    Pre-fix ``_iter_blocks`` yielded a 2x2 box lattice (the (0,0) full box plus
    three partial edge boxes of shape 2x5, 5x2 and 2x2), so ``n_boxes`` was 4 and
    the partial boxes biased ``S_b``.  Post-fix only the single full 5x5 box at
    the origin survives, so the result equals the full-box reference (the grid
    cropped to its leading 5x5 corner).
    """
    rng = np.random.default_rng(1)
    labels = rng.integers(1, 4, size=(7, 7))  # 7 is not a multiple of 5

    be = bas.basin_entropy(labels, box_size=5)
    # Only the leading 5x5 full box remains; the three partial edge boxes are gone.
    assert be.n_boxes == 1  # pre-fix this was 4 (2x2 lattice incl. partials)

    # Full-box reference: the same metric on the cropped 5x5 corner.
    cropped = labels[:5, :5]
    ref = bas.basin_entropy(cropped, box_size=5)
    assert be.n_boxes == ref.n_boxes
    assert be.sb == ref.sb
    assert be.n_boundary_boxes == ref.n_boundary_boxes
