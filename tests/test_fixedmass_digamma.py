r"""Regression tests for the digamma-corrected fixed-mass dimension.

Ticket ``FIX-FIXEDMASS-DIGAMMA``.  The fixed-mass estimator regresses the
neighbour count (the *mass* :math:`k`) against the mean log radius
:math:`\langle\log r_k\rangle` needed to enclose it.  Grassberger (P. Grassberger,
*Phys. Lett. A* **107**, 101, 1985) showed that the *unbiased* relation is

.. math::

    \langle\log r_k\rangle = D^{-1}\,\psi(k) + \text{const},

with the digamma :math:`\psi(k)`, **not** :math:`\log k`.  Because
:math:`\psi(k) = \log k - \tfrac{1}{2k} + O(k^{-2})`, using :math:`\log k`
systematically biases the slope at the small-:math:`k` end of the scaling region
— precisely the regime where the fixed-mass method earns its signal-to-noise
advantage.  These tests pin the digamma ordinate and show the small-:math:`k`
bias is removed; they fail on the pre-fix ``log(k)`` code.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import digamma

import tsdynamics as ts
from tsdynamics.analysis.dimensions.fixedmass import fixed_mass_dimension


def _uniform_cube(d: int, n: int, seed: int) -> np.ndarray:
    """``n`` i.i.d. uniform points in the unit ``d``-cube (topological dim ``d``)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, d))


def test_ordinate_is_digamma_not_log() -> None:
    """The returned ordinate is ``psi(k)`` (the fix), not ``log(k)`` (the bug).

    This is the defining behavioural change: the abscissa of the slope fit is the
    digamma of the probed masses.  ``log(k)`` and ``psi(k)`` differ by an
    ``O(1/k)`` term, so this assertion fails on the pre-fix code.
    """
    pts = _uniform_cube(2, 6000, seed=0)
    ks = np.array([1, 2, 3, 5, 8, 13, 21, 34], dtype=int)
    res = fixed_mass_dimension(pts, ks=ks, min_window=5)

    # ordinate ``y`` is psi(k) over the (sorted-ascending) probed masses
    y = np.asarray(res.ordinate, dtype=float)
    expected = np.sort(digamma(ks.astype(float)))
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-12)

    # and it is demonstrably NOT log(k): the gap is the digamma correction
    log_k = np.sort(np.log(ks.astype(float)))
    assert np.max(np.abs(y - log_k)) > 0.1, "ordinate coincides with log(k) — fix not applied"


@pytest.mark.parametrize("d,expected", [(2, 2.0), (3, 3.0)])
def test_uniform_cube_recovers_dimension(d: int, expected: float) -> None:
    """Digamma fixed-mass recovers the topological dimension of a uniform cube."""
    pts = _uniform_cube(d, 8000, seed=1)
    res = fixed_mass_dimension(pts)
    assert abs(float(res) - expected) < 0.15, (
        f"{d}-cube: fixed-mass D = {float(res):.3f}, expected {expected}"
    )


@pytest.mark.parametrize("d,expected", [(2, 2.0), (3, 3.0)])
def test_small_k_bias_is_reduced(d: int, expected: float) -> None:
    r"""Small-``k`` slope: ``psi(k)`` is near-unbiased where ``log(k)`` is not.

    On a uniform ``d``-cube the relation :math:`\langle\log r_k\rangle =
    D^{-1}\psi(k) + c` is exact, so a straight-line slope of :math:`\psi(k)` (vs
    :math:`\langle\log r_k\rangle`) over a *small-mass* window recovers ``d``.  The
    pre-fix abscissa :math:`\log k` grows faster than :math:`\psi(k)` at small
    ``k``, biasing that same slope downward — the bug this ticket fixes.  We drive
    ``fixed_mass_dimension`` to expose its per-mass curve, then read both slopes
    off the *same* :math:`\langle\log r_k\rangle` data so only the abscissa
    differs.
    """
    pts = _uniform_cube(d, 8000, seed=2)
    ks = np.array([1, 2, 3, 4, 5, 6, 8, 11, 15, 21, 30, 42], dtype=int)
    res = fixed_mass_dimension(pts, ks=ks, theiler=0, min_window=5)

    # res.abscissa is <log r_k>; res.ordinate is psi(k) — both sorted ascending.
    x = np.asarray(res.abscissa, dtype=float)  # <log r_k>
    y_psi = np.asarray(res.ordinate, dtype=float)  # psi(k)
    # ks sorted to the same order as the curve (ascending <log r_k> == ascending k)
    y_log = np.sort(np.log(ks.astype(float)))

    def _slope(xx: np.ndarray, yy: np.ndarray) -> float:
        coeffs = np.polyfit(xx, yy, 1)
        return float(coeffs[0])

    # small-mass window: the first six masses (k <= 6) — where the bias bites.
    s = slice(0, 6)
    slope_psi = _slope(x[s], y_psi[s])
    slope_log = _slope(x[s], y_log[s])

    err_psi = abs(slope_psi - expected)
    err_log = abs(slope_log - expected)

    # the digamma estimate is close to the true dimension on the small-k window ...
    assert err_psi < 0.12, f"{d}-cube small-k psi slope = {slope_psi:.3f} (err {err_psi:.3f})"
    # ... and it is strictly less biased than the log(k) estimate it replaced.
    assert err_psi < err_log, (
        f"{d}-cube: digamma slope err {err_psi:.3f} not below log(k) err {err_log:.3f}"
    )
    # the log(k) bias is real and downward (sanity-anchors the regression direction).
    assert slope_log < slope_psi


def test_top_level_export() -> None:
    """The fixed mass estimator stays reachable under the curated top level."""
    assert ts.fixed_mass_dimension is fixed_mass_dimension
