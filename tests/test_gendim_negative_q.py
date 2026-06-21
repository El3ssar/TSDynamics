r"""Regression: ``generalized_dimension`` must reject negative Rényi order ``q``.

A box-counting :math:`D_q` for :math:`q < 0` is dominated by the rarely-visited,
under-sampled boxes — :math:`p_i^q` with :math:`p_i \ll 1` and :math:`q < 0`
blows up — so the partition-function estimate is unreliable and divergent in
practice; the regime the fixed-mass estimators were designed for instead
(Badii & Politi, *Phys. Rev. Lett.* **52**, 1661, 1984; Hentschel & Procaccia,
*Physica D* **8**, 435, 1983).  Silently returning a garbage slope is a headline
wrong-input footgun, so the estimator now rejects ``q < 0`` with the canonical
:class:`~tsdynamics.errors.InvalidParameterError`.

Ticket FIX-GENDIM-NEGQ.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidParameterError


def _uniform_2cube(n: int = 1500, seed: int = 0) -> np.ndarray:
    """``n`` i.i.d. uniform points in the unit square (a clean ``D_q = 2`` set)."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, 2))


def test_generalized_dimension_rejects_negative_q() -> None:
    """``q = -2`` on a uniform 2-cube raises, instead of returning a garbage slope."""
    pts = _uniform_2cube()
    with pytest.raises(InvalidParameterError) as excinfo:
        ts.generalized_dimension(pts, q=-2.0)
    # Value-naming standard: the message names the offending parameter and value.
    msg = str(excinfo.value)
    assert "q" in msg
    assert "-2" in msg


def test_negative_q_error_is_a_valueerror() -> None:
    """``InvalidParameterError`` still subclasses ``ValueError`` (callers catching
    ``ValueError`` keep working)."""
    assert issubclass(InvalidParameterError, ValueError)
    pts = _uniform_2cube()
    with pytest.raises(ValueError):
        ts.generalized_dimension(pts, q=-2.0)


def test_dimension_spectrum_rejects_a_negative_order() -> None:
    """A negative entry anywhere in ``qs`` is rejected (not silently computed)."""
    pts = _uniform_2cube()
    with pytest.raises(InvalidParameterError):
        ts.dimension_spectrum(pts, qs=[0.0, 1.0, -2.0])


@pytest.mark.parametrize("q", [0.0, 1.0, 2.0])
def test_positive_q_unchanged(q: float) -> None:
    """The ``q >= 0`` path is untouched: a uniform 2-cube still gives ``D_q ~= 2``."""
    pts = _uniform_2cube(n=2000, seed=1)
    dq = float(ts.generalized_dimension(pts, q=q))
    assert np.isfinite(dq)
    # Uniform 2-cube: D_q ~= 2 for every order; finite-N box-counting band.
    assert abs(dq - 2.0) < 0.35


def test_q_zero_boundary_is_allowed() -> None:
    """``q = 0`` is the box-counting dimension — the boundary must NOT be rejected."""
    pts = _uniform_2cube(n=2000, seed=2)
    d0 = float(ts.box_counting_dimension(pts))
    assert abs(d0 - 2.0) < 0.35
