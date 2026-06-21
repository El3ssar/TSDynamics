"""Regression: DDE Lyapunov QR-norm floor (ticket FIX-DDELYAP-WINDOW, part 2).

The function-space Benettin renormalisation in
:func:`tsdynamics.families._dde_lyapunov._qr_segments` accumulates
``log|diag R|`` each chunk.  A rank-deficient or collapsed deviation segment
produces a (near-)zero ``R`` diagonal entry; the unfloored ``log(0) = -inf``
both poisons the time-average and — fatally, since the suite runs under
``filterwarnings = ["error"]`` (see ``pyproject.toml``) — raises a
``RuntimeWarning: divide by zero encountered in log`` that aborts the estimate.

The fix floors ``|R_ii|`` at ``numpy.finfo(float64).tiny`` before the log.  These
tests are *failing-first* against the pre-fix code: ``test_..._under_warnings_error``
errored out (the warning became an exception), and the floor is asserted to be
answer-preserving for a healthy, well-conditioned segment so it never perturbs a
converged spectrum.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("tsdynamics._rust")

from tsdynamics.families._dde_lyapunov import _qr_segments


def _rank_deficient_segments() -> np.ndarray:
    """``(n_seg+1, n_exp, dim)`` deviations where one deviation is identically zero.

    The zero deviation collapses to a zero ``R`` diagonal under QR — the exact
    condition that yields ``log(0)`` in the growth accumulation.
    """
    n_seg, n_exp, dim = 8, 2, 1
    dev = np.zeros((n_seg + 1, n_exp, dim))
    dev[:, 0, 0] = np.cos(np.linspace(0.0, np.pi, n_seg + 1))  # healthy deviation
    # dev[:, 1, 0] stays identically zero -> rank-deficient column
    return dev


def test_qr_floor_keeps_log_growth_finite_under_warnings_error() -> None:
    """A collapsed deviation must not raise ``divide by zero`` nor return ``-inf``.

    Failing-first: the pre-fix ``np.log(np.abs(np.diag(r)))`` raised a
    ``RuntimeWarning`` here, which is promoted to an error under the suite's
    ``filterwarnings = ["error"]`` profile.  The floor returns a large *finite*
    negative growth (``~log(tiny) = -708``) with no warning.
    """
    dev = _rank_deficient_segments()
    n_exp, dim = dev.shape[1], dev.shape[2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning -> test failure
        seg, log_growth = _qr_segments(dev, n_exp, dim)
    assert log_growth.shape == (n_exp,)
    assert np.all(np.isfinite(log_growth)), log_growth
    # The collapsed deviation is floored at log(tiny), not -inf.
    assert log_growth.min() <= np.log(np.finfo(np.float64).tiny) + 1.0
    assert log_growth.min() > -np.inf


def test_qr_floor_is_answer_preserving_for_a_healthy_segment() -> None:
    """The floor is a pure guard: a well-conditioned segment is untouched.

    For ``|R_ii| >> tiny`` the floored ``log`` equals the raw ``log|diag R|`` to
    machine precision, so the guard never perturbs a converged spectrum (this is
    what keeps ``interp == jit`` bit-for-bit).
    """
    rng = np.random.default_rng(0)
    n_seg, n_exp, dim = 12, 3, 2
    dev = rng.standard_normal((n_seg + 1, n_exp, dim))  # full-rank, well-conditioned
    seg, log_growth = _qr_segments(dev, n_exp, dim)
    # Recompute the raw (unfloored) growth from the same QR to compare.
    mat = dev.transpose(0, 2, 1).reshape((n_seg + 1) * dim, n_exp)
    _, r = np.linalg.qr(mat)
    raw = np.log(np.abs(np.diag(r)))
    assert np.all(np.isfinite(raw))  # sanity: the healthy case never hit the floor
    np.testing.assert_array_equal(log_growth, raw)
