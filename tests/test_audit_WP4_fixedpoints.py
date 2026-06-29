r"""Regression tests for WP4_fixedpoints (audit findings A7-1 and A7-5).

A7-1: the interval root finder (``method="interval"``) must enclose flows whose
symengine RHS contains ``exp`` of a state variable (canonicalized to
``Pow(E, ...)``) and a constant fractional power such as ``sqrt(2)``
(``Pow(2, 1/2)``).  Before the fix these surfaced as ``InvalidInputError`` even
though the roots are well defined, because ``_eval_jet`` only accepted integer
powers and rejected the ``E`` base.

A7-5: the FFT period estimator applies a Hann window to suppress spectral
leakage, roughly halving the sub-bin period bias on a record whose true period
is not an integer divisor of the record length.
"""

from __future__ import annotations

import numpy as np

import tsdynamics as ts
from tsdynamics.analysis.fixedpoints._interval import flow_interval_fn
from tsdynamics.analysis.fixedpoints.periodic import _fft_period_lag


def _equilibrium_residual(system: ts.ContinuousSystem, x: np.ndarray) -> float:
    rhs = system._rhs_numeric()
    return float(np.linalg.norm(rhs(np.asarray(x, dtype=float), 0.0)))


def test_interval_encloses_exp_of_state() -> None:
    """A7-1: ``exp(state)`` (Pow(E, ...)) must build and find roots, not raise.

    Pre-fix, ``flow_interval_fn`` raised ``InvalidInputError`` on the
    ``exp(y(0)*y(1))`` term of YuWang because the ``E`` base of the ``Pow`` node
    was unmodelled.
    """
    sys = ts.systems.YuWang()
    # Build the interval residual+Jacobian (this is the build-time probe that
    # used to raise pre-fix).
    flow_interval_fn(sys)  # must not raise

    region = ts.Box([-3.0] * sys.dim, [3.0] * sys.dim)
    fps = list(ts.fixed_points(sys, method="interval", region=region))
    assert len(fps) >= 1
    for p in fps:
        assert _equilibrium_residual(sys, p.x) < 1e-6


def test_interval_encloses_constant_sqrt_coefficient() -> None:
    """A7-1: a constant ``sqrt(2)`` coefficient (Pow(2, 1/2)) must fold, not raise.

    NuclearQuadrupole carries ``0.825*sqrt(2)*y(0)**2`` terms; pre-fix the
    ``Pow(2, 1/2)`` node tripped the ``non-integer power`` rejection.
    """
    sys = ts.systems.NuclearQuadrupole()
    flow_interval_fn(sys)  # must not raise

    region = ts.Box([-3.0] * sys.dim, [3.0] * sys.dim)
    fps = list(ts.fixed_points(sys, method="interval", region=region))
    assert len(fps) >= 1
    for p in fps:
        assert _equilibrium_residual(sys, p.x) < 1e-6


def test_interval_polynomial_flow_unaffected() -> None:
    """A7-1 regression guard: a pure-polynomial flow is answer-preserving.

    Lorenz has integer powers only; the exp/sqrt additions must not perturb its
    three equilibria.
    """
    sys = ts.systems.Lorenz()
    region = ts.Box([-30.0] * 3, [30.0] * 3)
    pts = sorted(
        tuple(round(c, 5) for c in p.x)
        for p in list(ts.fixed_points(sys, method="interval", region=region))
    )
    # origin + the symmetric pair C± = (±√(b(r−1)), ±√(b(r−1)), r−1), r=28, b=8/3
    c = float(np.sqrt(8.0 / 3.0 * 27.0))
    expected = sorted(
        [
            (0.0, 0.0, 0.0),
            (round(c, 5), round(c, 5), 27.0),
            (round(-c, 5), round(-c, 5), 27.0),
        ]
    )
    assert pts == expected


def _nowindow_fft_period(y: np.ndarray) -> float:
    """Reproduce the pre-fix (un-windowed) FFT period estimate."""
    n = y.size
    spec = np.abs(np.fft.rfft(y)) ** 2
    spec[0] = 0.0
    k = int(np.argmax(spec))
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    denom = a - 2.0 * b + c
    k_ref = k + 0.5 * (a - c) / denom if denom != 0.0 else float(k)
    return float(n / k_ref)


def test_fft_hann_window_reduces_period_bias() -> None:
    """A7-5: the Hann window roughly halves sub-bin period bias on a non-commensurate record."""
    n = 1000
    true_p = 40.5  # not an integer divisor of n -> leakage-biased without a taper
    t = np.arange(n)
    y = np.sin(2.0 * np.pi * t / true_p)

    hann_lag, _, _, _ = _fft_period_lag(y, 1.0)
    nowin_lag = _nowindow_fft_period(y)

    err_hann = abs(hann_lag - true_p)
    err_nowin = abs(nowin_lag - true_p)
    # The windowed estimate is strictly better, by at least a meaningful margin.
    assert err_hann < err_nowin
    assert err_hann < 0.6 * err_nowin
    assert err_hann < 0.25  # closing-note verified ~0.19
