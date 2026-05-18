"""Peak detection and inter-spike-interval primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks

from .._registry import trajectory_op


def _extract_component(y: np.ndarray, component: int) -> np.ndarray:
    """Return ``y[:, component]`` with bounds-checking."""
    dim = y.shape[1]
    if not isinstance(component, int | np.integer):
        raise TypeError(f"component must be an integer, got {type(component).__name__}")
    if component < -dim or component >= dim:
        raise IndexError(f"component {component} out of range for state dim {dim}")
    return y[:, component]


def _hermite_state(s, y_a, y_b, m_a, m_b, dt):
    """Local cubic-Hermite interpolant."""
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * y_a + h10 * dt * m_a + h01 * y_b + h11 * dt * m_b


def _hermite_slope(s, y_a, y_b, m_a, m_b, dt):
    """Time-derivative of the cubic-Hermite interpolant."""
    h00p = 6.0 * s * s - 6.0 * s
    h10p = 3.0 * s * s - 4.0 * s + 1.0
    h01p = -6.0 * s * s + 6.0 * s
    h11p = 3.0 * s * s - 2.0 * s
    return (h00p * y_a + h01p * y_b) / dt + h10p * m_a + h11p * m_b


def _refine_extrema(
    t: np.ndarray,
    y: np.ndarray,
    component: int,
    *,
    kind: str,
    find_peaks_kwargs: dict,
    rtol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sub-sample peak/trough refinement via cubic-Hermite derivative roots.

    Locates extrema of ``y[:, component]`` by:
    1. Finding integer-sample peaks via :func:`scipy.signal.find_peaks` so
       the user's ``prominence`` / ``distance`` / etc. kwargs are honoured.
    2. For each found peak at index ``k``, building the cubic-Hermite
       interpolant on the bracket ``[k-1, k+1]`` and locating the
       analytical root of its time-derivative.

    Returns ``(t_peaks, y_peaks)`` with shapes ``(K,)`` and ``(K,)``.
    """
    sig = _extract_component(y, component)
    work = -sig if kind == "min" else sig
    idx, _ = find_peaks(work, **find_peaks_kwargs)
    if idx.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    slopes = np.gradient(y, t, axis=0)
    t_peaks: list[float] = []
    y_peaks: list[float] = []
    for k in idx:
        if k <= 0 or k >= t.size - 1:
            t_peaks.append(float(t[k]))
            y_peaks.append(float(sig[k]))
            continue

        k_lo, k_hi = k - 1, k
        t_a, t_b = float(t[k_lo]), float(t[k_hi])
        dt_ab = t_b - t_a
        y_a, y_b = y[k_lo], y[k_hi]
        m_a, m_b = slopes[k_lo], slopes[k_hi]

        def dydt_left(
            s: float,
            dt: float = dt_ab,
            y_a: np.ndarray = y_a,
            y_b: np.ndarray = y_b,
            m_a: np.ndarray = m_a,
            m_b: np.ndarray = m_b,
            c: int = component,
        ) -> float:
            return float(_hermite_slope(s, y_a, y_b, m_a, m_b, dt)[c])

        d_lo_left = dydt_left(0.0)
        d_hi_left = dydt_left(1.0)

        k2_lo, k2_hi = k, k + 1
        t_a2, t_b2 = float(t[k2_lo]), float(t[k2_hi])
        dt_ab2 = t_b2 - t_a2
        y_a2, y_b2 = y[k2_lo], y[k2_hi]
        m_a2, m_b2 = slopes[k2_lo], slopes[k2_hi]

        def dydt_right(
            s: float,
            dt: float = dt_ab2,
            y_a: np.ndarray = y_a2,
            y_b: np.ndarray = y_b2,
            m_a: np.ndarray = m_a2,
            m_b: np.ndarray = m_b2,
            c: int = component,
        ) -> float:
            return float(_hermite_slope(s, y_a, y_b, m_a, m_b, dt)[c])

        d_lo_right = dydt_right(0.0)
        d_hi_right = dydt_right(1.0)

        if d_lo_left * d_hi_left <= 0.0:
            s_star = (
                0.0
                if d_lo_left == 0.0
                else 1.0
                if d_hi_left == 0.0
                else brentq(dydt_left, 0.0, 1.0, rtol=rtol, maxiter=100)
            )
            t_star = t_a + s_star * dt_ab
            y_star = _hermite_state(s_star, y_a, y_b, m_a, m_b, dt_ab)
        elif d_lo_right * d_hi_right <= 0.0:
            s_star = (
                0.0
                if d_lo_right == 0.0
                else 1.0
                if d_hi_right == 0.0
                else brentq(dydt_right, 0.0, 1.0, rtol=rtol, maxiter=100)
            )
            t_star = t_a2 + s_star * dt_ab2
            y_star = _hermite_state(s_star, y_a2, y_b2, m_a2, m_b2, dt_ab2)
        else:
            t_star = float(t[k])
            y_star = y[k]

        t_peaks.append(float(t_star))
        y_peaks.append(float(y_star[component]))

    return np.asarray(t_peaks, dtype=float), np.asarray(y_peaks, dtype=float)


@trajectory_op(returns="trajectory")
def local_maxima(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    *,
    refined: bool = False,
    **find_peaks_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate local maxima of one state component.

    By default uses :func:`scipy.signal.find_peaks` (sample-accurate);
    pass ``refined=True`` for sub-sample-accurate times and heights via
    cubic-Hermite refinement (same algorithm that backs
    :func:`detect_events`).

    Parameters
    ----------
    component : int, default 0
    refined : bool, default ``False``
        Refine peak times/heights below sample resolution.
    **find_peaks_kwargs
        Forwarded to :func:`scipy.signal.find_peaks`
        (``prominence``, ``distance``, ``height``, …).

    Returns
    -------
    Trajectory
        ``t`` = peak times; ``y`` has shape ``(K, 1)`` — peak heights of
        the chosen component.

    Examples
    --------
    >>> peaks = traj.local_maxima(component=2, prominence=1.0)
    >>> peaks.t.shape, peaks.y.shape
    ((K,), (K, 1))
    """
    if refined:
        tp, yp = _refine_extrema(t, y, component, kind="max", find_peaks_kwargs=find_peaks_kwargs)
    else:
        sig = _extract_component(y, component)
        idx, _ = find_peaks(sig, **find_peaks_kwargs)
        tp = t[idx].copy()
        yp = sig[idx].copy()
    return tp, yp.reshape(-1, 1)


@trajectory_op(returns="trajectory")
def local_minima(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    *,
    refined: bool = False,
    **find_peaks_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate local minima of one state component.

    See :func:`local_maxima` for parameter meanings; this is the same
    primitive applied to ``-y[:, component]``.

    Returns
    -------
    Trajectory
        ``t`` = trough times; ``y`` has shape ``(K, 1)`` — trough heights.
    """
    if refined:
        tp, yp = _refine_extrema(t, y, component, kind="min", find_peaks_kwargs=find_peaks_kwargs)
    else:
        sig = _extract_component(y, component)
        idx, _ = find_peaks(-sig, **find_peaks_kwargs)
        tp = t[idx].copy()
        yp = sig[idx].copy()
    return tp, yp.reshape(-1, 1)


@trajectory_op(returns="trajectory")
def return_times(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    **find_peaks_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inter-peak intervals on one state component — the "poor man's period".

    For a clean ``sin(ω t)`` every entry is close to ``2π / ω``.  The
    proper spectral version lands in M7.

    Parameters
    ----------
    component : int, default 0
    **find_peaks_kwargs
        Forwarded to :func:`scipy.signal.find_peaks`.

    Returns
    -------
    Trajectory
        ``t`` = times of the *first* peak in each pair (length ``K - 1``);
        ``y`` has shape ``(K - 1, 1)`` — the ISI values.  Empty trajectory
        if fewer than two peaks were found.
    """
    sig = _extract_component(y, component)
    idx, _ = find_peaks(sig, **find_peaks_kwargs)
    if idx.size < 2:
        return np.empty(0, dtype=float), np.empty((0, 1), dtype=float)
    tp = t[idx]
    isi = np.diff(tp)
    return tp[:-1].copy(), isi.reshape(-1, 1)


__all__ = ["local_maxima", "local_minima", "return_times"]
