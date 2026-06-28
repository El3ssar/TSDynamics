"""
Shape-preserving signal conditioning: detrend, normalize, and filter.

Each function takes a :class:`~tsdynamics.data.Trajectory` (or array) and returns
an object of the *same* type and shape — a ``Trajectory`` in yields a new
``Trajectory`` carrying the original time base, system back-reference and
provenance (plus a note of the transform), mirroring
:meth:`tsdynamics.data.Trajectory.standardize`; an array in yields an array out.
Time is taken to run along axis 0, so multi-channel ``(T, channels)`` signals are
conditioned channel-by-channel.

The filters are Butterworth designs (Butterworth 1930) applied as second-order
sections with zero-phase forward-backward filtering, so they introduce **no**
phase delay — important when a filtered signal is fed back into phase-sensitive
analysis (embeddings, recurrence).

References
----------
Butterworth, S. (1930). On the theory of filter amplifiers.
*Experimental Wireless and the Wireless Engineer*, 7, 536-541.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy import signal as _sig

from ..errors import invalid_value
from ._common import resolve_fs, to_signal, wrap_like

__all__ = [
    "bandpass",
    "bandstop",
    "butter_filter",
    "detrend",
    "highpass",
    "lowpass",
    "normalize",
]

_TINY = np.finfo(float).tiny


@lru_cache(maxsize=256)
def _design_butter_sos(
    order: int, cutoff_key: float | tuple[float, ...], btype: str, fs: float
) -> np.ndarray:
    """Design (and cache) a Butterworth SOS for a ``(order, cutoff, btype, fs)`` key.

    :func:`scipy.signal.butter` does a fresh pole/zero placement on every call;
    the design depends only on these four scalars, so caching it lets repeated
    filtering (the same filter swept over many signals, e.g. an ensemble) skip
    the redesign.  ``cutoff`` is passed through as ``list(cutoff_key)`` for a
    pair so SciPy sees exactly the original argument, and the returned SOS is
    marked read-only so a cached array can never be mutated in place by a caller.
    """
    cutoff: Any = list(cutoff_key) if isinstance(cutoff_key, tuple) else cutoff_key
    sos: np.ndarray = np.asarray(_sig.butter(order, cutoff, btype=btype, fs=fs, output="sos"))
    sos.flags.writeable = False
    return sos


def detrend(data: Any, *, method: str = "linear") -> Any:
    """
    Remove a constant or linear trend from each channel.

    Parameters
    ----------
    data : Trajectory or array-like
        Signal with time along axis 0.
    method : {"linear", "constant"}, default "linear"
        ``"linear"`` subtracts a least-squares straight line; ``"constant"``
        subtracts the mean.

    Returns
    -------
    Trajectory or ndarray
        Same type and shape as ``data``, detrended.
    """
    if method not in ("linear", "constant"):
        raise ValueError(f"detrend method must be 'linear' or 'constant', got {method!r}.")
    sig = to_signal(data)
    out = _sig.detrend(sig, axis=0, type=method)
    return wrap_like(data, out, detrended=method)


def normalize(data: Any, *, method: str = "zscore") -> Any:
    """
    Rescale each channel by a per-channel statistic.

    Parameters
    ----------
    data : Trajectory or array-like
        Signal with time along axis 0.
    method : {"zscore", "minmax", "l2", "demean"}, default "zscore"
        - ``"zscore"`` — subtract the mean, divide by the standard deviation.
        - ``"minmax"`` — affinely map each channel onto ``[0, 1]``.
        - ``"l2"`` — divide by the Euclidean norm.
        - ``"demean"`` — subtract the mean only.

        Channels with zero spread (constant signals) are passed through their
        location shift unscaled rather than producing ``inf``/``nan``.

    Returns
    -------
    Trajectory or ndarray
        Same type and shape as ``data``, normalised.
    """
    sig = to_signal(data)
    if method == "zscore":
        loc = sig.mean(axis=0)
        scale = sig.std(axis=0)
    elif method == "minmax":
        loc = sig.min(axis=0)
        scale = sig.max(axis=0) - loc
    elif method == "l2":
        loc = 0.0
        scale = np.sqrt((sig**2).sum(axis=0))
    elif method == "demean":
        loc = sig.mean(axis=0)
        scale = 1.0
    else:
        raise ValueError(
            f"unknown normalize method {method!r}; use 'zscore', 'minmax', 'l2', or 'demean'."
        )
    scale = np.where(np.abs(scale) < _TINY, 1.0, scale)
    out = (sig - loc) / scale
    return wrap_like(data, out, normalized=method)


def butter_filter(
    x: Any,
    cutoff: float | tuple[float, float] | Any,
    *,
    btype: str,
    fs: float | None = None,
    dt: float | None = None,
    order: int = 4,
) -> Any:
    """
    Zero-phase Butterworth filter.

    Designs a Butterworth filter of the given order and applies it forward and
    backward (zero phase) as second-order sections.

    Parameters
    ----------
    x : Trajectory or array-like
        Signal with time along axis 0.
    cutoff : float or (float, float)
        Cutoff frequency in the same units as ``fs``; a scalar for ``"lowpass"``
        / ``"highpass"``, a ``(low, high)`` pair for ``"bandpass"`` / ``"bandstop"``.
    btype : {"lowpass", "highpass", "bandpass", "bandstop"}
        Filter band type.
    fs, dt : float, optional
        Sampling frequency / spacing (see :func:`tsdynamics.transforms.power_spectral_density`).
    order : int, default 4
        Butterworth order (per band edge).  Must be a positive integer.

    Returns
    -------
    Trajectory or ndarray
        Same type and shape as ``x``, filtered.

    Raises
    ------
    InvalidParameterError
        If ``order`` is not at least 1 (a :class:`ValueError`).
    ValueError
        If ``cutoff`` does not lie strictly between 0 and the Nyquist frequency,
        or the number of cutoff edges does not match ``btype``.

    Notes
    -----
    Zero-phase filtering doubles the effective order and needs the signal to be
    longer than the filter's padding (a few times ``order``); very short signals
    raise from :func:`scipy.signal.sosfiltfilt`.

    References
    ----------
    Butterworth, S. (1930). On the theory of filter amplifiers.
    *Experimental Wireless and the Wireless Engineer*, 7, 536-541.
    """
    if order < 1:
        raise invalid_value("order", order, rule="must be a positive integer (>= 1)")
    rate = resolve_fs(x, fs=fs, dt=dt)
    nyquist = 0.5 * rate
    edges = np.atleast_1d(np.asarray(cutoff, dtype=float))
    if np.any(edges <= 0.0) or np.any(edges >= nyquist):
        raise ValueError(
            f"cutoff {cutoff!r} must lie strictly between 0 and the Nyquist "
            f"frequency ({nyquist:g}); reduce the cutoff or raise fs."
        )
    if btype in ("bandpass", "bandstop") and edges.size != 2:
        raise ValueError(f"{btype} needs a (low, high) cutoff pair, got {cutoff!r}.")
    if btype in ("lowpass", "highpass") and edges.size != 1:
        raise ValueError(f"{btype} needs a single scalar cutoff, got {cutoff!r}.")

    sig = to_signal(x)
    # Cache the design on the four scalars it depends on.  ``edges`` is the
    # float view of ``cutoff`` already built for validation; for a scalar cutoff
    # SciPy sees a Python float (a 0-d ndarray would broadcast the same Wn), and
    # for a pair it sees ``list((low, high))`` — both byte-identical designs to
    # passing the raw ``cutoff``.
    cutoff_key: float | tuple[float, ...] = (
        float(edges[0]) if edges.size == 1 else tuple(edges.tolist())
    )
    # sosfiltfilt needs a writeable SOS buffer; the cached design is read-only,
    # so hand it a fresh copy (a tiny (n_sections, 6) array — negligible vs the
    # pole/zero placement we just skipped).
    sos = _design_butter_sos(order, cutoff_key, btype, float(rate)).copy()
    out = _sig.sosfiltfilt(sos, sig, axis=0)
    return wrap_like(x, out, filtered={"btype": btype, "cutoff": cutoff, "order": order})


def lowpass(data: Any, cutoff: float, **kwargs: Any) -> Any:
    """Zero-phase Butterworth low-pass filter (see :func:`butter_filter`)."""
    return butter_filter(data, cutoff, btype="lowpass", **kwargs)


def highpass(data: Any, cutoff: float, **kwargs: Any) -> Any:
    """Zero-phase Butterworth high-pass filter (see :func:`butter_filter`)."""
    return butter_filter(data, cutoff, btype="highpass", **kwargs)


def bandpass(data: Any, low: float, high: float, **kwargs: Any) -> Any:
    """Zero-phase Butterworth band-pass filter (see :func:`butter_filter`)."""
    return butter_filter(data, (low, high), btype="bandpass", **kwargs)


def bandstop(data: Any, low: float, high: float, **kwargs: Any) -> Any:
    """Zero-phase Butterworth band-stop filter (see :func:`butter_filter`)."""
    return butter_filter(data, (low, high), btype="bandstop", **kwargs)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
