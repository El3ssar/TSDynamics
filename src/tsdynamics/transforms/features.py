"""
Generic per-channel feature extraction.

Turns a :class:`~tsdynamics.data.Trajectory` (or array) into a small, named bag
of scalar descriptors — the bridge between a raw signal and feature-consuming
analysis (clustering attractors, screening sweeps, surrogate test statistics).
:func:`extract_features` evaluates a selection of named features channel by
channel; the catalogue lives in :data:`FEATURE_FUNCTIONS` and is extensible.

The catalogue mixes time-domain shape statistics (moments, RMS, peak-to-peak,
zero-crossing rate), the Hjorth descriptors (Hjorth 1970) — a compact
activity / mobility / complexity triple long used to characterise time series —
and frequency-domain summaries reused from :mod:`tsdynamics.transforms.spectral`.

References
----------
Hjorth, B. (1970). EEG analysis based on time domain properties.
*Electroencephalography and Clinical Neurophysiology*, 29(3), 306-310.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ._common import channel_iter, resolve_fs, to_signal
from ._result import FeatureSet
from .spectral import dominant_frequency, spectral_centroid, spectral_entropy

__all__ = [
    "FEATURE_FUNCTIONS",
    "FeatureSet",
    "extract_features",
    "feature_names",
    "feature_set",
    "hjorth_parameters",
    "zero_crossing_rate",
]


# --- scalar cores (operate on a single 1-D channel) -------------------------


def _mean(sig: np.ndarray, fs: float) -> float:
    return float(sig.mean())


def _std(sig: np.ndarray, fs: float) -> float:
    return float(sig.std())


def _var(sig: np.ndarray, fs: float) -> float:
    return float(sig.var())


def _rms(sig: np.ndarray, fs: float) -> float:
    return float(np.sqrt(np.mean(sig**2)))


def _ptp(sig: np.ndarray, fs: float) -> float:
    return float(np.ptp(sig))


def _central_moment(sig: np.ndarray, order: int) -> float:
    centered = sig - sig.mean()
    return float(np.mean(centered**order))


def _skewness(sig: np.ndarray, fs: float) -> float:
    m2 = _central_moment(sig, 2)
    if m2 <= 0.0:  # constant channel — no shape
        return 0.0
    return float(_central_moment(sig, 3) / m2**1.5)


def _kurtosis(sig: np.ndarray, fs: float) -> float:
    """Excess (Fisher) kurtosis — 0 for a Gaussian."""
    m2 = _central_moment(sig, 2)
    if m2 <= 0.0:
        return 0.0
    return _central_moment(sig, 4) / m2**2 - 3.0


def _zcr(sig: np.ndarray, fs: float) -> float:
    """Zero-crossing rate: fraction of adjacent samples that change sign."""
    if sig.size < 2:
        return 0.0
    # signbit splits at 0.0 consistently (no spurious crossing on exact zeros).
    return float(np.count_nonzero(np.diff(np.signbit(sig)))) / (sig.size - 1)


def _hjorth_triple(sig: np.ndarray) -> tuple[float, float, float]:
    """Return (activity, mobility, complexity) for a 1-D channel."""
    var_zero = sig.var()
    if var_zero <= 0.0:
        return 0.0, 0.0, 0.0
    d1 = np.diff(sig)
    var_d1 = d1.var()
    mobility = float(np.sqrt(var_d1 / var_zero))
    if var_d1 <= 0.0:
        return float(var_zero), mobility, 0.0
    d2 = np.diff(d1)
    mobility_d1 = np.sqrt(d2.var() / var_d1)
    complexity = float(mobility_d1 / np.sqrt(var_d1 / var_zero))
    return float(var_zero), mobility, complexity


def _hjorth_activity(sig: np.ndarray, fs: float) -> float:
    return _hjorth_triple(sig)[0]


def _hjorth_mobility(sig: np.ndarray, fs: float) -> float:
    return _hjorth_triple(sig)[1]


def _hjorth_complexity(sig: np.ndarray, fs: float) -> float:
    return _hjorth_triple(sig)[2]


def _dominant_frequency(sig: np.ndarray, fs: float) -> float:
    return float(dominant_frequency(sig, fs=fs))


def _spectral_centroid(sig: np.ndarray, fs: float) -> float:
    return float(spectral_centroid(sig, fs=fs))


def _spectral_entropy(sig: np.ndarray, fs: float) -> float:
    return float(spectral_entropy(sig, fs=fs))


#: The feature catalogue: name → ``f(channel_1d, fs) -> float``.  Insertion order
#: is the default column order of :func:`extract_features`.  Add an entry (here
#: or, from outside, ``FEATURE_FUNCTIONS["name"] = fn``) to extend it.
FEATURE_FUNCTIONS: dict[str, Callable[[np.ndarray, float], float]] = {
    "mean": _mean,
    "std": _std,
    "var": _var,
    "rms": _rms,
    "ptp": _ptp,
    "skewness": _skewness,
    "kurtosis": _kurtosis,
    "zero_crossing_rate": _zcr,
    "hjorth_activity": _hjorth_activity,
    "hjorth_mobility": _hjorth_mobility,
    "hjorth_complexity": _hjorth_complexity,
    "dominant_frequency": _dominant_frequency,
    "spectral_centroid": _spectral_centroid,
    "spectral_entropy": _spectral_entropy,
}


def feature_names() -> list[str]:
    """Return the names of every catalogued feature, in default column order."""
    return list(FEATURE_FUNCTIONS)


def _per_channel(sig: np.ndarray, fn: Callable[[np.ndarray, float], float], fs: float) -> Any:
    """Apply a scalar-core feature over channels, scalar for 1-D, array for 2-D."""
    values = [fn(col, fs) for _, col in channel_iter(sig)]
    if sig.ndim == 1:
        return values[0]
    return np.asarray(values, dtype=float)


def extract_features(
    data: Any,
    *,
    fs: float | None = None,
    dt: float | None = None,
    features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Compute a named bag of scalar features for each channel of a signal.

    Parameters
    ----------
    data : Trajectory or array-like
        Signal with time along axis 0; a multi-channel ``(T, channels)`` signal
        is summarised channel-by-channel.
    fs, dt : float, optional
        Sampling frequency / spacing, needed by the frequency-domain features
        (see :func:`tsdynamics.transforms.power_spectral_density`).
    features : sequence of str, optional
        Which features to compute (names from :data:`FEATURE_FUNCTIONS`).
        Defaults to all of them, in catalogue order.

    Returns
    -------
    dict
        Maps each requested feature name to its value: a ``float`` for a single
        channel, a ``(channels,)`` ``ndarray`` for a multi-channel signal.

    Examples
    --------
    >>> import numpy as np
    >>> feats = extract_features(np.random.default_rng(0).standard_normal(2000))
    >>> round(feats["mean"], 1)
    0.0
    """
    sig = to_signal(data)
    rate = resolve_fs(data, fs=fs, dt=dt)
    names = feature_names() if features is None else list(features)
    out: dict[str, Any] = {}
    for name in names:
        try:
            fn = FEATURE_FUNCTIONS[name]
        except KeyError:
            raise KeyError(f"unknown feature {name!r}; available: {feature_names()}.") from None
        out[name] = _per_channel(sig, fn, rate)
    return out


def feature_set(
    data: Any,
    *,
    fs: float | None = None,
    dt: float | None = None,
    features: Sequence[str] | None = None,
    variant: str = "bar",
) -> FeatureSet:
    """Compute the feature vector as a self-describing, plottable :class:`FeatureSet`.

    A result-typed wrapper over :func:`extract_features`: it computes the same
    named bag of scalar features and returns a
    :class:`~tsdynamics.transforms.FeatureSet` carrying them, so the vector can
    ``.plot()`` (a ``FEATURE_BARS`` chart over a categorical feature-name axis)
    or ``.to_dict()``.  Use :func:`extract_features` directly for the bare
    ``{name: value}`` dict.

    Parameters
    ----------
    data : Trajectory or array-like
        Signal with time along axis 0; a multi-channel ``(T, channels)`` signal
        is summarised channel-by-channel.
    fs, dt : float, optional
        Sampling frequency / spacing, needed by the frequency-domain features
        (see :func:`power_spectral_density`).
    features : sequence of str, optional
        Which features to compute (names from :data:`FEATURE_FUNCTIONS`).
        Defaults to all of them, in catalogue order.
    variant : {"bar", "radar", "parallel"}, default "bar"
        The plot variant, recorded in ``meta["variant"]``: ``"bar"`` draws a bar
        per feature, ``"radar"`` / ``"parallel"`` a line over the same
        categorical feature axis (polar / parallel-coordinate layout).

    Returns
    -------
    FeatureSet
        The named feature bag, plottable as a feature-bar chart.

    Examples
    --------
    >>> import numpy as np
    >>> fset = feature_set(np.random.default_rng(0).standard_normal(2000))
    >>> round(fset.features["mean"], 1)
    0.0
    """
    feats = extract_features(data, fs=fs, dt=dt, features=features)
    meta = {
        "transform": "feature_set",
        "fs": resolve_fs(data, fs=fs, dt=dt),
        "variant": variant,
    }
    return FeatureSet(features=feats, meta=meta)


def zero_crossing_rate(x: Any) -> Any:
    """
    Fraction of adjacent samples that change sign, per channel.

    Parameters
    ----------
    x : Trajectory or array-like
        Signal with time along axis 0.

    Returns
    -------
    float or ndarray
        Scalar for a single channel, ``(channels,)`` otherwise.
    """
    sig = to_signal(x)
    return _per_channel(sig, _zcr, 1.0)


def hjorth_parameters(x: Any) -> dict[str, Any]:
    """
    Hjorth activity, mobility and complexity per channel (Hjorth 1970).

    - **activity** — the signal variance (its power).
    - **mobility** — ``sqrt(var(x') / var(x))``, a mean-frequency proxy.
    - **complexity** — ``mobility(x') / mobility(x)``, how much the signal departs
      from a pure sinusoid (1.0 for a sine wave).

    Parameters
    ----------
    x : Trajectory or array-like
        Signal with time along axis 0.

    Returns
    -------
    dict
        Keys ``"activity"``, ``"mobility"``, ``"complexity"``; each value is a
        ``float`` for a single channel or a ``(channels,)`` array otherwise.
    """
    sig = to_signal(x)
    triples = [_hjorth_triple(col) for _, col in channel_iter(sig)]
    keys = ("activity", "mobility", "complexity")
    if sig.ndim == 1:
        return dict(zip(keys, triples[0], strict=True))
    cols = np.asarray(triples, dtype=float)  # (channels, 3)
    return {k: cols[:, i] for i, k in enumerate(keys)}


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
