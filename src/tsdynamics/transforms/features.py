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
from .spectral import (
    _dominant_frequency_of,
    _spectral_centroid_of,
    _spectral_entropy_of,
    dominant_frequency,
    power_spectral_density,
    spectral_centroid,
    spectral_entropy,
)

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
    """Zero-crossing rate: fraction of adjacent samples that change sign.

    A crossing is a flip of the :func:`numpy.signbit` partition (the ``< 0`` vs
    ``>= 0`` half-line).  Exact zeros — including negative zero (``-0.0``), for
    which ``np.signbit`` is ``True`` even though the value is ``0.0`` — are first
    normalised to ``+0.0`` so a ``-0.0`` sample cannot fabricate a crossing
    against its ``+0.0`` neighbours (the defect this guards against).
    """
    if sig.size < 2:
        return 0.0
    # Collapse -0.0 to +0.0 before signbit: numpy's signbit(-0.0) is True, so a
    # signal grazing zero from below (... +x, -0.0, +y ...) would otherwise count
    # two spurious crossings.  Adding 0.0 turns every signed zero into +0.0.
    sign_lt_zero = np.signbit(sig + 0.0)
    return float(np.count_nonzero(np.diff(sign_lt_zero))) / (sig.size - 1)


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


#: Spectral features that are all reductions of the *same* per-channel PSD.
#: :func:`extract_features` computes that PSD once per channel and applies each
#: requested reduction, instead of letting every feature recompute the spectrum
#: (the catalogue entries above still work standalone — they each compute their
#: own PSD — but the bulk path reuses one spectrum across all three).
_PSD_FEATURES = frozenset({"dominant_frequency", "spectral_centroid", "spectral_entropy"})


def _psd_feature_of(name: str, freqs: np.ndarray, psd: np.ndarray) -> float:
    """Apply one spectral reduction to an already-computed single-channel PSD.

    Mirrors the bare-call defaults of the matching catalogue cores
    (``_dominant_frequency``/``_spectral_centroid``/``_spectral_entropy``), which
    call the public spectral functions with only ``fs=`` — i.e. every other PSD /
    reduction option at its default — so the result is identical.
    """
    if name == "dominant_frequency":
        return float(_dominant_frequency_of(freqs, psd))
    if name == "spectral_centroid":
        return float(_spectral_centroid_of(freqs, psd))
    return float(_spectral_entropy_of(psd))


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

    # Validate every requested name up front (preserving the original error and
    # available-list) so the shared-PSD fast path below never half-fills ``out``.
    for name in names:
        if name not in FEATURE_FUNCTIONS:
            raise KeyError(f"unknown feature {name!r}; available: {feature_names()}.")

    # Compute the per-channel PSD ONCE and feed it to every requested spectral
    # reduction, instead of each of dominant_frequency / spectral_centroid /
    # spectral_entropy recomputing the same Welch spectrum (up to a 3× saving on
    # the dominant cost of this routine).  The PSD here matches each spectral
    # core's bare ``f(col, fs)`` call exactly (default method/window/etc.).
    psd_names = [name for name in names if name in _PSD_FEATURES]
    psd_values: dict[str, Any] = {}
    if psd_names:
        per_channel: dict[str, list[float]] = {name: [] for name in psd_names}
        for _, col in channel_iter(sig):
            freqs, psd = power_spectral_density(col, fs=rate)
            for name in psd_names:
                per_channel[name].append(_psd_feature_of(name, freqs, psd))
        for name in psd_names:
            vals = per_channel[name]
            psd_values[name] = vals[0] if sig.ndim == 1 else np.asarray(vals, dtype=float)

    out: dict[str, Any] = {}
    for name in names:
        if name in psd_values:
            out[name] = psd_values[name]
        else:
            out[name] = _per_channel(sig, FEATURE_FUNCTIONS[name], rate)
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

    A crossing is a flip of the ``< 0`` / ``>= 0`` partition between consecutive
    samples; the rate is the crossing count divided by ``n_samples - 1``, so it
    lies in ``[0, 1]``.  For a clean tone of frequency ``f`` sampled at ``fs`` it
    is approximately ``2 f / fs``.  Signed zeros are normalised (``-0.0`` is
    treated as ``+0.0``) so a sample grazing zero cannot fabricate a crossing.

    Parameters
    ----------
    x : Trajectory or array-like
        Signal with time along axis 0.

    Returns
    -------
    float or ndarray
        Scalar for a single channel, ``(channels,)`` otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 200, endpoint=False)
    >>> round(float(zero_crossing_rate(np.sin(2 * np.pi * 5 * t))), 2)
    0.05
    """
    sig = to_signal(x)
    return _per_channel(sig, _zcr, 1.0)


def hjorth_parameters(x: Any) -> dict[str, Any]:
    """
    Hjorth activity, mobility and complexity per channel (Hjorth 1970).

    The three descriptors are built from the variances of the signal ``x`` and
    its first / second discrete differences (``x'``, ``x''``):

    - **activity** — ``var(x)``, the signal's power.
    - **mobility** — ``sqrt(var(x') / var(x))``, the standard deviation of the
      slope relative to that of the amplitude; a proxy for the mean frequency.
    - **complexity** — ``mobility(x') / mobility(x)``, i.e.
      ``sqrt(var(x'') / var(x')) / sqrt(var(x') / var(x))``; how much the signal's
      frequency content departs from a pure tone.  It is ``1`` for an ideal
      sinusoid and grows as the spectrum broadens.

    The differences are taken with :func:`numpy.diff` (unit sample spacing), so
    mobility is in cycles per sample.  A constant channel (zero variance) has no
    shape and returns ``0`` for all three rather than dividing by zero.

    Parameters
    ----------
    x : Trajectory or array-like
        Signal with time along axis 0.

    Returns
    -------
    dict
        Keys ``"activity"``, ``"mobility"``, ``"complexity"``; each value is a
        ``float`` for a single channel or a ``(channels,)`` array otherwise.

    References
    ----------
    Hjorth, B. (1970). EEG analysis based on time domain properties.
    *Electroencephalography and Clinical Neurophysiology*, 29(3), 306-310.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 4000, endpoint=False)
    >>> round(hjorth_parameters(np.sin(2 * np.pi * 3 * t))["complexity"], 2)
    1.0
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
