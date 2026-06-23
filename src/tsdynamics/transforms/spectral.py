"""
Spectral estimators: power spectral density and the scalar features built on it.

These turn a time-domain :class:`~tsdynamics.data.Trajectory` (or array) into a
frequency-domain description.  :func:`power_spectral_density` is the workhorse —
a one-sided PSD by Welch's method (Welch 1967) or a raw periodogram — and the
remaining functions are scalar summaries of that spectrum that feed the analysis
layer: :func:`spectral_entropy` (Powell & Percival 1979), :func:`spectral_centroid`,
and :func:`dominant_frequency`.

All four accept a single signal or a multi-channel ``(T, channels)`` signal and,
for the scalar summaries, return a Python ``float`` for the former and a
``(channels,)`` array for the latter.

References
----------
Welch, P. D. (1967). The use of fast Fourier transform for the estimation of
power spectra: a method based on time averaging over short, modified
periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70-73.

Powell, G. E., & Percival, I. C. (1979). A spectral entropy method for
distinguishing regular and irregular motion of Hamiltonian systems.
*Journal of Physics A: Mathematical and General*, 12(11), 2053-2071.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as _sig

from ._common import resolve_fs, to_signal
from ._result import Spectrogram, Spectrum

__all__ = [
    "Spectrogram",
    "Spectrum",
    "dominant_frequency",
    "power_spectral_density",
    "power_spectrum",
    "spectral_centroid",
    "spectral_entropy",
    "spectrogram",
]


def power_spectral_density(
    data: Any,
    *,
    fs: float | None = None,
    dt: float | None = None,
    method: str = "welch",
    window: str = "hann",
    nperseg: int | None = None,
    detrend: str | bool = "constant",
    scaling: str = "density",
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided power spectral density of a signal.

    Parameters
    ----------
    data : Trajectory or array-like
        Signal with time along axis 0; a multi-channel ``(T, channels)`` signal
        is transformed channel-by-channel.
    fs, dt : float, optional
        Sampling frequency or spacing.  At most one; if neither is given and
        ``data`` is a :class:`~tsdynamics.data.Trajectory`, the rate is read off
        its time vector, otherwise ``fs = 1.0``.
    method : {"welch", "periodogram"}, default "welch"
        Welch's averaged-periodogram estimator (lower variance) or a single raw
        periodogram (full frequency resolution, higher variance).
    window : str, default "hann"
        Window passed to the estimator (any :func:`scipy.signal.get_window` name).
    nperseg : int, optional
        Welch segment length.  Defaults to ``min(256, n_samples)`` so short
        signals do not trigger a segment-length warning.  Ignored by
        ``method="periodogram"``.
    detrend : str or False, default "constant"
        Per-segment detrending (``"constant"`` removes the mean, ``"linear"`` a
        least-squares line, ``False`` disables it).
    scaling : {"density", "spectrum"}, default "density"
        ``"density"`` gives a PSD (units²/Hz), ``"spectrum"`` a power spectrum
        (units²).

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
        Frequency bins, in the same units as ``fs``.
    psd : ndarray, shape (n_freqs,) or (n_freqs, channels)
        Power at each frequency, one column per channel.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 2000, endpoint=False)
    >>> f, p = power_spectral_density(np.sin(2 * np.pi * 5 * t), fs=200.0)
    >>> round(float(f[np.argmax(p)]))    # dominant line at 5 Hz
    5
    """
    sig = to_signal(data)
    rate = resolve_fs(data, fs=fs, dt=dt)
    n = sig.shape[0]

    if method == "welch":
        seg = min(256, n) if nperseg is None else int(nperseg)
        freqs, psd = _sig.welch(
            sig,
            fs=rate,
            window=window,
            nperseg=seg,
            detrend=detrend,
            scaling=scaling,
            axis=0,
        )
    elif method == "periodogram":
        freqs, psd = _sig.periodogram(
            sig,
            fs=rate,
            window=window,
            detrend=detrend,
            scaling=scaling,
            axis=0,
        )
    else:
        raise ValueError(f"unknown PSD method {method!r}; use 'welch' or 'periodogram'.")

    return freqs, psd


def spectral_entropy(
    data: Any,
    *,
    fs: float | None = None,
    dt: float | None = None,
    normalize: bool = True,
    base: float = 2.0,
    **psd_kwargs: Any,
) -> Any:
    """
    Shannon entropy of the normalised power spectrum (Powell & Percival 1979).

    The PSD is normalised to a probability distribution over frequency,
    ``p_i = P_i / Σ P_i``, and its Shannon entropy ``H = -Σ p_i log_base p_i`` is
    returned.  With ``normalize=True`` the result is divided by ``log_base(n_freqs)``
    so it lands in ``[0, 1]``: ``0`` for a pure tone (all power in one bin), ``1``
    for a flat (white) spectrum.  A useful regular-vs-chaotic discriminator.

    Parameters
    ----------
    data : Trajectory or array-like
        Signal with time along axis 0.
    fs, dt : float, optional
        Sampling frequency / spacing (see :func:`power_spectral_density`).
    normalize : bool, default True
        Divide by ``log_base(n_freqs)`` to rescale onto ``[0, 1]``.
    base : float, default 2.0
        Logarithm base (2 → bits).
    **psd_kwargs
        Forwarded to :func:`power_spectral_density` (``method``, ``window``, ...).

    Returns
    -------
    float or ndarray
        Spectral entropy; scalar for a single channel, ``(channels,)`` otherwise.
    """
    _, psd = power_spectral_density(data, fs=fs, dt=dt, **psd_kwargs)
    psd = np.asarray(psd, dtype=float)
    total = psd.sum(axis=0)
    # A constant (zero-power) channel has no spectral content → entropy 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0.0, psd / total, 0.0)
        logp = np.where(p > 0.0, np.log(p) / np.log(base), 0.0)
    entropy = -(p * logp).sum(axis=0)
    if normalize:
        n_freqs = psd.shape[0]
        if n_freqs > 1:
            entropy = entropy / (np.log(n_freqs) / np.log(base))
    return float(entropy) if np.ndim(entropy) == 0 else entropy


def spectral_centroid(
    data: Any,
    *,
    fs: float | None = None,
    dt: float | None = None,
    **psd_kwargs: Any,
) -> Any:
    """
    Power-weighted mean frequency (the spectral "centre of mass").

    ``centroid = Σ f_i P_i / Σ P_i`` — the frequency about which the spectrum is
    balanced.

    Parameters
    ----------
    data : Trajectory or array-like
        Signal with time along axis 0.
    fs, dt : float, optional
        Sampling frequency / spacing (see :func:`power_spectral_density`).
    **psd_kwargs
        Forwarded to :func:`power_spectral_density`.

    Returns
    -------
    float or ndarray
        Centroid frequency; scalar for a single channel, ``(channels,)`` otherwise.
    """
    freqs, psd = power_spectral_density(data, fs=fs, dt=dt, **psd_kwargs)
    psd = np.asarray(psd, dtype=float)
    total = psd.sum(axis=0)
    weighted = (freqs[:, None] * psd if psd.ndim == 2 else freqs * psd).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        centroid = np.where(total > 0.0, weighted / total, 0.0)
    return float(centroid) if np.ndim(centroid) == 0 else centroid


def dominant_frequency(
    data: Any,
    *,
    fs: float | None = None,
    dt: float | None = None,
    exclude_dc: bool = True,
    **psd_kwargs: Any,
) -> Any:
    """
    Frequency carrying the most power.

    Parameters
    ----------
    data : Trajectory or array-like
        Signal with time along axis 0.
    fs, dt : float, optional
        Sampling frequency / spacing (see :func:`power_spectral_density`).
    exclude_dc : bool, default True
        Ignore the zero-frequency bin so a non-zero mean does not masquerade as
        the dominant component.
    **psd_kwargs
        Forwarded to :func:`power_spectral_density`.

    Returns
    -------
    float or ndarray
        Peak frequency; scalar for a single channel, ``(channels,)`` otherwise.
    """
    freqs, psd = power_spectral_density(data, fs=fs, dt=dt, **psd_kwargs)
    psd = np.asarray(psd, dtype=float)
    if exclude_dc and freqs.size > 1 and freqs[0] == 0.0:
        freqs = freqs[1:]
        psd = psd[1:]
    peak = freqs[np.argmax(psd, axis=0)]
    return float(peak) if np.ndim(peak) == 0 else peak


def power_spectrum(
    data: Any,
    *,
    fs: float | None = None,
    dt: float | None = None,
    markers: bool = True,
    **psd_kwargs: Any,
) -> Spectrum:
    """One-sided power spectrum as a self-describing, plottable :class:`Spectrum`.

    A thin result-typed wrapper over :func:`power_spectral_density`: it computes
    the same PSD and, by default, the dominant frequency and spectral centroid,
    and returns a :class:`~tsdynamics.transforms.Spectrum` carrying them — so the
    spectrum can ``.plot()`` (a ``POWER_SPECTRUM`` with the two features as
    reference lines) or ``.to_dict()``.  Use :func:`power_spectral_density`
    directly for the bare ``(freqs, psd)`` arrays.

    Parameters
    ----------
    data : Trajectory or array-like
        Signal with time along axis 0.
    fs, dt : float, optional
        Sampling frequency / spacing (see :func:`power_spectral_density`).
    markers : bool, default True
        Compute the dominant frequency and spectral centroid markers.  Pass
        ``False`` to skip the two extra PSD reductions.
    **psd_kwargs
        Forwarded to :func:`power_spectral_density` (``method``, ``window``,
        ``nperseg``, ``detrend``, ``scaling``).

    Returns
    -------
    Spectrum
        The PSD plus its spectral-feature markers.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 2000, endpoint=False)
    >>> spec = power_spectrum(np.sin(2 * np.pi * 5 * t), fs=200.0)
    >>> round(spec.dominant_frequency)
    5
    """
    freqs, psd = power_spectral_density(data, fs=fs, dt=dt, **psd_kwargs)
    dom: Any = None
    cen: Any = None
    if markers:
        dom = dominant_frequency(data, fs=fs, dt=dt, **psd_kwargs)
        cen = spectral_centroid(data, fs=fs, dt=dt, **psd_kwargs)
    scaling = str(psd_kwargs.get("scaling", "density"))
    meta = {
        "transform": "power_spectrum",
        "method": psd_kwargs.get("method", "welch"),
        "fs": resolve_fs(data, fs=fs, dt=dt),
        "scaling": scaling,
    }
    return Spectrum(
        frequencies=freqs,
        psd=psd,
        dominant_frequency=dom,
        spectral_centroid=cen,
        scaling=scaling,
        meta=meta,
    )


def spectrogram(
    data: Any,
    *,
    fs: float | None = None,
    dt: float | None = None,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    detrend: str | bool = "constant",
    scaling: str = "density",
) -> Spectrogram:
    """Short-time Fourier power spectrogram (time x frequency x power).

    Slides a window over the signal and computes a PSD per segment (a
    short-time Fourier transform), the standard view of how spectral content
    evolves in time.  A multi-channel signal is reduced to its **first** channel
    (a spectrogram is a single time--frequency image); transform each channel
    separately for a per-channel image.

    Parameters
    ----------
    data : Trajectory or array-like
        Signal with time along axis 0.  A ``(T, channels)`` signal uses its
        first channel.
    fs, dt : float, optional
        Sampling frequency or spacing.  At most one; if neither is given and
        ``data`` is a :class:`~tsdynamics.data.Trajectory`, the rate is read off
        its time vector, otherwise ``fs = 1.0`` (cycles per sample).
    window : str, default "hann"
        Segment window (any :func:`scipy.signal.get_window` name).
    nperseg : int, optional
        Segment length.  Defaults to ``min(256, n_samples)`` so short signals do
        not trigger a segment-length warning.
    noverlap : int, optional
        Samples of overlap between adjacent segments.  Defaults to
        ``nperseg // 8`` (SciPy's spectrogram default).
    detrend : str or False, default "constant"
        Per-segment detrending (``"constant"`` / ``"linear"`` / ``False``).
    scaling : {"density", "spectrum"}, default "density"
        ``"density"`` gives a PSD (units²/Hz), ``"spectrum"`` a power spectrum.

    Returns
    -------
    Spectrogram
        ``times`` (segment centres), ``frequencies``, and ``power`` of shape
        ``(n_freqs, n_times)`` — a plottable ``SPECTROGRAM`` image.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 4000, endpoint=False)
    >>> sg = spectrogram(np.sin(2 * np.pi * 5 * t), fs=400.0)
    >>> sg.power.shape == (sg.frequencies.size, sg.times.size)
    True

    References
    ----------
    Welch, P. D. (1967). The use of fast Fourier transform for the estimation of
    power spectra: a method based on time averaging over short, modified
    periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2),
    70-73.
    """
    sig = to_signal(data)
    rate = resolve_fs(data, fs=fs, dt=dt)
    channel = sig if sig.ndim == 1 else sig[:, 0]
    n = channel.shape[0]
    seg = min(256, n) if nperseg is None else int(nperseg)

    freqs, times, power = _sig.spectrogram(
        channel,
        fs=rate,
        window=window,
        nperseg=seg,
        noverlap=noverlap,
        detrend=detrend,
        scaling=scaling,
        mode="psd",
    )

    meta = {
        "transform": "spectrogram",
        "window": window,
        "nperseg": seg,
        "noverlap": seg // 8 if noverlap is None else int(noverlap),
        "fs": rate,
        "scaling": scaling,
    }
    return Spectrogram(times=times, frequencies=freqs, power=power, meta=meta)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
