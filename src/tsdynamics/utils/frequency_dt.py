from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

__all__ = ["DtFromSpectrum", "estimate_dt_from_spectrum"]

@dataclass(frozen=True)
class DtFromSpectrum:
    """
    Result for frequency-based Δt selection.
    """
    delta_t: float                 # Δt* = safety / (2 f_h)
    f_hz: float                    # highest significant frequency f_h
    f_quantile_hz: float           # energy-quantile cutoff
    f_snr_hz: float                # SNR-based cutoff
    safety: float                  # c
    q_power: float                 # power quantile used
    snr_db: float                  # SNR threshold used
    freqs_hz: np.ndarray           # Welch frequency grid (one-sided)
    psd_total: np.ndarray          # Welch total PSD (sum over dims, after σ-normalization)
    converged: bool                # True unless degenerate/flat spectrum
    notes: str = ""                # info / warnings


# ------------------------ Welch (multivariate) ------------------------

def _welch_multivar(
    y: np.ndarray,
    dt: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multivariate Welch PSD. Features are σ-normalized per column; PSDs are summed.
    Returns frequencies (Hz) and total PSD (units of normalized^2/Hz).
    """
    y = np.asarray(y, float)
    T, d = y.shape
    if nperseg is None:
        # ~8 segments target; power-of-two helps FFT
        nperseg = 1 << int(np.floor(np.log2(max(256, T // 8))))
        nperseg = min(nperseg, T)
    if nperseg < 16 or nperseg > T:
        nperseg = min(max(16, T // 4), T)
    if noverlap is None:
        noverlap = nperseg // 2

    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("noverlap must be < nperseg.")
    nseg = 1 + (T - nperseg) // step
    if nseg <= 0:
        # fallback: single segment
        nseg = 1
        step = 0
        noverlap = 0
        nperseg = T

    # window
    if window == "hann":
        w = np.hanning(nperseg)
    elif window == "boxcar":
        w = np.ones(nperseg)
    else:
        raise ValueError("Only 'hann' and 'boxcar' supported without SciPy.")
    w_norm = (w**2).sum()

    # frequency grid
    nfft = 1 << int(np.ceil(np.log2(nperseg)))
    freqs = np.fft.rfftfreq(nfft, d=dt)  # Hz
    psd_total = np.zeros(freqs.size, dtype=float)

    # σ-normalize features to avoid scale dominance
    std = y.std(axis=0, ddof=1)
    std[~np.isfinite(std) | (std == 0.0)] = 1.0
    y_norm = y / std

    # segment loop
    for s in range(nseg):
        i0 = s * step
        i1 = i0 + nperseg
        if i1 > T:
            break
        # sum PSD across dims (not magnitudes, PSD additivity holds per dim)
        Ssum = np.zeros(freqs.size, dtype=float)
        for j in range(y_norm.shape[1]):
            seg = y_norm[i0:i1, j]
            seg = seg - seg.mean()
            xw = seg * w
            X = np.fft.rfft(xw, n=nfft)
            S = (np.abs(X) ** 2) / (w_norm)   # power per bin (discrete-time)
            Ssum += S
        psd_total += Ssum

    # Average over segments and convert to density (per Hz): divide by fs = 1/dt
    fs = 1.0 / dt
    psd_total = psd_total / max(nseg, 1) / fs
    return freqs, psd_total


def _smooth_moving(y: np.ndarray, m: int = 8) -> np.ndarray:
    """Simple centered moving-average smoothing with reflective edges."""
    if m <= 1:
        return y
    pad = m // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kern = np.ones(m, dtype=float) / m
    out = np.convolve(yp, kern, mode="valid")
    return out[: y.size]


# -------------------- robust f_h selection (idempotent) --------------------

def estimate_dt_from_spectrum(
    y: np.ndarray,
    dt0: float,
    *,
    q_power: float = 0.99,          # energy quantile (e.g., 99% power below f_q)
    snr_db: float = 8.0,            # SNR threshold above noise floor in dB
    smooth_bins: int = 8,           # frequency smoothing bins for PSD
    safety: float = 0.9,            # c in Δt* = c / (2 f_h)
    guard_frac: float = 0.9,        # only trust PSD below 0.9*Nyquist (stability)
    min_bins: int = 64,             # minimum number of frequency bins for reliability
    enforce_coarsening: bool = True # never suggest Δt* < dt0
) -> DtFromSpectrum:
    """
    Frequency-based Δt selection with practical idempotence.

    Steps:
      1) Welch PSD on σ-normalized features; sum across dims.
      2) Smooth PSD; ignore top (1-guard_frac) near Nyquist to avoid edge bias.
      3) Two cutoffs:
         - Energy: smallest f with cumulative power ≥ q_power.
         - SNR: largest f where PSD(f) ≥ floor * 10^(snr_db/10),
                with floor = median PSD over the top 20% of freq (tail).
      4) f_h = max(f_energy, f_snr); Δt* = safety / (2 f_h).
         If no significant energy (flat/constant), set f_h→0 ⇒ Δt*→∞.
      5) If enforce_coarsening and Δt* < dt0, clamp to dt0.

    Idempotence rationale:
      - With safety<1 and low-pass at c*Nyq before resampling, the next run sees
        the same bandlimited spectrum below f_h, so f_h (and Δt*) stays put.

    Returns details for inspection.
    """
    y = np.asarray(y, float)
    if y.ndim != 2:
        raise ValueError("y must be (T, n_dim).")
    T, d = y.shape
    if T < 4 * min_bins:
        # PSD will be noisy; still proceed but warn
        pass

    # Welch PSD
    freqs, psd = _welch_multivar(y, dt0)
    if freqs.size < min_bins:
        # enlarge segment to get more bins
        freqs, psd = _welch_multivar(y, dt0, nperseg=max(256, min_bins*2))

    # Smooth and guard
    psd_s = _smooth_moving(psd, smooth_bins)
    f_nyq = 0.5 / dt0
    f_guard = guard_frac * f_nyq
    mask = freqs <= f_guard
    f = freqs[mask]
    S = psd_s[mask]

    # If the spectrum is degenerate or zero
    if not np.isfinite(S).any() or S.max() <= 0:
        return DtFromSpectrum(
            delta_t=np.inf,
            f_hz=0.0,
            f_quantile_hz=0.0,
            f_snr_hz=0.0,
            safety=float(safety),
            q_power=float(q_power),
            snr_db=float(snr_db),
            freqs_hz=freqs,
            psd_total=psd,
            converged=False,
            notes="Degenerate/zero spectrum."
        )

    # Energy-quantile cutoff (integral via trapezoid)
    df = np.diff(f).mean() if f.size > 1 else f_guard
    cum_power = np.cumsum(S) * df
    total_power = cum_power[-1]
    # numeric guard
    if total_power <= 0 or not np.isfinite(total_power):
        total_power = np.maximum(total_power, 1e-12)
    idx_q = np.searchsorted(cum_power, q_power * total_power)
    idx_q = np.clip(idx_q, 0, f.size - 1)
    f_q = float(f[idx_q])

    # SNR cutoff relative to tail noise floor (robust)
    tail_mask = f >= (0.8 * f_guard)
    floor = np.median(S[tail_mask]) if tail_mask.any() else np.median(S[-max(8, S.size//10):])
    thr = floor * (10.0 ** (snr_db / 10.0))
    above = np.where(S >= thr)[0]
    f_snr = float(f[above[-1]]) if above.size else 0.0

    # Highest significant frequency
    f_h = max(f_q, f_snr)
    notes = ""

    if f_h <= 0.0 or not np.isfinite(f_h):
        # Essentially flat/lowpass: allow huge Δt (no dynamic content)
        dt_star = np.inf
        notes = "No significant high-frequency content; Δt* unbounded."
    else:
        dt_star = safety / (2.0 * f_h)
        if enforce_coarsening and dt_star < dt0:
            dt_star = dt0
            notes = "Clamped to dt0 (coarsening-only)."

    return DtFromSpectrum(
        delta_t=float(dt_star),
        f_hz=float(f_h),
        f_quantile_hz=float(f_q),
        f_snr_hz=float(f_snr),
        safety=float(safety),
        q_power=float(q_power),
        snr_db=float(snr_db),
        freqs_hz=freqs,
        psd_total=psd,
        converged=True,
        notes=notes,
    )
