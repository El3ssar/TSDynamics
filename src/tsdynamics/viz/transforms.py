"""Numerical transforms for visualization (no plotting here)."""

from __future__ import annotations
from typing import Iterable, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import welch, find_peaks
from scipy.spatial.distance import cdist


# ------------------------- Basic helpers -------------------------

def take_columns(Y: ArrayLike, cols: Iterable[int]) -> np.ndarray:
    """Return selected columns from trajectory Y (T, N)."""
    Y = np.asarray(Y, float)
    cols = list(cols)
    return Y[:, cols]


def autocorrelation(x: ArrayLike, max_lag: int) -> np.ndarray:
    """Unbiased autocorrelation up to max_lag."""
    x = np.asarray(x, float) - np.mean(x)
    n = x.size
    r = np.asarray([np.dot(x[:n - k], x[k:]) / (n - k) for k in range(max_lag + 1)])
    return r / r[0]


def delay_embedding(x: ArrayLike, m: int, tau: int) -> np.ndarray:
    """
    Takens delay embedding of a 1D series x into R^m with lag tau.
    Returns matrix shape (T_eff, m).
    """
    x = np.asarray(x, float)
    if m < 1 or tau < 1:
        raise ValueError("m and tau must be >= 1")
    T = x.size - (m - 1) * tau
    if T <= 0:
        raise ValueError("Time series too short for the requested embedding.")
    return np.column_stack([x[i:i + T] for i in range(0, m * tau, tau)])


def return_map(x: ArrayLike, lag: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_t, x_{t+lag}) pairs for a scalar series."""
    x = np.asarray(x, float)
    if lag < 1:
        raise ValueError("lag must be >= 1")
    return x[:-lag], x[lag:]


# ------------------------- Spectral analysis -------------------------

def power_spectrum_fft(x: ArrayLike, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Single-sided FFT power spectrum."""
    x = np.asarray(x, float)
    n = x.size
    X = np.fft.rfft(x - np.mean(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(X) ** 2) / (n * fs)
    return freqs, psd


def power_spectrum_welch(x: ArrayLike, fs: float = 1.0, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD."""
    x = np.asarray(x, float)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg or max(256, len(x)//8))
    return f, Pxx


def wavelet_scalogram(x: ArrayLike, fs: float, wavelet: str = "morl", widths: Optional[np.ndarray] = None):
    """
    Continuous wavelet scalogram using PyWavelets (optional dependency).
    Returns (t, scales, |CWT|).
    """
    try:
        import pywt  # optional
    except Exception as e:
        raise ImportError("wavelet_scalogram requires PyWavelets (`pywt`).") from e

    x = np.asarray(x, float)
    if widths is None:
        # default logarithmic widths
        widths = np.geomspace(1, max(8, len(x)//8), 64)
    cwtmatr, scales = pywt.cwt(x, widths, wavelet, sampling_period=1.0/fs)
    t = np.arange(x.size) / fs
    return t, scales, np.abs(cwtmatr)


# ------------------------- Recurrence analysis -------------------------

def recurrence_matrix(X: ArrayLike, eps: Optional[float] = None, percent: Optional[float] = 10.0,
                      metric: str = "euclidean", normalize: bool = True) -> np.ndarray:
    """
    Binary recurrence matrix R[i,j] (1 if distance <= threshold).
    If eps is None, choose threshold so that `percent`% of entries are 1.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    D = cdist(X, X, metric=metric)
    if normalize:
        D /= (np.nanmax(D) + 1e-12)
    if eps is None:
        if percent is None:
            raise ValueError("Either eps or percent must be given.")
        k = int(np.clip(percent / 100.0 * D.size, 1, D.size))
        eps = np.partition(D.ravel(), k - 1)[k - 1]
    return (D <= eps).astype(np.uint8)


def cross_recurrence_matrix(X: ArrayLike, Y: ArrayLike, eps: Optional[float] = None,
                            percent: Optional[float] = 10.0, metric: str = "euclidean",
                            normalize: bool = True) -> np.ndarray:
    """Binary cross-recurrence of two sequences."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.ndim == 1: 
        X = X[:, None]
    if Y.ndim == 1: 
        Y = Y[:, None]
    D = cdist(X, Y, metric=metric)
    if normalize:
        D /= (np.nanmax(D) + 1e-12)
    if eps is None:
        if percent is None:
            raise ValueError("Either eps or percent must be given.")
        k = int(np.clip(percent / 100.0 * D.size, 1, D.size))
        eps = np.partition(D.ravel(), k - 1)[k - 1]
    return (D <= eps).astype(np.uint8)


def joint_recurrence_matrix(*Rs: np.ndarray) -> np.ndarray:
    """Joint recurrence: logical AND over multiple binary recurrence matrices."""
    if not Rs:
        raise ValueError("Provide at least one recurrence matrix.")
    R = np.asarray(Rs[0], dtype=np.uint8)
    for Ri in Rs[1:]:
        R &= np.asarray(Ri, dtype=np.uint8)
    return R


# ------------------------- Poincaré & events -------------------------

def poincare_section(times: ArrayLike, X: ArrayLike, *,
                     section_dim: int = 0,
                     section_value: float = 0.0,
                     direction: str = "positive",
                     extract_dims: Tuple[int, int] | Tuple[int, int, int] = (1, 2)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Poincaré section hits from trajectory X(t).
    Linear interpolation between samples is used for crossings.

    Returns
    -------
    t_hits : (K,) times of intersections
    points : (K, len(extract_dims)) values of selected dims at crossings
    """
    t = np.asarray(times, float)
    X = np.asarray(X, float)
    s = X[:, section_dim] - section_value
    if direction == "positive":
        mask = (s[:-1] < 0) & (s[1:] >= 0)
    elif direction == "negative":
        mask = (s[:-1] > 0) & (s[1:] <= 0)
    else:
        mask = (s[:-1] * s[1:] <= 0)

    idx = np.where(mask)[0]
    t_hits = []
    pts = []
    for i in idx:
        a, b = s[i], s[i+1]
        if np.isclose(a, b):
            tau = 0.0
        else:
            tau = -a / (b - a)  # in [0,1]
        thit = t[i] + tau * (t[i+1] - t[i])
        xhit = X[i] + tau * (X[i+1] - X[i])
        t_hits.append(thit)
        pts.append(xhit[list(extract_dims)])
    return np.asarray(t_hits), np.asarray(pts)


def peaks(x: ArrayLike, height: Optional[float] = None, distance: Optional[int] = None) -> np.ndarray:
    """Indices of local maxima using scipy.signal.find_peaks."""
    x = np.asarray(x, float)
    idx, _ = find_peaks(x, height=height, distance=distance)
    return idx


# ------------------------- Bifurcation helpers -------------------------

def asymptotic_samples(x: ArrayLike, *,
                       tail: float = 0.2,
                       via_peaks: bool = True,
                       max_points: int = 400) -> np.ndarray:
    """
    Extract points for a bifurcation diagram from a scalar series x(t).
    Either take peaks on the tail, or raw tail samples.
    """
    x = np.asarray(x, float)
    n0 = int(max(1, (1.0 - tail) * x.size))
    tail_series = x[n0:]
    if via_peaks:
        idx = peaks(tail_series)
        vals = tail_series[idx]
    else:
        vals = tail_series
    if vals.size > max_points:
        # uniform subsample for stable density
        sel = np.linspace(0, vals.size - 1, max_points).astype(int)
        vals = vals[sel]
    return vals


# ------------------------- Projections & embeddings -------------------------

def pca_project(X: ArrayLike, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA using SVD. Return (components, explained_variance, mean).
    - components: (n_components, N)
    - variance: (n_components,)
    - mean: (N,)
    """
    X = np.asarray(X, float)
    mu = X.mean(axis=0)
    Z = X - mu
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    comps = Vt[:n_components]
    var = (S**2)[:n_components] / (X.shape[0] - 1)
    return comps, var, mu


def project_with_components(X: ArrayLike, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Project rows of X using precomputed PCA components and mean."""
    X = np.asarray(X, float)
    return (X - mean) @ components.T


# ------------------------- Distance matrices & density -------------------------

def distance_matrix(X: ArrayLike, metric: str = "euclidean") -> np.ndarray:
    """Pairwise distances for rows of X."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    return cdist(X, X, metric=metric)


def phase_space_density(X: ArrayLike, bins: int | Tuple[int, int] = 100, range=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D histogram density for (x,y) columns of X.
    Returns (H, xedges, yedges).
    """
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("X must be (T, >=2) for density.")
    H, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=bins, range=range, density=True)
    return H.T, xedges, yedges
