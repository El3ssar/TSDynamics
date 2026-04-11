"""
Numerical transforms for visualization.

Pure functions only — no matplotlib, no side effects, no classes.
Every function is independently testable and safe to import in headless environments.

Input contract
--------------
Trajectory functions accept ``t`` of shape ``(m,)`` and ``X`` of shape ``(m, n_dim)``.
This matches the output of ``DynSys.integrate()``, ``DynSysDelay.integrate()``,
and ``DynMap.iterate()``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks, welch
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Transient removal
# ---------------------------------------------------------------------------


def strip_transient(
    t: ArrayLike,
    X: ArrayLike,
    *,
    n: int | None = None,
    frac: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discard the transient prefix of a trajectory.

    At most one of *n* or *frac* may be supplied.  If neither is given,
    ``frac=0.1`` (10 %) is used as the default.

    Parameters
    ----------
    t : array_like, shape (m,)
        Time points.
    X : array_like, shape (m, n_dim)
        Trajectory.
    n : int, optional
        Absolute number of leading steps to discard.
    frac : float, optional
        Fraction of total steps to discard.  If neither *n* nor *frac* is
        provided the default fraction of 0.1 is applied.

    Returns
    -------
    t_trimmed : ndarray, shape (m - n0,)
    X_trimmed : ndarray, shape (m - n0, n_dim)

    Raises
    ------
    ValueError
        If both *n* and *frac* are given simultaneously, or *frac* is
        outside [0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 100)
    >>> X = np.random.default_rng(0).standard_normal((100, 3))
    >>> t2, X2 = strip_transient(t, X, frac=0.2)
    >>> t2.shape
    (80,)
    """
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    if n is not None and frac is not None:
        raise ValueError("Provide at most one of 'n' or 'frac', not both.")
    if n is None and frac is None:
        frac = 0.1
    if frac is not None:
        if not (0.0 <= frac < 1.0):
            raise ValueError(f"frac must be in [0, 1), got {frac}.")
        n = int(frac * len(t))
    return t[n:], X[n:]


# ---------------------------------------------------------------------------
# Basic signal helpers
# ---------------------------------------------------------------------------


def autocorrelation(x: ArrayLike, max_lag: int) -> np.ndarray:
    """
    Unbiased normalised autocorrelation of a scalar series.

    Parameters
    ----------
    x : array_like, shape (m,)
        Scalar time series.
    max_lag : int
        Maximum lag (inclusive).

    Returns
    -------
    r : ndarray, shape (max_lag + 1,)
        ACF values; ``r[0] == 1.0`` by construction.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(np.linspace(0, 4 * np.pi, 200))
    >>> r = autocorrelation(x, max_lag=50)
    >>> float(r[0])
    1.0
    """
    x = np.asarray(x, float) - np.mean(x)
    n = x.size
    r = np.asarray([np.dot(x[: n - k], x[k:]) / (n - k) for k in range(max_lag + 1)])
    return r / r[0]


def delay_embedding(x: ArrayLike, m: int, tau: int) -> np.ndarray:
    """
    Takens delay embedding of a univariate series into R^m.

    Parameters
    ----------
    x : array_like, shape (T,) or (T, 1)
        Scalar time series.
    m : int
        Embedding dimension (>= 1).
    tau : int
        Lag in samples (>= 1).

    Returns
    -------
    E : ndarray, shape (T - (m - 1) * tau, m)
        Delay-coordinate matrix; each row is a state vector.

    Raises
    ------
    ValueError
        If m or tau < 1, or the series is too short.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(20, dtype=float)
    >>> E = delay_embedding(x, m=3, tau=2)
    >>> E.shape
    (16, 3)
    """
    x = np.asarray(x, float).ravel()
    if m < 1 or tau < 1:
        raise ValueError("m and tau must be >= 1.")
    T = x.size - (m - 1) * tau
    if T <= 0:
        raise ValueError("Series too short for the requested embedding (m, tau).")
    return np.column_stack([x[i : i + T] for i in range(0, m * tau, tau)])


def return_map(x: ArrayLike, lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Lagged return-map pairs from a scalar series.

    Parameters
    ----------
    x : array_like, shape (m,)
        Scalar time series.
    lag : int
        Lag (>= 1).

    Returns
    -------
    x_t : ndarray, shape (m - lag,)
    x_t_lag : ndarray, shape (m - lag,)
        ``x_t_lag[i] == x[i + lag]``.

    Raises
    ------
    ValueError
        If lag < 1.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(10, dtype=float)
    >>> xn, xn1 = return_map(x, lag=1)
    >>> xn1[0]
    1.0
    """
    x = np.asarray(x, float)
    if lag < 1:
        raise ValueError("lag must be >= 1.")
    return x[:-lag], x[lag:]


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------


def power_spectrum_fft(x: ArrayLike, fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-sided FFT power spectral density.

    Parameters
    ----------
    x : array_like, shape (m,)
        Scalar time series.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    freqs : ndarray, shape (m // 2 + 1,)
    psd : ndarray, shape (m // 2 + 1,)

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(2 * np.pi * 5 * np.arange(256) / 256)
    >>> f, p = power_spectrum_fft(x, fs=256)
    >>> f.shape == p.shape
    True
    """
    x = np.asarray(x, float)
    n = x.size
    X = np.fft.rfft(x - np.mean(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(X) ** 2) / (n * fs)
    return freqs, psd


def power_spectrum_welch(
    x: ArrayLike,
    fs: float = 1.0,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Welch power spectral density estimate.

    Parameters
    ----------
    x : array_like, shape (m,)
        Scalar time series.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Segment length; defaults to ``max(256, len(x) // 8)``.

    Returns
    -------
    freqs : ndarray
    psd : ndarray

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(2 * np.pi * 3 * np.arange(512) / 512)
    >>> f, p = power_spectrum_welch(x, fs=512)
    >>> f.shape == p.shape
    True
    """
    x = np.asarray(x, float)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg or max(256, len(x) // 8))
    return f, Pxx


def wavelet_scalogram(
    x: ArrayLike,
    fs: float,
    wavelet: str = "morl",
    widths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Continuous wavelet transform scalogram (requires PyWavelets).

    Parameters
    ----------
    x : array_like, shape (m,)
        Scalar time series.
    fs : float
        Sampling frequency in Hz.
    wavelet : str
        Wavelet name recognised by ``pywt.cwt``.
    widths : ndarray, optional
        Scale array; defaults to 64 log-spaced values from 1 to ``max(8, m // 8)``.

    Returns
    -------
    t : ndarray, shape (m,)
    scales : ndarray, shape (n_scales,)
    amplitude : ndarray, shape (n_scales, m)
        ``|CWT|`` coefficients.

    Raises
    ------
    ImportError
        If PyWavelets is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(2 * np.pi * np.linspace(0, 4, 256))
    >>> t, scales, A = wavelet_scalogram(x, fs=64)
    >>> A.shape[1] == len(x)
    True
    """
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("wavelet_scalogram requires PyWavelets: pip install pywavelets") from exc
    x = np.asarray(x, float)
    if widths is None:
        widths = np.geomspace(1, max(8, len(x) // 8), 64)
    cwtmatr, scales = pywt.cwt(x, widths, wavelet, sampling_period=1.0 / fs)
    t = np.arange(x.size) / fs
    return t, scales, np.abs(cwtmatr)


# ---------------------------------------------------------------------------
# Recurrence analysis
# ---------------------------------------------------------------------------


def recurrence_matrix(
    X: ArrayLike,
    eps: float | None = None,
    percent: float | None = 10.0,
    metric: str = "euclidean",
    normalize: bool = True,
) -> np.ndarray:
    """
    Binary recurrence matrix R[i, j] == 1 if dist(X[i], X[j]) <= threshold.

    Parameters
    ----------
    X : array_like, shape (m,) or (m, d)
        Trajectory or scalar series.
    eps : float, optional
        Fixed distance threshold. If ``None``, *percent* is used instead.
    percent : float, optional
        If *eps* is ``None``, choose threshold so that *percent*% of all pairs
        are recurrent. Default 10.0.
    metric : str
        Distance metric passed to ``scipy.spatial.distance.cdist``.
    normalize : bool
        If True, distances are normalised by their maximum before thresholding.

    Returns
    -------
    R : ndarray, shape (m, m), dtype uint8

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).standard_normal((50, 2))
    >>> R = recurrence_matrix(X, percent=20.0)
    >>> R.shape
    (50, 50)
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    D = cdist(X, X, metric=metric)
    if normalize:
        D = D / (np.nanmax(D) + 1e-12)
    if eps is None:
        if percent is None:
            raise ValueError("Either eps or percent must be given.")
        k = int(np.clip(percent / 100.0 * D.size, 1, D.size))
        eps = float(np.partition(D.ravel(), k - 1)[k - 1])
    return (eps >= D).astype(np.uint8)


def cross_recurrence_matrix(
    X: ArrayLike,
    Y: ArrayLike,
    eps: float | None = None,
    percent: float | None = 10.0,
    metric: str = "euclidean",
    normalize: bool = True,
) -> np.ndarray:
    """
    Binary cross-recurrence matrix between two sequences.

    Parameters
    ----------
    X : array_like, shape (m,) or (m, d)
    Y : array_like, shape (n,) or (n, d)
    eps : float, optional
    percent : float, optional
    metric : str
    normalize : bool

    Returns
    -------
    R : ndarray, shape (m, n), dtype uint8

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> X = rng.standard_normal((40, 2))
    >>> Y = rng.standard_normal((40, 2))
    >>> R = cross_recurrence_matrix(X, Y, percent=15.0)
    >>> R.shape
    (40, 40)
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    D = cdist(X, Y, metric=metric)
    if normalize:
        D = D / (np.nanmax(D) + 1e-12)
    if eps is None:
        if percent is None:
            raise ValueError("Either eps or percent must be given.")
        k = int(np.clip(percent / 100.0 * D.size, 1, D.size))
        eps = float(np.partition(D.ravel(), k - 1)[k - 1])
    return (eps >= D).astype(np.uint8)


def joint_recurrence_matrix(*Rs: np.ndarray) -> np.ndarray:
    """
    Joint recurrence: element-wise logical AND over binary recurrence matrices.

    Parameters
    ----------
    *Rs : ndarray
        One or more binary recurrence matrices of identical shape.

    Returns
    -------
    R : ndarray, same shape as inputs, dtype uint8

    Raises
    ------
    ValueError
        If no matrices are provided.

    Examples
    --------
    >>> import numpy as np
    >>> R1 = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    >>> R2 = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    >>> joint_recurrence_matrix(R1, R2)
    array([[1, 0],
           [0, 1]], dtype=uint8)
    """
    if not Rs:
        raise ValueError("Provide at least one recurrence matrix.")
    R = np.asarray(Rs[0], dtype=np.uint8).copy()
    for Ri in Rs[1:]:
        R &= np.asarray(Ri, dtype=np.uint8)
    return R


# ---------------------------------------------------------------------------
# Poincaré section & event detection
# ---------------------------------------------------------------------------


def poincare_section(
    t: ArrayLike,
    X: ArrayLike,
    *,
    section_dim: int = 0,
    section_value: float = 0.0,
    direction: str = "positive",
    extract_dims: tuple[int, ...] = (1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract Poincaré section crossings from a trajectory.

    Linearly interpolates between adjacent samples to locate each crossing.

    Parameters
    ----------
    t : array_like, shape (m,)
        Time points.
    X : array_like, shape (m, n_dim)
        Trajectory; must have ``n_dim >= 2``.
    section_dim : int
        Index of the coordinate used to define the hyperplane.
    section_value : float
        Hyperplane value: ``X[:, section_dim] == section_value``.
    direction : {"positive", "negative", "both"}
        Cross only ascending (positive), only descending (negative), or both.
    extract_dims : tuple of int
        Indices of dimensions to extract at each crossing.

    Returns
    -------
    t_hits : ndarray, shape (K,)
        Interpolated times of each crossing.
    points : ndarray, shape (K, len(extract_dims))
        Interpolated state values at each crossing, restricted to *extract_dims*.
        If no crossings are found, both arrays are empty with the correct shape.

    Raises
    ------
    ValueError
        If *direction* is not one of the accepted strings.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 4 * np.pi, 1000)
    >>> X = np.column_stack([np.sin(t), np.cos(t), t / (4 * np.pi)])
    >>> t_hits, pts = poincare_section(t, X, section_dim=0, section_value=0.0)
    >>> pts.shape[1]
    2
    """
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    s = X[:, section_dim] - section_value
    if direction == "positive":
        mask = (s[:-1] < 0) & (s[1:] >= 0)
    elif direction == "negative":
        mask = (s[:-1] > 0) & (s[1:] <= 0)
    elif direction == "both":
        mask = s[:-1] * s[1:] <= 0
    else:
        raise ValueError(f"direction must be 'positive', 'negative', or 'both'; got {direction!r}.")

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.empty(0), np.empty((0, len(extract_dims)))

    t_hits, pts = [], []
    for i in idx:
        a, b = s[i], s[i + 1]
        tau = 0.0 if np.isclose(a, b) else -a / (b - a)
        t_hits.append(t[i] + tau * (t[i + 1] - t[i]))
        xhit = X[i] + tau * (X[i + 1] - X[i])
        pts.append(xhit[list(extract_dims)])
    return np.asarray(t_hits), np.asarray(pts)


def peaks(
    x: ArrayLike,
    height: float | None = None,
    distance: int | None = None,
) -> np.ndarray:
    """
    Find indices of local maxima in a scalar series.

    Parameters
    ----------
    x : array_like, shape (m,)
    height : float, optional
        Minimum height of peaks.
    distance : int, optional
        Minimum sample distance between peaks.

    Returns
    -------
    idx : ndarray
        Integer indices of detected peaks.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 1.0, 0.5, 2.0, 0.3])
    >>> peaks(x)
    array([1, 3])
    """
    x = np.asarray(x, float)
    idx, _ = find_peaks(x, height=height, distance=distance)
    return idx


def asymptotic_samples(
    x: ArrayLike,
    *,
    tail: float = 0.2,
    via_peaks: bool = True,
    max_points: int = 400,
) -> np.ndarray:
    """
    Extract asymptotic (post-transient) samples from a scalar series.

    Useful for constructing bifurcation diagrams: call once per parameter value
    and collect the returned arrays.

    Parameters
    ----------
    x : array_like, shape (m,)
        Scalar time series (e.g. one component of a trajectory).
    tail : float
        Fraction of the series to treat as the asymptotic tail (default 0.2).
    via_peaks : bool
        If True, extract local maxima from the tail; otherwise return raw tail samples.
    max_points : int
        Cap on returned points; uniformly sub-samples if exceeded.

    Returns
    -------
    vals : ndarray, shape (<=max_points,)

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(np.linspace(0, 40 * np.pi, 2000))
    >>> vals = asymptotic_samples(x, tail=0.3, via_peaks=True)
    >>> vals.size > 0
    True
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
        sel = np.linspace(0, vals.size - 1, max_points).astype(int)
        vals = vals[sel]
    return vals


# ---------------------------------------------------------------------------
# PCA / projections
# ---------------------------------------------------------------------------


def pca_project(
    X: ArrayLike,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Principal component analysis via truncated SVD.

    Parameters
    ----------
    X : array_like, shape (m, d)
        Trajectory or data matrix.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    components : ndarray, shape (n_components, d)
        Principal component directions (unit vectors).
    variance : ndarray, shape (n_components,)
        Variance explained by each component.
    mean : ndarray, shape (d,)
        Column means used for centring.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).standard_normal((100, 5))
    >>> comps, var, mu = pca_project(X, n_components=2)
    >>> comps.shape
    (2, 5)
    """
    X = np.asarray(X, float)
    mu = X.mean(axis=0)
    Z = X - mu
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    components = Vt[:n_components]
    variance = (S**2)[:n_components] / (X.shape[0] - 1)
    return components, variance, mu


def project_onto_pca(
    X: ArrayLike,
    components: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """
    Project data onto precomputed PCA components.

    Parameters
    ----------
    X : array_like, shape (m, d)
    components : ndarray, shape (n_components, d)
        Output of :func:`pca_project`.
    mean : ndarray, shape (d,)
        Column means from :func:`pca_project`.

    Returns
    -------
    Z : ndarray, shape (m, n_components)

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).standard_normal((100, 5))
    >>> comps, _, mu = pca_project(X, n_components=2)
    >>> Z = project_onto_pca(X, comps, mu)
    >>> Z.shape
    (100, 2)
    """
    X = np.asarray(X, float)
    return (X - mean) @ components.T


# ---------------------------------------------------------------------------
# Distance & density
# ---------------------------------------------------------------------------


def distance_matrix(X: ArrayLike, metric: str = "euclidean") -> np.ndarray:
    """
    Symmetric pairwise distance matrix for rows of X.

    Parameters
    ----------
    X : array_like, shape (m,) or (m, d)
    metric : str
        Metric passed to ``scipy.spatial.distance.cdist``.

    Returns
    -------
    D : ndarray, shape (m, m)
        Zero diagonal, symmetric.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> D = distance_matrix(X)
    >>> float(D[0, 1])
    1.0
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    return cdist(X, X, metric=metric)


def phase_space_density(
    X: ArrayLike,
    bins: int | tuple[int, int] = 100,
    range: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2-D histogram density of a two-column trajectory slice.

    The caller is responsible for selecting the desired columns *before* calling
    this function (e.g. ``phase_space_density(X[:, (0, 2)])``) so the
    function remains honest about its input shape.

    Parameters
    ----------
    X : array_like, shape (m, 2)
        Two-column array; exactly 2 columns required.
    bins : int or (int, int)
        Number of histogram bins along each axis.
    range : tuple, optional
        ``((x_min, x_max), (y_min, y_max))`` passed to ``np.histogram2d``.

    Returns
    -------
    H : ndarray, shape (bins_y, bins_x)
        Density estimate (transposed for ``imshow`` compatibility).
    xedges : ndarray
    yedges : ndarray

    Raises
    ------
    ValueError
        If X does not have exactly 2 columns.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).standard_normal((500, 2))
    >>> H, xe, ye = phase_space_density(X)
    >>> H.shape
    (100, 100)
    """
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            f"X must have shape (m, 2), got {X.shape}. "
            "Select two columns before calling, e.g. X[:, (0, 2)]."
        )
    H, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=bins, range=range, density=True)
    return H.T, xedges, yedges
