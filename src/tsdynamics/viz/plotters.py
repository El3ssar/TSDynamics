"""
Static plot functions for TSDynamics trajectories.

Every function follows the same contract:

* Accepts ``ax: matplotlib.axes.Axes | None = None``.
  - If *ax* is ``None`` a new figure is created internally.
  - If *ax* is provided the function draws into it unchanged.
* Returns the ``matplotlib.axes.Axes`` used (not a ``(fig, ax)`` tuple).
  The figure is always reachable via ``ax.figure``.
* Accepts ``**kwargs`` forwarded to the underlying matplotlib primitive.
* Never calls ``plt.show()``.
* Never mutates input arrays.

All functions are stateless; they carry no global state or registries.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers "3d" projection

from . import transforms as tf
from ._utils import _label_dims, _resolve_ax

# ---------------------------------------------------------------------------
# Trajectory plots
# ---------------------------------------------------------------------------


def trajectory_plot(
    t: np.ndarray,
    X: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    ax=None,
    **kwargs,
):
    """
    2-D trajectory: one phase-space component vs another, connected by lines.

    The time axis *t* is used only to set a time-ordered line; for a
    pure phase-space scatter use :func:`phase_portrait` instead.

    Parameters
    ----------
    t : ndarray, shape (m,)
        Time points (unused for layout but kept for API uniformity).
    X : ndarray, shape (m, n_dim)
        Trajectory.
    dims : (int, int)
        Column indices to plot on the x and y axes.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. Created internally if *None*.
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 200)
    >>> X = np.column_stack([np.sin(t), np.cos(t), t])
    >>> ax = trajectory_plot(t, X, dims=(0, 1))
    >>> ax.figure is not None
    True
    """
    X = np.asarray(X, float)
    ax = _resolve_ax(ax)
    kwargs.setdefault("lw", 0.8)
    ax.plot(X[:, dims[0]], X[:, dims[1]], **kwargs)
    _label_dims(ax, dims)
    return ax


def trajectory_plot_3d(
    t: np.ndarray,
    X: np.ndarray,
    dims: tuple[int, int, int] = (0, 1, 2),
    ax=None,
    **kwargs,
):
    """
    3-D trajectory line plot.

    Parameters
    ----------
    t : ndarray, shape (m,)
    X : ndarray, shape (m, n_dim), n_dim >= 3
    dims : (int, int, int)
    ax : Axes3D, optional
        Must be a 3D axes if provided.
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    ax : mpl_toolkits.mplot3d.Axes3D

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 200)
    >>> X = np.column_stack([np.sin(t), np.cos(t), t])
    >>> ax = trajectory_plot_3d(t, X)
    >>> hasattr(ax, 'set_zlabel')
    True
    """
    X = np.asarray(X, float)
    ax = _resolve_ax(ax, projection="3d")
    kwargs.setdefault("lw", 0.7)
    ax.plot(X[:, dims[0]], X[:, dims[1]], X[:, dims[2]], **kwargs)
    _label_dims(ax, dims)
    return ax


# ---------------------------------------------------------------------------
# Phase portraits & density
# ---------------------------------------------------------------------------


def phase_portrait(
    X: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    ax=None,
    **kwargs,
):
    """
    Phase portrait: scatter of trajectory points in phase space.

    Unlike :func:`trajectory_plot`, phase portraits do not draw connecting lines
    and do not accept a time vector (time-independence is semantically correct
    for phase-space representations).

    Parameters
    ----------
    X : ndarray, shape (m, n_dim)
    dims : (int, int)
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.plot``. Default marker is ``','`` (pixel).

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).standard_normal((500, 3))
    >>> ax = phase_portrait(X, dims=(0, 2))
    >>> len(ax.lines) > 0
    True
    """
    X = np.asarray(X, float)
    ax = _resolve_ax(ax)
    kwargs.setdefault("marker", ",")
    kwargs.setdefault("lw", 0)
    kwargs.setdefault("alpha", 0.5)
    ax.plot(X[:, dims[0]], X[:, dims[1]], **kwargs)
    _label_dims(ax, dims)
    return ax


def phase_density(
    X: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    bins: int | tuple[int, int] = 200,
    ax=None,
    **kwargs,
):
    """
    Phase-space density map rendered as a 2-D histogram (``imshow``).

    Faster than scatter for dense trajectories (> 10 k points) and gives a
    cleaner visual of the invariant measure.

    Parameters
    ----------
    X : ndarray, shape (m, n_dim)
    dims : (int, int)
    bins : int or (int, int)
        Histogram bin count.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.imshow``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).standard_normal((1000, 3))
    >>> ax = phase_density(X, dims=(0, 1), bins=50)
    >>> ax is not None
    True
    """
    X = np.asarray(X, float)
    ax = _resolve_ax(ax)
    H, xe, ye = tf.phase_space_density(X[:, list(dims)], bins=bins)
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("cmap", "inferno")
    im = ax.imshow(H, extent=[xe[0], xe[-1], ye[0], ye[-1]], **kwargs)
    ax.figure.colorbar(im, ax=ax, label="density")
    _label_dims(ax, dims)
    return ax


# ---------------------------------------------------------------------------
# Poincaré section & return maps
# ---------------------------------------------------------------------------


def poincare_scatter(
    t: np.ndarray,
    X: np.ndarray,
    *,
    section_dim: int = 0,
    section_value: float = 0.0,
    direction: str = "positive",
    extract_dims: tuple[int, ...] = (1, 2),
    ax=None,
    **kwargs,
):
    """
    Poincaré section scatter plot.

    Internally calls :func:`transforms.poincare_section` and plots the crossing
    points.

    Parameters
    ----------
    t : ndarray, shape (m,)
    X : ndarray, shape (m, n_dim)
    section_dim : int
        Dimension used to define the hyperplane.
    section_value : float
        Hyperplane value.
    direction : {"positive", "negative", "both"}
    extract_dims : tuple of int
        Dimensions to plot; 2 → 2D scatter, 3 → 3D scatter.
    ax : Axes or Axes3D, optional
    **kwargs
        Forwarded to ``ax.scatter``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 20 * np.pi, 5000)
    >>> X = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])
    >>> ax = poincare_scatter(t, X, section_dim=0, section_value=0.0)
    >>> ax is not None
    True
    """
    is_3d = len(extract_dims) == 3
    ax = _resolve_ax(ax, projection="3d" if is_3d else None)
    _, pts = tf.poincare_section(
        t,
        X,
        section_dim=section_dim,
        section_value=section_value,
        direction=direction,
        extract_dims=extract_dims,
    )
    if pts.size == 0:
        ax.set_title("No Poincaré crossings found")
        return ax
    kwargs.setdefault("s", 10)
    kwargs.setdefault("alpha", 0.7)
    if is_3d:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kwargs)
        _label_dims(ax, extract_dims)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], **kwargs)
        _label_dims(ax, extract_dims)
    return ax


def return_map_plot(
    x: np.ndarray,
    lag: int = 1,
    ax=None,
    **kwargs,
):
    """
    Lagged return map: x(t) vs x(t + lag).

    Parameters
    ----------
    x : ndarray, shape (m,)
        Scalar time series.
    lag : int
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(np.linspace(0, 20 * np.pi, 500))
    >>> ax = return_map_plot(x, lag=1)
    >>> len(ax.lines) > 0
    True
    """
    x0, x1 = tf.return_map(x, lag=lag)
    ax = _resolve_ax(ax)
    ax.plot(x0, x1, **kwargs)
    ax.set_xlabel("$x(t)$")
    ax.set_ylabel(f"$x(t+{lag})$")
    return ax


# ---------------------------------------------------------------------------
# Recurrence plots
# ---------------------------------------------------------------------------


def recurrence_plot(
    X: np.ndarray,
    eps: float | None = None,
    percent: float | None = 10.0,
    metric: str = "euclidean",
    ax=None,
    **kwargs,
):
    """
    Binary recurrence plot.

    Parameters
    ----------
    X : ndarray, shape (m,) or (m, d)
    eps : float, optional
    percent : float, optional
    metric : str
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.imshow``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).standard_normal((80, 2))
    >>> ax = recurrence_plot(X, percent=10.0)
    >>> ax is not None
    True
    """
    R = tf.recurrence_matrix(X, eps=eps, percent=percent, metric=metric)
    ax = _resolve_ax(ax)
    kwargs.setdefault("cmap", "binary")
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("interpolation", "nearest")
    ax.imshow(R, **kwargs)
    ax.set_xlabel("$i$")
    ax.set_ylabel("$j$")
    return ax


def cross_recurrence_plot(
    X: np.ndarray,
    Y: np.ndarray,
    eps: float | None = None,
    percent: float | None = 10.0,
    metric: str = "euclidean",
    ax=None,
    **kwargs,
):
    """
    Cross-recurrence plot between two trajectories.

    Parameters
    ----------
    X : ndarray, shape (m,) or (m, d)
    Y : ndarray, shape (n,) or (n, d)
    eps : float, optional
    percent : float, optional
    metric : str
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.imshow``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X, Y = rng.standard_normal((60, 2)), rng.standard_normal((60, 2))
    >>> ax = cross_recurrence_plot(X, Y, percent=10.0)
    >>> ax is not None
    True
    """
    R = tf.cross_recurrence_matrix(X, Y, eps=eps, percent=percent, metric=metric)
    ax = _resolve_ax(ax)
    kwargs.setdefault("cmap", "binary")
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("interpolation", "nearest")
    ax.imshow(R, **kwargs)
    ax.set_xlabel("$i$ (X)")
    ax.set_ylabel("$j$ (Y)")
    return ax


def joint_recurrence_plot(
    *Rs: np.ndarray,
    ax=None,
    **kwargs,
):
    """
    Joint-recurrence plot from precomputed binary recurrence matrices.

    Parameters
    ----------
    *Rs : ndarray
        Two or more binary recurrence matrices of the same shape.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.imshow``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 2))
    >>> R1 = tf.recurrence_matrix(X, percent=10.0)
    >>> R2 = tf.recurrence_matrix(X, percent=20.0)
    >>> ax = joint_recurrence_plot(R1, R2)
    >>> ax is not None
    True
    """
    R = tf.joint_recurrence_matrix(*Rs)
    ax = _resolve_ax(ax)
    kwargs.setdefault("cmap", "binary")
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("interpolation", "nearest")
    ax.imshow(R, **kwargs)
    ax.set_xlabel("$i$")
    ax.set_ylabel("$j$")
    return ax


# ---------------------------------------------------------------------------
# Spectral & wavelet
# ---------------------------------------------------------------------------


def power_spectrum_plot(
    x: np.ndarray,
    fs: float = 1.0,
    method: str = "welch",
    ax=None,
    **kwargs,
):
    """
    Power spectral density plot (Welch or FFT).

    Parameters
    ----------
    x : ndarray, shape (m,)
        Scalar time series.
    fs : float
        Sampling frequency.
    method : {"welch", "fft"}
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.semilogy``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(2 * np.pi * 5 * np.arange(512) / 512)
    >>> ax = power_spectrum_plot(x, fs=512)
    >>> ax is not None
    True
    """
    if x.ndim == 2:
        if x.shape[1] > 1:
            raise ValueError(f"x must have shape (m,) or (m, 1), got {x.shape}.")
        x = x[:, 0]
    elif x.ndim > 2:
        raise ValueError(f"x must have shape (m,) or (m, 1), got {x.shape}.")
    if method == "welch":
        f, Pxx = tf.power_spectrum_welch(x, fs=fs)
    elif method == "fft":
        f, Pxx = tf.power_spectrum_fft(x, fs=fs)
    else:
        raise ValueError(f"method must be 'welch' or 'fft', got {method!r}.")
    ax = _resolve_ax(ax)
    ax.semilogy(f, Pxx + 1e-20, **kwargs)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("PSD")
    return ax


def wavelet_scalogram_plot(
    x: np.ndarray,
    fs: float,
    wavelet: str = "morl",
    widths: np.ndarray | None = None,
    ax=None,
    **kwargs,
):
    """
    Wavelet scalogram plot (requires PyWavelets).

    Parameters
    ----------
    x : ndarray, shape (m,)
    fs : float
    wavelet : str
    widths : ndarray, optional
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.imshow``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(2 * np.pi * np.linspace(0, 4, 256))
    >>> ax = wavelet_scalogram_plot(x, fs=64)
    >>> ax is not None
    True
    """
    t_arr, scales, A = tf.wavelet_scalogram(x, fs=fs, wavelet=wavelet, widths=widths)
    ax = _resolve_ax(ax)
    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("cmap", "viridis")
    im = ax.imshow(A, extent=[t_arr[0], t_arr[-1], scales[-1], scales[0]], **kwargs)
    ax.figure.colorbar(im, ax=ax, label="|CWT|")
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale")
    return ax


# ---------------------------------------------------------------------------
# Embeddings & projections
# ---------------------------------------------------------------------------


def embedding_plot(
    x: np.ndarray,
    m: int,
    tau: int,
    ax=None,
    **kwargs,
):
    """
    Delay-embedding plot (2-D or 3-D depending on *m*).

    Parameters
    ----------
    x : ndarray, shape (T,)
        Univariate time series.
    m : int
        Embedding dimension (2 or 3).
    tau : int
        Lag in samples.
    ax : Axes or Axes3D, optional
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    ax : matplotlib.axes.Axes or Axes3D

    Raises
    ------
    ValueError
        If m < 2.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.sin(np.linspace(0, 20 * np.pi, 500))
    >>> ax = embedding_plot(x, m=2, tau=10)
    >>> ax is not None
    True
    """
    if x.ndim == 2:
        if x.shape[1] > 1:
            raise ValueError(f"x must have shape (m,) or (m, 1), got {x.shape}.")
        x = x[:, 0]
    elif x.ndim > 2:
        raise ValueError(f"x must have shape (m,) or (m, 1), got {x.shape}.")
    if m < 2:
        raise ValueError(f"m must be >= 2 for a meaningful embedding plot, got {m}.")
    E = tf.delay_embedding(x, m=m, tau=tau)
    if m == 2:
        ax = _resolve_ax(ax)
        kwargs.setdefault("marker", ",")
        kwargs.setdefault("lw", 0)
        kwargs.setdefault("alpha", 0.6)
        ax.plot(E[:, 0], E[:, 1], **kwargs)
        ax.set_xlabel("$x(t)$")
        ax.set_ylabel(f"$x(t+{tau})$")
    else:
        ax = _resolve_ax(ax, projection="3d")
        kwargs.setdefault("marker", ",")
        kwargs.setdefault("lw", 0)
        kwargs.setdefault("alpha", 0.6)
        ax.plot(E[:, 0], E[:, 1], E[:, 2], **kwargs)
        ax.set_xlabel("$x(t)$")
        ax.set_ylabel(f"$x(t+{tau})$")
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel(f"$x(t+{2 * tau})$")
    return ax


def pca_projection_plot(
    X: np.ndarray,
    n_components: int = 2,
    ax=None,
    **kwargs,
):
    """
    Project trajectory onto the first principal components and plot.

    Parameters
    ----------
    X : ndarray, shape (m, n_dim)
    n_components : int
        2 → 2-D plot, 3 → 3-D plot.
    ax : Axes or Axes3D, optional
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    ax : matplotlib.axes.Axes or Axes3D

    Raises
    ------
    ValueError
        If n_components is not 2 or 3.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).standard_normal((200, 5))
    >>> ax = pca_projection_plot(X, n_components=2)
    >>> ax is not None
    True
    """
    if n_components not in (2, 3):
        raise ValueError(f"n_components must be 2 or 3 for plotting, got {n_components}.")
    comps, var, mu = tf.pca_project(X, n_components=n_components)
    Z = tf.project_onto_pca(X, comps, mu)
    kwargs.setdefault("marker", ",")
    kwargs.setdefault("lw", 0)
    kwargs.setdefault("alpha", 0.6)
    if n_components == 2:
        ax = _resolve_ax(ax)
        ax.plot(Z[:, 0], Z[:, 1], **kwargs)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA (var: {var[0]:.2g}, {var[1]:.2g})")
    else:
        ax = _resolve_ax(ax, projection="3d")
        ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], **kwargs)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel("PC3")
    return ax


# ---------------------------------------------------------------------------
# Distance & bifurcation
# ---------------------------------------------------------------------------


def distance_heatmap(
    X: np.ndarray,
    metric: str = "euclidean",
    ax=None,
    **kwargs,
):
    """
    Pairwise distance heatmap.

    Parameters
    ----------
    X : ndarray, shape (m,) or (m, d)
    metric : str
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.imshow``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).standard_normal((50, 3))
    >>> ax = distance_heatmap(X)
    >>> ax is not None
    True
    """
    D = tf.distance_matrix(X, metric=metric)
    ax = _resolve_ax(ax)
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("aspect", "auto")
    im = ax.imshow(D, **kwargs)
    ax.figure.colorbar(im, ax=ax, label=f"{metric} distance")
    ax.set_xlabel("$i$")
    ax.set_ylabel("$j$")
    return ax


def bifurcation_diagram(
    params: np.ndarray,
    points: Sequence[np.ndarray],
    *,
    s: float = 2.0,
    alpha: float = 0.5,
    ax=None,
    **kwargs,
):
    """
    Plot a generic bifurcation diagram from a caller-driven parameter sweep.

    For each parameter value in *params*, plot the corresponding asymptotic
    samples from *points* (output of :func:`transforms.asymptotic_samples`).

    Parameters
    ----------
    params : ndarray, shape (n_params,)
        Parameter values, one per run.
    points : sequence of ndarray
        ``points[i]`` contains the asymptotic samples for ``params[i]``.
    s : float
        Scatter marker size.
    alpha : float
        Scatter transparency.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.scatter``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If ``len(points) != len(params)``.

    Examples
    --------
    >>> import numpy as np
    >>> params = np.linspace(3.5, 4.0, 5)
    >>> points = [np.array([0.5, 0.8]) for _ in params]
    >>> ax = bifurcation_diagram(params, points)
    >>> ax is not None
    True
    """
    params = np.asarray(params, float)
    if len(points) != params.size:
        raise ValueError("len(points) must equal len(params).")
    ax = _resolve_ax(ax)
    for p, vals in zip(params, points, strict=True):
        if vals is None or len(vals) == 0:
            continue
        ax.scatter(
            np.full_like(vals, p, dtype=float), vals, s=s, alpha=alpha, edgecolors="none", **kwargs
        )
    ax.set_xlabel("parameter")
    ax.set_ylabel("observable")
    return ax


# ---------------------------------------------------------------------------
# Lyapunov spectrum
# ---------------------------------------------------------------------------


def lyapunov_spectrum_plot(
    exponents: np.ndarray,
    ax=None,
    **kwargs,
):
    """
    Horizontal bar chart of a Lyapunov spectrum.

    Bars are coloured by sign: positive → red, zero (within ±0.01) → grey,
    negative → steelblue. The Kaplan–Yorke dimension D_KY is annotated.

    Parameters
    ----------
    exponents : ndarray, shape (n_lyap,)
        Lyapunov exponents (largest first, as returned by
        ``DynSys.lyapunov_spectrum()`` etc.).
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.barh``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> exponents = np.array([0.91, 0.0, -14.57])
    >>> ax = lyapunov_spectrum_plot(exponents)
    >>> ax is not None
    True
    """
    exponents = np.asarray(exponents, float).ravel()
    ax = _resolve_ax(ax)
    tol = 1e-2
    colors = ["red" if e > tol else "grey" if abs(e) <= tol else "steelblue" for e in exponents]
    indices = np.arange(len(exponents))
    ax.barh(indices, exponents, color=colors, **kwargs)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Lyapunov exponent")
    ax.set_ylabel("index")
    ax.set_yticks(indices)
    ax.invert_yaxis()

    # Kaplan-Yorke dimension
    cumsum = np.cumsum(exponents)
    j_arr = np.where(cumsum >= 0)[0]
    if len(j_arr) == 0:
        dky = 0.0
    else:
        j = j_arr[-1]
        if j + 1 < len(exponents) and abs(exponents[j + 1]) > 1e-12:
            dky = (j + 1) + cumsum[j] / abs(exponents[j + 1])
        else:
            dky = float(len(exponents))
    ax.set_title(f"Lyapunov spectrum  —  $D_{{KY}}$ = {dky:.3f}")
    return ax


# ---------------------------------------------------------------------------
# Space–time diagram
# ---------------------------------------------------------------------------


def spacetime_plot(
    t: np.ndarray,
    X: np.ndarray,
    *,
    nx: int | None = None,
    ny: int | None = None,
    ax=None,
    **kwargs,
):
    """
    Space–time diagram (kymograph) for high-dimensional trajectories.

    Intended for PDE-discretised systems such as Kuramoto–Sivashinsky or
    Lorenz-96 where *n_dim* represents a spatial discretisation of a 1-D field.

    For a 1-D field (``ny=None``):
    - X rows are time steps; X columns are spatial positions.
    - The image shows time on the x-axis and space on the y-axis.

    For a 2-D field (``nx`` and ``ny`` given, ``nx * ny == n_dim``):
    - Displays the final frame reshaped to (ny, nx).

    Parameters
    ----------
    t : ndarray, shape (T,)
    X : ndarray, shape (T, N)
    nx : int, optional
        Spatial width (1-D: must equal N; 2-D: required with ny).
    ny : int, optional
        Spatial height for 2-D fields.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``ax.imshow``. Default ``aspect="auto"``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        On inconsistent nx/ny vs X.shape.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 5, 100)
    >>> X = np.random.default_rng(0).standard_normal((100, 32))
    >>> ax = spacetime_plot(t, X)
    >>> ax is not None
    True
    """
    X = np.asarray(X, float)
    T, N = X.shape
    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("origin", "lower")

    ax = _resolve_ax(ax)
    if ny is None:
        # 1-D field: full kymograph
        _nx = nx if nx is not None else N
        if _nx != N:
            raise ValueError(f"nx={_nx} does not match X.shape[1]={N}.")
        im = ax.imshow(X.T, extent=[float(t[0]), float(t[-1]), 0, _nx], **kwargs)
        ax.set_xlabel("$t$")
        ax.set_ylabel("spatial index")
    else:
        # 2-D field: show last frame
        if nx is None or nx * ny != N:
            raise ValueError(f"nx * ny must equal X.shape[1]={N}, got nx={nx}, ny={ny}.")
        last = X[-1].reshape(ny, nx)
        im = ax.imshow(last, **kwargs)
        ax.set_title("field snapshot (last frame)")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    ax.figure.colorbar(im, ax=ax)
    return ax
