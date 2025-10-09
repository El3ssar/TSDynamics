"""Static plot creators using matplotlib. Consume arrays from transforms."""

from __future__ import annotations
from typing import Optional, Tuple, Sequence
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .base import new_fig_ax, PlotConfig
from . import transforms as tf


# ------------------------- Trajectories -------------------------

def trajectory2d(times: np.ndarray, Y: np.ndarray, dims: Tuple[int, int] = (0, 1),
                 cfg: Optional[PlotConfig] = None):
    """2D trajectory plot."""
    XY = tf.take_columns(Y, dims)
    fig, ax = new_fig_ax(cfg=cfg)
    ax.plot(XY[:, 0], XY[:, 1], lw=0.8)
    ax.set_xlabel(f"x[{dims[0]}]")
    ax.set_ylabel(f"x[{dims[1]}]")
    ax.set_title("Trajectory (2D)")
    return fig, ax


def trajectory3d(times: np.ndarray, Y: np.ndarray, dims: Tuple[int, int, int] = (0, 1, 2),
                 cfg: Optional[PlotConfig] = None):
    """3D trajectory plot."""
    XYZ = tf.take_columns(Y, dims)
    fig, ax = new_fig_ax(cfg=cfg, projection="3d")
    ax.plot(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], lw=0.7)
    ax.set_xlabel(f"x[{dims[0]}]")
    ax.set_ylabel(f"x[{dims[1]}]")
    ax.set_zlabel(f"x[{dims[2]}]")
    ax.set_title("Trajectory (3D)")
    return fig, ax


# ------------------------- Phase portraits & density -------------------------

def phase_portrait(Y: np.ndarray, dims: Tuple[int, int] = (0, 1),
                   cfg: Optional[PlotConfig] = None):
    """Scatter phase portrait (thin)."""
    XY = tf.take_columns(Y, dims)
    fig, ax = new_fig_ax(cfg=cfg)
    ax.plot(XY[:, 0], XY[:, 1], ",", alpha=0.5)
    ax.set_xlabel(f"x[{dims[0]}]")
    ax.set_ylabel(f"x[{dims[1]}]")
    ax.set_title("Phase portrait")
    return fig, ax


def phase_density(Y: np.ndarray, dims: Tuple[int, int] = (0, 1),
                  bins: int | Tuple[int, int] = 200,
                  cfg: Optional[PlotConfig] = None):
    """2D density map (hist2d) of phase space."""
    XY = tf.take_columns(Y, dims)
    H, xe, ye = tf.phase_space_density(XY, bins=bins)
    fig, ax = new_fig_ax(cfg=cfg)
    im = ax.imshow(H, origin="lower", aspect="auto",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]])
    ax.set_xlabel(f"x[{dims[0]}]")
    ax.set_ylabel(f"x[{dims[1]}]")
    ax.set_title("Phase-space density")
    fig.colorbar(im, ax=ax, label="density")
    return fig, ax


# ------------------------- Poincaré & return maps -------------------------

def poincare_scatter(times: np.ndarray, Y: np.ndarray, *,
                     section_dim: int = 0, section_value: float = 0.0,
                     direction: str = "positive",
                     extract_dims: Tuple[int, int] | Tuple[int, int, int] = (1, 2),
                     cfg: Optional[PlotConfig] = None):
    """Plot Poincaré section intersections."""
    t_hits, pts = tf.poincare_section(times, Y, section_dim=section_dim,
                                      section_value=section_value,
                                      direction=direction,
                                      extract_dims=extract_dims)
    fig, ax = new_fig_ax(cfg=cfg, projection="3d" if len(extract_dims) == 3 else None)
    if pts.size == 0:
        ax.set_title("No intersections found")
        return fig, ax
    if pts.shape[1] == 2:
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.7)
        ax.set_xlabel(f"x[{extract_dims[0]}]")
        ax.set_ylabel(f"x[{extract_dims[1]}]")
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=12, alpha=0.7)
        ax.set_xlabel(f"x[{extract_dims[0]}]")
        ax.set_ylabel(f"x[{extract_dims[1]}]")
        ax.set_zlabel(f"x[{extract_dims[2]}]")
    ax.set_title("Poincaré section")
    return fig, ax


def return_map(x: np.ndarray, lag: int = 1, cfg: Optional[PlotConfig] = None):
    """Lagged return map x_t vs x_{t+lag}."""
    x0, x1 = tf.return_map(x, lag=lag)
    fig, ax = new_fig_ax(cfg=cfg)
    ax.plot(x0, x1, ".", alpha=0.6)
    ax.set_xlabel("x(t)")
    ax.set_ylabel(f"x(t+{lag})")
    ax.set_title("Return map")
    return fig, ax


# ------------------------- Recurrence plots -------------------------

def recurrence_plot(X: np.ndarray, eps: Optional[float] = None, percent: Optional[float] = 10.0,
                    metric: str = "euclidean", cfg: Optional[PlotConfig] = None):
    """Binary recurrence plot."""
    R = tf.recurrence_matrix(X, eps=eps, percent=percent, metric=metric)
    fig, ax = new_fig_ax(cfg=cfg)
    im = ax.imshow(R, cmap="binary", origin="lower", interpolation="nearest")
    ax.set_xlabel("t")
    ax.set_ylabel("t'")
    ax.set_title("Recurrence plot")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def cross_recurrence_plot(X: np.ndarray, Y: np.ndarray, eps: Optional[float] = None, percent: Optional[float] = 10.0,
                          metric: str = "euclidean", cfg: Optional[PlotConfig] = None):
    """Cross-recurrence plot."""
    R = tf.cross_recurrence_matrix(X, Y, eps=eps, percent=percent, metric=metric)
    fig, ax = new_fig_ax(cfg=cfg)
    im = ax.imshow(R, cmap="binary", origin="lower", interpolation="nearest")
    ax.set_xlabel("t (X)")
    ax.set_ylabel("t' (Y)")
    ax.set_title("Cross-recurrence plot")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def joint_recurrence_plot(*Rs: np.ndarray, cfg: Optional[PlotConfig] = None):
    """Joint-recurrence plot from precomputed binary matrices."""
    R = tf.joint_recurrence_matrix(*Rs)
    fig, ax = new_fig_ax(cfg=cfg)
    im = ax.imshow(R, cmap="binary", origin="lower", interpolation="nearest")
    ax.set_title("Joint-recurrence plot")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


# ------------------------- Spectral & wavelet -------------------------

def power_spectrum(x: np.ndarray, fs: float = 1.0, method: str = "welch",
                   cfg: Optional[PlotConfig] = None):
    """Power spectrum via FFT or Welch."""
    if method == "welch":
        f, Pxx = tf.power_spectrum_welch(x, fs=fs)
    elif method == "fft":
        f, Pxx = tf.power_spectrum_fft(x, fs=fs)
    else:
        raise ValueError("method must be 'welch' or 'fft'")
    fig, ax = new_fig_ax(cfg=cfg)
    ax.semilogy(f, Pxx + 1e-20)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("PSD")
    ax.set_title(f"Power spectrum ({method})")
    return fig, ax


def wavelet_scalogram(x: np.ndarray, fs: float, wavelet: str = "morl",
                      widths: Optional[np.ndarray] = None,
                      cfg: Optional[PlotConfig] = None):
    """Wavelet scalogram (requires PyWavelets)."""
    t, scales, A = tf.wavelet_scalogram(x, fs=fs, wavelet=wavelet, widths=widths)
    fig, ax = new_fig_ax(cfg=cfg)
    im = ax.imshow(A, extent=[t[0], t[-1], scales[-1], scales[0]],
                   aspect="auto", cmap="viridis")
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale")
    ax.set_title("Wavelet scalogram")
    fig.colorbar(im, ax=ax, label="|CWT|")
    return fig, ax


# ------------------------- Embeddings & projections -------------------------

def embedding_plot(x: np.ndarray, m: int, tau: int, cfg: Optional[PlotConfig] = None):
    """Plot 2D/3D delay embedding of a scalar series."""
    E = tf.delay_embedding(x, m=m, tau=tau)
    if m == 2:
        fig, ax = new_fig_ax(cfg=cfg)
        ax.plot(E[:, 0], E[:, 1], ".", alpha=0.6)
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t+{tau})")
        ax.set_title("Delay embedding (2D)")
        return fig, ax
    elif m >= 3:
        fig, ax = new_fig_ax(cfg=cfg, projection="3d")
        ax.plot(E[:, 0], E[:, 1], E[:, 2], ",", alpha=0.6)
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t+{tau})")
        ax.set_zlabel(f"x(t+{2*tau})")
        ax.set_title("Delay embedding (3D)")
        return fig, ax
    else:
        raise ValueError("m must be >= 2 for a meaningful embedding plot.")


def pca_projection_plot(Y: np.ndarray, n_components: int = 2, cfg: Optional[PlotConfig] = None):
    """Project trajectory onto first principal components."""
    comps, var, mu = tf.pca_project(Y, n_components=n_components)
    Z = tf.project_with_components(Y, comps, mu)
    if n_components == 2:
        fig, ax = new_fig_ax(cfg=cfg)
        ax.plot(Z[:, 0], Z[:, 1], ",", alpha=0.6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA projection (var: {var[0]:.2g}, {var[1]:.2g})")
        return fig, ax
    elif n_components == 3:
        fig, ax = new_fig_ax(cfg=cfg, projection="3d")
        ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], ",", alpha=0.6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA projection (3D)")
        return fig, ax
    else:
        raise ValueError("n_components must be 2 or 3 for plotting.")


# ------------------------- Distance maps & bifurcations -------------------------

def distance_heatmap(X: np.ndarray, metric: str = "euclidean", cfg: Optional[PlotConfig] = None):
    """Pairwise distance heatmap."""
    D = tf.distance_matrix(X, metric=metric)
    fig, ax = new_fig_ax(cfg=cfg)
    im = ax.imshow(D, origin="lower", aspect="auto")
    ax.set_title(f"Distance matrix ({metric})")
    fig.colorbar(im, ax=ax)
    return fig, ax


def bifurcation_diagram(params: np.ndarray, points: Sequence[np.ndarray], *,
                        cfg: Optional[PlotConfig] = None,
                        s: float = 2.0, alpha: float = 0.5):
    """
    Generic bifurcation diagram.

    Parameters
    ----------
    params : (M,) array
        Parameter values (one per run).
    points : list of arrays
        For each parameter p, an array of asymptotic samples (e.g., peaks or tail samples).
        See `transforms.asymptotic_samples`.
    """
    params = np.asarray(params, float)
    if len(points) != params.size:
        raise ValueError("len(points) must equal len(params).")
    fig, ax = new_fig_ax(cfg=cfg)
    for p, vals in zip(params, points):
        if vals is None or len(vals) == 0:
            continue
        ax.scatter(np.full_like(vals, p, dtype=float), vals, s=s, alpha=alpha, edgecolor="none")
    ax.set_xlabel("parameter")
    ax.set_ylabel("observable")
    ax.set_title("Bifurcation diagram")
    return fig, ax


# ------------------------- Lyapunov spectrum -------------------------

def lyapunov_spectrum(exponents: np.ndarray, cfg: Optional[PlotConfig] = None):
    """Bar plot of Lyapunov exponents."""
    exponents = np.asarray(exponents, float).reshape(-1)
    fig, ax = new_fig_ax(cfg=cfg)
    ax.bar(np.arange(exponents.size), exponents)
    ax.set_xlabel("index")
    ax.set_ylabel("Lyapunov exponent")
    ax.set_title("Lyapunov spectrum")
    return fig, ax


# ------------------------- Space–time diagrams -------------------------

def kymograph(times: np.ndarray, Y: np.ndarray, *,
              nx: Optional[int] = None, ny: Optional[int] = None,
              cfg: Optional[PlotConfig] = None):
    """
    Space–time diagram (kymograph).

    Y shape expected (T, N). If ny is None, assume 1D field of length nx (=N).
    If nx*ny == N, reshape each row to (ny, nx) and display as image sequence in time,
    but here we aggregate into a single image (time vs space for 1D).
    """
    T, N = Y.shape
    if ny is None:
        if nx is None:
            nx = N
        if nx != N:
            raise ValueError("For 1D kymograph, nx must equal N.")
        # time (rows) vs space (cols)
        fig, ax = new_fig_ax(cfg=cfg)
        im = ax.imshow(Y, aspect="auto", origin="lower", extent=[0, nx, times[0], times[-1]])
        ax.set_xlabel("space index")
        ax.set_ylabel("time")
        ax.set_title("Kymograph (1D)")
        fig.colorbar(im, ax=ax)
        return fig, ax
    else:
        if nx is None or nx * ny != N:
            raise ValueError("Provide consistent nx, ny with nx*ny == Y.shape[1].")
        # Show a snapshot grid? For static, we can show mean over time or the last frame.
        last = Y[-1].reshape(ny, nx)
        fig, ax = new_fig_ax(cfg=cfg)
        im = ax.imshow(last, origin="lower", aspect="auto")
        ax.set_title("Field snapshot (last frame)")
        fig.colorbar(im, ax=ax)
        return fig, ax
