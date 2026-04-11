"""
Matplotlib animation wrappers for TSDynamics trajectories.

Every function follows the ``ax=None`` contract (see ``plotters.py``) and
returns a ``matplotlib.animation.FuncAnimation`` rather than ``ax``.
Never calls ``plt.show()`` — the caller decides when to display or save.
"""

from __future__ import annotations

import numpy as np
from matplotlib.animation import FuncAnimation

from ._utils import _resolve_ax


def animate_trajectory(
    t: np.ndarray,
    X: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    interval_ms: int = 30,
    trail: int = 300,
    ax=None,
    **kwargs,
) -> FuncAnimation:
    """
    Animate a 2-D trajectory with a trailing path.

    Parameters
    ----------
    t : ndarray, shape (m,)
    X : ndarray, shape (m, n_dim)
    dims : (int, int)
        Column indices to use.
    interval_ms : int
        Delay between frames in milliseconds.
    trail : int
        Number of past points to keep visible.
    ax : matplotlib.axes.Axes, optional
        Created internally if *None*.
    **kwargs
        Forwarded to the trail ``ax.plot`` call.

    Returns
    -------
    anim : FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 4 * np.pi, 200)
    >>> X = np.column_stack([np.sin(t), np.cos(t), t])
    >>> anim = animate_trajectory(t, X)
    >>> type(anim).__name__
    'FuncAnimation'
    """
    X = np.asarray(X, float)
    ax = _resolve_ax(ax)
    XY = X[:, list(dims)]

    kwargs.setdefault("lw", 1.0)
    (line,) = ax.plot([], [], **kwargs)
    (head,) = ax.plot([], [], "o", ms=4, color=line.get_color())
    ax.set_xlabel(f"$x_{{{dims[0]}}}$")
    ax.set_ylabel(f"$x_{{{dims[1]}}}$")
    ax.set_xlim(float(XY[:, 0].min()), float(XY[:, 0].max()))
    ax.set_ylim(float(XY[:, 1].min()), float(XY[:, 1].max()))

    def init():
        line.set_data([], [])
        head.set_data([], [])
        return line, head

    def update(i):
        j0 = max(0, i - trail)
        line.set_data(XY[j0 : i + 1, 0], XY[j0 : i + 1, 1])
        head.set_data([XY[i, 0]], [XY[i, 1]])
        return line, head

    return FuncAnimation(
        ax.figure, update, frames=len(t), init_func=init, interval=interval_ms, blit=True
    )


def animate_trajectory_3d(
    t: np.ndarray,
    X: np.ndarray,
    dims: tuple[int, int, int] = (0, 1, 2),
    interval_ms: int = 30,
    trail: int = 300,
    ax=None,
    **kwargs,
) -> FuncAnimation:
    """
    Animate a 3-D trajectory with a trailing path.

    Parameters
    ----------
    t : ndarray, shape (m,)
    X : ndarray, shape (m, n_dim), n_dim >= 3
    dims : (int, int, int)
    interval_ms : int
    trail : int
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        A 3D axes. Created internally if *None*.
    **kwargs
        Forwarded to the trail ``ax.plot`` call.

    Returns
    -------
    anim : FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 4 * np.pi, 200)
    >>> X = np.column_stack([np.sin(t), np.cos(t), t / (4 * np.pi)])
    >>> anim = animate_trajectory_3d(t, X)
    >>> type(anim).__name__
    'FuncAnimation'
    """
    X = np.asarray(X, float)
    ax = _resolve_ax(ax, projection="3d")
    XYZ = X[:, list(dims)]

    kwargs.setdefault("lw", 1.0)
    (line,) = ax.plot([], [], [], **kwargs)
    (head,) = ax.plot([], [], [], "o", ms=4, color=line.get_color())
    ax.set_xlabel(f"$x_{{{dims[0]}}}$")
    ax.set_ylabel(f"$x_{{{dims[1]}}}$")
    if hasattr(ax, "set_zlabel"):
        ax.set_zlabel(f"$x_{{{dims[2]}}}$")
    ax.set_xlim(float(XYZ[:, 0].min()), float(XYZ[:, 0].max()))
    ax.set_ylim(float(XYZ[:, 1].min()), float(XYZ[:, 1].max()))
    ax.set_zlim(float(XYZ[:, 2].min()), float(XYZ[:, 2].max()))

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        head.set_data([], [])
        head.set_3d_properties([])
        return line, head

    def update(i):
        j0 = max(0, i - trail)
        xs, ys, zs = XYZ[j0 : i + 1, 0], XYZ[j0 : i + 1, 1], XYZ[j0 : i + 1, 2]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        head.set_data([XYZ[i, 0]], [XYZ[i, 1]])
        head.set_3d_properties([XYZ[i, 2]])
        return line, head

    return FuncAnimation(
        ax.figure, update, frames=len(t), init_func=init, interval=interval_ms, blit=True
    )


def animate_spacetime(
    t: np.ndarray,
    X: np.ndarray,
    *,
    interval_ms: int = 30,
    ax: object | None = None,
    **kwargs,
) -> FuncAnimation:
    """
    Animate a 1-D spatial field y(x, t) as a line evolving over time.

    Each frame shows the field profile at one time step.

    Parameters
    ----------
    t : ndarray, shape (T,)
    X : ndarray, shape (T, N)
        Each row is the field at one time step; N is the spatial dimension.
    interval_ms : int
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to the initial ``ax.plot`` call.

    Returns
    -------
    anim : FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 5, 100)
    >>> X = np.random.default_rng(0).standard_normal((100, 32))
    >>> anim = animate_spacetime(t, X)
    >>> type(anim).__name__
    'FuncAnimation'
    """
    X = np.asarray(X, float)
    T, N = X.shape
    x_idx = np.arange(N)
    ax = _resolve_ax(ax)

    kwargs.setdefault("lw", 1.0)
    (line,) = ax.plot(x_idx, X[0], **kwargs)
    ax.set_xlim(0, N - 1)
    ymin, ymax = float(np.min(X)), float(np.max(X))
    if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel("spatial index")
    ax.set_ylabel("value")

    def update(i):
        line.set_ydata(X[i])
        return (line,)

    return FuncAnimation(ax.figure, update, frames=T, interval=interval_ms, blit=True)
