"""Matplotlib animations (trajectories and space–time)."""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from matplotlib.animation import FuncAnimation

from .base import new_fig_ax, PlotConfig
from . import transforms as tf


def animate_trajectory2d(times: np.ndarray, Y: np.ndarray, dims: Tuple[int, int] = (0, 1),
                         interval_ms: int = 30, trail: int = 300,
                         cfg: Optional[PlotConfig] = None):
    """Animate a 2D trajectory with a trailing path."""
    XY = tf.take_columns(Y, dims)
    fig, ax = new_fig_ax(cfg=cfg)
    line, = ax.plot([], [], lw=1.0)
    head, = ax.plot([], [], "o", ms=4)
    ax.set_xlabel(f"x[{dims[0]}]")
    ax.set_ylabel(f"x[{dims[1]}]")
    ax.set_title("Animated trajectory (2D)")
    ax.set_xlim(np.min(XY[:, 0]), np.max(XY[:, 0]))
    ax.set_ylim(np.min(XY[:, 1]), np.max(XY[:, 1]))

    def init():
        line.set_data([], [])
        head.set_data([], [])
        return line, head

    def update(i):
        j0 = max(0, i - trail)
        xs, ys = XY[j0:i+1, 0], XY[j0:i+1, 1]
        line.set_data(xs, ys)
        head.set_data([XY[i, 0]], [XY[i, 1]])
        return line, head

    anim = FuncAnimation(fig, update, frames=len(times), init_func=init,
                         interval=interval_ms, blit=True)
    return anim


def animate_trajectory3d(times: np.ndarray, Y: np.ndarray, dims: Tuple[int, int, int] = (0, 1, 2),
                         interval_ms: int = 30, trail: int = 300,
                         cfg: Optional[PlotConfig] = None):
    """Animate a 3D trajectory with a trailing path."""
    XYZ = tf.take_columns(Y, dims)
    fig, ax = new_fig_ax(cfg=cfg, projection="3d")
    line, = ax.plot([], [], [], lw=1.0)
    head, = ax.plot([], [], [], "o", ms=4)
    ax.set_xlabel(f"x[{dims[0]}]")
    ax.set_ylabel(f"x[{dims[1]}]")
    ax.set_zlabel(f"x[{dims[2]}]")
    ax.set_title("Animated trajectory (3D)")
    ax.set_xlim(np.min(XYZ[:, 0]), np.max(XYZ[:, 0]))
    ax.set_ylim(np.min(XYZ[:, 1]), np.max(XYZ[:, 1]))
    ax.set_zlim(np.min(XYZ[:, 2]), np.max(XYZ[:, 2]))

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        head.set_data([], [])
        head.set_3d_properties([])
        return line, head

    def update(i):
        j0 = max(0, i - trail)
        xs, ys, zs = XYZ[j0:i+1, 0], XYZ[j0:i+1, 1], XYZ[j0:i+1, 2]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        head.set_data([XYZ[i, 0]], [XYZ[i, 1]])
        head.set_3d_properties([XYZ[i, 2]])
        return line, head

    anim = FuncAnimation(fig, update, frames=len(times), init_func=init,
                         interval=interval_ms, blit=True)
    return anim


def animate_space_time(times: np.ndarray, Y: np.ndarray, *,
                       interval_ms: int = 30,
                       cfg: Optional[PlotConfig] = None):
    """
    Animate a 1D field y(x,t) given as Y shape (T, N): show y vs x over time.
    """
    T, N = Y.shape
    x = np.arange(N)
    fig, ax = new_fig_ax(cfg=cfg)
    (line,) = ax.plot(x, Y[0], lw=1.0)
    ax.set_xlim(0, N - 1)
    ymin, ymax = np.min(Y), np.max(Y)
    if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel("space index")
    ax.set_ylabel("value")
    ax.set_title("Space–time curve evolution")

    def update(i):
        line.set_ydata(Y[i])
        return (line,)

    anim = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=True)
    return anim
