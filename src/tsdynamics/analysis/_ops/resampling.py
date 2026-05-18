"""Resampling and windowing primitives."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline

from .._registry import trajectory_op


@trajectory_op(returns="trajectory")
def decimate(
    t: np.ndarray,
    y: np.ndarray,
    every: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep every ``every``-th sample.

    Parameters
    ----------
    every : int
        Stride.  Must be ``>= 1``.

    Returns
    -------
    Trajectory

    Examples
    --------
    >>> traj.decimate(every=10)            # 10x fewer samples
    >>> decimate(traj, every=10)           # equivalent free-function form
    """
    if not isinstance(every, int | np.integer) or every < 1:
        raise ValueError(f"decimate: 'every' must be a positive integer, got {every!r}")
    return t[::every].copy(), y[::every].copy()


@trajectory_op(returns="trajectory")
def resample(
    t: np.ndarray,
    y: np.ndarray,
    dt_new: float,
    kind: str = "cubic",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample onto a uniform grid with spacing ``dt_new``.

    The new time grid is ``np.arange(t[0], t[-1], dt_new)``; the final value
    of ``t`` may not be hit exactly to keep the grid uniform.

    Parameters
    ----------
    dt_new : float
        New step.  Must be positive.
    kind : {"linear", "cubic"}, default "cubic"
        Interpolation kind, via :func:`scipy.interpolate.make_interp_spline`.

    Examples
    --------
    >>> traj.resample(dt_new=0.01, kind="cubic")
    """
    if dt_new <= 0:
        raise ValueError(f"resample: 'dt_new' must be positive, got {dt_new}")
    if t.size < 2:
        raise ValueError("resample: need at least 2 samples")
    if not np.all(np.diff(t) > 0):
        raise ValueError("resample: input 't' must be strictly increasing")

    if kind == "linear":
        k = 1
    elif kind == "cubic":
        k = 3
    else:
        raise ValueError(f"resample: unknown kind {kind!r} (allowed: 'linear', 'cubic')")

    if t.size <= k:
        raise ValueError(f"resample: kind={kind!r} needs at least {k + 1} samples, got {t.size}")

    t_new = np.arange(t[0], t[-1], dt_new)
    spline = make_interp_spline(t, y, k=k, axis=0)
    y_new = spline(t_new)
    return t_new, np.asarray(y_new)


@trajectory_op(returns="trajectory")
def project(
    t: np.ndarray,
    y: np.ndarray,
    dims: tuple[int, ...] | list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a subset of state components.

    Parameters
    ----------
    dims : iterable of int
        Component indices to keep.  Order is preserved; repeats are allowed.

    Examples
    --------
    >>> traj.project(dims=(0, 2))          # only x and z
    """
    dims = list(dims)
    if not dims:
        raise ValueError("project: 'dims' must be non-empty")
    dim = y.shape[1]
    for d in dims:
        if not isinstance(d, int | np.integer):
            raise TypeError(f"project: dims must be integers, got {type(d).__name__}")
        if d < -dim or d >= dim:
            raise IndexError(f"project: dim {d} out of range for state dim {dim}")
    return t.copy(), y[:, dims].copy()


@trajectory_op(returns="trajectory")
def window(
    t: np.ndarray,
    y: np.ndarray,
    t0: float | None = None,
    t1: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Restrict to ``[t0, t1]`` (inclusive on both ends).

    Either bound may be ``None`` to leave that side open.  Bounds outside
    the available time range are clipped silently.

    Parameters
    ----------
    t0, t1 : float or None

    Examples
    --------
    >>> traj.window(t0=50.0, t1=100.0)
    """
    if t0 is not None and t1 is not None and t0 > t1:
        raise ValueError(f"window: t0={t0} > t1={t1}")
    mask = np.ones_like(t, dtype=bool)
    if t0 is not None:
        mask &= t >= t0
    if t1 is not None:
        mask &= t <= t1
    return t[mask].copy(), y[mask].copy()


__all__ = ["decimate", "resample", "project", "window"]
