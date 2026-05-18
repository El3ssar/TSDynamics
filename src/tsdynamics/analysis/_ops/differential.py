"""Differential and norm primitives."""

from __future__ import annotations

import numpy as np

from .._registry import trajectory_op


@trajectory_op(returns="ndarray_keep_t")
def derivative(
    t: np.ndarray,
    y: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """
    Numerical derivative along time, via repeated central differences.

    Uses :func:`numpy.gradient` (second-order accurate central differences
    in the interior, first-order at the edges); higher orders recurse so
    edge accuracy degrades with each application.

    Parameters
    ----------
    order : int, default 1
        Differentiation order.  Must be ``>= 1``.

    Examples
    --------
    >>> dtraj = traj.derivative(order=1)   # ẏ(t)
    """
    if not isinstance(order, int | np.integer) or order < 1:
        raise ValueError(f"derivative: 'order' must be a positive integer, got {order!r}")
    out = np.asarray(y, dtype=float)
    for _ in range(int(order)):
        out = np.gradient(out, t, axis=0)
    return out


@trajectory_op(returns="trajectory")
def norm(
    t: np.ndarray,
    y: np.ndarray,
    axis: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean norm of the state at each time, ``||y(t)||``.

    Parameters
    ----------
    axis : int, default 1
        Axis to reduce over.  The default reduces the state axis so the
        result is one scalar per time point.

    Returns
    -------
    Trajectory
        ``t`` unchanged; ``y`` has shape ``(T, 1)`` — the norm value.

    Examples
    --------
    >>> r = traj.norm()
    >>> r.y.shape
    (T, 1)
    """
    r = np.linalg.norm(y, axis=axis)
    return t.copy(), r.reshape(-1, 1)


__all__ = ["derivative", "norm"]
