"""
M1 trajectory-enrichment primitives.

Every function here takes ``(t, y, *args, **kwargs)`` and returns a
shape declared by its :func:`trajectory_op` decoration.  The decorator
turns each one into a polymorphic free function (accepts a
:class:`Trajectory`, a ``(t, y)`` tuple, *or* bare arrays) and queues
it for installation as a method on :class:`Trajectory`.

There is no hand-written wrapper anywhere else: the function below is
the single source of truth for both ``decimate(traj, every=5)`` and
``traj.decimate(every=5)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks

from ._registry import trajectory_op

# ---------------------------------------------------------------------------
# Resampling / windowing
# ---------------------------------------------------------------------------


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
    Trajectory (method form) or (t_new, y_new) tuple (free-function form
    with bare arrays / ``(t, y)`` tuple input).

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
        Interpolation kind, implemented via
        :func:`scipy.interpolate.make_interp_spline`.

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


# ---------------------------------------------------------------------------
# Differential / norm
# ---------------------------------------------------------------------------


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


@trajectory_op(returns="passthrough")
def norm(t: np.ndarray, y: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Euclidean norm of the state at each time, ``||y(t)||``.

    The ``t`` argument is unused but kept to satisfy the universal
    ``(t, y, ...)`` calling convention.

    Parameters
    ----------
    axis : int, default 1

    Returns
    -------
    ndarray, shape ``(T,)``

    Examples
    --------
    >>> r = traj.norm()
    """
    del t  # universal-signature placeholder
    return np.linalg.norm(y, axis=axis)


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------


def _extract_component(y: np.ndarray, component: int) -> np.ndarray:
    """Return ``y[:, component]`` with bounds-checking."""
    dim = y.shape[1]
    if not isinstance(component, int | np.integer):
        raise TypeError(f"component must be an integer, got {type(component).__name__}")
    if component < -dim or component >= dim:
        raise IndexError(f"component {component} out of range for state dim {dim}")
    return y[:, component]


@trajectory_op(returns="passthrough")
def local_maxima(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    **find_peaks_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate local maxima of one state component.

    Thin wrapper around :func:`scipy.signal.find_peaks`; any additional
    keyword arguments (``prominence``, ``distance``, ``height``, …) are
    forwarded directly.  For sub-sample-accurate extrema, use
    :class:`tsdynamics.analysis.LocalExtremum` with
    :func:`detect_events` instead — that uses cubic-Hermite refinement.

    Parameters
    ----------
    component : int, default 0
    **find_peaks_kwargs
        Forwarded to :func:`scipy.signal.find_peaks`.

    Returns
    -------
    (t_peaks, y_peaks) : tuple of ndarrays

    Examples
    --------
    >>> tp, yp = traj.local_maxima(component=2, prominence=1.0)
    """
    sig = _extract_component(y, component)
    idx, _ = find_peaks(sig, **find_peaks_kwargs)
    return t[idx].copy(), sig[idx].copy()


@trajectory_op(returns="passthrough")
def local_minima(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    **find_peaks_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate local minima of one state component.

    Implemented as :func:`local_maxima` applied to ``-y[:, component]`` so
    every kwarg accepted by ``find_peaks`` works the same way for troughs.

    Parameters
    ----------
    component : int, default 0
    **find_peaks_kwargs
        Forwarded to :func:`scipy.signal.find_peaks`.

    Returns
    -------
    (t_mins, y_mins) : tuple of ndarrays
    """
    sig = _extract_component(y, component)
    idx, _ = find_peaks(-sig, **find_peaks_kwargs)
    return t[idx].copy(), sig[idx].copy()


@trajectory_op(returns="passthrough")
def return_times(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    **find_peaks_kwargs: Any,
) -> np.ndarray:
    """
    Inter-peak intervals on one state component — the "poor man's period".

    For a clean ``sin(ω t)`` the result is a vector whose entries are all
    close to ``2π / ω``.  The proper spectral version lands in M7.

    Parameters
    ----------
    component : int, default 0
    **find_peaks_kwargs
        Forwarded to :func:`scipy.signal.find_peaks`.

    Returns
    -------
    ndarray, shape ``(n_peaks - 1,)``
        Empty if fewer than two peaks were found.
    """
    sig = _extract_component(y, component)
    idx, _ = find_peaks(sig, **find_peaks_kwargs)
    if idx.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(t[idx])


# ---------------------------------------------------------------------------
# DataSpec placeholder (V1 will replace this with the real class)
# ---------------------------------------------------------------------------

_DATASPEC_SCHEMAS: dict[str, tuple[str, ...]] = {
    "timeseries": ("dims",),
    "phase_portrait_2d": ("dims",),
    "phase_portrait_3d": ("dims",),
}

_DATASPEC_REQUIRED_DIMS: dict[str, int] = {
    "phase_portrait_2d": 2,
    "phase_portrait_3d": 3,
}


@trajectory_op(returns="passthrough")
def to_dataspec(
    t: np.ndarray,
    y: np.ndarray,
    kind: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Build a plain dict describing a trajectory view — V1 ``DataSpec`` shim.

    The schema is intentionally simple — V1 will swap in a real
    ``DataSpec`` class without breaking call sites because the keys here
    mirror what that class will store.

    Parameters
    ----------
    kind : {"timeseries", "phase_portrait_2d", "phase_portrait_3d"}
    dims : iterable of int, optional
        Component indices.  Required for the phase-portrait kinds;
        defaults to all components for ``timeseries``.
    """
    if kind not in _DATASPEC_SCHEMAS:
        raise ValueError(
            f"to_dataspec: unknown kind {kind!r} (allowed: {sorted(_DATASPEC_SCHEMAS)})"
        )

    dims = kwargs.pop("dims", None)
    if dims is None:
        if kind in _DATASPEC_REQUIRED_DIMS:
            raise ValueError(f"to_dataspec: kind={kind!r} requires 'dims='")
        dims = tuple(range(y.shape[1]))
    dims = tuple(int(d) for d in dims)

    required = _DATASPEC_REQUIRED_DIMS.get(kind)
    if required is not None and len(dims) != required:
        raise ValueError(
            f"to_dataspec: kind={kind!r} requires exactly {required} dims, got {len(dims)}"
        )

    for d in dims:
        if d < -y.shape[1] or d >= y.shape[1]:
            raise IndexError(f"to_dataspec: dim {d} out of range for state dim {y.shape[1]}")

    return {"kind": kind, "t": t, "y": y, "dims": dims, **kwargs}


__all__ = [
    "decimate",
    "derivative",
    "local_maxima",
    "local_minima",
    "norm",
    "project",
    "resample",
    "return_times",
    "to_dataspec",
    "window",
]
