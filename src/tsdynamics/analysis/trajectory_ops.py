"""
Pure (t, y) operations behind ``Trajectory`` enrichment methods.

Each function here takes the trajectory's ``t`` (shape ``(T,)``) and ``y``
(shape ``(T, dim)``) arrays plus operation-specific arguments, and returns
new ndarrays (or auxiliary scalars).  No reference to :class:`Trajectory` is
needed, which keeps the algorithms independently unit-testable and reusable
from later analysis primitives that don't carry a system back-reference.

All functions are pure — they never mutate their inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Resampling / windowing
# ---------------------------------------------------------------------------


def decimate(
    t: np.ndarray,
    y: np.ndarray,
    every: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep every ``every``-th sample.

    Parameters
    ----------
    t : ndarray, shape (T,)
    y : ndarray, shape (T, dim)
    every : int
        Stride.  Must be ``>= 1``.

    Returns
    -------
    (t_new, y_new) : tuple of ndarrays
        Both views are copies (not slices) so downstream mutation cannot
        leak back into the source arrays.

    Examples
    --------
    >>> t = np.arange(10.0)
    >>> y = np.arange(20.0).reshape(10, 2)
    >>> tn, yn = decimate(t, y, every=3)
    >>> tn
    array([0., 3., 6., 9.])
    """
    if not isinstance(every, int | np.integer) or every < 1:
        raise ValueError(f"decimate: 'every' must be a positive integer, got {every!r}")
    return t[::every].copy(), y[::every].copy()


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
    t : ndarray, shape (T,)
        Must be strictly increasing.
    y : ndarray, shape (T, dim)
    dt_new : float
        New step.  Must be positive.
    kind : {"linear", "cubic"}, default "cubic"
        Interpolation kind.

    Returns
    -------
    (t_new, y_new) : tuple of ndarrays

    Raises
    ------
    ValueError
        If ``t`` is not strictly increasing, ``dt_new`` is non-positive,
        or ``kind`` is unrecognised.

    Examples
    --------
    >>> t = np.linspace(0.0, 1.0, 11)
    >>> y = np.sin(2 * np.pi * t)[:, None]
    >>> tn, yn = resample(t, y, dt_new=0.05, kind="cubic")
    >>> tn.shape, yn.shape
    ((20,), (20, 1))
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
    # make_interp_spline with k=1 returns float arrays but may return a 1-D shape
    # if y was 1-D; we always operate on (T, dim), so y_new is (T_new, dim).
    return t_new, np.asarray(y_new)


def project(
    t: np.ndarray,
    y: np.ndarray,
    dims: tuple[int, ...] | list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a subset of components.

    Parameters
    ----------
    t : ndarray, shape (T,)
    y : ndarray, shape (T, dim)
    dims : iterable of int
        Component indices to keep.  Order is preserved; repeats are allowed.

    Returns
    -------
    (t_new, y_new) : tuple of ndarrays
        ``y_new`` has shape ``(T, len(dims))``.

    Raises
    ------
    IndexError
        If any element of ``dims`` is out of range for ``y``.
    ValueError
        If ``dims`` is empty.

    Examples
    --------
    >>> t = np.arange(3.0)
    >>> y = np.arange(9.0).reshape(3, 3)
    >>> _, yp = project(t, y, dims=(0, 2))
    >>> yp
    array([[0., 2.],
           [3., 5.],
           [6., 8.]])
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


def window(
    t: np.ndarray,
    y: np.ndarray,
    t0: float | None = None,
    t1: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Restrict to the time interval ``[t0, t1]`` (inclusive on both ends).

    Either bound may be ``None`` to leave that side open.  Bounds that fall
    outside the available time range are clipped silently.

    Parameters
    ----------
    t : ndarray, shape (T,)
    y : ndarray, shape (T, dim)
    t0, t1 : float or None

    Returns
    -------
    (t_new, y_new) : tuple of ndarrays

    Raises
    ------
    ValueError
        If both bounds are provided and ``t0 > t1``.

    Examples
    --------
    >>> t = np.linspace(0.0, 10.0, 11)
    >>> y = t[:, None]
    >>> tw, _ = window(t, y, t0=2.5, t1=7.5)
    >>> tw
    array([3., 4., 5., 6., 7.])
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


def derivative(
    t: np.ndarray,
    y: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """
    Numerical derivative along time, via repeated central differences.

    Uses :func:`numpy.gradient`, which applies second-order accurate central
    differences in the interior and first-order at the edges.  For higher
    orders the operation is applied recursively, so accuracy degrades at the
    boundaries with each application.

    Parameters
    ----------
    t : ndarray, shape (T,)
    y : ndarray, shape (T, dim)
    order : int, default 1
        Differentiation order.  Must be ``>= 1``.

    Returns
    -------
    ndarray, shape (T, dim)

    Examples
    --------
    >>> t = np.linspace(0.0, 2 * np.pi, 200)
    >>> y = np.sin(t)[:, None]
    >>> dy = derivative(t, y, order=1)
    >>> bool(np.allclose(dy[1:-1, 0], np.cos(t[1:-1]), atol=1e-3))
    True
    """
    if not isinstance(order, int | np.integer) or order < 1:
        raise ValueError(f"derivative: 'order' must be a positive integer, got {order!r}")
    out = np.asarray(y, dtype=float)
    for _ in range(int(order)):
        out = np.gradient(out, t, axis=0)
    return out


def norm(y: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Euclidean norm of ``y`` along ``axis``.

    Parameters
    ----------
    y : ndarray
    axis : int, default 1
        Axis to reduce.  ``axis=1`` collapses the state-space dimension,
        producing a 1-D array of length ``T``.

    Returns
    -------
    ndarray
        Same shape as ``y`` with ``axis`` removed.

    Examples
    --------
    >>> y = np.array([[3.0, 4.0], [6.0, 8.0]])
    >>> norm(y)
    array([ 5., 10.])
    """
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


def local_maxima(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    **find_peaks_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate local maxima of one state component.

    Thin wrapper around :func:`scipy.signal.find_peaks`; any additional
    keyword arguments (``prominence``, ``distance``, ``height``, ``width``,
    …) are forwarded directly.

    Parameters
    ----------
    t : ndarray, shape (T,)
    y : ndarray, shape (T, dim)
    component : int, default 0
    **find_peaks_kwargs
        Forwarded to :func:`scipy.signal.find_peaks`.

    Returns
    -------
    (t_peaks, y_peaks) : tuple of ndarrays
        Times and values of the located maxima.

    Examples
    --------
    >>> t = np.linspace(0.0, 4 * np.pi, 4000)
    >>> y = np.sin(t)[:, None]
    >>> tp, _ = local_maxima(t, y, component=0)
    >>> tp.size
    2
    """
    sig = _extract_component(y, component)
    idx, _ = find_peaks(sig, **find_peaks_kwargs)
    return t[idx].copy(), sig[idx].copy()


def local_minima(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    **find_peaks_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate local minima of one state component.

    Implemented as :func:`local_maxima` applied to ``-y[:, component]`` so
    every kwarg accepted by ``find_peaks`` (e.g. ``prominence``) works the
    same way for troughs.

    Parameters
    ----------
    t : ndarray, shape (T,)
    y : ndarray, shape (T, dim)
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


def return_times(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    **find_peaks_kwargs: Any,
) -> np.ndarray:
    """
    Inter-peak intervals on one state component — the "poor man's period".

    For a clean ``sin(ω t)`` the result is a vector whose entries are all
    close to ``2π / ω``.  This is the rough analogue of a period estimate
    by zero-crossing; the proper spectral version lands in M7.

    Parameters
    ----------
    t : ndarray, shape (T,)
    y : ndarray, shape (T, dim)
    component : int, default 0
    **find_peaks_kwargs
        Forwarded to :func:`scipy.signal.find_peaks`.

    Returns
    -------
    ndarray, shape (n_peaks - 1,)
        Empty if fewer than two peaks were found.

    Examples
    --------
    >>> t = np.linspace(0.0, 10.0, 10000)
    >>> y = np.sin(2 * np.pi * t)[:, None]
    >>> isi = return_times(t, y)
    >>> bool(np.allclose(isi.mean(), 1.0, atol=1e-3))
    True
    """
    tp, _ = local_maxima(t, y, component=component, **find_peaks_kwargs)
    if tp.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(tp)


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


def to_dataspec(
    t: np.ndarray,
    y: np.ndarray,
    kind: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Build a plain dict describing a trajectory view, the V1 ``DataSpec`` shim.

    The schema is intentionally simple — V1 will swap in a real ``DataSpec``
    class without breaking the call sites because the keys here mirror what
    that class will store.

    Parameters
    ----------
    t : ndarray, shape (T,)
    y : ndarray, shape (T, dim)
    kind : {"timeseries", "phase_portrait_2d", "phase_portrait_3d"}
    dims : iterable of int, optional
        Component indices.  Required for the phase-portrait kinds; defaults
        to all components for ``timeseries``.

    Returns
    -------
    dict
        ``{"kind": ..., "t": ..., "y": ..., "dims": (..., ...)}``.

    Raises
    ------
    ValueError
        If ``kind`` is unknown or ``dims`` has the wrong cardinality.
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
