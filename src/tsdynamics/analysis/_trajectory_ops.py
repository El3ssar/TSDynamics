"""
Trajectory enrichment primitives.

Every function here takes ``(t, y, *args, **kwargs)`` and returns shape
declared by its :func:`trajectory_op` decoration.  The decorator turns
each one into a polymorphic free function (accepts a
:class:`~tsdynamics.base.Trajectory`, a ``(t, y)`` tuple, *or* bare
arrays) and queues it for installation as a method on
:class:`~tsdynamics.base.Trajectory`.

There is no hand-written wrapper anywhere else: the function below is
the single source of truth for both ``decimate(traj, every=5)`` and
``traj.decimate(every=5)``.

Uniform output contract
-----------------------

Every op here returns a :class:`~tsdynamics.base.Trajectory`.  Ops that
reduce the state to a scalar (norm, peaks, ISIs) return a Trajectory
with ``y.shape == (K, 1)`` — the scalar is just a one-column state.
This keeps downstream code (plotters, chained ops) shape-agnostic.

The single non-Trajectory exit point is :func:`to_dataspec`, which
builds a plain dict describing a trajectory view for the future V1
``DataSpec`` class.  See its docstring for the recognised kinds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq
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


def _refine_extrema(
    t: np.ndarray,
    y: np.ndarray,
    component: int,
    *,
    kind: str,
    find_peaks_kwargs: dict,
    rtol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sub-sample peak/trough refinement via cubic-Hermite derivative roots.

    Locates extrema of ``y[:, component]`` by:
    1. Finding integer-sample peaks via :func:`scipy.signal.find_peaks` so
       the user's ``prominence`` / ``distance`` / etc. kwargs are honoured.
    2. For each found peak at index ``k``, building the cubic-Hermite
       interpolant on the bracket ``[k-1, k+1]`` and locating the
       analytical root of its time-derivative.

    Returns ``(t_peaks, y_peaks)`` with shapes ``(K,)`` and ``(K,)``.
    """
    sig = _extract_component(y, component)
    work = -sig if kind == "min" else sig
    idx, _ = find_peaks(work, **find_peaks_kwargs)
    if idx.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    slopes = np.gradient(y, t, axis=0)
    t_peaks: list[float] = []
    y_peaks: list[float] = []
    for k in idx:
        # We need samples on both sides of the integer peak so the
        # cubic-Hermite derivative can change sign across [k-1, k+1].
        # Pick the half-bracket that the discrete peak straddles.
        if k <= 0 or k >= t.size - 1:
            # Edge peak — no interior bracket; keep the sample value.
            t_peaks.append(float(t[k]))
            y_peaks.append(float(sig[k]))
            continue

        k_lo, k_hi = k - 1, k
        t_a, t_b = float(t[k_lo]), float(t[k_hi])
        dt_ab = t_b - t_a
        y_a, y_b = y[k_lo], y[k_hi]
        m_a, m_b = slopes[k_lo], slopes[k_hi]

        def dydt_left(
            s: float,
            dt: float = dt_ab,
            y_a: np.ndarray = y_a,
            y_b: np.ndarray = y_b,
            m_a: np.ndarray = m_a,
            m_b: np.ndarray = m_b,
            c: int = component,
        ) -> float:
            return float(_hermite_slope(s, y_a, y_b, m_a, m_b, dt)[c])

        d_lo_left = dydt_left(0.0)
        d_hi_left = dydt_left(1.0)

        k2_lo, k2_hi = k, k + 1
        t_a2, t_b2 = float(t[k2_lo]), float(t[k2_hi])
        dt_ab2 = t_b2 - t_a2
        y_a2, y_b2 = y[k2_lo], y[k2_hi]
        m_a2, m_b2 = slopes[k2_lo], slopes[k2_hi]

        def dydt_right(
            s: float,
            dt: float = dt_ab2,
            y_a: np.ndarray = y_a2,
            y_b: np.ndarray = y_b2,
            m_a: np.ndarray = m_a2,
            m_b: np.ndarray = m_b2,
            c: int = component,
        ) -> float:
            return float(_hermite_slope(s, y_a, y_b, m_a, m_b, dt)[c])

        d_lo_right = dydt_right(0.0)
        d_hi_right = dydt_right(1.0)

        # Pick the bracket whose endpoints straddle zero.
        if d_lo_left * d_hi_left <= 0.0:
            s_star = (
                0.0
                if d_lo_left == 0.0
                else 1.0
                if d_hi_left == 0.0
                else brentq(dydt_left, 0.0, 1.0, rtol=rtol, maxiter=100)
            )
            t_star = t_a + s_star * dt_ab
            y_star = _hermite_state(s_star, y_a, y_b, m_a, m_b, dt_ab)
        elif d_lo_right * d_hi_right <= 0.0:
            s_star = (
                0.0
                if d_lo_right == 0.0
                else 1.0
                if d_hi_right == 0.0
                else brentq(dydt_right, 0.0, 1.0, rtol=rtol, maxiter=100)
            )
            t_star = t_a2 + s_star * dt_ab2
            y_star = _hermite_state(s_star, y_a2, y_b2, m_a2, m_b2, dt_ab2)
        else:
            # Neither bracket straddles zero (e.g. flat peak); keep the
            # discrete sample.
            t_star = float(t[k])
            y_star = y[k]

        t_peaks.append(float(t_star))
        y_peaks.append(float(y_star[component]))

    return np.asarray(t_peaks, dtype=float), np.asarray(y_peaks, dtype=float)


def _hermite_state(s, y_a, y_b, m_a, m_b, dt):
    """Local cubic-Hermite interpolant (re-implemented to avoid circular import)."""
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * y_a + h10 * dt * m_a + h01 * y_b + h11 * dt * m_b


def _hermite_slope(s, y_a, y_b, m_a, m_b, dt):
    """Time-derivative of the cubic-Hermite interpolant."""
    h00p = 6.0 * s * s - 6.0 * s
    h10p = 3.0 * s * s - 4.0 * s + 1.0
    h01p = -6.0 * s * s + 6.0 * s
    h11p = 3.0 * s * s - 2.0 * s
    return (h00p * y_a + h01p * y_b) / dt + h10p * m_a + h11p * m_b


@trajectory_op(returns="trajectory")
def local_maxima(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    *,
    refined: bool = False,
    **find_peaks_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate local maxima of one state component.

    By default uses :func:`scipy.signal.find_peaks` (sample-accurate);
    pass ``refined=True`` for sub-sample-accurate times and heights via
    cubic-Hermite refinement (same algorithm that backs
    :func:`detect_events`).

    Parameters
    ----------
    component : int, default 0
    refined : bool, default ``False``
        Refine peak times/heights below sample resolution.
    **find_peaks_kwargs
        Forwarded to :func:`scipy.signal.find_peaks`
        (``prominence``, ``distance``, ``height``, …).

    Returns
    -------
    Trajectory
        ``t`` = peak times; ``y`` has shape ``(K, 1)`` — peak heights of
        the chosen component.

    Examples
    --------
    >>> peaks = traj.local_maxima(component=2, prominence=1.0)
    >>> peaks.t.shape, peaks.y.shape
    ((K,), (K, 1))
    """
    if refined:
        tp, yp = _refine_extrema(t, y, component, kind="max", find_peaks_kwargs=find_peaks_kwargs)
    else:
        sig = _extract_component(y, component)
        idx, _ = find_peaks(sig, **find_peaks_kwargs)
        tp = t[idx].copy()
        yp = sig[idx].copy()
    return tp, yp.reshape(-1, 1)


@trajectory_op(returns="trajectory")
def local_minima(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    *,
    refined: bool = False,
    **find_peaks_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate local minima of one state component.

    See :func:`local_maxima` for parameter meanings; this is the same
    primitive applied to ``-y[:, component]``.

    Returns
    -------
    Trajectory
        ``t`` = trough times; ``y`` has shape ``(K, 1)`` — trough heights.
    """
    if refined:
        tp, yp = _refine_extrema(t, y, component, kind="min", find_peaks_kwargs=find_peaks_kwargs)
    else:
        sig = _extract_component(y, component)
        idx, _ = find_peaks(-sig, **find_peaks_kwargs)
        tp = t[idx].copy()
        yp = sig[idx].copy()
    return tp, yp.reshape(-1, 1)


@trajectory_op(returns="trajectory")
def return_times(
    t: np.ndarray,
    y: np.ndarray,
    component: int = 0,
    **find_peaks_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inter-peak intervals on one state component — the "poor man's period".

    For a clean ``sin(ω t)`` every entry is close to ``2π / ω``.  The
    proper spectral version lands in M7.

    Parameters
    ----------
    component : int, default 0
    **find_peaks_kwargs
        Forwarded to :func:`scipy.signal.find_peaks`.

    Returns
    -------
    Trajectory
        ``t`` = times of the *first* peak in each pair (length ``K - 1``);
        ``y`` has shape ``(K - 1, 1)`` — the ISI values.  Empty trajectory
        if fewer than two peaks were found.
    """
    sig = _extract_component(y, component)
    idx, _ = find_peaks(sig, **find_peaks_kwargs)
    if idx.size < 2:
        return np.empty(0, dtype=float), np.empty((0, 1), dtype=float)
    tp = t[idx]
    isi = np.diff(tp)
    return tp[:-1].copy(), isi.reshape(-1, 1)


# ---------------------------------------------------------------------------
# to_dataspec — V1 DataSpec placeholder
# ---------------------------------------------------------------------------

# Mapping from ``kind`` to required positional dims (None = no required cardinality).
_DATASPEC_REQUIRED_DIMS: dict[str, int | None] = {
    "timeseries": None,
    "phase_portrait_2d": 2,
    "phase_portrait_3d": 3,
    "scatter": None,
    "return_map": None,
    "events": None,
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

    Recognised kinds
    ----------------

    ``"timeseries"``
        ``{kind, t, y, dims}``.  ``dims`` defaults to ``tuple(range(dim))``.

    ``"phase_portrait_2d"`` / ``"phase_portrait_3d"``
        ``{kind, t, y, dims}``.  ``dims=`` is required and must have the
        right cardinality.

    ``"scatter"``
        ``{kind, t, y, dims}``.  Like ``"timeseries"`` but signals to the
        plotter to render as a scatter rather than a line.  Useful for
        Poincaré-section data.

    ``"return_map"``
        ``{kind, x, y, t, step, observable, component}``.  Builds the
        ``(x_k, x_{k+step})`` pair view from the trajectory's observable
        column.

        kwargs:
          - ``step`` (int, default 1) — return step.
          - ``observable`` (int, default 0) — which column of ``y`` to pair.

    ``"events"``
        ``{kind, t, y, dims}``.  Same shape as ``timeseries`` but
        signals that ``t`` are isolated event instants (renderer should
        use markers).
    """
    if kind not in _DATASPEC_REQUIRED_DIMS:
        raise ValueError(
            f"to_dataspec: unknown kind {kind!r} (allowed: {sorted(_DATASPEC_REQUIRED_DIMS)})"
        )

    if kind == "return_map":
        return _build_return_map_spec(t, y, **kwargs)

    dims = kwargs.pop("dims", None)
    if dims is None:
        if kind in ("phase_portrait_2d", "phase_portrait_3d"):
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


def _build_return_map_spec(
    t: np.ndarray,
    y: np.ndarray,
    *,
    step: int = 1,
    observable: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    """Build the {x, y, t, step, observable} spec for kind='return_map'."""
    if not isinstance(step, int | np.integer) or step < 1:
        raise ValueError(
            f"to_dataspec(kind='return_map'): 'step' must be a positive integer, got {step!r}"
        )
    if y.ndim != 2:
        raise ValueError(
            f"to_dataspec(kind='return_map'): y must be 2-D (T, dim), got shape {y.shape}"
        )
    dim = y.shape[1]
    c = int(observable)
    if not (-dim <= c < dim):
        raise IndexError(
            f"to_dataspec(kind='return_map'): observable component {c} out of range for dim {dim}"
        )
    obs = y[:, c]
    if obs.size <= step:
        empty = np.empty(0, dtype=float)
        return {
            "kind": "return_map",
            "x": empty,
            "y": empty,
            "t": empty,
            "step": int(step),
            "observable": c,
            **extra,
        }
    return {
        "kind": "return_map",
        "x": obs[:-step].astype(float, copy=True),
        "y": obs[step:].astype(float, copy=True),
        "t": t[:-step].astype(float, copy=True),
        "step": int(step),
        "observable": c,
        **extra,
    }


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
