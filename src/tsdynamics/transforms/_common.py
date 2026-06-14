"""
Shared plumbing for the transform layer.

Every transform consumes either a :class:`~tsdynamics.data.Trajectory` or a raw
array, with time running along axis 0 (the convention a ``Trajectory.y`` of
shape ``(T, dim)`` already obeys).  The helpers here centralise the two pieces of
boilerplate that would otherwise be copy-pasted across every transform:

- :func:`to_signal` — coerce the polymorphic input to a contiguous ``float``
  array, leaving 1-D signals 1-D and multi-channel signals ``(T, channels)``.
- :func:`resolve_fs` — settle the sampling frequency.  Spectral estimators need
  it; a :class:`~tsdynamics.data.Trajectory` already carries the information in
  its ``t`` vector, so it is inferred from there when the caller does not pass an
  explicit ``fs`` / ``dt``.

Shape-preserving transforms (detrend, normalize, filters) additionally use
:func:`wrap_like` so that ``Trajectory`` in yields ``Trajectory`` out — the new
trajectory keeps the original time base and provenance and records the transform
it went through, mirroring :meth:`tsdynamics.data.Trajectory.standardize`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Relative tolerance within which a time vector counts as uniformly sampled.
_UNIFORM_RTOL = 1e-3


def _is_trajectory(x: Any) -> bool:
    """Duck-typed ``Trajectory`` check that avoids importing the data layer eagerly."""
    # A local import keeps ``transforms`` importable even mid-initialisation and
    # avoids a hard module-level dependency cycle; the class is tiny to import.
    from tsdynamics.data import Trajectory

    return isinstance(x, Trajectory)


def to_signal(x: Any) -> np.ndarray:
    """
    Coerce a transform input to a float array with time along axis 0.

    Parameters
    ----------
    x : Trajectory or array-like
        A :class:`~tsdynamics.data.Trajectory` (its ``y`` of shape ``(T, dim)``
        is used) or any array-like.  1-D inputs stay 1-D; 2-D inputs are read as
        ``(n_samples, n_channels)``.

    Returns
    -------
    ndarray
        ``float`` array, contiguous, shape unchanged from the underlying data.
    """
    if _is_trajectory(x):
        return np.ascontiguousarray(x.y, dtype=float)
    # Rank-check the raw array first: ``ascontiguousarray`` would promote a 0-D
    # scalar to 1-D and hide it, so the scalar guard has to come before it.
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        raise ValueError("a transform needs a 1-D or 2-D signal, got a scalar.")
    if arr.ndim > 2:
        raise ValueError(f"signals must be 1-D or 2-D (time, channels); got ndim={arr.ndim}.")
    return np.ascontiguousarray(arr)


def _uniform_dt(t: np.ndarray) -> float:
    """
    Sample spacing of a (near-)uniform time vector.

    Raises
    ------
    ValueError
        If the time vector is too short, non-increasing, or not uniform within
        :data:`_UNIFORM_RTOL` — spectral/filter methods assume even sampling, so
        a silent wrong answer is worse than a clear error.
    """
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        raise ValueError("cannot infer a sampling rate from fewer than two time points.")
    diffs = np.diff(t)
    dt = float(np.mean(diffs))
    if dt <= 0.0:
        raise ValueError("time vector must be strictly increasing to infer a sampling rate.")
    spread = float(np.max(np.abs(diffs - dt)))
    if spread > _UNIFORM_RTOL * dt:
        raise ValueError(
            "time vector is not uniformly sampled (spread "
            f"{spread:.3g} > {_UNIFORM_RTOL:.0e}·dt); pass an explicit fs= or dt=, "
            "or resample first."
        )
    return dt


def resolve_fs(x: Any, *, fs: float | None = None, dt: float | None = None) -> float:
    """
    Settle the sampling frequency for a spectral / filtering transform.

    Resolution order: an explicit ``fs`` wins; else an explicit ``dt`` (``fs =
    1/dt``); else, for a :class:`~tsdynamics.data.Trajectory`, the rate inferred
    from its ``t`` vector; else ``1.0`` (cycles per sample).

    Parameters
    ----------
    x : Trajectory or array-like
        The signal whose sampling rate is being resolved (only its time base is
        consulted, and only when it is a ``Trajectory``).
    fs : float, optional
        Sampling frequency in samples per unit time.  Mutually exclusive with
        ``dt``.
    dt : float, optional
        Sample spacing.  Mutually exclusive with ``fs``.

    Returns
    -------
    float
        The sampling frequency.
    """
    if fs is not None and dt is not None:
        raise ValueError("pass at most one of fs= / dt=.")
    if fs is not None:
        if not np.isfinite(fs) or fs <= 0.0:
            raise ValueError(f"fs must be a positive finite number, got {fs!r}.")
        return float(fs)
    if dt is not None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be a positive finite number, got {dt!r}.")
        return 1.0 / float(dt)
    if _is_trajectory(x):
        return 1.0 / _uniform_dt(x.t)
    return 1.0


def wrap_like(x: Any, y: np.ndarray, **meta_update: Any) -> Any:
    """
    Return a shape-preserving transform's output in the caller's own type.

    Given the original input ``x`` and the transformed values ``y`` (same shape
    as ``to_signal(x)``), return a fresh :class:`~tsdynamics.data.Trajectory`
    when ``x`` was one — keeping its time base, back-reference, and provenance,
    plus any ``meta_update`` describing the transform — or the bare array
    otherwise.

    Parameters
    ----------
    x : Trajectory or array-like
        The original input.
    y : ndarray
        The transformed values.
    **meta_update
        Extra provenance keys merged into the new trajectory's ``meta`` (ignored
        when ``x`` is a plain array).

    Returns
    -------
    Trajectory or ndarray
    """
    if _is_trajectory(x):
        from tsdynamics.data import Trajectory

        meta = dict(x.meta)
        meta.update(meta_update)
        return Trajectory(x.t, np.asarray(y), x.system, meta=meta)
    return np.asarray(y)


def channel_iter(sig: np.ndarray):
    """
    Iterate ``(index, column)`` over the channels of a 1-D or 2-D signal.

    A 1-D signal yields a single ``(0, signal)`` pair; a 2-D ``(T, C)`` signal
    yields one pair per column.  Lets per-channel reductions (features, spectral
    entropy) share one code path regardless of input rank.
    """
    if sig.ndim == 1:
        yield 0, sig
    else:
        for j in range(sig.shape[1]):
            yield j, sig[:, j]
