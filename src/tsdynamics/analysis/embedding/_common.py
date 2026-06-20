r"""
Shared input coercion for the delay-embedding toolkit.

The estimators in this subpackage operate on a single scalar time series
(:func:`_as_series`) or, for multivariate reconstruction, on a bundle of
synchronous channels (:func:`_as_channels`).  Both accept a
:class:`~tsdynamics.data.Trajectory`, a raw array, or a plain sequence, so the
public functions read a trajectory and a NumPy array the same way.

The :class:`~tsdynamics.data.Trajectory` is duck-typed (``.y`` / ``.component``)
to avoid an import cycle through :mod:`tsdynamics.families` / :mod:`tsdynamics.data`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__: list[str] = []


def _is_trajectory(x: Any) -> bool:
    """Return whether ``x`` quacks like a :class:`~tsdynamics.data.Trajectory`."""
    return hasattr(x, "y") and hasattr(x, "component") and not isinstance(x, np.ndarray)


def _as_series(x: Any, component: int | str | None = None) -> np.ndarray:
    """Coerce ``x`` into a contiguous 1-D ``float64`` array (one scalar series).

    Parameters
    ----------
    x : array-like or Trajectory
        A 1-D sequence (used directly), a 2-D array (a column is selected by
        ``component``), or a :class:`~tsdynamics.data.Trajectory` (a component
        is selected by index, or by name when the system declares ``variables``).
    component : int or str, optional
        Which column/component to extract from a multi-component input.  Required
        when the input has more than one component; for a single-component input
        it must be left ``None`` (or ``0``).

    Returns
    -------
    ndarray, shape (N,)
        The scalar series.

    Raises
    ------
    ValueError
        If ``component`` is ambiguous/meaningless, the series is not 1-D after
        selection, has fewer than two samples, or contains non-finite values.
    """
    if _is_trajectory(x):
        if component is None:
            if x.y.ndim == 1 or x.y.shape[1] == 1:
                arr = np.asarray(x.y, dtype=float).ravel()
            else:
                raise ValueError(
                    "trajectory has multiple components; pass component= to select one "
                    f"(e.g. component=0 or a name from {getattr(x, 'variables', None)})."
                )
        else:
            arr = np.asarray(x.component(component), dtype=float).ravel()
    else:
        a = np.asarray(x, dtype=float)
        if a.ndim == 1:
            if component not in (None, 0):
                raise ValueError("component= is meaningless for a 1-D series.")
            arr = a
        elif a.ndim == 2:
            if a.shape[1] == 1 and component in (None, 0):
                arr = a[:, 0]
            elif component is None:
                raise ValueError(f"input has {a.shape[1]} columns; pass component= to select one.")
            else:
                arr = a[:, int(component)]
        else:
            raise ValueError(f"expected a 1-D or 2-D input, got array of shape {a.shape}.")

    arr = np.ascontiguousarray(arr, dtype=float)
    if arr.size < 2:
        raise ValueError(f"need at least two samples, got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("series contains non-finite values (nan/inf).")
    return arr


def _as_channels(x: Any) -> np.ndarray:
    """Coerce ``x`` into a contiguous ``(N, d)`` array of synchronous channels.

    Used by multivariate embedding.  Accepts a
    :class:`~tsdynamics.data.Trajectory` (its full state array), a 2-D
    ``(N, d)`` array, a 1-D ``(N,)`` series (treated as a single channel,
    ``(N, 1)``), or a sequence of equal-length 1-D series.

    Parameters
    ----------
    x : array-like or Trajectory

    Returns
    -------
    ndarray, shape (N, d)

    Raises
    ------
    ValueError
        If the channels have unequal length, the result is not 2-D, there are
        fewer than two samples, or any value is non-finite.
    """
    if _is_trajectory(x):
        arr = np.asarray(x.y, dtype=float)
    else:
        # A list/tuple of 1-D series -> stack as columns; reject ragged input.
        if isinstance(x, (list, tuple)) and x and all(np.ndim(c) == 1 for c in x):
            lengths = {len(c) for c in x}
            if len(lengths) != 1:
                raise ValueError(f"channels have unequal lengths {sorted(lengths)}.")
            arr = np.column_stack([np.asarray(c, dtype=float) for c in x])
        else:
            arr = np.asarray(x, dtype=float)

    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(
            f"expected a (N, d) channel bundle or a 1-D series, got shape {arr.shape}."
        )
    arr = np.ascontiguousarray(arr, dtype=float)
    if arr.shape[0] < 2:
        raise ValueError(f"need at least two samples, got {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("channel data contains non-finite values (nan/inf).")
    return arr


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
