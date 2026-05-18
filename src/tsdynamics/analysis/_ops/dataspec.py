"""DataSpec construction — V1 DataSpec placeholder."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._registry import trajectory_op

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


__all__ = ["to_dataspec"]
