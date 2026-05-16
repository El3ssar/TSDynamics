"""
First- and N-step return maps built from Poincaré crossings.

The "return map" of a section is the discrete dynamical system you get by
sampling a scalar observable at successive intersections of the trajectory
with the section.  For Lorenz it gives the classical tent-like map of
``z_{k+1}`` vs ``z_k`` at the ``z = 27`` section — the canonical visual
demonstration of low-dimensional chaos.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._registry import trajectory_op
from .events import EventCondition
from .sections import _override_direction

Observable = int | Callable[[float, np.ndarray], float]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ReturnMap:
    """
    Pairs ``(x_k, x_{k+step})`` of a scalar observable at section crossings.

    Attributes
    ----------
    x : ndarray, shape (K - step,)
        Observable values at crossing ``k``.
    y : ndarray, shape (K - step,)
        Observable values at crossing ``k + step``.
    t : ndarray, shape (K - step,)
        Times of the ``x`` crossings.
    step : int
        Return step.  ``step=1`` is the classical first return map.
    observable_meta : str
        Human-readable description of the observable.

    Examples
    --------
    >>> # constructed below by :func:`return_map`
    >>> rmap = ReturnMap(x=np.array([1.0, 2.0]), y=np.array([2.0, 1.0]),
    ...                  t=np.array([0.1, 0.2]), step=1, observable_meta="y[0]")
    >>> len(rmap)
    2
    """

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    step: int = 1
    observable_meta: str = ""
    condition: EventCondition | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return int(self.x.size)

    def __iter__(self):
        return iter((self.x, self.y))

    def __repr__(self) -> str:
        if len(self):
            return (
                f"ReturnMap(n={len(self)}, step={self.step}, observable={self.observable_meta!r})"
            )
        return f"ReturnMap(n=0, step={self.step})"

    def to_dataspec(self, kind: str = "return_map") -> dict[str, Any]:
        """Plain-dict representation mirroring M1's :func:`to_dataspec` shape.

        V2 (the Poincaré / return-map plotter milestone) will replace this
        with a real ``DataSpec`` class; the dict keys here match what that
        class is expected to expose so call sites do not change.
        """
        if kind != "return_map":
            raise ValueError(f"ReturnMap.to_dataspec: unknown kind {kind!r} (only 'return_map')")
        return {
            "kind": kind,
            "x": self.x,
            "y": self.y,
            "t": self.t,
            "step": self.step,
            "observable": self.observable_meta,
        }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@trajectory_op(returns="passthrough")
def return_map(
    t: np.ndarray,
    y: np.ndarray,
    plane: EventCondition,
    observable: Observable = 0,
    *,
    step: int = 1,
    direction: str | None = "up",
    rtol: float = 1e-8,
) -> ReturnMap:
    """
    Build a return map from a section ``plane`` and scalar ``observable``.

    Available equivalently as a free function and as a
    :class:`Trajectory` method::

        return_map(traj, plane, observable=0)
        return_map((t, y), plane, observable=0)
        return_map(t, y, plane, observable=0)
        traj.return_map(plane, observable=0)

    Parameters
    ----------
    plane : EventCondition
        The Poincaré section; typically a :class:`Plane` or
        :class:`LinearPlane`.
    observable : int or callable, default ``0``
        Scalar observable evaluated at each crossing.  If an integer, it
        names a state-vector component; if a callable, it must accept
        ``(t, y)`` and return a float.
    step : int, default ``1``
        Return step.  ``step=1`` gives the standard first-return map;
        ``step=N`` gives the N-th return.
    direction : {"up", "down", "either", None}, default ``"up"``
        Canonical Poincaré usage is upward crossings.  Pass ``None`` to
        honour ``plane``'s own ``direction``.
    rtol : float, default ``1e-8``

    Returns
    -------
    ReturnMap

    Examples
    --------
    >>> from tsdynamics.analysis import Plane
    >>> rmap = traj.return_map(Plane(axis=2, value=27.0), observable=0)
    """
    if not isinstance(step, int | np.integer) or step < 1:
        raise ValueError(f"return_map: 'step' must be a positive integer, got {step!r}")

    cond = _override_direction(plane, direction)
    events = cond.detect(t, y, rtol=rtol)

    if events.t.size <= step:
        return ReturnMap(
            x=np.empty(0, dtype=float),
            y=np.empty(0, dtype=float),
            t=np.empty(0, dtype=float),
            step=int(step),
            observable_meta=_observable_label(observable),
            condition=cond,
        )

    vals = _evaluate_observable(observable, events.t, events.y)
    xk = vals[:-step]
    xk_next = vals[step:]
    tk = events.t[:-step]
    return ReturnMap(
        x=np.asarray(xk, dtype=float),
        y=np.asarray(xk_next, dtype=float),
        t=np.asarray(tk, dtype=float),
        step=int(step),
        observable_meta=_observable_label(observable),
        condition=cond,
    )


def _evaluate_observable(observable: Observable, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a scalar observable at every crossing."""
    if isinstance(observable, int | np.integer):
        dim = y.shape[1]
        c = int(observable)
        if not (-dim <= c < dim):
            raise IndexError(f"return_map: observable component {c} out of range for dim {dim}")
        return y[:, c]
    if callable(observable):
        return np.array(
            [float(observable(float(tk), yk)) for tk, yk in zip(t, y, strict=True)],
            dtype=float,
        )
    raise TypeError(
        f"return_map: observable must be int or callable, got {type(observable).__name__}"
    )


def _observable_label(observable: Observable) -> str:
    if isinstance(observable, int | np.integer):
        return f"y[{int(observable)}]"
    name = getattr(observable, "__name__", None) or type(observable).__name__
    return f"{name}(t, y)"


__all__ = ["Observable", "ReturnMap", "return_map"]
