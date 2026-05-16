"""
Poincaré section construction on top of :mod:`tsdynamics.analysis.events`.

A Poincaré section is just an :class:`EventCondition` evaluated along a
trajectory, with the resulting refined crossings packaged as a new
``Trajectory``.  The full state is kept at each crossing; users who want
the ``(dim - 1)``-dim view can ``.project()`` the result.

The canonical Poincaré convention is *upward crossings only*, so the
``direction="up"`` default below overrides any :class:`Plane` /
:class:`LinearPlane` ``direction`` attribute unless the caller passes
``direction=None``.
"""

from __future__ import annotations

import numpy as np

from ._registry import trajectory_op
from .events import EventCondition, LinearPlane, Plane


@trajectory_op(returns="trajectory")
def poincare_section(
    t: np.ndarray,
    y: np.ndarray,
    plane: EventCondition,
    *,
    direction: str | None = "up",
    rtol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Poincaré section of a trajectory through ``plane``.

    Available equivalently as a free function and as a
    :class:`Trajectory` method::

        poincare_section(traj, plane)
        poincare_section((t, y), plane)
        poincare_section(t, y, plane)
        traj.poincare_section(plane)

    Parameters
    ----------
    plane : EventCondition
        Usually a :class:`Plane` or :class:`LinearPlane`.  Its
        ``direction`` attribute is overridden by the keyword below
        (default ``"up"`` for canonical Poincaré).
    direction : {"up", "down", "either", None}, default ``"up"``
        Pass ``None`` to honour the ``plane`` object's own ``direction``.
    rtol : float, default ``1e-8``

    Returns
    -------
    Trajectory (method form / Trajectory-input free-function form) or
    ``(t, y)`` tuple (bare-array / tuple-input free-function form).  The
    section axis is *kept*; project it away with ``.project(...)`` if
    you want a ``(dim - 1)``-dim view.

    Examples
    --------
    >>> from tsdynamics.analysis import Plane
    >>> sec = traj.poincare_section(Plane(axis=2, value=27.0))
    """
    cond = _override_direction(plane, direction)
    result = cond.detect(t, y, rtol=rtol)
    return result.t, result.y


def _override_direction(plane: EventCondition, direction: str | None) -> EventCondition:
    """
    Return a (possibly new) condition with ``direction`` overridden.

    For :class:`Plane` / :class:`LinearPlane` we build a fresh dataclass
    instance so the caller's plane object is not mutated.  Any other
    condition is passed through unchanged: the user-supplied condition's
    own ``direction`` semantics are respected, since rewriting it could
    silently change behaviour for custom subclasses.
    """
    if direction is None:
        return plane
    if isinstance(plane, Plane):
        return Plane(axis=plane.axis, value=plane.value, direction=direction)
    if isinstance(plane, LinearPlane):
        return LinearPlane(
            normal=np.asarray(plane.normal, dtype=float).copy(),
            offset=plane.offset,
            direction=direction,
        )
    return plane


__all__ = ["poincare_section"]
