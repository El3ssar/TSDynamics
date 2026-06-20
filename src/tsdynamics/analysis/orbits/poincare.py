"""Poincaré sections from systems (exact crossings) or trajectories (interpolation)."""

from __future__ import annotations

from typing import Any

import numpy as np

from tsdynamics.derived import PoincareMap
from tsdynamics.families import Trajectory

__all__ = ["poincare_section"]


def poincare_section(
    sys_or_traj: Any,
    plane: tuple,
    *,
    direction: int = +1,
    steps: int = 1000,
    transient: int = 0,
    dt: float = 0.01,
    max_time: float = 1e4,
) -> Trajectory:
    """
    Poincaré surface of section.

    Two input modes:

    - **System** → wraps it in a :class:`~tsdynamics.derived.PoincareMap`
      and collects ``steps`` root-refined crossings (accurate).
    - **Trajectory** → finds the plane crossings between consecutive samples
      by linear interpolation (pure data path; accuracy limited by the
      trajectory's sampling interval).

    Parameters
    ----------
    sys_or_traj : System or Trajectory
    plane : tuple
        ``(i, c)`` for the section ``y_i = c``, or ``(normal, offset)``.
    direction : {+1, -1, 0}
        Crossing direction filter.
    steps, transient, dt, max_time
        System mode only — see :class:`~tsdynamics.derived.PoincareMap`.

    Returns
    -------
    Trajectory
        ``t`` = crossing times, ``y`` = full-dimensional crossing states.

    Examples
    --------
    >>> section = poincare_section(Rossler(), plane=(0, 0.0), steps=500)
    >>> section = poincare_section(traj, plane=(2, 25.0))     # from data
    """
    if isinstance(sys_or_traj, Trajectory):
        return _section_from_data(sys_or_traj, plane, direction)
    pmap = PoincareMap(sys_or_traj, plane, direction=direction, dt=dt, max_time=max_time)
    return pmap.trajectory(steps, transient=transient)


def _section_from_data(traj: Trajectory, plane: tuple, direction: int) -> Trajectory:
    normal, offset = PoincareMap._parse_plane(traj.dim, plane)
    g = traj.y @ normal - offset
    g_prev, g_next = g[:-1], g[1:]

    up = (g_prev < 0.0) & (g_next >= 0.0)
    down = (g_prev > 0.0) & (g_next <= 0.0)
    if direction > 0:
        hits = up
    elif direction < 0:
        hits = down
    else:
        hits = up | down
    (i_hits,) = np.nonzero(hits)

    if i_hits.size == 0:
        return Trajectory(
            t=np.empty(0),
            y=np.empty((0, traj.dim)),
            system=traj.system,
            meta={
                **traj.meta,
                "derived": "poincare_section",
                # Section intent (viz.PlotKind.POINCARE_SECTION) so a renderer
                # draws the in-plane scatter, never the source flow.
                "plot_kind": "poincare_section",
                "plane": plane,
            },
        )

    s = g[i_hits] / (g[i_hits] - g[i_hits + 1])  # linear interpolation fraction
    points = traj.y[i_hits] + s[:, None] * (traj.y[i_hits + 1] - traj.y[i_hits])
    times = traj.t[i_hits] + s * (traj.t[i_hits + 1] - traj.t[i_hits])

    meta = {
        **traj.meta,
        "derived": "poincare_section",
        # Section intent (viz.PlotKind.POINCARE_SECTION) so a renderer draws the
        # in-plane scatter, never the source flow.
        "plot_kind": "poincare_section",
        "plane": plane,
        "direction": direction,
    }
    return Trajectory(t=times, y=points, system=traj.system, meta=meta)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
