"""Poincaré sections from systems (exact crossings) or trajectories (interpolation)."""

from __future__ import annotations

from typing import Any

import numpy as np

from tsdynamics.derived import PoincareMap
from tsdynamics.families import Trajectory

__all__ = ["poincare_section"]


def _seeded_ic(system: Any, ic: Any | None, seed: int | None) -> np.ndarray | None:
    """Return a reproducible random initial condition, or ``None`` for the default.

    Returns a seeded ``U[0, 1)^dim`` draw only when the run would *otherwise*
    fall back to a random IC (no explicit ``ic``, no ``system.ic``, no class
    ``default_ic``); in every other case the existing resolution wins and this
    returns ``None`` (so ``seed`` never overrides a deliberate initial state).
    """
    if ic is not None or seed is None:
        return None
    if getattr(system, "ic", None) is not None:
        return None
    if getattr(type(system), "default_ic", None) is not None:
        return None
    return np.random.default_rng(seed).random(system.dim)


def poincare_section(
    system: Any,
    plane: tuple,
    *,
    direction: int = +1,
    n: int = 1000,
    skip_crossings: int = 0,
    dt: float = 0.01,
    max_time: float = 1e4,
    seed: int | None = None,
) -> Trajectory:
    """
    Poincaré surface of section.

    Two input modes:

    - **System** → wraps it in a :class:`~tsdynamics.derived.PoincareMap`
      and collects ``n`` root-refined crossings (accurate).
    - **Trajectory** → finds the plane crossings between consecutive samples
      by linear interpolation (pure data path; accuracy limited by the
      trajectory's sampling interval).

    Parameters
    ----------
    system : System or Trajectory
        A flow to section, or measured trajectory data (the ``data`` overload).
    plane : tuple
        ``(i, c)`` for the section ``y_i = c``, or ``(normal, offset)``.
    direction : {+1, -1, 0}
        Crossing direction filter.
    n : int, default 1000
        Number of crossings to collect (system mode).
    skip_crossings : int, default 0
        Number of leading crossings to discard before recording (system mode).
    dt, max_time : float
        Detection step and integration ceiling (system mode) — see
        :class:`~tsdynamics.derived.PoincareMap`.
    seed : int, optional
        Seed for the random initial condition when the system has none
        (system mode); makes the section reproducible.

    Returns
    -------
    Trajectory
        ``t`` = crossing times, ``y`` = full-dimensional crossing states.

    Examples
    --------
    >>> section = poincare_section(Rossler(), plane=(0, 0.0), n=500)
    >>> section = poincare_section(traj, plane=(2, 25.0))     # from data
    """
    if isinstance(system, Trajectory):
        return _section_from_data(system, plane, direction)
    seeded = _seeded_ic(system, None, seed)
    if seeded is not None:
        system = system.copy()
        system.reinit(seeded)
    pmap = PoincareMap(system, plane, direction=direction, dt=dt, max_time=max_time)
    return pmap.trajectory(n, transient=skip_crossings)


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
