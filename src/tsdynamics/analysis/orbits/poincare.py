"""Poincaré sections from systems (exact crossings) or trajectories (interpolation)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from tsdynamics.derived import PoincareMap
from tsdynamics.derived.poincare import PoincareSection, _resolve_section_plane
from tsdynamics.errors import remedy
from tsdynamics.families import Trajectory

__all__ = ["PoincareSection", "poincare_section"]


def _seeded_ic(system: Any, ic: Any | None, seed: int | None) -> np.ndarray | None:
    """Return a reproducible random initial condition, or ``None`` for the default.

    Returns a seeded ``U[0, 1)^dim`` draw only when the run would *otherwise*
    fall back to a random IC (no explicit ``ic``, no ``system.ic``, no class
    ``_default_ic``); in every other case the existing resolution wins and this
    returns ``None`` (so ``seed`` never overrides a deliberate initial state).

    The ClassVar is read by its v6 name, ``_default_ic``.  Reading the pre-v6
    ``default_ic`` here returned ``None`` for **every** system — a legal value,
    so nothing raised — and a seeded section of a system that declares a default
    initial state silently started somewhere else instead.
    """
    if ic is not None or seed is None:
        return None
    if getattr(system, "ic", None) is not None:
        return None
    # Instance first, then the class: a variable-dimension system sizes its own
    # default in ``__init__`` (``MultiChua``), so the class attribute is ``None``
    # there while the instance has a real one.
    if getattr(system, "_default_ic", None) is not None:
        return None
    return cast(np.ndarray, np.random.default_rng(seed).random(system.dim))


def poincare_section(
    system: Any,
    plane: tuple[Any, ...] | None = None,
    *,
    direction: int | str = +1,
    crossings: int = 1000,
    skip_crossings: int = 0,
    ic: Any | None = None,
    dt: float = 0.01,
    max_time: float = 1e4,
    seed: int | None = 0,
) -> PoincareSection:
    """
    Poincaré surface of section.

    Two input modes:

    - **System** → wraps it in a :class:`~tsdynamics.derived.PoincareMap`
      and collects ``crossings`` root-refined crossings on the fast Rust event engine
      (stream WS-CROSSKERNEL).
    - **Trajectory** → finds the plane crossings between consecutive samples
      by linear interpolation (pure data path; accuracy limited by the
      trajectory's sampling interval).

    Parameters
    ----------
    system : System or Trajectory
        A flow to section, or measured trajectory data (the ``data`` overload).
    plane : tuple, optional
        The section, in any of three spellings:

        - ``(axis, c)`` — ``axis`` a component **name** (resolved against the
          system's ``variables``, e.g. ``"y"``) or an integer index, for the
          section ``y_axis = c``;
        - ``(axis, c, direction)`` — the same, with the crossing direction
          (``"up"`` / ``"down"`` / ``"both"``) as a third element, which
          overrides the ``direction`` argument;
        - ``(normal, offset)`` — an arbitrary normal **vector**, for the section
          ``normal · y = offset``.

        For example ``plane=("y", 0.0, "up")``, ``plane=(1, 0.0)``, or
        ``plane=([1, 0, 0], 0.0)``.

        **Omit it and one is chosen**, from a short probe orbit: the
        widest-spread component (largest interquartile range) *among those the
        orbit repeatedly returns to*, sectioned at its median — an offset the
        probe orbit provably straddles (so the auto section cannot miss *it*; on
        a system with no ``default_ic`` the section run starts from a different
        random draw, which only matters when the two orbits need not share an
        attractor — see :func:`~tsdynamics.derived.poincare.auto_plane`), and one
        it comes back to, so the map has crossings to return to.  A
        monotone coordinate (a carried drive phase) has the widest spread of all
        and is not a section; it is skipped for that reason.  See
        :func:`tsdynamics.derived.poincare.auto_plane` for the rule and why it is
        that one.  The choice is recorded in ``result.meta["plane"]`` with
        ``meta["plane_auto"] = True`` and printed by the section's repr; name a
        plane whenever you have one.  In **data** mode (a ``Trajectory`` in) the
        same rule applies, read off the data itself.
    direction : {+1, -1, 0} or {"up", "down", "both"}, default +1
        Crossing direction filter (``+1`` / ``"up"`` keeps only crossings where
        the section function is increasing).  Ignored when ``plane`` carries its
        own direction (third element).
    crossings : int, default 1000
        Number of crossings to collect (system mode) — the number of points the
        returned section holds.

        .. versionchanged:: 6.0
            Named ``n`` before v6.  ``n`` meant four different things across the
            public surface; here it counts crossings, so it says so — and it now
            reads in the same unit as its neighbour ``skip_crossings``.
    skip_crossings : int, default 0
        Number of leading crossings to discard before recording.  (A *section*
        transient is a count of crossings, deliberately distinct from the
        time/step ``transient`` of other analyses.)
    ic : array-like, optional
        Where to start the section run — ``dim`` numbers, one per state
        component (system mode).  Falls back to ``system.ic``, the declared
        default, then the seeded random draw.

        .. versionadded:: 6.0
            Every sibling that runs a system took ``ic=`` — ``orbit_diagram``,
            ``lyapunov_spectrum``, ``zero_one_test`` — and this
            one refused it with a bare ``TypeError``, leaving ``seed=`` as the
            only control over where the orbit starts.  On a system with a finite
            basin that is not a control at all: 3 of the first 4 seeds can land
            off-attractor.

    dt : float, default 0.01
        Crossing-**detection** step, in **time units** (system mode).  The march
        is fixed-step ``rk4`` at this step, so ``dt`` bounds how finely a crossing
        is bracketed before the Hermite refinement — it is not an output grid and
        not a tolerance.  Ignored for a ``Trajectory``, whose sampling interval is
        already fixed.
    max_time : float, default 1e4
        Ceiling on the integration horizon, in **time units** (system mode):
        how long to march before giving up on collecting ``crossings`` crossings.
    seed : int, default 0
        Seed for the random initial condition when the system has none
        (system mode), so the section is reproducible.  Pass ``seed=None`` for an
        explicitly unseeded draw.

        .. versionchanged:: 6.0
            Was ``None``: the same call drew a different starting point, and so a
            different set of crossings, every time it ran.

    Returns
    -------
    PoincareSection
        A :class:`~tsdynamics.data.Trajectory` of the crossings (``t`` = crossing
        times, ``y`` = full-dimensional crossing states) carrying
        ``POINCARE_SECTION`` plot intent and a repr / ``.to_dict()`` /
        ``.plot`` result surface.

    Examples
    --------
    >>> section = poincare_section(Rossler(), plane=("y", 0.0, "up"), crossings=500)
    >>> section = poincare_section(Rossler(), crossings=500)     # section chosen + recorded
    >>> section = poincare_section(traj, plane=("z", 25.0))     # from data
    """
    _reject_axes_pair(system, plane)
    if isinstance(system, Trajectory):
        if ic is not None:
            from tsdynamics.errors import InvalidParameterError

            raise InvalidParameterError(
                "ic= chooses where to START a run, and a Trajectory has already been "
                "run — its section is cut from the samples it holds."
                + remedy(
                    "ts.analysis.poincare_section(system, plane, ic=[1.0, 1.0, 1.0])",
                    "ts.analysis.poincare_section(traj[100:], plane)   # ...or cut later data",
                )
            )
        return _section_from_data(system, plane, direction)
    start = _seeded_ic(system, ic, seed) if ic is None else np.asarray(ic, dtype=float)
    if start is not None:
        system = system.copy()
        system.reinit(start)
    pmap = PoincareMap(system, plane, direction=direction, dt=dt, max_time=max_time)
    return pmap.run(crossings, transient=skip_crossings)


def _reject_axes_pair(subject: Any, plane: tuple[Any, ...] | None) -> None:
    """Refuse a *view-axes* pair where a cutting SECTION is wanted.

    ``plane=`` is two different words in this library — a section ``(axis,
    offset)`` here, a pair of view axes at the eight field analyses in
    :mod:`tsdynamics.analysis.planar` — and each door used to read the other's
    spelling without complaint.  Two entries that both NAME components can only
    be the axes spelling (a section's second entry is a number), so that half is
    detectable and is named rather than reinterpreted.
    """
    if plane is None or len(tuple(plane)) != 2:
        return
    first, second = tuple(plane)
    if not (isinstance(first, str) and isinstance(second, str)):
        return
    names = getattr(subject, "variables", None) or ()
    if first in names and second in names:
        from tsdynamics.errors import InvalidParameterError

        raise InvalidParameterError(
            f"plane={(first, second)!r} names two COORDINATES, which is the view-axes "
            f"spelling the field analyses take; a Poincaré section is a cutting surface, "
            f"so it is an axis and the VALUE it is held at.\n"
            f'    ts.analysis.poincare_section(system, ("{first}", 0.0))'
            f"   # the section {first} = 0\n"
            f"    ts.analysis.flow_field(system, plane={(first, second)!r})"
            f"   # the {first}-{second} plane"
        )


def _section_from_data(
    traj: Trajectory, plane: tuple[Any, ...] | None, direction: int | str
) -> PoincareSection:
    resolved_plane, direction = _resolve_section_plane(traj, plane, direction)
    normal, offset = PoincareMap._parse_plane(traj.dim, resolved_plane)
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

    # Section intent (viz.PlotKind.POINCARE_SECTION) so a renderer draws the
    # in-plane scatter, never the source flow.  ``plane`` is the resolved
    # (index, offset) form so Trajectory._section_axes can drop the normal axis.
    meta = {
        **traj.meta,
        "derived": "poincare_section",
        "plot_kind": "poincare_section",
        "plane": resolved_plane,
        # An auto-chosen section is recorded, never silent (same contract as the
        # system path and as orbit_diagram's auto discrete view).
        "plane_auto": plane is None,
        "direction": direction,
    }
    if i_hits.size == 0:
        return PoincareSection(
            t=np.empty(0),
            y=np.empty((0, traj.dim)),
            system=traj.system,
            meta=meta,
        )

    s = g[i_hits] / (g[i_hits] - g[i_hits + 1])  # linear interpolation fraction
    points = traj.y[i_hits] + s[:, None] * (traj.y[i_hits + 1] - traj.y[i_hits])
    times = traj.t[i_hits] + s * (traj.t[i_hits + 1] - traj.t[i_hits])

    return PoincareSection(t=times, y=points, system=traj.system, meta=meta)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
