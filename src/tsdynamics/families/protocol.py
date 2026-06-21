"""
The ``System`` runtime protocol — the contract every analysis function consumes.

All three families (:class:`~tsdynamics.families.ContinuousSystem`,
:class:`~tsdynamics.families.DelaySystem`, :class:`~tsdynamics.families.DiscreteMap`)
and every derived-system wrapper (:mod:`tsdynamics.derived`) implement this
interface, so analysis code can be written once against ``System`` and applied
to anything that steps:

- ``step(n_or_dt)`` advances the system and **returns the new state** —
  the number of map iterations for discrete systems, the time increment for
  continuous ones (each has a sensible default).
- ``state()`` / ``set_state(u)`` read/write the current state.  DDEs raise
  ``NotImplementedError`` from ``set_state`` — their state is a whole history
  function, not a point; use ``reinit(u)`` to restart from a constant past.
- ``time()`` is the current time (or iteration count for maps).
- ``reinit(u, t=..., params=...)`` restarts the internal stepper.
- ``run(...)`` is **the** canonical verb that produces a
  :class:`~tsdynamics.families.Trajectory`, dispatching on ``is_discrete``:
  ``final_time``/``dt`` for a continuous-time flow, ``n`` for a map.  A flow and
  a map answer the same call (``Lorenz().run(final_time=100, dt=0.01)`` and
  ``Henon().run(n=5000)``).  The family-specific spellings ``integrate`` (flows
  / DDEs / SDEs) and ``iterate`` (maps), and the protocol method
  ``trajectory(...)``, remain as permanent thin aliases.
- ``trajectory(...)`` produces a :class:`~tsdynamics.families.Trajectory` on a
  uniform grid (delegates to ``integrate`` / ``iterate``); it is the *structural*
  member the protocol requires (``run`` is the canonical verb, but every family
  and wrapper implements ``trajectory`` — see the ``trajectory`` method note).

Stepping state is lazily initialised: the first ``step()`` or ``state()``
call on a fresh system performs an implicit ``reinit()``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .base import Trajectory

__all__ = ["System"]


@runtime_checkable
class System(Protocol):
    """Structural type for steppable dynamical systems."""

    dim: int

    @property
    def is_discrete(self) -> bool:
        """True for iterated maps (and map-like wrappers such as Poincaré maps)."""
        ...

    def step(self, n_or_dt: float | int | None = None) -> np.ndarray:
        """Advance by ``n`` iterations (discrete) or ``dt`` time (continuous)."""
        ...

    def state(self) -> np.ndarray:
        """Return a copy of the current state vector."""
        ...

    def set_state(self, u: Any) -> None:
        """Overwrite the current state (not available for DDEs)."""
        ...

    def time(self) -> float:
        """Return the current time (continuous) or iteration count (discrete)."""
        ...

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Restart the stepper from state ``u`` at time ``t``."""
        ...

    def trajectory(self, *args: Any, **kwargs: Any) -> Trajectory:
        """Produce a trajectory on a uniform output grid (the alias of :meth:`run`).

        .. note::
            ``run`` is the canonical trajectory-producer verb (see the module
            docstring), but ``trajectory`` is the member the *structural*
            protocol requires: every family and wrapper implements it, whereas a
            few (``WrappedSystem`` and the derived wrappers) expose only
            ``trajectory`` — not ``run`` — so requiring ``run`` here would make
            them fail ``isinstance(obj, System)``.  Code written against
            ``System`` should call ``trajectory``; code holding a concrete flow
            or map should prefer ``run``.
        """
        ...


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
