"""
The ``System`` runtime protocol — the contract every analysis function consumes.

All four families (:class:`~tsdynamics.families.ContinuousSystem`,
:class:`~tsdynamics.families.DelaySystem`,
:class:`~tsdynamics.families.DiscreteMap`,
:class:`~tsdynamics.families.StochasticSystem`) and every derived-system wrapper
(:mod:`tsdynamics.derived`) implement this interface, so analysis code can be
written once against ``System`` and applied to anything that steps:

- ``run(...)`` is **the** verb that produces a
  :class:`~tsdynamics.families.Trajectory`.  ``trajectory`` / ``integrate`` /
  ``iterate`` were three more names for it and are gone in v6.  The horizon word
  follows the family — ``Lorenz().run(final_time=100, dt=0.01)`` and
  ``Henon().run(steps=5000)`` — because the two horizons are different
  quantities in different units.
- ``step(n_or_dt)`` advances the system and **returns the new state** — a count
  of iterations for a map, a time increment for a flow (each has a default).
- ``state()`` reads the current state.
- ``time()`` is the current time (or iteration count for a map).
- ``reinit(u, t=..., params=...)`` restarts the internal stepper.
- ``family`` is ``"ode" | "dde" | "map" | "sde"``.

**``set_state`` is deliberately NOT a protocol member.**  It is a per-family
*capability*: a delay system's state is a whole history function, so it cannot
be seated from a point, and on Python >= 3.12 ``isinstance`` checks data members
— keeping ``set_state`` in the protocol while removing it from ``DelaySystem``
would make ``isinstance(mg, System)`` **False**.  Ask for it with ``hasattr``.

Stepping state is lazily initialised: the first ``step()`` or ``state()``
call on a fresh system performs an implicit ``reinit()``.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .base import Trajectory

__all__ = ["System"]


@runtime_checkable
class System(Protocol):
    """Structural type for steppable dynamical systems."""

    dim: int
    family: Literal["ode", "dde", "map", "sde"]

    def run(self, *args: Any, **kwargs: Any) -> Trajectory:
        """Produce a :class:`~tsdynamics.families.Trajectory`.

        The one trajectory verb on every family and every wrapper.  The horizon
        keyword follows the family (``final_time`` for a flow, ``steps`` for a
        map); everything family-independent — ``ic``, ``seed``, ``backend``,
        ``transient`` — is spelled identically everywhere.
        """
        ...

    def step(self, n_or_dt: float | int | None = None) -> np.ndarray:
        """Advance the system and return the new state.

        The argument is the number of iterations for a discrete map and the time
        increment ``dt`` for a continuous flow (each family supplies a sensible
        default when ``None``).  Calling ``step`` on a fresh system performs an
        implicit :meth:`reinit` first.

        Returns
        -------
        ndarray
            The state after advancing.
        """
        ...

    def state(self) -> np.ndarray:
        """Return a copy of the current state vector.

        Calling ``state`` on a fresh system performs an implicit :meth:`reinit`
        first.  The returned array is a copy, so mutating it never disturbs the
        live stepper.
        """
        ...

    def time(self) -> float:
        """Return the current time (continuous) or iteration count (discrete)."""
        ...

    def reinit(self, u: Any | None = None, **kwargs: Any) -> None:
        """Restart the stepper from state ``u``."""
        ...


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
