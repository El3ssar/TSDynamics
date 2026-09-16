"""Adapt an arbitrary external stepper to the System protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from tsdynamics.errors import ConvergenceError, InvalidInputError

from ._derive import DeriveMixin
from ._plottable import SystemPlottable
from .base import Absent, Trajectory, _absent_slot

__all__ = ["WrappedSystem"]


class WrappedSystem(DeriveMixin, SystemPlottable):
    """
    Wrap any external stepping rule as a first-class :class:`System`.

    Give it a ``step_fn(state, n_or_dt) -> new_state`` and a dimension, and the
    whole analysis toolkit (orbit diagrams, Lyapunov-from-rescaling, Poincaré
    sections, ensembles, basins) applies to your own simulation code — a
    foreign ODE solver, an agent-based model, a hardware-in-the-loop rig,
    anything that advances a state vector.

    Parameters
    ----------
    step_fn : callable
        ``step_fn(state, n_or_dt) -> new_state``.  ``new_state`` is array-like
        of length ``dim``.  ``n_or_dt`` is the iteration count (discrete) or
        the time increment (continuous); pass it through to your stepper.
    dim : int
        State-space dimension.
    family : {"map", "ode"}, default "map"
        Whether ``n_or_dt`` counts iterations (True) or measures time (False).
    initial : array-like, optional
        Default initial state used when ``reinit`` gets no explicit state.
    default_dt : float, default 1.0
        Step taken by ``step()`` when called with no argument.
    variables : tuple of str, optional
        Component names, enabling ``traj["x"]`` on produced trajectories.
        Defaults to ``y0 ... y{dim-1}`` — never ``None``.

    Examples
    --------
    >>> # a plain logistic map written by hand
    >>> import numpy as np
    >>> def step(u, n):
    ...     x = u[0]
    ...     for _ in range(int(n)):
    ...         x = 3.9 * x * (1 - x)
    ...     return [x]
    >>> sysm = WrappedSystem(step, dim=1, family="map", initial=[0.5])
    >>> traj = sysm.run(500)
    >>> import tsdynamics as ts
    >>> ts.analysis.max_lyapunov(sysm, ic=[0.3]) > 0   # chaotic
    True
    """

    def __init__(
        self,
        step_fn: Callable[[np.ndarray, float], Any],
        *,
        dim: int,
        family: str = "map",
        initial: Any | None = None,
        default_dt: float = 1.0,
        variables: tuple[str, ...] | None = None,
    ) -> None:
        self._step_fn = step_fn
        self.dim = int(dim)
        if family not in ("ode", "map"):
            from tsdynamics.errors import invalid_value

            raise invalid_value(
                "family",
                family,
                rule='must be "ode" (a flow you step by dt) or "map" (a rule you iterate)',
                hint="WrappedSystem adapts an opaque stepper; a delay or noise "
                "kernel needs DelaySystem / StochasticSystem, which own the history "
                "and the Wiener stream.",
            )
        self._family = family
        self._is_discrete = family == "map"
        self._initial = None if initial is None else np.asarray(initial, dtype=float).reshape(dim)
        self._default_dt = float(default_dt)
        # v6: every system names its own components.  ``None`` used to mean
        # "unnamed", which made ``traj["x"]`` and a section plane by name a guess.
        self.variables: tuple[str, ...] = (
            tuple(str(v) for v in variables)
            if variables is not None
            else tuple(f"y{i}" for i in range(self.dim))
        )

        self._state: np.ndarray | None = None
        self._t: float = 0.0

    # --- System protocol ---

    #: A wrapper holds an opaque stepper, so there is no right-hand side.
    jacobian = Absent(
        "a WrappedSystem wraps an opaque stepper, so there is no right-hand side to differentiate",
        "ts.analysis.max_lyapunov(system, ic=[0.3])",
    )

    #: ... and no symbolic tree either.
    jacobian_sym = Absent(
        "a WrappedSystem wraps an opaque stepper, so there is no right-hand side to differentiate",
        "ts.analysis.max_lyapunov(system, ic=[0.3])",
    )

    def __dir__(self) -> list[str]:
        """List what this wrapper actually answers — no ``Absent`` slots."""
        cls = type(self)
        return sorted(n for n in set(super().__dir__()) if _absent_slot(cls, n) is None)

    @property
    def family(self) -> str:
        """``"ode"`` or ``"map"`` — the kind of dynamics this stepper implements."""
        return self._family

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Restart from state ``u`` (falls back to ``initial``, then zeros)."""
        if u is not None:
            self._state = np.asarray(u, dtype=float).reshape(self.dim)
        elif self._initial is not None:
            self._state = self._initial.copy()
        else:
            self._state = np.zeros(self.dim)
        self._t = float(t) if t is not None else 0.0

    def step(self, n_or_dt: float | None = None) -> np.ndarray:
        """Advance by ``n_or_dt`` (default ``default_dt``) and return the new state."""
        if self._state is None:
            self.reinit()
        assert self._state is not None
        amount = self._default_dt if n_or_dt is None else n_or_dt
        new = np.asarray(self._step_fn(self._state, amount), dtype=np.float64).reshape(self.dim)
        if not np.all(np.isfinite(new)):
            raise ConvergenceError("WrappedSystem.step produced non-finite state")
        self._state = new
        self._t += amount
        return self._state.copy()

    def state(self) -> np.ndarray:
        """Return a copy of the current state."""
        if self._state is None:
            self.reinit()
        assert self._state is not None
        return self._state.copy()

    def set_state(self, u: Any) -> None:
        """Overwrite the current state."""
        self._state = np.asarray(u, dtype=float).reshape(self.dim)

    def time(self) -> float:
        """Return the current time (continuous) or iteration count (discrete)."""
        return self._t

    def copy(self) -> WrappedSystem:
        """Return a fresh wrapper sharing the same step rule (independent state)."""
        return WrappedSystem(
            self._step_fn,
            dim=self.dim,
            family=self._family,
            initial=self._initial,
            default_dt=self._default_dt,
            variables=self.variables,
        )

    def run(
        self,
        steps: int | None = None,
        *,
        transient: int = 0,
        ic: Any | None = None,
        final_time: float | None = None,
        dt: float | None = None,
    ) -> Trajectory:
        """Step the wrapper repeatedly (after ``transient``) and collect a Trajectory.

        The wrapper accepts **both** the map-style positional sample count ``n``
        and — for a *continuous* wrapper — the family-uniform ``final_time`` /
        ``dt`` spelling, so a generic protocol caller (which calls
        ``trajectory(final_time=…, dt=…)``) works without raising ``TypeError``.

        Exactly one count source must be supplied: either ``n`` (the number of
        samples to collect), or ``final_time``.  When ``final_time`` is given the
        sample count is ``round(final_time / dt)`` and the per-step increment is
        ``dt``, which defaults to ``default_dt`` for **both** continuous and
        discrete wrappers.  (A discrete wrapper that wants the "one time unit =
        one iteration" convention should construct itself with ``default_dt=1.0``,
        the constructor default.)

        Parameters
        ----------
        steps : int, optional
            Number of samples to collect.  Mutually exclusive with ``final_time``.
        transient : int, default 0
            Number of leading samples to discard (steps taken but not recorded).
        ic : array-like, optional
            Initial state for the run (falls back to ``initial``, then zeros).
        final_time : float, optional
            Integration horizon, an alternative to ``n`` for continuous wrappers
            (and accepted for discrete ones).  The sample count is
            ``round(final_time / dt)``.
        dt : float, optional
            Per-step increment used with ``final_time`` (and as the ``step``
            argument).  Defaults to ``default_dt``.

        Returns
        -------
        Trajectory
            ``n`` recorded samples with their times.

        Raises
        ------
        InvalidInputError
            If neither ``n`` nor ``final_time`` is given, if both are, or if the
            resolved sample count is not positive.

        Examples
        --------
        >>> import numpy as np
        >>> flow = lambda u, dt: [u[0] * np.exp(0.5 * dt)]
        >>> w = WrappedSystem(flow, dim=1, family="ode", default_dt=0.1)
        >>> traj = w.run(final_time=1.0, dt=0.1)   # the one trajectory verb
        >>> traj.y.shape
        (10, 1)
        """
        step_dt = self._default_dt if dt is None else float(dt)
        if final_time is not None:
            if steps is not None:
                raise InvalidInputError(
                    "WrappedSystem.run: pass either steps or final_time, not both."
                )
            steps = int(round(float(final_time) / step_dt))
        elif steps is None:
            raise InvalidInputError("WrappedSystem.run requires steps (a count) or final_time.")
        n = int(steps)
        if n <= 0:
            raise InvalidInputError(f"WrappedSystem.run needs a positive count, got {n}.")

        self.reinit(ic)
        for _ in range(transient):
            self.step(step_dt)
        ts = np.empty(n)
        ys = np.empty((n, self.dim))
        for i in range(n):
            ys[i] = self.step(step_dt)
            ts[i] = self._t
        meta = {"system": "WrappedSystem", "family": self._family}
        return Trajectory(t=ts, y=ys, system=self, meta=meta)

    def __repr__(self) -> str:
        return f"WrappedSystem(dim={self.dim}, family={self._family!r})"


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
