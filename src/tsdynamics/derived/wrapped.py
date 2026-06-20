"""Adapt an arbitrary external stepper to the System protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from tsdynamics.families import Trajectory

__all__ = ["WrappedSystem"]


class WrappedSystem:
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
    is_discrete : bool, default True
        Whether ``n_or_dt`` counts iterations (True) or measures time (False).
    initial : array-like, optional
        Default initial state used when ``reinit`` gets no explicit state.
    default_dt : float, default 1.0
        Step taken by ``step()`` when called with no argument.
    variables : tuple of str, optional
        Component names, enabling ``traj["x"]`` on produced trajectories.

    Examples
    --------
    >>> # a plain logistic map written by hand
    >>> import numpy as np
    >>> def step(u, n):
    ...     x = u[0]
    ...     for _ in range(int(n)):
    ...         x = 3.9 * x * (1 - x)
    ...     return [x]
    >>> sysm = WrappedSystem(step, dim=1, is_discrete=True, initial=[0.5])
    >>> traj = sysm.trajectory(500)
    >>> import tsdynamics as ts
    >>> ts.max_lyapunov(sysm, ic=[0.3]) > 0          # chaotic
    True
    """

    def __init__(
        self,
        step_fn: Callable[[np.ndarray, float], Any],
        *,
        dim: int,
        is_discrete: bool = True,
        initial: Any | None = None,
        default_dt: float = 1.0,
        variables: tuple[str, ...] | None = None,
    ) -> None:
        self._step_fn = step_fn
        self.dim = int(dim)
        self._is_discrete = bool(is_discrete)
        self._initial = None if initial is None else np.asarray(initial, dtype=float).reshape(dim)
        self._default_dt = float(default_dt)
        self.variables = variables

        self._state: np.ndarray | None = None
        self._t: float = 0.0

    # --- System protocol ---

    @property
    def is_discrete(self) -> bool:
        """Whether stepping counts iterations rather than time."""
        return self._is_discrete

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict | None = None,
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
        amount = self._default_dt if n_or_dt is None else n_or_dt
        new = np.asarray(self._step_fn(self._state, amount), dtype=float).reshape(self.dim)
        if not np.all(np.isfinite(new)):
            raise RuntimeError("WrappedSystem.step produced non-finite state")
        self._state = new
        self._t += amount
        return self._state.copy()

    def state(self) -> np.ndarray:
        """Return a copy of the current state."""
        if self._state is None:
            self.reinit()
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
            is_discrete=self._is_discrete,
            initial=self._initial,
            default_dt=self._default_dt,
            variables=self.variables,
        )

    def trajectory(
        self,
        n: int,
        *,
        transient: int = 0,
        ic: Any | None = None,
    ) -> Trajectory:
        """Step ``n`` times (after ``transient``) and collect a Trajectory."""
        self.reinit(ic)
        for _ in range(transient):
            self.step()
        ts = np.empty(n)
        ys = np.empty((n, self.dim))
        for i in range(n):
            ys[i] = self.step()
            ts[i] = self._t
        meta = {"system": "WrappedSystem", "is_discrete": self._is_discrete}
        return Trajectory(t=ts, y=ys, system=self, meta=meta)

    def __repr__(self) -> str:
        kind = "discrete" if self._is_discrete else "continuous"
        return f"WrappedSystem(dim={self.dim}, {kind})"


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
