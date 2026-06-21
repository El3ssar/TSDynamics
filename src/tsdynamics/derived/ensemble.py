"""Ensemble of identical systems stepped in lockstep."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

__all__ = ["EnsembleSystem"]


class EnsembleSystem:
    """
    Many copies of one system, advanced synchronously from different states.

    Used for two-trajectory Lyapunov estimates, basin sampling, and ensemble
    statistics.  Members are independent copies — parameters are shared at
    construction, states are per-member.

    Parameters
    ----------
    system : System
        The template system (copied per member; the original is untouched).
    states : array-like, shape (m, dim)
        One initial state per member.

    Examples
    --------
    >>> ens = EnsembleSystem(Lorenz(), [[1, 1, 1], [1.001, 1, 1]])
    >>> ens.step(0.01)
    array([[...], [...]])
    """

    def __init__(self, system: Any, states: Any) -> None:
        states_arr = np.atleast_2d(np.asarray(states, dtype=float))
        if states_arr.shape[1] != system.dim:
            raise ValueError(f"states must have shape (m, {system.dim}), got {states_arr.shape}")
        self.template = system
        self.members = []
        for s in states_arr:
            member = system.copy()
            member.reinit(s)
            self.members.append(member)

    @property
    def size(self) -> int:
        """Number of ensemble members."""
        return len(self.members)

    @property
    def dim(self) -> int:
        """State-space dimension of each member."""
        return cast(int, self.template.dim)

    @property
    def is_discrete(self) -> bool:
        """Match the template system's time semantics."""
        return cast(bool, self.template.is_discrete)

    def step(self, n_or_dt: float | int | None = None) -> np.ndarray:
        """Advance every member and return the stacked states, shape (m, dim)."""
        return np.array([m.step(n_or_dt) for m in self.members])

    def states(self) -> np.ndarray:
        """Return the current states, shape (m, dim)."""
        return np.array([m.state() for m in self.members])

    def set_states(self, states: Any) -> None:
        """Overwrite every member's state."""
        states_arr = np.atleast_2d(np.asarray(states, dtype=float))
        if states_arr.shape != (self.size, self.dim):
            raise ValueError(f"expected shape {(self.size, self.dim)}, got {states_arr.shape}")
        for member, s in zip(self.members, states_arr, strict=True):
            member.set_state(s)

    def time(self) -> float:
        """Return the common member time."""
        return self.members[0].time() if self.members else 0.0

    def __len__(self) -> int:
        return len(self.members)

    def __repr__(self) -> str:
        return f"EnsembleSystem({type(self.template).__name__}, size={self.size})"


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
