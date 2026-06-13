"""Stroboscopic map: a forced flow sampled once per forcing period."""

from __future__ import annotations

from typing import Any

import numpy as np

from tsdynamics.base import Trajectory

from ._base import DerivedSystem

__all__ = ["StroboscopicMap"]


class StroboscopicMap(DerivedSystem):
    """
    Present a forced flow as the discrete map of once-per-period samples.

    One ``step()`` advances the underlying continuous system by exactly one
    forcing period and returns the new state.  Orbit diagrams over a
    ``StroboscopicMap`` are the standard way to study forced oscillators
    (Duffing, forced van der Pol, ...).

    Parameters
    ----------
    system : System
        A continuous-time system.
    period : float
        Sampling period (the forcing period).

    Examples
    --------
    >>> smap = StroboscopicMap(ForcedVanDerPol(), period=2 * np.pi / 0.63)
    >>> samples = smap.trajectory(300, transient=100)
    """

    def __init__(self, system: Any, period: float) -> None:
        super().__init__(system)
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        self.period = float(period)

    def _rebuild(self, inner: Any) -> StroboscopicMap:
        return StroboscopicMap(inner, self.period)

    @property
    def is_discrete(self) -> bool:
        """A stroboscopic map is a discrete view of the flow."""
        return True

    def step(self, n_or_dt: int | None = None) -> np.ndarray:
        """Advance ``n`` periods (default 1) and return the new state."""
        n = int(n_or_dt) if n_or_dt is not None else 1
        return self.system.step(n * self.period)

    def time(self) -> float:
        """Return the inner flow time."""
        return self.system.time()

    def trajectory(self, steps: int = 100, *, transient: int = 0, **kwargs: Any) -> Trajectory:
        """Collect ``steps`` once-per-period samples (after ``transient`` periods)."""
        if kwargs:
            self.reinit(kwargs.pop("ic", None), **kwargs)
        if transient:
            self.system.step(transient * self.period)
        times = np.empty(steps)
        points = np.empty((steps, self.system.dim))
        for k in range(steps):
            points[k] = self.step()
            times[k] = self.system.time()
        meta = {
            "derived": "StroboscopicMap",
            "period": self.period,
            "system": type(self.system).__name__,
            "params": self.params.as_dict(),
        }
        return Trajectory(t=times, y=points, system=self.system, meta=meta)
