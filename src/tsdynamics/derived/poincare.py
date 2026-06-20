"""Poincaré (first-return) map of a continuous-time system."""

from __future__ import annotations

from typing import Any

import numpy as np

from tsdynamics.families import Trajectory

from . import _crossings
from ._base import DerivedSystem

__all__ = ["PoincareMap"]


class PoincareMap(DerivedSystem):
    """
    Present a flow as the discrete map of its crossings through a hyperplane.

    One ``step()`` advances the underlying system until the trajectory
    crosses the section plane in the chosen direction, refines the crossing
    point by cubic Hermite interpolation of the bracketing samples (using the
    system's numeric RHS for endpoint derivatives — O(dt⁴) accuracy), and
    returns the full-dimensional crossing state.

    Because a ``PoincareMap`` *is* a discrete system, everything written for
    maps applies to flows through it — e.g. an orbit diagram over a
    ``PoincareMap`` is a bifurcation diagram of the flow.

    Parameters
    ----------
    system : System
        A continuous-time system (ODE or DDE).
    plane : tuple
        Either ``(i, c)`` — the section ``y_i = c`` — or
        ``(normal, offset)`` with an arbitrary normal vector, for the
        section ``normal · y = offset``.
    direction : {+1, -1, 0}
        Count only crossings with ``d(normal·y)/dt > 0`` (+1, default),
        ``< 0`` (-1), or both (0).
    dt : float
        March step used for crossing detection.  The refinement makes the
        crossing itself far more accurate than ``dt``; this only needs to be
        small enough not to skip crossings.
    max_time : float
        Raise if no crossing is found within this much time (e.g. the plane
        misses the attractor).

    Examples
    --------
    >>> pmap = PoincareMap(Rossler(), plane=(0, 0.0), direction=+1)
    >>> section = pmap.trajectory(500)         # 500 crossings
    >>> section.y.shape
    (500, 3)
    """

    def __init__(
        self,
        system: Any,
        plane: tuple,
        *,
        direction: int = +1,
        dt: float = 0.01,
        max_time: float = 1e4,
    ) -> None:
        super().__init__(system)
        normal, offset = self._parse_plane(system.dim, plane)
        self.plane = plane
        self._normal = normal
        self._offset = offset
        self.direction = int(np.sign(direction))
        self.dt = float(dt)
        self.max_time = float(max_time)

        # Numeric RHS for Hermite endpoint derivatives; falls back to linear
        # interpolation (O(dt²)) for systems without one (e.g. DDEs).
        self._rhs = system._rhs_numeric() if hasattr(system, "_rhs_numeric") else None

        self._u_cross: np.ndarray | None = None
        self._t_cross: float | None = None
        self._n_cross = 0

    @staticmethod
    def _parse_plane(dim: int, plane: tuple) -> tuple[np.ndarray, float]:
        if len(plane) != 2:
            raise ValueError("plane must be (component_index, value) or (normal, offset)")
        first, second = plane
        if np.isscalar(first):
            i = int(first)
            if not 0 <= i < dim:
                raise ValueError(f"plane component index {i} out of range for dim={dim}")
            normal = np.zeros(dim)
            normal[i] = 1.0
            return normal, float(second)
        normal = np.asarray(first, dtype=float).reshape(dim)
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            raise ValueError("plane normal must be non-zero")
        return normal / norm, float(second) / norm

    def _rebuild(self, inner: Any) -> PoincareMap:
        return PoincareMap(
            inner,
            self.plane,
            direction=self.direction,
            dt=self.dt,
            max_time=self.max_time,
        )

    # --- section geometry ---

    def _g(self, u: np.ndarray) -> float:
        """Signed distance of state ``u`` from the section plane."""
        return float(self._normal @ u - self._offset)

    def _is_crossing(self, g_prev: float, g_now: float) -> bool:
        up = g_prev < 0.0 <= g_now
        down = g_prev > 0.0 >= g_now
        if self.direction > 0:
            return up
        if self.direction < 0:
            return down
        return up or down

    def _refine(
        self, t0: float, u0: np.ndarray, t1: float, u1: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Locate the crossing inside the bracket [t0, t1]."""
        from scipy.optimize import brentq

        g0, g1 = self._g(u0), self._g(u1)
        if self._rhs is None:
            # Linear interpolation — O(dt²)
            s = g0 / (g0 - g1)
            return t0 + s * (t1 - t0), u0 + s * (u1 - u0)

        # Cubic Hermite on [0, 1] with endpoint derivatives from the RHS — O(dt⁴)
        h = t1 - t0
        f0 = self._rhs(u0, t0)
        f1 = self._rhs(u1, t1)

        def u_at(s: float) -> np.ndarray:
            s2, s3 = s * s, s * s * s
            return (
                (2 * s3 - 3 * s2 + 1) * u0
                + (s3 - 2 * s2 + s) * h * f0
                + (-2 * s3 + 3 * s2) * u1
                + (s3 - s2) * h * f1
            )

        s_star = brentq(lambda s: self._g(u_at(s)), 0.0, 1.0, xtol=1e-14)
        return t0 + s_star * h, u_at(s_star)

    # --- System protocol (discrete view) ---

    @property
    def is_discrete(self) -> bool:
        """A Poincaré map is a discrete view of the flow."""
        return True

    def _advance_to_crossing(self) -> None:
        sys = self.system
        u_prev = sys.state()
        t_prev = sys.time()
        g_prev = self._g(u_prev)
        deadline = t_prev + self.max_time

        while True:
            u = sys.step(self.dt)
            t = sys.time()
            g = self._g(u)
            if self._is_crossing(g_prev, g):
                self._t_cross, self._u_cross = self._refine(t_prev, u_prev, t, u)
                self._n_cross += 1
                return
            if t >= deadline:
                raise RuntimeError(
                    f"PoincareMap: no section crossing within max_time={self.max_time} "
                    f"(plane may miss the attractor, or direction={self.direction} is wrong)."
                )
            u_prev, t_prev, g_prev = u, t, g

    def step(self, n_or_dt: int | None = None) -> np.ndarray:
        """Advance to the ``n``-th next crossing and return it (full-dim coords)."""
        n = int(n_or_dt) if n_or_dt is not None else 1
        for _ in range(n):
            self._advance_to_crossing()
        return self._u_cross.copy()

    def state(self) -> np.ndarray:
        """Return the last crossing point (or the inner state before any crossing)."""
        if self._u_cross is not None:
            return self._u_cross.copy()
        return self.system.state()

    def set_state(self, u: Any) -> None:
        """Overwrite the inner flow state and reset crossing bookkeeping."""
        self.system.set_state(u)
        self._u_cross = None
        self._t_cross = None

    def time(self) -> float:
        """Return the continuous time of the last crossing (inner time before any)."""
        return self._t_cross if self._t_cross is not None else self.system.time()

    @property
    def crossing_count(self) -> int:
        """Return the number of crossings recorded so far."""
        return self._n_cross

    def reinit(self, u: Any | None = None, **kwargs: Any) -> None:
        """Restart the inner flow and clear crossing bookkeeping."""
        self.system.reinit(u, **kwargs)
        # Parameter values are baked into the numeric RHS — rebuild it so the
        # Hermite refinement matches the (possibly re-parametrized) dynamics.
        if hasattr(self.system, "_rhs_numeric"):
            self._rhs = self.system._rhs_numeric()
        self._u_cross = None
        self._t_cross = None
        self._n_cross = 0

    def _python_trajectory(self, steps: int, transient: int) -> tuple[np.ndarray, np.ndarray]:
        """Collect crossings with the per-``dt`` Python march (DDE / no-RHS path)."""
        for _ in range(transient):
            self._advance_to_crossing()
        times = np.empty(steps)
        points = np.empty((steps, self.system.dim))
        for k in range(steps):
            self._advance_to_crossing()
            times[k] = self._t_cross
            points[k] = self._u_cross
        return times, points

    def _engine_trajectory(
        self, steps: int, transient: int, backend: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect crossings with the wired Rust event engine (one call)."""
        be = backend if backend is not None else getattr(self.system, "_default_backend", "interp")
        ic = self.system.state()
        t0 = self.system.time()
        times, points, t_final, u_final = _crossings.section_crossings(
            self.system,
            self._normal,
            self._offset,
            direction=self.direction,
            n_crossings=steps,
            transient=transient,
            dt=self.dt,
            max_time=self.max_time,
            backend=be,
            ic=ic,
            t0=t0,
        )
        # Advance the inner flow past the marched crossings so a subsequent
        # ``step()`` continues forward (the per-``dt`` loop advanced it too); the
        # cursor is the span end, just past the collected crossings.
        self.system.reinit(u_final, t=t_final)
        if steps:
            self._u_cross = np.asarray(points[-1], dtype=float).copy()
            self._t_cross = float(times[-1])
            self._n_cross += transient + steps
        return times, points

    def trajectory(
        self,
        steps: int = 100,
        *,
        transient: int = 0,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Trajectory:
        """
        Collect crossings as a trajectory.

        ``t`` holds the continuous crossing times; ``y`` the full-dimensional
        crossing states.  ``transient`` crossings are discarded first.

        For an ordinary (non-stiff) ODE on the compiled engine this marches the
        whole attractor and refines every crossing in **one engine call** (the
        wired Rust ``integrate_events``, stream WS-CROSSKERNEL) — ~100× faster than
        the per-``dt`` Python loop it replaces.  DDEs, systems without a numeric
        RHS, stiff defaults, and ``backend="reference"`` keep the Python loop.  The
        engine path is answer-identical to that loop's fixed-step (``rk4``)
        refinement; see :mod:`tsdynamics.derived._crossings`.

        Parameters
        ----------
        steps : int
            Number of crossings to collect.
        transient : int
            Number of leading crossings to discard.
        backend : {"interp", "jit", "reference"}, optional
            Engine evaluator for the fast path; defaults to the inner system's
            backend.  ``"reference"`` forces the pure-Python loop.
        """
        if kwargs:
            self.reinit(kwargs.pop("ic", None), **kwargs)

        if _crossings.engine_eligible(self.system, backend):
            from tsdynamics.engine.run import EngineNotAvailableError

            try:
                times, points = self._engine_trajectory(steps, transient, backend)
            except EngineNotAvailableError:
                times, points = self._python_trajectory(steps, transient)
        else:
            times, points = self._python_trajectory(steps, transient)

        meta = {
            "derived": "PoincareMap",
            # Section intent, so a renderer draws the 2-D in-plane scatter rather
            # than mistaking the full-dimensional crossing states for a flow line
            # (the string value of viz.PlotKind.POINCARE_SECTION).
            "plot_kind": "poincare_section",
            "plane": self.plane,
            "direction": self.direction,
            "dt": self.dt,
            "system": type(self.system).__name__,
            "params": self.params.as_dict(),
        }
        return Trajectory(t=times, y=points, system=self.system, meta=meta)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
