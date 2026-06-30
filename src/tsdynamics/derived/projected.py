"""Observation-side projection of a system onto a subset of components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from tsdynamics.errors import InvalidInputError
from tsdynamics.families import Trajectory

from ._base import DerivedSystem

__all__ = ["ProjectedSystem"]


class ProjectedSystem(DerivedSystem):
    """
    View a system through a subset of its components.

    The full system is stepped underneath; only ``state()``/``step()``
    *outputs* are projected.  ``set_state`` needs the inverse direction and
    therefore requires a ``complete`` callable mapping a projected state back
    to a full state.

    Parameters
    ----------
    system : System
        The full system.
    components : sequence of int or str
        Component indices (or names, when the system declares ``variables``).
    complete : callable, optional
        ``complete(u_projected) -> u_full`` used by ``set_state``/``reinit``
        when given projected-dimensional inputs.

    Examples
    --------
    >>> proj = ProjectedSystem(Lorenz(), ["x", "z"])
    >>> proj.step(0.01).shape
    (2,)
    """

    def __init__(
        self,
        system: Any,
        components: Any,
        *,
        complete: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        super().__init__(system)
        names = getattr(type(system), "variables", None)
        idx = []
        for c in components:
            if isinstance(c, str):
                if names is None:
                    raise ValueError(
                        f"{type(system).__name__} declares no `variables`; "
                        f"use integer component indices."
                    )
                idx.append(names.index(c))
            else:
                idx.append(int(c))
        if not idx:
            raise ValueError("components must be non-empty")
        self.components = tuple(idx)
        self.complete = complete

    def _rebuild(self, inner: Any) -> ProjectedSystem:
        return ProjectedSystem(inner, self.components, complete=self.complete)

    @property
    def dim(self) -> int:
        """Dimension of the projected view."""
        return len(self.components)

    @property
    def variables(self) -> tuple[str, ...] | None:
        """Component names of the *projected* view (the inner names, subset).

        Overrides :class:`DerivedSystem`'s pass-through (which would return the
        inner system's *full* names and mislabel the projected columns).  Returns
        ``None`` when the inner system declares no ``variables``.
        """
        inner = getattr(type(self.system), "variables", None)
        if inner is None:
            return None
        return tuple(inner[i] for i in self.components)

    def step(self, n_or_dt: float | int | None = None) -> np.ndarray:
        """Advance the full system; return the projected new state."""
        return cast(np.ndarray, self.system.step(n_or_dt)[list(self.components)])

    def state(self) -> np.ndarray:
        """Return the projected current state."""
        return cast(np.ndarray, self.system.state()[list(self.components)])

    def _to_full_state(self, u_arr: np.ndarray, *, verb: str) -> np.ndarray:
        """Map a user-supplied state to a full inner state.

        The full-vs-projected interpretation is disambiguated on **intent**, not
        purely on size, so a dimension-preserving projection (a permutation, e.g.
        ``components=[2, 1, 0]`` on a 3-D system) is handled correctly rather than
        silently written through untransformed:

        - When a ``complete`` callable is supplied, a projected-dimensional input
          (``size == self.dim``) is always reconstructed through it.  This takes
          precedence over the full-state shortcut, so the ambiguous case
          ``self.dim == self.system.dim`` resolves to the **projected** reading —
          the one the user opted into by supplying ``complete``.
        - With no ``complete`` callable, only a full-dimensional state can be set;
          a full state is written directly and a projected-dimensional one raises
          (it cannot be reconstructed).
        """
        if self.complete is not None and u_arr.size == self.dim:
            return np.asarray(self.complete(u_arr), dtype=float)
        if u_arr.size == self.system.dim:
            return u_arr
        if self.complete is None:
            raise NotImplementedError(
                f"ProjectedSystem.{verb} with a projected-dimensional state "
                f"(size {u_arr.size}) needs a `complete=` callable to reconstruct the "
                f"full {self.system.dim}-D state."
            )
        raise InvalidInputError(
            f"ProjectedSystem.{verb}: state of size {u_arr.size} matches neither the "
            f"projected dimension {self.dim} nor the full dimension {self.system.dim}."
        )

    def set_state(self, u: Any) -> None:
        """Overwrite the state (projected inputs need a ``complete`` callable)."""
        u_arr = np.asarray(u, dtype=float)
        self.system.set_state(self._to_full_state(u_arr, verb="set_state"))

    def reinit(self, u: Any | None = None, **kwargs: Any) -> None:
        """Restart the full system (projected inputs need a ``complete`` callable)."""
        if u is not None:
            u = self._to_full_state(np.asarray(u, dtype=float), verb="reinit")
        self.system.reinit(u, **kwargs)

    def trajectory(self, *args: Any, **kwargs: Any) -> Trajectory:
        """Full-system trajectory with projected columns."""
        traj = self.system.trajectory(*args, **kwargs)
        meta = {**traj.meta, "projected": self.components}
        # Back-reference ``self`` (not the inner system): the returned ``y`` holds
        # only the projected columns, and ``self.variables`` names exactly those —
        # so ``traj["x"]`` resolves to the right column and an unknown name raises
        # KeyError rather than silently mislabelling or IndexError-ing.
        return Trajectory(traj.t, traj.y[:, list(self.components)], self, meta=meta)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
