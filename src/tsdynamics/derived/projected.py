"""Observation-side projection of a system onto a subset of components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from tsdynamics.errors import InvalidInputError, InvalidParameterError, remedy
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
        if isinstance(components, str | int | np.integer):
            components = (components,)
        # The INSTANCE names, not ``type(system).variables``: every system names
        # every component since v6, but a generated tuple (Lorenz96's ``y0..y4``,
        # a field system's blocks) lives only on the instance — so reading the
        # class refused the names of the 5 built-ins that generate theirs, and of
        # every user system that does not declare them on the class.
        names = tuple(getattr(system, "variables", ()) or ())
        idx = []
        for c in components:
            if isinstance(c, str):
                if c not in names:
                    raise InvalidParameterError(
                        f"{type(system).__name__} has no component {c!r}; it names "
                        f"{names if names else 'nothing'}."
                        + remedy(f"ts.derived.ProjectedSystem(system, {list(range(2))})")
                    )
                idx.append(names.index(c))
            else:
                idx.append(int(c))
        if not idx:
            raise InvalidParameterError(
                "components must name at least one component of the system."
                + remedy(
                    "ts.derived.ProjectedSystem(system, ['x', 'z'])",
                    "ts.derived.ProjectedSystem(system, [0, 2])",
                )
            )
        self.components = tuple(idx)
        self.complete = complete

    def _rebuild(self, inner: Any) -> ProjectedSystem:
        return ProjectedSystem(inner, self.components, complete=self.complete)

    @property
    def dim(self) -> int:
        """Dimension of the projected view."""
        return len(self.components)

    @property
    def variables(self) -> tuple[str, ...]:
        """Component names of the *projected* view (the inner names, subset).

        Overrides :class:`DerivedSystem`'s pass-through, which would return the
        inner system's *full* names and mislabel the projected columns.
        """
        inner = tuple(self.system.variables)
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
                + remedy(
                    "ts.derived.ProjectedSystem(system, [0, 2], "
                    "complete=lambda v: [v[0], 0.0, v[1]])"
                )
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

    def __repr__(self) -> str:
        """Name the inner system AND which components survive the projection."""
        names = tuple(self.system.variables)
        shown = ", ".join(names[i] if 0 <= i < len(names) else str(i) for i in self.components)
        return f"ProjectedSystem({type(self.system).__name__}, {shown})"

    def run(self, *args: Any, **run_kw: Any) -> Trajectory:
        """Run the **full** system and keep only the projected columns.

        Takes the inner family's own ``run`` vocabulary verbatim — ``final_time``
        / ``dt`` for a flow, ``steps`` for a map — so an unknown keyword is
        refused by the family that owns the word, with its reason.

        Returns
        -------
        Trajectory
            ``(T, len(components))``, back-referencing *this* wrapper so
            ``traj["x"]`` names the surviving columns.
        """
        traj = self.system.run(*args, **run_kw)
        # Overwrite the INNER system's recorded names: a run records the component
        # names it was produced with, and ``Trajectory.variables`` reads
        # ``meta["variables"]`` before the system — so carrying the full tuple
        # through would leave a 1-column projection with a 2-name record, which
        # resolves to the generated ``y0`` and mislabels every column.
        meta = {**traj.meta, "projected": self.components, "variables": self.variables}
        # Back-reference ``self`` (not the inner system): the returned ``y`` holds
        # only the projected columns, and ``self.variables`` names exactly those —
        # so ``traj["x"]`` resolves to the right column and an unknown name raises
        # KeyError rather than silently mislabelling or IndexError-ing.
        return Trajectory(traj.t, traj.y[:, list(self.components)], self, meta=meta)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
