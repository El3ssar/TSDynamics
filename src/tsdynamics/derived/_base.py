"""Common machinery for derived-system wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.data import Trajectory
    from tsdynamics.families.base import MetaStore, ParamSet
    from tsdynamics.viz.spec import PlotSpec

__all__ = ["DerivedSystem"]


class DerivedSystem:
    """
    Base for wrappers that present an existing system through a new lens.

    A derived system implements the :class:`~tsdynamics.families.System` protocol
    by delegating to a wrapped system, transforming what "one step" or "the
    state" means (Poincaré crossings, stroboscopic samples, projections...).

    Parameters and metadata are forwarded to the wrapped system, and
    ``with_params`` re-parametrizes the *inner* system and rebuilds the
    wrapper, so parameter sweeps compose: an orbit diagram over a
    ``PoincareMap`` is a bifurcation diagram of the underlying flow.
    """

    def __init__(self, system: Any) -> None:
        self.system = system

    # --- forwarded surface ---

    @property
    def dim(self) -> int:
        return cast(int, self.system.dim)

    @property
    def params(self) -> ParamSet:
        return cast("ParamSet", self.system.params)

    @property
    def meta(self) -> MetaStore:
        return cast("MetaStore", self.system.meta)

    @property
    def variables(self) -> tuple[str, ...] | None:
        return getattr(type(self.system), "variables", None)

    def with_params(self, **overrides: Any) -> DerivedSystem:
        """Return a new wrapper of the same kind around a re-parametrized copy."""
        return self._rebuild(self.system.with_params(**overrides))

    def copy(self) -> DerivedSystem:
        """Return a new wrapper of the same kind around a copy of the inner system."""
        return self._rebuild(self.system.copy())

    def _rebuild(self, inner: Any) -> DerivedSystem:
        """Construct a new wrapper of the same kind around ``inner``."""
        raise NotImplementedError

    # --- default protocol delegation (subclasses override what differs) ---

    @property
    def is_discrete(self) -> bool:
        return cast(bool, self.system.is_discrete)

    def state(self) -> np.ndarray:
        return cast(np.ndarray, self.system.state())

    def set_state(self, u: Any) -> None:
        self.system.set_state(u)

    def time(self) -> float:
        return cast(float, self.system.time())

    def reinit(self, u: Any | None = None, **kwargs: Any) -> None:
        self.system.reinit(u, **kwargs)

    def trajectory(self, *args: Any, **kwargs: Any) -> Trajectory:
        """Produce the wrapper's trajectory — subclasses implement the lens-specific collection."""
        raise NotImplementedError

    def run(self, *args: Any, **kwargs: Any) -> Trajectory:
        """Produce the wrapper's trajectory — the alias of :meth:`trajectory`.

        ``run`` is the library's canonical trajectory-producer verb (a flow's
        ``Lorenz().run(...)``, a map's ``Henon().run(...)``), so a fluent
        derived view reads left-to-right with the same verb at the end::

            section = Rossler().poincare(section="y", at=0.0).run(steps=500)

        It forwards verbatim to this wrapper's :meth:`trajectory`, so the two are
        byte-identical and every wrapper-specific keyword (``transient``, ...) is
        honoured.  ``trajectory`` remains the member the structural ``System``
        protocol requires; ``run`` is the discoverable spelling.
        """
        return self.trajectory(*args, **kwargs)

    # --- visualization seam ---

    def to_plot_spec(self, kind: str | None = None) -> PlotSpec:
        """Describe this derived view as a backend-agnostic :class:`PlotSpec`.

        The default delegates to the wrapper's own :meth:`trajectory`: it
        collects the lens-specific trajectory (Poincaré crossings, projected
        columns, ...) and forwards to that trajectory's
        :meth:`~tsdynamics.data.Trajectory.to_plot_spec`.  Subclasses whose
        natural picture is *not* a single trajectory line — a stroboscopic
        scatter, an ensemble fan, a Lyapunov convergence curve — override this
        with their own spec builder.

        The :mod:`tsdynamics.viz.spec` import stays inside the trajectory method
        (lazy), so building a spec never pulls in a plotting backend.

        Parameters
        ----------
        kind : str, optional
            Override the auto-dispatched semantic kind with any member of the
            closed :class:`~tsdynamics.viz.spec.PlotKind` vocabulary.  ``None``
            (the default) lets the underlying trajectory auto-dispatch.

        Returns
        -------
        PlotSpec
        """
        return self.trajectory().to_plot_spec(kind=kind)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.system!r})"


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
