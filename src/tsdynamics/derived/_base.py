"""Common machinery for derived-system wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.data import Trajectory
    from tsdynamics.families.base import ParamSet
    from tsdynamics.viz.spec import PlotSpec

__all__ = ["DerivedSystem"]


def _reject_wrapper_keywords(
    view: Any, unknown: dict[str, Any], *, accepted: tuple[str, ...]
) -> None:
    """Refuse a leftover keyword **in the wrapper's own name**.

    A wrapper's ``run`` used to forward its leftovers straight into the inner
    system's ``reinit``, so ``pmap.run(steps=3, nonsense_kw=1)`` was reported as
    ``nonsense_kw is not a valid Rossler.reinit() keyword`` — a method and a class
    the caller never mentioned.  The word reached ``PoincareMap.run``; that is the
    door that must answer for it.
    """
    if not unknown:
        return
    from tsdynamics.families._kwargs import run_keyword_error

    bad = next(iter(unknown))
    raise run_keyword_error(
        view, bad, unknown[bad], family=view.family, accepted=accepted, verb="run"
    )


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

    def __getattr__(self, name: str) -> Any:
        """Answer a miss with the same teaching a system gives.

        A wrapper is a *system* as far as a user is concerned — ``pm.integrate``
        used to print a bare miss while ``lor.integrate`` printed the retired
        name, its reason and the line to type, so the wrappers were the one
        place in the library where guessing taught nothing.  This routes through
        the very same builder (:func:`~tsdynamics.families.base._absent_name_error`),
        which covers the retired verbs, the four deleted analysis namespaces and
        the near-miss ranking.

        A wrapper still does **not** forward unknown attributes to the inner
        system: this raises, it never delegates.
        """
        if name.startswith("_") or name in ("system",):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        if name == "to_plot_spec":
            from tsdynamics.utils.plot_namespace import plot_seam_error

            raise plot_seam_error(type(self).__name__, "view")
        from tsdynamics.families.base import _absent_name_error

        raise _absent_name_error(self, name)

    # --- forwarded surface ---

    @property
    def dim(self) -> int:
        return cast(int, self.system.dim)

    @property
    def params(self) -> ParamSet:
        """The inner system's parameters — an empty set when it declares none.

        Blindly forwarding ``self.system.params`` made ``PoincareMap(wrapped)``
        march correctly and then die with a raw ``AttributeError`` from this
        line, because a :class:`~tsdynamics.families.WrappedSystem` has no
        parameter set at all.
        """
        from tsdynamics.families.base import ParamSet as _ParamSet

        return cast("ParamSet", getattr(self.system, "params", None) or _ParamSet({}))

    @property
    def family(self) -> str:
        """The inner system's family word."""
        return cast(str, getattr(self.system, "family", "ode"))

    @property
    def variables(self) -> tuple[str, ...]:
        """The inner system's component names."""
        return cast("tuple[str, ...]", self.system.variables)

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
    def _is_discrete(self) -> bool:
        return cast(bool, self.system._is_discrete)

    def state(self) -> np.ndarray:
        return cast(np.ndarray, self.system.state())

    def set_state(self, u: Any) -> None:
        self.system.set_state(u)

    def time(self) -> float:
        return cast(float, self.system.time())

    def reinit(self, u: Any | None = None, **kwargs: Any) -> None:
        self.system.reinit(u, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Trajectory:
        """Produce this derived view's trajectory — **always from a fresh start**.

        The one trajectory verb, on a wrapper too::

            section = Rossler().poincare("y", 0.0).run(steps=500)

        Before v6 a wrapper's ``run`` continued the live stepper, so
        ``pmap.run(steps=5)`` twice returned *different* data.  One rule now, on
        every family and every wrapper: ``run()`` is a fresh run from ``ic``, and
        ``step()`` is the one that continues.
        """
        raise NotImplementedError(
            f"{type(self).__name__} inherits run() but does not implement it. "
            "A wrapper that cannot produce a trajectory must bind "
            "tsdynamics.families.base.Absent instead, so hasattr() and "
            "isinstance(view, System) tell the truth."
        )

    # --- visualization seam ---

    def __plot_spec__(self, kind: str | None = None, **kwargs: Any) -> PlotSpec:
        """Describe this derived view as a backend-agnostic :class:`PlotSpec`.

        The default delegates to the wrapper's own :meth:`run`: it
        collects the lens-specific trajectory (Poincaré crossings, projected
        columns, ...) and forwards to that trajectory's
        :meth:`~tsdynamics.data.Trajectory.__plot_spec__`.  Subclasses whose
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
        traj = self.run(**kwargs)
        return traj.__plot_spec__(kind=kind)

    def plot(self, *transforms: Any, **kwargs: Any) -> PlotSpec:
        """Draw this derived view.

        All five wrappers used to answer ``hasattr(pm, "plot") -> False`` while
        the plot seam was ``True``: the seam was there and the verb was not.
        """
        from tsdynamics.viz import plot as _plot

        return _plot(self, *transforms, **kwargs)

    def __repr__(self) -> str:
        inner = type(self.system).__name__
        return f"{type(self).__name__}({inner})"


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
