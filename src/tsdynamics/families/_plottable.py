"""Make the system families plottable (stream VIZ-SYSTEM-PLOT).

Gives every :class:`~tsdynamics.families.base.SystemBase` subclass — the
continuous / delay / discrete / stochastic families — a ``.plot()`` accessor and
a default ``to_plot_spec()`` so ``ts.Lorenz().plot()`` resolves end-to-end
through the visualization seam, exactly as the analysis result types already do.

The default :meth:`SystemPlottable.to_plot_spec` integrates a short default
trajectory (each family's own :meth:`trajectory` with its defaults) and delegates
to :meth:`tsdynamics.data.Trajectory.to_plot_spec`, which already dispatches on
``is_discrete`` (a map → scatter orbit, a flow → time series / phase portrait).
Richer system draw-views (vector fields, cobwebs, component triples) are layered
on by the gap-fill stream; this is the safe default.

**Import-light:** this module imports :mod:`tsdynamics.viz` only *lazily*, inside
the methods, so ``import tsdynamics`` (which imports the family bases) never pulls
in the visualization package — the ``tsdynamics.viz``-stays-lazy guarantee holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tsdynamics.data import Trajectory
    from tsdynamics.viz.spec import PlotSpec

__all__ = ["SystemPlottable"]


class SystemPlottable:
    """Mixin adding ``to_plot_spec`` / ``.plot`` / a notebook hook to a system.

    Mixed into :class:`~tsdynamics.families.base.SystemBase`, so every system
    family inherits it.  A system describes itself by integrating a default
    trajectory and delegating to :meth:`tsdynamics.data.Trajectory.to_plot_spec`;
    the rendering sugar mirrors :class:`tsdynamics.viz.spec.Plottable` but is
    spelled out here with lazy imports so importing the family bases never drags
    in the visualization package.
    """

    if TYPE_CHECKING:
        # Provided by the concrete families (Continuous/Delay/Discrete/Stochastic)
        # this mixin is combined with; declared for the type checker only.
        def trajectory(self, *args: Any, **kwargs: Any) -> Trajectory: ...

    def to_plot_spec(self, kind: str | None = None, **trajectory_kwargs: Any) -> PlotSpec:
        """Describe this system as a :class:`PlotSpec` via a default trajectory.

        Integrates the system with its family's :meth:`trajectory` defaults (or
        the ``trajectory_kwargs`` you pass — ``final_time`` / ``dt`` / ``steps`` /
        ``ic`` / …) and delegates to the trajectory's own ``to_plot_spec``, which
        dispatches on ``is_discrete`` (map orbit → scatter, flow → time series /
        phase portrait).

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind, forwarded to the trajectory's
            ``to_plot_spec``.
        **trajectory_kwargs
            Forwarded to the family's :meth:`trajectory` (e.g. ``final_time``,
            ``dt``, ``steps``, ``ic``).

        Returns
        -------
        PlotSpec
        """
        traj = self.trajectory(**trajectory_kwargs)
        return traj.to_plot_spec(kind=kind)

    def plot(self, backend: str | None = None, **tweaks: Any) -> Any:
        """Render this system via a backend, applying inline tweaks first.

        Builds the default :meth:`to_plot_spec`, applies any recognised inline
        tweaks (``xlabel`` / ``yscale`` / ``title`` / …), and renders through the
        backend dispatch.  Raises
        :class:`~tsdynamics.analysis._result.VisualizationNotInstalled` until a
        rendering backend is registered.

        Parameters
        ----------
        backend : str, optional
            Renderer name; ``None`` uses the default capable backend.
        **tweaks
            Inline spec tweaks and/or backend keyword arguments.

        Returns
        -------
        Any
            Whatever the backend returns.
        """
        from tsdynamics.viz.spec import _apply_inline_tweaks

        spec = self.to_plot_spec()
        backend_kw = _apply_inline_tweaks(spec, tweaks)
        return spec.render(backend, **backend_kw)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Rich notebook display — renders inline once a backend is installed.

        Returns ``None`` (so IPython falls back to ``__repr__``) when no rendering
        backend is registered, keeping notebook import of core plot-library-free.
        """
        from tsdynamics.viz.spec import _resolve_renderers

        if _resolve_renderers() is None:
            return None
        try:  # pragma: no cover - exercised only once a backend is installed
            return self.plot()
        except Exception:  # pragma: no cover - never break repr on a render error
            return None
