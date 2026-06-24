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

    def to_plot_spec(self, kind: str | None = None, **kwargs: Any) -> PlotSpec:
        """Describe this system as a :class:`PlotSpec` via a default trajectory.

        Integrates the system with its family's :meth:`trajectory` (defaults, or
        the integration keywords you pass — ``final_time`` / ``dt`` / ``steps`` /
        ``ic`` / …) and delegates to the trajectory's own ``to_plot_spec``.  The
        plot-shaping keywords (``components`` and the per-kind options ``tau`` /
        ``color_by`` / ``transpose``) are split out and forwarded to the
        trajectory's ``to_plot_spec``; every other keyword goes to
        :meth:`trajectory`.  This split keys off the **closed** set of plot
        keywords (``tsdynamics.data.trajectory._PLOT_SPEC_KEYS``), so a system's
        own — possibly heterogeneous — ``trajectory`` signature stays open-ended.

        Parameters
        ----------
        kind : str, optional
            Override / select the semantic kind, forwarded to the trajectory's
            ``to_plot_spec`` (a ``PlotKind`` value or the ``"delay"`` recipe).
        **kwargs
            Plot-shaping keywords (``components`` / ``tau`` / ``color_by`` /
            ``transpose``) forwarded to the trajectory's ``to_plot_spec``; all
            other keywords forwarded to :meth:`trajectory` (``final_time``,
            ``dt``, ``steps``, ``ic``, …).

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.data.trajectory import _PLOT_SPEC_KEYS

        plot_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _PLOT_SPEC_KEYS}
        traj = self.trajectory(**kwargs)
        return traj.to_plot_spec(kind=kind, **plot_kw)

    def plot(self, backend: str | None = None, **kwargs: Any) -> Any:
        """Render this system via a backend, applying inline tweaks first.

        Peels off the plot-shaping keywords (``kind`` / ``components`` and the
        per-kind options), forwards them to :meth:`to_plot_spec`, applies any
        recognised inline tweaks (``xlabel`` / ``yscale`` / ``title`` / …) to the
        spec, and renders through the backend dispatch.  Raises
        :class:`~tsdynamics.analysis._result.VisualizationNotInstalled` until a
        rendering backend is registered.

        Parameters
        ----------
        backend : str, optional
            Renderer name; ``None`` uses the default capable backend.
        **kwargs
            Plot-shaping keywords (``kind`` / ``components`` / per-kind options),
            inline spec tweaks, and/or backend keyword arguments.

        Returns
        -------
        Any
            Whatever the backend returns.
        """
        from tsdynamics.data.trajectory import _PLOT_SPEC_KEYS
        from tsdynamics.viz.spec import _apply_inline_tweaks

        spec_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _PLOT_SPEC_KEYS}
        spec = self.to_plot_spec(**spec_kw)
        backend_kw = _apply_inline_tweaks(spec, kwargs)
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
