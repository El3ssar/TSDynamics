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

#: Renderer keywords ``system.plot(...)`` forwards to the drawing backend rather
#: than to the integration.  Deliberately a **closed, small** set: on a *system*
#: (unlike a bare :class:`~tsdynamics.data.Trajectory`) an unrecognised keyword is
#: far more likely to be a misspelt integration keyword than a backend option, and
#: the old catch-all "everything left over goes to the renderer" rule made
#: ``lor.plot(final_time=2.0)`` a silent no-op — the renderer swallows ``**kwargs``.
#: Anything exotic goes through the explicit ``backend_kwargs=`` escape hatch.
#:
#: The set is the union of the keyword arguments the **in-tree** renderers name
#: (matplotlib ``figsize`` / ``dpi`` / ``fps`` / ``size``, plotly ``html`` /
#: ``path`` / ``full_html`` / ``include_plotlyjs``, json + threejs ``path`` /
#: ``indent`` / ``raw``, plus the dispatcher's ``warn``), so ``system.plot`` and
#: :meth:`Trajectory.plot <tsdynamics.data.Trajectory.plot>` accept the same
#: backend options.  None of them collide with an integration keyword.  An
#: out-of-tree renderer's own options go through ``backend_kwargs=``.
_RENDER_KEYS: frozenset[str] = frozenset(
    {
        "ax",
        "figsize",
        "dpi",
        "fps",
        "size",
        "warn",
        "html",
        "path",
        "full_html",
        "include_plotlyjs",
        "indent",
        "raw",
    }
)


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

    def plot(
        self,
        backend: str | None = None,
        *,
        backend_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render this system via a backend, applying inline tweaks first.

        Keywords are routed by category, in this order:

        1. plot-shaping keywords (``kind`` / ``components`` / the per-kind
           options) → :meth:`to_plot_spec`;
        2. inline spec tweaks (``xlabel`` / ``yscale`` / ``title`` / ``xlim`` /
           …) → applied to the spec;
        3. renderer options (:data:`_RENDER_KEYS`, plus anything in
           ``backend_kwargs``) → the drawing backend;
        4. **everything else → the integration** — forwarded through
           :meth:`to_plot_spec` to the family's ``trajectory`` (``final_time`` /
           ``dt`` / ``steps`` / ``ic`` / ``method`` / …), which validates them and
           raises :class:`~tsdynamics.errors.InvalidParameterError` on a typo.

        .. versionchanged:: 6.0
           Category 4 is new.  Previously every keyword that was not plot-shaping
           or an inline tweak was handed to the renderer, whose ``**kwargs`` then
           swallowed it — so ``lor.plot(final_time=2.0, dt=0.1)`` silently drew
           the *default* 100-time-unit trajectory, and ``lor.plot(finaltime=2.0)``
           was silently accepted.  Both now behave as written / raise.

        Parameters
        ----------
        backend : str, optional
            Renderer name; ``None`` uses the default capable backend.
        backend_kwargs : dict, optional
            Extra keyword arguments passed verbatim to the renderer — the escape
            hatch for backend options outside :data:`_RENDER_KEYS`.
        **kwargs
            Plot-shaping keywords, inline spec tweaks, renderer options, and
            integration keywords (see above).

        Returns
        -------
        Any
            Whatever the backend returns.
        """
        from tsdynamics.data.trajectory import _PLOT_SPEC_KEYS
        from tsdynamics.viz.spec import _COLORIZE_TWEAKS, _INLINE_TWEAKS, _apply_inline_tweaks

        def _take(keys: Any) -> dict[str, Any]:
            return {k: kwargs.pop(k) for k in list(kwargs) if k in keys}

        spec_kw = _take(_PLOT_SPEC_KEYS)
        tweak_kw = _take(_INLINE_TWEAKS.keys() | _COLORIZE_TWEAKS)
        render_kw = _take(_RENDER_KEYS)
        # Whatever is left is an integration keyword; ``to_plot_spec`` hands it to
        # the family's ``trajectory``, which is the one place that knows the valid
        # names and rejects a typo.
        spec = self.to_plot_spec(**spec_kw, **kwargs)
        _apply_inline_tweaks(spec, tweak_kw)
        return spec.render(backend, **render_kw, **(backend_kwargs or {}))

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
