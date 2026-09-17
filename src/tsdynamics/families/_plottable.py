"""Make the system families plottable (stream VIZ-SYSTEM-PLOT).

Gives every :class:`~tsdynamics.families.base.SystemBase` subclass — the
continuous / delay / discrete / stochastic families — a ``.plot()`` accessor and
a default ``__plot_spec__()`` so ``ts.Lorenz().plot()`` resolves end-to-end
through the visualization seam, exactly as the analysis result types already do.

The default :meth:`SystemPlottable.__plot_spec__` integrates a short default
trajectory (each family's own ``run`` with its defaults) and delegates
to :meth:`tsdynamics.data.Trajectory.__plot_spec__`, which already dispatches on
``family`` (a map → scatter orbit, a flow → time series / phase portrait).
Richer system draw-views (vector fields, cobwebs, component triples) are layered
on by the gap-fill stream; this is the safe default.

**Import-light:** this module imports :mod:`tsdynamics.viz` only *lazily*, inside
the methods, so ``import tsdynamics`` (which imports the family bases) never pulls
in the visualization package — the ``tsdynamics.viz``-stays-lazy guarantee holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tsdynamics.utils.plot_namespace import plot_namespace as _plot_namespace

if TYPE_CHECKING:
    from tsdynamics.data import Trajectory
    from tsdynamics.viz.spec import Plot

__all__ = ["SystemPlottable"]


class SystemPlottable:
    """Mixin adding ``__plot_spec__`` / ``.plot`` / a notebook hook to a system.

    Mixed into :class:`~tsdynamics.families.base.SystemBase`, so every system
    family inherits it.  A system describes itself by integrating a default
    trajectory and delegating to :meth:`tsdynamics.data.Trajectory.__plot_spec__`;
    the rendering sugar mirrors :class:`tsdynamics.viz.spec.Plottable` but is
    spelled out here with lazy imports so importing the family bases never drags
    in the visualization package.
    """

    if TYPE_CHECKING:
        # Provided by the concrete families (Continuous/Delay/Discrete/Stochastic)
        # this mixin is combined with; declared for the type checker only.
        def run(self, *args: Any, **kwargs: Any) -> Trajectory: ...

    def __plot_spec__(self, kind: str | None = None, **kwargs: Any) -> Plot:
        """Describe this system as a :class:`PlotSpec` via a default trajectory.

        Integrates the system with its family's ``run`` (defaults, or
        the integration keywords you pass — ``final_time`` / ``dt`` / ``steps`` /
        ``ic`` / …) and delegates to the trajectory's own ``__plot_spec__``.  The
        plot-shaping keywords (``components`` and the per-kind options ``delay`` /
        ``delay_time`` / ``color_by`` / ``transpose``) are split out and forwarded to the
        trajectory's ``__plot_spec__``; every other keyword goes to
        ``run``.  This split keys off the **closed** set of plot
        keywords (``tsdynamics.data.trajectory._PLOT_SPEC_KEYS``), so a system's
        own — possibly heterogeneous — ``trajectory`` signature stays open-ended.

        Parameters
        ----------
        kind : str, optional
            Override / select the semantic kind, forwarded to the trajectory's
            ``__plot_spec__`` (a ``PlotKind`` value or the ``"delay"`` recipe).
        **kwargs
            Plot-shaping keywords (``components`` / ``primitive`` / ``delay`` /
            ``delay_time`` / ``color_by`` / ``transpose``) forwarded to the trajectory's
            ``__plot_spec__``; all other keywords forwarded to ``run``
            (``final_time``, ``dt``, ``steps``, ``ic``, …).  ``primitive=``
            selects **how** the view is drawn (``"points"`` / ``"density"`` / …),
            validated against the transform's declared row — see
            :meth:`tsdynamics.data.Trajectory.__plot_spec__` and
            ``ts.viz.compatibility()``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.data.trajectory import _PLOT_SPEC_KEYS

        plot_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _PLOT_SPEC_KEYS}
        traj = self.run(**kwargs)
        return traj.__plot_spec__(kind=kind, **plot_kw)

    def _plot_impl(self, *transforms: Any, **kwargs: Any) -> Plot:
        """Build this system's :class:`PlotSpec`, applying inline tweaks first.

        ``plot`` **builds**, ``render`` **draws**, ``save`` **writes** — one word,
        one return type, everywhere::

            lor.plot()                             # a PlotSpec
            lor.plot(final_time=20.0).save("a.png")
            lor.plot().render("plotly")

        Keywords are routed by category, in this order:

        1. plot-shaping keywords (``kind`` / ``components`` / ``primitive`` /
           the per-kind options) → :meth:`__plot_spec__`;
        2. the **style** vocabulary (:data:`~tsdynamics.viz.style.STYLE_KEYS`
           and its aliases — ``color`` / ``lw`` / ``alpha`` / …) and ``theme``
           → applied to the finished spec.  These are the same spellings, with
           the same meanings, that ``ts.plot(system, color=...)`` accepts::

               lor.plot(final_time=20.0, color="crimson", title="Lorenz")

        3. inline spec tweaks (``xlabel`` / ``yscale`` / ``title`` / ``xlim`` /
           …) → applied to the spec;
        4. **everything else → the integration** — forwarded through
           :meth:`__plot_spec__` to the family's ``run`` (``final_time`` /
           ``dt`` / ``steps`` / ``ic`` / ``method`` / …), which validates them and
           raises :class:`~tsdynamics.errors.InvalidParameterError` on a typo.

        Renderer options (``ax`` / ``figsize`` / ``dpi`` / ``html`` / …) belong to
        :meth:`~tsdynamics.viz.spec.PlotSpec.render`, which is also where the
        backend is chosen — so there is no ``backend=`` here to be confused with
        the integration keywords.

        .. versionchanged:: 6.0
           Returns the :class:`~tsdynamics.viz.spec.PlotSpec` rather than the
           backend figure, so ``system.plot()`` and ``ts.plot(system)`` are the
           same kind of thing; and every keyword that is not plot-shaping or a
           tweak now reaches the integration (previously the renderer's
           ``**kwargs`` swallowed it, so ``lor.plot(final_time=2.0)`` silently
           drew the default 100-time-unit trajectory).

        Parameters
        ----------
        **kwargs
            Plot-shaping keywords, inline spec tweaks, and integration keywords
            (see above).

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.data.trajectory import _PLOT_SPEC_KEYS
        from tsdynamics.viz.spec import (
            _COLORIZE_TWEAKS,
            _INLINE_TWEAKS,
            reject_positional_transform,
        )
        from tsdynamics.viz.style import style_names

        reject_positional_transform(transforms, "system")

        def _take(keys: Any) -> dict[str, Any]:
            return {k: kwargs.pop(k) for k in list(kwargs) if k in keys}

        spec_kw = _take(_PLOT_SPEC_KEYS)
        # Style is peeled BEFORE the leftovers reach the integration: a style word
        # that fell through used to be reported as an invalid *integrate()*
        # keyword ("color is not a valid integrate()/run() keyword"), which names
        # the wrong vocabulary entirely.
        style_kw = _take(style_names())
        theme = kwargs.pop("theme", None)
        tweak_kw = _take(_INLINE_TWEAKS.keys() | _COLORIZE_TWEAKS)
        # Whatever is left is an integration keyword; ``__plot_spec__`` hands it to
        # the family's ``run``, which is the one place that knows the valid
        # names and rejects a typo.
        spec = self.__plot_spec__(**spec_kw, **kwargs)
        if theme is not None:
            spec.theme(theme)
        if style_kw:
            spec.style(**style_kw)
        return spec.tweak(**tweak_kw)

    #: ``subject.plot`` is BOTH the verb and the namespace (§6.7): ``plot()``
    #: draws the default view, ``plot.psd()`` / ``plot.nullclines()`` name a
    #: transform, and ``plot.<TAB>`` lists every transform that draws THIS
    #: subject — the discovery route ruling A2 promised when it took the
    #: analyses off the object.
    plot = _plot_namespace(_plot_impl)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Rich notebook display — renders inline once a backend is installed.

        Returns ``None`` (so IPython falls back to ``__repr__``) outside a
        notebook or when no rendering backend is installed, keeping notebook
        import of core plot-library-free.
        """
        from tsdynamics.viz.spec import _notebook_mimebundle

        return _notebook_mimebundle(lambda: self.__plot_spec__().render(), include, exclude)
