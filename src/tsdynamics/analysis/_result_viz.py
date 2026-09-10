"""Visualization seam for the analysis result classes.

Holds the plotting machinery split out of ``analysis/_result.py``:
:class:`VisualizationNotInstalled` (the canonical public exception the whole viz
layer raises), the renderer-registry probe ``_available_renderers``, and the
:class:`_PlotAccessor` that backs ``AnalysisResult.plot`` (callable *and* a
namespace of typed kind methods).  Nothing here imports a plotting library at
module scope — every viz import is deferred to call time.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tsdynamics.analysis._result_base import AnalysisResult

#: Keyword arguments that belong to a *rendering backend* rather than to a
#: result's ``to_plot_spec``.  ``result.plot(**kw)`` splits its keywords on this
#: table: a name ``to_plot_spec`` declares goes to the spec builder, a name here
#: goes to the renderer, and **anything else raises** (see :meth:`_PlotAccessor._render`).
#: A keyword that *some* backend accepts but the **chosen** one does not is caught
#: one layer down, by :func:`~tsdynamics.viz.render.render_spec`.
#:
#: Before this table existed, ``.plot()`` called ``to_plot_spec()`` with *no*
#: arguments and forwarded every keyword to ``render`` — where the in-tree
#: backends absorb unknown keywords in a ``**_kw`` catch-all.  The result was
#: total silence: ``od.plot()``, ``od.plot(annotate=True)`` and
#: ``od.plot(totally_bogus_kwarg=42)`` produced **byte-identical** PNGs.  Every
#: result-plot parameter (``OrbitDiagram(annotate=)`` and friends) was therefore
#: unreachable, and every typo was invisible.
#:
#: The union of the in-tree backends' render keywords, **derived** from the
#: per-backend declarations in :data:`tsdynamics.viz.render.caps._BUILTIN_RENDER_KWARGS`
#: rather than hand-copied here.  The hand-copied version had drifted: it listed
#: ``"standalone"``, which no backend has ever accepted, and omitted four three.js
#: keywords (``loader_url`` / ``poster`` / ``background`` / ``title``), so
#: ``result.plot(poster=False)`` raised "unexpected keyword" for an option that
#: exists.  A *union* is the right shape here only because this split happens
#: before a backend is chosen; the per-backend check that catches a keyword the
#: chosen backend cannot read runs in
#: :func:`~tsdynamics.viz.render.render_spec`.  A third-party backend's keyword is
#: reached through ``spec.render(backend, **kw)`` directly.
#:
#: Resolved lazily (never at import) so the analysis layer still pulls in no
#: plotting library — ``caps`` is import-light but lives under ``tsdynamics.viz``,
#: which is deliberately bound lazily.
_RENDERER_KWARGS_FALLBACK: frozenset[str] = frozenset(
    {"figsize", "full_html", "html", "include_plotlyjs", "indent", "path", "raw"}
)


def _renderer_kwargs() -> frozenset[str]:
    """Return every render keyword any in-tree backend accepts (lazily resolved)."""
    try:
        from tsdynamics.viz.render.caps import _BUILTIN_RENDER_KWARGS
    except Exception:  # pragma: no cover - viz layer unavailable
        return _RENDERER_KWARGS_FALLBACK
    return frozenset().union(*_BUILTIN_RENDER_KWARGS.values())


_VIZ_HINT = (
    "No visualization backend is available: install one (e.g. `pip install "
    "matplotlib` or `pip install plotly`), or export the data with .to_dict() / "
    ".to_frame() and plot it yourself."
)


class VisualizationNotInstalled(ImportError):  # noqa: N818 — canonical v4 public name
    """Raised by ``result.plot`` when no visualization backend is available.

    Subclasses :class:`ImportError` so the plotting machinery reads like any
    other optional dependency: ``except ImportError`` catches it.  It is raised
    only when *no* rendering backend can be registered — e.g. a wheel-free
    environment with neither matplotlib nor plotly installed.
    """


def _available_renderers() -> Any | None:
    """Return the renderer registry if it has a usable backend, else ``None``.

    Seeds the in-tree rendering backends first (matplotlib / plotly / json /
    threejs) via :func:`tsdynamics.viz.render.register_builtin_renderers`, then
    reports the registry only if it ended up non-empty.  The seeding step is what
    makes ``result.plot()`` work on the **very first** viz action in a fresh
    process: without it the registry is empty until some other code path happens
    to register a backend, so the gate would spuriously raise
    :class:`VisualizationNotInstalled` even with matplotlib installed.

    The viz import is deferred to call time (never at module scope), so importing
    the result layer still pulls in no plotting library.  The registration helper
    is reached through the module object (``render.register_builtin_renderers``)
    rather than a bound import so tests can monkeypatch it.

    Returns
    -------
    tsdynamics.registry.Registry or None
        The renderer registry when at least one backend registered, else
        ``None`` (no backend installed → the caller raises).
    """
    try:
        from tsdynamics.viz import render

        render.register_builtin_renderers()
    except Exception:  # best-effort seeding — fall through to whatever is registered
        pass
    try:
        from tsdynamics.registry import renderers
    except Exception:  # pragma: no cover - defensive
        return None
    try:
        return renderers if len(renderers) else None
    except Exception:  # pragma: no cover - defensive
        return None


def _spec_keywords(to_spec: Any) -> frozenset[str]:
    """Return the keyword names ``to_spec`` declares (empty if it takes ``**kwargs``).

    A ``to_plot_spec`` with a ``**kwargs`` catch-all is treated as accepting
    *everything*, signalled by returning ``None``-like behaviour at the call site
    (see :func:`_split_plot_kwargs`).
    """
    try:
        sig = inspect.signature(to_spec)
    except (TypeError, ValueError):  # pragma: no cover - builtin / C callable
        return frozenset()
    return frozenset(
        name
        for name, p in sig.parameters.items()
        if name != "kind" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and name != "self"
    )


def _accepts_var_keyword(to_spec: Any) -> bool:
    """Whether ``to_spec`` has a ``**kwargs`` catch-all (so it accepts anything)."""
    try:
        sig = inspect.signature(to_spec)
    except (TypeError, ValueError):  # pragma: no cover - builtin / C callable
        return False
    return any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())


def _split_plot_kwargs(
    to_spec: Any, tweaks: dict[str, Any], owner: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ``.plot(**tweaks)`` into ``(spec_kwargs, backend_kwargs)``, or raise.

    See :meth:`_PlotAccessor._render` for why this exists.  A keyword the result's
    ``to_plot_spec`` declares wins over the renderer table, because that is where
    the caller's intent lives (a result that declares ``path=`` means its own
    ``path``).
    """
    spec_names = _spec_keywords(to_spec)
    takes_any = _accepts_var_keyword(to_spec)
    renderer_names = _renderer_kwargs()
    spec_kw: dict[str, Any] = {}
    backend_kw: dict[str, Any] = {}
    unknown: list[str] = []
    for name, value in tweaks.items():
        if name in spec_names or takes_any:
            spec_kw[name] = value
        elif name in renderer_names:
            backend_kw[name] = value
        else:
            unknown.append(name)
    if unknown:
        from tsdynamics.errors import InvalidParameterError

        accepted = ", ".join(sorted(spec_names)) or "(none)"
        raise InvalidParameterError(
            f"{owner}.plot() got unexpected keyword argument(s) "
            f"{', '.join(repr(u) for u in sorted(unknown))}. "
            f"Accepted spec keywords for this result: {accepted}. "
            f"Accepted backend keywords: {', '.join(sorted(renderer_names))}."
        )
    return spec_kw, backend_kw


class _PlotAccessor:
    """The ``result.plot`` seam: callable *and* a namespace of typed methods.

    ``result.plot()`` renders the result's default view; the named methods
    (``result.plot.scaling()``, ``.phase()``, …) force a particular semantic
    plot kind.  Every entry point funnels through :meth:`_render`, which seeds the
    in-tree backends and renders, raising :class:`VisualizationNotInstalled` only
    when no rendering backend can be registered (a wheel-free environment).
    """

    __slots__ = ("_result",)

    def __init__(self, result: AnalysisResult) -> None:
        self._result = result

    def __call__(
        self, backend: str | None = None, *, kind: str | None = None, **tweaks: Any
    ) -> Any:
        """Render the result's default view (see :meth:`_render`)."""
        return self._render(kind=kind, backend=backend, **tweaks)

    # -- typed kind methods (the closed plot vocabulary; see viz.PlotKind) --

    def scaling(self, **kw: Any) -> Any:
        """Plot as a log--log scaling fit (dimensions / Lyapunov-from-data)."""
        return self._render(kind="scaling_fit", **kw)

    def diagnostic(self, **kw: Any) -> Any:
        """Plot as a diagnostic growth/decay curve (GALI, divergence)."""
        return self._render(kind="diagnostic_curve", **kw)

    def time_series(self, **kw: Any) -> Any:
        """Plot as a one-dimensional time series."""
        return self._render(kind="time_series", **kw)

    def phase(self, **kw: Any) -> Any:
        """Plot as a 2-D phase portrait (3-D results pass ``kind="phase_portrait_3d"``)."""
        return self._render(kind="phase_portrait_2d", **kw)

    def image(self, **kw: Any) -> Any:
        """Plot as a 2-D image (recurrence matrix, basins)."""
        return self._render(kind="image", **kw)

    def bifurcation(self, **kw: Any) -> Any:
        """Plot as a bifurcation / orbit diagram."""
        return self._render(kind="bifurcation", **kw)

    def return_map(self, **kw: Any) -> Any:
        """Plot as a first-return / next-amplitude map."""
        return self._render(kind="return_map", **kw)

    # NOTE (v6, sanctioned public API break): ``.histogram()`` and ``.spectrum()``
    # are gone.  Their data producers — the surrogate null distribution and the
    # power spectrum — left with the generic time-series layer in the v6 scope
    # surgery, so nothing in the library builds either kind.  The methods did not
    # fail: ``_render(kind=...)`` only *relabels* the spec, so the caller got the
    # result's ordinary layers under a histogram/spectrum name.  Verified on a
    # DimensionResult: ``.plot.histogram()`` drew the log C(r) scaling fit, axes
    # and title intact, titled "correlation dimension" — a plot of the wrong
    # thing, presented as the right thing.  A method that silently draws
    # something else is worse than one that is absent, so it is absent.

    def section(self, **kw: Any) -> Any:
        """Plot as a Poincaré section."""
        return self._render(kind="poincare_section", **kw)

    def _render(self, *, kind: str | None = None, backend: str | None = None, **tweaks: Any) -> Any:
        """Resolve a backend and render, or raise :class:`VisualizationNotInstalled`.

        The ``kind`` requested by a typed method routes into ``to_plot_spec`` (so
        the *spec* carries the semantic kind), and rendering goes through the
        documented ``PlotSpec.render(backend, **backend_kw)`` contract — ``kind``
        is never passed to ``render``.

        **Keyword routing.**  ``**tweaks`` is split three ways: a keyword the
        result's own ``to_plot_spec`` declares is forwarded there (this is what
        makes ``OrbitDiagram.plot(annotate=True)`` reachable at all), a keyword in
        :func:`_renderer_kwargs` goes to the renderer, and anything else raises
        :class:`~tsdynamics.errors.InvalidParameterError` naming both accepted
        sets.  Previously *every* keyword went to the renderer, whose ``**_kw``
        catch-all absorbed it — so a spec-shaping parameter was silently dropped
        and a misspelled one was silently ignored.

        Parameters
        ----------
        kind : str, optional
            The semantic plot kind a typed accessor method forces (e.g.
            ``"scaling_fit"``), or ``None`` for the result's natural kind.
        backend : str, optional
            The renderer to use (``"matplotlib"`` / ``"plotly"`` / …); ``None``
            lets :meth:`~tsdynamics.viz.spec.PlotSpec.render` pick the default.
        **tweaks
            Spec-shaping keywords (whatever this result's ``to_plot_spec``
            accepts) and/or backend keywords (:func:`_renderer_kwargs`).

        Returns
        -------
        PlotSpec
            The built spec — **not** a backend figure.  ``plot`` **builds**,
            ``render`` **draws**, ``show`` **displays**, ``save`` **writes**, on
            every plottable object in the library.

            .. versionchanged:: 6.0
                Returned the backend's figure before v6, so
                ``rm.plot().save("a.png")`` was an ``AttributeError`` while
                ``traj.plot().save("a.png")`` worked — the same verb handing back
                two types with disjoint methods (``.savefig`` vs ``.save``).  Use
                ``.render(backend, **backend_kw)`` for the figure.

        Raises
        ------
        VisualizationNotInstalled
            If no rendering backend can be registered, or the result has no
            ``to_plot_spec`` method.
        InvalidParameterError
            If a keyword is recognised by neither ``to_plot_spec`` nor the
            renderer contract.
        """
        renderers = _available_renderers()
        if renderers is None:
            raise VisualizationNotInstalled(_VIZ_HINT)
        to_spec = getattr(self._result, "to_plot_spec", None)
        if to_spec is None:  # pragma: no cover - every result has to_plot_spec
            raise VisualizationNotInstalled(
                f"{type(self._result).__name__} has no to_plot_spec() yet, so it cannot be plotted."
            )
        spec_kw, backend_kw = _split_plot_kwargs(to_spec, tweaks, type(self._result).__name__)
        # A typed method (e.g. .scaling()) requests a kind; pass it to to_plot_spec
        # when that result accepts an override, else fall back to its natural kind.
        try:
            spec = to_spec(kind=kind, **spec_kw) if kind is not None else to_spec(**spec_kw)
        except TypeError:
            if spec_kw:  # a declared keyword must not be silently dropped
                raise
            spec = to_spec()
        # ``plot`` builds.  A caller who named a backend or passed a renderer
        # option is asking for that drawing to happen, so it still happens — but
        # what comes back is the spec, which is what every other ``.plot()`` in
        # the library returns and what ``.save()`` / ``.show()`` hang off.
        if backend is not None or backend_kw:
            spec.render(backend, **backend_kw)
        return spec

    def __repr__(self) -> str:  # noqa: D105
        return f"<plot accessor for {type(self._result).__name__}>"
