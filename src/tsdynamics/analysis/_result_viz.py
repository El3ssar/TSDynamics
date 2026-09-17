"""Visualization seam for the analysis result classes.

Holds the plotting machinery split out of ``analysis/_result.py``:
:class:`VisualizationNotInstalled` (the canonical public exception the whole viz
layer raises), the renderer-registry probe ``_available_renderers``, and the
:class:`_PlotAccessor` that backs ``AnalysisResult.plot`` (callable *and* a
namespace of the transform names that admit this result).  Nothing here imports a
plotting library at module scope — every viz import is deferred to call time.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tsdynamics.analysis._result_base import AnalysisResult

#: Keyword arguments that belong to a *rendering backend* rather than to a
#: result's ``__plot_spec__``.  ``result.plot(**kw)`` splits its keywords on this
#: table: a name ``__plot_spec__`` declares goes to the spec builder, a name here
#: goes to the renderer, and **anything else raises** (see :meth:`_PlotAccessor._render`).
#: A keyword that *some* backend accepts but the **chosen** one does not is caught
#: one layer down, by :func:`~tsdynamics.viz.render.render_spec`.
#:
#: Before this table existed, ``.plot()`` called ``__plot_spec__()`` with *no*
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

    A ``__plot_spec__`` with a ``**kwargs`` catch-all is treated as accepting
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


def _style_vocabulary() -> frozenset[str]:
    """Return the canonical per-layer style words plus ``theme`` (lazily resolved)."""
    try:
        from tsdynamics.viz.style import style_names
    except Exception:  # pragma: no cover - viz layer unavailable
        return frozenset({"theme"})
    return frozenset(style_names()) | {"theme"}


def _figure_vocabulary() -> frozenset[str]:
    """Return the figure words every plotting door applies (``title`` / ``xscale`` / …)."""
    try:
        from tsdynamics.viz.spec import FIGURE_KEYS
    except Exception:  # pragma: no cover - viz layer unavailable
        return frozenset()
    return frozenset(FIGURE_KEYS) - {"theme"}


def _split_plot_kwargs(
    to_spec: Any, tweaks: dict[str, Any], owner: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split ``.plot(**tweaks)`` four ways, or raise naming every accepted set.

    Returns ``(spec_kwargs, style_kwargs, figure_kwargs, backend_kwargs)``.

    See :meth:`_PlotAccessor._render` for why this exists.  Precedence is
    deliberate: a keyword the result's ``__plot_spec__`` declares wins, because
    that is where the caller's intent lives (a result that declares ``path=``
    means its own ``path``); then style, then figure, then the renderer table.
    """
    spec_names = _spec_keywords(to_spec)
    takes_any = _accepts_var_keyword(to_spec)
    style_names_ = _style_vocabulary()
    figure_names = _figure_vocabulary()
    renderer_names = _renderer_kwargs()
    spec_kw: dict[str, Any] = {}
    style_kw: dict[str, Any] = {}
    figure_kw: dict[str, Any] = {}
    backend_kw: dict[str, Any] = {}
    unknown: list[str] = []
    for name, value in tweaks.items():
        if name in spec_names:
            spec_kw[name] = value
        elif name in style_names_:
            style_kw[name] = value
        elif name in figure_names:
            figure_kw[name] = value
        elif name in renderer_names:
            backend_kw[name] = value
        elif takes_any:
            spec_kw[name] = value
        else:
            unknown.append(name)
    if unknown:
        from tsdynamics.errors import InvalidParameterError

        accepted = ", ".join(sorted(spec_names)) or "(none)"
        raise InvalidParameterError(
            f"{owner}.plot() got unexpected keyword argument(s) "
            f"{', '.join(repr(u) for u in sorted(unknown))}.\n"
            f"    Spec keywords for this result: {accepted}\n"
            f"    Style keywords: {', '.join(sorted(style_names_))}\n"
            f"    Figure keywords: {', '.join(sorted(figure_names))}\n"
            f"    Legend keywords: labels\n"
            f"    Backend keywords (at render()): {', '.join(sorted(renderer_names))}"
        )
    return spec_kw, style_kw, figure_kw, backend_kw


#: The eight ``kind``-forcing methods v6 removed from :class:`_PlotAccessor`.
#: Named in the ``AttributeError`` so a reader who learned one is told what it
#: actually did, not merely that it is gone.  See the class docstring for the
#: measurement.
_RETIRED_KIND_METHODS: frozenset[str] = frozenset(
    {
        "scaling",
        "diagnostic",
        "time_series",
        "phase",
        "image",
        "bifurcation",
        "return_map",
        "section",
    }
)


def _transform_names(result: Any) -> list[str]:
    """Every registered plot transform whose declared subjects admit ``result``.

    The same question :class:`~tsdynamics.utils.plot_namespace.PlotNamespace`
    asks for a trajectory or a system, asked for a result — so ``.plot.<TAB>``
    means one thing everywhere in the library.
    """
    try:
        from tsdynamics.viz import transforms as _transforms

        found = _transforms.find(subject=result)
    except Exception:  # pragma: no cover - viz layer unavailable
        return []
    return sorted(r if isinstance(r, str) else getattr(r, "name", str(r)) for r in found)


class _PlotAccessor:
    """The ``result.plot`` seam: callable *and* a namespace of transform names.

    ``result.plot()`` draws the result's own view.  ``result.plot.<TAB>`` lists
    the registered transforms that admit **this** result and
    ``result.plot.scaling_fit()`` is exactly ``ts.plot(result, "scaling_fit")`` —
    the same gesture, meaning the same thing, that ``traj.plot`` and
    ``system.plot`` already carry (:class:`~tsdynamics.utils.plot_namespace.PlotNamespace`).

    .. versionchanged:: 6.0
        **The eight kind-forcing methods are gone** (``.scaling()`` /
        ``.diagnostic()`` / ``.time_series()`` / ``.phase()`` / ``.image()`` /
        ``.bifurcation()`` / ``.return_map()`` / ``.section()``).  They did not
        build a different picture — they **relabelled** the result's own spec
        with a different :class:`~tsdynamics.viz.spec.PlotKind`.  Measured over
        the 30 result fixtures: not one of the eight changed a single byte of
        layer data, and each was *truthful* only when the kind it forced already
        was the result's natural kind — a no-op.  ``.time_series()``,
        ``.image()``, ``.bifurcation()`` and ``.section()`` were truthful on
        **zero** results, so
        ``lyapunov_spectrum(lor).plot.section()`` returned a bar chart of
        exponents labelled ``poincare_section``.  This module already deleted
        ``.histogram()`` and ``.spectrum()`` on exactly that reasoning; the
        remaining eight fail the same test.

        The replacement is strictly more capable, because a transform *computes*
        where a relabel only renamed: ``ts.plot(dim, "scaling_fit")`` builds
        **five** layers (the curve, the fitted line, the window markers) where
        ``dim.plot.scaling()`` built one.
    """

    __slots__ = ("_result",)

    def __init__(self, result: AnalysisResult) -> None:
        self._result = result

    def __call__(self, backend: str | None = None, **tweaks: Any) -> Any:
        """Build this result's plot (see :meth:`_render`)."""
        return self._render(backend=backend, **tweaks)

    def _names(self) -> list[str]:
        """Every transform that admits this result (the tab surface)."""
        return _transform_names(self._result)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        import functools

        names = self._names()
        if name in names:
            from tsdynamics.viz import plot as _plot

            bound = functools.partial(_plot, self._result, name)
            bound.__doc__ = f"ts.plot(result, {name!r}, **kwargs) — see ts.viz.compatibility()."
            return bound

        import difflib

        who = type(self._result).__name__
        close = difflib.get_close_matches(name, names, n=2, cutoff=0.55)
        hint = "".join(f"\n    result.plot.{c}()" for c in close)
        extra = ""
        if name in _RETIRED_KIND_METHODS:
            extra = (
                f"\n  ({name!r} forced a plot KIND: it relabelled this result's own spec "
                f"without changing a byte of it, so it drew the wrong name over the right "
                f"picture — or the right name over an unchanged one.)"
            )
        raise AttributeError(
            f"no plot transform named {name!r} for a {who}."
            + extra
            + (f"  Did you mean:{hint}" if close else "")
            + "\n    result.plot()                 # this result's own view"
            + f"\n    result.plot.<TAB>             # the {len(names)} that draw this"
            + "\n    ts.viz.compatibility()        # the whole matrix"
        )

    def __dir__(self) -> list[str]:
        """Tab-completion is the point: the transforms that draw this result."""
        return self._names()

    def _render(self, *, backend: str | None = None, **tweaks: Any) -> Any:
        """Build this result's plot, or raise :class:`VisualizationNotInstalled`.

        **One vocabulary, five doors.**  ``**tweaks`` is split the way
        ``traj.plot`` and ``system.plot`` split theirs: a keyword the result's own
        ``__plot_spec__`` declares goes to the spec builder (this is what makes
        ``OrbitDiagram.plot(annotate=True)`` reachable), the **style** vocabulary
        (:data:`~tsdynamics.viz.style.STYLE_KEYS` and its aliases) plus ``theme``
        is applied to the finished spec, the **figure** vocabulary
        (:data:`~tsdynamics.viz.spec.FIGURE_KEYS` — ``title`` / ``xlabel`` /
        ``xscale`` / …) is applied as an inline tweak, a keyword in
        :func:`_renderer_kwargs` goes to the renderer, and anything else raises
        naming every accepted set.

        .. versionchanged:: 6.0
            The style and figure vocabularies used to raise here while working at
            the other four doors — measured, ``result.plot(color="red")``,
            ``(title=…)``, ``(theme=…)`` and ``(xscale=…)`` all raised
            ``InvalidParameterError`` while ``ts.plot(result, …)`` accepted every
            one.  Style at the call site is the library's rule; a door that is
            the exception is the defect.

        Parameters
        ----------
        backend : str, optional
            The renderer to use (``"matplotlib"`` / ``"plotly"`` / …); ``None``
            lets :meth:`~tsdynamics.viz.spec.PlotSpec.render` pick the default.
        **tweaks
            Spec-shaping keywords (whatever this result's ``__plot_spec__``
            accepts), style keywords, figure keywords, and/or backend keywords.

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
            ``__plot_spec__`` method.
        InvalidParameterError
            If a keyword is recognised by none of the four vocabularies, or if
            ``kind=`` is passed — v6 collapsed the plot ``kind`` into the
            transform name, so the spelling is ``ts.plot(result, "<name>")``.
        """
        from tsdynamics.errors import InvalidParameterError

        if "kind" in tweaks:
            names = self._names()
            offered = ", ".join(f"result.plot.{n}()" for n in names) or "(none for this result)"
            raise InvalidParameterError(
                f"kind= is not how a picture is chosen any more: v6 collapsed the plot "
                f"kind into the TRANSFORM NAME, so there is one spelling to learn and it "
                f"is the same at every door.\n"
                f"    result.plot()                 # this result's own view\n"
                f"    {offered}"
            )
        renderers = _available_renderers()
        if renderers is None:
            raise VisualizationNotInstalled(_VIZ_HINT)
        to_spec = getattr(self._result, "__plot_spec__", None)
        if to_spec is None:  # pragma: no cover - every result has __plot_spec__
            raise VisualizationNotInstalled(
                f"{type(self._result).__name__} has no __plot_spec__() yet, so it cannot be plotted."
            )
        # ``labels=`` names the curve, at every plotting door.  Peeled here rather
        # than in the four-way split because it is applied to the built spec, not
        # handed to any of the four; a result whose own ``__plot_spec__`` declares
        # the word keeps it (that precedence is the split's, and it holds here).
        labels = None
        if "labels" in tweaks and "labels" not in _spec_keywords(to_spec):
            labels = tweaks.pop("labels")
        spec_kw, style_kw, figure_kw, backend_kw = _split_plot_kwargs(
            to_spec, tweaks, type(self._result).__name__
        )
        spec = to_spec(**spec_kw)
        theme = style_kw.pop("theme", None)
        if theme is not None:
            spec.theme(theme)
        if style_kw:
            spec.style(**style_kw)
        if figure_kw:
            spec.tweak(**figure_kw)
        if labels is not None:
            from tsdynamics.viz.compose import apply_labels

            apply_labels([spec], labels)
        # ``plot`` builds.  A caller who named a backend or passed a renderer
        # option is asking for that drawing to happen, so it still happens — but
        # what comes back is the spec, which is what every other ``.plot()`` in
        # the library returns and what ``.save()`` / ``.show()`` hang off.
        if backend is not None or backend_kw:
            spec.render(backend, **backend_kw)
        return spec

    def __repr__(self) -> str:
        """Say what this draws and what to type next."""
        names = self._names()
        who = type(self._result).__name__
        head = f"<plot namespace for {who}: {len(names)} transforms>"
        shown = ", ".join(names) if names else "(none registered for this result)"
        return f"{head}\n    {shown}\n    call it — result.plot() — for this result's own view"
