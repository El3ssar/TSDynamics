"""Visualization seam for the analysis result classes.

Holds the plotting machinery split out of ``analysis/_result.py``:
:class:`VisualizationNotInstalled` (the canonical public exception the whole viz
layer raises), the renderer-registry probe ``_available_renderers``, and the
:class:`_PlotAccessor` that backs ``AnalysisResult.plot`` (callable *and* a
namespace of typed kind methods).  Nothing here imports a plotting library at
module scope — every viz import is deferred to call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tsdynamics.analysis._result_base import AnalysisResult

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

    def histogram(self, **kw: Any) -> Any:
        """Plot as a null-distribution histogram (surrogate tests)."""
        return self._render(kind="histogram_null", **kw)

    def spectrum(self, **kw: Any) -> Any:
        """Plot as a power spectrum."""
        return self._render(kind="power_spectrum", **kw)

    def section(self, **kw: Any) -> Any:
        """Plot as a Poincaré section."""
        return self._render(kind="poincare_section", **kw)

    def _render(self, *, kind: str | None = None, backend: str | None = None, **tweaks: Any) -> Any:
        """Resolve a backend and render, or raise :class:`VisualizationNotInstalled`.

        The ``kind`` requested by a typed method routes into ``to_plot_spec`` (so
        the *spec* carries the semantic kind), and rendering goes through the
        documented ``PlotSpec.render(backend, **backend_kw)`` contract — ``kind``
        is never passed to ``render``.

        Parameters
        ----------
        kind : str, optional
            The semantic plot kind a typed accessor method forces (e.g.
            ``"scaling_fit"``), or ``None`` for the result's natural kind.
        backend : str, optional
            The renderer to use (``"matplotlib"`` / ``"plotly"`` / …); ``None``
            lets :meth:`~tsdynamics.viz.spec.PlotSpec.render` pick the default.
        **tweaks
            Backend keyword arguments forwarded verbatim to ``PlotSpec.render``.

        Returns
        -------
        object
            Whatever the chosen backend's ``render`` returns (a Matplotlib
            figure, a Plotly figure, …).

        Raises
        ------
        VisualizationNotInstalled
            If no rendering backend can be registered, or the result has no
            ``to_plot_spec`` method.
        """
        renderers = _available_renderers()
        if renderers is None:
            raise VisualizationNotInstalled(_VIZ_HINT)
        to_spec = getattr(self._result, "to_plot_spec", None)
        if to_spec is None:  # pragma: no cover - every result has to_plot_spec
            raise VisualizationNotInstalled(
                f"{type(self._result).__name__} has no to_plot_spec() yet, so it cannot be plotted."
            )
        # A typed method (e.g. .scaling()) requests a kind; pass it to to_plot_spec
        # when that result accepts an override, else fall back to its natural kind.
        try:
            spec = to_spec(kind=kind) if kind is not None else to_spec()
        except TypeError:
            spec = to_spec()
        if backend is not None:
            return spec.render(backend, **tweaks)
        return spec.render(**tweaks)

    def __repr__(self) -> str:  # noqa: D105
        return f"<plot accessor for {type(self._result).__name__}>"
