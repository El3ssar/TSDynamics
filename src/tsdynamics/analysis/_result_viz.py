"""Visualization seam for the analysis result classes.

Holds the deferred-plotting machinery split out of ``analysis/_result.py``:
:class:`VisualizationNotInstalled` (the canonical public exception the whole viz
layer raises), the defensive renderer-registry probe ``_available_renderers``,
and the :class:`_PlotAccessor` that backs ``AnalysisResult.plot`` (callable *and*
a namespace of typed kind methods).  Nothing here imports a plotting library at
module scope — every viz import is deferred to call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tsdynamics.analysis._result_base import AnalysisResult

_VIZ_HINT = (
    "Visualization is deferred in this release: no rendering backend is "
    "registered. Export the data with .to_dict() / .to_frame() and plot it "
    "yourself, or install a backend once one is available."
)


class VisualizationNotInstalled(ImportError):  # noqa: N818 — canonical v4 public name
    """Raised by ``result.plot`` when no visualization backend is available.

    Subclasses :class:`ImportError` so the (deferred) plotting machinery reads
    like any other optional dependency: ``except ImportError`` catches it.  When
    a renderer registers itself in ``tsdynamics.registry.renderers`` the seam
    starts working with no change to result types.
    """


def _available_renderers() -> Any | None:
    """Return the renderer registry if it exists and is non-empty, else ``None``.

    The renderer registry (``registry.renderers``) and the rendering backends
    are added by later visualization streams.  Resolving it lazily and
    defensively keeps :class:`AnalysisResult` self-contained today while wiring
    itself up automatically the moment a backend lands.
    """
    try:
        from tsdynamics.registry import renderers
    except Exception:  # pragma: no cover - registry has no renderers yet
        return None
    try:
        return renderers if len(renderers) else None
    except Exception:  # pragma: no cover - defensive
        return None


class _PlotAccessor:
    """The ``result.plot`` seam: callable *and* a namespace of typed methods.

    ``result.plot()`` renders the result's default view; the named methods
    (``result.plot.scaling()``, ``.phase()``, …) force a particular semantic
    plot kind.  Every entry point funnels through :meth:`_render`, which raises
    :class:`VisualizationNotInstalled` until a rendering backend is registered.
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
        is never passed to ``render``.  This path only runs once a backend lands
        (the visualization streams own the spec); until then it is unreachable.
        """
        renderers = _available_renderers()
        if renderers is None:
            raise VisualizationNotInstalled(_VIZ_HINT)
        to_spec = getattr(self._result, "to_plot_spec", None)
        if to_spec is None:  # pragma: no cover - wired by the to_plot_spec stream
            raise VisualizationNotInstalled(
                f"{type(self._result).__name__} has no to_plot_spec() yet, so it cannot be plotted."
            )
        # A typed method (e.g. .scaling()) requests a kind; pass it to to_plot_spec
        # when that result accepts an override, else fall back to its natural kind.
        try:  # pragma: no cover - exercised once a backend registers
            spec = to_spec(kind=kind) if kind is not None else to_spec()
        except TypeError:
            spec = to_spec()
        if backend is not None:  # pragma: no cover - exercised once a backend registers
            return spec.render(backend, **tweaks)
        return spec.render(**tweaks)

    def __repr__(self) -> str:  # noqa: D105
        return f"<plot accessor for {type(self._result).__name__}>"
