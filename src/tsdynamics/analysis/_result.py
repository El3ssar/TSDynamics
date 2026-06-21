"""Shared base class for the analysis layer's result objects.

Every analysis in TSDynamics returns a *self-describing result object* rather
than a bare ``float``/``ndarray``/``list``.  :class:`AnalysisResult` is the one
base those results share.  It is purely additive — it adds a uniform surface
without changing any existing return value — so the analysis subpackages can be
reparented onto it one at a time.

The surface every result inherits
---------------------------------
- ``meta`` — a provenance mapping (system, params, version, run settings),
  built from :meth:`tsdynamics.families.base.SystemBase._provenance` at the
  call site via :meth:`AnalysisResult.build_meta`.
- ``__repr__`` — a compact, ``_repr_fields``-driven one-liner.
- ``_repr_html_`` — a small table for Jupyter / IPython.
- :meth:`summary` — a human-readable multi-line readout plus an optional
  interpretation line (subclasses supply the interpretation).
- :meth:`to_dict` — a stdlib, JSON-friendly mapping (arrays become lists).
- :meth:`to_frame` — a :mod:`pandas` ``DataFrame`` (``pandas`` is a soft
  dependency, imported lazily, with an install hint if it is missing).
- ``plot`` — the visualization seam: an accessor that is both callable
  (``result.plot()``) and a namespace of typed kind methods
  (``result.plot.scaling()``).  Visualization ships in a later release, so until
  a rendering backend registers itself the seam raises
  :class:`VisualizationNotInstalled`.

Subclassing contract
--------------------
:class:`AnalysisResult` is a *frozen* dataclass.  A subclass must re-apply the
decorator and stay frozen::

    @dataclass(frozen=True)
    class LyapunovSpectrum(AnalysisResult):
        _repr_fields = ("exponents", "kaplan_yorke")
        exponents: np.ndarray = field(repr=False, compare=False)
        kaplan_yorke: float

        def _interpretation(self) -> str:
            n_pos = int((self.exponents > 0).sum())
            return "chaotic" if n_pos else "regular"

``meta`` is keyword-only on the base, so subclasses are free to declare their own
positional fields without tripping the "non-default argument follows default
argument" rule.

Array-valued fields **must** be declared ``field(compare=False)``.  A frozen
dataclass derives ``__eq__`` / ``__hash__`` from a tuple of its fields, and a
NumPy array is neither boolean-comparable (``arr == arr`` is an array, so a
generated ``__eq__`` would raise ``ValueError``) nor hashable.  Keeping arrays
out of equality and hashing lets two results compare on their scalar summary
fields; ``meta`` (provenance) is already excluded the same way on the base.

The scaling-curve family
------------------------
A large family of estimators reads a single number off the slope of a log--log
(or semi-log) scaling curve: every fractal dimension, the Lyapunov exponent from
a measured time series, expansion entropy, and the Cao / false-nearest-neighbour
embedding-dimension diagnostics.  :class:`ScalingResult` gives that whole family
**one** canonical schema — ``estimate`` / ``stderr`` / ``abscissa`` /
``ordinate`` / ``fit_region`` / ``intercept`` (plus the ``local_slopes`` and
``scaling_window`` diagnostics) — so a single ``result.plot.scaling()`` renders
any of them.  It is additive: existing results are *reparented* onto it by a
later stream, not changed here.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, cast

import numpy as np

__all__ = [
    "AnalysisResult",
    "ArrayResult",
    "CollectionResult",
    "CountResult",
    "ScalarResult",
    "ScalingResult",
    "VisualizationNotInstalled",
]


# ---------------------------------------------------------------------------
# Visualization seam
# ---------------------------------------------------------------------------

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
        """Plot as a 2-D / 3-D phase portrait."""
        return self._render(kind="phase_portrait", **kw)

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


# ---------------------------------------------------------------------------
# JSON / repr helpers
# ---------------------------------------------------------------------------


def _jsonify(obj: Any) -> Any:
    """Coerce ``obj`` into a JSON-serializable form (arrays → lists).

    Handles the value shapes analysis results actually carry: NumPy arrays and
    scalars, mappings, ``list``/``tuple``/``set`` containers (recursively), nested
    result objects (an ``Attractor`` inside an ``AttractorSet``, an ``RQAResult``
    inside a ``WindowedRQA``), SciPy sparse matrices (a recurrence matrix), and the
    plain state-space dataclasses (``Box``/``Ball``/``Grid``) a result may embed.
    Plain Python scalars pass through unchanged; anything else is returned as-is
    (the caller owns exotic payloads).
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonify(v) for v in obj]
    # A nested result object (e.g. an ``Attractor`` inside an ``AttractorSet``):
    # serialize it through its own JSON-friendly ``to_dict`` rather than leaving a
    # non-serializable object behind.  ``int``-backed results (``CountResult`` *is*
    # an ``int``) are excluded so they stay native JSON scalars.
    if isinstance(obj, AnalysisResult) and not isinstance(obj, (int, float, str)):
        return obj.to_dict()
    # A SciPy sparse matrix (the recurrence matrix): emit a COO triplet so it
    # round-trips without densifying to ``O(N^2)``.
    if hasattr(obj, "tocoo") and hasattr(obj, "shape") and hasattr(obj, "nnz"):
        coo = obj.tocoo()
        return {
            "format": "coo",
            "shape": list(coo.shape),
            "row": coo.row.tolist(),
            "col": coo.col.tolist(),
            "data": _jsonify(coo.data),
        }
    # A plain data-carrying dataclass embedded in a result (``Box``/``Ball``/
    # ``Grid``): serialize its fields recursively.  ``AnalysisResult`` is handled
    # above (its curated ``to_dict``); the scalar/class guard keeps int-backed
    # results native and never expands a dataclass *type*.
    if is_dataclass(obj) and not isinstance(obj, (int, float, str, type)):
        return {f.name: _jsonify(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def _is_frame_scalar(value: Any) -> bool:
    """Return whether ``value`` belongs in a single DataFrame cell.

    True for plain scalars and 0-d arrays; False for containers and n-d arrays.
    Tests the type directly rather than via ``numpy.ndim`` so a ragged/mixed
    container (e.g. ``(1, [2, 3])``) is excluded rather than raising when NumPy
    tries to coerce it to an array.
    """
    if isinstance(value, (list, tuple, set, frozenset, Mapping)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return True


def _fmt(value: Any) -> str:
    """Format a single value for the compact :meth:`AnalysisResult.__repr__`."""
    if value is None:
        return repr(value)
    if isinstance(value, (bool, np.bool_)):
        return repr(bool(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _fmt(value.item())
        if value.size <= 4:
            return "[" + ", ".join(_fmt(v) for v in value.ravel().tolist()) + "]"
        return f"array(shape={tuple(value.shape)})"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        if len(value) <= 4:
            inner = ", ".join(_fmt(v) for v in value)
            return f"[{inner}]" if isinstance(value, list) else f"({inner})"
        return f"{type(value).__name__}(len={len(value)})"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisResult:
    """Base class for every analysis result object.

    See the module docstring for the full contract.  Subclasses re-apply
    ``@dataclass(frozen=True)``, declare their own fields, list the ones to show
    in ``__repr__``/:meth:`summary` via the ``_repr_fields`` class attribute, and
    optionally override :meth:`_interpretation`.

    Attributes
    ----------
    meta : Mapping
        Provenance for the computation: the originating system, its parameters,
        the library version, and the run settings.  Keyword-only; defaults to an
        empty dict.  Build it with :meth:`build_meta`.
    """

    #: Names of the fields shown in ``__repr__`` / :meth:`summary` / the HTML
    #: table.  When empty the dataclass fields are used (skipping ``meta`` and
    #: any declared with ``field(repr=False)``).
    _repr_fields: ClassVar[tuple[str, ...]] = ()

    #: ``meta`` is provenance, not identity: ``compare=False`` keeps it out of the
    #: generated ``__eq__`` / ``__hash__`` (a dict is unhashable, and two otherwise
    #: identical results should compare equal even with differing run provenance),
    #: while it stays in :func:`dataclasses.fields` so :meth:`to_dict` still emits it.
    meta: Mapping[str, Any] = field(default_factory=dict, kw_only=True, repr=False, compare=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Keep the inherited ``__repr__`` on every subclass.

        ``@dataclass`` regenerates ``__repr__`` for each subclass (shadowing the
        inherited one), which would drop ``_repr_fields`` formatting.  This hook
        runs *before* the decorator, so claiming the MRO-resolved ``__repr__`` in
        the subclass's own namespace makes ``dataclass`` leave it alone.  A
        subclass keeps the nearest ancestor's repr — the base ``_repr_fields``
        one for the common case, or a custom repr a parent defined (so a
        grandchild inherits it too) — unless it writes its own.
        """
        super().__init_subclass__(**kwargs)
        if "__repr__" not in cls.__dict__:
            inherited_repr = cls.__repr__
            cls.__repr__ = inherited_repr  # type: ignore[method-assign]

    # -- provenance -------------------------------------------------------

    @staticmethod
    def build_meta(system: Any = None, **extra: Any) -> dict[str, Any]:
        """Build a provenance dict for a result, from a system plus run settings.

        Delegates to ``system._provenance(**extra)`` when ``system`` exposes it
        (every :class:`~tsdynamics.families.base.SystemBase`), so a result's
        ``meta`` matches the provenance attached to trajectories.  Falls back to
        a minimal ``{"system": ..., **extra}`` for plain objects or ``None``.

        Parameters
        ----------
        system : object, optional
            The system the analysis ran on.
        **extra
            Additional run settings to record (e.g. ``transient=50``).

        Returns
        -------
        dict
        """
        provenance = getattr(system, "_provenance", None)
        if callable(provenance):
            return dict(provenance(**extra))
        if system is None:
            return dict(extra)
        name = getattr(type(system), "__name__", str(system))
        return {"system": name, **extra}

    # -- repr / summary ---------------------------------------------------

    def _display_fields(self) -> tuple[str, ...]:
        """Resolve the field names to display (``_repr_fields`` or introspection)."""
        if self._repr_fields:
            return self._repr_fields
        return tuple(f.name for f in fields(self) if f.repr and f.name != "meta")

    def __repr__(self) -> str:  # noqa: D105
        parts = []
        for name in self._display_fields():
            try:
                value = getattr(self, name)
            except AttributeError:
                continue
            parts.append(f"{name}={_fmt(value)}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def _interpretation(self) -> str | None:
        """Return a one-line human interpretation, or ``None`` for none.

        Overridden by subclasses (e.g. "chaotic: 1 positive exponent").  The
        base returns ``None`` so :meth:`summary` simply omits the line.
        """
        return None

    def _system_label(self) -> str | None:
        """Return the originating system's name from ``meta``, if recorded."""
        system = self.meta.get("system") if self.meta else None
        return str(system) if system else None

    def summary(self) -> str:
        """Return a human-readable multi-line readout of the result.

        The header names the result type (and the originating system, when
        recorded in ``meta``); the body lists the display fields; an optional
        trailing ``→`` line carries the subclass's interpretation.

        Returns
        -------
        str
        """
        label = self._system_label()
        header = type(self).__name__ + (f"  ({label})" if label else "")
        lines = [header]
        for name in self._display_fields():
            try:
                value = getattr(self, name)
            except AttributeError:
                continue
            lines.append(f"  {name} = {_fmt(value)}")
        interpretation = self._interpretation()
        if interpretation:
            lines.append(f"  → {interpretation}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return a small HTML table for Jupyter / IPython display."""
        label = self._system_label()
        caption = html.escape(type(self).__name__ + (f" ({label})" if label else ""))
        rows = []
        for name in self._display_fields():
            try:
                value = getattr(self, name)
            except AttributeError:
                continue
            rows.append(
                f"<tr><th style='text-align:left;padding-right:1em'>{html.escape(name)}</th>"
                f"<td style='text-align:left'>{html.escape(_fmt(value))}</td></tr>"
            )
        interpretation = self._interpretation()
        footer = (
            f"<tr><td colspan='2' style='padding-top:0.4em'><em>{html.escape(interpretation)}"
            "</em></td></tr>"
            if interpretation
            else ""
        )
        return (
            "<table>"
            f"<caption style='text-align:left;font-weight:bold'>{caption}</caption>"
            f"{''.join(rows)}{footer}</table>"
        )

    # -- export -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of every field (arrays become lists).

        Uses only the standard library.  Every dataclass field is included
        (``meta`` too), recursively coerced to JSON-serializable types.

        Returns
        -------
        dict
        """
        return {f.name: _jsonify(getattr(self, f.name)) for f in fields(self)}

    @staticmethod
    def _require_pandas() -> Any:
        """Import :mod:`pandas` lazily, raising a friendly hint if it is absent."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_frame() needs pandas, which is an optional dependency. "
                "Install it with `pip install tsdynamics[frame]` (or "
                "`pip install pandas`). Use .to_dict() for a stdlib-only export."
            ) from exc
        return pd

    def to_frame(self) -> Any:
        """Return a :class:`pandas.DataFrame` view of the result.

        ``pandas`` is a soft dependency, imported lazily; a missing install
        raises an :class:`ImportError` naming the ``tsdynamics[frame]`` extra.

        The base produces a single-row frame of the scalar display fields, with
        ``meta`` carried on ``frame.attrs["meta"]``.  Subclasses that carry a
        natural table (e.g. a scaling curve) override this with a tidy,
        column-per-array frame.

        Returns
        -------
        pandas.DataFrame
        """
        pd = self._require_pandas()
        row: dict[str, Any] = {}
        for name in self._display_fields():
            try:
                value = getattr(self, name)
            except AttributeError:
                continue
            if _is_frame_scalar(value):
                row[name] = _jsonify(value)
        frame = pd.DataFrame([row]) if row else pd.DataFrame()
        frame.attrs["meta"] = dict(self.meta) if self.meta else {}
        return frame

    # -- visualization seam ----------------------------------------------

    @property
    def plot(self) -> _PlotAccessor:
        """The visualization seam (callable + typed kind methods).

        ``result.plot()`` renders the default view; ``result.plot.scaling()``
        and the sibling methods force a particular plot kind.  Raises
        :class:`VisualizationNotInstalled` until a rendering backend registers
        itself, since visualization is deferred in this release.
        """
        return _PlotAccessor(self)


# ---------------------------------------------------------------------------
# ScalingResult — the canonical scaling-curve schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingResult(AnalysisResult):
    r"""A quantity read off the slope of a scaling curve, with that curve.

    Many estimators in TSDynamics share one shape: build a scaling curve, fit a
    straight line over its linear (scaling) region, and report the slope.  Every
    fractal dimension (slope of :math:`\log C(r)` against :math:`\log r`), the
    maximal Lyapunov exponent from a measured series (slope of the mean
    log-divergence against time), expansion entropy, and the Cao /
    false-nearest-neighbour embedding diagnostics all fit this mould.

    :class:`ScalingResult` is the *one* schema for that whole family, so a single
    generic ``result.plot.scaling()`` renders any of them and any consumer can
    find "the curve" and "the fit" without knowing which estimator produced it.

    The number is :attr:`estimate`; ``float(result)`` returns it, so a
    :class:`ScalingResult` drops straight into arithmetic and comparisons.

    Attributes
    ----------
    estimate : float
        The estimated quantity — the fitted slope of the scaling region (a
        dimension, a Lyapunov exponent, an entropy, …).  ``float(self)`` returns
        this value.
    stderr : float
        Standard error of :attr:`estimate` from the line fit.
    abscissa : numpy.ndarray
        The horizontal scaling coordinate at every point of the curve (e.g.
        :math:`\log r`, or time).  Same length as :attr:`ordinate`.
    ordinate : numpy.ndarray
        The vertical coordinate at every point of the curve (e.g.
        :math:`\log C(r)`, or mean log-divergence).
    fit_region : tuple of int
        The ``(lo, hi)`` inclusive index bounds (into :attr:`abscissa` /
        :attr:`ordinate`) of the scaling region the line was fitted over.
    intercept : float
        Intercept of the fitted line, so ``ordinate ≈ intercept + estimate *
        abscissa`` over the fit region — what a renderer draws the fit line from.

    Notes
    -----
    Construct one with the canonical names::

        ScalingResult(
            estimate=2.05, stderr=0.03,
            abscissa=log_r, ordinate=log_C,
            fit_region=(8, 24), intercept=-1.2,
            meta=AnalysisResult.build_meta(system, ...),
        )

    The two curve arrays are declared ``field(compare=False)`` per the
    :class:`AnalysisResult` subclassing contract (a frozen dataclass derives
    ``__eq__`` / ``__hash__`` from its fields, and NumPy arrays are neither
    boolean-comparable nor hashable), so two scaling results compare on their
    scalar summary fields.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("estimate", "stderr")

    estimate: float = 0.0
    stderr: float = 0.0
    abscissa: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    ordinate: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    fit_region: tuple[int, int] = (0, 0)
    intercept: float = 0.0

    # -- the value -------------------------------------------------------

    def __float__(self) -> float:
        """Return :attr:`estimate`, so the result drops into arithmetic."""
        return float(self.estimate)

    # -- scaling diagnostics --------------------------------------------

    @property
    def local_slopes(self) -> np.ndarray:
        r"""Pointwise local slope ``d(ordinate)/d(abscissa)`` of the curve.

        Centered differences (one-sided at the ends, via
        :func:`numpy.gradient`), so non-uniform spacing is handled correctly.
        The plateau of this curve *is* the scaling region; inspecting it is the
        standard sanity check on any reported scaling estimate.  Returns an
        all-``nan`` array of the same shape when there are fewer than two points.

        Returns
        -------
        numpy.ndarray
            Local slope at every point of the curve.
        """
        x = np.asarray(self.abscissa, dtype=float)
        y = np.asarray(self.ordinate, dtype=float)
        if x.size < 2:
            return np.full(x.shape, np.nan)
        return cast("np.ndarray", np.gradient(y, x))

    @property
    def scaling_window(self) -> tuple[float, float]:
        """Return the abscissa span ``(lo, hi)`` the fit was taken over.

        The :attr:`abscissa` values at the two endpoints of :attr:`fit_region`
        — the actual coordinate window of the scaling region (not the index
        bounds).

        Returns
        -------
        tuple of float
            ``(abscissa[lo], abscissa[hi])``.
        """
        lo, hi = self.fit_region
        x = np.asarray(self.abscissa, dtype=float)
        return float(x[lo]), float(x[hi])

    def _interpretation(self) -> str | None:
        """Report the estimate with its scaling-window width and point count."""
        lo, hi = self.fit_region
        n_fit = hi - lo + 1
        return f"estimate = {float(self.estimate):.4g} ± {float(self.stderr):.2g}  (fit over {n_fit} points)"

    # -- visualization ---------------------------------------------------

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe this scaling result as a backend-agnostic :class:`PlotSpec`.

        Builds a ``SCALING_FIT`` spec — the curve as a scatter layer, the fitted
        scaling region highlighted, and the fit line drawn from
        :attr:`intercept` and :attr:`estimate` — so any registered backend can
        render it identically.  The :mod:`tsdynamics.viz.spec` import is lazy, so
        building a result (or importing :mod:`tsdynamics`) never pulls a plot
        library; the spec itself carries no rendering code.

        Parameters
        ----------
        kind : str, optional
            An override for the semantic spec kind (the closed
            :class:`~tsdynamics.viz.spec.PlotKind` vocabulary).  ``None`` (the
            default) uses ``SCALING_FIT``; the ``.plot.scaling()`` seam passes
            ``"scaling_fit"`` explicitly, which resolves to the same kind.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.SCALING_FIT
        x = np.asarray(self.abscissa, dtype=float)
        y = np.asarray(self.ordinate, dtype=float)
        lo, hi = self.fit_region

        layers = [
            Layer(PlotKind.SCATTER, {"x": x, "y": y}, label="curve"),
        ]
        if x.size and hi >= lo:
            layers.append(
                Layer(
                    PlotKind.MARKERS,
                    {"x": x[lo : hi + 1], "y": y[lo : hi + 1]},
                    label="fit region",
                )
            )
            fit_x = np.array([x[lo], x[hi]], dtype=float)
            layers.append(
                Layer(
                    PlotKind.LINE,
                    {"x": fit_x, "y": self.intercept + self.estimate * fit_x},
                    label=f"slope = {self.estimate:.3g}",
                )
            )

        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=type(self).__name__,
            x=Axis(label="abscissa"),
            y=Axis(label="ordinate"),
            layers=layers,
            meta=dict(self.meta) if self.meta else {},
        )


# ---------------------------------------------------------------------------
# ScalarResult — a single number that behaves like the number
# ---------------------------------------------------------------------------


def _coerce_float(value: Any) -> float:
    """Return ``float(value)`` or raise — the gate for numeric dunder forwarding."""
    return float(value)


class _NumericOps:
    """Mixin giving a result the full numeric protocol of ``float(self)``.

    A :class:`ScalarResult` wraps a bare ``float``/``int`` return so it can carry
    ``.meta`` and the result surface, but it must stay a *drop-in* for the number
    it replaces: ``result > 0.9``, ``result == pytest.approx(x)``, ``abs(result)``
    and ``2 * result`` all have to keep working without callers unwrapping it.

    Dunder methods are resolved on the *type*, never via ``__getattr__``, so the
    operators are spelled out here.  Each forwards to ``float(self)`` and coerces
    the other operand with :func:`_coerce_float`; an operand that is not
    float-convertible (an ``ndarray``, a ``pytest.approx`` sentinel) yields
    :data:`NotImplemented` so Python falls back to *its* reflected operator — which
    is exactly how ``result == pytest.approx(x)`` resolves (approx then calls
    ``float(self)`` itself).
    """

    def __float__(self) -> float:  # noqa: D105
        return _coerce_float(self.value)  # type: ignore[attr-defined]

    def __int__(self) -> int:  # noqa: D105
        return int(_coerce_float(self.value))  # type: ignore[attr-defined]

    def __bool__(self) -> bool:  # noqa: D105
        return bool(_coerce_float(self.value))  # type: ignore[attr-defined]

    def __round__(self, ndigits: int | None = None) -> float | int:  # noqa: D105
        return round(float(self), ndigits) if ndigits is not None else round(float(self))

    def __array__(self, dtype: Any = None) -> np.ndarray:  # noqa: D105
        arr = np.asarray(float(self))
        return arr.astype(dtype) if dtype is not None else arr

    # -- comparisons (NotImplemented → reflected op, e.g. pytest.approx) ------

    def __eq__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) == _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __ne__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) != _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __lt__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) < _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __le__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) <= _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __gt__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) > _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __ge__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) >= _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __hash__(self) -> int:  # noqa: D105
        return hash(float(self))

    # -- arithmetic ----------------------------------------------------------

    def __add__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) + _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) - _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __rsub__(self, other: Any) -> Any:  # noqa: D105
        try:
            return _coerce_float(other) - float(self)
        except (TypeError, ValueError):
            return NotImplemented

    def __mul__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) * _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) / _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __rtruediv__(self, other: Any) -> Any:  # noqa: D105
        try:
            return _coerce_float(other) / float(self)
        except (TypeError, ValueError):
            return NotImplemented

    def __neg__(self) -> float:  # noqa: D105
        return -float(self)

    def __pos__(self) -> float:  # noqa: D105
        return +float(self)

    def __abs__(self) -> float:  # noqa: D105
        return abs(float(self))


@dataclass(frozen=True, eq=False)
class ScalarResult(_NumericOps, AnalysisResult):
    """A single scalar measurement that still behaves like its number.

    Wraps a bare ``float`` return (a maximal Lyapunov exponent, an entropy, a
    0--1-test ``K``, …) so it carries the :class:`AnalysisResult` surface —
    ``.meta``, ``.summary()``, ``.to_dict()``, the ``.plot`` seam — while
    ``float(result)`` and every comparison / arithmetic operator keep working via
    :class:`_NumericOps`, so it is a drop-in for the value it replaces.

    Subclasses may add domain context fields (e.g. ``base``, ``normalized``);
    they must re-apply ``@dataclass(frozen=True, eq=False)`` so the numeric
    ``__eq__`` / ``__hash__`` are not regenerated by the dataclass machinery.

    Attributes
    ----------
    value : float
        The measured number.  ``float(result)`` returns it.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("value",)

    value: float = 0.0

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe the scalar as a one-point :class:`PlotSpec` (rarely plotted).

        A lone number has no natural figure; this emits a minimal
        ``DIAGNOSTIC_CURVE`` carrying the value so the ``.plot`` seam resolves
        uniformly.  The :mod:`tsdynamics.viz.spec` import is lazy.
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.DIAGNOSTIC_CURVE
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=type(self).__name__,
            x=Axis(label="index"),
            y=Axis(label="value"),
            layers=[Layer(PlotKind.MARKERS, {"x": np.array([0.0]), "y": np.array([float(self)])})],
            meta=dict(self.meta) if self.meta else {},
        )


class CountResult(int, AnalysisResult):
    """A scalar *integer* result that genuinely **is** an ``int``.

    Subclasses ``int`` (rather than wrapping one) so a count read off the data —
    an estimated delay from ``optimal_delay``, a dimension — is a complete drop-in
    for the bare integer it replaces: ``isinstance(result, int)`` holds, it indexes
    and slices arrays, it survives ``delay=result`` round-trips into estimators
    that type-check their arguments, and all integer arithmetic / comparisons work
    natively.  It *also* carries the :class:`AnalysisResult` surface — ``.meta`` /
    ``.summary()`` / ``.to_dict()`` / the ``.plot`` seam.

    Attributes
    ----------
    value : int
        The measured count (an alias for the integer itself).
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("value",)

    def __new__(cls, value: Any = 0, *, meta: Mapping[str, Any] | None = None) -> CountResult:
        """Construct the integer (``int.__new__``); ``meta`` is set in ``__init__``."""
        return super().__new__(cls, int(value))

    def __init__(self, value: Any = 0, *, meta: Mapping[str, Any] | None = None) -> None:
        """Attach provenance; ``AnalysisResult`` is frozen, so set it via ``object``."""
        object.__setattr__(self, "meta", dict(meta) if meta else {})

    @property
    def value(self) -> int:
        """The measured count (the integer value itself)."""
        return int(self)

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}({int(self)})"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the value and provenance."""
        return {"value": int(self), "meta": _jsonify(self.meta)}

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe the count as a one-point :class:`PlotSpec` (rarely plotted)."""
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.DIAGNOSTIC_CURVE
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=type(self).__name__,
            x=Axis(label="index"),
            y=Axis(label="value"),
            layers=[Layer(PlotKind.MARKERS, {"x": np.array([0.0]), "y": np.array([float(self)])})],
            meta=dict(self.meta) if self.meta else {},
        )


# ---------------------------------------------------------------------------
# ArrayResult — an ndarray-valued result that behaves like the array
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ArrayResult(AnalysisResult):
    """An array-valued measurement that stays a drop-in for its ``ndarray``.

    Wraps a bare ``ndarray`` return (a Lyapunov spectrum, a delay-embedded point
    cloud, a mutual-information-vs-lag curve, a surrogate ensemble) so it carries
    the result surface while ``np.asarray(result)``, indexing/slicing, ``len`` /
    iteration, elementwise comparisons (``result >= 0``) and attribute access
    (``result.shape``, ``result.max()``) all defer to the underlying array.

    Indexing and slicing return the *raw* array element/sub-array (never another
    wrapper), so ``result[:, 0]`` flows straight into NumPy as before.  Operators
    are resolved on the type (``__getattr__`` cannot intercept them), so the
    comparison/arithmetic dunders are spelled out and return raw arrays.

    Attributes
    ----------
    values : numpy.ndarray
        The wrapped array.  ``np.asarray(result)`` returns it.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ()

    values: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}({_fmt(np.asarray(self.values))})"

    # -- ndarray protocol ----------------------------------------------------

    def __array__(self, dtype: Any = None) -> np.ndarray:  # noqa: D105
        arr = np.asarray(self.values)
        return arr.astype(dtype) if dtype is not None else arr

    def __getitem__(self, key: Any) -> Any:  # noqa: D105
        return self.values[key]

    def __len__(self) -> int:  # noqa: D105
        return len(self.values)

    def __iter__(self) -> Any:  # noqa: D105
        return iter(self.values)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown public attributes to the wrapped array (``.shape``, ``.max``)."""
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            values = object.__getattribute__(self, "values")
        except AttributeError as exc:  # pragma: no cover - during unpickling
            raise AttributeError(name) from exc
        return getattr(values, name)

    # -- elementwise comparisons / arithmetic (return raw arrays) ------------

    def __eq__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) == other

    def __ne__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) != other

    def __lt__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) < other

    def __le__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) <= other

    def __gt__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) > other

    def __ge__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) >= other

    __hash__ = None  # type: ignore[assignment]  # array-valued → unhashable, like ndarray

    def __add__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) + other

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) - other

    def __mul__(self, other: Any) -> Any:  # noqa: D105
        return np.asarray(self) * other

    __rmul__ = __mul__

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping (the array as a nested list + ``meta``)."""
        return {"values": _jsonify(self.values), "meta": _jsonify(self.meta)}


# ---------------------------------------------------------------------------
# CollectionResult — a sequence of sub-results (fixed points, periodic orbits)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CollectionResult(AnalysisResult):
    """A homogeneous collection of result items that behaves like a ``list``.

    Wraps a bare ``list`` return (``fixed_points`` → fixed points,
    ``periodic_orbits`` → orbits) so it carries the result surface while
    ``for item in result``, ``result[0]`` and ``len(result)`` keep working.
    Indexing with an ``int`` returns the item; slicing returns a plain ``list``
    of items, matching list semantics.

    Subclasses add domain selectors (``.stable`` / ``.unstable``) and a tidy
    :meth:`to_frame`.

    Attributes
    ----------
    items : tuple
        The collected result items, in order.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ()

    items: tuple[Any, ...] = ()

    def __iter__(self) -> Any:  # noqa: D105
        return iter(self.items)

    def __len__(self) -> int:  # noqa: D105
        return len(self.items)

    def __getitem__(self, key: Any) -> Any:  # noqa: D105
        if isinstance(key, slice):
            return list(self.items[key])
        return self.items[key]

    def __bool__(self) -> bool:  # noqa: D105
        return bool(self.items)

    def __eq__(self, other: Any) -> Any:
        """Compare element-wise — also equal to a plain ``list``/``tuple`` of items.

        Keeps ``result == [...]`` working for callers that treated the old bare
        ``list`` return as a list (e.g. ``tipping_points(...) == []``).
        """
        if isinstance(other, CollectionResult):
            return list(self.items) == list(other.items)
        if isinstance(other, (list, tuple)):
            return list(self.items) == list(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]  # mutable-sequence-like → unhashable, like list

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}({len(self.items)} items)"

    def summary(self) -> str:
        """Return a header naming the collection size, then each item's repr."""
        label = self._system_label()
        header = f"{type(self).__name__} — {len(self.items)} items" + (
            f"  ({label})" if label else ""
        )
        return "\n".join([header, *(f"  {item!r}" for item in self.items)])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping: each item's ``to_dict`` (or repr) + ``meta``."""
        items = [
            item.to_dict() if hasattr(item, "to_dict") else _jsonify(item) for item in self.items
        ]
        return {"items": items, "meta": _jsonify(self.meta)}

    def to_frame(self) -> Any:
        """Return a :class:`pandas.DataFrame` with one row per item.

        Each row carries that item's scalar display fields (when the item is an
        :class:`AnalysisResult`), so a fixed-point / orbit set tabulates cleanly;
        ``meta`` rides on ``frame.attrs["meta"]``.  ``pandas`` is a soft
        dependency, imported lazily.

        Returns
        -------
        pandas.DataFrame
        """
        pd = self._require_pandas()
        rows: list[dict[str, Any]] = []
        for item in self.items:
            display = getattr(item, "_display_fields", None)
            if callable(display):
                row: dict[str, Any] = {}
                for name in display():
                    try:
                        value = getattr(item, name)
                    except AttributeError:
                        continue
                    if _is_frame_scalar(value):
                        row[name] = _jsonify(value)
                rows.append(row)
            else:
                rows.append({"value": _jsonify(item)})
        frame = pd.DataFrame(rows)
        frame.attrs["meta"] = dict(self.meta) if self.meta else {}
        return frame


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
