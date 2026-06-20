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
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

import numpy as np

__all__ = ["AnalysisResult", "VisualizationNotInstalled"]


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

    def __call__(self, backend: str | None = None, *, kind: str | None = None, **tweaks: Any):
        """Render the result's default view (see :meth:`_render`)."""
        return self._render(kind=kind, backend=backend, **tweaks)

    # -- typed kind methods (the closed plot vocabulary; see viz.PlotKind) --

    def scaling(self, **kw: Any):
        """Plot as a log--log scaling fit (dimensions / Lyapunov-from-data)."""
        return self._render(kind="scaling_fit", **kw)

    def diagnostic(self, **kw: Any):
        """Plot as a diagnostic growth/decay curve (GALI, divergence)."""
        return self._render(kind="diagnostic_curve", **kw)

    def time_series(self, **kw: Any):
        """Plot as a one-dimensional time series."""
        return self._render(kind="time_series", **kw)

    def phase(self, **kw: Any):
        """Plot as a 2-D / 3-D phase portrait."""
        return self._render(kind="phase_portrait", **kw)

    def image(self, **kw: Any):
        """Plot as a 2-D image (recurrence matrix, basins)."""
        return self._render(kind="image", **kw)

    def bifurcation(self, **kw: Any):
        """Plot as a bifurcation / orbit diagram."""
        return self._render(kind="bifurcation", **kw)

    def return_map(self, **kw: Any):
        """Plot as a first-return / next-amplitude map."""
        return self._render(kind="return_map", **kw)

    def histogram(self, **kw: Any):
        """Plot as a null-distribution histogram (surrogate tests)."""
        return self._render(kind="histogram_null", **kw)

    def spectrum(self, **kw: Any):
        """Plot as a power spectrum."""
        return self._render(kind="power_spectrum", **kw)

    def section(self, **kw: Any):
        """Plot as a Poincaré section."""
        return self._render(kind="poincare_section", **kw)

    def _render(self, *, kind: str | None = None, backend: str | None = None, **tweaks: Any):
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
    scalars, mappings, and ``list``/``tuple``/``set`` containers (recursively).
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
