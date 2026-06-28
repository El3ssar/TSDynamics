"""The :class:`AnalysisResult` base class.

Split out of ``analysis/_result.py``; see that module's docstring (now the
re-exporting facade) for the full result-layer contract.  The repr/JSON helpers
live in :mod:`tsdynamics.analysis._result_json` and the ``.plot`` seam in
:mod:`tsdynamics.analysis._result_viz`; this module holds only the base class.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from tsdynamics._result_common import resolve_plot_kind
from tsdynamics.analysis._result_json import _fmt, _is_frame_scalar, _jsonify
from tsdynamics.analysis._result_viz import VisualizationNotInstalled, _PlotAccessor

if TYPE_CHECKING:
    from tsdynamics.viz.spec import PlotSpec


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
        natural table (e.g. a scaling curve, a wrapped array) override this with a
        tidy, column-per-array frame.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ImportError
            If :mod:`pandas` is not installed (the message names the
            ``tsdynamics[frame]`` extra).
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

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe this result as a backend-agnostic :class:`PlotSpec` (generic fallback).

        Result types with a *natural* figure override this with a bespoke spec
        (a scaling fit, a recurrence image, a phase portrait, …).  This base
        provides the **fallback** every other result inherits, so the ``.plot``
        seam and any registered renderer resolve uniformly across the whole
        analysis layer rather than tripping over a result that never grew a
        bespoke method.

        The fallback inspects the result's display fields (see
        :meth:`_display_fields`) and builds a ``DIAGNOSTIC_CURVE``:

        - the first pair of equal-length 1-D numeric arrays becomes an ``(x, y)``
          ``LINE`` (e.g. a basin-metric's :math:`f(\\varepsilon)` curve);
        - failing that, the first 1-D numeric array becomes a ``LINE`` against
          its index;
        - failing that, the numeric scalar fields become ``MARKERS`` at one point
          per field (e.g. a basin entropy's :math:`S_b` / :math:`S_{bb}`).

        The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never
        pulls a plotting library and the spec itself carries no rendering code.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (a :class:`~tsdynamics.viz.spec.PlotKind`
            value).  ``None`` uses ``DIAGNOSTIC_CURVE``.

        Returns
        -------
        PlotSpec

        See Also
        --------
        _overlay_on : the ``base=`` overlay convention a result with a host view
            (fixed points over a phase portrait) uses in its own ``to_plot_spec``.

        Raises
        ------
        VisualizationNotInstalled
            If the result carries **no** plottable numeric field (no array and no
            scalar) — there is nothing for a generic fallback to draw, so a
            bespoke ``to_plot_spec`` is required.  Subclasses with such a result
            override this method.
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = resolve_plot_kind(kind, PlotKind.DIAGNOSTIC_CURVE)
        arrays, scalars = self._plottable_fields()

        layers: list[Layer] = []
        xlabel, ylabel = "index", "value"
        if len(arrays) >= 2 and arrays[0][1].size == arrays[1][1].size:
            (xname, x), (yname, y) = arrays[0], arrays[1]
            layers.append(Layer(PlotKind.LINE, {"x": x, "y": y}, label=yname))
            xlabel, ylabel = xname, yname
        elif arrays:
            yname, y = arrays[0]
            layers.append(
                Layer(PlotKind.LINE, {"x": np.arange(y.size, dtype=float), "y": y}, label=yname)
            )
            ylabel = yname
        elif scalars:
            names = [n for n, _ in scalars]
            vals = np.array([v for _, v in scalars], dtype=float)
            layers.append(
                Layer(PlotKind.MARKERS, {"x": np.arange(vals.size, dtype=float), "y": vals})
            )
            return PlotSpec(
                kind=spec_kind,
                ndim=2,
                title=type(self).__name__,
                x=Axis(label="field", ticks=list(range(len(names)))),
                y=Axis(label="value"),
                layers=layers,
                meta=dict(self.meta) if self.meta else {},
            )
        else:
            raise VisualizationNotInstalled(
                f"{type(self).__name__} carries no plottable numeric field, so the generic "
                "to_plot_spec() fallback has nothing to draw. A result of this kind needs a "
                "bespoke to_plot_spec(); export it with .to_dict() meanwhile."
            )

        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=type(self).__name__,
            x=Axis(label=xlabel),
            y=Axis(label=ylabel),
            layers=layers,
            meta=dict(self.meta) if self.meta else {},
        )

    def overlay_on(self, base: PlotSpec, *, kind: str | None = None) -> PlotSpec:
        """Overlay this result's figure onto a host ``base`` spec (host drawn first).

        The ``base=`` overlay convention, as a method that does not perturb the
        uniform ``to_plot_spec(self, kind=None)`` signature: build this result's
        spec and append its layers / annotations *after* the host's, so e.g.
        fixed-point markers land over a phase portrait or an attractor scatter
        over a basin image.  The merged ``base`` is mutated and returned.

        Parameters
        ----------
        base : PlotSpec
            The host spec to draw under this result.
        kind : str, optional
            Forwarded to :meth:`to_plot_spec`.

        Returns
        -------
        PlotSpec
            ``base``, with this result's layers / annotations appended.
        """
        return self._overlay_on(self.to_plot_spec(kind=kind), base)

    @staticmethod
    def _overlay_on(spec: PlotSpec, base: PlotSpec | None) -> PlotSpec:
        """Overlay ``spec``'s layers/annotations onto ``base`` (host first), or pass through.

        The ``base=`` overlay convention: when a host ``base`` spec is given, its
        layers are drawn first and ``spec``'s layers/annotations are appended, so
        e.g. fixed-point markers land *over* a phase portrait.  Returns ``spec``
        unchanged when ``base`` is ``None``.
        """
        if base is None:
            return spec
        base.layers = list(base.layers) + list(spec.layers)
        base.annotations = list(base.annotations) + list(spec.annotations)
        if base.legend is None and len(base.layers) > 1:
            from tsdynamics.viz.spec import Legend

            base.legend = Legend()
        return base

    def _plottable_fields(
        self,
    ) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, float]]]:
        """Split the display fields into ``(1-D numeric arrays, numeric scalars)``.

        Used by the generic :meth:`to_plot_spec` fallback.  Booleans are treated
        as scalars (``0`` / ``1``); non-numeric and higher-dimensional fields are
        skipped.  ``meta`` is never included (it is provenance, not plot data).
        """
        arrays: list[tuple[str, np.ndarray]] = []
        scalars: list[tuple[str, float]] = []
        for name in self._display_fields():
            try:
                value = getattr(self, name)
            except AttributeError:
                continue
            if isinstance(value, (bool, np.bool_)):
                scalars.append((name, float(value)))
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                scalars.append((name, float(value)))
                continue
            if (
                isinstance(value, np.ndarray)
                and value.ndim == 1
                and value.size
                and np.issubdtype(value.dtype, np.number)
            ):
                arrays.append((name, np.asarray(value, dtype=float)))
        return arrays, scalars

    @property
    def plot(self) -> _PlotAccessor:
        """The visualization seam (callable + typed kind methods).

        ``result.plot()`` renders the default view; ``result.plot.scaling()``
        and the sibling methods force a particular plot kind.  The in-tree
        backends seed themselves on first use, so it works out of the box when a
        plotting library is installed; with none it raises
        :class:`VisualizationNotInstalled`.

        Returns
        -------
        _PlotAccessor
            A callable that is also a namespace of typed kind methods.
        """
        return _PlotAccessor(self)
