"""The :class:`ArrayResult` ndarray-valued result.

Split out of ``analysis/_result.py``; wraps a bare ``ndarray`` return so it
stays a drop-in for its array while carrying the result surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from tsdynamics._result_common import resolve_plot_kind
from tsdynamics.analysis._result_base import AnalysisResult
from tsdynamics.analysis._result_json import _fmt, _jsonify
from tsdynamics.analysis._result_viz import VisualizationNotInstalled


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

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe the wrapped array as a :class:`PlotSpec` (safe generic view).

        A 1-D array becomes a ``DIAGNOSTIC_CURVE`` ``LINE`` against its index; a
        2-column / 3-column array becomes a phase-portrait point cloud (2-D
        ``SCATTER`` / 3-D ``LINE3D``); any other 2-D array becomes a
        ``LINE_FAMILY`` (one line per column).  Subclasses with a natural figure
        (a Lyapunov spectrum, a mutual-information curve) override this.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` picks the natural kind above.

        Raises
        ------
        VisualizationNotInstalled
            If the array is empty or higher-than-2-D (nothing generic to draw).
        """
        from tsdynamics.viz.spec import Axis, Layer, Legend, PlotKind, PlotSpec

        arr = np.asarray(self.values, dtype=float)
        title = type(self).__name__
        meta = dict(self.meta) if self.meta else {}

        if arr.ndim == 1 and arr.size:
            spec_kind = resolve_plot_kind(kind, PlotKind.DIAGNOSTIC_CURVE)
            layer = Layer(PlotKind.LINE, {"x": np.arange(arr.size, dtype=float), "y": arr})
            return PlotSpec(
                kind=spec_kind,
                ndim=2,
                title=title,
                x=Axis(label="index"),
                y=Axis(label="value"),
                layers=[layer],
                meta=meta,
            )

        if arr.ndim == 2 and arr.shape[0] and arr.shape[1] in (2, 3):
            if arr.shape[1] == 3:
                spec_kind = resolve_plot_kind(kind, PlotKind.PHASE_PORTRAIT_3D)
                layer = Layer(PlotKind.LINE3D, {"x": arr[:, 0], "y": arr[:, 1], "z": arr[:, 2]})
                return PlotSpec(
                    kind=spec_kind,
                    ndim=3,
                    title=title,
                    x=Axis(label="x1"),
                    y=Axis(label="x2"),
                    z=Axis(label="x3"),
                    layers=[layer],
                    meta=meta,
                )
            spec_kind = resolve_plot_kind(kind, PlotKind.PHASE_PORTRAIT_2D)
            layer = Layer(PlotKind.SCATTER, {"x": arr[:, 0], "y": arr[:, 1]})
            return PlotSpec(
                kind=spec_kind,
                ndim=2,
                aspect="equal",
                title=title,
                x=Axis(label="x1"),
                y=Axis(label="x2"),
                layers=[layer],
                meta=meta,
            )

        if arr.ndim == 2 and arr.shape[0]:
            spec_kind = resolve_plot_kind(kind, PlotKind.LINE_FAMILY)
            idx = np.arange(arr.shape[0], dtype=float)
            layers = [
                Layer(PlotKind.LINE, {"x": idx, "y": arr[:, j]}, label=f"[{j}]")
                for j in range(min(arr.shape[1], 12))
            ]
            return PlotSpec(
                kind=spec_kind,
                ndim=2,
                title=title,
                x=Axis(label="index"),
                y=Axis(label="value"),
                layers=layers,
                legend=Legend(),
                meta=meta,
            )

        raise VisualizationNotInstalled(
            f"{title} wraps an empty or higher-than-2-D array, so the generic ArrayResult "
            "to_plot_spec() has nothing to draw; export it with .to_dict() instead."
        )
