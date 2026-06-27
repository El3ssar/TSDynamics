"""The :class:`CollectionResult` sequence-of-sub-results result.

Split out of ``analysis/_result.py``; wraps a bare ``list`` return (fixed
points, periodic orbits) so it behaves like a list while carrying the result
surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from tsdynamics._result_common import resolve_plot_kind
from tsdynamics.analysis._result_base import AnalysisResult
from tsdynamics.analysis._result_json import _is_frame_scalar, _jsonify
from tsdynamics.analysis._result_viz import VisualizationNotInstalled


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

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe the collection as a :class:`PlotSpec` (safe generic scatter).

        Each item contributes one representative point (its ``x`` attribute, the
        mean of its ``points``, or the item itself if it is a 1-D array); the
        points become a ``SCATTER`` phase portrait (first two coordinates) or, in
        1-D, ``MARKERS`` against their index.  Subclasses with a richer figure (a
        fixed-point overlay with eigenvalue markers) override this.  To draw the
        collection *over* a host figure, use :meth:`AnalysisResult.overlay_on`.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.

        Raises
        ------
        VisualizationNotInstalled
            If no item yields a numeric point (nothing generic to scatter).
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        raw = [self._item_point(item) for item in self.items]
        points = [p for p in raw if p is not None and p.size]
        if not points:
            raise VisualizationNotInstalled(
                f"{type(self).__name__} has no item with a numeric point to scatter, so the "
                "generic CollectionResult to_plot_spec() has nothing to draw; export it with "
                ".to_dict() instead."
            )
        dim = min(p.size for p in points)
        pts = np.asarray([p[:dim] for p in points], dtype=float)
        title = type(self).__name__
        meta = dict(self.meta) if self.meta else {}

        if dim == 1:
            spec_kind = resolve_plot_kind(kind, PlotKind.DIAGNOSTIC_CURVE)
            layer = Layer(PlotKind.MARKERS, {"x": np.arange(len(pts), dtype=float), "y": pts[:, 0]})
            return PlotSpec(
                kind=spec_kind,
                ndim=2,
                title=title,
                x=Axis(label="index"),
                y=Axis(label="value"),
                layers=[layer],
                meta=meta,
            )
        spec_kind = resolve_plot_kind(kind, PlotKind.PHASE_PORTRAIT_2D)
        layer = Layer(PlotKind.SCATTER, {"x": pts[:, 0], "y": pts[:, 1]}, label=title)
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

    @staticmethod
    def _item_point(item: Any) -> np.ndarray | None:
        """Return a 1-D representative point for ``item``, or ``None`` if it has none."""
        x = getattr(item, "x", None)
        if x is not None and np.ndim(x) == 1:
            point: np.ndarray = np.asarray(x, dtype=float)
            return point
        pts = getattr(item, "points", None)
        if pts is not None and np.ndim(pts) == 2:
            centroid: np.ndarray = np.asarray(pts, dtype=float).mean(axis=0)
            return centroid
        if isinstance(item, np.ndarray) and item.ndim == 1:
            arr: np.ndarray = np.asarray(item, dtype=float)
            return arr
        return None

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
