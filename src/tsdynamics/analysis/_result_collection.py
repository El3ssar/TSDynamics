"""The :class:`CollectionResult` sequence-of-sub-results result.

Split out of ``analysis/_result.py``; wraps a bare ``list`` return (fixed
points, periodic orbits) so it behaves like a list while carrying the result
surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from tsdynamics.analysis._result_base import _MAX_ITEMS, AnalysisResult
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

    def __contains__(self, item: Any) -> bool:  # noqa: D105
        return any(x is item or x == item for x in self.items)

    def __reversed__(self) -> Any:  # noqa: D105
        return reversed(self.items)

    def index(self, item: Any) -> int:
        """Return the position of ``item``, like :meth:`list.index`."""
        return list(self.items).index(item)

    def count(self, item: Any) -> int:
        """Return how many items equal ``item``, like :meth:`list.count`."""
        return list(self.items).count(item)

    def by_id(self, key: Any) -> Any:
        """Return the item whose ``id`` attribute is ``key``.

        ``[]`` indexes a collection **by position** (contract §4.2 rule 6), the
        way every Python sequence does; when the items carry their own integer
        labels this is the explicit spelling for looking one up.  Raises
        :class:`KeyError` when no item carries that id.
        """
        for item in self.items:
            if getattr(item, "id", None) == key:
                return item
        raise KeyError(f"{type(self).__name__} has no item with id {key!r}")

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return the collected points as a real ``(n, dim)`` float array.

        ``np.asarray(fixed_points)`` used to yield a ``(n,)`` array of *objects*
        — a shape that plots as nothing and arithmetics into a ``TypeError`` —
        because NumPy fell back to iterating the wrappers.  Each item
        contributes its representative point (see :meth:`_item_point`), so a
        fixed-point set becomes the ``(n, dim)`` array of the points themselves.
        An item with no numeric point, or a ragged set, keeps the old object
        array rather than silently dropping or padding rows.
        """
        points = [self._item_point(item) for item in self.items]
        if points and all(p is not None and p.size for p in points):
            sizes = {int(p.size) for p in points if p is not None}
            if len(sizes) == 1:
                arr = np.asarray([np.asarray(p, dtype=float) for p in points], dtype=float)
                return arr.astype(dtype, copy=bool(copy)) if dtype is not None else arr
        return np.asarray(list(self.items), dtype=object)

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

    # -- the readout ---------------------------------------------------------

    def _noun(self) -> str:
        """Return the word for one item (``"item"``; subclasses say ``"point"``)."""
        return "item"

    def _answer(self) -> str:
        """Return the count, or ``none found`` when the collection is empty.

        An empty collection is a real answer, not a failure, so it says so in
        words.  An estimator that can explain *what* it did not find records the
        clause under ``meta["means_none"]`` (e.g. ``tipping_points`` →
        ``"no basin annihilates over the sweep"``) and it is appended here — the
        alternative, a table of analysis names inside this generic module, would
        put every estimator's vocabulary in the wrong file.
        """
        n = len(self.items)
        if n == 0:
            clause = (self.meta.get("means_none") if self.meta else None) or ""
            return "none found" + (f" ({clause})" if clause else "")
        return f"{n} {self._noun()}" + ("s" if n != 1 else "")

    def _item_line(self, index: int, item: Any) -> str:
        """Return one item's line in the repr's list (``[0] …``)."""
        text = item._as_item() if isinstance(item, AnalysisResult) else str(item)
        return f"[{index}] {text}"

    def _item_lines(self) -> tuple[str, ...]:
        """Return the item list, truncated to :data:`_MAX_ITEMS` with a total."""
        shown = [self._item_line(i, item) for i, item in enumerate(self.items[:_MAX_ITEMS])]
        if len(self.items) > _MAX_ITEMS:
            shown.append(f"... [{len(self.items)} total]")
        return tuple(shown)

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly mapping: each item's ``to_dict`` (or repr) + ``meta``.

        Parameters
        ----------
        full : bool, default False
            Also emit the derived quantities the repr reports, and pass ``full``
            down to each item.
        """
        items = [
            item.to_dict(full) if isinstance(item, AnalysisResult) else _jsonify(item)
            for item in self.items
        ]
        data: dict[str, Any] = {"items": items, "meta": _jsonify(self.meta)}
        if full:
            data.update({k: _jsonify(v) for k, v in self._derived().items()})
        return data

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
        from . import _plotbuilder as pb

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

        if dim == 1:
            return pb.spec(
                kind,
                "diagnostic_curve",
                layers=[pb.markers(np.arange(len(pts), dtype=float), pts[:, 0])],
                xlabel="index",
                ylabel="value",
                title=title,
                meta=self.meta,
            )
        return pb.spec(
            kind,
            "phase_portrait_2d",
            layers=[pb.scatter(pts[:, 0], pts[:, 1], label=title)],
            aspect="equal",
            xlabel="x1",
            ylabel="x2",
            title=title,
            meta=self.meta,
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

        Raises
        ------
        ImportError
            If :mod:`pandas` is not installed.
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
