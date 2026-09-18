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
from tsdynamics.analysis._result_json import _jsonify, _row_for
from tsdynamics.analysis._result_viz import VisualizationNotInstalled


@dataclass(frozen=True, eq=False)
class CollectionResult(AnalysisResult):
    """A homogeneous collection of measurements that indexes to **numbers**.

    Wraps a bare ``list`` return (``fixed_points`` → fixed points,
    ``periodic_orbits`` → orbits) so it carries the result surface while
    ``for x in result``, ``result[0]`` and ``len(result)`` keep working.

    **Indexing and iteration give numbers** (contract §4.2 rule 6).
    ``fixed_points(sys)[0]`` is the ``(dim,)`` point as a plain
    :class:`numpy.ndarray` — not a ``FixedPoint`` the caller has to learn to
    unwrap — so it has ``.shape``, ``.tolist()``, a numeric ``dtype``, and
    ``np.asarray(result)[i]`` is the very same row.  The per-member *records*,
    which carry the repr, the verdict and the diagnostics, did not go away: they
    are :attr:`details`, one dot out of the way, indexed in the same order.

    Every per-member diagnostic is additionally a **vectorised** attribute on the
    collection (``fps.points`` / ``fps.eigenvalues`` / ``fps.is_stable``), so the
    common case needs neither a loop nor a record.

    Subclasses add those selectors (``.stable`` / ``.unstable``) and a tidy
    :meth:`to_frame`.

    Attributes
    ----------
    details : tuple
        The collected result records, in order.  The **one public spelling**;
        ``items`` is the dataclass field behind it and holds the same object.

    .. versionchanged:: 6.0
        Three names left ``dir()`` (all still resolve — this is a listing, not a
        permission):

        * ``items`` — measured, ``fps.items is fps.details`` was ``True``, and
         only :attr:`details` is taught.  Two names for one tuple is the
         second-spelling defect the v6 curation exists to remove;
         :meth:`to_dict` still emits the ``"items"`` key.
        * ``count`` and ``index``, the two :class:`~collections.abc.Sequence`
         mixins.  Their plain-English reading ("how many did you find") is not
         their meaning — ``count`` takes a *member* and counts equal ones —
         and ``len(result)`` is the question people actually have.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ()

    #: R2 for ``items`` (``details`` is the taught spelling of the same tuple)
    #: and a mis-reading guard for the two ``Sequence`` mixins — see the class
    #: ``versionchanged`` note.  A listing edit only: all three still resolve.
    _HIDDEN_ATTRIBUTES: ClassVar[frozenset[str]] = frozenset({"count", "index", "items"})

    items: tuple[Any, ...] = ()

    # -- the sequence protocol, over NUMBERS ---------------------------------

    def _item_value(self, item: Any) -> Any:
        """Return the numbers member ``item`` stands for — what ``[]`` hands back.

        A member that is a plain value (``tipping_points`` collects plain dicts)
        is handed back unchanged; a member that is a *result* contributes the
        numbers it stands for — its representative point (see
        :meth:`_item_point`), else the scalar it converts to.  Overridden where
        the natural answer is richer than one point: an :class:`OrbitSet` member
        is its whole ``(p, dim)`` orbit, not the orbit's centroid.

        A result with neither is handed back as itself rather than dropped; no
        in-tree collection holds one, and ``test_indexing_a_collection_gives_numbers``
        fails if one ever ships.
        """
        if not isinstance(item, AnalysisResult):
            return item
        point = self._item_point(item)
        if point is not None:
            return point
        number = item._as_number()
        return item if number is None else number

    def __iter__(self) -> Any:
        """Iterate the members' **numbers** (:attr:`details` iterates the records)."""
        return iter([self._item_value(item) for item in self.items])

    def __len__(self) -> int:  # noqa: D105
        return len(self.items)

    def __getitem__(self, key: Any) -> Any:
        """Return member ``key``'s numbers; a slice gives the list of them.

        ``fps[0]`` is a ``(dim,)`` :class:`numpy.ndarray`, equal to
        ``np.asarray(fps)[0]``.  ``fps.details[0]`` is the record it came from.
        """
        if isinstance(key, slice):
            return [self._item_value(item) for item in self.items[key]]
        return self._item_value(self.items[key])

    def __bool__(self) -> bool:  # noqa: D105
        return bool(self.items)

    @staticmethod
    def _matches(value: Any, probe: Any) -> bool:
        """Whether ``value`` equals ``probe``, for array-valued members too.

        ``value == probe`` is *elementwise* once the members are arrays, and
        ``any()`` over an array raises "truth value is ambiguous" — so ``in`` /
        :meth:`index` / :meth:`count` compare whole values here rather than
        relying on a scalar ``__eq__``.
        """
        if value is probe:
            return True
        try:
            return bool(np.array_equal(np.asarray(value), np.asarray(probe)))
        except (TypeError, ValueError):
            return bool(value == probe)

    def __contains__(self, item: Any) -> bool:
        """Whether any member equals ``item`` — compared as numbers, or as records."""
        return any(
            self._matches(value, item) or self._matches(record, item)
            for value, record in zip(self, self.items, strict=True)
        )

    def __reversed__(self) -> Any:
        """Iterate the members' numbers, last to first."""
        return reversed([self._item_value(item) for item in self.items])

    def index(self, item: Any) -> int:
        """Return the position of ``item``, like :meth:`list.index`.

        Matches on the member's numbers or on its record, so both
        ``fps.index(fps[0])`` and ``fps.index(fps.details[0])`` answer ``0``.
        """
        for i, (value, record) in enumerate(zip(self, self.items, strict=True)):
            if self._matches(value, item) or self._matches(record, item):
                return i
        raise ValueError(f"{item!r} is not in this {type(self).__name__}")

    def count(self, item: Any) -> int:
        """Return how many members equal ``item``, like :meth:`list.count`."""
        return sum(
            1
            for value, record in zip(self, self.items, strict=True)
            if self._matches(value, item) or self._matches(record, item)
        )

    @property
    def details(self) -> tuple[Any, ...]:
        """The member **records**, in the order ``[]`` indexes — the objects ``[]`` no longer hands back.

        ``fps[0]`` is the point; ``fps.details[0]`` is the
        :class:`~tsdynamics.analysis.results.FixedPoint` that point came from,
        with its repr, its ``eigenvalues`` and its ``stable`` verdict.  One name
        across every collection in the library (an
        :class:`~tsdynamics.analysis.results.AttractorSet`'s attractors, an
        :class:`~tsdynamics.analysis.results.OrbitSet`'s orbits, a
        :class:`~tsdynamics.analysis.results.WindowedRQA`'s per-window
        readouts), so it is learned once.
        """  # noqa: E501
        return self.items

    def by_id(self, key: Any) -> Any:
        """Return the **record** whose ``id`` attribute is ``key``.

        ``[]`` indexes a collection by *position* and hands back numbers
        (contract §4.2 rule 6); when the members carry their own integer labels
        this is the explicit spelling for looking one up, and — because a label
        is something you read off a picture and then want to know *about* — it
        hands back the record, exactly as :attr:`details` does.

        Raises
        ------
        KeyError
            If no member carries that id.
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
        if not self.items:
            # An EMPTY collection is an empty array of numbers, not an empty
            # array of objects: ``np.asarray(tipping_points(...))`` must stay
            # usable in arithmetic when nothing was found.
            return np.empty(0, dtype=float if dtype is None else dtype)
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
        """Return one item's line in the repr's list (``details[0] …``).

        The label is the **accessor that hands back what the line describes**.
        It used to be a bare ``[0]``, and three independent readers took that
        literally and guessed twice: the line shows a fixed point's stability and
        ``Re(λ)max``, so ``fp[0].stable`` is the obvious next keystroke — and
        ``fp[0]`` is the bare coordinate array (contract §4.2 rule 6: indexing a
        collection gives *numbers*).  Printing the accessor closes the gap
        without reversing that rule: the token on the line is the expression to
        type.
        """
        text = item._as_item() if isinstance(item, AnalysisResult) else str(item)
        return f"details[{index}] {text}"

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
            data.update({k: _jsonify(v) for k, v in self._full_extras().items()})
        return data

    def __plot_spec__(self, kind: str | None = None) -> Any:
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
                "generic CollectionResult __plot_spec__() has nothing to draw; export it with "
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
        # A VECTOR field is the content, not decoration: a table of fixed points
        # used to arrive with the coordinates and the eigenvalues silently
        # dropped, three columns of booleans where the answer should be.  Since
        # v6 ``_row_for`` builds a member's row and a *singular* member's whole
        # frame, so one fixed point tabulates exactly like one row of its set.
        #
        # The declared component names live on the SET's provenance (the members
        # are per-point records and carry little of their own), so they are lent
        # to each row — otherwise a pendulum's fixed points tabulate as
        # ``x0``/``x1`` for a class that declares ``("theta", "omega")``.
        names = self.meta.get("variables") if self.meta else None
        rows = [_row_for(item, variables=names) for item in self.items]
        frame = pd.DataFrame(rows)
        frame.attrs["meta"] = dict(self.meta) if self.meta else {}
        return frame
