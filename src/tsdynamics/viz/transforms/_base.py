"""The transform substrate — :class:`Geometry`, :class:`PlotTransform`, :class:`Primitive`.

Three records, and one rule that connects them:

.. code-block:: text

    subject  ──transform.compute──▶  Geometry  ──primitive.build──▶  [Layer]  ──▶  Plot
             (what to measure)      (numbers)   (how to draw it)     (the IR)

A **transform** turns a subject (a :class:`~tsdynamics.data.Trajectory`, a system,
an analysis result) into :class:`Geometry`: named, typed channels of plain
``ndarray`` sitting in a declared :class:`~tsdynamics.viz._frames.Frame`.  A
**primitive** turns that geometry into :class:`~tsdynamics.viz.spec.Layer`
objects.  Neither knows about the other beyond the channel names, which is
exactly what makes the primitive **swappable**: ``basins`` computes a label
image once, and ``image`` / ``contour`` / ``points`` are three ways of drawing
the same numbers.

Why ``Geometry`` is not an IR
-----------------------------
It has no ``to_dict``, no schema version, and no renderer ever sees one — it
dies at spec-build time.  The serializable IR is, and stays,
:class:`~tsdynamics.viz.spec.Plot`.  ``Geometry`` exists for one reason: if
``compute`` took the primitive as an argument, every transform would have to
implement every primitive and the swap would not be a swap.

The channel *type* (``quantitative`` / ``nominal`` / ``ordinal`` / ``temporal``)
is the one idea worth borrowing from Vega-Lite, and the library already needed
it: the matplotlib renderer re-derives "these are category labels, give them a
discrete colormap" from a ``meta`` side-channel.  A channel that says so itself
is right on every backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from .._frames import Frame, FrameSpace, OverlayRole, axis_name, space_arity
from .._visibility import dir_without as _dir_without
from .._visibility import listing_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..spec import Layer, Plot, PlotKind

__all__ = [
    "SUBJECT_KINDS",
    "Channel",
    "ChannelType",
    "Geometry",
    "Part",
    "PlotTransform",
    "Presentation",
    "Primitive",
    "Source",
    "make_frame",
    "subject_kinds",
]

#: The two — and only two — source categories a transform can declare.
#:
#: ``"data"``
#:     Computable from the samples you already have: a series, a point set, a
#:     label image.  A ``data`` transform **also** accepts a system, because a
#:     model gives you data for free (it integrates one and goes on).
#: ``"model"``
#:     Must evaluate or integrate the right-hand side at points that are *not*
#:     in the input.  The greppable rule: if the transform calls
#:     ``_rhs_numeric`` / ``jacobian`` / ``integrate`` / ``iterate`` / ``_step``
#:     at a point not already in the input array, it is ``model``.  Handed a bare
#:     :class:`~tsdynamics.data.Trajectory`, a ``model`` transform raises
#:     :class:`~tsdynamics.errors.InvalidInputError` naming the system it needs.
#:
#: There is deliberately no third category.
Source = Literal["data", "model"]

#: The closed vocabulary a transform may declare in ``subjects=``, beside any
#: analysis-result **class name**.  Five words, each meaning exactly what its
#: consumers need:
#:
#: ``system``
#:     A system **object** — something with equations, which can be run.
#: ``flow``
#:     A **continuous system**: one with a vector field to evaluate at points no
#:     trajectory visits.  A flow's *samples* are not a flow, because they carry
#:     no right-hand side — they are ``trajectory`` / ``array``.
#: ``map``
#:     Discrete-time dynamics: a :class:`~tsdynamics.families.DiscreteMap` **or a
#:     trajectory of one**.  Here the data *does* carry the distinction — the
#:     iterates are what a cobweb staircases — which is why this word reaches
#:     further than ``flow``.
#: ``trajectory`` / ``array``
#:     Measured samples, with or without a time axis.  ``array`` is the permissive
#:     bucket: anything that is not a system, a trajectory, a result or a bare
#:     right-hand side is read as numbers, because a user holding numbers should
#:     not have to construct a library type to look at them.
#: ``function``
#:     A bare right-hand side ``f(u, t)`` — a model without a system around it.
SUBJECT_KINDS: frozenset[str] = frozenset(
    {"flow", "map", "system", "trajectory", "array", "function"}
)


def subject_kinds(subject: Any) -> frozenset[str]:
    """Classify a plotting subject into the :data:`SUBJECT_KINDS` it answers to.

    One function, consulted by :meth:`PlotTransform.accepts_subject` — which is
    in turn what ``ts.plot`` pairing, ``ts.viz.transforms.find(subject=…)`` and
    ``subject.plot.<TAB>`` all read — so *"can this be drawn from that?"* has a
    single answer in the library rather than one per door.

    An **analysis result** answers to every class name in its MRO, so declaring
    ``subjects=("ScalingResult",)`` admits ``DimensionResult`` and
    ``LyapunovFromData`` without naming either.  A result is deliberately not
    ``"array"``: it may be array-like, but a Lyapunov spectrum drawn as a power
    spectrum is a wrong picture, not a permissive one.  A result is recognised by
    carrying ``__plot_spec__`` (the one seam every plottable has) while being
    neither a system nor a trajectory — so this module still imports nothing from
    :mod:`tsdynamics.analysis`.

    Everything else falls into ``array``, which keeps the door as wide as it was:
    a pandas Series, a list of lists, anything ``as_trajectory`` can coerce.
    """
    from tsdynamics.families import SystemBase

    if isinstance(subject, SystemBase) or (
        hasattr(subject, "family") and hasattr(subject, "dim") and hasattr(subject, "run")
    ):
        family = getattr(subject, "family", None)
        return frozenset({"system", "map" if family == "map" else "flow"})
    if isinstance(subject, np.ndarray | list | tuple):
        return frozenset({"array"})
    if hasattr(subject, "y") and hasattr(subject, "t"):  # a Trajectory (or subclass)
        # Deliberately **not** ``"array"``: a trajectory is a trajectory, and
        # every ``data`` transform lists both words anyway.  The one transform
        # that declares ``array`` alone reads a recorded ``(times, estimates)``
        # pair — which a Trajectory is not, and used to be offered.
        system = getattr(subject, "system", None)
        family = getattr(system, "family", None) if system is not None else None
        return frozenset({"trajectory", "map"} if family == "map" else {"trajectory"})
    if hasattr(subject, "__plot_spec__"):  # an analysis result (or an out-of-tree plottable)
        return frozenset(cls.__name__ for cls in type(subject).__mro__ if cls is not object)
    if callable(subject):
        return frozenset({"function"})
    return frozenset({"array"})


class ChannelType(StrEnum):
    """What the numbers in a :class:`Channel` *mean*.

    Members
    -------
    QUANTITATIVE
        A continuous magnitude (a coordinate, a speed, a field value).  The
        default, and the only type that maps onto a continuous colour scale.
    NOMINAL
        Unordered category labels — basin indices, attractor ids, cluster
        labels.  A renderer must give these a **discrete** colormap; drawing
        them on a continuous scale is the live plotly ``basins_image`` bug.
    ORDINAL
        Ordered categories (a period number, a symbol rank).
    TEMPORAL
        Time.  Distinguished from ``QUANTITATIVE`` so an animator can find the
        clock without guessing a channel name.
    """

    QUANTITATIVE = "quantitative"
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    TEMPORAL = "temporal"


@dataclass(frozen=True)
class Channel:
    """One named array of numbers plus what they mean.

    Parameters
    ----------
    name : str
        The channel name.  The vocabulary is the :class:`~tsdynamics.viz.spec.Layer`
        one (``x`` / ``y`` / ``z`` / ``c`` / ``u`` / ``v`` / ``lo`` / ``hi`` /
        ``err`` / ``cat`` / ``size`` / ``frames``) because a primitive's job is to
        put channels into a layer; a transform may add its own names for a
        primitive that knows them.
    values : ndarray
        The data.  Coerced with :func:`numpy.asarray` (never copied when it is
        already an array), so a geometry is cheap to build.
    type : ChannelType, optional
        See :class:`ChannelType`.  Default :data:`ChannelType.QUANTITATIVE`.
    label : str, optional
        A human label for an axis / colorbar built from this channel.
    """

    name: str
    values: np.ndarray
    type: ChannelType = ChannelType.QUANTITATIVE
    label: str | None = None

    def __post_init__(self) -> None:
        """Coerce ``values`` to an ndarray and ``type`` to a :class:`ChannelType`."""
        object.__setattr__(self, "values", np.asarray(self.values))
        object.__setattr__(self, "type", ChannelType(self.type))

    @property
    def is_categorical(self) -> bool:
        """Whether this channel holds category labels (``NOMINAL`` / ``ORDINAL``)."""
        return self.type in (ChannelType.NOMINAL, ChannelType.ORDINAL)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return the underlying array, so ``np.asarray(channel)`` is the data.

        Without this a channel is an opaque object and
        ``np.asarray(part.channels["x"]).shape`` is ``()`` — a 0-d array holding
        the ``Channel`` itself, which is a silent wrong answer for anyone who
        reached into a geometry to get their numbers back.
        """
        values = self.values if dtype is None else self.values.astype(dtype)
        return np.array(values, copy=True) if copy else np.asarray(values)

    def __len__(self) -> int:
        """Return the length of the underlying array."""
        return len(self.values)

    def __repr__(self) -> str:  # noqa: D105
        return f"Channel({self.name!r}, {self.values.shape}, {self.type.value})"


def _as_channels(
    channels: Mapping[str, Any],
    types: Mapping[str, ChannelType | str] | None = None,
) -> dict[str, Channel]:
    """Coerce a ``name -> array | Channel`` mapping into ``name -> Channel``."""
    types = types or {}
    out: dict[str, Channel] = {}
    for name, value in channels.items():
        if isinstance(value, Channel):
            out[name] = value
        else:
            out[name] = Channel(name, value, ChannelType(types.get(name, ChannelType.QUANTITATIVE)))
    return out


@dataclass(frozen=True, init=False)
class Part:
    """One drawable piece of a :class:`Geometry`.

    Most geometries are a single part (an image, a curve, a point cloud).  A part
    list exists for the genuinely plural cases the library already has — one
    curve per component in a time series, the ``y = x`` diagonal *and* the
    staircase of a cobweb, one polyline per contour level — where the pieces have
    different lengths and different labels and so cannot be columns of one array.

    Parameters
    ----------
    channels : mapping of str to Channel or ndarray
        The part's data.  Bare arrays are wrapped as
        :data:`~ChannelType.QUANTITATIVE` channels.
    label : str, optional
        The legend entry for this part.
    style : mapping, optional
        Per-part style overrides, in the canonical
        :data:`~tsdynamics.viz.style.STYLE_KEYS` vocabulary.
    primitive : str, optional
        Pin this part to one primitive regardless of the caller's choice.  This
        is for a geometry that is genuinely heterogeneous — a vector field with a
        host orbit drawn over it is a ``quiver`` part *and* a ``line`` part, and
        no single primitive draws both.  ``None`` (the default) means "draw me
        with whatever primitive was chosen", which is the case that makes the
        swap a swap.
    """

    channels: Mapping[str, Channel]
    label: str | None = None
    style: Mapping[str, Any] = field(default_factory=dict)
    primitive: str | None = None

    def __init__(
        self,
        channels: Mapping[str, Any],
        label: str | None = None,
        style: Mapping[str, Any] | None = None,
        primitive: str | None = None,
    ) -> None:
        """Wrap bare arrays as :class:`Channel` values behind a read-only view.

        Written by hand rather than generated so a transform author can pass
        plain ``ndarray`` values (the overwhelmingly common case) while the
        stored field keeps its honest :class:`Channel` type.
        """
        object.__setattr__(self, "channels", MappingProxyType(_as_channels(channels)))
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "style", MappingProxyType(dict(style or {})))
        object.__setattr__(self, "primitive", primitive)

    def array(self, name: str) -> np.ndarray | None:
        """Return channel ``name``'s array, or ``None`` when it is absent."""
        chan = self.channels.get(name)
        return None if chan is None else chan.values

    def __getitem__(self, name: str) -> np.ndarray:
        """Return channel ``name``'s **array** — ``part["x"]`` is the numbers.

        The subscript is the one a person reaching into a geometry writes, so it
        returns the array rather than the :class:`Channel` wrapper (which stays
        available as ``part.channels[name]`` for a renderer that needs the type).

        Raises
        ------
        KeyError
            Naming the channels this part does carry.
        """
        chan = self.channels.get(name)
        if chan is None:
            raise KeyError(f"{name!r}; this part carries {sorted(self.channels)}")
        return chan.values

    def __contains__(self, name: object) -> bool:
        """Whether this part carries a channel of that name."""
        return name in self.channels

    def __repr__(self) -> str:  # noqa: D105
        return f"Part({sorted(self.channels)}, label={self.label!r})"


#: The keys a channel mapping may carry that are **not** channels.  Everything
#: else in the mapping is data — which is what lets a transform author return
#: ``{"x": …, "y": …}``, and a primitive author return
#: ``[{"mark": "line", …}, {"mark": "points", …}]``, without importing an IR type.
PART_KEYS: frozenset[str] = frozenset({"label", "style", "primitive", "mark"})


def part_from_mapping(mapping: Mapping[str, Any]) -> Part:
    """Build one :class:`Part` from a plain mapping of channels plus :data:`PART_KEYS`.

    The single reader of the mapping convention, shared by the transform door
    (:func:`~tsdynamics.viz.transforms.register`), the primitive door
    (:func:`~tsdynamics.viz.transforms.register_primitive`) and the arrays door
    (:func:`tsdynamics.viz.draw`), so the three cannot drift.  ``mark`` is
    accepted as a synonym of ``primitive``: a primitive that emits several marks
    pins its pieces by mark, and a transform pins a part to a primitive — the
    same field, spelled the way each author thinks about it.
    """
    channels = {k: v for k, v in mapping.items() if k not in PART_KEYS}
    pinned = mapping.get("primitive", mapping.get("mark"))
    return Part(
        channels,
        label=mapping.get("label"),
        style=dict(mapping.get("style") or {}),
        primitive=str(pinned) if pinned is not None else None,
    )


def parts_from_return(result: Any) -> list[Part] | None:
    """Read a mapping / sequence-of-mappings return as parts; ``None`` if it is neither."""
    if isinstance(result, Mapping):
        return [part_from_mapping(result)]
    if (
        isinstance(result, Sequence)
        and not isinstance(result, (str, bytes))
        and len(result) > 0
        and all(isinstance(item, Mapping) for item in result)
    ):
        return [part_from_mapping(item) for item in result]
    return None


@dataclass(frozen=True)
class Geometry:
    """Plottable numbers in a declared coordinate frame — what a transform returns.

    Parameters
    ----------
    transform : str
        The name of the transform that produced this geometry.  Stamped onto
        every :class:`~tsdynamics.viz.spec.Layer` it lowers to, which is what
        makes per-source restyling inside an overlay
        (``spec.style("basins", cmap=...)``) possible at all.
    frame : Frame
        The coordinate space, its dimension, and its axis names — the thing that
        decides whether this geometry may share axes with another.  Build it with
        :func:`make_frame` so the axis names come from the same normalization the
        overlay check uses.
    parts : sequence of Part
        The drawable pieces, in draw order.  Passing ``channels=`` instead builds
        a one-part geometry.
    channels : mapping, optional
        Shorthand for a single-part geometry.  Mutually exclusive with ``parts``.
    axis_labels : tuple of str, optional
        The label of each *drawn* axis (x, y, and z when 3-D).  This is
        presentation text — ``"$x$"``, ``"x(t - 4)"`` — as opposed to
        :attr:`Frame.axes`, which are the normalized names the overlay check
        compares.  A ``time`` frame is ``ndim=1`` but draws two axes, so the two
        tuples legitimately differ in length.
    axis_limits : tuple, optional
        Optional ``(lo, hi)`` per drawn axis (``None`` to autoscale that axis) —
        for a transform that samples a box and must draw exactly that box.
    axis_scales : tuple, optional
        Optional ``"linear"`` / ``"log"`` / ``"symlog"`` per drawn axis (``None``
        to leave the default).  For the transforms whose quantity is *only*
        readable on a logarithmic axis: a power spectrum on linear axes is a
        spike at ``f = 0`` and a flat line, which is a picture of nothing.  It is
        a default, not a lock — ``spec.rescale(y="linear")`` still wins, because
        the tweak runs after the spec is assembled.
    kind : PlotKind, optional
        The semantic kind of the assembled spec, when it depends on the data
        rather than on the transform (a portrait is 2-D or 3-D).  ``None`` uses
        the transform's declared kind.
    chosen_primitive : str, optional
        Override the transform's default primitive **for this geometry only**.
        The default sometimes depends on the data rather than on the transform:
        a discrete-map orbit is a point sequence and a flow is a connected
        curve, and drawing either as the other is wrong.  An explicit
        ``primitive=`` from the caller still wins, and the override is checked
        against the transform's declared row like any other choice.

        .. versionchanged:: 6.0
            Spelled ``primitive`` — one letter from :attr:`primitives`, on the
            same record, meaning the opposite thing (*the one chosen* versus *the
            whole legal row*).  The old spelling raises and names this one.
    primitives : frozenset of str, optional
        Narrow the transform's declared row **for this geometry**.  A row is a
        statement about the transform; some transforms produce geometry whose
        shape decides which of those primitives can honestly draw it (a 2-D
        spatial field is an image, its 1-D profile is a line, and drawing either
        with the other's primitive is a plot of nothing).  ``None`` (the default)
        keeps the whole row.  The narrowing can only remove, never add.
    aspect : {"auto", "equal"}, optional
        Override the transform's declared aspect for this geometry (a 2-D field
        is drawn on equal axes, its 1-D profile is not).
    title : str, optional
        Figure title.
    color_label : str, optional
        The colorbar label, when this geometry has a colour dimension.
    legend : bool, optional
        Force a legend on / off for this geometry, overriding the transform's
        :attr:`Presentation.legend` policy.  For the case where legibility
        depends on the *data* — a field with a host orbit over it wants a legend,
        the same field alone does not.
    clim : tuple of float, optional
        An explicit colour range — for a geometry whose colour scale must be
        fixed across frames rather than inferred from the drawn one.
    meta : mapping, optional
        Provenance, carried onto the spec.  **Every auto-chosen default a
        model transform made belongs here** (the sampled region, the grid
        resolution, the integration time), because the alternative is a plot of
        half the story with nothing to say so.

    Notes
    -----
    A ``Geometry`` is frozen and holds only arrays and plain data.  It is not
    serializable, has no schema version, and no renderer ever receives one.
    """

    transform: str
    frame: Frame
    parts: tuple[Part, ...] = ()
    axis_labels: tuple[str, ...] = ()
    axis_limits: tuple[tuple[float, float] | None, ...] = ()
    axis_scales: tuple[str | None, ...] = ()
    kind: PlotKind | None = None
    chosen_primitive: str | None = None
    primitives: frozenset[str] | None = None
    aspect: Literal["auto", "equal"] | None = None
    title: str = ""
    color_label: str | None = None
    legend: bool | None = None
    clim: tuple[float, float] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        transform: str,
        frame: Frame,
        parts: Sequence[Part] | None = None,
        *,
        channels: Mapping[str, Any] | None = None,
        label: str | None = None,
        style: Mapping[str, Any] | None = None,
        axis_labels: Sequence[str] = (),
        axis_limits: Sequence[tuple[float, float] | None] = (),
        axis_scales: Sequence[str | None] = (),
        kind: PlotKind | None = None,
        chosen_primitive: str | None = None,
        primitives: Sequence[str] | None = None,
        aspect: Literal["auto", "equal"] | None = None,
        title: str = "",
        color_label: str | None = None,
        legend: bool | None = None,
        clim: tuple[float, float] | None = None,
        meta: Mapping[str, Any] | None = None,
        **moved: Any,
    ) -> None:
        """Build a geometry from either ``parts`` or the single-part ``channels=``.

        ``**moved`` exists to answer the **one** renamed keyword by name: v6 spelled
        :attr:`chosen_primitive` ``primitive``, and an unnamed ``TypeError:
        __init__() got an unexpected keyword argument`` would be the worst possible
        answer for a transform author mid-migration.
        """
        from tsdynamics.errors import InvalidParameterError

        if "primitive" in moved:
            raise InvalidParameterError(
                "Geometry(primitive=…) is now Geometry(chosen_primitive=…) — one letter "
                "from primitives= (the whole legal row) meant the opposite thing.\n"
                f"    Geometry(..., chosen_primitive={moved['primitive']!r})"
            )
        if moved:
            raise InvalidParameterError(
                f"Geometry got unexpected keyword(s) {sorted(moved)}; it takes "
                "transform, frame, parts/channels, label, style, axis_labels, axis_limits, "
                "axis_scales, kind, chosen_primitive, primitives, aspect, title, "
                "color_label, legend, clim and meta."
            )
        if (parts is None) == (channels is None):
            raise InvalidParameterError(
                "Geometry takes exactly one of parts= (several drawable pieces) or "
                "channels= (the single-part shorthand)."
            )
        resolved = (
            tuple(parts)
            if parts is not None
            else (Part(channels or {}, label=label, style=style or {}),)
        )
        object.__setattr__(self, "transform", str(transform))
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "parts", resolved)
        object.__setattr__(self, "axis_labels", tuple(str(a) for a in axis_labels))
        object.__setattr__(self, "axis_limits", tuple(axis_limits))
        object.__setattr__(self, "axis_scales", tuple(axis_scales))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "chosen_primitive", chosen_primitive)
        object.__setattr__(
            self, "primitives", frozenset(primitives) if primitives is not None else None
        )
        object.__setattr__(self, "aspect", aspect)
        object.__setattr__(self, "title", str(title))
        object.__setattr__(self, "color_label", color_label)
        object.__setattr__(self, "legend", legend)
        object.__setattr__(self, "clim", clim)
        object.__setattr__(self, "meta", MappingProxyType(dict(meta or {})))

    # -- introspection (the rung-4 escape hatch) ---------------------------

    @property
    def space(self) -> FrameSpace:
        """The coordinate space this geometry is drawn in."""
        return self.frame.space

    @property
    def axes(self) -> tuple[str, ...]:
        """The normalized coordinate names of the frame (what the overlay check compares)."""
        return self.frame.axes

    @property
    def channels(self) -> Mapping[str, Channel]:
        """The channels of a **single-part** geometry.

        Raises
        ------
        tsdynamics.errors.InvalidParameterError
            If this geometry has several parts — returning the first part's
            channels would silently hide the rest, which is the class of quiet
            wrongness this whole layer exists to remove.  Use :attr:`parts`.
        """
        from tsdynamics.errors import InvalidParameterError

        if len(self.parts) != 1:
            raise InvalidParameterError(
                f"geometry {self.transform!r} has {len(self.parts)} parts, so it has no single "
                "channel set; iterate g.parts (each has .channels) instead."
            )
        return self.parts[0].channels

    def channel_names(self) -> frozenset[str]:
        """Every channel name present in **any** part (what a primitive is checked against)."""
        return frozenset(name for part in self.parts for name in part.channels)

    def __len__(self) -> int:
        """Return the number of drawable parts."""
        return len(self.parts)

    def __iter__(self) -> Any:
        """Iterate the drawable parts, in draw order."""
        return iter(self.parts)

    def __getitem__(self, key: str | int) -> Any:
        """``g["x"]`` is the channel's **array**; ``g[0]`` is one :class:`Part`.

        The two subscripts read differently and cannot be confused: a string names
        a channel (and asks the same question ``part["x"]`` does), an integer
        picks one of several drawable pieces.

        On a **multi-part** geometry a string returns the channel **stacked over
        the parts** — shape ``(n_parts, …)`` — which is what a caller reaching
        into a three-component time series means by ``g["y"]``.  It used to raise,
        sending them to ``g.parts[i]["y"]`` and a loop.  Parts whose channel has a
        different length still raise, because stacking those would invent
        alignment that is not there.

        Raises
        ------
        tsdynamics.errors.InvalidParameterError
            For a channel this geometry does not carry, or one whose length
            differs between parts.
        """
        if not isinstance(key, str):
            return self.parts[key]
        if len(self.parts) == 1:
            return self.channels[key].values
        return self._stacked(key)

    def _stacked(self, name: str) -> np.ndarray:
        """Return channel ``name`` stacked over every part that carries it."""
        from tsdynamics.errors import InvalidParameterError

        rows = [part.channels[name].values for part in self.parts if name in part.channels]
        if not rows:
            raise InvalidParameterError(
                f"geometry {self.transform!r} carries no channel {name!r}; it has "
                f"{sorted(self.channel_names())}."
            )
        widths = {np.shape(r) for r in rows}
        if len(widths) != 1:
            raise InvalidParameterError(
                f"geometry {self.transform!r} has {name!r} at differing shapes across its "
                f"{len(self.parts)} parts ({sorted(widths)}), so there is no array to stack; "
                "iterate g.parts (each has .channels) instead."
            )
        return np.stack([np.asarray(r, dtype=float) for r in rows])

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return this geometry's numbers as a **numeric** array.

        ``ts.viz.geometry(...)`` is advertised as *the arrays escape hatch*, and
        ``np.asarray`` of one used to be a ``(n_parts,)`` array of **objects** —
        a silent non-answer that plots as nothing and arithmetics into a
        ``TypeError`` far from the call site.

        The array is the geometry's drawn channels in axis order (``x``, ``y``,
        ``z`` where present), stacked: a single-part 2-D geometry gives
        ``(2, n)``, a three-part time series ``(2, 3, n)``.  A geometry whose
        parts carry different channels or differing lengths has no such array and
        **raises**, naming ``g.parts`` — never an object array.
        """
        from tsdynamics.errors import InvalidInputError

        shared = [c for c in ("x", "y", "z") if all(c in part.channels for part in self.parts)]
        if not shared:
            raise InvalidInputError(
                f"geometry {self.transform!r} is not one array of numbers: its "
                f"{len(self.parts)} part(s) share no coordinate channel (they carry "
                f"{sorted(self.channel_names())}). Ask for one — np.asarray(g['u']) — "
                "or iterate g.parts."
            )
        stacked = np.stack([np.asarray(self[c], dtype=float) for c in shared])
        return stacked.astype(dtype) if dtype is not None else stacked

    def __getattr__(self, name: str) -> Any:
        """Answer the one renamed field by name; everything else is a plain miss.

        Reached only when normal lookup fails, so it costs nothing on a hit and
        leaves ``hasattr(g, anything)`` answering ``False`` as it should.
        """
        moved = _GEOMETRY_MOVED.get(name)
        if moved is not None:
            raise AttributeError(f"Geometry has no {name!r}. {moved}")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __dir__(self) -> list[str]:
        """Expose the documented doors: ``g["x"]``, ``.parts``, ``.frame``, ``.primitives``, ``.meta``.

        ``ts.viz.geometry(subject, name)`` is *the arrays escape hatch*, and what
        a caller does with one is read a channel (``g["x"]``), reach the drawable
        pieces behind it (``.parts`` — each with its own ``channels``, ``label``
        and ``style``), ask what space it is in (``.frame``), ask what may legally
        draw it (``.primitives``), and read the choices the transform made for
        them (``.meta`` — the sampled region, the grid resolution, the
        integration time).

        ``parts`` is listed because **this class hands it back**: both the
        differing-shape branch of ``g["x"]`` and the no-shared-channel branch of
        ``np.asarray(g)`` end their message with *"iterate g.parts"*, and a
        remedy a library prints must be a name the reader can then tab-complete.
        It is also the only route to a *per-piece* label, style or primitive,
        which is what a multi-part geometry has that a stacked array does not.

        The other fourteen names — ``channels`` ``channel_names`` ``axis_labels``
        ``axis_limits`` ``axis_scales`` ``space`` ``axes`` ``kind`` ``aspect``
        ``title`` ``clim`` ``color_label`` ``legend`` ``transform``
        ``chosen_primitive`` — are what the *lowering* reads on the way to a
        :class:`~tsdynamics.viz.spec.Plot`.  Every one is still a readable,
        tested attribute; ``dir`` has no part in attribute lookup.  Iteration
        (``for part in g``) and ``len(g)`` are untouched.
        """
        return ["__getitem__", "frame", "meta", "parts", "primitives"]

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Geometry({self.transform!r}, frame={self.frame.describe()}, "
            f"parts={len(self.parts)}, channels={sorted(self.channel_names())})"
        )


#: ``old Geometry attribute -> the sentence naming the working spelling``.
#: Read by :meth:`Geometry.__getattr__`; a rename's message *is* its migration guide.
_GEOMETRY_MOVED: dict[str, str] = {
    "primitive": (
        "It is `chosen_primitive` now — one letter from `primitives` (the whole "
        "legal row), on the same record, meaning the opposite thing. Read it with "
        "`g.chosen_primitive`; the row is `g.primitives`."
    ),
}


def make_frame(
    space: FrameSpace | str, labels: Sequence[str | None], ndim: int | None = None
) -> Frame:
    """Build a :class:`~tsdynamics.viz._frames.Frame` from presentation labels.

    The axis *names* a frame compares are the labels put through
    :func:`~tsdynamics.viz._frames.axis_name` — the same normalization the
    overlay check applies to a hand-built spec — so a transform that labels its
    axis ``"$x$"`` and one that labels it ``"x"`` are recognised as the same
    coordinate.  Missing labels become
    :data:`~tsdynamics.viz._frames.ANY_AXIS` (an explicit "I did not say"),
    never a silently-compatible blank.

    Parameters
    ----------
    space : FrameSpace or str
        The coordinate space.
    labels : sequence of str or None
        The axis labels, in axis order.  Only the first ``ndim`` are used; a
        shorter sequence is padded.
    ndim : int, optional
        How many of the drawn axes are *coordinates* of that space.  **Derived
        from the space** (:func:`~tsdynamics.viz._frames.space_arity`) and only
        worth passing for a geometry whose shape genuinely varies *within* one
        space — a spatial field is a 2-D lattice or a 1-D profile.

        .. versionchanged:: 6.0
            It used to be the required second positional argument, which made
            every caller state a number the space already fixes.
    """
    width = space_arity(space) if ndim is None else int(ndim)
    names = [axis_name(label) for label in list(labels)[:width]]
    names += [axis_name(None)] * (width - len(names))
    return Frame(FrameSpace(space), width, tuple(names))


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Primitive:
    """How geometry is drawn — one or more marks plus a data-shaping rule.

    "Primitive" does not mean simple.  A basin image, a quiver field, a 3-D
    surface and a marching-squares contour are all primitives; what they share is
    that they consume channels and emit :class:`~tsdynamics.viz.spec.Layer`
    objects, and that they know nothing about which transform produced the
    numbers.

    Parameters
    ----------
    name : str
        The name used in ``primitive=`` and in a transform's declared row.
    build : callable
        ``build(geometry, part, options) -> list[Layer]``.  Called once per
        :class:`Part`; the returned layers are concatenated in part order.
    marks : frozenset of PlotKind
        The layer marks this primitive lowers to.  Documentation and governance
        only — no new :class:`~tsdynamics.viz.spec.PlotKind` member is ever
        needed to add a primitive, which is the point.
    requires : frozenset of str
        The channel names a part must carry for this primitive to draw it.  The
        declared-cell check is exactly ``requires <= geometry.channel_names()``,
        so a row cannot claim a pair that cannot physically work.
    frames : frozenset of FrameSpace, optional
        The coordinate spaces this primitive can draw in.  ``None`` means any.
    options : frozenset of str, optional
        The keyword names this primitive accepts (``bins``, ``levels``, …).
        Anything else passed through ``primitive_options`` raises rather than
        being silently dropped.
    consumes : frozenset of str, optional
        The channels this primitive can actually *draw* (a superset of
        :attr:`requires`: ``line`` requires ``x``/``y`` and consumes ``c`` when
        it is there).  Read by the check that refuses to hand a primitive a
        colour channel it would throw away.  Empty means "not declared", which is
        only honest for a primitive that reshapes its input wholesale.
    emits_frame : FrameSpace, optional
        Set when the primitive *changes* the coordinate space of what it draws
        (a space-filling-curve image consumes a 1-D series and emits a lattice).
        ``None`` (the default) keeps the geometry's frame.
    doc : str, optional
        One line, shown by :func:`tsdynamics.viz.compatibility`.
    """

    name: str
    build: Callable[[Geometry, Part, Mapping[str, Any]], list[Layer]]
    marks: frozenset[PlotKind]
    requires: frozenset[str] = frozenset()
    frames: frozenset[FrameSpace] | None = None
    options: frozenset[str] = frozenset()
    emits_frame: FrameSpace | None = None
    doc: str = ""
    consumes: frozenset[str] = frozenset()

    def accepts_frame(self, space: FrameSpace) -> bool:
        """Whether this primitive can draw in coordinate space ``space``."""
        return self.frames is None or space in self.frames

    def __dir__(self) -> list[str]:
        """Expose what ``primitives.get("line")`` is asked: ``name doc requires marks``.

        ``ts.viz.primitives.get("line").requires`` is the advertised read — *what
        channels must my geometry carry for this to draw it?* — and ``marks`` is
        the governance answer to *what does it lower to?*.

        ``build`` (the drawing function), ``frames`` / ``accepts_frame`` (the
        coordinate-space guard), ``options`` (checked for you when you pass one)
        and ``consumes`` / ``emits_frame`` are how the lowering drives the record.
        All six stay public; a primitive author still writes and reads them.
        """
        return _dir_without(self, _PRIMITIVE_MACHINERY)

    def __repr__(self) -> str:  # noqa: D105
        return f"Primitive({self.name!r})"


#: What a :class:`Primitive` record carries for the lowering to drive it — as
#: opposed to the four names a caller asks it about.  Hidden from ``dir()`` only.
_PRIMITIVE_MACHINERY: frozenset[str] = frozenset(
    {"accepts_frame", "build", "consumes", "emits_frame", "frames", "options"}
)


# ---------------------------------------------------------------------------
# Presentation + the transform record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Presentation:
    """Presentation intent a transform declares once, honored by every backend.

    This is where a *cross-backend* presentation fact lives — the aspect ratio a
    phase portrait needs, the discrete colormap a label image needs.  Keeping it
    on the transform rather than in one renderer's private preset table is not
    cosmetic: the library shipped a live bug where ``basins_image`` drew with a
    categorical colormap on matplotlib and plotly's continuous default on
    plotly, because only matplotlib had the table.

    Parameters
    ----------
    aspect : {"auto", "equal"}, optional
        The axes aspect ratio.  ``"equal"`` for anything drawn in a metric space
        (a phase portrait, a basin image, a section).
    autocolor : bool, optional
        Run :meth:`~tsdynamics.viz.spec.Plot.autocolor` on the assembled
        spec — attach a colorbar and infer the colour range from the drawn data.
    legend : bool, optional
        ``True`` / ``False`` force a legend; ``None`` (default) attaches one when
        more than one part carries a label.
    cmap : str, optional
        The default colormap for this transform's colour dimension.
    discrete : bool, optional
        Force a discrete (categorical) colormap.  Usually unnecessary — a
        :data:`~ChannelType.NOMINAL` colour channel implies it — but available
        for a transform whose labels arrive as floats.
    """

    aspect: Literal["auto", "equal"] = "auto"
    autocolor: bool = False
    legend: bool | None = None
    cmap: str | None = None
    discrete: bool = False


#: What a transform's ``example`` callable must return: the subject to run the
#: transform on, and the options to run it with.
ExampleFactory = Callable[[str], "tuple[Any, Mapping[str, Any]]"]

#: What a :class:`PlotTransform` record carries so the *library* can drive it,
#: as opposed to what it declares so a *reader* can understand it.  Hidden from
#: ``dir()`` only — every one is public, importable, tested, and called on every
#: single plot this library draws.
_TRANSFORM_MACHINERY: frozenset[str] = frozenset(
    {
        "accepts_subject",
        "analysis",
        "compute",
        "describe_primitives",
        "example",
        "labels",
        "ndim",
        "presentation",
        "role",
        "shape_dependent",
    }
)

#: ``deleted PlotTransform attribute -> the sentence that answers the guess``.
#: Read by :meth:`PlotTransform.__getattr__`, for the same reason
#: :data:`_GEOMETRY_MOVED` exists: this record is handed to third-party
#: transform authors by ``ts.viz.transforms.get(...)``, and a field the docs
#: once described must answer by name rather than with a bare miss.
_TRANSFORM_MOVED: dict[str, str] = {
    "exclusive": (
        "It was deleted in v6: measured `frozenset()` on all 39 registered "
        "transforms, it was never a parameter of `register()`, and the `!` marker "
        "it drove could not appear in `ts.viz.compatibility()`. The declared "
        "compatibility row is `t.primitives`, and `ts.viz.transforms.allow(name, "
        "primitive)` widens it."
    ),
}


@dataclass(frozen=True)
class PlotTransform:
    """One registered way of turning a subject into plottable geometry.

    The record **is** the contract: the compatibility matrix is its
    :attr:`primitives` row, declared at the definition site so it cannot rot
    away from the code that computes the geometry.

    Parameters
    ----------
    name : str
        The name used in ``ts.plot(subject, "<name>")`` and as the registry key.
    source : {"data", "model"}
        See :data:`Source`.  There is no third category.
    compute : callable
        ``compute(subject, **options) -> Geometry``.
    default_primitive : str
        The primitive used when the caller names none.  Must be in
        :attr:`primitives`.
    primitives : frozenset of str
        **The declared compatibility row** — every primitive that may draw this
        transform's geometry.  Anything else raises
        :class:`~tsdynamics.errors.InvalidParameterError`; never a fallback,
        never a warning.  A capability mismatch (this backend cannot draw that
        mark) is a different thing and keeps its warn-and-fall-back behaviour;
        a *semantic* mismatch has no correct drawing, so there is nothing to
        fall back to.
    frame : FrameSpace or tuple of FrameSpace
        The coordinate space(s) this transform draws in.  The axis *names* come
        from the geometry, not from here.
    role : OverlayRole
        Draw order by meaning — a field under a curve under a marker — so an
        overlay call is order-free.
    ndim : int or tuple of int
        How many coordinate axes the frame has (a portrait declares ``(2, 3)``).
    requires : str, optional
        An **optional dependency** distribution name this transform needs
        (``"hilbertplot"``).  Listed as *unavailable* rather than omitted when
        absent, because a silently-missing row reads as "the feature does not
        exist".
    doc : str, optional
        One line, shown by :func:`tsdynamics.viz.compatibility`.
    kind : PlotKind, optional
        The semantic kind of the assembled spec.  A geometry may override it
        (``Geometry.kind``) when it depends on the data.
    presentation : Presentation, optional
        See :class:`Presentation`.
    analysis : str, optional
        The dotted path of the analysis function this transform adapts.  **A
        transform is a thin adapter and owns no new math**: anything needing new
        numerics gets an analysis function, with its own citation and its own
        tests, first.  ``None`` for a transform that only reshapes its input.
    example : callable, optional
        ``example(primitive) -> (subject, options)`` — a small, fast subject the
        governance gate can build and render this transform on, for the given
        primitive.  It is on the record rather than in the test file so that
        adding a transform is **one registration and nothing else**: the gate
        picks the new row up with no test edit.
    labels : tuple of str, optional
        The presentation label of each drawn axis.  Used **only** when
        ``compute`` returns a plain mapping of channels instead of a
        :class:`Geometry`: the registry then builds the frame itself, and these
        are the labels it puts on it.  A transform that builds its own
        ``Geometry`` (because its labels depend on the subject) leaves this
        empty and passes ``axis_labels=`` there.
    """

    name: str
    source: Source
    compute: Callable[..., Geometry]
    default_primitive: str
    primitives: frozenset[str]
    frame: tuple[FrameSpace, ...]
    role: OverlayRole
    ndim: tuple[int, ...]
    requires: str | None = None
    doc: str = ""
    kind: PlotKind | None = None
    presentation: Presentation = field(default_factory=Presentation)
    analysis: str | None = None
    example: ExampleFactory | None = None
    labels: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    #: The declared subject vocabulary (see :attr:`subjects`).  Empty means
    #: *"derive it from* :attr:`source` *"*, which is what 26 of the 39 in-tree
    #: transforms do.
    _subjects: tuple[str, ...] = ()

    @property
    def subjects(self) -> tuple[str, ...]:
        """What this transform can be handed — declared, or derived from :attr:`source`.

        The vocabulary is closed and small (:data:`SUBJECT_KINDS`) plus **any
        result class name**:

        ``flow``
            A continuous system — one with a vector field to evaluate.
        ``map``
            A discrete map (or a trajectory of one).
        ``system``
            Either; shorthand for ``("flow", "map")``.
        ``trajectory`` / ``array``
            Measured samples, with or without a time axis.
        A class name (``"ScalingResult"``, ``"GALIResult"``)
            That analysis result, or any subclass of it.

        Derived when not declared: ``model`` → ``("system", "function")`` (it
        needs a right-hand side, in a system or on its own),
        ``data`` → ``("trajectory", "array", "system")``
        (samples, and a system supplies those by being run).  An analysis result
        is **never** admitted by the derived default — a Lyapunov spectrum is not
        a time series — so a transform that reads one says which one.
        """
        if self._subjects:
            return self._subjects
        if self.source == "model":
            return ("system", "function")
        return ("trajectory", "array", "system")

    def accepts_subject(self, subject: Any) -> bool:
        """Whether this transform can be handed ``subject`` (by declared :attr:`subjects`).

        The check behind ``ts.plot(vdp, t1, t2, "vector_field", "nullclines")``:
        a named transform is applied to **every subject its source admits**, so a
        field transform skips the trajectories instead of raising on them.  It is
        also what ``ts.viz.transforms.find(subject=…)`` answers with, and what
        ``subject.plot.<TAB>`` lists — so it must be *true*, not permissive.

        .. versionchanged:: 6.0
            Measured before: ``find(subject=henon)`` advertised all **38**
            transforms and **14 raised**, including every 2-D-flow field
            transform; ``find(subject=lyapunov_spectrum)`` advertised 38 and
            **22 raised**.  The old rule accepted everything that was not literally
            an array or a trajectory, so a *map* was offered ``nullclines`` and a
            *result* was offered ``vector_field``.  Declared subjects replace the
            guess.
        """
        return bool(set(self.subjects) & subject_kinds(subject))

    @property
    def available(self) -> bool:
        """Whether this transform's optional dependency (:attr:`requires`) is importable."""
        if self.requires is None:
            return True
        from importlib.util import find_spec

        try:
            return find_spec(self.requires.replace("-", "_")) is not None
        except (ImportError, ValueError):  # pragma: no cover - defensive
            return False

    def describe_primitives(self) -> tuple[str, ...]:
        """Return the row with its reading mark: ``*`` marks the default primitive.

        Sorted, so the row is stable output rather than set-iteration order.
        A ``†`` after the row (see :attr:`shape_dependent`) warns that only the
        primitives fitting the geometry you actually computed are legal.
        """
        out = [
            f"{name}*" if name == self.default_primitive else name
            for name in sorted(self.primitives)
        ]
        return tuple(out) + (("†",) if self.shape_dependent else ())

    @property
    def shape_dependent(self) -> bool:
        """Whether this transform's geometry *shape* — and so its legal row — varies.

        The declared row is a statement about the **transform**; the real
        constraint is per-**geometry**.  Three transforms produce geometry whose
        shape depends on the subject — ``phase_portrait`` (2-D or 3-D),
        ``spatial_field`` (a 1-D profile or a 2-D field), ``invariant_density``
        (a histogram or an image) — and for those the row is a union, not a
        promise: printing ``phase_portrait -> density, line3d, ...`` flat tells a
        3-D trajectory it can have a density plot, which is not a drawing that
        exists.  Reported as a footnote by :func:`tsdynamics.viz.compatibility`;
        the exact narrowed row for one subject is
        ``ts.viz.geometry(subject, name).primitives``.
        """
        return len(set(self.ndim)) > 1 or len(self.frame) > 1

    def __dir__(self) -> list[str]:
        """Expose the **declaration** a registry reader wants; hide the machinery.

        ``ts.viz.transforms.get("psd")`` hands back this record so a caller can
        ask what it is and what may draw it — ``name`` ``doc`` ``source``
        ``subjects`` ``frame`` ``kind`` ``primitives`` ``default_primitive``
        ``requires`` ``available`` ``aliases``.  That is the eleven names left.

        The ten removed are how the library *uses* the record, not what it says:
        ``compute`` (the function — reached by calling the transform through
        ``ts.plot``), ``ndim`` ``role`` ``presentation`` ``labels`` (spec-assembly
        inputs), ``example`` (the governance gate's fixture), ``analysis`` (the
        provenance string), and the three derived helpers ``accepts_subject``
        ``describe_primitives`` ``shape_dependent`` whose answers
        ``ts.viz.transforms.find(subject=…)`` and ``ts.viz.compatibility()``
        already print.  All ten stay public and tested.
        """
        return _dir_without(self, _TRANSFORM_MACHINERY)

    def __getattr__(self, name: str) -> Any:
        """Answer the one deleted field by name; everything else is a plain miss.

        Reached only when normal lookup fails, so it costs nothing on a hit and
        leaves ``hasattr(t, anything)`` answering ``False`` as it should.
        """
        moved = _TRANSFORM_MOVED.get(name)
        if moved is not None:
            raise AttributeError(f"PlotTransform has no {name!r}. {moved}")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"PlotTransform({self.name!r}, source={self.source!r}, "
            f"primitives={list(self.describe_primitives())})"
        )


def spec_of(geometry: Geometry, transform: PlotTransform, layers: list[Layer]) -> Plot:
    """Assemble the :class:`~tsdynamics.viz.spec.Plot` for a lowered geometry.

    The one assembler every transform shares.  It reads only what the geometry
    and the transform *declared* — the kind, the axis labels and limits, the
    colour label and range, the presentation intent — so a new transform gets a
    correct spec without writing any spec-assembly code, which is the whole of
    the "one registration and nothing else" claim.

    Parameters
    ----------
    geometry : Geometry
        The computed geometry (supplies title, axes, colour, meta and frame).
    transform : PlotTransform
        The record (supplies the semantic kind and the presentation intent).
    layers : list of Layer
        The lowered layers, in draw order.

    Returns
    -------
    Plot
    """
    from tsdynamics.errors import InvalidParameterError

    from ..spec import Axis, Colorbar, Legend, PlotKind, PlotSpec

    kind = geometry.kind if geometry.kind is not None else transform.kind
    if kind is None:  # pragma: no cover - registration rejects this
        raise InvalidParameterError(
            f"transform {transform.name!r} declares no semantic kind and its geometry "
            "supplied none; a spec cannot be assembled without one."
        )
    kind = PlotKind(kind)

    labels = list(geometry.axis_labels) + [""] * 3
    limits = list(geometry.axis_limits) + [None] * 3
    scales = list(geometry.axis_scales) + [None] * 3
    three_d = geometry.frame.ndim >= 3 or any(lyr.kind in _THREE_D_MARKS for lyr in layers)

    present = transform.presentation
    labelled = sum(1 for lyr in layers if lyr.label)
    legend = geometry.legend
    if legend is None:
        legend = present.legend if present.legend is not None else labelled > 1

    spec = PlotSpec(
        kind=kind,
        ndim=cast("Any", 3 if three_d else max(1, min(2, geometry.frame.ndim))),
        aspect=geometry.aspect if geometry.aspect is not None else present.aspect,
        title=geometry.title,
        x=_axis(Axis, labels[0], limits[0], scales[0]),
        y=_axis(Axis, labels[1], limits[1], scales[1]),
        z=_axis(Axis, labels[2], limits[2], scales[2]) if three_d else None,
        clim=geometry.clim,
        colorbar=Colorbar(label=geometry.color_label or "", cmap=present.cmap)
        if _has_color(layers, geometry)
        else None,
        legend=Legend() if legend else None,
        layers=layers,
        meta=dict(geometry.meta),
        frame=geometry.frame,
    )
    if present.discrete and spec.colorbar is not None:
        spec.colorbar.discrete = True
    if present.autocolor:
        spec.autocolor()
    return spec


def _axis(axis_cls: Any, label: str, limits: Any, scale: str | None) -> Any:
    """Build one :class:`~tsdynamics.viz.spec.Axis`, honouring a declared scale."""
    if scale is None:
        return axis_cls(label=label, limits=limits)
    return axis_cls(label=label, limits=limits, scale=scale)


#: Marks that force a 3-D axes (kept in step with ``spec._THREE_D_MARKS``).
_THREE_D_MARKS: frozenset[str] = frozenset({"line3d", "surface3d"})


def _has_color(layers: list[Layer], geometry: Geometry) -> bool:
    """Whether the assembled spec has a colour dimension worth a colorbar."""
    if geometry.color_label is not None or geometry.clim is not None:
        return True
    return any("c" in lyr.data or str(lyr.kind) in ("image", "surface3d") for lyr in layers)


__dir__ = listing_dir(__all__)
