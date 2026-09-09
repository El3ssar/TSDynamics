"""What a set of axes *means* — the :class:`Frame` record and the overlay rule.

Two plots may share one set of axes when, and only when, **they are drawings of
the same space**.  Before v6 the composition front door approximated that with a
three-member whitelist of :class:`~tsdynamics.viz.spec.PlotKind` values
(``compose._OVERLAYABLE``), which got the question wrong in both directions:

- it **refused** legal overlays — a basin image, its attractors, a trajectory and
  the equilibria are four different kinds all drawn on the *same* ``(x, v)``
  plane, and the renderers already draw them correctly on one axes;
- it **permitted** illegal ones — two ``PHASE_PORTRAIT_2D`` specs pass the kind
  test whether they are both ``(x, y)`` or one is ``(x, y)`` and the other
  ``(x, z)``, and the second case silently draws the equilibria in the wrong
  place.  That is the bug ``analysis/fixedpoints/fixed.py`` used to patch by hand
  for its own overlay path only.

A :class:`Frame` states the thing the kind was standing in for: the coordinate
``space``, its dimensionality ``ndim``, and the ``axes`` — one normalized
coordinate name per axis.  Overlay legality is **frame compatibility**, which
widens composition and makes an axis-mismatch overlay impossible at the same
time.

The vocabulary is closed
------------------------
:class:`FrameSpace` is a governance-gated, ten-member enum, exactly like
:class:`~tsdynamics.viz.spec.PlotKind` — ``tests/test_plotspec.py`` pins the
membership, so adding a space is a deliberate, reviewed contract change.

This module imports **nothing** from :mod:`tsdynamics.viz.spec` at runtime (it
reads a spec structurally — ``kind`` / ``x`` / ``y`` / ``z`` / ``layers``), so
``spec.py`` can import it without a cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from .spec import PlotSpec

__all__ = [
    "ANY_AXIS",
    "FRAME_SPACES",
    "Frame",
    "FrameSpace",
    "OverlayRole",
    "axis_name",
    "check_overlay",
    "force_requested",
    "frame_of",
    "role_of",
]

#: The accepted values of the ``on=`` frame-check escape.
ON_VALUES: frozenset[str] = frozenset({"force"})


def force_requested(on: str | None) -> bool:
    """Validate an ``on=`` value and return whether the frame check is forced.

    Every door that takes ``on=`` (:func:`tsdynamics.viz.plot`,
    :meth:`tsdynamics.viz.spec.PlotSpec.add`,
    :meth:`tsdynamics.analysis.AnalysisResult.overlay_on`) validates it here, so
    a typo is an error at all three rather than a silently-not-forced overlay at
    one of them.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``on`` is neither ``None`` nor ``"force"``.
    """
    from tsdynamics.errors import InvalidParameterError

    if on is None:
        return False
    if on not in ON_VALUES:
        raise InvalidParameterError(
            f"unknown on={on!r}; the only accepted value is 'force' (overlay a "
            "deliberate frame mismatch with a warning instead of an error)."
        )
    return True


class FrameSpace(StrEnum):
    """The closed vocabulary of coordinate spaces a plot can be drawn in.

    A :class:`~enum.StrEnum`, so a member compares equal to its value
    (``FrameSpace.STATE2 == "state2"``) and serializes as a plain string.
    Frozen: the membership is pinned by ``tests/test_plotspec.py``, the way
    ``tests/test_viz_vocab.py`` pins :class:`~tsdynamics.viz.spec.PlotKind`.

    Members
    -------
    TIME
        A curve over time — the shared coordinate is ``t`` and the vertical axis
        is a free value axis (so ``x(t)`` and ``y(t)`` legitimately overlay).
    STATE2, STATE3
        A 2-D / 3-D projection of state space: phase portraits, Poincaré
        sections, basin images, vector fields, equilibrium overlays, return maps.
    PARAM1, PARAM2
        A one- / two-parameter sweep (orbit diagrams, continuation; ``PARAM2`` is
        reserved for the two-parameter scans that need an engine kernel).
    INDEX
        Discrete index coordinates — a recurrence plot's ``(i, j)``, a spectrum's
        exponent index.
    GRID2
        A spatial / spacetime lattice: a spacetime image, a spatial field, a
        space-filling-curve image.
    COMPLEX
        The complex plane (eigenvalues / Floquet multipliers).
    SCALING
        A log-log scaling diagnostic (``log C(r)`` vs ``log r``, ``D(q)`` vs
        ``q``, a generic diagnostic curve).
    CATEGORY
        A categorical axis (basin fractions, RQA measure bars).
    """

    TIME = "time"
    STATE2 = "state2"
    STATE3 = "state3"
    PARAM1 = "param1"
    PARAM2 = "param2"
    INDEX = "index"
    GRID2 = "grid2"
    COMPLEX = "complex"
    SCALING = "scaling"
    CATEGORY = "category"


#: The frozen membership of :class:`FrameSpace` (the governance gate reads this).
FRAME_SPACES: frozenset[FrameSpace] = frozenset(FrameSpace)

#: The axis name that means *"this producer did not say which coordinate it
#: draws"*.  It is compatible with every name in the same position — the frame
#: check is exactly as strong as the information the producers give it, never
#: stronger (see :func:`axis_name`).
ANY_AXIS = ""

#: ``$x_{0}$`` / ``x_0`` — the indexed fallback spelling
#: :func:`tsdynamics.analysis._plotbuilder.axis_labels` emits when a system
#: declares no ``variables``.  The index is real information, so it is kept as an
#: ordinal (``#0``) rather than thrown away.
_ORDINAL_LABEL = re.compile(r"\$?([A-Za-z])_\{?(\d+)\}?\$?")

#: A **bare** ``<letter><digits>`` label (``y0`` from ``Trajectory``, ``x1`` /
#: ``x2`` hard-coded by ``AttractorSet.to_plot_spec``).  These name a *position*
#: with no agreed base — ``y0`` is 0-based, ``x1`` is 1-based — so they are
#: read as "unnamed" rather than guessed at.  A **declared** variable arrives
#: dollar-wrapped (``$x1$``) or as a non-numeric bare name (``x``), so it is not
#: caught here.  Documented limitation: a system whose variables are themselves
#: digit-suffixed (``ArnoldWeb``'s ``x1`` / ``x2``) plotted through
#: ``Trajectory.to_plot_spec`` reads as unnamed, so the axis check cannot
#: separate its ``(x1, x2)`` plane from its ``(p1, p2)`` plane.  The fix is for
#: the producers to name their coordinates (P1's typed geometry channels), not
#: for this regex to guess harder.
_BARE_INDEXED_LABEL = re.compile(r"[A-Za-z]\d+")


def axis_name(label: str | None) -> str:
    """Normalize an axis *label* to a comparable coordinate **name**.

    Axis labels are presentation strings and the in-tree producers spell the same
    coordinate three ways — ``"x"`` (``Trajectory``), ``"$x$"``
    (:func:`~tsdynamics.analysis._plotbuilder.axis_labels`) and ``"x1"``
    (``AttractorSet``).  Comparing them raw would make the flagship overlay
    (basins + attractors + trajectory + equilibria, all on one plane) fail on
    typography.  This is the single normalization every frame comparison goes
    through:

    - ``""`` / ``None`` → :data:`ANY_AXIS` (nothing was claimed);
    - ``"$x_{0}$"`` / ``"x_0"`` → ``"#0"`` (an ordinal — the index is real
      information and is kept);
    - a bare ``"y0"`` / ``"x1"`` → :data:`ANY_AXIS` (see
      :data:`_BARE_INDEXED_LABEL`);
    - anything else → the label with ``$`` and ``{}`` stripped, so ``"$x$"`` and
      ``"x"`` are the same coordinate.

    Parameters
    ----------
    label : str, optional
        The raw :attr:`~tsdynamics.viz.spec.Axis.label`.

    Returns
    -------
    str
        The comparable name, or :data:`ANY_AXIS` for an unnamed axis.
    """
    text = (label or "").strip()
    if not text:
        return ANY_AXIS
    ordinal = _ORDINAL_LABEL.fullmatch(text)
    if ordinal is not None:
        return f"#{int(ordinal.group(2))}"
    if _BARE_INDEXED_LABEL.fullmatch(text):
        return ANY_AXIS
    return text.strip("$").replace("{", "").replace("}", "").strip()


def _is_ordinal(name: str) -> bool:
    """Whether ``name`` is an ordinal (``#0``) rather than a coordinate name."""
    return name.startswith("#")


def _axis_rank(name: str) -> int:
    """How informative an axis name is: a name (2) beats an ordinal (1) beats nothing (0)."""
    if name == ANY_AXIS:
        return 0
    return 1 if _is_ordinal(name) else 2


def _axes_compatible(a: str, b: str) -> bool:
    """Whether two normalized axis names may sit on the same axis.

    Three cases, in order of how much the two sides actually claim:

    - either side claims nothing (:data:`ANY_AXIS`) → compatible;
    - both claim the same thing → compatible;
    - one claims a **coordinate name** (``"x"``) and the other an **ordinal**
      (``"#0"``, the indexed fallback of a system that declares no ``variables``)
      → compatible, because neither statement contradicts the other: nothing in
      the library says which ordinal ``"x"`` is.  Two ordinals, or two names,
      must match exactly.
    """
    if a == b or ANY_AXIS in (a, b):
        return True
    return _is_ordinal(a) != _is_ordinal(b)


@dataclass(frozen=True)
class Frame:
    """What one set of axes *means*: a coordinate space, its size, its axes.

    Parameters
    ----------
    space : FrameSpace
        The coordinate space (see :class:`FrameSpace`).
    ndim : int
        How many of the plot's axes are **coordinates** of that space.  This is
        deliberately *not* the figure's dimensionality: a ``TIME`` frame is
        ``ndim=1`` because only the horizontal axis is a coordinate and the
        vertical one is a free value axis — which is why ``x(t)`` and ``y(t)``
        overlay while an ``(x, y)`` portrait and an ``(x, z)`` portrait do not.
    axes : tuple of str
        One normalized coordinate name per axis, in axis order — **required**,
        never defaulted.  A frame with no axes would read as "compatible with
        everything", which is precisely how the wrong-plane overlay used to get
        through; an axis the producer did not name is spelled :data:`ANY_AXIS`
        by :func:`axis_name`, which is an explicit statement rather than an
        absence.

    Notes
    -----
    Compatibility is **not** ``==``: an unnamed axis, and an ordinal one, are
    compatible with a named axis (see :meth:`compatible_with` and
    :func:`_axes_compatible`).  Two frames that are ``==`` are always compatible.
    """

    space: FrameSpace
    ndim: int
    axes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize ``space`` / ``axes`` and enforce ``len(axes) == ndim``."""
        from tsdynamics.errors import InvalidParameterError

        object.__setattr__(self, "space", FrameSpace(self.space))
        object.__setattr__(self, "axes", tuple(str(a) for a in self.axes))
        if self.ndim < 1:
            raise InvalidParameterError(f"a Frame needs ndim >= 1, got {self.ndim}.")
        if len(self.axes) != self.ndim:
            raise InvalidParameterError(
                f"Frame(space={self.space.value!r}, ndim={self.ndim}) needs {self.ndim} axis "
                f"name(s), got {len(self.axes)}: {list(self.axes)}. Axes are required at "
                "construction — a frame that names no axes would overlay onto anything."
            )

    def compatible_with(self, other: Frame) -> bool:
        """Whether ``self`` and ``other`` may share one set of axes.

        Requires the same ``space`` and ``ndim``, and per-axis names that do not
        contradict each other (:func:`_axes_compatible`).
        """
        return (
            self.space is other.space
            and self.ndim == other.ndim
            and all(_axes_compatible(a, b) for a, b in zip(self.axes, other.axes, strict=True))
        )

    def merge(self, other: Frame) -> Frame:
        """Return the more *informative* of two compatible frames.

        Axis-wise by :func:`_axis_rank` — a coordinate name beats an ordinal
        beats nothing — so overlaying a producer that names its coordinates onto
        one that does not keeps the names for the merged figure.
        """
        axes = tuple(
            a if _axis_rank(a) >= _axis_rank(b) else b
            for a, b in zip(self.axes, other.axes, strict=True)
        )
        return Frame(self.space, self.ndim, axes)

    def describe(self) -> str:
        """Return a short human spelling, e.g. ``state2(x, v)``."""
        names = ", ".join(a if a != ANY_AXIS else "?" for a in self.axes)
        return f"{self.space.value}({names})"

    def __str__(self) -> str:  # noqa: D105
        return self.describe()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this frame."""
        return {"space": self.space.value, "ndim": int(self.ndim), "axes": list(self.axes)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Frame:
        """Rebuild a :class:`Frame` from :meth:`to_dict` output."""
        return cls(
            space=FrameSpace(d["space"]),
            ndim=int(d["ndim"]),
            axes=tuple(str(a) for a in d.get("axes", ())),
        )


class OverlayRole(IntEnum):
    """Draw order **by meaning**, so a composition call is order-free.

    A field is a backdrop, a curve is drawn on it, and a marker annotates the
    result — that ordering is a property of *what the thing is*, not of where the
    caller happened to type it.  Sorting an overlay by role therefore makes
    ``plot(basins, traj)`` and ``plot(traj, basins)`` the same picture, which is
    the difference between a composition API and a stack of arguments.

    The enum value **is** the z-order (lower draws first).
    """

    FIELD = 0
    BASE = 1
    OVERLAY = 2


#: Semantic kinds whose figure is a *backdrop* — an image / field the rest of the
#: composition is drawn on top of.
_FIELD_KINDS: frozenset[str] = frozenset(
    {
        "basins_image",
        "recurrence_plot",
        "spacetime",
        "spatial_field",
        "vector_field",
        "phase_portrait_field",
    }
)

#: Semantic kinds that exist *to be drawn on something else* — annotations.
_OVERLAY_KINDS: frozenset[str] = frozenset({"fixed_points_overlay"})

#: Layer marks that make a spec a backdrop when its kind says nothing (the
#: fallback for hand-built specs carrying no semantic kind of their own).
_FIELD_MARKS: frozenset[str] = frozenset({"image", "quiver", "surface3d"})

#: Layer marks that are pure annotation (ditto).
_OVERLAY_MARKS: frozenset[str] = frozenset({"markers"})


def role_of(spec: PlotSpec) -> OverlayRole:
    """Return the z-order :class:`OverlayRole` of ``spec``.

    Resolved from the *semantic kind* first (the producer's own statement about
    what the figure is) and from the layer marks only as a fallback, so a
    hand-built spec carrying a bare mark still sorts sensibly.
    """
    kind = str(spec.kind)
    if kind in _FIELD_KINDS:
        return OverlayRole.FIELD
    if kind in _OVERLAY_KINDS:
        return OverlayRole.OVERLAY
    marks = {str(layer.kind) for layer in spec.layers}
    if marks and marks <= _OVERLAY_MARKS:
        return OverlayRole.OVERLAY
    if marks & _FIELD_MARKS:
        return OverlayRole.FIELD
    return OverlayRole.BASE


#: ``kind`` → ``(space, ndim)``.  ``ndim`` counts **coordinate** axes, so a curve
#: over one coordinate (time, a parameter, a scaling variable) is ``1`` and its
#: vertical axis stays free.  A kind absent from the table falls back to
#: ``(SCALING, 1)`` — a curve against whatever its x axis is, the weakest claim
#: available, and the one that lets two diagnostic curves overlay.
_KIND_FRAME: dict[str, tuple[FrameSpace, int]] = {
    "time_series": (FrameSpace.TIME, 1),
    "ensemble_fan": (FrameSpace.TIME, 1),
    "phase_portrait_2d": (FrameSpace.STATE2, 2),
    "phase_portrait_3d": (FrameSpace.STATE3, 3),
    "poincare_section": (FrameSpace.STATE2, 2),
    "fixed_points_overlay": (FrameSpace.STATE2, 2),
    "basins_image": (FrameSpace.STATE2, 2),
    "vector_field": (FrameSpace.STATE2, 2),
    "phase_portrait_field": (FrameSpace.STATE2, 2),
    "return_map": (FrameSpace.STATE2, 2),
    "cobweb": (FrameSpace.STATE2, 2),
    "orbit_diagram": (FrameSpace.PARAM1, 1),
    "bifurcation": (FrameSpace.PARAM1, 1),
    "continuation": (FrameSpace.PARAM1, 1),
    "recurrence_plot": (FrameSpace.INDEX, 2),
    "lyapunov_spectrum": (FrameSpace.INDEX, 1),
    "spacetime": (FrameSpace.GRID2, 2),
    "eigenvalue_plane": (FrameSpace.COMPLEX, 2),
    "scaling_fit": (FrameSpace.SCALING, 1),
    "dimension_spectrum": (FrameSpace.SCALING, 1),
    "diagnostic_curve": (FrameSpace.SCALING, 1),
    "line_family": (FrameSpace.SCALING, 1),
    "categorical_bar": (FrameSpace.CATEGORY, 1),
    # ``IMAGE`` is a *mark*, but ``_KIND_ALIAS`` lets it be a spec kind (the
    # ``result.plot.image()`` accessor), and a bare image is a lattice — not the
    # ``SCALING`` curve the default would make it.
    "image": (FrameSpace.GRID2, 2),
}

#: The fallback for a kind the table does not name (including a bare layer mark
#: used as a spec kind).
_DEFAULT_KIND_FRAME: tuple[FrameSpace, int] = (FrameSpace.SCALING, 1)


def frame_of(spec: PlotSpec) -> Frame:
    """Return ``spec``'s :class:`Frame` — the declared one, else derived.

    A spec that carries an explicit :attr:`~tsdynamics.viz.spec.PlotSpec.frame`
    is taken at its word.  Otherwise the frame is derived from the semantic kind
    (:data:`_KIND_FRAME`) plus the axis labels, so every spec the library builds
    today — and every hand-built one — has a frame without a single producer
    having to change.

    Two derivations are dynamic rather than tabular, because the kind alone does
    not settle them:

    - a spec that draws in 3-D (:attr:`~tsdynamics.viz.spec.PlotSpec.is_three_d`)
      is promoted to a 3-coordinate frame (``STATE2`` → ``STATE3``);
    - a ``spatial_field`` is a 2-coordinate lattice when it carries an ``IMAGE``
      layer (a 2-D field) and a 1-coordinate profile otherwise (a 1-D field).

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``spec`` is a composite — a multi-panel figure owns no single set of
        axes, so it has no frame.
    """
    from tsdynamics.errors import InvalidParameterError

    declared: Frame | None = getattr(spec, "frame", None)
    if declared is not None:
        return declared
    if spec.is_composite:
        raise InvalidParameterError(
            "a COMPOSITE figure has no frame: it owns no single set of axes. "
            "Overlay the panels themselves, or arrange with layout='stack' / 'row' / 'grid'."
        )

    kind = str(spec.kind)
    if kind == "spatial_field":
        two_d = any(str(layer.kind) == "image" for layer in spec.layers)
        space, ndim = FrameSpace.GRID2, (2 if two_d else 1)
    else:
        space, ndim = _KIND_FRAME.get(kind, _DEFAULT_KIND_FRAME)
        if spec.is_three_d and ndim < 3:
            space, ndim = (FrameSpace.STATE3 if space is FrameSpace.STATE2 else space), 3

    labels = [spec.x, spec.y, spec.z][:ndim]
    axes = tuple(axis_name(getattr(axis, "label", None)) for axis in labels)
    # A 3-D promotion on a spec with no ``z`` axis still needs three names.
    axes = axes + (ANY_AXIS,) * (ndim - len(axes))
    return Frame(space, ndim, axes)


def check_overlay(specs: Sequence[PlotSpec], *, force: bool = False) -> Frame:
    """Validate that every spec in ``specs`` may share one set of axes.

    This is the **one** overlay policy in the library:
    :func:`tsdynamics.viz.plot` and
    :meth:`tsdynamics.analysis.AnalysisResult.overlay_on` both go through it, so
    a pair cannot be legal through one door and refused at the other (before v6
    they disagreed — ``fps.overlay_on(portrait)`` succeeded on the exact pair
    ``viz.plot(portrait, fps)`` refused).

    Parameters
    ----------
    specs : sequence of PlotSpec
        The specs to merge, in argument order (the first is the reference).
    force : bool, default False
        Overlay a deliberate frame mismatch anyway, warning once
        (:class:`~tsdynamics.viz.render.caps.VisualizationDegraded`).  The escape
        exists because without one users route around the API and the wrong-plane
        bug class comes back through the door; it makes the guarantee advisory
        **for that call only**.

    Returns
    -------
    Frame
        The merged frame of the whole overlay (the most informative axis names).

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If two specs are drawings of different spaces, or of the same space on
        different axes.  The message names both frames and what to do instead.
    """
    import warnings

    from tsdynamics.errors import InvalidParameterError

    from .render.caps import VisualizationDegraded

    frames = [frame_of(s) for s in specs]
    merged = frames[0]
    for spec, frame in zip(specs[1:], frames[1:], strict=True):
        if merged.compatible_with(frame):
            merged = merged.merge(frame)
            continue
        message = _mismatch_message(specs[0], merged, spec, frame)
        if not force:
            raise InvalidParameterError(message)
        warnings.warn(
            f"on='force': {message} Drawing them on one set of axes anyway — "
            "the picture may not mean what it looks like.",
            VisualizationDegraded,
            stacklevel=3,
        )
    return merged


def _mismatch_message(base_spec: PlotSpec, base: Frame, other_spec: PlotSpec, other: Frame) -> str:
    """Build the frame-mismatch error text (a different space vs. different axes)."""
    if base.space is not other.space or base.ndim != other.ndim:
        return (
            f"cannot overlay frame {other.space.value!r} ({other_spec.kind}) on frame "
            f"{base.space.value!r} ({base_spec.kind}): they are drawings of different "
            "spaces. Use layout='stack' / 'row' / 'grid' to give each its own panel, "
            "or on='force' to overlay them anyway."
        )
    return (
        f"axes mismatch {base.describe()} vs {other.describe()} — the two are different "
        f"planes, so the {other_spec.kind} would land in the wrong place. Pass the same "
        "components= to both, or on='force' to overlay them anyway."
    )
