"""Registration, resolution and the declared compatibility matrix.

This module owns the four things that make a transform *first-class*:

1. :func:`plot_transform` — the decorator.  Registering a transform is **one
   call at its definition site and nothing else**: no renderer edit, no
   :class:`~tsdynamics.viz.spec.PlotKind` edit, no ``compose`` edit, no test
   edit.  The compatibility row, the presentation intent and the gate's example
   subject all ride on the same decorator.
2. The **declared matrix** — :attr:`PlotTransform.primitives`, one row per
   transform, checked at registration time against the primitives' own
   structural claims (which coordinate spaces they draw in) so a row cannot
   advertise a pair that cannot physically work.
3. The build path — :func:`geometry` (numbers), :func:`draw` (numbers → spec),
   :func:`build_spec` (both), which every front door goes through.
4. :func:`compatibility` — the matrix, printable and introspectable, with an
   optional-dependency row shown as **unavailable** rather than omitted.  A
   silently-absent row reads as "the feature does not exist", which is a
   different and worse statement than "install the extra".
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ... import registry as _registry
from .._frames import FrameSpace, OverlayRole
from ..spec import PlotKind
from ._base import (
    ExampleFactory,
    Geometry,
    PlotTransform,
    Presentation,
    Source,
    spec_of,
)
from ._primitives import PRIMITIVES, get_primitive, primitive_names

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..spec import Layer, PlotSpec

__all__ = [
    "ADMITTED_SERIES_DIAGNOSTICS",
    "EXCLUDED_SERIES_TOOLBOX",
    "T",
    "TransformCall",
    "build_spec",
    "compatibility",
    "draw",
    "geometry",
    "get",
    "lower",
    "names",
    "row_option_names",
    "plot_transform",
    "transforms",
]


# ---------------------------------------------------------------------------
# The scope boundary, as a checked constant
# ---------------------------------------------------------------------------

#: The **only** series diagnostics admitted into the plot-transform registry.
#:
#: The v6 scope surgery deleted the generic time-series layer (PSD, detrending,
#: filter design, entropy estimators, surrogates) on the principle *phase-space
#: methods stay; generic series statistics go*.  One member of that layer was
#: re-admitted by the owner, under a rule narrow enough to be checkable:
#:
#:     **The power spectrum of a phase-space trajectory is a phase-space
#:     diagnostic; a PSD toolbox with windowing options, detrending and filter
#:     design is not.**
#:
#: ``S(f)`` of an orbit is how the literature separates periodic (discrete
#: lines) from quasiperiodic (incommensurate lines) from chaotic (broadband)
#: motion, and a plotting module claiming completeness cannot omit it.  Nothing
#: else follows it in: the admitted set is this frozen constant, and
#: ``tests/test_viz_transforms.py`` fails if it grows or if any member of
#: :data:`EXCLUDED_SERIES_TOOLBOX` appears as a registered transform.  Without
#: that gate, ``spectrogram`` and ``detrend`` are back within two releases and
#: the scope decision is undone by accretion.
ADMITTED_SERIES_DIAGNOSTICS: frozenset[str] = frozenset({"psd"})

#: Names that must **never** be registered as plot transforms — the generic
#: time-series toolbox the v6 scope surgery removed.  They live in the companion
#: time-series library; a transform is not a back door into this one.
EXCLUDED_SERIES_TOOLBOX: frozenset[str] = frozenset(
    {
        "spectrogram",
        "detrend",
        "filter",
        "bandpass",
        "butterworth",
        "normalize",
        "hjorth_features",
        "extract_features",
        "permutation_entropy",
        "dispersion_entropy",
        "sample_entropy",
        "multiscale_entropy",
        "lz76_complexity",
        "surrogate_null",
        "surrogates",
        "surrogate_test",
        "time_reversal_asymmetry",
        "nonlinear_prediction_error",
    }
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _as_tuple(value: Any) -> tuple[Any, ...]:
    """Coerce a scalar-or-sequence declaration into a tuple."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)


def plot_transform(
    *,
    name: str,
    source: Source,
    default_primitive: str,
    primitives: Iterable[str],
    frame: FrameSpace | str | Sequence[FrameSpace | str],
    ndim: int | Sequence[int],
    kind: PlotKind | str | None = None,
    role: OverlayRole | str = OverlayRole.BASE,
    exclusive: Iterable[str] = (),
    requires: str | None = None,
    presentation: Presentation | None = None,
    analysis: str | None = None,
    example: ExampleFactory | None = None,
    doc: str = "",
    replace: bool = False,
) -> Callable[[Callable[..., Geometry]], Callable[..., Geometry]]:
    """Register a geometry-computing function as a named plot transform.

    This decorator **is** the extension point.  Adding a plot to TSDynamics is:
    write a function that returns a :class:`~tsdynamics.viz.transforms.Geometry`,
    decorate it with this, done.  The compatibility row, the presentation intent
    and the governance gate's example all ride on the same call, so nothing else
    in the library has to learn the new name — not the renderers, not the
    :class:`~tsdynamics.viz.spec.PlotKind` vocabulary, not ``viz.compose``, not
    the test suite.

    Parameters
    ----------
    name : str
        The registry key and the spelling used in ``ts.plot(subject, "<name>")``.
    source : {"data", "model"}
        See :data:`~tsdynamics.viz.transforms._base.Source`.  ``"model"`` if the
        function evaluates or integrates the right-hand side at a point that is
        not already in its input.
    default_primitive : str
        Used when the caller names none.  Must appear in ``primitives``.
    primitives : iterable of str
        **The declared compatibility row.**  Every primitive that may draw this
        transform's geometry; anything else raises.
    frame : FrameSpace or sequence
        The coordinate space(s) the geometry is drawn in.  A sequence for a
        transform whose space depends on the data (a portrait is ``state2`` or
        ``state3``).
    ndim : int or sequence of int
        The matching coordinate-axis count(s).
    kind : PlotKind or str, optional
        The semantic kind of the assembled spec.  Omit only when the geometry
        always supplies its own (``Geometry.kind``).
    role : OverlayRole or str, optional
        Draw order by meaning — ``field`` under ``base`` under ``overlay`` — so
        an overlay call is order-free.  Default ``base``.
    exclusive : iterable of str, optional
        The primitives in the row that are valid **only** here.
    requires : str, optional
        An optional-dependency distribution name.  The row is listed as
        *unavailable* (never omitted) when it is not installed.
    presentation : Presentation, optional
        Cross-backend presentation intent (aspect, colormap, colorbar policy).
    analysis : str, optional
        The dotted path of the analysis function this transform adapts.  **A
        transform is a thin adapter and owns no new math**: if you find yourself
        writing an estimator here, it belongs in :mod:`tsdynamics.analysis` with
        its own citation and tests first.
    example : callable, optional
        ``example(primitive) -> (subject, options)``.  A small, fast subject the
        compatibility gate builds and renders every declared cell on.  Required
        of in-tree transforms (a gate enforces it) so a declared pair that cannot
        draw fails CI without anyone editing a test file.
    doc : str, optional
        One line, shown by :func:`compatibility`.
    replace : bool, optional
        Overwrite an existing registration of the same name.

    Returns
    -------
    callable
        The undecorated function, unchanged — so a transform stays directly
        callable and unit-testable.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If the declared row names an unknown primitive, omits the default, marks
        an exclusive primitive that is not in the row, or claims a
        (primitive, coordinate-space) pair the primitive itself refuses.
    """
    from tsdynamics.errors import InvalidParameterError

    def decorator(fn: Callable[..., Geometry]) -> Callable[..., Geometry]:
        spaces = tuple(FrameSpace(s) for s in _as_tuple(frame))
        row = frozenset(str(p) for p in primitives)
        excl = frozenset(str(p) for p in exclusive)

        if name in EXCLUDED_SERIES_TOOLBOX:
            raise InvalidParameterError(
                f"{name!r} is part of the generic time-series toolbox the v6 scope surgery "
                "removed; it belongs in the companion time-series library, not in a plot "
                "transform. See ADMITTED_SERIES_DIAGNOSTICS for the one exception and the "
                "rule that admits it."
            )
        if default_primitive not in row:
            raise InvalidParameterError(
                f"transform {name!r} declares default_primitive={default_primitive!r}, which "
                f"is not in its row {sorted(row)}."
            )
        unknown = sorted(row - set(PRIMITIVES))
        if unknown:
            raise InvalidParameterError(
                f"transform {name!r} declares unknown primitive(s) {unknown}; the registered "
                f"primitives are {list(primitive_names())}."
            )
        stray = sorted(excl - row)
        if stray:
            raise InvalidParameterError(
                f"transform {name!r} marks {stray} exclusive but does not list them in its row."
            )
        for prim_name in sorted(row):
            prim = PRIMITIVES[prim_name]
            bad = [s.value for s in spaces if not prim.accepts_frame(s)]
            if bad:
                raise InvalidParameterError(
                    f"transform {name!r} claims primitive {prim_name!r}, which cannot draw in "
                    f"coordinate space(s) {bad} (it draws in "
                    f"{sorted(s.value for s in prim.frames or ())}). The declared matrix may "
                    "not contain a pair that is structurally impossible."
                )

        # The caller's keywords are split between ``compute`` and the primitive by
        # name, so a transform whose compute takes a parameter a primitive in its
        # row also names would be routed ambiguously.  Catch it at registration,
        # where it is a two-word fix, rather than at call time as a silently
        # dropped argument.
        import inspect

        prim_options = frozenset().union(*(PRIMITIVES[p].options for p in row))
        clash = sorted(prim_options & set(inspect.signature(fn).parameters))
        if clash:
            raise InvalidParameterError(
                f"transform {name!r} takes parameter(s) {clash}, which a primitive in its "
                f"row also accepts as an option; rename one so the keyword split is "
                "unambiguous."
            )

        summary = doc
        if not summary and fn.__doc__:
            summary = fn.__doc__.strip().splitlines()[0]

        record = PlotTransform(
            name=name,
            source=source,
            compute=fn,
            default_primitive=default_primitive,
            primitives=row,
            frame=spaces,
            role=OverlayRole[role.upper()] if isinstance(role, str) else OverlayRole(role),
            ndim=tuple(int(n) for n in _as_tuple(ndim)),
            exclusive=excl,
            requires=requires,
            doc=summary,
            kind=PlotKind(kind) if kind is not None else None,
            presentation=presentation or Presentation(),
            analysis=analysis,
            example=example,
        )
        _registry.plot_transforms.register(
            name,
            record,
            replace=replace,
            source=source,
            default_primitive=default_primitive,
            requires=requires,
        )
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def names() -> list[str]:
    """Return every registered transform name, in registration order."""
    return _registry.plot_transforms.names()


def get(name: str) -> PlotTransform:
    """Return the :class:`PlotTransform` registered as ``name``.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If no transform of that name is registered.  The message lists the
        registered names, because a name that is *almost* right
        (``"nullcline"`` for ``"nullclines"``) is the common case.
    """
    from tsdynamics.errors import InvalidParameterError

    try:
        record = _registry.plot_transforms.get(name)
    except KeyError:
        close = [n for n in names() if n.lower().startswith(name.lower()[:4])]
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise InvalidParameterError(
            f"unknown plot transform {name!r}.{hint} Registered transforms: {names()}."
        ) from None
    return record  # type: ignore[no-any-return]


def resolve(spec: str) -> tuple[PlotTransform, str | None]:
    """Split a ``"transform"`` / ``"transform.primitive"`` spelling.

    The dotted form is sugar, not a second mechanism:
    ``"phase_portrait.density"`` is exactly ``("phase_portrait",
    primitive="density")``, validated by the same row.
    """
    if "." in spec:
        head, _, tail = spec.partition(".")
        return get(head), tail
    return get(spec), None


def validate_primitive(
    transform: PlotTransform, primitive: str | None, *, geometry: Geometry | None = None
) -> str:
    """Return the primitive to use, checking it against ``transform``'s declared row.

    Resolution order: the caller's explicit choice, then the geometry's own
    data-dependent default (a map orbit is points, a flow is a line), then the
    transform's declared default.

    An invalid pair **raises** — never a fallback, never a warning.  The library
    already made this call once, when v6 deleted the accessors that silently drew
    something else: *a method that silently draws something else is worse than
    one that is absent*.  A renderer's ``VisualizationDegraded`` fallback is a
    different situation (the same plot, a different backend); here no correct
    drawing exists, so there is nothing to fall back to.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        Naming the valid set and — when the requested primitive is the default of
        some other transform — naming that transform, because that is almost
        always what the caller actually wanted.
    """
    from tsdynamics.errors import InvalidParameterError

    allowed = transform.primitives
    if geometry is not None and geometry.primitives is not None:
        allowed = allowed & geometry.primitives
    if primitive is None:
        fallback = geometry.primitive if geometry is not None else None
        primitive = fallback if fallback is not None else transform.default_primitive
    if primitive in allowed:
        return primitive
    if primitive in transform.primitives:
        raise InvalidParameterError(
            f"primitive {primitive!r} is valid for transform {transform.name!r}, but not for "
            f"*this* geometry (a {geometry.frame.describe() if geometry else '?'} shape); "
            f"here you can use: {', '.join(sorted(allowed))}."
        )
    raise InvalidParameterError(_invalid_primitive_message(transform, primitive))


def _invalid_primitive_message(transform: PlotTransform, primitive: str) -> str:
    """Compose the "not valid for this transform" message, with the useful hint."""
    valid = ", ".join(sorted(transform.primitives))
    text = f"primitive {primitive!r} is not valid for transform {transform.name!r}; valid: {valid}."
    if primitive not in PRIMITIVES:
        return (
            f"{text} ({primitive!r} is not a registered primitive at all; the registered "
            f"ones are {list(primitive_names())}.)"
        )
    owner = next(
        (t.name for t in _all_records() if t.default_primitive == primitive),
        None,
    )
    if owner is not None and owner != transform.name:
        return f"{text} ({primitive!r} is the default primitive of transform {owner!r}.)"
    others = sorted(t.name for t in _all_records() if primitive in t.primitives)
    if others:
        return f"{text} ({primitive!r} is valid for {others}.)"
    return text


def _all_records() -> list[PlotTransform]:
    """Every registered :class:`PlotTransform`, in registration order."""
    return [entry.obj for entry in _registry.plot_transforms.all()]


# ---------------------------------------------------------------------------
# Build path
# ---------------------------------------------------------------------------


def row_option_names(transform: PlotTransform) -> frozenset[str]:
    """Every option keyword any primitive in ``transform``'s row accepts.

    Splitting the caller's keywords by this union (rather than by the *chosen*
    primitive's options) keeps the split independent of which primitive wins —
    which matters because the winner can be decided by the geometry, i.e. after
    the options have already been routed.  Registration rejects a transform whose
    ``compute`` takes a parameter of the same name, so the union is unambiguous.
    """
    return frozenset().union(*(PRIMITIVES[p].options for p in transform.primitives))


def _split_options(
    transform: PlotTransform, options: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split caller options into ``(compute_kwargs, primitive_kwargs)``.

    A keyword any primitive in the transform's row declares (``bins``,
    ``levels``) goes to the primitive; everything else goes to ``compute``.  The
    explicit ``primitive_options={...}`` escape wins over the automatic split,
    for the day a transform and a primitive genuinely want one keyword name.
    """
    explicit = dict(options.pop("primitive_options", None) or {})
    prim_names = row_option_names(transform)
    compute_kw: dict[str, Any] = {}
    prim_kw: dict[str, Any] = {}
    for key, value in options.items():
        (prim_kw if key in prim_names else compute_kw)[key] = value
    prim_kw.update(explicit)
    return compute_kw, prim_kw


def geometry(subject: Any, name: str, /, **options: Any) -> Geometry:
    """Compute a transform's :class:`Geometry` and stop there — the raw-arrays escape hatch.

    The rung researchers actually want: get the numbers, look at them, do your
    own thing with them.  ``ts.viz.draw(g, "image")`` hands one back to the
    library.

    Parameters
    ----------
    subject : Any
        A :class:`~tsdynamics.data.Trajectory`, a system, an analysis result —
        whatever this transform accepts.
    name : str
        The transform name (the ``"transform.primitive"`` spelling is accepted;
        the primitive part is ignored here, since no drawing happens).
    **options
        Forwarded to the transform's ``compute``.

    Returns
    -------
    Geometry
    """
    transform, _ = resolve(name)
    _check_available(transform)
    result = _compute(transform, subject, options)
    _check_geometry(transform, result)
    return result


#: Integration keywords the fallback may forward when it runs a system for a
#: ``data`` transform.  Deliberately the run keywords and nothing else: a
#: transform's own options must not be swallowed by the coercion.
_RUN_KEYS: frozenset[str] = frozenset(
    {"final_time", "dt", "steps", "ic", "transient", "seed", "method", "backend", "rtol", "atol"}
)


def _compute(transform: PlotTransform, subject: Any, options: dict[str, Any]) -> Geometry:
    """Call ``compute``, running the system first if that is the only way it works.

    The library declares exactly two source categories, and the asymmetry between
    them is the point: a ``model`` transform *needs* the right-hand side, while a
    ``data`` transform needs only samples — and therefore accepts a system too,
    because having the model is having the data.

    That promise was not being kept: thirteen of the twenty-two registered
    ``data`` transforms refused a system outright, ``time_series`` and
    ``phase_portrait`` among them.

    The fix is a **fallback, not a pre-conversion**, and the distinction matters.
    Fourteen of those transforms already handle a system themselves, and better
    than a generic rule could — they iterate a map rather than integrating it,
    they run their own tangent-space analysis, they record what they ran in
    ``meta["integrated_for_plot"]``.  Converting up front would have taken that
    away and replaced it with something cruder.  So the transform gets first
    refusal, and only if it cannot cope does the registry run the system and try
    once more.  A new transform inherits the behaviour without declaring
    anything, which is what "one registration and nothing else" has to mean.

    If the retry also fails, the transform's ORIGINAL error is raised: it knows
    what it wanted, and a message about a failed coercion would bury that.
    """
    try:
        return transform.compute(subject, **options)
    except (TypeError, ValueError) as first:
        if transform.source != "data":
            raise
        from tsdynamics.families import SystemBase

        if not isinstance(subject, SystemBase):
            raise
        run_kw = {k: options.pop(k) for k in list(options) if k in _RUN_KEYS}
        try:
            return transform.compute(subject.trajectory(**run_kw), **options)
        except Exception:
            raise first from None


def lower(geom: Geometry, primitive: str | None = None, /, **primitive_options: Any) -> list[Layer]:
    """Lower a geometry to :class:`~tsdynamics.viz.spec.Layer` objects via one primitive.

    Each :class:`~tsdynamics.viz.transforms._base.Part` is drawn by the chosen
    primitive, except a part that pins its own (a vector field's host orbit is a
    line no matter how the field is drawn).
    """
    from tsdynamics.errors import InvalidParameterError

    transform = get(geom.transform)
    chosen = validate_primitive(transform, primitive, geometry=geom)
    prim = get_primitive(chosen)
    unknown = sorted(set(primitive_options) - prim.options)
    if unknown:
        raise InvalidParameterError(
            f"primitive {chosen!r} does not accept keyword(s) {unknown}; "
            f"it accepts {sorted(prim.options) or '(none)'}."
        )
    layers: list[Layer] = []
    for part in geom.parts:
        this = get_primitive(part.primitive) if part.primitive is not None else prim
        missing = sorted(this.requires - set(part.channels))
        if missing:
            raise InvalidParameterError(
                f"primitive {this.name!r} needs channel(s) {missing}, which the geometry of "
                f"transform {geom.transform!r} does not carry (it has "
                f"{sorted(part.channels)}). This is a declared-row bug, not a user error."
            )
        layers.extend(this.build(geom, part, primitive_options if this is prim else {}))
    return layers


def draw(geom: Geometry, primitive: str | None = None, /, **primitive_options: Any) -> PlotSpec:
    """Turn a :class:`Geometry` into a renderable :class:`~tsdynamics.viz.spec.PlotSpec`.

    The second half of the escape hatch: ``ts.viz.geometry(...)`` gets the
    numbers, this hands them back to the library with a chosen primitive.
    """
    transform = get(geom.transform)
    layers = lower(geom, primitive, **primitive_options)
    return spec_of(geom, transform, layers)


def build_spec(
    subject: Any, name: str, /, *, primitive: str | None = None, **options: Any
) -> PlotSpec:
    """Compute *and* draw — the one path every front door goes through.

    Parameters
    ----------
    subject : Any
        What to transform.
    name : str
        The transform name, or the ``"transform.primitive"`` dotted sugar.
    primitive : str, optional
        How to draw it.  Validated against the transform's declared row; an
        invalid pair raises.  ``None`` uses the transform's default.
    **options
        Split automatically: a keyword the chosen primitive declares goes to the
        primitive, everything else to the transform's ``compute``.  Force the
        split with ``primitive_options={...}``.

    Returns
    -------
    PlotSpec
    """
    transform, dotted = resolve(name)
    if dotted is not None and primitive is not None and dotted != primitive:
        from tsdynamics.errors import InvalidParameterError

        raise InvalidParameterError(
            f"{name!r} names primitive {dotted!r} but primitive={primitive!r} was also "
            "passed; use one spelling."
        )
    wanted = primitive if dotted is None else dotted
    if wanted is not None:  # fail on an invalid pair *before* doing any work
        validate_primitive(transform, wanted)
    compute_kw, prim_kw = _split_options(transform, dict(options))
    geom = geometry(subject, transform.name, **compute_kw)
    return draw(geom, wanted, **prim_kw)


def _check_available(transform: PlotTransform) -> None:
    """Raise a clear, actionable error when an optional dependency is missing."""
    if transform.available:
        return
    from tsdynamics.analysis._result_viz import VisualizationNotInstalled

    raise VisualizationNotInstalled(
        f"transform {transform.name!r} needs the optional package {transform.requires!r}; "
        f"install it (pip install {transform.requires}) to use this plot. "
        "It is listed as unavailable rather than hidden, because a missing row reads as "
        "'this plot does not exist'."
    )


def _check_geometry(transform: PlotTransform, result: Any) -> None:
    """Raise unless ``compute`` returned a well-formed :class:`Geometry`."""
    from tsdynamics.errors import InvalidInputError, InvalidParameterError

    if not isinstance(result, Geometry):
        raise InvalidInputError(
            f"transform {transform.name!r} returned {type(result).__name__}, not a Geometry. "
            "A transform computes geometry; the primitive step builds the PlotSpec."
        )
    if result.transform != transform.name:
        raise InvalidParameterError(
            f"transform {transform.name!r} returned a geometry stamped "
            f"{result.transform!r}; the stamp is the provenance every layer carries."
        )
    if result.frame.space not in transform.frame:
        raise InvalidParameterError(
            f"transform {transform.name!r} declares coordinate space(s) "
            f"{[s.value for s in transform.frame]} but returned {result.frame.space.value!r}."
        )


# ---------------------------------------------------------------------------
# Per-transform options inside an overlay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformCall:
    """One transform plus the options it should be built with — what :func:`T` returns.

    Inside an overlay every transform would otherwise have to share one keyword
    namespace, which is how ``grid=`` ends up meaning two things at once::

        ts.plot(vdp, T("flow_speed", log=True, alpha=0.55),
                     T("streamlines", seeds=8, color="w"),
                     T("nullclines", linewidth=2.0))

    Style keywords (the canonical
    :data:`~tsdynamics.viz.style.STYLE_KEYS` vocabulary) are split out and
    applied to that transform's layers, so a per-source restyle needs no second
    call.
    """

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)
    primitive: str | None = None

    def __repr__(self) -> str:  # noqa: D105
        opts = ", ".join(f"{k}={v!r}" for k, v in self.options.items())
        prim = f", primitive={self.primitive!r}" if self.primitive else ""
        return f"T({self.name!r}{prim}{', ' + opts if opts else ''})"


def T(name: str, /, *, primitive: str | None = None, **options: Any) -> TransformCall:  # noqa: N802
    """Name a transform **with its own options**, for use inside a composition.

    ``ts.plot(subject, "flow_speed", "streamlines")`` builds both with the
    defaults; ``ts.plot(subject, T("flow_speed", log=True), T("streamlines",
    seeds=8))`` gives each its own.  The dotted primitive sugar works here too:
    ``T("phase_portrait.density")``.

    The options are checked against the named transform when :func:`ts.plot
    <tsdynamics.viz.transforms.plot>` builds the figure — the same check the
    shared keywords get — so a typo (``T("phase_portrait", nonsense=1)``) is
    answered with the keywords that *are* accepted and a did-you-mean, rather
    than with a bare ``TypeError`` raised from inside the compute.

    Parameters
    ----------
    name : str
        The transform name, or ``"transform.primitive"``.
    primitive : str, optional
        How to draw this one.  Validated against its declared row.
    **options
        The transform's compute options, its chosen primitive's options, and any
        canonical style keys (``color``, ``linewidth``, ``alpha``, …), which are
        applied to this transform's layers only.

    Returns
    -------
    TransformCall
    """
    if "." in name and primitive is None:
        name, _, primitive = name.partition(".")
    return TransformCall(name=name, options=dict(options), primitive=primitive)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def transforms(
    *, source: Source | None = None, available: bool | None = None
) -> list[PlotTransform]:
    """List the registered transforms, optionally filtered.

    Parameters
    ----------
    source : {"data", "model"}, optional
        Keep only one source category.
    available : bool, optional
        Keep only transforms whose optional dependency is (or is not) installed.
        ``None`` (default) lists everything — an unavailable transform is
        **listed as unavailable, never omitted**.
    """
    out = _all_records()
    if source is not None:
        out = [t for t in out if t.source == source]
    if available is not None:
        out = [t for t in out if t.available is available]
    return out


class CompatibilityMatrix(dict):  # type: ignore[type-arg]
    """The declared matrix: ``transform -> (primitive, …)``, with a readable repr.

    A plain ``dict`` subclass, so it is programmable (``m["phase_portrait"]``,
    ``pandas.DataFrame(m.rows())``) *and* prints as a table when you just look at
    it.  Reading marks: ``*`` the default primitive, ``!`` exclusive to this row.
    """

    def rows(self) -> list[dict[str, Any]]:
        """Return one record per transform — the DataFrame-able form."""
        return [
            {
                "transform": t.name,
                "source": t.source,
                "space": "/".join(s.value for s in t.frame),
                "default": t.default_primitive,
                "primitives": sorted(t.primitives),
                "exclusive": sorted(t.exclusive),
                "available": t.available,
                "requires": t.requires,
                "doc": t.doc,
            }
            for t in transforms()
        ]

    def __repr__(self) -> str:  # noqa: D105
        if not self:
            return "CompatibilityMatrix(empty)"
        width = max(len(k) for k in self)
        lines = [f"{'transform'.ljust(width)}  primitives  (* default, ! exclusive)"]
        for name, row in sorted(self.items()):
            record = get(name)
            flag = "" if record.available else f"   [unavailable: needs {record.requires}]"
            lines.append(f"{name.ljust(width)}  {', '.join(row)}{flag}")
        return "\n".join(lines)


def compatibility(name: str | None = None) -> Any:
    """Return the declared compatibility matrix — the whole thing, or one row.

    ``compatibility()`` is a printable, DataFrame-able mapping of every
    transform to its valid primitives; ``compatibility("phase_portrait")`` is
    that one row.  Reading marks: ``*`` marks the default primitive, ``!`` marks a
    primitive that is exclusive to that row.

    A transform whose optional dependency is missing is **listed, flagged
    unavailable** — omitting it would read as "that plot does not exist", which
    is a different and less useful statement than "install the extra".

    Parameters
    ----------
    name : str, optional
        One transform.  ``None`` returns the whole matrix.

    Returns
    -------
    CompatibilityMatrix or tuple of str
    """
    if name is not None:
        return get(name).describe_primitives()
    return CompatibilityMatrix({t.name: t.describe_primitives() for t in transforms()})
