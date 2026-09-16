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
from .._frames import FrameSpace, OverlayRole, space_arity
from ..spec import PlotKind
from ._base import (
    PART_KEYS,
    ExampleFactory,
    Geometry,
    Part,
    PlotTransform,
    Presentation,
    Source,
    make_frame,
    part_from_mapping,
    parts_from_return,
    spec_of,
)
from ._primitives import PRIMITIVES, get_primitive, primitive_names

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..spec import Layer, PlotSpec

__all__ = [
    "ADMITTED_SERIES_DIAGNOSTICS",
    "EXCLUDED_SERIES_TOOLBOX",
    "PART_KEYS",
    "T",
    "TransformCall",
    "build_spec",
    "compatibility",
    "draw",
    "find",
    "geometry",
    "get",
    "lower",
    "names",
    "part_from_mapping",
    "plot_transform",
    "record_for",
    "register",
    "row_option_names",
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


def register(
    *,
    source: Source,
    frame: FrameSpace | str | Sequence[FrameSpace | str],
    primitives: Iterable[str],
    kind: PlotKind | str | None = None,
    name: str | None = None,
    default_primitive: str | None = None,
    ndim: int | Sequence[int] | None = None,
    role: OverlayRole | str = OverlayRole.BASE,
    aliases: Iterable[str] = (),
    requires: str | None = None,
    presentation: Presentation | None = None,
    analysis: str | None = None,
    example: ExampleFactory | None = None,
    labels: Sequence[str] = (),
    doc: str = "",
    replace: bool = False,
) -> Callable[[Callable[..., Geometry]], Callable[..., Geometry]]:
    """Register a geometry-computing function as a named plot transform.

    **Four declarations.  Everything else is derived.**

    .. code-block:: python

        import numpy as np, tsdynamics as ts

        @ts.viz.transforms.register(source="data", frame="time", kind="diagnostic_curve",
                                    primitives=("line", "points", "steps"))
        def speed(traj):
            '''Instantaneous speed |dx/dt| along the orbit.'''
            dt = np.diff(traj.t)
            return {"x": traj.t[1:],
                    "y": np.linalg.norm(np.diff(traj.y, axis=0), axis=1) / dt}

    That is the whole extension point, and it buys — with **no other edit
    anywhere** — ``ts.plot(traj, "speed")``, ``primitive="steps"``,
    ``traj.plot.speed()``, a row in :func:`tsdynamics.viz.compatibility`, an
    entry in ``ts.viz.transforms.names()``, a generated gallery figure, and every
    declared cell rendered by the compatibility gate.

    Derived, never declared: ``name`` from ``fn.__name__``, ``doc`` from the
    docstring's first line, **``ndim`` from the frame's arity**
    (:func:`~tsdynamics.viz._frames.space_arity`), ``default_primitive`` from the
    first entry of ``primitives``, and :attr:`~PlotTransform.subjects` from
    ``source``.  (Axis *labels* are deliberately **not** derived from the channel
    names: a ``{"x": t, "y": speed}`` return would label the time axis ``x``,
    which is a wrong label rather than a missing one.  Pass ``labels=`` when the
    axes have names.)

    Parameters
    ----------
    source : {"data", "model"}
        See :data:`~tsdynamics.viz.transforms._base.Source`.  ``"model"`` if the
        function evaluates or integrates the right-hand side at a point that is
        not already in its input — the registry then *enforces* it, refusing a
        bare array with a message naming what the transform needs.
    frame : FrameSpace or sequence
        The coordinate space(s) the geometry is drawn in.  A sequence for a
        transform whose space depends on the data (a portrait is ``state2`` or
        ``state3``).
    primitives : iterable of str
        **The declared compatibility row**, best first.  Every primitive that may
        draw this transform's geometry; anything else raises.
    kind : PlotKind or str, optional
        The semantic kind of the assembled spec.  **Required**, except for a
        transform declaring several frames — whose geometry must supply its own,
        since the kind is exactly what varies.  (It was de jure optional and de
        facto mandatory before v6: the authoring template in the shipped gallery
        registered fine and then failed at *plot* time, on a user's machine.)
    name : str, optional
        The registry key and the spelling used in ``ts.plot(subject, "<name>")``.
        Defaults to the decorated function's ``__name__``.
    default_primitive : str, optional
        Used when the caller names none.  Defaults to the first of ``primitives``.
    ndim : int or sequence of int, optional
        The coordinate-axis count(s).  Derived from ``frame``; pass it only for a
        geometry whose shape varies *within* one space (a spatial field is a 2-D
        lattice or a 1-D profile).
    role : OverlayRole or str, optional
        Draw order by meaning — ``field`` under ``base`` under ``overlay`` — so
        an overlay call is order-free.  Default ``base``.
    aliases : iterable of str, optional
        Extra names resolving to this transform (``direction_field`` →
        ``vector_field``).
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
    labels : sequence of str, optional
        The axis labels to stamp when ``compute`` returns a plain **mapping of
        channels** (see below).  Ignored when it returns a ``Geometry``, which
        carries its own.
    doc : str, optional
        One line, shown by :func:`compatibility`.  Defaults to the docstring's
        first line.
    replace : bool, optional
        Overwrite an existing registration of the same name.

    Returns
    -------
    callable
        The undecorated function, unchanged — so a transform stays directly
        callable and unit-testable.

    Notes
    -----
    **``compute`` may return a plain mapping of channels, or a list of them.**
    The name, the coordinate space and the labels are declared *here*, so a
    transform that repeats them in a hand-built :class:`Geometry` declares each
    of them twice and the registry's only contribution is to check the author
    against themselves.  A list gives one :class:`Part` per mapping, reading four
    reserved keys — ``label``, ``style``, ``primitive``, ``mark`` — with
    everything else a channel::

        return [{"x": r, "y": c, "label": "data"},
                {"x": r, "y": fit, "label": "fit", "style": {"linestyle": "dashed"}}]

    With this, **no transform author ever needs an IR type**.  Build a
    :class:`Geometry` when the frame or the kind depends on the subject.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``source`` is not one of the two categories, if ``kind`` is missing
        where it is required, if the declared row names an unknown primitive or
        omits the default, or if it claims a (primitive, coordinate-space) pair
        the primitive itself refuses.
    """
    from tsdynamics.errors import InvalidParameterError

    def decorator(fn: Callable[..., Geometry]) -> Callable[..., Geometry]:
        # -- derive what was not declared -----------------------------------
        key = name if name is not None else fn.__name__
        spaces = tuple(FrameSpace(s) for s in _as_tuple(frame))
        ordered = tuple(str(p) for p in primitives)
        row = frozenset(ordered)
        chosen_default = default_primitive if default_primitive is not None else ordered[0]
        arities = (
            tuple(int(n) for n in _as_tuple(ndim))
            if ndim is not None
            else tuple(space_arity(s) for s in spaces)
        )

        if source not in ("data", "model"):
            raise InvalidParameterError(
                f"transform {key!r} declares source={source!r}; there are exactly two "
                "categories — 'data' (computable from the samples you have) and 'model' "
                "(must evaluate or integrate the right-hand side somewhere new)."
            )
        if kind is None and len(spaces) == 1:
            raise InvalidParameterError(
                f"transform {key!r} declares no kind=. The semantic kind is what the "
                "renderers dispatch on, so a transform without one registers cleanly and "
                "then fails at PLOT time, on a user's machine. Declare one of "
                f"{[k.value for k in PlotKind]}; only a transform declaring several frames "
                "may leave it to its geometry."
            )
        if key in EXCLUDED_SERIES_TOOLBOX:
            raise InvalidParameterError(
                f"{key!r} is part of the generic time-series toolbox the v6 scope surgery "
                "removed; it belongs in the companion time-series library, not in a plot "
                "transform. See ADMITTED_SERIES_DIAGNOSTICS for the one exception and the "
                "rule that admits it."
            )
        if chosen_default not in row:
            raise InvalidParameterError(
                f"transform {key!r} declares default_primitive={chosen_default!r}, which "
                f"is not in its row {sorted(row)}."
            )
        if len(arities) != len(spaces) and len(spaces) != 1:
            # One arity per space, or — for the one case ``ndim=`` survives for —
            # several arities within a single space, because the geometry's SHAPE
            # varies there (a spatial field is a 2-D lattice or a 1-D profile).
            raise InvalidParameterError(
                f"transform {key!r} declares {len(spaces)} coordinate space(s) but "
                f"{len(arities)} ndim value(s); give one per space, or omit ndim entirely "
                "and let the space decide."
            )
        unknown = sorted(row - set(PRIMITIVES))
        if unknown:
            raise InvalidParameterError(
                f"transform {key!r} declares unknown primitive(s) {unknown}; the registered "
                f"primitives are {list(primitive_names())}."
            )
        for prim_name in sorted(row):
            prim = PRIMITIVES[prim_name]
            bad = [s.value for s in spaces if not prim.accepts_frame(s)]
            if bad:
                raise InvalidParameterError(
                    f"transform {key!r} claims primitive {prim_name!r}, which cannot draw in "
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
                f"transform {key!r} takes parameter(s) {clash}, which a primitive in its "
                f"row also accepts as an option; rename one so the keyword split is "
                "unambiguous."
            )

        summary = doc
        if not summary and fn.__doc__:
            summary = fn.__doc__.strip().splitlines()[0]

        record = PlotTransform(
            name=key,
            source=source,
            compute=fn,
            default_primitive=chosen_default,
            primitives=row,
            frame=spaces,
            role=OverlayRole[role.upper()] if isinstance(role, str) else OverlayRole(role),
            ndim=arities,
            requires=requires,
            doc=summary,
            kind=PlotKind(kind) if kind is not None else None,
            presentation=presentation or Presentation(),
            analysis=analysis,
            example=example,
            labels=tuple(str(label) for label in labels),
            aliases=tuple(str(a) for a in aliases),
        )
        _registry.plot_transforms.register(
            key,
            record,
            replace=replace,
            source=source,
            default_primitive=chosen_default,
            requires=requires,
        )
        for alias in record.aliases:
            _ALIASES[alias] = key
        return fn

    return decorator


#: ``alias -> registered name``.  An alias is a second *spelling*, never a second
#: record: ``ts.plot(sys, "direction_field")`` and ``"vector_field"`` build the
#: identical geometry, and the matrix lists one row.
_ALIASES: dict[str, str] = {}

#: The pre-v6 spelling of :func:`register`.  The same function object — the
#: in-tree transforms were written against it and it is what the published
#: authoring recipe named.
plot_transform = register


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def names() -> list[str]:
    """Return every registered transform name, in registration order."""
    return _registry.plot_transforms.names()


def get(name: str) -> PlotTransform:
    """Return the :class:`PlotTransform` registered as ``name`` (or as one of its aliases).

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If no transform of that name is registered.  The message lists the
        registered names, because a name that is *almost* right
        (``"nullcline"`` for ``"nullclines"``) is the common case.
    """
    import difflib

    from tsdynamics.errors import InvalidParameterError

    try:
        record = _registry.plot_transforms.get(_ALIASES.get(name, name))
    except (KeyError, ValueError):
        close = difflib.get_close_matches(name, names(), n=1, cutoff=0.5) or [
            n for n in names() if n.lower().startswith(name.lower()[:4])
        ]
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise InvalidParameterError(
            f"unknown plot transform {name!r}.{hint} Registered transforms: {names()}."
        ) from None
    return record  # type: ignore[no-any-return]


def find(
    what: str | None = None,
    /,
    *,
    subject: Any = None,
    source: Source | None = None,
    frame: FrameSpace | str | None = None,
    primitive: str | None = None,
    available: bool | None = None,
) -> list[str]:
    """Answer *"what can I draw?"* — the names matching every filter given.

    The fourth shared registry verb (``register`` / ``names`` / ``find`` /
    ``get``).  ``subject=`` is the **user's** question, and the one worth typing::

        ts.viz.transforms.find(subject=traj)     # what can I draw from THIS?
        ts.viz.transforms.find(subject=lorenz)
        ts.viz.transforms.find("spectrum")       # free text over name + summary
        ts.viz.transforms.find(source="model")   # the author's question
        ts.viz.transforms.find(primitive="contour")

    Parameters
    ----------
    what : str, optional
        Free text matched against the transform name and its one-line summary.
    subject : Any, optional
        A trajectory / system / array.  Keeps the transforms whose declared
        :attr:`~PlotTransform.subjects` admit it.
    source : {"data", "model"}, optional
        Keep one source category.
    frame : FrameSpace or str, optional
        Keep the transforms that draw in that coordinate space.
    primitive : str, optional
        Keep the transforms whose declared row contains it.
    available : bool, optional
        Keep the transforms whose optional dependency is (or is not) installed.

    Returns
    -------
    list of str
        The matching names, sorted.
    """
    text = (what or "").lower()
    space = FrameSpace(frame) if frame is not None else None
    out = []
    for record in _all_records():
        if text and text not in record.name.lower() and text not in record.doc.lower():
            continue
        if subject is not None and not record.accepts_subject(subject):
            continue
        if source is not None and record.source != source:
            continue
        if space is not None and space not in record.frame:
            continue
        if primitive is not None and primitive not in record.primitives:
            continue
        if available is not None and record.available is not available:
            continue
        out.append(record.name)
    return sorted(out)


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
    return _check_geometry(transform, _stamp(transform, _compute(transform, subject, options)))


#: Integration keywords the fallback may forward when it runs a system for a
#: ``data`` transform.  Deliberately the run keywords and nothing else: a
#: transform's own options must not be swallowed by the coercion.
_RUN_KEYS: frozenset[str] = frozenset(
    {"final_time", "dt", "steps", "ic", "transient", "seed", "method", "backend", "rtol", "atol"}
)


def _compute(transform: PlotTransform, subject: Any, options: dict[str, Any]) -> Any:
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

    The ``model`` half is **enforced here rather than by a hand-written guard per
    transform**, because the hand-written guards did not exist: handed a bare
    array, a model transform used to raise ``AttributeError: 'numpy.ndarray'
    object has no attribute 'jacobian'`` from somewhere inside its own numerics.
    """
    if transform.source == "model" and not transform.accepts_subject(subject):
        from tsdynamics.errors import InvalidInputError

        raise InvalidInputError(
            f"transform {transform.name!r} needs a dynamical system: it evaluates the "
            f"right-hand side at points that are not in your data, and a "
            f"{type(subject).__name__} cannot be asked for them. Pass the system "
            f"(ts.plot(lorenz, {transform.name!r})), or pick a 'data' transform — "
            f"ts.viz.transforms.find(subject=your_data) lists them."
        )
    from tsdynamics.families import SystemBase as _SystemBase

    if transform.source == "data" and isinstance(subject, _SystemBase):
        # Peel the run vocabulary the transform does not itself declare, and run
        # the system with it.  ``psd`` names ``final_time``/``dt``/``steps`` in
        # its own signature and therefore keeps them (unchanged); ``time_series``
        # does not, and used to REFUSE them at the front door rather than
        # integrate.  Doing it HERE rather than only in the exception fallback
        # below is what lets the transform answer in its own words when the run
        # is not the problem — the fallback re-raises the transform's first
        # error, which for a run keyword is always "unexpected keyword argument".
        import inspect

        declared = set(inspect.signature(transform.compute).parameters)
        run_kw = {k: options.pop(k) for k in list(options) if k in _RUN_KEYS - declared}
        if run_kw:
            try:
                subject = subject.run(**run_kw)
            except Exception:
                options.update(run_kw)  # let the transform (or the error) speak
    try:
        return transform.compute(subject, **options)
    except (TypeError, ValueError) as first:
        if transform.source != "data":
            raise
        from tsdynamics.data.trajectory import Trajectory, as_trajectory
        from tsdynamics.families import SystemBase

        if isinstance(subject, SystemBase):
            run_kw = {k: options.pop(k) for k in list(options) if k in _RUN_KEYS}
            try:
                return transform.compute(subject.run(**run_kw), **options)
            except Exception:
                raise first from None
        # The same fallback, for measured data: a transform that wants a
        # trajectory gets one built from the caller's arrays.  Some transforms
        # (``psd``) already read a bare series and never reach here; the two a
        # newcomer tries first (``time_series``, ``phase_portrait``) did not,
        # so "a subject may be a bare array" was true of some transforms and
        # not others, with no way to tell which.
        if isinstance(subject, Trajectory):
            raise
        try:
            coerced = as_trajectory(subject, dt=options.pop("dt", None))
        except Exception:
            raise first from None
        try:
            return transform.compute(coerced, **options)
        except Exception:
            raise first from None


def record_for(geom: Geometry) -> PlotTransform:
    """Return the :class:`PlotTransform` behind ``geom`` — or an ad-hoc one for hand-built arrays.

    **``Geometry.transform`` is provenance, not a lookup key.**  It is the stamp
    every layer carries so per-source restyling works; it used to double as the
    registry key, which closed the layer in both directions — you could not hand
    the library a geometry you built yourself without first registering a fake
    transform for it.  A geometry whose stamp names no registered transform now
    gets a record that allows **every** primitive and declares its own frame:
    there is no declared row to violate, because nobody declared one.
    """
    from tsdynamics.errors import InvalidParameterError

    try:
        return get(geom.transform)
    except InvalidParameterError:
        return PlotTransform(
            name=geom.transform,
            source="data",
            compute=lambda subject, **kw: geom,
            default_primitive="line",
            primitives=frozenset(PRIMITIVES),
            frame=(geom.frame.space,),
            role=OverlayRole.BASE,
            ndim=(geom.frame.ndim,),
            doc="hand-built geometry",
        )


def lower(geom: Geometry, primitive: str | None = None, /, **primitive_options: Any) -> list[Layer]:
    """Lower a geometry to :class:`~tsdynamics.viz.spec.Layer` objects via one primitive.

    Each :class:`~tsdynamics.viz.transforms._base.Part` is drawn by the chosen
    primitive, except a part that pins its own (a vector field's host orbit is a
    line no matter how the field is drawn).
    """
    from tsdynamics.errors import InvalidParameterError

    transform = record_for(geom)
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
        built = this.build(geom, part, primitive_options if this is prim else {})
        _check_colour_survived(geom, this, part, built)
        layers.extend(built)
    return layers


def _check_colour_survived(geom: Geometry, prim: Any, part: Part, layers: Sequence[Layer]) -> None:
    """Raise when a primitive silently **discards** a part's colour channel.

    Measured before v6: ``ts.plot(traj, "phase_portrait.steps")`` on a
    colour-by-time portrait dropped the ``c`` channel *and kept the colorbar* —
    a figure with a legend for a dimension it is not drawing, at no warning
    level.  A primitive that cannot colour per-vertex is a legitimate primitive;
    being handed colour it will throw away is a user error, and it has a
    one-word fix.
    """
    from tsdynamics.errors import InvalidParameterError

    if not prim.consumes or prim.consumes & {"c", "z"}:
        # Undeclared ``consumes`` means the primitive reshapes its input wholesale
        # (``density`` bins it, ``contour`` colours by level) and owns its own
        # colour story.  A primitive that consumes ``z`` draws the same scalar as
        # height, so nothing was lost either.  What is left is the real case:
        # a primitive that draws neither, handed a colour channel.
        return
    if "c" not in part.channels or any("c" in layer.data for layer in layers):
        return
    keeps = sorted(name for name, other in PRIMITIVES.items() if "c" in other.consumes)
    raise InvalidParameterError(
        f"primitive {prim.name!r} cannot draw the colour channel that transform "
        f"{geom.transform!r} computed, so the colour would be silently dropped. Either "
        f"draw it with one that can ({', '.join(keeps)}), or drop the colour "
        "(color_by=None)."
    )


def draw(
    data: Geometry | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    primitive: str | None = None,
    /,
    *,
    labels: Sequence[str] = (),
    title: str = "",
    kind: PlotKind | str | None = None,
    frame: FrameSpace | str = FrameSpace.FREE,
    **primitive_options: Any,
) -> PlotSpec:
    """**Hand arrays to a primitive and get a `Plot` back** — no transform required.

    The other half of the escape hatch (:func:`geometry` gets the numbers, this
    takes numbers back), and the door for the plot the library does not have::

        ts.viz.draw({"x": r, "y": C}, "line", labels=("log r", "log C(r)"))

        ts.viz.draw([{"x": r, "y": C,   "label": "data"},
                     {"x": r, "y": fit, "label": "fit",
                      "style": {"linestyle": "dashed"}},
                     {"x": r, "y": lo, "y2": hi, "primitive": "band",
                      "style": {"alpha": 0.2}}],
                    "line", labels=("log r", "log C(r)"), title="correlation sum")

    Because it returns a :class:`~tsdynamics.viz.spec.Plot`, it **composes with
    everything** — that is closure, and it is why this is a function rather than
    a subsystem::

        ts.plot(traj, "phase_portrait") + ts.viz.draw({"x": xs, "y": ys}, "line")
        ts.viz.grid(ts.viz.draw({"x": r, "y": C}, "points"), ts.plot(traj, "psd"))

    Parameters
    ----------
    data : Geometry, mapping, or sequence of mappings
        A mapping of **channel name → array** (``x`` / ``y`` / ``z`` / ``c`` /
        ``u`` / ``v`` / ``lo`` / ``hi`` / ``err`` / …), optionally with the four
        reserved keys ``label`` / ``style`` / ``primitive`` (or ``mark``); a list
        of such mappings for several pieces; or a :class:`Geometry` you built.
    primitive : str, optional
        How to draw it — ``ts.viz.primitives.names()`` lists them.  Defaults to
        ``"line"`` for a mapping (and to the geometry's own default otherwise).
    labels : sequence of str, optional
        Axis labels, in axis order.
    title : str, optional
        Figure title.
    kind : PlotKind or str, optional
        The semantic kind to stamp.  Defaults to the primitive's own mark, which
        is the honest answer for arrays that carry no further meaning.
    frame : FrameSpace or str, optional
        The coordinate space to claim.  Defaults to
        :data:`~tsdynamics.viz._frames.FrameSpace.FREE` — *"I did not say"* —
        which overlays with anything, because a caller who hand-built the arrays
        made no coordinate claim to violate.
    **primitive_options
        Forwarded to the primitive (``bins=``, ``levels=``, …).

    Returns
    -------
    Plot
    """
    from dataclasses import replace as _replace

    geom = data if isinstance(data, Geometry) else _hand_built(data, frame, labels, title)
    transform = record_for(geom)
    layers = lower(geom, primitive, **primitive_options)

    # A hand-built geometry carries no semantic kind and no transform declared one
    # for it, so the honest kind is the mark the chosen primitive emitted.
    if kind is None and geom.kind is None and transform.kind is None:
        transform = _replace(
            transform, kind=PlotKind(str(layers[0].kind)) if layers else PlotKind.LINE
        )
    spec = spec_of(geom, transform, layers)
    if kind is not None:
        spec.kind = PlotKind(kind)
    return spec


def _hand_built(
    data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    frame: FrameSpace | str,
    labels: Sequence[str],
    title: str,
) -> Geometry:
    """Build a hand-arrays :class:`Geometry` from a mapping / list of mappings."""
    from tsdynamics.errors import InvalidInputError

    parts = parts_from_return(data)
    if parts is None:
        raise InvalidInputError(
            "draw() takes a channel mapping ({'x': …, 'y': …}), a list of them, or a "
            f"Geometry; got {type(data).__name__}. For a trajectory / system / result, use "
            "ts.plot(subject, ...)."
        )
    space = FrameSpace(frame)
    return Geometry(
        "(arrays)",
        make_frame(space, labels, space_arity(space)),
        parts=parts,
        axis_labels=tuple(str(label) for label in labels),
        title=title,
    )


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


def _stamp(transform: PlotTransform, result: Any) -> Any:
    """Wrap a plain channel mapping — or a **list** of them — in a :class:`Geometry`.

    ``name``, the coordinate space, the axis count and the axis labels are all
    declared on the decorator.  A transform that also spells them out in a
    hand-built ``Geometry`` declares each of them twice, and the registry's only
    contribution is to check the author against themselves.  So a mapping is a
    legal return: the registry stamps it from the declaration, which cannot
    disagree with itself.  A **sequence of mappings** gives one :class:`Part`
    each (``label`` / ``style`` / ``primitive`` / ``mark`` are read, everything
    else is a channel) — which is what makes the plural cases, a data curve plus
    its fit plus its confidence band, reachable without an IR type either.

    Available only for a transform declaring exactly one ``(frame, ndim)`` pair.
    A transform whose space depends on the data must *say* which one it
    produced, and there is no honest default to guess.
    """
    parts = parts_from_return(result)
    if parts is None:
        return result
    from tsdynamics.errors import InvalidParameterError

    if len(transform.frame) != 1 or len(transform.ndim) != 1:
        raise InvalidParameterError(
            f"transform {transform.name!r} declares coordinate space(s) "
            f"{[s.value for s in transform.frame]} with ndim {list(transform.ndim)}, so a plain "
            "channel mapping cannot say which it produced — return a Geometry naming the frame."
        )
    return Geometry(
        transform=transform.name,
        frame=make_frame(transform.frame[0], transform.labels, transform.ndim[0]),
        parts=parts,
        axis_labels=transform.labels,
    )


def _check_geometry(transform: PlotTransform, result: Any) -> Geometry:
    """Return ``result`` unless it is not a well-formed :class:`Geometry`, in which case raise."""
    from tsdynamics.errors import InvalidInputError, InvalidParameterError

    if not isinstance(result, Geometry):
        raise InvalidInputError(
            f"transform {transform.name!r} returned {type(result).__name__}, not a Geometry "
            "(nor a plain mapping of channels). A transform computes geometry; the primitive "
            "step builds the PlotSpec."
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
    # The axis COUNT was declared and never checked: a transform could declare
    # ndim=1 and return a 3-D frame, and it plotted.  ndim is what decides
    # whether two geometries may share axes, so an unchecked claim is an overlay
    # bug waiting for a second layer.
    if result.frame.ndim not in transform.ndim:
        raise InvalidParameterError(
            f"transform {transform.name!r} declares ndim {list(transform.ndim)} but returned a "
            f"frame with ndim={result.frame.ndim}. The axis count is what decides whether two "
            "geometries may overlay, so it has to be the declared one."
        )
    return result


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
        records = [get(name) for name in sorted(self)]
        lines: list[str] = []
        varying: list[str] = []
        # Grouped by SOURCE, because that is the first thing a newcomer needs to
        # know: a `model` transform wants the system, a `data` one takes the
        # numbers you already have.  A flat alphabetical list answered "what can
        # I draw" with 35 names and no way to tell which of them apply to the
        # object in your hand.
        for source, header in (
            ("data", "FROM DATA — a trajectory, an array, or a system (it runs one)"),
            ("model", "FROM A MODEL — needs the system: evaluates the RHS somewhere new"),
        ):
            rows = [r for r in records if r.source == source]
            if not rows:
                continue
            lines += [header, "-" * len(header)]
            for record in rows:
                row = self[record.name]
                flag = "" if record.available else f"   [unavailable: needs {record.requires}]"
                doc = f"   {record.doc}" if record.doc else ""
                lines.append(f"  {record.name.ljust(width)}  {', '.join(row)}{flag}")
                if doc:
                    lines.append(f"  {' ' * width}  {record.doc}")
                if record.shape_dependent:
                    varying.append(record.name)
            lines.append("")
        lines += ["* = the default primitive (what you get when you name none)"]
        if varying:
            # The row is declared per TRANSFORM; the real constraint is per
            # GEOMETRY.  Printed flat, `phase_portrait -> density, line3d, ...`
            # promises a 3-D trajectory a density plot, which is not a drawing
            # that exists.  Say so, and name the call that answers exactly.
            lines += [
                f"† {', '.join(varying)} produce geometry whose SHAPE depends on the subject",
                "  (2-D vs 3-D; a 1-D profile vs a 2-D field), so the row above is a union.",
                "  The legal row for one subject: ts.viz.geometry(subject, name).primitives",
            ]
        lines += [
            "",
            "ts.plot(subject, 'name')              draw it",
            "ts.plot(subject, 'name.primitive')    ...drawn another way",
            "ts.viz.transforms.find(subject=x)     what can I draw from THIS?",
        ]
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
