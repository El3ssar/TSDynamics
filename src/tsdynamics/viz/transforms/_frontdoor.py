"""``ts.plot`` — the one-liner front door over transforms and composition.

**The rule, in one sentence:**

    ``ts.plot`` draws everything you hand it on one figure and gives you back a
    :class:`~tsdynamics.viz.spec.Plot`: a positional **string** (or
    ``("name", {…})`` pair, or ``T(...)``) says *how* to draw, everything else
    says *what* to draw — and **a `Plot` handed back in is just another thing to
    draw**.

That third clause is *closure*, and it is why grids-of-different-plots,
movies-of-anything and escape-and-return all follow with no additional API.

.. code-block:: python

    ts.plot(traj)                                          # the default view
    ts.plot(lor, final_time=20.0, dt=0.005)                # ...with your integration
    ts.plot(np.sin(np.linspace(0, 40, 2000)))              # bare arrays plot too
    ts.plot(traj, "delay_embedding", delay=7)              # a transform by name
    ts.plot(traj, "phase_portrait.density")                # dotted primitive sugar
    ts.plot(traj_a, traj_b, "phase_portrait")              # two orbits, one figure
    ts.plot(vdp, t1, t2, t3, "vector_field", "nullclines") # the flagship figure
    ts.plot(traj, "time_series", "psd", layout="grid", cols=2)
    ts.plot(traj, ax=my_existing_axes)

**Subjects × transforms is filtered by declared source, not a full cross
product.**  A named transform applies to every subject its
:attr:`~tsdynamics.viz.transforms.PlotTransform.subjects` admits; a subject no
named transform admits draws its **default view**; a named transform no subject
admits raises, naming what it needs.  A full cross product would make the
flagship line above impossible — ``vector_field`` would be tried on a
``Trajectory`` and raise.

Draw order is by **role**, not by argument order, so the call is order-free:
``plot(field, orbit)`` and ``plot(orbit, field)`` are the same picture.
"""

from __future__ import annotations

from typing import Any, cast

from ..compose import _plot_spec_of, unwrap_container
from ..compose import plot as compose_plot
from ..spec import (
    FIGURE_KEYS,
    Plot,
    apply_figure_keywords,
    nearest_keyword,
    reject_on_keyword,
    split_figure_keywords,
)
from ._registry import TransformCall, build_spec, get

__all__ = ["plot"]


def _as_call(sel: Any) -> Any:
    """Normalise a plain ``("name", {options})`` pair to a :class:`TransformCall`.

    No library type is needed to give a transform its own options: a tuple of a
    name and a mapping is a transform call, and the dotted-primitive sugar
    (``("phase_portrait.density", {...})``) works there too.  Anything else is
    returned unchanged.

    This is sugar, not a second grammar — the pair funnels into exactly the
    :class:`TransformCall` that :func:`~tsdynamics.viz.transforms.T` builds, so
    it inherits the did-you-mean validation and produces the same spec.  Before
    this, ``ts.plot(fhn, ("flow_speed", {"log": True}), "streamlines")`` was read
    as a second *subject* and refused with a message about subject counts, which
    pointed nowhere near the mistake.
    """
    from collections.abc import Mapping

    if (
        isinstance(sel, tuple)
        and len(sel) == 2
        and isinstance(sel[0], str)
        and isinstance(sel[1], Mapping)
    ):
        head, _, prim = sel[0].partition(".")
        return TransformCall(name=head, options=dict(sel[1]), primitive=prim or None)
    return sel


def _flatten_positionals(things: tuple[Any, ...]) -> list[Any]:
    """Splice every list/tuple **of plottables** into the positional stream.

    CONTRACT §6.2's grammar table says a ``list``/``tuple`` of plottables is
    unwrapped to subjects, so ``plot([a, b]) == plot(a, b)``.  It used to hold
    only when the container was the *sole* argument, which made the obvious
    comparison call ``ts.plot([t1, t2], "phase_portrait")`` answer with advice
    about raw arrays — a list of trajectories diagnosed as a malformed array.

    Alongside a transform name, a container is spliced only when **every** item
    in it is something the library can plot in its own right — a trajectory, a
    system, a result, a finished ``Plot``.  Anything looser would eat the shapes
    that are legitimately one subject: a ``("name", {options})`` transform call,
    a numeric list (``plot([0.1, 0.2, 0.3])`` is one series) and the recorded
    ``(times, estimates)`` pair some transforms take.  The sole-argument spelling
    keeps :func:`~tsdynamics.viz.compose.unwrap_container`'s older, looser rule.
    """
    if len(things) == 1:
        return unwrap_container(things)
    out: list[Any] = []
    for thing in things:
        if (
            isinstance(thing, (list, tuple))
            and len(thing) > 0
            and _as_call(thing) is thing
            and all(_plot_spec_of(item) is not None for item in thing)
        ):
            out.extend(thing)
        else:
            out.append(thing)
    return out


def _is_selector(thing: Any) -> bool:
    """Whether a positional argument names a *transform* rather than a subject."""
    return isinstance(thing, (str, TransformCall)) or _as_call(thing) is not thing


def _selector_name(selector: str | TransformCall) -> str:
    """Return the transform name a selector refers to (without the dotted primitive)."""
    call = _as_call(selector)
    name = call.name if isinstance(call, TransformCall) else str(call)
    return name.partition(".")[0]


def _split_style(options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a transform's options into ``(build_options, style)``.

    A per-transform style keyword inside an overlay (``T("flow_speed", alpha=0.55)``)
    is applied to *that* transform's layers only — which is the whole point of
    naming them separately.  The split keys off the canonical
    :data:`~tsdynamics.viz.style.STYLE_KEYS` vocabulary and its aliases, so
    ``lw=`` and ``linewidth=`` are both recognised as style.
    """
    style_names = _style_names()
    build: dict[str, Any] = {}
    style: dict[str, Any] = {}
    for name, value in options.items():
        (style if name in style_names else build)[name] = value
    return build, style


#: Composition knobs that must survive the per-transform keyword filter.  They
#: are *not* figure keywords (they change the shape of the figure rather than
#: naming it) and they are *not* transform options.  ``animate`` used to be
#: dropped by that filter — ``ts.plot(traj, "phase_portrait", animate=True)``
#: returned a static spec and ``.save("x.gif")`` wrote a one-frame gif that
#: looked like a working animation.
_COMPOSITION_KEYS: frozenset[str] = frozenset({"animate", "fps"})

#: The words on :func:`plot`'s own signature.  Never keywords a transform sees —
#: they are bound positionally by the door — but a near miss must be able to
#: *suggest* them, or ``label=`` gets answered with ``zlabel=`` (an axis name).
_DOOR_KEYWORDS: frozenset[str] = frozenset(
    {"layout", "rows", "cols", "share_x", "share_y", "share_color", "primitive", "labels", "ax"}
)


def _style_names() -> frozenset[str]:
    """Return the canonical style vocabulary plus every alias, as one lookup set."""
    from ..style import STYLE_KEYS

    names = set(STYLE_KEYS)
    for key in STYLE_KEYS.values():
        names.update(key.aliases)
    return frozenset(names)


def _build_one(
    subject: Any, selector: str | TransformCall, shared: dict[str, Any], primitive: str | None
) -> Any:
    """Build one transform's spec, merging the shared options under its own."""
    from ..style import normalize_style

    selector = _as_call(selector)
    if isinstance(selector, TransformCall):
        name, own, chosen = selector.name, dict(selector.options), selector.primitive
        # A T()'s OWN options were the one keyword path through this door that
        # nobody validated: they are merged in below and handed straight to the
        # compute, so ``T("phase_portrait", nonsense=1)`` surfaced as a bare
        # ``TypeError: phase_portrait() got an unexpected keyword argument`` from
        # deep inside the registry.  Route them through the same check the shared
        # keywords get, so every keyword entering the front door is answered the
        # same way — with the accepted names and a did-you-mean.
        _reject_unaccepted(own, [selector], dropped=False)
    else:
        name, own, chosen = selector, {}, None
    chosen = chosen if chosen is not None else primitive

    # A shared keyword only reaches a transform that can accept it: `ts.plot(sys,
    # "escape_time", "nullclines", grid=400)` must not hand `grid=` to `nullclines`.
    # A *style* keyword is accepted by every transform — it is applied to the
    # layers, not passed to the compute — so it must survive this filter, or
    # `ts.plot(traj, "phase_portrait", color="red")` (the most obvious styling
    # call in the API) silently drops the colour.
    record = get(name.partition(".")[0])
    accepted = _accepted_names(record) | _style_names() | _COMPOSITION_KEYS
    merged = {k: v for k, v in shared.items() if k in accepted}
    merged.update(own)
    build, style = _split_style(merged)
    composition = {k: build.pop(k) for k in list(build) if k in _COMPOSITION_KEYS}

    spec = build_spec(subject, name, primitive=chosen, **build)
    animate = composition.get("animate", False)
    if animate or composition.get("fps") is not None:
        spec = _apply_animation(spec, animate or True, composition.get("fps"))
    if style:
        canon = normalize_style(style)
        for layer in spec.layers:
            layer.style = {**layer.style, **canon}
    return spec


def _apply_animation(spec: Any, animate: Any, fps: float | None = None) -> Any:
    """Attach an :class:`~tsdynamics.viz.spec.Animation` to a freshly built spec.

    Accepts the same three spellings the single-panel front door does — ``True``
    for the defaults, a dict of knobs, or a ready-made ``Animation`` — so
    ``ts.plot(traj, "phase_portrait", animate=True)`` and
    ``traj.plot(animate=True)`` mean the same thing.
    """
    from ..spec import Animation

    if isinstance(animate, Animation):
        spec.animation = animate
    elif isinstance(animate, dict):
        spec.animate(**animate)
    else:
        spec.animate()
    if fps is not None and spec.animation is not None:
        spec.animate(fps=fps)
    return spec


def _accepted_names(record: Any) -> frozenset[str]:
    """Return the keyword names a transform accepts.

    Its ``compute`` parameters, its primitive's options — and, for a ``data``
    transform, **the run vocabulary**.  A ``data`` transform accepts a system
    (having the model is having the data), so ``ts.plot(lorenz, "time_series",
    final_time=5.0)`` has to say how long to run; before this the answer was a
    coin flip: 13 of 26 data transforms hand-copied ``final_time``/``dt`` into
    their own signature and the other 13 refused those words outright — with the
    contract's own field-movie example among the refusals.
    """
    import inspect

    from ._registry import _RUN_KEYS, row_option_names

    params = inspect.signature(record.compute).parameters
    names = frozenset(params) | row_option_names(record) | {"primitive_options"}
    if record.source == "data":
        names |= _RUN_KEYS
    return names


def _reject_unaccepted(
    kw: dict[str, Any], selectors: list[str | TransformCall], *, dropped: bool = True
) -> None:
    """Raise if a keyword is accepted by *none* of the named transforms.

    The per-transform filter in :func:`_build_one` exists so ``plot(sys,
    "escape_time", "nullclines", grid=400)`` does not hand ``grid=`` to
    ``nullclines`` — but a
    keyword no transform accepts falls through the same sieve and is **silently
    dropped**, which is exactly the defect ``_COMPOSITION_KEYS`` was added to fix
    and exactly what the sibling front door (``traj.plot(colour="red")``) already
    refuses.  One typo (``colour=``, ``componets=``) must not cost a wrong picture
    and no message, so the two doors agree: an unusable keyword is an error that
    names the closest keyword the named transforms *do* take.

    Parameters
    ----------
    kw : dict
        The keywords to check.
    selectors : list
        The transform selectors they would reach.
    dropped : bool, optional
        Whether an unaccepted keyword would be *silently dropped* (the shared-``kw``
        case) rather than reaching a compute that raises (a ``T()``'s own
        options).  Only the wording of the message differs; the check does not.
    """
    from tsdynamics.errors import InvalidParameterError

    accepted: set[str] = set(_style_names() | _COMPOSITION_KEYS | FIGURE_KEYS)
    selectors = [_as_call(s) for s in selectors]
    for selector in selectors:
        name = selector.name if isinstance(selector, TransformCall) else selector
        accepted |= _accepted_names(get(name.partition(".")[0]))
    unused = sorted(set(kw) - accepted)
    if not unused:
        return
    # ``source`` is the subject (passed positionally) and ``primitive_options`` is
    # the escape hatch — neither is something a caller types, so listing them
    # would send a reader looking for a keyword that is not the one they want.
    listed = sorted(
        accepted
        - _style_names()
        - _COMPOSITION_KEYS
        - FIGURE_KEYS
        - {"source", "primitive_options"}
    )
    named = [str(s) for s in selectors]
    # Suggest against the style vocabulary too — ``colour=`` is a misspelling of
    # the style key ``color=``, not of the transform's own ``color_by=``.  And
    # against ``plot``'s own signature, so the singular ``label=`` — what a user
    # types before learning the plural — is answered with ``labels=`` instead of
    # the measured ``did you mean zlabel=?``, which named an *axis*.
    pool = sorted(
        set(listed)
        | set(_style_names())
        | set(_COMPOSITION_KEYS)
        | set(FIGURE_KEYS)
        | _DOOR_KEYWORDS
    )
    close = {u: nearest_keyword(u, pool) for u in unused}
    hints = "".join(f"\n    {bad}= — did you mean {near}=?" for bad, near in close.items() if near)
    tail = ", so they would be silently ignored." if dropped else "."
    raise InvalidParameterError(
        f"{named} does not accept keyword(s) {unused}{tail}"
        f"{hints}\nKeywords accepted here: {listed} (plus any style keyword — color=, "
        "linewidth=, alpha=, … — plus title=, xlabel=, ylabel=, theme= and animate=)."
    )


def _pair_up(
    subjects: list[Any], selectors: list[str | TransformCall]
) -> list[tuple[Any, str | TransformCall | None]]:
    """Pair every subject with every transform its declared ``source`` admits.

    **This is the cross product, filtered — the heart of the v6 grammar.**

    - ``ts.plot(a, b, "phase_portrait")`` → both orbits, one figure.  (Before v6
      this raised *"needs exactly one subject to apply them to, got 2"* — for
      the most obvious comparison plot in dynamics.)
    - ``ts.plot(vdp, t1, t2, "vector_field", "nullclines")`` → the two model
      transforms take the system, the three orbits draw their default view.  A
      *full* cross product would hand ``vector_field`` a ``Trajectory`` and
      raise, which is why the filter is by declared source rather than by
      position.
    - A :class:`~tsdynamics.viz.spec.Plot` handed in is already drawn (closure),
      so it is never fed to a transform.

    Returns
    -------
    list of tuple
        ``(subject, selector)`` pairs in a deterministic order — subject-major,
        then declaration order — with ``selector=None`` meaning *draw this
        subject's default view*.
    """
    from tsdynamics.errors import InvalidParameterError

    from ..spec import PlotSpec

    pairs: list[tuple[Any, str | TransformCall | None]] = []
    used: set[int] = set()
    for subject in subjects:
        if isinstance(subject, PlotSpec):
            pairs.append((subject, None))
            continue
        taken = [(i, sel) for i, sel in enumerate(selectors) if _admits(sel, subject)]
        used.update(i for i, _ in taken)
        pairs.extend((subject, sel) for _, sel in taken)
        if not taken:
            pairs.append((subject, None))
    orphans = [str(sel) for i, sel in enumerate(selectors) if i not in used]
    if orphans:
        needed = sorted({_needs(sel) for sel in selectors if str(sel) in orphans})
        raise InvalidParameterError(
            f"transform(s) {orphans} were named but none of the {len(subjects)} subject(s) "
            f"can be handed to them: they need {' / '.join(needed)}. "
            f"Pass one — ts.plot(lorenz, {orphans[0]!r}) — or drop the transform."
        )
    return pairs


def _admits(selector: str | TransformCall, subject: Any) -> bool:
    """Whether the named transform's declared subjects admit this subject."""
    return bool(get(_selector_name(selector)).accepts_subject(subject))


#: How each declared subject word reads in a sentence.  A transform says what it
#: takes at its registration site; this turns that declaration into the phrase the
#: refusal prints, so the message cannot drift from the check.
_SUBJECT_PHRASES: dict[str, str] = {
    "system": "a system (it runs the equations)",
    "flow": "a continuous system (it evaluates the vector field)",
    "map": "a discrete map, or a trajectory of one",
    "trajectory": "a trajectory",
    "array": "an array of samples",
}


def _needs(selector: str | TransformCall) -> str:
    """Return a short phrase naming what a transform must be handed.

    Built from the transform's **declared** ``subjects``, so ``ts.plot(traj,
    "cobweb")`` now says *"a discrete map, or a trajectory of one"* instead of
    the source-derived "a trajectory / array / system" — which was the phrase
    that made the refusal unactionable, since a trajectory is one of those.
    """
    record = get(_selector_name(selector))
    words = [_SUBJECT_PHRASES.get(s, f"a {s}") for s in record.subjects]
    return " or ".join(dict.fromkeys(words))


def plot(
    *things: Any,
    layout: str = "overlay",
    rows: int | None = None,
    cols: int | None = None,
    share_x: bool | None = None,
    share_y: bool | None = None,
    share_color: bool | None = None,
    primitive: str | None = None,
    force: bool = False,
    animate: Any = False,
    fps: float | None = None,
    labels: Any = None,
    ax: Any = None,
    **kw: Any,
) -> Plot:
    """Plot one or more things, optionally through named transforms.

    Parameters
    ----------
    *things
        The **subjects** (a :class:`~tsdynamics.data.Trajectory`, a system, an
        analysis result, a bare array, or a finished
        :class:`~tsdynamics.viz.spec.Plot`) mixed freely with **transform
        selectors** — a name (``"phase_portrait"``, ``"phase_portrait.density"``),
        a ``("name", {options})`` pair, or a
        :func:`~tsdynamics.viz.transforms.T` carrying that transform's own
        options.  Each named transform is applied to every subject its declared
        source admits; a subject no transform admits draws its default view.
    layout : {"overlay", "stack", "row", "grid", "frames"}, optional
        ``"overlay"`` (the default) draws everything on one set of axes; the
        others give each thing its own panel.  Overlay legality is *frame*
        compatibility — the same coordinate space, dimension and axes — so a
        basin image, its attractors, an orbit and the equilibria share one axes,
        while an ``(x, y)`` portrait refuses an ``(x, z)`` overlay.

        ``"frames"`` is the parameter-sweep **movie**: the panels are consecutive
        in *time* rather than in space, so they are played one after another
        instead of tiled (``fps=`` is enough to start it)::

            ts.plot(*[ts.plot(sys.with_params(r=r), "cobweb") for r in rs],
                    layout="frames", fps=15).save("cascade.mp4")
    rows, cols : int, optional
        The panel grid shape for ``layout="grid"``.
    share_x, share_y : bool, optional
        Force shared axes across the panels.
    share_color : bool, optional
        Put every panel on **one** colour scale and draw **one** colorbar.
    primitive : str, optional
        How to draw the named transform(s) — validated against each one's
        declared row, so an invalid pair raises (naming the valid set) rather
        than quietly drawing something else.  A ``T(..., primitive=...)`` wins
        over this for its own transform.
    force : bool, default False
        Overlay a deliberate frame mismatch with a warning instead of raising.
    animate : bool or dict or Animation, optional
        Animate the figure (a comet on a curve, a movie of a field).
    fps : float, optional
        Frames per second; implies ``animate=True``.
    labels : sequence of str, optional
        **Name the curves** — one entry per *subject*, in argument order, with
        ``None`` leaving that subject's automatic label alone::

            ts.plot(a, b, labels=["mu = 1", "mu = 3"])

        Comparing two parameter values is the commonest figure in this field and
        before v6 it had no spelling at all: ``label=``/``labels=`` were refused
        (suggesting ``zlabel=``, an *axis* name), leaving ``p.layers[i].label``
        — reaching into the IR — as the only route.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes **now** and still return the ``Plot`` — the
        way to put a tsdynamics figure inside a layout you are building
        yourself.  ``.render(...)`` is the verb that leaves the library and hands
        back the backend artifact; this one keeps you in it.
    **kw
        Figure keywords (``title`` / ``xlabel`` / ``xlim`` / ``xscale`` /
        ``clim`` / ``theme`` / …, the 17 of
        :data:`~tsdynamics.viz.spec.FIGURE_KEYS`), style keywords (``color`` /
        ``linewidth`` / ``alpha`` / …), and the named transforms' own options —
        routed only to the transforms that accept them, so ``plot(sys,
        "escape_time", "nullclines", grid=400)`` does not hand ``grid=`` to
        ``nullclines``.  A keyword that reaches *nothing* is an **error**, not a
        silent drop, so a typo (``colour=``) costs a message rather than a wrong
        picture.

    Returns
    -------
    Plot
        Always — which is why a result feeds straight back in, and why this is
        the *same* return type as ``traj.plot()`` / ``system.plot()``.  ``plot``
        builds, ``render`` draws, ``show`` displays, ``save`` writes.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If a named transform matches no subject, if a keyword reaches none of
        the named transforms, if a (transform, primitive) pair is not declared,
        or if the frames do not allow an overlay.

    Examples
    --------
    Every name below is a **registered** transform.  The current list is
    ``print(ts.viz.compatibility())`` — the matrix *is* the repr, so a script
    needs the ``print``, where a REPL does not — or the documentation gallery,
    which is generated from the same registry.

    >>> ts.plot(traj)                                        # doctest: +SKIP
    >>> ts.plot(traj, "delay_embedding", delay=7)            # doctest: +SKIP
    >>> ts.plot(traj_a, traj_b, "phase_portrait")            # doctest: +SKIP
    >>> ts.plot(vdp, t1, t2, "vector_field", "nullclines")   # doctest: +SKIP
    >>> ts.plot(vdp, ts.viz.T("flow_speed", log=True),
    ...              ts.viz.T("streamlines", color="w"))     # doctest: +SKIP
    """
    from tsdynamics.errors import InvalidParameterError

    reject_on_keyword(kw, "ts.plot()", "ts.plot(a, b, force=True)")
    # A lone ``("name", {options})`` pair is a transform call, not a container of
    # plottables to unwrap — so the message is about the missing subject rather
    # than about an unplottable string.
    items = (
        [things[0]]
        if len(things) == 1 and _as_call(things[0]) is not things[0]
        else _flatten_positionals(things)
    )
    selectors = [t for t in items if _is_selector(t)]
    subjects = [t for t in items if not _is_selector(t)]
    if not selectors:
        if primitive is not None:
            raise InvalidParameterError(
                "primitive= selects how a *named transform* is drawn, but no transform was "
                "named; pass one, e.g. ts.plot(traj, 'phase_portrait', primitive='density')."
            )
        result = compose_plot(
            *subjects,
            layout=layout,
            rows=rows,
            cols=cols,
            share_x=share_x,
            share_y=share_y,
            share_color=share_color,
            force=force,
            animate=animate,
            fps=fps,
            labels=labels,
            **kw,
        )
        return _finish(result, ax)

    if not subjects:
        raise InvalidParameterError(
            f"naming transform(s) {[str(s) for s in selectors]} needs something to apply them "
            "to; pass the subject too, e.g. ts.plot(traj, 'phase_portrait')."
        )
    # Naming the figure is not the same as computing it: ``title=`` / ``xlim=`` /
    # ``theme=`` describe the result, so they are peeled off here and applied to
    # the composed spec rather than offered to a transform that has no idea what
    # to do with them.
    figure = split_figure_keywords(kw)
    kw.update(_shared_with_transforms(figure, selectors))
    _reject_unaccepted(kw, selectors)
    if animate is not False or fps is not None:
        kw = {**kw, "animate": animate or True, "fps": fps}
    pairs = _pair_up(subjects, selectors)
    # The subjects no transform claimed draw their own view; build those FIRST,
    # because a model transform's auto window is derived from them (see
    # ``_window_over_data``).
    drawn: dict[int, Any] = {
        i: _default_view(subject, dict(kw)) for i, (subject, sel) in enumerate(pairs) if sel is None
    }
    field_kw = {**kw, **_window_over_data(drawn.values(), kw)}
    specs = [
        drawn[i] if sel is None else _build_one(subject, sel, dict(field_kw), primitive)
        for i, (subject, sel) in enumerate(pairs)
    ]
    if labels is not None:
        _label_by_subject(specs, [subject for subject, _ in pairs], subjects, labels)
    result = compose_plot(
        *specs,
        layout=layout,
        rows=rows,
        cols=cols,
        share_x=share_x,
        share_y=share_y,
        share_color=share_color,
        force=force,
    )
    apply_figure_keywords(result, figure)
    return _finish(result, ax)


#: How much room to leave around the data when a model transform's window is
#: derived from it.  A curve exactly tangent to the axes reads as clipped.
_WINDOW_PAD = 0.05


def _window_over_data(views: Any, kw: dict[str, Any]) -> dict[str, Any]:
    """Return the ``xlim``/``ylim`` a model transform should cover, from the data.

    **The window is the union over every data subject.**  Before this, a field
    transform chose its window from the *system* alone and — as the overlay's
    base spec — imposed it on the whole figure, so

    >>> ts.plot(pendulum, *five_orbits, "vector_field")   # doctest: +SKIP

    picked ``[-2, 2]²``, drew **two orbits entirely outside the axes**, and still
    legended them: a figure asserting that a curve is somewhere it is not.  The
    field is now computed over what is actually being drawn, which fixes the
    clipping and the white bands at once (widening only the axes would trade one
    for the other).

    Only a default: an explicit ``xlim=``/``ylim=`` — shared or inside a
    ``("name", {...})`` pair — still wins, and a non-finite or degenerate extent
    falls back to the transform's own choice.
    """
    from ..spec import PlotKind

    wanted = [axis for axis in ("xlim", "ylim") if axis not in kw]
    if not wanted:
        return {}
    planar = [
        spec
        for spec in views
        if spec is not None and spec.ndim == 2 and spec.kind is not PlotKind.COMPOSITE
    ]
    if not planar:
        return {}
    out: dict[str, Any] = {}
    for axis, channel in (("xlim", "x"), ("ylim", "y")):
        if axis not in wanted:
            continue
        span = _channel_span([layer.data.get(channel) for s in planar for layer in s.layers])
        if span is not None:
            out[axis] = span
    return out


def _channel_span(arrays: list[Any]) -> tuple[float, float] | None:
    """Return the padded ``(lo, hi)`` covering every finite sample, or ``None``."""
    import numpy as np

    lo, hi = np.inf, -np.inf
    for arr in arrays:
        if arr is None:
            continue
        values = np.asarray(arr, dtype=float).ravel()
        finite = values[np.isfinite(values)]
        if finite.size:
            lo, hi = min(lo, float(finite.min())), max(hi, float(finite.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    pad = (hi - lo) * _WINDOW_PAD
    return (lo - pad, hi + pad)


def _shared_with_transforms(
    figure: dict[str, Any], selectors: list[str | TransformCall]
) -> dict[str, Any]:
    """Return the figure keywords a named transform **also** declares.

    ``xlim=`` means *axis limits* to a figure and *the domain to evaluate the
    field on* to ``vector_field`` / ``flow_speed`` / ``ftle`` / ``escape_time``
    / ``nullclines`` / ``streamlines`` / ``transient_time``, and the figure used
    to win in silence: ``ts.plot(sys, "flow_speed", ylim=(-8, 8))`` moved the
    axes and left the field on its auto window, so three panels of a comparison
    figure came out with white bands and **no warning** — while the very same
    keyword at the geometry door (``ts.viz.geometry(sys, "flow_speed",
    ylim=…)``) windowed the field correctly.  Two doors, one word, two pictures.

    A caller writing one window means one thing, so the word now reaches
    **both**: the transform computes over it and the axes are set to match.  The
    escape hatch stays exact — a transform's *own* option wins over the shared
    one (``_build_one`` merges ``own`` last), so

    >>> ts.plot(vdp, ("flow_speed", {"xlim": (-3, 3)}), xlim=(-10, 10))  # doctest: +SKIP

    computes the field over ``[-3, 3]`` and draws axes over ``[-10, 10]``.
    """
    if not figure:
        return {}
    accepted: set[str] = set()
    for selector in selectors:
        call = _as_call(selector)
        name = call.name if isinstance(call, TransformCall) else str(call)
        accepted |= _accepted_names(get(name.partition(".")[0]))
    return {k: v for k, v in figure.items() if k in accepted}


def _label_by_subject(
    specs: list[Any], owners: list[Any], subjects: list[Any], labels: Any
) -> None:
    """Apply ``labels=`` to the specs, matched to **subjects** in argument order.

    A subject can produce several specs (one per transform it was paired with),
    so the labels are matched to what the caller typed — the subjects — and then
    fanned out to whatever those subjects drew.  Matching specs instead would
    make ``ts.plot(vdp, t1, t2, "vector_field", "nullclines", labels=[...])``
    require a count no caller can predict.
    """
    from ..compose import apply_labels, label_count_message

    names: list[Any] = [labels] if isinstance(labels, str) else list(labels)
    if len(names) != len(subjects):
        from tsdynamics.errors import InvalidParameterError

        raise InvalidParameterError(label_count_message(len(subjects), len(names)))
    by_subject = dict(zip(map(id, subjects), names, strict=True))
    apply_labels(specs, [by_subject[id(owner)] for owner in owners])


def _default_view(subject: Any, kw: dict[str, Any]) -> Any:
    """Build a subject's own default view — what a subject no transform admits gets.

    The keywords are **not** forwarded: they were validated against the named
    transforms, and a ``Trajectory`` sitting under a ``vector_field`` call has no
    business being handed ``grid=``.
    """
    from ..compose import to_spec

    return to_spec(subject, {})


def _finish(result: Any, ax: Any) -> Plot:
    """Render eagerly into ``ax`` when one was given, and return the ``Plot`` either way.

    ``ax=`` is an *in* door: you are still in the library afterwards, holding the
    same chainable object.  ``.render(...)`` is the *out* door, and hands back
    the backend artifact.  Two verbs, two meanings.
    """
    if ax is not None:
        result.render(ax=ax)
    return cast("Plot", result)
