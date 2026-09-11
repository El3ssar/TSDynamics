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

from typing import Any

from ..compose import plot as compose_plot
from ..compose import unwrap_container
from ..spec import FIGURE_KEYS, apply_figure_keywords, split_figure_keywords
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
    ``traj.to_plot_spec(animate=True)`` mean the same thing.
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
    """Return the keyword names a transform accepts (compute parameters + primitive options)."""
    import inspect

    from ._registry import row_option_names

    params = inspect.signature(record.compute).parameters
    return frozenset(params) | row_option_names(record) | {"primitive_options"}


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
    import difflib

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
    # the style key ``color=``, not of the transform's own ``color_by=``.
    pool = sorted(set(listed) | set(_style_names()) | set(_COMPOSITION_KEYS) | set(FIGURE_KEYS))
    close = {u: difflib.get_close_matches(u, pool, n=1, cutoff=0.6) for u in unused}
    hints = "".join(
        f"\n    {bad}= — did you mean {near[0]}=?" for bad, near in close.items() if near
    )
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


def _needs(selector: str | TransformCall) -> str:
    """Return a short phrase naming what a transform must be handed."""
    record = get(_selector_name(selector))
    return "a dynamical system" if record.source == "model" else "a trajectory / array / system"


def plot(
    *things: Any,
    layout: str = "overlay",
    rows: int | None = None,
    cols: int | None = None,
    share_x: bool | None = None,
    share_y: bool | None = None,
    share_color: bool | None = None,
    primitive: str | None = None,
    on: str | None = None,
    animate: Any = False,
    fps: float | None = None,
    ax: Any = None,
    **kw: Any,
) -> Any:
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
    layout : {"overlay", "stack", "row", "grid"}, optional
        ``"overlay"`` (the default) draws everything on one set of axes; the
        others give each thing its own panel.  Overlay legality is *frame*
        compatibility — the same coordinate space, dimension and axes — so a
        basin image, its attractors, an orbit and the equilibria share one axes,
        while an ``(x, y)`` portrait refuses an ``(x, z)`` overlay.
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
    on : {"force"}, optional
        Overlay a deliberate frame mismatch with a warning instead of raising.
    animate : bool or dict or Animation, optional
        Animate the figure (a comet on a curve, a movie of a field).
    fps : float, optional
        Frames per second; implies ``animate=True``.
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
    Every name below is a **registered** transform — see
    :func:`tsdynamics.viz.compatibility` for the current list, or the
    documentation gallery, which is generated from it.

    >>> ts.plot(traj)                                        # doctest: +SKIP
    >>> ts.plot(traj, "delay_embedding", delay=7)            # doctest: +SKIP
    >>> ts.plot(traj_a, traj_b, "phase_portrait")            # doctest: +SKIP
    >>> ts.plot(vdp, t1, t2, "vector_field", "nullclines")   # doctest: +SKIP
    >>> ts.plot(vdp, ts.viz.T("flow_speed", log=True),
    ...              ts.viz.T("streamlines", color="w"))     # doctest: +SKIP
    """
    from tsdynamics.errors import InvalidParameterError

    # A lone ``("name", {options})`` pair is a transform call, not a container of
    # plottables to unwrap — so the message is about the missing subject rather
    # than about an unplottable string.
    items = (
        [things[0]]
        if len(things) == 1 and _as_call(things[0]) is not things[0]
        else unwrap_container(things)
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
            on=on,
            animate=animate,
            fps=fps,
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
    _reject_unaccepted(kw, selectors)
    if animate is not False or fps is not None:
        kw = {**kw, "animate": animate or True, "fps": fps}
    specs = [
        _build_one(subject, sel, dict(kw), primitive)
        if sel is not None
        else _default_view(subject, dict(kw))
        for subject, sel in _pair_up(subjects, selectors)
    ]
    result = compose_plot(
        *specs,
        layout=layout,
        rows=rows,
        cols=cols,
        share_x=share_x,
        share_y=share_y,
        share_color=share_color,
        on=on,
    )
    apply_figure_keywords(result, figure)
    return _finish(result, ax)


def _default_view(subject: Any, kw: dict[str, Any]) -> Any:
    """Build a subject's own default view — what a subject no transform admits gets.

    The keywords are **not** forwarded: they were validated against the named
    transforms, and a ``Trajectory`` sitting under a ``vector_field`` call has no
    business being handed ``grid=``.
    """
    from ..compose import to_spec

    return to_spec(subject, {})


def _finish(result: Any, ax: Any) -> Any:
    """Render eagerly into ``ax`` when one was given, and return the ``Plot`` either way.

    ``ax=`` is an *in* door: you are still in the library afterwards, holding the
    same chainable object.  ``.render(...)`` is the *out* door, and hands back
    the backend artifact.  Two verbs, two meanings.
    """
    if ax is not None:
        result.render(ax=ax)
    return result
