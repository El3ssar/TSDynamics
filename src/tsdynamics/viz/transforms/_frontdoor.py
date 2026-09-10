"""``ts.plot`` — the one-liner front door over transforms and composition.

Three rungs, one function, one return type:

.. code-block:: python

    ts.plot(traj)                                     # what you already had
    ts.plot(traj, "delay_embedding", delay=7)         # a transform by name
    ts.plot(vdp, "flow_speed", "streamlines", "nullclines")   # several, one subject

Every name in an example on this page is a **registered** transform
(:func:`~tsdynamics.viz.compatibility` lists them, and the documentation gallery
is generated from that list).  The rung-3 line above used to name four
transforms — ``basins``, ``attractors``, ``trajectory``, ``fixed_points`` — none
of which is registered, so the module's flagship example answered a reader who
pasted it with ``InvalidParameterError: unknown plot transform 'basins'``.  A
gate now checks the names in every such example against the live registry.

The rule that keeps it one function rather than three: **a positional string (or
a** :func:`~tsdynamics.viz.transforms.T` **) is a transform to apply to the
subject; anything else is a subject.**  With no transform named, this is exactly
:func:`tsdynamics.viz.plot`, which stays the composition layer underneath — so
there is one merge policy, one frame check, one z-ordering, and one return type
(always a :class:`~tsdynamics.viz.spec.PlotSpec`, which renders itself).

Draw order is by **role**, not by argument order, so the call is order-free:
``plot(field, orbit)`` and ``plot(orbit, field)`` are the same picture.
"""

from __future__ import annotations

from typing import Any

from ..compose import apply_presentation, unwrap_container
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


#: Keywords that are about the FIGURE, not about the transform's computation, and
#: so must survive the per-transform keyword filter.  ``animate`` was silently
#: dropped by that filter — ``ts.plot(traj, "phase_portrait", animate=True)``
#: returned a static spec and ``.save("x.gif")`` wrote a one-frame gif that looked
#: like a working animation.  It is the same class of defect as a dropped
#: ``color=``: a keyword accepted and ignored.
_FIGURE_KEYS: frozenset[str] = frozenset({"animate"})

#: Keywords that *name* the figure rather than compute it.  They are peeled off
#: before the per-transform keyword filter and applied to the composed spec, so
#: ``ts.plot(traj, "phase_portrait", title="Lorenz")`` — half of the most common
#: plot request in existence — arrives instead of being refused.
_LABEL_KEYS: frozenset[str] = frozenset({"title", "xlabel", "ylabel", "zlabel", "theme"})


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
    accepted = _accepted_names(record) | _style_names() | _FIGURE_KEYS
    merged = {k: v for k, v in shared.items() if k in accepted}
    merged.update(own)
    build, style = _split_style(merged)
    figure_kw = {k: build.pop(k) for k in list(build) if k in _FIGURE_KEYS}

    spec = build_spec(subject, name, primitive=chosen, **build)
    animate = figure_kw.get("animate", False)
    if animate:
        spec = _apply_animation(spec, animate)
    if style:
        canon = normalize_style(style)
        for layer in spec.layers:
            layer.style = {**layer.style, **canon}
    return spec


def _apply_animation(spec: Any, animate: Any) -> Any:
    """Attach an :class:`~tsdynamics.viz.spec.Animation` to a freshly built spec.

    Accepts the same three spellings the single-panel front door does — ``True``
    for the defaults, a dict of knobs, or a ready-made ``Animation`` — so
    ``ts.plot(traj, "phase_portrait", animate=True)`` and
    ``traj.to_plot_spec(animate=True)`` mean the same thing.
    """
    from ..spec import Animation

    if isinstance(animate, Animation):
        spec.animation = animate
        return spec
    if isinstance(animate, dict):
        return spec.animate(**animate)
    return spec.animate()


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
    dropped**, which is exactly the defect ``_FIGURE_KEYS`` was added to fix and
    exactly what the sibling front door (``traj.plot(colour="red")``) already
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

    accepted: set[str] = set(_style_names() | _FIGURE_KEYS | _LABEL_KEYS)
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
        accepted - _style_names() - _FIGURE_KEYS - _LABEL_KEYS - {"source", "primitive_options"}
    )
    named = [str(s) for s in selectors]
    # Suggest against the style vocabulary too — ``colour=`` is a misspelling of
    # the style key ``color=``, not of the transform's own ``color_by=``.
    pool = sorted(set(listed) | set(_style_names()) | set(_FIGURE_KEYS) | set(_LABEL_KEYS))
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


def plot(
    *things: Any,
    layout: str = "overlay",
    primitive: str | None = None,
    on: str | None = None,
    **kw: Any,
) -> Any:
    """Plot one or more things, optionally through named transforms.

    Parameters
    ----------
    *things
        The **subject** (a :class:`~tsdynamics.data.Trajectory`, a system, an
        analysis result, a :class:`~tsdynamics.viz.spec.PlotSpec`) followed by
        any number of **transform selectors** — a name (``"phase_portrait"``,
        ``"phase_portrait.density"``) or a :func:`~tsdynamics.viz.transforms.T` carrying
        that transform's own options.  With no selector, every positional is a
        subject and this is :func:`tsdynamics.viz.plot`.
    layout : {"overlay", "stack", "row", "grid"}, optional
        ``"overlay"`` (the default) draws everything on one set of axes; the
        others give each thing its own panel.  Overlay legality is *frame*
        compatibility — the same coordinate space, dimension and axes — so a
        basin image, its attractors, an orbit and the equilibria share one axes,
        while an ``(x, y)`` portrait refuses an ``(x, z)`` overlay.
    primitive : str, optional
        How to draw the named transform(s) — validated against each one's
        declared row, so an invalid pair raises (naming the valid set) rather
        than quietly drawing something else.  A ``T(..., primitive=...)`` wins
        over this for its own transform.
    on : {"force"}, optional
        Overlay a deliberate frame mismatch with a warning instead of raising.
    **kw
        Options shared by the named transforms — routed only to the transforms
        that actually accept them, so ``plot(sys, "escape_time", "nullclines",
        grid=400)`` does not hand ``grid=`` to ``nullclines``.  A keyword that
        reaches *no* named transform is an **error**, not a silent drop, so a
        typo (``colour=``) costs a message rather than a wrong picture.  With no
        selector, forwarded to each subject's ``to_plot_spec`` (as
        :func:`tsdynamics.viz.plot` does).

    Returns
    -------
    PlotSpec
        Always — which is why a result feeds straight back in, and why this is
        the *same* return type as ``traj.plot()`` / ``system.plot()`` /
        ``spec.tweak()``.  ``plot`` builds, ``render`` draws, ``show`` displays, ``save`` writes:
        ``.save("fig.pdf")`` / ``.render("plotly")``.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If transform names are given without exactly one subject, if a shared
        keyword reaches none of the named transforms, if a (transform,
        primitive) pair is not declared, or if the frames do not allow an
        overlay.

    Examples
    --------
    Every name below is a **registered** transform — see
    :func:`tsdynamics.viz.compatibility` for the current list, or the
    documentation gallery, which is generated from it.

    >>> ts.plot(traj)                                        # doctest: +SKIP
    >>> ts.plot(traj, "delay_embedding", delay=7)            # doctest: +SKIP
    >>> ts.plot(traj, "phase_portrait", components=("x", "z"),
    ...         primitive="density")                         # doctest: +SKIP
    >>> ts.plot(fhn, "flow_speed", "streamlines", "nullclines",
    ...         xlim=(-2.5, 2.5), ylim=(-1.0, 2.0))          # doctest: +SKIP
    >>> ts.plot(vdp, ts.T("flow_speed", log=True, alpha=0.6),
    ...              ts.T("streamlines", seeds=6, color="w")) # doctest: +SKIP
    """
    from tsdynamics.errors import InvalidParameterError

    from ..compose import plot as compose_plot

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
        return compose_plot(*subjects, layout=layout, on=on, **kw)

    if len(subjects) != 1:
        raise InvalidParameterError(
            f"naming transform(s) {[str(s) for s in selectors]} needs exactly one subject to "
            f"apply them to, got {len(subjects)}. Build each subject's spec separately and "
            "compose them with tsdynamics.viz.plot(...)."
        )
    subject = subjects[0]
    # Naming the figure is not the same as computing it: ``title=`` / ``xlabel=``
    # / ``theme=`` describe the result, so they are peeled off here and applied
    # to the composed spec rather than offered to a transform that has no idea
    # what to do with them.
    figure = {k: kw.pop(k) for k in list(kw) if k in _LABEL_KEYS}
    _reject_unaccepted(kw, selectors)
    specs = [_build_one(subject, sel, dict(kw), primitive) for sel in selectors]
    result = compose_plot(*specs, layout=layout, on=on)
    apply_presentation(result, {}, figure)
    return result
