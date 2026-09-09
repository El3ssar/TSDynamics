"""``ts.plot`` — the one-liner front door over transforms and composition.

Three rungs, one function, one return type:

.. code-block:: python

    ts.plot(traj)                                     # what you already had
    ts.plot(traj, "delay_embedding", delay=7)         # a transform by name
    ts.plot(duff, "basins", "attractors", "trajectory", "fixed_points")

The rule that keeps it one function rather than three: **a positional string (or
a** :func:`~tsdynamics.viz.transforms.T` **) is a transform to apply to the
subject; anything else is a subject.**  With no transform named, this is exactly
:func:`tsdynamics.viz.plot`, which stays the composition layer underneath — so
there is one merge policy, one frame check, one z-ordering, and one return type
(always a :class:`~tsdynamics.viz.spec.PlotSpec`, which renders itself).

Draw order is by **role**, not by argument order, so the call is order-free:
``plot(basins, traj)`` and ``plot(traj, basins)`` are the same picture.
"""

from __future__ import annotations

from typing import Any

from ._registry import TransformCall, build_spec, get

__all__ = ["plot"]


def _is_selector(thing: Any) -> bool:
    """Whether a positional argument names a *transform* rather than a subject."""
    return isinstance(thing, (str, TransformCall))


def _split_style(options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a transform's options into ``(build_options, style)``.

    A per-transform style keyword inside an overlay (``T("basins", alpha=0.55)``)
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

    if isinstance(selector, TransformCall):
        name, own, chosen = selector.name, dict(selector.options), selector.primitive
    else:
        name, own, chosen = selector, {}, None
    chosen = chosen if chosen is not None else primitive

    # A shared keyword only reaches a transform that can accept it: `ts.plot(sys,
    # "basins", "trajectory", grid=400)` must not hand `grid=` to `trajectory`.
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


def _reject_unused(kw: dict[str, Any], selectors: list[str | TransformCall]) -> None:
    """Raise if a shared keyword reaches *none* of the named transforms.

    The per-transform filter in :func:`_build_one` exists so ``plot(sys, "basins",
    "trajectory", grid=400)`` does not hand ``grid=`` to ``trajectory`` — but a
    keyword no transform accepts falls through the same sieve and is **silently
    dropped**, which is exactly the defect ``_FIGURE_KEYS`` was added to fix and
    exactly what the sibling front door (``traj.plot(colour="red")``) already
    refuses.  One typo (``colour=``, ``componets=``) must not cost a wrong picture
    and no message, so the two doors agree: an unusable keyword is an error that
    names the closest keyword the named transforms *do* take.
    """
    import difflib

    from tsdynamics.errors import InvalidParameterError

    accepted: set[str] = set(_style_names() | _FIGURE_KEYS)
    for selector in selectors:
        name = selector.name if isinstance(selector, TransformCall) else selector
        accepted |= _accepted_names(get(name.partition(".")[0]))
    unused = sorted(set(kw) - accepted)
    if not unused:
        return
    # ``source`` is the subject (passed positionally) and ``primitive_options`` is
    # the escape hatch — neither is something a caller types, so listing them
    # would send a reader looking for a keyword that is not the one they want.
    listed = sorted(accepted - _style_names() - _FIGURE_KEYS - {"source", "primitive_options"})
    named = [str(s) for s in selectors]
    # Suggest against the style vocabulary too — ``colour=`` is a misspelling of
    # the style key ``color=``, not of the transform's own ``color_by=``.
    pool = sorted(set(listed) | set(_style_names()) | set(_FIGURE_KEYS))
    close = {u: difflib.get_close_matches(u, pool, n=1, cutoff=0.6) for u in unused}
    hints = "".join(
        f"\n    {bad}= — did you mean {near[0]}=?" for bad, near in close.items() if near
    )
    raise InvalidParameterError(
        f"{named} does not accept keyword(s) {unused}, so they would be silently ignored."
        f"{hints}\nKeywords accepted here: {listed} (plus any style keyword — color=, "
        "linewidth=, alpha=, … — and animate=)."
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
        any number of **transform selectors** — a name (``"basins"``,
        ``"basins.boundary"``) or a :func:`~tsdynamics.viz.transforms.T` carrying
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
        that actually accept them, so ``plot(sys, "basins", "trajectory",
        grid=400)`` does not hand ``grid=`` to ``trajectory``.  A keyword that
        reaches *no* named transform is an **error**, not a silent drop, so a
        typo (``colour=``) costs a message rather than a wrong picture.  With no
        selector, forwarded to each subject's ``to_plot_spec`` (as
        :func:`tsdynamics.viz.plot` does).

    Returns
    -------
    PlotSpec
        Always — which is why a result feeds straight back in, and why this is
        the *same* return type as ``traj.plot()`` / ``system.plot()`` /
        ``spec.plot()``.  ``plot`` builds, ``render`` draws, ``save`` writes:
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
    >>> ts.plot(traj, "phase_portrait", primitive="density") # doctest: +SKIP
    >>> ts.plot(fhn, "flow_speed", "streamlines", "nullclines",
    ...         xlim=(-2.5, 2.5), ylim=(-1.0, 2.0))          # doctest: +SKIP
    >>> ts.plot(vdp, ts.T("flow_speed", log=True, alpha=0.6),
    ...              ts.T("streamlines", seeds=6, color="w")) # doctest: +SKIP
    """
    from tsdynamics.errors import InvalidParameterError

    from ..compose import plot as compose_plot

    items = (
        list(things[0])
        if len(things) == 1 and isinstance(things[0], (list, tuple))
        else list(things)
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
    _reject_unused(kw, selectors)
    specs = [_build_one(subject, sel, dict(kw), primitive) for sel in selectors]
    return compose_plot(*specs, layout=layout, on=on)
