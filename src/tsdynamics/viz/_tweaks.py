"""The panel-scoped / figure-scoped partition of :class:`PlotSpec`'s fluent tweaks.

Every public fluent tweak on :class:`~tsdynamics.viz.spec.PlotSpec` (the
mutate-and-return-``self`` methods that chain — ``relabel`` / ``rescale`` /
``limits`` / ``style`` / ``theme`` / …) is exactly one of two things:

**panel-scoped**
    It describes *a panel*: axis labels, scales, limits, ticks, gridlines, layer
    style, colours, the colour range.  On a :data:`~tsdynamics.viz.spec.PlotKind.COMPOSITE`
    spec — which has no axes and no layers of its own, only ``panels`` — such a
    tweak **must recurse into every panel**, or it is a silent no-op: the user
    sees a figure and believes the tweak landed.  (Before v6 all twelve of them
    were exactly that: ``ts.viz.plot(a, b, layout="stack").recolor("red")``
    left ``to_dict()["panels"]`` byte-identical.)

**figure-scoped**
    It describes *the figure*: the title, the theme, the background, the figure
    size, the animation timeline.  These belong on the composite itself and must
    **not** be pushed down — a panel that pins its own theme would stop
    inheriting a later ``composite.theme("dark")``, and per-panel animations
    would desynchronise the lockstep master clock.

This module owns that partition and the recursion; the tweak *bodies* stay in
``spec.py`` (they differ in what they write — only the recursion is shared).

The partition is **enforced**, not documented: :func:`tweak_scopes` reads the
marker each decorator stamps on the function, and the governance gate in
``tests/test_viz_spec.py`` fails when any public method returning a ``PlotSpec``
carries no scope.  A *new* tweak method therefore cannot be added without making
a forwarding decision.

This module imports nothing from :mod:`tsdynamics` — it is a leaf helper, so it
cannot create a cycle with ``spec.py`` (which imports it).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Literal, TypeVar

__all__ = [
    "SCOPE_ATTR",
    "figure_scoped",
    "panel_scoped",
    "panel_scoped_custom",
    "tweak_scopes",
]

#: The attribute each decorator stamps on a tweak method naming its scope
#: (``"panel"`` / ``"figure"``).  Read by :func:`tweak_scopes` and the gate.
SCOPE_ATTR = "__tsd_tweak_scope__"

Scope = Literal["panel", "figure"]

F = TypeVar("F", bound=Callable[..., Any])


def figure_scoped[F: Callable[..., Any]](func: F) -> F:
    """Mark a tweak as **figure-scoped**: it applies to the composite, not its panels.

    A pure marker — it adds no behaviour.  Use it for tweaks that describe the
    whole figure (``title``, ``theme``, ``background``, ``size``, the animation
    timeline), which panels either inherit at render time or must not carry.
    """
    setattr(func, SCOPE_ATTR, "figure")
    return func


def panel_scoped_custom[F: Callable[..., Any]](func: F) -> F:
    """Mark a tweak as **panel-scoped** with its own hand-written recursion.

    For the one tweak whose composite semantics are not "call me on each panel
    with the same arguments" — :meth:`~tsdynamics.viz.spec.PlotSpec.recolor`,
    where colour *i* addresses *panel* i rather than *layer* i.  The method body
    owns the recursion; this marker only records the scope for the gate.
    """
    setattr(func, SCOPE_ATTR, "panel")
    return func


def panel_scoped(
    *, figure_only: tuple[str, ...] = (), writes_theme: bool = False
) -> Callable[[F], F]:
    """Mark a tweak **panel-scoped** and forward it into every child panel.

    The wrapper calls the same method on each entry of ``self.panels`` (with the
    same arguments, minus any keyword named in ``figure_only``) and *then* runs
    the original body on ``self``.  A single-panel spec has no ``panels``, so the
    wrapper is inert there — the pre-v6 behaviour of every non-composite spec is
    bit-for-bit unchanged.

    Recursion is naturally depth-first: a panel that is itself a composite
    forwards on down.

    Parameters
    ----------
    figure_only : tuple of str, optional
        Keyword names that describe the *figure* and must not be pushed into the
        panels — e.g. ``relabel(title=...)``, which is one figure title, not one
        title per panel.  When a call passes *only* figure-only keywords (and no
        positional arguments), no panel call is made at all.
    writes_theme : bool, optional
        Whether the tweak pins a :class:`~tsdynamics.viz.style.Theme` on the spec
        it runs against (``palette`` / ``font`` / ``grid(color=, alpha=)``).  Such
        a tweak resolves its base theme as "mine, else the *global* default" —
        correct for a lone spec, but wrong for a panel, which should start from
        the theme it inherits from its composite.  Left unhandled, forwarding
        ``composite.theme("dark").palette(...)`` stamps a ``default``-based theme
        on every panel and the figure renders dark chrome on white axes, with
        light-on-white tick labels.  So when the composite pins a theme, this
        wrapper **seeds each themeless panel with the composite's resolved theme
        first**, making the inheritance explicit at the one moment the panel is
        about to overwrite it.  Inert when the composite pins no theme (the panel
        would have resolved to the same global default anyway).  Default ``False``.

    Returns
    -------
    callable
        The decorator.
    """

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            panels = getattr(self, "panels", None)
            if panels:
                forwarded = {k: v for k, v in kwargs.items() if k not in figure_only}
                # Skip only when the caller passed nothing but figure-only keys —
                # a no-argument call (``autocolor()``, ``grid()``) must still reach
                # the panels.
                if args or forwarded or not figure_only:
                    inherited = getattr(self, "_theme", None) if writes_theme else None
                    for panel in panels:
                        if inherited is not None and getattr(panel, "_theme", None) is None:
                            panel._theme = inherited
                        getattr(panel, func.__name__)(*args, **forwarded)
            return func(self, *args, **kwargs)

        setattr(wrapper, SCOPE_ATTR, "panel")
        return wrapper  # type: ignore[return-value]

    return decorate


def tweak_scopes(cls: type) -> dict[str, Scope]:
    """Return ``{method name: scope}`` for every scope-marked method on ``cls``.

    The introspection half of the governance gate: pair it with the set of public
    methods that return the class itself (the fluent tweaks) to prove none is
    unclassified.
    """
    out: dict[str, Scope] = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = getattr(cls, name, None)
        scope = getattr(attr, SCOPE_ATTR, None)
        if scope is not None:
            out[name] = scope
    return out
