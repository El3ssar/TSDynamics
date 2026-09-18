"""``subject.plot`` as a callable NAMESPACE — the compensation ruling A2 promised.

Lives in :mod:`tsdynamics._utils` (the LEAF package) because both
:mod:`tsdynamics.data` and :mod:`tsdynamics.families` bind it, and it must import
no plotting library at module scope — ``import tsdynamics`` still pulls in none.

Ruling A2 took every analysis off the system and the trajectory and made
tab-completion plus error messages the whole discovery story.  For plotting that
left the string-naming front door as the only route — and **no way to discover
the string**, because the transform registry is module-level while the thing a
user is holding is an object.

So ``subject.plot`` is both:

- a **verb** — ``traj.plot()``, ``traj.plot(color="red")``, exactly as before;
- a **namespace** — ``traj.plot.psd()``, ``lor.plot.nullclines()``,
  ``traj.plot.<TAB>`` listing every transform that admits *this* subject.

``x.plot.<name>(**kw)`` is defined as ``ts.plot(x, "<name>", **kw)``, so there is
one implementation and one grammar; the namespace only makes the strings
discoverable.  A wrong guess names the nearest transform that admits this
subject, and says how many there are.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PlotNamespace", "plot_namespace", "plot_seam_error"]


def plot_seam_error(owner: str, subject: str) -> AttributeError:
    """Build the ``to_plot_spec`` refusal — one sentence, two runnable lines.

    ``to_plot_spec`` was the third public way to say *build a plot and do not
    draw it*, alongside ``ts.plot(x)`` (which returns the very same
    :class:`~tsdynamics.viz.spec.Plot`, rendering nothing) and ``x.plot()``.
    v6 keeps the machinery under the dunder ``__plot_spec__`` — the one
    predicate ``ts.plot`` uses to recognise a subject — and retires the name.

    The message is built here rather than written out three times so that a
    :class:`~tsdynamics.data.Trajectory`, a system and all 32 analysis results
    answer the guess identically, in their own nouns.

    Parameters
    ----------
    owner : str
        The class name to quote, e.g. ``"Trajectory"``.
    subject : str
        The expression to use in the runnable lines, e.g. ``"traj"``.

    Returns
    -------
    AttributeError
        Raise it; do not return it.
    """
    from tsdynamics.errors import taught

    return taught(
        AttributeError(
            f"{owner!r} object has no attribute 'to_plot_spec': building a plot without "
            f"drawing it is what ts.plot({subject}) already does.\n"
            f"    ts.plot({subject})     # the Plot object — nothing is rendered\n"
            f"    {subject}.plot()       # the same thing, styled at the door\n"
            f"    (the seam itself is the dunder {subject}.__plot_spec__, not a verb you type)"
        ),
        "to_plot_spec",
    )


def _bound_transform(subject: Any, name: str) -> Any:
    """Return ``subject.plot.<name>`` — a function carrying THAT transform's signature.

    The namespace entry used to be a bare :func:`functools.partial`, so
    ``help(traj.plot.orbit_diagram)`` printed ``ts.plot``'s generic
    ``(*things, layout, rows, cols, share_x, …)`` — the composition vocabulary,
    which is the one thing that call is *not* asking about.  The owner went
    looking for the orbit diagram's ``param=`` / ``values=`` / ``points=`` and
    found none of them, though all three are real and a *wrong* keyword already
    listed them back.

    So the entry is a real function with ``__name__``, ``__doc__`` and
    ``__signature__`` taken from the transform's own ``compute`` — which makes
    ``help()``, ``inspect.signature`` and IPython's ``?`` all correct, from one
    source, for every registered transform including one a user wrote this
    morning.
    """
    from tsdynamics.viz import plot as _plot
    from tsdynamics.viz import transforms as _transforms

    record = _transforms.get(name)

    def bound(**options: Any) -> Any:
        return _plot(subject, name, **options)

    bound.__name__ = name
    bound.__qualname__ = f"{type(subject).__name__}.plot.{name}"
    bound.__doc__ = record.help_text()
    bound.__signature__ = record.call_signature()  # type: ignore[attr-defined]
    return bound


class PlotNamespace:
    """A bound ``subject.plot`` — callable, and a namespace of transform names."""

    __slots__ = ("_subject", "_verb")

    def __init__(self, subject: Any, verb: Any) -> None:
        self._subject = subject
        self._verb = verb

    def __call__(self, *transforms: Any, **kwargs: Any) -> Any:
        """Draw this subject — the pre-v6 spelling, unchanged."""
        return self._verb(*transforms, **kwargs)

    def _names(self) -> list[str]:
        """Every registered transform whose declared source admits this subject."""
        try:
            from tsdynamics.viz import transforms as _transforms

            found = _transforms.find(subject=self._subject)
        except Exception:  # pragma: no cover - viz layer unavailable
            return []
        return sorted(r if isinstance(r, str) else getattr(r, "name", str(r)) for r in found)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        names = self._names()
        if name in names:
            return _bound_transform(self._subject, name)

        import difflib

        close = difflib.get_close_matches(name, names, n=2, cutoff=0.55)
        who = type(self._subject).__name__
        hint = "".join(f"\n    subject.plot.{c}()" for c in close)
        raise AttributeError(
            f"no plot transform named {name!r} for a {who}."
            + (f"  Did you mean:{hint}" if close else "")
            + f"\n    subject.plot.<TAB>            # the {len(names)} that draw this"
            + "\n    ts.viz.compatibility()        # the whole matrix"
        )

    def __dir__(self) -> list[str]:
        """Tab-completion is the point: list the transforms that draw this subject."""
        return self._names()

    def __repr__(self) -> str:
        names = self._names()
        head = f"<plot namespace for {type(self._subject).__name__}: {len(names)} transforms>"
        shown = ", ".join(names[:8]) + (" …" if len(names) > 8 else "")
        return f"{head}\n    {shown}\n    call it — subject.plot() — for the default view"


class plot_namespace:  # noqa: N801 — a descriptor reads as an attribute, not a class
    """Descriptor binding :class:`PlotNamespace` over an existing ``plot`` method.

    Used as ``plot = plot_namespace(_plot_impl)``: reading ``obj.plot`` gives the
    callable namespace, ``obj.plot(...)`` calls ``_plot_impl`` unchanged.
    """

    __slots__ = ("_impl", "_doc")

    def __init__(self, impl: Any) -> None:
        self._impl = impl
        self._doc = impl.__doc__

    @property
    def __doc__(self) -> Any:  # type: ignore[override]
        """The wrapped method's docstring, so ``help(obj.plot)`` still works."""
        return self._doc

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self._impl
        return PlotNamespace(obj, self._impl.__get__(obj, objtype))


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
