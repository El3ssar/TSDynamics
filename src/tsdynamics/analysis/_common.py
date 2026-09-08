r"""Input guards shared by every *data-first* analysis.

The analysis layer has two calling conventions (the frozen glossary §1): a
**system-first** analysis takes a live ``System`` and integrates it, while a
**data-first** analysis takes a *measured series* — a
:class:`~tsdynamics.data.Trajectory`, an ``(N, dim)`` point set, or a 1-D
signal.  Handing a ``System`` to a data-first analysis is the single most
common front-door mistake, and until v6 it produced::

    >>> ts.correlation_dimension(ts.Lorenz())        # doctest: +SKIP
    TypeError: float() argument must be a string or a real number, not 'Lorenz'

— a NumPy message that names neither the mistake nor the fix.  :func:`reject_system`
is the one guard every point-set / series coercion calls first, so all of them
answer with the same actionable :class:`~tsdynamics.errors.InvalidInputError`.

The ``System`` is **duck-typed** against the runtime protocol
(:mod:`tsdynamics.families.protocol`) rather than imported, because
:mod:`tsdynamics.analysis` must not import :mod:`tsdynamics.families` at module
scope — that is the deliberate families→analysis layering seam
(``families/_accessors.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__: list[str] = []

#: The ``System`` runtime protocol's method set (``families/protocol.py``).  An
#: object that implements *all* of these is a system, not a measured series.
_SYSTEM_METHODS = ("step", "state", "reinit", "trajectory")


def is_system(obj: Any) -> bool:
    """Return whether ``obj`` implements the ``System`` runtime protocol.

    Duck-typed, so it recognises a built-in family instance, a
    :class:`~tsdynamics.derived.DerivedSystem` wrapper and a user
    :class:`~tsdynamics.families.WrappedSystem` alike, without importing
    :mod:`tsdynamics.families`.

    Arrays and :class:`~tsdynamics.data.Trajectory` objects are excluded up
    front: a ``Trajectory`` carries a ``.system`` reference and could otherwise
    be confused for one by an over-eager ``getattr`` walk.

    Parameters
    ----------
    obj : Any
        The object to classify.

    Returns
    -------
    bool
        ``True`` if ``obj`` is a system (and therefore *not* measured data).
    """
    if isinstance(obj, np.ndarray) or obj is None:
        return False
    # A Trajectory is data: it carries both a time base and a state array.
    if hasattr(obj, "t") and hasattr(obj, "y"):
        return False
    return all(callable(getattr(obj, name, None)) for name in _SYSTEM_METHODS)


def front_door(system: Any) -> str:
    """Return the run-me call this particular system actually has, as source text.

    ``integrate`` for a flow, ``iterate`` for a map, and the protocol's
    ``trajectory`` for a derived wrapper (a ``PoincareMap`` has neither of the
    first two).
    """
    if callable(getattr(system, "integrate", None)):
        return "integrate(final_time=100.0, dt=0.01)"
    if callable(getattr(system, "iterate", None)):
        return "iterate(steps=10000)"
    return "trajectory(10000)"


def reject_system(data: Any, *, analysis: str | None = None, hint: str | None = None) -> None:
    """Raise if ``data`` is a ``System`` where already-computed data is wanted.

    The guard every data-first analysis (and every shared coercion —
    ``_as_points`` / ``_coerce_signal`` / ``_as_label_array`` / …) calls
    **first**, so the whole data-first surface answers this mistake with one
    message that names the system and shows the fix.

    Parameters
    ----------
    data : Any
        The first positional argument the analysis received.
    analysis : str, optional
        The public function's name, used to open the message.  Callers that sit
        in a shared coercion helper may omit it.
    hint : str, optional
        Replaces the default remedy block for an analysis whose input is *not* a
        trajectory.  The basin metrics, for instance, read a label image, so
        "run the system and pass its trajectory" would send the caller the wrong
        way; they pass the ``basins_of_attraction`` recipe instead.  ``{run}`` in
        the text is substituted with the system's own front-door call.

    Raises
    ------
    InvalidInputError
        If ``data`` implements the ``System`` protocol.  ``InvalidInputError``
        subclasses :class:`TypeError`, so an existing ``except TypeError``
        keeps catching it.

    Examples
    --------
    >>> import tsdynamics as ts
    >>> from tsdynamics.analysis._common import reject_system
    >>> from tsdynamics.errors import InvalidInputError
    >>> try:
    ...     reject_system(ts.Lorenz(), analysis="correlation_dimension")
    ... except InvalidInputError as err:
    ...     print(str(err).splitlines()[0])
    correlation_dimension() expects measured data, not a System (got Lorenz). Run the system first and pass its trajectory:
    >>> reject_system(np.zeros((10, 3)))          # measured data passes through
    """
    if not is_system(data):
        return

    from tsdynamics.errors import InvalidInputError

    name = type(data).__name__
    who = f"{analysis}()" if analysis else "this analysis"
    run = front_door(data)
    call = f"{analysis}(" if analysis else "analysis("
    remedy = (
        hint.format(run=run)
        if hint is not None
        else (
            "Run the system first and pass its trajectory:\n"
            f"    traj = system.{run}\n"
            f"    {call}traj)            # the full (N, dim) point set\n"
            f"    {call}traj.y[:, 0])    # a single scalar component"
        )
    )
    raise InvalidInputError(f"{who} expects measured data, not a System (got {name}). {remedy}")


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
