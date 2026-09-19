"""The one guard that stands in front of every ``value in <set/dict>`` on user input.

A keyword whose vocabulary is a closed set is validated the obvious way::

    if kind not in _KIND_ALIASES:
        raise InvalidParameterError(...)   # names the valid spellings

and that line is correct for every value a caller might *mean* — and wrong for
every value they might mistype as a container.  ``in`` HASHES its left operand
first, so a list, a dict, a slice or a numpy array dies inside the membership
test with the interpreter's own words::

    TypeError: cannot use 'list' as a set element (unhashable type: 'list')

naming a set the caller has never heard of, one line before the library's typed
error would have named the spellings they wanted.  Measured on v6.0.0 at six
doors — ``force=``, ``theme=``, ``primitive=``, ``layout=``, ``backend=`` and
``kind=`` — of which ``backend=`` and ``kind=`` are the two most commonly typed.

So membership is only ever asked of a value that can answer it, and an
unhashable one falls through to the typed error that was always waiting::

    if not is_hashable(kind) or kind not in _KIND_ALIASES:
        raise InvalidParameterError(...)

This is a leaf module: it imports nothing from :mod:`tsdynamics`, so ``viz`` and
``data`` can both take it at module scope with no cycle.
"""

from __future__ import annotations

from collections.abc import Hashable

__all__ = ["is_hashable"]


def is_hashable(value: object) -> bool:
    """Whether ``value`` may be used as a set element or dict key at all.

    Ask this **before** any ``value in <set>`` / ``<dict>.get(value)`` over
    caller-supplied input, so an unhashable argument reaches the library's typed
    error instead of dying in the membership test (see the module docstring).

    Parameters
    ----------
    value : object
        The caller-supplied value about to be looked up.

    Returns
    -------
    bool
        ``True`` when ``value`` can be hashed, ``False`` otherwise.

    Examples
    --------
    >>> from tsdynamics._utils.lookup import is_hashable
    >>> is_hashable("dark"), is_hashable(["dark"])
    (True, False)
    """
    return isinstance(value, Hashable)


def __dir__() -> list[str]:
    return sorted(__all__)
