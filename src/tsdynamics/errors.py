"""The TSDynamics exception hierarchy and the value-naming error standard.

Every error TSDynamics raises *on purpose* descends from
:class:`TSDynamicsError`, so a caller can write ``except ts.TSDynamicsError`` to
catch "anything the library rejected deliberately" without also swallowing
genuine bugs (a ``KeyError`` from the caller's own dict, say).

Why multiple inheritance off the stdlib types
---------------------------------------------
The leaf classes deliberately inherit from **both** :class:`TSDynamicsError`
**and** the stdlib exception a user would already be catching:

- :class:`InvalidParameterError` is a :class:`ValueError`
- :class:`InvalidInputError` is a :class:`TypeError`

So existing ``except ValueError`` / ``except TypeError`` handlers keep catching
these — the hierarchy is purely *additive*.  ``isinstance(err, ValueError)`` and
``isinstance(err, ts.TSDynamicsError)`` are both true.  This mirrors the NumPy 2.0
choice (``np.exceptions.AxisError`` is a ``ValueError`` *and* an ``IndexError``)
and is the recommendation the v4 design dossier landed on.

The value-naming standard
-------------------------
Every deliberate raise that rejects a *value* follows one shape — the shape the
already-excellent ``method=`` / ``backend=`` / ``set_state`` messages established:

1. **name the offending value** (``final_time must be > 0, got -5``),
2. **state the rule or list the valid options**, and
3. **suggest the fix** where one exists.

:func:`invalid_value` builds exactly that message so the standard is applied
uniformly rather than re-spelled at every raise site.

Examples
--------
>>> from tsdynamics.errors import TSDynamicsError, InvalidParameterError
>>> issubclass(InvalidParameterError, ValueError)
True
>>> issubclass(InvalidParameterError, TSDynamicsError)
True
>>> try:
...     raise InvalidParameterError("dt must be > 0, got 0.0")
... except ValueError as err:          # still caught by plain ValueError
...     print(type(err).__name__)
InvalidParameterError
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = [
    "BackendError",
    "ConvergenceError",
    "InvalidInputError",
    "InvalidParameterError",
    "TSDynamicsError",
    "invalid_value",
]


class TSDynamicsError(Exception):
    """Base class for every error TSDynamics raises on purpose.

    Catch this to handle "anything the library rejected deliberately" without
    also swallowing unrelated bugs.  Concrete leaf classes additionally inherit
    from the relevant stdlib exception (e.g. :class:`InvalidParameterError` is
    also a :class:`ValueError`), so existing ``except ValueError`` handlers keep
    working — the hierarchy only *adds* a common ancestor.
    """


class InvalidParameterError(TSDynamicsError, ValueError):
    """A parameter / keyword value is out of range or otherwise unacceptable.

    Use for a *value* that is the wrong magnitude, sign, or choice — a
    non-positive ``final_time`` / ``dt``, an unknown solver ``method=``, an
    unknown parameter name, a typo'd system attribute.  Subclasses
    :class:`ValueError`, so ``except ValueError`` still catches it.
    """


class InvalidInputError(TSDynamicsError, TypeError):
    """An argument is the wrong *type* or *shape* for what the call needs.

    Use when the caller handed in the wrong kind of object — a ``System`` where a
    measured series was expected, an initial condition of the wrong length, an
    array of the wrong dimensionality.  Subclasses :class:`TypeError`, so
    ``except TypeError`` still catches it.
    """


class ConvergenceError(TSDynamicsError, RuntimeError):
    """An iterative numerical routine failed to converge.

    Use for Newton / shooting / fixed-point iterations that exhaust their
    budget, and for an integration that diverged.  Subclasses
    :class:`RuntimeError` so existing ``except RuntimeError`` handlers (the
    divergence convention) keep working.
    """


class BackendError(TSDynamicsError, RuntimeError):
    """A compute backend (the Rust engine, a solver kernel) failed or is absent.

    A **reserved** base for backend-side failures — an engine that is not built, a
    kernel that refused a problem, or an FFI-boundary failure surfaced with domain
    framing rather than a raw extension traceback.  It is the designed home for
    such conditions; today it has no concrete leaves
    (:class:`tsdynamics.engine.run.EngineNotAvailableError` still derives from
    :class:`RuntimeError` directly, *not* from this class, so do not assume
    ``isinstance(err, BackendError)`` for it yet).  Subclasses
    :class:`RuntimeError`.
    """


def invalid_value(
    name: str,
    value: Any,
    *,
    rule: str | None = None,
    options: Iterable[Any] | None = None,
    hint: str | None = None,
) -> InvalidParameterError:
    """Build an :class:`InvalidParameterError` in the value-naming standard.

    The single helper behind the v4 error standard: it always *names the
    offending value*, then states the rule or lists the valid options, and
    finally appends a fix hint when one is given.  Returning (not raising) the
    exception keeps the call site's ``raise`` visible to readers and linters::

        raise invalid_value("final_time", final_time, rule="must be > 0")

    Parameters
    ----------
    name : str
        The parameter / value being rejected (e.g. ``"final_time"``).
    value : Any
        The bad value, rendered with ``repr`` so strings are quoted.
    rule : str, optional
        The rule the value violated, phrased to read after the name
        (``"must be > 0"`` → ``"final_time must be > 0, got -5"``).  Mutually
        complementary with ``options``; supply at least one.
    options : iterable, optional
        The valid choices to list (``"got 'gpu'; choose from [...]"``).
    hint : str, optional
        A trailing fix suggestion (a full sentence).

    Returns
    -------
    InvalidParameterError
        The constructed (not yet raised) exception.

    Examples
    --------
    >>> str(invalid_value("final_time", -5, rule="must be > 0"))
    'final_time must be > 0, got -5'
    >>> str(invalid_value("backend", "gpu", options=["interp", "jit"]))
    "unknown backend 'gpu'; choose from ['interp', 'jit']"
    """
    if rule is not None:
        msg = f"{name} {rule}, got {value!r}"
    elif options is not None:
        opts = list(options)
        msg = f"unknown {name} {value!r}; choose from {opts}"
    else:
        msg = f"invalid {name}: {value!r}"
    if hint:
        msg = f"{msg}. {hint}"
    return InvalidParameterError(msg)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
