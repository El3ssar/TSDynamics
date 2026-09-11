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

The runnable-line standard
--------------------------
Naming the mistake is necessary and *not sufficient*.  When a call fails because
the user typed the wrong shape of call — not merely a wrong number — the message
must end with **the line they should type instead**, indented and complete enough
to paste::

    orbit_diagram needs a discrete-time view ...          # describes the mistake
    wrap the flow in a section first:                     # ... and shows the fix
        ts.analysis.orbit_diagram(system.poincare("z", 27.0), "rho", values)

:func:`remedy` formats that block, so every site spells it the same way and the
polish gate (``tests/test_polish_standards.py``) can *decide* whether a message
carries a runnable line by parsing its indented lines as Python.  Two rules make
the block worth pasting:

- **name the call the user actually made**, not the internal function it
  delegates to (a user who typed ``ts.basins`` must not be answered about
  ``basins_of_attraction``); and
- **fill in their own values** — the system's real dimension, its declared
  component names, the length their array actually had — so the line runs as-is
  rather than needing to be decoded first.

The hierarchy at a glance
-------------------------
============================ =================== ===========================
Class                        Stdlib base         Raised for
============================ =================== ===========================
:class:`TSDynamicsError`     :class:`Exception`  (abstract common ancestor)
:class:`InvalidParameterError` :class:`ValueError` a bad *value* (range/choice)
:class:`InvalidInputError`   :class:`TypeError`  a bad argument *type*/*shape*
:class:`ConvergenceError`    :class:`RuntimeError` divergence / non-convergence
:class:`StepBudgetError`     :class:`ConvergenceError` a *stalled* run (finite state)
:class:`BackendError`        :class:`RuntimeError` a compute-backend failure
:class:`MovedInV6`           :class:`ImportError` a name that moved or left in v6
============================ =================== ===========================

:class:`MovedInV6` is the odd one out and deliberately so: it is the only class
here that inherits :class:`ImportError` rather than the stdlib type its *raise
site* would otherwise use.  See its docstring for the measurement behind that.

One concrete leaf lives outside this module to keep it import-light (it must load
while the package is still initialising):
:class:`tsdynamics.engine.run.EngineNotAvailableError` subclasses
:class:`BackendError` and is raised when the compiled ``tsdynamics._rust``
extension is absent — ``except BackendError`` (or ``except RuntimeError``) catches
it.

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
    "MovedInV6",
    "StepBudgetError",
    "TSDynamicsError",
    "invalid_value",
    "remedy",
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


class StepBudgetError(ConvergenceError):
    """An integration ran out of solver steps while its state stayed finite.

    The *stalled*, not *diverged*, half of "the run did not reach the final
    time".  The engine caps the number of solver steps it will spend on one
    output segment; hitting that cap with a perfectly finite state means the
    model is fine and the **solver settings** are not — an explicit method on a
    stiff problem, or a tolerance tighter than the dynamics can meet.  The
    remedy is a looser ``rtol``/``atol``, an implicit ``method="bdf"``, or a
    shorter span; it is *not* to go hunting for a blow-up.

    Subclasses :class:`ConvergenceError` (and so :class:`RuntimeError`), so
    every handler that already catches engine divergence keeps catching this —
    the split is additive, for callers that want to tell the two apart.
    """


class BackendError(TSDynamicsError, RuntimeError):
    """A compute backend (the Rust engine, a solver kernel) failed or is absent.

    A base for backend-side failures — an engine that is not built, a kernel that
    refused a problem, or an FFI-boundary failure surfaced with domain framing
    rather than a raw extension traceback.  Its concrete leaf
    :class:`tsdynamics.engine.run.EngineNotAvailableError` is raised when the
    compiled ``tsdynamics._rust`` extension is missing, so
    ``isinstance(err, BackendError)`` catches it.  Subclasses
    :class:`RuntimeError`, so legacy ``except RuntimeError`` handlers still apply.
    """


class MovedInV6(TSDynamicsError, ImportError):  # noqa: N818 - see the docstring
    """A public name that v6 moved to another address, renamed, or removed.

    Raised by a package ``__getattr__`` on an **exact hit** in one of the v6
    redirect tables (:mod:`tsdynamics._redirects`) or in a public submodule's
    ``__all__``.  The message carries the line to type instead — it *is* the
    migration guide.

    Why this is an ``ImportError`` and not an ``AttributeError``
    -----------------------------------------------------------
    Because the failing spelling that matters is ``from tsdynamics import X``,
    and CPython **discards** a module ``__getattr__``'s message for that spelling
    whenever the exception matches ``AttributeError``.  Measured on CPython
    3.14.2 against a two-line probe package:

    .. code-block:: text

        __getattr__ raises AttributeError  ->  ImportError: cannot import name
                                               'attr_case' from 'pkg'   (TEXT LOST)
        __getattr__ raises ImportError     ->  ImportError: <custom text>
                                                                        (TEXT KEPT)

    A class inheriting *both* is impossible — ``class M(AttributeError,
    ImportError)`` raises ``TypeError: multiple bases have instance lay-out
    conflict`` — so the two spellings cannot be served by one type, and the
    import spelling wins.

    The cost, stated
    ----------------
    ``hasattr(ts, name)`` **raises** instead of returning ``False`` for a name in
    a redirect table, because :func:`hasattr` only swallows ``AttributeError``.
    That is confined to the enumerated dead names and is the point: a v5 script
    probing ``hasattr(ts, "permutation_entropy")`` should not silently take the
    "not installed" branch when the honest answer is "that moved out of this
    library".  A *guess* (``ts.random_typo``) stays an ``AttributeError``, so
    ``hasattr`` keeps working for every other name in the universe.

    Subclasses :class:`ImportError`, so ``except ImportError`` — the handler a
    caller already wraps an optional dependency in — catches it.

    Examples
    --------
    >>> from tsdynamics.errors import MovedInV6
    >>> issubclass(MovedInV6, ImportError)
    True
    >>> issubclass(MovedInV6, TSDynamicsError)
    True
    >>> issubclass(MovedInV6, AttributeError)
    False
    """


def remedy(*lines: str, lead: str | None = None) -> str:
    """Format the *runnable* fix block that closes an error message.

    The formatter behind the runnable-line standard documented at the top of this
    module: a short lead-in sentence, then one indented line per statement the
    user should type.  Keeping it in one helper means every site spells the fix
    the same way, and lets the polish gate decide mechanically whether a message
    carries a line that actually parses as Python.

    Parameters
    ----------
    *lines : str
        The source lines to show, in the order they should be typed.  Each must
        be a complete statement (a call, or an assignment whose value is a call)
        — a fragment the user still has to finish is not a remedy.
    lead : str, optional
        A sentence introducing the block (e.g. ``"wrap the flow in a section
        first:"``).  Rendered on its own line above the code.

    Returns
    -------
    str
        The block, opening with a newline so it appends directly to a message.

    Examples
    --------
    >>> print("orbit_diagram needs a discrete-time view." + remedy(
    ...     'ts.analysis.orbit_diagram(sys.poincare("z", 27.0), "rho", values)',
    ...     lead="Wrap the flow in a section first:",
    ... ))
    orbit_diagram needs a discrete-time view.
    Wrap the flow in a section first:
        ts.analysis.orbit_diagram(sys.poincare("z", 27.0), "rho", values)
    """
    body = "\n".join(f"    {line}" for line in lines)
    return f"\n{lead}\n{body}" if lead else f"\n{body}"


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
