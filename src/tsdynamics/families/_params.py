"""``ParamSet`` — the ordered, fixed-key parameter container.

Since v6 it is a **``dict`` subclass**.  It was a ``MutableMapping`` wrapping a
private dict, which made three ordinary lines fail on the object a user reaches
for most::

    isinstance(lor.params, dict)   -> False
    json.dumps(lor.params)         -> TypeError: Object of type ParamSet ...
    repr(lor.params)               -> "ParamSet({'sigma': 10.0, ...})"

Everything that made it worth having is kept: the keys are frozen at
construction (a typo is refused, never silently stored), ``as_tuple()`` is the
**ordered** value tuple the engine's tape lowering consumes, and
``param_hash()`` is the process-stable key behind the DDE tape cache.

Subclassing ``dict`` *adds* five mutators the wrapper never had — ``pop``,
``popitem``, ``clear``, ``setdefault``, ``update`` — and every one of them can
add or remove a key.  All five are overridden here to route through the same key
check as ``__setitem__``, so the fixed-key contract survives the change.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ParamSet"]


class ParamSet(dict[str, Any]):
    """Ordered, fixed-key parameter container (a ``dict`` you cannot re-key).

    Keys are frozen at construction — you can change values but not add or
    remove keys.  Supports both dict-style (``p["sigma"]``) and attribute-style
    (``p.sigma``) read/write.

    Parameters
    ----------
    data : dict
        Initial key→value mapping.  All future writes must use existing keys.

    Raises
    ------
    AttributeError
        On attribute-style read or write of an undeclared key
        (``p.unknown`` / ``p.unknown = ...``).
    KeyError
        On item-style write of an undeclared key (``p["unknown"] = ...``).
    InvalidInputError
        On any structural mutation (``del``, ``pop``, ``clear``, ...).

    Examples
    --------
    >>> p = ParamSet({"sigma": 10.0, "rho": 28.0})
    >>> p
    {'sigma': 10.0, 'rho': 28.0}
    >>> isinstance(p, dict)
    True
    >>> p.as_tuple()
    (10.0, 28.0)
    >>> p["sigma"] = 12.0
    >>> p.sigma
    12.0
    >>> p["sigmaa"] = 1.0
    Traceback (most recent call last):
        ...
    KeyError: "Unknown parameter 'sigmaa'. Declared params: ['sigma', 'rho']"
    """

    __slots__ = ()

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__(data)

    # --- attribute access routes to the mapping ---

    def __getattr__(self, key: str) -> Any:
        # Only reached when normal attribute lookup fails, so ``as_tuple`` and
        # friends never come through here.
        if key in self:
            return self[key]
        raise AttributeError(f"Unknown parameter {key!r}. Declared params: {list(self)}")

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self:
            dict.__setitem__(self, key, value)
            return
        raise AttributeError(f"Unknown parameter {key!r}. Declared params: {list(self)}")

    # --- the fixed-key contract ---

    def _reject(self, key: str) -> None:
        raise KeyError(f"Unknown parameter {key!r}. Declared params: {list(self)}")

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self:
            self._reject(key)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key: str) -> None:
        from tsdynamics.errors import InvalidInputError

        raise InvalidInputError("Parameters are fixed-key — cannot delete.")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse — a parameter set has fixed keys."""
        from tsdynamics.errors import InvalidInputError

        raise InvalidInputError("Parameters are fixed-key — cannot pop.")

    def popitem(self) -> tuple[str, Any]:
        """Refuse — a parameter set has fixed keys."""
        from tsdynamics.errors import InvalidInputError

        raise InvalidInputError("Parameters are fixed-key — cannot popitem.")

    def clear(self) -> None:
        """Refuse — a parameter set has fixed keys."""
        from tsdynamics.errors import InvalidInputError

        raise InvalidInputError("Parameters are fixed-key — cannot clear.")

    def setdefault(self, key: str, default: Any = None, /) -> Any:
        """Return ``self[key]``; refuse to *insert* an undeclared key."""
        if key not in self:
            self._reject(key)
        return self[key]

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update values in place; refuse any key that is not declared."""
        incoming: dict[str, Any] = dict(*args, **kwargs)
        unknown = [k for k in incoming if k not in self]
        if unknown:
            self._reject(unknown[0])
        for k, v in incoming.items():
            dict.__setitem__(self, k, v)

    # --- helpers ---

    def as_tuple(self) -> tuple[Any, ...]:
        """Return parameter values as a tuple (declaration order).

        This ordering is the **tape contract**: the engine's lowering hands the
        values to the kernel positionally, so re-ordering them silently swaps
        parameters.  ``dict`` preserves insertion order, which is why a plain
        ``dict`` subclass is a safe home for it.
        """
        return tuple(self.values())

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy as a plain ``dict``."""
        return dict(self)

    def copy(self) -> ParamSet:
        """Return an independent :class:`ParamSet` with the same keys and values."""
        return ParamSet(dict(self))

    # --- pickling / copying ---

    def __reduce__(self) -> tuple[Any, ...]:
        """Reconstruct through ``ParamSet(dict)`` — the validating constructor."""
        return (ParamSet, (dict(self),))

    def param_hash(self) -> int:
        """
        Return a process-stable 64-bit integer hash of the current parameter values.

        Uses MD5 over a JSON-serialised representation so the result is
        reproducible across Python process restarts (unlike ``hash()``).

        The hash backs cache keys for per-system lowering / lambdify caches.
        At 64 bits the birthday-paradox collision probability reaches 50 % only
        around ``2^32 ≈ 4·10⁹`` distinct parameter sets, which is well beyond any
        realistic parameter sweep.
        """
        import hashlib
        import json

        s = json.dumps(list(self.items()), default=str)
        return int(hashlib.md5(s.encode()).hexdigest()[:16], 16)
