"""Core abstractions: ParamSet, MetaStore, SystemBase.

:class:`Trajectory` is *not* defined here — it is a data type the families
merely produce, so it lives in :mod:`tsdynamics.data`.  It is re-exported below
(and from :mod:`tsdynamics.families`) so the family modules and existing
import sites keep resolving ``Trajectory`` to the one canonical object.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Any, ClassVar, cast

import numpy as np

# Trajectory's canonical home is the data layer; re-exported here for the
# family modules (``from .base import SystemBase, Trajectory``) and back-compat.
# tsdynamics.data is a leaf package (no top-level tsdynamics imports), so this
# is cycle-safe.
from tsdynamics.data.trajectory import Trajectory
from tsdynamics.errors import InvalidInputError, remedy

# The Plottable mixin (stream VIZ-SYSTEM-PLOT) gives every system a ``.plot()`` /
# ``to_plot_spec()``.  It imports tsdynamics.viz only lazily (inside its methods),
# so importing the family bases here keeps ``import tsdynamics`` visualization-free.
from ._plottable import SystemPlottable

__all__ = ["MetaStore", "ParamSet", "SystemBase", "Trajectory"]

# ---------------------------------------------------------------------------
# ParamSet
# ---------------------------------------------------------------------------


class ParamSet(MutableMapping[str, Any]):
    """
    Ordered, fixed-key parameter container.

    Keys are frozen at construction time — you can change values but not add or
    remove keys. Supports both dict-style (``p["sigma"]``) and attribute-style
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
        On any attempt to delete a key (the key set is frozen).

    Examples
    --------
    >>> p = ParamSet({"sigma": 10.0, "rho": 28.0})
    >>> p.sigma
    10.0
    >>> p.sigma = 15.0
    >>> p["sigma"]
    15.0
    >>> p.unknown = 5.0            # raises AttributeError
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    # --- attribute access routes to _data ---

    def __getattr__(self, key: str) -> Any:
        d = object.__getattribute__(self, "_data")
        if key in d:
            return d[key]
        raise AttributeError(f"Unknown parameter {key!r}. Declared params: {list(d)}")

    def __setattr__(self, key: str, value: Any) -> None:
        d = object.__getattribute__(self, "_data")
        if key in d:
            d[key] = value
        else:
            raise AttributeError(f"Unknown parameter {key!r}. Declared params: {list(d)}")

    # --- MutableMapping protocol ---

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._data:
            raise KeyError(f"Unknown parameter {key!r}. Declared params: {list(self._data)}")
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        from tsdynamics.errors import InvalidInputError

        raise InvalidInputError("Parameters are fixed-key — cannot delete.")

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # --- helpers ---

    def as_tuple(self) -> tuple[Any, ...]:
        """Return parameter values as a tuple (insertion order)."""
        return tuple(self._data.values())

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy as a plain dict."""
        return dict(self._data)

    # --- pickling / copying ---
    #
    # ``__slots__`` plus a validating ``__setattr__`` defeats the default
    # ``object.__reduce_ex__`` path: the generic slot restorer calls
    # ``setattr(inst, "_data", ...)`` on a *fresh* instance whose ``_data`` does
    # not exist yet, so :meth:`__setattr__` raises ``AttributeError: '...ParamSet'
    # object has no attribute '_data'``.  That single AttributeError made every
    # system, and every trajectory holding one, impossible to pickle **or**
    # deep-copy — which blocks ``multiprocessing`` / ``joblib`` parameter sweeps
    # and disk caching.  Declaring the state protocol explicitly fixes both
    # (``copy.deepcopy`` goes through the same ``__reduce_ex__``).

    def __getstate__(self) -> dict[str, Any]:
        """Return the picklable state (a plain dict of the parameter values)."""
        return dict(self._data)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__`, bypassing the validating setattr."""
        object.__setattr__(self, "_data", dict(state))

    def param_hash(self) -> int:
        """
        Return a process-stable 64-bit integer hash of the current parameter values.

        Uses MD5 over a JSON-serialised representation so the result is
        reproducible across Python process restarts (unlike ``hash()``).

        The hash backs cache keys for per-system lowering / lambdify caches.
        At 64 bits the birthday-paradox
        collision probability reaches 50 % only around ``2^32 ≈ 4·10⁹``
        distinct parameter sets, which is well beyond any realistic
        parameter sweep.  The previous 32-bit width hit the same threshold
        at only ``2^16 ≈ 65 000`` sets, which a sufficiently large sweep
        could plausibly reach — and a collision there would silently
        return a compiled artifact built for a different parameter set.
        """
        import hashlib
        import json

        s = json.dumps(list(self._data.items()), default=str)
        return int(hashlib.md5(s.encode()).hexdigest()[:16], 16)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"ParamSet({{{items}}})"


# ---------------------------------------------------------------------------
# MetaStore
# ---------------------------------------------------------------------------

#: Maximum length of a single metadata value's repr in :meth:`MetaStore.__repr__`.
#: Longer reprs (e.g. large Lyapunov spectra) are truncated with a trailing
#: ``...`` so the store's repr stays a readable single line.
_MAX_META_VALUE_REPR = 60


def _short_value_repr(value: Any, maxlen: int = _MAX_META_VALUE_REPR) -> str:
    """Return a compact, single-line ``repr`` of a metadata value.

    Internal whitespace is collapsed (so a multi-line array repr renders on one
    line) and an over-long repr is truncated with a trailing ``...`` so the
    enclosing :class:`MetaStore` repr remains a readable single line even when a
    value is a large array or object.
    """
    try:
        text = repr(value)
    except Exception:
        # A value whose __repr__ raises must not break the store's repr.
        return f"<{type(value).__name__}>"
    text = " ".join(text.split())
    if len(text) > maxlen:
        text = text[: maxlen - 3] + "..."
    return text


class MetaStore(MutableMapping[str, Any]):
    """
    Append-with-history metadata store for computed results.

    Behaves like a dict for everyday use (``meta["lyapunov_spectrum"]``
    reads/writes the *latest* value), but every write is appended rather
    than overwritten, so earlier results survive::

        sys.meta.record("lyapunov_spectrum", spec, dt=0.1, final_time=200.0)
        sys.meta["lyapunov_spectrum"]            # latest value
        sys.meta.history("lyapunov_spectrum")    # every record, with context

    Equality compares the latest values against a plain dict (or another
    MetaStore), preserving ``sys.meta == {}`` style assertions.
    """

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: dict[str, list[dict[str, Any]]] = {}

    def record(self, key: str, value: Any, **context: Any) -> Any:
        """Append ``value`` under ``key`` with optional context kwargs.

        Each call stores a new record ``{"value", "context", "timestamp"}`` —
        earlier records under the same key are preserved (retrievable with
        :meth:`history`), and ``meta[key]`` returns the most recent value.

        Parameters
        ----------
        key : str
            The result name (e.g. ``"lyapunov_spectrum"``).
        value : Any
            The computed value to store.
        **context
            Free-form context recorded alongside the value (e.g. ``dt``,
            ``final_time``), surfaced by :meth:`history`.

        Returns
        -------
        Any
            ``value`` unchanged, so a result can be recorded and returned in one
            expression (``return self.meta.record("k", v)``).
        """
        import time

        self._records.setdefault(key, []).append(
            {"value": value, "context": context, "timestamp": time.time()}
        )
        return value

    def history(self, key: str) -> list[dict[str, Any]]:
        """Return every record for ``key`` (oldest first), with context."""
        return list(self._records.get(key, []))

    def latest(self) -> dict[str, Any]:
        """Return a plain dict of the latest value per key."""
        return {k: recs[-1]["value"] for k, recs in self._records.items()}

    def copy(self) -> MetaStore:
        """Return an independent store holding the same records.

        The record *lists* are fresh, so recording on the copy never appends to
        the original's history; the recorded values themselves are shared (a
        shallow copy, matching :func:`copy.copy` semantics).
        """
        clone = MetaStore()
        clone._records = {k: [dict(r) for r in recs] for k, recs in self._records.items()}
        return clone

    # --- pickling (``__slots__`` needs an explicit state protocol) ---

    def __getstate__(self) -> dict[str, list[dict[str, Any]]]:
        """Return the picklable state (the full record history)."""
        return self._records

    def __setstate__(self, state: dict[str, list[dict[str, Any]]]) -> None:
        """Restore from :meth:`__getstate__`."""
        object.__setattr__(self, "_records", dict(state))

    # --- MutableMapping protocol (operates on latest values) ---

    def __setitem__(self, key: str, value: Any) -> None:
        self.record(key, value)

    def __getitem__(self, key: str) -> Any:
        recs = self._records.get(key)
        if not recs:
            raise KeyError(key)
        return recs[-1]["value"]

    def __delitem__(self, key: str) -> None:
        del self._records[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MetaStore):
            return self.latest() == other.latest()
        if isinstance(other, dict):
            return self.latest() == other
        return NotImplemented

    def __repr__(self) -> str:
        """Show the latest value per key, annotating overwritten keys with ``(xN)``.

        Examples
        --------
        >>> m = MetaStore()
        >>> m["dt"] = 0.01
        >>> m["system"] = "lorenz"
        >>> m["system"] = "lorenz"
        >>> m
        MetaStore(dt=0.01, system='lorenz' (x2))
        """
        if not self._records:
            return "MetaStore()"
        parts = []
        for key, recs in self._records.items():
            value_repr = _short_value_repr(recs[-1]["value"])
            suffix = f" (x{len(recs)})" if len(recs) > 1 else ""
            parts.append(f"{key}={value_repr}{suffix}")
        return f"MetaStore({', '.join(parts)})"


# ---------------------------------------------------------------------------
# SystemBase
# ---------------------------------------------------------------------------

#: The named keyword arguments of :meth:`SystemBase.__init__`.  Because the
#: constructor also accepts free ``**param_kwargs`` (``Lorenz(sigma=12.0)``), a
#: declared parameter sharing one of these names could never be reached as a
#: keyword — :meth:`SystemBase.__init_subclass__` refuses such a class outright.
#: Filled in from the real signature immediately after the class body, so it can
#: never drift from the constructor it describes.
_RESERVED_INIT_KEYWORDS: frozenset[str] = frozenset()


class SystemBase(SystemPlottable):
    """
    Abstract base class for all dynamical systems.

    Provides:
    - ``params`` — a :class:`ParamSet` holding the system's parameter values.
      Attribute access on the system is transparently forwarded to ``params``.
    - ``dim`` — integer state-space dimension.
    - ``ic`` — optional initial conditions array.
    - ``meta`` — dict for storing computed metadata (Lyapunov spectra, etc.).
    - ``copy()`` / ``with_params()`` for safe cloning.
    - ``resolve_ic()`` for uniform IC resolution across subclasses.

    Class-level declarations
    ------------------------
    Subclasses should declare at class level::

        class Lorenz(ContinuousSystem):
            params = {"sigma": 10.0, "rho": 28.0, "beta": 8/3}
            dim = 3

    Constructor overrides
    ---------------------
    Every declared parameter is a plain constructor keyword — the most natural
    line a user will type::

        lor = Lorenz(rho=30.0, ic=[1.0, 0.0, 0.0])

    The ``params=`` dict spelling stays supported and means exactly the same
    thing (it is the way to pass a name computed at runtime)::

        lor = Lorenz(params={"rho": 30.0}, ic=[1.0, 0.0, 0.0])

    The constructor raises
    :class:`~tsdynamics.errors.InvalidParameterError` (a ``ValueError``
    subclass) for any unknown parameter key — whether it arrives as
    ``params={"rhoo": 30.0}`` or as ``rhoo=30.0`` — so a typo fails loudly,
    naming the declared parameters, instead of being silently ignored.  Giving
    the *same* parameter through both channels is also an error: there is
    deliberately no precedence rule to memorise.

    The five keywords :meth:`__init__` reserves for itself (``params``, ``ic``,
    ``dim``, ``field_shape``, ``seed``) cannot also be parameter names — a class
    declaring one is refused at definition time by :meth:`__init_subclass__`
    rather than silently shadowing it.

    See Also
    --------
    ParamSet : the fixed-key parameter container behind ``params``.
    MetaStore : the append-with-history store behind ``meta``.
    resolve_ic : the uniform initial-condition resolution helper.
    """

    #: Class-level parameter defaults.  Keys are frozen once the instance
    #: is created — only values may change.
    params: ClassVar[dict[str, Any]] = {}

    #: State-space dimension.  Set at class level for fixed-dim systems;
    #: override in ``__init__`` for variable-dim systems (e.g. Lorenz96).
    dim: ClassVar[int | None] = None

    #: Optional class-level default initial conditions.  Used when no ``ic``
    #: argument is supplied to the constructor or to ``resolve_ic``.  Useful
    #: for systems whose attractor basin is small (e.g. Tinkerbell) so random
    #: ICs in ``U[0, 1)^dim`` always diverge.
    default_ic: ClassVar[Any | None] = None

    #: Optional component names, e.g. ``("x", "y", "z")``.  Enables named
    #: access on trajectories (``traj["x"]``) and labelled docs figures.
    variables: ClassVar[tuple[str, ...] | None] = None

    #: Optional **spatial** grid shape for a spatially-extended system whose state
    #: vector is a flattened field — e.g. ``(Ny, Nx)`` for a 2-D
    #: reaction-diffusion / PDE lattice, or ``(N,)`` for a 1-D profile.  When set,
    #: it is recorded on ``traj.meta["field_shape"]`` so the visualization layer's
    #: ``kind="field"`` recipe (stream VIZ-SPATIAL-FIELD) reshapes each per-time
    #: state vector to that grid and plays it as a *spatial-field movie* (a line
    #: for a 1-D field, an ``imshow`` heatmap for a 2-D field).  A system that
    #: packs several field blocks into one state vector (e.g. Gray–Scott's
    #: ``[u, v]``) also declares :attr:`field_labels` and gives the grid of one
    #: block here.  ``None`` for an ordinary low-dimensional system.
    _field_shape: ClassVar[tuple[int, ...] | None] = None

    #: Optional names of the field blocks packed into the flattened state vector
    #: of a spatially-extended system (e.g. ``("u", "v")`` for a two-species
    #: reaction-diffusion model).  Each block has :attr:`_field_shape` grid cells,
    #: laid out contiguously in state order.  Lets the ``kind="field"`` recipe
    #: select which block to plot via ``components="u"|"v"`` (the first block is
    #: the default).  ``None`` for a single-block field.
    field_labels: ClassVar[tuple[str, ...] | None] = None

    #: Optional literature reference for the system, e.g.
    #: ``"Lorenz (1963), J. Atmos. Sci. 20, 130"``.  Surfaced in the docs.
    reference: ClassVar[str | None] = None

    #: Optional known Lyapunov data used by the bulk known-value tests::
    #:
    #:     known_lyapunov = {
    #:         "spectrum": (0.906, 0.0, -14.57),   # literature values
    #:         "atol": 0.1,                        # per-exponent tolerance
    #:         "kwargs": {"final_time": 300.0},    # forwarded to lyapunov_spectrum
    #:         "source": "Sprott (2003)",
    #:     }
    known_lyapunov: ClassVar[dict[str, Any] | None] = None

    #: The runtime backend this family's engine-dispatch seam uses when a caller
    #: does not name one.  Every concrete family sets it to ``"jit"`` (the Rust
    #: engine's Cranelift JIT, with the compiled-evaluator cache paying the
    #: compile once per distinct system — it was ``"interp"`` before v6); the
    #: abstract base keeps ``"reference"`` (the wheel-free pure-Python oracle).
    #: Read by the family ``integrate`` / ``iterate`` methods and by
    #: :meth:`_dispatch`, so "the default backend" lives in exactly one place.
    _default_backend: ClassVar[str] = "reference"

    #: Instance-dict keys that are **runtime caches / live stepping state**, not
    #: part of a system's identity.  They are dropped by :meth:`__getstate__`
    #: (pickle / ``copy.deepcopy``) and by :meth:`__copy__`, because several hold
    #: objects that cannot be pickled at all — most notably the Rust
    #: ``OdeStepper`` behind ``_ode_stepper`` and the compiled problem behind
    #: ``_engine_problem``.  A restored system is *cold*: call ``reinit()`` to
    #: resume stepping.  **Any new per-instance runtime cache added by a family
    #: must be listed here**, or pickling that family will start failing.
    _TRANSIENT_STATE: ClassVar[frozenset[str]] = frozenset(
        {
            "_accessor_cache",  # SystemBase topical accessors (hold self)
            "_ic_rng",  # the IC Generator (rebuilt from ``_ic_seed``)
            "_ode_stepper",  # ContinuousSystem: the Rust resumable stepper
            "_engine_problem",  # ContinuousSystem: the lowered Problem
            "_step_tape_arrays",  # ContinuousSystem: the FFI tape arrays
            "_stepper",  # StochasticSystem: the per-step SDE context
        }
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A declared parameter whose name collides with one of this constructor's
        # own keyword arguments would be *unreachable* as a constructor keyword —
        # ``Sys(dim=3)`` would silently set the state-space dimension instead of
        # the parameter ``dim``.  Refuse the class at definition time (the same
        # moment ``DiscreteMap`` rejects a params/_step signature mismatch) rather
        # than shipping a silent shadow.  Only classes that actually *use* this
        # constructor are checked: a subclass with its own ``__init__`` (the
        # variable-dimension systems) owns its own signature.
        if cls.__init__ is SystemBase.__init__:
            shadowed = sorted(_RESERVED_INIT_KEYWORDS & set(cls.params or {}))
            if shadowed:
                raise TypeError(
                    f"{cls.__name__}: parameter name(s) {shadowed} collide with "
                    f"SystemBase.__init__'s own keyword argument(s) "
                    f"{sorted(_RESERVED_INIT_KEYWORDS)}, so they could never be "
                    f"passed as constructor keywords. Rename the parameter(s), or "
                    f"give the class its own __init__."
                )
        # The framework bases (ContinuousSystem, DelaySystem, DiscreteMap, ...)
        # live under tsdynamics.families and are not registrable systems themselves.
        if not cls.__module__.startswith("tsdynamics.families"):
            from tsdynamics.registry import register_class

            register_class(cls)

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        ic: Any | None = None,
        dim: int | None = None,
        field_shape: tuple[int, ...] | None = None,
        seed: int | None = None,
        **param_kwargs: Any,
    ) -> None:
        """Initialise a system from its class defaults plus instance overrides.

        Parameters
        ----------
        params : dict, optional
            Per-instance parameter overrides.  Every key must already exist in
            the class-level :attr:`params` defaults; unknown keys raise.
        ic : array-like, optional
            Initial conditions.  Stored on ``self.ic`` (as a ``float`` array) and
            used by :meth:`resolve_ic` when no explicit ``ic`` is later supplied.
            An ``ic`` given here is **explicit**: it is never silently replaced by
            a random draw (a diverging one raises instead).
        dim : int, optional
            State-space dimension override for variable-dimension systems.  Falls
            back to the class-level :attr:`dim` when omitted.
        field_shape : tuple of int, optional
            Spatial grid shape override for a spatially-extended system (see
            :attr:`_field_shape`).  Falls back to the class-level value.
        seed : int, optional
            Seed for the **initial-condition draw** used when neither ``ic`` nor
            :attr:`default_ic` supplies one.  The draw runs on a private
            :class:`numpy.random.Generator`, so it is reproducible *and* never
            touches the global ``numpy.random`` stream.  When omitted a fresh
            OS-entropy seed is drawn on first use and recorded on
            ``traj.meta["ic_seed"]``, so any run can be reproduced exactly by
            passing that value back as ``seed=``.
        **param_kwargs
            Per-parameter overrides given as plain keywords — ``Lorenz(sigma=12.0)``
            is exactly ``Lorenz(params={"sigma": 12.0})``.  Each name must be a
            declared parameter of this class; an unknown one raises and names the
            valid options.  A name given **both** here and inside ``params=`` is an
            error rather than a silent precedence rule.

        Raises
        ------
        InvalidParameterError
            If ``params`` or ``**param_kwargs`` contains a key that is not a
            declared parameter, or if the same parameter is given twice (once in
            ``params=`` and once as a keyword).

        Examples
        --------
        >>> Lorenz(sigma=12.0).sigma            # the natural spelling
        12.0
        >>> Lorenz(params={"sigma": 12.0}).sigma   # equivalent, still supported
        12.0
        """
        # Build ParamSet from class defaults + constructor overrides.  The two
        # override channels (``params=`` and free keywords) are merged here, and
        # both are validated against the *declared* parameter names.
        defaults = dict(type(self).params)
        overrides: dict[str, Any] = dict(params) if params else {}
        if param_kwargs:
            duplicated = sorted(set(param_kwargs) & set(overrides))
            if duplicated:
                from tsdynamics.errors import InvalidParameterError

                raise InvalidParameterError(
                    f"{type(self).__name__}: parameter(s) {duplicated} given twice — "
                    f"once in params= and once as a keyword. Pass each parameter "
                    f"exactly once (there is deliberately no precedence rule)."
                )
            overrides.update(param_kwargs)
        if overrides:
            unknown = set(overrides) - set(defaults)
            if unknown:
                # A misspelt parameter is nearly always one character off a
                # declared one, so name the closest match and spell the whole
                # constructor call rather than only listing what was valid.
                import difflib

                from tsdynamics.errors import InvalidParameterError

                cls_name = type(self).__name__
                guesses = {
                    bad: (
                        [n for n in sorted(defaults) if n.lower() == bad.lower()]
                        or difflib.get_close_matches(bad, sorted(defaults), n=1, cutoff=0.5)
                    )
                    for bad in sorted(unknown)
                }
                fixed = {bad: g[0] for bad, g in guesses.items() if g}
                did_you_mean = (
                    " Did you mean " + ", ".join(f"{b!r} -> {g!r}" for b, g in fixed.items()) + "?"
                    if fixed
                    else ""
                )
                call = ", ".join(
                    f"{fixed.get(bad, bad)}={overrides[bad]!r}" for bad in sorted(unknown)
                )
                raise InvalidParameterError(
                    f"{cls_name}: unknown parameter(s) {sorted(unknown)}. "
                    f"Declared: {sorted(defaults)}.{did_you_mean}"
                    + remedy(
                        f"{cls_name}({call})"
                        if fixed
                        else f"{cls_name}({sorted(defaults)[0]}={defaults[sorted(defaults)[0]]!r})",
                        lead="Spell it as:" if fixed else "The declared parameters are set like:",
                    )
                )
            defaults.update(overrides)
        object.__setattr__(self, "params", ParamSet(defaults))

        # dim: constructor arg > class attribute
        resolved_dim = dim if dim is not None else type(self).dim
        if resolved_dim is None:
            # Every downstream consumer does ``int(self.dim)``, so an undeclared
            # dimension used to surface as NumPy's ``int() argument must be ...
            # not 'NoneType'`` from deep inside resolve_ic -- which names neither
            # the class nor the attribute.  Writing a system class is the first
            # thing a new user does, so refuse it here, where the fix is one line.
            raise InvalidInputError(
                f"{type(self).__name__} does not declare its state-space dimension, "
                f"so the library cannot tell how many components its state has. "
                f"`dim` is the number of equations `_equations` returns."
                + remedy(
                    f"class {type(self).__name__}({type(self).__bases__[0].__name__}):",
                    "    dim = 3                       # the number of state components",
                    '    variables = ("x", "y", "z")   # optional, names them',
                    lead="Declare it on the class:",
                )
                + remedy(
                    f"{type(self).__name__}(dim=3)",
                    lead="...or pass it for this one instance:",
                )
            )
        object.__setattr__(self, "dim", resolved_dim)

        # field_shape (spatially-extended systems): constructor arg > class
        # attribute.  Set via object.__setattr__ so a custom-N instance overrides
        # the class default without tripping the ClassVar assignment rule (mirrors
        # ``dim``).  ``_provenance`` reads the instance value onto ``traj.meta``.
        resolved_field_shape = field_shape if field_shape is not None else type(self)._field_shape
        object.__setattr__(self, "_field_shape", resolved_field_shape)

        # Initial conditions.  ``_ic_explicit`` records whether the CURRENT
        # ``self.ic`` was chosen by the user (constructor / an explicit ``ic=``)
        # or merely auto-resolved (a random draw).  A user-chosen IC must never
        # be silently swapped for a random one — see ``DiscreteMap.iterate``.
        # ``np.array(..., copy=True)``, not ``asarray``: a float64 array passed in
        # would otherwise be *shared* with the caller, so mutating either side
        # silently moved the other's initial condition.  It is also what made
        # ``with_params`` (which forwards ``ic=self.ic``) hand back a system
        # aliasing the source's ic array.
        ic_arr = np.array(ic, dtype=float, copy=True) if ic is not None else None
        object.__setattr__(self, "ic", ic_arr)
        object.__setattr__(self, "_ic_explicit", ic is not None)

        # The private initial-condition RNG (see ``resolve_ic``).  Only the seed
        # is persisted; the Generator itself is a transient cache.
        object.__setattr__(self, "_ic_seed", None if seed is None else int(seed))
        object.__setattr__(self, "_ic_rng", None)

        # Metadata store: computed properties (Lyapunov, etc.) accumulate here
        # with history — repeated runs append instead of overwriting.
        object.__setattr__(self, "meta", MetaStore())

    # --- transparent attribute routing through params ---

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails
        try:
            params = object.__getattribute__(self, "params")
            return params[name]
        except (AttributeError, KeyError) as err:
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}") from err

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            params = object.__getattribute__(self, "params")
            if name in params:
                params[name] = value
                return
        except AttributeError:
            # No ParamSet yet (mid-construction) — let the assignment through.
            object.__setattr__(self, name, value)
            return

        # A public name that is neither a declared parameter, an already-set
        # instance attribute, nor a class-level attribute/method is almost
        # always a typo for a parameter (``lor.sigmaa = 99`` for ``sigma``).
        # ``with_params(sigmaa=99)`` already rejects exactly this — so reject it
        # here too rather than silently storing a stray attribute while the real
        # parameter stays unchanged.  Private/dunder names (``self._t_now``, the
        # families' step state) and anything the class already defines pass
        # straight through.
        if not name.startswith("_") and name not in self.__dict__ and not hasattr(type(self), name):
            from tsdynamics.errors import invalid_value

            declared = list(params)
            raise invalid_value(
                f"attribute {name!r} on {type(self).__name__}",
                value,
                rule=f"is not a declared parameter {declared}",
                hint=(
                    f"check the spelling, or set a known parameter "
                    f"(e.g. {type(self).__name__.lower()}.{declared[0]} = ...)"
                    if declared
                    else "this system declares no parameters"
                ),
            )
        object.__setattr__(self, name, value)

    # --- cloning / pickling ---

    def copy(self) -> SystemBase:
        """
        Return a deep copy with the same class, params, and ic.

        The copy has its own independent ``params`` and ``meta`` stores, so
        mutating the clone's parameters or recording metadata on it never
        affects the original.

        Returns
        -------
        SystemBase
            A fresh instance of the same subclass with copied ``params`` and
            ``ic`` and an empty ``meta`` store.

        See Also
        --------
        __copy__ : ``copy.copy(system)`` — the same independence, but it *keeps*
            the recorded ``meta`` (and does not re-run ``__init__``).
        """
        return type(self)(
            params=cast(ParamSet, self.params).as_dict(),
            ic=self.ic.copy() if self.ic is not None else None,
        )

    def _clone_state(self) -> dict[str, Any]:
        """Return the identity-defining instance state, with fresh containers.

        The mutable containers (``params`` / ``meta`` / ``ic``) are rebuilt so a
        clone can never write through to the original, and the runtime caches in
        :attr:`_TRANSIENT_STATE` (the Rust stepper, the lowered problem, the
        accessor cache, the IC Generator) are dropped — they are rebuilt on
        demand and several cannot be pickled at all.
        """
        state = {k: v for k, v in self.__dict__.items() if k not in self._TRANSIENT_STATE}
        state["params"] = ParamSet(cast(ParamSet, self.params).as_dict())
        meta = self.__dict__.get("meta")
        state["meta"] = meta.copy() if isinstance(meta, MetaStore) else MetaStore()
        ic = self.__dict__.get("ic")
        state["ic"] = None if ic is None else np.array(ic, dtype=float, copy=True)
        return state

    def __copy__(self) -> SystemBase:
        """Return an **independent** shallow copy (``copy.copy(system)``).

        Historically the default ``copy.copy`` produced an *aliased* system: the
        clone shared the original's :class:`ParamSet`, :class:`MetaStore` and
        ``ic`` array, so ``copy.copy(lor).sigma = 99`` silently rewrote the
        original's ``sigma``.  The copy now owns those three containers (the
        recorded *values* are still shared, which is what a shallow copy means),
        so mutating a copy can never reach back into the original.
        """
        clone = type(self).__new__(type(self))
        clone.__dict__.update(self._clone_state())
        return clone

    def __deepcopy__(self, memo: dict[int, Any]) -> SystemBase:
        """Return a fully independent deep copy (``copy.deepcopy(system)``)."""
        import copy as _copy

        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        clone.__dict__.update(_copy.deepcopy(self._clone_state(), memo))
        return clone

    def __getstate__(self) -> dict[str, Any]:
        """Return the picklable state — identity only, no runtime caches.

        A pickled/unpickled system is **cold**: the live stepping state
        (``reinit``/``step``) is not carried over, because it is backed by a Rust
        stepper handle that cannot cross a process boundary.  Call ``reinit()``
        on the restored system to resume stepping.  Everything that defines the
        system — class, ``params``, ``ic``, ``dim``, ``meta``, the IC seed — is
        preserved, which is what a ``multiprocessing`` / ``joblib`` parameter
        sweep or an on-disk cache needs.
        """
        return {k: v for k, v in self.__dict__.items() if k not in self._TRANSIENT_STATE}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__`, bypassing the validating setattr."""
        self.__dict__.update(state)
        # Older pickles / hand-built states may predate these fields.
        self.__dict__.setdefault("_ic_explicit", self.__dict__.get("ic") is not None)
        self.__dict__.setdefault("_ic_seed", None)
        self.__dict__["_ic_rng"] = None

    def with_params(self, **overrides: Any) -> SystemBase:
        """
        Return a **new** system with some parameters overridden.

        Does not mutate ``self``.  Designed for parameter sweeps::

            for rho in np.linspace(0, 50, 200):
                traj = base_system.with_params(rho=rho).integrate(final_time=50)

        Parameters
        ----------
        **overrides
            New parameter values.  Keys must exist in ``params``.

        Returns
        -------
        SystemBase
            New instance of the same subclass.
        """
        new_p = {**cast(ParamSet, self.params).as_dict(), **overrides}
        return type(self)(params=new_p, ic=self.ic)

    # --- IC resolution ---

    def ic_generator(self, seed: int | None = None) -> np.random.Generator:
        """Return this system's **private** initial-condition ``Generator``.

        The random-IC fallback in :meth:`resolve_ic` draws from here, never from
        the global ``numpy.random`` stream: a plain ``system.run()`` must not
        perturb a caller's own ``np.random.seed(0)`` reproducibility.

        The generator is seeded from (in priority order) the ``seed`` argument,
        the ``seed=`` given to the constructor, or — when neither is supplied — a
        fresh OS-entropy seed drawn once per instance.  Whichever it is, the
        resolved seed is remembered and published on ``traj.meta["ic_seed"]``, so
        a run started from a random IC can always be reproduced by constructing
        the system again with that ``seed=``.

        Parameters
        ----------
        seed : int, optional
            Re-seed the generator (and adopt this seed as the system's).

        Returns
        -------
        numpy.random.Generator
        """
        if seed is not None:
            gen = np.random.default_rng(int(seed))
            object.__setattr__(self, "_ic_seed", int(seed))
            object.__setattr__(self, "_ic_rng", gen)
            return gen
        cached: np.random.Generator | None = self.__dict__.get("_ic_rng")
        if cached is not None:
            return cached
        resolved = self.__dict__.get("_ic_seed")
        if resolved is None:
            # No seed was asked for: draw one from OS entropy *and record it*,
            # so the run stays as random as before but is now reproducible.
            resolved = int(cast(int, np.random.SeedSequence().entropy))
            object.__setattr__(self, "_ic_seed", resolved)
        gen = np.random.default_rng(resolved)
        object.__setattr__(self, "_ic_rng", gen)
        return gen

    def _coerce_ic(self, value: Any, source: str) -> np.ndarray:
        """Coerce one initial-condition candidate to a ``(dim,)`` float array.

        The single place a wrong-length / non-numeric initial condition is
        rejected, so the three sources :meth:`resolve_ic` draws from (the ``ic=``
        argument, the latched ``self.ic``, the class-level ``default_ic``) all
        answer with one message that names the system's dimension, its component
        names, and the line to type — instead of leaking NumPy's
        ``cannot reshape array of size 2 into shape (3,)``.

        Parameters
        ----------
        value : array-like
            The candidate initial condition.
        source : str
            Where it came from, quoted back to the user (``"ic="`` /
            ``"system.ic"`` / ``"default_ic"``) so a stale latched IC is not
            mistaken for the one just passed.

        Returns
        -------
        ndarray, shape (dim,)

        Raises
        ------
        InvalidInputError
            If the value has the wrong number of components or is not numeric.
        """
        dim = int(cast(int, self.dim))
        name = type(self).__name__
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as err:
            raise InvalidInputError(
                f"{name}: {source} must be {dim} numbers (one per state component), "
                f"got {value!r}." + self._ic_example(dim)
            ) from err
        if arr.size != dim:
            names = getattr(type(self), "variables", None)
            components = f" ({', '.join(names)})" if names and len(names) == dim else ""
            raise InvalidInputError(
                f"{name} has {dim} state components{components}, so {source} needs "
                f"{dim} numbers — got {arr.size}: {np.asarray(value).tolist()!r}."
                + self._ic_example(dim)
            )
        return arr.reshape(dim)

    def _ic_example(self, dim: int) -> str:
        """Return the runnable one-liner showing a correctly sized ``ic=`` for this system.

        The receiver is spelled as the class constructor (``Lorenz()``) so the
        line pastes into a bare REPL — except for a variable-dimension system,
        whose dimension comes from a structural parameter the constructor would
        have to repeat, where ``system`` keeps the line honest.
        """
        example = "[" + ", ".join(["1.0"] * dim) + "]" if dim <= 8 else f"np.ones({dim})"
        run = (
            f"iterate(steps=1000, ic={example})"
            if callable(getattr(self, "iterate", None))
            else f"run(final_time=100.0, ic={example})"
        )
        structural: frozenset[str] = getattr(type(self), "_structural_params", frozenset())
        who = "system" if structural else f"{type(self).__name__}()"
        return remedy(f"{who}.{run}")

    def resolve_ic(self, ic: Any | None = None, *, seed: int | None = None) -> np.ndarray:
        """
        Resolve initial conditions consistently.

        Priority:

        1. ``ic`` argument (if provided)
        2. ``self.ic`` (set by a previous integration / iteration)
        3. ``type(self).default_ic`` (class-level default, if declared)
        4. Random ``U[0, 1)^dim`` from the system's **private**
           :meth:`ic_generator` (never the global ``numpy.random`` stream)

        The resolved IC is stored in ``self.ic`` so subsequent calls without
        an explicit ``ic`` reproduce the same initial state.

        Parameters
        ----------
        ic : array-like or None
            An explicit initial condition.  Marks ``self.ic`` as user-chosen, so
            it is never silently replaced by a random draw — *unless* it is
            bit-for-bit the value already on ``self.ic``, which is an internal
            re-resolution (the engine problem builders round-trip the resolved
            array) and leaves the existing user-chosen/auto-drawn flag alone.
        seed : int, optional
            Seed for the random-IC fallback (cases 1–3 ignore it, since no draw
            happens).  Equivalent to the constructor's ``seed=``, applied here.

        Returns
        -------
        ndarray, shape (dim,)
        """
        explicit = True
        if ic is not None:
            arr = self._coerce_ic(ic, "ic=")
            # A *re-resolution of the value already on the instance* is not a new
            # user choice.  The engine problem builders (``ode_problem`` /
            # ``map_problem``) hand the array ``resolve_ic`` just returned straight
            # back into ``resolve_ic``, so without this an internally drawn random
            # IC was promoted to "user-chosen" the moment it was used — which
            # silently disabled ``DiscreteMap.iterate``'s random-IC retry from the
            # second call onwards, and made its diagnostic claim the IC "was
            # supplied explicitly" about an IC the library itself had drawn.
            prior = self.__dict__.get("ic")
            if prior is not None and prior.shape == arr.shape and np.array_equal(prior, arr):
                explicit = bool(self.__dict__.get("_ic_explicit", True))
        elif self.ic is not None:
            arr = self._coerce_ic(self.ic, "system.ic")
            explicit = bool(self.__dict__.get("_ic_explicit", False))
        elif type(self).default_ic is not None:
            arr = self._coerce_ic(type(self).default_ic, "default_ic")
            # A class-declared default is not a *user* choice: a system whose
            # declared default lands off-basin keeps the random-IC retry.
            explicit = False
        else:
            arr = self.ic_generator(seed).random(cast(int, self.dim))
            explicit = False
        object.__setattr__(self, "ic", arr.copy())
        object.__setattr__(self, "_ic_explicit", explicit)
        return arr

    def _ic_rollback(self) -> Any:
        """Return a context manager restoring ``ic`` if the block raises.

        :meth:`resolve_ic` commits the resolved initial condition to ``self.ic``
        *before* the run happens, so a run that then fails used to leave the bad
        IC latched on the instance — and every later, unrelated analysis silently
        started from it.  Wrapping a run in this guard makes a failure leave the
        object exactly as it was.
        """
        from contextlib import contextmanager

        @contextmanager
        def _guard() -> Iterator[None]:
            prior_ic = self.__dict__.get("ic")
            prior_flag = self.__dict__.get("_ic_explicit", False)
            try:
                yield
            except BaseException:
                object.__setattr__(self, "ic", prior_ic)
                object.__setattr__(self, "_ic_explicit", prior_flag)
                raise

        return _guard()

    # --- engine-dispatch seam ---

    def _dispatch(self, *, backend: str, seed: int | None = None, **kwargs: Any) -> Trajectory:
        """Route this system's engine-path run through the one engine seam.

        Every family's ``interp`` / ``jit`` / ``reference`` integration branch
        funnels here, so the FFI marshalling, the divergence guards and the
        engine-path provenance live once in
        :func:`tsdynamics.engine.run.integrate` rather than being re-implemented
        per family.  Family-specific run inputs pass straight through as keyword
        arguments — ``history`` for a delay system, ``ic`` / ``method`` /
        ``rtol`` / ``atol`` / ``t0`` for the continuous families, ``final_time``
        (the step count) for a map.

        Diagonal-Itô SDEs are the one family that does **not** route here: the
        generic seam cannot carry their noise seed and step-as-noise-scale, so
        :class:`~tsdynamics.families.stochastic.StochasticSystem` drives the
        dedicated ``run.sde_integrate_dense`` / ``run.sde_ensemble_final`` seam
        instead (and ``run.integrate`` refuses an SDE problem).

        ``seed`` is the **initial-condition** seed — the one every family's
        trajectory producer accepts (``DiscreteMap.iterate(seed=)`` has always
        had it; the flow families gained it for symmetry).  It is resolved here,
        *inside* the rollback guard, and only bites when a random draw actually
        happens: an explicit ``ic``, a previously resolved ``self.ic`` and a
        class-level ``default_ic`` all take priority, exactly as in
        :meth:`resolve_ic`.  The resolved array is then handed down as ``ic=``,
        which ``run.integrate``'s own ``resolve_ic`` recognises as a
        re-resolution of the value already on the instance (so the
        user-chosen/auto-drawn flag is preserved and the random-IC retry logic
        is unaffected).

        The call is wrapped in :meth:`_ic_rollback`, so a run that raises (a
        divergence, an interrupt, an engine fault) leaves ``self.ic`` untouched
        instead of latching the offending initial condition onto the instance.
        """
        from tsdynamics.engine import run

        with self._ic_rollback():
            if seed is not None:
                kwargs["ic"] = self.resolve_ic(kwargs.get("ic"), seed=seed)
            return run.integrate(self, backend=backend, **kwargs)

    # --- misc ---

    def _provenance(self, **extra: Any) -> dict[str, Any]:
        """Build the provenance dict attached to trajectories as ``traj.meta``."""
        from tsdynamics import __version__

        prov: dict[str, Any] = {
            "system": type(self).__name__,
            "params": cast(ParamSet, self.params).as_dict(),
            "tsdynamics": __version__,
        }
        # The initial-condition seed, whenever one exists (the user passed
        # ``seed=``, or a random draw happened and recorded its OS-entropy seed).
        # With it and ``meta["ic"]`` a run started from a random IC is exactly
        # reproducible: ``type(sys)(params=..., seed=meta["ic_seed"])``.
        ic_seed = self.__dict__.get("_ic_seed")
        if ic_seed is not None:
            prov["ic_seed"] = int(ic_seed)
        prov.update(extra)
        # A spatially-extended system carries its grid shape (and field-block
        # labels) so a bare Trajectory can be played as a spatial-field movie
        # (stream VIZ-SPATIAL-FIELD) without the producing system in hand.  Read
        # the *instance* attribute first so a custom-N instance (which sets
        # ``self._field_shape`` in __init__) wins over the class default.
        field_shape = getattr(self, "_field_shape", None)
        if field_shape is not None:
            prov["field_shape"] = tuple(int(n) for n in field_shape)
            field_labels = getattr(self, "field_labels", None)
            if field_labels is not None:
                prov["field_labels"] = tuple(str(s) for s in field_labels)
        return prov

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{type(self).__name__}({params_str})"

    # --- object-side analysis surface (additive convenience) ------------- #
    #
    # The canonical analysis surface is the *free functions* in
    # ``tsdynamics.analysis``.  The accessors below make that toolkit
    # *discoverable from a system in hand*: pressing ``<TAB>`` on ``sys.``
    # reveals a handful of grouped topical namespaces (the xarray accessor
    # pattern) instead of nothing, and each delegates to the matching free
    # function with the system bound — adding zero behaviour.  All analysis /
    # derived imports are function-local (here and in ``_accessors.py``) so this
    # module stays free of import cycles.
    #
    # The accessors cache on the instance (so ``sys.lyap is sys.lyap``) via a
    # small helper that stows the built accessor in an ``_accessor_cache`` dict
    # set with ``object.__setattr__`` (the family ``__getattr__`` /
    # ``__setattr__`` route ordinary attribute access through ``params``, so a
    # plain ``functools.cached_property`` cannot be used).

    def _topical_accessor(self, name: str, factory: Any) -> Any:
        """Return the cached topical accessor ``name``, building it once."""
        cache = self.__dict__.get("_accessor_cache")
        if cache is None:
            cache = {}
            object.__setattr__(self, "_accessor_cache", cache)
        acc = cache.get(name)
        if acc is None:
            acc = factory(self)
            cache[name] = acc
        return acc

    @property
    def lyap(self) -> Any:
        """Lyapunov-exponent estimators bound to this system.

        A cached :class:`~tsdynamics.families._accessors.LyapunovAccessor`
        exposing ``.spectrum()`` / ``.maximal()`` / ``.from_data()`` — each
        delegating to :func:`tsdynamics.analysis.lyapunov_spectrum`,
        :func:`~tsdynamics.analysis.max_lyapunov` and
        :func:`~tsdynamics.analysis.lyapunov_from_data` with this system bound.
        """
        from tsdynamics.families._accessors import LyapunovAccessor

        return self._topical_accessor("lyap", LyapunovAccessor)

    @property
    def chaos(self) -> Any:
        """Chaos indicators bound to this system.

        A cached :class:`~tsdynamics.families._accessors.ChaosAccessor` exposing
        ``.gali()`` / ``.expansion_entropy()`` / ``.zero_one()`` — delegating to
        :func:`tsdynamics.analysis.gali`,
        :func:`~tsdynamics.analysis.expansion_entropy` and
        :func:`~tsdynamics.analysis.zero_one_test`.
        """
        from tsdynamics.families._accessors import ChaosAccessor

        return self._topical_accessor("chaos", ChaosAccessor)

    @property
    def dims(self) -> Any:
        """Fractal-dimension estimators bound to this system.

        A cached :class:`~tsdynamics.families._accessors.DimensionsAccessor`
        (``.correlation()`` / ``.generalized()`` / …) delegating to the
        ``*_dimension`` free functions.  These consume a point set; omitting the
        ``data`` argument runs the system first (an implicit integration).
        """
        from tsdynamics.families._accessors import DimensionsAccessor

        return self._topical_accessor("dims", DimensionsAccessor)

    @property
    def recurrence(self) -> Any:
        """Recurrence-quantification estimators bound to this system.

        A cached :class:`~tsdynamics.families._accessors.RecurrenceAccessor`
        (``.matrix()`` / ``.rqa()`` / ``.windowed()``) delegating to
        :func:`tsdynamics.analysis.recurrence_matrix`,
        :func:`~tsdynamics.analysis.rqa` and
        :func:`~tsdynamics.analysis.windowed_rqa`.
        """
        from tsdynamics.families._accessors import RecurrenceAccessor

        return self._topical_accessor("recurrence", RecurrenceAccessor)

    # --- first-class analysis / derived verbs (additive convenience) ----- #

    def fixed_points(self, **kwargs: Any) -> Any:
        """Find fixed points / equilibria of this system.

        Delegates to :func:`tsdynamics.analysis.fixed_points` with this system
        bound — returns the same list of
        :class:`~tsdynamics.analysis.FixedPoint`.
        """
        from tsdynamics.analysis import fixed_points

        return fixed_points(self, **kwargs)

    def poincare(
        self,
        section: Any = None,
        at: float = 0.0,
        *,
        plane: tuple[Any, ...] | None = None,
        direction: int = +1,
        **kwargs: Any,
    ) -> Any:
        """Build a :class:`~tsdynamics.derived.PoincareMap` of this flow.

        The friendly ``section=`` (a component index or name) + ``at=`` (the
        crossing value) spelling is sugar over the wrapper's ``plane`` tuple; an
        explicit ``plane=(normal, offset)`` may be passed instead for an
        arbitrary-normal plane.  Calling ``.run(...)`` (or ``.trajectory(...)``)
        on the returned map collects crossings — the returned object is exactly
        ``PoincareMap(self, plane, direction=...)``.

        Parameters
        ----------
        section : int or str, optional
            State component whose level set defines the section.  A string is
            resolved against the system's ``variables``.  Ignored when an
            explicit ``plane`` is given.
        at : float, default 0.0
            The crossing value for ``section`` (the plane offset).
        plane : tuple, optional
            The raw ``(component_index, value)`` or ``(normal, offset)`` tuple
            passed straight to :class:`~tsdynamics.derived.PoincareMap`.  Takes
            precedence over ``section`` / ``at``.
        direction : int, default +1
            Crossing direction (sign).
        **kwargs
            Forwarded to :class:`~tsdynamics.derived.PoincareMap` (``dt``,
            ``max_time``).
        """
        from tsdynamics.derived import PoincareMap

        if plane is None:
            from tsdynamics.errors import InvalidParameterError

            if section is None:
                raise InvalidParameterError(
                    "poincare() needs either `section=` (with `at=`) or an explicit `plane=`."
                )
            comp = section
            if isinstance(comp, str):
                names = getattr(type(self), "variables", None)
                if names is None:
                    raise InvalidParameterError(
                        f"{type(self).__name__} declares no `variables`; "
                        f"pass an integer `section=` (component index)."
                    )
                comp = names.index(comp)
            plane = (int(comp), float(at))
        return PoincareMap(self, plane, direction=direction, **kwargs)

    def stroboscope(self, period: float | None = None, **kwargs: Any) -> Any:
        """Build a :class:`~tsdynamics.derived.StroboscopicMap` of this forced flow.

        When ``period`` is omitted the forcing period is **inferred from the
        system** — from a ``forcing_period`` / ``drive_period`` hook (used
        verbatim) or a ``drive_frequency`` / ``omega`` hook (taken as the
        angular drive frequency, so the period is ``2*pi / omega``).  The
        catalogue's forced systems (e.g. :class:`~tsdynamics.systems.Duffing`,
        whose autonomising phase obeys ``zdot = omega``) follow the ``omega``
        convention, so ``ForcedDuffing().stroboscope()`` just works.  Pass
        ``period=`` to override the inference (or when no drive hook exists).
        Equivalent to ``StroboscopicMap(self, period)``.

        Parameters
        ----------
        period : float, optional
            The forcing period.  When ``None`` (the default) it is inferred from
            the system; a system with no recognised drive hook raises, asking for
            an explicit ``period=``.
        **kwargs
            Forwarded to :class:`~tsdynamics.derived.StroboscopicMap`.

        Raises
        ------
        InvalidParameterError
            When ``period`` is omitted and the system exposes no drive hook to
            infer it from.
        """
        from tsdynamics.derived import StroboscopicMap

        if period is None:
            from tsdynamics.errors import invalid_value
            from tsdynamics.families._accessors import infer_forcing_period

            try:
                period = infer_forcing_period(self)
            except KeyError as err:
                raise invalid_value(
                    f"period for {type(self).__name__}.stroboscope()",
                    value=None,
                    rule="could not be inferred from the system",
                    hint=(
                        "pass an explicit `period=` (e.g. `2 * np.pi / omega`), or give "
                        "the system a `drive_frequency` / `forcing_period` attribute."
                    ),
                ) from err
        return StroboscopicMap(self, period, **kwargs)

    def tangent(self, k: int | None = None, **kwargs: Any) -> Any:
        """Build a :class:`~tsdynamics.derived.TangentSystem` (state plus ``k`` deviation vectors).

        Equivalent to ``TangentSystem(self, k, ...)`` — the Lyapunov engine.
        """
        from tsdynamics.derived import TangentSystem

        return TangentSystem(self, k, **kwargs)

    def project(self, *components: Any, **kwargs: Any) -> Any:
        """Build a :class:`~tsdynamics.derived.ProjectedSystem` onto ``components``.

        Accepts component indices or names (resolved against ``variables``), as
        positional arguments (``self.project("x", "z")``) or a single sequence
        (``self.project(["x", "z"])``).  Equivalent to
        ``ProjectedSystem(self, components)``.
        """
        from tsdynamics.derived import ProjectedSystem

        if len(components) == 1 and not isinstance(components[0], (str, bytes)):
            first = components[0]
            try:
                comps = list(first)
            except TypeError:
                comps = [first]
        else:
            comps = list(components)
        return ProjectedSystem(self, comps, **kwargs)

    def ensemble(self, states: Any) -> Any:
        """Build an :class:`~tsdynamics.derived.EnsembleSystem` over ``states``.

        Equivalent to ``EnsembleSystem(self, states)`` — many copies stepped in
        lockstep.
        """
        from tsdynamics.derived import EnsembleSystem

        return EnsembleSystem(self, states)


def _reserved_init_keywords() -> frozenset[str]:
    """Return the named (non-``**``) keyword arguments of ``SystemBase.__init__``."""
    import inspect

    return frozenset(
        name
        for name, p in inspect.signature(SystemBase.__init__).parameters.items()
        if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
    )


_RESERVED_INIT_KEYWORDS = _reserved_init_keywords()


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
