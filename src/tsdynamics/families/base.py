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
        raise TypeError("Parameters are fixed-key — cannot delete.")

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
        """Append ``value`` under ``key`` with optional context kwargs."""
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
    Individual instances can override params and/or ic::

        lor = Lorenz(params={"rho": 30.0}, ic=[1.0, 0.0, 0.0])

    Constructor will raise ``ValueError`` for unknown param keys.
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
    #: does not name one.  Every concrete family sets it to ``"interp"`` (the Rust
    #: engine interpreter — the sole integration backend post-M3); the abstract
    #: base keeps ``"reference"`` (the wheel-free pure-Python oracle).  Read by the
    #: family ``integrate`` / ``iterate`` methods and by :meth:`_dispatch`, so "the
    #: default backend" lives in exactly one place.
    _default_backend: ClassVar[str] = "reference"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
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
    ) -> None:
        # Build ParamSet from class defaults + constructor overrides
        defaults = dict(type(self).params)
        if params:
            unknown = set(params) - set(defaults)
            if unknown:
                from tsdynamics.errors import InvalidParameterError

                raise InvalidParameterError(
                    f"{type(self).__name__}: unknown parameter(s) "
                    f"{sorted(unknown)}. Declared: {sorted(defaults)}"
                )
            defaults.update(params)
        object.__setattr__(self, "params", ParamSet(defaults))

        # dim: constructor arg > class attribute
        resolved_dim = dim if dim is not None else type(self).dim
        object.__setattr__(self, "dim", resolved_dim)

        # Initial conditions
        ic_arr = np.asarray(ic, dtype=float) if ic is not None else None
        object.__setattr__(self, "ic", ic_arr)

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

    # --- cloning ---

    def copy(self) -> SystemBase:
        """
        Return a deep copy with the same class, params, and ic.

        The copy has its own independent ``params`` and ``meta`` stores.
        """
        return type(self)(
            params=cast(ParamSet, self.params).as_dict(),
            ic=self.ic.copy() if self.ic is not None else None,
        )

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

    def resolve_ic(self, ic: Any | None = None) -> np.ndarray:
        """
        Resolve initial conditions consistently.

        Priority:

        1. ``ic`` argument (if provided)
        2. ``self.ic`` (set by a previous integration / iteration)
        3. ``type(self).default_ic`` (class-level default, if declared)
        4. Random ``U[0, 1)^dim``

        The resolved IC is stored in ``self.ic`` so subsequent calls without
        an explicit ``ic`` reproduce the same initial state.

        Parameters
        ----------
        ic : array-like or None

        Returns
        -------
        ndarray, shape (dim,)
        """
        if ic is not None:
            arr = np.asarray(ic, dtype=float).reshape(self.dim)
        elif self.ic is not None:
            arr = np.asarray(self.ic, dtype=float).reshape(self.dim)
        elif type(self).default_ic is not None:
            arr = np.asarray(type(self).default_ic, dtype=float).reshape(self.dim)
        else:
            arr = np.random.rand(cast(int, self.dim))
        object.__setattr__(self, "ic", arr.copy())
        return arr

    # --- engine-dispatch seam ---

    def _dispatch(self, *, backend: str, **kwargs: Any) -> Trajectory:
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
        """
        from tsdynamics.engine import run

        return run.integrate(self, backend=backend, **kwargs)

    # --- misc ---

    def _provenance(self, **extra: Any) -> dict[str, Any]:
        """Build the provenance dict attached to trajectories as ``traj.meta``."""
        from tsdynamics import __version__

        return {
            "system": type(self).__name__,
            "params": cast(ParamSet, self.params).as_dict(),
            "tsdynamics": __version__,
            **extra,
        }

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

    @property
    def entropy(self) -> Any:
        """Entropy / complexity estimators bound to this system.

        A cached :class:`~tsdynamics.families._accessors.EntropyAccessor`
        (``.permutation()`` / ``.sample()`` / …) delegating to the entropy free
        functions.  These consume a scalar series; omitting ``data`` runs the
        system first.
        """
        from tsdynamics.families._accessors import EntropyAccessor

        return self._topical_accessor("entropy", EntropyAccessor)

    @property
    def surrogate(self) -> Any:
        """Surrogate generators + nonlinearity tests bound to this system.

        A cached :class:`~tsdynamics.families._accessors.SurrogateAccessor`
        (``.test()`` / ``.generate()`` / …) delegating to
        :func:`tsdynamics.analysis.surrogate_test`,
        :func:`~tsdynamics.analysis.surrogates` and the surrogate statistics.
        """
        from tsdynamics.families._accessors import SurrogateAccessor

        return self._topical_accessor("surrogate", SurrogateAccessor)

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
            if section is None:
                raise ValueError(
                    "poincare() needs either `section=` (with `at=`) or an explicit `plane=`."
                )
            comp = section
            if isinstance(comp, str):
                names = getattr(type(self), "variables", None)
                if names is None:
                    raise ValueError(
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


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
