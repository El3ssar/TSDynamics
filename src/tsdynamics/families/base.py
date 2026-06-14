"""Core abstractions: ParamSet, Trajectory, SystemBase."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, ClassVar

import numpy as np

# ---------------------------------------------------------------------------
# ParamSet
# ---------------------------------------------------------------------------


class ParamSet(MutableMapping):
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

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # --- helpers ---

    def as_tuple(self) -> tuple:
        """Return parameter values as a tuple (insertion order)."""
        return tuple(self._data.values())

    def as_dict(self) -> dict:
        """Return a shallow copy as a plain dict."""
        return dict(self._data)

    def param_hash(self) -> int:
        """
        Return a process-stable 64-bit integer hash of the current parameter values.

        Uses MD5 over a JSON-serialised representation so the result is
        reproducible across Python process restarts (unlike ``hash()``).

        The hash backs cache keys for compiled JiTCODE / JiTCDDE modules
        and the Numba-iterate cache.  At 64 bits the birthday-paradox
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


class MetaStore(MutableMapping):
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
        self._records: dict[str, list[dict]] = {}

    def record(self, key: str, value: Any, **context: Any) -> Any:
        """Append ``value`` under ``key`` with optional context kwargs."""
        import time

        self._records.setdefault(key, []).append(
            {"value": value, "context": context, "timestamp": time.time()}
        )
        return value

    def history(self, key: str) -> list[dict]:
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

    def __iter__(self):
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
        keys = {k: len(v) for k, v in self._records.items()}
        return f"MetaStore({keys})"


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class Trajectory:
    """
    The result of integrating or iterating a dynamical system.

    Supports tuple-unpacking for backward compatibility::

        t, y = system.integrate(final_time=100)

    Attributes
    ----------
    t : ndarray, shape (T,)
        Time points (or step indices for discrete maps).
    y : ndarray, shape (T, dim)
        State at each time point.
    system : SystemBase
        Back-reference to the system that produced this trajectory.
    meta : dict
        Provenance: system name, params snapshot, solver, tolerances, ic.

    Examples
    --------
    >>> traj = lor.integrate(final_time=100)
    >>> traj.dim
    3
    >>> traj["x"]            # named component (via the class's ``variables``)
    array([...])
    >>> traj.after(20.0)     # drop transient
    Trajectory(n_steps=..., dim=3, t=[20.0, 100.0])
    >>> t, y = traj          # tuple-unpack still works
    """

    __slots__ = ("t", "y", "system", "meta", "_kdtree")

    def __init__(
        self,
        t: np.ndarray,
        y: np.ndarray,
        system: Any,
        meta: dict | None = None,
    ) -> None:
        self.t = np.asarray(t)
        self.y = np.asarray(y)
        self.system = system
        self.meta = dict(meta) if meta else {}
        self._kdtree = None

    # --- compatibility / convenience ---

    def __iter__(self):
        """Allow ``t, y = trajectory``."""
        return iter((self.t, self.y))

    def __getitem__(self, key):
        """
        Component access by name, or joint row slicing.

        - ``traj["x"]`` → 1-D component array (requires the system class to
          declare ``variables``).
        - ``traj[["x", "z"]]`` → ``(T, 2)`` array.
        - anything else (int/slice/mask) slices ``t`` and ``y`` together and
          returns a new :class:`Trajectory`.
        """
        if isinstance(key, str):
            return self.y[:, self._component_index(key)]
        if isinstance(key, list | tuple) and key and all(isinstance(k, str) for k in key):
            return self.y[:, [self._component_index(k) for k in key]]
        if isinstance(key, int | np.integer):
            # Keep the result a well-formed Trajectory (one row), not a
            # corrupted one built from scalars.
            return Trajectory(
                np.atleast_1d(self.t[key]),
                np.atleast_2d(self.y[key]),
                self.system,
                meta=self.meta,
            )
        return Trajectory(self.t[key], self.y[key], self.system, meta=self.meta)

    @property
    def variables(self) -> tuple[str, ...] | None:
        """Component names declared by the system (instance attr or class ClassVar)."""
        if self.system is None:
            return None
        # Instance lookup falls back to the ClassVar for the built-in families,
        # and also honours per-instance names (e.g. WrappedSystem).
        return getattr(self.system, "variables", None)

    def _component_index(self, name: str) -> int:
        names = self.variables
        if names is None:
            raise KeyError(
                f"{type(self.system).__name__ if self.system else 'This system'} declares no "
                f"`variables`; use integer indexing or add e.g. variables = ('x', 'y', 'z') "
                f"to the system class."
            )
        try:
            return names.index(name)
        except ValueError:
            raise KeyError(f"Unknown component {name!r}. Declared variables: {names}") from None

    @property
    def dim(self) -> int:
        """State-space dimension."""
        return self.y.shape[1]

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return len(self.t)

    def component(self, i: int | str) -> np.ndarray:
        """
        Return a single state component.

        Parameters
        ----------
        i : int or str
            Component index, or component name when the system declares
            ``variables``.

        Returns
        -------
        ndarray, shape (T,)
        """
        if isinstance(i, str):
            i = self._component_index(i)
        return self.y[:, i]

    def after(self, t0: float) -> Trajectory:
        """
        Drop the initial transient.

        Parameters
        ----------
        t0 : float
            Keep only time points ``t >= t0``.

        Returns
        -------
        Trajectory
        """
        mask = self.t >= t0
        return Trajectory(self.t[mask], self.y[mask], self.system, meta=self.meta)

    # --- point-set operations ---

    def minmax(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-component ``(minima, maxima)``, each of shape ``(dim,)``."""
        return self.y.min(axis=0), self.y.max(axis=0)

    def standardize(self) -> Trajectory:
        """
        Return a copy with zero mean and unit standard deviation per component.

        The applied transform is recorded in ``meta["standardized"]``.
        """
        mean = self.y.mean(axis=0)
        std = self.y.std(axis=0)
        std = np.where(std < np.finfo(float).tiny, 1.0, std)
        return Trajectory(
            self.t,
            (self.y - mean) / std,
            self.system,
            meta={**self.meta, "standardized": {"mean": mean, "std": std}},
        )

    def neighbors(self, q: Any, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Nearest trajectory points to query point(s) ``q``.

        Builds a KD-tree lazily on first call and caches it; subsequent
        queries are O(log T).

        Parameters
        ----------
        q : array-like, shape (dim,) or (m, dim)
            Query point(s).
        k : int
            Number of neighbours per query point.

        Returns
        -------
        (distances, indices)
            As returned by :meth:`scipy.spatial.cKDTree.query`.
        """
        from scipy.spatial import cKDTree

        if self._kdtree is None:
            self._kdtree = cKDTree(self.y)
        return self._kdtree.query(np.asarray(q, dtype=float), k=k)

    def set_distance(self, other: Any, *, method: str = "centroid") -> float:
        """
        Distance to another point set (Trajectory or array), as a set.

        ``method`` is ``"centroid"`` (default), ``"hausdorff"``, or
        ``"minimum"`` — see :func:`tsdynamics.data.set_distance`.  The
        matching primitive behind attractor deduplication and continuation.
        """
        from tsdynamics.data import set_distance

        return set_distance(self, other, method=method)

    def __repr__(self) -> str:
        return (
            f"Trajectory(n_steps={self.n_steps}, dim={self.dim}, "
            f"t=[{self.t[0]:.3g}, {self.t[-1]:.3g}])"
        )


# ---------------------------------------------------------------------------
# SystemBase
# ---------------------------------------------------------------------------


class SystemBase:
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
    params: ClassVar[dict] = {}

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
    known_lyapunov: ClassVar[dict | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # The framework bases (ContinuousSystem, DelaySystem, DiscreteMap, ...)
        # live under tsdynamics.families and are not registrable systems themselves.
        if not cls.__module__.startswith("tsdynamics.families"):
            from tsdynamics.registry import register_class

            register_class(cls)

    def __init__(
        self,
        params: dict | None = None,
        ic: Any | None = None,
        dim: int | None = None,
    ) -> None:
        # Build ParamSet from class defaults + constructor overrides
        defaults = dict(type(self).params)
        if params:
            unknown = set(params) - set(defaults)
            if unknown:
                raise ValueError(
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
            pass
        object.__setattr__(self, name, value)

    # --- cloning ---

    def copy(self) -> SystemBase:
        """
        Return a deep copy with the same class, params, and ic.

        The copy has its own independent ``params`` and ``meta`` stores.
        """
        return type(self)(
            params=self.params.as_dict(),
            ic=self.ic.copy() if self.ic is not None else None,
        )

    def with_params(self, **overrides) -> SystemBase:
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
        new_p = {**self.params.as_dict(), **overrides}
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
            arr = np.random.rand(self.dim)
        object.__setattr__(self, "ic", arr.copy())
        return arr

    # --- misc ---

    def _provenance(self, **extra: Any) -> dict:
        """Build the provenance dict attached to trajectories as ``traj.meta``."""
        from tsdynamics import __version__

        return {
            "system": type(self).__name__,
            "params": self.params.as_dict(),
            "tsdynamics": __version__,
            **extra,
        }

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{type(self).__name__}({params_str})"
