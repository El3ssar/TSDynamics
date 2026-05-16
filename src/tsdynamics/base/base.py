"""Core abstractions: ParamSet, Trajectory, SystemBase."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, ClassVar

import numpy as np

from ..analysis._registry import install_methods

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
        Free-form per-result metadata.  Analysis ops that need to surface
        extra information about the result (e.g. ``detect_events`` may
        stash per-event direction / source-index arrays here) write to
        this dict.  Empty by default.

    Examples
    --------
    >>> traj = lor.integrate(final_time=100)
    >>> traj.dim
    3
    >>> traj.after(20.0)    # drop transient
    Trajectory(n_steps=..., dim=3, t=[20.0, 100.0])
    >>> t, y = traj          # tuple-unpack still works
    """

    __slots__ = ("t", "y", "system", "meta")

    def __init__(
        self,
        t: np.ndarray,
        y: np.ndarray,
        system: Any = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.t = np.asarray(t)
        self.y = np.asarray(y)
        self.system = system
        self.meta = dict(meta) if meta else {}

    # --- compatibility / convenience ---

    def __iter__(self):
        """Allow ``t, y = trajectory``."""
        return iter((self.t, self.y))

    def __getitem__(self, key):
        """Slice both arrays together: ``traj[100:]`` → new Trajectory."""
        return Trajectory(self.t[key], self.y[key], self.system, meta=self.meta)

    @property
    def dim(self) -> int:
        """State-space dimension."""
        return self.y.shape[1]

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return len(self.t)

    def component(self, i: int) -> np.ndarray:
        """
        Return a single state component.

        Parameters
        ----------
        i : int
            Component index.

        Returns
        -------
        ndarray, shape (T,)
        """
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

    # ------------------------------------------------------------------
    # Analysis methods (decimate, resample, project, window, derivative,
    # norm, local_maxima, local_minima, return_times, to_dataspec,
    # detect_events, poincare_section, return_map, ...) are *not* written
    # out here.  They live as decorated free functions in
    # :mod:`tsdynamics.analysis` and are installed on this class by the
    # ``install_methods(Trajectory)`` call at the bottom of this module.
    # See :mod:`tsdynamics.analysis._registry` for the design.
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Trajectory(n_steps={self.n_steps}, dim={self.dim}, "
            f"t=[{self.t[0]:.3g}, {self.t[-1]:.3g}])"
        )


# Wire up every ``@trajectory_op``-decorated function as a Trajectory method.
# The registry is populated at import time when ``tsdynamics.analysis``
# loads; calling it here is what turns ``traj.decimate(...)``,
# ``traj.detect_events(...)``, ``traj.poincare_section(...)`` etc. into
# real methods.  See :mod:`tsdynamics.analysis._registry` for details.
install_methods(Trajectory)


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

        # Metadata store: store computed properties (Lyapunov, etc.) here
        object.__setattr__(self, "meta", {})

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

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{type(self).__name__}({params_str})"
