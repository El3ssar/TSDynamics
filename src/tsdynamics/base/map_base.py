"""DiscreteMap — base class for discrete dynamical maps."""

from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np

from .base import SystemBase, Trajectory

try:
    from numba import njit as _njit

    _NUMBA = True
except ImportError:

    def _njit(fn):
        return fn

    _NUMBA = False


# ---------------------------------------------------------------------------
# Signature validation helpers
# ---------------------------------------------------------------------------


def _unwrap_static(obj: Any) -> Any:
    """Peel ``staticmethod`` and Numba-dispatcher wrappers off a map method."""
    fn = getattr(obj, "__func__", obj)  # staticmethod → wrapped callable
    return getattr(fn, "py_func", fn)  # numba CPUDispatcher → python function


def _positional_param_names(fn: Any) -> list[str] | None:
    """
    Return the parameter names after the state argument, or None if unknowable.

    ``None`` is returned for catch-all signatures like ``(X, *params)`` (the
    abstract methods on :class:`DiscreteMap`) and for callables that
    ``inspect.signature`` cannot introspect.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    names: list[str] = []
    for p in list(sig.parameters.values())[1:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return None
        names.append(p.name)
    return names


# ---------------------------------------------------------------------------
# DiscreteMap
# ---------------------------------------------------------------------------


class DiscreteMap(SystemBase):
    """
    Base class for discrete maps with Numba-accelerated iteration.

    Subclass contract
    -----------------
    1. Declare ``params = {...}`` and ``dim = N``.
    2. Implement ``_step`` and ``_jacobian`` as ``@staticjit`` static methods.
       Parameters arrive as **positional arguments** in the order they appear
       in the class-level ``params`` dict.

    Compilation
    -----------
    On the first call to ``iterate``, a Numba-compiled loop is built that
    inlines ``_step`` with the current parameter values baked in.  This
    eliminates per-step Python overhead.  The compiled loop is cached in
    ``_iter_cache`` per ``(class, params_hash)``; changing a parameter
    triggers a fresh compile on the next call.

    Lyapunov spectrum
    -----------------
    Computed in a single forward pass via QR decomposition of the Jacobian
    product — no redundant second iteration over the trajectory.

    Examples
    --------
    >>> h = Henon()
    >>> traj = h.iterate(steps=10_000)
    >>> t_idx, X = traj          # tuple-unpack
    >>> exps = h.lyapunov_spectrum(steps=5_000)
    >>> h_variant = h.with_params(a=1.2)
    >>> traj2 = h_variant.iterate(steps=10_000)
    """

    # Class-level cache: (class_name, params_hash) → compiled iterate fn
    _iter_cache: ClassVar[dict[tuple, Any]] = {}

    #: Set to False on maps whose orbit visits discontinuities, where the
    #: finite-difference Jacobian validation in the test suite cannot apply.
    _jacobian_fd_check: ClassVar[bool] = True

    # Protocol stepping state (instances shadow these class defaults).
    _state_now: np.ndarray | None = None
    _n_now: int = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Validate the subclass contract at class-definition time.

        ``_step`` / ``_jacobian`` receive parameters *positionally* in the
        insertion order of the class-level ``params`` dict.  A mismatch
        between the method signature and the dict order silently swaps
        parameter values (this bit the Circle map once), so it is promoted
        to an import-time ``TypeError``.
        """
        # Validate BEFORE super().__init_subclass__ so a failing class is
        # never registered in the system registry.
        declared = list(getattr(cls, "params", {}))
        for name in ("_step", "_jacobian"):
            method = getattr(cls, name, None)
            if method is None:
                continue
            sig_names = _positional_param_names(_unwrap_static(method))
            if sig_names is None:  # catch-all (X, *params) or non-introspectable
                continue
            if sig_names != declared:
                raise TypeError(
                    f"{cls.__name__}.{name} takes parameters {sig_names} but the "
                    f"params dict declares {declared} — names and ORDER must match, "
                    f"because parameters are passed positionally."
                )
        super().__init_subclass__(**kwargs)

    # ------------------------------------------------------------------ #
    # Subclass interface
    # ------------------------------------------------------------------ #

    @staticmethod
    @abstractmethod
    def _step(X: np.ndarray, *params) -> Any:
        """
        Evaluate the map at state ``X``.

        Decorate with ``@staticjit``.  Parameters arrive positionally in
        the order they appear in the class-level ``params`` dict.

        Parameters
        ----------
        X : ndarray, shape (dim,)
            Current state.
        *params
            Parameter values in declaration order.

        Returns
        -------
        array-like of shape (dim,).
        """
        ...

    @staticmethod
    @abstractmethod
    def _jacobian(X: np.ndarray, *params) -> Any:
        """
        Return the (dim × dim) Jacobian at state ``X``.

        Decorate with ``@staticjit``.  Parameters positional, same order as
        the class-level ``params`` dict.

        Returns
        -------
        array-like of shape (dim, dim).
        """
        ...

    # ------------------------------------------------------------------ #
    # System protocol — incremental stepping
    # ------------------------------------------------------------------ #

    @property
    def is_discrete(self) -> bool:
        """Maps are discrete-time systems."""
        return True

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict | None = None,
    ) -> None:
        """(Re)start stepping from state ``u`` at iteration count ``t``."""
        if params:
            for k, v in params.items():
                self.params[k] = v
        self._state_now = self.resolve_ic(u)
        self._n_now = int(t) if t is not None else 0

    def step(self, n_or_dt: int | None = None) -> np.ndarray:
        """Advance ``n`` iterations (default 1) and return the new state."""
        if self._state_now is None:
            self.reinit()
        if n_or_dt is None:
            n = 1
        else:
            nf = float(n_or_dt)
            if not nf.is_integer() or nf < 1:
                raise ValueError(
                    f"{type(self).__name__}.step takes a positive whole number of "
                    f"iterations, got {n_or_dt!r} (fractional time steps have no "
                    f"meaning for discrete maps)."
                )
            n = int(nf)
        x = self._state_now
        params = self.params.as_tuple()

        iterate_fn = self._get_iterate_fn() if n > 16 else None
        if iterate_fn is not None:
            x = iterate_fn(x.copy(), n)[-1].copy()
        else:
            step_fn = type(self)._step
            for _ in range(n):
                x = np.asarray(step_fn(x, *params), dtype=float).ravel()
        if not np.isfinite(x).all():
            raise RuntimeError(
                f"{type(self).__name__}: map diverged at iteration {self._n_now + n}."
            )
        self._state_now = np.asarray(x, dtype=float).reshape(self.dim)
        self._n_now += n
        return self._state_now.copy()

    def state(self) -> np.ndarray:
        """Return a copy of the current state (implicit ``reinit`` if cold)."""
        if self._state_now is None:
            self.reinit()
        return self._state_now.copy()

    def set_state(self, u: Any) -> None:
        """Overwrite the current state."""
        self._state_now = np.asarray(u, dtype=float).reshape(self.dim)

    def time(self) -> float:
        """Return the current iteration count."""
        return float(self._n_now)

    def trajectory(
        self,
        steps: int = 1000,
        *,
        transient: int = 0,
        **kwargs,
    ) -> Trajectory:
        """Protocol-uniform trajectory: ``iterate`` plus optional transient drop."""
        traj = self.iterate(steps=transient + steps, **kwargs)
        return traj[transient:] if transient > 0 else traj

    # ------------------------------------------------------------------ #
    # Compiled iterate loop
    # ------------------------------------------------------------------ #

    def _get_iterate_fn(self):
        """
        Return a Numba-compiled iterate function for the current params.

        The function signature is ``iterate(ic, steps) → (steps, dim) array``.

        When Numba is unavailable, returns ``None`` and the caller falls back
        to a pure-Python loop.  The compiled function is cached per
        ``(class_name, params_hash)`` — a param change triggers one re-JIT.
        """
        if not _NUMBA:
            return None

        # Key on the class OBJECT, not its name: a same-named user class (or a
        # notebook redefinition with edited _step) must never reuse another
        # definition's compiled loop.
        cache_key = (type(self), self.params.param_hash())
        cache = type(self)._iter_cache

        if cache_key in cache:
            return cache[cache_key]

        step_fn = type(self)._step  # @njit-compiled static method
        params_tuple = self.params.as_tuple()
        dim = self.dim

        @_njit
        def _iterate(ic: np.ndarray, steps: int) -> np.ndarray:
            out = np.empty((steps, dim))
            x = ic.copy()
            for i in range(steps):
                nxt = step_fn(x, *params_tuple)
                for j in range(dim):
                    out[i, j] = nxt[j]
                    x[j] = nxt[j]
            return out

        cache[cache_key] = _iterate
        return _iterate

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #

    def iterate(
        self,
        steps: int = 1000,
        ic: Any | None = None,
        max_retries: int = 10,
    ) -> Trajectory:
        """
        Iterate the map for ``steps`` steps.

        Uses a Numba-compiled loop when available (significant speedup for
        large ``steps`` or high-dim maps).  Falls back to a Python loop
        when Numba is not installed.

        Parameters
        ----------
        steps : int
            Number of iterations. Default 1000.
        ic : array-like, optional
            Initial state. Falls back to ``self.ic``, then random.
        max_retries : int
            Retry with a new random IC if divergence is detected.

        Returns
        -------
        Trajectory
            ``t`` is ``arange(steps)`` (integer step indices, not float times).
        """
        ic_arr = self.resolve_ic(ic)
        iterate_fn = self._get_iterate_fn()

        for attempt in range(max_retries):
            try:
                if iterate_fn is not None:
                    out = iterate_fn(ic_arr.copy(), steps)
                else:
                    out = self._iterate_python(ic_arr.copy(), steps)

                if not np.all(np.isfinite(out)):
                    bad = np.argmax(~np.isfinite(out).all(axis=1))
                    raise ValueError(f"Divergence detected at step {bad}")

                return Trajectory(
                    t=np.arange(steps),
                    y=out,
                    system=self,
                    meta=self._provenance(family="map", steps=steps, ic=ic_arr.copy()),
                )

            except ValueError as exc:
                if attempt == max_retries - 1:
                    raise
                print(f"Warning: {exc}. Retrying with a new random IC.")
                ic_arr = np.random.rand(self.dim)
                object.__setattr__(self, "ic", ic_arr.copy())

    def _iterate_python(self, ic: np.ndarray, steps: int) -> np.ndarray:
        """Pure-Python fallback (used when Numba is unavailable)."""
        params = self.params.as_tuple()
        out = np.empty((steps, self.dim))
        x = ic
        for i in range(steps):
            x = np.asarray(type(self)._step(x, *params), dtype=float)
            out[i] = x
        return out

    # ------------------------------------------------------------------ #
    # Lyapunov spectrum
    # ------------------------------------------------------------------ #

    def lyapunov_spectrum(
        self,
        steps: int = 5000,
        ic: Any | None = None,
        n_exp: int | None = None,
        reortho_interval: int = 1,
    ) -> np.ndarray:
        """
        QR-based Lyapunov spectrum.

        Computes the spectrum in a **single forward pass** — the Jacobian is
        evaluated at each step alongside the trajectory, avoiding the
        redundant double iteration of the previous implementation.

        Results are stored in ``self.meta['lyapunov_spectrum']``.

        Parameters
        ----------
        steps : int
            Number of iterations. Default 5000.
        ic : array-like, optional
            Initial state. Falls back to ``self.ic``, then random.
        n_exp : int, optional
            Number of exponents. Defaults to ``dim``.
        reortho_interval : int
            Reorthonormalise every this many steps. Default 1.

        Returns
        -------
        ndarray, shape (n_exp,)
        """
        n_exp = n_exp or self.dim
        params = self.params.as_tuple()
        step = type(self)._step
        jac = type(self)._jacobian
        max_retries = 10

        # The QR loop can encounter overflow on transient huge iterates before
        # the divergence check triggers; silence the float warnings locally so
        # they don't elevate to errors under strict filterwarnings policies.
        # We track non-finiteness explicitly and treat it as a soft failure.
        for attempt in range(max_retries):
            x = self.resolve_ic(ic if attempt == 0 else None)
            Q = np.eye(self.dim)[:n_exp]
            lyap_sums = np.zeros(n_exp)
            intervals = 0
            failed = False

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                for i in range(steps):
                    x = np.asarray(step(x, *params), dtype=float)
                    if not np.all(np.isfinite(x)):
                        failed = True
                        break

                    J = np.atleast_2d(np.asarray(jac(x, *params), dtype=float))
                    if not np.all(np.isfinite(J)):
                        failed = True
                        break

                    Q = (J @ Q.T).T
                    if not np.all(np.isfinite(Q)):
                        failed = True
                        break

                    if (i + 1) % reortho_interval == 0:
                        Q_new, R = np.linalg.qr(Q.T)
                        if not np.all(np.isfinite(Q_new)):
                            failed = True
                            break
                        Q = Q_new.T
                        diag = np.abs(np.diag(R))
                        diag = np.where(
                            diag < np.finfo(float).tiny,
                            np.finfo(float).tiny,
                            diag,
                        )
                        lyap_sums += np.log(diag)
                        intervals += 1

            if not failed and intervals > 0:
                exponents = lyap_sums / (intervals * reortho_interval)
                self.meta.record(
                    "lyapunov_spectrum",
                    exponents,
                    steps=steps,
                    n_exp=n_exp,
                    reortho_interval=reortho_interval,
                )
                return exponents

            if attempt < max_retries - 1:
                # Clear stored IC so resolve_ic generates a fresh random one.
                object.__setattr__(self, "ic", None)

        raise ValueError(
            f"{type(self).__name__}.lyapunov_spectrum: failed after "
            f"{max_retries} retries — iterates diverge from every tried IC. "
            f"Try a larger `steps` budget or pass an `ic` from a known basin point."
        )
