"""DiscreteMap — base class for discrete dynamical maps."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np

from tsdynamics._native import iterate_map as _iterate_map
from tsdynamics._native import lyapunov_spectrum_map as _lyapunov_spectrum_map

from ._ir import CompiledMap, NotLowerableError
from ._lowering import lower_to_ir
from .base import SystemBase, Trajectory

try:
    from numba import njit as _njit

    _NUMBA = True
except ImportError:

    def _njit(fn):
        return fn

    _NUMBA = False


# ---------------------------------------------------------------------------
# DiscreteMap
# ---------------------------------------------------------------------------


class DiscreteMap(SystemBase):
    """
    Base class for discrete maps.

    Subclass contract
    -----------------
    1. Declare ``params = {...}`` and ``dim = N``.
    2. Implement ``_step`` and ``_jacobian`` as ``@staticjit`` static methods.
       Parameters arrive as **positional arguments** in the order they appear
       in the class-level ``params`` dict.
    3. Use NumPy operations (``np.cos``, ``np.sin``, ``np.where``, …) rather
       than Python ``if`` / ``and`` on state or parameter values — those
       can't be traced into the IR and force the slower fallback path.

    Compilation
    -----------
    First call: ``_step`` / ``_jacobian`` are traced with placeholder Tracer
    inputs, the resulting expressions are lowered to an IR bytecode, and
    every subsequent ``iterate`` / ``lyapunov_spectrum`` runs in Rust. If
    a map uses an op the IR doesn't yet support, lowering raises
    :class:`NotLowerableError` and the call falls back to the Numba-compiled
    path.

    Lyapunov spectrum
    -----------------
    Single forward pass with QR reorthonormalisation of the variational
    bundle — same algorithm as before, now in Rust under the hood.

    Examples
    --------
    >>> h = Henon()
    >>> traj = h.iterate(steps=10_000)
    >>> exps = h.lyapunov_spectrum(steps=5_000)        # ≈ [0.42, -1.62]
    """

    # Class-level caches:
    #   * _iter_cache: Numba-compiled iterate function (fallback path).
    #   * _ir_cache:   CompiledMap bytecode (None sentinel = "not lowerable").
    _iter_cache: ClassVar[dict[tuple, Any]] = {}
    _ir_cache: ClassVar[dict[tuple, CompiledMap | None]] = {}

    # ------------------------------------------------------------------ #
    # Subclass interface
    # ------------------------------------------------------------------ #

    @staticmethod
    @abstractmethod
    def _step(X: np.ndarray, *params) -> Any:
        """Evaluate the map at state ``X``. Decorate with ``@staticjit``."""

    @staticmethod
    @abstractmethod
    def _jacobian(X: np.ndarray, *params) -> Any:
        """Return the (dim × dim) Jacobian at state ``X``. Decorate with ``@staticjit``."""

    # ------------------------------------------------------------------ #
    # IR compilation (Rust path)
    # ------------------------------------------------------------------ #

    def _compile_ir(self) -> CompiledMap | None:
        """Trace + lower the map to an IR bytecode payload.

        Returns ``None`` if the map can't be lowered (caller falls back to
        the Numba path). Cached per ``(class, params_hash)`` since changing
        parameter values doesn't change the IR structure (params are bound
        per-call in Rust), but a future structural change might.
        """
        cache_key = (type(self).__name__, self.params.param_hash())
        cache = type(self)._ir_cache
        if cache_key in cache:
            return cache[cache_key]
        try:
            compiled = lower_to_ir(type(self), self.params.as_tuple(), self.dim)
        except NotLowerableError:
            cache[cache_key] = None
            return None
        cache[cache_key] = compiled
        return compiled

    # ------------------------------------------------------------------ #
    # Numba fallback (used when IR lowering fails)
    # ------------------------------------------------------------------ #

    def _get_numba_iterate_fn(self):
        """Return a Numba-compiled iterate function. Used only when IR fails."""
        if not _NUMBA:
            return None

        cache_key = (type(self).__name__, self.params.param_hash())
        cache = type(self)._iter_cache
        if cache_key in cache:
            return cache[cache_key]

        step_fn = type(self)._step
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

        Tries the Rust IR-interpreted path first; falls back to the
        Numba-compiled loop if the system can't be lowered. Falls back to
        a pure-Python loop if neither IR nor Numba is available.

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
            ``t`` is ``arange(steps)`` (integer step indices).
        """
        ic_arr = self.resolve_ic(ic)
        compiled = self._compile_ir()
        numba_fn = None if compiled is not None else self._get_numba_iterate_fn()
        params_arr = np.asarray(self.params.as_tuple(), dtype=float)

        for attempt in range(max_retries):
            try:
                if compiled is not None:
                    out = _iterate_map(
                        compiled.bytecode,
                        ic_arr.astype(float, copy=True),
                        params_arr,
                        steps,
                    )
                elif numba_fn is not None:
                    out = numba_fn(ic_arr.copy(), steps)
                else:
                    out = self._iterate_python(ic_arr.copy(), steps)

                if not np.all(np.isfinite(out)):
                    bad = np.argmax(~np.isfinite(out).all(axis=1))
                    raise ValueError(f"Divergence detected at step {bad}")

                return Trajectory(t=np.arange(steps), y=out, system=self)

            except ValueError as exc:
                if attempt == max_retries - 1:
                    raise
                print(f"Warning: {exc}. Retrying with a new random IC.")
                ic_arr = np.random.rand(self.dim)
                object.__setattr__(self, "ic", ic_arr.copy())

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

        Runs in Rust when the map lowers to IR; falls back to the
        existing Python QR loop calling Numba-compiled ``_step`` /
        ``_jacobian`` when it doesn't.

        Results are stored in ``self.meta['lyapunov_spectrum']``.

        Parameters
        ----------
        steps : int
            Number of iterations. Default 5000.
        ic : array-like, optional
            Initial state.
        n_exp : int, optional
            Number of exponents. Defaults to ``dim``.
        reortho_interval : int
            Reorthonormalise every this many steps. Default 1.

        Returns
        -------
        ndarray, shape (n_exp,)
        """
        n_exp = n_exp or self.dim
        compiled = self._compile_ir()

        if compiled is not None:
            return self._lyapunov_rust(compiled, steps, ic, n_exp, reortho_interval)
        return self._lyapunov_python(steps, ic, n_exp, reortho_interval)

    # -- Rust path ---------------------------------------------------------

    def _lyapunov_rust(
        self,
        compiled: CompiledMap,
        steps: int,
        ic: Any | None,
        n_exp: int,
        reortho_interval: int,
    ) -> np.ndarray:
        max_retries = 10
        params_arr = np.asarray(self.params.as_tuple(), dtype=float)
        for attempt in range(max_retries):
            ic_arr = self.resolve_ic(ic if attempt == 0 else None).astype(float, copy=True)
            try:
                exps = _lyapunov_spectrum_map(
                    compiled.bytecode,
                    ic_arr,
                    params_arr,
                    int(steps),
                    int(n_exp),
                    int(reortho_interval),
                )
            except ValueError as exc:
                if "divergence" not in str(exc).lower():
                    raise
                if attempt < max_retries - 1:
                    object.__setattr__(self, "ic", None)
                    continue
                raise
            self.meta["lyapunov_spectrum"] = exps
            return exps

        raise ValueError(
            f"{type(self).__name__}.lyapunov_spectrum: failed after "
            f"{max_retries} retries — iterates diverge from every tried IC. "
            f"Try a larger `steps` budget or pass an `ic` from a known basin point."
        )

    # -- Python fallback (for maps that can't lower) -----------------------

    def _lyapunov_python(
        self,
        steps: int,
        ic: Any | None,
        n_exp: int,
        reortho_interval: int,
    ) -> np.ndarray:
        params = self.params.as_tuple()
        step = type(self)._step
        jac = type(self)._jacobian
        max_retries = 10

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
                self.meta["lyapunov_spectrum"] = exponents
                return exponents

            if attempt < max_retries - 1:
                object.__setattr__(self, "ic", None)

        raise ValueError(
            f"{type(self).__name__}.lyapunov_spectrum: failed after "
            f"{max_retries} retries — iterates diverge from every tried IC. "
            f"Try a larger `steps` budget or pass an `ic` from a known basin point."
        )
