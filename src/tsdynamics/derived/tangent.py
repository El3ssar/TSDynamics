"""Tangent dynamics: state + deviation vectors, the one Lyapunov engine."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from tsdynamics.families import ContinuousSystem, DelaySystem, DiscreteMap

from ._base import DerivedSystem
from ._variational import build_variational_tape, embed_extended, split_extended

__all__ = ["TangentSystem"]

#: ODE backends that integrate the *extended* variational system on the engine
#: (interpreter / Cranelift JIT / pure-Python reference).
_ENGINE_BACKENDS = frozenset({"interp", "jit", "reference"})


def _qr_growths(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorthonormalise deviation vectors; return ``(Q, log|diag R|)``.

    The logarithmic stretch factors ``log|diag R|`` are the per-step Lyapunov
    contributions; their sign-independence (``|·|``) absorbs the arbitrary
    column-sign convention of ``numpy.linalg.qr``.
    """
    q, r = np.linalg.qr(w)
    diag = np.abs(np.diag(r))
    tiny = np.finfo(float).tiny
    diag = np.where(diag < tiny, tiny, diag)
    return q, np.log(diag)


class TangentSystem(DerivedSystem):
    """
    Evolve a system together with ``k`` deviation (tangent) vectors.

    Each ``step()`` advances the state and the deviation vectors, then
    QR-reorthonormalises; :meth:`growths` exposes the per-step logarithmic
    stretch factors ``log |diag R|`` and :meth:`exponents` their running
    time-average — the Lyapunov spectrum estimate.  :meth:`lyapunov_spectrum`
    wraps that into the standard burn-in + time-averaged estimate, and is the
    single implementation every family's ``lyapunov_spectrum`` delegates to.

    Implementation per family
    -------------------------
    - **Maps**: pure NumPy — ``W ← J(x)·W`` with ``_jacobian`` evaluated at the
      pre-image (the correct tangent-map convention), then QR.
    - **ODEs**: the *extended* ODE (state ⊕ ``k`` tangent vectors, see
      :mod:`tsdynamics.derived._variational`) is lowered to an engine tape and
      integrated per step on the Rust engine (or the pure-Python reference
      oracle), then QR-reorthonormalised here.  Select the variational backend
      with ``backend=``: ``"interp"`` (default), ``"jit"``, or ``"reference"``.
    - **DDEs**: not supported — tangent dynamics of a DDE lives in an
      infinite-dimensional history space; use
      ``DelaySystem.lyapunov_spectrum`` (the engine DDE Lyapunov estimator).

    Parameters
    ----------
    system : DiscreteMap or ContinuousSystem
        The base system whose tangent dynamics to evolve.
    k : int, optional
        Number of deviation vectors (``1 ≤ k ≤ system.dim``).  Defaults to the
        full state dimension.
    backend : str, optional
        ODE variational backend (ignored for maps): ``"interp"`` (default),
        ``"jit"``, or ``"reference"``.

    Examples
    --------
    >>> tang = TangentSystem(Henon(), k=2)
    >>> tang.reinit([0.1, 0.1])
    >>> for _ in range(5000):
    ...     tang.step()
    >>> tang.exponents()        # ≈ [0.42, -1.62]
    """

    _ODE_DEFAULT_DT: ClassVar[float] = 0.1

    def __init__(self, system: Any, k: int | None = None, *, backend: str | None = None) -> None:
        if isinstance(system, DelaySystem):
            raise NotImplementedError(
                "TangentSystem does not support delay systems — their tangent space is the "
                "infinite-dimensional history space. Use DelaySystem.lyapunov_spectrum."
            )
        if isinstance(system, DiscreteMap):
            self._mode = "map"
        elif isinstance(system, ContinuousSystem):
            self._mode = "ode"
        else:
            raise TypeError(
                f"TangentSystem needs a DiscreteMap or ContinuousSystem, "
                f"got {type(system).__name__}."
            )
        super().__init__(system)
        self.k = int(k) if k is not None else system.dim
        if not 1 <= self.k <= system.dim:
            raise ValueError(f"k must be in [1, {system.dim}], got {self.k}")

        if self._mode == "map":
            self._backend = "map"
        else:
            self._backend = (backend or "interp").lower()
            if self._backend not in _ENGINE_BACKENDS:
                raise ValueError(
                    f"unknown ODE tangent backend {backend!r}; choose from "
                    f"{sorted(_ENGINE_BACKENDS)} (the engine variational path)."
                )

        self._W: np.ndarray | None = None  # (dim, k) deviation vectors (map mode)
        self._ext_tape: Any = None  # extended variational tape (ode engine mode)
        self._z: np.ndarray | None = None  # extended state (ode engine mode)
        self._t = 0.0
        # ODE integration options (engine mode), captured at reinit.
        self._method: str | None = None
        self._rtol = 1e-6
        self._atol = 1e-9
        self._integrator_kwargs: dict[str, Any] = {}
        self._last_growths = np.zeros(self.k)
        self._sum_growths = np.zeros(self.k)
        self._elapsed = 0.0

    def _rebuild(self, inner: Any) -> TangentSystem:
        backend = None if self._mode == "map" else self._backend
        return TangentSystem(inner, self.k, backend=backend)

    # --- lifecycle ---

    def _reset_accumulators(self) -> None:
        """Zero the accumulated and last growth records (start a fresh average)."""
        self._last_growths = np.zeros(self.k)
        self._sum_growths = np.zeros(self.k)
        self._elapsed = 0.0

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Restart state, deviation vectors, and accumulated growth sums."""
        self._reset_accumulators()

        if self._mode == "map":
            self.system.reinit(u, t=t, params=params)
            self._W = np.eye(self.system.dim)[:, : self.k]
            return

        # ode mode — apply any parameter overrides to the inner system first.
        if params:
            for key, value in params.items():
                self.params[key] = value
        ic_arr = self.system.resolve_ic(u)
        t0 = float(t) if t is not None else 0.0
        self._t = t0
        self._method = kwargs.pop("method", None)
        self._rtol = kwargs.pop("rtol", 1e-6)
        self._atol = kwargs.pop("atol", 1e-9)
        self._integrator_kwargs = dict(kwargs)

        self._reinit_ode_engine(ic_arr)

    def _reinit_ode_engine(self, ic_arr: np.ndarray) -> None:
        """Lower the extended variational tape and seed the extended state."""
        if self._ext_tape is None:
            self._ext_tape = build_variational_tape(self.system, self.k)
        w0 = np.eye(self.system.dim)[:, : self.k]
        self._z = embed_extended(ic_arr, w0)

    # --- protocol ---

    @property
    def is_discrete(self) -> bool:
        """Match the wrapped system's time semantics."""
        return self.system.is_discrete

    def step(self, n_or_dt: float | None = None) -> np.ndarray:
        """
        Advance state + deviation vectors and reorthonormalise.

        For maps ``n_or_dt`` is the number of iterations (QR after each);
        for ODEs it is the time increment (default 0.1).
        Returns the new state.
        """
        if self._mode == "map":
            return self._step_map(int(n_or_dt) if n_or_dt is not None else 1)
        dt = float(n_or_dt) if n_or_dt is not None else self._ODE_DEFAULT_DT
        return self._step_ode_engine(dt)

    def _step_map(self, n: int) -> np.ndarray:
        if self._W is None:
            self.reinit()
        sys = self.system
        jac = type(sys)._jacobian
        params = sys.params.as_tuple()
        for _ in range(n):
            x = sys.state()
            j = np.atleast_2d(np.asarray(jac(x, *params), dtype=float))
            sys.step()
            self._W = j @ self._W
            self._W, growths = _qr_growths(self._W)
            self._last_growths = growths
            self._sum_growths += growths
            self._elapsed += 1.0
        return sys.state()

    def _step_ode_engine(self, dt: float) -> np.ndarray:
        if self._z is None:
            self.reinit()
        from tsdynamics.engine import run
        from tsdynamics.engine.problem import ODEProblem

        dim = self.system.dim
        t0 = self._t
        method = self._method or self.system._default_method
        problem = ODEProblem(tape=self._ext_tape, ic=self._z, t0=t0, system=self.system)
        traj = run.integrate(
            problem,
            final_time=t0 + dt,
            dt=dt,
            t0=t0,
            method=method,
            rtol=self._rtol,
            atol=self._atol,
            backend=self._backend,
        )
        x, w = split_extended(traj.y[-1], dim, self.k)
        q, growths = _qr_growths(w)
        self._z = embed_extended(x, q)
        self._t = t0 + dt
        self._last_growths = growths
        self._sum_growths += growths
        self._elapsed += dt
        return x

    def state(self) -> np.ndarray:
        """Return a copy of the current base-system state."""
        if self._mode == "map":
            return self.system.state()
        # ODE mode: the constructor guarantees an engine backend, so the extended
        # state ``self._z`` always carries the base state in its leading slots.
        if self._z is None:
            self.reinit()
        return self._z[: self.system.dim].copy()

    def set_state(self, u: Any) -> None:
        """Overwrite the base state (map mode only — ODE tangent vectors would desync)."""
        if self._mode == "map":
            self.system.set_state(u)
        else:
            raise NotImplementedError(
                "TangentSystem(ode).set_state is not supported — the tangent vectors "
                "would desynchronise. Use reinit(u)."
            )

    def time(self) -> float:
        """Return the current time / iteration count."""
        return self.system.time() if self._mode == "map" else self._t

    # --- Lyapunov accessors ---

    def deviations(self) -> np.ndarray:
        """Return the current orthonormal deviation vectors, shape ``(dim, k)``.

        Available on every supported backend — maps and the ODE engine
        backends both carry the deviation matrix explicitly.
        """
        if self._mode == "map":
            if self._W is None:
                raise RuntimeError("deviations() is available after reinit()")
            return self._W.copy()
        # ODE engine backend (interp/jit/reference): the constructor rejects any
        # other backend, so the deviation vectors are always carried explicitly.
        if self._z is None:
            raise RuntimeError("deviations() is available after reinit()")
        return split_extended(self._z, self.system.dim, self.k)[1]

    def growths(self) -> np.ndarray:
        """Return the log stretch factors ``log|diag R|`` from the most recent step."""
        return self._last_growths.copy()

    def exponents(self) -> np.ndarray:
        """Return the running Lyapunov-spectrum estimate (accumulated growths / elapsed)."""
        if self._elapsed == 0.0:
            return np.zeros(self.k)
        return self._sum_growths / self._elapsed

    # --- the one Lyapunov engine: burn-in + time-averaged spectrum ---

    def lyapunov_spectrum(self, **kwargs: Any) -> np.ndarray:
        """
        Estimate the Lyapunov spectrum — the unified engine for every family.

        Map and ODE families both delegate their ``lyapunov_spectrum`` here, so
        the QR/variational machinery lives in exactly one place.  Mode-specific
        keywords:

        - **maps**: ``steps`` (default 5000), ``ic``, ``reortho_interval`` (1).
        - **ODEs**: ``final_time`` (200.0), ``dt`` (0.1), ``ic``, ``burn_in``
          (50.0), ``method``, ``rtol`` (1e-6), ``atol`` (1e-9), and any extra
          integrator keywords.

        The estimate is recorded in ``self.meta['lyapunov_spectrum']`` (the inner
        system's :class:`~tsdynamics.families.base.MetaStore`).

        Returns
        -------
        ndarray, shape (k,)
            Lyapunov exponents, largest first (QR order).
        """
        if self._mode == "map":
            return self._lyapunov_spectrum_map(**kwargs)
        return self._lyapunov_spectrum_ode(**kwargs)

    def _lyapunov_spectrum_map(
        self,
        steps: int = 5000,
        ic: Any | None = None,
        reortho_interval: int = 1,
    ) -> np.ndarray:
        """Single-pass QR spectrum for a map, with random-IC retry on divergence."""
        max_retries = 10
        for attempt in range(max_retries):
            use_ic = ic if attempt == 0 else None
            if self._accumulate_map(steps, use_ic, reortho_interval):
                exponents = self.exponents()
                self.meta.record(
                    "lyapunov_spectrum",
                    exponents,
                    steps=steps,
                    n_exp=self.k,
                    reortho_interval=reortho_interval,
                )
                return exponents
            if attempt < max_retries - 1:
                # Force a fresh random IC: clear any stored one on the inner map.
                object.__setattr__(self.system, "ic", None)

        raise ValueError(
            f"{type(self.system).__name__}.lyapunov_spectrum: failed after "
            f"{max_retries} retries — iterates diverge from every tried IC. "
            f"Try a larger `steps` budget or pass an `ic` from a known basin point."
        )

    def _accumulate_map(self, steps: int, ic: Any | None, reortho_interval: int) -> bool:
        """Run the map QR loop; return ``False`` (soft failure) if it diverges.

        Evaluates the Jacobian at the pre-image ``x_n`` (so the accumulated
        product is ``J(x_{N-1})···J(x_0)`` — the exact tangent map), QR-ing every
        ``reortho_interval`` steps.  Float overflow on transient huge iterates is
        silenced locally and tracked explicitly as a soft failure.
        """
        self.reinit(ic)
        sys = self.system
        jac = type(sys)._jacobian
        params = sys.params.as_tuple()
        w = self._W
        sums = np.zeros(self.k)
        last = np.zeros(self.k)
        intervals = 0
        tiny = np.finfo(float).tiny

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for i in range(steps):
                x = sys.state()
                j = np.atleast_2d(np.asarray(jac(x, *params), dtype=float))
                if not np.all(np.isfinite(j)):
                    return False
                try:
                    sys.step()  # raises RuntimeError on a non-finite iterate
                except RuntimeError:
                    return False
                w = j @ w
                if not np.all(np.isfinite(w)):
                    return False
                if (i + 1) % reortho_interval == 0:
                    q, r = np.linalg.qr(w)
                    if not np.all(np.isfinite(q)):
                        return False
                    w = q
                    diag = np.abs(np.diag(r))
                    diag = np.where(diag < tiny, tiny, diag)
                    last = np.log(diag)
                    sums += last
                    intervals += 1

        if intervals == 0:
            return False
        self._W = w
        self._last_growths = last  # most recent reorthonormalisation's stretch
        self._sum_growths = sums
        self._elapsed = float(intervals * reortho_interval)
        return True

    def _lyapunov_spectrum_ode(
        self,
        final_time: float = 200.0,
        dt: float = 0.1,
        *,
        ic: Any | None = None,
        burn_in: float = 50.0,
        method: str | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **integrator_kwargs: Any,
    ) -> np.ndarray:
        """Burn-in then time-averaged spectrum for a flow (either ODE backend)."""
        if final_time <= 0:
            raise ValueError(f"final_time must be positive, got {final_time!r}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")

        self.reinit(ic, method=method, rtol=rtol, atol=atol, **integrator_kwargs)

        # Burn-in: advance the trajectory + tangent frame without accumulating.
        t_burn = self._t + max(0.0, burn_in)
        while self._t < t_burn - 1e-12:
            self.step(min(dt, t_burn - self._t))
        self._reset_accumulators()

        # Production: uniform-dt steps, time-weighted growth average.
        t_end = self._t + final_time
        while self._t < t_end - 1e-12:
            self.step(min(dt, t_end - self._t))

        exponents = self.exponents()
        self.meta.record(
            "lyapunov_spectrum",
            exponents,
            dt=dt,
            final_time=final_time,
            burn_in=burn_in,
            n_exp=self.k,
            method=method or self.system._default_method,
            backend=self._backend,
        )
        return exponents


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
