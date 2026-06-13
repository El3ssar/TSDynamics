"""Tangent dynamics: state + deviation vectors, the engine behind Lyapunov spectra."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from tsdynamics.base import ContinuousSystem, DelaySystem, DiscreteMap

from ._base import DerivedSystem

__all__ = ["TangentSystem"]


class TangentSystem(DerivedSystem):
    """
    Evolve a system together with ``k`` deviation (tangent) vectors.

    Each ``step()`` advances the state and the deviation vectors, then
    QR-reorthonormalises; :meth:`growths` exposes the per-step logarithmic
    stretch factors ``log |diag R|`` and :meth:`exponents` their running
    time-average — the Lyapunov spectrum estimate.

    Implementation per family
    -------------------------
    - **Maps**: pure NumPy — ``W ← J(x)·W`` with the hand-written/compiled
      ``_jacobian``, then QR.
    - **ODEs**: a thin wrapper over the compiled variational equations
      (``jitcode_lyap``), which JiTCODE builds by symbolic differentiation —
      no Python-side Jacobian evaluation in the hot loop.
    - **DDEs**: not supported — tangent dynamics of a DDE lives in an
      infinite-dimensional history space; use
      ``DelaySystem.lyapunov_spectrum`` which wraps ``jitcdde_lyap``.

    Examples
    --------
    >>> tang = TangentSystem(Henon(), k=2)
    >>> tang.reinit([0.1, 0.1])
    >>> for _ in range(5000):
    ...     tang.step()
    >>> tang.exponents()        # ≈ [0.42, -1.62]
    """

    _ODE_DEFAULT_DT: ClassVar[float] = 0.1

    def __init__(self, system: Any, k: int | None = None) -> None:
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

        self._W: np.ndarray | None = None  # (dim, k) deviation vectors (map mode)
        self._ode_lyap: Any = None  # jitcode_lyap wrapper (ode mode)
        self._t = 0.0
        self._last_growths = np.zeros(self.k)
        self._sum_growths = np.zeros(self.k)
        self._elapsed = 0.0

    def _rebuild(self, inner: Any) -> TangentSystem:
        return TangentSystem(inner, self.k)

    # --- lifecycle ---

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Restart state, deviation vectors, and accumulated growth sums."""
        self._last_growths = np.zeros(self.k)
        self._sum_growths = np.zeros(self.k)
        self._elapsed = 0.0

        if self._mode == "map":
            self.system.reinit(u, t=t, params=params)
            self._W = np.eye(self.system.dim)[:, : self.k]
            return

        # ode mode — fresh jitcode_lyap from the cached compiled module
        if params:
            for key, value in params.items():
                self.params[key] = value
        ic_arr = self.system.resolve_ic(u)
        t0 = float(t) if t is not None else 0.0

        method = kwargs.pop("method", None) or self.system._default_method
        rtol = kwargs.pop("rtol", 1e-6)
        atol = kwargs.pop("atol", 1e-9)
        from tsdynamics.base.ode_base import _INTEGRATOR_MAP

        ode = self.system._ensure_compiled(for_lyap=True, n_lyap=self.k)
        ode.set_integrator(_INTEGRATOR_MAP.get(method, method), rtol=rtol, atol=atol, **kwargs)
        ode.set_parameters(*self.system._control_params().values())
        ode.set_initial_value(ic_arr, t0)
        self._ode_lyap = ode
        self._t = t0

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
        return self._step_ode(float(n_or_dt) if n_or_dt is not None else self._ODE_DEFAULT_DT)

    def _step_map(self, n: int) -> np.ndarray:
        if self._W is None:
            self.reinit()
        sys = self.system
        jac = type(sys)._jacobian
        params = sys.params.as_tuple()
        x = sys.state()
        for _ in range(n):
            J = np.atleast_2d(np.asarray(jac(x, *params), dtype=float))
            x = sys.step()
            self._W = J @ self._W
            Q, R = np.linalg.qr(self._W)
            self._W = Q
            diag = np.abs(np.diag(R))
            diag = np.where(diag < np.finfo(float).tiny, np.finfo(float).tiny, diag)
            self._last_growths = np.log(diag)
            self._sum_growths += self._last_growths
            self._elapsed += 1.0
        return x

    def _step_ode(self, dt: float) -> np.ndarray:
        if self._ode_lyap is None:
            self.reinit()
        self._t += dt
        ret = self._ode_lyap.integrate(self._t)
        if not (isinstance(ret, tuple) and len(ret) >= 2):
            raise RuntimeError("jitcode_lyap.integrate returned an unexpected value")
        state = np.asarray(ret[0], dtype=float)
        local = np.asarray(ret[1], dtype=float).ravel()[: self.k]
        self._last_growths = local * dt  # local exponents → log stretch over dt
        self._sum_growths += self._last_growths
        self._elapsed += dt
        return state

    def state(self) -> np.ndarray:
        """Return a copy of the current base-system state."""
        if self._mode == "map":
            return self.system.state()
        if self._ode_lyap is None:
            self.reinit()
        return np.asarray(self._ode_lyap.y[: self.system.dim], dtype=float).copy()

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
        """Return the current orthonormal deviation vectors, shape ``(dim, k)`` (maps only)."""
        if self._mode != "map" or self._W is None:
            raise RuntimeError("deviations() is available for map mode after reinit()")
        return self._W.copy()

    def growths(self) -> np.ndarray:
        """Return the log stretch factors ``log|diag R|`` from the most recent step."""
        return self._last_growths.copy()

    def exponents(self) -> np.ndarray:
        """Return the running Lyapunov-spectrum estimate (accumulated growths / elapsed)."""
        if self._elapsed == 0.0:
            return np.zeros(self.k)
        return self._sum_growths / self._elapsed
