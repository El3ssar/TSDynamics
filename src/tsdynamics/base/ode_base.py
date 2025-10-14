from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import numpy as np
from jitcode import jitcode, jitcode_lyap, y, t   # pip install jitcode
from .base import BaseDyn


# Map SciPy-style names to JiTCODE integrators
_INTEGRATOR_MAP = {
    None:        "dopri5",
    "auto":      "dopri5",
    "dopri5":    "dopri5",
    "RK45":      "dopri5",
    "dop853":    "dop853",
    "DOP853":    "dop853",
    "lsoda":     "lsoda",
    "LSODA":     "lsoda",
    "vode":      "vode",
    "VODE":      "vode",
}

class DynSys(BaseDyn, ABC):
    """Class for continuous dynamical systems (ODEs) using JiTCODE."""

    # --------- Symbolic interface (mirrors DynSysDelay) ---------
    def rhs(self, y_sym, t_sym):
        """Wrapper to pass params into subclass equations (static)."""
        return self._rhs(y_sym, t_sym, **self.params)

    def jac(self, y_sym, t_sym):
        """Optional symbolic Jacobian if subclass provides it; else None."""
        if hasattr(self, "_jac"):
            return self._jac(y_sym, t_sym, **self.params)  # may be NotImplemented
        return None

    @abstractmethod
    def _rhs(y_sym, t_sym, **params):
        """
        Return a sequence (len n_dim) with JiTCODE expressions.
        Use y_sym(i) for state components and t_sym for time.
        """
        raise NotImplementedError

    # Optional; subclasses may omit
    @staticmethod
    def _jac(y_sym, t_sym, **params):
        """Return Jacobian expressions or NotImplemented to skip."""
        return NotImplemented

    # --------- Integration ---------
    def integrate(
        self,
        dt: float = 0.02,
        steps: Optional[int] = None,
        final_time: float = 100.0,
        initial_conds: Optional[Sequence[float]] = None,
        method: Optional[str] = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the ODE with JiTCODE. Returns (t_eval, y_eval) where
        y_eval has shape (len(t_eval), n_dim).
        """
        if self.n_dim is None:
            raise ValueError("n_dim must be set.")

        # Initial conditions
        if initial_conds is None:
            if self.initial_conds is None:
                initial_conds = np.random.rand(self.n_dim)
                initial_conds = np.asarray(initial_conds, float).reshape(self.n_dim)
                self.initial_conds = np.array(initial_conds, copy=True)
        else:
            self.initial_conds = np.array(initial_conds, copy=True)


        # Output grid
        t_eval = self.generate_timesteps(dt=dt, steps=steps, final_time=final_time)
        if t_eval[0] < 0.0:
            raise ValueError("t_eval must be nonnegative.")

        ode = jitcode(self.rhs(y, t))

        # Compile the RHS (and Jacobian if set)
        ode.generate_f_C()          # generate & compile C code for f
        ode.generate_jac_C()    # compile Jacobian if provided

        # Set integrator (maps to SciPy’s ODE integrators)
        integ_name = _INTEGRATOR_MAP.get(method, method)
        if integ_name is None:
            integ_name = "dopri5"
        ode.set_integrator(integ_name, rtol=rtol, atol=atol, **kwargs)

        # Initial state at t=0
        ode.set_initial_value(self.initial_conds, 0.0)

        # March over t_eval. If a requested tk is <= current ode.t, evaluate from spline.
        y_out = np.empty((t_eval.size, self.n_dim), float)

        # Fill t=0 directly from initial conditions (no integrate at 0)
        i0 = 0
        if t_eval[0] == 0.0:
            y_out[0] = self.initial_conds
            i0 = 1

        # Helper: evaluate CHS spline when target not ahead
        def _eval_spline(tk: float) -> np.ndarray:
            spline = ode.get_state()                         # CubicHermiteSpline
            return np.asarray(spline.get_state([tk]))[0]     # shape (n_dim,)

        for k in range(i0, t_eval.size):
            tk = float(t_eval[k])
            t_curr = float(ode.t)
            if tk <= t_curr:
                y_out[k] = _eval_spline(tk)
            else:
                y_out[k] = ode.integrate(tk)

        return t_eval, y_out

    def lyapunov_spectrum(
        self,
        dt: float = 0.1,
        final_time: float = 200.0,
        initial_conds: Optional[Sequence[float]] = None,
        n_lyap: Optional[int] = None,
        burn_in: float = 50.0,
        method: Optional[str] = "dopri5",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **integrator_kwargs,
    ) -> np.ndarray:
        """
        Estimate the Lyapunov spectrum of an ODE using :func:`jitcode.jitcode_lyap`.

        This integrates the system together with `n_lyap` orthonormal tangent
        vectors and accumulates the per-step *local* Lyapunov exponents returned by
        JiTCODE. The reported spectrum is a time-weighted average of those local values
        after an optional burn-in period.

        Parameters
        ----------
        dt : float, optional
            Sampling interval (in integration time units) at which local exponents are
            recorded and averaged. This does **not** constrain the adaptive internal
            stepper; it only sets the spacing of requests to ``integrate``. Default is
            ``0.1``.
        final_time : float, optional
            Length of the averaging window *after* burn-in. Default is ``200.0``.
        initial_conds : sequence of float, optional
            Initial state at ``t=0``. If omitted, uses ``self.initial_conds`` if
            available, else random ``U[0,1)`` of length ``n_dim``.
        n_lyap : int, optional
            Number of Lyapunov exponents to estimate. Defaults to ``self.n_dim``.
        burn_in : float, optional
            Time to discard at the beginning (aligns tangent vectors before averaging).
            Default is ``50.0``.
        method : {"dopri5","dop853","vode","lsoda",None,"auto"}, optional
            JiTCODE integrator to use. ``None`` or ``"auto"`` map to ``"dopri5"``.
            For explicit RK methods (``dopri5``, ``dop853``) no Jacobian is used.
        rtol : float, optional
            Relative tolerance for the integrator. Default ``1e-6``.
        atol : float, optional
            Absolute tolerance for the integrator. Default ``1e-9``.
        **integrator_kwargs
            Additional keyword args forwarded to :meth:`jitcode.jitcode.set_integrator`
            (e.g., ``max_step``, ``first_step``).

        Returns
        -------
        exponents : (n_lyap,) ndarray of float
            Estimated (time-weighted) Lyapunov exponents ordered from largest to
            smallest as produced by JiTCODE.

        Raises
        ------
        ValueError
            If ``n_dim`` is not set, or the subclass ``_rhs`` returns the wrong length.

        Notes
        -----
        - JiTCODE’s ``jitcode_lyap.integrate(T)`` returns a tuple whose second element
        is the vector of local Lyapunov exponents at time ``T``; some versions also
        return the Lyapunov vectors as a third element. This method extracts and
        averages only the local exponents.
        - Averaging uses time weights equal to the elapsed time between successive
        sampling calls. If ``dt`` is constant, this is equivalent to a simple mean.
        - Convergence depends on burn-in, history of the trajectory, and tolerances.
        Increase ``final_time`` and tighten ``rtol/atol`` to verify stability of the
        reported spectrum.

        Examples
        --------
        >>> lor = Lorenz()  # sigma=10, rho=28, beta=8/3 by default
        >>> exps = lor.lyapunov_spectrum(dt=0.1, burn_in=50.0, final_time=300.0,
        ...                              initial_conds=[1.0, 1.0, 1.0],
        ...                              method="dop853", rtol=1e-8, atol=1e-10)
        >>> exps  # doctest: +SKIP
        array([ 0.91...,  0.00..., -14.57... ])
        """
        if initial_conds is None:
            if self.initial_conds is None:
                initial_conds = np.random.rand(self.n_dim)
                initial_conds = np.asarray(initial_conds, float).reshape(self.n_dim)
                self.initial_conds = np.array(initial_conds, copy=True)

        if n_lyap is None:
            n_lyap = self.n_dim

        # Symbolic vector field
        f = tuple(self.rhs(y, t))
        if len(f) != self.n_dim:
            raise ValueError(f"_rhs must return length {self.n_dim}, got {len(f)}")

        ode = jitcode_lyap(f, n_lyap=n_lyap)
        integ = _INTEGRATOR_MAP.get(method, method or "dopri5")
        ode.set_integrator(integ, rtol=rtol, atol=atol, **integrator_kwargs)
        ode.set_initial_value(initial_conds, 0.0)

        # Burn-in
        T = 0.0
        while T < burn_in:
            Tn = min(burn_in, T + dt)
            _ = ode.integrate(Tn)          # may return (state, lyaps) or (state, lyaps, lyap_vectors)
            T = Tn

        # Production: time-weighted average of local exponents
        T_end = float(ode.t) + final_time
        weights = []
        ly_steps = []
        T = float(ode.t)
        while T < T_end:
            Tn = min(T_end, T + dt)
            ret = ode.integrate(Tn)

            # Robust extraction of local LEs across JiTCODE versions
            if isinstance(ret, tuple):
                if len(ret) >= 2:
                    local_lyaps = ret[1]   # (state, lyaps, [lyap_vectors])
                else:
                    local_lyaps = ret
            else:
                local_lyaps = ret

            v = np.asarray(local_lyaps, float).reshape(-1)
            if v.size != n_lyap:
                raise ValueError(f"Expected {n_lyap} local LEs, got shape {v.shape}")

            ly_steps.append(v)
            weights.append(Tn - T)
            T = Tn

        W = np.asarray(weights, float)
        L = np.vstack(ly_steps) if ly_steps else np.empty((0, n_lyap), float)
        exponents = (W[:, None] * L).sum(axis=0) / W.sum() if L.size else np.zeros(n_lyap)
        return exponents