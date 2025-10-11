import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from jitcdde import jitcdde, jitcdde_lyap, t, y

from .base import BaseDyn

warnings.filterwarnings(
    "ignore",
    message=".*target time is smaller than the current time.*",
    category=UserWarning,
)

class DynSysDelay(BaseDyn, ABC):
    """
    Base class for delay differential systems (DDEs) using jitcdde.

    Subclasses implement `_rhs(y, t, **params)` and return a list/tuple
    of length `n_dim` with expressions built from jitcdde's `y(i, time)` and `t`.
    """

    # --------- Interface similar to DynSys ---------
    def rhs(self, y_sym, t_sym):
        """
        Wrapper to pass parameters into subclass rhs.
        Returns a tuple/list of expressions of length n_dim.
        """
        return self._rhs(y_sym, t_sym, **self.params)

    @abstractmethod
    def _rhs(self, y_sym, t_sym, **params):
        """
        Provide the DDE right-hand side as jitcdde expressions.

        Parameters
        ----------
        y_sym : callable
            jitcdde symbol: y(index, time)
        t_sym : symbol
            jitcdde symbol for time t
        **params : dict
            Parameters made available as attributes and kwargs.

        Returns
        -------
        Sequence of length n_dim with expressions referencing y_sym(., t_sym) and delays.
        """
        raise NotImplementedError

    # --------- DDE integration (jitcdde) ---------
    def integrate(
        self,
        dt: float = 0.02,
        steps: Optional[int] = None,
        final_time: float = 100.0,
        initial_conds: Optional[Sequence[float]] = None,
        rtol: float = 1e-3,
        atol: float = 1e-3,
        history: Optional[Callable[[float], Sequence[float]]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the DDE system with adaptive control via jitcdde.

        Parameters
        ----------
        dt : float
            Output sampling step for returned trajectory (not the internal stepper step).
        steps : int, optional
            Number of output points; if given, overrides final_time.
        final_time : float
            Final simulation time if `steps` is not provided.
        initial_conds : sequence of float, optional
            Used to set a constant past if `history` is None.
        rtol, atol : float
            Relative and absolute tolerances passed to jitcdde.
        history : callable, optional
            Function h(s) -> sequence at time s (s ≤ 0) to define the past.
            If None, a constant past is used from `initial_conds`.
        **kwargs :
            Passed to `set_integration_parameters` (e.g., max_step, first_step, min_step).

        Returns
        -------
        t_eval : ndarray, shape (m,)
            Output times.
        y_eval : ndarray, shape (m, n_dim)
            Solution values at `t_eval`.

        Raises
        ------
        ValueError
            If shapes/params are inconsistent.
        """
        # Determine dimensions and initial conditions / past
        if initial_conds is None:
            if self.initial_conds is None:
                initial_conds = np.random.rand(self.n_dim)
                initial_conds = np.asarray(initial_conds, float).reshape(self.n_dim)
                self.initial_conds = np.array(initial_conds, copy=True)

        # Output grid
        t_eval = self.generate_timesteps(dt=dt, steps=steps, final_time=final_time)
        if t_eval[0] < 0.0:
            raise ValueError("DDE integration requires nonnegative output times (t >= 0).")

        # Build jitcdde system
        rhs = tuple(self.rhs(y, t))
        if len(rhs) != self.n_dim:
            raise ValueError(f"_rhs must return length {self.n_dim}, got {len(rhs)}")

        dde = jitcdde(rhs)

        # Past / history
        if history is None:
            dde.constant_past(self.initial_conds)
            hist0 = self.initial_conds
        else:
            # history(s) must return a sequence of length n_dim for any s ≤ 0
            def _hist(s: float) -> np.ndarray:
                return np.asarray(history(s), dtype=float).reshape(self.n_dim)
            dde.past_from_function(_hist)
            hist0 = _hist(0.0)

        # Tolerances and optional steps
        dde.set_integration_parameters(rtol=rtol, atol=atol, **kwargs)

        # If you are confident your history is compatible and want silence:
        dde.initial_discontinuities_handled = True

        # Integrate to requested times
        y_out = np.empty((t_eval.size, self.n_dim), dtype=float)

        # t=0 value: use the history at 0
        y_out[0] = hist0
        i0 = 1

        # march forward
        for k in range(i0, t_eval.size):
            tk = float(t_eval[k])
            y_out[k] = dde.integrate(tk)

        return t_eval, y_out

    def lyapunov_spectrum(
        self,
        dt: float = 0.1,
        final_time: float = 200.0,
        initial_conds: Optional[Sequence[float]] = None,
        n_lyap: int = 1,  # user chooses how many (DDEs have infinitely many)
        history: Optional[Callable[[float], Sequence[float]]] = None,
        burn_in: float = 50.0,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **integration_kwargs,
    ) -> np.ndarray:
        """
        Estimate the first ``n_lyap`` Lyapunov exponents of a DDE using
        :func:`jitcdde.jitcdde_lyap`.

        This integrates the delay system together with ``n_lyap`` separation functions.
        At each sampling time, JiTCDDE returns *local* exponents and a **weight**
        (the effective integration time they represent). The reported spectrum is the
        weight-averaged mean of those local values after an optional burn-in.

        Parameters
        ----------
        dt : float, optional
            Sampling interval (in integration time units) at which local exponents are
            requested. This does **not** constrain the adaptive internal stepper.
            Default is ``0.1``.
        final_time : float, optional
            Length of the averaging window *after* burn-in. Default is ``200.0``.
        initial_conds : sequence of float, optional
            Used to define a constant past if ``history`` is not provided. Shape
            ``(n_dim,)`` at time ``t=0``. If omitted, uses ``self.initial_conds`` if
            available, else random ``U[0,1)``.
        n_lyap : int, optional
            Number of leading Lyapunov exponents to estimate (DDEs have infinitely many).
            Default is ``1``.
        history : callable, optional
            Function ``h(s) -> sequence`` defining the past for ``s <= 0``. If omitted,
            a constant past equal to ``initial_conds`` is used.
        burn_in : float, optional
            Time to discard before averaging (aligns separation functions). Default
            ``50.0``.
        rtol : float, optional
            Relative tolerance for JiTCDDE. Default ``1e-6``.
        atol : float, optional
            Absolute tolerance for JiTCDDE. Default ``1e-9``.
        **integration_kwargs
            Additional keyword args forwarded to
            :meth:`jitcdde.jitcdde.set_integration_parameters` (e.g., ``max_step``,
            ``first_step``).

        Returns
        -------
        exponents : (n_lyap,) ndarray of float
            Estimated weight-averaged Lyapunov exponents (largest first, as produced
            by JiTCDDE).

        Raises
        ------
        ValueError
            If ``n_dim`` is not set, or the subclass ``_rhs`` returns the wrong length.

        Notes
        -----
        - This method sets the past (constant or from ``history``), calls
        :meth:`jitcdde.jitcdde_lyap.step_on_discontinuities`, and then starts
        sampling from ``dde.t`` as recommended in the JiTCDDE docs.
        - Each call to ``jitcdde_lyap.integrate(T)`` returns a tuple
        ``(state, local_lyaps, weight)``. **You must use** the returned ``weight``
        when averaging local exponents; it may be zero if no real integration
        occurred between two sampling times.
        - Avoid histories that place the system exactly at an equilibrium (they can
        yield exponents near zero). For Mackey–Glass, for example, a strictly
        constant past at the fixed point produces trivial dynamics.

        Examples
        --------
        >>> mg = MackeyGlass()  # beta=0.2, gamma=0.1, tau=17, n=10 by default
        >>> # Provide a nontrivial past (constant 1.0 is an equilibrium here):
        >>> hist = lambda s: [1.0 + 0.1*np.sin(0.2*s)]
        >>> exps = mg.lyapunov_spectrum(n_lyap=2, dt=0.2, burn_in=100.0, final_time=300.0,
        ...                             history=hist, rtol=1e-8, atol=1e-10)
        >>> exps  # doctest: +SKIP
        array([ 2.8e-03, -... ])
        """
        if self.n_dim is None:
            raise ValueError("n_dim must be set.")

        # Past / ICs
        if initial_conds is None:
            if self.initial_conds is None:
                initial_conds = np.random.rand(self.n_dim)
                initial_conds = np.asarray(initial_conds, float).reshape(self.n_dim)
                self.initial_conds = np.array(initial_conds, copy=True)

        # Build symbolic field
        f = tuple(self.rhs(y, t))
        if len(f) != self.n_dim:
            raise ValueError(f"_rhs must return length {self.n_dim}, got {len(f)}")

        dde = jitcdde_lyap(f, n_lyap=n_lyap)
        dde.set_integration_parameters(rtol=rtol, atol=atol, **integration_kwargs)

        if history is None:
            dde.constant_past(initial_conds)
        else:
            dde.past_from_function(lambda s: np.asarray(history(s), float).reshape(self.n_dim))

        # Handle initial discontinuities; start sampling from dde.t afterwards
        dde.step_on_discontinuities()

        # Burn-in: align separation functions (discard output)
        T_end_burn = float(dde.t) + max(0.0, burn_in)
        while dde.t < T_end_burn:
            Tn = min(T_end_burn, dde.t + dt)
            _ = dde.integrate(Tn)  # returns (state, local_lyaps, weight)

        # Production: weight-average the local LEs using the returned weights
        T_end = float(dde.t) + final_time
        weights = []
        ly_steps = []
        prev_time = float(dde.t)

        while dde.t < T_end:
            Tn = min(T_end, dde.t + dt)
            ret = dde.integrate(Tn)  # expected: (state, local_lyaps, weight)
            if not isinstance(ret, tuple):
                raise RuntimeError("jitcdde_lyap.integrate did not return a tuple")
            if len(ret) >= 3:
                _, local_lyaps, w = ret
                weight = float(w)
            else:
                # Fallback (shouldn't happen): use elapsed time as weight
                _, local_lyaps = ret
                weight = Tn - prev_time

            v = np.asarray(local_lyaps, float).reshape(-1)
            if v.size != n_lyap:
                raise ValueError(f"Expected {n_lyap} local LEs, got {v.shape}")

            ly_steps.append(v)
            weights.append(weight)
            prev_time = float(dde.t)

        W = np.asarray(weights, float)
        mask = W > 0.0
        if not np.any(mask):
            return np.zeros(n_lyap)
        L = np.vstack([ly_steps[i] for i, m in enumerate(mask) if m])
        exponents = (W[mask, None] * L).sum(axis=0) / W[mask].sum()
        return exponents