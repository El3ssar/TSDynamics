import warnings
from abc import abstractmethod
from typing import Callable, Optional, Sequence, Tuple

warnings.filterwarnings(
    "ignore",
    message=".*target time is smaller than the current time.*",
    category=UserWarning,
)


import numpy as np
from jitcdde import jitcdde, t, y

from .base import BaseDyn


class DynSysDelay(BaseDyn):
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
        method: Optional[str] = "auto",
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
        if self.n_dim is None:
            raise ValueError("n_dim must be set on the system.")

        if initial_conds is None:
            initial_conds = np.random.rand(self.n_dim)
        initial_conds = np.asarray(initial_conds, dtype=float).reshape(self.n_dim)

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
            dde.constant_past(initial_conds)
            hist0 = initial_conds
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


        # for k in range(i0, t_eval.size):
        #     tk = float(t_eval[k])
        #     if tk <= 0.0:
        #         y_out[k] = hist0
        #         continue

        #     t_curr = float(dde.t)
        #     if tk <= t_curr:
        #         # Evaluate from the spline, not by integrating “backwards”
        #         spline = dde.get_state()  # CHSPy CubicHermiteSpline
        #         # get_state accepts an array; returns shape (m, n_dim)
        #         y_out[k] = np.asarray(spline.get_state([tk]))[0]
        #     else:
        #         # Advance strictly forward
        #         y_out[k] = dde.integrate(tk)

        # spline = dde.get_state()   # CHSPy cubic Hermite spline view

        # for k in range(i0, t_eval.size):
        #     tk = float(t_eval[k])
        #     if tk <= 0.0:
        #         y_out[k] = hist0
        #         continue

        #     # if target already covered by integrator, evaluate directly from spline
        #     if tk <= dde.t:
        #         y_out[k] = np.asarray(spline.get_state([tk]))[0]
        #     else:
        #         y_out[k] = dde.integrate(tk)
        #         spline = dde.get_state()  # refresh after each advance

        self.initial_conds = np.array(initial_conds, copy=True)
        return t_eval, y_out