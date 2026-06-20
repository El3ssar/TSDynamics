"""DelaySystem — DDE base class on the Rust method-of-steps engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np

from .base import SystemBase, Trajectory

__all__ = ["DelaySystem"]

# ---------------------------------------------------------------------------
# Type alias for history functions
# ---------------------------------------------------------------------------

History = Callable[[float], Sequence[float]] | None


# ---------------------------------------------------------------------------
# DelaySystem
# ---------------------------------------------------------------------------


class DelaySystem(SystemBase, ABC):
    """
    Base class for delay differential systems (DDEs), integrated on the engine.

    Subclass contract
    -----------------
    1. Declare ``params = {...}`` and ``dim = N``.
    2. Implement ``_equations`` as a ``@staticmethod`` returning a
       length-``dim`` sequence of SymEngine symbolic expressions.
       Use ``y(i, t - tau)`` for delayed state access.

    Lowering
    --------
    Each system is lowered once to an in-process IR tape with no warmup.  Delay
    values directly affect the history-buffer structure, so they are baked into
    the tape rather than read live like the other parameters; a delay change
    re-lowers, while ordinary parameters are read live with no re-lowering.

    DDEs typically need looser tolerances than ODEs (start with ``rtol=atol=1e-3``).

    History
    -------
    Pass a ``history`` callable ``h(s) → sequence`` defining the past for
    ``s ≤ 0``.  If omitted, a constant past equal to ``ic`` is used.

    .. note::
        Provide a non-equilibrium history to avoid trivial Lyapunov exponents.
        ``lyapunov_spectrum`` starts from a constant past, so the workaround is
        to run ``integrate`` first with the desired history, then pass the
        end-state as ``ic`` to ``lyapunov_spectrum``.

    Examples
    --------
    >>> mg = MackeyGlass()
    >>> hist = lambda s: [1.0 + 0.1 * np.sin(0.2 * s)]
    >>> traj = mg.integrate(final_time=500, history=hist)
    >>> exps = mg.lyapunov_spectrum(n_exp=2, ic=traj.y[-1])
    """

    _default_rtol: ClassVar[float] = 1e-3
    _default_atol: ClassVar[float] = 1e-3

    #: The default runtime backend (see :attr:`SystemBase._default_backend`).
    #: ``"interp"`` — the Rust method-of-steps DDE engine (the sole DDE backend
    #: since the M3 migration retired the v2 backends).
    _default_backend: ClassVar[str] = "interp"

    #: Names of parameters that hold delay values (must be positive floats).
    #: Subclasses with custom delay-naming conventions should override this.
    #: The default ``("tau",)`` matches the convention used throughout the
    #: built-in DDE systems.  Override with ``("tau1", "tau2")`` etc. for
    #: multi-delay systems, or override ``_delays()`` for delays computed
    #: from other parameters.
    _delay_params: ClassVar[tuple[str, ...]] = ("tau",)

    # Protocol stepping state (instances shadow these class defaults).  DDE
    # stepping re-integrates from the constant past each call (no stateful
    # one-step restart in the method of steps).
    _past_ic: np.ndarray | None = None
    _state_now: np.ndarray | None = None
    _t_now: float = 0.0
    _default_step_dt: ClassVar[float] = 0.1

    # ------------------------------------------------------------------ #
    # Subclass interface
    # ------------------------------------------------------------------ #

    @staticmethod
    @abstractmethod
    def _equations(y, t, **params) -> Sequence:
        """
        Build the symbolic DDE RHS.

        Parameters
        ----------
        y : symbolic state accessor.
            ``y(i)`` for current state component ``i``;
            ``y(i, t - tau)`` for delayed access.
        t : symbolic time variable.
        **params
            Current parameter values as Python floats.

        Returns
        -------
        list of ``dim`` SymEngine expressions.
        """
        ...

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _cache_key(self) -> str:
        """Return a unique key for this (class, params, equations) combination."""
        import hashlib
        import inspect

        fn = type(self)._equations
        fn = getattr(fn, "__func__", fn)
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            src = repr(getattr(fn, "__code__", fn).co_code)
        eq = hashlib.md5(src.encode()).hexdigest()[:8]
        return f"{type(self).__name__}_{self.params.param_hash():016x}_{eq}"

    def _delays(self) -> list[float]:
        """
        Return the list of delay values used by this system.

        Default implementation reads each parameter name listed in
        ``_delay_params`` and returns them as floats.  Subclasses with
        delays computed from other parameters (e.g. ``tau = pi / omega``)
        should override this method.

        Returns
        -------
        list[float]
            One positive value per delay channel in the RHS.

        Raises
        ------
        ValueError
            If a declared delay parameter is missing, non-numeric, or
            non-positive.
        """
        try:
            delays = [float(self.params[k]) for k in type(self)._delay_params]
        except KeyError as err:
            missing = err.args[0]
            raise ValueError(
                f"{type(self).__name__}: delay parameter {missing!r} listed in "
                f"_delay_params but not found in self.params. Declared params: "
                f"{list(self.params)}"
            ) from err
        for k, d in zip(type(self)._delay_params, delays, strict=True):
            if not (d > 0.0):
                raise ValueError(
                    f"{type(self).__name__}: delay parameter {k!r} = {d!r} must be "
                    f"strictly positive."
                )
        return delays

    def _max_delay(self) -> float:
        """
        Return the maximum delay value with a small safety margin.

        The engine requires ``max_delay >= max(delays)``; we add a 1% margin to
        avoid edge effects in the history evaluation.
        """
        delays = self._delays()
        return max(delays) * 1.01 if delays else 1.0

    # ------------------------------------------------------------------ #
    # System protocol — incremental stepping (forward-only)
    # ------------------------------------------------------------------ #

    @property
    def is_discrete(self) -> bool:
        """DDEs are continuous-time systems."""
        return False

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        **kwargs,
    ) -> None:
        """
        (Re)start the incremental stepper from a constant past equal to ``u``.

        DDE state is a history *function*; the protocol restart uses a
        constant past (the same convention as ``lyapunov_spectrum``).  For a
        custom history, use :meth:`integrate` with ``history=`` and continue
        from ``traj.y[-1]``.

        Stepping is forward-only and re-integrates from the constant past on the
        Rust DDE engine each call (the method of steps has no stateful one-step
        restart), so it is correct but ``O(steps²)`` — use :meth:`integrate` for
        a full trajectory.
        """
        if params:
            for k, v in params.items():
                self.params[k] = v
        if t is not None and float(t) != 0.0:
            raise NotImplementedError(
                "DelaySystem.reinit only supports t=0 (the past starts there)."
            )
        self._past_ic = self.resolve_ic(u)
        self._step_rtol = rtol
        self._step_atol = atol
        self._state_now = self._past_ic.copy()
        self._t_now = 0.0

    def step(self, n_or_dt: float | None = None) -> np.ndarray:
        """Advance by ``dt`` (default 0.1, forward-only) and return the new state."""
        if self._past_ic is None:
            self.reinit()
        dt = float(n_or_dt) if n_or_dt is not None else self._default_step_dt
        self._t_now = self._t_now + dt
        traj = self.integrate(
            final_time=self._t_now,
            dt=min(dt, self._t_now),
            ic=self._past_ic,
            rtol=self._step_rtol if self._step_rtol is not None else self._default_rtol,
            atol=self._step_atol if self._step_atol is not None else self._default_atol,
        )
        state = np.asarray(traj.y[-1], dtype=float)
        if not np.isfinite(state).all():
            raise RuntimeError(
                f"{type(self).__name__}: DDE diverged at t={self._t_now:.6g} during step()."
            )
        self._state_now = state.copy()
        return state

    def state(self) -> np.ndarray:
        """Return a copy of the current state (implicit ``reinit`` if cold)."""
        if self._state_now is None:
            self.reinit()
        return self._state_now.copy()

    def set_state(self, u: Any) -> None:
        """Not available for DDEs — their state is a whole history function."""
        raise NotImplementedError(
            f"{type(self).__name__}.set_state is impossible for delay systems: the "
            f"instantaneous state is a history function over [t - max_delay, t], not a "
            f"point.  Use reinit(u) to restart from a constant past, or integrate(...) "
            f"with a history callable."
        )

    def time(self) -> float:
        """Return the current stepper time."""
        return self._t_now

    def trajectory(
        self,
        final_time: float = 100.0,
        *,
        dt: float = 0.02,
        transient: float = 0.0,
        **kwargs,
    ) -> Trajectory:
        """Protocol-uniform trajectory: ``integrate`` plus optional transient drop."""
        traj = self.integrate(final_time=transient + final_time, dt=dt, **kwargs)
        return traj.after(transient) if transient > 0 else traj

    # ------------------------------------------------------------------ #
    # Integration
    # ------------------------------------------------------------------ #

    def integrate(
        self,
        final_time: float = 100.0,
        dt: float = 0.02,
        *,
        ic: Any | None = None,
        history: History = None,
        rtol: float | None = None,
        atol: float | None = None,
        backend: str | None = None,
        method: str = "rk45",
        **kwargs,
    ) -> Trajectory:
        """
        Integrate the DDE and return a :class:`~tsdynamics.families.Trajectory`.

        Parameters
        ----------
        final_time : float
            Integration end time. Default 100.0.
        dt : float
            Output sampling interval.
        ic : array-like, optional
            Used for constant past when ``history`` is ``None``.
            Falls back to ``self.ic``, then random.
        history : callable, optional
            ``h(s) → sequence`` of length ``dim`` for ``s ≤ 0``.
            If ``None``, a constant past equal to ``ic`` is used.
        rtol, atol : float
            Integration tolerances.  DDEs typically need 1e-3; very tight
            tolerances can stall the solver.
        backend : str, optional
            Which engine integrates the DDE.  Defaults to ``_default_backend``
            (``"interp"``).  ``"interp"`` / ``"jit"`` route — through the shared
            engine seam (:func:`tsdynamics.engine.run.integrate`) — to the Rust
            method-of-steps engine (history ring buffer + cubic-Hermite dense
            interpolation; stream E-DDE), reusing the explicit solver kernels.
            Only **constant** delays lower; a state-dependent delay raises.
            ``backend="reference"`` is unsupported for DDEs (there is no
            pure-Python delay integrator).
        method : str, default "rk45"
            The explicit kernel (``"rk45"``, ``"tsit5"``, ``"dop853"``,
            ``"rk4"``); the method of steps drives explicit kernels only.

        Returns
        -------
        Trajectory
        """
        backend = backend if backend is not None else self._default_backend
        return self._integrate_engine(
            final_time,
            dt,
            ic=ic,
            history=history,
            rtol=rtol,
            atol=atol,
            backend=backend,
            method=method,
        )

    # ------------------------------------------------------------------ #
    # Rust engine integration (method of steps) — stream E-DDE
    # ------------------------------------------------------------------ #

    def _integrate_engine(
        self,
        final_time: float,
        dt: float,
        *,
        ic: Any | None,
        history: History,
        rtol: float | None,
        atol: float | None,
        backend: str,
        method: str,
    ) -> Trajectory:
        """Integrate the DDE on the Rust method-of-steps engine (stream E-DDE).

        Routes through the shared engine-dispatch seam
        (:func:`tsdynamics.engine.run.integrate`), which lowers the delay system
        (via :func:`tsdynamics.engine.compile.lower_dde`) to a tape over
        ``dim + n_slots`` inputs (the delay slots), samples the past, and drives
        the Rust method-of-steps integrator — a history ring buffer with
        cubic-Hermite dense interpolation, reusing the explicit solver kernels.
        Only constant delays lower; a state-dependent delay raises
        ``TapeCompileError``, and ``backend="reference"`` raises (there is no
        pure-Python delay integrator).

        DDE tolerances default to ``_default_rtol`` / ``_default_atol`` (both
        ``1e-3``) and are resolved here before handing off, since the generic
        seam's ODE-style ``1e-6`` / ``1e-9`` defaults are too tight for delay
        systems.
        """
        rtol = rtol if rtol is not None else self._default_rtol
        atol = atol if atol is not None else self._default_atol
        return self._dispatch(
            backend=backend,
            final_time=final_time,
            dt=dt,
            ic=ic,
            history=history,
            method=method,
            rtol=rtol,
            atol=atol,
        )

    # ------------------------------------------------------------------ #
    # Lyapunov spectrum
    # ------------------------------------------------------------------ #

    def lyapunov_spectrum(
        self,
        final_time: float = 200.0,
        dt: float = 0.1,
        *,
        ic: Any | None = None,
        n_exp: int = 1,
        burn_in: float = 50.0,
        rtol: float | None = None,
        atol: float | None = None,
        backend: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Estimate the ``n_exp`` leading Lyapunov exponents of the delay system.

        The **engine** estimator (stream E-DDE-LYAP, result stored in
        ``self.meta['lyapunov_spectrum']``) integrates the extended variational
        DDE on the Rust engine with a function-space Benettin renormalisation
        (:func:`tsdynamics.families._dde_lyapunov.dde_lyapunov_spectrum`):
        ``backend="interp"`` / ``"jit"``.  ``"reference"`` is rejected (the engine
        has no pure-Python DDE integrator).

        Parameters
        ----------
        final_time : float
            Averaging window after burn-in. Default 200.0.
        dt : float
            Sampling interval (should divide the maximum delay).
        ic : array-like, optional
            Initial state. Provide the end-state of a prior ``integrate``
            call so the trajectory starts on the attractor (recommended).
        n_exp : int
            Number of leading exponents to estimate. DDEs have infinitely
            many; choose consciously. Default 1.
        burn_in : float
            Discard interval. Default 50.0.
        rtol, atol : float, optional
            Integration tolerances.  The engine path renormalises every delay
            window and uses defaults of ``1e-7`` / ``1e-9``.
        backend : str, optional
            ``"interp"`` or ``"jit"``.  Defaults to :attr:`_default_backend`
            (``"interp"``).

        Notes
        -----
        For best results, pass ``ic=traj.y[-1]`` from a prior ``integrate`` run —
        this places the trajectory on the attractor and avoids trivial exponents
        from equilibrium pasts.

        Returns
        -------
        ndarray, shape (n_exp,)
        """
        backend = backend if backend is not None else self._default_backend
        from tsdynamics.families._dde_lyapunov import dde_lyapunov_spectrum

        if kwargs:
            raise TypeError(
                f"lyapunov_spectrum(backend={backend!r}) does not accept the "
                f"extra integration keyword(s) {sorted(kwargs)}."
            )
        exps = dde_lyapunov_spectrum(
            self,
            n_exp=n_exp,
            final_time=final_time,
            dt=dt,
            burn_in=burn_in,
            ic=ic,
            backend=backend,
            rtol=rtol if rtol is not None else 1e-7,
            atol=atol if atol is not None else 1e-9,
        )
        self.meta.record(
            "lyapunov_spectrum",
            exps,
            backend=backend,
            n_exp=n_exp,
            final_time=final_time,
            dt=dt,
            burn_in=burn_in,
        )
        return exps


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
