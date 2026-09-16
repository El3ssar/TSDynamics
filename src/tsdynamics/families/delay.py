"""DelaySystem — DDE base class on the Rust method-of-steps engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from tsdynamics.errors import ConvergenceError, InvalidInputError, InvalidParameterError
from tsdynamics.utils.tolerances import (
    DDE_ATOL,
    DDE_LYAPUNOV_ATOL,
    DDE_LYAPUNOV_RTOL,
    DDE_RTOL,
)

from ._kwargs import reject_unknown_run_keywords
from .base import Absent, SystemBase, Trajectory, resolve_transient

#: ``DelaySystem.run``'s keywords, in signature order.
_DDE_RUN_KEYWORDS = (
    "final_time",
    "dt",
    "ic",
    "history",
    "transient",
    "solver",
    "rtol",
    "atol",
    "backend",
    "seed",
)

if TYPE_CHECKING:
    from .base import ParamSet

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
    Each system is lowered once to an in-process IR tape, then JIT-compiled on
    first use (both memoised, so neither cost repeats).  Delay
    values directly affect the history-buffer structure, so they are baked into
    the tape rather than read live like the other parameters; a delay change
    re-lowers, while ordinary parameters are read live with no re-lowering.

    Tolerances
    ----------
    The DDE family keeps its **own**, looser default —
    :data:`~tsdynamics.utils.tolerances.DDE_RTOL` /
    :data:`~tsdynamics.utils.tolerances.DDE_ATOL` (``1e-3`` / ``1e-3``) — rather
    than the ODE :data:`~tsdynamics.utils.tolerances.DEFAULT_RTOL` /
    :data:`~tsdynamics.utils.tolerances.DEFAULT_ATOL`.  The reason is *not* that
    the solver struggles when tightened (it does not: all six built-in DDEs
    complete at ``rtol=1e-12``/``atol=1e-15``).  It is that the tolerance is
    largely **inert** here: the method of steps lands on every output sample, so
    ``dt`` already bounds the internal step below the natural error.  Measured
    over all six built-in DDEs at the default ``dt=0.02`` to ``T=10``, five
    return a **bit-identical** final state at ``rtol=1e-3`` and at
    ``rtol=1e-9``; the sixth (``IkedaDelay``, the only one whose step is
    tolerance-bound) costs 1.4x for a 6x accuracy gain.  Tighten it explicitly if
    you need that.

    .. versionchanged:: 6.0
        The ODE default tightened to ``1e-9``/``1e-12`` to compensate for native
        dense output.  DDEs never had dense output — the march always landed on
        every sample — so they lost nothing and this default is unchanged.

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
    >>> import tsdynamics as ts
    >>> mg = MackeyGlass()
    >>> hist = lambda s: [1.0 + 0.1 * np.sin(0.2 * s)]
    >>> traj = mg.run(final_time=500, history=hist)
    >>> exps = ts.analysis.lyapunov_spectrum(mg, k=2, ic=traj.y[-1])
    """

    #: The DDE family's own integration tolerances (see the class docstring for
    #: the measurement that justifies keeping them looser than the ODE default).
    _default_rtol: ClassVar[float] = DDE_RTOL
    _default_atol: ClassVar[float] = DDE_ATOL

    #: The default runtime backend (see :attr:`SystemBase._default_backend`).
    #: ``"jit"`` — the Rust method-of-steps DDE engine driven by the Cranelift
    #: JIT, with the compiled-evaluator cache paying the compile once per
    #: distinct system.  Was ``"interp"`` before v6.  There is no ``"reference"``
    #: DDE integrator, so the two engine evaluators are the only options.
    _default_backend: ClassVar[str] = "jit"

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
    def _equations(y: Any, t: Any, **params: Any) -> Sequence[Any]:
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
            code = getattr(fn, "__code__", None)
            src = repr(code.co_code) if code is not None else repr(fn)
        eq = hashlib.md5(src.encode()).hexdigest()[:8]
        return f"{type(self).__name__}_{cast('ParamSet', self.params).param_hash():016x}_{eq}"

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
            raise InvalidParameterError(
                f"{type(self).__name__}: delay parameter {missing!r} listed in "
                f"_delay_params but not found in self.params. Declared params: "
                f"{list(self.params)}"
            ) from err
        for k, d in zip(type(self)._delay_params, delays, strict=True):
            if not (d > 0.0):
                raise InvalidParameterError(
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

    #: The output sampling interval ``run`` uses when given no ``dt``.  The
    #: method of steps lands on every sample, so it *does* bound the step here.
    _default_dt: ClassVar[float] = 0.02

    #: The solver kernel each delay window is integrated with by default.
    _default_method: ClassVar[str] = "rk45"

    #: The one-word family, read by :attr:`SystemBase.family`.
    _family: ClassVar[str] = "dde"

    #: A delay system's right-hand side is not a function of one state vector.
    jacobian = Absent(
        "a delay system's right-hand side reads the state at several past times, "
        "so d f/d u at one point is not defined without those",
        "ts.analysis.lyapunov_spectrum(system, k=1, dt=0.5)",
    )

    #: A delay system's kernel is lowered with its delay slots baked in.
    jacobian_sym = Absent(
        "a delay system's right-hand side reads the state at several past times, "
        "so there is no single square Jacobian to hand back",
        "ts.analysis.lyapunov_spectrum(system, k=1, dt=0.5)",
    )

    #: A delay system's state is a whole history function.
    set_state = Absent(
        "a delay system's state is a whole history function on [-tau_max, 0], not "
        "a point, so it cannot be seated from one",
        "mg.reinit(history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)])",
    )

    @property
    def _is_discrete(self) -> bool:
        """DDEs are continuous-time systems."""
        return False

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict[str, Any] | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        **unknown: Any,
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
        reject_unknown_run_keywords(
            self,
            unknown,
            family="dde",
            accepted=("t", "params", "rtol", "atol"),
            verb="reinit",
        )
        if params:
            for k, v in params.items():
                self.params[k] = v
        if t is not None and float(t) != 0.0:
            raise NotImplementedError(
                "DelaySystem.reinit only supports t=0 (the past starts there)."
            )
        # ``resolve_ic`` commits the resolved IC to ``self.ic`` before anything
        # else, so a malformed ``u`` must leave the object exactly as it was —
        # the same contract ``integrate`` gets from ``_dispatch``.
        with self._ic_rollback():
            past_ic = self._resolve_ic(u)
        self._past_ic = past_ic
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
        traj = self.run(
            final_time=self._t_now,
            dt=min(dt, self._t_now),
            ic=self._past_ic,
            rtol=self._step_rtol if self._step_rtol is not None else self._default_rtol,
            atol=self._step_atol if self._step_atol is not None else self._default_atol,
        )
        state = np.asarray(traj.y[-1], dtype=float)
        if not np.isfinite(state).all():
            raise ConvergenceError(
                f"{type(self).__name__}: DDE diverged at t={self._t_now:.6g} during step()."
            )
        self._state_now = state.copy()
        return state

    def state(self) -> np.ndarray:
        """Return a copy of the current state (implicit ``reinit`` if cold)."""
        if self._state_now is None:
            self.reinit()
        assert self._state_now is not None
        return self._state_now.copy()

    def time(self) -> float:
        """Return the current stepper time."""
        return self._t_now

    # ------------------------------------------------------------------ #
    # Integration
    # ------------------------------------------------------------------ #

    def run(
        self,
        final_time: float = 100.0,
        dt: float | None = None,
        *,
        ic: Any | None = None,
        history: History = None,
        transient: float = 0.0,
        solver: str | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        backend: str | None = None,
        seed: int | None = None,
        **solver_options: Any,
    ) -> Trajectory:
        """
        Integrate the delay system and return a :class:`~tsdynamics.families.Trajectory`.

        ``run`` is **the** trajectory verb — one word on flows, maps, delay and
        stochastic systems.  ``integrate`` and ``trajectory`` were two more names
        for this method and are gone in v6.

        The signature is **closed**.  Before v6 this method accepted — and
        silently dropped — ``max_step``, ``t0``, ``events`` and outright typos:
        the run completed, the number was wrong, and nothing said so.

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
            Integration tolerances.  Default
            :data:`~tsdynamics.utils.tolerances.DDE_RTOL` /
            :data:`~tsdynamics.utils.tolerances.DDE_ATOL` (both ``1e-3``) — the
            DDE family's own, deliberately looser than the ODE default because
            the method of steps lands on every output sample, so ``dt`` bounds
            the step and the tolerance is largely inert.  See the class
            docstring for the measurement.  Tightening is *safe* (no stall) if
            you need it.
        backend : {"jit", "interp"}, optional
            Which evaluator drives the DDE engine.  Defaults to
            ``_default_backend`` (``"jit"``).  Both route — through the shared
            engine seam (:func:`tsdynamics.engine.run.integrate`) — to the Rust
            method-of-steps engine (history ring buffer + cubic-Hermite dense
            interpolation; stream E-DDE), reusing the explicit solver kernels.

            - ``"jit"`` (default) — the Cranelift JIT, compiled once per distinct
              system and served from the process-wide compiled-evaluator cache.
            - ``"interp"`` — the SSA-tape interpreter; bit-for-bit identical, and
              the way to avoid the one-off compile.

            Only **constant** delays lower; a state-dependent delay raises.
            ``backend="reference"`` is unsupported for DDEs (there is no
            pure-Python delay integrator).

            .. versionchanged:: 6.0
               The default moved from ``"interp"`` to ``"jit"``, once the v6
               compiled-evaluator cache removed the JIT's per-call recompile.
        solver : str, default "rk45"
            The explicit kernel (``"rk45"``, ``"tsit5"``, ``"dop853"``,
            ``"rk4"``); the method of steps drives explicit kernels only.
            ``"auto"`` is an explicit **no-op** here — it resolves to the DDE
            default (``rk45``) without an auto-stiffness probe.  Auto-stiffness is
            an ODE-only feature: the one-point heuristic reads only the
            instantaneous Jacobian and would ignore the delay terms that shape a
            DDE's spectrum (it could even select an implicit kernel the
            method-of-steps engine cannot drive), so pass another explicit
            ``method=`` directly if you need one.
        seed : int, optional
            Seed for the **random initial-condition draw** (the constant past when
            ``history`` is ``None``) — the same meaning ``seed=`` has on every
            other family's trajectory producer and on the constructor
            (:meth:`SystemBase.ic_generator`).  Inert unless a draw actually
            happens: an explicit ``ic``, an already-resolved ``self.ic`` and a
            class-level ``default_ic`` all take priority.  The resolved seed is
            recorded on ``traj.meta["ic_seed"]``.

            .. versionadded:: 6.0
        transient : float, optional
            Leading stretch of the run to discard, in **time units** (the same
            unit as ``final_time``).  The window is extended to
            ``transient + final_time`` and everything before ``transient``
            dropped.  Spelled identically on every family and every
            trajectory-producing verb.

            .. versionadded:: 6.0

        Returns
        -------
        Trajectory
        """
        reject_unknown_run_keywords(self, solver_options, family="dde", accepted=_DDE_RUN_KEYWORDS)
        dt = self._default_dt if dt is None else dt
        method = self._default_method if solver is None else solver
        transient = resolve_transient(transient, discrete=False)
        if transient > 0.0:
            traj = self.run(
                final_time + transient,
                dt,
                ic=ic,
                history=history,
                rtol=rtol,
                atol=atol,
                backend=backend,
                solver=method,
                seed=seed,
            )
            return traj.after(transient)
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
            seed=seed,
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
        seed: int | None = None,
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

        DDE tolerances default to ``_default_rtol`` / ``_default_atol``
        (:data:`~tsdynamics.utils.tolerances.DDE_RTOL` /
        :data:`~tsdynamics.utils.tolerances.DDE_ATOL`, both ``1e-3``) and are
        resolved here before handing off, so the generic seam's ODE-style
        ``1e-9`` / ``1e-12`` default never reaches a delay system.
        """
        rtol = rtol if rtol is not None else self._default_rtol
        atol = atol if atol is not None else self._default_atol
        return self._dispatch(
            backend=backend,
            seed=seed,
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

    def _lyapunov_spectrum(
        self,
        final_time: float = 200.0,
        dt: float = 0.1,
        *,
        ic: Any | None = None,
        k: int = 1,
        transient: float = 50.0,
        rtol: float | None = None,
        atol: float | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Estimate the ``k`` leading Lyapunov exponents of the delay system.

        The **engine** estimator (stream E-DDE-LYAP, result stored in
        the DDE Lyapunov estimator) integrates the extended variational
        DDE on the Rust engine with a function-space Benettin renormalisation
        (:func:`tsdynamics.families._dde_lyapunov.dde_lyapunov_spectrum`):
        ``backend="jit"`` (the default) / ``"interp"``.  ``"reference"`` is
        rejected (the engine has no pure-Python DDE integrator).

        Parameters
        ----------
        final_time : float
            Averaging window after burn-in. Default 200.0.
        dt : float
            Sampling interval (should divide the maximum delay).
        ic : array-like, optional
            Initial state. Provide the end-state of a prior ``integrate``
            call so the trajectory starts on the attractor (recommended).
        k : int
            Number of leading exponents to estimate. DDEs have infinitely
            many; choose consciously. Default 1.
        transient : float
            Discard this much time before averaging, in **time units**. Default
            50.0.  Spelled ``transient`` on every entry point in the library —
            it was ``burn_in`` here until v6.
        rtol, atol : float, optional
            Integration tolerances.  The engine path renormalises every delay
            window and defaults to
            :data:`~tsdynamics.utils.tolerances.DDE_LYAPUNOV_RTOL` /
            :data:`~tsdynamics.utils.tolerances.DDE_LYAPUNOV_ATOL` (``1e-7`` /
            ``1e-9``) — tighter than plain DDE integration, looser than the ODE
            default, and measured to agree with ``1e-9``/``1e-12`` to within the
            estimator's own finite-time scatter on all six built-in DDEs.
        backend : {"jit", "interp"}, optional
            Which evaluator drives the engine.  Defaults to
            :attr:`_default_backend` (``"jit"``, the Cranelift JIT);
            ``"interp"`` is the bit-for-bit identical SSA-tape interpreter.

            .. versionchanged:: 6.0
               Default moved from ``"interp"`` to ``"jit"`` (see
               :meth:`integrate`).

        Notes
        -----
        For best results, pass ``ic=traj.y[-1]`` from a prior ``integrate`` run —
        this places the trajectory on the attractor and avoids trivial exponents
        from equilibrium pasts.

        Returns
        -------
        ndarray, shape (k,)
        """
        backend = backend if backend is not None else self._default_backend
        from tsdynamics.families._dde_lyapunov import dde_lyapunov_spectrum

        if kwargs:
            raise InvalidInputError(
                f"lyapunov_spectrum(backend={backend!r}) does not accept the "
                f"extra integration keyword(s) {sorted(kwargs)}."
            )
        exps = dde_lyapunov_spectrum(
            self,
            k=k,
            final_time=final_time,
            dt=dt,
            burn_in=transient,
            ic=ic,
            backend=backend,
            rtol=rtol if rtol is not None else DDE_LYAPUNOV_RTOL,
            atol=atol if atol is not None else DDE_LYAPUNOV_ATOL,
        )
        return exps


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
