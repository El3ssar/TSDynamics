"""DelaySystem — DDE base class via JiTCDDE."""

from __future__ import annotations

import os
import pathlib
import sysconfig
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np

from .base import SystemBase, Trajectory

__all__ = ["DelaySystem"]

# Suppress the expected "target time is smaller than current time" warning
# that JiTCDDE emits during spline evaluation — it is harmless.
warnings.filterwarnings(
    "ignore",
    message=".*target time is smaller than the current time.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Cache directory (shared with ODE module)
# ---------------------------------------------------------------------------

_CACHE_DIR = pathlib.Path(
    os.environ.get("TSDYNAMICS_CACHE", pathlib.Path.home() / ".cache" / "tsdynamics")
)

_EXT_SUFFIX: str = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

# ---------------------------------------------------------------------------
# Type alias for history functions
# ---------------------------------------------------------------------------

History = Callable[[float], Sequence[float]] | None


def _make_t_eval(t0: float, tf: float, dt: float) -> np.ndarray:
    t_arr = np.arange(t0, tf, dt)
    if t_arr.size == 0 or t_arr[-1] < tf - 1e-12:
        t_arr = np.append(t_arr, tf)
    return t_arr


# ---------------------------------------------------------------------------
# DelaySystem
# ---------------------------------------------------------------------------


class DelaySystem(SystemBase, ABC):
    """
    Base class for delay differential systems (DDEs), compiled via JiTCDDE.

    Subclass contract
    -----------------
    1. Declare ``params = {...}`` and ``dim = N``.
    2. Implement ``_equations`` as a ``@staticmethod`` returning a
       length-``dim`` sequence of JiTCDDE symbolic expressions.
       Use ``y(i, t - tau)`` for delayed state access.

    Compilation & caching
    ---------------------
    DDE systems cache compiled modules per ``(class, params_hash)``.  Unlike
    ODEs, delay values directly affect the history-buffer structure and cannot
    easily be treated as runtime control parameters.  The compiled ``.so`` is
    saved to disk so subsequent runs with the same parameters reload instantly.

    DDEs typically need looser tolerances than ODEs (start with ``rtol=atol=1e-3``).

    History
    -------
    Pass a ``history`` callable ``h(s) → sequence`` defining the past for
    ``s ≤ 0``.  If omitted, a constant past equal to ``ic`` is used.

    .. note::
        Provide a non-equilibrium history to avoid trivial Lyapunov exponents.
        For ``lyapunov_spectrum``, ``past_from_function`` is incompatible with
        ``jitcdde_lyap``; the workaround is to run ``integrate`` first with the
        desired history, then pass the end-state as ``ic`` to
        ``lyapunov_spectrum`` which uses ``constant_past`` internally.

    Examples
    --------
    >>> mg = MackeyGlass()
    >>> hist = lambda s: [1.0 + 0.1 * np.sin(0.2 * s)]
    >>> traj = mg.integrate(final_time=500, history=hist)
    >>> exps = mg.lyapunov_spectrum(n_exp=2, ic=traj.y[-1])
    """

    _default_rtol: ClassVar[float] = 1e-3
    _default_atol: ClassVar[float] = 1e-3

    #: Names of parameters that hold delay values (must be positive floats).
    #: Subclasses with custom delay-naming conventions should override this.
    #: The default ``("tau",)`` matches the convention used throughout the
    #: built-in DDE systems.  Override with ``("tau1", "tau2")`` etc. for
    #: multi-delay systems, or override ``_delays()`` for delays computed
    #: from other parameters.
    _delay_params: ClassVar[tuple[str, ...]] = ("tau",)

    # In-process path caches: cache_key (str) → absolute path of compiled .so.
    _compiled_ddes: ClassVar[dict[str, str]] = {}
    _compiled_lyap: ClassVar[dict[str, str]] = {}

    # Protocol stepping state (instances shadow these class defaults).
    _stepper: Any = None
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
        y : JiTCDDE ``y``-accessor.
            ``y(i)`` for current state component ``i``;
            ``y(i, t - tau)`` for delayed access.
        t : JiTCDDE time symbol.
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

    def _module_path(self) -> pathlib.Path:
        return _CACHE_DIR / f"tsdyn_dde_{self._cache_key()}"

    def _ensure_compiled(self, for_lyap: bool = False, n_lyap: int = 0) -> pathlib.Path:
        """
        Ensure the compiled JiTCDDE module exists on disk and return its path.

        DDE objects are never cached in-process — every integration call needs a
        fresh instance with its own past-buffer.  Only the ``.so`` path is cached.

        Lookup order
        ------------
        1. In-process path cache.
        2. Disk cache (``_CACHE_DIR``).
        3. Check ``sys.modules`` for name collision from an interrupted prior
           compilation, recovering from the temp dir or using a unique name.
        4. Fresh compilation.
        """
        import shutil
        import sys

        from jitcdde import jitcdde as _jitcdde
        from jitcdde import jitcdde_lyap as _jitcdde_lyap
        from jitcdde import t as t_sym
        from jitcdde import y
        from jitcxde_common.modules import modulename_from_path

        cls_jitc = _jitcdde_lyap if for_lyap else _jitcdde
        lyap_kw = {"n_lyap": n_lyap} if for_lyap else {}
        cache = type(self)._compiled_lyap if for_lyap else type(self)._compiled_ddes
        so_suffix = f"_lyap{n_lyap}" if for_lyap else ""
        dest_path = pathlib.Path(str(self._module_path()) + so_suffix)
        cache_key = str(dest_path)

        def _find_so(base: pathlib.Path) -> pathlib.Path | None:
            # save_compiled writes plain "<name>.so"; interrupted-compile
            # recovery writes the full EXT_SUFFIX form — accept both.
            hits = [
                f
                for f in _CACHE_DIR.glob(f"{base.name}.*")
                if (f.name.endswith(_EXT_SUFFIX) or f.name.endswith(".so")) and f.stat().st_size > 0
            ]
            return hits[0] if hits else None

        # 1. In-process path cache
        cached = cache.get(cache_key)
        if cached and pathlib.Path(cached).exists():
            return pathlib.Path(cached)
        if cached:
            del cache[cache_key]  # stale — clear it

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 2. Disk cache
        so = _find_so(dest_path)
        if so:
            cache[cache_key] = str(so)
            return so

        # 3. Check for name collision from a prior interrupted compilation.
        #    JiTCDDE's save_compiled calls compile_C internally, which registers
        #    the module name in sys.modules before the .so reaches disk.
        #    Recover from the temp dir if it's still alive, or redirect to a
        #    unique per-process name to avoid NameError.
        dest = dest_path
        mname = modulename_from_path(str(dest_path))
        if mname in sys.modules:
            live_so = getattr(sys.modules[mname], "__file__", None)
            if live_so and pathlib.Path(live_so).exists():
                so = pathlib.Path(str(dest_path) + _EXT_SUFFIX)
                try:
                    shutil.copy(live_so, so)
                except OSError:
                    so = pathlib.Path(live_so)
                cache[cache_key] = str(so)
                return so
            dest = pathlib.Path(f"{dest_path}_{os.getpid()}")

        # 4. Fresh compilation
        rhs = list(type(self)._equations(y, t_sym, **self.params.as_dict()))
        if len(rhs) != self.dim:
            raise ValueError(f"_equations must return {self.dim} expressions, got {len(rhs)}")

        dde = cls_jitc(rhs, verbose=False, **lyap_kw)
        dde.compile_C()
        so = pathlib.Path(dde.save_compiled(destination=str(dest), overwrite=True))
        cache[cache_key] = str(so)
        return so

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

        JiTCDDE requires ``max_delay >= max(delays)``; we add a 1% margin to
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
        """
        from jitcdde import jitcdde as _jitcdde

        if params:
            for k, v in params.items():
                self.params[k] = v
        if t is not None and float(t) != 0.0:
            raise NotImplementedError(
                "DelaySystem.reinit only supports t=0 — JiTCDDE pasts start there."
            )
        ic_arr = self.resolve_ic(u)
        so_path = self._ensure_compiled(for_lyap=False)

        stepper = _jitcdde(
            module_location=str(so_path),
            n=self.dim,
            delays=self._delays(),
            max_delay=self._max_delay(),
            verbose=False,
        )
        stepper.constant_past(ic_arr)
        stepper.set_integration_parameters(
            rtol=rtol if rtol is not None else self._default_rtol,
            atol=atol if atol is not None else self._default_atol,
            **kwargs,
        )
        stepper.initial_discontinuities_handled = True

        self._stepper = stepper
        self._state_now = ic_arr.copy()
        self._t_now = float(stepper.t)

    def step(self, n_or_dt: float | None = None) -> np.ndarray:
        """Advance by ``dt`` (default 0.1, forward-only) and return the new state."""
        if self._stepper is None:
            self.reinit()
        dt = float(n_or_dt) if n_or_dt is not None else self._default_step_dt
        self._t_now = self._t_now + dt
        state = np.asarray(self._stepper.integrate(self._t_now), dtype=float)
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
        backend: str = "jitcdde",
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
        backend : str, default "jitcdde"
            Which engine integrates the DDE.  ``"jitcdde"`` (the default) is the
            v2 compiled backend.  ``"interp"`` / ``"jit"`` route to the Rust
            method-of-steps engine (history ring buffer + cubic-Hermite dense
            interpolation; stream E-DDE), reusing the explicit solver kernels.
            The Rust path supports **constant** delays; a state-dependent delay
            raises (use ``"jitcdde"`` for those until the lowering grows dynamic
            delay slots).  ``**kwargs`` are ignored by the engine backends.
        method : str, default "rk45"
            The explicit kernel for the engine backends (``"rk45"``, ``"tsit5"``,
            ``"dop853"``, ``"rk4"``); ignored when ``backend="jitcdde"``.
        **kwargs
            Forwarded to ``jitcdde.set_integration_parameters``
            (e.g. ``max_step``, ``first_step``) on the ``"jitcdde"`` backend.

        Returns
        -------
        Trajectory
        """
        if backend not in (None, "jitcdde", "v2"):
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

        from jitcdde import jitcdde as _jitcdde

        rtol = rtol if rtol is not None else self._default_rtol
        atol = atol if atol is not None else self._default_atol

        ic_arr = self.resolve_ic(ic)
        t_eval = _make_t_eval(0.0, final_time, dt)
        so_path = self._ensure_compiled(for_lyap=False)
        max_delay = self._max_delay()

        # Fresh jitcdde instance each call (DDE objects are stateful —
        # they own a past-buffer that must start clean every integration).
        dde = _jitcdde(
            module_location=str(so_path),
            n=self.dim,
            delays=self._delays(),
            max_delay=max_delay,
            verbose=False,
        )

        # Set past
        if history is None:
            dde.constant_past(ic_arr)
            y0 = ic_arr.copy()
        else:

            def hist_fn(s):
                return np.asarray(history(s), dtype=float).reshape(self.dim)

            dde.past_from_function(hist_fn)
            y0 = hist_fn(0.0)

        dde.set_integration_parameters(rtol=rtol, atol=atol, **kwargs)
        dde.initial_discontinuities_handled = True

        y_out = np.empty((t_eval.size, self.dim), dtype=float)
        y_out[0] = y0
        for k in range(1, t_eval.size):
            y_out[k] = dde.integrate(float(t_eval[k]))

        return Trajectory(
            t=t_eval,
            y=y_out,
            system=self,
            meta=self._provenance(
                family="dde",
                dt=dt,
                rtol=rtol,
                atol=atol,
                ic=ic_arr.copy(),
                history="callable" if history is not None else "constant",
            ),
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

        The delay system lowers (via :func:`tsdynamics.engine.compile.lower_dde`)
        to a tape over ``dim + n_slots`` inputs whose extra inputs are the delay
        slots; the engine fills those from a history buffer it interpolates with
        cubic Hermite, reusing the explicit solver kernels for each step.  Only
        constant delays lower; a state-dependent delay raises ``TapeCompileError``
        (use ``backend="jitcdde"``).
        """
        from tsdynamics.engine.problem import dde_problem
        from tsdynamics.engine.run import resolve_backend

        backend = resolve_backend(backend)
        if backend == "reference":
            raise NotImplementedError(
                f"{type(self).__name__}: backend='reference' has no DDE integrator; use "
                f"backend='interp' (the Rust engine) or backend='jitcdde' (the v2 backend)."
            )
        try:
            from tsdynamics import _rust
        except ImportError as err:
            from tsdynamics.engine.run import EngineNotAvailableError

            raise EngineNotAvailableError(
                "the Rust engine extension (tsdynamics._rust, stream E7) is not built; use "
                "backend='jitcdde' for the v2 DDE backend."
            ) from err

        rtol = rtol if rtol is not None else self._default_rtol
        atol = atol if atol is not None else self._default_atol

        problem = dde_problem(self, ic=ic)
        dim = problem.dim
        max_delay = problem.max_delay
        t_eval = _make_t_eval(0.0, final_time, dt)

        # The t0 state and the past on [-max_delay, 0]; a constant past (no
        # history callable) is a single sample that the engine treats as constant.
        past_t, past_y, ic0 = self._sample_past(history, problem.ic, dim, max_delay, dt)

        slot_components = np.array([s.component for s in problem.delay_slots], dtype=np.int32)
        slot_delays = np.array([s.delay for s in problem.delay_slots], dtype=np.float64)

        y = np.asarray(
            _rust.integrate_dde_dense(
                *problem.tape.to_arrays(),
                slot_components,
                slot_delays,
                np.ascontiguousarray(ic0, dtype=np.float64),
                np.ascontiguousarray(past_t, dtype=np.float64),
                np.ascontiguousarray(past_y.ravel(), dtype=np.float64),
                np.ascontiguousarray(t_eval, dtype=np.float64),
                method,
                float(rtol),
                float(atol),
                backend == "jit",
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(y)):
            raise RuntimeError(
                f"{type(self).__name__}: DDE integration diverged or the step collapsed before "
                f"reaching the final time (backend={backend!r}, method={method!r})."
            )

        return Trajectory(
            t=t_eval,
            y=y,
            system=self,
            meta=self._provenance(
                family="dde",
                engine="rust",
                backend=backend,
                method=method,
                dt=dt,
                rtol=rtol,
                atol=atol,
                ic=np.asarray(ic0, dtype=float).copy(),
                history="callable" if history is not None else "constant",
            ),
        )

    def _sample_past(
        self,
        history: History,
        ic_arr: np.ndarray,
        dim: int,
        max_delay: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the past on ``[-max_delay, 0]``, returning ``(past_t, past_y, ic0)``.

        A ``None`` history is a constant past equal to ``ic_arr`` — a single
        sample the engine reads as constant.  A callable is sampled densely (so
        the engine's finite-difference Hermite tangents stay accurate), with the
        ``t0`` state taken as ``history(0)``.
        """
        if history is None:
            past_t = np.array([0.0], dtype=float)
            past_y = np.asarray(ic_arr, dtype=float).reshape(1, dim)
            return past_t, past_y, np.asarray(ic_arr, dtype=float)

        # Dense sampling: step no coarser than the output dt and well below the
        # delay, capped so a tiny delay or dt cannot explode the sample count.
        step = min(dt, max_delay / 200.0) if max_delay > 0.0 else dt
        step = max(step, 1e-9)
        n = max(2, int(np.ceil(max_delay / step)) + 1)
        n = min(n, 200_000)
        past_t = np.linspace(-max_delay, 0.0, n)
        past_y = np.array(
            [np.asarray(history(s), dtype=float).reshape(dim) for s in past_t],
            dtype=float,
        )
        return past_t, past_y, past_y[-1].copy()

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
        **kwargs,
    ) -> np.ndarray:
        """
        Estimate the ``n_exp`` leading Lyapunov exponents via :class:`jitcdde.jitcdde_lyap`.

        Results are stored in ``self.meta['lyapunov_spectrum']``.

        Parameters
        ----------
        final_time : float
            Averaging window after burn-in. Default 200.0.
        dt : float
            Sampling interval. Default 0.1.
        ic : array-like, optional
            Initial state. Provide the end-state of a prior ``integrate``
            call so the trajectory starts on the attractor (recommended).
        n_exp : int
            Number of leading exponents to estimate. DDEs have infinitely
            many; choose consciously. Default 1.
        burn_in : float
            Discard interval. Default 50.0.
        rtol, atol : float, optional
            Integration tolerances.  Defaults to ``_default_rtol`` /
            ``_default_atol`` (both ``1e-3``).  Do **not** use ODE-style
            tight values (e.g. ``1e-6``/``1e-9``) — DDE solvers are stiff and
            tight tolerances cause the variational equations to accumulate
            floating-point garbage before the first Lyapunov renormalisation,
            producing ``Inf`` / ``NaN`` exponents.

        Notes
        -----
        ``past_from_function`` is NOT used here because it is incompatible
        with ``jitcdde_lyap`` internally.  A constant past from ``ic`` is
        used instead.  For best results, pass ``ic=traj.y[-1]`` from a
        prior ``integrate`` run — this places the trajectory on the
        attractor and avoids trivial exponents from equilibrium pasts.

        Returns
        -------
        ndarray, shape (n_exp,)
        """
        from jitcdde import jitcdde_lyap as _jitcdde_lyap

        rtol = rtol if rtol is not None else self._default_rtol
        atol = atol if atol is not None else self._default_atol

        ic_arr = self.resolve_ic(ic)
        so_path = self._ensure_compiled(for_lyap=True, n_lyap=n_exp)
        max_delay = self._max_delay()

        dde = _jitcdde_lyap(
            module_location=str(so_path),
            n=self.dim,
            delays=self._delays(),
            max_delay=max_delay,
            n_lyap=n_exp,
            verbose=False,
        )
        dde.set_integration_parameters(rtol=rtol, atol=atol, **kwargs)
        dde.constant_past(ic_arr)
        dde.step_on_discontinuities()

        # Burn-in
        T_burn = float(dde.t) + max(0.0, burn_in)
        while dde.t < T_burn:
            dde.integrate(min(T_burn, dde.t + dt))

        # Production: weight-averaged local LEs
        T_end = float(dde.t) + final_time
        weights = []
        ly_steps = []

        while dde.t < T_end:
            ret = dde.integrate(min(T_end, dde.t + dt))
            if not isinstance(ret, tuple):
                raise RuntimeError("jitcdde_lyap.integrate did not return a tuple")
            _, local_lyaps, w = ret
            v = np.asarray(local_lyaps, float).ravel()
            if v.size != n_exp:
                raise ValueError(f"Expected {n_exp} local LEs, got {v.shape}")
            weights.append(float(w))
            ly_steps.append(v)

        W = np.asarray(weights, float)
        mask = W > 0.0
        if not mask.any():
            warnings.warn(
                f"{type(self).__name__}.lyapunov_spectrum: every integration "
                f"step returned zero weight from jitcdde_lyap. This means the "
                f"sampling step `dt={dt}` is smaller than the internal step "
                f"size, so no Lyapunov estimates were accumulated. Increase "
                f"`dt` (typical: 0.1–1.0 for DDEs) and rerun.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros(n_exp)

        L = np.vstack([ly_steps[i] for i, m in enumerate(mask) if m])
        exponents = (W[mask, None] * L).sum(0) / W[mask].sum()

        self.meta.record(
            "lyapunov_spectrum",
            exponents,
            dt=dt,
            final_time=final_time,
            burn_in=burn_in,
            n_exp=n_exp,
        )
        return exponents
