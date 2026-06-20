"""ContinuousSystem — ODE base class on the Rust engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from .base import SystemBase, Trajectory


def _resolve_derivative_nodes(expr):
    """
    Replace unevaluated SymEngine ``Derivative`` nodes with a.e. derivatives.

    SymEngine leaves ``d|u|/du``, ``d sign(u)/du`` and ``d floor(u)/du`` /
    ``d ceil(u)/du`` unevaluated (the last two wrapped in a ``Subs``), which
    ``Lambdify`` and the tape emitter cannot compile.  Almost everywhere,
    ``d sign(u)/d· = 0``, ``d|u|/du = sign(u)`` and ``d floor(u)/d· =
    d ceil(u)/d· = 0`` (floor/ceil are piecewise-constant) — the measure-zero
    kink is irrelevant for numeric Jacobian evaluation along an orbit.

    Works directly on the SymEngine tree: these nodes may differentiate with
    respect to *expressions* (chain-rule dummies), which cannot round-trip
    through SymPy at all.
    """
    import symengine

    s = str(expr)
    if "Derivative" not in s and "Subs" not in s:
        return expr

    def walk(e):
        name = type(e).__name__
        if name == "Derivative":
            target = e.args[0]
            wrt = e.args[1]
            tname = type(target).__name__
            if tname in ("sign", "floor", "ceiling"):
                # Piecewise-constant a.e.: derivative is zero off the kinks.
                return symengine.Integer(0)
            if tname == "Abs":
                g = target.args[0]
                if g == wrt:
                    return symengine.sign(g)
                try:  # wrt may be a symbol or a y(i) application — diff handles both
                    return symengine.sign(g) * walk(g.diff(wrt))
                except RuntimeError:
                    return e
            return e  # unknown derivative — leave it; Lambdify will fail loudly
        if name == "Subs":
            # ``Subs(Derivative(floor(ξ), ξ), ξ, g)`` — the chain-rule form of a
            # floor/ceil derivative.  If the substituted-into expression resolves
            # to a constant, the substitution is that constant.
            inner = walk(e.args[0])
            if not inner.free_symbols and inner == symengine.Integer(0):
                return symengine.Integer(0)
            if inner is e.args[0]:
                return e
            try:
                return e.func(inner, *e.args[1:])
            except (TypeError, RuntimeError):
                return e
        if name == "Piecewise":
            # ``args`` is flattened ``(expr0, cond0, expr1, cond1, …)`` and a
            # Piecewise cannot be rebuilt via ``func(*args)`` — re-pair the
            # resolved expressions with their (unchanged) conditions.
            args = e.args
            new_args = [walk(a) for a in args]
            if all(na is a for na, a in zip(new_args, args, strict=True)):
                return e
            pairs = [(new_args[i], new_args[i + 1]) for i in range(0, len(new_args), 2)]
            return symengine.Piecewise(*pairs)
        args = e.args
        if not args:
            return e
        new_args = [walk(a) for a in args]
        if all(na is a for na, a in zip(new_args, args, strict=True)):
            return e
        try:
            return e.func(*new_args)
        except (TypeError, RuntimeError):
            # Some node types (e.g. Piecewise) flatten their args and cannot
            # be reconstructed via func(*args) — keep the original node.
            return e

    return walk(symengine.sympify(expr))


# ---------------------------------------------------------------------------
# ContinuousSystem
# ---------------------------------------------------------------------------


class ContinuousSystem(SystemBase, ABC):
    """
    Base class for ODE-based dynamical systems, integrated on the engine.

    Subclass contract
    -----------------
    1. Declare ``params = {...}`` and ``dim = N`` at class level.
    2. Implement ``_equations`` as a ``@staticmethod`` returning a
       length-``dim`` sequence of SymEngine symbolic expressions.
    3. Optionally mark integer or loop-structural parameters in
       ``_structural_params`` — these are baked into the lowered tape
       rather than exposed as runtime control parameters.

    Lowering
    --------
    Each system is lowered once to an in-process IR tape with no warmup; the
    engine reads non-structural parameters live from the system on every run,
    so a parameter change never triggers a re-lowering.

    Class-level attributes
    ----------------------
    _structural_params : frozenset[str]
        Parameter names that appear as integer loop bounds or affect the
        symbolic structure of ``_equations``.  These are baked in at compile
        time.  For most systems this is empty (the default).

        Example — Lorenz96 uses ``N`` to build the list comprehension::

            _structural_params = frozenset({"N"})

    _default_method : str
        Default integrator name (default ``"RK45"``).

    Examples
    --------
    >>> lor = Lorenz()
    >>> traj = lor.integrate(final_time=100, dt=0.01)
    >>> t, y = traj          # tuple-unpack
    >>> lor.sigma = 15.0     # change param — zero recompile cost
    >>> traj2 = lor.integrate(final_time=100)
    """

    _default_method: ClassVar[str] = "RK45"

    #: The default runtime backend (see :attr:`SystemBase._default_backend`).
    #: ``"interp"`` — the zero-warmup Rust engine interpreter (the sole engine
    #: since the M3 migration retired the v2 backends).
    _default_backend: ClassVar[str] = "interp"

    #: Parameters whose values affect the symbolic *structure* of _equations
    #: (e.g. integer loop bounds). These are baked in at lowering time.
    _structural_params: ClassVar[frozenset[str]] = frozenset()

    # Per-class in-process cache:
    # _lambdified : cache_key (str) → (rhs_fn, jac_fn, control_names) —
    #               SymEngine-Lambdified numeric RHS/Jacobian evaluators
    #               (used by figures, Poincaré Hermite refinement and analyses),
    #               keyed by class + dim + structural params.
    _lambdified: ClassVar[dict[str, tuple]] = {}

    # Protocol stepping state (instances shadow these class defaults on first
    # ``reinit``).  The engine lowers the tape once in ``reinit`` and ``step``
    # reuses it with the current state as the initial condition.
    _engine_problem: Any = None
    _state_now: np.ndarray | None = None
    _t_now: float = 0.0
    _default_step_dt: ClassVar[float] = 0.01

    # Per-step integration context cached by ``reinit`` so a repeated constant-``dt``
    # stepping loop pays the fixed per-call overhead (solver-registry resolve,
    # Jacobian decision, output-grid build, provenance, Trajectory wrap) once at
    # reinit instead of on every ``step`` (stream WS-STEPBUF).
    _step_method_canonical: str | None = None

    # ------------------------------------------------------------------ #
    # Subclass interface
    # ------------------------------------------------------------------ #

    @staticmethod
    @abstractmethod
    def _equations(y, t, **params) -> Sequence:
        """
        Build the symbolic RHS.

        Parameters
        ----------
        y : symbolic state accessor — call ``y(i)`` for state component ``i``.
        t : symbolic time variable.
        **params
            Current parameter values.  For non-structural params these are
            SymEngine symbols during lowering and float values during any
            Python-fallback evaluation.

        Returns
        -------
        Sequence of ``dim`` SymEngine expressions.

        Notes
        -----
        Use only symbolic / arithmetic operations.  No NumPy, no ``math``,
        no Python ``if``.  Periodic index: ``y((i+1) % N)``.
        """
        ...

    # ------------------------------------------------------------------ #
    # Internal compilation helpers
    # ------------------------------------------------------------------ #

    def _structural_vals(self) -> dict[str, Any]:
        """Return the structural parameter key→value pairs (baked in)."""
        return {k: self.params[k] for k in type(self)._structural_params}

    def _equations_hash(self) -> str:
        """
        Content hash of the RHS definition, part of every compile-cache key.

        Without it, two same-named classes (user shadowing a builtin, or a
        notebook cell redefining a class with edited equations) would silently
        reuse each other's compiled dynamics.
        """
        import hashlib
        import inspect

        fn = type(self)._equations
        fn = getattr(fn, "__func__", fn)
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):  # dynamically defined without source
            src = repr(getattr(fn, "__code__", fn).co_code)
        return hashlib.md5(src.encode()).hexdigest()[:8]

    def _control_params(self) -> dict[str, Any]:
        """Return the non-structural parameters (the engine's live control parameters)."""
        structural = type(self)._structural_params
        return {k: v for k, v in self.params.items() if k not in structural}

    def _cache_key(self) -> str:
        """Stable in-process cache key (class name + dim + structural + RHS hash).

        Keys the per-class numeric-evaluator cache (:meth:`_build_lambdified`).
        Changing non-structural params does NOT change the key — those become
        runtime control parameters.
        """
        import hashlib
        import json

        eq = self._equations_hash()
        struct_vals = self._structural_vals()
        if struct_vals:
            h = hashlib.md5(
                json.dumps(sorted(struct_vals.items()), default=str).encode()
            ).hexdigest()[:16]
            return f"tsdyn_{type(self).__name__}_{self.dim}_{h}_{eq}"
        return f"tsdyn_{type(self).__name__}_{self.dim}_{eq}"

    # ------------------------------------------------------------------ #
    # System protocol — incremental stepping
    # ------------------------------------------------------------------ #

    @property
    def is_discrete(self) -> bool:
        """ODEs are continuous-time systems."""
        return False

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict | None = None,
        method: str | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        backend: str | None = None,
    ) -> None:
        """
        (Re)start the incremental stepper from state ``u`` at time ``t``.

        Parameters
        ----------
        u : array-like, optional
            Initial state (falls back to ``self.ic``, then random).
        t : float, optional
            Start time (default 0.0).
        params : dict, optional
            Parameter overrides applied (in place) before restarting.
        method, rtol, atol, backend
            Stepper configuration, as in :meth:`integrate`.
        """
        if params:
            for k, v in params.items():
                self.params[k] = v
        t0 = float(t) if t is not None else 0.0
        ic_arr = self.resolve_ic(u)
        be = backend if backend is not None else self._default_backend
        self._step_backend = be if be in ("interp", "jit") else "interp"

        from tsdynamics import solvers
        from tsdynamics.engine.problem import ode_problem

        # Lower the tape ONCE here and reuse it for every step() — a sweep reinits
        # thousands of times, so re-lowering per step would dominate the cost.
        # Resolve the step method first so an implicit kernel (bdf/rosenbrock/
        # trbdf2) gets a Jacobian-carrying tape — step() reuses this exact tape,
        # so the Jacobian must be baked in here or the engine refuses the step.
        self._step_method = method or self._default_method
        resolution = solvers.resolve(self._step_method)
        # Cache the canonical kernel name (``"RK45"`` → ``"rk45"``) so the per-step
        # loop hits the engine core directly without re-resolving every call.
        self._step_method_canonical = resolution.name
        self._engine_problem = ode_problem(self, ic=ic_arr, t0=t0, **resolution.build_kwargs)
        self._step_rtol = float(rtol)
        self._step_atol = float(atol)
        self._state_now = ic_arr.copy()
        self._t_now = t0

    def step(self, n_or_dt: float | None = None) -> np.ndarray:
        """
        Advance the system by ``dt`` (default 0.01) and return the new state.

        The first call performs an implicit :meth:`reinit`.  Parameter changes
        made after ``reinit`` take effect on the next ``reinit``, not on a
        live stepper.

        Notes
        -----
        Each call integrates **exactly one** ``dt`` from the live ``(state, t)``,
        returning byte-for-byte the trajectory the released per-``dt`` path
        produced — there is no batching, so the numbers are unchanged for every
        method, adaptive or fixed-step (stream WS-STEPBUF).  The speedup comes
        only from skipping fixed per-call overhead: ``step`` reuses the tape and
        canonical kernel name cached by :meth:`reinit` and calls the engine's lean
        dense-output core (:func:`~tsdynamics.engine.run._run_continuous`)
        directly, so a constant-``dt`` stepping loop (Poincaré refinement, basins
        over flows) skips the solver-registry resolve, the implicit-Jacobian
        decision, provenance assembly and the :class:`Trajectory` wrap that the
        full :meth:`integrate` entry point pays on every call.

        A batch-ahead variant that integrated a whole chunk in one engine call was
        rejected: a chunked adaptive integration is *not* equal to N single-``dt``
        integrations (the adaptive controller carries its step-size/error state
        across output nodes), which silently corrupted sensitive consumers such as
        ``max_lyapunov``.  The durable amortisation is the resumable engine stepper
        (WS-STEPPER), which advances without re-seeding and stays answer-exact.
        """
        from tsdynamics.engine.problem import ODEProblem
        from tsdynamics.engine.run import _run_continuous
        from tsdynamics.utils.grids import make_output_grid

        if self._engine_problem is None:
            self.reinit()
        dt = float(n_or_dt) if n_or_dt is not None else self._default_step_dt

        t0 = self._t_now
        # The same two-node output grid the full ``integrate`` would build for a
        # single ``dt`` span — so ``step`` is bit-identical to a per-``dt``
        # ``integrate`` from the live state.
        t_eval = make_output_grid(t0, t0 + dt, dt)
        prob = ODEProblem(tape=self._engine_problem.tape, ic=self._state_now, t0=t0, system=self)
        y = _run_continuous(
            prob,
            t_eval,
            method=self._step_method_canonical,
            rtol=self._step_rtol,
            atol=self._step_atol,
            backend=self._step_backend,
        )
        state = np.asarray(y[-1], dtype=float)
        self._t_now = t0 + dt
        self._state_now = state.copy()
        return state.copy()

    def state(self) -> np.ndarray:
        """Return a copy of the current state (implicit ``reinit`` if cold)."""
        if self._state_now is None:
            self.reinit()
        return self._state_now.copy()

    def set_state(self, u: Any) -> None:
        """Overwrite the current state without changing the current time."""
        u_arr = np.asarray(u, dtype=float).reshape(self.dim)
        if self._engine_problem is None:
            self.reinit(u_arr)
        else:
            self._state_now = u_arr.copy()

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
    # Symbolic Jacobian autogeneration + numeric RHS
    # ------------------------------------------------------------------ #

    def jacobian_sym(self) -> list[list[Any]]:
        """
        Return the symbolic Jacobian of ``_equations``, differentiated by SymEngine.

        Rows are ``d f_i / d y(j)`` for the *current* structural parameters;
        non-structural parameters appear as symbols.  Hand-written
        ``_jacobian`` methods on system classes are never used at runtime —
        this autogenerated form is the single source of truth (the test suite
        cross-checks hand-written ones against it).

        Returns
        -------
        list of ``dim`` rows, each a list of ``dim`` SymEngine expressions.
        """
        import symengine

        from tsdynamics.engine.symbols import state_time_symbols

        y, t_sym = state_time_symbols()

        struct_vals = self._structural_vals()
        control_syms = {k: symengine.Symbol(k) for k in self._control_params()}
        f_sym = list(type(self)._equations(y, t_sym, **{**struct_vals, **control_syms}))
        if len(f_sym) != self.dim:
            raise ValueError(f"_equations must return {self.dim} expressions, got {len(f_sym)}")
        return [
            [_resolve_derivative_nodes(symengine.sympify(fi).diff(y(j))) for j in range(self.dim)]
            for fi in f_sym
        ]

    def _build_lambdified(self) -> tuple[Any, Any, list[str]]:
        """
        Build (and cache) SymEngine-Lambdified numeric RHS and Jacobian.

        Both take a flat argument vector ``[y_0..y_{dim-1}, t, *control_params]``.
        Cached per (class, dim, structural-hash) — parameter value changes
        need no rebuild because control params are call-time arguments.
        """
        key = self._cache_key()
        cached = type(self)._lambdified.get(key)
        if cached is not None:
            return cached

        import symengine

        from tsdynamics.engine.symbols import state_time_symbols

        y, t_sym = state_time_symbols()

        struct_vals = self._structural_vals()
        control_names = list(self._control_params())
        control_syms = {k: symengine.Symbol(k) for k in control_names}
        f_sym = [
            symengine.sympify(e)
            for e in type(self)._equations(y, t_sym, **{**struct_vals, **control_syms})
        ]
        jac_rows = [
            [_resolve_derivative_nodes(fi.diff(y(j))) for j in range(self.dim)] for fi in f_sym
        ]

        # Lambdify needs plain symbols — swap the y(i) function applications out.
        y_syms = [symengine.Symbol(f"y_{i}") for i in range(self.dim)]
        subs = {y(i): y_syms[i] for i in range(self.dim)}
        args = [*y_syms, t_sym, *(control_syms[k] for k in control_names)]
        rhs_fn = symengine.Lambdify(args, [e.subs(subs) for e in f_sym])
        jac_fn = symengine.Lambdify(args, [e.subs(subs) for row in jac_rows for e in row])

        entry = (rhs_fn, jac_fn, control_names)
        type(self)._lambdified[key] = entry
        return entry

    def jacobian(self, u: Any, t: float = 0.0) -> np.ndarray:
        """
        Evaluate the (autogenerated) Jacobian numerically at state ``u``.

        Parameters
        ----------
        u : array-like, shape (dim,)
            State at which to evaluate.
        t : float
            Time (matters only for non-autonomous systems).

        Returns
        -------
        ndarray, shape (dim, dim)
        """
        _, jac_fn, control_names = self._build_lambdified()
        vals = [float(self.params[k]) for k in control_names]
        arg = np.concatenate([np.asarray(u, dtype=float).ravel(), [t], vals])
        return np.asarray(jac_fn(arg), dtype=float).reshape(self.dim, self.dim)

    def _rhs_numeric(self):
        """
        Return a fast numeric RHS callable ``f(u, t) -> ndarray``.

        Parameter values are captured at call time of this method; build a
        fresh callable after changing parameters.  Used by figure tooling,
        Poincaré crossing refinement, and backend cross-validation — the
        engine remains the integrator of record.
        """
        rhs_fn, _, control_names = self._build_lambdified()
        vals = np.array([float(self.params[k]) for k in control_names])

        def rhs(u: Any, t: float = 0.0) -> np.ndarray:
            arg = np.concatenate([np.asarray(u, dtype=float).ravel(), [t], vals])
            return np.asarray(rhs_fn(arg), dtype=float).ravel()

        return rhs

    # ------------------------------------------------------------------ #
    # Trajectory production — the canonical ``run`` verb
    # ------------------------------------------------------------------ #

    def run(
        self,
        final_time: float = 100.0,
        dt: float = 0.02,
        **kwargs,
    ) -> Trajectory:
        """
        Produce a trajectory — the one canonical verb for every family.

        ``run`` is the unified trajectory producer: it answers the same call for
        flows, maps, DDEs and SDEs, dispatching on :attr:`is_discrete`.  For a
        continuous-time system (this family) it integrates the flow, so
        ``run`` is a thin alias of :meth:`integrate` and forwards every keyword
        to it unchanged.

        Parameters
        ----------
        final_time : float
            End of the integration window. Default 100.0.
        dt : float
            Output sampling interval. The internal stepper is adaptive.
        **kwargs
            Forwarded verbatim to :meth:`integrate` (``t0``, ``ic``, ``method``,
            ``rtol``, ``atol``, ``backend``, …).

        Returns
        -------
        Trajectory
            Identical to :meth:`integrate` — ``run`` adds no behaviour.

        See Also
        --------
        integrate : The family-specific spelling (a permanent alias of ``run``).

        Examples
        --------
        >>> traj = Lorenz().run(final_time=100, dt=0.01)
        >>> Henon().run(n=5000)            # the same verb iterates a map
        """
        return self.integrate(final_time=final_time, dt=dt, **kwargs)

    # ------------------------------------------------------------------ #
    # Integration
    # ------------------------------------------------------------------ #

    def integrate(
        self,
        final_time: float = 100.0,
        dt: float = 0.02,
        *,
        t0: float = 0.0,
        ic: Any | None = None,
        method: str | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        backend: str | None = None,
        **integrator_kwargs,
    ) -> Trajectory:
        """
        Integrate the ODE and return a :class:`~tsdynamics.families.Trajectory`.

        Parameters
        ----------
        final_time : float
            End of integration window. Default 100.0.
        dt : float
            Output sampling interval. The internal stepper is adaptive.
        t0 : float
            Start time. Default 0.0. Allows warm restarts from a non-zero
            time (the IC is interpreted as the state at ``t0``).
        ic : array-like, optional
            Initial state at ``t0``. Falls back to ``self.ic``, then
            ``U[0, 1)^dim``.
        method : str, optional
            Solver name, resolved by the solver registry (default ``"RK45"``):
            explicit (``RK45`` / ``DOP853`` / ``tsit5`` / ``dop853``) or implicit
            / stiff (``bdf`` / ``rosenbrock`` / ``trbdf2``).
        rtol, atol : float
            Solver tolerances (default 1e-6 / 1e-9).
        backend : {"interp", "jit", "reference"}, optional
            Where the ODE is integrated.  Defaults to ``_default_backend``
            (``"interp"``).

            - ``"interp"`` / ``"jit"`` — the **Rust engine** (the zero-warmup
              SSA-tape interpreter or the Cranelift JIT) via the shared engine
              seam (:func:`tsdynamics.engine.run.integrate`).
            - ``"reference"`` — the dependency-light pure-Python oracle (the
              lowered tape integrated with SciPy); the engine's validation
              backend, usable without the compiled wheel.

        Returns
        -------
        Trajectory
            Supports tuple-unpacking: ``t, y = sys.integrate(...)``.
        """
        backend = backend if backend is not None else self._default_backend
        return self._dispatch(
            backend=backend,
            final_time=final_time,
            dt=dt,
            t0=t0,
            ic=ic,
            method=method or self._default_method,
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
        n_exp: int | None = None,
        burn_in: float = 50.0,
        method: str | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **integrator_kwargs,
    ) -> np.ndarray:
        """
        Estimate the Lyapunov spectrum of the flow.

        Delegates to :class:`~tsdynamics.derived.tangent.TangentSystem`, the one
        backend-neutral variational/Lyapunov engine shared across families: the
        *extended* variational ODE (state ⊕ ``k`` tangent vectors) is integrated
        on the Rust engine per dt-chunk and QR-reorthonormalised.

        Results are stored in ``self.meta['lyapunov_spectrum']``.

        Parameters
        ----------
        final_time : float
            Averaging window length after burn-in. Default 200.0.
        dt : float
            Sampling interval for local exponent accumulation. Default 0.1.
        ic : array-like, optional
            Initial state. Falls back to ``self.ic``, then random.
        n_exp : int, optional
            Number of exponents. Defaults to ``dim``.
        burn_in : float
            Discard this much time before averaging. Default 50.0.
        method : str, optional
            Integrator (default ``"RK45"``).
        rtol, atol : float
            Tolerances.

        Returns
        -------
        ndarray, shape (n_exp,)
            Lyapunov exponents ordered from largest to smallest.
        """
        if n_exp is not None and n_exp <= 0:
            raise ValueError(f"n_exp must be a positive integer, got {n_exp!r}")
        from tsdynamics.derived.tangent import TangentSystem

        k = n_exp if n_exp is not None else self.dim
        return TangentSystem(self, k=k, backend="interp").lyapunov_spectrum(
            final_time=final_time,
            dt=dt,
            ic=ic,
            burn_in=burn_in,
            method=method,
            rtol=rtol,
            atol=atol,
            **integrator_kwargs,
        )
