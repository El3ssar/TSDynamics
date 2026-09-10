"""ContinuousSystem — ODE base class on the Rust engine."""

from __future__ import annotations

import itertools
import math
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, cast

import numpy as np

from tsdynamics.errors import InvalidParameterError
from tsdynamics.utils.tolerances import DEFAULT_ATOL, DEFAULT_RTOL

from .base import SystemBase, Trajectory, as_lyapunov_result, resolve_transient

#: Memo for :meth:`ContinuousSystem._equations_hash`, keyed by the ``_equations``
#: kernel **function object itself** — the same key discipline the lowered-tape
#: cache uses (``engine/compile.py``), and for the same reason: a kernel's source
#: text cannot change without the function object being replaced, so identity is
#: both cheap and exactly as invalidating as re-reading the source would be.
#: A monkeypatched or redefined ``_equations`` is a *different* object, hence a
#: miss, hence a fresh token and a fresh ``_cache_key`` — so the numeric-evaluator
#: cache still rebuilds.  Weak keys mean the memo dies with the function (a
#: notebook redefining a class in a loop cannot leak entries).
_EQUATIONS_HASH_MEMO: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()

#: Per-kernel-object serial, appended to the source hash so the token identifies
#: the *function object*, not merely its source text.  Source alone is not
#: injective: a factory that closes over a value (``def make(m): def _eq(y, t, a,
#: _m=m): ...``) produces distinct kernels with byte-identical source, which
#: previously collided on one ``_lambdified`` entry — so the second system
#: silently evaluated the first one's RHS/Jacobian.  Identity is the correct
#: discriminator and is what the lowered-tape cache already uses; the serial is
#: how it is expressed inside a string cache key.
_EQUATIONS_HASH_SERIAL = itertools.count()


def _kernel_token(fn: Any) -> str:
    """Return the source content hash of ``fn`` (the un-memoised core).

    Eight hex digits of md5 over the kernel's source text, or — for a kernel
    with no retrievable source — over its bytecode.
    """
    import hashlib
    import inspect

    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):  # dynamically defined without source
        code = getattr(fn, "__code__", None)
        src = repr(code.co_code) if code is not None else repr(fn)
    return hashlib.md5(src.encode()).hexdigest()[:8]


class _NumericRHS:
    """A picklable ``f(u, t) -> ndarray`` over a SymEngine-lambdified RHS.

    The numeric-RHS helper used to be a closure over ``rhs_fn`` / ``vals``
    defined inside :meth:`ContinuousSystem._rhs_numeric`.  A local function
    cannot be pickled, so every object that *cached* one became unpicklable —
    most visibly :class:`~tsdynamics.derived.poincare.PoincareMap`, the wrapper
    a user reaches for to parallelise a bifurcation sweep.  Hoisting the closure
    into a module-level callable with the same behaviour fixes that without
    changing a single number: the two ``__call__`` bodies are identical, and both
    the lambdified callable and the captured parameter vector pickle natively.
    """

    __slots__ = ("_rhs_fn", "_vals")

    def __init__(self, rhs_fn: Any, vals: np.ndarray) -> None:
        self._rhs_fn = rhs_fn
        self._vals = vals

    def __call__(self, u: Any, t: float = 0.0) -> np.ndarray:
        arg = np.concatenate([np.asarray(u, dtype=float).ravel(), [t], self._vals])
        return np.asarray(self._rhs_fn(arg), dtype=float).ravel()

    # ``__slots__`` classes get no ``__dict__``, so spell the pickle protocol out.
    def __getstate__(self) -> tuple[Any, np.ndarray]:
        return (self._rhs_fn, self._vals)

    def __setstate__(self, state: tuple[Any, np.ndarray]) -> None:
        self._rhs_fn, self._vals = state


def _resolve_extremum_derivative(
    is_max: bool, args: Sequence[Any], wrt: Any, walk: Callable[[Any], Any]
) -> Any:
    """Build the a.e. derivative of ``min``/``max`` of ``args`` w.r.t. ``wrt``.

    The derivative of ``max(a₀, …, aₙ)`` (resp. ``min``) is, almost everywhere,
    the derivative of whichever argument is currently the active extremum.  This is
    expressed as a nested SymEngine ``Piecewise`` over the existing comparison
    opcodes — the kink set (where two arguments tie) is measure-zero, so the
    numeric Jacobian along an orbit is unaffected (the same a.e. convention the
    ``Abs``/``sign`` resolution uses).

    Each branch is the (finite) chain-rule derivative ``daᵢ/dwrt`` of one argument,
    so the arithmetic-blend ``Piecewise`` lowering — which evaluates *every* branch
    and masks — stays well-defined.

    Parameters
    ----------
    is_max : bool
        ``True`` for ``max`` (select the largest argument), ``False`` for ``min``.
    args : sequence of SymEngine expressions
        The extremum's arguments ``a₀, …, aₙ``.
    wrt : SymEngine symbol or ``y(i)`` application
        The variable to differentiate with respect to.
    walk : callable
        The recursive resolver (handles nested unevaluated derivatives in each
        argument's own derivative via the chain rule).
    """
    import symengine

    running_val = symengine.sympify(args[0])
    running_der = walk(running_val.diff(wrt))
    for raw in args[1:]:
        cand_val = symengine.sympify(raw)
        cand_der = walk(cand_val.diff(wrt))
        # ``max``: select the candidate when it is ≥ the running extremum;
        # ``min``: when it is ≤ it.  Ties fall through to the running branch.
        cond = (
            symengine.Ge(cand_val, running_val) if is_max else symengine.Le(cand_val, running_val)
        )
        running_der = symengine.Piecewise((cand_der, cond), (running_der, True))
        running_val = (
            symengine.Max(cand_val, running_val) if is_max else symengine.Min(cand_val, running_val)
        )
    return running_der


def _resolve_derivative_nodes(expr: Any) -> Any:
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

    def walk(e: Any) -> Any:
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
            if tname in ("Min", "Max"):
                # ``d/dx min(a₀, …, aₙ) = Σ_i 1[aᵢ is the active extremum] · daᵢ/dx``
                # a.e.: the derivative follows whichever argument is currently the
                # min/max.  Express it with the existing comparison/Piecewise
                # opcodes — every branch is the (finite) derivative of an argument,
                # so the arithmetic-blend Piecewise lowering stays well-defined.
                try:
                    return _resolve_extremum_derivative(tname == "Max", target.args, wrt, walk)
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


#: Keywords ``lyapunov_spectrum`` forwards to the integrator.  Everything a
#: caller may legitimately pass is bound to an explicit parameter, so anything
#: else in ``**integrator_kwargs`` is a typo.
_LYAPUNOV_FORWARDED: frozenset[str] = frozenset(
    {"t0", "max_step", "max_steps", "first_step", "seed"}
)


def _reject_unknown_lyapunov_keywords(extra: dict[str, Any]) -> None:
    """Raise on an unrecognised ``lyapunov_spectrum`` keyword instead of dropping it.

    ``**integrator_kwargs`` silently swallowed anything it did not recognise, and
    the keyword it swallowed most often was ``n_exp`` — this method's own
    parameter until the v4 glossary renamed it to ``k``.  So
    ``lorenz.lyapunov_spectrum(n_exp=1)`` returned THREE exponents: the caller
    asked for one, the request went into the void, and the answer looked fine.
    A wrong number returned confidently is the worst outcome available here, and
    it is the same footgun ``integrate`` already closed.
    """
    unknown = sorted(set(extra) - _LYAPUNOV_FORWARDED)
    if not unknown:
        return
    from tsdynamics.errors import invalid_value

    bad = unknown[0]
    hint = (
        "did you mean k=? (the number of exponents was renamed n_exp -> k in v4)."
        if bad in {"n_exp", "nexp", "n_exponents"}
        else "valid keywords: final_time, dt, ic, k, transient, method, rtol, atol, backend."
    )
    raise invalid_value(
        bad, extra[bad], rule="is not a valid lyapunov_spectrum() keyword", hint=hint
    )


class ContinuousSystem(SystemBase, ABC):
    """
    Base class for ODE-based dynamical systems, integrated on the engine.

    Subclass contract
    -----------------
    1. Declare ``params = {...}`` and ``dim = N`` at class level.
    2. Implement ``_equations`` as a ``@staticmethod`` returning a
       length-``dim`` sequence of SymEngine symbolic expressions.  Its
       signature is ``_equations(y, t, **params)``, and **``y`` is a state
       ACCESSOR, not an array**: component ``i`` is ``y(i)`` — a *call*, which
       is also what lets a :class:`~tsdynamics.families.delay.DelaySystem`
       write a delayed access as ``y(i, t - tau)``.  So neither ``y[0]`` nor
       ``x, y, z = y`` works here (a ``DiscreteMap``'s ``_step`` does take a
       plain state vector — that is the one place the two families differ)::

           class MyLorenz(ts.ContinuousSystem):
               params = {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3}
               variables = ("x", "y", "z")
               dim = 3

               @staticmethod
               def _equations(y, t, sigma, rho, beta):
                   x, u, z = y(0), y(1), y(2)          # NOT `x, u, z = y`
                   return [sigma * (u - x), x * (rho - z) - u, x * u - beta * z]

    3. Optionally mark integer or loop-structural parameters in
       ``_structural_params`` — these are baked into the lowered tape
       rather than exposed as runtime control parameters.

    Lowering
    --------
    Each system is lowered once to an in-process IR tape, then JIT-compiled on
    first use (both memoised, so neither cost repeats); the
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
    >>> t, y = traj.unpack()   # the two columns
    >>> lor.sigma = 15.0     # change param — zero recompile cost
    >>> traj2 = lor.integrate(final_time=100)
    """

    _default_method: ClassVar[str] = "RK45"

    #: The default runtime backend (see :attr:`SystemBase._default_backend`).
    #: ``"jit"`` — the Cranelift JIT, with the process-wide compiled-evaluator
    #: cache paying the compile once per distinct system.  Was ``"interp"``
    #: before v6, when every call recompiled the tape.
    _default_backend: ClassVar[str] = "jit"

    #: Parameters whose values affect the symbolic *structure* of _equations
    #: (e.g. integer loop bounds). These are baked in at lowering time.
    _structural_params: ClassVar[frozenset[str]] = frozenset()

    # Per-class in-process cache:
    # _lambdified : cache_key (str) → (rhs_fn, jac_fn, control_names) —
    #               SymEngine-Lambdified numeric RHS/Jacobian evaluators
    #               (used by figures, Poincaré Hermite refinement and analyses),
    #               keyed by class + dim + structural params.
    # Bounded LRU (move-to-end on hit, evict-oldest past the cap) so a long-lived
    # process that lowers many distinct structural variants of one system (e.g. a
    # Lorenz96 swept over its dimension ``N``) cannot grow this without bound; the
    # control-param sweep that re-runs one structural variant is served from a
    # single entry, so the common case never evicts.  The cap is generous because
    # each entry is a couple of compiled SymEngine callables, not a heavy buffer.
    _lambdified: ClassVar[OrderedDict[str, tuple[Any, Any, list[str]]]] = OrderedDict()

    #: Maximum number of distinct numeric-evaluator entries kept per class in
    #: :attr:`_lambdified` before the least-recently-used entry is evicted.
    _LAMBDIFIED_CACHE_MAXSIZE: ClassVar[int] = 64

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
    # reinit instead of on every ``step`` (stream WS-STEPBUF).  ``_step_tape_arrays``
    # caches the engine wire arrays of the (loop-invariant) tape so the per-``dt``
    # loop reuses them rather than re-marshalling the tuple each call (WS-INVHOIST).
    _step_method_canonical: str | None = None
    _step_tape_arrays: Any = None
    #: The per-step size ceiling ``reinit`` recorded (``None`` = no ceiling); every
    #: subsequent ``step`` forwards it to the engine.
    _step_max_step: float | None = None

    # The durable resumable engine stepper handle (stream WS-STEPPER): an opaque
    # ``tsdynamics._rust.OdeStepper`` that owns the built tape evaluator + solver
    # once and carries the live integration point across ``step`` calls, so a
    # per-``dt`` loop never re-marshals the tape into the engine.  Built lazily on
    # the first ``step`` after a ``reinit`` (so a cold ``state()`` / ``time()`` —
    # or a ``reference``-backed flow — never forces the compiled engine), and
    # discarded by ``reinit`` so a re-lowered tape / new state is always picked up.
    _ode_stepper: Any = None

    # ------------------------------------------------------------------ #
    # Subclass interface
    # ------------------------------------------------------------------ #

    @staticmethod
    @abstractmethod
    def _equations(y: Any, t: Any, **params: Any) -> Sequence[Any]:
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
        Identity token for the RHS definition, part of every compile-cache key.

        Without it, two same-named classes (user shadowing a builtin, or a
        notebook cell redefining a class with edited equations) would silently
        reuse each other's compiled dynamics.

        The token is **memoised on the kernel function object**, and identifies
        that object rather than only its source text.  Two properties follow:

        * It is *cheap*.  Reading and hashing the source on every call defeated
          the very cache this key serves — ``inspect.getsource`` dominated
          :meth:`jacobian` (69 µs of a 92 µs Lorenz call, a 2.9x tax on every
          flow variational analysis).  A warm call now costs one weak-dict
          lookup.
        * It is *at least as invalidating*.  A monkeypatched or redefined
          ``_equations`` is a different object, so it misses the memo, re-reads
          the source and gets a fresh token — and because the token carries a
          per-object serial it is injective in the function object, where the
          bare source hash was not (a factory closing over a value produces
          distinct kernels with byte-identical source; those used to collide on
          one :attr:`_lambdified` entry, so the second system silently evaluated
          the first one's RHS and Jacobian).

        This is the key discipline the lowered-tape cache already uses.
        """
        fn = type(self)._equations
        fn = getattr(fn, "__func__", fn)
        try:
            memo = _EQUATIONS_HASH_MEMO.get(fn)
        except TypeError:
            # Neither hashable nor weak-referenceable (a builtin / C callable).
            # It cannot be memoised, so it cannot carry a stable serial either —
            # minting a fresh one per call would make *every* lookup a miss and
            # rebuild the evaluator.  Fall back to the bare source hash, i.e.
            # exactly the pre-memo behaviour for this (exotic) kernel shape.
            return _kernel_token(fn)
        if memo is not None:
            return memo
        token = f"{_kernel_token(fn)}{next(_EQUATIONS_HASH_SERIAL):x}"
        _EQUATIONS_HASH_MEMO[fn] = token
        return token

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
        params: dict[str, Any] | None = None,
        method: str | None = None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        max_step: float | None = None,
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
        method, rtol, atol, max_step, backend
            Stepper configuration, as in :meth:`integrate`.  ``max_step`` is
            stored and applied to every subsequent :meth:`step`.

        Notes
        -----
        A ``reinit`` that raises leaves the system exactly as it was — ``self.ic``
        is rolled back rather than left holding the initial condition that failed.
        """
        if params:
            for k, v in params.items():
                self.params[k] = v
        # ``resolve_ic`` commits the resolved IC to ``self.ic`` before any of the
        # work below happens, and plenty below can raise (an unknown backend, an
        # unresolvable method, a tape that will not lower, an unavailable engine).
        # Without the guard a failed ``reinit`` latched the offending IC onto the
        # instance and every later, unrelated call silently started from it.
        with self._ic_rollback():
            self._reinit_resolved(
                u, t=t, method=method, rtol=rtol, atol=atol, max_step=max_step, backend=backend
            )

    def _reinit_resolved(
        self,
        u: Any | None,
        *,
        t: float | None,
        method: str | None,
        rtol: float,
        atol: float,
        max_step: float | None,
        backend: str | None,
    ) -> None:
        """Run :meth:`reinit`'s body (wrapped by its IC rollback guard)."""
        t0 = float(t) if t is not None else 0.0
        ic_arr = self.resolve_ic(u)
        from tsdynamics.engine.run import resolve_backend

        # Honour the requested backend through the stepping protocol: ``reference``
        # is the wheel-free pure-Python oracle and a real, supported stepping path
        # (one ``dt`` chunk per ``step`` through the reference ODE integrator), not
        # something to silently coerce to ``"interp"`` (the old behaviour, which
        # made the oracle unreachable via ``reinit``/``step``/``state`` — diagnosis
        # #5).  ``resolve_backend`` raises ``InvalidParameterError`` for an unknown
        # name (it subclasses ``ValueError``).
        self._step_backend = resolve_backend(
            backend if backend is not None else self._default_backend
        )

        from tsdynamics import solvers
        from tsdynamics.engine.problem import ode_problem

        # Lower the tape ONCE here and reuse it for every step() — a sweep reinits
        # thousands of times, so re-lowering per step would dominate the cost.
        # Resolve the step method first so an implicit kernel (bdf/rosenbrock/
        # trbdf2) gets a Jacobian-carrying tape — step() reuses this exact tape,
        # so the Jacobian must be baked in here or the engine refuses the step.
        #
        # ``method="auto"`` is the same a-priori auto-stiffness contract
        # ``integrate``/``ensemble`` honour (FIX-AUTOSTIFF): probe the Jacobian
        # spectrum at the start state and let the solver registry pick the implicit
        # ``bdf`` on a stiff RHS or the explicit ``rk45`` otherwise
        # (:func:`tsdynamics.solvers.recommend`).  Without this branch the stepping
        # path crashed with an opaque "unknown solver method 'auto'" — the one
        # entry point that rejected the advertised value (diagnosis P2-1), exactly
        # where auto-stiffness matters for a stiff system stepped incrementally
        # (Poincaré / basins / streaming).
        self._step_method = method or self._default_method
        if solvers.normalize(self._step_method) == "auto":
            resolution = solvers.recommend(self, family="ode", ic=ic_arr, t=t0)
        else:
            resolution = solvers.resolve(self._step_method)
        # Cache the canonical kernel name (``"RK45"`` → ``"rk45"``) so the per-step
        # loop hits the engine core directly without re-resolving every call.
        self._step_method_canonical = resolution.name
        prob = ode_problem(self, ic=ic_arr, t0=t0, **resolution.build_kwargs)
        # Marshal the (loop-invariant) tape to its engine wire arrays once here so the
        # per-``dt`` ``step`` loop reuses them instead of rebuilding the tuple on every
        # call (stream WS-INVHOIST).  Re-derived on each ``reinit`` (a sweep reinits
        # per parameter value), so a re-lowered tape is always picked up.  Marshal before
        # publishing the problem so a (theoretical) marshalling failure can't leave a new
        # ``_engine_problem`` paired with a stale ``_step_tape_arrays``.
        step_arrays = prob.tape.to_arrays()
        self._engine_problem = prob
        self._step_tape_arrays = step_arrays
        self._step_rtol = float(rtol)
        self._step_atol = float(atol)
        self._step_max_step = max_step
        self._state_now = ic_arr.copy()
        self._t_now = t0
        # Drop any prior durable stepper handle: the next ``step`` rebuilds it from
        # the freshly lowered tape + new live state (WS-STEPPER).  Built lazily so a
        # cold ``state()``/``time()`` (or a ``reference`` flow) never forces the
        # compiled engine here — exactly as the pre-handle path reached the engine
        # only inside ``step``.
        self._ode_stepper = None

    def step(self, n_or_dt: float | None = None) -> np.ndarray:
        """
        Advance the system by ``dt`` (default 0.01) and return the new state.

        The first call performs an implicit :meth:`reinit`.  Parameter changes
        made after ``reinit`` take effect on the next ``reinit``, not on a
        live stepper.

        Notes
        -----
        Each call advances **exactly one** ``dt`` from the live ``(state, t)``,
        returning byte-for-byte the trajectory the released per-``dt`` path
        produced — there is no batching, so the numbers are unchanged for every
        method, adaptive or fixed-step (streams WS-STEPBUF, WS-INVHOIST,
        WS-STEPPER).  The amortisation is durable: the first ``step`` after a
        :meth:`reinit` builds an opaque resumable engine handle
        (:class:`tsdynamics._rust.OdeStepper`, via
        :func:`~tsdynamics.engine.run.make_ode_stepper`) that owns the built tape
        evaluator + solver once and carries the live ``(u, t)`` across calls; every
        later ``step`` is one :func:`~tsdynamics.engine.run.step_advance` on that
        handle — the tape is **never re-marshalled into the engine again**.  So a
        constant-``dt`` stepping loop (Poincaré refinement, basins over flows) skips
        not only the solver-registry resolve, the implicit-Jacobian decision, the
        output-grid build, provenance assembly and the :class:`Trajectory` wrap that
        the full :meth:`integrate` entry point pays, but also the per-step tape
        re-marshalling and tape *rebuild* the pre-handle stepping core paid.  The
        control-parameter vector is still read live each step, so the live-stepper
        semantics are unchanged.

        Why this stays answer-exact: the engine handle's ``advance(dt)`` re-seeds a
        *fresh* solver and state for each ``dt`` segment (the adaptive controller is
        re-seeded each step, exactly as the released per-``dt`` ``integrate_dense``
        did), so the numbers are bit-for-bit identical to that path — verified in
        the engine's own test.  A batch-ahead variant that integrated a whole chunk
        in one engine call was rejected (WS-STEPBUF): a chunked adaptive integration
        is *not* equal to N single-``dt`` integrations (the controller would carry
        its step/error state across output nodes), which silently corrupted
        sensitive consumers such as ``max_lyapunov``.  The durable handle amortises
        the *build/marshalling*, never the numerics.
        """
        from tsdynamics.engine.run import make_ode_stepper, step_advance

        if self._engine_problem is None:
            self.reinit()
        # After ``reinit`` (run above when cold) these stepping-state attributes are
        # always populated; narrow them for the typed engine call below.
        assert self._state_now is not None
        assert self._step_method_canonical is not None
        dt = float(n_or_dt) if n_or_dt is not None else self._default_step_dt

        # The wheel-free pure-Python oracle: advance one ``dt`` chunk through the
        # reference ODE integrator (the same path ``integrate(backend="reference")``
        # uses), so the protocol exposes the oracle honestly instead of secretly
        # running the compiled engine (diagnosis #5).  Off the durable engine-handle
        # fast path (reference owns no ``OdeStepper``); it is the validation backend,
        # not a hot loop.
        if self._step_backend == "reference":
            return self._step_reference(dt)

        t0 = self._t_now
        tf = t0 + dt
        # Preserve the released ``step`` span contract exactly: a non-positive ``dt``
        # or a non-forward window (``t0 + dt == t0`` at a large ``t0``) must raise the
        # canonical :class:`~tsdynamics.errors.InvalidParameterError` (a
        # ``ValueError``), not silently no-op on the engine handle.  When the span
        # clears ``1e-9`` the regime is unambiguously forward+positive, so the happy
        # path goes straight to the handle (never touching ``make_output_grid``);
        # only the sub-``1e-9`` remainder defers to the helper for that identical
        # loud-footgun error (and the byte-identical degenerate-grid behaviour).
        if not tf - t0 > 1e-9:
            from tsdynamics.utils.grids import make_output_grid

            # Raises InvalidParameterError for dt <= 0 / a non-forward window; for a
            # valid-but-tiny span it returns the (possibly single-node) grid, which
            # the per-``dt`` engine core integrated identically — reproduce that here
            # via the same lean core to stay byte-identical for the rare small step.
            t_eval = make_output_grid(t0, tf, dt)
            from tsdynamics.engine.run import _step_continuous

            y = _step_continuous(
                self._step_tape_arrays,
                self._state_now,
                self._engine_problem.params_vec(),
                t_eval,
                method=self._step_method_canonical,
                rtol=self._step_rtol,
                atol=self._step_atol,
                max_step=math.inf if self._step_max_step is None else float(self._step_max_step),
                jit=self._step_backend == "jit",
                name=type(self).__name__,
            )
            state = np.asarray(y[-1], dtype=float)
            # The handle (if already built) no longer mirrors the live point after
            # this off-handle advance; drop it so the next ``step`` rebuilds from the
            # synced ``_state_now``/``_t_now``.
            object.__setattr__(self, "_ode_stepper", None)
            object.__setattr__(self, "_t_now", tf)
            object.__setattr__(self, "_state_now", state.copy())
            return state.copy()

        # Build the durable resumable handle lazily on the first ``step`` after a
        # ``reinit`` (so a cold ``state()``/``time()`` never forces the engine), from
        # the live state + the tape arrays + the solver config ``reinit`` cached.
        if self._ode_stepper is None:
            stepper = make_ode_stepper(
                self._step_tape_arrays,
                self._state_now,
                self._t_now,
                method=self._step_method_canonical,
                rtol=self._step_rtol,
                atol=self._step_atol,
                jit=self._step_backend == "jit",
            )
            object.__setattr__(self, "_ode_stepper", stepper)

        state = step_advance(
            self._ode_stepper,
            dt,
            self._engine_problem.params_vec(),
            name=type(self).__name__,
            max_step=self._step_max_step,
        )
        # The state/time advance writes private framework attributes that always pass
        # straight through ``SystemBase.__setattr__`` (underscore-prefixed) — go direct
        # to ``object.__setattr__`` so the hot loop skips the param-typo guard's
        # ``params`` membership check on every step (WS-INVHOIST).  The engine handle
        # is the live-state authority; ``_state_now``/``_t_now`` mirror it so
        # ``state()``/``time()`` and ``set_state`` stay consistent.
        object.__setattr__(self, "_t_now", self._t_now + dt)
        object.__setattr__(self, "_state_now", state.copy())
        return state.copy()

    def _step_reference(self, dt: float) -> np.ndarray:
        """Advance one ``dt`` chunk on the pure-Python reference ODE integrator.

        The ``backend="reference"`` stepping path: it integrates the cached
        (loop-invariant) tape from the live ``(state, t)`` over a one-segment grid
        with the same :func:`tsdynamics.engine.run._run_continuous` the
        ``integrate(backend="reference")`` entry point uses, so the wheel-free
        oracle is reachable through ``reinit``/``step``/``state`` exactly as it is
        through ``integrate`` (diagnosis #5).  Answer-identical to
        ``integrate(backend="reference")`` over the same ``dt`` discretisation.

        A non-positive ``dt`` / non-forward window raises
        :class:`~tsdynamics.errors.InvalidParameterError` (a ``ValueError``), the
        same loud-footgun contract as the engine stepping path
        (:func:`~tsdynamics.utils.grids.make_output_grid` enforces it).
        """
        import dataclasses

        from tsdynamics.engine.run import _run_continuous
        from tsdynamics.utils.grids import make_output_grid

        assert self._state_now is not None
        assert self._step_method_canonical is not None
        t0 = self._t_now
        tf = t0 + dt
        # Raises InvalidParameterError for dt <= 0 / a non-forward window; otherwise
        # the (possibly single-node) one-segment grid the reference integrator samples.
        t_eval = make_output_grid(t0, tf, dt)
        # Re-point the cached, already-lowered problem at the live state/time without
        # re-lowering (a frozen-dataclass copy); the reference integrator reads the
        # tape + live params off it.
        prob = dataclasses.replace(
            self._engine_problem, ic=np.ascontiguousarray(self._state_now, dtype=np.float64), t0=t0
        )
        y = _run_continuous(
            prob,
            t_eval,
            method=self._step_method_canonical,
            rtol=self._step_rtol,
            atol=self._step_atol,
            backend="reference",
            max_step=math.inf if self._step_max_step is None else float(self._step_max_step),
        )
        state = np.asarray(y[-1], dtype=float)
        object.__setattr__(self, "_t_now", tf)
        object.__setattr__(self, "_state_now", state.copy())
        return state.copy()

    def state(self) -> np.ndarray:
        """Return a copy of the current state (implicit ``reinit`` if cold)."""
        if self._state_now is None:
            self.reinit()
        assert self._state_now is not None  # set by reinit above
        return self._state_now.copy()

    def set_state(self, u: Any) -> None:
        """Overwrite the current state without changing the current time."""
        u_arr = np.asarray(u, dtype=float).reshape(self.dim)
        if self._engine_problem is None:
            self.reinit(u_arr)
        else:
            self._state_now = u_arr.copy()
            # Drop the durable stepper handle: its live point no longer mirrors the
            # reseated state, so the next ``step`` rebuilds it from ``_state_now`` /
            # ``_t_now`` (WS-STEPPER).  Cheaper and less error-prone than reseating
            # the handle in place, and ``set_state`` is not on the hot stepping path.
            self._ode_stepper = None

    def time(self) -> float:
        """Return the current stepper time."""
        return self._t_now

    def trajectory(
        self,
        final_time: float = 100.0,
        *,
        dt: float = 0.02,
        transient: float = 0.0,
        **kwargs: Any,
    ) -> Trajectory:
        """Protocol-uniform trajectory: ``integrate`` plus optional transient drop.

        A permanent alias of ``integrate(transient=...)``, which owns the one
        implementation — so the transient cut respects a non-zero ``t0`` here
        too (it previously cut at the absolute time ``transient``, discarding
        the whole run whenever ``t0 >= transient``).
        """
        return self.integrate(final_time=final_time, dt=dt, transient=transient, **kwargs)

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

        dim = cast(int, self.dim)
        struct_vals = self._structural_vals()
        control_syms = {k: symengine.Symbol(k) for k in self._control_params()}
        f_sym = list(type(self)._equations(y, t_sym, **{**struct_vals, **control_syms}))
        if len(f_sym) != dim:
            raise InvalidParameterError(
                f"_equations must return {dim} expressions, got {len(f_sym)}"
            )
        return [
            [_resolve_derivative_nodes(symengine.sympify(fi).diff(y(j))) for j in range(dim)]
            for fi in f_sym
        ]

    def _build_lambdified(self) -> tuple[Any, Any, list[str]]:
        """
        Build (and cache) SymEngine-Lambdified numeric RHS and Jacobian.

        Both take a flat argument vector ``[y_0..y_{dim-1}, t, *control_params]``.
        Cached per (class, dim, structural-hash) — parameter value changes
        need no rebuild because control params are call-time arguments.  The
        per-class cache is a bounded LRU (:attr:`_LAMBDIFIED_CACHE_MAXSIZE`):
        a cache hit is moved to the most-recently-used end, and an insertion past
        the cap evicts the least-recently-used entry, so a process that lowers many
        distinct structural variants stays bounded.

        Returns
        -------
        tuple of (rhs_fn, jac_fn, control_names)
            ``rhs_fn`` / ``jac_fn`` are SymEngine ``Lambdify`` callables and
            ``control_names`` the ordered non-structural parameter names.
        """
        cache = type(self)._lambdified
        key = self._cache_key()
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)  # mark most-recently-used
            return cached

        import symengine

        from tsdynamics.engine.symbols import state_time_symbols

        y, t_sym = state_time_symbols()

        dim = cast(int, self.dim)
        struct_vals = self._structural_vals()
        control_names = list(self._control_params())
        control_syms = {k: symengine.Symbol(k) for k in control_names}
        f_sym = [
            symengine.sympify(e)
            for e in type(self)._equations(y, t_sym, **{**struct_vals, **control_syms})
        ]
        jac_rows = [[_resolve_derivative_nodes(fi.diff(y(j))) for j in range(dim)] for fi in f_sym]

        # Lambdify needs plain symbols — swap the y(i) function applications out.
        y_syms = [symengine.Symbol(f"y_{i}") for i in range(dim)]
        subs = {y(i): y_syms[i] for i in range(dim)}
        args = [*y_syms, t_sym, *(control_syms[k] for k in control_names)]
        rhs_fn = symengine.Lambdify(args, [e.subs(subs) for e in f_sym])
        jac_fn = symengine.Lambdify(args, [e.subs(subs) for row in jac_rows for e in row])

        entry = (rhs_fn, jac_fn, control_names)
        cache[key] = entry
        cache.move_to_end(key)
        while len(cache) > type(self)._LAMBDIFIED_CACHE_MAXSIZE:
            cache.popitem(last=False)  # evict least-recently-used
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
        dim = cast(int, self.dim)
        _, jac_fn, control_names = self._build_lambdified()
        vals = [float(self.params[k]) for k in control_names]
        arg = np.concatenate([np.asarray(u, dtype=float).ravel(), [t], vals])
        return np.asarray(jac_fn(arg), dtype=float).reshape(dim, dim)

    def _rhs_numeric(self) -> Callable[..., np.ndarray]:
        """
        Return a fast numeric RHS callable ``f(u, t) -> ndarray``.

        Parameter values are captured at call time of this method; build a
        fresh callable after changing parameters.  Used by figure tooling,
        Poincaré crossing refinement, and backend cross-validation — the
        engine remains the integrator of record.

        The returned object is a module-level :class:`_NumericRHS` instance, not
        a closure, so it **pickles**: a wrapper holding one (notably
        :class:`~tsdynamics.derived.poincare.PoincareMap`, which caches it for
        Hermite refinement) can cross a ``multiprocessing`` / ``joblib`` boundary,
        which is exactly what a parallel bifurcation sweep needs.
        """
        rhs_fn, _, control_names = self._build_lambdified()
        vals = np.array([float(self.params[k]) for k in control_names])
        return _NumericRHS(rhs_fn, vals)

    # ------------------------------------------------------------------ #
    # Trajectory production — the canonical ``run`` verb
    # ------------------------------------------------------------------ #

    def run(
        self,
        final_time: float = 100.0,
        dt: float = 0.02,
        *,
        t0: float = 0.0,
        ic: Any | None = None,
        transient: float = 0.0,
        method: str | None = None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        max_step: float | None = None,
        backend: str | None = None,
        seed: int | None = None,
        events: Any = None,
        **integrator_kwargs: Any,
    ) -> Trajectory:
        """
        Produce a trajectory — the one canonical verb for every family.

        ``run`` is the unified trajectory producer: one verb for flows, maps,
        DDEs and SDEs.  For a continuous-time system (this family) it integrates
        the flow, so ``run`` is a thin alias of :meth:`integrate` and forwards
        every keyword to it unchanged.

        .. note::
           The **verb** is shared; the **horizon keyword follows the family**,
           because the two horizons are different quantities in different units.
           A flow has ``final_time`` (time units), a map has ``n`` (a count of
           iterations) — ``lor.run(final_time=100)`` and ``hen.run(n=1000)``.
           Passing ``final_time`` to a map is refused by name, not silently
           reinterpreted.  The **positional** form is the spelling that reads
           the same on both: ``lor.run(100)``, ``hen.run(1000)``.  Everything
           that *is* family-independent — ``ic``, ``seed``, ``backend``,
           ``transient`` — is spelled identically everywhere.

        Parameters
        ----------
        final_time : float
            End of the integration window. Default 100.0.
        dt : float
            Output sampling interval — the spacing of the returned grid; the
            internal stepper is adaptive and controlled by ``rtol``/``atol``
            (see :meth:`integrate`).  ``dt`` is **sampling only**: it does not
            bound the internal step.  Use ``max_step`` for that.
        t0 : float
            Start time of the window. Default 0.0.
        ic : array_like, optional
            Initial condition.  Priority ``ic`` > ``self.ic`` > ``default_ic`` >
            a seeded random draw.
        transient : float
            Time to integrate and **discard** before recording, so the returned
            trajectory starts on the attractor.  In time units (a map's
            ``transient`` is a count of iterations).
        method : str, optional
            Solver kernel (``"rk45"``, ``"dop853"``, ``"bdf"``, …), or
            ``"auto"`` to probe stiffness and choose.  Defaults to the system's
            ``_default_method``.
        rtol, atol : float
            Relative / absolute error tolerances — **this** is the accuracy
            knob.  Defaults are the library-wide
            :data:`~tsdynamics.utils.tolerances.DEFAULT_RTOL` /
            :data:`~tsdynamics.utils.tolerances.DEFAULT_ATOL`.
        max_step : float, optional
            Upper bound on any single internal step (a step **size** — not to be
            confused with ``max_steps``, a step *count*).  ``None`` is unbounded.
        backend : str, optional
            ``"jit"`` (default), ``"interp"``, or ``"reference"``.
        seed : int, optional
            Seeds the random initial-condition draw, making a run with no ``ic``
            reproducible.
        events : sequence, optional
            Detect events along the flow (a SciPy-shaped ``events=`` API).  Each
            element is an :class:`~tsdynamics.engine.run.Event`, a bare
            ``g(y, t)`` callable carrying ``.direction`` / ``.terminal``
            attributes (the SciPy convention), or a plane tuple
            (``("y", 0.0, "up")``).  A **terminal** event stops the integration at
            its first crossing (arbitrary stopping).  The returned trajectory
            carries each event's crossings in ``meta["t_events"]`` /
            ``meta["y_events"]`` (one array per event, aligned with ``events``),
            plus ``meta["terminated"]``.  This wires the same compiled event
            engine :class:`~tsdynamics.derived.poincare.PoincareMap` uses;
            ``PoincareMap.as_events()`` shows the section as one such event.
        **integrator_kwargs
            Any remaining solver option, forwarded verbatim to
            :meth:`integrate`.

        Returns
        -------
        Trajectory
            Identical to :meth:`integrate` when ``events`` is ``None`` — ``run``
            adds no behaviour.  With ``events`` set, the dense trajectory
            (truncated at the first terminal crossing) plus the per-event
            crossings in ``meta``.

        See Also
        --------
        integrate : The family-specific spelling (a permanent alias of ``run``).

        Examples
        --------
        >>> traj = Lorenz().run(final_time=100, dt=0.01)
        >>> Henon().run(n=5000)            # the same verb iterates a map
        >>> sol = Lorenz().run(final_time=50, events=[("z", 27.0, "up")])
        >>> sol.meta["t_events"][0].shape    # times z=27 was crossed upward
        (... ,)
        """
        return self.integrate(
            final_time=final_time,
            dt=dt,
            t0=t0,
            ic=ic,
            transient=transient,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            backend=backend,
            seed=seed,
            events=events,
            **integrator_kwargs,
        )

    def _run_events(
        self,
        *,
        final_time: float,
        dt: float,
        events: Any,
        t0: float = 0.0,
        ic: Any | None = None,
        method: str | None = None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        max_step: float | None = None,
        backend: str | None = None,
        seed: int | None = None,
    ) -> Trajectory:
        """Integrate with event detection and wrap the result as a Trajectory.

        Builds the ODE problem, hands it to the engine event seam
        (:func:`tsdynamics.engine.run.integrate_events`), and attaches the
        per-event crossings to ``meta`` (the SciPy-shaped ``t_events`` /
        ``y_events``).
        """
        # ``resolve_ic`` commits the IC before the engine march runs, so an event
        # run that diverges (or is interrupted) must not leave it latched — the
        # same contract ``_dispatch`` gives plain ``integrate``.
        with self._ic_rollback():
            return self._run_events_resolved(
                final_time=final_time,
                dt=dt,
                events=events,
                t0=t0,
                ic=ic,
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                backend=backend,
                seed=seed,
            )

    def _run_events_resolved(
        self,
        *,
        final_time: float,
        dt: float,
        events: Any,
        t0: float,
        ic: Any | None,
        method: str | None,
        rtol: float,
        atol: float,
        max_step: float | None,
        backend: str | None,
        seed: int | None,
    ) -> Trajectory:
        """Run :meth:`_run_events`' body (wrapped by its IC rollback guard)."""
        from tsdynamics.engine import run as engine_run
        from tsdynamics.engine.problem import ode_problem

        be = backend if backend is not None else self._default_backend
        meth = method or self._default_method
        # ``seed=`` is the initial-condition seed (inert unless a draw happens) —
        # the same contract ``integrate``/``_dispatch`` honour.
        ic_arr = self.resolve_ic(ic, seed=seed)
        prob = ode_problem(self, ic=ic_arr, t0=float(t0))
        # Resolve ``method=`` through the shared ``auto``-aware contract so the
        # events path honours ``method="auto"`` identically to integrate/ensemble
        # (diagnosis P3-1) and records the canonical kernel name in ``meta`` rather
        # than the raw ``"auto"`` alias.  ``integrate_events`` re-resolves the
        # canonical name to itself (idempotent), so this is the single resolution
        # point that drives both the engine call and the provenance.
        from tsdynamics.engine.run_methods import _resolve_method_for

        meth = _resolve_method_for(meth, prob).name
        sol = engine_run.integrate_events(
            prob,
            events,
            final_time=final_time,
            dt=dt,
            t0=float(t0),
            method=meth,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            backend=be,
        )
        meta = self._provenance(
            family="ode",
            engine="rust" if be in ("interp", "jit") else "reference",
            backend=be,
            method=meth,
            dt=dt,
            t0=float(t0),
            rtol=rtol,
            atol=atol,
            max_step=math.inf if max_step is None else float(max_step),
            ic=np.asarray(ic_arr, dtype=float).copy(),
            n_events=len(sol.events),
            terminated=sol.terminated,
            events=[
                {"name": e.name, "direction": e.direction, "terminal": e.terminal}
                for e in sol.events
            ],
            t_events=sol.t_events,
            y_events=sol.y_events,
        )
        return Trajectory(t=sol.t, y=sol.y, system=self, meta=meta)

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
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        max_step: float | None = None,
        backend: str | None = None,
        seed: int | None = None,
        transient: float = 0.0,
        events: Any = None,
        **integrator_kwargs: Any,
    ) -> Trajectory:
        """
        Integrate the ODE and return a :class:`~tsdynamics.families.Trajectory`.

        Parameters
        ----------
        final_time : float
            End of integration window. Default 100.0.
        dt : float
            Output sampling interval — the spacing of the returned grid.  **It
            does not set the integration accuracy:** the internal stepper is
            adaptive and controlled by ``rtol``/``atol``; interior samples are
            produced by the kernel's own continuous extension.  Use ``rtol`` /
            ``atol`` to control accuracy and ``max_step`` to bound the internal
            step.

            .. versionchanged:: 6.0
                Before v6 the stepper was forced to land on every output sample,
                so a fine ``dt`` silently bought extra accuracy and a coarse one
                silently lost it.  It no longer does.  Kernels without a native
                continuous extension (everything but ``rk45`` / ``tsit5`` /
                ``dop853``) still land on every sample.
        t0 : float
            Start time. Default 0.0. Allows warm restarts from a non-zero
            time (the IC is interpreted as the state at ``t0``).
        ic : array-like, optional
            Initial state at ``t0``. Falls back to ``self.ic``, then
            ``U[0, 1)^dim``.
        method : str, optional
            Solver name, resolved by the solver registry (default ``"RK45"``):
            explicit (``RK45`` / ``DOP853`` / ``tsit5`` / ``dop853``) or implicit
            / stiff (``bdf`` / ``rosenbrock`` / ``trbdf2``).  Pass ``"auto"`` to
            select a kernel by a-priori auto-stiffness — the Jacobian spectrum at
            the start state is probed and ``bdf`` chosen on a stiff RHS, ``rk45``
            otherwise (:func:`tsdynamics.solvers.recommend`; a one-point heuristic,
            so a reliably-stiff system should still declare ``_default_method``).
        rtol, atol : float
            Solver tolerances — the accuracy knob.  Default
            :data:`~tsdynamics.utils.tolerances.DEFAULT_RTOL` /
            :data:`~tsdynamics.utils.tolerances.DEFAULT_ATOL` (``1e-9`` /
            ``1e-12``).

            .. versionchanged:: 6.0
                Tightened from ``1e-6`` / ``1e-9``.  ``dt`` is now purely an
                output grid, so ``rtol`` is the *only* accuracy knob and the
                default had to carry the accuracy the old forced landing supplied
                for free.  Measured median 1459x more accurate for 1.74x the
                cost; pass ``rtol=1e-6`` for the pre-v6 trade.
        max_step : float, optional
            Upper bound on any single internal solver step, in time units.
            ``None`` (default) means no ceiling — the adaptive controller chooses
            freely from ``rtol``/``atol``.  Use it to stop an adaptive kernel
            stepping over a narrow feature (a thin resonance, a fast transient)
            in an otherwise smooth region, or to bound the detection resolution
            of an event march.  Note this is a step *size*; the engine's
            ``max_steps`` is a step *count* and is not exposed here.  A
            ``max_step`` far below the tolerance-driven natural step multiplies
            cost with no accuracy benefit and can trip
            :class:`~tsdynamics.errors.StepBudgetError`.
        backend : {"jit", "interp", "reference"}, optional
            Where the ODE is integrated.  Defaults to ``_default_backend``
            (``"jit"``).  The first two go through the shared engine seam
            (:func:`tsdynamics.engine.run.integrate`).

            - ``"jit"`` (default) — the **Cranelift JIT**: the lowered tape
              compiled to native code, with a process-wide compiled-evaluator
              cache, so the compile is paid once per distinct system rather than
              on every call.
            - ``"interp"`` — the **SSA-tape interpreter**.  Same answers as the
              JIT, bit-for-bit; marginally faster on very small tapes, and the
              way to avoid the one-off compile.
            - ``"reference"`` — the dependency-light pure-Python/SciPy oracle.
              Not for production use: it is the independent cross-check the
              engine is validated against, and the wheel-free fallback.

            .. versionchanged:: 6.0
               The default moved from ``"interp"`` to ``"jit"``.  Before v6 the
               JIT recompiled the whole tape on every FFI call, which made it
               slower than the interpreter for short runs; the v6
               compiled-evaluator cache removed that per-call compile.
        seed : int, optional
            Seed for the **random initial-condition draw** — the same meaning it
            has on :meth:`DiscreteMap.iterate` and on the constructor
            (:meth:`SystemBase.ic_generator`), so ``seed=`` reads identically on
            every family.  It only bites when a draw actually happens: an
            explicit ``ic``, an ``ic`` already resolved onto the system, and a
            class-level ``default_ic`` all take priority.  The resolved seed is
            recorded on ``traj.meta["ic_seed"]``.

            .. versionadded:: 6.0
        events : sequence, optional
            Detect events along the flow (the SciPy-shaped ``events=`` API; see
            :meth:`run`).  Each element is an
            :class:`~tsdynamics.engine.run.Event`, a bare ``g(y, t)`` callable
            carrying ``.direction`` / ``.terminal`` attributes, or a plane tuple
            (``("y", 0.0, "up")``).  A **terminal** event stops the integration at
            its first crossing; the returned trajectory carries each event's
            crossings in ``meta["t_events"]`` / ``meta["y_events"]`` (aligned with
            ``events``) plus ``meta["terminated"]``.
        transient : float, optional
            Leading stretch of the run to discard, in **time units** (the same
            unit as ``final_time``).  The window is extended to
            ``transient + final_time`` and everything before ``t0 + transient``
            is dropped, so the returned trajectory still spans ``final_time``.
            Spelled identically on every family and every trajectory-producing
            verb — ``run`` / ``integrate`` / ``trajectory`` (and ``iterate`` on a
            map, where the unit is iterations).

            .. versionadded:: 6.0
                Previously only :meth:`trajectory` accepted it, so a user who
                found the canonical ``run`` verb could not discard a transient.

        Returns
        -------
        Trajectory
            A container of samples: ``len(traj)`` time points, iterating
            yields ``(t_i, y_i)`` pairs; ``traj.unpack()`` gives the two
            column arrays ``(t, y)``.
        """
        transient = resolve_transient(transient, discrete=False)
        if transient > 0.0:
            traj = self.integrate(
                final_time=final_time + transient,
                dt=dt,
                t0=t0,
                ic=ic,
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                backend=backend,
                seed=seed,
                events=events,
                **integrator_kwargs,
            )
            return traj.after(t0 + transient)
        if integrator_kwargs:
            # Anything left in **integrator_kwargs is an unrecognised keyword (every
            # valid argument is bound to an explicit parameter above). Reject it
            # instead of silently dropping a typo'd keyword (the WS-ERRADOPT footgun).
            from tsdynamics.errors import invalid_value

            bad = sorted(integrator_kwargs)[0]
            # The mirror of the map's *flow*-keyword refusal: a map's horizon
            # words reaching a flow must teach the same lesson from this side,
            # or the pair reads as a rule in one direction and a typo in the
            # other.
            count_words = {"n", "steps", "n_steps", "iterations"}
            hint = (
                (
                    f"{bad} is a *map* keyword: a flow runs for a span of continuous "
                    f"time, not a count of iterations, so its horizon is final_time "
                    f"(and dt is the output sampling interval)."
                )
                if bad in count_words
                else (
                    "check the keyword spelling (e.g. final_time, dt, t0, ic, method, "
                    "rtol, atol, max_step, seed)."
                )
            )
            if bad in count_words:
                from tsdynamics.errors import remedy

                hint += remedy(f"{type(self).__name__}().run(final_time=100.0, dt=0.01)")
            raise invalid_value(
                bad,
                integrator_kwargs[bad],
                rule="is not a valid integrate()/run() keyword",
                hint=hint,
            )
        if events is not None:
            return self._run_events(
                final_time=final_time,
                dt=dt,
                events=events,
                t0=t0,
                ic=ic,
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                backend=backend,
                seed=seed,
            )
        backend = backend if backend is not None else self._default_backend
        return self._dispatch(
            backend=backend,
            seed=seed,
            final_time=final_time,
            dt=dt,
            t0=t0,
            ic=ic,
            method=method or self._default_method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
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
        k: int | None = None,
        transient: float = 50.0,
        method: str | None = None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        backend: str = "jit",
        **integrator_kwargs: Any,
    ) -> Any:
        """
        Estimate the Lyapunov spectrum of the flow.

        Delegates to :class:`~tsdynamics.derived.tangent.TangentSystem`, the one
        backend-neutral variational/Lyapunov engine shared across families: the
        *extended* variational ODE (state ⊕ ``k`` tangent vectors) is integrated
        on the chosen ``backend`` per dt-chunk and QR-reorthonormalised.  The
        Benettin time-averaging of the log-stretch rates follows the classical
        construction of Benettin et al. [1]_.

        Results are stored in ``self.meta['lyapunov_spectrum']``.

        Parameters
        ----------
        final_time : float
            Averaging window length after burn-in. Default 200.0.
        dt : float
            Sampling interval for local exponent accumulation. Default 0.1.
        ic : array-like, optional
            Initial state. Falls back to ``self.ic``, then random.
        k : int, optional
            Number of exponents to compute.  Defaults to ``dim``.
            (Renamed from ``n_exp`` in v4; the old spelling was silently swallowed
            by ``**integrator_kwargs`` on this method until v6.)
        transient : float
            Discard this much time before averaging, in **time units**. Default
            50.0.  Spelled ``transient`` on every entry point in the library —
            it was ``burn_in`` here until v6, the one place the concept had a
            second name.
        method : str, optional
            Integrator (default ``"RK45"``).
        rtol, atol : float
            Tolerances.
        backend : {"jit", "interp", "reference"}, optional
            Backend on which the extended variational ODE is integrated.
            Defaults to ``"jit"`` (the Cranelift JIT, served from the
            compiled-evaluator cache).  ``"interp"`` is the SSA-tape interpreter
            (bit-for-bit identical); ``"reference"`` is the dependency-light
            pure-Python oracle, usable without the compiled wheel but not
            intended for production.  Any other name is rejected by
            :class:`~tsdynamics.derived.tangent.TangentSystem`.

            .. versionchanged:: 6.0
               Default moved from ``"interp"`` to ``"jit"`` (see
               :meth:`integrate`).

        Returns
        -------
        ndarray, shape (k,)
            Lyapunov exponents ordered from largest to smallest.

        Raises
        ------
        InvalidParameterError
            If ``k`` is given and not a positive integer.

        References
        ----------
        .. [1] G. Benettin, L. Galgani, A. Giorgilli, and J.-M. Strelcyn,
           "Lyapunov characteristic exponents for smooth dynamical systems and
           for Hamiltonian systems; a method for computing all of them,"
           *Meccanica* 15, 9-30 (1980).
        """
        if k is not None and k <= 0:
            raise InvalidParameterError(
                f"k (number of exponents) must be a positive integer, got {k!r}"
            )
        from tsdynamics.derived.tangent import TangentSystem

        _reject_unknown_lyapunov_keywords(integrator_kwargs)
        k = k if k is not None else self.dim
        exponents = TangentSystem(self, k=k, backend=backend).lyapunov_spectrum(
            final_time=final_time,
            dt=dt,
            ic=ic,
            transient=transient,
            method=method,
            rtol=rtol,
            atol=atol,
            **integrator_kwargs,
        )
        return as_lyapunov_result(
            self,
            exponents,
            final_time=final_time,
            dt=dt,
            transient=transient,
            method=method,
            backend=backend,
        )
