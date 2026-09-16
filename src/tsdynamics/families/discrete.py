"""DiscreteMap — base class for discrete dynamical maps."""

from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from tsdynamics.errors import (
    ConvergenceError,
    InvalidInputError,
    InvalidParameterError,
    remedy,
)

from ._kwargs import reject_unknown_run_keywords
from .base import Absent, SystemBase, Trajectory, as_lyapunov_result, resolve_transient

#: ``DiscreteMap.run``'s keywords, in signature order.
_MAP_RUN_KEYWORDS = ("steps", "ic", "transient", "backend", "seed", "max_retries")

if TYPE_CHECKING:
    from .base import ParamSet

# ---------------------------------------------------------------------------
# Signature validation helpers
# ---------------------------------------------------------------------------


def _unwrap_static(obj: Any) -> Any:
    """Peel the ``staticmethod`` wrapper off a map method to get the raw callable."""
    return getattr(obj, "__func__", obj)


class _KeywordOnlyParamError(Exception):
    """A map kernel declares a keyword-only parameter (cannot bind positionally)."""

    def __init__(self, param_name: str) -> None:
        self.param_name = param_name
        super().__init__(param_name)


def _positional_param_names(fn: Any) -> list[str] | None:
    """
    Return the parameter names after the state argument, or None if unknowable.

    ``None`` is returned for catch-all signatures like ``(X, *params)`` (the
    abstract methods on :class:`DiscreteMap`) and for callables that
    ``inspect.signature`` cannot introspect.

    Raises
    ------
    _KeywordOnlyParamError
        If the signature carries a keyword-only parameter after the state
        argument.  The engine calls a kernel positionally as
        ``step_fn(x, *params)``, so a keyword-only parameter can never receive
        its value — that is an unconditional contract violation (independent of
        whether the names happen to match the ``params`` dict), surfaced as an
        import-time error by the caller.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    names: list[str] = []
    for p in list(sig.parameters.values())[1:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return None
        if p.kind == p.KEYWORD_ONLY:
            raise _KeywordOnlyParamError(p.name)
        names.append(p.name)
    return names


# ---------------------------------------------------------------------------
# DiscreteMap
# ---------------------------------------------------------------------------


class DiscreteMap(SystemBase, ABC):
    """
    Base class for discrete maps iterated on the engine.

    Subclass contract
    -----------------
    1. Declare ``params = {...}`` and ``dim = N``.
    2. Implement ``_step`` as a ``@staticmethod``.  Parameters arrive as
       **positional arguments** in the order they appear in the class-level
       ``params`` dict.  It is :func:`abc.abstractmethod` and — since this class
       is an :class:`abc.ABC`, like every other family base — a subclass that
       omits it **cannot be instantiated**.  (Before v6 ``DiscreteMap`` was the
       one family base that was *not* an ABC, so the abstract marker was inert
       and a missing kernel surfaced far downstream as a mystifying
       ``TapeCompileError`` from the lowering pass.)
    3. ``_jacobian`` is **optional**: by default the library differentiates
       ``_step`` symbolically, exactly as ``ContinuousSystem`` autogenerates its
       Jacobian from ``_equations``.  Write one only when the step cannot be
       traced (a Python ``if`` on the state) or when a one-sided slope on a
       discontinuity is the meaningful answer.  Every catalogue map still ships
       a hand-written one, which the test suite uses as a cross-check oracle.

    Iteration
    ---------
    ``iterate`` lowers ``_step`` to an in-process IR tape and runs the engine's
    native map loop, on a tape lowered and JIT-compiled once per system (both
    memoised).  The engine reads the current parameter
    values live on every run, so a parameter change never triggers a
    re-lowering.

    Lyapunov spectrum
    -----------------
    Computed in a single forward pass via QR decomposition of the Jacobian
    product — no redundant second iteration over the trajectory.

    Examples
    --------
    >>> h = Henon()
    >>> traj = h.run(steps=10_000)
    >>> t_idx, X = traj.unpack()   # the two columns
    >>> exps = h._lyapunov_spectrum(n=5_000)
    >>> h_variant = h.with_params(a=1.2)
    >>> traj2 = h_variant.run(steps=10_000)
    """

    #: Set to False on maps whose orbit visits discontinuities, where the
    #: finite-difference Jacobian validation in the test suite cannot apply.
    _jacobian_fd_check: ClassVar[bool] = True

    #: The default runtime backend (see :attr:`SystemBase._default_backend`).
    #: ``"jit"`` — the Rust engine's native map loop driven by the Cranelift JIT,
    #: with the compiled-evaluator cache paying the compile once per distinct
    #: system.  Was ``"interp"`` before v6.
    _default_backend: ClassVar[str] = "jit"

    # Protocol stepping state (instances shadow these class defaults).
    _state_now: np.ndarray | None = None
    _n_now: int = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Validate the subclass contract at class-definition time.

        ``_step`` / ``_jacobian`` receive parameters *positionally* in the
        insertion order of the class-level ``params`` dict.  A mismatch
        between the method signature and the dict order silently swaps
        parameter values (this bit the Circle map once), so it is promoted
        to an import-time ``TypeError``.
        """
        # Adopt a kernel written as an ordinary method *first*, so the signature
        # this validates is the corrected one (a stray ``self`` would otherwise
        # be reported as a parameter-order mismatch).
        cls._adopt_class_called_kernels()
        # Validate BEFORE super().__init_subclass__ so a failing class is
        # never registered in the system registry.
        declared = list(getattr(cls, "params", {}))
        for name in ("_step", "_jacobian"):
            method = getattr(cls, name, None)
            if method is None:
                continue
            try:
                sig_names = _positional_param_names(_unwrap_static(method))
            except _KeywordOnlyParamError as exc:
                raise InvalidInputError(
                    f"{cls.__name__}.{name} declares keyword-only parameter "
                    f"{exc.param_name!r}, but map kernels are called positionally as "
                    f"step_fn(x, *params) — a keyword-only parameter can never receive "
                    f"its value. Make every parameter positional-or-keyword."
                ) from None
            if sig_names is None:  # catch-all (X, *params) or non-introspectable
                continue
            if sig_names != declared:
                raise InvalidInputError(
                    f"{cls.__name__}.{name} takes parameters {sig_names} but the "
                    f"params dict declares {declared} — names and ORDER must match, "
                    f"because parameters are passed positionally."
                )
        super().__init_subclass__(**kwargs)

    # ------------------------------------------------------------------ #
    # Subclass interface
    # ------------------------------------------------------------------ #

    @staticmethod
    @abstractmethod
    def _step(X: np.ndarray, *params: Any) -> Any:
        """
        Evaluate the map at state ``X``.

        Decorate with ``@staticmethod``.  Parameters arrive positionally in
        the order they appear in the class-level ``params`` dict.

        Parameters
        ----------
        X : ndarray, shape (dim,)
            Current state.
        *params
            Parameter values in declaration order.

        Returns
        -------
        array-like of shape (dim,).
        """
        ...

    def jacobian(self, u: Any, t: float = 0.0) -> np.ndarray:
        """Return the tangent map ``J = df/dx`` at state ``u``.

        The same verb, the same signature and the same return type as a flow's
        :meth:`~tsdynamics.families.ContinuousSystem.jacobian` — a map simply
        has no ``t`` dependence, so the argument is accepted and ignored.

        Parameters
        ----------
        u : array-like, shape (dim,)
            The state to linearise at.
        t : float, optional
            Ignored; present so a caller holding *any* system can write
            ``sys.jacobian(u, t)``.

        Returns
        -------
        ndarray, shape (dim, dim)

        Examples
        --------
        >>> import tsdynamics as ts
        >>> ts.systems.Henon().jacobian([0.5, 0.2]).shape
        (2, 2)
        """
        state = np.asarray(u, dtype=float)
        jac = type(self)._jacobian(state, *cast(Any, self.params).as_tuple())
        n = int(cast(int, self.dim))
        return np.asarray(jac, dtype=float).reshape(n, n)

    @classmethod
    def _jacobian(cls, X: np.ndarray, *params: Any) -> Any:
        """
        Return the (dim × dim) Jacobian at state ``X``.

        **Optional.**  By default it is derived symbolically from ``_step`` —
        the same way :class:`~tsdynamics.families.ContinuousSystem`
        autogenerates its Jacobian from ``_equations`` — so defining a map means
        writing ``_step`` and nothing else.

        Override it (as a ``@staticmethod``, parameters positional in the order
        of the class-level ``params`` dict) when the derivative cannot be taken
        from the step: a map that branches on the state with a Python ``if``, or
        one living on a discontinuity where a one-sided slope is the meaningful
        answer.  A hand-written override always wins, and the test suite
        cross-checks it against a finite difference of ``_step``
        (:attr:`_jacobian_fd_check`).

        Returns
        -------
        array-like of shape (dim, dim).
        """
        from tsdynamics.engine.compile import map_jacobian_fn

        # ``_jacobian`` is called off the CLASS (``type(sys)._jacobian(x, *params)``),
        # so a map whose dimension is only fixed per instance has no dimension to
        # trace against here.  Say so, rather than failing inside the trace.
        dim = cls.dim
        if dim is None:
            raise InvalidInputError(
                f"{cls.__name__}: cannot autogenerate `_jacobian` because `dim` is set "
                f"per instance, and the Jacobian is derived off the class. Declare a "
                f"class-level `dim`, or write the Jacobian yourself:\n"
                f"    @staticmethod\n"
                f"    def _jacobian(X, {', '.join(cls.params) or '...'}): ..."
            )
        return map_jacobian_fn(cls, params, dim=dim)(X)

    # ------------------------------------------------------------------ #
    # System protocol — incremental stepping
    # ------------------------------------------------------------------ #

    #: The one-word family, read by :attr:`SystemBase.family`.
    _family: ClassVar[str] = "map"

    #: A section is the transversal crossing of a CONTINUOUS trajectory.
    poincare = Absent(
        "a section is the transversal crossing of a CONTINUOUS trajectory, and a "
        "map has no in-between to cross",
        "ts.analysis.orbit_diagram(system, 'a', values)",
    )

    #: A map's kernel is traced numerically, not held as a symbolic tree.
    jacobian_sym = Absent(
        "a map's kernel is traced numerically, not held as a symbolic tree",
        "system.jacobian(x)",
    )

    @property
    def _is_discrete(self) -> bool:
        """Maps are discrete-time systems."""
        return True

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """(Re)start stepping from state ``u`` at iteration count ``t``."""
        if params:
            for k, v in params.items():
                self.params[k] = v
        # ``resolve_ic`` commits the resolved IC to ``self.ic`` up front, so a
        # malformed ``u`` (wrong length → a reshape ValueError) or a malformed
        # ``t`` must not leave a half-applied IC behind (the ``_dispatch`` /
        # ``iterate`` contract, applied to the stepping entry point).
        with self._ic_rollback():
            state = self._resolve_ic(u)
            n_now = int(t) if t is not None else 0
        self._state_now = state
        self._n_now = n_now

    def step(self, n_or_dt: int | None = None) -> np.ndarray:
        """
        Advance ``n`` iterations and return the new state.

        The first call performs an implicit :meth:`reinit`.

        Parameters
        ----------
        n_or_dt : int, optional
            Number of iterations to advance (default 1).  Must be a positive
            whole number — a discrete map has no notion of a fractional step.

        Returns
        -------
        ndarray, shape (dim,)
            A copy of the state after ``n`` iterations.

        Raises
        ------
        InvalidParameterError
            If ``n_or_dt`` is non-positive or not a whole number.
        ConvergenceError
            If the orbit diverges to a non-finite state within the ``n`` steps.
        """
        if self._state_now is None:
            self.reinit()
        if n_or_dt is None:
            n = 1
        else:
            nf = float(n_or_dt)
            if not nf.is_integer() or nf < 1:
                raise InvalidParameterError(
                    f"{type(self).__name__}.step takes a positive whole number of "
                    f"iterations, got {n_or_dt!r} (fractional time steps have no "
                    f"meaning for discrete maps)."
                )
            n = int(nf)
        assert self._state_now is not None
        x = self._state_now
        params = cast("ParamSet", self.params).as_tuple()

        step_fn = type(self)._step
        # A diverging orbit overflows to ``inf`` in pure-Python float64 arithmetic;
        # that is *expected* and is caught by the finite check below, so silence the
        # spurious NumPy over/under/invalid FP warnings the loop would otherwise emit
        # (under ``filterwarnings=error`` they would mask the real divergence signal).
        with np.errstate(all="ignore"):
            for _ in range(n):
                x = np.asarray(step_fn(x, *params), dtype=np.float64).ravel()
        if not np.isfinite(x).all():
            raise ConvergenceError(
                f"{type(self).__name__}: map diverged at iteration {self._n_now + n}."
            )
        self._state_now = np.asarray(x, dtype=float).reshape(self.dim)
        self._n_now += n
        return self._state_now.copy()

    def state(self) -> np.ndarray:
        """Return a copy of the current state (implicit ``reinit`` if cold)."""
        if self._state_now is None:
            self.reinit()
        assert self._state_now is not None
        return self._state_now.copy()

    def set_state(self, u: Any) -> None:
        """Overwrite the current state."""
        self._state_now = np.asarray(u, dtype=float).reshape(self.dim)

    def time(self) -> float:
        """Return the current iteration count."""
        return float(self._n_now)

    def _resolve_iteration_count(
        self, n: int | None, kwargs: dict[str, Any], *, where: str, default: int
    ) -> int:
        """Read a map's horizon from ``n`` or its ``steps`` alias, never both.

        ``iterate`` names the iteration count ``steps`` and ``run`` names it
        ``n``.  One concept, so **every** map horizon door accepts either word
        and resolves it here — otherwise ``steps=`` collides with the forward on
        one door and surfaces as a raw "got multiple values for keyword argument
        'steps'" ``TypeError``, or (worse) as an "unexpected keyword argument"
        on a door that simply forgot the alias.

        Each door declares ``n: int | None = None`` and names its default here,
        so "the user passed both" is always distinguishable from "the user
        passed neither".  ``kwargs`` is consumed in place: ``steps`` is popped.
        """
        alias = kwargs.pop("steps", None)
        if alias is not None and n is not None:
            raise InvalidParameterError(
                f"n and steps are the same argument (the iteration count), so pass "
                f"only one; got n={n!r} and steps={alias!r}."
                + remedy(f"{type(self).__name__}().{where}(n={default})")
            )
        count = n if n is not None else alias
        return int(default if count is None else count)

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #

    def run(
        self,
        steps: int = 1000,
        *,
        ic: Any | None = None,
        transient: int = 0,
        backend: str | None = None,
        seed: int | None = None,
        max_retries: int = 10,
        **solver_options: Any,
    ) -> Trajectory:
        """
        Iterate the map and return a :class:`~tsdynamics.families.Trajectory`.

        ``run`` is **the** trajectory verb — one word on flows, maps, delay and
        stochastic systems.  ``iterate`` and ``trajectory`` were two more names
        for this method and are gone in v6.

        A map's horizon is ``steps``, a **count of iterations**, because a map
        has no continuous time; ``final_time`` is refused by name rather than
        reinterpreted.  The positional form reads the same on both families:
        ``hen.run(1000)``, ``lor.run(100.0)``.

        Parameters
        ----------
        steps : int
            Number of iterations. Default 1000.
        ic : array-like, optional
            Initial state. Falls back to ``self.ic``, then random.
        max_retries : int
            Retry with a new random IC if divergence is detected — only when the
            initial condition was **not chosen by the user**.  An explicit ``ic``
            (here *or* on the constructor) that diverges raises instead of being
            silently swapped for a random one that traces a different orbit.
        seed : int, optional
            Seed for the random-IC fallback and for the divergence retries, so an
            unseeded-IC run is reproducible.  Equivalent to the constructor's
            ``seed=`` (see :meth:`SystemBase.ic_generator`); the resolved seed is
            recorded on ``traj.meta["ic_seed"]``.
        backend : {"jit", "interp", "reference"}, optional
            Where the iteration runs.  Defaults to ``_default_backend``
            (``"jit"``).

            - ``"jit"`` (default) — the Rust engine's native map loop driven by
              the **Cranelift JIT**: the lowered tape compiled to native code and
              served from a process-wide compiled-evaluator cache, so the compile
              is paid once per distinct system rather than on every call.
            - ``"interp"`` — the same native map loop driven by the **SSA-tape
              interpreter**.  Bit-for-bit identical to the JIT; marginally faster
              on very small tapes, and the way to skip the one-off compile.
              Both require the compiled extension (:mod:`tsdynamics._rust`);
              until it is built they raise
              :class:`~tsdynamics.engine.run.EngineNotAvailableError`.
            - ``"reference"`` — the lowered next-state tape, iterated in pure
              Python.  Not for production use: it is the dependency-light oracle
              the engine is validated against.

            .. versionchanged:: 6.0
               The default moved from ``"interp"`` to ``"jit"``, once the v6
               compiled-evaluator cache removed the JIT's per-call recompile.

            Every backend lowers ``_step`` to the engine IR, so it requires a
            map whose step traces symbolically (see
            :func:`tsdynamics.engine.compile.lower_map`); piecewise or
            ``numpy``-ufunc steps raise
            :class:`~tsdynamics.engine.compile.TapeCompileError`.
        transient : int, optional
            Leading stretch of the orbit to discard, in **iterations** (the same
            unit as ``steps``).  ``transient + steps`` iterations are run and the
            first ``transient`` dropped, so the returned trajectory still has
            ``steps`` samples.  Spelled identically on every family and every
            trajectory-producing verb — ``run`` / ``iterate`` / ``trajectory``
            (on a flow the unit is time).

            .. versionadded:: 6.0
                Previously only :meth:`trajectory` accepted it, so a user who
                found the canonical ``run`` verb could not discard a transient.

        Returns
        -------
        Trajectory
            ``t`` is ``arange(steps)`` (integer step indices, not float times).

        Raises
        ------
        ConvergenceError
            If an explicit ``ic`` diverges, or every random-IC retry diverges.
        EngineNotAvailableError
            If a Rust-engine backend (``"interp"`` / ``"jit"``) is requested but
            the compiled extension is not built.  This propagates immediately
            (it is not divergence) rather than consuming the retry budget.
        TapeCompileError
            If ``_step`` cannot be lowered to the engine IR (piecewise / ufunc).
        InvalidParameterError
            If ``steps < 1``, or an unrecognised keyword is passed — including a
            *flow* keyword (``final_time`` / ``dt`` / ``method`` / a tolerance),
            which a map has no meaning for.  ``**kwargs`` exists only to catch
            those: nothing is silently dropped.
        """
        reject_unknown_run_keywords(self, solver_options, family="map", accepted=_MAP_RUN_KEYWORDS)
        steps = int(steps)
        if steps < 1:
            raise InvalidParameterError(
                f"steps is a number of iterations, so it must be >= 1; got {steps}."
                + remedy(f"{type(self).__name__}().run(steps=1000)")
            )
        drop = int(resolve_transient(transient, discrete=True))
        backend = backend if backend is not None else self._default_backend

        with self._ic_rollback():
            traj = self._iterate_with_retries(
                steps=steps + drop, ic=ic, max_retries=max_retries, backend=backend, seed=seed
            )
        return traj[drop:] if drop > 0 else traj

    def _iterate_with_retries(
        self, *, steps: int, ic: Any | None, max_retries: int, backend: str, seed: int | None
    ) -> Trajectory:
        """Run :meth:`iterate`'s retry loop (wrapped by its IC rollback guard)."""
        # Iterate on the Rust engine.  Preserve the random-IC retry only when the
        # initial condition was not chosen by the user (a random draw can land
        # off-basin); a *user* ic — passed here or to the constructor — that
        # diverges raises loudly, the engine's contract.  ``resolve_ic`` records
        # which of the two it resolved on ``_ic_explicit``.
        ic_arr = self._resolve_ic(ic, seed=seed)
        ic_explicit = ic is not None or bool(self.__dict__.get("_ic_explicit", False))
        for attempt in range(max_retries):
            try:
                return self._iterate_engine(steps=steps, ic=ic_arr, backend=backend)
            except (ConvergenceError, ArithmeticError) as exc:
                # Catch ONLY divergence — :class:`ConvergenceError` (the engine /
                # reference "diverge loudly" signal, also a ``RuntimeError``) and the
                # arithmetic blow-ups (``OverflowError`` / ``FloatingPointError`` /
                # ``ZeroDivisionError``, all :class:`ArithmeticError`).  A missing /
                # broken engine surfaces as
                # :class:`~tsdynamics.engine.run.EngineNotAvailableError` (a
                # :class:`~tsdynamics.errors.BackendError`, hence a ``RuntimeError`` but
                # NOT a ``ConvergenceError``); narrowing the catch lets it — and any
                # other genuine fault, e.g. a ``backend="jit"`` compile failure —
                # propagate loudly instead of being mistaken for divergence and
                # silently burning the whole retry budget.
                if ic_explicit:
                    # Re-raise the engine seam's own message verbatim (callers pin
                    # it) and *annotate* why no retry happened: the initial
                    # condition was chosen by the user — here or on the
                    # constructor — and an explicit IC is never silently swapped
                    # for a random one that would trace a different orbit.
                    exc.add_note(
                        f"The initial condition {np.array2string(ic_arr, precision=6)} "
                        f"was supplied explicitly, so {type(self).__name__}.iterate did "
                        f"not retry from a random one: pass a different ic=, or omit it "
                        f"to let the random-IC retry find the attractor."
                    )
                    raise
                if attempt == max_retries - 1:
                    raise
                # Off-basin random draw diverged; warn (not stdout) and retry from
                # a fresh random IC. Final exhaustion raises loudly below.
                warnings.warn(
                    f"{type(self).__name__}.iterate: {exc} "
                    "Retrying from a new random initial condition.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                ic_arr = self._ic_generator().random(cast(int, self.dim))
                object.__setattr__(self, "ic", ic_arr.copy())
                object.__setattr__(self, "_ic_explicit", False)
        raise ConvergenceError(
            f"{type(self).__name__}.iterate exhausted {max_retries} "
            f"retries without a finite trajectory."
        )

    def _iterate_engine(self, *, steps: int, ic: Any | None, backend: str) -> Trajectory:
        """Iterate on the Rust engine (or its pure-Python reference evaluator).

        Routes through the shared engine-dispatch seam
        (:meth:`SystemBase._dispatch` → :func:`tsdynamics.engine.run.integrate`),
        which lowers ``_step`` to the engine IR and runs the native map loop
        (stream E-MAP).  This is the seam that makes a map iterate on the same
        engine as every other family.

        Divergence is reported (it is not silently returned and there is no
        random-IC retry — the engine's "diverge loudly" contract); the
        random-IC retry lives in :meth:`iterate` for the implicit-ic case.

        Notes
        -----
        There is deliberately **no finiteness scan here**.  Every backend already
        diverges loudly *before* returning — the Rust map loop raises
        ``EngineError::Diverged`` → :class:`~tsdynamics.errors.ConvergenceError`
        at the first non-finite iterate, ``_reference_map`` raises per-iterate,
        and :func:`tsdynamics.engine._families._run_map` keeps one full
        ``np.all(np.isfinite(...))`` guard at the engine seam covering *all*
        callers of the map path.  A second, row-wise
        ``np.isfinite(traj.y).all(axis=1)`` here was unreachable (no backend has
        ever reached it) and cost ~12 ms of a 25 ms 1e6-step Hénon run — a 48%
        Python tax over a 9 ms Rust kernel — so it was removed.  Divergence
        behaviour is unchanged: the message a caller sees is, as before, the one
        ``_run_map`` (engine) or ``_reference_map`` (reference) raises, and the
        random-IC retry in :meth:`iterate` catches the same
        :class:`ConvergenceError` type from the same place.
        """
        return self._dispatch(backend=backend, final_time=steps, ic=ic)

    # ------------------------------------------------------------------ #
    # Lyapunov spectrum
    # ------------------------------------------------------------------ #

    def _lyapunov_spectrum(
        self,
        n: int | None = None,
        ic: Any | None = None,
        k: int | None = None,
        reortho_interval: int = 1,
        *,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        QR-based Lyapunov spectrum.

        Delegates to :class:`~tsdynamics.derived.tangent.TangentSystem`, the one
        backend-neutral variational/Lyapunov engine shared across families — a
        single forward pass evaluating the Jacobian alongside the trajectory,
        QR-reorthonormalising every ``reortho_interval`` steps, with a random-IC
        retry on divergence.

        On the compiled-engine backends (``"jit"`` default / ``"interp"``) the whole
        QR tangent-map iteration runs in one Rust kernel call
        (:func:`tsdynamics.engine.run.map_lyapunov`) — no per-step Python→FFI
        round-trip, so it is dramatically faster than the per-step NumPy loop.
        ``backend="reference"`` (and any map whose ``_step`` will not lower to the
        engine IR, or a wheel-free environment) runs the pure-Python QR loop — the
        oracle the engine is validated against.



        Parameters
        ----------
        n : int
            Number of iterations — a map's horizon word, exactly as on
            :meth:`run`. Default 5000.  ``steps=`` is accepted as its alias
            here too (:meth:`iterate` names it that), so the same word works at
            every map horizon door; passing both raises.
        ic : array-like, optional
            Initial state. Falls back to ``self.ic``, then random.
        k : int, optional
            Number of exponents to compute.  Defaults to ``dim``.
            (Renamed from ``n_exp`` in v4; the old spelling was silently swallowed
            by ``**integrator_kwargs`` on this method until v6.)
        reortho_interval : int
            Reorthonormalise every this many steps. Default 1.
        backend : {"jit", "interp", "reference"}, optional
            ``"jit"`` (default, the Rust kernel on Cranelift-compiled code) /
            ``"interp"`` (the same kernel on the SSA-tape interpreter,
            bit-for-bit identical) / ``"reference"`` (the pure-Python QR loop —
            the oracle, not for production use).

            .. versionchanged:: 6.0
               Default moved from ``"interp"`` to ``"jit"`` (see :meth:`iterate`).

        Returns
        -------
        ndarray, shape (k,)
            Lyapunov exponents ordered from largest to smallest.

        References
        ----------
        .. [1] G. Benettin, L. Galgani, A. Giorgilli, and J.-M. Strelcyn,
           "Lyapunov characteristic exponents for smooth dynamical systems and
           for Hamiltonian systems; a method for computing all of them,"
           *Meccanica* 15, 9-30 (1980).
        """
        from tsdynamics.derived.tangent import TangentSystem

        n = self._resolve_iteration_count(n, kwargs, where="lyapunov_spectrum", default=5000)
        k = k or self.dim
        exponents = TangentSystem(self, k=k, backend=backend)._lyapunov_spectrum(
            n=n, ic=ic, reortho_interval=reortho_interval
        )
        return as_lyapunov_result(
            self, exponents, n=n, reortho_interval=reortho_interval, backend=backend
        )
