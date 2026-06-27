"""DiscreteMap — base class for discrete dynamical maps."""

from __future__ import annotations

import inspect
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from tsdynamics.errors import ConvergenceError, InvalidInputError, InvalidParameterError

from .base import SystemBase, Trajectory

if TYPE_CHECKING:
    from .base import ParamSet

# ---------------------------------------------------------------------------
# Signature validation helpers
# ---------------------------------------------------------------------------


def _unwrap_static(obj: Any) -> Any:
    """Peel the ``staticmethod`` wrapper off a map method to get the raw callable."""
    return getattr(obj, "__func__", obj)


def _positional_param_names(fn: Any) -> list[str] | None:
    """
    Return the parameter names after the state argument, or None if unknowable.

    ``None`` is returned for catch-all signatures like ``(X, *params)`` (the
    abstract methods on :class:`DiscreteMap`) and for callables that
    ``inspect.signature`` cannot introspect.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    names: list[str] = []
    for p in list(sig.parameters.values())[1:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return None
        names.append(p.name)
    return names


# ---------------------------------------------------------------------------
# DiscreteMap
# ---------------------------------------------------------------------------


class DiscreteMap(SystemBase):
    """
    Base class for discrete maps iterated on the engine.

    Subclass contract
    -----------------
    1. Declare ``params = {...}`` and ``dim = N``.
    2. Implement ``_step`` and ``_jacobian`` as ``@staticmethod`` static methods.
       Parameters arrive as **positional arguments** in the order they appear
       in the class-level ``params`` dict.

    Iteration
    ---------
    ``iterate`` lowers ``_step`` to an in-process IR tape and runs the engine's
    native map loop, with no warmup.  The engine reads the current parameter
    values live on every run, so a parameter change never triggers a
    re-lowering.

    Lyapunov spectrum
    -----------------
    Computed in a single forward pass via QR decomposition of the Jacobian
    product — no redundant second iteration over the trajectory.

    Examples
    --------
    >>> h = Henon()
    >>> traj = h.iterate(steps=10_000)
    >>> t_idx, X = traj          # tuple-unpack
    >>> exps = h.lyapunov_spectrum(steps=5_000)
    >>> h_variant = h.with_params(a=1.2)
    >>> traj2 = h_variant.iterate(steps=10_000)
    """

    #: Set to False on maps whose orbit visits discontinuities, where the
    #: finite-difference Jacobian validation in the test suite cannot apply.
    _jacobian_fd_check: ClassVar[bool] = True

    #: The default runtime backend (see :attr:`SystemBase._default_backend`).
    #: ``"interp"`` — the Rust engine's native map loop (the sole map backend
    #: since the M3 migration retired the v2 backends).
    _default_backend: ClassVar[str] = "interp"

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
        # Validate BEFORE super().__init_subclass__ so a failing class is
        # never registered in the system registry.
        declared = list(getattr(cls, "params", {}))
        for name in ("_step", "_jacobian"):
            method = getattr(cls, name, None)
            if method is None:
                continue
            sig_names = _positional_param_names(_unwrap_static(method))
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

    @staticmethod
    @abstractmethod
    def _jacobian(X: np.ndarray, *params: Any) -> Any:
        """
        Return the (dim × dim) Jacobian at state ``X``.

        Decorate with ``@staticmethod``.  Parameters positional, same order as
        the class-level ``params`` dict.

        Returns
        -------
        array-like of shape (dim, dim).
        """
        ...

    # ------------------------------------------------------------------ #
    # System protocol — incremental stepping
    # ------------------------------------------------------------------ #

    @property
    def is_discrete(self) -> bool:
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
        self._state_now = self.resolve_ic(u)
        self._n_now = int(t) if t is not None else 0

    def step(self, n_or_dt: int | None = None) -> np.ndarray:
        """Advance ``n`` iterations (default 1) and return the new state."""
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

    def trajectory(
        self,
        steps: int = 1000,
        *,
        transient: int = 0,
        **kwargs: Any,
    ) -> Trajectory:
        """Protocol-uniform trajectory: ``iterate`` plus optional transient drop."""
        traj = self.iterate(steps=transient + steps, **kwargs)
        return traj[transient:] if transient > 0 else traj

    # ------------------------------------------------------------------ #
    # Trajectory production — the canonical ``run`` verb
    # ------------------------------------------------------------------ #

    def run(
        self,
        n: int = 1000,
        **kwargs: Any,
    ) -> Trajectory:
        """
        Produce a trajectory — the one canonical verb for every family.

        ``run`` is the unified trajectory producer: it answers the same call for
        flows, maps, DDEs and SDEs, dispatching on :attr:`is_discrete`.  For a
        discrete map (this family) it iterates the map, so ``run`` is a thin
        alias of :meth:`iterate`.  The number of iterations is named ``n`` (the
        canonical step-count keyword), forwarded to :meth:`iterate` as its
        ``steps`` argument; every other keyword is forwarded unchanged.

        Parameters
        ----------
        n : int
            Number of iterations. Default 1000.
        **kwargs
            Forwarded verbatim to :meth:`iterate` (``ic``, ``max_retries``,
            ``backend``).

        Returns
        -------
        Trajectory
            Identical to :meth:`iterate` — ``run`` adds no behaviour.

        See Also
        --------
        iterate : The family-specific spelling (a permanent alias of ``run``).

        Examples
        --------
        >>> Henon().run(n=5000)
        >>> Lorenz().run(final_time=100, dt=0.01)   # the same verb integrates a flow
        """
        return self.iterate(steps=n, **kwargs)

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #

    def iterate(
        self,
        steps: int = 1000,
        ic: Any | None = None,
        max_retries: int = 10,
        *,
        backend: str | None = None,
    ) -> Trajectory:
        """
        Iterate the map for ``steps`` steps on the engine.

        Parameters
        ----------
        steps : int
            Number of iterations. Default 1000.
        ic : array-like, optional
            Initial state. Falls back to ``self.ic``, then random.
        max_retries : int
            Retry with a new random IC if divergence is detected (only when no
            explicit ``ic`` was given; an explicit ic that diverges raises).
        backend : str, optional
            Where the iteration runs.  Defaults to ``_default_backend``
            (``"interp"``).

            - ``"interp"`` (default) / ``"jit"`` — the Rust engine's native
              map loop (interpreter or Cranelift JIT).  Requires the compiled
              extension (:mod:`tsdynamics._rust`); until it is built these
              raise :class:`~tsdynamics.engine.run.EngineNotAvailableError`.
            - ``"reference"`` — the lowered next-state tape, iterated in pure
              Python (the dependency-light oracle the engine is validated
              against).

            Every backend lowers ``_step`` to the engine IR, so it requires a
            map whose step traces symbolically (see
            :func:`tsdynamics.engine.compile.lower_map`); piecewise or
            ``numpy``-ufunc steps raise
            :class:`~tsdynamics.engine.compile.TapeCompileError`.

        Returns
        -------
        Trajectory
            ``t`` is ``arange(steps)`` (integer step indices, not float times).
        """
        backend = backend if backend is not None else self._default_backend

        # Iterate on the Rust engine.  Preserve the random-IC retry only when no
        # explicit ``ic`` was given (a random draw can land off-basin); an
        # explicit ic that diverges raises loudly, the engine's contract.
        ic_explicit = ic is not None
        ic_arr = self.resolve_ic(ic)
        for attempt in range(max_retries):
            try:
                return self._iterate_engine(steps=steps, ic=ic_arr, backend=backend)
            except RuntimeError as exc:
                if ic_explicit or attempt == max_retries - 1:
                    raise
                # Off-basin random draw diverged; warn (not stdout) and retry from
                # a fresh random IC. Final exhaustion raises loudly below.
                warnings.warn(
                    f"{type(self).__name__}.iterate: {exc} "
                    "Retrying from a new random initial condition.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                ic_arr = np.random.rand(cast(int, self.dim))
                object.__setattr__(self, "ic", ic_arr.copy())
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
        """
        traj = self._dispatch(backend=backend, final_time=steps, ic=ic)
        # Enforce "diverge loudly" at the family boundary so every backend behaves
        # alike: the Rust engine path raises on a non-finite iterate, but the
        # pure-Python reference iterator returns the offending rows as-is — catch
        # those here rather than handing back a quietly poisoned trajectory.
        finite_rows = np.isfinite(traj.y).all(axis=1)
        if not finite_rows.all():
            bad = int(np.argmin(finite_rows))
            raise ConvergenceError(
                f"{type(self).__name__}: map diverged at iteration {bad} (backend={backend!r})."
            )
        return traj

    # ------------------------------------------------------------------ #
    # Lyapunov spectrum
    # ------------------------------------------------------------------ #

    def lyapunov_spectrum(
        self,
        steps: int = 5000,
        ic: Any | None = None,
        n_exp: int | None = None,
        reortho_interval: int = 1,
    ) -> np.ndarray:
        """
        QR-based Lyapunov spectrum.

        Delegates to :class:`~tsdynamics.derived.tangent.TangentSystem`, the one
        backend-neutral variational/Lyapunov engine shared across families — a
        single forward pass evaluating the Jacobian alongside the trajectory,
        QR-reorthonormalising every ``reortho_interval`` steps, with a random-IC
        retry on divergence.

        Results are stored in ``self.meta['lyapunov_spectrum']``.

        Parameters
        ----------
        steps : int
            Number of iterations. Default 5000.
        ic : array-like, optional
            Initial state. Falls back to ``self.ic``, then random.
        n_exp : int, optional
            Number of exponents. Defaults to ``dim``.
        reortho_interval : int
            Reorthonormalise every this many steps. Default 1.

        Returns
        -------
        ndarray, shape (n_exp,)
        """
        from tsdynamics.derived.tangent import TangentSystem

        k = n_exp or self.dim
        return TangentSystem(self, k=k).lyapunov_spectrum(
            steps=steps, ic=ic, reortho_interval=reortho_interval
        )
