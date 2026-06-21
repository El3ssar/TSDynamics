"""
Stochastic differential equation family — ``StochasticSystem`` (stream E-SDE).

The home for the SDE family base class. Per the resolved noise contract
(ROADMAP §11, **diagonal-Itô**), a stochastic system is defined by

- ``_drift(y, t, **params)`` — the deterministic part, exactly like an ODE's
  ``_equations``; and
- ``_diffusion(y, t, **params)`` — **one noise coefficient per state
  component**, each multiplying an independent Wiener increment,

so the SDE is ``dX_k = f_k(X, t) dt + g_k(X, t) dW_k`` with independent
``dW_k`` (Itô interpretation). Both are written symbolically with the SymEngine
``y``/``t`` accessors, just like every other family, and lower to instruction
tapes through :func:`tsdynamics.engine.compile.lower_sde` (drift + diffusion,
the diffusion carrying ``∂g/∂u`` for Milstein).

Solvers (the real engine, ``tsdyn-solvers``/``tsdyn-engine``): **Euler–Maruyama**
(strong order 0.5) and **Milstein** (strong order 1.0, which reads ``∂g/∂u``).
The default ``backend="interp"`` (and ``"jit"``) dispatches the two-tape SDE call
to the compiled Rust engine (``tsdynamics._rust``, the FFI surface in
``tsdyn-core``; stream E-WIRE) — the interpreter and the Cranelift JIT.
``backend="reference"`` runs a dependency-light **pure-Python reference
integrator** that mirrors the engine's semantics — the same drift/diffusion
tapes, the same diagonal Wiener substrate (a faithful port of the engine's
``SplitMix64`` / ``seed_for`` / ``fill_wiener``), and the same
per-trajectory-index seeding — reproducing the engine path bit-for-bit under a
fixed seed; it is the wheel-free oracle.

Registry note
-------------
The runtime registry's family detection keys off ``_equations`` / ``_step`` and
a fixed table of family bases (:mod:`tsdynamics.registry`); neither covers SDEs
yet, so concrete ``StochasticSystem`` subclasses are **not** auto-registered.
That is deliberate — adding the ``stochastic`` family tag is stream **C-FAM**'s
acceptance. SDE systems still integrate and lower today (the engine's
:func:`~tsdynamics.engine.problem.build_problem` detects them by duck-typing
``_drift`` / ``_diffusion``); they simply do not yet appear in
``registry.all_systems()``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from tsdynamics.utils.grids import make_output_grid

from .base import SystemBase, Trajectory

if TYPE_CHECKING:
    from tsdynamics.engine.problem import SDEProblem

__all__ = ["StochasticSystem"]

# The two diagonal-Itô schemes, mapped from user-facing aliases to a canonical
# name. The canonical names match the Rust SDE kernel registry
# (``tsdyn_solvers::sde``) so a future compiled-engine path selects the same
# kernel by the same string.
_METHODS: dict[str, str] = {
    "euler_maruyama": "euler_maruyama",
    "euler-maruyama": "euler_maruyama",
    "em": "euler_maruyama",
    "milstein": "milstein",
}

#: Schemes that read the diffusion Jacobian ``∂g/∂u`` (so the diffusion tape must
#: be lowered with it). Milstein's order-1.0 correction needs the diagonal.
_NEEDS_DIFFUSION_JACOBIAN: frozenset[str] = frozenset({"milstein"})


# ---------------------------------------------------------------------------
# Seeded RNG / Wiener substrate — a faithful port of the engine's rng.rs.
#
# Mirrors ``tsdyn-engine``'s SplitMix64 generator, the per-stream ``seed_for``
# function and the diagonal ``fill_wiener`` increment, so the reference
# integrator draws the *same integer stream* as the compiled engine and seeds an
# ensemble the same way (per trajectory index). Box–Muller uses the platform
# ``math`` library, so the normal draws are reproducible across runs but may
# differ from the Rust ``sin_cos`` in the last ULP — this is the lowering oracle,
# not a bit-for-bit cross-check of the engine (that is I-XVAL's job).
# ---------------------------------------------------------------------------

_U64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_MIX_MULT_1 = 0xBF58476D1CE4E5B9
_MIX_MULT_2 = 0x94D049BB133111EB
_F64_SCALE = 1.0 / (1 << 53)


class _SplitMix64:
    """SplitMix64 generator with a cached Box–Muller normal (port of ``rng.rs``)."""

    __slots__ = ("_state", "_cached_normal")

    def __init__(self, seed: int) -> None:
        self._state = seed & _U64
        self._cached_normal: float | None = None

    def next_u64(self) -> int:
        """Return the next 64 raw pseudo-random bits (the SplitMix64 finalizer)."""
        self._state = (self._state + _GOLDEN_GAMMA) & _U64
        z = self._state
        z = ((z ^ (z >> 30)) * _MIX_MULT_1) & _U64
        z = ((z ^ (z >> 27)) * _MIX_MULT_2) & _U64
        return z ^ (z >> 31)

    def next_f64(self) -> float:
        """Return a uniform double in ``[0, 1)`` with 53 bits of resolution."""
        return (self.next_u64() >> 11) * _F64_SCALE

    def next_normal(self) -> float:
        """Return a draw from the standard normal ``N(0, 1)`` (basic Box–Muller)."""
        if self._cached_normal is not None:
            z = self._cached_normal
            self._cached_normal = None
            return z
        u1 = 1.0 - self.next_f64()  # (0, 1] keeps ln finite
        u2 = self.next_f64()
        r = math.sqrt(-2.0 * math.log(u1))
        angle = math.tau * u2
        self._cached_normal = r * math.sin(angle)
        return r * math.cos(angle)


def _seed_for(base_seed: int, index: int) -> int:
    """Derive the seed for trajectory ``index`` from a run-wide ``base_seed``.

    The keystone of the parallel-equals-serial guarantee: a trajectory's stream
    depends only on ``(base_seed, index)``. Mirrors ``rng.rs``'s ``seed_for``.
    """
    z = (base_seed + (index * _GOLDEN_GAMMA)) & _U64
    z = ((z ^ (z >> 30)) * _MIX_MULT_1) & _U64
    z = ((z ^ (z >> 27)) * _MIX_MULT_2) & _U64
    return z ^ (z >> 31)


def _wiener(rng: _SplitMix64, dt: float, dim: int) -> np.ndarray:
    """One diagonal Wiener increment per component: ``dW_k ~ N(0, dt)``."""
    scale = math.sqrt(dt)
    return np.array([scale * rng.next_normal() for _ in range(dim)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Reference SDE kernels (pure Python; mirror the Rust kernels' update formulas)
# ---------------------------------------------------------------------------


def _sde_step(
    method: str,
    drift: Any,
    diffusion: Any,
    u: np.ndarray,
    p: np.ndarray,
    t: float,
    dw: np.ndarray,
    h: float,
) -> np.ndarray:
    """One diagonal-Itô step; the pure-Python twin of the Rust ``SdeKernel``s.

    Euler–Maruyama: ``u + f h + g ⊙ dW``. Milstein adds the diagonal correction
    ``½ g ⊙ g' ⊙ (dW² − h)`` using ``g' = diag(∂g/∂u)``.
    """
    from tsdynamics.engine.compile import eval_tape, eval_tape_jac

    f = eval_tape(drift, u, p, t)
    if method == "milstein":
        g, gjac = eval_tape_jac(diffusion, u, p, t)
        gprime = np.diagonal(gjac)
        return cast(np.ndarray, u + f * h + g * dw + 0.5 * g * gprime * (dw * dw - h))
    g = eval_tape(diffusion, u, p, t)
    return cast(np.ndarray, u + f * h + g * dw)


def _advance_to(
    method: str,
    drift: Any,
    diffusion: Any,
    u: np.ndarray,
    p: np.ndarray,
    t: float,
    t_end: float,
    dt: float,
    rng: _SplitMix64,
    name: str,
) -> tuple[np.ndarray, float]:
    """Step ``(u, t)`` to ``t_end`` with fixed step ``dt``; raise on divergence.

    Mirrors the Rust ``sde_advance_to``: a landing step is shortened to the
    remaining span (its increment drawn ``~ N(0, h)``) and the time pinned to
    ``t_end``. A non-finite state raises rather than returning silent garbage.
    """
    dim = u.size
    while t < t_end:
        remaining = t_end - t
        landing = dt >= remaining
        h = remaining if landing else dt
        dw = _wiener(rng, h, dim)
        u = _sde_step(method, drift, diffusion, u, p, t, dw, h)
        if not np.all(np.isfinite(u)):
            raise RuntimeError(f"{name}: SDE diverged — non-finite state at t={t + h:.6g}.")
        t = t_end if landing else t + h
    return u, t


# ---------------------------------------------------------------------------
# StochasticSystem
# ---------------------------------------------------------------------------


class StochasticSystem(SystemBase, ABC):
    """
    Base class for diagonal-Itô stochastic differential equations.

    Subclass contract
    -----------------
    1. Declare ``params = {...}`` and ``dim = N`` at class level.
    2. Implement ``_drift`` and ``_diffusion`` as ``@staticmethod``s, each
       returning a length-``dim`` sequence of JiTCODE / SymEngine symbolic
       expressions (use ``y(i)`` for component ``i`` and ``t`` for time — no
       NumPy, no ``math``, no Python ``if``):

       - ``_drift(y, t, **params)`` is the deterministic part ``f``;
       - ``_diffusion(y, t, **params)`` is the per-component diagonal noise
         coefficient ``g`` (so ``dX_k = f_k dt + g_k dW_k``).

    3. Optionally mark integer / loop-structural parameters in
       ``_structural_params`` (baked in at lowering time, like the ODE family).

    Example
    -------
    Geometric Brownian motion ``dX = μX dt + σX dW``::

        class GeometricBrownianMotion(StochasticSystem):
            params = {"mu": 0.1, "sigma": 0.3}
            dim = 1
            variables = ("x",)

            @staticmethod
            def _drift(y, t, mu, sigma):
                return [mu * y(0)]

            @staticmethod
            def _diffusion(y, t, mu, sigma):
                return [sigma * y(0)]

        gbm = GeometricBrownianMotion()
        traj = gbm.integrate(final_time=1.0, dt=0.01, ic=[1.0], seed=0)

    Notes
    -----
    Integration uses a fixed step (the step *is* the noise scale ``√dt``).
    ``method`` selects ``"euler_maruyama"`` (default, order 0.5) or ``"milstein"``
    (order 1.0). Pass ``seed`` for a reproducible noise realisation; the resolved
    seed is recorded in the trajectory's ``meta``.
    """

    _default_method: ClassVar[str] = "euler_maruyama"
    _default_step_dt: ClassVar[float] = 0.01

    #: The default runtime backend (see :attr:`SystemBase._default_backend`).
    #: ``"interp"`` — the Rust SDE engine.  Unlike the other families the SDE
    #: engine path does not go through ``run.integrate`` (which cannot carry the
    #: noise seed/step); it uses the dedicated ``run.sde_integrate_dense`` /
    #: ``run.sde_ensemble_final`` seam.  ``"reference"`` is the wheel-free
    #: pure-Python oracle.
    _default_backend: ClassVar[str] = "interp"

    #: Parameters whose values affect the symbolic *structure* of the dynamics
    #: (e.g. integer loop bounds); baked in at lowering time, like the ODE family.
    _structural_params: ClassVar[frozenset[str]] = frozenset()

    # Protocol stepping state (instances shadow these on first ``reinit``).
    _state_now: np.ndarray | None = None
    _t_now: float = 0.0
    _stepper: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Subclass interface
    # ------------------------------------------------------------------ #

    @staticmethod
    @abstractmethod
    def _drift(y: Any, t: Any, **params: Any) -> Sequence[Any]:
        """Return the ``dim`` symbolic drift expressions ``f_k(y, t)``."""
        ...

    @staticmethod
    @abstractmethod
    def _diffusion(y: Any, t: Any, **params: Any) -> Sequence[Any]:
        """Return the ``dim`` symbolic diagonal diffusion expressions ``g_k(y, t)``."""
        ...

    # ------------------------------------------------------------------ #
    # Lowering helpers (consumed by engine.compile.lower_sde)
    # ------------------------------------------------------------------ #

    def _structural_vals(self) -> dict[str, Any]:
        """Return the structural parameter key→value pairs (baked in at lowering)."""
        return {k: self.params[k] for k in type(self)._structural_params}

    def _control_params(self) -> dict[str, Any]:
        """Return the non-structural parameters (become tape control inputs)."""
        structural = type(self)._structural_params
        return {k: v for k, v in self.params.items() if k not in structural}

    # ------------------------------------------------------------------ #
    # System protocol
    # ------------------------------------------------------------------ #

    @property
    def is_discrete(self) -> bool:
        """SDEs are continuous-time systems."""
        return False

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict[str, Any] | None = None,
        method: str | None = None,
        seed: int | None = None,
        dt: float | None = None,
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
        method : str, optional
            ``"euler_maruyama"`` (default) or ``"milstein"``.
        seed : int, optional
            Seed for the noise stream (random if omitted) — set it for a
            reproducible path.
        dt : float, optional
            Default step size for :meth:`step` (default ``0.01``).
        """
        if params:
            for k, v in params.items():
                self.params[k] = v
        t0 = float(t) if t is not None else 0.0
        ic_arr = self.resolve_ic(u)
        canon = self._resolve_method(method)
        base_seed = _resolve_seed(seed)
        step_dt = float(dt) if dt is not None else type(self)._default_step_dt

        problem = self._problem(ic=ic_arr, t0=t0, method=canon)
        self._stepper = {
            "method": canon,
            "drift": problem.drift,
            "diffusion": problem.diffusion,
            "p": problem.params_vec(),
            "rng": _SplitMix64(base_seed),
            "seed": base_seed,
            "dt": step_dt,
        }
        self._state_now = ic_arr.copy()
        self._t_now = t0

    def step(self, n_or_dt: float | None = None) -> np.ndarray:
        """
        Advance by ``dt`` (default ``0.01``) and return the new state.

        The first call performs an implicit :meth:`reinit`. Each call draws a
        fresh diagonal Wiener increment from the stepper's seeded stream, so
        repeated stepping traces one reproducible sample path.
        """
        if self._stepper is None:
            self.reinit()
        assert self._stepper is not None
        s = self._stepper
        dt = float(n_or_dt) if n_or_dt is not None else s["dt"]
        u = np.asarray(self._state_now, dtype=float)
        dw = _wiener(s["rng"], dt, u.size)
        u = _sde_step(s["method"], s["drift"], s["diffusion"], u, s["p"], self._t_now, dw, dt)
        if not np.all(np.isfinite(u)):
            raise RuntimeError(
                f"{type(self).__name__}: SDE diverged at t={self._t_now + dt:.6g} during step()."
            )
        self._t_now += dt
        self._state_now = u.copy()
        return u

    def state(self) -> np.ndarray:
        """Return a copy of the current state (implicit ``reinit`` if cold)."""
        if self._state_now is None:
            self.reinit()
        assert self._state_now is not None
        return self._state_now.copy()

    def set_state(self, u: Any) -> None:
        """Overwrite the current state without changing the current time.

        Unlike a DDE (whose state is a whole history function), an SDE's state is
        a single Markovian point, so ``set_state`` is well-defined.
        """
        u_arr = np.asarray(u, dtype=float).reshape(self.dim)
        if self._stepper is None:
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
        **kwargs: Any,
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
        t0: float = 0.0,
        ic: Any | None = None,
        method: str | None = None,
        seed: int | None = None,
        backend: str | None = None,
    ) -> Trajectory:
        """
        Integrate the SDE and return a :class:`~tsdynamics.families.Trajectory`.

        Parameters
        ----------
        final_time : float, default 100.0
            End of the integration window.
        dt : float, default 0.02
            Fixed step size *and* output sampling interval — for an SDE the step
            is the noise scale (each increment is drawn ``~ N(0, dt)``), so the
            output grid and the discretisation share one ``dt``.
        t0 : float, default 0.0
            Start time (the IC is the state at ``t0``).
        ic : array-like, optional
            Initial state. Falls back to ``self.ic``, then ``U[0, 1)^dim``.
        method : str, optional
            ``"euler_maruyama"`` (default, order 0.5) or ``"milstein"``
            (order 1.0).
        seed : int, optional
            Seed for the noise realisation (random if omitted). The resolved seed
            is recorded in ``traj.meta["seed"]`` so a run can be reproduced.
        backend : str, optional
            Defaults to ``_default_backend`` (``"interp"``).  ``"interp"`` /
            ``"jit"`` dispatch the two-tape
            SDE call to the compiled Rust engine (:mod:`tsdynamics._rust`) via the
            dedicated ``run.sde_integrate_dense`` seam.  The engine path reproduces
            the reference under a fixed seed and raises
            :class:`~tsdynamics.engine.run.EngineNotAvailableError` if the
            extension is not built.

        Returns
        -------
        Trajectory
            Supports tuple-unpacking: ``t, y = sys.integrate(...)``.
        """
        from tsdynamics.engine import run

        backend = backend if backend is not None else self._default_backend
        canon = self._resolve_method(method)
        base_seed = _resolve_seed(seed)
        ic_arr = self.resolve_ic(ic)
        problem = self._problem(ic=ic_arr, t0=t0, method=canon)
        t_eval = make_output_grid(t0, final_time, dt)

        backend_canon = run.resolve_backend(backend)
        if backend_canon == "reference":
            rng = _SplitMix64(base_seed)
            y = self._run_reference(problem, t_eval, dt, canon, rng)
            engine = "reference"
        else:
            y = run.sde_integrate_dense(
                problem, t_eval, dt=dt, method=canon, seed=base_seed, backend=backend_canon
            )
            engine = "rust"

        return Trajectory(
            t=t_eval,
            y=y,
            system=self,
            meta=self._provenance(
                family="sde",
                engine=engine,
                backend=backend_canon,
                method=canon,
                dt=dt,
                t0=t0,
                seed=base_seed,
                ic=ic_arr.copy(),
            ),
        )

    def ensemble(
        self,
        ics: Any,
        *,
        final_time: float = 100.0,
        dt: float = 0.02,
        t0: float = 0.0,
        method: str | None = None,
        seed: int | None = None,
        backend: str | None = None,
    ) -> np.ndarray:
        """
        Integrate a batch of initial conditions and return their final states.

        ``ics`` is ``(n, dim)``; each row is integrated from ``t0`` to
        ``final_time`` and its final state returned as a row of the ``(n, dim)``
        result. Trajectory ``i`` draws its noise from a stream seeded with
        ``seed_for(seed, i)`` — depending only on the index — so the batch is
        **reproducible** and matches the compiled engine's per-index seeding
        (the parallel-equals-serial contract). A diverging trajectory yields a
        row of ``NaN`` rather than aborting the batch.

        Parameters
        ----------
        ics : array-like, shape (n, dim)
            The batch of initial conditions.
        final_time, dt, t0, method, seed
            As in :meth:`integrate` (``seed`` is the ensemble's base seed).
        backend : str, optional
            Defaults to ``_default_backend`` (``"interp"``).  ``"interp"`` /
            ``"jit"`` fan the batch out on the compiled engine's rayon pool;
            ``"reference"`` is a pure-Python loop.  All seed each trajectory by index, so the final
            states match across backends to floating-point tolerance.

        Returns
        -------
        ndarray, shape (n, dim)
            Final states (rows of ``NaN`` for diverged trajectories).
        """
        from tsdynamics.engine import run

        backend = backend if backend is not None else self._default_backend
        canon = self._resolve_method(method)
        base_seed = _resolve_seed(seed)
        ics = np.ascontiguousarray(ics, dtype=np.float64)
        if ics.ndim != 2 or ics.shape[1] != self.dim:
            raise ValueError(f"ics must be (n, {self.dim}); got shape {ics.shape}")

        backend_canon = run.resolve_backend(backend)
        if backend_canon != "reference":
            problem = self._problem(ic=ics[0], t0=t0, method=canon)
            return run.sde_ensemble_final(
                problem,
                ics,
                t0=t0,
                t1=float(final_time),
                dt=dt,
                method=canon,
                seed=base_seed,
                backend=backend_canon,
            )

        problem = self._problem(ic=ics[0], t0=t0, method=canon)
        drift, diffusion = problem.drift, problem.diffusion
        p = problem.params_vec()
        out = np.empty_like(ics)
        for i, ic in enumerate(ics):
            rng = _SplitMix64(_seed_for(base_seed, i))
            try:
                u, _ = _advance_to(
                    canon,
                    drift,
                    diffusion,
                    np.asarray(ic, dtype=float),
                    p,
                    t0,
                    float(final_time),
                    dt,
                    rng,
                    type(self).__name__,
                )
                out[i] = u
            except RuntimeError:
                out[i] = np.nan
        return cast(np.ndarray, out)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _problem(self, *, ic: np.ndarray, t0: float, method: str) -> SDEProblem:
        """Build the :class:`~tsdynamics.engine.problem.SDEProblem` for ``method``."""
        from tsdynamics.engine.problem import sde_problem

        return sde_problem(
            self,
            ic=ic,
            t0=t0,
            with_diffusion_jacobian=method in _NEEDS_DIFFUSION_JACOBIAN,
        )

    def _run_reference(
        self, problem: Any, t_eval: np.ndarray, dt: float, method: str, rng: _SplitMix64
    ) -> np.ndarray:
        """Integrate the lowered drift/diffusion tapes on a grid (pure Python)."""
        drift, diffusion = problem.drift, problem.diffusion
        p = problem.params_vec()
        y = np.empty((t_eval.size, cast(int, self.dim)), dtype=np.float64)
        u = np.asarray(problem.ic, dtype=float).reshape(self.dim)
        y[0] = u
        t = float(t_eval[0])
        for k in range(1, t_eval.size):
            u, t = _advance_to(
                method,
                drift,
                diffusion,
                u,
                p,
                t,
                float(t_eval[k]),
                dt,
                rng,
                type(self).__name__,
            )
            y[k] = u
        return y

    @staticmethod
    def _resolve_method(method: str | None) -> str:
        """Canonicalise a scheme name; raise on an unknown one."""
        if method is None:
            return StochasticSystem._default_method
        canon = _METHODS.get(str(method).lower())
        if canon is None:
            raise ValueError(
                f"unknown SDE method {method!r}; choose from "
                f"{sorted(set(_METHODS.values()))} (aliases: {sorted(_METHODS)})."
            )
        return canon


def _resolve_seed(seed: int | None) -> int:
    """Resolve a user seed to a concrete ``u64`` (random when omitted)."""
    if seed is not None:
        return int(seed) & _U64
    return int(np.random.randint(0, 1 << 63, dtype=np.int64))


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
