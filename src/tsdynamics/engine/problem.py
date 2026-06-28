"""Problem builders — bundle a compiled tape with its runtime data, per family.

A :class:`Tape` (from :mod:`tsdynamics.engine.compile`) is pure dynamics: the
straight-line instructions for a right-hand side.  To *run* a system the engine
also needs the runtime context — the initial state, the control-parameter
values, the start time, and any family-specific structure (a DDE's delay slots,
an SDE's diffusion tape).  A **Problem** is that bundle: the immutable,
engine-ready description of one integrable system, the unit
:mod:`tsdynamics.engine.run` hands across the FFI boundary.

There is one Problem type per family — :class:`ODEProblem`, :class:`MapProblem`,
:class:`DDEProblem`, :class:`SDEProblem` — each built from a system instance by
the matching ``*_problem`` factory (or the family-dispatching
:func:`build_problem`).  Control-parameter *values* are read live from the
system at :meth:`~ODEProblem.params_vec` time, not snapshotted, so a
control-parameter sweep reuses the same compiled tape with no recompile — change
``system.params`` and rebuild only the (cheap) parameter vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .compile import (
    DelaySlot,
    LoweredSDE,
    Tape,
    lower_dde_cached,
    lower_map_cached,
    lower_ode_cached,
    lower_sde_cached,
)

__all__ = [
    "DDEProblem",
    "DelaySlot",
    "MapProblem",
    "ODEProblem",
    "Problem",
    "SDEProblem",
    "build_problem",
    "dde_problem",
    "map_problem",
    "ode_problem",
    "sde_problem",
]


# ---------------------------------------------------------------------------
# Problem types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ODEProblem:
    """An ODE ready to integrate: a lowered RHS tape plus its runtime context.

    Attributes
    ----------
    tape : Tape
        The lowered RHS (carries the analytic Jacobian when built with
        ``with_jacobian=True``).
    ic : ndarray, shape (dim,)
        The resolved initial state.
    t0 : float
        Start time.
    system : object
        Back-reference to the source system; control-parameter values are read
        from it live (see :meth:`params_vec`).
    """

    tape: Tape
    ic: np.ndarray
    t0: float = 0.0
    system: Any = None
    family: str = field(default="ode", init=False)

    @property
    def dim(self) -> int:
        """State-space dimension."""
        return self.tape.dim

    def params_vec(self) -> np.ndarray:
        """Return the control-parameter values in ``tape.control_names`` order.

        Read live from ``system.params`` so a control-parameter sweep needs no
        recompile — only this (cheap) vector changes.
        """
        return _params_vec(self.system, self.tape)


@dataclass(frozen=True)
class MapProblem:
    """A discrete map ready to iterate: a lowered next-state tape plus context.

    Map parameters are folded into the tape as constants (``tape.n_param == 0``),
    so there is no runtime parameter vector — a parameter change rebuilds the
    tape.

    Attributes
    ----------
    tape : Tape
        The lowered next-state map (carries the step Jacobian when built with
        ``with_jacobian=True``).
    ic : ndarray, shape (dim,)
        The resolved initial state.
    n0 : int
        Starting iteration index.
    system : object
        Back-reference to the source map.
    """

    tape: Tape
    ic: np.ndarray
    n0: int = 0
    system: Any = None
    family: str = field(default="map", init=False)

    @property
    def dim(self) -> int:
        """State-space dimension."""
        return self.tape.dim

    def params_vec(self) -> np.ndarray:
        """Empty — map parameters are folded into the tape as constants."""
        return np.empty(0, dtype=np.float64)


@dataclass(frozen=True)
class DDEProblem:
    """A delay system ready to integrate: an extended-input RHS tape + delay slots.

    The tape is an ordinary RHS over ``dim + len(delay_slots)`` inputs; the
    engine fills inputs ``dim … dim + n_slots - 1`` each step from the history
    buffer, as directed by :attr:`delay_slots`.  Parameters are folded into the
    tape (``tape.n_param == 0``): a delay value bakes into the tape, so a DDE
    re-lowers on any parameter change and carries no runtime parameter vector.

    Attributes
    ----------
    tape : Tape
        The lowered RHS over the extended (state + delay-slot) input space.
    delay_slots : list[DelaySlot]
        One per extra input: which component is delayed and by how much.
    ic : ndarray, shape (dim,)
        Constant-past state (used when no history callable is supplied at run).
    t0 : float
        Start time.
    system : object
        Back-reference to the source system.
    """

    tape: Tape
    delay_slots: list[DelaySlot]
    ic: np.ndarray
    t0: float = 0.0
    system: Any = None
    family: str = field(default="dde", init=False)

    @property
    def dim(self) -> int:
        """State-space dimension (the real components, excluding delay slots)."""
        return self.tape.dim

    @property
    def delays(self) -> list[float]:
        """The distinct delay magnitudes, in slot order."""
        return [s.delay for s in self.delay_slots]

    @property
    def max_delay(self) -> float:
        """The largest delay magnitude (the history-buffer horizon); 0 if none."""
        return max((s.delay for s in self.delay_slots), default=0.0)

    def params_vec(self) -> np.ndarray:
        """Empty — DDE parameters are folded into the tape as constants."""
        return np.empty(0, dtype=np.float64)


@dataclass(frozen=True)
class SDEProblem:
    """A diagonal-Itô SDE ready to integrate: drift + diffusion tapes plus context.

    Attributes
    ----------
    drift : Tape
        The deterministic part ``f`` (one output per component).
    diffusion : Tape
        The per-component diagonal noise coefficients ``g``; carries ``∂g/∂u``
        when built with ``with_diffusion_jacobian=True`` (for Milstein).
    ic : ndarray, shape (dim,)
        The resolved initial state.
    t0 : float
        Start time.
    system : object
        Back-reference to the source system.
    """

    drift: Tape
    diffusion: Tape
    ic: np.ndarray
    t0: float = 0.0
    system: Any = None
    family: str = field(default="sde", init=False)

    @property
    def dim(self) -> int:
        """State-space dimension."""
        return self.drift.dim

    def params_vec(self) -> np.ndarray:
        """Control-parameter values in ``drift.control_names`` order.

        The drift and diffusion tapes share the same parameter layout, so one
        vector feeds both.
        """
        return _params_vec(self.system, self.drift)


#: Union of the concrete Problem types, for annotations.
Problem = ODEProblem | MapProblem | DDEProblem | SDEProblem


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def ode_problem(
    system: Any,
    *,
    ic: Any = None,
    t0: float = 0.0,
    with_jacobian: bool = False,
) -> ODEProblem:
    """Build an :class:`ODEProblem` from a continuous system.

    Parameters
    ----------
    system : ContinuousSystem
        The source system.
    ic : array-like, optional
        Initial state; resolved via ``system.resolve_ic`` (arg > ``system.ic`` >
        ``default_ic`` > random).
    t0 : float, default 0.0
        Start time.
    with_jacobian : bool, default False
        Lower the analytic Jacobian alongside the RHS (for stiff solvers).
    """
    tape = lower_ode_cached(system, with_jacobian=with_jacobian)
    ic_arr = system.resolve_ic(ic)
    return ODEProblem(tape=tape, ic=ic_arr, t0=float(t0), system=system)


def map_problem(
    system: Any,
    *,
    ic: Any = None,
    n0: int = 0,
    with_jacobian: bool = False,
) -> MapProblem:
    """Build a :class:`MapProblem` from a discrete map.

    Parameters
    ----------
    system : DiscreteMap
        The source map.
    ic : array-like, optional
        Initial state; resolved via ``system.resolve_ic``.
    n0 : int, default 0
        Starting iteration index.
    with_jacobian : bool, default False
        Lower the step Jacobian alongside the next-state map (for Lyapunov).

    Raises
    ------
    TapeCompileError
        If the map's ``_step`` cannot be traced symbolically (see
        :func:`tsdynamics.engine.compile.lower_map`).
    """
    tape = lower_map_cached(system, with_jacobian=with_jacobian)
    ic_arr = system.resolve_ic(ic)
    return MapProblem(tape=tape, ic=ic_arr, n0=int(n0), system=system)


def dde_problem(
    system: Any,
    *,
    ic: Any = None,
    t0: float = 0.0,
) -> DDEProblem:
    """Build a :class:`DDEProblem` from a delay system.

    Parameters
    ----------
    system : DelaySystem
        The source system.
    ic : array-like, optional
        Constant-past state; resolved via ``system.resolve_ic``.
    t0 : float, default 0.0
        Start time.

    Raises
    ------
    TapeCompileError
        If a delayed access has a state-dependent delay (see
        :func:`tsdynamics.engine.compile.lower_dde`).
    """
    tape, slots = lower_dde_cached(system)
    ic_arr = system.resolve_ic(ic)
    return DDEProblem(tape=tape, delay_slots=slots, ic=ic_arr, t0=float(t0), system=system)


def sde_problem(
    system: Any,
    *,
    ic: Any = None,
    t0: float = 0.0,
    with_diffusion_jacobian: bool = False,
) -> SDEProblem:
    """Build an :class:`SDEProblem` from a diagonal-Itô stochastic system.

    Parameters
    ----------
    system : object
        Anything exposing ``_drift`` / ``_diffusion`` staticmethods plus
        ``dim`` / ``params`` — the
        :class:`~tsdynamics.families.stochastic.StochasticSystem` contract
        (duck-typed so the engine layer stays below ``families``).
    ic : array-like, optional
        Initial state; resolved via ``system.resolve_ic``.
    t0 : float, default 0.0
        Start time.
    with_diffusion_jacobian : bool, default False
        Lower ``∂g/∂u`` into the diffusion tape (required by Milstein).
    """
    lowered: LoweredSDE = lower_sde_cached(system, with_diffusion_jacobian=with_diffusion_jacobian)
    ic_arr = system.resolve_ic(ic)
    return SDEProblem(
        drift=lowered.drift,
        diffusion=lowered.diffusion,
        ic=ic_arr,
        t0=float(t0),
        system=system,
    )


def build_problem(system: Any, **kwargs: Any) -> Problem:
    """Build the right Problem for ``system`` by detecting its family.

    Dispatches on the family base class (map → DDE → SDE → ODE, checked in that
    order so the more specific families win).  Keyword arguments are forwarded
    to the matching ``*_problem`` factory.

    Parameters
    ----------
    system : SystemBase
        Any built-in or user system instance.
    **kwargs
        Forwarded to the family factory (e.g. ``ic``, ``t0``, ``with_jacobian``).

    Returns
    -------
    Problem

    Raises
    ------
    TypeError
        If ``system``'s family is not recognised.
    """
    from tsdynamics.families.continuous import ContinuousSystem
    from tsdynamics.families.delay import DelaySystem
    from tsdynamics.families.discrete import DiscreteMap

    if isinstance(system, DiscreteMap):
        return map_problem(system, **kwargs)
    if isinstance(system, DelaySystem):
        return dde_problem(system, **kwargs)
    if _is_sde(system):
        return sde_problem(system, **kwargs)
    if isinstance(system, ContinuousSystem):
        return ode_problem(system, **kwargs)
    raise TypeError(
        f"build_problem: unrecognised family for {type(system).__name__}; expected an "
        f"ODE (ContinuousSystem), map (DiscreteMap), DDE (DelaySystem), or diagonal-Itô "
        f"SDE (with _drift/_diffusion)."
    )


def _is_sde(system: Any) -> bool:
    """Whether ``system`` is a diagonal-Itô SDE (defines ``_drift`` + ``_diffusion``).

    Detection is duck-typed on the contract methods rather than an
    ``isinstance(system, StochasticSystem)`` check, a deliberate layering choice:
    the engine sits below ``families`` in the import graph, so it must not import
    the family base class.
    """
    cls = type(system)
    return hasattr(cls, "_drift") and hasattr(cls, "_diffusion")


def _params_vec(system: Any, tape: Tape) -> np.ndarray:
    """Read the control-parameter values for ``tape`` from ``system.params``.

    A *fresh* ``float64`` vector is built on every call, by design: it is the live
    read that lets a control-parameter sweep reuse one compiled tape (the values
    are never snapshotted into the Problem).  The rebuild is deliberately cheap —
    one float per control name — and must stay so, because the engine reads it once
    per ``integrate`` / per stepping ``dt``; it is not cached because a cache would
    have to be invalidated on every ``system.params`` mutation, reintroducing the
    very snapshot staleness this live read exists to avoid.
    """
    if system is None or not tape.control_names:
        return np.empty(0, dtype=np.float64)
    return np.asarray([float(system.params[k]) for k in tape.control_names], dtype=np.float64)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
