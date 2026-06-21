"""Engine-backed Poincaré crossing detection (stream WS-CROSSKERNEL).

The Rust event engine (:func:`tsdyn_engine::integrate_events`, wired through
:func:`tsdynamics.engine.run.crossings`) marches a whole attractor and refines
its section crossings in **one FFI call per span**, replacing the Python
``PoincareMap`` loop that drove the flow one ``dt`` at a time through a full
``integrate()`` round-trip (the named "Poincaré sections are slow" culprit).

This module is the geometry + accumulation layer over that engine seam: it builds
the plane event tape ``g(u) = normal · u − offset`` once, sizes the integration
spans, and accumulates the requested number of crossings.
:class:`~tsdynamics.derived.poincare.PoincareMap` consumes it through
``trajectory``, so the ``trajectory``-driven consumers — ``poincare_section`` and
``return_map(method="poincare")`` — inherit the speedup with no change.
(``orbit_diagram`` drives the wrapper with ``step()`` rather than ``trajectory``,
so a bifurcation diagram over a ``PoincareMap`` is *not* accelerated here — that
awaits a resumable ``PoincareMap.step()`` (WS-STEPPER) or routing
``orbit_diagram`` through ``trajectory`` (WS-MAPITER).)

Accuracy / answer-preservation
------------------------------
The march uses the **fixed-step** ``rk4`` kernel at the detection step ``dt`` (the
engine's adaptive kernels carry no step ceiling, so an adaptive march would grow
the step, skip crossings, and degrade the O(h⁴) Hermite refinement that pins the
crossing).  With ``rk4`` at ``dt`` the engine marches the exact same ``dt`` grid
as the Python ``PoincareMap`` and refines with the identical cubic-Hermite
formula, so the crossings are answer-identical to that reference to ~1e-9 (only
the bracketed root solver differs — Illinois vs ``brentq``, both to ``xtol=1e-14``).
Spans are snapped to whole ``dt`` steps and resumed from the previous end state,
so a multi-span march reproduces one long march to ~1e-9 (``rk4`` is memoryless and
every step is ``dt`` — up to a ULP in the accumulated landing step at each span
boundary).
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The first span to probe, in time units, before the crossing rate is known.
_INITIAL_SPAN_TIME: float = 50.0

#: Explicit kernels whose fixed-step ``rk4`` march is a faithful, stable detection
#: grid.  A stiff default (``bdf``/``rosenbrock``/``trbdf2``) keeps the Python loop
#: — ``rk4`` at ``dt`` would be inaccurate or unstable on a stiff flow.
_EXPLICIT_METHODS: frozenset[str] = frozenset({"rk4", "rk45", "tsit5", "dop853"})


def plane_event_tape(dim: int, normal: np.ndarray, offset: float) -> Any:
    """Lower the section plane ``g(u) = normal · u − offset`` to a one-output tape.

    The event function is pure geometry (no parameters, no time), so it lowers to a
    single-output :class:`~tsdynamics.engine.compile.Tape` over the ``dim`` state
    inputs — exactly the channel :func:`tsdynamics.engine.run.crossings` watches.
    """
    import symengine

    from tsdynamics.engine.compile import lower_expressions

    syms = [symengine.Symbol(f"u{i}") for i in range(dim)]
    expr: Any = symengine.sympify(0)
    for i in range(dim):
        coeff = float(normal[i])
        if coeff != 0.0:
            expr = expr + coeff * syms[i]
    expr = expr - float(offset)
    return lower_expressions([expr], syms)


def engine_eligible(system: Any, backend: str | None) -> bool:
    """Whether ``system`` can take the fast engine crossing path.

    Eligible when the inner system is a non-stiff ODE with a numeric RHS and the
    requested backend is the compiled engine.  DDEs (no ``_rhs_numeric``), the
    pure-Python ``"reference"`` backend, and stiff defaults fall back to the
    Python ``PoincareMap`` loop.
    """
    from tsdynamics.families.continuous import ContinuousSystem

    if not isinstance(system, ContinuousSystem):
        return False
    if not hasattr(system, "_rhs_numeric"):
        return False
    if backend is not None and str(backend).lower() == "reference":
        return False
    from tsdynamics import solvers

    try:
        name = solvers.resolve(getattr(system, "_default_method", "rk45")).name
    except Exception:
        return False
    return name in _EXPLICIT_METHODS


def section_crossings(
    system: Any,
    normal: np.ndarray,
    offset: float,
    *,
    direction: int,
    n_crossings: int,
    transient: int = 0,
    dt: float,
    max_time: float,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    backend: str = "interp",
    t0: float = 0.0,
    ic: Any,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Collect ``n_crossings`` section crossings on the Rust engine.

    Returns ``(times, states, t_final, u_final)``: the crossing times and
    ``(n_crossings, dim)`` states after ``transient`` are discarded, plus the
    time and state the march ended at — the live cursor for the caller to advance
    its system to (so a subsequent ``step()`` continues forward rather than
    re-yielding crossings).  Raises :class:`RuntimeError` if no crossing is found
    within ``max_time`` of marching (the plane misses the attractor or the
    direction is wrong) — matching the Python loop's contract.
    """
    from tsdynamics.engine import run
    from tsdynamics.engine.problem import ODEProblem, ode_problem

    dim = int(system.dim)
    need = int(n_crossings) + int(transient)
    dt = float(dt)
    max_time = float(max_time)

    g_tape = plane_event_tape(dim, np.asarray(normal, dtype=float), offset)
    base_tape = ode_problem(system, ic=np.asarray(ic, dtype=float), t0=float(t0)).tape

    t_cur = float(t0)
    u_cur = np.ascontiguousarray(ic, dtype=np.float64)

    times_chunks: list[np.ndarray] = []
    states_chunks: list[np.ndarray] = []
    count = 0
    span_time = min(max_time, _INITIAL_SPAN_TIME)
    time_since_cross = 0.0

    while count < need:
        n_steps = max(1, int(np.ceil(span_time / dt)))
        t1 = t_cur + n_steps * dt
        prob = ODEProblem(tape=base_tape, ic=u_cur, t0=t_cur, system=system)
        times, states, t_final, u_final, _terminated = run.crossings(
            prob,
            g_tape,
            t0=t_cur,
            t1=t1,
            first_step=dt,
            direction=direction,
            method="rk4",
            rtol=rtol,
            atol=atol,
            backend=backend,
        )
        actual_span = t_final - t_cur
        if t_final <= t_cur:
            raise RuntimeError(
                f"{type(system).__name__}: the crossing march made no progress "
                f"(the step collapsed at t={t_cur:g})."
            )

        if times.size:
            times_chunks.append(times)
            states_chunks.append(states)
            count += int(times.size)
            time_since_cross = t_final - float(times[-1])
            remaining = need - count
            rate = times.size / actual_span  # crossings per unit time
            if remaining > 0 and rate > 0.0:
                span_time = min(max_time, remaining / rate * 1.3 + 10.0 * dt)
        else:
            time_since_cross += actual_span
            span_time = min(max_time, span_time * 2.0)

        if count < need and time_since_cross >= max_time:
            raise RuntimeError(
                f"{type(system).__name__}: no section crossing within "
                f"max_time={max_time:g} (the plane may miss the attractor, or "
                f"direction={direction} is wrong)."
            )

        t_cur = t_final
        u_cur = np.ascontiguousarray(u_final, dtype=np.float64)

    # ``t_cur`` / ``u_cur`` now hold the last span's end (or ``t0`` / ``ic`` when
    # ``need == 0`` and the loop never ran) — the cursor to resume from.
    if not times_chunks:
        return np.empty(0), np.empty((0, dim)), t_cur, u_cur
    all_times = np.concatenate(times_chunks)[:need]
    all_states = np.concatenate(states_chunks, axis=0)[:need]
    return all_times[transient:], all_states[transient:], t_cur, u_cur
