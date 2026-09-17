"""Closed ``run()`` signatures — the one place an unknown keyword is refused.

Before v6 every family's ``run`` ended in ``**kwargs`` and each family invented
its own guard, so :class:`~tsdynamics.families.DelaySystem` accepted and
*silently dropped* ``max_step``, ``t0``, ``events`` and outright typos.  A
keyword a user typed and the library ignored is the silent-wrong-answer defect
in its purest form: the run completes, the number is wrong, nothing says so.

Every family now routes its leftovers through :func:`reject_unknown_run_keywords`,
which answers from a per-name **why** table.  The table is the interesting part:
the words that reach the wrong family are not typos, they are *the right word for
a different kind of dynamics*, and the answer that helps is the mathematical
reason plus a runnable line — never "unexpected keyword argument".
"""

from __future__ import annotations

import difflib
from typing import Any

from tsdynamics.errors import InvalidParameterError, remedy

__all__ = ["WHY", "reject_unknown_run_keywords", "run_keyword_error"]


#: Why a keyword that is valid *somewhere* does not exist *here*.
#:
#: Keyed by ``(keyword, family)``; the value is ``(reason, runnable_line)`` or
#: ``(reason, runnable_line, lead)``.  ``{cls}`` interpolates the class name and
#: ``{lower}`` a short instance name.  A keyword with no entry for this family
#: falls through to the did-you-mean path.
WHY: dict[tuple[str, str], tuple[str, str | None] | tuple[str, str | None, str]] = {
    # --- horizon words -----------------------------------------------------
    ("final_time", "map"): (
        "final_time is a *flow* keyword: a map has no continuous time — its "
        "horizon is a count of iterations.",
        "{cls}().run(steps=1000)",
    ),
    ("steps", "ode"): (
        "steps is a *map* keyword: a flow runs for a span of continuous time, "
        "not a count of iterations, so its horizon is final_time.",
        "{cls}().run(final_time=100.0, dt=0.01)",
    ),
    ("steps", "dde"): (
        "steps is a *map* keyword: a delay system runs for a span of continuous "
        "time, so its horizon is final_time.",
        "{cls}().run(final_time=100.0, dt=0.1)",
    ),
    ("steps", "sde"): (
        "steps is a *map* keyword: an SDE runs for a span of continuous time, "
        "so its horizon is final_time (and dt is the Ito increment).",
        "{cls}().run(final_time=10.0, dt=0.01)",
    ),
    ("n", "ode"): (
        "n is a *map* keyword: a flow runs for a span of continuous time, not a "
        "count of iterations, so its horizon is final_time.",
        "{cls}().run(final_time=100.0, dt=0.01)",
    ),
    ("n", "dde"): (
        "n is a *map* keyword: a delay system runs for a span of continuous "
        "time, so its horizon is final_time.",
        "{cls}().run(final_time=100.0, dt=0.1)",
    ),
    ("n", "sde"): (
        "n is a *map* keyword: an SDE runs for a span of continuous time, so "
        "its horizon is final_time.",
        "{cls}().run(final_time=10.0, dt=0.01)",
    ),
    # --- the clock ---------------------------------------------------------
    ("dt", "map"): (
        "dt is a *flow* keyword: consecutive iterates ARE the output grid of a "
        "map, so there is no sampling interval to choose.",
        "{cls}().run(steps=1000)",
    ),
    ("t0", "map"): (
        "t0 is a *flow* keyword: a map's independent variable is an integer "
        "iterate index, which always starts at 0.",
        "{cls}().run(steps=1000)",
    ),
    ("t0", "dde"): (
        "a delay system's clock is pinned to its history window [-tau_max, 0], "
        "so moving t0 would silently reinterpret the past.",
        "{cls}().run(final_time=100.0, dt=0.1)",
    ),
    # --- the engine ---------------------------------------------------------
    # ``backend`` IS a ``run`` keyword on every family; these rows answer it at
    # ``reinit``, where only the two families with a resumable engine handle
    # (ODE, DDE) can honour it.  Accepting it and doing nothing is the
    # silent-wrong-answer defect this module exists to prevent.
    ("backend", "map"): (
        "backend chooses the engine that RUNS a trajectory; a map's step() drives "
        "the pure-Python _step kernel, so a live stepper has no engine to choose.",
        "{cls}().run(steps=1000, backend='interp')",
    ),
    ("backend", "sde"): (
        "backend chooses the engine that RUNS a path; step() draws its own Wiener "
        "increment in Python, so a live SDE stepper has no engine to choose.",
        "{cls}().run(final_time=10.0, dt=0.01, backend='interp')",
    ),
    # --- the initial condition --------------------------------------------
    ("history", "ode"): (
        "history is a *delay* keyword: a delay system's initial condition is a "
        "whole function on [-tau_max, 0]; an ODE starts from one point.",
        "{cls}().run(final_time=100.0, ic=[1.0, 1.0, 1.0])",
    ),
    ("history", "map"): (
        "history is a *delay* keyword: a map starts from one point.",
        "{cls}().run(steps=1000, ic=[0.1, 0.1])",
    ),
    ("history", "sde"): (
        "history is a *delay* keyword: an SDE starts from one point.",
        "{cls}().run(final_time=10.0, ic=[1.0])",
    ),
    # --- the solver --------------------------------------------------------
    ("solver", "map"): (
        "a map has no solver — it IS the update rule, applied exactly.",
        "{cls}().run(steps=1000)",
    ),
    ("method", "ode"): (
        "method= selects an *estimator* in v6; the numerical kernel is solver=.",
        '{cls}().run(final_time=100.0, solver="dop853")',
    ),
    ("method", "dde"): (
        "method= selects an *estimator* in v6; the numerical kernel is solver=.",
        '{cls}().run(final_time=100.0, solver="rk45")',
    ),
    ("method", "sde"): (
        "method= selects an *estimator* in v6; the numerical kernel is solver=.",
        '{cls}().run(final_time=10.0, solver="milstein")',
    ),
    ("method", "map"): (
        "a map has no solver — it IS the update rule, applied exactly.",
        "{cls}().run(steps=1000)",
    ),
    # --- error control -----------------------------------------------------
    ("rtol", "map"): (
        "a map iterates in exact arithmetic — there is no local error to control.",
        "{cls}().run(steps=1000)",
    ),
    ("atol", "map"): (
        "a map iterates in exact arithmetic — there is no local error to control.",
        "{cls}().run(steps=1000)",
    ),
    ("rtol", "sde"): (
        "an SDE runs a fixed-step Ito scheme with no embedded error estimate; "
        "dt is the accuracy knob.",
        "{cls}().run(final_time=10.0, dt=0.001)",
    ),
    ("atol", "sde"): (
        "an SDE runs a fixed-step Ito scheme with no embedded error estimate; "
        "dt is the accuracy knob.",
        "{cls}().run(final_time=10.0, dt=0.001)",
    ),
    ("max_step", "map"): (
        "only an adaptive kernel has an internal step to bound; a map takes one "
        "iteration at a time.",
        "{cls}().run(steps=1000)",
    ),
    ("max_step", "dde"): (
        "the method of steps lands on every output sample, so dt already bounds the step.",
        "{cls}().run(final_time=100.0, dt=0.05)",
    ),
    ("max_step", "sde"): (
        "an SDE takes fixed dt steps, so dt already is the step.",
        "{cls}().run(final_time=10.0, dt=0.001)",
    ),
    # --- events ------------------------------------------------------------
    ("events", "map"): (
        "an event is a zero of a scalar BETWEEN samples, and a map has no in-between.",
        None,
    ),
    ("events", "dde"): (
        "Event detection is wired for ODEs only — the delay engine has no dense "
        "event root-finder yet. This is an engine limitation, not a property of "
        "delay systems.",
        "traj = {lower}.run(final_time=500.0, dt=0.01)",
        "Detect crossings from the samples instead:",
    ),
    ("events", "sde"): (
        "Event detection is wired for ODEs only — a fixed-step Ito path has no "
        "dense interpolant to root-find on. This is an engine limitation.",
        "traj = {lower}.run(final_time=10.0, dt=0.001)",
        "Detect crossings from the samples instead:",
    ),
    # --- retries -----------------------------------------------------------
    ("max_retries", "ode"): (
        "only a map re-draws a random initial condition on divergence.",
        "{cls}().run(final_time=100.0, ic=[1.0, 1.0, 1.0])",
    ),
    ("max_retries", "dde"): (
        "only a map re-draws a random initial condition on divergence.",
        "{cls}().run(final_time=100.0)",
    ),
    ("max_retries", "sde"): (
        "only a map re-draws a random initial condition on divergence.",
        "{cls}().run(final_time=10.0)",
    ),
}


def _did_you_mean(bad: str, accepted: tuple[str, ...]) -> str:
    """Return a ``Did you mean 'x'?`` clause, or the empty string."""
    exact = [n for n in accepted if n.lower() == bad.lower()]
    close = exact or difflib.get_close_matches(bad, list(accepted), n=1, cutoff=0.6)
    return f"Did you mean {close[0]!r}?  " if close else ""


def run_keyword_error(
    system: Any,
    bad: str,
    value: Any,
    *,
    family: str,
    accepted: tuple[str, ...],
    verb: str = "run",
) -> InvalidParameterError:
    """Build the ``InvalidParameterError`` for one rejected ``run()`` keyword."""
    cls = type(system).__name__
    lower = cls[0].lower() + cls[1:3].lower()
    head = f"{bad} is not a valid {cls}.{verb}() keyword, got {value!r}."
    entry = WHY.get((bad, family))
    if entry is not None:
        why, line = entry[0], entry[1]
        lead = entry[2] if len(entry) == 3 else None
        text = f"{head}\n{why}"
        if line is not None:
            text += remedy(line.format(cls=cls, lower=lower), lead=lead)
        return InvalidParameterError(text)
    accepted_list = ", ".join(accepted)
    return InvalidParameterError(
        f"{head}\n{_did_you_mean(bad, accepted)}{cls}.{verb} accepts: {accepted_list}."
    )


def reject_unknown_run_keywords(
    system: Any,
    leftovers: dict[str, Any],
    *,
    family: str,
    accepted: tuple[str, ...],
    verb: str = "run",
) -> None:
    """Raise on the first leftover keyword, naming why it does not exist here.

    Parameters
    ----------
    system : object
        The system whose ``run`` was called — only its class name is used.
    leftovers : dict
        Whatever reached ``**solver_options``.  Empty is the normal case.
    family : {"ode", "dde", "map", "sde"}
        Selects the row of :data:`WHY`.
    accepted : tuple of str
        This family's ``run`` keywords, in signature order — printed verbatim,
        so the message never has to be kept in sync by hand.
    verb : str
        The method name to print (``"run"``, ``"reinit"``, ``"ensemble"``).

    Raises
    ------
    InvalidParameterError
        On the first unknown keyword.
    """
    for bad in leftovers:
        raise run_keyword_error(
            system, bad, leftovers[bad], family=family, accepted=accepted, verb=verb
        )
