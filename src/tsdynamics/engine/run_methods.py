"""Solver-method resolution + auto-stiffness — the ``method=`` contract.

Split out of :mod:`tsdynamics.engine.run` (the run-split refactor); every name
here stays reachable as ``tsdynamics.engine.run.<name>`` via re-export, so this is
a pure move.

This module owns the shared ``method=`` resolution both :func:`integrate` and
:func:`ensemble` route through, so ``"auto"`` (a-priori auto-stiffness),
spelling/alias normalisation ("RK45" → "rk45", "dopri5" → "rk45"), v2-only-name
rejection ("LSODA"), and the implicit-kernel Jacobian rebuild behave identically
across both entry points (diagnosis #11/#13).

The problem-coercion helper (``_as_problem``) is late-imported from
:mod:`tsdynamics.engine.run` inside the functions that need it, so importing this
module does not create an import cycle.
"""

from __future__ import annotations

from typing import Any

from .problem import DDEProblem, MapProblem, ODEProblem, Problem


def _recommend_method(problem: Problem) -> Any:
    """Resolve ``method="auto"`` to a kernel by a-priori auto-stiffness selection.

    The auto-stiffness wiring (ticket FIX-AUTOSTIFF) applies to **ODEs only**:
    probe the problem's Jacobian spectrum at its start state and let the solver
    registry recommend an implicit kernel (``bdf``) on a stiff RHS or the explicit
    default (``rk45``) otherwise (:func:`tsdynamics.solvers.recommend`).  The probe
    is the one-point heuristic :func:`tsdynamics.solvers.is_stiff` — useful but not
    a guarantee (see its docstring): the verdict is read at the problem's resolved
    start state (``problem.ic``) and at ``problem.t0`` (``0.0`` unless a builder set
    it; the integrate/ensemble ``t0=`` argument is not threaded here).  A system
    declares a stiff default via ``_default_method`` when the heuristic is too
    coarse for it.

    Maps and DDEs treat ``"auto"`` as an explicit, documented **no-op** — it
    resolves to the family default rather than probing:

    - **Maps** iterate without a solver kernel, so the resolved kernel is ignored
      by the map branch (``"auto"`` simply does not raise).
    - **DDEs** are driven by the method of steps, which reuses the *explicit*
      stage kernels only.  A stiffness probe would be meaningless **and** a trap:
      :func:`tsdynamics.solvers.is_stiff` reads only the *instantaneous* Jacobian
      ``∂f/∂u``, ignoring the delay terms that actually shape a DDE's spectrum, so
      a spurious "stiff" verdict could otherwise select the implicit
      ``rosenbrock`` the DDE engine cannot drive.  ``"auto"`` therefore resolves to
      the DDE default (``rk45``); a delay system that genuinely needs another
      explicit kernel should pass ``method=`` explicitly.

    SDEs never reach here (the SDE path refuses ``run.integrate`` upstream).

    Parameters
    ----------
    problem : Problem
        The already-lowered problem (built once by :func:`integrate`).

    Returns
    -------
    solvers.Resolution
        The recommended kernel, carrying its ``build_kwargs``.
    """
    from tsdynamics import solvers

    # Maps (no kernel) and DDEs (explicit method-of-steps; no meaningful stiffness
    # probe) fall back to their family default — auto-stiffness is ODE-only.
    if isinstance(problem, MapProblem | DDEProblem):
        # A map has no solver family of its own (DEFAULT_METHOD has no "map" key)
        # and the map branch discards the resolved kernel entirely, so it borrows
        # the "ode" default purely to return a valid Resolution; only DDEs carry a
        # genuine family here.
        family = "dde" if isinstance(problem, DDEProblem) else "ode"
        return solvers.resolve(solvers.DEFAULT_METHOD[family], family=family)
    return solvers.recommend(
        problem.system,
        family=problem.family,
        ic=problem.ic,
        t=getattr(problem, "t0", 0.0),
    )


def _resolve_method_for(method: str, problem: Problem) -> Any:
    """Resolve ``method=`` to a :class:`solvers.Resolution` for an already-built problem.

    The single home for the ``method=`` contract shared by :func:`integrate` and
    :func:`ensemble`, so ``"auto"`` means the same thing through both entry points
    (diagnosis #11/#13).  A literal ``method="auto"`` triggers a-priori
    auto-stiffness selection over *problem* (:func:`_recommend_method` →
    :func:`solvers.recommend`); every other name normalises spellings/aliases
    ("RK45" → "rk45", "dopri5" → "rk45") and rejects unknown or v2-only names
    ("LSODA") with a listing error + a stiff hint.  Resolving is harmless for the
    families that ignore ``method=`` (maps fold params in; SDE/DDE batches are
    refused by the caller before stepping).
    """
    from tsdynamics import solvers

    if solvers.normalize(method) == "auto":
        return _recommend_method(problem)
    return solvers.resolve(method)


def _resolve_method_and_prepare(
    system_or_problem: Any,
    problem: Problem,
    method: str,
    build_kwargs: dict[str, Any],
    **rebuild_extra: Any,
) -> tuple[str, Problem]:
    """Resolve ``method=`` and rebuild the ODE problem ``with_jacobian`` if the kernel needs it.

    The single shared contract for both :func:`integrate` and :func:`ensemble`, so
    ``method="auto"`` *and* the implicit-kernel Jacobian requirement behave
    identically through both entry points.  (This previously drifted: ``ensemble``
    resolved ``"auto"`` but skipped the ``with_jacobian`` rebuild, so a stiff
    ``ensemble(method="auto")`` dispatched a Jacobian-less tape and the engine
    refused it — factoring the two halves into one helper closes that gap.)

    The implicit ODE kernels (``bdf``/``rosenbrock``/``trbdf2``) need ``∂f/∂u`` on
    the tape or the engine refuses the step.  Rebuild once with the Jacobian when
    the resolved method needs it and the tape lacks it — only for an ODE built from
    a *system* (DDEs drive explicit kernels only; maps fold params in; a pre-built
    ``Problem`` we cannot re-lower, so the engine raises its clear guard instead).
    ``rebuild_extra`` carries entry-point-specific build kwargs (``integrate``
    passes ``ic=``; ``ensemble`` seeds each trajectory later, so it passes none).

    Returns the canonical method name and the (possibly rebuilt) problem.
    """
    from .run import _as_problem

    resolution = _resolve_method_for(method, problem)
    if (
        resolution.build_kwargs.get("with_jacobian")
        and isinstance(problem, ODEProblem)
        and not problem.tape.has_jacobian
        and "with_jacobian" not in build_kwargs
        and not isinstance(system_or_problem, ODEProblem)
    ):
        problem = _as_problem(
            system_or_problem, with_jacobian=True, **rebuild_extra, **build_kwargs
        )
    return resolution.name, problem
