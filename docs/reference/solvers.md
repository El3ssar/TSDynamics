---
description: API reference for tsdynamics.solvers — the by-name solver registry, capability flags, solver resolution and auto-stiffness selection.
---

<span class="ts-kicker">Reference</span>

# Solvers

The numerical-method layer. Each solver is a `SolverSpec` carrying capability
flags (`SolverCaps`: explicit/implicit, adaptive, needs-Jacobian, supported
families); `solver=` strings resolve against the registry, an unknown name
raises with the available list, and an auto-stiffness heuristic can pick an
implicit kernel from the Jacobian spectrum. Third-party solvers register
through the same entry point.

Most users never call these directly — they pass `solver="rk45"` (or similar)
to `run` (see
[Integration & methods](../analysis/integration-and-methods.md)). This page
documents the registry itself.

!!! note "`solver=` picks a kernel; `method=` picks an estimator"
    One concept, one spelling. `lor.run(solver="dop853")` chooses the numerical
    kernel; `ts.analysis.optimal_delay(x, method="mi")` chooses an estimation
    algorithm. `run(method=...)` raises and names `solver=`.

## Specs & capabilities

::: tsdynamics.solvers.SolverSpec

::: tsdynamics.solvers.SolverCaps

## Resolution & selection

::: tsdynamics.solvers.select.resolve

::: tsdynamics.solvers.select.default_method

::: tsdynamics.solvers.select.available_for

::: tsdynamics.solvers.select.recommend

::: tsdynamics.solvers.select.is_stiff

::: tsdynamics.solvers.select.build_kwargs

## Registration

::: tsdynamics.solvers.register
