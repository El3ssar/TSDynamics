---
description: API reference for tsdynamics.solvers — the by-name solver registry, capability flags, method resolution and auto-stiffness selection.
---

<span class="ts-kicker">Reference</span>

# Solvers

The numerical-method layer. Each solver is a `SolverSpec` carrying capability
flags (`SolverCaps`: explicit/implicit, adaptive, needs-Jacobian, supported
families); `method=` strings resolve against the registry, an unknown name
raises with the available list, and an auto-stiffness heuristic can pick an
implicit method from the Jacobian spectrum. Third-party solvers register
through the same entry point.

Most users never call these directly — they pass `method="rk45"` (or similar)
to [`integrate`](../analysis/integrate.md). This page documents the registry
itself.

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
