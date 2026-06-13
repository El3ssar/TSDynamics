---
description: The complete public API of TSDynamics, with a map of where each symbol lives.
---

<span class="ts-kicker">Reference</span>

# Reference

The complete public API, one page per area, generated from the docstrings
in the source. If a signature here disagrees with your installed version,
trust your installed version.

## Where do I find X

| You are looking for | Page |
| ------------------- | ---- |
| Everything importable straight from `tsdynamics` | [Top level](top-level.md) |
| `ContinuousSystem`, `DelaySystem`, `DiscreteMap`, `Trajectory`, the `System` protocol | [Base classes](base.md) |
| `PoincareMap`, `StroboscopicMap`, `TangentSystem`, `EnsembleSystem`, `ProjectedSystem` | [Derived systems](derived.md) |
| `orbit_diagram`, `poincare_section`, `lyapunov_spectrum`, `max_lyapunov`, `fixed_points` | [Analysis](analysis.md) |
| Programmatic access to the 149 built-in systems | [Registry](registry.md) |
| `staticjit`, timestep estimation | [Utilities](utils.md) |
| A specific built-in system (equations, defaults, figures) | [Systems](../systems/index.md) |

## Scope

This reference covers the public API only; names with a leading underscore
are private and not documented. The built-in system classes are documented
on their auto-generated pages under [Systems](../systems/index.md) rather
than here. For prose-first treatments, start from
[Start](../start/index.md) and [Analysis](../analysis/index.md).
