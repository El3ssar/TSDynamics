---
description: The complete public API of TSDynamics, with a map of where each symbol lives.
---

<span class="ts-kicker">Reference</span>

# Reference

The complete public API, one page per area, generated from the docstrings
in the source. If a signature here disagrees with your installed version,
check `ts.__version__`, `help(function)` and the release notes for the matching
version.

## Where do I find X

| You are looking for | Page |
| ------------------- | ---- |
| The top-level names, where everything else lives, and migration messages | [Top level](top-level.md) |
| `ContinuousSystem`, `DelaySystem`, `DiscreteMap`, `StochasticSystem`, `Trajectory`, `ParamSet`, the `System` protocol | [Base classes](base.md) |
| `PoincareMap`, `StroboscopicMap`, `TangentSystem`, `ProjectedSystem`, `Ensemble`, `WrappedSystem` | [Derived systems](derived.md) |
| Lyapunov, chaos indicators, fixed/periodic orbits, orbit diagrams, basins, dimensions, embeddings, recurrence | [Analysis](analysis.md) |
| `Box`, `Ball`, `Grid`, `sampler`, `grid_points`, `set_distance` | [Data & state-space](data.md) |
| `solver=` resolution, `SolverSpec`, auto-stiffness, registering a solver | [Solvers](solvers.md) |
| Programmatic access to the built-in system catalogue | [Registry](registry.md) |
| timestep estimation | [Utilities](utils.md) |
| A specific built-in system (equations, defaults, figures) | [Systems](../systems/index.md) |

## Scope

The main entries cover the public API. The solver and utility pages also
describe selected private implementation interfaces; these may change between
releases. The built-in system classes are documented
on their auto-generated pages under [Systems](../systems/index.md) rather
than here. For prose-first treatments, start from
[Start](../start/index.md) and [Analysis](../analysis/index.md).
