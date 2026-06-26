---
description: Every symbol importable directly from the tsdynamics namespace, mapped to its canonical reference page.
---

<span class="ts-kicker">Reference</span>

# Top level

Everything below is importable directly from the `tsdynamics` namespace.
Each is a re-export; the canonical page listed alongside documents it in
full.

```python
import tsdynamics as ts

traj = ts.Lorenz().integrate(final_time=100.0, dt=0.01)
spec = ts.lyapunov_spectrum(ts.Henon())
```

## Base classes & result type

| Symbol | Canonical home |
| ------ | -------------- |
| [`ContinuousSystem`][tsdynamics.families.continuous.ContinuousSystem] | `tsdynamics.families.continuous` |
| [`DelaySystem`][tsdynamics.families.delay.DelaySystem] | `tsdynamics.families.delay` |
| [`DiscreteMap`][tsdynamics.families.discrete.DiscreteMap] | `tsdynamics.families.discrete` |
| [`Trajectory`][tsdynamics.data.Trajectory] | `tsdynamics.data` |

## Derived-system wrappers

| Symbol | Canonical home |
| ------ | -------------- |
| [`PoincareMap`][tsdynamics.derived.poincare.PoincareMap] | `tsdynamics.derived.poincare` |
| [`StroboscopicMap`][tsdynamics.derived.stroboscopic.StroboscopicMap] | `tsdynamics.derived.stroboscopic` |
| [`TangentSystem`][tsdynamics.derived.tangent.TangentSystem] | `tsdynamics.derived.tangent` |
| [`EnsembleSystem`][tsdynamics.derived.ensemble.EnsembleSystem] | `tsdynamics.derived.ensemble` |
| [`ProjectedSystem`][tsdynamics.derived.projected.ProjectedSystem] | `tsdynamics.derived.projected` |

## Analysis toolkit

| Symbol | Canonical home |
| ------ | -------------- |
| [`orbit_diagram`][tsdynamics.analysis.orbits.orbit_diagram.orbit_diagram] | `tsdynamics.analysis.orbits.orbit_diagram` |
| [`OrbitDiagram`][tsdynamics.analysis.orbits.orbit_diagram.OrbitDiagram] | `tsdynamics.analysis.orbits.orbit_diagram` |
| [`poincare_section`][tsdynamics.analysis.orbits.poincare.poincare_section] | `tsdynamics.analysis.orbits.poincare` |
| [`lyapunov_spectrum`][tsdynamics.analysis.lyapunov.lyapunov_spectrum] | `tsdynamics.analysis.lyapunov` |
| [`max_lyapunov`][tsdynamics.analysis.lyapunov.max_lyapunov] | `tsdynamics.analysis.lyapunov` |
| [`kaplan_yorke_dimension`][tsdynamics.analysis.lyapunov.kaplan_yorke_dimension] | `tsdynamics.analysis.lyapunov` |
| [`fixed_points`][tsdynamics.analysis.fixedpoints.fixed_points] | `tsdynamics.analysis.fixedpoints` |
| [`FixedPoint`][tsdynamics.analysis.fixedpoints.FixedPoint] | `tsdynamics.analysis.fixedpoints` |

## The 151 built-in systems

Built-in system classes live under [`tsdynamics.systems`](../systems/index.md) —
the canonical path is `tsdynamics.systems.Lorenz` (flat, no need to remember
whether a model is `continuous` or `discrete`). They are kept out of the
top-level namespace so it stays focused on the base classes, wrappers, analysis
functions, and submodules. For backwards compatibility `tsdynamics.Lorenz` (and
`from tsdynamics import Lorenz`) still resolve lazily. The classes are documented
on their generated pages under [Systems](../systems/index.md), and discoverable
programmatically through the [registry](registry.md).

## Submodules

`tsdynamics.families`, `tsdynamics.systems`, `tsdynamics.derived`,
`tsdynamics.analysis`, `tsdynamics.registry`, and `tsdynamics.utils` are
importable as attributes. Lower-level surface that is deliberately *not*
re-exported at the top level — `SystemBase`, `ParamSet`, `MetaStore`, the
`System` protocol — lives under `tsdynamics.families`; see
[Base classes](base.md) and [Utilities](utils.md).

`tsdynamics.__version__` is the installed package version.
