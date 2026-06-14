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
| [`orbit_diagram`][tsdynamics.analysis.orbit_diagram.orbit_diagram] | `tsdynamics.analysis.orbit_diagram` |
| [`OrbitDiagram`][tsdynamics.analysis.orbit_diagram.OrbitDiagram] | `tsdynamics.analysis.orbit_diagram` |
| [`poincare_section`][tsdynamics.analysis.poincare.poincare_section] | `tsdynamics.analysis.poincare` |
| [`lyapunov_spectrum`][tsdynamics.analysis.lyapunov.lyapunov_spectrum] | `tsdynamics.analysis.lyapunov` |
| [`max_lyapunov`][tsdynamics.analysis.lyapunov.max_lyapunov] | `tsdynamics.analysis.lyapunov` |
| [`kaplan_yorke_dimension`][tsdynamics.analysis.lyapunov.kaplan_yorke_dimension] | `tsdynamics.analysis.lyapunov` |
| [`fixed_points`][tsdynamics.analysis.fixed_points.fixed_points] | `tsdynamics.analysis.fixed_points` |
| [`FixedPoint`][tsdynamics.analysis.fixed_points.FixedPoint] | `tsdynamics.analysis.fixed_points` |

## The 149 built-in systems

Every registered system class is re-exported at the top level, so
`from tsdynamics import Lorenz` works without remembering submodule
paths. The classes are documented on their generated pages under
[Systems](../systems/index.md), and discoverable programmatically through
the [registry](registry.md).

## Submodules

`tsdynamics.families`, `tsdynamics.systems`, `tsdynamics.derived`,
`tsdynamics.analysis`, `tsdynamics.registry`, and `tsdynamics.utils` are
importable as attributes. Lower-level surface that is deliberately *not*
re-exported at the top level — `SystemBase`, `ParamSet`, `MetaStore`, the
`System` protocol, `staticjit` — lives under `tsdynamics.families` and
`tsdynamics.utils`; see [Base classes](base.md) and [Utilities](utils.md).

`tsdynamics.__version__` is the installed package version.
