---
description: API reference for tsdynamics.families — the family base classes, Trajectory, ParamSet, and the System protocol.
---

<span class="ts-kicker">Reference</span>

# Base classes

The shared machinery (`ParamSet`, `Trajectory`, `SystemBase`), the four family
bases users subclass — `ContinuousSystem` (ODEs), `DelaySystem` (DDEs),
`DiscreteMap` (maps) and `StochasticSystem` (diagonal-Itô SDEs) — and the
`System` protocol the analysis toolkit is written against.

`run` is the one verb that produces data on every one of them; `reinit` /
`step` / `state` / `time` are the stepping protocol underneath it.

::: tsdynamics.families.base.ParamSet

::: tsdynamics.data.Trajectory

::: tsdynamics.families.base.SystemBase

::: tsdynamics.families.continuous.ContinuousSystem

::: tsdynamics.families.delay.DelaySystem

::: tsdynamics.families.discrete.DiscreteMap

::: tsdynamics.families.stochastic.StochasticSystem

::: tsdynamics.families.protocol.System
