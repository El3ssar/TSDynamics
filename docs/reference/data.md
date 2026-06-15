---
description: API reference for tsdynamics.data — the Trajectory result type and the state-space regions, samplers and set distances the basin/attractor layer builds on.
---

<span class="ts-kicker">Reference</span>

# Data & state-space

The lingua franca every analysis consumes. `Trajectory` is the result type
all families produce (documented with the [base classes](base.md#tsdynamics.data.Trajectory));
the regions, samplers and distances below are the state-space primitives the
attractor/basin layer is built on — Monte-Carlo and full-grid sampling, and
the attractor-matching distance used by continuation.

## Regions

A `Region` is a `Box` or a `Ball`; both describe a bounded subset of state
space and can draw uniform samples from it.

::: tsdynamics.data.sampling.Box

::: tsdynamics.data.sampling.Ball

::: tsdynamics.data.sampling.Grid

## Sampling

::: tsdynamics.data.sampling.sampler

::: tsdynamics.data.sampling.grid_points

## Set distances

::: tsdynamics.data.sampling.set_distance
