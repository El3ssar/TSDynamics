---
description: API reference for tsdynamics.analysis — orbit diagrams, Poincaré sections, Lyapunov quantifiers, and fixed points.
---

<span class="ts-kicker">Reference</span>

# Analysis

The quantifiers that consume any [`System`](base.md). Prose-first
treatments live in the [Analysis](../analysis/index.md) section.

## Orbit diagrams

::: tsdynamics.analysis.orbits.orbit_diagram.orbit_diagram

::: tsdynamics.analysis.orbits.orbit_diagram.OrbitDiagram

## Poincaré sections

::: tsdynamics.analysis.orbits.poincare.poincare_section

## Lyapunov quantifiers

::: tsdynamics.analysis.lyapunov.lyapunov_spectrum

::: tsdynamics.analysis.lyapunov.max_lyapunov

::: tsdynamics.analysis.lyapunov.kaplan_yorke_dimension

## Fixed points

::: tsdynamics.analysis.fixedpoints.fixed_points

::: tsdynamics.analysis.fixedpoints.FixedPoint

## Entropy & complexity

Composable estimation — an [`OutcomeSpace`](#tsdynamics.analysis.entropy.core.OutcomeSpace)
(how a series is symbolised), a probability estimator, and an information
measure — plus the named measures built on it.

::: tsdynamics.analysis.entropy.core.entropy

::: tsdynamics.analysis.entropy.permutation.permutation_entropy

::: tsdynamics.analysis.entropy.permutation.weighted_permutation_entropy

::: tsdynamics.analysis.entropy.dispersion.dispersion_entropy

::: tsdynamics.analysis.entropy.sample.sample_entropy

::: tsdynamics.analysis.entropy.sample.approximate_entropy

::: tsdynamics.analysis.entropy.multiscale.multiscale_entropy

::: tsdynamics.analysis.entropy.lz.lz76_complexity

::: tsdynamics.analysis.entropy.lz.lz76_entropy

### Composable building blocks

::: tsdynamics.analysis.entropy.core.OutcomeSpace

::: tsdynamics.analysis.entropy.core.OrdinalPatterns

::: tsdynamics.analysis.entropy.core.Dispersion

::: tsdynamics.analysis.entropy.core.Shannon

::: tsdynamics.analysis.entropy.core.Renyi

::: tsdynamics.analysis.entropy.core.Tsallis
