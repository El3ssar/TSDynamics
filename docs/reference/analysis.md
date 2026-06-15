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

::: tsdynamics.analysis.lyapunov.from_data.lyapunov_from_data

::: tsdynamics.analysis.lyapunov.from_data.LyapunovFromData

::: tsdynamics.analysis.lyapunov.kaplan_yorke_dimension

## Fixed points & periodic orbits

::: tsdynamics.analysis.fixedpoints.fixed.fixed_points

::: tsdynamics.analysis.fixedpoints.fixed.FixedPoint

::: tsdynamics.analysis.fixedpoints.periodic.periodic_orbits

::: tsdynamics.analysis.fixedpoints.periodic.periodic_orbit

::: tsdynamics.analysis.fixedpoints.periodic.PeriodicOrbit

::: tsdynamics.analysis.fixedpoints.periodic.estimate_period

## Chaos indicators

Three literature-validated answers to "is this orbit chaotic?": the Generalized
Alignment Index (GALI), the 0--1 test, and Hunt--Ott expansion entropy.

::: tsdynamics.analysis.chaos.gali.gali

::: tsdynamics.analysis.chaos.gali.GALIResult

::: tsdynamics.analysis.chaos.zero_one.zero_one_test

::: tsdynamics.analysis.chaos.expansion.expansion_entropy

::: tsdynamics.analysis.chaos.expansion.ExpansionEntropyResult

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

## Fractal dimensions

::: tsdynamics.analysis.dimensions.correlation.correlation_dimension

::: tsdynamics.analysis.dimensions.correlation.correlation_sum

::: tsdynamics.analysis.dimensions.generalized.generalized_dimension

::: tsdynamics.analysis.dimensions.generalized.box_counting_dimension

::: tsdynamics.analysis.dimensions.generalized.information_dimension

::: tsdynamics.analysis.dimensions.generalized.dimension_spectrum

::: tsdynamics.analysis.dimensions.fixedmass.fixed_mass_dimension

::: tsdynamics.analysis.dimensions.DimensionResult

## Delay embeddings

State-space reconstruction from a scalar (or multivariate) measurement
(Takens, 1981): the time-delay map, plus the delay- and dimension-selection
heuristics that parameterise it.

::: tsdynamics.analysis.embedding.embed.embed

### Delay selection

::: tsdynamics.analysis.embedding.delay.optimal_delay

::: tsdynamics.analysis.embedding.delay.mutual_information

::: tsdynamics.analysis.embedding.delay.autocorrelation

### Dimension selection

::: tsdynamics.analysis.embedding.dimension.embedding_dimension

::: tsdynamics.analysis.embedding.dimension.cao_dimension

::: tsdynamics.analysis.embedding.dimension.false_nearest_neighbors

::: tsdynamics.analysis.embedding.dimension.EmbeddingDimension
