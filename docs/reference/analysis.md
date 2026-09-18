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

## Return maps

::: tsdynamics.analysis.orbits.return_map.return_map

::: tsdynamics.analysis.orbits.return_map.ReturnMap

## Lyapunov quantifiers

::: tsdynamics.analysis.lyapunov.lyapunov_spectrum

::: tsdynamics.analysis.lyapunov.from_data.lyapunov_from_data

::: tsdynamics.analysis.lyapunov.from_data.LyapunovFromData

::: tsdynamics.analysis.lyapunov.kaplan_yorke_dimension

## Fixed points & periodic orbits

::: tsdynamics.analysis.fixedpoints.fixed.fixed_points

::: tsdynamics.analysis.fixedpoints.fixed.FixedPoint

::: tsdynamics.analysis.fixedpoints.periodic.periodic_orbits

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

## Recurrence & RQA

Recurrence plots (Eckmann, Kamphorst & Ruelle, 1987) record when a trajectory
revisits its own past; recurrence quantification analysis (Marwan et al., 2007)
reduces that structure to scalar measures of determinism and laminarity, run
globally or in a sliding window.

::: tsdynamics.analysis.recurrence.matrix.recurrence_matrix

::: tsdynamics.analysis.recurrence.matrix.RecurrenceMatrix

::: tsdynamics.analysis.recurrence.rqa.rqa

::: tsdynamics.analysis.recurrence.rqa.RQAResult

::: tsdynamics.analysis.recurrence.windowed.windowed_rqa

::: tsdynamics.analysis.recurrence.windowed.WindowedRQA

## Attractors & basins

The global stability picture of a multistable system. Attractors are located by
following trajectories through a cell tessellation until they recurrently revisit
cells (Datseris & Wagemakers, 2022); the basin of each is the set of initial
conditions reaching it. Basin *stability* (Menck et al., 2013) is an attractor's
share of a sampled region; basin *entropy* (Daza et al., 2016) and the
*uncertainty exponent* (Grebogi et al., 1983) quantify how fractal the boundaries
are; continuation tracks attractors and their basins across a parameter.

::: tsdynamics.analysis.basins.attractors.attractors

::: tsdynamics.analysis.basins.attractors.AttractorSet

::: tsdynamics.analysis.basins.attractors.Attractor

::: tsdynamics.analysis.basins.basins.basins

::: tsdynamics.analysis.basins.basins.BasinsResult

::: tsdynamics.analysis.basins.basins.basin_fractions

::: tsdynamics.analysis.basins.basins.BasinFractions

### Boundary structure

::: tsdynamics.analysis.basins.metrics.basin_entropy

::: tsdynamics.analysis.basins.metrics.BasinEntropy

::: tsdynamics.analysis.basins.metrics.uncertainty_exponent

::: tsdynamics.analysis.basins.metrics.UncertaintyExponent

::: tsdynamics.analysis.basins.metrics.wada_property

::: tsdynamics.analysis.basins.metrics.WadaResult

::: tsdynamics.analysis.basins.metrics.resilience

### Continuation & tipping

::: tsdynamics.analysis.basins.continuation.continuation

::: tsdynamics.analysis.basins.continuation.ContinuationResult

::: tsdynamics.analysis.basins.continuation.tipping_points

## Sampling

Sagitta-based tools for time-ordered samples: choose an output `dt` (the
largest stride whose mid-point bow off the chord — the *sagitta* — stays under a
geometric tolerance), or read the per-point sagitta as a local "how sharply it
bends" field.

::: tsdynamics.analysis.sampling.sagitta.estimate_dt_from_sagitta

::: tsdynamics.analysis.sampling.sagitta.sagitta_profile
