---
title: Tutorials
description: Guided experiments in chaos, bifurcations, reconstruction, multistability and noise, with runnable code and scientific figures.
hide:
  - toc
---

<div class="tsd-page-intro" markdown="1">

<span class="tsd-eyebrow">Guided experiments</span>

# Tutorials {#learn-by-doing}

Start with a question. Run the model, make a figure, then use the numerical
tools to investigate what you see. Each tutorial connects the equations to a
complete experiment, with code you can run in order.

[Start with seven tasks](seven-tasks.md){ .md-button .md-button--primary }
[Explore the plotting guide](../visualization/index.md){ .md-button }

</div>

<figure class="tsd-showcase" markdown="1">

![Four real library outputs: a Lorenz phase portrait, logistic bifurcations, a component time series and a noisy mean-reverting trajectory](../assets/figures/tutorials/overview.svg){ loading=lazy }

<figcaption><span class="tsd-caption-label">Example workflows</span><br>Geometry, parameter changes, time evolution and noise. These panels are assembled with the same <code>|</code> and <code>/</code> operators used in the visualization guide.</figcaption>
</figure>

## Your first complete workflow

The [seven-task introduction](seven-tasks.md) takes you from a first trajectory
to chaos indicators, bifurcation diagrams, equilibria, basins and an animated
phase portrait. Finish by defining a system of your own. Use it as a tour of the
library, then choose a deeper experiment below.

## Choose an experiment

<div class="tsd-feature-grid tsd-tutorial-grid" markdown="1">

<article class="tsd-feature-card" markdown="1">

[![The two lobes of the Lorenz attractor](../assets/figures/viz/kind-phase-3d.svg){ loading=lazy }](chaotic-attractor.md)

<span class="tsd-card-meta">Chaotic flows</span>

### [Anatomy of a chaotic attractor](chaotic-attractor.md)

Watch nearby states separate. Read a Lyapunov spectrum, inspect the dimension
estimate, and turn the trajectory into a movie.

</article>

<article class="tsd-feature-card" markdown="1">

[![The logistic map's branching orbit diagram](../assets/figures/analysis/orbit-diagram.svg){ loading=lazy }](bifurcations.md)

<span class="tsd-card-meta">Parameter sweeps</span>

### [The road to chaos](bifurcations.md)

Find period doubling and periodic windows. Begin with a map, then build a
Rössler sweep through a Poincaré section.

</article>

<article class="tsd-feature-card" markdown="1">

[![The measured and delay-coordinate views of a Rössler record](../assets/figures/tutorials/reconstruction-compare.svg){ loading=lazy }](reconstruction.md)

<span class="tsd-card-meta">Working from data</span>

### [Reconstruction from one signal](reconstruction.md)

Choose a delay and embedding dimension from diagnostic curves. Rebuild a
state-space picture and inspect data-driven estimates.

</article>

<article class="tsd-feature-card" markdown="1">

[![A flow section and the folded geometry of its return map](../assets/figures/tutorials/return-map.svg){ loading=lazy }](poincare-return-maps.md)

<span class="tsd-card-meta">Reduced dynamics</span>

### [Poincaré sections and return maps](poincare-return-maps.md)

Turn a flow into a map of crossings. Reveal structure hidden inside a
three-dimensional orbit.

</article>

<article class="tsd-feature-card" markdown="1">

[![Two differently coloured basins of attraction in the Duffing plane](../assets/figures/tutorials/basins-image.svg){ loading=lazy }](basins-multistability.md)

<span class="tsd-card-meta">Global structure</span>

### [Basins and multistability](basins-multistability.md)

Find competing attractors, map their basins, then tilt the system and watch
one possible outcome disappear.

</article>

<article class="tsd-feature-card" markdown="1">

[![A noisy double-well path beside the potential that shapes its motion](../assets/figures/tutorials/double-well-switching.svg){ loading=lazy }](noise-driven.md)

<span class="tsd-card-meta">Stochastic models</span>

### [Noise-driven dynamics](noise-driven.md)

Compare sample paths with ensemble statistics and analytic laws. Distinguish
barrier recrossings from completed transitions between wells.

</article>

</div>

## Make the experiment your own

Run the blocks in order with the stated initial conditions and seeds. Inspect
the figures before interpreting a number, then change a parameter, observation
window or plotting style and repeat. Numerical values are illustrative estimates;
chaotic trajectories and fitted quantities can vary with solver settings,
software versions and the length of the record.

Keep the [analysis reference](../analysis/index.md) nearby for assumptions and
options. Use [plot composition](../visualization/composition.md) to assemble a
figure, [styling](../visualization/styling.md) to give it a consistent look, and
[animation](../visualization/animation.md) to explain how the motion unfolds.
