# Start

TSDynamics simulates ordinary, delay and stochastic differential equations and
discrete maps. You can use a built-in model or define your own, then study its
trajectories with numerical analyses and plots. The compiled Rust engine is
included in the standard installation.

This section introduces installation, a first calculation, the main types of
system and the equations required to define a new model.

Read it in order, or jump to what you need:

- **[01 · Install](install.md)** — one `pip install`; the compiled engine ships
  in the wheel, so there is no build step and no compiler toolchain.
- **[02 · First trajectory](first-trajectory.md)** — instantiate the Lorenz
  system, `run` it, read named components off the result, plot it, and measure
  its Lyapunov spectrum.
- **[03 · The mental model](concepts.md)** — the handful of ideas the whole
  library rests on: the four families, the one stepping protocol, the three
  execution backends, and the derived-system wrappers.
- **[04 · Defining systems](defining-systems.md)** — subclass a family base and
  write the dynamics as one symbolic method. A worked example end to end, plus
  maps, SDEs, and automatic registration.

In a hurry? [**Seven tasks in one sitting**](../tutorials/seven-tasks.md) is
the same material as one page of runnable code: simulate, test for chaos, draw a
bifurcation diagram, find fixed points, paint basins, animate a phase portrait,
and define a system of your own.

Once you are comfortable, the rest of the site opens up:

- The **[Systems](../systems/index.md)** catalogue — 177 built-in systems
  (142 ODEs, 6 delay equations, 3 SDEs, and 26 maps), each with its equations,
  a rendered attractor, and its literature reference.
- The **[Analysis](../analysis/index.md)** toolkit — Lyapunov spectra,
  bifurcation diagrams, Poincaré sections, fractal dimensions, recurrence
  quantification, delay embeddings, basins of attraction, and more. Each routine
  documents its supported families and inputs; `ts.analysis.find(system)` helps
  select compatible tools.

## Choose a starting point for your question

| What you want to do | Where to start | What to look for |
| --- | --- | --- |
| Understand the motion of a model | [First trajectory](first-trajectory.md) | Compare its phase portrait with its component time series. |
| See how a parameter changes the behaviour | [Orbit diagrams](../analysis/orbit-diagrams.md) | Branch splitting, periodic windows and changes in the attractor. |
| Estimate sensitivity to initial conditions | [Lyapunov spectra](../analysis/lyapunov.md) | The signs of the exponents, and whether they have converged. |
| Find stationary states and their stability | [Fixed points](../analysis/fixed-points.md) | The residual, eigenvalues and stability classification of each candidate. |
| Find which initial conditions reach which attractor | [Basins of attraction](../analysis/basins.md) | Basin boundaries, competing outcomes and dependence on the search region. |
| Work from recorded observations rather than equations | [Delay embeddings](../analysis/embedding.md) | Sampling assumptions, delay choice and reconstruction dimension. |
| Study a model you wrote yourself | [Defining systems](defining-systems.md) | The family contract and one complete runnable example. |

Each analysis page states which inputs it supports. A flow, a discrete map,
a delay equation and a noisy model have different mathematical assumptions;
choose a compatible tool before interpreting its answer.

## Read the result, then check the assumptions

A picture helps you decide what to measure. A numerical result needs context:
which parameters and starting state were used, how long the calculation ran,
and whether increasing that horizon changes the conclusion. The worked
tutorials show the code beside the plot, and the system catalogue distinguishes
published reference values from finite-time measurements.
