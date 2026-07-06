# Start

TSDynamics is a Python library for studying dynamical systems: you describe
the math once, symbolically, and a compiled Rust engine integrates flows,
iterates maps, marches delay and stochastic equations, and feeds a full
chaos-analysis toolkit. This section is the getting-started journey — from a
fresh `pip install` to defining a system of your own.

Read it in order, or jump to what you need:

- **[01 · Install](install.md)** — one `pip install`; the compiled engine ships
  in the wheel, so there is no build step, no compiler, and no warmup.
- **[02 · First trajectory](first-trajectory.md)** — instantiate the Lorenz
  system, integrate it, read named components off the result, plot it, and
  compute its Lyapunov spectrum.
- **[03 · The mental model](concepts.md)** — the handful of ideas the whole
  library rests on: the four families, the one stepping protocol, the three
  execution backends, and the derived-system wrappers.
- **[04 · Defining systems](defining-systems.md)** — subclass a family base and
  write the dynamics as one symbolic method. A worked example end to end, plus
  maps, SDEs, and automatic registration.

Once you are comfortable, the rest of the site opens up:

- The **[Systems](../systems/index.md)** catalogue — 171 built-in systems
  (136 ODEs, 6 delay equations, 3 SDEs, and 26 maps), each with its equations,
  a rendered attractor, and its literature reference.
- The **[Analysis](../analysis/index.md)** toolkit — Lyapunov spectra,
  bifurcation diagrams, Poincaré sections, fractal dimensions, recurrence
  quantification, surrogate tests, basins of attraction, and more, all composing
  over any built-in or user-defined system.
</content>
</invoke>
