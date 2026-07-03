# Analysis

The quantifier toolkit. Everything that steps — a map, a flow, or a Poincaré
section of a flow — implements the same `System` protocol, so the analyses
below compose over any [built-in or user-defined system](../systems/).

This section will cover:

- **Integration & solver methods** — explicit and stiff kernels, adaptive and
  fixed-step, and how to choose one.
- **Lyapunov spectra** — full spectra, the maximal exponent, and estimation
  from a measured time series.
- **Orbit & bifurcation diagrams** — parameter sweeps over maps and flows.
- **Poincaré sections** — root-refined crossings of an arbitrary plane.
- **Fixed & periodic points** — equilibria, limit cycles, and rigorous root
  enclosures.
- **Chaos indicators** — GALI, the 0–1 test, and expansion entropy.
- **Recurrence & RQA** — recurrence matrices and their quantification.
- **Fractal dimensions** — correlation, generalized Rényi, and box-counting.
- **Delay embeddings** — reconstructing state space from a scalar signal.
- **Entropy & complexity** — permutation, dispersion, sample, and multiscale.
- **Surrogates** — surrogate generators and nonlinearity tests.
- **Attractors & basins** — finding attractors, painting basins, and
  continuation.
