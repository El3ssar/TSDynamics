---
description: The Analysis toolkit — the Trajectory object, the uniform stepping protocol, and every quantifier that composes over any built-in or user-defined system.
---

<span class="ts-kicker">Analysis · Overview</span>

# Analysis

The quantifier toolkit. Everything that steps — a map, a flow, a delay or
stochastic system, or a Poincaré section of a flow — implements the same
`System` protocol, so every analysis below composes over any
[built-in or user-defined system](../systems/index.md) without special-casing.
You define the dynamics once; the whole toolkit follows.

Two things tie it together. First, **the `Trajectory`** — every `integrate` or
`iterate` call returns one: a `(T, dim)` array of states with named components
(`traj["x"]`), point-set operations (`traj.after(t)`, `traj.minmax()`,
`traj.standardize()`) and provenance carried in `traj.meta`. Second, **the
calling convention** — every quantifier takes a *system or data* as its first
argument, dispatches on its family, and returns a rich result object that is a
drop-in for its underlying value (a `float`, an array) while also carrying
`.meta`, `.summary()`, `.to_dict()` and a `.plot` seam.

```python
import tsdynamics as ts

lor  = ts.systems.Lorenz()
traj = lor.integrate(final_time=100.0, dt=0.01)      # a Trajectory
exps = ts.lyapunov_spectrum(lor, final_time=300.0)   # → [0.91, ~0, -14.57]
ts.kaplan_yorke_dimension(exps)                      # → ≈ 2.06
```

## The toolkit, by capability

Every page below is a self-contained guide with runnable examples and the
literature it implements.

### Integration & methods

The two verbs that drive everything — `integrate` for flows, `iterate` for
maps — the fixed / adaptive / implicit solver families, automatic stiffness
selection, and the `interp` / `jit` / `reference` backends.

- [**Integration & methods**](integration-and-methods.md) — choosing a solver and a backend, plus the complete solver capability table.

### Lyapunov exponents

The average exponential separation rate — the defining quantifier of chaos.

- [**Lyapunov spectra**](lyapunov.md) — full spectra for flows / maps / DDEs, the Jacobian-free `max_lyapunov`, `lyapunov_from_data` (Kantz / Rosenstein), and the Kaplan–Yorke dimension.

### Orbits, bifurcations & sections

How the asymptotic dynamics reorganise as a parameter is swept, and the
lower-dimensional maps that expose their structure.

- [**Orbit & bifurcation diagrams**](orbit-diagrams.md) — parameter sweeps over maps and flow sections, plus first-return / next-amplitude maps.
- [**Poincaré sections**](poincare.md) — root-refined crossings of an arbitrary plane, the engine-native fast march.

### Fixed & periodic points

The invariant sets that organise the flow — equilibria, cycles, and their
linear stability.

- [**Fixed points & periodic orbits**](fixed-points.md) — equilibria and map fixed points (Newton / SD / DL, rigorous interval enclosure), period-$p$ orbits, and flow limit cycles by shooting.

### Chaos indicators

Fast "is this orbit chaotic?" verdicts that stand in for the full spectrum.

- [**Chaos indicators**](chaos.md) — GALI (Skokos), the 0–1 test (Gottwald–Melbourne), and expansion entropy (Hunt–Ott).

### Recurrence & complexity

Structure read off the geometry and the symbol statistics of an orbit.

- [**Recurrence & RQA**](recurrence.md) — recurrence matrices and their quantification (determinism, laminarity, entropy).
- [**Entropy & complexity**](entropy.md) — permutation, dispersion, sample, multiscale entropy, and Lempel–Ziv complexity.

### Geometry of the attractor

How much of state space the attractor fills, and how to rebuild it from a single
observable.

- [**Fractal dimensions**](dimensions.md) — correlation, generalized Rényi, box-counting and information dimensions with scaling-region fits.
- [**Delay embeddings**](embedding.md) — reconstructing state space from a scalar signal (Takens), with optimal-delay and embedding-dimension selection.

### Statistical tests

A principled null for "is there nonlinear structure here at all?"

- [**Surrogates**](surrogate.md) — surrogate generators (FT / AAFT / IAAFT) and nonlinearity tests (time-reversal asymmetry, nonlinear prediction error).

### Attractors & basins

Which attractor a given start ends up on, and how that partition of state space
is structured.

- [**Attractors & basins**](basins.md) — finding attractors, painting basins, basin stability / entropy / Wada, and continuation across a parameter.

## Beyond the prepackaged routines

Every routine above is built on the same uniform **stepping protocol** —
`reinit(u)` / `step(n_or_dt)` / `state()` / `set_state(u)` / `time()` — that
every system implements. When a prepackaged analysis does not fit (covariant
Lyapunov vectors, a custom event, finite-time statistics), you can drive the
protocol directly, or reach for the derived wrappers — `PoincareMap`,
`StroboscopicMap`, `TangentSystem`, `EnsembleSystem`, `ProjectedSystem` — which
are themselves systems, so the whole toolkit composes back over them.

```python
lor.reinit([1.0, 1.0, 1.0])
for _ in range(1000):
    u = lor.step(0.01)          # advance one dt chunk, inspect, decide
```

## See also

- [Systems](../systems/index.md) — the 171 built-in systems every analysis composes over
- [Integration & methods](integration-and-methods.md) — the march underneath every quantifier
- [Lyapunov spectra](lyapunov.md) — the natural first stop for a new attractor
