---
description: The analysis toolkit — trajectories, the uniform stepping protocol, and quantifiers for supported systems and sampled data.
---

<span class="ts-kicker">Analysis · Overview</span>

# Analysis

The quantifier toolkit. Everything that steps — a map, a flow, a delay or
stochastic system, or a Poincaré section of a flow — implements the same
`System` protocol. Each analysis declares the inputs and families it supports:
for example, a deterministic Lyapunov spectrum and a stochastic sample-path
statistic have different assumptions. Use `ts.analysis.find(system)` to find
compatible routines for a [built-in or user-defined system](../systems/index.md).

Two things tie it together. First, **the `Trajectory`** — a single-system `run`
returns one (an ensemble returns a batch): a `(T, dim)` array of states with named components (`traj["x"]`),
row/column selection (`traj[10:50]`, `traj["x", "z"]`, `traj.after(t)`) and
provenance carried in `traj.meta`. Second, **the calling convention**:

> **Every analysis is a free function whose FIRST argument is the thing it is
> about** — a system, a trajectory, or a result another analysis returned.

There is no bound method on a system, on a `Trajectory`, or on a result. One
concept, one spelling; and the free function is the one that cannot hide a
sampling choice from you. Most quantifiers return a rich result object that is a drop-in for
its underlying value (a `float`, an array) while also carrying `.meta`,
`.to_dict()`, `.to_frame()` and a `.plot` seam — **and whose `repr` is the
answer**, so `print(result)` is the whole reporting story.

Some analyses have different return shapes, documented in their `Returns`
sections: field dataclasses or curve lists; a `PoincareSection` trajectory;
the named `(radii, sums)` correlation-sum tuple; a `{q: DimensionResult}` spectrum;
sampling results and arrays; a `(centres, density, edges)` density tuple; or the
plain `float` from `set_distance`. These alternatives do not all provide the
result-object methods above.

```python
import tsdynamics as ts

lor  = ts.systems.Lorenz()
traj = lor.run(final_time=100.0, dt=0.01)                     # a Trajectory
exps = ts.analysis.lyapunov_spectrum(lor, final_time=300.0)   # a property of the equations
ts.analysis.correlation_dimension(traj)                       # a property of a point set
ts.analysis.kaplan_yorke_dimension(exps)                      # a property of the answer above
```

## Finding the one you want

`ts.analysis` is **52 flat, sorted names** — 49 analyses plus `find`, `register`
and `results` — so `ts.analysis.<TAB>` is the index. Two questions get answered
without leaving the REPL:

```python
ts.analysis.find(traj)             # what can I measure on THIS?
ts.analysis.find("is this chaotic")  # who answers THIS question?
```

`find` takes **one** positional argument: a free-text string, or a subject (a
system, a trajectory, an array, a result — or any of their classes), or nothing
at all for the complete list. It returns the analysis *functions*, so
`ts.analysis.find("chaotic")[0](lor)` runs the first hit; its `repr` is the
grouped table. `help(ts.analysis)` prints that same table, grouped by **what you
are holding**.

For a concrete system, discovery checks the registered family restrictions,
minimum state dimension and any autonomous-equation requirement. For example,
it excludes deterministic tangent estimators from SDEs, GALI from scalar models,
and equilibrium searches from explicitly forced equations. Class queries know
the family but cannot check parameter-dependent forcing; use an instance for
that. Plugins can declare these restrictions with `register(families=...,
min_dim=..., autonomous=...)`; omitting them preserves existing plugin behavior.
Equation-based plugins can additionally declare `system_types=(...)` when a
family name alone is insufficient to guarantee access to the required equations.

Guess a name that used to exist and the error is the migration guide:

```
>>> lor.lyapunov_spectrum()
AttributeError: 'Lorenz' object has no attribute 'lyapunov_spectrum': analyses are
free functions in v6, and the subject is the first argument.
    ts.analysis.lyapunov_spectrum(system)
    ts.analysis.find(system)   # all 20 that take a Lorenz flow
```

## The toolkit, by capability

Every page below is a self-contained guide with runnable examples and the
literature it implements.

### Integration & methods

The one verb that drives everything — `run`, on every family — the fixed /
adaptive / implicit solver families, automatic stiffness selection, and the
`interp` / `jit` / `reference` backends.

- [**Integration & methods**](integration-and-methods.md) — choosing a solver and a backend, plus the complete solver capability table.

### Lyapunov exponents

The average exponential separation rate — the defining quantifier of chaos.

- [**Lyapunov spectra**](lyapunov.md) — full spectra for flows / maps / DDEs, `lyapunov_from_data` (Kantz / Rosenstein), and the Kaplan–Yorke dimension.

### Orbits, bifurcations & sections

How the asymptotic dynamics reorganise as a parameter is swept, and the
lower-dimensional maps that expose their structure.

- [**Orbit & bifurcation diagrams**](orbit-diagrams.md) — parameter sweeps over maps and flow sections, plus first-return / next-amplitude maps.
- [**Poincaré sections**](poincare.md) — root-refined crossings of an arbitrary plane, the engine-native fast march.

### Fixed & periodic points

The invariant sets that organise the flow — equilibria, cycles, and their
linear stability.

- [**Fixed points & periodic orbits**](fixed-points.md) — equilibria and map fixed points (Newton / SD / DL, numerical interval search), period-$p$ orbits, and flow limit cycles by shooting.

### Chaos indicators

Fast "is this orbit chaotic?" verdicts that stand in for the full spectrum.

- [**Chaos indicators**](chaos.md) — GALI (Skokos), the 0–1 test (Gottwald–Melbourne), and expansion entropy (Hunt–Ott).

### Recurrence

Structure read off the geometry of an orbit — which states the trajectory
revisits, and when.

- [**Recurrence & RQA**](recurrence.md) — recurrence matrices and their quantification (determinism, laminarity, entropy).

### Geometry of the attractor

How much of state space the attractor fills, and how to rebuild it from a single
observable.

- [**Fractal dimensions**](dimensions.md) — correlation, generalized Rényi, box-counting and information dimensions with scaling-region fits.
- [**Delay embeddings**](embedding.md) — reconstructing state space from a scalar signal (Takens), with optimal-delay and embedding-dimension selection.

### Attractors & basins

Which attractor a given start ends up on, and how that partition of state space
is structured.

- [**Attractors & basins**](basins.md) — finding attractors, painting basins, basin stability / entropy / Wada, and continuation across a parameter.

## Beyond the prepackaged routines

Every routine above is built on the same uniform **stepping protocol** —
`reinit(...)` / `step(n_or_dt)` / `state()` / `time()` — that
every system implements. Most also offer `set_state(u)`; DDEs require an entire
history and use `reinit(history=...)` instead. When a prepackaged analysis does not fit (covariant
Lyapunov vectors, a custom event, finite-time statistics), you can drive the
protocol directly, or reach for the derived views — `sys.poincare(...)`,
`sys.ensemble(...)`, and `ts.derived.TangentSystem` / `ts.derived.ProjectedSystem`
— which are themselves systems and work with analyses that support their capabilities.

```python
lor.reinit([1.0, 1.0, 1.0])
for _ in range(1000):
    u = lor.step(0.01)          # advance one dt chunk, inspect, decide
```

## See also

- [Systems](../systems/index.md) — the 177 built-in systems and their families
- [Integration & methods](integration-and-methods.md) — the march underneath every quantifier
- [Lyapunov spectra](lyapunov.md) — the natural first stop for a new attractor
