---
description: The analysis toolkit — trajectories, the uniform stepping protocol, and quantifiers for supported systems and sampled data.
---

<span class="ts-kicker">Analysis · Overview</span>

# Analysis

Use these routines to study stability, sensitivity, recurrence and attractor
geometry. Systems share a stepping protocol, but each analysis declares its
supported inputs and families:
for example, a deterministic Lyapunov spectrum and a stochastic sample-path
statistic have different assumptions. Use `ts.analysis.find(system)` to find
compatible routines for a [built-in or user-defined system](../systems/index.md).

Call an analysis as `ts.analysis.name(subject, ...)`, where the first argument
is a system, sampled data or a previous result. A single-system `run` returns a
`Trajectory` containing `(T, dim)` states, component names, sample times and
provenance. Ensembles return a batch of trajectories.

Most analyses return a result object with a numerical value, calculation settings
in `.meta`, export methods and a `.plot` interface. `print(result)` gives a short
summary; inspect its diagnostic curves and settings before interpreting an
estimate. Some results can also be converted to a `float` or NumPy array.

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

Use tab completion to browse `ts.analysis`, or search by subject and task:

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

Older method names produce a message with the current function call:

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

Simulate each family with `run`, choose fixed-step, adaptive or implicit solvers,
and compare the `interp`, `jit` and `reference` execution backends.

- [**Integration & methods**](integration-and-methods.md) — choosing a solver and a backend, plus the complete solver capability table.

### Lyapunov exponents

Estimate the average exponential growth and decay of infinitesimal perturbations.

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

Measure tangent-vector alignment, the 0–1 statistic or regional expansion.
These indicators have different assumptions and are not interchangeable chaos
classifiers.

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

For custom model-based calculations, use the **stepping protocol** —
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
- [Integration & methods](integration-and-methods.md) — simulation settings and their numerical meaning
- [Lyapunov spectra](lyapunov.md) — the natural first stop for a new attractor
