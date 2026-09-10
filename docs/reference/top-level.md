---
description: The twelve names in the tsdynamics namespace, the rule that decides membership, and where everything else lives.
---

<span class="ts-kicker">Reference</span>

# Top level

`tsdynamics.<TAB>` shows **twelve** names. That is not minimalism for its own
sake — it is one rule applied mechanically:

> A name earns a top-level slot only if you **type** it in ordinary work. And you
> should never have to construct a library type to make a call — so a name that
> is exported *because a signature demands it* is evidence of a signature bug,
> not of a needed export.

```python
import tsdynamics as ts

traj = ts.systems.Lorenz().run(final_time=100.0, dt=0.01)
exps = ts.analysis.lyapunov_spectrum(ts.systems.Henon())
ts.plot(traj, color="crimson", title="Lorenz").save("lorenz.png")
```

## The twelve

| Symbol | What it is | Canonical home |
| ------ | ---------- | -------------- |
| [`ContinuousSystem`][tsdynamics.families.continuous.ContinuousSystem] | subclass it to define an ODE | `tsdynamics.families.continuous` |
| [`DelaySystem`][tsdynamics.families.delay.DelaySystem] | …a DDE | `tsdynamics.families.delay` |
| [`DiscreteMap`][tsdynamics.families.discrete.DiscreteMap] | …a map | `tsdynamics.families.discrete` |
| [`StochasticSystem`][tsdynamics.families.stochastic.StochasticSystem] | …an SDE | `tsdynamics.families.stochastic` |
| [`WrappedSystem`][tsdynamics.families.wrapped.WrappedSystem] | …or adapt an external stepper | `tsdynamics.families.wrapped` |
| [`Trajectory`][tsdynamics.data.Trajectory] | what every run returns — and what you build from measured data | `tsdynamics.data` |
| `plot` | the plotting front door | `tsdynamics.viz` |
| `systems` | the 177 built-in models | — |
| `analysis` | every quantifier | — |
| `viz` | specs, themes, renderers, transforms | — |
| `errors` | the exception hierarchy you catch | — |
| `__version__` | the installed version | — |

The five family bases are here because each is the **sole spelling of a
capability**: there is no verb and no plain-Python path that defines a delay
system for you. `Trajectory` is the one type you *receive* that is also a type
you *type* — you annotate it, you `isinstance` it, and since v6 you construct it
(`ts.Trajectory(t, y)`) from data you measured elsewhere.

## Where everything else lives

Nothing was removed. Every name below still resolves as `ts.<name>` and still
imports with `from tsdynamics import <name>` — it is only off the tab surface,
because **no call requires it**.

| Go here | For | When you would |
| ------- | --- | -------------- |
| [`ts.analysis`](analysis.md) | the 44 quantifiers + their result types | always — this is where analyses live |
| [`ts.systems`](../systems/index.md) | the 177 models | `ts.systems.Lorenz()` |
| [`ts.viz`](../visualization/index.md) | `PlotSpec`, themes, renderers, transforms, `T` | you are styling, exporting or writing a transform |
| [`ts.derived`](derived.md) | `PoincareMap`, `StroboscopicMap`, `TangentSystem`, `EnsembleSystem`, `ProjectedSystem` | rarely — each has a verb on the system (below) |
| [`ts.data`](data.md) | `Trajectory`, `Box`, `Ball`, `Grid`, `sampler`, `grid_points` | rarely — every region argument takes plain bounds |
| [`ts.errors`](../contributing/glossary.md) | `InvalidParameterError`, `ConvergenceError`, … | writing an `except` clause |
| [`ts.registry`](registry.md) | the system / analysis / renderer / transform registries | enumerating the catalogue programmatically |
| [`ts.families`](base.md) | `SystemBase`, `ParamSet`, `MetaStore`, the `System` protocol | writing a new family, not a new system |
| [`ts.utils`](utils.md) | `make_output_grid`, the tolerance constants | reading what a default actually is |
| `ts.engine` / `ts.solvers` | the Rust-facing compile/run seam, the solver table | debugging the engine |

### The wrappers have verbs

Every derived wrapper is built by a verb on the system it wraps, and the verb
takes the same vocabulary the class does:

```python
import numpy as np

lor = ts.systems.Lorenz()
states = np.random.default_rng(0).normal(size=(8, 3))

lor.poincare("y", 0.0)                 # a PoincareMap  (also ("y", 0.0, "up"))
lor.stroboscope(period=6.28)           # a StroboscopicMap
lor.tangent(k=2)                       # a TangentSystem — the Lyapunov engine
lor.project(0, 2)                      # a ProjectedSystem — the (x, z) shadow
lor.copies(states)                     # an EnsembleSystem you drive yourself
lor.ensemble(states, final_time=5.0)   # ...or just run the batch → (8, 3)
```

### The analyses are also on the object

Pressing ++tab++ on a system shows its own verbs, never the ~60 free functions
that take it as a first argument. Those are grouped into four cached *topical
accessors*, so the toolkit is navigable from the object you already hold:

```python
lor = ts.systems.Lorenz()

lor.lyap.spectrum(final_time=20.0)              # → ts.lyapunov_spectrum(lor, ...)
lor.chaos.gali(k=2, final_time=20.0)            # → ts.gali(lor, k=2, ...)
lor.dims.correlation(                           # → runs the system, then
    run_kwargs={"final_time": 40.0, "dt": 0.05},  # ts.correlation_dimension
)
lor.recurrence.rqa(                             # → ...likewise
    recurrence_rate=0.05, run_kwargs={"final_time": 10.0, "dt": 0.05}
)
```

An accessor adds **zero behaviour** — the result is the free function's result
on the same input.

Three of them — the ones whose analyses consume a *measured point set* — are on
the [`Trajectory`][tsdynamics.data.Trajectory] too, which is what you have when
the data came from somewhere else:

```python
traj = lor.run(final_time=40.0, dt=0.05, ic=[1.0, 1.0, 1.0])

traj.dims.correlation()                    # the series is used verbatim
traj.recurrence.rqa(recurrence_rate=0.05)
traj.lyap.from_data(k_max=40)              # the estimator a bare series supports
```

Reached from a system these run it first to produce a trajectory (pass `data=`
to supply your own); reached from a trajectory there is nothing to run.
`traj.lyap.spectrum()` therefore raises — a trajectory is numbers, not a
right-hand side — and says so, naming `system.lyap.spectrum()` instead.

### Regions take plain bounds

Everything that asks for a region reads **one `(lo, hi)` pair per state
component** — add a third entry, `(lo, hi, n)`, to say how finely to grid it.
No library type is ever required:

```python
vdp, henon = ts.systems.VanDerPol(), ts.systems.Henon()

ts.fixed_points(vdp, region=[(-3, 3), (-3, 3)])
ts.basins_of_attraction(henon, [(-2, 2, 20), (-2, 2, 20)], max_steps=500)
ts.expansion_entropy(henon, [(-1.6, 1.6), (-0.5, 0.5)], n_samples=100, n=8)
```

A `ts.data.Box` / `Ball` / `Grid` is still accepted at every one of those doors.
It is simply never the price of entry.

## The 177 built-in systems

Built-in system classes live under [`tsdynamics.systems`](../systems/index.md) —
the canonical path is `tsdynamics.systems.Lorenz` (flat, no need to remember
whether a model is `continuous` or `discrete`). They are kept out of the
top-level namespace so it stays focused on what you type. For backwards
compatibility `tsdynamics.Lorenz` (and `from tsdynamics import Lorenz`) still
resolve lazily. The classes are documented on their generated pages under
[Systems](../systems/index.md), and discoverable programmatically through the
[registry](registry.md).

## Renamed in v6

| Was | Now | Why |
| --- | --- | --- |
| `ts.basins(...)` | `ts.basins_of_attraction(...)` | the short alias was a *function* while `ts.analysis.basins` is the *subpackage* — one word, two objects, one dot apart |

`ts.basins` answers with an `AttributeError` naming the surviving spelling, not
a bare "no attribute".
