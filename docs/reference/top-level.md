---
description: The seventeen names in the tsdynamics namespace, the rule that decides membership, where everything else lives, and the error ladder that hands you the new address.
---

<span class="ts-kicker">Reference</span>

# Top level

`tsdynamics.<TAB>` shows **seventeen** names. That is not minimalism for its own
sake — it is one rule applied mechanically:

> A public name is a **verb** you call on the thing you are already holding, a
> **fact** about that thing, or a member of exactly one named **registry**.
> Everything else still exists — importable, reachable, tested — it just stops
> shouting.

```python
import tsdynamics as ts

traj = ts.systems.Lorenz().run(final_time=100.0, dt=0.01)
exps = ts.analysis.lyapunov_spectrum(ts.systems.Henon())
ts.plot(traj, color="crimson", title="Lorenz")
```

## The seventeen

```text
BackendError           DiscreteMap             StochasticSystem       plot
ContinuousSystem       InvalidInputError       StepBudgetError        systems
ConvergenceError       InvalidParameterError   TSDynamicsError        viz
DelaySystem            Trajectory              WrappedSystem          __version__
                                               analysis
```

Five classes you subclass · one type you receive and annotate · one plotting verb
· three registries · six names you type inside `except` · the version.

| Symbol | What it is | Canonical home |
| ------ | ---------- | -------------- |
| [`ContinuousSystem`][tsdynamics.families.continuous.ContinuousSystem] | subclass it to define an ODE | `tsdynamics.families.continuous` |
| [`DelaySystem`][tsdynamics.families.delay.DelaySystem] | …a DDE | `tsdynamics.families.delay` |
| [`DiscreteMap`][tsdynamics.families.discrete.DiscreteMap] | …a map | `tsdynamics.families.discrete` |
| [`StochasticSystem`][tsdynamics.families.stochastic.StochasticSystem] | …an SDE | `tsdynamics.families.stochastic` |
| [`WrappedSystem`][tsdynamics.families.wrapped.WrappedSystem] | …or adapt an external stepper | `tsdynamics.families.wrapped` |
| [`Trajectory`][tsdynamics.data.Trajectory] | what every `run` returns — and what you build from measured data | `tsdynamics.data` |
| `plot` | the plotting front door (`ts.plot is ts.viz.plot`) | `tsdynamics.viz` |
| `systems` | the 177 built-in models | — |
| `analysis` | the 50 quantifiers | — |
| `viz` | plots, primitives, transforms, themes, renderers | — |
| `TSDynamicsError` … `StepBudgetError` | the six exception types you catch | `tsdynamics.errors` |
| `__version__` | the installed version | — |

The five family bases are here because each is the **sole spelling of a
capability**: there is no verb and no plain-Python path that defines a delay
system for you. `Trajectory` is the one type you *receive* that is also a type
you *type* — you annotate it, you `isinstance` it, and you construct it
(`ts.Trajectory(t, y)`) from data measured elsewhere.

The six error classes are here for the same reason: `except ConvergenceError:` is
something you type, and reaching for `ts.errors.ConvergenceError` to write a
single `except` clause is a toll. Each one subclasses the builtin you would
historically have caught, so a `RuntimeError` / `ValueError` / `TypeError`
`except` keeps catching the same failure.

| You catch | It is a | Raised when |
| --------- | ------- | ----------- |
| `InvalidParameterError` | `ValueError` | a bad `dt`, solver, backend, section, parameter value |
| `InvalidInputError` | `TypeError` | a malformed argument — an array of the wrong shape, the wrong subject |
| `ConvergenceError` | `RuntimeError` | divergence, non-convergence, no section crossing |
| `StepBudgetError` | a `ConvergenceError` | the step budget ran out with a **finite** state — a settings problem, not a blow-up |
| `BackendError` | `RuntimeError` | the compiled engine is unavailable |
| `TSDynamicsError` | the base | catch-all for everything the library raises |

## Where everything else lives

Nothing was deleted. Roughly 230 names are merely **demoted** — they live at one
importable address each, and the error you get from guessing wrong hands you that
address (see the ladder below).

| Go here | For | When you would |
| ------- | --- | -------------- |
| [`ts.analysis`](analysis.md) | the 50 quantifiers + `find` + `register` + `results` | always — this is where analyses live |
| [`ts.systems`](../systems/index.md) | the 177 models, plus `names()` / `find()` / `get()` | `ts.systems.Lorenz()` |
| [`ts.viz`](../visualization/index.md) | `Plot`, `draw`, `grid`, `geometry`, `transforms`, `primitives`, `themes`, `renderers`, `spec` | you are styling, composing, exporting or extending a plot |
| [`ts.analysis.results`](analysis.md) | the 32 result classes | annotating a function that returns one |
| [`ts.viz.spec`](../visualization/index.md) | the IR nouns (`Layer`, `Axis`, `PlotKind`, `Geometry`, `T`, …) | writing a renderer or a transform |
| [`ts.derived`](derived.md) | `PoincareMap`, `StroboscopicMap`, `TangentSystem`, `ProjectedSystem`, `Ensemble` | rarely — the first two and `Ensemble` have verbs on the system |
| [`ts.data`](data.md) | `Trajectory`, `Box`, `Ball`, `Grid`, `region`, `sampler`, `grid_points`, `set_distance` | rarely — every region argument takes plain bounds |
| [`ts.errors`](../contributing/glossary.md) | the same six, plus `MovedInV6` and the warning types | reading the hierarchy |
| [`ts.registry`](registry.md) | the system / analysis / renderer / transform registries | enumerating a catalogue programmatically |
| [`ts.families`](base.md) | `SystemBase`, `ParamSet`, the `System` protocol | writing a new *family*, not a new system |
| [`ts.utils`](utils.md) | `make_output_grid`, the tolerance constants | reading what a default actually is |
| `ts.engine` / `ts.solvers` | the Rust-facing compile/run seam, the solver table | debugging the engine |

### The derived views have verbs

Two derived systems are built by a verb on the system they wrap, because they
speak the same vocabulary the class does:

```python
import numpy as np

lor = ts.systems.Lorenz()
states = np.random.default_rng(0).normal(size=(8, 3))

lor.poincare("y", 0.0)                 # a PoincareMap  (also ("y", 0.0, "up"))
ts.systems.Duffing().poincare(period=4.488)    # a strobe — the same verb
band = lor.ensemble(states)            # an Ensemble — a system holding 8 copies
band.run(final_time=5.0).final         # ...run it → the (8, 3) array of end states
```

`poincare` takes **either** a plane **or** a period, never both: a plane is an
affine surface $g(\mathbf u) = \mathbf n\cdot\mathbf u - c$, a period samples the
phase circle, and they are different sections of the same flow.

The two that are machinery rather than everyday verbs live at their address:

```python
from tsdynamics.derived import TangentSystem, ProjectedSystem

TangentSystem(lor, k=2)          # the Lyapunov engine, steppable
ProjectedSystem(lor, (0, 2))     # the (x, z) shadow, as a live system
```

### The analyses are free functions

An analysis is **not** a method. It is a free function whose first argument is
the thing it is about — a system, a trajectory, or a result:

```python
ts.analysis.lyapunov_spectrum(lor)               # a property of the equations
ts.analysis.correlation_dimension(traj)          # a property of a point set
ts.analysis.kaplan_yorke_dimension(exps)         # a property of the answer above
```

Pressing ++tab++ on a system therefore shows **19 names**, every one a verb or a
fact about that system, and `ts.analysis.<TAB>` shows the 50 quantifiers, flat
and sorted. `ts.analysis.find(subject_or_query)` is the search:

```python
ts.analysis.find(traj)               # what can I measure on THIS?
ts.analysis.find("is this chaotic")  # who answers THIS question?
```

### Regions take plain bounds

Everything that asks for a region reads **one `(lo, hi)` pair per state
component** — add a third entry, `(lo, hi, n)`, to say how finely to grid it.
No library type is ever required:

```python
vdp, henon = ts.systems.VanDerPol(), ts.systems.Henon()

ts.analysis.fixed_points(vdp, region=[(-3, 3), (-3, 3)])
ts.analysis.basins(henon, [(-2, 2, 20), (-2, 2, 20)], max_steps=500)
ts.analysis.expansion_entropy(henon, [(-1.6, 1.6), (-0.5, 0.5)], n_samples=100, n=8)
```

A `ts.data.Box` / `Ball` / `Grid` is still accepted at every one of those doors.
It is simply never the price of entry.

## The 177 built-in systems

Built-in system classes live under [`tsdynamics.systems`](../systems/index.md) —
the canonical path is `ts.systems.Lorenz` (flat, no need to remember whether a
model is `continuous` or `discrete`). `ts.Lorenz` **no longer resolves**: the
listing above is the whole top level, and a name that is not in it does not
answer. What you get instead is its address (below). The classes are documented
on their generated pages under [Systems](../systems/index.md), and discoverable
programmatically through the [registry](registry.md) or `ts.systems.find(...)`.

## The error ladder — how you find a name that moved

Curation hides ~230 reachable names from autocomplete, so **the error message is
the discovery mechanism**, and it is built to be one. A miss is answered in five
ordered cases.

**1 – 3. An exact hit in a redirect table raises `MovedInV6`** — a subclass of
`ImportError`, deliberately, because `from tsdynamics import X` discards a module
`__getattr__`'s message when it is an `AttributeError` and keeps it verbatim when
it is an `ImportError`. Both spellings therefore teach:

```pycon
>>> ts.Lorenz
tsdynamics.errors.MovedInV6: ts.Lorenz moved in v6: the top level is 17 names now,
and this one lives at its own address.
    ts.systems.Lorenz()

>>> from tsdynamics import PlotSpec
tsdynamics.errors.MovedInV6: ts.PlotSpec was renamed in v6. Same capability, one
spelling:
    ts.viz.Plot
(it is the type ts.plot hands back, and nothing about it is a spec any more — same
class, renamed, nothing wrapped)
```

**4. A near miss on a name that does exist gets suggestions:**

```pycon
>>> ts.lyapunov
AttributeError: module 'tsdynamics' has no attribute 'lyapunov'. Did you mean:
    ts.analysis.lyapunov_spectrum
    ts.analysis.lyapunov_from_data
```

**5. A genuine miss names the two catalogues worth tab-completing:**

```pycon
>>> ts.random_typo
AttributeError: module 'tsdynamics' has no attribute 'random_typo'.
Tab-complete a registry, or search:
    ts.systems.<TAB>             # the 177 built-in systems
    ts.analysis.<TAB>            # the 50 quantifiers
    ts.analysis.find('random_typo') # ...or search by what it does
```

Cases 4 and 5 stay `AttributeError`, so `hasattr(ts, "anything")` is still
`False` for a name nobody has ever used, and `dir()`,
`inspect.getmembers()` and `from tsdynamics import *` are untouched.

## Renamed or removed in v6

| Was | Now | Why |
| --- | --- | --- |
| `sys.integrate(...)` / `sys.iterate(...)` / `sys.trajectory(...)` | `sys.run(...)` | one verb produces data, on every family |
| `run(method="rk45")` | `run(solver="rk45")` | `solver=` picks a kernel, `method=` picks an *estimator* |
| `sys.is_discrete` | `sys.family` | `"ode"` / `"dde"` / `"map"` / `"sde"` — four answers a boolean could not give |
| `sys.stroboscope(period=T)` | `sys.poincare(period=T)` | one section verb, two kinds of section |
| `sys.project(...)` / `sys.tangent(...)` / `sys.copies(...)` | `ts.derived.ProjectedSystem` / `TangentSystem` / `sys.ensemble` | not everyday verbs; `copies` was one word from `ensemble` |
| `sys.lyap.spectrum()`, `traj.dims.correlation()`, … | `ts.analysis.<name>(subject)` | an analysis has one door, and it is the one that cannot hide a sampling choice |
| `result.summary()` | `repr(result)` | the good text already existed inside `summary()`, which nothing advertised |
| `PlotSpec` | `Plot` | same class — escalating from easy to expert is not a type change |
| `ts.basins_of_attraction` / `ts.find_attractors` | `ts.analysis.basins` / `ts.analysis.attractors` | the nouns a user types |
| `ts.bifurcation_diagram` | `ts.analysis.orbit_diagram` | one implementation cannot name two spellings in an error message |
| `ts.basins` (the top-level function) | **deleted** — `ts.analysis.basins` | it was a *function* shadowing a *subpackage*: one word, two objects |
