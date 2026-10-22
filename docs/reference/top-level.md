---
description: Main imports, analysis and system discovery, typed errors, and migration paths for names moved in v6.
---

<span class="ts-kicker">Reference</span>

# Top level

The top-level namespace contains the family classes, `Trajectory`, plotting,
the system and analysis catalogues, and the installed version. Begin with:

```python
import tsdynamics as ts

traj = ts.systems.Lorenz().run(final_time=100.0, dt=0.01)
exps = ts.analysis.lyapunov_spectrum(ts.systems.Henon())
ts.plot(traj, color="crimson", title="Lorenz")
```

## The eleven

```text
ContinuousSystem       StochasticSystem        plot
DelaySystem            Trajectory              systems
DiscreteMap            WrappedSystem           viz
                       __version__             analysis
```

Five classes you subclass · one type you receive and annotate · one plotting verb
· three registries · the version.

| Symbol | What it is | Canonical home |
| ------ | ---------- | -------------- |
| [`ContinuousSystem`][tsdynamics.families.continuous.ContinuousSystem] | subclass it to define an ODE | `tsdynamics.families.continuous` |
| [`DelaySystem`][tsdynamics.families.delay.DelaySystem] | …a DDE | `tsdynamics.families.delay` |
| [`DiscreteMap`][tsdynamics.families.discrete.DiscreteMap] | …a map | `tsdynamics.families.discrete` |
| [`StochasticSystem`][tsdynamics.families.stochastic.StochasticSystem] | …an SDE | `tsdynamics.families.stochastic` |
| [`WrappedSystem`][tsdynamics.families.wrapped.WrappedSystem] | …or adapt an external stepper | `tsdynamics.families.wrapped` |
| [`Trajectory`][tsdynamics.data.Trajectory] | single-system simulation output, also constructible from measured data | `tsdynamics.data` |
| `plot` | the plotting function (`ts.plot is ts.viz.plot`) | `tsdynamics.viz` |
| `systems` | the 177 built-in models | — |
| `analysis` | registered analysis functions | — |
| `viz` | plots, primitives, transforms, themes, renderers | — |
| `__version__` | the installed version | — |

Subclass a family to define a model, or use `WrappedSystem` to adapt an external
stepper. Construct `Trajectory(t, y)` when working with measured arrays.

The six error classes live at `ts.errors.<Name>`, outside the top-level listing.
Each subclasses the builtin you would historically have caught, so a
`RuntimeError` / `ValueError` / `TypeError` `except` keeps catching the same
failure. Use the typed names when you need to distinguish library failures:

```python
try:
    ts.systems.Lorenz().run(final_time=1.0, dt=-0.1)
except ts.errors.InvalidParameterError as error:
    print(error)
```

| You catch | It is a | Raised when |
| --------- | ------- | ----------- |
| `InvalidParameterError` | `ValueError` | a bad `dt`, solver, backend, section, parameter value |
| `InvalidInputError` | `TypeError` | a malformed argument — an array of the wrong shape, the wrong subject |
| `ConvergenceError` | `RuntimeError` | divergence, non-convergence, no section crossing |
| `StepBudgetError` | a `ConvergenceError` | the step budget ran out with a **finite** state — a settings problem, not a blow-up |
| `BackendError` | `RuntimeError` | the compiled engine is unavailable |
| `TSDynamicsError` | the base | base class for library-specific errors |

## Where everything else lives

Use these submodules for more specialized classes and functions. Names moved in
v6 raise a migration message with their current location.

| Go here | For | When you would |
| ------- | --- | -------------- |
| [`ts.analysis`](analysis.md) | quantifiers, `find`, `register`, and `results` | finding or running an analysis |
| [`ts.systems`](../systems/index.md) | the 177 models, plus `names()` / `find()` / `get()` | `ts.systems.Lorenz()` |
| [`ts.viz`](../visualization/index.md) | `Plot`, `draw`, `grid`, `geometry`, `transforms`, `primitives`, `themes`, `renderers`, `spec` | you are styling, composing, exporting or extending a plot |
| [`ts.analysis.results`](analysis.md) | the 32 result classes | annotating a function that returns one |
| [`ts.viz.spec`](../visualization/index.md) | plot-description types (`Layer`, `Axis`, `PlotKind`, `Geometry`, `T`, …) | writing a renderer or a transform |
| [`ts.derived`](derived.md) | `PoincareMap`, `StroboscopicMap`, `TangentSystem`, `ProjectedSystem`, `Ensemble` | rarely — the first two and `Ensemble` have verbs on the system |
| [`ts.data`](data.md) | `Trajectory`, `Box`, `Ball`, `Grid`, `region`, `sampler`, `grid_points`, `set_distance` | rarely — every region argument takes plain bounds |
| [`ts.errors`](../contributing/glossary.md) | the six typed errors, plus `MovedInV6` and the warning types | catching a specific library failure |
| [`ts.registry`](registry.md) | the system / analysis / renderer / transform registries | enumerating a catalogue programmatically |
| [`ts.families`](base.md) | `SystemBase`, `ParamSet`, the `System` protocol | writing a new *family*, not a new system |
| [`tsdynamics._utils`](utils.md) | `make_output_grid`, the tolerance constants | internal implementation reference |
| `tsdynamics._engine` / `tsdynamics._solvers` | engine integration and the solver table | debugging internals; these are private modules |

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

Pressing ++tab++ on a Lorenz system shows **20 names**, every one a verb or a
fact about that system, and `ts.analysis.<TAB>` shows registered analysis functions, flat
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

Use `ts.data.Box`, `Ball` or `Grid` when you need a reusable region object;
check the analysis signature for accepted region types.

## The 177 built-in systems

Built-in system classes live under [`tsdynamics.systems`](../systems/index.md) —
the canonical path is `ts.systems.Lorenz` (flat, no need to remember whether a
model is `continuous` or `discrete`). `ts.Lorenz` **no longer resolves**: the
listing above is the whole top level, and a name that is not in it does not
answer. What you get instead is its address (below). The classes are documented
on their generated pages under [Systems](../systems/index.md), and discoverable
programmatically through the [registry](registry.md) or `ts.systems.find(...)`.

## The error ladder — how you find a name that moved

For a name moved in v6, `MovedInV6` (an `ImportError` subclass) explains the new
import or method. For example:

```pycon
>>> ts.Lorenz
tsdynamics.errors.MovedInV6: ts.Lorenz moved: the top level is 11 names now,
and this one lives at its own address.
    ts.systems.Lorenz()

>>> from tsdynamics import PlotSpec
tsdynamics.errors.MovedInV6: ts.PlotSpec was renamed in v6. Same capability, one
spelling:
    ts.viz.Plot
(it is the type ts.plot hands back, and nothing about it is a spec any more — same
class, renamed, nothing wrapped)
```

**Similar names produce suggestions:**

```pycon
>>> ts.lyapunov
AttributeError: module 'tsdynamics' has no attribute 'lyapunov'. Did you mean:
    ts.analysis.lyapunov_spectrum
    ts.analysis.lyapunov_from_data
```

**Unknown names point to catalogue discovery:**

```pycon
>>> ts.random_typo
AttributeError: module 'tsdynamics' has no attribute 'random_typo'.
Tab-complete a registry, or search:
    ts.systems.<TAB>             # the 177 built-in systems
    ts.analysis.<TAB>            # registered analysis functions
    ts.analysis.find('random_typo') # ...or search by what it does
```

Unknown-name errors remain `AttributeError`, so `hasattr(ts, "anything")` is still
`False` for a name nobody has ever used, and `dir()`,
`inspect.getmembers()` and `from tsdynamics import *` are untouched.

## Renamed or removed in v6

| Previous API | Current API |
|---|---|
| `sys.integrate(...)`, `sys.iterate(...)`, `sys.trajectory(...)` | `sys.run(...)` |
| `run(method="rk45")` | `run(solver="rk45")` |
| `sys.is_discrete` | `sys.family`: `"ode"`, `"dde"`, `"map"` or `"sde"` |
| `sys.stroboscope(period=T)` | `sys.poincare(period=T)` |
| `sys.project(...)`, `sys.tangent(...)`, `sys.copies(...)` | `ts.derived.ProjectedSystem`, `ts.derived.TangentSystem`, `sys.ensemble(...)` |
| `sys.lyap.spectrum()`, `traj.dims.correlation()`, … | `ts.analysis.name(subject, ...)` |
| `result.summary()` | `repr(result)` or `print(result)` |
| `PlotSpec` | `Plot` |
| `ts.basins_of_attraction`, `ts.find_attractors` | `ts.analysis.basins`, `ts.analysis.attractors` |
| `ts.bifurcation_diagram` | `ts.analysis.orbit_diagram` |
| `ts.basins` | `ts.analysis.basins` |
