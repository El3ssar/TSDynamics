---
description: Package your own models, analyses, plots and rendering backends, install them once, and use them through the normal TSDynamics APIs. Includes a complete model package and the limits of solver extensions.
---

<span class="ts-kicker">Project · Plugins & extensions</span>

# Plugins & extensions

A plugin is an ordinary installed Python package. A small declaration in its
`pyproject.toml` tells TSDynamics which class or function to import. You install
the package in your Python environment, restart Python, and use its additions
through the usual `ts.systems`, `ts.analysis` and `ts.viz` APIs.

You do **not** need a plugin to experiment in a notebook. Define a system class
and call it directly, or register a plotting function in your current session.
Packaging is useful when you want to reuse or share those additions without
copying the definitions into every script.

## Choose what you are adding

| Addition | Entry-point group | What the package provides | Where you use it |
| --- | --- | --- | --- |
| A model | `tsdynamics.systems` | A concrete system class | `ts.systems.MyOscillator()` |
| A measurement | `tsdynamics.analyses` | An analysis function | `ts.analysis.sampled_amplitude(traj)` |
| A plot transform | `tsdynamics.plot_transforms` | A registered function that produces plot coordinates | `ts.plot(traj, "oscillator_phase")` |
| A drawing primitive | `tsdynamics.plot_primitives` | A function that turns channels into drawing marks | `ts.viz.draw(data, "oscillator_stems")` |
| A rendering backend | `tsdynamics.renderers` | A callable that consumes a plot description | `plot.render(backend="model_summary")` |
| A solver declaration | `tsdynamics._solvers` | A `SolverSpec` describing an implemented numerical kernel | `system.run(solver="oscillator_rk4")` |

The first five can be written in Python. A new integration algorithm
also needs a Rust implementation; [the solver section](#solver-extensions) explains
the boundary.

## A complete model package

Create this directory:

```text
oscillator-addon/
├── pyproject.toml
└── src/
    └── oscillator_addon/
        ├── __init__.py
        └── models.py
```

Leave `__init__.py` empty. Put the following in `models.py`:

```python title="src/oscillator_addon/models.py"
import tsdynamics as ts


class MyOscillator(ts.ContinuousSystem):
    """A harmonic oscillator: x' = v, v' = -omega² x."""

    params = {"omega": 1.0}
    dim = 2
    variables = ("x", "v")

    @staticmethod
    def _equations(y, t, *, omega):
        return (y(1), -(omega ** 2) * y(0))
```

`y(i)` reads a state component. The function returns symbolic expressions,
which the existing engine compiles; adding a model does not require changing or
rebuilding the Rust engine. The other family contracts are covered in
[Defining systems](../start/defining-systems.md).

Put this in `pyproject.toml`:

```toml title="pyproject.toml"
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "oscillator-addon"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["tsdynamics"]

[tool.setuptools.packages.find]
where = ["src"]

[project.entry-points."tsdynamics.systems"]
MyOscillator = "oscillator_addon.models:MyOscillator"
```

The last line means: import `oscillator_addon.models`, then expose its
`MyOscillator` class under that same name. The package name, `oscillator-addon`,
and import name, `oscillator_addon`, may differ; the entry point must use the
**import** name.

Install it into the environment where you use TSDynamics. From the add-on's
directory, with that environment active:

```bash
python -m pip install -e .
```

The editable install lets you develop the package in place. With a uv-managed
environment, `uv pip install -e .` is the corresponding command. Published
packages work the same way with `python -m pip install oscillator-addon`.

Restart Python, or restart your notebook's kernel, after installing the package
or changing entry-point declarations. Discovery happens when the corresponding
TSDynamics namespace first loads; reinstalling a package does not refresh a
namespace already in memory.

In a fresh session:

```python
# skip-doctest — run after installing the separate package above
import tsdynamics as ts

model = ts.systems.MyOscillator(omega=2.0, ic=[1.0, 0.0])
traj = model.run(final_time=5.0, dt=0.01)

ts.systems.get("MyOscillator") is ts.systems.MyOscillator  # True
ts.systems.find("MyOscillator")                          # the installed class
ts.plot(traj)                                           # the usual plotting API
```

The entry-point name must be a public Python identifier and match the class's
name. A plugin class must implement one of the four model families. Installed
models remain external entries: they do not enter the built-in system sweeps or
acquire generated catalogue pages automatically.

## Add an analysis

Add `measurements.py` to the same package. This measurement reports half the
sampled range of one component; its value depends on the observation window and
sampling, so it is not a guarantee of the underlying oscillator's true amplitude.

```python title="src/oscillator_addon/measurements.py"
# skip-doctest — this module registers an installed package's analysis
import numpy as np
from tsdynamics.analysis import register
from tsdynamics.analysis.results import ScalarResult


@register(subjects=("trajectory",), area="geometry", returns=ScalarResult,
          keywords="oscillation amplitude sampled range")
def sampled_amplitude(traj, *, component="x") -> ScalarResult:
    """Half the sampled range of a trajectory component."""
    values = np.asarray(traj[component], dtype=float)
    if values.size == 0:
        raise ValueError("sampled_amplitude needs at least one sample")
    return ScalarResult(value=float(np.ptp(values) / 2.0),
                        meta={"component": component})
```

Add this table to `pyproject.toml`:

```toml
[project.entry-points."tsdynamics.analyses"]
sampled_amplitude = "oscillator_addon.measurements:sampled_amplitude"
```

The decorator supplies discovery metadata: which inputs the function accepts,
its area, return type and search terms. The docstring supplies its summary.
Choose an existing area such as `geometry`, `chaos`, `orbits`, `dimensions` or
`sampling`; arbitrary area names are rejected. A bare callable is also a valid
entry-point target, but without registration metadata it cannot participate in
subject-based discovery as usefully.

Reinstall the add-on and restart Python. The function then appears in
`ts.analysis.find("sampled amplitude")` and `ts.analysis.find(traj)`:

```python
# skip-doctest — requires the installed measurements module
result = ts.analysis.sampled_amplitude(traj, component="x")
amplitude = float(result)
```

## Add a plot transform

A transform chooses the coordinates to draw. It should adapt existing data or
an analysis result, with new numerical estimators kept in the analysis layer.
This one selects the oscillator's position and velocity:

```python title="src/oscillator_addon/plots.py"
# skip-doctest — this module registers an installed package's plot transform
import tsdynamics as ts


@ts.viz.transforms.register(source="data", frame="state2",
                            kind="phase_portrait_2d", primitives=("line", "points"),
                            labels=("position", "velocity"))
def oscillator_phase(traj):
    """Velocity against position for the oscillator."""
    return {"x": traj["x"], "y": traj["v"]}
```

```toml
[project.entry-points."tsdynamics.plot_transforms"]
oscillator_phase = "oscillator_addon.plots:oscillator_phase"
```

The decorator performs registration when the module is imported. Keep the
entry-point name equal to the registered transform's name. After reinstalling
and restarting, use `ts.plot(traj, "oscillator_phase")`, choose
`primitive="points"`, or inspect its row in `ts.viz.compatibility()`.
For a local experiment, importing the decorated function is enough; packaging
is what makes that import automatic. See [Transforms & primitives](../visualization/plotting.md#writing-a-transform)
for inferred declarations and more options.

## Add a drawing primitive

A primitive turns channels such as `x` and `y` into the marks existing renderers
already understand. This example draws all stems as one line layer and all
markers as one point layer, avoiding a separate layer for each sample:

```python title="src/oscillator_addon/primitives.py"
# skip-doctest — this module registers an installed package's drawing primitive
import numpy as np
import tsdynamics as ts


@ts.viz.primitives.register("oscillator_stems", requires=("x", "y"),
                           marks=("line", "points"))
def oscillator_stems(part, **options):
    """A line from zero to each sample, with a point at its tip."""
    x, y = np.asarray(part["x"]), np.asarray(part["y"])
    gaps = np.full(x.shape, np.nan)
    xs = np.column_stack((x, x, gaps)).ravel()
    ys = np.column_stack((np.zeros_like(y), y, gaps)).ravel()
    return [{"mark": "line", "x": xs, "y": ys},
            {"mark": "points", "x": x, "y": y}]
```

```toml
[project.entry-points."tsdynamics.plot_primitives"]
oscillator_stems = "oscillator_addon.primitives:oscillator_stems"
```

Use it directly with `ts.viz.draw({"x": x, "y": y}, "oscillator_stems")`.
To offer it on a transform as well, allow it **after both have loaded**:

```python
# skip-doctest — requires the installed transform and primitive
ts.viz.transforms.allow("oscillator_phase", "oscillator_stems")
ts.plot(traj, "oscillator_phase", primitive="oscillator_stems")
```

The transform-to-primitive compatibility row is checked. Installing a primitive
alone does not make every transform accept it.

## Add a rendering backend

A renderer consumes the finished plot description and returns a figure or an
export payload. It does not compute the model or the analysis. This small
diagnostic exporter returns a summary, demonstrating the complete callable and
capability declaration without requiring a graphics library:

```python title="src/oscillator_addon/renderers.py"
# skip-doctest — this callable is loaded from the installed package
from tsdynamics.viz.render import RendererCapabilities, RenderResult


def model_summary(spec):
    """Describe a plot without drawing it."""
    return RenderResult(backend="model_summary", kind=spec.kind,
                        payload={"kind": spec.kind.value, "layers": len(spec.layers)},
                        mimetype="application/json")


model_summary.capabilities = RendererCapabilities(
    name="model_summary", supports_3d=True, data_export=True,
    render_kwargs=frozenset(),
)
```

```toml
[project.entry-points."tsdynamics.renderers"]
model_summary = "oscillator_addon.renderers:model_summary"
```

```python
# skip-doctest — requires the installed renderer
payload = ts.plot(traj, "oscillator_phase").render(backend="model_summary").payload
```

`data_export=True` keeps this exporter out of the default drawing choice.
For a graphical backend, declare the plot kinds, 3-D support and render keywords
you actually implement. File saving additionally needs truthful
`writes_static` / `writes_animated` declarations and a result that the save path
can write; registering a renderer does not automatically implement new file
formats. See [Backends & export](../visualization/backends.md).

## Solver extensions

`SolverSpec` describes a solver's name, capabilities and implementation name.
It contains **no numerical algorithm**. A Python package can declare an
alternative selectable name for a kernel that is already compiled into the
engine, for example:

```python title="src/oscillator_addon/solvers.py"
# skip-doctest — this declaration is loaded from the installed package
from tsdynamics._solvers import SolverCaps, SolverSpec

OSCILLATOR_RK4 = SolverSpec(
    name="oscillator_rk4", kernel="rk4", origin="plugin",
    caps=SolverCaps(kind="explicit", supports=frozenset({"ode"})),
    description="Classic RK4 under this package's selectable name",
)
```

```toml
[project.entry-points."tsdynamics._solvers"]
oscillator_rk4 = "oscillator_addon.solvers:OSCILLATOR_RK4"
```

After installation, `model.run(final_time=5, dt=0.01, solver="oscillator_rk4")`
uses the existing `rk4` implementation. Trajectory metadata records the kernel
that actually ran. Capability flags must describe that implementation; setting
`kernel="new_algorithm"` cannot create a new solver.

To add a new method, implement and register it in Rust, mirror its
capabilities in Python, and build an engine extension that includes it. The
registration is part of that compiled engine, not a runtime loader for arbitrary
Rust libraries. Adaptive methods also need their tolerance handling wired into
the binding. The Python SDE family currently implements only
`euler_maruyama` and `milstein`; a new SDE method needs the corresponding family
and reference-step support as well. Follow [Adding a solver](adding-a-solver.md).

## Names, failures and debugging

- **Use distinctive names.** Existing registered entries are protected by the
  normal discovery paths. Do not use `replace=True` in an add-on's decorators
  to replace somebody else's model, measurement or plot.
- **System-name duplicates are rejected together.** If two distributions publish
  the same system entry-point name, neither target is imported. The other plugin
  groups use their own registration rules; avoid relying on installation order
  to select among duplicate names.
- **Import failures warn and skip.** A missing optional dependency or an exception
  during loading names the failing entry. A failing solver factory also rolls
  back its registry changes. This is error isolation, not a sandbox for untrusted
  Python code.
- **Check the same environment.** The add-on must be installed for the Python
  interpreter running your script or notebook. After editing `pyproject.toml`,
  reinstall it and restart that interpreter.
- **Check discovery before plotting or running.** Look in `ts.systems.names()`,
  `ts.analysis.find("your name")`, `ts.viz.transforms.names()`,
  `ts.viz.primitives.names()` or `ts.viz.renderers.names()` as appropriate.

To inspect Python's installed declarations without loading their targets:

```python
from importlib.metadata import entry_points

declared_models = [(ep.name, ep.value) for ep in entry_points(group="tsdynamics.systems")]
```

For detailed import tracebacks during development, the internal discovery
helpers for analyses, solvers and visualization accept `strict=True`. The normal
user workflow is still install, restart and use the public registry; there is no
separate activation command.
