---
title: Plot gallery
description: Every plot TSDynamics can draw, generated from the plot-transform registry — each transform, each primitive it declares, the figure it makes, and the one line of code that made it.
---

<span class="ts-kicker">Visualization</span>

# The gallery

Browse the available views and drawing choices below. Each figure is generated
from the code beside it. Start with a nearby example, then change the data,
components and style. The [recipes](recipes.md) show how to combine views into
complete scientific figures.

The gallery is generated from the transform registry. An extension can supply
an `example=` factory for automatic inclusion; unchanged figures are cached.

## How to read it

Three ideas structure the whole visualization layer, and this page is laid out
along them.

**A transform** turns a subject — a `Trajectory`, a system, an analysis result,
a bare array — into *plottable geometry*: coordinates, channels, labels, a frame.
It is named, registered and documented, and it owns no new mathematics (it adapts
an estimator from [`tsdynamics.analysis`](../analysis/index.md)).

**A primitive** is how that geometry is *drawn* — a line, a point cloud, an
image, a quiver field, a 3-D surface. "Primitive" does not mean "simple".

**The compatibility matrix** says which primitives may draw which transform. It
is **declared**, one row per transform, at the transform's definition site — so
an invalid pair raises, naming the valid set, instead of quietly drawing
something that looks plausible and is wrong. The tabs under each entry below are
exactly that row.

Transforms come in exactly **two source categories**, and the page is split by
them:

<div class="grid cards" markdown>

-   **`data`** — computable from a series or a point set.

    A `Trajectory`, a NumPy array, an analysis result. It **also** accepts a
    system, which is simulated before a data view is built.

-   **`model`** — must evaluate or integrate the right-hand side.

    At points that are *not* in the input: a lattice of initial conditions, a
    Jacobian at an equilibrium. A bare array cannot serve, and it says so.

</div>

## The one-liner

Everything on this page is reachable through one function:

```python
import tsdynamics as ts

# skip-doctest — the calling pattern; `subject` and the transform name are yours
ts.plot(subject, "transform_name", **options)                 # draw it
ts.plot(subject, "transform_name", primitive="contour")       # draw it differently
ts.plot(subject, "flow_speed", "nullclines", "streamlines")   # overlay several
```

`ts.plot` always returns a [`Plot`](plotting.md), which renders itself —
`.show()`, `.save("figure.pdf")`, `.render("plotly")` — so a gallery snippet is
also a working first line of your own figure. Use the lower-level functions when
you want plot coordinates or drawing layers instead of a rendered figure:

```python
# skip-doctest — the escape-hatch pattern; `subject` is yours
g = ts.viz.geometry(subject, "ftle", grid=201)   # the arrays, and stop there
ts.viz.draw(g, "contour")                        # hand them back to the library
```

And `ts.viz.compatibility()` prints the same matrix this page draws.

{{ gallery }}

## Adding a plot

A transform is a registered function returning a channel mapping. The decorator
can infer its name and compatible drawing choices from that mapping. Explicit
declarations make a diagnostic's coordinate meaning and supported primitives
clearer:

```python
# skip-doctest — a template: `measure` is yours, and registering a placeholder
# transform here would put it in every listing for the rest of the session
import tsdynamics as ts

@ts.viz.transforms.register(source="data",              # or "model"
                            frame="scaling",
                            kind="scaling_curve",
                            primitives=("line", "points"))
def my_diagnostic(subject, *, option=1.0):
    """One line — it becomes the summary in the matrix above."""
    r, c = measure(subject, option)
    return {"x": r, "y": c}              # a plain channel mapping — no IR type
```

Everything else is **derived**: the name from the function, the summary from the
docstring, the axis count from the frame's arity, the default primitive from the
first one listed, the accepted subjects from `source`. The optional keywords are
`name=`, `default_primitive=`, `aliases=`, `labels=`, `role=`, `analysis=` (the
estimator this adapts) and `example=` (a small subject the gallery can draw).

The registration carries the compatibility row, so
`tests/test_viz_compatibility.py` starts rendering every cell of it, and this
gallery starts showing it — on the small fixture the `example=` factory returns,
until someone curates a nicer subject in `docs/_tooling/gallery.py`. The
generator prints any uncurated transform at the end of a run, so a new one is
visible the moment it lands.

Out of tree, declare the same function against the `tsdynamics.plot_transforms`
entry-point group so it can join the registry when discovered. Include an
`example=` factory for tools that generate a gallery or check every supported
primitive. See [writing a transform](plotting.md#writing-a-transform).
