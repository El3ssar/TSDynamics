---
description: Every plot TSDynamics can draw, generated from the plot-transform registry — each transform, each primitive it declares, the figure it makes, and the one line of code that made it.
---

<span class="ts-kicker">Visualization</span>

# The gallery

This page is the answer to *"what plots can this library actually draw?"* — and
it is **generated from the registry at build time**, not written by hand. Every
figure below was produced by the code printed beside it, on this machine, during
this documentation build.

That matters more than it sounds. A hand-written gallery drifts: a transform is
renamed, an option changes, a picture stays. Here the page is built by walking
`registry.plot_transforms`, and each snippet string is *executed* to make the
figure it is shown next to. A transform that stopped drawing would fail the docs
build; a transform added tomorrow appears here tomorrow with no edit to this
page.

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
    system, because a model gives you data for free.

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
ts.plot(subject, "nullclines", "direction_field", "streamlines")   # overlay several
```

`ts.plot` always returns a [`PlotSpec`](plotting.md), which renders itself —
`.plot()`, `.save("figure.pdf")`, `.render("plotly")` — so a gallery snippet is
also a working first line of your own figure. Two rungs below it are public when
you want the numbers rather than the picture:

```python
# skip-doctest — the escape-hatch pattern; `subject` is yours
g = ts.viz.geometry(subject, "ftle", grid=201)   # the arrays, and stop there
spec = ts.viz.draw(g, "contour")                 # hand them back to the library
```

And `ts.viz.compatibility()` prints the same matrix this page draws.

{{ gallery }}

## Adding a plot

A new transform is **one registration and nothing else** — no renderer edit, no
`PlotKind` edit, no `compose` edit, no test edit:

```python
from tsdynamics.viz.transforms import Geometry, plot_transform

@plot_transform(
    name="my_diagnostic",
    source="data",                       # or "model"
    default_primitive="line",
    primitives=("line", "points"),       # the declared compatibility row
    frame="scaling",
    ndim=1,
    doc="One line, shown in the matrix above.",
    example=lambda primitive: (my_small_fixture(), {}),
)
def my_diagnostic(subject, *, option=1.0) -> Geometry:
    ...
```

The registration carries the compatibility row, so
`tests/test_viz_compatibility.py` starts rendering every cell of it, and this
gallery starts showing it — on the small fixture the `example=` factory returns,
until someone curates a nicer subject in `docs/_tooling/gallery.py`.

Out of tree, declare the same function against the `tsdynamics.plot_transforms`
entry-point group and it joins the registry — and this page — on import.
