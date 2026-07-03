---
description: Compose one or more plots into a figure with ts.viz.plot — overlaying compatible panels onto shared axes, tiling them into a stack / row / grid of COMPOSITE panels, and recursing spec-in-spec because the input and output are the same PlotSpec.
---

<span class="ts-kicker">Visualization</span>

# Composition

A single `to_plot_spec(...)` describes **one panel**. Real figures — a
before/after pair, a small-multiples grid of attractors, two state variables
stacked over a shared time axis — are *arrangements* of panels. `ts.viz.plot`
is the figure-level front door that does the arranging.

Its defining property is that it is **closed over `PlotSpec`**: it takes
plottables (a [`Trajectory`](../reference/top-level.md), a system, an analysis
result) or already-built specs, and it *returns a `PlotSpec`*. Because the input
type and the output type are the same, a `plot(...)` result feeds straight back
into `plot(...)` — you build each panel with one flat call, then arrange the
panels with another. The result renders itself: `.plot()`, `.save(path)`,
`.render(backend=)`, and a notebook display hook, exactly like a single-panel
spec.

<figure markdown>
![Two Rossler orbits overlaid on one set of x-y axes: a small teal limit cycle at c=2.3 and a wide indigo chaotic band at c=5.7, distinguished by an auto-generated legend](../assets/figures/viz/compose-overlay.svg){ loading=lazy }
<figcaption>An <strong>overlay</strong>: two Rossler orbits — the small teal limit cycle at <code>c = 2.3</code> and the wide indigo chaotic band at <code>c = 5.7</code> — merged onto <em>one</em> set of <code>(x, y)</code> axes with <code>layout="overlay"</code>. The legend is auto-disambiguated by source, since both panels came from a system named "Rossler".</figcaption>
</figure>

## The one verb

```python
import tsdynamics as ts

ts.viz.plot(*things, layout="overlay", animate=False, **build_kw)  # -> PlotSpec
```

- **`*things`** — any mix of plottables and already-built `PlotSpec` objects.
  A single list/tuple argument is unwrapped, so `plot([a, b])` and `plot(a, b)`
  are the same.
- **`layout`** — `"overlay"` (default) draws everything on one set of axes;
  `"stack"` / `"row"` / `"grid"` give each thing its own panel in a
  [`COMPOSITE`](#composite-tiling-panels-into-a-figure) figure.
- **`build_kw`** — `components=`, `kind=`, and the per-kind options are forwarded
  to each non-spec thing's `to_plot_spec`, so `plot(a, b, components="x")`
  composes the *same view* of each. `build_kw` **cannot** be combined with an
  already-built `PlotSpec` argument (you passed the build options when you first
  built it) — mixing the two raises `InvalidParameterError`.
- **`animate`** — animate the whole figure (see
  [Animating a figure](#animating-a-figure)).

---

## Overlay: many drawables, one set of axes

`layout="overlay"` (the default) merges compatible single-panel specs onto **one**
panel. It is the right layout when the drawables share a coordinate frame — two
orbits in the same phase plane, several state variables over the same time axis,
a signal and its surrogate.

The overlay figure above is one call:

```python
import tsdynamics as ts

r_cycle = ts.Rossler().with_params(c=2.3)   # a small limit cycle
r_chaos = ts.Rossler().with_params(c=5.7)   # the classic chaotic band
ic = [0.1, 0.0, 0.0]

t_cycle = r_cycle.trajectory(final_time=200.0, dt=0.05, ic=ic)
t_chaos = r_chaos.trajectory(final_time=200.0, dt=0.05, ic=ic)

fig = ts.viz.plot(t_cycle, t_chaos, components=["x", "y"])
fig.recolor("#11857A", "#574FCF").save("compose-overlay.svg")
```

```pycon
>>> fig.kind
<PlotKind.PHASE_PORTRAIT_2D: 'phase_portrait_2d'>
>>> len(fig.layers)              # both orbits are layers on one panel
2
>>> [layer.label for layer in fig.layers]
['Rossler (1)', 'Rossler (2)']
>>> fig.meta["composed"]         # the source tags, in order
['Rossler (1)', 'Rossler (2)']
```

Both sources carry the title "Rossler", so the overlay makes the legend
unambiguous by tagging each `Rossler (1)` / `Rossler (2)` and prefixing every
layer label with its source. When the sources have **distinct** titles the tags
are those titles verbatim (no counter). Give a source a clearer name by
relabelling its spec's `title` before you compose, or pin it inline:

```python
a = t_cycle.to_plot_spec(components=["x", "y"]).relabel(title="limit cycle")
b = t_chaos.to_plot_spec(components=["x", "y"]).relabel(title="chaos")
ts.viz.plot(a, b)   # legend entries: "limit cycle" / "chaos"
```

### What can be overlaid

Only kinds that genuinely share axes may overlay: `TIME_SERIES`,
`PHASE_PORTRAIT_2D`, and `PHASE_PORTRAIT_3D`. Every input must be the **same**
one of those. Mixing kinds — a time series and a 2-D portrait, or a 2-D and a
3-D portrait — cannot share one set of axes and raises, pointing you at a
panelled layout instead:

```pycon
>>> lor = ts.Lorenz()
>>> ts_spec = lor.to_plot_spec(components="x", final_time=40.0, dt=0.01, ic=[1.0, 1.0, 1.0])
>>> ph_spec = lor.to_plot_spec(components=["x", "y"], final_time=40.0, dt=0.01, ic=[1.0, 1.0, 1.0])
>>> ts.viz.plot(ts_spec, ph_spec)
Traceback (most recent call last):
    ...
tsdynamics.errors.InvalidParameterError: cannot overlay specs of kinds
['phase_portrait_2d', 'time_series'] on one set of axes; use layout='stack'
(or 'row' / 'grid') to give each its own panel.
```

Image / section / composite kinds (a recurrence plot, a spacetime image, a basin
image) each need their own axes and colorbar — they are **never** overlaid;
arrange them into panels.

!!! note "A single input passes through"
    `plot(one_thing)` with one argument returns that thing's spec unchanged
    (there is nothing to arrange). This is what lets `ts.viz.plot` double as a
    convenience wrapper — `ts.viz.plot(system).save("x.png")`.

---

## Composite: tiling panels into a figure

`layout="stack"` / `"row"` / `"grid"` build a `PlotKind.COMPOSITE` spec: each
thing keeps its own axes and becomes one **panel**, and a
[`Layout`](../reference/top-level.md) records how to tile them. Every renderer
that draws composites (matplotlib and plotly) maps the `Layout.mode` to its own
subplot grid.

<figure markdown>
![A 2x2 grid of four classic attractors — Lorenz, Rossler, Halvorsen and Thomas — each in its own panel and brand colour](../assets/figures/viz/compose-grid.svg){ loading=lazy }
<figcaption>A <strong>grid</strong> composite: four classic attractors, each its own panel, tiled 2×2 with <code>layout="grid"</code>. Small multiples like this are the natural way to compare systems side by side — the panels do not share axes, so each attractor keeps its own scale.</figcaption>
</figure>

```python
import tsdynamics as ts

lor, ros, hal, tho = ts.Lorenz(), ts.Rossler(), ts.Halvorsen(), ts.Thomas()

grid = ts.viz.plot(
    lor.to_plot_spec(components=[0, 2], final_time=100.0, dt=0.01, ic=[1.0, 1.0, 1.0]),
    ros.to_plot_spec(components=[0, 1], final_time=200.0, dt=0.05, ic=[0.1, 0.0, 0.0]),
    hal.to_plot_spec(components=[0, 1], final_time=100.0, dt=0.01, ic=[-5.0, 0.0, 0.0]),
    tho.to_plot_spec(components=[0, 1], final_time=300.0, dt=0.05, ic=[1.1, 1.1, -0.01]),
    layout="grid",
)
grid.save("compose-grid.svg")
```

```pycon
>>> grid.kind
<PlotKind.COMPOSITE: 'composite'>
>>> grid.is_composite
True
>>> len(grid.panels)
4
>>> grid.layout.mode
'grid'
>>> [p.title for p in grid.panels]
['Lorenz', 'Rossler', 'Halvorsen', 'Thomas']
```

Two of these systems (`Halvorsen`, `Thomas`) declare no named `variables`, so
their components are selected by **index** (`[0, 1]`); `Lorenz` and `Rossler`
would also accept `["x", "y"]`. Selecting by index always works — see
[Plotting — the front door](plotting.md).

### The three arrangements

| `layout` | `Layout.mode` | Shape |
| --- | --- | --- |
| `"stack"` | `stack` | one column of stacked panels |
| `"row"` | `row` | one row of side-by-side panels |
| `"grid"` | `grid` | a 2-D grid, near-square unless you set `rows` / `cols` |

`"stack"` has one convenience default: when **every** panel is a `TIME_SERIES`
that names the same x axis (the canonical "`x(t)` above `y(t)` over one shared
time axis" case), the panels auto-share their x axis.

```python
lz = ts.Lorenz()
px = lz.to_plot_spec(components="x", final_time=40.0, dt=0.01, ic=[1.0, 1.0, 1.0])
py = lz.to_plot_spec(components="y", final_time=40.0, dt=0.01, ic=[1.0, 1.0, 1.0])

stack = ts.viz.plot(px, py, layout="stack")
```

```pycon
>>> stack.layout.mode, stack.layout.share_x
('stack', True)
```

Any other arrangement — a `row`, a `grid`, or a stack of mixed kinds — keeps
independent axes. To force axis sharing, or to fix an explicit grid shape, build
the `Layout` by hand and attach it:

```python
from tsdynamics.viz.spec import Layout

grid.layout = Layout(mode="grid", rows=2, cols=2, share_x=True, share_y=True)
```

---

## Recursion: spec in, spec out

Because a composite *is* a `PlotSpec`, you can compose composites. When a
composite is passed into `plot`, its panels are **flattened one level** into the
new figure — so you build the pieces bottom-up and combine them:

```python
import tsdynamics as ts

lz = ts.Lorenz()
ic = [1.0, 1.0, 1.0]

# Panel 1: x(t) and y(t) stacked on a shared time axis.
px = lz.to_plot_spec(components="x", final_time=40.0, dt=0.01, ic=ic)
py = lz.to_plot_spec(components="y", final_time=40.0, dt=0.01, ic=ic)
timeline = ts.viz.plot(px, py, layout="stack")

# Panel 2: the phase portrait.
portrait = lz.to_plot_spec(components=["x", "z"], final_time=40.0, dt=0.01, ic=ic)

# Combine — the stack's two panels flatten in alongside the portrait.
figure = ts.viz.plot(timeline, portrait, layout="row")
```

```pycon
>>> figure.kind, figure.layout.mode
(<PlotKind.COMPOSITE: 'composite'>, 'row')
>>> len(figure.panels)          # timeline (2 panels, flattened) + portrait
3
```

The flattening is one level deep, which is what makes the pattern predictable:
the panels of any composite you pass in become panels of the new figure, never a
nested sub-figure.

---

## Animating a figure

`animate=` (or the chainable `.animate()` on the result) turns the whole figure
into a movie. A **composite** plays every panel in **lockstep on one master
clock**: the composite carries the master `Animation`, and each panel that is not
already animated gets the same timeline — with its own per-kind head default (a
head marker on portraits and spacetime, off on a plain time series).

```python
import tsdynamics as ts

lor, ros = ts.Lorenz(), ts.Rossler()
comp = ts.viz.plot(
    lor.to_plot_spec(components=[0, 2], final_time=30.0, dt=0.01, ic=[1.0, 1.0, 1.0]),
    ros.to_plot_spec(components=[0, 1], final_time=60.0, dt=0.05, ic=[0.1, 0.0, 0.0]),
    layout="row",
    animate=True,
)
comp.animate(fps=30, n_frames=120).save("two-attractors.mp4")
```

```pycon
>>> comp.is_animated
True
>>> [p.animation is not None for p in comp.panels]   # every panel plays in lockstep
[True, True]
```

An animated composite is a matplotlib job (its `FuncAnimation` drives every panel
on the shared clock); the plotly backend declines an animated composite and
dispatch falls back to matplotlib. See [Animation](animation.md) for the full
comet / trail / head / camera vocabulary and the export formats.

---

## Themes across a composite

A composite resolves its theme **per panel**, in this order:

```
panel.theme  →  composite.theme  →  the active global default
```

So to give every panel one look, set the theme on the composite result; to make
one panel different, set its theme *before* you pass it in.

```python
grid.theme("dark")                       # every panel goes dark
grid.panels[0].theme("publication")      # …except the first
```

An overlay is a single panel, so it just uses `spec.theme or the global default`.
See [Styling & themes](styling.md) for the full theme system.

---

## Pitfalls

| Situation | What happens |
| --- | --- |
| Overlaying incompatible kinds (image + portrait, 2-D + 3-D) | Raises `InvalidParameterError` — use `layout="stack"` / `"row"` / `"grid"`. |
| `build_kw` (`components=`, `kind=`) with an already-built `PlotSpec` | Raises `InvalidParameterError` — pass build options when you first build the spec. |
| Passing a plain array / a non-plottable | Raises `InvalidInputError` — wrap it in a `Trajectory` or build a `PlotSpec`. |
| Two sources with the same title | Legend tags are auto-disambiguated (`Name (1)` / `Name (2)`); relabel a spec's `title` for clearer entries. |
| Wanting a nested sub-figure | Not supported — composites flatten one level; keep the structure flat. |
| A random-IC system in a snippet | Always pass an explicit `ic=` (see [the front door](plotting.md)) so the panel is deterministic. |

---

## See also

- [Plotting — the front door](plotting.md) — the single-panel `to_plot_spec`
  each panel is built from.
- [Styling & themes](styling.md) — the fluent tweaks (`recolor` / `theme` / …)
  that also chain on a composite result.
- [Animation](animation.md) — the orthogonal animation modifier, including the
  lockstep composite movie.
- [Backends & export](backends.md) — how the returned spec renders and saves
  itself, and how composites tile natively in matplotlib and plotly.
