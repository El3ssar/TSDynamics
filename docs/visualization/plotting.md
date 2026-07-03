---
description: The plotting front door — traj.to_plot_spec and .plot, the auto-dispatch on component count, the PlotKind vocabulary (time series, 2-D & 3-D phase portraits, spacetime, delay embeddings, spatial fields), components= selection, and per-kind options, each with a runnable IC-pinned example.
---

<span class="ts-kicker">Visualization · Plotting</span>

# Plotting — the front door

There is exactly **one** entry point for turning a trajectory into a figure:
`to_plot_spec`. Every common view — a time series, a phase portrait, a spacetime
image, a delay embedding, a spatial-field movie — comes out of it, and its
sibling `.plot(...)` builds the spec and renders it in a single call. Learn this
one method and you can draw everything the library produces.

```python
import tsdynamics as ts

traj = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).integrate(final_time=100.0, dt=0.01)

traj.plot()                         # render immediately (auto-dispatched)
spec = traj.to_plot_spec()          # …or keep the spec, tweak it, then render
```

The two forms differ only in *when* they render. `to_plot_spec(...)` returns the
backend-agnostic [`PlotSpec`](index.md) so you can chain tweaks
(`.relabel`, `.recolor`, `.grid`, …) and choose an output; `.plot(...)` is the
same thing plus an immediate render. Everything below is written against
`to_plot_spec` — swap in `.plot` whenever you just want a picture.

!!! tip "Plot straight from a system"

    `system.to_plot_spec(...)` / `system.plot(...)` accept **integration**
    keywords too (`final_time`, `dt`, `ic`, `method`, …) — the system integrates
    first, then builds the spec, splitting the two kinds of keyword for you:

    ```python
    ts.systems.Rossler(ic=[1.0, 1.0, 1.0]).plot(final_time=200, dt=0.05,
                                                 components=["x", "y"])
    ```

## Auto-dispatch: the kind follows the components

With no `kind=`, `to_plot_spec` picks the semantic kind from **how many
components you are drawing** — the natural view for that dimensionality:

| Components drawn | Auto kind | What you get |
| --- | --- | --- |
| 1 | `TIME_SERIES` | the channel against time |
| 2 | `PHASE_PORTRAIT_2D` | a 2-D orbit on equal axes |
| 3 | `PHASE_PORTRAIT_3D` | a 3-D attractor |
| 4 or more | `SPACETIME` | a component-vs-time field image |

```python
traj = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).integrate(final_time=100, dt=0.01)

traj.to_plot_spec().kind                          # → 'phase_portrait_3d'  (all 3)
traj.to_plot_spec(components="x").kind             # → 'time_series'        (1)
traj.to_plot_spec(components=["x", "z"]).kind      # → 'phase_portrait_2d'  (2)
```

The 4+-case is deliberate: a high-dimensional flow (a Lorenz-96 lattice) reads
as a *spacetime field*, never as a misleading 3-D portrait of its first three
coordinates. A discrete-map orbit is drawn with a `SCATTER` mark (a point
sequence), not a joined line, because successive iterates are not continuous.

You override the auto-dispatch with `kind=` (any member of the closed
`PlotKind` vocabulary, plus the `"delay"` and `"field"` recipes) and you pick
*which* channels with `components=`. The rest of this page walks every kind.

---

## Time series

One or more components against time. With `kind=None` you get this whenever a
single component is selected; passing `kind="time_series"` overlays *every*
selected component as its own line (a legend appears automatically for two or
more).

```python
ros = ts.systems.Rossler(ic=[1.0, 1.0, 1.0]).integrate(final_time=200.0, dt=0.05)

# all three components, x(t) / y(t) / z(t), overlaid with a legend
ros.to_plot_spec(kind="time_series").save("rossler-ts.pdf")

# just one channel
ros.to_plot_spec(components="x").save("rossler-x.pdf")
```

<figure markdown>
![Three overlaid time series x(t), y(t), z(t) of the Rössler system, in the brand palette, with a legend; z spikes periodically while x and y oscillate smoothly](../assets/figures/viz/kind-time-series.svg){ loading=lazy }
<figcaption>The Rössler system's three components overlaid via <code>kind="time_series"</code>. The smooth <code>x</code>/<code>y</code> oscillation and the sharp periodic <code>z</code>-spikes (the reinjection kicks) are the signature of the Rössler folding — one legend, one set of axes.</figcaption>
</figure>

**Reading it.** Time runs along `x`; each layer is one state component. The
`z`-spikes are Rössler's fast reinjection events — the slow spiral in the
`(x, y)` plane is punctuated by a jump up in `z` that folds the orbit back.

Colour the line by a scalar with `color_by=` (see [below](#colour-by-a-scalar));
force a single-channel view of a 3-D trajectory with `components=`.

---

## 2-D phase portrait

Two components plotted against each other — the classic orbit view, on **equal
axes** so the geometry is undistorted. Auto-selected when you draw two
components.

```python
ros = ts.systems.Rossler(ic=[1.0, 1.0, 1.0]).integrate(final_time=200.0, dt=0.05)

ros.to_plot_spec(components=["x", "y"], color_by="time").save("rossler-xy.pdf")
```

<figure markdown>
![A 2-D phase portrait of the Rössler system in the x-y plane, the spiral orbit coloured from dark to bright along a viridis time colorbar](../assets/figures/viz/kind-phase-2d.svg){ loading=lazy }
<figcaption>The Rössler orbit projected onto the <code>(x, y)</code> plane, coloured by elapsed time (<code>color_by="time"</code>) along a viridis colorbar — the spiral winds outward, then folds. Colour-by-time turns a static orbit into a legible history: you can see which way it is going.</figcaption>
</figure>

**Reading it.** The orbit spirals outward in the plane and is periodically
folded back inward — the colour ramp (dark → bright with elapsed time) shows the
direction of travel, which a bare line cannot. Equal axes keep the spiral round
rather than squashed.

---

## 3-D phase portrait

Three components in space — the attractor itself. Auto-selected for a 3-D
trajectory. Hide the axes for a clean "object floating in space" look and frame
it with the camera.

```python
lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).integrate(final_time=100.0, dt=0.01)

(
    lor.to_plot_spec()               # all three → phase_portrait_3d
       .style(axes=False)            # no ticks / labels / grey panes
       .camera(elev=22, azim=-60)    # frame the butterfly
       .save("lorenz-3d.pdf")
)
```

<figure markdown>
![The Lorenz butterfly as a clean 3-D phase portrait, its two indigo wings, axes hidden, on a transparent background](../assets/figures/viz/kind-phase-3d.svg){ loading=lazy }
<figcaption>The Lorenz attractor drawn as a 3-D <code>LINE3D</code> portrait. <code>.style(axes=False)</code> strips the ticks, labels, and 3-D background panes; <code>.camera(elev=22, azim=-60)</code> frames the two wings. This is the "attractor floating in space" look the catalogue pages use.</figcaption>
</figure>

**Reading it.** The two lobes are the Lorenz butterfly — the orbit loops around
one wing, then crosses to the other, never settling. `.style(axes=False)` is the
right choice when the *shape* is the message and axis numbers add nothing.

!!! note "Interactive 3-D"

    A 3-D portrait becomes a **rotatable** page with the plotly or three.js
    backend — `spec.save("lorenz.html")`. See [three.js export](backends.md#threejs-export)
    for the WebGL viewer that ships on the catalogue's 3-D system pages.

---

## Spacetime image

A high-dimensional flow — a spatially-extended lattice like Lorenz-96 — imaged
as a **field**: one axis is time, the other is component index, and the state is
the colour. Auto-selected when you draw four or more components.

```python
import numpy as np

ic = np.full(20, 8.0)
ic[0] += 0.01                        # a small bump breaks the symmetry
l96 = ts.systems.Lorenz96().integrate(final_time=30.0, dt=0.05, ic=ic)

l96.to_plot_spec().save("lorenz96-spacetime.pdf")   # 20 components → SPACETIME
```

<figure markdown>
![A spacetime image of the 20-site Lorenz-96 lattice: component index on the vertical axis, time on the horizontal, state value in viridis, with diagonal travelling-wave stripes](../assets/figures/viz/kind-spacetime.svg){ loading=lazy }
<figcaption>The 20-site Lorenz-96 lattice as a <code>SPACETIME</code> image — time horizontal, site index vertical, state in viridis. The diagonal stripes are travelling waves circulating around the ring of sites; the small initial bump has spread into full spatiotemporal chaos.</figcaption>
</figure>

**Reading it.** Each column is the whole lattice state at one instant; scanning
left to right plays time forward. The diagonal bands are travelling waves
propagating around the periodic ring of sites — the field grows a colorbar and
an inferred colour range automatically. `transpose=True` swaps the two axes if
you prefer time on the vertical.

---

## Delay embedding

Reconstruct the attractor of a *single scalar observable* by plotting it against
a delayed copy of itself — `x(t)` vs `x(t − τ)`. This is the natural 2-D view of
a delay system, where the true state lives in an infinite-dimensional history
space you cannot plot directly.

```python
import numpy as np

mg = ts.systems.MackeyGlass()
traj = mg.integrate(final_time=500.0, dt=0.5,
                    history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)])

traj.to_plot_spec(kind="delay", tau=17.0).save("mackey-glass-delay.pdf")
```

<figure markdown>
![A delay embedding of the Mackey-Glass system: x(t) against x(t minus tau) tracing a folded chaotic band](../assets/figures/viz/kind-delay.svg){ loading=lazy }
<figcaption>The Mackey–Glass delay system embedded via <code>kind="delay"</code>, <code>tau=17.0</code>. At <code>dt=0.5</code> the delay of 17 time units is a 34-sample lag (the axis reads <code>x(t - 34)</code>), and the folded band is the attractor of the scalar <code>x(t)</code> reconstructed by Takens' theorem.</figcaption>
</figure>

**`tau` is in time units.** It is converted to a sample lag through the
trajectory's `dt` (here 17.0 / 0.5 = 34 samples), so the same `tau` reads the
same physical delay regardless of your output spacing. `kind="delay"` embeds one
component — with no `components=` it uses the first; select exactly one channel
otherwise. See the [embedding analysis](../analysis/embedding.md) for choosing an
optimal delay from data.

---

## Spatial field

The field of a spatially-extended system (a method-of-lines PDE) drawn on its
**spatial grid** — a 1-D profile as a line, a 2-D field as a heatmap. Use
`kind="field"`; the spatial layout comes from the system's `_field_shape`, so
you never pass a `shape` by hand.

```python
# a 2-D reaction–diffusion field → heatmap of the activator
gs = ts.systems.GrayScott().integrate(final_time=2000.0, dt=5.0)
gs.to_plot_spec(kind="field").save("gray-scott.pdf")           # last field, imshow
gs.to_plot_spec(kind="field", components="v").save("gs-v.pdf")  # pick a field block
```

A multi-block field (Gray–Scott packs an activator `u` and inhibitor `v`)
declares `field_labels`; `components=` picks the block, defaulting to the last
(the activator). A system with **no** `_field_shape` (or a 1-D one) plots as a
1-D profile — honest, never guessing a 2-D grid.

<figure markdown>
![Two panels: left, a Gray-Scott 2-D activator field as a viridis heatmap of self-replicating spots; right, a Kuramoto-Sivashinsky 1-D space-time diagram in viridis showing chaotic cellular stripes](../assets/figures/viz/spatial-field.svg){ loading=lazy }
<figcaption>Left: the Gray–Scott activator field via <code>kind="field"</code> — a viridis heatmap of the reaction–diffusion pattern. Right: the Kuramoto–Sivashinsky 1-D field (<code>N=128</code>, <code>L=60</code>) as a space-time diagram (<code>kind="spacetime"</code>, viridis) — time horizontal, site index vertical, the chaotic cellular flame front.</figcaption>
</figure>

**A field over time is a movie.** `to_plot_spec(kind="field", animate=True)`
plays the field frame by frame — a travelling wave for a 1-D field, an evolving
heatmap for a 2-D one. See [animation](#animation) below.

---

## Selecting components

`components=` chooses *what* to draw — a name, an index, or a sequence — and the
auto-dispatch then keys off how many you selected:

```python
traj = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).integrate(final_time=100, dt=0.01)

traj.to_plot_spec(components="x")             # one channel  → time series
traj.to_plot_spec(components=["x", "z"])       # two channels → 2-D portrait
traj.to_plot_spec(components=[0, 2])           # …by index, same thing
```

Names resolve against the system's declared `variables` (Lorenz's are
`('x', 'y', 'z')`). A system with no declared names uses generated `y0`, `y1`, …
labels. An unknown name or an out-of-range index raises `InvalidParameterError`
rather than plotting the wrong thing.

## Colour by a scalar

On a time series or a phase portrait, `color_by=` maps a per-point scalar onto
the line — turning a static orbit into a legible history (which way is it going?
where is it fast?). It accepts a **named field**, a **per-point array**, or a
**callable** `f(trajectory) -> array`:

```python
traj = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).integrate(final_time=100, dt=0.01)

traj.to_plot_spec(components=["x", "z"], color_by="time")     # elapsed time
traj.to_plot_spec(components=["x", "z"], color_by="speed")    # |velocity|
traj.to_plot_spec(components=["x", "z"], color_by=lambda tr: tr["y"])  # any channel
```

The named fields are computed from the drawn points:

| `color_by` | Meaning |
| --- | --- |
| `"time"` | elapsed time along the orbit |
| `"index"` | sample index (0, 1, 2, …) |
| `"speed"` | instantaneous speed \|dx/dt\| |
| `"acceleration"` (`"accel"`) | \|d²x/dt²\| |
| `"arclength"` | cumulative distance travelled |
| `"sagitta"` | per-point bow (the sampling-error field) |
| `"curvature"` | local path curvature |

A `color_by` spec grows a viridis colorbar automatically. Passing `color_by` to
a kind that does not accept it (a spacetime image, say) raises — each per-kind
option is valid for one kind only.

## Per-kind options

Options that make sense for one kind only ride on `**kind_kw` rather than
cluttering the signature. Passing one to the wrong kind raises
`InvalidParameterError`:

| Kind | Option | Meaning |
| --- | --- | --- |
| `delay` | `tau` *(required)* | delay in **time units** (→ sample lag via `dt`) |
| `time_series`, `phase_portrait_2d`/`_3d` | `color_by` | colour the line by a scalar (above) |
| `spacetime` | `transpose` | swap the time / component axes |

## Rendering and saving

A built `PlotSpec` renders itself. `.save(path)` picks the backend from the file
extension:

```python
spec = traj.to_plot_spec()

spec.save("fig.pdf")     # vector, for a manuscript  → matplotlib
spec.save("fig.png")     # raster                    → matplotlib
spec.save("fig.svg")     # scalable vector           → matplotlib
spec.save("fig.html")    # interactive, rotatable    → plotly
spec.save("fig.json")    # raw data payload          → json exporter

spec.plot()                       # render inline (a notebook shows it)
spec.render(backend="plotly")     # force a backend
```

`.plot(...)` and `.save(...)` also accept **inline tweaks** — `title=`,
`xlabel=`, `yscale=`, `xlim=` — applied to the spec before rendering, so a
quick one-off needs no chain:

```python
traj.plot(components=["x", "z"], title="Lorenz (x, z)", yscale="linear")
```

## Composing panels

`to_plot_spec` builds **one panel**. To arrange several things into one figure —
overlaid on shared axes, or tiled — use `ts.viz.plot(*things, layout=...)`,
which returns a spec that itself renders:

```python
a = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).integrate(final_time=100, dt=0.01)
b = ts.systems.Rossler(ic=[1.0, 1.0, 1.0]).integrate(final_time=200, dt=0.05)

ts.viz.plot(a, b, layout="grid").save("two-attractors.pdf")
```

<figure markdown>
![A two-by-two grid of four classic strange attractors — Lorenz, Rössler, Halvorsen, Thomas — each its own panel in a distinct brand colour](../assets/figures/viz/compose-grid.svg){ loading=lazy }
<figcaption>Four attractors tiled into a <code>COMPOSITE</code> grid via <code>ts.viz.plot(lorenz, rossler, halvorsen, thomas, layout="grid")</code> — each its own panel and brand colour. <code>layout=</code> is <code>"overlay"</code> (shared axes), <code>"stack"</code>, <code>"row"</code>, or <code>"grid"</code>.</figcaption>
</figure>

Overlay merges compatible single-panel specs onto **one** set of axes, with the
legend auto-disambiguated by source:

```python
r1 = ts.systems.Rossler(params={"c": 2.3}, ic=[1.0, 1.0, 1.0]).integrate(400, 0.05).after(100)
r2 = ts.systems.Rossler(params={"c": 5.7}, ic=[1.0, 1.0, 1.0]).integrate(400, 0.05).after(100)

ts.viz.plot(r1, r2, components=["x", "y"], layout="overlay").save("rossler-overlay.pdf")
```

<figure markdown>
![Two Rössler orbits overlaid on one x-y plane: a small teal limit cycle at c=2.3 and a wide indigo chaotic band at c=5.7, with a legend distinguishing them](../assets/figures/viz/compose-overlay.svg){ loading=lazy }
<figcaption>Two Rössler orbits merged onto one <code>(x, y)</code> plane via <code>layout="overlay"</code>: the small teal limit cycle (<code>c=2.3</code>) sits inside the wide indigo chaotic band (<code>c=5.7</code>), the legend disambiguated by source. Because <code>plot</code> takes and returns a <code>PlotSpec</code>, a composed figure feeds straight back into another <code>plot</code> call.</figcaption>
</figure>

## Animation

Any spec of any kind becomes a movie by carrying an `Animation` — an *orthogonal
modifier*, so the semantic kind is unchanged and a backend that cannot animate
draws the final frame. Turn it on with `animate=True` (or a dict / `Animation`),
then tune with the chainable `.animate` / `.trail` / `.head` / `.camera` /
`.clock` methods:

```python
lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).integrate(final_time=60.0, dt=0.01)

(
    lor.to_plot_spec(animate=True)
       .animate(n_frames=100, fps=25)
       .trail(("time", 6.0), fade=True)   # a 6-time-unit comet tail, fading
       .head(color="#E8912D")             # an amber "current state" marker
       .style(axes=False)
       .background("#0B0F14")             # a solid dark stage (a GIF has no alpha)
       .save("lorenz-reveal.gif")
)
```

<figure markdown>
![An animated GIF of the Lorenz attractor drawing itself in: a bright amber head sweeps the two wings, trailing a fading comet tail on a dark stage, axes hidden](../assets/figures/viz/animation-lorenz-reveal.gif){ loading=lazy }
<figcaption>A looping reveal-comet of the Lorenz attractor: <code>animate=True</code> + <code>.trail(("time", 6.0), fade=True)</code> gives a fading 6-time-unit tail behind an amber <code>.head</code>, axes hidden and drawn on the brand dark stage. matplotlib writes <code>.mp4</code>/<code>.gif</code>; plotly exports a real-time rotatable-while-playing <code>.html</code>. See <a href="animation.md">Animation</a> for the field movie, the spinning attractor, and more.</figcaption>
</figure>

!!! tip "Animate on a solid background"
    A GIF has no alpha channel — a transparent animation is flattened to one fill
    colour (often a jarring green). Give a movie a **solid** background:
    `.background("#0B0F14")` here. A still `.png`/`.svg` keeps real transparency.

There are two frame models. **`reveal`** (the default) keeps the full static
data and shows a comet — a head at the current sample, a tail reaching back
`trail_length` (`None` = a persistent "draws itself in" trail). **`frames`** is
the spatial-field movie — each frame is a fresh spatial snapshot (used
automatically by `kind="field", animate=True`). Export goes to matplotlib for
`.mp4`/`.gif` and to plotly for an interactive real-time `.html`.

## Pitfalls

- **Random initial conditions.** Many catalogue systems have no default IC, so a
  bare `integrate()` starts from a fresh random point each run. **Always pass an
  explicit `ic=`** (as every example here does) when you want a figure to
  reproduce.
- **`kind="delay"` needs `tau`.** It is required, and it is in *time units* — the
  trajectory must carry a `dt` (it does, from `integrate`) so the lag can be
  resolved.
- **Overlaying incompatible kinds.** You cannot overlay an image and a portrait,
  or a 2-D and a 3-D plot, on one set of axes — use `layout="stack"` / `"grid"`
  to give each its own panel.
- **Forcing too few components.** `kind="phase_portrait_3d"` on a 2-D selection
  raises — pick enough channels, or use `"time_series"`.

## See also

- [Styling & themes](styling.md) — restyle any spec: colours, widths, themes.
- [Figure conventions](conventions.md) — the rules for paper-ready output.
- [three.js export](backends.md#threejs-export) — interactive WebGL attractor viewers.
- [Analysis](../analysis/index.md) — every analysis result carries a `.plot`.
