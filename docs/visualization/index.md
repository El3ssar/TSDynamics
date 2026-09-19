---
description: The TSDynamics plotting grammar — one verb, ts.plot, that draws everything you hand it on one figure; a positional string says how to draw, everything else says what to draw, and a Plot handed back in is just another thing to draw.
---

<span class="ts-kicker">Visualization</span>

# Visualization

There is **one plotting verb**, and it has one rule:

> **`ts.plot` draws everything you hand it on one figure and gives you back a
> `Plot`: a positional *string* says **how** to draw, everything else says
> **what** to draw — and a `Plot` handed back in is just another thing to draw.**

That last clause is *closure*, and it is why there is no separate API for grids,
for movies, or for dropping down to matplotlib and coming back. They all follow
from it.

```python
import numpy as np
import tsdynamics as ts

traj = ts.systems.Lorenz().run(final_time=100.0, dt=0.01, ic=[1.0, 1.0, 1.0])

ts.plot(traj)                                     # the default view
ts.plot(traj, color="crimson", title="Lorenz")    # ...styled at the door
ts.plot(traj, "psd", xscale="log", yscale="log")  # a named transform
ts.plot(np.sin(np.linspace(0, 40, 2000)))         # bare arrays plot too
```

<figure markdown>
![The Lorenz butterfly rendered as a clean 3-D phase portrait, axes hidden, drawn on a transparent background with the two indigo wings of the attractor](../assets/figures/viz/kind-phase-3d.svg){ loading=lazy }
<figcaption>One call: <code>ts.plot(traj)</code>. Three components means a 3-D phase portrait; the axes are hidden with <code>.style(axes=False)</code> and the camera framed with <code>.camera(elev=22, azim=-60)</code> for the attractor "floating in space".</figcaption>
</figure>

## Everything `ts.plot` accepts

```python
# skip-doctest — the signature, for reference
plot(*things,
     layout="overlay", rows=None, cols=None,
     share_x=None, share_y=None, share_color=None,
     primitive=None, on=None, animate=False, fps=None, ax=None,
     **options) -> Plot
```

A positional argument is classified by **what it is**:

| You pass | It means | Example |
| -------- | -------- | ------- |
| a `str` | a **transform** — how to draw | `ts.plot(traj, "psd")` |
| `"name.primitive"` | ...and which drawing to use | `ts.plot(traj, "phase_portrait.density")` |
| `("name", {...})` | a transform with its own options | `ts.plot(vdp, ("streamlines", {"color": "w"}))` |
| `T("name", **opts)` | the same thing, typed (`ts.viz.spec.T`) | `ts.plot(vdp, T("flow_speed", log=True))` |
| a **`Plot`** | a **subject** — closure | `ts.plot(p1, p2, layout="row")` |
| a system / `Trajectory` / result | a subject | `ts.plot(lorenz)` |
| an array | a subject (coerced to an index-time trajectory) | `ts.plot(signal)` |
| a list of any of those | unwrapped to several subjects | `ts.plot([a, b])` |

**Subjects and transforms are matched by what each transform declares it needs**,
not by a blind cross product. A transform applies to every subject it admits; a
subject no named transform admits draws its own default view. That is what makes
the flagship two-dimensional-dynamics figure a single call:

```python
vdp = ts.systems.VanDerPol()
a = vdp.run(final_time=30.0, dt=0.01, ic=[0.5, 0.0])
b = vdp.run(final_time=30.0, dt=0.01, ic=[2.5, 0.0])

ts.plot(vdp, a, b, "flow_speed", "nullclines")
```

`flow_speed` and `nullclines` need the *equations*, so they consume `vdp`; the
two trajectories have no named transform that admits them, so they draw their
default view — two orbits over the field. Order does not matter: layers are drawn
by role, fields behind orbits behind overlays.

## Four keyword vocabularies, peeled in order

Keywords are routed, not guessed, and an unroutable one raises naming the
nearest match across **all** the vocabularies.

1. **Composition** — `layout`, `rows`, `cols`, `share_x`, `share_y`,
   `share_color`, `primitive`, `animate`, `fps`, `ax`. They are on the signature,
   so `help(ts.plot)` shows them.
2. **Figure** — `title`, `xlabel`/`ylabel`/`zlabel`, `xlim`/`ylim`/`zlim`,
   `xscale`/`yscale`/`zscale`, `xticks`/`yticks`/`zticks`, `clim`, `colorbar`,
   `legend`, `theme`. All seventeen work at **every** plotting door.
3. **Style** — `color`, `linewidth`, `linestyle`, `marker`, `markersize`,
   `alpha`, `cmap`, `fill`, `fillalpha`, `zorder` (plus the usual aliases: `lw`,
   `c`, `ms`, `"--"`, `"o"`).
4. **Run** — `final_time`, `steps`, `dt`, `t0`, `ic`, `transient`, `solver`,
   `rtol`, `atol`, `max_step`, `seed`, `backend`, `history`, `events`. Only
   meaningful when the subject is a *system*, which `ts.plot` runs for you.

Anything left over is a **transform option**, routed to the transforms whose
signature accepts it.

```python
ts.plot(ts.systems.Rossler(), final_time=200.0, dt=0.02,   # (4) run it like this
        color="#11857A", linewidth=0.8,                    # (3) draw it like this
        title="Rössler", theme="dark")                     # (2) label it like this
```

Because style is peeled *before* the leftovers reach `run()`, an integration typo
is still reported as an integration typo — never as "`color` is not a valid
`run()` keyword".

## What comes back: a `Plot`

`ts.plot` always returns a `Plot`, and a `Plot` is the whole story — the
description *and* the thing you render. There is nothing to unwrap and no second
type to learn when you want more control.

```python
p = ts.plot(traj, "phase_portrait", color="#4B3F9E")

p.save("lorenz.png")         # write it: .png .pdf .svg .html .json .mp4 .gif
p.relabel(title="Lorenz").gridlines().limits(x=(-20, 20))   # fluent, chainable
p[0]                         # select a panel  (p["psd"] works too)
p.style("phase_portrait", alpha=0.6)                # select layers by name
p.to_json()                  # the full figure as JSON — cache it, ship it, diff it
```

`p.show()` displays it in a notebook or a GUI session.

### The escape hatch, and the way back

When the library does not have the knob you want, take the matplotlib objects:

```python
# skip-doctest — needs the optional tsdynamics[viz] backend
p = ts.plot(traj, "phase_portrait")
p.fig                        # the matplotlib Figure — rendered once, then cached
p.ax                         # the Axes  (.axes for a grid, in panel order)
p.axes[0].set_yscale("symlog")
```

…and the way *in* is the `ax=` keyword, so a TSDynamics plot can be one panel of
a figure you are building by hand:

```python
# skip-doctest — needs the optional tsdynamics[viz] backend
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(9, 4))
ts.plot(traj, "phase_portrait", ax=axs[0])
ts.plot(traj, "psd", ax=axs[1])
```

!!! warning "Library tweaks first, matplotlib last"
    A `Plot` caches its figure, and every mutating method drops that cache. If
    you have already taken `.fig`/`.ax` and hand-edited it, a later `.style(...)`
    re-renders and your hand edits are gone — so the library warns, once, rather
    than silently losing your work. Do the library tweaks first, then take the
    figure.

## Composing: closure does the work

Because a `Plot` is a legal subject, composition needs no new API.

=== "Overlay (the default)"

    ```python
    a = ts.systems.Lorenz().run(final_time=40.0, dt=0.01, ic=[1.0, 1.0, 1.0])
    b = ts.systems.Lorenz().run(final_time=40.0, dt=0.01, ic=[1.0001, 1.0, 1.0])

    ts.plot(a, b, "phase_portrait")        # two orbits, one set of axes
    ```

    Overlay is legal by **frame identity** — the same coordinate space, the same
    dimension, the same axis names — and every orbit gets its own legend entry.
    Mixing incompatible frames raises and names the panelled layout instead.

=== "A grid of different plots"

    ```python
    ts.plot(
        ts.plot(traj, "phase_portrait", title="orbit"),
        ts.plot(traj, "time_series", components="x", title="x(t)"),
        ts.plot(traj, "psd", xscale="log", yscale="log", title="spectrum"),
        layout="grid", rows=1, cols=3, theme="publication",
    )
    ```

    `ts.viz.grid(*plots, rows=, cols=)` is the same thing under a shorter name.
    Panels hold the *same* objects you passed in, so `p.panels[0].style(...)`
    after the fact still lands in the render.

=== "Straight from arrays"

    ```python
    r = np.logspace(-1, 1, 40)
    c = r**2.06

    ts.viz.draw({"x": r, "y": c}, "line", labels=("log r", "log C(r)"))
    ```

    No transform, no registration, no library type — a mapping of channels and
    the name of a primitive. And because `draw` returns a `Plot`, it composes
    with everything else.

## A map of this section

<div class="grid cards" markdown>

- **[Transforms & primitives](plotting.md)**

    ---

    *What* can be drawn and *how*: the 38 transforms, the 16 primitives, the
    declared compatibility matrix, `ts.viz.geometry` for the arrays alone, and
    the one-decorator recipe for adding your own.

- **[Composition](composition.md)**

    ---

    Overlay by frame identity, `layout="stack"/"row"/"grid"`, `share_x` /
    `share_y` / `share_color`, per-panel selection and styling, and the closure
    rule that makes a figure of figures just another figure.

- **[Styling & themes](styling.md)**

    ---

    The canonical style vocabulary, the fluent chainable tweaks, the four
    built-in themes, registering your own house style, and the per-backend
    *honoring* contract that makes an unsupported key warn rather than vanish.

- **[Animation](animation.md)**

    ---

    Animation as an orthogonal modifier — `animate=True` plus `.trail` /
    `.head` / `.camera` / `.clock` — the reveal comet, the spatial-field movie,
    and export to `.mp4` / `.gif` or a live interactive `.html`.

- **[Backends & export](backends.md)**

    ---

    The four renderers (matplotlib · plotly · json · three.js), how dispatch and
    fallback work, `p.save(path)` by extension, and the self-contained WebGL
    export that ships on the 3-D catalogue pages.

- **[The gallery](gallery.md)**

    ---

    Every registered transform, drawn with every primitive it declares, with the
    code that produced each picture printed beside it — generated from the
    registry at docs-build time, so the code and the picture cannot drift.

- **[Figure conventions](conventions.md)**

    ---

    The rules the catalogue figures follow — viridis for sequential fields,
    twilight for cyclic ones, one figure per concept, colorbars and legends — so
    your figures come out paper-ready.

</div>

## Under the hood: one description, four renderers

A `Plot` is a **backend-agnostic, JSON-serializable description** of a figure. It
holds NumPy arrays and typed presentation metadata and imports no plotting
library; a *renderer* consumes it.

```text
   your system / trajectory / array / analysis result
                        │
                   transform            ← turns a subject into Geometry
                        │
                   primitive            ← chooses how that geometry is drawn
                        │
                        ▼
   ┌────────────────────────────────────────────────┐
   │                      Plot                       │
   │   kind · layers · axes · colorbar · legend ·    │
   │   theme · annotations · animation · panels      │
   └────────────────────────────────────────────────┘
        │            │            │            │
   matplotlib      plotly      three.js       json
   (raster/      (interactive   (WebGL       (data
    vector)         HTML)       viewer)      export)
```

Three properties fall out of the split, and all three matter for research work:

1. **Backend independence.** A tweak like `.rescale(x="log")` or
   `.recolor("#11857A")` touches the description, not a renderer, so it renders
   identically everywhere. Compose once; export four ways.
2. **Serialisation.** `p.to_json()` round-trips the whole figure — arrays, axes,
   colorbar, annotations — to plain JSON, and `ts.viz.load(...)` reads a path
   *or* a JSON document back. A computed figure can be cached, version
   controlled, or replotted months later **without rerunning the analysis**.
3. **Zero import cost.** A plain `import tsdynamics` pulls in **no** plotting
   library. `ts.viz` is bound lazily and each renderer's import is deferred to
   its first render, so the core stays light on a headless cluster.

```python
p = ts.plot(traj, "phase_portrait")
blob = p.to_json()                 # plain JSON — cache it, ship it, diff it
same = ts.viz.load(blob)           # rebuilt, no recomputation
```

`PlotKind` — the semantic vocabulary a renderer dispatches on (`TIME_SERIES`,
`PHASE_PORTRAIT_2D`/`_3D`, `SPACETIME`, `SPATIAL_FIELD`, `RECURRENCE_PLOT`, …) —
is a **closed, reviewed contract**. Adding a transform never needs a new kind;
adding a primitive never needs a new kind. That is the invariant that lets the
drawing vocabulary grow without touching a single renderer.

## Installing a backend

The description ships with the core library; a backend is an optional extra:

| Extra | Backend | Draws |
| --- | --- | --- |
| `tsdynamics[viz]` | matplotlib | everything — raster (`.png`/`.jpg`) and vector (`.pdf`/`.svg`), plus `.mp4`/`.gif` animation |
| `tsdynamics[interactive]` | plotly | interactive 2-D/3-D HTML, real-time animated `.html` |
| *(bundled)* | json, three.js | data export — no plotting dependency |

With no backend installed, `p.to_json()` still works — you can render the payload
yourself. Once matplotlib is present it is the deterministic default, and
`p.save(path)` picks the backend from the file extension.

## See also

- [Transforms & primitives](plotting.md) — start here to draw something other than the default view.
- [The gallery](gallery.md) — every transform, every primitive, with its code.
- [Tutorials](../tutorials/index.md) — end-to-end journeys that dogfood this module.
