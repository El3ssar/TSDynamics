---
description: The four rendering backends — matplotlib (the universal default), plotly (interactive HTML), three.js (a BufferGeometry web payload with a live orbitable viewer) and json (lossless serialization) — plus backend selection, capability negotiation, the VisualizationDegraded warning, and saving / rendering / exporting a PlotSpec by file extension.
---

<span class="ts-kicker">Visualization</span>

# Backends & export

A [`PlotSpec`](../reference/top-level.md) is a *semantic* description of a figure
— a kind, some layers of array data, typed axes, a theme. It holds no drawing
state and imports no plotting library. **Rendering** is the separate step that
turns that description into a concrete artifact: a matplotlib figure, an
interactive plotly page, a three.js web payload, or a serialized JSON document.

The same spec drives all four. Because the spec is the single source of truth,
every fluent tweak — `rescale(x="log")`, `theme("dark")`, `.recolor(...)` —
renders identically on whichever backend draws it. You pick the backend by name,
or let the file extension pick it for you.

```python
import tsdynamics as ts

spec = ts.systems.Lorenz().to_plot_spec(
    components=["x", "y", "z"], final_time=50.0, dt=0.01, ic=[1.0, 1.0, 1.0]
)

spec.show()                    # draw with the default backend (matplotlib)
spec.save("lorenz.png")        # → matplotlib (raster / vector image)
spec.save("lorenz.html")       # → plotly (a self-contained interactive page)
spec.save("lorenz.json")       # → json (a lossless, re-loadable payload)
spec.render("threejs")         # → a three.js BufferGeometry-ready dict
```

Everything on a spec is inherited by results, trajectories and systems: a
`Trajectory` or an analysis result carries the same `.plot()` / `.save()` /
`.render()` methods (they build a spec, then render it), and so does the
`ts.viz.plot(...)` composition result.

---

## The four backends

| Backend | Draws | 3-D | Interactive | Web export | Data export | Needs |
| --- | --- | :---: | :---: | :---: | :---: | --- |
| **matplotlib** | every kind | ✓ | | | | `matplotlib` |
| **plotly** | 2-D + 3-D + composite | ✓ | ✓ | ✓ | | `plotly` |
| **three.js** | line / points / surface geometry | ✓ | (in the browser) | ✓ | ✓ | — |
| **json** | serializes any kind | ✓ | | | ✓ | — |

The two *data-export* backends (`json`, `threejs`) return a serializable
**payload** rather than a live figure — they draw nothing themselves, they hand
a browser (or a cache) the data to draw. Both are pure standard library over the
spec IR, so they need **no** third-party dependency and always register.

### matplotlib — the universal reference renderer

matplotlib is the **default** and the fallback. It declares `kinds=None` —
meaning it can draw *every* `PlotKind`, in 2-D and 3-D — so it is the backend a
no-`backend=` render resolves to, and the one dispatch falls back to when a
partial backend declines a spec. If you never think about backends, this is the
one you get. It renders to any raster / vector format matplotlib supports
(`.png` / `.pdf` / `.svg` / `.jpg`) and writes `.mp4` / `.gif`
[animations](animation.md) via ffmpeg / pillow.

```python
fig = spec.render("matplotlib")     # a matplotlib.figure.Figure
spec.save("lorenz.pdf")             # a vector figure for a paper
```

### plotly — interactive, web-native

plotly produces an **interactive** figure — pan, zoom, hover, and orbit a 3-D
attractor with the mouse. It draws the 2-D and 3-D kinds and tiles composites
natively (a `make_subplots` grid, each panel on its own `xy` / `scene` cell), and
it serializes to a **self-contained HTML page** with `.save("*.html")` — the
zero-extra-dependency way to ship a live figure into docs, a dashboard, or a
paper-as-web-page.

```python
fig = spec.render("plotly")         # a plotly.graph_objs.Figure
spec.save("lorenz.html")            # one self-contained interactive page
```

plotly declines only the deferred *animation kinds* and — for animation — an
animated **composite** and the animated **spatial-field movie** (both are
matplotlib's job); those fall back automatically. Its animated-HTML export is a
real-time comet you can rotate while it plays — see [Animation](animation.md).

### json — lossless serialization

The `json` backend serializes the **whole** spec to the versioned JSON envelope
(`tsdynamics.viz.export.to_json`) — every layer, axis, annotation, color range,
theme, and `meta` field, with NumPy arrays as nested lists. It round-trips
exactly, so a spec computed on one machine can be cached, shipped to a web
frontend, or replayed **without re-running the analysis** and without a plotting
library installed.

```python
spec.render("json", path="lorenz-spec.json")   # write the payload to a file
text = spec.render("json", raw=True)            # …or get the JSON string back
```

```pycon
>>> from tsdynamics.viz.export import to_json, from_json
>>> import numpy as np
>>> spec2 = from_json(to_json(spec))
>>> spec2.kind == spec.kind and len(spec2.layers) == len(spec.layers)
True
>>> np.allclose(spec2.layers[0].data["x"], spec.layers[0].data["x"])
True
```

By default the renderer returns a `RenderResult` carrying the JSON string as its
`.payload` and `mimetype="application/json"`; pass `raw=True` for the bare string,
`indent=N` to pretty-print, or `path=` to write it straight to a file (it returns
the `Path`). The envelope is versioned (`schema_version`) so the schema can evolve
without breaking old payloads — `from_json` reads current, legacy-versioned, and
bare-`to_dict()` documents alike.

### three.js — a BufferGeometry web payload

The `threejs` backend lowers a spec to a **three.js BufferGeometry-ready** JSON
payload — a plain dict of flat float / int lists a browser front-end turns into
an orbitable WebGL scene with no Python kernel in the loop — and, via
`save(".html", backend="threejs")`, wraps it in one **self-contained page** that
opens from `file://`. It is the natural format for putting a 3-D attractor
*inside someone else's web page*. Covered in full [below](#threejs-export).

---

## How a backend is chosen

`PlotSpec.render(backend=...)` delegates to the dispatcher, which:

1. **Registers** the installed in-tree backends lazily, on the first render —
   never at import, so `import tsdynamics` stays plot-free. A backend whose
   library is absent simply does not register.
2. **Selects** a backend — the one you named, or a default.
3. **Negotiates capabilities** — if the chosen backend cannot draw this spec's
   kind (or its 3-D-ness), it **falls back** to a capable one, warning you.
4. **Reports honoring gaps** — one consolidated warning names any style key,
   animation knob, or theme field the chosen backend will silently ignore.

### The default is deterministic

With **no** `backend=`, matplotlib is preferred whenever it is installed — it is
the universal reference renderer, selected *by name*, so the choice never depends
on which backends happen to be registered or in what order. If matplotlib is
absent, dispatch picks the first registered **drawing** backend that can handle
the spec (the data-export backends are skipped — a caller with no `backend=`
wants a figure, not a payload).

### Named selection, with aliases

Naming a backend reaches it directly. The friendly aliases are accepted
(`"mpl"` → `matplotlib`), and an unknown name raises a clear `KeyError`:

```pycon
>>> spec.render(backend="mpl")          # the alias resolves
>>> spec.render(backend="gnuplot")
Traceback (most recent call last):
    ...
KeyError: "No renderer registered as 'gnuplot'."
```

---

## Capability negotiation & `VisualizationDegraded`

Not every backend can draw every kind, and not every backend honors every style
knob. Rather than fail or silently misrender, the dispatcher **degrades loudly**
with a `VisualizationDegraded` warning. There are two triggers.

**1. Backend fallback.** You named a backend that declines the spec, and dispatch
routed to a capable one (matplotlib) instead:

```pycon
>>> px = ts.systems.Lorenz().to_plot_spec(components="x", final_time=20, dt=0.02, ic=[1.0, 1.0, 1.0])
>>> py = ts.systems.Lorenz().to_plot_spec(components="y", final_time=20, dt=0.02, ic=[1.0, 1.0, 1.0])
>>> comp = ts.viz.plot(px, py, layout="stack", animate=True)   # an animated composite
>>> comp.render("plotly")
VisualizationDegraded: backend 'plotly' cannot draw a 'composite' spec;
falling back to 'matplotlib'.
```

**2. Knob degradation.** The chosen backend draws the spec, but it does not honor
one or more of the style keys / animation directives / theme fields the spec
carries. The dispatcher collects **all** of them and emits **one** consolidated
warning before drawing:

```pycon
>>> spec = ts.systems.Lorenz().to_plot_spec(components=["x", "y", "z"], final_time=20, dt=0.02, ic=[1.0, 1.0, 1.0])
>>> spec.style(linestyle="dashed")     # three.js has no line dashing
>>> spec.render("threejs")
VisualizationDegraded: threejs: ignoring linestyle
```

Each backend's honored vocabulary is an **enforced contract** — see
[Styling & themes](styling.md) for the per-key `honored_by` table and the
`caps.style_honoring_gaps` helper that drives these warnings. `json` is exempt
(it serializes every field faithfully, so it drops nothing).

`VisualizationDegraded` is a `UserWarning`, so the call still returns a figure — a
*hard* failure (no capable backend registered at all) raises
`VisualizationNotInstalled` instead. Under `filterwarnings=["error"]` (a strict
test suite) a degradation becomes an error; scope an
`ignore::tsdynamics.viz.render.caps.VisualizationDegraded` — or a `pytest.warns`
around the one call — narrowly, so a genuinely unintended degradation still
surfaces everywhere else.

---

## Saving by extension

`PlotSpec.save(path)` picks the backend from the file extension when you do not
name one — the fast path for "just write me the figure".

| Extension | Backend | Produces |
| --- | --- | --- |
| `.png` `.pdf` `.svg` `.jpg` | matplotlib | a raster / vector image |
| `.html` | plotly | a self-contained interactive page |
| `.json` | json | the lossless, re-loadable payload |
| `.mp4` `.gif` (animated spec) | matplotlib | a movie (ffmpeg / pillow) |
| `.html` (animated spec) | plotly | a real-time, rotatable-while-playing page |
| `.html` with `backend="threejs"` | three.js | a self-contained WebGL viewer page |
| `.json` with `backend="threejs"` | three.js | the BufferGeometry payload (**not** the spec IR) |

!!! warning "`.json` means two different documents"

    `save("x.json")` writes the **PlotSpec IR envelope** — reloadable with
    `tsdynamics.viz.export.from_json`. `save("x.json", backend="threejs")` writes
    the **BufferGeometry payload** — geometry for a browser, *not* a spec. They
    share an extension and are not interchangeable; `backend=` is the
    disambiguator.

```python
spec.save("fig.png")                 # matplotlib image
spec.save("fig.html")                # interactive plotly page
spec.save("fig.png", size=(1600, 1200), dpi=200)   # pixel size + resolution
spec.save("fig.svg", backend="matplotlib")         # force a backend explicitly
```

`.save` returns the path it wrote. A **still** save of an animated spec renders
its final, fully-revealed frame; a **movie** save (`.mp4` / `.gif`) drives the
matplotlib `FuncAnimation`. `size` is in **pixels** (converted to a matplotlib
figure size via `dpi`), and `fps` / `dpi` override the spec's animation settings
for that one write. See [Animation](animation.md) for the movie details.

---

## three.js export {#threejs-export}

Besides the matplotlib and plotly renderers, a spec can be exported to a
**three.js BufferGeometry-ready payload** — a plain JSON object a browser
front-end turns into an orbitable WebGL scene with no Python kernel in the loop.
It is the natural format for 3-D attractors on the web (docs, dashboards,
papers-as-web-pages).

### One file you can email: `save(".html", backend="threejs")`

The headline surface. It writes **one self-contained HTML document** — the
geometry payload and the reference loader are both inlined — so it opens by
double-clicking, from a `file://` path, from inside a zip, or pasted into a CMS.
No web server, no sibling `.json`, no `fetch`.

```python
import tsdynamics as ts

spec = ts.systems.Lorenz().to_plot_spec(
    final_time=80.0, dt=0.0025, ic=[1.0, 1.0, 1.0], animate=True,
)
spec.save("lorenz.html", backend="threejs")     # 32k vertices, ~1.3 MB
```

The one external reference is the **pinned three.js build**, declared through an
ES module import map (three.js is ~600 KB of library that is not ours to vendor).
When that CDN is unreachable — or WebGL is off, or JavaScript is disabled — the
page falls back to an inlined **poster PNG** rendered through the matplotlib
backend, so the reader always sees the attractor.

Knobs (all on `render` / `save`):

| Argument | Default | What it does |
| --- | --- | --- |
| `max_points` | `40000` | Vertex ceiling per geometry; `None` exports everything |
| `decimals` | `4` | Float rounding in the position / `c` buffers |
| `assets` | `"inline"` | `"inline"` embeds the loader; `"link"` imports it from `loader_url` |
| `poster` | `True` | Inline a matplotlib PNG as the no-WebGL / no-JS fallback |
| `axes` | `True` | Draw the labelled scale frame (box, ticks, axis names) |
| `background` | theme | Scene + page background colour |

For a site that ships *many* viewers, write the loader once and link it:

```python
from tsdynamics.viz.render.threejs import write_loader_asset

write_loader_asset("site/_static")                       # one shared copy
spec.save("a.html", backend="threejs", assets="link",
          loader_url="/_static/tsdyn-threejs-loader.js")
```

### Putting it on your own page {#embedding}

The whole path, end to end. There are exactly three steps, and the middle one is
copy-paste.

**1. Write the file.** One line:

```python
import tsdynamics as ts

ts.systems.Lorenz().to_plot_spec(
    final_time=80.0, dt=0.0025, ic=[1.0, 1.0, 1.0],
).save("lorenz.html", backend="threejs")
```

That is a complete, standalone document. Open it by double-clicking to check it
before you ship it — it needs no server.

**2. Drop it next to your HTML and reference it in an `<iframe>`.** An iframe is
the right container here, not a `<div>`: the page brings its own import map,
its own module script and its own stylesheet, and an iframe keeps all three out
of your document's scope. It also means the viewer cannot break your page's
layout or its CSP.

```html
<iframe src="lorenz.html"
        title="Lorenz attractor"
        loading="lazy"
        style="width:100%;aspect-ratio:16/10;border:0;border-radius:8px"></iframe>
```

`loading="lazy"` is worth keeping: the viewer idles when scrolled off-screen, but
lazy loading means a page with a dozen attractors does not parse a dozen payloads
up front either.

**3. Nothing.** There is no build step, no bundler, and no `npm install`. The one
thing the page fetches from the network is the pinned three.js build from a CDN;
if that is blocked, the reader sees the inlined poster PNG instead of a blank
frame (see below).

If you would rather mount the viewer into your own `<div>` — a React component, a
dashboard panel — export the payload instead of the page and call the loader
yourself; that is [the reference loader](#the-reference-loader) section.

#### The size budget

The number that decides whether this is embeddable is the size of the HTML file.
Two things drive it: the vertex buffer, and the inlined poster PNG. The loader
(~58 KB) is noise, and three.js is not embedded at all. Measured:

| What you export | `poster=False` | `poster=True` |
| --- | --- | --- |
| 3-D flow, `c` channel, default cap (40 000) | 1.34 MB | 1.89 MB |
| 3-D flow, `c` channel, `max_points=25_000` | 0.86 MB | 1.41 MB |
| 3-D flow, `c` channel, `max_points=10_000` | 0.38 MB | 0.93 MB |
| A map's iterate cloud, 200 000 iterates at the default cap | 0.80 MB | 0.85 MB |
| 1e6-sample flow, default cap | 1.00 MB | — |
| 1e6-sample flow, **uncapped** (`max_points=None`) | 23.3 MB | don't |

Rules of thumb: **budget ~1 MB per viewer**, treat 5 MB as the point where a phone
on mobile data gives up, and reach for `max_points` before anything else — the
capped 1e6-sample flow is 23x lighter than the uncapped one and, because the
thinning is by arc length, visually indistinguishable.

The poster is not free: it costs ~50 KB on a sparse scatter but ~550 KB on a dense
3-D flow, where it can be a third of the file. `poster=False` drops it — a real
trade, not a micro-optimisation, so make it deliberately (see the next section).

#### The fallback, for readers who cannot run it

Every emitted page carries a matplotlib-rendered **poster PNG**, inlined as a
`data:` URI, and reveals it when anything in the WebGL chain fails — an
unreachable CDN, WebGL disabled, an old browser. A reader with JavaScript off
entirely gets the same image through `<noscript>`. Both paths are the same
picture, so the page degrades to a static figure rather than to a blank rectangle.

This is why every `import` in the page is *dynamic*: a static `import` of an
unreachable module aborts the whole script before any `try`/`catch` can run, and
the poster would never appear.

#### A worked example

```python
import os
import warnings

import tsdynamics as ts

# A long, finely sampled run — more samples than we will ship, deliberately:
# the exporter thins by arc length, so the *shape* is set by the integration and
# the *weight* by the cap. Thinning is not silent: it raises
# VisualizationDegraded naming the layer and the counts. That is the intent
# here, so this snippet accepts it rather than letting it escape.
traj = ts.systems.Rossler().run(final_time=400.0, dt=0.002, ic=[1.0, 1.0, 1.0])
print(traj.y.shape)                       # (200001, 3)

spec = traj.to_plot_spec(color_by="time")
spec.relabel(title="Rössler attractor")

# Use the exporter's own `background=` rather than `.theme("dark")`: threejs
# honors the background colour but not a theme's fonts/foreground/grid, so a
# full theme would warn about the fields it has to drop.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")     # accept the documented thinning notice
    path = spec.save(                   # returns the path it wrote, as a str
        "rossler.html",
        backend="threejs",
        max_points=25_000,              # ~1.2 MB with the poster, ~0.7 MB without
        background="#111318",
    )
print(os.path.getsize(path))
```

Then, in your page:

```html
<figure>
  <iframe src="rossler.html" title="Rössler attractor" loading="lazy"
          style="width:100%;aspect-ratio:16/10;border:0"></iframe>
  <figcaption>Rössler attractor, a = 0.2, b = 0.2, c = 5.7. Drag to orbit.</figcaption>
</figure>
```

### Exporting the raw payload

The `threejs` renderer ships in-tree and needs **no extra dependency** (it is
pure Python over the spec IR):

```python
# a modestly sampled spec, so nothing is thinned by the vertex cap
spec = ts.systems.Lorenz().to_plot_spec(final_time=80.0, dt=0.0025, ic=[1.0, 1.0, 1.0])

payload = spec.render("threejs")             # a RenderResult carrying the dict
spec.render("threejs", path="lorenz.json")   # …or write the payload to a file
page = spec.render("threejs", html=True)     # …or get the viewer page as a str
```

```pycon
>>> result = spec.render("threejs")
>>> tj = result.payload
>>> list(tj.keys())
['schema_version', 'kind', 'title', 'geometries', 'metadata']
>>> tj["geometries"][0]["type"]        # a 3-D line → a "line" geometry
'line'
>>> "c" in tj["geometries"][0]         # color_by="time" adds a per-vertex "c" channel
True
```

A 2-D phase portrait exports the same way (its `LINE` / `SCATTER` layers are
lifted to `z = 0`).

### Payload size is a budget, and the budget is enforced

A browser is not a plot device with an unlimited budget. The exporter therefore
caps each geometry at `max_points` (40 000 by default). Capping emits a
`VisualizationDegraded` warning naming both counts, and records them in
`metadata.resample`; it is never silent.

Measured on a 1 000 000-sample Lorenz (`final_time=2000, dt=0.002`):

| Export | Payload |
| --- | --- |
| default (`max_points=40_000`) | **0.94 MB** (1.50 MB as a saved page, poster included) |
| `max_points=None` | 23.2 MB |
| `max_points=None, decimals=None` | 56.6 MB |
| *(pre-v6 exporter, for reference)* | *~122 MB* |

The pre-v6 number is historical, not a current option: it predates the cap, the
dropped line indices, the scalar `c` channel and position rounding. Today even the
uncapped path is ~5x lighter than it was — but 23 MB is still a payload no page
should carry, which is why the cap is on by default.

**Publish to this budget:**

| Payload | Verdict |
| --- | --- |
| ≤ 500 KB | comfortable — embed as many as you like |
| ≤ 2 MB | fine for a hero / landing page |
| > 5 MB | you passed `max_points=None`; be sure you meant it |

**The cap is applied by arc length, never by a stride** — and that is not a
stylistic preference. The scale-free readability criterion is the worst-case
*sagitta*: the bow of the curve off its local chord, over the bounding-box
diagonal, which should stay under ~0.01 for a curve to read as an arc rather than
a polygon. Measured at a 40 000-vertex budget on `final_time=90, dt=0.001`
(transient dropped):

| Attractor | stride `y[::k]` | arc length |
| --- | --- | --- |
| HyperQi | **0.245** | 0.0069 |
| DequanLi | 0.0087 | 0.0062 |
| QiChen | 0.0057 | 0.0015 |
| ZhouChen | 0.0051 | 0.0001 |
| Lorenz | 0.0002 | 0.0001 |

A stride spends its budget where the trajectory is *slow* and starves the fast,
tightly-curved turns — exactly the features that carry the geometry. HyperQi at
a stride is 35x off target and visibly chorded; arc-length resampled at the same
byte count it is smooth. (A **point cloud** — a map's iterate set — has no chord
to bow off, so it is thinned instead by a deterministic seeded uniform draw,
which preserves the invariant density and cannot alias a resonant orbit the way
a stride can.)

#### When 40 000 vertices is not enough

That table is one operating point, and the sagitta is **not** a constant of the
attractor — it is a function of how much curve you asked to fit in the box. The
governing quantity is the resampled chord length in bounding-box units,

$$
h = \frac{L}{D \cdot n}
$$

for total arc length $L$, bounding-box diagonal $D$ and vertex budget $n$. Measured
across the catalogue, the worst-case sagitta tracks $h$ in one of two regimes:

- **smooth attractors** are *resolution-limited*: sagitta $\sim h^2$. Lorenz at
  $L/D = 394$ gives $h = 0.0099$ and sagitta $0.0016$ at the default cap — an order
  of magnitude inside target, so doubling the trajectory length costs nothing
  visible.
- **cusp-like hyperchaotic attractors** are *curvature-limited*: sagitta $\approx h$,
  because the local radius of curvature at the sharpest turns is comparable to the
  chord itself. No resample can beat that; only more vertices can.

So a long trace of a fast attractor **can** exceed the target at the default cap.
HyperQi over `final_time=400` reaches $L/D = 1661$, giving $h = 0.042$ and a measured
sagitta of **0.038** — about 5x over. The fix is to spend vertices, not to change
the resampling:

```python
# skip-doctest — `traj` is a long HyperQi run (final_time=400); the sagitta and
# payload figures quoted around this block come from the benchmark below, so the
# snippet documents that measured configuration rather than re-deriving it.
# HyperQi is 4-D, so pick the three components you want to see
spec = traj.to_plot_spec(components=[0, 1, 2])
spec.save("hyperqi.html", backend="threejs", max_points=200_000)
```

That lands sagitta at **0.0080** — on target — for a **5.35 MB** payload. Note what
the budget table above says about 5 MB: this is a deliberate, measured choice for a
hero page, not a default.

Rule of thumb: to hold sagitta under `0.008` on a curvature-limited attractor, take
`max_points >= L / (D * 0.008)` — for HyperQi that is 207 632, which is where the
200 000 above comes from. A smooth attractor needs far fewer, since its error falls
as $h^2$. The export records `metadata.resample`, so you can check what you actually
got rather than guessing.

Three further weight decisions, each measured against the old payload:

- a `"line"` carries **no index buffer** (its vertex order *is* its draw order) —
  12.9% of the bytes, for the same picture at half the GPU index work;
- a per-vertex scalar `"c"` channel replaces pre-expanded RGB — 38.6% of the
  bytes; the loader owns the colour ramp, which is a renderer's job anyway;
- positions are rounded to `decimals=4`, well inside `Float32` precision at
  attractor scales.

### plotly or three.js?

Both ship, and they are different products — neither is a fallback for the other.

| | plotly | three.js |
| --- | --- | --- |
| Best for | one interactive figure, axes and all | embedding an attractor *in a page* |
| Axes / ticks / legend | yes | no (a bare WebGL scene) |
| 2-D plots | yes, first-class | lifted to `z = 0`; not the point |
| Big 3-D curves | slows down; gl3d cannot stream while you orbit | built for it |
| Output | one `.html` (0.05 MB via `render(path=)`) | one `.html` (~1.5 MB inlined) |
| Customising the look | plotly's API | edit the reference loader — it is yours |

Reach for plotly when you want *a plot*. Reach for three.js when you want the
attractor itself living in someone else's web page.

### The payload schema

```json
{
  "schema_version": 3,
  "kind": "phase_portrait_3d",
  "title": "...",
  "geometries": [
    {
      "type": "line | points | surface",
      "label": "...",
      "positions": [x0, y0, z0, x1, y1, z1, ...],   // FLAT, Float32-ready
      "indices":   [],                              // "surface" triangles only; [] for line/points
      "material": {                                 // the three.js-honored style keys
        "color": "#1f77b4", "linewidth": null,
        "markersize": null, "alpha": null, "zorder": null
      },
      "c":         [c0, c1, ...],                   // OPTIONAL per-vertex SCALAR field
      "n_vertices": 40000,
      "n_vertices_original": 100001                 // before the cap
    }
  ],
  "metadata": {
    "schema_version": 3,
    "labels": {"x": "x", "y": "y", "z": "z"},
    "bounds": {"x": [min, max], "y": [min, max], "z": [min, max]},
    "camera": {"position": [x, y, z], "target": [x, y, z], "up": [x, y, z]},
    "resample": {"max_points": 40000, "original_vertices": 100001,
                 "vertices": 40000, "capped": true},
    "theme":  {"background": "...", "palette": ["#...", "..."]},
    "animation": {                              // ONLY when the spec is animated
      "fps": 30.0, "duration": null, "n_frames": null,
      "loop": true, "pingpong": false,
      "trail_length_samples": 200,              // comet tail length; null = persistent
      "head": true, "head_size": 6.0, "head_color": null,
      "n_samples": 40000                        // vertices on the longest animated line
    }
  }
}
```

A web frontend reads each geometry's flat `positions` into a `Float32Array` and
maps the optional scalar `c` through a colour ramp of its choice — exactly what a
`THREE.BufferGeometry` wants. `metadata.camera` seeds the initial view, and
`metadata.theme` gives the scene background + the palette for auto-coloured layers.

!!! note "Schema 3 (v6)"

    Three redundant blocks were removed and one added. Gone: the `LineSegments`
    `indices` for lines, the pre-expanded per-vertex `colors` RGB (now the scalar
    `c`), and `metadata.units` (which carried axis *tickformats*, not units, and
    which no consumer read). Added: `metadata.resample` and the per-geometry
    vertex counts. The reference loader reads **both** generations, so an older
    payload still renders.

    `n_samples` and `trail_length_samples` are reported in **capped** vertices —
    a reveal sized against the uncapped count would index past the buffer.

A few honest exclusions the payload makes deliberately:

- **`material`** carries only the style keys three.js genuinely honors —
  `color` / `linewidth` / `markersize` / `alpha` / `zorder` (→ `renderOrder`).
  `linestyle`, marker *shape*, and `cmap` are **not** emitted (the loader has no
  way to honor them — it owns a fixed built-in colormap ramp for `c`), so they
  are dropped rather than serialized as dead fields. `caps` warns about any of them.
- **`c`** appears only for a genuine per-vertex gradient (`color_by="time"`, a
  speed field, …). A *solid* colour rides once on `material.color` — repeating it
  per vertex would be waste.
- **`metadata.theme`** carries only `background` and `palette` — the other theme
  fields (foreground, font, grid) have no analogue in a bare three.js scene.

`metadata.animation` is present **only** for an animated spec (e.g.
`spec.to_plot_spec(..., animate=True)`); a static export omits it and the geometry
buffers are byte-for-byte the non-animated payload. When it is present, the loader
plays a **reveal comet** — a faint full-curve backdrop with a bright windowed
trail and a head marker sweeping the line — by advancing
a fixed-length windowed trail per frame (no full buffer re-upload). Because
`OrbitControls` is independent of that draw-range update, you can **orbit the
attractor with the mouse while it plays**.

The reveal comet sweeps a **line** index buffer (`LINE` / `LINE3D`), so the
animation block is emitted only when the spec has a line layer. An animated
`points`-only (`SCATTER` / `MARKERS`) or `surface`-only spec has nothing to
reveal: the exporter drops the animation to a valid static payload (which the
loader still renders, auto-rotating) and emits a `VisualizationDegraded` warning
— the animation is **never silently dropped**.

### Composite payloads

A composite spec has no top-level geometry — its content is in the panels. The
threejs exporter detects a composite, lowers each panel recursively, and emits a
`"panels"` list (each with its own `geometries` + `metadata`, a `grid` cell, and
a layout `offset` for dropping every panel into one shared scene) plus a
top-level `layout` block. `geometries` is then always empty at the top level.

The reference loader mounts each panel in its own group at the panel's `offset`,
with its own axes frame built from its own local bounds, so a composite `.html`
draws every panel in one orbitable scene.

!!! note "A composite renders in *one* 3-D scene, not in tiled subplots"

    matplotlib and plotly tile a composite into a real subplot grid; three.js has
    no subplots, so the panels are laid out side by side **in the scene** and
    viewed through one camera. For 2-D panels that reads much like a subplot grid.
    For several 3-D panels it reads as several boxes at different depths, which is
    honest but rarely what you want — save each panel separately (a panel *is* a
    `PlotSpec`), or use `backend="plotly"` for a genuine interactive multi-panel
    page.

    Earlier versions **refused** a composite `.html` outright, because the loader
    read only the top-level `geometries` — `[]` for a composite — and the page came
    up blank: a multi-megabyte file that existed, opened in a browser, and showed
    nothing. That is fixed at the loader, so the refusal is gone.

### The reference loader

`tsdyn-threejs-loader.js` is a small reference ES module that does the whole
conversion — `BufferGeometry` for `line` (a contiguous `Line`), `points`
(`Points`) and `surface` (a normals-computed `Mesh`), a colour ramp over the
scalar `c`, a camera from `metadata.camera`, `OrbitControls`, and (when
`metadata.animation` is set) a reveal comet with a play/pause overlay.

**It ships inside the wheel** — it is library code, not a docs file, and
`save(".html", backend="threejs")` inlines it. Reach it from Python:

```python
from tsdynamics.viz.render.threejs import loader_path, loader_source, write_loader_asset

loader_path()                       # where it lives in the installed package
loader_source()                     # its JavaScript, as text
write_loader_asset("site/_static")  # drop a copy next to your pages
```

```javascript
import { renderThreejsPayload } from "./tsdyn-threejs-loader.js";

const payload = await (await fetch("lorenz.json")).json();
const handle = renderThreejsPayload(document.querySelector("#viewer"), payload);
// …later, when the component unmounts:
handle.dispose();
```

It draws a **labelled scale frame** — box edges, 1/2/5-ladder ticks and the axis
names from `metadata.labels`. On a 3-D box the ticks are re-seated every frame onto
whichever parallel edge currently projects furthest from the box centre, so they
stay on the silhouette and never end up written across the attractor as you orbit.
The frame is on by default for a static payload and off while a reveal comet plays
(there the motion is the subject); `opts.axes` — surfaced as `axes=` on
`render` / `save` — forces either way.

It also sizes point clouds in **screen pixels**, not world units, with the dot
diameter and opacity chosen from the point count. That distinction is not cosmetic:
a 200 000-iterate Hénon export sized in world units drew every dot 23% as wide as
the whole attractor and rendered as a solid opaque slab, with the Cantor banding —
the entire content of the picture — buried underneath it.

It is written for real embedding contexts, not just a full-page demo:

- it sizes itself from a **`ResizeObserver` on its container**, so a viewer inside
  a tab panel or an accordion — hidden at boot, measuring 0x0 — renders at the
  right size the moment it is revealed (a `window.resize` listener never fires for
  that, which used to leave such an embed permanently at 640x420);
- it **idles when scrolled off-screen** (`IntersectionObserver`), so a page with
  six attractors is not running six WebGL loops for nobody;
- `dispose()` really tears down — both animation loops, both observers, and every
  geometry / material / texture — so mounting and unmounting does not leak;
- `devicePixelRatio` is capped at 2, because a thin-line scene gains nothing from
  9x the fragments on a 3x-DPR phone.

### In the system catalogue

Every **3-D ODE** system page in the [catalogue](../systems/index.md) embeds
exactly this pipeline: at docs-build time an animated `PHASE_PORTRAIT_3D` payload
of the attractor is exported, inlined into a small self-contained HTML document
(the import map + the inlined payload + this reference loader), and dropped onto
the page in an `<iframe>` — so the headline figure on a chaotic-attractor page is
the **live, orbitable comet** rather than a static PNG. Maps, DDEs, spatial-field
and stiff / discontinuous systems keep their static figure, and any page degrades
to the PNG when WebGL or the CDN is unavailable. The emitter is
`docs/_tooling/threejs_viewer.py`.

### Live demo

The exact payload the 3-D catalogue pages ship — the Lorenz attractor as an
**animated reveal comet** (a thin teal trail fading into the dark stage behind an
indigo state head, sweeping the full faint attractor) rendered in your browser by
the reference loader. **The camera is yours: drag to orbit, scroll to zoom — the
comet keeps playing while you move it.** The play/pause + restart controls sit
bottom-left.

<div id="tsdyn-threejs-viewer"
     style="width:100%;height:460px;border-radius:8px;overflow:hidden;background:#0b0f14"></div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
  import { renderThreejsPayload } from "../../_static/tsdyn-threejs-loader.js";
  const el = document.getElementById("tsdyn-threejs-viewer");
  try {
    const payload = await (await fetch("../../assets/threejs-demo/lorenz-threejs.json")).json();
    // An animated payload: the loader plays the reveal comet and holds the camera
    // still by default (orbitable by mouse) — no autoRotate override needed.
    renderThreejsPayload(el, payload);
  } catch (err) {
    el.textContent = "three.js demo unavailable (needs a network connection for the CDN build): " + err;
    el.style.color = "#9aa4c0";
    el.style.padding = "1rem";
  }
</script>

---

## Pitfalls

| Situation | What happens / what to do |
| --- | --- |
| `import tsdynamics` and expecting a plot library | None is imported — `ts.viz` is lazy, and backends register on first render. |
| No plotting library installed | `matplotlib` / `plotly` renders raise `VisualizationNotInstalled`; `json` / `threejs` still work (they need no dependency). |
| Naming a backend that declines the spec | Dispatch warns (`VisualizationDegraded`) and falls back to matplotlib. |
| A style / theme knob a backend ignores | One consolidated `VisualizationDegraded` names every dropped knob (json is exempt). |
| Strict `filterwarnings=["error"]` test suite | A degradation becomes an error — scope a narrow `pytest.warns` / ignore around the one call. |
| Saving `.mp4` with no ffmpeg | matplotlib raises — install ffmpeg, or save `.gif` (pillow) / `.html` (plotly). |
| `.render("json")` returns a `RenderResult`, not a string | Pass `raw=True` for the bare string, or read `.payload`. |
| A `VisualizationDegraded` about capped vertices | Expected — the export was thinned to `max_points` by arc length. Pass `max_points=None` to keep every vertex (and read the size budget first). |
| A `.html` three.js page shows the poster, not the attractor | The pinned three.js CDN was unreachable, or WebGL is off. The poster *is* the fallback working as designed. |
| `save("x.json")` vs `save("x.json", backend="threejs")` | Two different documents (spec IR vs BufferGeometry) behind one extension — name the backend. |

---

## See also

- [Plotting — the front door](plotting.md) — the `to_plot_spec` front door every
  render starts from.
- [Styling & themes](styling.md) — the per-key `honored_by` contract these
  degradation warnings enforce.
- [Composition](composition.md) — how composites tile natively in matplotlib and
  plotly (and lower to panelled payloads in three.js / json).
- [Animation](animation.md) — the movie export formats (`.mp4` / `.gif` / `.html`)
  and the three.js reveal comet.
