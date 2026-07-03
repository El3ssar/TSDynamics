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

spec = ts.Lorenz().to_plot_spec(
    components=["x", "y", "z"], final_time=50.0, dt=0.01, ic=[1.0, 1.0, 1.0]
)

spec.plot()                    # draw with the default backend (matplotlib)
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
an orbitable WebGL scene with no Python kernel in the loop. It is the natural
format for 3-D attractors on the web. It is covered in full [below](#threejs-export).

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
>>> px = ts.Lorenz().to_plot_spec(components="x", final_time=20, dt=0.02, ic=[1.0, 1.0, 1.0])
>>> py = ts.Lorenz().to_plot_spec(components="y", final_time=20, dt=0.02, ic=[1.0, 1.0, 1.0])
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
>>> spec = ts.Lorenz().to_plot_spec(components=["x", "y", "z"], final_time=20, dt=0.02, ic=[1.0, 1.0, 1.0])
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

### Exporting a payload

The `threejs` renderer ships in-tree and needs **no extra dependency** (it is
pure Python over the spec IR):

```python
import tsdynamics as ts

# A 3-D attractor, coloured by elapsed time.
spec = ts.Lorenz().to_plot_spec(
    components=["x", "y", "z"], color_by="time",
    final_time=50.0, dt=0.01, ic=[1.0, 1.0, 1.0],
)

payload = spec.render("threejs")             # a RenderResult carrying the dict
spec.render("threejs", path="lorenz.json")   # …or write it straight to a file
```

```pycon
>>> result = spec.render("threejs")
>>> tj = result.payload
>>> list(tj.keys())
['schema_version', 'kind', 'title', 'geometries', 'metadata']
>>> tj["geometries"][0]["type"]        # a 3-D line → a "line" geometry
'line'
>>> "colors" in tj["geometries"][0]    # color_by="time" adds a per-vertex "c" channel
True
```

A 2-D phase portrait exports the same way (its `LINE` / `SCATTER` layers are
lifted to `z = 0`).

### The payload schema

```json
{
  "schema_version": 2,
  "kind": "phase_portrait_3d",
  "title": "...",
  "geometries": [
    {
      "type": "line | points | surface",
      "label": "...",
      "positions": [x0, y0, z0, x1, y1, z1, ...],   // FLAT, Float32-ready
      "indices":   [0, 1, 1, 2, 2, 3, ...],         // line segments / surface triangles
      "material": {                                 // the three.js-honored style keys
        "color": "#1f77b4", "linewidth": null,
        "markersize": null, "alpha": null, "zorder": null
      },
      "colors":    [r0, g0, b0, r1, g1, b1, ...]    // OPTIONAL, per-vertex 0-1 (a "c" gradient)
    }
  ],
  "metadata": {
    "schema_version": 2,
    "labels": {"x": "x", "y": "y", "z": "z"},
    "units":  {"x": "", "y": "", "z": ""},
    "bounds": {"x": [min, max], "y": [min, max], "z": [min, max]},
    "camera": {"position": [x, y, z], "target": [x, y, z], "up": [x, y, z]},
    "theme":  {"background": "...", "palette": ["#...", "..."]},
    "animation": {                              // ONLY when the spec is animated
      "fps": 30.0, "duration": null, "n_frames": null,
      "loop": true, "pingpong": false,
      "trail_length_samples": 200,              // comet tail length; null = persistent
      "head": true, "head_size": 6.0, "head_color": null,
      "n_samples": 5001                         // vertices on the longest animated line
    }
  }
}
```

A web frontend reads each geometry's flat `positions` into a `Float32Array`, the
optional `colors` into a second `Float32Array` (vertex colours), and `indices`
into the draw-index buffer — exactly the three inputs a `THREE.BufferGeometry`
wants. `metadata.camera` seeds the initial view, and `metadata.theme` gives the
scene background + the palette for auto-coloured layers.

A few honest exclusions the payload makes deliberately:

- **`material`** carries only the style keys three.js genuinely honors —
  `color` / `linewidth` / `markersize` / `alpha` / `zorder` (→ `renderOrder`).
  `linestyle`, marker *shape*, and `cmap` are **not** emitted (the loader has no
  way to honor them — the per-vertex `colors` use a fixed built-in colormap
  ramp), so they are dropped rather than serialized as dead fields. `caps` warns
  about any of them.
- **`colors`** appears only for a genuine per-vertex `"c"` gradient
  (`color_by="time"`, a speed field, …). A *solid* colour rides once on
  `material.color` — repeating it per vertex would be waste.
- **`metadata.theme`** carries only `background` and `palette` — the other theme
  fields (foreground, font, grid) have no analogue in a bare three.js scene.

`metadata.animation` is present **only** for an animated spec (e.g.
`spec.to_plot_spec(..., animate=True)`); a static export omits it and the geometry
buffers are byte-for-byte the non-animated payload. When it is present, the loader
plays a **reveal comet** — a faint full-curve backdrop with a bright windowed
trail and a head marker sweeping the line — by advancing
`geometry.setDrawRange(start, count)` per frame (no buffer re-upload). Because
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

### The reference loader

[`tsdyn-threejs-loader.js`](https://github.com/El3ssar/TSDynamics/blob/main/docs/_static/tsdyn-threejs-loader.js) is a tiny
reference ES module that does the whole conversion — `BufferGeometry` for `line`
(`LineSegments`), `points` (`Points`) and `surface` (a normals-computed `Mesh`),
per-vertex colours, a camera from `metadata.camera`, `OrbitControls`, and (when
`metadata.animation` is set) a `setDrawRange`-driven reveal comet with a
play/pause overlay:

```javascript
import { renderThreejsPayload } from "./tsdyn-threejs-loader.js";

const payload = await (await fetch("lorenz.json")).json();
renderThreejsPayload(document.querySelector("#viewer"), payload);
```

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

The Lorenz payload rendered in your browser by the reference loader (drag to
orbit, scroll to zoom):

<div id="tsdyn-threejs-viewer"
     style="width:100%;height:440px;border-radius:8px;overflow:hidden;background:#0b1020"></div>

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
    renderThreejsPayload(el, payload, { autoRotate: true });
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
