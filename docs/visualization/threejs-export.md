# three.js export

Every TSDynamics result, trajectory, or system describes itself as a backend-agnostic
[`PlotSpec`](../reference/top-level.md). Besides the matplotlib and plotly renderers, the spec can be
exported to a **three.js BufferGeometry-ready payload** — a plain JSON object a browser front-end
turns into an orbitable WebGL scene with no Python kernel in the loop. It is the natural format for
3-D attractors on the web (docs, dashboards, papers-as-web-pages).

## Exporting a payload

The `threejs` renderer ships in-tree and needs **no extra dependency** (it is pure-Python over the
spec IR):

```python
import tsdynamics as ts
from tsdynamics.viz.spec import PlotSpec, PlotKind, Layer, Axis
import numpy as np

# A 3-D attractor, coloured by time.
y = np.asarray(ts.Lorenz().trajectory(final_time=50.0, dt=0.01).y)
spec = PlotSpec(
    kind=PlotKind.PHASE_PORTRAIT_3D, ndim=3,
    x=Axis(label="x"), y=Axis(label="y"), z=Axis(label="z"),
    layers=[Layer(PlotKind.LINE3D, {"x": y[:, 0], "y": y[:, 1], "z": y[:, 2],
                                    "c": np.linspace(0, 1, len(y))})],
)

payload = spec.render("threejs")          # a JSON-able dict (a RenderResult.payload)
spec.render("threejs", path="lorenz.json")  # …or write it straight to a file
```

A 2-D phase portrait exports just the same (its `LINE` / `SCATTER` layers are lifted to `z = 0`).

## The payload schema

```json
{
  "schema_version": 1,
  "kind": "phase_portrait_3d",
  "title": "...",
  "geometries": [
    {
      "type": "line | points | surface",
      "label": "...",
      "positions": [x0, y0, z0, x1, y1, z1, ...],   // FLAT, Float32-ready
      "colors":    [r0, g0, b0, r1, g1, b1, ...],   // optional, per-vertex 0–1
      "indices":   [0, 1, 1, 2, 2, 3, ...]          // line segments / surface triangles
    }
  ],
  "metadata": {
    "schema_version": 1,
    "labels": {"x": "x", "y": "y", "z": "z"},
    "units":  {"x": "", "y": "", "z": ""},
    "bounds": {"x": [min, max], "y": [min, max], "z": [min, max]},
    "camera": {"position": [x, y, z], "target": [x, y, z], "up": [x, y, z]},
    "animation": {                              // ONLY when the spec is animated
      "fps": 30.0,
      "duration": null,                         // wall-clock seconds, or null
      "n_frames": null,
      "loop": true,
      "pingpong": false,
      "trail_length_samples": 400,              // comet tail length; null = persistent
      "head": true,
      "head_size": 6.0,
      "head_color": [r, g, b],                  // or null (inherit the line colour)
      "n_samples": 3701                         // vertices on the longest animated line
    }
  }
}
```

A web frontend reads each geometry's flat `positions` into a `Float32Array`, the optional `colors`
into a second `Float32Array` (vertex colours), and `indices` into the draw-index buffer — exactly the
three inputs a `THREE.BufferGeometry` wants. `metadata.camera` seeds the initial view.

`metadata.animation` is present **only** for an animated spec (e.g.
`tr.to_plot_spec(animate=True)`); a static export omits it and the geometry buffers are byte-for-byte
the non-animated payload. When it is present, the loader plays a **reveal comet** — a faint full-curve
backdrop with a bright windowed trail and a head marker sweeping the line — by advancing
`geometry.setDrawRange(start, count)` per frame (no buffer re-upload). Because `OrbitControls` is
independent of that draw-range update, you can **orbit the attractor with the mouse while it plays**.

The reveal comet sweeps a **line** index buffer (`LINE` / `LINE3D`), so the block is emitted only
when the spec has a line layer. An animated `points`-only (`SCATTER` / `MARKERS`) or `surface`-only
spec has nothing to reveal: the exporter drops the animation to a valid static payload (which the
loader renders, auto-rotating, as usual) and emits a `VisualizationDegraded` warning — the animation
is **never silently dropped**.

## The reference loader

[`tsdyn-threejs-loader.js`](../_static/tsdyn-threejs-loader.js) is a tiny reference ES module that
does the whole conversion — `BufferGeometry` for `line` (`LineSegments`), `points` (`Points`) and
`surface` (a normals-computed `Mesh`), per-vertex colours, a camera from `metadata.camera`,
`OrbitControls`, and (when `metadata.animation` is set) a `setDrawRange`-driven reveal comet with a
play/pause overlay:

```javascript
import { renderThreejsPayload } from "./tsdyn-threejs-loader.js";

const payload = await (await fetch("lorenz.json")).json();
renderThreejsPayload(document.querySelector("#viewer"), payload);
```

## In the system catalogue

Every **3-D ODE** system page in the [catalogue](../systems/continuous/index.md) embeds exactly this
pipeline: at docs-build time an animated `PHASE_PORTRAIT_3D` payload of the attractor is exported,
inlined into a small self-contained HTML document (the import map + the inlined payload + this
reference loader), and dropped onto the page in an `<iframe>` — so the headline figure on, say, the
[Lorenz](../systems/continuous/chaotic-attractors/Lorenz.md) page is the **live, orbitable comet**
rather than a static PNG. Maps, DDEs, spatial-field and stiff/discontinuous systems keep their static
figure, and any page degrades to the PNG when WebGL or the CDN is unavailable. The emitter is
`docs/_tooling/threejs_viewer.py`.

## Live demo

The Lorenz payload exported above, rendered in your browser by the reference loader (drag to orbit,
scroll to zoom):

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
    const payload = await (await fetch("../../_static/lorenz-threejs.json")).json();
    renderThreejsPayload(el, payload, { autoRotate: true });
  } catch (err) {
    el.textContent = "three.js demo unavailable (needs a network connection for the CDN build): " + err;
    el.style.color = "#9aa4c0";
    el.style.padding = "1rem";
  }
</script>
