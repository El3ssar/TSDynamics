# TSDynamics Visualization Cheatsheet (`VIZ-CHSH`)

Everything you can do with the plotting layer: how to build a `PlotSpec`, every knob you can turn,
every kind, every parameter, and how to render/save. The whole layer is **backend-agnostic** — you
build a `PlotSpec` (plain data), then a renderer draws it. `import tsdynamics` pulls in **no**
plotting library; `ts.viz` and the renderers load lazily on first use.

---

## 1. Mental model

```
data / system / result ──▶ PlotSpec ──▶ renderer ──▶ figure / file
        to_plot_spec()       (knobs)      .plot()/.save()/.render()
        ts.viz.plot()      (chainable)
```

- **One panel** = `Trajectory.to_plot_spec(...)` (or `system.to_plot_spec(...)`, or a result's).
- **A figure** (one or many panels) = `ts.viz.plot(*things, layout=...)`. **Spec in → spec out**, so
  a `plot(...)` result feeds straight back into `plot(...)` (fully recursive composition).
- A `PlotSpec` **renders itself**: `.plot()`, `.render(backend=)`, `.save(path)`, and notebook display.
- Every **knob is a chainable method** that *mutates the spec and returns `self`*:
  `spec.relabel(...).rescale(...).style(...).animate(...)`. Because they touch the spec (not a
  renderer), the same tweak renders identically on every backend.

```python
import tsdynamics as ts
tr = ts.Lorenz().integrate(final_time=100, dt=0.01)

tr.plot()                                   # quickest: render with defaults
tr.to_plot_spec().relabel(title="Lorenz").save("lorenz.png")
ts.viz.plot(tr, components="x").save("x.html")
```

---

## 2. Quickstart one-liners

```python
tr.plot()                                          # auto kind, default backend (matplotlib)
tr.plot(backend="plotly")                          # interactive
tr.plot(components="x")                             # just the x component → time series
tr.plot(components=["x", "z"])                      # 2-D phase portrait
tr.to_plot_spec(kind="delay", tau=0.5)             # delay embedding x(t) vs x(t-τ)
tr.to_plot_spec(animate=True).save("orbit.mp4")    # animation → video (matplotlib)
tr.to_plot_spec(animate=True).save("orbit.html")   # animation → interactive HTML (plotly)
ts.viz.plot(tr1, tr2, components="x")              # overlay two runs on one axes
ts.viz.plot(p1, p2, layout="stack")                # stack two panels
system.plot(final_time=200, dt=0.01, backend="plotly")   # integrate + plot in one call
```

---

## 3. Building a spec

### 3.1 From a `Trajectory`

```python
Trajectory.to_plot_spec(
    kind=None,            # str | None  — override the auto kind, or a recipe ("delay")
    *,
    components=None,      # int | str | sequence | None — which state components to draw
    animate=False,        # bool | dict | Animation — turn into a reveal animation
    **kind_kw,            # per-kind options (tau / color_by / transpose)
) -> PlotSpec
```

- **`components`** — names (from the system's `variables`) or integer indices; a lone value is fine
  (`components="x"` or `components=0`); a sequence picks several; `None` = all.
- **Auto kind** is chosen from the **number of selected components**:

  | # selected | kind | meaning |
  |---|---|---|
  | 1 | `TIME_SERIES` | value vs time |
  | 2 | `PHASE_PORTRAIT_2D` | x–y portrait |
  | 3 | `PHASE_PORTRAIT_3D` | x–y–z portrait |
  | **4+** | `SPACETIME` | field image (never a misleading 3-D portrait of the first 3 coords) |

- **`kind=`** forces any kind (a `PlotKind` value as a string, e.g. `"time_series"`), **or** a
  *recipe*: `"delay"` (an `x(t)` vs `x(t-τ)` embedding) or **`"field"`** (a spatial-field plot of
  a spatially-extended system — see §3.4).
- **`**kind_kw`** — options valid for one kind only (passing one to the wrong kind raises
  `InvalidParameterError`):

  | option | valid for | meaning |
  |---|---|---|
  | `tau` | `kind="delay"` (**required**) | embedding lag in **time units** (→ samples via `meta["dt"]`) |
  | `color_by` | `time_series`, `phase_portrait_2d/3d` | `"time"` or `"speed"` — colour the curve |
  | `transpose` | `spacetime` | swap the image axes (bool) |

`Trajectory.plot(backend=None, **kwargs)` is sugar: spec-shaping keys (`kind`, `components`, `tau`,
`color_by`, `transpose`, `animate`) go to `to_plot_spec`; everything else is an inline tweak
(§6) or a backend kwarg.

### 3.2 From a system

```python
system.to_plot_spec(kind=None, **kwargs) -> PlotSpec
system.plot(backend=None, **kwargs)
```

The system **integrates first**, then delegates to the trajectory's `to_plot_spec`. Keywords split
on the closed plot-keyword set:

- **Plot-shaping** (`components`, `tau`, `color_by`, `transpose`, `animate`, `kind`) → `to_plot_spec`.
- **Everything else** → `trajectory(...)`: `final_time`, `dt`, `steps`, `ic`, `backend=` (integration backend), …

```python
ts.Lorenz().plot(final_time=200, dt=0.005, components=["x", "z"], title="Lorenz x–z")
```

### 3.3 From a result or an existing `PlotSpec`

Analysis results (`OrbitDiagram`, `RecurrenceMatrix`, …) implement `to_plot_spec()` / `.plot()` too.
An already-built `PlotSpec` is itself plottable (it *is* what `ts.viz.plot` returns).

### 3.4 Composition — `ts.viz.plot`

```python
ts.viz.plot(
    *things,              # Trajectory / system / result / PlotSpec (mix freely); a single list is unwrapped
    layout="overlay",     # "overlay" | "stack" | "row" | "grid"
    animate=False,        # bool | dict | Animation — animate the WHOLE figure
    **build_kw,           # forwarded to each non-spec thing's to_plot_spec (components / kind / per-kind opts)
) -> PlotSpec
```

| `layout` | result | notes |
|---|---|---|
| `"overlay"` (default) | one panel, all on shared axes | only **overlay-compatible** kinds (`TIME_SERIES`, `PHASE_PORTRAIT_2D`, `PHASE_PORTRAIT_3D`); mixing an image with a portrait, or 2-D with 3-D, raises — use a panelled layout |
| `"stack"` | `COMPOSITE`, one column | auto-shares x **only** when every panel is a `TIME_SERIES` with the same x label (else independent axes) |
| `"row"` | `COMPOSITE`, one row | side by side |
| `"grid"` | `COMPOSITE`, 2-D grid | near-square unless `rows`/`cols` set on the `Layout` |

- `build_kw` (e.g. `components="x"`) is applied to **each** thing — `plot(a, b, components="x")` overlays
  the same view of each. Cannot be combined with an already-built `PlotSpec` argument.
- **Recursive:** `plot(plot(...), plot(...), layout="stack")` composes (composite inputs flatten one level).

```python
px = ts.viz.plot(lor1, lor2, components="x")     # overlay → one panel
py = ts.viz.plot(lor1, lor2, components="y")
ts.viz.plot(px, py, layout="stack")              # two stacked panels
```

### 3.5 Spatial-field movies — `kind="field"`

A **spatially-extended** system (a method-of-lines PDE whose state vector is a flattened field)
plays its field *over time* with the `"field"` recipe. Each frame is the field's **spatial** state
at that instant; the per-frame plot's shape follows the field's spatial dimensionality (one kind,
`SPATIAL_FIELD`, covers both — the renderer dispatches on the spatial ndim):

| field | each frame | example |
|---|---|---|
| **1-D** `u(x)` | a **line** (the profile) — a travelling-wave movie | `KuramotoSivashinsky` |
| **2-D** `u(x,y)` | an `imshow` **heatmap** — an evolving-field movie | `GrayScott`, `SwiftHohenberg` |

```python
ts.systems.SwiftHohenberg().to_plot_spec(kind="field", animate=True).save("sh.mp4")  # 2-D heatmap movie
ts.systems.GrayScott().to_plot_spec(kind="field", animate=True).save("gs.gif")        # reaction-diffusion
ts.KuramotoSivashinsky().to_plot_spec(kind="field", animate=True).save("ks.mp4")      # 1-D profile movie
ts.systems.SwiftHohenberg().to_plot_spec(kind="field").save("final.png")              # static = the FINAL field
```

- **No `shape` kwarg.** The spatial grid comes from the **system**, via the optional
  `_field_shape: tuple[int, ...]` ClassVar (recorded onto `traj.meta["field_shape"]` at integration
  time, so a bare `Trajectory` carries it). A system with no `_field_shape` (or a 1-D one) is a 1-D
  profile — honest, never guessing a 2-D grid.
- **Multi-block fields.** A state that packs several field blocks declares `field_labels` (e.g.
  Gray–Scott's `("u", "v")`); `components="u"|"v"` picks the block, defaulting to the **last** (the
  activator convention).
- **Static = the final field** (no `animate`); **animation = the field movie** (the frames model — see
  §9). The field movie is **matplotlib-only** (`.mp4`/`.gif`): plotly *declines* an animated
  `SPATIAL_FIELD` (it draws a static field), threejs draws a 1-D profile / declines a 2-D field.

---

## 4. Kinds — the vocabulary

The `PlotKind` enum is a **frozen, reviewed contract** (governance gate `tests/test_viz_vocab.py`). It
partitions into **semantic kinds** (what a whole spec *means* — a renderer dispatches on
`PlotSpec.kind`) and **layer marks** (how one `Layer` is drawn — dispatch on `Layer.kind`).

### 4.1 Semantic kinds (pass to `kind=` as the string value)

| `PlotKind` | string value |
|---|---|
| `TIME_SERIES` | `"time_series"` |
| `PHASE_PORTRAIT_2D` | `"phase_portrait_2d"` |
| `PHASE_PORTRAIT_3D` | `"phase_portrait_3d"` |
| `SPACETIME` | `"spacetime"` |
| `SPATIAL_FIELD` | `"spatial_field"` (a field at one instant — 1-D line / 2-D heatmap; via the `"field"` recipe — see §3.5) |
| `COMPOSITE` | `"composite"` (multi-panel figure) |
| `BIFURCATION` | `"bifurcation"` |
| `ORBIT_DIAGRAM` | `"orbit_diagram"` |
| `COBWEB` | `"cobweb"` |
| `RETURN_MAP` | `"return_map"` |
| `POINCARE_SECTION` | `"poincare_section"` |
| `BASINS_IMAGE` | `"basins_image"` |
| `RECURRENCE_PLOT` | `"recurrence_plot"` |
| `POWER_SPECTRUM` | `"power_spectrum"` |
| `SPECTROGRAM` | `"spectrogram"` |
| `SCALING_FIT` | `"scaling_fit"` |
| `DIMENSION_SPECTRUM` | `"dimension_spectrum"` |
| `DIAGNOSTIC_CURVE` | `"diagnostic_curve"` |
| `COMPLEXITY_CURVE` | `"complexity_curve"` |
| `LINE_FAMILY` | `"line_family"` |
| `ENSEMBLE_FAN` | `"ensemble_fan"` |
| `HISTOGRAM_NULL` | `"histogram_null"` |
| `LYAPUNOV_SPECTRUM` | `"lyapunov_spectrum"` |
| `EIGENVALUE_PLANE` | `"eigenvalue_plane"` |
| `FIXED_POINTS_OVERLAY` | `"fixed_points_overlay"` |
| `VECTOR_FIELD` | `"vector_field"` |
| `PHASE_PORTRAIT_FIELD` | `"phase_portrait_field"` |
| `CONTINUATION` | `"continuation"` |
| `CATEGORICAL_BAR` | `"categorical_bar"` |
| `FEATURE_BARS` | `"feature_bars"` |
| `TRAJECTORY_ANIMATION` | `"trajectory_animation"` |
| `ENSEMBLE_ANIMATION` | `"ensemble_animation"` |

> Animation is normally an **orthogonal modifier** (`spec.animation`, §9), *not* a kind — you rarely set
> the `*_ANIMATION` kinds directly.

### 4.2 Layer marks (how a `Layer` draws)

`LINE` `"line"` · `LINE3D` `"line3d"` · `SCATTER` `"scatter"` · `MARKERS` `"markers"` ·
`IMAGE` `"image"` · `QUIVER` `"quiver"` · `SURFACE3D` `"surface3d"` · `HISTOGRAM` `"histogram"` ·
`BAR` `"bar"` · `AREA` `"area"` · `ERRORBAR` `"errorbar"`.

### 4.3 Kind aliases / recipes

| you pass `kind=` | resolves to | extra |
|---|---|---|
| `"delay"` / `"delay_embedding"` | `delay_embedding` (emits a `PHASE_PORTRAIT_2D`) | needs `tau=` (time units) |

---

## 5. The knobs — chainable spec methods

All of these **mutate the spec and return `self`**, so they chain. (`PlotSpec` also has the `is_composite`,
`is_animated`, and `has_color_channel()` read-only helpers.)

### Labels / scales / limits / ticks

```python
spec.relabel(*, x=None, y=None, z=None, title=None)        # set axis labels and/or title
spec.rescale(*, x=None, y=None, z=None)                    # "linear" | "log" | "symlog" (| "categorical")
spec.limits (*, x=None, y=None, z=None)                    # (lo, hi) per axis
spec.ticks  (*, x=None, y=None, z=None)                    # explicit tick locations (sequences)
```
`z=` is ignored when the spec has no z axis. Example:
```python
spec.relabel(x="r", y="x*", title="Logistic").rescale(y="log").limits(x=(2.8, 4.0)).ticks(x=[3, 3.5, 4])
```

### Style (per-layer visual keys + axis visibility)

```python
spec.style(*, layer=None, axes=None, **kw)
```
- `layer=None` styles **every** layer; `layer=i` only layer `i`.
- `axes=False` **hides the axes entirely** — ticks, labels, gridlines, and (in 3-D) the grey
  background panes — for a clean "object floating in space" look. `axes=True` re-enables. `None`
  leaves visibility unchanged.
- `**kw` are backend-neutral per-layer style keys: `color`, `cmap`, `lw` (line width), `alpha`,
  `marker`, `s` (marker size). Unknown keys are passed through / ignored by a renderer.

```python
spec.style(color="crimson", lw=2.0, alpha=0.8)
spec.style(layer=0, color="black").style(layer=1, color="orange")
spec.style(axes=False)                                     # clean attractor, no axes
```

### Colour / colorbar / legend

```python
spec.colorize(*, clim=None, colorbar=None, legend=None)
```
- `clim=(vmin, vmax)` — colour range.
- `colorbar=` — a `Colorbar` object, or `True` (default colorbar) / `False` (drop).
- `legend=` — a `Legend` object, or `True` (default legend) / `False` (drop).

```python
spec.colorize(clim=(0, 1), colorbar=True, legend=False)
```

```python
spec.autocolor()        # attach a default Colorbar + infer clim for a colored spec (no-op otherwise;
                        # never overwrites a clim/colorbar you set). has_color_channel() reports eligibility.
```

### Animation knobs (each also **turns animation on**)

```python
spec.animate(*, fps=None, duration=None, n_frames=None, loop=None, pingpong=None, mode=None)
spec.trail(length=<unchanged>, *, fade=None)
spec.head(show=None, *, size=None, color=None, symbol=None)
spec.camera(*, elev=None, azim=None, spin=None)
spec.clock(show=True, *, fmt=None)
```

| method | parameter | values / meaning | default* |
|---|---|---|---|
| `animate` | `fps` | playback frames/sec | `30` |
| | `duration` | total seconds (`n_frames = round(duration*fps)`) | `None` |
| | `n_frames` | explicit frame count (overrides duration/fps) | `None` |
| | `loop` | repeat | `True` |
| | `pingpong` | forward then reverse each loop | `False` |
| | `mode` | `"reveal"` (comet) or `"frames"` (spatial-field movie, via `kind="field"`) | `"reveal"` |
| `trail` | `length` | `("time", t)` / `("steps", n)` / `None` (persistent). Omit ⇒ unchanged | `None` (persistent) |
| | `fade` | fade opacity head→tail | `False` |
| `head` | `show` | draw the moving "current state" marker | `True` |
| | `size` | marker size | `6` |
| | `color` | marker colour (`None` inherits layer colour) | `None` |
| | `symbol` | marker symbol (e.g. `"o"`) | `"o"` |
| `camera` | `elev`, `azim` | fixed view angle (deg); recorded in `meta["camera"]` | mpl default |
| | `spin` | camera revolutions over the animation (0 = still) | `0` |
| `clock` | `show` | draw a live time readout | `True` (when called) |
| | `fmt` | label format, `{t}` = current time | `"t = {t:.2f}"` |

\* These are the `Animation` dataclass defaults (what `.animate()` on a *static* spec gives — a
**persistent** trail). Note `to_plot_spec(animate=True)` instead seeds a **windowed comet**
(`trail_kind="steps"`, `trail_length=max(2, min(n_steps//10, 200))`, head on except for a plain time
series) — see §9.

```python
(tr.to_plot_spec(animate=True)
   .animate(fps=60, duration=12)
   .trail(length=("time", 3.0), fade=True)
   .head(size=10, color="white", symbol="o")
   .camera(elev=25, azim=-60, spin=1)
   .clock(fmt="t = {t:.1f}s")
   .style(axes=False)
   .save("wow.mp4"))
```

---

## 6. Inline tweaks — `.plot(...)` / `.save(...)` shorthand

When you call `.plot(**tweaks)` (on a trajectory, system, result, or spec), these keywords are applied
to the spec before rendering; **anything else is forwarded to the backend**.

| inline keyword | does | inline keyword | does |
|---|---|---|---|
| `xlabel` / `ylabel` / `zlabel` | `relabel(...)` | `xlim` / `ylim` / `zlim` | `limits(...)` |
| `title` | `relabel(title=...)` | `xticks` / `yticks` / `zticks` | `ticks(...)` |
| `xscale` / `yscale` / `zscale` | `rescale(...)` | `clim` / `colorbar` / `legend` | `colorize(...)` |

```python
tr.plot(xlabel="time", ylabel="x", yscale="log", ylim=(1e-3, 1), title="decay")
spec.plot(title="overridden", colorbar=True)     # PlotSpec.plot has the same sugar
```

---

## 7. Rendering & saving

```python
spec.render(backend=None, **backend_kw)              # raw render → figure / payload
spec.plot(backend=None, **tweaks)                    # inline tweaks (§6) + render
spec.save(path, *, backend=None, fps=None, dpi=None, size=None, **backend_kw) -> path
```

- `render` picks a backend by name or capability; falls back to matplotlib when a backend declines a kind.
- `save` picks a backend from the **file extension** unless you pass `backend=`. `size=(w, h)` is in
  **pixels** (converted via `dpi`); `fps`/`dpi` apply to video/gif (and image dpi).

### Backends

| name | role | strengths | formats |
|---|---|---|---|
| `"matplotlib"` | universal **reference** renderer (draws every kind; the fallback) | stills + animations | `.png` `.pdf` `.svg` `.jpg`; `.mp4` `.gif` (FuncAnimation via ffmpeg/pillow) |
| `"plotly"` | interactive | 2-D + 3-D, orbit/zoom, animated real-time HTML | `.html` |
| `"json"` | data export | JSON of the spec | `.json` |
| `"threejs"` | data export | three.js `BufferGeometry`-ready JSON (static geometry) | JSON via `render("threejs", path=...)` |

### `save()` extension routing (when `backend=` not given)

| spec | extension | backend |
|---|---|---|
| **static** | `.json` | json |
| | `.html` | plotly (interactive page) |
| | `.png`/`.pdf`/`.svg`/`.jpg`/… | matplotlib |
| **animated** | `.html` | plotly (real-time, rotatable-while-playing) |
| | `.mp4`/`.gif`/`.webm`/`.mov`/… | matplotlib |
| | a still image (`.png`/…) | matplotlib — renders the **final, fully-revealed frame** |

```python
spec.save("fig.png", dpi=200, size=(1600, 1000))
spec.save("orbit.mp4", fps=30, dpi=150)
spec.save("orbit.html")                              # animated → plotly real-time
spec.render("threejs", path="lorenz.json")           # threejs payload (.save(".json") → json backend, not threejs)
```

---

## 8. Composition layouts in depth

- `layout="overlay"` merges overlay-compatible single-panel specs onto **one** axes, disambiguating
  legend labels by source. Incompatible kinds (image vs portrait, 2-D vs 3-D) raise.
- `layout="stack"|"row"|"grid"` builds a `COMPOSITE` spec carrying child `panels` + a `Layout`. The
  **matplotlib** renderer tiles panels into a subplot grid; **plotly declines `COMPOSITE`** and falls
  back to matplotlib (an *overlay* single-panel spec still renders natively in plotly).
- Composites animate in **lockstep** (one master clock; each panel keeps its per-kind head default).

---

## 9. Animation deep-dive

Animation is an **orthogonal modifier**: any spec of any kind (single-panel or composite) becomes a
movie by carrying an `Animation` (`spec.animation`); the semantic `kind` is unchanged, and a backend
that cannot animate draws the **final frame**.

- **Turn it on:** `to_plot_spec(animate=True | dict | Animation)`, `ts.viz.plot(..., animate=...)`, or
  any of `.animate()/.trail()/.head()/.camera()/.clock()` on a static spec.
- **Two frame models (`Animation.mode`):**
  - **`reveal`** (the default): the layer keeps its full static data and each frame shows a *slice* — a
    comet whose head is the current sample and whose tail reaches back `trail_length` (`None` ⇒ the whole
    curve persists, the classic "orbit draws itself in").
  - **`frames`**: a **spatial-field movie** — the field of a spatially-extended system *played over
    time* (see §3.5). Each frame is the field's spatial state at that instant: a **1-D** field is a
    travelling-wave line, a **2-D** field an `imshow` heatmap, and consecutive frames carry genuinely
    different data. Built via `to_plot_spec(kind="field", animate=True)`, which stamps `mode="frames"`
    with the field defaults (no comet window, no head — the field *is* the motion). The producer keeps
    the **final** field as the static layer data, so a still save draws the final field. **mpl-only.**
- **`animate=True` defaults** (via `to_plot_spec`, *reveal* kinds): a **windowed** comet —
  `trail_kind="steps"`, `trail_length ≈ n_steps/10` (capped 2–200), head **on** for portraits/spacetime,
  **off** for a plain time series. (A bare `Animation()` from `.animate()` instead defaults to a
  *persistent* trail.) For a persistent "draw-it-in", call `.trail(length=None)`.
- **Per-backend:**
  - **matplotlib** → `FuncAnimation`; `.save("x.mp4" / "x.gif")` (ffmpeg / pillow). Camera **spin** and
    the **clock** work here. ✅ The solid, working path — **and the only backend for a field movie**.
  - **plotly** → `.save("x.html")`: a real-time `requestAnimationFrame` *reveal* animation,
    camera-rotatable while playing. **Declines an animated `SPATIAL_FIELD`** (falls back to mpl mp4/gif).
    Camera-spin/clock are matplotlib-only.
  - **threejs** → static geometry only; **does not animate** (camera `autoRotate` only).

```python
tr.to_plot_spec(animate=True).save("orbit.mp4")                      # smooth comet video (reveal)
tr.to_plot_spec(animate={"fps": 60, "loop": False}).save("once.gif") # dict overrides Animation fields
tr.to_plot_spec(animate=True).trail(length=None).camera(spin=2).save("drawin.mp4")
ts.systems.GrayScott().to_plot_spec(kind="field", animate=True).save("field.mp4")  # 2-D field movie
```

---

## 10. Dataclass reference (fields + defaults)

You rarely build these by hand (the front door does), but knowing the fields tells you exactly what a
knob sets and what round-trips through `to_dict`/`from_dict`.

### `PlotSpec`
`kind` · `layers=[]` · `x=Axis()` · `y=Axis()` · `z=None` · `clim=None` · `colorbar=None` ·
`legend=None` · `title=""` · `ndim=2` · `aspect="auto"` · `annotations=[]` · `meta={}` ·
`panels=[]` · `layout=None` · `animation=None`.

### `Axis`
`label=""` · `scale="linear"` · `limits=None` · `ticks=None` · `tickformat=None` · `categories=None`.

### `Layer`
`kind` · `data={}` · `label=None` · `style={}`.
**Data channels** (closed vocab; a renderer ignores channels it doesn't consume):
`x`, `y`, `z` (coords) · `c` (per-vertex colour/scalar) · `u`, `v` (quiver/vector) ·
`lo`, `hi` (area/fan band edges) · `err` (errorbar) · `cat` (category index) · `size` (per-point
marker size) · `frame` (animation axis, deferred).
**Style keys:** `color`, `cmap`, `lw`, `alpha`, `marker`, `s`.

### `Colorbar`
`label=""` · `location="right"` · `ticks=None` · `tickformat=None` · `show=True` · `cmap=None` ·
`norm=None` · `discrete=False`.

### `Legend`
`show=True` · `location="best"` · `title=""`.

### `Layout`
`mode="stack"` · `rows=None` · `cols=None` · `share_x=False` · `share_y=False`.

### `Annotation`
`kind` (`"vline"|"hline"|"text"|"span"`) · `text=""` · `x=None` · `y=None` · `span=None` ·
`axis="x"` · `style={}`.

### `Animation`
`fps=30.0` · `duration=None` · `n_frames=None` · `loop=True` · `pingpong=False` · `mode="reveal"` ·
`trail_kind=None` · `trail_length=None` · `trail_fade=False` · `head=True` · `head_size=6.0` ·
`head_color=None` · `head_symbol="o"` · `spin=0.0` · `clock=False` · `clock_format="t = {t:.2f}"`.
(`DEFAULT_FRAMES = 360` when neither `n_frames` nor `duration` is set.)

---

## 11. Allowed-value quick tables

| field | allowed values |
|---|---|
| axis `scale` (`rescale`) | `linear`, `log`, `symlog`, `categorical` |
| colour `norm` | `linear`, `log`, `symlog` |
| `aspect` | `auto`, `equal` |
| `ndim` | `1`, `2`, `3` |
| colorbar `location` | `right`, `left`, `top`, `bottom` |
| legend `location` | `best`, `upper right`, `upper left`, `lower left`, `lower right`, `right`, `center left`, `center right`, `lower center`, `upper center`, `center` |
| annotation `kind` | `vline`, `hline`, `text`, `span` |
| annotation `axis` | `x`, `y` |
| animation `mode` | `reveal`, `frames` |
| `trail_kind` | `time`, `steps` |
| `Layout.mode` | `stack`, `row`, `grid` |
| `ts.viz.plot(layout=)` | `overlay`, `stack`, `row`, `grid` |
| `color_by` | `time`, `speed` |

---

## 12. Recipes (combinations)

```python
import tsdynamics as ts
import numpy as np

tr = ts.Lorenz().integrate(final_time=100, dt=0.01).after(5)

# --- static stills -------------------------------------------------------
tr.to_plot_spec().style(color="teal", lw=0.6).relabel(title="Lorenz").save("attractor.png")
tr.to_plot_spec(components=["x", "z"]).limits(x=(-25, 25)).save("xz.pdf")
tr.to_plot_spec(components="x").rescale(y="symlog").save("x.svg")
tr.to_plot_spec(kind="delay", tau=0.5).style(color="black").save("delay.png")
tr.to_plot_spec(components="x", color_by="speed").save("speed.png")
tr.to_plot_spec().style(axes=False).save("clean.png")          # no axes/panes

# --- interactive ---------------------------------------------------------
tr.plot(backend="plotly")
tr.to_plot_spec().save("attractor.html")                       # interactive 3-D

# --- composition ---------------------------------------------------------
ts.viz.plot(tr, components="x").save("x.png")                  # overlay (single thing)
ts.viz.plot(ts.Lorenz(), ts.Rossler(), components="x")        # overlay two systems' x(t)
ts.viz.plot(
    tr.to_plot_spec(components="x"),
    tr.to_plot_spec(components=["x", "y"]),
    tr.to_plot_spec(),
    layout="grid",
).save("dashboard.png")

# --- animation -----------------------------------------------------------
tr.to_plot_spec(animate=True).save("orbit.mp4")                # windowed comet → video
tr.to_plot_spec(animate=True).trail(length=None).save("drawin.gif")          # persistent
(tr.to_plot_spec(animate=True)
   .animate(fps=60, duration=15).trail(length=("time", 4), fade=True)
   .head(size=9, color="white").camera(spin=1).style(axes=False)
   .save("stunning.mp4"))
ts.viz.plot(p1, p2, layout="stack", animate=True).save("panels.mp4")         # lockstep composite

# --- system path (integrate + plot) --------------------------------------
ts.Lorenz().plot(final_time=200, dt=0.005, components=["x", "z"], title="x–z")
ts.Lorenz().to_plot_spec(final_time=200, dt=0.01, animate=True).save("lorenz.mp4")

# --- serialize / round-trip ----------------------------------------------
d = tr.to_plot_spec().to_dict()                                # JSON-able
from tsdynamics.viz.spec import PlotSpec
spec2 = PlotSpec.from_dict(d)                                  # rebuilt, no recompute
```

---

## 13. Gotchas

- **`components` count drives the auto kind** (1/2/3/4+). Force any kind with `kind="..."`.
- **`kind="delay"` requires `tau`** (in *time units*; converted via `meta["dt"]`).
- **`.save("x.json")` → the `json` backend**, not threejs; use `render("threejs", path=...)` for the
  three.js payload.
- **plotly declines `COMPOSITE`** (multi-panel) → falls back to matplotlib. Overlay single panels render
  natively in plotly.
- **`animate=True` (via `to_plot_spec`) = windowed comet**; a bare `.animate()` on a static spec =
  *persistent* trail. Use `.trail(length=...)` to choose.
- **plotly animated HTML is currently not animating** (open issue) — use matplotlib `.mp4`/`.gif`.
  **threejs never animates** (static geometry; camera auto-rotate only).
- **Knobs mutate in place.** `spec.copy()`-style isolation isn't automatic; `to_dict`/`from_dict`
  gives you a fresh spec if you need one.
- Importing `tsdynamics` loads **no** plot library; the first `.plot()`/`.render()`/`.save()` (or
  `ts.viz`) pulls in the backend.
```
