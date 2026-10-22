---
title: Animation
description: Nine real movies show orbit reveals, trails, signed camera spin, overlays, linked and nested layouts, map iterations, field snapshots, and reverse playback.
---

<span class="ts-kicker">Visualization · Animation</span>

# Make the motion explain the picture

<div class="tsd-page-intro tsd-animation-intro" markdown="1">
<div markdown="1">

Make a plot. Add `.animate()`. Save the same object as a movie. Start with the
short example below, then change one thing at a time: what stays visible,
where the camera looks, or how related views sit together.

Every movie on this page comes from its runnable example. Press play when
you want motion; the players have native controls and never autoplay.

</div>

<figure class="tsd-demo-card">
<video width="960" height="600" controls playsinline preload="none" aria-label="A Lorenz orbit gradually draws itself in three-dimensional state space" poster="../../assets/figures/viz/animation-reveal-poster.png">
<source src="../../assets/figures/viz/animation-reveal.mp4" type="video/mp4">
<a href="../../assets/figures/viz/animation-reveal.mp4">Download the orbit reveal</a>.
</video>
<figcaption>The recorded path builds up and stays visible. Ten seconds of playback
show thirty simulation time units; the labels describe state coordinates.</figcaption>
</figure>
</div>

<nav class="tsd-recipe-nav" aria-label="Animation recipes">
<a href="#your-first-movie">First movie</a>
<a href="#leave-a-trail">Trails</a>
<a href="#move-the-camera">Camera</a>
<a href="#overlay-two-orbits">Overlays</a>
<a href="#link-the-views">Linked views</a>
<a href="#nest-the-layout">Layouts</a>
<a href="#show-map-iterations">Maps</a>
<a href="#play-a-spatial-field">Fields</a>
<a href="#play-forward-and-back">Ping-pong</a>
<a href="#all-the-controls">All controls</a>
</nav>

## Your first movie

Install the optional drawing backend with `pip install "tsdynamics[viz]"`.
For MP4, install FFmpeg on your path; on macOS, `brew install ffmpeg`.

<!-- tsd-example: animation-reveal -->
```python
import tsdynamics as ts

trajectory = ts.systems.Lorenz(ic=[1, 1, 1]).run(final_time=30, dt=0.01)
movie = ts.plot(trajectory, color="#2CC5AE").theme("dark")
movie.animate(duration=10, fps=24)
```



```python
# skip-doctest — writes a movie using the external FFmpeg executable
movie.save("lorenz.mp4", size=(960, 600), dpi=120)
```

The output grid (`dt=0.01`) controls the recorded data. Playback duration and
frame rate control how those samples are shown. More movie frames do not make
the simulation more accurate.

## Leave a trail

Keep the whole path with `.trail(None)`, or keep the most recent samples with
`.trail(300)`. A fading trail makes recent motion stand out. Use `.copy()` when
you want independent versions of the same plot.

<!-- tsd-example: animation-trails -->
```python
import tsdynamics as ts

trajectory = ts.systems.Lorenz(ic=[1, 1, 1]).run(final_time=30, dt=0.01)
base = ts.plot(trajectory, components=["x", "z"], color="#2CC5AE").theme("dark")
persistent = base.copy().relabel(title="Keep the path").trail(None)
window = base.copy().relabel(title="Last 300 samples").trail(300, backdrop=False)
fading = base.copy().relabel(title="Fade those 300").trail(
    300, fade=True, backdrop=False
)
movie = (persistent | window | fading).font(family="sans-serif", size=10)
movie.animate(duration=12, fps=24)
```

<figure class="tsd-showcase tsd-demo-card">
<video width="1104" height="432" controls playsinline preload="none" aria-label="The same Lorenz orbit with a persistent path, a moving window, and a fading window" poster="../../assets/figures/viz/animation-trails-poster.png">
<source src="../../assets/figures/viz/animation-trails.mp4" type="video/mp4">
<a href="../../assets/figures/viz/animation-trails.mp4">Download the trail comparison</a>.
</video>
<figcaption>Same samples, same playback position. Left keeps the past; middle and
right keep 300 samples, about three simulation time units on this grid. Their
full-path backdrops are hidden so you can see exactly what the tail retains.</figcaption>
</figure>

To specify a physical duration instead, use `.trail(("time", 3))` on uniformly
sampled data. `.trail(("steps", 300))` is the explicit spelling of `.trail(300)`.
By default, a short trail has a faint full-path backdrop; tune it with
`.trail(backdrop=True, backdrop_alpha=0.10)`. Turn it off for a reveal that should
not show the ending early.

## Move the camera

`.camera(elev=..., azim=...)` sets a 3-D viewing angle in degrees. Add `spin` to
turn the camera over one forward pass: negative turns reverse the direction,
zero holds the view still, and positive turns move it the other way.

<!-- tsd-example: animation-camera -->
```python
import tsdynamics as ts

trajectory = ts.systems.Lorenz(ic=[1, 1, 1]).run(final_time=30, dt=0.01)
base = ts.plot(trajectory, color="#2CC5AE").theme("dark").style(axes=False)
base.trail(250, fade=True, backdrop_alpha=0.12).head(size=7, color="#E8912D")
reverse = base.copy().relabel(title="spin = -0.25").camera(
    elev=24, azim=-60, spin=-0.25
)
fixed = base.copy().relabel(title="spin = 0").camera(
    elev=24, azim=-60, spin=0
)
forward = base.copy().relabel(title="spin = +0.25").camera(
    elev=24, azim=-60, spin=0.25
)
movie = (reverse | fixed | forward).font(family="sans-serif", size=10)
movie.animate(duration=12, fps=24)
```

<figure class="tsd-showcase tsd-demo-card">
<video width="1152" height="480" controls playsinline preload="none" aria-label="The same three-dimensional orbit with a reverse quarter-turn, fixed camera, and forward quarter-turn" poster="../../assets/figures/viz/animation-camera-poster.png">
<source src="../../assets/figures/viz/animation-camera.mp4" type="video/mp4">
<a href="../../assets/figures/viz/animation-camera.mp4">Download the camera comparison</a>.
</video>
<figcaption>Each moving camera turns 90 degrees over twelve seconds. The middle
camera stays fixed. Axes are hidden here to isolate the viewing effect; use the
labelled first movie when coordinate scales matter.</figcaption>
</figure>

Set the individual panels' cameras before composing them. A later
`movie.camera(spin=0.25)` applies that spin to the whole figure. Camera motion
changes the view, not the trajectory.

## Overlay two orbits

Use `+` when the coordinates and units match. Here two nearby Lorenz starts
share one state-space panel, with distinct colours and a visible legend.

<!-- tsd-example: animation-overlay -->
```python
import tsdynamics as ts

model = ts.systems.Lorenz()
a = model.run(final_time=30, dt=0.01, ic=[1, 1, 1])
b = model.run(final_time=30, dt=0.01, ic=[1.005, 1, 1])
first = ts.plot(
    a, components=["x", "z"], labels=["x₀ = 1.000"], color="#2CC5AE"
)
second = ts.plot(
    b, components=["x", "z"], labels=["x₀ = 1.005"], color="#E8912D"
)
movie = (first + second).theme("dark").font(family="sans-serif")
movie.animate(duration=10, fps=24).trail(250, fade=True, backdrop_alpha=0.10)
movie.head(size=7, symbol="s").clock("t = {t:.1f}")
```

<figure class="tsd-showcase tsd-demo-card">
<video width="960" height="560" controls playsinline preload="none" aria-label="Two nearby Lorenz initial states evolve together on one labelled phase plane" poster="../../assets/figures/viz/animation-overlay-poster.png">
<source src="../../assets/figures/viz/animation-overlay.mp4" type="video/mp4">
<a href="../../assets/figures/viz/animation-overlay.mp4">Download the orbit overlay</a>.
</video>
<figcaption>Both runs use the same physical time grid. This shows the separation
of these two trajectories over this window; the movie alone does not estimate a
Lyapunov exponent.</figcaption>
</figure>

Use `|` or `/` when axes represent different quantities. A spectrum and a phase
portrait belong in separate panels, even if their colours match.

## Link the views

A row is just `portrait | trace`. Animate the composed figure once; both panels
advance along the same recorded trajectory.

<!-- tsd-example: animation-linked -->
```python
import tsdynamics as ts

trajectory = ts.systems.Lorenz(ic=[1, 1, 1]).run(final_time=30, dt=0.01)
portrait = ts.plot(
    trajectory, components=["x", "z"], title="State space", color="#2CC5AE"
)
trace = ts.plot(
    trajectory, "time_series", components="x", title="x over time", color="#2CC5AE"
)

movie = (portrait | trace).theme("dark").font(family="sans-serif").size(9, 4)
movie.animate(duration=12, fps=24).trail(("time", 4), fade=True)
```

<figure class="tsd-showcase tsd-demo-card">
<video width="1080" height="480" controls playsinline preload="none" aria-label="One Lorenz orbit linked to its x time series" poster="../../assets/figures/viz/animation-linked-poster.png">
<source src="../../assets/figures/viz/animation-linked.mp4" type="video/mp4">
<a href="../../assets/figures/viz/animation-linked.mp4">Download the linked views</a>.
</video>
<figcaption>The point in state space and the revealed time trace represent the
same sample. Both retain a four-time-unit fading trail.</figcaption>
</figure>

Inputs with different lengths follow their own relative progress over the same
playback window. That is not a guarantee of equal physical times: use the same
trajectory or a common time grid when that relationship matters.

## Nest the layout

Put a row above a timeline with `(front | side) / trace`. The bottom panel spans
the full width; each panel keeps its own axes.

<!-- tsd-example: animation-nested -->
```python
import tsdynamics as ts

trajectory = ts.systems.Lorenz(ic=[1, 1, 1]).run(final_time=30, dt=0.01)
front = ts.plot(
    trajectory, components=["x", "z"], title="Front: x, z", color="#2CC5AE"
)
side = ts.plot(
    trajectory, components=["y", "z"], title="Side: y, z", color="#E8912D"
)
trace = ts.plot(
    trajectory, "time_series", components="x",
    title="One shared timeline", color="#2CC5AE"
)
movie = ((front | side) / trace).theme("dark").font(family="sans-serif")
movie.animate(duration=12, fps=24).trail(300, fade=True, backdrop_alpha=0.10)
```

<figure class="tsd-showcase tsd-demo-card">
<video width="1080" height="720" controls playsinline preload="none" aria-label="Two Lorenz projections above a full-width synchronized time trace" poster="../../assets/figures/viz/animation-nested-poster.png">
<source src="../../assets/figures/viz/animation-nested.mp4" type="video/mp4">
<a href="../../assets/figures/viz/animation-nested.mp4">Download the nested movie</a>.
</video>
<figcaption>Two projections of one state, above its time history. Parentheses
make the layout explicit. This is a genuine three-panel movie.</figcaption>
</figure>

Tweaks on the figure can apply to all panels. Select a child for a local change:
`movie[0][1]` is the upper-right projection; `movie[1]` is the lower trace.
See [composition](composition.md) for grids and shared axes.

## Show map iterations

A map gives ordered iterates, not a continuously traversed curve. Draw them as
points and keep the iteration label explicit.

<!-- tsd-example: animation-map -->
```python
import tsdynamics as ts

iterates = ts.systems.Henon(ic=[0, 0]).run(steps=600)
movie = ts.plot(
    iterates, "phase_portrait", components=["x", "y"],
    primitive="points", color="#2CC5AE"
)
movie.theme("dark").font(family="sans-serif").style(markersize=2)
movie.animate(n_frames=240, fps=24).trail(None).head(False)
movie.clock("n = {t:.0f} · frame {i}")
```

<figure class="tsd-showcase tsd-demo-card">
<video width="960" height="600" controls playsinline preload="none" aria-label="The Henon map reveals its points in their actual iteration order" poster="../../assets/figures/viz/animation-map-poster.png">
<source src="../../assets/figures/viz/animation-map.mp4" type="video/mp4">
<a href="../../assets/figures/viz/animation-map.mp4">Download the ordered map reveal</a>.
</video>
<figcaption>Points appear in the order the map visits them. They are not joined
by invented continuous segments. The clock distinguishes the stored iteration
index from the output frame. Here 240 frames at 24 fps give ten seconds.</figcaption>
</figure>

For maps, use sample-count trails such as `.trail(100)`. The movie may reveal
several stored iterates per output frame; its frame rate is not a dynamical
integration step.

## Play a spatial field

A field movie replaces the whole spatial state each frame. Select the field by
name, then animate the same plot; the trajectory carries its grid metadata.

<!-- tsd-example: animation-field -->
```python
import tsdynamics as ts

model = ts.systems.GrayScott(N=48, F=0.06, k=0.062, seed=0)
trajectory = model.run(final_time=6000, dt=25)
movie = ts.plot(trajectory, "spatial_field", components="v", cmap="magma")
movie.theme("dark").font(family="sans-serif").relabel(title="Gray–Scott activator v")
movie.animate(duration=10, fps=24).clock("t = {t:.0f}")
```

<figure class="tsd-showcase tsd-demo-card">
<video width="960" height="600" controls playsinline preload="none" aria-label="The Gray Scott activator evolves from a seeded patch into a spatial pattern" poster="../../assets/figures/viz/animation-field-poster.png">
<source src="../../assets/figures/viz/animation-field.mp4" type="video/mp4">
<a href="../../assets/figures/viz/animation-field.mp4">Download the field movie</a>.
</video>
<figcaption>Real snapshots of the activator on a periodic 48 × 48 grid, from a
seeded initial patch. The stored snapshots are 25 simulation time units apart;
this recipe plays the 6000-unit window over ten seconds.</figcaption>
</figure>

A one-dimensional field uses a changing profile line instead of a heatmap.
Keep a fixed colour range when comparing field frames. Trails, moving heads and
camera spin describe orbit reveals; they do not turn a heatmap into a moving
particle. Pattern formation depends on the model, parameters, grid and time
window, so this example states all four.

## Play forward and back

`pingpong=True` adds a reverse pass. This is reverse **playback** of recorded
samples; it does not solve the dynamics backward.

<!-- tsd-example: animation-pingpong -->
```python
import tsdynamics as ts

trajectory = ts.systems.Lorenz(ic=[1, 1, 1]).run(final_time=12, dt=0.02)
movie = ts.plot(trajectory, components=["x", "z"], color="#2CC5AE")
movie.theme("dark").font(family="sans-serif")
movie.animate(duration=5, fps=24, pingpong=True, loop=False)
movie.trail(100, fade=True, backdrop=False).head(color="#E8912D", size=8)
movie.clock("t = {t:.1f}")
```

<figure class="tsd-showcase tsd-demo-card">
<video width="960" height="560" controls playsinline preload="none" aria-label="A Lorenz path plays forward and then back while its time readout reverses" poster="../../assets/figures/viz/animation-pingpong-poster.png">
<source src="../../assets/figures/viz/animation-pingpong.mp4" type="video/mp4">
<a href="../../assets/figures/viz/animation-pingpong.mp4">Download the ping-pong movie</a>.
</video>
<figcaption>A five-second forward pass plus a return pass, without repeated
endpoints: 238 frames at 24 fps, about 9.92 seconds in total. Watch the time
readout change direction.</figcaption>
</figure>

MP4 stores that one sequence. The video player's repeat setting controls whether
it starts again; `.animate(loop=False)` controls repeat behavior in animation
players that support it. GIF playback timing is quantized, so an arbitrary
requested frame rate may be approximated.

## All the controls

Start with the recipes above, then adjust the same `movie` object. Fluent calls
mutate and return the plot; `movie.copy()` gives an independent version.

<div class="tsd-control-table" markdown="1">

| Control | Meaning | Visible example |
|---|---|---|
| `.animate(duration=10, fps=24)` | Forward-pass playback seconds and encoded frames per second | Every recipe |
| `.animate(n_frames=240, fps=24)` | Exact forward frame count; overrides `duration` | [Map iterations](#show-map-iterations) |
| `.animate(loop=False)` | Repetition in players that support it; MP4 repetition belongs to its player | [Forward and back](#play-forward-and-back) |
| `.animate(pingpong=True)` | Append a reverse pass without repeating its endpoints | [Forward and back](#play-forward-and-back) |
| `.animate(mode="reveal")` | Progressively show recorded orbit samples; the default | [First movie](#your-first-movie) |
| `.animate(mode="frames")` | Play a stored spatial-field stack; detected automatically for field data | [Spatial field](#play-a-spatial-field) |
| `.trail(None)` | Keep the whole revealed path | [First movie](#your-first-movie), [maps](#show-map-iterations) |
| `.trail(300)` or `.trail(("steps", 300))` | Keep 300 stored samples | [Trail comparison](#leave-a-trail) |
| `.trail(("time", 4))` | Keep four physical time units on a uniform time grid | [Linked views](#link-the-views) |
| `.trail(fade=True)` | Fade older tail segments | [Trail comparison](#leave-a-trail) |
| `.trail(backdrop=False)` | Hide the faint full-path context | [Trail comparison](#leave-a-trail) |
| `.trail(backdrop_alpha=0.10)` | Set the full-path context opacity | [Overlay](#overlay-two-orbits) |
| `.head(False)` | Hide the current-state marker | [Maps](#show-map-iterations) |
| `.head(size=8, color="#E8912D", symbol="o")` | Set marker size, colour and shape | Amber circles in [ping-pong](#play-forward-and-back); square heads in the [overlay](#overlay-two-orbits) |
| `.camera(elev=24, azim=-60)` | Fixed 3-D angles in degrees | [Camera comparison](#move-the-camera) |
| `.camera(spin=-0.25)` / `0` / `0.25` | Reverse / fixed / forward quarter-turn per forward pass | [Camera comparison](#move-the-camera) |
| `.clock("t = {t:.1f}")` | Live time or iteration readout; `.clock(False)` hides it | [Overlay](#overlay-two-orbits), [maps](#show-map-iterations) |
| `.clock("frame {i}: t = {t:.1f}")` | Include the playback-frame index | [Map iterations](#show-map-iterations) |

</div>

A shorter recording can repeat sample positions to fill an explicit frame count.
A smoother recording needs a finer output grid, not just a larger movie frame
count. For ping-pong, `duration` and `n_frames` describe the forward pass; the
return leg adds `max(n_frames - 2, 0)` frames.

### Customize the figure, then export it

Themes, axes, annotations and layout remain ordinary plotting controls. Add
`.style(linewidth=0.9)`, `.font(family="sans-serif", size=11)`, `.background(...)`,
`.limits(...)`, `.gridlines(...)` or `.camera(...)` to the plot you already made.
The camera comparison uses `.style(axes=False)` deliberately; keep axes visible
for figures whose units or scales matter.

```python
# A separate presentation variant: the earlier movie keeps its settings.
presentation = movie.copy().theme("publication").font(family="sans-serif", size=11)
presentation.head(size=7, color="#D64562", symbol="s")
```

```python
# skip-doctest — exports files with optional drawing tools
presentation.save("talk.mp4", size=(1600, 960), dpi=160)
presentation.save("talk.gif", size=(800, 480), dpi=100)
```

`size=` on `.save()` is in pixels. `.size(width, height)` on a plot is in inches;
`dpi=` determines their conversion. MP4 needs FFmpeg. GIF uses Pillow and needs
no FFmpeg, but usually produces a larger file with fewer colours. Choose a solid
background for either movie format.

### Choose the output deliberately

| Output | Animation support |
|---|---|
| Matplotlib → MP4 / GIF | All recipes here, including fading trails, signed spin, clocks, overlays, nested layouts and field stacks |
| Plotly → HTML | Supported single-panel reveals; animated composites and spatial-field movies are refused rather than saved as partial output |
| three.js → HTML / geometry | Supported single-panel line/point reveals; no fading tails, camera spin, clock text or arbitrary marker shapes |
| PNG / SVG / PDF | A still of the final animation frame |
| JSON | Serializable plot data and animation settings; no encoded video |

Use the [backend guide](backends.md) for exact fallback and export rules.
For a supported single-panel interactive view, `movie.save("orbit.html")` uses
the interactive extra. An animated composite belongs in MP4 or GIF. These
examples use native video controls, `preload="none"`, and no autoplay, so motion
starts only when the reader requests it. Use the player's fullscreen control
when comparing small panels.
