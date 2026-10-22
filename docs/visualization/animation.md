---
title: Animation
description: Turn a plot into a controlled, labelled movie, link multiple views, and export real MP4 or GIF output.
---

<span class="ts-kicker">Visualization · Animation</span>

# Let the motion explain the picture

<div class="tsd-page-intro" markdown="1">

A still shows where an orbit goes. A movie shows how it gets there. Animate the
same plot, choose a short trail or a persistent reveal, and keep the playback
long enough to follow. Simulation time and playback time are separate choices.

</div>

<figure class="tsd-showcase">
<video controls playsinline preload="none" aria-label="Lorenz orbit in state space linked to its x time series" poster="../../assets/figures/viz/animation-linked-poster.png">
<source src="../../assets/figures/viz/animation-linked.mp4" type="video/mp4">
Your browser cannot play this video. <a href="../../assets/figures/viz/animation-linked.mp4">Download the linked-view MP4</a>.
</video>
<figcaption>One Lorenz orbit, shown in state space and over time. Press play to
follow the moving state in both panels; playback starts only when you choose.</figcaption>
</figure>

## Build a linked view

<!-- tsd-example: animation-linked -->
```python
import tsdynamics as ts

trajectory = ts.systems.Lorenz(ic=[1, 1, 1]).run(final_time=30, dt=0.01)
portrait = ts.plot(trajectory, components=["x", "z"], title="State space", color="#2CC5AE")
trace = ts.plot(trajectory, "time_series", components="x", title="x over time", color="#2CC5AE")

movie = (portrait | trace).theme("dark").font(family="sans-serif").size(9, 4)
movie.animate(duration=12, fps=24).trail(("time", 4), fade=True)
```

Twelve seconds of playback reveal thirty time units of simulation. `fps=24` sets
the encoded frame rate; `duration=12` sets playback length. The four-time-unit
trail follows the trajectory's sampling grid, independently of the playback
speed. Both panels use the same recorded trajectory and advance together.

## Save a movie

MP4 needs the optional Matplotlib renderer (`pip install "tsdynamics[viz]"`) and
an FFmpeg executable on your path. For example, install FFmpeg with
`brew install ffmpeg` on macOS or your operating system's package manager.

```python
# skip-doctest — writes a video and requires an external FFmpeg executable
movie.save("lorenz-linked.mp4", dpi=120)
```

For a short loop where a video player is inconvenient, save a GIF instead:

```python
# skip-doctest — writes an encoded animation with the optional drawing backend
movie.save("lorenz-linked.gif", dpi=85)
```

MP4 usually gives a smaller, smoother artifact; GIF uses Pillow and works
without FFmpeg. Use a solid `.background(...)` or a theme for either format.
GIF has limited transparency and colour depth, so a transparent still's look
will not necessarily survive animation encoding.

## Shape the reveal

```python
orbit = ts.plot(trajectory, components=["x", "y", "z"])
orbit.theme("dark").recolor("#2CC5AE").style(lw=0.8)
orbit.animate(duration=8, fps=30)
orbit.trail(("time", 4), fade=True)
orbit.head(size=8, color="#E8912D")
orbit.camera(elev=22, azim=-60)
```

- `.trail(("time", 4))` keeps the last four time units.
- `.trail(("steps", 200))` keeps a fixed number of samples.
- `.trail(None)` reveals the whole path and leaves it visible.
- `.head(show=False)` removes the moving marker.

A trail measured in time requires a uniformly sampled trajectory. For map data,
use a trail in steps: map iterations have no continuous integration time.

```python
orbit.camera(spin=0.5)       # half a camera revolution over the movie
orbit.clock(fmt="t = {t:.1f}")
```

Camera spin and the physical-time clock are Matplotlib features. Keep axis
labels and a clock when units matter. An unlabeled, floating attractor can be a
useful illustration, but it does not communicate its coordinate scale.

## A presentation-ready recipe

Use a quiet background, one accent for the orbit and a contrasting moving
marker. Let the camera move slowly enough that the state remains the subject.

```python
presentation = ts.plot(trajectory, components=["x", "y", "z"])
presentation.theme("dark").font(family="sans-serif", size=11).size(10, 6)
presentation.recolor("#2CC5AE").style(linewidth=0.9)
presentation.animate(duration=12, fps=30)
presentation.trail(("time", 5), fade=True, backdrop=True, backdrop_alpha=0.10)
presentation.head(size=7, color="#E8912D")
presentation.camera(elev=24, azim=-65, spin=0.15)
presentation.clock("t = {t:.1f}")
```

```python
# skip-doctest — writes an MP4 using FFmpeg
presentation.save("lorenz-talk.mp4", size=(1600, 960), dpi=160)
```

`.size(10, 6)` sets the figure size in inches. The `size=` on `.save()` sets
the exported pixel dimensions; `dpi=` determines the conversion between physical
size and pixels. Check the result at the size your audience will actually see.
For a reveal that keeps the geometry secret until the state reaches it, use
`.trail(("time", 5), backdrop=False)`. For a growing drawing that never erases,
use `.trail(None)`.

### Playback and appearance controls

| Control | What it changes |
|---|---|
| `.animate(duration=12, fps=30)` | Playback length and frame rate; reuses the recorded samples |
| `.animate(n_frames=360, fps=30)` | An explicit frame count; takes precedence over `duration` |
| `.animate(loop=False)` | Repeat behavior in players that support the setting |
| `.animate(pingpong=True)` | Forward then reverse playback; the return leg runs backward in simulation time |
| `.trail(("steps", 150), fade=True)` | A fading window of 150 recorded samples |
| `.trail(backdrop=False)` | Hide the faint full path without changing the trail length |
| `.head(size=7, color="#E8912D", symbol="o")` | The current-state marker |
| `.camera(elev=24, azim=-65, spin=0.15)` | View angles in degrees and camera revolutions over the movie |
| `.clock("frame {i}: t = {t:.1f}")` | A labelled clock; `{i}` is the encoded frame index |

More output frames can make playback last longer, but do not add simulation
accuracy: when necessary, samples repeat. Integrate with a finer output grid
when the path itself needs more detail. With `pingpong=True`, `duration` and
`n_frames` describe the forward pass; the return pass adds playback time without
repeating the two endpoints. A six-second forward pass therefore produces
almost twelve seconds of forward-and-reverse footage. MP4 stores one playback sequence;
whether it repeats is controlled by the video player. GIF timing is quantized
by the format, so an arbitrary requested frame rate may be approximated.

These settings describe the Matplotlib movie path. The
[backend guide](backends.md#capability-negotiation-visualizationdegraded) explains
which options an interactive export can honor and how unsupported effects are
reported.

## Play a spatial field frame by frame

A field movie is different from an orbit reveal: each frame is the entire
spatial state at one instant. The model's field metadata chooses a 1-D profile
or 2-D heatmap automatically.

```python
field = ts.systems.GrayScott(N=24)
trajectory_field = field.run(final_time=1200, dt=20)
field_movie = ts.plot(trajectory_field, "spatial_field").animate(duration=6, fps=10)
field_movie.theme("dark")
```

This smaller grid is a quick starting point. Pattern formation depends on the
parameters, initial field and time window; extend the simulation to see slower
structures develop. The generator records genuine spatial snapshots rather
than manufacturing frames from a static image.

## Share an interactive reveal

```python
# skip-doctest — requires the interactive extra and writes an HTML artifact
orbit.save("lorenz.html")
```

With `pip install "tsdynamics[interactive]"`, the HTML export lets readers
inspect supported single-panel views in their browser. The three.js export is
another option for line, point and surface geometry; see
[backends and export](backends.md). Browser playback and encoded video are
different renderers, so check the relevant capability warning before depending
on a particular effect.

Matplotlib renders nested layouts and composite movies. Plotly declines
animated composites and spatial-field movies; use MP4 or GIF for those figures.
The embedded MP4 above has native playback controls and never autoplays, so it
also remains still for readers who prefer reduced motion.
