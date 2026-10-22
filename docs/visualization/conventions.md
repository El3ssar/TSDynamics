---
title: Figure conventions
description: Choose meaningful color, readable labels, comparable scales and the right export format for a scientific figure.
---

<span class="ts-kicker">Visualization · Conventions</span>

# Figure conventions

The [catalogue](../systems/index.md) and [examples](index.md) aim to make one
useful fact visible at a time. Meaningful color, readable labels and comparable
scales matter more than decoration. A page illustration can emphasize an orbit's
shape; a quantitative figure also needs its coordinates, units and sampling
conditions. Check the exported artifact at the size where readers will use it.

This page is the house style, so your figures can match it. The mechanics for
*applying* any of this — `.recolor`, `.style`, `.theme`, `.gridlines` — live in
[styling & themes](styling.md); here we cover *what* to choose and *why*.

The [scientific recipes](recipes.md) apply these choices in complete, editable
figures. Their settings are visible beside each picture so that a reader can
reproduce the sampling, projection, time window and color range.

## Match color to the quantity {#the-one-rule-colour-encodes-meaning}

Colour is a data channel, not ornament. The choice of colormap should follow the
*kind* of quantity being shown, because the human eye reads different maps in
different ways.

| Quantity | Colormap | Why |
| --- | --- | --- |
| A **sequential scalar field** — a spacetime image, a 2-D reaction–diffusion field, elapsed time along an orbit | **viridis** | perceptually uniform (equal data steps look equal), monotone in lightness (survives greyscale printing), colour-blind safe |
| A **cyclic / phase** quantity — an angle, a phase, a periodic coordinate | **twilight** | wraps back to its start, so 0 and 2π read as the same colour (a sequential map would show a false seam) |
| A **spectral density** (a spectrogram) | **magma** on a log norm | high dynamic range, dark-to-bright reads as low-to-high power |
| A **categorical label** — basins of attraction, attractor ids | **tab20** (discrete swatches) | distinct hues, no false ordering between unrelated categories |
| A **binary mask** — a recurrence plot | **binary** | ink-on-white, maximal contrast for a black/white structure |

These are the library's built-in per-kind defaults — a `SPACETIME` /
`SPATIAL_FIELD` / `IMAGE` renders viridis, a `BASINS_IMAGE` renders tab20, a
`RECURRENCE_PLOT` renders binary, a `SPECTROGRAM` renders magma with a log norm —
Use `cmap=` to choose a different map when it better represents your quantity.

!!! note "Avoid artificial contrast"

    Rainbow maps such as `jet` have uneven lightness, which can emphasize some
    intervals and obscure others. For sequential values, start with a map such
    as viridis, then check grayscale reproduction and color-vision accessibility.

## Keep comparisons easy to follow

The documentation's accent colors sit comfortably beside the field colormaps:

- **Teal `#11857A`** — use it for the main trace, the object of interest, links and marks.
- **Indigo `#574FCF`** — use it for a
  second series, integrator state, highlights, the "other" thing in a comparison.
- **Amber `#ffb000`** — a warm highlight for a moving head, an event marker, a
  reference value against a cool field. Reserved for the *one thing the eye
  should land on*.

```python
import tsdynamics as ts

lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).run(final_time=100, dt=0.01)

ts.plot(lor, components=["x", "z"]).recolor("#11857A")   # the house teal
```

When two traces are compared, teal-then-indigo is the default pairing (small
thing teal, big thing indigo, or reference teal, perturbed indigo). For three or
more, fall back to a full themed [palette](styling.md#palette-grid-font-background-size) rather than
hand-picking hues.

## Label the quantity, then check the typography

The site's body and code use IBM Plex. Plot themes choose their own type;
the scientific SVG examples use generic sans-serif labels so they stay readable
on machines without an installed scientific font. `.font(family="sans-serif")`
sets that choice explicitly. For quantitative figures, label every axis and
scalar color channel. Add physical units where the model has them.

```python
(
    ts.plot(lor, components=["x", "z"], color_by="time")
       .relabel(x="x", y="z", title="Lorenz (x, z)")   # axes + title
       # a color_by= channel grows a labelled colorbar automatically
)
```

The `variables` a system declares (Lorenz's `('x', 'y', 'z')`) become the axis
labels for free, so a portrait of named components is self-documenting. For a
manuscript, the **publication** theme switches to a serif face and larger type;
see [themes](styling.md#themes).

## One figure, one concept

A figure should answer a clear question. Several coordinated panels can answer
it together: geometry beside time, a result beside its convergence evidence, or
the same field at two times. Give each panel one readable job:

- **An orbit's shape** → a clean phase portrait, optionally with `.style(axes=False)`
  for an illustrative thumbnail. Keep labels for a quantitative comparison.
- **A comparison** → an [overlay](plotting.md#composing-panels) (two orbits on
  one plane, teal vs indigo) *or* a small-multiples [grid](plotting.md#composing-panels)
  (one system per panel) — never both crammed into a single busy axes.
- **A spatiotemporal field** → a [spacetime](plotting.md#spacetime-image) image
  with a colorbar, not a tangle of 20 overlaid line plots.

When quantities need different coordinates or units, give them separate panels. The
`ts.plot(..., layout="grid" | "row" | "stack")` composer exists precisely so
that "several related views" is *panels*, not clutter.

<figure markdown>
![A two-by-two grid showing the same Lorenz (x, z) portrait under the four built-in themes — default, dark, minimal, publication — the dark panel keeping a near-black background](../assets/figures/viz/themes.svg){ loading=lazy }
<figcaption>The same Lorenz <code>(x, z)</code> portrait under the four built-in themes, tiled with <code>ts.plot(layout="grid")</code>. A theme is a coherent set of conventions — palette, background, font, grid — applied at once; <strong>publication</strong> (serif, colour-blind-safe palette, larger type) is the one to reach for when the figure is bound for a journal.</figcaption>
</figure>

## Colorbars and legends

- **Label scalar color channels.** By default, A `color_by=` line, a
  spacetime image, a 2-D field — each grows a labelled colorbar and an inferred
  colour range (`clim`). Check the label and units, especially for a custom
  scalar channel.
- **Two or more labelled layers get a legend.** A single-trace figure does not —
  the axis labels already say what it is. An overlay disambiguates its legend by
  source automatically.
- **Fix the colour range across a comparison.** When two panels image the same
  quantity, set the same `clim=` on both so their colours are comparable —
  otherwise equal colors can represent different magnitudes.

```python
# a shared colour range across two field panels, so the colours mean the same thing
ks = ts.systems.KuramotoSivashinsky(N=64, L=22.0)
spec_a = ts.plot(ks, final_time=100.0, dt=0.5, ic=None)
spec_b = ts.plot(ks, final_time=200.0, dt=0.5, ic=None)

spec_a.colorize(clim=(-12.0, 15.0))
spec_b.colorize(clim=(-12.0, 15.0))
```

## State what the picture can establish

| Picture | State alongside it |
|---|---|
| A trajectory projection | Selected components, initial condition, retained time window and sampling interval |
| A power spectrum | Sampling units, estimator and record or segment length; broadband appearance alone does not prove chaos |
| An ensemble fan | Number of members, source of variation, percentile bounds and common time grid; a percentile band is not a mean confidence interval |
| A Lyapunov or dimension estimate | Finite observation window, numerical settings and convergence or scaling evidence |
| A planar field of a higher-dimensional model | Which coordinates vary and where the remaining coordinates are frozen |
| Two field snapshots | Same scalar, same color limits, grid resolution and snapshot times |

A projected curve can cross itself without the full state revisiting the same
point. A frozen two-dimensional slice need not be an invariant plane. A smooth
looking trace need not resolve fast dynamics. Labels and captions should preserve
these distinctions instead of assigning a stronger conclusion to the picture.

## Transparency and output format

- **Vector for line art, raster for fields.** A phase portrait or time series is
  line art — save it as `.pdf` or `.svg` so it scales crisply in a manuscript. A
  dense spacetime image or a high-resolution field is better as a `.png` at a
  publication `dpi` (300+). `.save(path)` picks the backend from the extension;
  set resolution with `.size(dpi=300)` or `save(..., dpi=300)`.
- **Choose the background with the ink.** Transparent output can suit a slide,
  but dark labels become unreadable on a dark surface. The new scientific
  examples use an explicit light paper surface; movies use a solid dark stage
  with light labels. Preview either format on its intended background.

```python
ts.plot(lor).style(axes=False).size(dpi=300).save("lorenz.png")
```

## A paper-ready recipe

Putting the conventions together — a colour-blind-safe portrait, labelled, on a
serif publication theme, exported as a vector:

```python
import tsdynamics as ts

lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).run(final_time=100.0, dt=0.01)

(
    ts.plot(lor, components=["x", "z"], color_by="time")
       .theme("publication")          # serif, Wong colour-blind-safe palette
       .relabel(x="x", y="z", title="Lorenz attractor")
       .gridlines(alpha=0.3)
       .save("lorenz-figure.pdf")      # vector output for a manuscript
)
```

For a whole document, set the theme once globally so every figure matches
without repeating `.theme(...)`:

```python
ts.viz.themes.use("publication")   # every subsequent plot uses it
# ...and when you are done with that document:
ts.viz.themes.use("default")
```

## Checklist

Before a figure leaves the notebook for a paper:

- [ ] Colormap matches the quantity — **viridis** sequential, **twilight**
      cyclic, **tab20** categorical; check accessibility and grayscale contrast.
- [ ] Every axis is labelled; a colour channel has a labelled colorbar.
- [ ] The figure makes **one** point; a second point is a second panel.
- [ ] Compared panels share a `clim=` so colours are comparable.
- [ ] Line art is vector (`.pdf`/`.svg`); dense fields are raster at 300+ dpi.
- [ ] The **publication** theme (or an equivalent colour-blind-safe palette) is
      on for journal output.

## See also

- [Styling & themes](styling.md) — how to apply all of the above.
- [Transforms and primitives](plotting.md) — building the figure in the first place.
- [System catalogue](../systems/index.md) — every page is a worked example of these conventions.
