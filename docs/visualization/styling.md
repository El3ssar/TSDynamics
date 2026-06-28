---
description: The visualization styling and theme system — the canonical per-layer style vocabulary, the fluent tweak methods, the four built-in themes, the global default, and per-backend honoring.
---

<span class="ts-kicker">Visualization</span>

# Styling & themes

Every plot TSDynamics produces is a backend-agnostic
[`PlotSpec`](../reference/top-level.md): the *semantic* description of a figure,
rendered on demand by matplotlib, plotly, or the three.js / JSON exporters. The
**styling layer** controls how that figure *looks* — line colours and widths,
markers, opacity, gridlines, fonts, backgrounds — through one **canonical,
validated, introspectable** vocabulary that every visual backend honors
consistently.

There are two levers:

- **Per-layer style** — the look of one drawable (a line, a scatter, an image):
  `color`, `linewidth`, `marker`, `alpha`, …  Set with `.style(...)` /
  `.recolor(...)`.
- **The figure-level [`Theme`](#themes)** — palette, font, background, grid, and
  line/marker defaults shared across the whole figure. Set with `.theme(...)` and
  its conveniences `.palette/.grid/.font/.background/.size`, or globally with
  `set_theme`.

A theme is applied first (it auto-colours every layer that carries no explicit
colour); per-layer style then overrides it.

All of these are **fluent tweaks** — they mutate the spec and return it, so they
chain:

```python
import tsdynamics as ts

spec = (
    ts.Lorenz()
    .to_plot_spec()
    .theme("dark")              # figure-level look
    .grid()                     # gridlines on
    .recolor("crimson")         # this layer's colour
    .style(lw=2, linestyle="dashed")  # this layer's line
)
spec.save("lorenz-dark.png")
```

---

## The style vocabulary

`STYLE_KEYS` is the **closed, reviewed** per-layer vocabulary. Each entry is a
`StyleKey` carrying its canonical name, the accepted aliases, the backends that
genuinely honor it, a value validator, and a one-line doc.

| Canonical | Aliases | Honored by | Meaning |
| --- | --- | --- | --- |
| `color` | `c` | mpl, plotly, three.js | line / marker / fill colour (CSS name, hex, or rgb tuple) |
| `linewidth` | `lw` | mpl, plotly, three.js | line width, pt (≥ 0) |
| `linestyle` | `ls` | mpl, plotly | one of `solid`, `dashed`, `dotted`, `dashdot` |
| `marker` | — | mpl, plotly | one of `circle`, `square`, `triangle`, `diamond`, `cross`, `x`, `star`, `none` |
| `markersize` | `ms`, `s` | mpl, plotly, three.js | marker size, pt (≥ 0) |
| `alpha` | `opacity` | mpl, plotly, three.js | opacity in `[0, 1]` |
| `cmap` | `colormap`, `colorscale` | mpl, plotly | colormap name for the colour / image channel |
| `fill` | — | mpl, plotly | fill the area under / between (bool) — `AREA` marks only |
| `fillalpha` | — | mpl, plotly | fill opacity in `[0, 1]` — `AREA` marks only |
| `zorder` | — | mpl, plotly, three.js | integer draw order (three.js maps it to `renderOrder`) |

### Aliases

Every key accepts the friendly short spellings shown above, and a few values are
canonicalised too — the matplotlib short line styles (`-`, `--`, `:`, `-.`) and
single-character markers (`o`, `s`, `^`, `D`, `+`, `x`, `*`). They are rewritten
to their canonical form before storage, so the spec only ever holds canonical
keys:

```pycon
>>> from tsdynamics.viz import normalize_style
>>> normalize_style({"lw": 2, "c": "red", "s": 8, "ls": "--"})
{'linewidth': 2.0, 'color': 'red', 'markersize': 8.0, 'linestyle': 'dashed'}
```

### Validation

Values are coerced and range-checked by each key's validator. A bad value raises
`ValueError` at the call site rather than silently producing a broken plot:

```pycon
>>> normalize_style({"alpha": 1.5})
Traceback (most recent call last):
    ...
ValueError: invalid value for style key 'alpha': expected a value in [0, 1], got 1.5
```

Sizes (`linewidth`, `markersize`) must be non-negative; opacities (`alpha`,
`fillalpha`) must lie in `[0, 1]`; `zorder` must be a genuine `int`. A `bool` is
*rejected* where a magnitude is expected — `alpha=True` is almost always a
mistake, so it raises instead of quietly becoming `1.0`.

### Unknown keys warn — they are never silently swallowed

A typo or an unsupported key is **dropped with a single warning**, naming the
offending key(s) and listing the vocabulary. There is no more "I set
`colour='red'` and nothing happened":

```pycon
>>> import warnings
>>> with warnings.catch_warnings(record=True) as w:
...     warnings.simplefilter("always")
...     normalize_style({"colour": "red", "lw": 2})
{'linewidth': 2.0}
>>> w[0].category.__name__
'VisualizationDegraded'
>>> str(w[0].message)
"unknown style key(s) dropped: 'colour'; known keys are alpha, ..."
```

`normalize_style` is the single choke point every style dict passes through —
both the `.style(...)` tweak and each renderer call it — so this guarantee holds
everywhere.

---

## Upgrading from v3 — breaking changes

The v5 redesign makes the style vocabulary a **closed, validated set**. Calls
that silently worked on the old v3 matplotlib path — because they were forwarded
straight to matplotlib — now go through `normalize_style` and raise a
`ValueError` **at the call site** if the value is not in the canonical vocabulary.
This is deliberate: the same spec must render the same on plotly and the
three.js / JSON exporters, so a knob that only matplotlib understands cannot be
allowed through. The trade is a louder upgrade for a portable, predictable spec.

```python
# v3 — forwarded to matplotlib, silently accepted
spec.style(marker="v", linestyle=(0, (3, 1)), alpha=1.5, zorder=2.0)

# v5 — each of these raises ValueError at the call site; map to the canonical set
spec.style(marker="triangle", linestyle="dashed", alpha=1.0, zorder=2)
```

### Markers — canonical shapes only

`marker` is the canonical set `circle`, `square`, `triangle`, `diamond`,
`cross`, `x`, `star`, `none`, plus the documented single-character aliases
`o s ^ D + x *`. The matplotlib-only spellings that have **no** canonical
equivalent now raise:

| v3 matplotlib marker | v5 — use instead |
| --- | --- |
| `'v'`, `'<'`, `'>'` (oriented triangles) | `"triangle"` |
| `'^'` | `"triangle"` (kept as an alias — still works) |
| `'d'` (thin diamond) | `"diamond"` |
| `'8'` (octagon) | `"circle"` (nearest) |
| `'P'` (filled plus) | `"cross"` |
| `'p'`, `'h'`, `'H'` (pentagon / hexagons) | `"star"` or `"circle"` (nearest) |

This is a deliberate **cross-backend** choice: three.js and the JSON exporter
cannot render oriented or exotic matplotlib markers, so the vocabulary is
restricted to shapes every visual backend can draw (or honestly decline). We
keep it a **hard error** rather than silently substituting — a quiet
down-triangle → up-triangle swap would corrupt a figure more confusingly than a
clear `ValueError` naming the unknown marker.

### Linestyles — named set only

`linestyle` is the named set `solid`, `dashed`, `dotted`, `dashdot`, plus the
short aliases `- -- : -.`. The matplotlib **dash-tuple** form (e.g.
`(0, (3, 1, 1, 1))`) now raises — pick the nearest named style:

```python
spec.style(linestyle=(0, (5, 2)))   # v3 — raises ValueError now
spec.style(linestyle="dashed")      # v5
```

### Range / type checks

Values are now coerced and range-checked, so out-of-range or wrong-typed values
that v3 tolerated raise:

- `alpha` (and `fillalpha`) must lie in `[0, 1]` — v3 tolerated `alpha > 1`
  (matplotlib clamped it); it now raises.
- `linewidth` / `markersize` must be `≥ 0`.
- `zorder` must be an **integer** — a float like `2.0` raises (v3 silently
  floored it via `int(...)`). A NumPy integer is accepted.
- A `bool` is rejected where a magnitude is expected (`alpha=True`,
  `linewidth=True`) rather than quietly becoming `1.0`.

Unknown keys — including a near-miss typo like `colour=` — are **not** an error:
they are dropped with a single [`VisualizationDegraded`](#per-backend-honoring)
warning naming the offending key(s) (see *Unknown keys warn*, above), so the
silent no-op of v3 (`colour='red'` did nothing) is gone.

### Migrating: introspect the vocabulary

`STYLE_KEYS` is the introspectable source of truth — there is no guesswork about
what a key accepts:

```pycon
>>> from tsdynamics.viz import STYLE_KEYS
>>> sorted(STYLE_KEYS)
['alpha', 'cmap', 'color', 'fill', 'fillalpha', 'linestyle', 'linewidth', 'marker', 'markersize', 'zorder']
>>> STYLE_KEYS["marker"].aliases          # the accepted alternate spellings
()
>>> STYLE_KEYS["linewidth"].aliases
('lw',)
>>> STYLE_KEYS["markersize"].aliases
('ms', 's')
```

(`marker` carries no `aliases` tuple — its alternate spellings are *value*
aliases, the single-character `o s ^ D + x *`, canonicalised by the key's
validator rather than the key-name index.) When a `ValueError` surprises you on
upgrade, the message names the offending key and lists the accepted values; the
table above plus `STYLE_KEYS` is the full map.

---

## Fluent tweak methods

All tweaks mutate the spec and **return `self`**, so they chain in any order and
render identically on every backend.

### `.style(**keys, layer=None, axes=None)`

Merge style keys into one layer or every layer. `**keys` go through
`normalize_style` (aliases canonicalised, values validated, unknown keys warned).

```python
spec = ts.Rossler().to_plot_spec()

spec.style(color="teal", lw=2, alpha=0.8)   # apply to every layer
spec.style(linestyle="dotted", layer=0)     # apply to layer 0 only
spec.style(axes=False)                        # hide axes entirely (object "in space")
```

`axes=False` is a figure-level switch (not a style key): it hides ticks, labels,
gridlines, and the 3-D background panes — handy for a clean attractor still or
animation. Every backend honors it.

### `.recolor(*colors, layer=None)`

The per-layer colour shorthand. With no `layer`, colour `i` is assigned to layer
`i`, cycling if there are more layers than colours; with an `int` layer, that one
layer is set to `colors[0]`.

```python
ts.viz.plot(ts.Lorenz(), ts.Rossler()).recolor("crimson", "royalblue")
```

### `.theme(name | Theme, /, **overrides)`

The figure-level look (palette / font / background / grid / line defaults). The
`theme` argument is **positional-only** — call `spec.theme("dark")`, never
`spec.theme(theme="dark")` (the keyword name is reserved so a theme field
literally called `theme` could go in `**overrides`). Pass a registered name, a
`Theme` instance, or `None` (snapshot the active global default *now*), and
optionally override individual fields on top:

```python
spec.theme("publication")                       # a named built-in
spec.theme("dark", font_family="monospace")     # built-in + override
spec.theme(None, line_width=2.0)                # global default + override
```

Note the snapshot semantics: `.theme("dark")` *pins* the dark theme onto the
spec. A spec that never calls `.theme(...)` carries no theme of its own and
resolves to the active global default at render time (so it tracks a later
[`set_theme`](#the-global-default)); calling `.theme(...)` detaches it from
future global changes.

### `.palette`, `.grid`, `.font`, `.background`, `.size`

Conveniences that each touch one facet of the theme (or, for `.size`, the figure
geometry):

```python
spec.palette(["#1b9e77", "#d95f02", "#7570b3"])  # explicit colour cycle
spec.palette("dark")                              # borrow a registered theme's palette
spec.grid(show=True, axis="y", alpha=0.3)         # gridlines on the y-axis
spec.font(family="serif", size=12)                # theme font
spec.background("#0b1020")                         # figure / axes facecolor
spec.size(width=8, height=5, dpi=150)              # figure geometry (lives in meta, not the theme)
```

`figsize` / `dpi` are deliberately **not** theme fields — they are figure
geometry, stored in `spec.meta` by `.size(...)`.

### Chaining

Because every tweak returns `self`, a full figure recipe reads top to bottom:

```python
spec = (
    ts.Lorenz()
    .to_plot_spec()
    .theme("dark")
    .grid()
    .recolor("crimson")
    .style(lw=2, linestyle="dashed")
    .font(family="monospace")
    .size(width=7, height=7)
)
spec.save("lorenz.html")   # interactive (plotly, by extension)
```

The static tweaks compose with the [animation](../visualization/threejs-export.md)
modifiers (`.animate/.trail/.head/.camera/.clock`) the same way — they are all
mutate-and-return-self.

---

## Themes

A `Theme` is a **frozen** dataclass describing the figure-level look applied
*before* per-layer style:

| Field | Meaning |
| --- | --- |
| `palette` | colour cycle for auto-coloured layers (those with no explicit `color`) |
| `background` | figure / axes facecolor |
| `foreground` | default ink — text / axes / ticks colour |
| `font_family`, `font_size`, `title_size` | typography |
| `grid`, `grid_color`, `grid_alpha` | default gridline visibility / look |
| `line_width`, `marker_size` | default line / marker sizes |

A theme round-trips through `to_dict` / `from_dict`, and `theme.merged(**fields)`
returns a copy with fields overridden (the original is unchanged — it is frozen).

```python
from tsdynamics.viz import Theme

base = Theme(name="brand", palette=("#0b6", "#e63"), font_family="serif")
warm = base.merged(background="#fffaf0", grid=True)
```

### The built-in themes

Four themes ship out of the box:

| Theme | Look |
| --- | --- |
| `default` | light background, a clean 10-colour qualitative palette, sans-serif, grid off |
| `dark` | near-black background (`#11131a`), light ink, a brighter palette |
| `minimal` | light, muted palette, slightly thinner lines, grid off |
| `publication` | serif font, larger labels / titles, a colourblind-safe palette (Wong 2011) |

```pycon
>>> import tsdynamics as ts
>>> ts.viz.themes()
['dark', 'default', 'minimal', 'publication']
```

Apply one per figure with `.theme(...)`:

```python
ts.Lorenz().to_plot_spec().theme("publication").save("lorenz-pub.pdf")
```

### Registering your own

Build a `Theme` and register it under its name; it then joins `themes()` and is
reachable by name everywhere a name is accepted:

```python
from tsdynamics.viz import Theme, register_theme

register_theme(Theme(
    name="paper",
    palette=("#222222", "#999999", "#0072b2"),
    background="#ffffff",
    font_family="serif",
    grid=True,
))

ts.Lorenz().to_plot_spec().theme("paper")   # now usable by name
```

### The global default

`set_theme` / `get_theme` read and write the **single** mutable global in the viz
layer — the active default theme that any spec without its own theme resolves to
at render time:

```pycon
>>> ts.viz.get_theme().name          # the active default
'default'
>>> ts.viz.set_theme("dark")         # every un-themed plot is now dark
>>> ts.viz.get_theme().name
'dark'
>>> ts.viz.set_theme("default")      # back to baseline
```

`set_theme` accepts a registered name *or* a `Theme` instance (which it registers
and activates in one step). `set_theme("default")` always returns to the
baseline.

Because a spec that never called `.theme(...)` resolves lazily, switching the
global default re-themes every such spec on its next render — set it once at the
top of a notebook or script and every plot follows:

```python
ts.viz.set_theme("publication")
ts.Lorenz().to_plot_spec().save("fig1.pdf")   # publication-themed
ts.Rossler().to_plot_spec().save("fig2.pdf")  # …and so is this
```

---

## Per-backend honoring

The three *visual* backends differ in what they can render. Each `StyleKey`
declares its `honored_by` set — the backends that genuinely apply it — and this
is an **enforced contract**: where a backend cannot honor a knob you set, the
renderer emits **one consolidated `VisualizationDegraded`** naming the dropped
knobs, rather than silently ignoring them. (The JSON exporter does not draw — it
faithfully *serialises* every field — so it is exempt from this negotiation.)

| Style key | matplotlib | plotly | three.js |
| --- | :---: | :---: | :---: |
| `color` | ✓ | ✓ | ✓ |
| `linewidth` | ✓ | ✓ | ✓ |
| `markersize` | ✓ | ✓ | ✓ |
| `alpha` | ✓ | ✓ | ✓ |
| `zorder` | ✓ | ✓ | ✓ (→ `renderOrder`) |
| `linestyle` | ✓ | ✓ | — |
| `marker` (shape) | ✓ | ✓ | — (points only) |
| `cmap` | ✓ | ✓ | — (fixed ramp) |
| `fill`, `fillalpha` | ✓ | ✓ | — |

The notable gaps:

- **three.js** does not honor `linestyle`, marker *shape*, or an arbitrary
  `cmap` — it draws solid lines, points (size honored, shape not), and a fixed
  built-in colour ramp.
- **`fill` / `fillalpha`** apply to `AREA` marks (a band, a filled region) only.

When a backend would drop something, it tells you:

```python
spec = ts.Lorenz().to_plot_spec().style(linestyle="dashed")
spec.render("threejs")   # warns: three.js does not honor linestyle
```

The warning is a single message per render — it also covers any unhonored
animation knobs and theme fields for the chosen backend — so degradation is
always *visible and named*, never silent. You can introspect a key's reach up
front:

```pycon
>>> sorted(ts.viz.STYLE_KEYS["linestyle"].honored_by)
['matplotlib', 'plotly']
```

For a fully portable figure (including three.js), prefer `color` / `linewidth` /
`markersize` / `alpha` / `zorder`, which every backend honors.

---

## Interaction with `filterwarnings = ["error"]`

This repo's own pytest policy is `filterwarnings = ["error"]` — every warning
becomes an exception — and a fair number of downstream projects run the same
strict policy. Under it, the **two new warning paths** the styling layer adds
turn into *errors*:

1. **An unknown style key.** `.style(unknown_key=...)` (or any `normalize_style`
   call that drops a key) emits a `VisualizationDegraded`. With warnings as
   errors, `.style(colour="red")` *raises* instead of warning-and-dropping.
2. **An unhonored knob at render time.** Rendering a spec that sets a knob the
   chosen backend cannot honor (e.g. `linestyle` on three.js) emits the
   consolidated `VisualizationDegraded` described above — which becomes an error
   under the strict policy.

There are **no false positives on a default render**: a spec that sets only
honored knobs (or no style at all) emits nothing, so the strict policy never
fires on an ordinary plot. Only specs that set an *unknown* key or an
*unhonored* knob are affected — exactly the cases the warning is meant to flag.

If a downstream test treats warnings as errors and you *want* the degraded path
(you know three.js drops `linestyle`, say), scope the warning rather than
loosening your global policy. The warning class is
`tsdynamics.viz.render.caps.VisualizationDegraded`:

```python
import warnings
from tsdynamics.viz.render.caps import VisualizationDegraded

# 1. Swallow it locally
with warnings.catch_warnings():
    warnings.simplefilter("ignore", VisualizationDegraded)
    spec.style(linestyle="dashed").render("threejs")
```

```python
# 2. Assert it in a test (pytest), which also consumes it so it can't escalate
import pytest

with pytest.warns(VisualizationDegraded):
    spec.style(colour="red")            # the typo path
```

```ini
# 3. Or down-grade just this class in pyproject.toml / pytest.ini
[tool.pytest.ini_options]
filterwarnings = [
    "error",
    "ignore::tsdynamics.viz.render.caps.VisualizationDegraded",
]
```

Prefer the narrowest scope that fits — `catch_warnings` / `pytest.warns` around
the one call, not a project-wide ignore — so a genuinely-unintended degradation
still surfaces everywhere else.

---

## Introspection

The whole styling surface is discoverable from `tsdynamics.viz`:

| Entry point | What it gives |
| --- | --- |
| `ts.viz.STYLE_KEYS` | the canonical vocabulary — `name → StyleKey` (each with `aliases`, `honored_by`, `validate`, `doc`) |
| `ts.viz.normalize_style` | the alias / validation / unknown-key choke point |
| `ts.viz.themes()` | sorted names of all registered themes |
| `ts.viz.get_theme()` / `ts.viz.set_theme()` | read / write the active global default |
| `ts.viz.register_theme()` | add a named theme |
| `ts.viz.Theme` | the theme dataclass |

```pycon
>>> sorted(ts.viz.STYLE_KEYS)
['alpha', 'cmap', 'color', 'fill', 'fillalpha', 'linestyle', 'linewidth', 'marker', 'markersize', 'zorder']
>>> key = ts.viz.STYLE_KEYS["linewidth"]
>>> key.aliases, key.doc
(('lw',), 'line width (pt)')
```

`ts.viz` is bound lazily — a plain `import tsdynamics` pulls in no plotting
library — so reach the styling API through the `ts.viz` namespace (it resolves
and caches on first access).
