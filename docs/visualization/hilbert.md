---
description: Lay a long 1-D record on a space-filling curve — the four hilbert* plot transforms, the optional hilbertplot extra, the pixel-to-sample map, and the locality trade-off measured rather than asserted.
---

<span class="ts-kicker">Visualization</span>

# Space-filling curves

A long scalar record is a bad plot. Twenty thousand samples of `x(t)` become a
strip a few hundred pixels wide by the time it reaches a page, so structure at
the scale of a few hundred samples — the thing you are usually looking for — is
smeared into ink.

A **space-filling curve** fixes the aspect ratio instead of the data. Walk a
curve that visits every cell of a square grid exactly once, drop sample `i` into
the `i`-th cell it visits, and the record becomes an image whose *locality* is
inherited from the curve: on a Hilbert curve, samples that are close in the
record land close in the picture. Long-range texture becomes visible texture.

This is the Hilbert plot of Estévez-Rams *et al.* (2015), and TSDynamics ships
four views of it as ordinary [plot transforms](plotting.md) — so they compose,
theme, style and export exactly like a phase portrait.

| transform | what it draws |
|---|---|
| `hilbert` | the data image itself |
| `hilbert_fourier` | its centred 2-D power spectrum — periodicity as symmetry |
| `hilbert_difference` | the locality-loss field: **where the curve's locality breaks** |
| `hilbert_labels` | the visit order — the curve's own path, drawn as an image |

---

## The one-liner

```python
import tsdynamics as ts

lorenz = ts.systems.Lorenz().run(final_time=200.0, dt=0.01, ic=[1, 1, 1])

ts.plot(lorenz, "hilbert", components="x")
```

That is 20,001 samples on a 142 × 142 grid. Everything else is a keyword:

```python
ts.plot(lorenz, "hilbert", components="x", curve="Moore")        # another of the 40 curves
ts.plot(lorenz, "hilbert", components="x", granularity=16)       # coarsen first
ts.plot(lorenz, "hilbert", components="x", fit="truncate")       # drop the tail, no padding
ts.plot(lorenz, "hilbert", components="x", primitive="surface3d")  # the image as a relief
```

The subject can be a **`Trajectory`**, a **bare array**, or a **system** (which
is integrated for you, with the choice recorded in `spec.meta`). A bare array is
the interesting case, because it is how every other 1-D record in the library
gets here:

```python
# a windowed RQA measure over time
windowed = ts.analysis.windowed_rqa(lorenz.y[::20], window=200, step=4, recurrence_rate=0.1)
ts.plot(windowed.determinism, "hilbert")

# an inter-event series: the successive return times to a level set
returns = ts.viz.geometry(lorenz, "return_time", components="z").meta["return_times"]
ts.plot(returns, "hilbert")

# a symbolic sequence, encoded as integers
symbols = (lorenz.y[:, 0] > 0).astype(float)
ts.plot(symbols, "hilbert", curve="Hilbert")
```

---

## Installing the forty curves

The real Hilbert-type curves come from **[hilbertplot](https://github.com/El3ssar/hilbertplot)**,
the reference implementation of all forty two-dimensional Hilbert curves proved
to exist by Estévez-Rams *et al.* (2017):

```bash
pip install "tsdynamics[hilbert]"
```

Only its **numpy core** is used. TSDynamics deliberately never installs or
touches hilbertplot's own `[plot]` extra: a Hilbert plot must arrive as a
`Plot` `IMAGE` layer like every other picture in this library, or it would
bypass the [`Theme`](styling.md), [`STYLE_KEYS`](styling.md) and three of the
four [backends](backends.md).

**Without the extra**, asking for a Hilbert curve raises and names it:

```pycon
>>> ts.plot(lorenz, "hilbert", components="x")
VisualizationNotInstalled: curve='Hilbert' is one of the forty Hilbert-type curves,
which come from the optional 'hilbertplot' package: install it with
`pip install tsdynamics[hilbert]` (or `pip install hilbertplot`).
It is not substituted silently, because the locality of the Hilbert curve is the
whole claim of this plot. If you want a picture now, ask for one of the
dependency-free orderings explicitly: curve=['morton', 'rowmajor', 'snake'].
```

It is **never** silently replaced, because the substitute would be a different
plot making the same claim (see [the measurement](#the-locality-trade-off)).
Three dependency-free orderings do ship in-tree, and asking for one explicitly
always works:

| ordering | path |
|---|---|
| `"rowmajor"` | left to right, bottom to top — no locality, the honest baseline |
| `"snake"` | boustrophedon: row-major with alternate rows reversed |
| `"morton"` | Z-order (bit-interleaved); needs a power-of-two side |

```python
ts.viz.transforms.hilbert.curve_names()
# ('morton', 'rowmajor', 'snake')                  # without the extra
# ('morton', 'rowmajor', 'snake', 'Hilbert', ...)  # with it: 43 in all
```

---

## Sizing the grid: `fit`

`n` samples do not generally make a square. Three rules, and TSDynamics always
picks one **explicitly** — it never leaves the choice to another package:

| `fit` | side | keeps every sample? |
|---|---|---|
| `"square"` | `ceil(sqrt(n))` — the tight grid | yes (trailing cells are padding) |
| `"pad"` | the smallest `2**k` with `4**k >= n` | yes (padding, sometimes a lot) |
| `"truncate"` | the largest `2**k` with `4**k <= n` | **no** — the tail is dropped |

The default is `"square"` when the curve can fill a grid of any side (the eight
*generalizing* curves, plus `rowmajor` and `snake`) and `"pad"` otherwise.

Padding cells are `NaN`, which every backend renders transparent, and their count
is reported:

```pycon
>>> ts.viz.geometry(lorenz, "hilbert", components="x", curve="Moore").meta["n_padding_cells"]
45535
```

45,535 of 65,536 cells are padding, because `Moore` is not a generalizing curve
and 20,001 samples must be padded up to a 256 × 256 grid. That is not a bug and
it is not hidden — it is the honest cost of that curve at that length, and
`fit="truncate"` (a 128 × 128 grid, 3,617 samples dropped) is the other trade.

---

## `granularity`: coarsening, not decimating

The *l*-granularity transform of Estévez-Rams *et al.* replaces each block of
`l` consecutive samples with `l` copies of the block mean. The record keeps its
length — so the grid, and the meaning of every pixel, is **unchanged** — and only
the sample-scale variation is flattened, leaving the long-range texture:

```python
ts.plot(lorenz, "hilbert", components="x", granularity=16)
```

---

## Reading a Hilbert plot honestly

### The locality trade-off

Every space-filling curve trades locality somewhere. Measured as the
record-distance between every pair of 8-adjacent cells of a 128 × 128 grid
(this table is pinned by `tests/test_viz_hilbert.py`):

| ordering | median gap | mean gap | P(gap ≤ 8) |
|---|---|---|---|
| Hilbert | **3.0** | 114.3 | **0.683** |
| Morton (Z) | 5.0 | 94.5 | 0.656 |
| snake | 85.5 | **96.1** | 0.275 |
| row-major | 127.5 | **96.1** | 0.251 |

The median says Hilbert is 43× more local than row-major. The mean says it is
19% *worse*. **Both are true**: Hilbert buys that median by making its few worst
jumps much longer. Quote one of these numbers and you are advertising, not
measuring.

### Where the locality breaks

`hilbert_difference` maps it. Bright ridges are the curve's locality barriers:
two pixels touching across a ridge are far apart in the record, so a texture that
straddles one is an artefact of the layout, not of your data.

```python
ts.plot(
    ts.plot(lorenz, "hilbert_difference", components="x", curve="Hilbert"),
    ts.plot(lorenz, "hilbert_difference", components="x", curve="rowmajor"),
    layout="row",
)
```

The Hilbert panel is a recursive tree of ridges. The row-major panel is a
**featureless plateau** — every vertical neighbour is exactly one grid-width
apart in the record, which is the precise sense in which a raster layout has no
locality at all.

The field depends only on the curve and the grid size, never on your values.

### Which sample is this pixel?

A wrong pixel-to-sample map is the worst failure available here: the image still
renders and only the annotation lies. So the map comes from public API and is
checked on every call.

```python
index = ts.viz.transforms.hilbert.sample_index_map(lorenz, components="x")

t_of_pixel = np.full(index.shape, np.nan)     # same shape as the image
inside = index < lorenz.t.size                # everything else is padding
t_of_pixel[inside] = lorenz.t[index[inside]]
```

The map is the curve's raw visit order — the same array `hilbert_labels` draws —
so a cell whose index is `>= len(series)` is a padding cell the record never
reached. Mask it, as above, rather than indexing blindly. The map is kept **off**
`spec.meta` by default — it is exactly as large as the image, and `meta` is
serialized by [`ts.viz.to_json`](backends.md) — but `with_sample_index=True`
puts it there when you want it travelling with the spec.

Three independent locks make it trustworthy:

1. **The fit is always explicit**, so nothing depends on how another package
   resolves `fit="auto"`.
2. **The map is read, not reconstructed** — `hilbertplot`'s public `label_map()`
   is built from the *same* `(granularity, order, fit)` call as `image()`, so the
   two are consistent by construction rather than by assumption.
3. **The result is verified**: the returned grid's side must equal the side
   `grid_side` computed, and a
   sample of cells must carry exactly the values the map names. A disagreement
   raises `BackendError` instead of drawing.

---

## Composing and exporting

A `hilbert*` spec is an ordinary `Plot` in the `grid2` frame, so everything
else in the visualization layer applies:

```python
image = ts.plot(lorenz, "hilbert", components="x")
spectrum = ts.plot(lorenz, "hilbert_fourier", components="x")

ts.plot(image, spectrum, layout="row").save("hilbert.pdf")
image.render("plotly")            # interactive heatmap
image.style(cmap="magma")   # the usual style vocabulary
```

The two do **not** overlay: the image lives on `(cell x, cell y)` and the Fourier
map on `(k_x, k_y)`, and the [frame check](conventions.md) refuses to put two
different coordinate systems on one set of axes.

`hilbert_fourier` and `hilbert_difference` additionally accept
`primitive="contour"`, and `hilbert` / `hilbert_fourier` / `hilbert_difference`
accept `primitive="surface3d"`:

```pycon
>>> ts.viz.compatibility("hilbert_fourier")
('contour', 'image*', 'surface3d')
```

---

## References

- Estévez-Rams, E., Lora Serrano, R., Aragón Fernández, B. & Brito Reyes, I.
  (2015). "Visualizing long vectors of measurements by use of the Hilbert
  curve." *Computer Physics Communications* **197**, 118–128.
- Estévez-Rams, E., Pérez-Cruz, J. A. & Rodríguez Hoyos, O. (2017). "Hilbert
  curves in two dimensions." *Revista Cubana de Física* **34**(1), 9–14.
