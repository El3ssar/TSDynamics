# Viz Styling Overhaul — API Design Contract (BUILD SOURCE OF TRUTH)

> This file is the **single source of truth** for the visualization styling/theming
> redesign. Every build agent binds to the exact names, fields, and signatures here.
> It is a *build artifact* — the docs agent folds the user-facing parts into
> `docs/` and this file is DELETED before the PR lands. Do not ship it.

Branch: `viz-styling-overhaul`. Worktree: `.claude/worktrees/viz-styling-overhaul/`.
Decisions (from the user): **clean redesign allowed (breaking, major bump)**;
**full theme system incl. a global default**; **fold in the related cross-backend bugs**.

The 3 *visual* backends are **matplotlib, plotly, three.js**; **json** is a
data-export backend that must faithfully **serialize** every new field (it draws
nothing). "Honored by all 3 backends" ⇒ mpl + plotly + threejs render it (or
honestly warn); json round-trips it.

---

## 0. Goals

Make the look of every plot **fully, discoverably, consistently customizable**:

1. A **canonical, documented, validated, introspectable** per-layer style
   vocabulary (no more free-form dict that silently swallows typos).
2. A **figure-level `Theme`** (palette, font, background, grid, line defaults) with
   **named built-ins** and an **optional global default**.
3. **Fluent tweak methods** for every knob (`.style/.recolor/.theme/.palette/.grid/.font/.background/.size`).
4. Every knob **honored consistently across mpl/plotly/threejs**, serialized by json,
   and where a backend *cannot* honor a knob it **warns** (no silent drops).
5. Public introspection: `ts.viz.STYLE_KEYS`, `ts.viz.themes()`, `ts.viz.get_theme/set_theme`.

Keep the sound parts of the IR (PlotKind, PlotSpec/Layer structure, animation as an
orthogonal modifier). The redesign touches the **presentation surface** only.

---

## 1. New module: `src/tsdynamics/viz/style.py`

Pure data + helpers. **No import of `spec.py`** (spec.py imports *from* style.py — keep
the dependency one-way to avoid cycles). Public objects:

### 1.1 `StyleKey` + `STYLE_KEYS`

```python
@dataclass(frozen=True)
class StyleKey:
    name: str                                   # canonical key
    aliases: tuple[str, ...] = ()               # accepted spellings -> name
    honored_by: frozenset[str] = frozenset({"matplotlib", "plotly", "threejs"})
    validate: Callable[[Any], Any] | None = None  # coerce/validate a value (raises ValueError)
    doc: str = ""
```

`STYLE_KEYS: dict[str, StyleKey]` — the canonical per-layer vocabulary. Define exactly
these keys (canonical name → aliases, honored_by, doc):

| canonical    | aliases               | honored_by                  | meaning |
|--------------|-----------------------|-----------------------------|---------|
| `color`      | `c`                   | mpl, plotly, threejs        | line/marker/fill color (CSS name, hex, or rgb tuple) |
| `linewidth`  | `lw`                  | mpl, plotly, threejs        | line width (pt) |
| `linestyle`  | `ls`                  | mpl, plotly                 | one of `solid,dashed,dotted,dashdot` (also accept `- -- : -.`); threejs does **not** honor (warns) |
| `marker`     | —                     | mpl, plotly                 | one of `circle,square,triangle,diamond,cross,x,star,none` (also accept mpl single-chars `o s ^ D + x *`); threejs draws points only (size honored, shape not) |
| `markersize` | `ms`, `s`             | mpl, plotly, threejs        | marker size (pt) |
| `alpha`      | `opacity`             | mpl, plotly, threejs        | 0..1 opacity |
| `cmap`       | `colormap`, `colorscale` | mpl, plotly, threejs     | colormap name for the `c`/image channel |
| `fill`       | —                     | mpl, plotly                 | bool: fill area under/between |
| `fillalpha`  | —                     | mpl, plotly                 | 0..1 fill opacity |
| `zorder`     | —                     | mpl, plotly                 | int draw order; threejs maps to renderOrder (honored) → put threejs in honored_by for zorder |

> Pick the honored_by sets above as written. They drive the honest-degradation
> warnings (§5). When in doubt about a backend, **include it and have the renderer
> actually honor it**; only exclude when honoring is genuinely impossible.

### 1.2 `normalize_style`

```python
def normalize_style(style: Mapping[str, Any], *, warn: bool = True) -> dict[str, Any]:
    """Canonicalize aliases -> canonical names, validate values, drop unknown keys.

    Unknown keys are dropped; if warn, emit ONE VisualizationDegraded naming them.
    Returns a new dict (never mutates the input)."""
```

Renderers and the `.style()` tweak both pass user style through this. It is the
single choke point that kills the "typo silently ignored" problem.

### 1.3 `Theme`

```python
@dataclass
class Theme:
    name: str = "default"
    palette: tuple[str, ...] = DEFAULT_PALETTE   # color cycle for auto-colored layers
    background: str | None = None                # figure/axes facecolor
    foreground: str | None = None                # default ink: text/axes/ticks color
    font_family: str | None = None
    font_size: float | None = None
    title_size: float | None = None
    grid: bool = False                           # default gridline visibility
    grid_color: str | None = None
    grid_alpha: float | None = None
    line_width: float | None = None              # default line width
    marker_size: float | None = None             # default marker size

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, d) -> "Theme": ...         # tolerate missing keys (defaults)
    def merged(self, **overrides: Any) -> "Theme": # return a copy with overrides applied
```

### 1.4 Built-in themes + registry + global default

```python
THEMES: dict[str, Theme]          # name -> Theme; built-ins below
def register_theme(theme: Theme) -> None
def get_theme(name: str | None = None) -> Theme   # None -> the active global default
def set_theme(theme: str | Theme) -> None         # set the global default (by name or instance)
def themes() -> list[str]                          # sorted built-in + registered names
def resolve_palette(p: str | Sequence[str]) -> tuple[str, ...]  # named palette or explicit list
```

Built-in themes (define all four, tasteful, distinct):
- `default` — light, thin grey grid off by default, matplotlib-ish palette (a clean
  10-color qualitative cycle), sans-serif.
- `dark` — dark background (`#11131a`-ish), light ink, brighter palette.
- `minimal` — no grid, no top/right "chrome" feel, muted palette, slightly thinner lines.
- `publication` — serif font, larger label/title sizes, grid off, a colorblind-safe palette.

Global default: module-level `_ACTIVE = "default"`; `set_theme`/`get_theme` read/write it.
`get_theme(None)` returns `THEMES[_ACTIVE]`. This is the ONLY mutable global state in viz;
keep it isolated in `style.py` and reset-safe (a `set_theme("default")` returns to baseline).

---

## 2. `src/tsdynamics/viz/spec.py` changes

`spec.py` imports `Theme`, `normalize_style`, `get_theme` from `.style`.

### 2.1 Enriched `Axis` (ADD fields; keep existing)

Add: `grid: bool | None = None`, `color: str | None = None`,
`label_size: float | None = None`, `tick_size: float | None = None`,
`tick_rotation: float | None = None`. Update `to_dict`/`from_dict` (from_dict uses
`.get` so old payloads still load).

### 2.2 Enriched `Legend`

Add: `font_size: float | None = None`, `ncol: int = 1`, `frame: bool = True`.
Update dict round-trip.

### 2.3 Enriched `Colorbar`

Add: `label_size: float | None = None`. Update dict round-trip. (Keep the rest.)

### 2.4 `PlotSpec` gains a `theme` field

Add `theme: Theme | None = None` to the dataclass (after `animation`). `None` ⇒ resolve
to the global default at render time (renderers call `get_theme(None)` when `spec.theme is None`).
Update `to_dict` (serialize `theme.to_dict()` or `None`) and `from_dict`
(`Theme.from_dict` when present). **figsize/dpi are NOT on Theme** — they live in
`meta` (`meta["figsize"]`, `meta["dpi"]`), set by `.size()`.

### 2.5 New / changed fluent tweak methods (all mutate + return `self`)

```python
def style(self, *, layer: int | None = None, axes: bool | None = None, **kw) -> PlotSpec:
    # ENHANCED: route **kw through normalize_style() before merging into the layer(s).
    # Keep the `axes=` (axes visibility) and `layer=` behavior.

def recolor(self, *colors: str, layer: int | None = None) -> PlotSpec:
    # layer is None: assign colors[i % len] to layer i across all layers.
    # layer is int: set that one layer's color to colors[0].

def theme(self, theme: str | Theme | None = None, /, **overrides) -> PlotSpec:
    # Resolve `theme` (name -> THEMES, Theme -> itself, None -> active default),
    # apply **overrides via Theme.merged(), store on self.theme.

def palette(self, colors: str | Sequence[str]) -> PlotSpec:
    # Convenience: self.theme = (self.theme or get_theme()).merged(palette=resolve_palette(colors))

def grid(self, show: bool = True, *, axis: Literal["x","y","both"] = "both",
         color: str | None = None, alpha: float | None = None) -> PlotSpec:
    # Set self.x.grid / self.y.grid (per `axis`); set grid_color/alpha on those axes if given.

def font(self, family: str | None = None, size: float | None = None) -> PlotSpec:
    # self.theme = (self.theme or get_theme()).merged(font_family=..., font_size=...)

def background(self, color: str) -> PlotSpec:
    # self.theme = (self.theme or get_theme()).merged(background=color)

def size(self, width: float | None = None, height: float | None = None,
         dpi: float | None = None) -> PlotSpec:
    # store into self.meta["figsize"]=(width,height) (only set provided dims) and meta["dpi"].
```

Keep all existing tweaks (`relabel/rescale/limits/ticks/colorize/animate/trail/head/camera/clock`).

---

## 3. Renderer contracts (mpl / plotly / threejs / json)

Each visual renderer implements a small shared shape:

1. **Resolve the theme**: `theme = spec.theme or get_theme(None)`. Apply theme-level
   presentation FIRST (background/facecolor, font family+size, default grid, palette as
   the color cycle for layers that carry no explicit `color`, default line_width/marker_size).
2. **Per-layer style**: `canon = normalize_style(layer.style, warn=False)` (the dispatcher
   already emitted the consolidated warning — §5 — so renderers pass `warn=False`), then
   translate canonical keys to backend kwargs and apply, OVERRIDING theme defaults.
3. **Honor the enriched Axis/Legend/Colorbar fields** (grid, label_size, tick_size,
   tick_rotation, ncol, frame, font_size, tickformat).
4. A backend **must not silently drop** a knob it lists in `honored_by`; if it cannot,
   remove itself from that key's `honored_by` (so §5 warns) — do not pretend.

Backend-specific notes:
- **mpl** (`render/mpl/`): the reference backend — honor *everything*, including
  `tickformat` (currently ignored — FIX: apply via `FormatStrFormatter`/`FuncFormatter`)
  and `zorder`. Apply theme font via `rcParams`-local context or per-artist kwargs (no
  global rcParams mutation — keep it figure-local). Background → `fig`/`ax` facecolor.
- **plotly** (`render/plotly/`): map canonical → plotly (`line.width`, `line.dash`,
  `marker.symbol`, `marker.size`, `opacity`, `colorscale`). `marker` unknown → keep the
  `_MARKER_MAP.get(..., "circle")` behavior BUT the dispatcher warns (§5). Theme
  background → `paper_bgcolor`/`plot_bgcolor`; font → `layout.font`; grid → axis
  `showgrid`. Honor `Legend.ncol`/`frame`/`font_size` where plotly allows.
- **threejs** (`render/threejs/` + `docs/_static/tsdyn-threejs-loader.js`): serialize
  `color/linewidth/markersize/alpha/cmap/zorder` into the material/metadata; serialize
  the resolved `theme` into a `metadata.theme` block (background, palette, foreground) so
  the loader can apply scene background + colors. `linestyle`/`marker`-shape are NOT
  honored (excluded from their honored_by) — the loader ignores them. Keep the JS loader
  in lockstep: any field the exporter emits, the loader must read or it is dead weight.
- **json** (`render/json.py`, `export.py`): purely serialize the full spec incl. `theme`
  and all new fields; round-trip fidelity is the only requirement.

---

## 4. Front door wiring (`data/trajectory.py`, `families/_plottable.py`, `viz/compose.py`, `viz/producers.py`, `viz/__init__.py`)

- `to_plot_spec(...)` / `.plot(...)`: nothing changes structurally, but the spec they
  return must accept the new tweaks (they already return a `PlotSpec`). Ensure
  `Trajectory.plot()` still forwards the spec-shaping kwargs correctly.
- **FOLD-IN BUG (Animation in-place mutation)**: in the SPATIAL_FIELD branch
  (`data/trajectory.py`, the `animate=Animation(...)` handling near the field producer),
  do **not** mutate the passed `Animation`; use `dataclasses.replace(anim, mode="frames", ...)`.
- `viz/compose.py` (`plot`): when composing, the composite spec may carry a top-level
  `theme`; panels inherit the composite theme unless they set their own (renderers resolve
  per-panel: `panel.theme or composite.theme or get_theme()`).
- **`viz/__init__.py`**: export `Theme, set_theme, get_theme, themes, register_theme,
  STYLE_KEYS` (add to `__all__` and the lazy binding). FIX the stale module docstring
  ("ships no rendering backend" → "imports no plot library at import time; four
  renderers self-register lazily"). `STYLE_KEYS` is exposed as the public introspection
  mapping (`name -> {aliases, honored_by, doc}`); you may expose a read-only view.

---

## 5. FOLD-IN BUGS (cross-backend correctness)

1. **Default-backend flip** (`render/mpl/__init__.py` `_reseat_last` + `render/__init__.py`
   default selection): make the default drawing backend **deterministic = matplotlib on
   the very first registration**, idempotent across repeated `register_builtin_renderers`
   calls. Align the `select_renderer` docstring to say matplotlib is the default. Add a
   regression test: render twice with no explicit backend → identical backend type.
2. **Silent knob/mark degradation → honest negotiation** (centralize in `render/caps.py`
   + `render/__init__.py`): add
   `def style_honoring_gaps(spec, backend_name) -> list[str]` that collects every
   per-layer canonical style key, every `Animation` knob (spin/clock/trail_fade/elev-azim/…),
   and every theme/axis presentation field the chosen backend does **not** honor. The
   dispatcher emits **ONE consolidated `VisualizationDegraded`** per render naming the
   dropped knobs (e.g. "plotly: ignoring camera spin, clock; threejs: ignoring linestyle").
   This is the systemic fix — renderers then run with `warn=False`.
3. **Animation in-place mutation** — see §4 (`dataclasses.replace`).
4. (Opportunistic, if cheap) **mpl `tickformat` ignored** — honor it (§3 mpl). **plotly
   unknown-marker → circle** — now covered by the §5.2 warning.

> Out of scope for THIS PR (leave as follow-ups, do not attempt): the plotly camera
> `elev/azim`→`eye` conversion (real feature work), the three.js empty-payload-on-image
> redesign (just make it WARN via §5.2, don't fix the geometry), and the test-suite
> reorg of `gapfill_*`. If `elev/azim` is trivially warnable, just include it in §5.2's gap list.

---

## 6. Tests (`tests/`) — breaking changes ripple here

The test agent migrates/extends the viz tests for the new API and adds coverage:
- Round-trip EVERY new field through `to_dict`/`from_dict` (Theme, enriched Axis/Legend/Colorbar, `PlotSpec.theme`).
- `normalize_style`: aliases resolve (`lw`→`linewidth`, `c`→`color`, `s`→`markersize`),
  unknown key warns + is dropped, validators reject bad values.
- Each new tweak method mutates + returns self + chains.
- Theme registry: built-ins exist, `set_theme`/`get_theme` round-trip, `set_theme("default")` resets.
- **Per-backend honoring (structural, not "doesn't raise")**: render a spec carrying a
  full style/theme and assert the artifact reflects it — mpl: artist props (`get_color`,
  `get_linewidth`, `get_linestyle`, axes facecolor, grid on); plotly: trace/layout attrs
  (`line.dash`, `marker.size`, `paper_bgcolor`, `font.family`); threejs: the serialized
  material/`metadata.theme`; json: dict equality.
- **Honest negotiation**: a spec with a knob a backend cannot honor emits exactly one
  `VisualizationDegraded` naming it; a spec with no such knob emits none.
- Backend-flip regression (render twice → same backend type).
- Keep/repair existing viz tests; update any that asserted the old free-form behavior.

Canonical commands (run FROM the worktree root, NO `uv run`):
```
VPY=/home/elessar/Projects/TSDynamics/.venv/bin
$VPY/ruff format src/ tests/ && $VPY/ruff check src/ tests/
$VPY/mypy --strict src/tsdynamics
PYTHONPATH="$PWD/src" MPLBACKEND=Agg $VPY/python -m pytest tests/test_viz*.py tests/test_plotspec*.py tests/test_to_plot_spec.py tests/test_plot_accessor_kinds.py --no-cov -p no:cacheprovider -q
```

`mypy --strict src/tsdynamics` MUST stay green (CI gate). `ruff` MUST be clean.

---

## 7. Naming & conventions (frozen)

- Public viz names: `Theme`, `set_theme`, `get_theme`, `themes`, `register_theme`,
  `STYLE_KEYS`, `normalize_style` (the last two reachable via `tsdynamics.viz`).
- Tweak methods: `style, recolor, theme, palette, grid, font, background, size`.
- Canonical style keys: exactly the §1.1 table (with the listed aliases).
- NumPy docstrings; cite no competitor software; ruff line length 100; `mypy --strict` clean.
- This is a **breaking** change → the PR title is `feat(viz)!: ...` (major bump).
