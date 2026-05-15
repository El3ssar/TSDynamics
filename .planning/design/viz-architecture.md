# Visualization architecture

This document is the contract for everything under `tsdynamics.viz`. The goal is a
plotting layer that:

- Lets a user say `phase_portrait(traj)` and get a publication-ready figure.
- Lets a power user compose transforms, drop down to the backend figure, and
  customise freely.
- Lets a contributor add a new analysis output, a new transform, or a new backend
  without touching anything else.

## The pipeline

```
analysis primitive (Trajectory, BifurcationData, BasinMap, …)
        ↓
   DataSpec(kind, data, meta)        ← canonical, backend-agnostic
        ↓
   [Transform, Transform, …]         ← pure DataSpec → DataSpec
        ↓
   Plotter(backend, kind)            ← registered renderer
        ↓
   backend figure object             ← returned to user
```

Two-axis separation: **what the data is** (`kind`) is orthogonal to **how it's
rendered** (`backend`). The registry is keyed by both.

## Core types

### `DataSpec`

A frozen dataclass:

```python
@dataclass(frozen=True)
class DataSpec:
    kind: str                    # canonical name; see registry below
    data: Mapping[str, Any]      # kind-specific payload; schema documented per kind
    meta: Mapping[str, Any]      # axis labels, units, param values, system ref, …
```

- `data` is mapping-style for forward compatibility (adding a field doesn't break old
  consumers).
- `meta` is purely cosmetic / informational — plotters use it for axis labels,
  legends, titles. Removing meta must never change the rendered geometry.

### Canonical `kind` values (shipped from V1 onwards)

| Kind | `data` schema | Source |
|---|---|---|
| `timeseries` | `t: (T,), y: (T,) or (T, d), labels: list[str]` | M1 |
| `phase_portrait_2d` | `x: (T,), y: (T,)` | V1 |
| `phase_portrait_3d` | `x, y, z: (T,)` | V1 |
| `poincare` | `points: (N, d-1), section_meta: dict` | M2 + V2 |
| `bifurcation` | `param: (P,), observable: (P, K) ragged, kind: scatter\|density` | M4 + V2 |
| `basin_field` | `grid: (G1, G2), labels: (G1, G2) int, palette: dict` | M12 + V3 |
| `recurrence_matrix` | `R: (T, T) bool` | M9 + V3 |
| `spectrum` | `freq: (F,), power: (F,)` | M7 + V3 |
| `rqa_lines` | `scalars: dict[str, float]` | M9 |

Adding a new kind is a one-liner in the registry plus a doc entry in this table.

### `Transform`

```python
Transform = Callable[[DataSpec], DataSpec]
```

Library transforms shipped from V1:

- `Decimate(n)` — drop to every n-th point. For `timeseries` and `phase_portrait_*`.
- `Project(dims)` — select subset of state-space dimensions.
- `Smooth(window, kind="savgol")` — smooth `timeseries`.
- `Log(axis="y")` — flag for the plotter to use log scaling on an axis.
- `ColorBy(field)` — attach a per-point color computed from another DataSpec or from
  `meta` (V4 adds the lyapunov / time / density variants).
- `Normalize` / `Clip` — value ranges.

Transforms are *pure*: same input → same output. They never call into a backend.

### `Plotter`

A registered function:

```python
@register_plotter(backend="matplotlib", kind="phase_portrait_3d")
def _plot_phase_portrait_3d(spec: DataSpec, ax_or_fig, **style) -> Any:
    ...
```

The first argument is the `DataSpec`. The second is either an `Axes` (single panel)
or a `Figure` (multi-panel composites; rare). The return is the backend object
created so the caller can keep customising (e.g., an mpl `Line3D` collection).

### Registry

```python
class PlotterRegistry:
    def register(self, backend: str, kind: str, fn): ...
    def get(self, backend: str, kind: str) -> Plotter: ...
    def kinds(self, backend: str) -> set[str]: ...
    def backends(self) -> set[str]: ...
```

Module-level singleton `PLOTTERS`. The decorator `register_plotter` writes to it at
import time. Same pattern for `TRANSFORMS` (less critical, but useful for
serialisable pipelines).

### `Figure` (high-level composer)

```python
def make_figure(
    specs: Sequence[Tuple[DataSpec, list[Transform], dict]],
    *,
    backend: str = "matplotlib",
    layout: str | None = None,
    fig=None,
) -> Any:
    ...
```

Iterates the list, applies transforms, looks up the plotter for `(backend, kind)`,
renders, returns the backend Figure. The user can either consume the returned
Figure directly or call lower-level convenience wrappers.

## The convenience surface

`tsdynamics.viz` exports one shell function per common task. Each shell:

1. Pulls the right analysis primitive from `tsdynamics.analysis` (or the trajectory
   if it's already that).
2. Builds a single-element `specs` list with sensible defaults.
3. Calls `make_figure`.
4. Returns the mpl Figure/Axes.

```python
def phase_portrait(traj, dims=(0, 1, 2), *, ax=None, **style): ...
def time_series(traj, dims=None, *, ax=None, **style): ...
def poincare(traj, section, *, ax=None, **style): ...
def bifurcation(system, param, grid, observable="poincare", *, ax=None, **style): ...
def basin_map(system, grid_shape, *, axes=(0, 1), ax=None, **style): ...
def recurrence_plot(traj, *, threshold=None, ax=None, **style): ...
def spectrum(traj, *, dim=0, method="welch", ax=None, **style): ...
```

Every shell follows the same conventions: returns the backend object, accepts `ax=`
for embedding into existing figures, forwards `**style` to the plotter.

## How users extend it

Three contracts:

1. **New `kind`.** Decide a name and `data` schema. Register at least one plotter for
   it. Document the schema in `docs/viz/kinds.md`.

2. **New `Transform`.** Write a function with the `Transform` signature. No registration
   required (transforms are just callables). Optionally register a name under
   `TRANSFORMS` for serialisable pipelines.

3. **New backend.** Implement a function with the plotter signature for each `kind`
   the backend supports. Register with `register_plotter(backend="myviz", kind="...")`.
   Backends do not need to support every kind — partial coverage is fine; the
   registry raises `PlotterNotFound` with a helpful suggestion.

V6 ships a tutorial walking through each of these.

## Out of scope

- **Interactive widgets.** The matplotlib path stays static. Interactivity lands when
  the plotly backend arrives (V5).
- **Animation.** Future track. The DataSpec schema already supports it (a
  `timeseries` with extra `t` dim is enough); plotters just don't render frames yet.
- **Subplot grid layout DSL.** `make_figure` accepts `layout=` but the V1 impl only
  supports `None` (single axes) and `"grid"` (auto m×n). Anything fancier is user-
  side.

## Open questions to revisit when V1 lands

- Should `DataSpec` be Pydantic-validated or stay plain dataclass? Lean plain
  dataclass for V1; revisit if user-extension errors get confusing.
- Should `meta` carry a `provenance` field linking back to the system + params?
  Useful for cache keys and reproducibility. Probably yes from V1.
- Should `Transform` chains be serialisable to a string (so a CLI can replay a
  pipeline)? Defer to V6.
