# Golden docs-figure corpus

A small, committed set of the auto-generated per-system documentation figures,
frozen as a **golden corpus** with a **perceptual-hash drift checker**. This is
the **renderer oracle** for the docs-figure pipeline: the reference picture every
future renderer must reproduce.

## What is here

| File | Purpose |
|---|---|
| `figures/<System>.png` | The frozen golden render of one representative system. |
| `manifest.json` | Per-figure perceptual hash (`ahash`) and family. |
| `__init__.py` | The corpus API: `render_png`, `average_hash`, `regenerate`, … |
| `__main__.py` | The re-baseline CLI. |

The subset (`GOLDEN_SUBSET`) is chosen to cover **every render branch** in
[`docs/_tooling/figures.py`](../figures.py):

- `Lorenz`, `Rossler` — 3-D ODE phase portraits
- `Lorenz96` — the space-time `imshow` branch
- `Henon` — 2-D map scatter
- `Logistic` — 1-D map return-map scatter
- `MackeyGlass` — DDE time series + delay-embedding view

## How the drift check works

A docs figure is a Matplotlib raster of a numerically integrated trajectory.
Exact PNG bytes are brittle — a Matplotlib/FreeType bump, a platform font, or a
roundoff-level engine difference all change the bytes while the *picture* (the
attractor's shape) is visually identical. So the checker compares a **16×16
average hash (aHash)** by Hamming distance, with a small tolerance
(`DEFAULT_TOLERANCE`, ~6 % of the 256-bit budget). That is robust to nuisance
jitter yet still trips on a genuine structural change (a system integrating to a
different attractor, a renderer dropping an axis, a regime change in the
dynamics).

`tests/test_docs_figures_golden.py` (marked `slow`) re-renders the subset and
asserts each fresh hash is within tolerance of its golden hash.

## The oracle for the DOCS-ENG streams

This corpus is the structural-equivalence oracle the downstream figure work
re-baselines against:

- **DOCS-ENG-ENGINEFIG** — re-renders the ODE figures through the shipped engine
  (replacing today's SciPy `_rhs_numeric` path) and must regenerate this corpus,
  proving the engine render is the same picture.
- **DOCS-ENG-PLOTSEAM** — introduces a `to_plot_spec()`-driven figure generator
  whose output is validated against this same golden set.

## Re-baselining (after an intentional change)

Run, from the repo root, with the `docs` dependency group active (it provides
Matplotlib):

```bash
uv run --group docs python docs/_tooling/golden/__main__.py
```

This rewrites every `figures/<System>.png` and `manifest.json`. **Review the
regenerated PNGs by eye** to confirm the change is intended, then commit them.
