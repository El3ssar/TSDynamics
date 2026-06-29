# `benchmarks/` — cross-library dynamical-systems benchmark

A head-to-head comparison of **TSDynamics** against the canonical pre-existing
libraries for integrating and analysing dynamical systems, on a shared set of
classic tasks: short/long ODE integration, integration accuracy, the Lyapunov
family (full spectrum, maximal exponent, maximal exponent from a time series),
correlation dimension, the logistic-map bifurcation diagram, basins of
attraction, fixed points, and the Poincaré section.

It measures **speed** (every task) and, where a task has a ground truth,
**precision** (the estimate and its deviation from the literature value). A
library that does not provide a capability — or that does not install in this
environment — leaves that cell **blank**, exactly as requested.

> This folder is distinct from the repo's existing `benches/`, which is the
> internal Rust-engine performance-regression harness. `benchmarks/` is the
> *external* cross-library comparison.

---

## Libraries compared

| Library | Language | What it contributes | Status here |
|---|---|---|---|
| **TSDynamics** (`interp` + `jit`) | Python + Rust | the library under test — every task | ✅ both engine backends |
| **DynamicalSystems.jl** | Julia | the most complete competitor — every task | ✅ v3.x (ChaosTools / Attractors) |
| **SciPy** | Python | the integration baseline (`solve_ivp`), fixed points (`fsolve`), Poincaré (events) | ✅ |
| **pynamical** | Python (numba) | logistic-map bifurcation diagram | ✅ |
| **dysts** | Python | GilpinLab chaotic-systems catalogue + SciPy integrator; correlation dimension (`gp_dim`), DFA | ✅ v0.96 |
| **nolds** | Python | from-data correlation dimension, Rosenstein Lyapunov, sample entropy, DFA, Hurst | ✅ |
| **nolitsa** | Python (numba) | from-data correlation dimension (`d2`), MLE Lyapunov, FNN embedding dim, IAAFT surrogates | ✅ |
| **antropy** | Python | sample/permutation entropy, DFA | ✅ v0.2 |
| **neurokit2** | Python | broad from-data complexity: entropy (sample/perm/multiscale), DFA, Hurst, correlation dim, RQA, embedding dim, surrogates | ✅ v0.2 |
| **pyunicorn** | Python | recurrence quantification (RQA determinism) | ✅ v0.9 |
| **PyDSTool** | Python | continuation / generators | ❌ does not import on NumPy ≥ 2 (`numpy.distutils` removed) |
| **TISEAN** | C / Fortran | `lyap_k`, `lyap_r`, `d2`, `c2` CLI tools | ❌ no binaries; legacy C+Fortran does not build with the current toolchain |

PyDSTool and TISEAN are kept as documented all-blank columns so the comparison
records that they were evaluated.

> **On comparing against DynamicalSystems.jl.** The TSDynamics source tree
> deliberately never references the Julia dynamical-systems ecosystem (see
> `CLAUDE.md`). This benchmark folder is the one place it is named, because a
> head-to-head performance comparison is its entire purpose and the maintainer
> explicitly asked for it. No idea or citation flows from it into the library.

---

## What the comparison shows

The rendered numbers live in [`RESULTS.md`](RESULTS.md); the patterns below are
the durable, machine-independent takeaways from a full run.

- **Integration is TSDynamics' clear strength.** Its Rust engine integrates the
  Lorenz system ~100–150× faster than SciPy's `solve_ivp` on both the short and
  long horizons, and is within ~2× of DynamicalSystems.jl — while producing the
  whole dense trajectory in one call. The `jit` backend roughly halves `interp`.
  All adaptive integrators hit the same trajectory to ~1e-9 at matched tolerance
  (DynamicalSystems.jl's Vern9 edges ahead at ~1e-10).
- **Precision is excellent where there is a ground truth.** TSDynamics gives the
  most accurate Lorenz λ_max of any library here (Δ < 1e-3), matches nolitsa on
  the embedded correlation dimension (≈2.05), and nails the Hénon fixed point and
  maximal exponent. DynamicalSystems.jl is comparably accurate throughout.
- **The iterative *analysis* routines are where DynamicalSystems.jl leads.** For
  the tasks that loop in Python over many small engine calls — the Lyapunov
  spectrum, basins of attraction, the bifurcation sweep, and the map maximal
  exponent — TSDynamics trails DynamicalSystems.jl's native-Julia implementations,
  by a large margin for basins and the bifurcation sweep. This is the honest
  counterpart to the integration win: it points squarely at where moving the
  analysis loops into the engine (as the trajectory path already is) would pay off.
- **From-data Lyapunov is method-sensitive across *every* library.** On a
  deliberately oversampled Lorenz series all estimators miss the literature 0.906
  — the Rosenstein-family ones (nolds, nolitsa, TSDynamics) cluster near 1.3,
  DynamicalSystems.jl's Kantz + automatic linear-region fit lands near 0.61. The
  spread is a property of the problem, not a ranking of the libraries.
- **The data-only libraries (nolds, nolitsa)** do exactly their niche — fractal
  dimension and Lyapunov from a scalar series — and nothing else, so most of
  their columns are blank. **pynamical** is logistic-bifurcation-only.
  **PyDSTool** and **TISEAN** could not be run here at all (see the table above).

## How to run

Everything runs out of a dedicated, isolated virtual-env so it never perturbs
the project's own `.venv` (the competitors pin **NumPy < 2.5** because
numba — and therefore pynamical and nolitsa — does not yet support NumPy 2.5).

### One-time setup

```bash
# 1. Python competitors + TSDynamics in one consistent env (Python 3.12, numpy<2.5)
uv venv --python 3.12 benchmarks/.venv-bench
BP=benchmarks/.venv-bench/bin/python
uv pip install --python $BP "numpy<2.5" scipy numba nolds sdeint matplotlib pandas
uv pip install --python $BP "numpy<2.5" dysts antropy neurokit2 pyunicorn
uv pip install --python $BP --no-deps pynamical
uv pip install --python $BP --no-deps "git+https://github.com/manu-mannattil/nolitsa.git"
uv pip install --python $BP .            # build + install TSDynamics into it

# 2. DynamicalSystems.jl (Julia ≥ 1.10)
julia --project=benchmarks/julia -e 'using Pkg; Pkg.add(["DynamicalSystems","OrdinaryDiffEq","JSON3","StaticArrays"]); Pkg.precompile()'
```

### Run

```bash
BP=benchmarks/.venv-bench/bin/python

$BP benchmarks/run_benchmarks.py                       # full run (writes RESULTS.md)
$BP benchmarks/run_benchmarks.py --quick               # CI-sized inputs, fewer reps
$BP benchmarks/run_benchmarks.py --skip-julia          # Python libraries only
$BP benchmarks/run_benchmarks.py --only lyapunov_spectrum,correlation_dimension
$BP benchmarks/run_benchmarks.py --libs tsdynamics-jit,scipy --skip-julia
```

> Use the **`.venv-bench` interpreter directly** — never `uv run`, which would
> re-sync the environment to the project lockfile and drop the competitor
> packages.

Outputs land in:

- `benchmarks/RESULTS.md` — the rendered Markdown comparison tables;
- `benchmarks/results/all.json` — the merged machine-readable results;
- `benchmarks/results/speed.csv`, `precision.csv` — the two tables as CSV;
- `benchmarks/results/<library>.json` — each library's raw record;
- `benchmarks/results/config.json` + `series_*.txt` — the frozen shared inputs.

---

## Methodology

- **Best-of-N timing.** Each task is timed N times and the **minimum** is kept —
  the most reproducible estimator of intrinsic cost (the machine can only ever
  add noise, never remove it). A warm-up call is made *before* timing so
  tape-lowering / numba-JIT / Julia-JIT compilation is paid once and excluded.
- **Process isolation.** Every library runs in its own subprocess (Python via
  `runworker.py`, Julia via `julia/bench.jl`) and writes a JSON record the
  orchestrator merges. A crash, a slow library, or an import side effect cannot
  take the rest of the suite down, and each library gets a clean interpreter.
- **Frozen, shared inputs.** All parameters live in `config.py`, dumped to
  `results/config.json`; the Julia script reads the same JSON, so Python and
  Julia use byte-identical parameters. The *from-data* tasks additionally read
  the **same dumped time series** (`results/series_*.txt`), generated once by
  SciPy/NumPy independently of any benchmarked library — so every estimator sees
  identical numbers, and the comparison measures the estimator, not the input.
- **Integration speed vs analysis.** The three integration tasks each use the
  library's *own* integrator (that is the point). The from-data analysis tasks
  feed every library the *same* generated series, isolating the analysis
  algorithm from integration speed.

### Per-task protocol

| Task | System | Protocol | Precision reference |
|---|---|---|---|
| Integration — short | Lorenz | `T=100`, `dt=0.01`, DOP853, `rtol=atol=1e-9` | — (speed) |
| Integration — long | Lorenz | `T=10000`, `dt=0.01`, DOP853, `rtol=atol=1e-9` | — (speed) |
| Integration accuracy | Lorenz | `T=8`, DOP853 `rtol=atol=1e-10`, ‖final − ref‖∞ | ref = SciPy DOP853 @ 1e-13 |
| Lyapunov spectrum | Lorenz | full spectrum over `T≈500`, report λ_max | 0.9056 (Sprott 2003) |
| Maximal Lyapunov | Hénon | from the **system** (TSDynamics/DSjl) or a generated **orbit** (nolds/nolitsa) | 0.419 (Sprott 2003) |
| Maximal Lyapunov from data | Lorenz x(t) | shared series, delay-embedded, **Rosenstein** across all three estimators | 0.9056 |
| Correlation dimension | Lorenz x(t) | shared series, delay-embedded (dim 5, delay 11), data-scale radii | 2.05 (Grassberger–Procaccia 1983) |
| Bifurcation diagram | logistic map | 1000 rates × 100 gens, 200 transient | — (speed) |
| Basins of attraction | Newton z³−1 map | 200×200 grid over [−1.5,1.5]² (three cube-root basins) | — (speed) |
| Fixed points | Hénon | find the saddle on the attractor; report x* | 0.6314 (analytic) |
| Poincaré section | Rössler | y=0 upward crossings, ~1000 of them | — (speed) |
| Sample entropy | Lorenz x(t) | shared series (3000 pts), m=2, r=0.2·std | — (cross-library agreement) |
| Permutation entropy | Lorenz x(t) | order 3, normalized | — (agreement) |
| Multiscale entropy | Lorenz x(t) | 5 scales | — (conventions differ, see notes) |
| DFA | white noise | DFA scaling exponent α | 0.5 (white noise has α=0.5) |
| Hurst | white noise | rescaled-range Hurst H | 0.5 (white noise has H=0.5) |
| RQA determinism | Lorenz x(t) | shared series (1200 pts), embed dim 3 / τ 5, recurrence rate 0.05 | — (agreement) |
| Embedding dimension | Lorenz x(t) | FNN (nolitsa) / Cao-AFNN (neurokit2, TSDynamics) | ≈3 (Lorenz; method-dependent) |
| Surrogate generation | Lorenz x(t) | one IAAFT surrogate | — (speed) |

dysts joins the integration **short/long** rows (it integrates in physical time when
`m.dt=None`) and the correlation-dimension + DFA rows; box-counting and generalized
(Rényi) dimensions are **TSDynamics-only** capabilities, so they are noted here rather
than added as all-blank rows.

### Fairness notes

- **From-data Lyapunov** is famously method- and parameter-sensitive. All three
  estimators (nolds, nolitsa, TSDynamics) therefore use the **same Rosenstein
  algorithm**, the same embedding, and the same Theiler window on the **same**
  series. They agree with each other (~1.3) and all overestimate the literature
  0.906 — a property of the (deliberately oversampled) problem, not of any one
  library.
- **Correlation dimension** depends on the radii scaling-region. TSDynamics and
  nolitsa are both given an explicit radii range spanning the same fraction of
  the attractor extent, so the scaling-region selection is identical; nolds
  uses its own RANSAC fit; DynamicalSystems.jl uses `grassberger_proccacia_dim`'s
  automatic fit. TSDynamics also natively measures D₂ on the **full attractor**
  (≈2.05) — the embedded-scalar row is the harder, apples-to-apples reconstruction.
- **Maximal Lyapunov (Hénon)** mixes from-system (TSDynamics, DSjl) and from-data
  (nolds, nolitsa) methods in one row; both target the same exponent and the
  cells are not directly speed-comparable (data methods also pay neighbour
  searches). It is a capability comparison.
- **Entropy / DFA / Hurst / RQA** feed every library the **same** series with
  matched parameters, so the cross-library *agreement* is the validation (sample
  entropy ≈0.143 and permutation entropy ≈0.451 agree to ~3 digits across all
  libraries; DFA/Hurst land ≈0.5 on white noise; RQA determinism ≈0.99). The one
  exception is **multiscale entropy**: TSDynamics reports the mean across scales
  while neurokit2 reports its single MSEn summary index — *different definitions*,
  so those two numbers are not expected to match (the row shows each library does
  MSE and how fast, not a like-for-like value).
- **dysts** is deliberately **off the Lyapunov and integration-accuracy rows**: it
  rescales each system's time axis per characteristic period (for ML benchmarking),
  so its Lyapunov exponents are in rescaled-time units (Lorenz ≈0.44, not 0.906)
  and its trajectory cannot be pinned to this benchmark's fixed `[1,1,1]`/physical-
  time reference. Its integration **speed** (physical time, `m.dt=None`) and its
  geometric measures (`gp_dim`, DFA) are clean comparisons.

---

## Layout

```
benchmarks/
├── README.md              # this file
├── RESULTS.md             # generated comparison tables
├── config.py              # frozen shared parameters + literature references
├── series.py              # shared from-data series generators (SciPy/NumPy)
├── harness.py             # task catalogue, best-of-N timer, table emitters
├── runworker.py           # runs ONE Python adapter in isolation → JSON
├── run_benchmarks.py      # orchestrator: spawn workers + Julia, merge, render
├── adapters/              # one adapter per Python library
│   ├── tsdynamics_adapter.py   # the library under test (interp + jit)
│   ├── scipy_adapter.py
│   ├── pynamical_adapter.py
│   ├── nolds_adapter.py
│   └── nolitsa_adapter.py
├── julia/
│   ├── bench.jl           # the DynamicalSystems.jl column (same schema)
│   └── Project.toml       # pinned Julia environment
└── results/               # generated JSON / CSV / dumped series (git-ignored)
```

### Adding a task or a library

- **A new task:** add a `TaskSpec` to `harness.TASKS`, then implement
  `task_<key>(self, quick)` on each adapter that supports it (and a branch in
  `julia/bench.jl`). Unsupported adapters need no change — the cell stays blank.
- **A new Python library:** add an adapter subclassing `adapters._base.BaseAdapter`
  and register it in `adapters/__init__.py::REGISTRY`.
