---
description: Head-to-head timing and accuracy of TSDynamics against the Python dynamical-systems ecosystem — integration ~60–150× faster than SciPy and ~180–390× faster than dysts, a competitive analysis toolkit, and where the other libraries still win.
---

<span class="ts-kicker">References · Benchmarks</span>

# Benchmarks

A library is only as good as the numbers it produces and the time it takes to
produce them. This page is the honest, reproducible comparison of TSDynamics
against the established Python ecosystem for integrating and analysing
dynamical systems — the same classic tasks run through each library's own code,
timed the same way, on the same machine.

The headline is the integration engine: on the Lorenz system TSDynamics'
Rust backend produces the whole dense trajectory **~68× faster than SciPy's
`solve_ivp`** with the interpreter, **~130× faster** with the JIT, and roughly
**180–390× faster than dysts** — while hitting the same reference trajectory to
$\approx 10^{-9}$ at matched tolerance. The analysis toolkit is competitive to
strong across the from-data routines, and this page is equally clear about the
handful of tasks where a specialised library is faster or where TSDynamics has
no entry at all.

<figure markdown>
![Horizontal bar chart of integration speedups: TSDynamics interp and jit versus SciPy and dysts on the short, long, and Poincaré tasks](../assets/figures/references/integration-speedup.svg){ loading=lazy }
<figcaption>Integration is TSDynamics' clear strength. Bars are the recorded best-of-N wall-time ratios: the Rust engine (teal · <code>interp</code>, indigo · <code>jit</code>) integrates the Lorenz system in one dense call ~68–150× faster than SciPy and ~180–390× faster than dysts, and marches the Rössler Poincaré section ~27× faster than SciPy's event integrator.</figcaption>
</figure>

## Methodology

Every number on this page comes from a single full run of the cross-library
harness in the repository's `benchmarks/` folder. The methodology is designed to
be fair to every library and reproducible on any machine.

- **Best-of-N wall time.** Each task is timed $N$ times and the **minimum** is
  kept — the most reproducible estimator of intrinsic cost, since the machine can
  only ever *add* noise, never remove it. A warm-up call is made **before** timing,
  so one-time compilation (TSDynamics' tape lowering, numba's JIT) is paid once
  and excluded from the measured time.
- **The library's own code, each task.** The three integration tasks each use the
  library's *own* integrator — that is the entire point of an integration
  benchmark. The from-data analysis tasks feed **every** library the *same*
  generated time series (dumped once, independently of any benchmarked library),
  so the comparison measures the estimator, not the input.
- **Process isolation.** Every library runs in its own subprocess and writes a
  JSON record the orchestrator merges. A crash, a slow library, or an import
  side-effect cannot take the rest of the suite down, and each library gets a
  clean interpreter.
- **Frozen, shared inputs.** All parameters live in one config file; the from-data
  tasks additionally read the same dumped series, so every estimator sees
  byte-identical numbers.
- **Precision against a tight reference.** Where a task has a ground truth — a
  literature Lyapunov exponent, an analytic fixed point, a $10^{-13}$ reference
  trajectory — the table reports both the estimate and its deviation $\Delta$ from
  that reference.

### The environment

| | |
|---|---|
| **Platform** | Linux x86-64 (`glibc` 2.43) |
| **Python** | 3.12.12 |
| **TSDynamics** | 5.2.4 (`interp` + `jit` backends) |
| **NumPy** | `< 2.5` (pinned — numba, and therefore pynamical/nolitsa, does not yet build on NumPy 2.5) |

!!! note "Reproduce it yourself"
    The harness, adapters, frozen inputs and rendered tables all live under
    `benchmarks/` in the source tree. It runs out of a **dedicated** virtual-env
    so it never perturbs the project's own — see `benchmarks/README.md` for the
    one-time setup and the `run_benchmarks.py` invocations. The numbers below are
    the committed output of one full run; re-running on your own hardware will
    shift the absolute times but not the relative story.

### The libraries compared

Only the **Python** ecosystem is compared here — the libraries a Python user
would actually reach for. Each contributes what it is designed for; a library
that does not provide a capability leaves that cell **blank**.

| Library | Version | What it contributes to the comparison |
|---|---|---|
| **TSDynamics** (`interp` + `jit`) | 5.2.4 | the library under test — every task, both engine backends |
| **SciPy** | 1.18.0 | the integration baseline (`solve_ivp`), fixed points (`fsolve`), Poincaré (events) |
| **dysts** | 0.96 | a chaotic-systems catalogue on a SciPy integrator; correlation dimension (`gp_dim`), DFA |
| **pynamical** | 0.3.3 | the logistic-map bifurcation diagram (numba) |
| **nolitsa** | — | from-data correlation dimension, MLE Lyapunov, FNN embedding dimension, IAAFT surrogates (numba) |
| **antropy** | 0.2.2 | sample / permutation entropy, DFA |
| **neurokit2** | 0.2.13 | broad from-data complexity — entropy, DFA, Hurst, correlation dim, RQA, embedding dim, surrogates |
| *a nonlinear-time-series toolkit* | — | from-data correlation dimension, Rosenstein Lyapunov, sample entropy, DFA, Hurst |
| *a recurrence-network package* | 0.9.0 | recurrence quantification (RQA determinism) |

Two further libraries were evaluated but could not be run in this environment,
and are recorded here for completeness: **PyDSTool** (does not import on
NumPy ≥ 2 — the removed `numpy.distutils`) and **TISEAN** (legacy C/Fortran CLI
tools that do not build with the current toolchain).

## Integration speed

The core task. Integrate the Lorenz system with DOP853 at `rtol=atol=1e-9` and
return the trajectory. dysts joins the integration rows (it integrates in
physical time when `m.dt=None`); the from-data-only libraries — pynamical,
nolitsa, antropy, neurokit2 and the two nonlinear-time-series toolkits — correctly
leave these cells blank.

| Task | TSDynamics `interp` | TSDynamics `jit` | SciPy | dysts |
|---|---:|---:|---:|---:|
| Integration — short (Lorenz, $T=100$) | **7.19 ms** | **3.71 ms** | 486.06 ms | 1.431 s |
| Integration — long (Lorenz, $T=10000$) | **773.49 ms** | **363.29 ms** | 54.427 s | 138.035 s |
| Poincaré section (Rössler, $y=0$, ≈1000 crossings) | **197.04 ms** | 198.78 ms | 5.293 s | — |

Reading the ratios:

- **Short integration:** `interp` is **67.6×** faster than SciPy and **199×**
  faster than dysts; `jit` is **131×** faster than SciPy and **386×** faster than
  dysts.
- **Long integration:** the win holds at scale — **70×** / **150×** faster than
  SciPy (`interp` / `jit`), **178×** / **380×** faster than dysts.
- **Poincaré section:** the engine marches the whole attractor and refines every
  crossing in one call, **~27×** faster than SciPy's event-based `solve_ivp`. Here
  the JIT gives no edge — the cost is dominated by the crossing refinement, not
  the raw stepping.

The `jit` backend (Cranelift) roughly **halves** the interpreter's time on the
raw integration tasks; on the iterative and event-driven tasks, where per-call
Python overhead or refinement dominates, the two backends are within noise of
each other.

## Integration accuracy

Speed is only half the story. Integrated to $T=8$ with DOP853 at
`rtol=atol=1e-10`, how close is the final state to a $10^{-13}$ reference
trajectory (itself a SciPy DOP853 run at that tolerance)?

| Task | TSDynamics `interp` | TSDynamics `jit` | SciPy |
|---|---:|---:|---:|
| $\lVert \Delta \rVert_\infty$ vs $10^{-13}$ reference | $3.33\times10^{-9}$ | $3.33\times10^{-9}$ | $2.83\times10^{-9}$ |
| Wall time for that run | 303 µs | 322 µs | 25.74 ms |

All three adaptive integrators hit the same trajectory to $\approx 10^{-9}$ at
matched tolerance — the tiny residual difference between $3.33\times10^{-9}$ and
$2.83\times10^{-9}$ is the expected float-order difference between two DOP853
implementations, not an accuracy gap. The point of the row is that the ~85×
speed advantage costs **nothing** in accuracy: `interp` and `jit` are bit-for-bit
identical here, and both land within a hair of SciPy's own answer.

## The analysis toolkit

For the from-data analysis routines the harness feeds **every** library the same
generated series, so the comparison isolates the estimator. TSDynamics ranges
from competitive to comfortably fastest across most of these — and, honestly,
loses two of them.

<figure markdown>
![Horizontal bar chart on a log axis showing TSDynamics analysis-toolkit speed relative to the fastest competitor per task; five wins in teal, two losses in amber](../assets/figures/references/analysis-speedup.svg){ loading=lazy }
<figcaption>Same series, every library, fastest competitor per task. Teal bars (to the right of the 1× line) are tasks where TSDynamics is fastest — embedding dimension, correlation dimension, the from-data Lyapunov, RQA and multiscale entropy. Amber bars are the honest losses: the specialised IAAFT surrogate generator and sample-entropy estimators edge it out.</figcaption>
</figure>

### Where TSDynamics leads

| Task | TSDynamics | Best competitor | Speedup |
|---|---:|---|---:|
| Embedding dimension (Cao / FNN) | **27.43 ms** | neurokit2 178.35 ms · nolitsa 1.237 s | **6.5×** / **45×** |
| Correlation dimension (embedded) | **216.87 ms** | nolitsa 250.22 ms · dysts 1.028 s · a time-series toolkit 1.637 s | **1.2×** – **7.5×** |
| Maximal Lyapunov from data | **34.81 ms** | nolitsa 130.15 ms · a time-series toolkit 252.68 ms | **3.7×** / **7.3×** |
| RQA determinism | **18.27 ms** | a recurrence-network package 28.24 ms · neurokit2 118.98 ms | **1.5×** / **6.5×** |
| Multiscale entropy | **31.36 ms** | neurokit2 146.10 ms | **4.7×** |

### Where the other libraries win

TSDynamics is **not** universally fastest, and the benchmark says so plainly:

| Task | TSDynamics | Fastest competitor | Verdict |
|---|---:|---|---|
| Sample entropy | 21.10 ms | neurokit2 12.24 ms · antropy 14.83 ms | ~1.7× **slower** than the specialised C-accelerated estimators (but ~20× faster than a pure-Python one) |
| Permutation entropy | 216 µs | antropy 81 µs | ~2.7× **slower** than antropy's tight NumPy kernel (but ~8× faster than neurokit2) |
| IAAFT surrogate | 26.13 ms | nolitsa 10.73 ms · neurokit2 11.55 ms | ~2.4× **slower** than the numba-JIT surrogate generators |

These are all cheap, tight inner loops where a single-purpose kernel (antropy's
sample-entropy C loop, nolitsa's numba IAAFT) has the edge — and all three land
in the tens-of-milliseconds-or-less range, so the absolute cost is small either
way.

### Capability rows

Some tasks are not a like-for-like speed race but a record that the library does
the job at all, on a shared series:

| Task | TSDynamics | Others |
|---|---:|---|
| Lyapunov spectrum (Lorenz, full) | 633.22 ms | *system-based; no from-data competitor here* |
| Maximal Lyapunov (Hénon map) | 453.20 ms | a time-series toolkit 1.695 s · nolitsa 279.59 ms *(from-data, not directly comparable)* |
| Bifurcation diagram (logistic sweep) | 339.39 ms | pynamical 37.50 ms *(numba, logistic-only)* |
| Basins of attraction (Newton $z^3\!-\!1$) | 85.27 ms | *no Python competitor here* |
| Fixed points (Hénon) | 35.15 ms | SciPy `fsolve` 52 µs |

Two of these deserve the honest footnote:

- **Bifurcation and basins** are the iterative analyses that loop in Python over
  many small engine calls. pynamical's numba-compiled logistic sweep is faster on
  that one specialised map; more broadly, these loops are exactly the ones that
  would benefit from moving into the engine the way the trajectory path already
  has — the honest counterpart to the integration win.
- **Fixed points** is a root-find, not an integration: SciPy's `fsolve` on a bare
  residual is unbeatable at 52 µs. TSDynamics spends its 35 ms building the
  analytic Jacobian, running a multi-start search *and* certifying the roots — a
  different, heavier deliverable — but if you only need one root fast, `fsolve`
  wins.

## Precision where there is a ground truth

Speed means nothing without the right answer. On every task with a literature or
analytic reference, here is the estimate and its deviation $\Delta$.

| Task (reference) | TSDynamics | $\Delta$ | Notable others |
|---|---:|---:|---|
| Lyapunov spectrum — Lorenz $\lambda_\max = 0.9056$ | 0.9064 | **$7.8\times10^{-4}$** | — |
| Maximal Lyapunov — Hénon $= 0.419$ | 0.4204 | **$1.4\times10^{-3}$** | a time-series toolkit 0.3721 ($\Delta\,0.047$) · nolitsa 0.4176 ($\Delta\,0.0014$) |
| Correlation dimension — Lorenz $= 2.05$ | 2.054 | **$3.9\times10^{-3}$** | nolitsa 2.055 · a time-series toolkit 1.905 · neurokit2 1.819 · dysts 2.014 |
| Fixed point — Hénon $x^* = 0.6314$ | 0.6314 | **$6.3\times10^{-15}$** | SciPy 0.6314 ($\Delta\,2.3\times10^{-14}$) |
| Integration accuracy (Lorenz $T=8$) | — | $3.33\times10^{-9}$ | SciPy $2.83\times10^{-9}$ |

The takeaways:

- TSDynamics gives the **most accurate Lorenz $\lambda_\max$** on this page
  ($\Delta < 10^{-3}$), matches nolitsa on the embedded correlation dimension
  ($\approx 2.05$), and pins the Hénon fixed point to machine precision.
- **Cross-library agreement validates the shared-series tasks.** Fed identical
  input, sample entropy lands at $\approx 0.143$ and permutation entropy at
  $\approx 0.451$ to three digits across TSDynamics, antropy and neurokit2;
  DFA/Hurst sit at $\approx 0.5$ on white noise; RQA determinism at $\approx 0.99$.
  Agreement is the point — it means every library, including TSDynamics, computes
  the same quantity the same way.

### One honest caveat: from-data Lyapunov

The maximal-Lyapunov-**from-data** task is famously method- and
parameter-sensitive, and it is worth calling out because *every* library misses
the literature value:

| Method | Estimate ($\lambda_\max$, ref $= 0.9056$) |
|---|---:|
| TSDynamics (Rosenstein) | 1.29 |
| nolitsa (Rosenstein) | 1.299 |
| a time-series toolkit (Rosenstein) | 1.24 |

On a deliberately oversampled Lorenz series the Rosenstein-family estimators all
cluster near $1.3$. This is a property of the (hard, oversampled) problem, not a
ranking of the libraries — all three use the *same* algorithm on the *same*
series and agree with each other. It is exactly the kind of result the
[from-data Lyapunov](../analysis/lyapunov.md#lyapunov_from_data-from-a-measured-series)
page tells you to
treat with care: inspect the scaling region before trusting the slope.

## What the comparison shows

The durable, machine-independent takeaways:

- **Integration is the decisive win.** The Rust engine is ~68–150× faster than
  SciPy and ~180–390× faster than dysts, at the same accuracy, returning the whole
  dense trajectory in one call. If you integrate, this is the reason to be here.
- **Precision is excellent wherever there is a ground truth** — the best Lorenz
  $\lambda_\max$ on the page, a machine-precision fixed point, an embedded
  correlation dimension that matches the reference.
- **The analysis toolkit is competitive to fastest across most from-data
  routines** — embedding dimension, correlation dimension, from-data Lyapunov,
  RQA and multiscale entropy — and it is **one library** for all of them, rather
  than five single-purpose packages.
- **It is not fastest everywhere, and this page says so.** Specialised kernels win
  the tightest inner loops (sample/permutation entropy, IAAFT surrogates), SciPy's
  `fsolve` wins a bare root-find, and pynamical's numba sweep wins the logistic
  bifurcation. The iterative analysis loops (basins, bifurcation) are the clear
  place future engine work would pay off — the honest flip-side of the integration
  story.

## See also

- [Integration & methods](../analysis/integration-and-methods.md) — the Rust
  engine, backends and solver families behind the integration numbers
- [Lyapunov spectra](../analysis/lyapunov.md) — the spectrum and from-data
  estimators benchmarked above
- [Fractal dimensions](../analysis/dimensions.md) — the correlation-dimension
  routines, including the full-attractor $D_2$
- [Recurrence & RQA](../analysis/recurrence.md) — the recurrence quantification
  compared against neurokit2 and a recurrence-network package
- [Bibliography](bibliography.md) — the original papers behind every method and
  reference value used here
