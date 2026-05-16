# TSDynamics Roadmap

This is the master roadmap. It lists every milestone in the order they should be
executed, with a status badge. To work on a milestone, open its file under
`milestones/` and follow the template.

The mission, in one line: **become the reference dynamical-systems library, superseding
DynamicalSystems.jl, with a pure-Rust compute layer that is invisible to users.**

For protocol see [CONTRIBUTING-CLAUDE.md](CONTRIBUTING-CLAUDE.md). For current state see
[STATUS.md](STATUS.md).

## Status legend

- `TODO`   — not started
- `WIP`    — in progress (a chat is actively working on it; see STATUS.md)
- `DONE`   — landed, tests green, docs updated
- `BLOCKED` — see Open questions in the milestone file

## Tracks

The roadmap runs four tracks in parallel:

- **Track A — Analysis primitives** (M-prefix): Trajectory enrichment, events,
  bifurcations, equilibria, spectra, fractals, recurrence, surrogates, entropy,
  basins, periodic orbits, continuation, manifolds, codim-2.
- **Track B — Visualization** (V-prefix): DataSpec/Transform/Plotter abstractions,
  matplotlib reference plotters, plotly backend, user-extension docs.
- **Track C — Rust analysis kernels** (R-prefix): sweep, correlation, recurrence,
  basin, continuation kernels. Build toolchain.
- **Track D — Polish & adoption** (P-prefix): benchmarks, gallery, docs, paper.
- **Track E — Native solver migration** (N-prefix): replace JiTCODE / JiTCDDE /
  Numba with pure-Rust steppers. Phased, invisible to users.

## Milestones, in execution order

| ID | Title | Status | Track | Notes |
|----|-------|--------|-------|-------|
| M0 | Bootstrap `.planning/` framework | DONE | meta | This commit. |
| R1 | Rust toolchain + maturin + CI wheels | DONE | C | Unblocks everything below. |
| N1 | Rust map stepper (drops Numba dispatch) | DONE | E | First end-to-end Rust proof. IR + tracer + interpreter shipped. |
| M1 | Trajectory enrichment | DONE | A | Slice/decimate/derivative/project/window. Landed 2026-05-16. |
| M2 | Event & section detection | DONE | A | Powers Poincaré, return maps. Landed 2026-05-16. |
| R2 | Rust parameter-sweep kernel | TODO | C | rayon-backed. |
| M3 | Parameter-sweep API on top of R2 | TODO | A | No Python fallback. |
| V1 | Viz skeleton (mpl: timeseries, phase portrait) | TODO | B | DataSpec/Transform/Plotter live here. |
| M4 | Bifurcation diagrams (brute-force) | TODO | A | Atop M2 + M3. |
| V2 | Poincaré + bifurcation plotters | TODO | B | |
| M5 | Equilibria & local stability | TODO | A | Newton/Krylov + eigenvalue classification. |
| M6 | Embedding utilities (promote internals) | TODO | A | AMI, FNN, Takens — currently hidden. |
| M7 | Spectral toolkit | TODO | A | FFT, Welch PSD, spectrogram, CWT. |
| N2 | Pure-Rust ODE stepper suite (RHS via IR) | WIP | E | Multi-chat. N2.a landed (IR extension + SymEngine lowering + PyO3 RHS + goldens). N2.b–d pending: Rust stepper crate, stiff family, polish. |
| N3 | Variational ODE Lyapunov in Rust | TODO | E | Builds on N2's stepper + IR. Drops `jitcode_lyap`. |
| R3 | correlation_sum + boxcount kernels | TODO | C | Feeds M8. |
| M8 | Fractal dimensions | TODO | A | On R3. Includes Kaplan-Yorke. |
| R4 | recurrence kernel | TODO | C | Matrix + line histograms. |
| M9 | Recurrence analysis | TODO | A | On R4. RR, DET, LAM, L_max, ENTR, TT. |
| V3 | Field plotters | TODO | B | Basins, recurrence, spectrum. |
| M10 | Surrogates | TODO | A | Shuffle, AAFT, IAAFT, twin, pseudo-periodic. |
| M11 | Entropy & complexity | TODO | A | Perm/sample/ApEn/dispersion/TE. Soft hook to lzcomplexity. |
| R5 | Rust basin kernel | TODO | C | IC sweep + attractor labelling. |
| M12 | Basins of attraction | TODO | A | On R5. |
| V4 | Color/alpha transforms | TODO | B | Color-by-Lyapunov, alpha-by-density. |
| N4 | Cranelift JIT for the IR | TODO | E | Multi-chat. Replaces the IR interpreter from N1+N2 with cranelift-codegen'd native code. Drops JiTCODE entirely (the C-compile path stops being used even for symbolic→IR — SymEngine alone is enough). Performance milestone, not a correctness one. |
| M13 | Periodic orbit detection | TODO | A | Multi-chat. Shooting + Floquet. |
| R6 | Rust continuation kernel | TODO | C | Predictor/corrector. |
| M14 | Continuation | TODO | A | Multi-chat. On R6. |
| V5 | Plotly backend | TODO | B | Proves the abstraction. |
| N5 | Pure-Rust DDE solver suite | TODO | E | Multi-chat. Hardest. Layered on N2's stepper + a Rust-side history buffer with Hermite continuous extension. Multi-method like N2. Drops JiTCDDE. |
| M15 | Manifolds (local stable/unstable) | TODO | A | Multi-chat. |
| M16 | Codim-2 bifurcation curves | TODO | A | Multi-chat. |
| N6 | Compatibility validation & default flip | TODO | E | Regression suite green; flip default. |
| N7 | Deprecate JiTCODE/JiTCDDE/Numba | TODO | E | Remove runtime deps. |
| V6 | User-extension tutorials | TODO | B | New kind, transform, backend. |
| P1 | Benchmark harness vs DynamicalSystems.jl | TODO | D | bench/RESULTS.md, CI weekly. |
| P2 | Examples gallery | TODO | D | One notebook per "story." |
| P3 | Sphinx tutorials & API docs | TODO | D | furo theme already configured. |
| P4 | Pickling & HDF5/Zarr persistence | TODO | D | |
| P5 | Paper + Zenodo DOI + JOSS submission | TODO | D | |

## Dependency graph (the parts that matter)

```
M0 ──┬─ R1 ──┬─ N1 ─── (Rust path for maps lands)
     │       ├─ R2 ─── M3 ─── M4 ──┐
     │       ├─ R3 ─── M8         │
     │       ├─ R4 ─── M9         ├─ V2/V3 (plotters)
     │       ├─ R5 ─── M12        │
     │       └─ R6 ─── M14        │
     │                            │
     ├─ M1 ── M2 ──────────────────┘
     ├─ M5    M6    M7    M10   M11
     │
     ├─ V1 ── V2 ── V3 ── V4 ── V5 ── V6
     │
     └─ N2 ── N3 ── N4 ── N5 ── N6 ── N7
                (multi-chat each)

P1–P5 run alongside from when there's something to benchmark / gallerise.
```

Once R1 lands, almost everything below it can be parallelised across chats. Track E is
the only strictly serial track (each N depends on the previous).
