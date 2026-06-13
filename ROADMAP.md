# TSDynamics Roadmap

**Goal:** evolve TSDynamics from a compiled-integration + Lyapunov library into the
reference Python platform for nonlinear dynamics — system definition, trajectory
generation, chaos quantification, attractor/basin analysis, fractal dimensions,
time-series analysis, and visualization — while keeping the current dead-simple
subclass contract (`params` + `dim` + one `_equations`/`_step` method).

This document is the working plan. Phases are ordered by dependency, not strict
calendar. Each phase has acceptance criteria so progress is measurable.

---

## Status — as of v2.1.0

> Single source of truth for *where we are*. Update this block when a phase
> moves. The plan below this section is the destination; this is the position.

**Shipped to PyPI:** v2.0.0 (platform relaunch) · v2.0.1 (Blasius basin fix) ·
v2.1.0 (Phase-1 tail). Release + docs deploy are automated on every push to
`main` (python-semantic-release + GitHub Pages) and proven end-to-end. Docs
live at <https://el3ssar.github.io/TSDynamics/>.

| Phase | State | Notes |
|---|---|---|
| **1 — Core abstraction** | ✅ **done** | `System` protocol on all families; `PoincareMap`/`StroboscopicMap`/`TangentSystem`/`EnsembleSystem`/`ProjectedSystem`/`WrappedSystem`; `Trajectory` (named components, KD-tree, set distances, provenance `MetaStore`); `tsdynamics.sampling` (Box/Ball/Grid, samplers, `grid_points`, `set_distance`); symbolic Jacobian autogen (13 real bugs fixed); registry-driven tests. *Deferred:* route per-family `lyapunov_spectrum` through `TangentSystem`; parallelize `EnsembleSystem`. |
| **2 — Rust backend** | 🟡 **~50%** | **Part 1 (ODE on diffsol) nearly done:** `backend="auto"`/`"diffsol"` (pydiffsol, prebuilt LLVM wheels) **translates 100% of the 118 ODEs**, cross-validated vs JiTCODE (sample on `slow`, full catalogue nightly via `test_diffsol_backend.py::test_cross_validation_full_catalogue`), benchmarked (`benches/bench_backends.py`: 9–29× JiTCODE, ~180× SciPy), documented (Theory → Backends). *Left in part 1:* flip the literal default to `auto` once the nightly full-catalogue gate is green. **Part 2 (its own milestone):** the bespoke `tsdynamics-core` PyO3/maturin crate — Rust **DDE** solver, **SDE** family, rayon ensembles, cross-platform prebuilt wheels. |
| **3 — Chaos quantification** | 🟡 **~50%** | Done: `orbit_diagram`, `poincare_section`, `max_lyapunov`, `kaplan_yorke_dimension`, `fixed_points` (maps), `lyapunov_spectrum` dispatcher. *Left:* `lyapunov_from_data` (Kantz), GALI, 0–1 test, expansion entropy, periodic orbits, `estimate_period`, `fixed_points` for flows. |
| **4 — Attractors & basins (the moat)** | ⬜ **0%** | **Next milestone.** Depends only on Phase 1 (done) — *not* on 2 or 3. |
| **5 — Time-series & geometry** | ⬜ **0%** | embeddings, fractal dimensions, entropy/complexity, recurrence, surrogates. |
| **6 — Visualization** | ⬜ **0%** (user-facing) | figure tooling exists inside the docs build; no `tsdynamics.plot` module yet. |
| **7 — Ecosystem & credibility** | 🟡 **~40%** | docs site + citations + known-value tests + semantic-release done. *Left:* docstring-citation lint rule, hypothesis tests, conda-forge, benchmark notebook, JOSS. |

**Solver coverage (Phase-2 pillar).** No single solver fits every system —
explicit methods fail on stiff RHS, implicit ones can fail elsewhere. Each
system therefore declares its own `_default_method`; stiff built-ins
(Oregonator, Duffing, SprottL/P, SprottJerk) default to `LSODA` so
`integrate()` works out of the box. The broader goal — a discoverable,
extensible multi-solver layer with automatic stiffness detection across both
backends — is tracked under Phase 2 (and is what the diffsol/Rust work must
preserve). Two systems still resist all adaptive solvers (BlinkingRotlet) or
diffsol specifically (WindmiReduced) and are documented exclusions.

**Dependency reality (not the numbering):** Phase 4's prerequisites are all in
Phase 1, which is complete. **Phase 2 (Rust) is a *performance* multiplier for
Phase 4, not a prerequisite** — basins run on the current JiTCODE backend, just
slower at large grid sizes (the cheap mitigation is parallelizing
`EnsembleSystem`, a deferred Phase-1 item, not the full Rust migration). The
**rest of Phase 3 is independent of Phase 4** — neither blocks the other. So
Phase 4 can proceed now; do Phase 2 as a parallel track.

**Open follow-ups:** Dependabot security alerts on the repo (7); parallelize
`EnsembleSystem` if basin compute is slow.

---

## 0. Where we are (audit summary, June 2026)

**Strong (keep):**
- Three system families with a unified feel: `ContinuousSystem` (JiTCODE),
  `DelaySystem` (JiTCDDE), `DiscreteMap` (Numba).
- Compile caching with structural/control parameter split — param sweeps don't recompile.
- `ParamSet` fixed-key container kills silent param bugs.
- 149 built-in systems (123 ODE, 5 DDE, 26 maps) — already a larger predefined
  catalogue than any comparable library.
- Lyapunov spectra for *all three* families. DDE Lyapunov spectra in particular are
  something no mainstream competitor offers.
- Clean CI (lint + tests on Linux/macOS, py3.12/3.13), fast test suite.

**Weak (fix):**
- Analysis breadth: Lyapunov spectra is essentially the *only* analysis tool.
  No basins, no dimensions, no Poincaré sections, no orbit diagrams, no embeddings,
  no recurrence/entropy/surrogates, no plotting.
- 116/123 ODE systems lack `_jacobian`; all DDE systems lack it.
- Map `_step`/`_jacobian` signature order vs `params` order is unchecked at runtime.
- Jacobian correctness is never validated against finite differences in tests.
- DDE Lyapunov requires the clunky two-call workflow (integrate → pass `ic=`).
- JiTCODE/JiTCDDE: runtime C-compiler dependency, slow first compile, fragile cache,
  upstream maintenance risk.
- `meta` storage overwrites results; no provenance.

**Competitive context.** The benchmark ecosystem to surpass is a Julia umbrella of
11 packages (~550 exported functions) organized around one idea: *a single
`DynamicalSystem` abstraction that every analysis function consumes*, plus
composable derived systems (Poincaré map of a flow is itself a map → orbit diagram
of it is a bifurcation diagram, for free). Its known gaps — no DDE support at all,
young/shallow SDE support, JIT latency at every session start, heavyweight GUI
dependencies — are precisely where a precompiled Python library can win.

---

## 1. Phase 1 — Core abstraction redesign (the multiplier)

Everything later depends on this. The single most important architectural move:
**decouple "being a dynamical system" from "being one of our three base classes"**
so that every analysis function can consume any system, including derived ones.

### 1.1 The `System` runtime protocol

A minimal stepping interface implemented by all families and all wrappers:

```python
class System(Protocol):
    dim: int
    def step(self, n_or_dt) -> None
    def state(self) -> np.ndarray
    def set_state(self, u) -> None
    def time(self) -> float | int
    def reinit(self, u=None, *, t=None, params=None) -> None
    def trajectory(self, T, *, dt=..., transient=0.0, ...) -> Trajectory
    @property
    def is_discrete(self) -> bool
```

- `ContinuousSystem`, `DelaySystem`, `DiscreteMap` keep their current subclass
  contract *unchanged*; they gain the protocol methods.
- A `WrappedSystem` adapter lets users plug any external stepper (their own
  simulation code) into the whole analysis library. Cheap to build, huge reach.

### 1.2 Derived systems (composition layer)

| Wrapper | What it does | What it unlocks |
|---|---|---|
| `PoincareMap(sys, plane)` | root-found plane crossings of a flow, exposed as a discrete map | return maps, bifurcation diagrams of flows, PSOS |
| `StroboscopicMap(sys, period)` | sample a forced flow every period | forced-oscillator analysis |
| `TangentSystem(sys, k)` | state + k deviation vectors via Jacobian | Lyapunov spectra (unified!), GALI, covariant vectors later |
| `EnsembleSystem(sys, states)` | many ICs stepped in lockstep (Rust/rayon) | max-Lyapunov by rescaling, basin sampling, ensemble statistics |
| `ProjectedSystem(sys, proj, complete)` | analysis on a subspace | high-dim systems |

Lyapunov code then collapses: one QR/Benettin implementation on `TangentSystem`
replaces the three per-family implementations (the DDE family keeps its special
path — that one is a differentiator to preserve and document loudly).

### 1.3 `Trajectory` → first-class state-space data

`Trajectory` grows into the lingua franca every analysis function accepts
(system *or* trajectory where meaningful):

- Named components (`traj["x"]`, names from an optional class attr `variables`).
- Point-set semantics: `minmax`, `standardize`, fancy indexing, iteration over points.
- Neighbor searches (KD-tree; SciPy now, Rust later) — one shared API used by
  embeddings, dimensions, data-Lyapunov, recurrence.
- Set distances (centroid / Hausdorff / minimum) — needed for attractor matching.
- `sample_statespace(region)` helpers (box/sphere/grid samplers) for basin work.
- Provenance: `traj.meta` records system, params, solver, tolerances, seed.
- `system.meta` results become append-with-key (`meta.record("lyapunov", value, **ctx)`),
  no silent overwrites.

### 1.4 Housekeeping in the same phase

- Runtime introspection check: map `_step`/`_jacobian` signature must match
  `params` insertion order (raise at class definition time, not silently).
- Finite-difference Jacobian validation test, parametrized over all systems
  defining `_jacobian`.
- Symbolic Jacobian autogeneration (SymEngine `diff`) so *every* ODE system gets
  a Jacobian for free; hand-written ones become optional overrides.
- Fix dependabot security alerts (8 on the repo) and re-accept the CI action bumps.

**Acceptance:** all existing tests pass unchanged; `lyapunov_spectrum` for ODEs and
maps routed through `TangentSystem`; Poincaré section of the Rössler attractor and
a bifurcation diagram of the logistic map each take ≤ 5 lines of user code.

---

## 2. Phase 2 — Rust numerical backend

Replace the JiTCODE/JiTCDDE C-codegen pipeline with a Rust extension crate
(`tsdynamics-core`) shipped as prebuilt wheels via PyO3 + maturin. Reuse existing
crates; do not write integrators from scratch.

### 2.1 Chosen components (research conclusions)

| Concern | Component | Why |
|---|---|---|
| ODE/DAE solving | **diffsol** crate (MIT, JOSS-published, active) | BDF + SDIRK (stiff) + explicit RK, dense output, event detection, forward/adjoint sensitivities, sparse algebra |
| Equation JIT | **DiffSL** (diffsol's DSL, Cranelift backend) | runtime JIT ≪ 1 s, pure-Rust dependency tree → trivially wheelable; no C compiler on user machines, no compile cache needed |
| RHS authoring | translator: SymEngine expression tree → DiffSL source (~few hundred lines of Python) | users keep writing the exact same `_equations`; strictly simpler than the current C codegen |
| DDE solving | **differential-equations** crate (Apache-2.0) | the only maintained Rust DDE solver: method of steps, dense history interpolation, constant/time-/state-dependent delays, events |
| SDE solving (new family) | same crate (Euler–Maruyama, Milstein) | cheapest path to an `StochasticSystem` family — a known weak spot of the competition |
| Maps | pure-Rust iteration via the same translator (Numba kept as fallback during transition) | kills JIT warmup; rayon ensembles |
| Python interop | PyO3 (abi3-py312), rust-numpy (zero-copy), GIL released for whole integrations, rayon inside | the polars model |
| Escape hatch | Numba `@cfunc` address passed as `extern "C"` fn pointer | plain-Python RHS that the DSL can't express (numbalsoda-proven pattern) |

### 2.2 Migration strategy

1. `backend="rust" | "jitcode"` switch on `ContinuousSystem`; land Rust as
   experimental, promote to default after one minor release of parity testing,
   then drop JiTCODE (and the C-compiler requirement, and `~/.cache/tsdynamics`).
2. Cross-validation suite: every built-in system integrated with both backends,
   trajectories compared within tolerance; Lyapunov spectra compared to literature.
3. Benchmark suite (`benches/`) tracked in CI: vs SciPy, vs old backend, and vs
   the Julia ecosystem on identical problems (Lorenz, Rössler, MackeyGlass,
   Lorenz-96 N=128, stiff Robertson). Publish results in docs.
4. Wheels: manylinux + macOS (arm64/x86_64) + Windows via maturin-action.
   Cranelift only in wheels; LLVM backend optional for source builds.

**Acceptance:** `pip install tsdynamics` with zero compiler toolchain; first
integration of a fresh system < 1 s end-to-end; ≥ 10× SciPy on small chaotic
systems; DDE + SDE families running on the Rust backend; ensemble of 10⁴ Lorenz
ICs integrated in parallel through one call.

---

## 3. Phase 3 — Chaos quantification layer (`tsdynamics.analysis`)

All functions take a `System` (or `Trajectory` where data-driven). Ship with
citations in every docstring.

- **Lyapunov:** `lyapunov_spectrum(sys)` (QR/Benettin on `TangentSystem`),
  `max_lyapunov(sys)` (two-trajectory rescaling on `EnsembleSystem`, no Jacobian
  needed), `lyapunov_from_data(traj)` (Kantz-style neighborhood divergence).
- **Derived quantities:** `kaplan_yorke_dimension(spectrum)`, sum-vs-divergence
  consistency check, finite-time/local growth rates.
- **Chaos tests:** GALI_k (Skokos), 0–1 test (Gottwald–Melbourne) on plain
  time series, expansion entropy (Hunt–Ott) on sampled regions.
- **Structure:** `orbit_diagram(map_like, param, values)` — works on
  `DiscreteMap`, `PoincareMap`, `StroboscopicMap` (= bifurcation diagrams of
  flows by composition), carries state across parameter values;
  `poincare_section(sys_or_traj, plane)`; `fixed_points(sys, box)` (interval
  root finding + stability via Jacobian eigenvalues); periodic orbits
  (Schmelcher–Diakonos and Davidchack–Lai for maps; least-squares shooting for
  flows); `estimate_period(x)` (autocorrelation, periodogram, YIN).

**Acceptance:** logistic orbit diagram, Rössler bifurcation diagram via
`PoincareMap`, Hénon fixed points with stability, Lorenz Kaplan–Yorke ≈ 2.06 —
each as a tested example.

---

## 4. Phase 4 — Attractors, basins, global stability (the moat)

This is the flagship capability that has no Python equivalent anywhere — landing
it credibly is what makes the "surpass" claim real.

- **Attractor finding** (`find_attractors`): recurrences-in-grid finite-state
  machine (Datseris & Wagemakers 2022), featurize-and-cluster (DBSCAN, bSTAB-style
  per Stender & Hoffmann 2021), and proximity-to-known-attractors. Grid
  subdivision for multiscale attractors.
- **Basins:** `basins_of_attraction(finder, grid)` (full grids),
  `basin_fractions(finder, sampler)` (Monte Carlo), convergence diagnostics.
  Rust-parallel ensemble integration underneath.
- **Basin geometry:** basin entropy (Daza 2016), basin boundary fractal dimension,
  uncertainty exponent (Grebogi 1983), Wada tests.
- **Global continuation:** seed-continue-match across a parameter range —
  attractors *and* basin fractions tracked, with set-distance matchers
  (Hungarian assignment) — plus `tipping_probabilities` (Kaszás 2019).
- **Resilience:** minimal fatal shock (Halekotte & Feudel 2020), edge tracking
  between attractors.

**Acceptance:** Duffing and magnetic-pendulum basins reproduced (plots in docs);
multistable Lorenz-84 continuation showing attractor birth/death across forcing;
basin fraction error < 1% vs published values on at least two literature systems.

---

## 5. Phase 5 — Time-series & geometry layer

- **Delay embeddings:** `embed(x, dim, delay)` + generalized multivariate
  version; delay estimation (autocorrelation zero/min, first minimum of mutual
  information); dimension estimation (Cao's AFNN, Kennel's FNN); PECUZAL-style
  unified optimization later.
- **Fractal dimensions:** correlation sum (with Theiler window) +
  Grassberger–Procaccia, box-assisted fast variant, generalized/Rényi
  box-counting dimension, fixed-mass estimator, automated scaling-region fitting
  (largest-linear-region with confidence intervals — shared infrastructure).
- **Entropy & complexity:** permutation entropy, sample/approximate entropy,
  dispersion entropy, multiscale wrappers. Architecture: composable
  (outcome-space × estimator × measure) so the catalogue multiplies; integrate
  **lzcomplexity** (our own LZ76 C++ library) as the Lempel–Ziv provider.
- **Recurrence:** recurrence matrices (fixed ε / fixed rate), RQA measures
  (DET, LAM, L_max, entropy, trapping time...), windowed RQA. Sparse + Rust.
- **Surrogates:** shuffle, Fourier phase, AAFT, IAAFT + a proper
  `SurrogateTest(statistic, x, method) → p-value` harness.

**Acceptance:** correlation dimension of Lorenz ≈ 2.05 from a trajectory;
embedding pipeline reconstructs Rössler attractor from x-component only; RQA on
logistic map regime change; surrogate test rejects linearity for Lorenz data.

---

## 6. Phase 6 — Visualization (`tsdynamics.plot`)

Two tiers, both optional extras (no hard plotting dependency in core):

- **Static (matplotlib, `tsdynamics[plot]`):** trajectory/phase portraits (2D/3D),
  orbit diagrams, Poincaré sections, basins heatmaps (attractors overlaid),
  continuation curves, recurrence plots, embedding diagnostics, spectrum plots.
  One consistent style; every analysis result type knows how to draw itself
  (`result.plot(ax=...)`).
- **Interactive (`tsdynamics[interactive]`, plotly or pyqtgraph — decide by
  prototype):** live trajectory explorer with parameter sliders, zoomable
  orbit diagram with on-demand recomputation, click-to-seed Poincaré explorer,
  cobweb diagrams for 1D maps, animated basin/continuation evolution.
- Notebook-first: everything must look right in Jupyter; GUI windows optional.

**Acceptance:** docs gallery with ≥ 15 one-snippet plots; interactive orbit
diagram zoom-recompute demo.

---

## 7. Phase 7 — Ecosystem & credibility

- **Docs:** mkdocs-material site — tutorial ("from equations to basins in 10
  minutes"), per-function reference with paper citations, theory notes, gallery,
  benchmark page. Docstring citations enforced by lint rule.
- **Quality:** property-based tests (hypothesis) for embeddings/dimensions;
  cross-validation against published values catalogued in one test module;
  deterministic seeding everywhere (`rng=` argument convention).
- **Distribution:** conda-forge feedstock, versioned changelog, semver.
- **Paper:** JOSS submission once Phases 2–4 land (the DDE Lyapunov + basins +
  Rust backend combination is the headline).
- **Performance marketing:** published, reproducible benchmark notebook vs the
  Julia ecosystem and SciPy — speed *and* time-to-first-result (their JIT warmup
  vs our precompiled wheels).

---

## 8. Differentiators to protect (why we win, not just tie)

1. **DDEs as first-class citizens** — integration *and* Lyapunov spectra. The
   competition has zero DDE support in its unified interface.
2. **Zero warmup** — prebuilt wheels + sub-second Cranelift JIT vs minutes of
   session-start compilation latency.
3. **Simplest system definition in any language** — `params` + `dim` +
   `_equations`, already proven over 149 systems. Never regress this.
4. **Python ecosystem gravity** — NumPy/pandas/sklearn/ML interop for free;
   basin classifiers, surrogate ML models, neural operators all one import away.
5. **149+ built-in systems** with literature defaults, growing.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| DiffSL can't express some RHS (heavy branching) | Numba-cfunc pointer escape hatch; keep symbolic→C fallback until parity proven |
| `differential-equations` crate is young (one maintainer) | vendor the DDE module; cross-validate against JiTCDDE before switching defaults; upstream fixes |
| Basin/continuation phase is research-grade work | implement against published reference results; keep scope per-release small (finder → basins → continuation) |
| Scope explosion (550-function target) | phases gated by acceptance criteria; analysis functions land only with literature-validated tests |
| Breaking users during backend swap | `backend=` flag + one deprecation cycle; trajectory cross-validation suite in CI |

## 10. Suggested order of attack

1. **Now:** Phase 1 (protocol + derived systems + Trajectory + housekeeping) — pure
   Python, immediately unlocks Phase 3 features even on the old backend.
2. **Parallel track:** Phase 2 prototype — SymEngine→DiffSL translator + pydiffsol
   spike on Lorenz/Rössler/Robertson to de-risk before writing our own bindings.
3. Phase 3 → 4 → 5 in order (4 is the moat; 5 is parallelizable among contributors).
4. Phase 6 starts as soon as Phase 3 produces plottable results (gallery-driven
   development).
5. Phase 7 continuous, JOSS after 4.
