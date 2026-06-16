# CLAUDE.md — TSDynamics

Architecture, conventions and patterns for any code change in this repo.
Keep this file in sync with the code — if you rename a method or attribute,
update this doc in the same PR.

---

## Project overview

**TSDynamics** is a Python library for studying dynamical systems. As of v3.0.0
(milestone **M3**) the **Rust engine (`tsdynamics._rust`) is the sole integration
backend** — the v2 backends (JiTCODE / JiTCDDE / Numba / diffsol) are gone. It
provides:

- ODE / DDE / SDE integration and discrete-map iteration on the Rust engine
  (`ContinuousSystem`, `DelaySystem`, `StochasticSystem`, `DiscreteMap`), reached
  through `backend="interp"` (the SSA-tape interpreter, default), `"jit"` (the
  Cranelift JIT), or `"reference"` (a dependency-light pure-Python SciPy oracle,
  ODE + maps only)
- A uniform stepping protocol (`System`) implemented by all families
- Derived-system wrappers (`PoincareMap`, `StroboscopicMap`, `TangentSystem`,
  `EnsembleSystem`, `ProjectedSystem`)
- An analysis toolkit (`orbit_diagram`, `poincare_section`, Lyapunov tools,
  `fixed_points`)
- A runtime registry of all systems powering bulk tests and auto-generated docs

The user defines the math (one symbolic `_equations` method, or `_step` +
`_jacobian` for maps); the library lowers it to an engine IR tape (via SymEngine)
and handles integration, output grids, and documentation. There is no
compilation/warmup step and no on-disk compile cache.

**Author:** Daniel Estevez
**Python:** ≥ 3.12
**Package manager:** uv
**License:** MIT

---

## Repository layout

The v3 modular layout (stream F3 reorg; old `base/`, `backends/`, `sampling.py`
paths have MOVED, no shims):

```
src/tsdynamics/
├── __init__.py               # __version__ (managed by python-semantic-release) + re-exports
├── registry.py               # system registry (SystemEntry + all_systems/…) + reserved generic analyses/transforms (solvers live in tsdynamics.solvers)
├── families/                 # base classes + the System protocol (was base/)
│   ├── base.py               # SystemBase, ParamSet, MetaStore (re-exports Trajectory from data)
│   ├── protocol.py           # the System runtime Protocol
│   ├── continuous.py         # ContinuousSystem (engine integrate + jacobian autogen)
│   ├── delay.py              # DelaySystem (engine method-of-steps, forward-only)
│   ├── discrete.py           # DiscreteMap (engine iterate + signature validation)
│   └── stochastic.py         # StochasticSystem — diagonal-Itô SDEs (_drift+_diffusion; EM/Milstein)
├── engine/                   # Rust-facing engine layer; tsdynamics._rust is the sole backend
│   ├── symbols.py            # engine-native symbolic frontend: state_time_symbols() → (Function("y"), Symbol("t"))
│   ├── compile.py            # symbolic dynamics → IR Tape (all families) + reference evaluator
│   ├── problem.py            # per-family Problem builders bundling a tape + runtime context
│   └── run.py                # backend select (interp|jit|reference) + integrate/ensemble + solver resolve/with_jacobian wiring
├── solvers/                  # F2 registry mechanism + C-SOLV in-tree specs (explicit/implicit/stochastic) + method= resolution/aliases + auto-stiffness (select.py)
├── derived/
│   ├── _base.py              # DerivedSystem (wrapper base, with_params rebuilds)
│   ├── poincare.py           # PoincareMap (Hermite-refined crossings)
│   ├── stroboscopic.py       # StroboscopicMap
│   ├── tangent.py            # TangentSystem (Lyapunov engine)
│   ├── ensemble.py           # EnsembleSystem
│   └── projected.py          # ProjectedSystem
├── data/                     # state-space geometry + trajectory lingua franca (was sampling.py)
│   ├── trajectory.py         # Trajectory (canonical home; re-exported via families + top level)
│   └── sampling.py           # Box/Ball/Grid, sampler, grid_points, set_distance
├── analysis/                 # quantifiers, one subpackage per A-* stream (A-LAYOUT reorg)
│   ├── __init__.py           # flat re-exports (public API) + analyses plugin discovery
│   ├── orbits/               # A-ORBIT: orbit_diagram + OrbitDiagram (+ periods/bifurcation_points; orbit_diagram.py); poincare_section (poincare.py); return_map + ReturnMap (first-return/next-amplitude map; return_map.py); self-registers into registry.analyses
│   ├── lyapunov/             # A-LYAP: lyapunov_spectrum, max_lyapunov, kaplan_yorke_dimension + lyapunov_from_data (Kantz/Rosenstein, from_data.py); self-registers into registry.analyses
│   ├── fixedpoints/          # A-FP: fixed_points/FixedPoint (maps+flow equilibria, Newton/SD/DL — fixed.py), periodic_orbits/periodic_orbit/PeriodicOrbit + estimate_period (periodic.py), shared primitives (_common.py); self-registers
│   ├── dimensions/           # A-DIM: correlation/generalized-Rényi/fixed-mass fractal dims + scaling-region fit
│   ├── entropy/              # A-ENT: permutation/dispersion/sample/multiscale entropy + LZ76 (composable OutcomeSpace×estimator×measure)
│   ├── chaos/               # A-CHAOS: GALI_k (Skokos) + 0–1 test (Gottwald–Melbourne) + expansion entropy (Hunt–Ott); maps via _jacobian, flows via self-contained RK4 variational core (no engine/compile)
│   ├── recurrence/          # A-RQA: recurrence_matrix (fixed ε / target rate, sparse cKDTree) + rqa (DET/LAM/L_max/ENTR/TT) + windowed_rqa; self-registers into registry.analyses
│   ├── surrogate/           # A-SURR: surrogates (shuffle/FT/AAFT/IAAFT generators) + time_reversal_asymmetry/nonlinear_prediction_error stats + surrogate_test→SurrogateTest (rank p + sigma); self-registers into registry.analyses
│   ├── basins/              # A-BASIN: find_attractors/basins_of_attraction (recurrence-FSM AttractorMapper) + basin_fractions (basin stability) + basin_entropy/uncertainty_exponent/wada_property (boundary structure) + continuation/tipping_points + resilience; cell tessellation in _common.py; self-registers into registry.analyses
│   └── embedding/           # owned by A-EMBED
├── transforms/               # signal/feature transforms (stream T-XFORM): spectral.py (PSD/entropy/centroid/dominant freq), preprocessing.py (detrend/normalize/Butterworth filters), features.py (FEATURE_FUNCTIONS + extract_features/Hjorth), _common.py (Trajectory↔array coercion + fs/dt resolution); self-register into registry.transforms
├── viz/                      # DEFERRED stub only (decision D6)
├── systems/
│   ├── continuous/           # 8 ODE category modules + delayed_systems.py (DDEs!)
│   └── discrete/             # 5 map category modules
└── utils/
    ├── general.py            # staticjit decorator
    └── sagitta_dt.py         # estimate_dt_from_sagitta

hooks/docs_autogen.py          # mkdocs hook: per-system pages + figures at build time
docs/_tooling/equations.py     # symbolic → LaTeX rendering for docs
docs/_tooling/figures.py       # cached figure rendering (scipy for ODEs)
tests/_sampling.py             # curated slow-tier sample + DDE histories + exclusions
```

---

## Public API surface

Built-in system classes are **not** in the top-level `__all__` (nor `dir()`):
they live under `tsdynamics.systems` — the canonical path is
`tsdynamics.systems.<Name>` (e.g. `tsdynamics.systems.Lorenz`, flat across
`continuous`/`discrete`), kept off the top level so the submodules stay findable.
For backwards compatibility a module-level `__getattr__` still resolves
`tsdynamics.Lorenz` / `from tsdynamics import Lorenz` lazily. `systems/__init__.py`
flat-re-exports every catalogue class automatically (driven by each category
module's `__all__`), so a new system needs no manual edit there.

`tsdynamics.__all__` exports:

- The 149 built-in systems are reachable via `tsdynamics.systems` (149 today:
  118 ODE + 5 DDE + 26 maps), not the top-level `__all__`
- Base classes: `ContinuousSystem`, `DelaySystem`, `DiscreteMap`,
  `StochasticSystem`; result type `Trajectory`
- Derived wrappers: `PoincareMap`, `StroboscopicMap`, `TangentSystem`,
  `EnsembleSystem`, `ProjectedSystem`
- Analysis: `orbit_diagram`, `OrbitDiagram` (+ `.periods()` /
  `.bifurcation_points()` cascade quantifiers), `poincare_section`, `return_map`,
  `ReturnMap` (A-ORBIT: first-return / next-amplitude map — Lorenz z-maxima cusp
  + Poincaré-crossing variant),
  `lyapunov_spectrum`, `max_lyapunov`, `kaplan_yorke_dimension`,
  `lyapunov_from_data`, `LyapunovFromData` (A-LYAP: maximal exponent from a
  time series, Kantz/Rosenstein),
  `fixed_points`, `FixedPoint` (A-FP: maps *and* flow equilibria, Newton +
  Schmelcher–Diakonos/Davidchack–Lai), `periodic_orbits` (map period-p orbits),
  `periodic_orbit` (flow single shooting), `PeriodicOrbit`, `estimate_period`;
  fractal dimensions (A-DIM)
  `correlation_dimension`, `correlation_sum`, `generalized_dimension`,
  `box_counting_dimension`, `information_dimension`, `dimension_spectrum`,
  `fixed_mass_dimension`, `DimensionResult`; chaos indicators (A-CHAOS)
  `gali`, `GALIResult`, `zero_one_test`, `expansion_entropy`,
  `ExpansionEntropyResult`; recurrence & RQA (A-RQA) `recurrence_matrix`,
  `RecurrenceMatrix`, `rqa`, `RQAResult`, `windowed_rqa`, `WindowedRQA`;
  surrogates & nonlinearity tests (A-SURR) `surrogates`, `random_shuffle`,
  `fourier_surrogate`, `aaft_surrogate`, `iaaft_surrogate`,
  `time_reversal_asymmetry`, `nonlinear_prediction_error`, `surrogate_test`,
  `SurrogateTest`; attractors & basins (A-BASIN) `find_attractors`,
  `basins_of_attraction`, `basin_fractions`, `basin_entropy`,
  `uncertainty_exponent`, `wada_property`, `continuation`, `tipping_points`,
  `resilience`, `Attractor`, `AttractorSet`, `BasinsResult`, `BasinFractions`,
  `BasinEntropy`, `UncertaintyExponent`, `WadaResult`, `ContinuationResult`
- Derived: `WrappedSystem` (adapt any external stepper to the protocol)
- State-space geometry (`data`): `Box`, `Ball`, `Grid`, `sampler`,
  `grid_points`, `set_distance` — the primitives the basin/attractor layer
  builds on (Monte-Carlo + full-grid sampling, attractor-matching distances)
- Submodules: `analysis`, `data`, `derived`, `families`, `registry`,
  `systems`, `utils`

Reachable but not top-level: `SystemBase`, `ParamSet`, `MetaStore`, `System`
(protocol) via `tsdynamics.families`; `staticjit` via `tsdynamics.utils`.
The engine layer (`tsdynamics.engine`) and the `transforms` / skeleton `viz`
packages are importable (`from tsdynamics import transforms`) but not advertised
in the top-level `__all__`; `transforms` carries its own flat public surface
(`tsdynamics.transforms.__all__`).

---

## The registry (load-bearing!)

`registry.py` hosts the specialised *system* registry (below) plus two
generic name→object `Registry` containers — `registry.analyses` and
`registry.transforms` — for the analysis/transform streams to register into.
Out-of-tree plugins are wired in (A-LAYOUT): `tsdynamics.analysis`/
`tsdynamics.transforms` call `plugins.register_entry_points` at import to load
the `tsdynamics.analyses`/`tsdynamics.transforms` entry-point groups; in-tree
analyses/transforms self-register from their own subpackages (the A-* streams).
**Solvers are not registered here**: they live in the
richer `tsdynamics.solvers` registry (a `name → SolverSpec` table with
capability flags + `solvers/` directory and entry-point discovery via
`plugins.py`, stream F2). Do not re-add a `solvers` registry to `registry.py`.
**Family detection keys off the module
prefix `tsdynamics.families`** (the `_BASE_PREFIX` in `registry.py` and the guard
in `SystemBase.__init_subclass__`) — both moved from `tsdynamics.base` in the F3
reorg; keep them in lock-step if the families package ever moves again.

Every concrete `SystemBase` subclass auto-registers at class-definition time
(`SystemBase.__init_subclass__` → `registry.register_class`). `SystemEntry`
records name/cls/family/category/dim/params/reference/known_lyapunov.

- `registry.all_systems(family=, category=, builtin=True)` — iteration default
  is builtin-only (module under `tsdynamics.systems`); user classes register
  as non-builtin.
- **Family detection walks the MRO** (`DiscreteMap` → map, `DelaySystem` → dde,
  `StochasticSystem` → sde, `ContinuousSystem` → ode) — NOT the module path; the
  DDE systems live in `systems/continuous/delayed_systems.py`.  A class is
  *concrete* (registrable) when it defines `_equations` / `_step` / `_drift`
  outside the framework bases (`_has_concrete_rhs`); `_drift` is the SDE marker.
  There are no built-in SDE systems yet, so `registry.families()` stays
  `{'ode': …, 'dde': …, 'map': …}` for builtins, but a user `StochasticSystem`
  subclass now registers (non-builtin) with family `sde`.
- Duplicate builtin class names raise at import.
- Consumers: registry-driven test parametrization (`tests/conftest.py`
  `pytest_generate_tests`), the docs autogen hook, and users.

Optional per-system metadata ClassVars: `variables` (component names →
`traj["x"]`, docs labels), `reference` (literature citation shown in docs),
`known_lyapunov` (drives `tests/test_known_values.py`; keys: `spectrum`+`atol`,
or `n_positive`, plus `params`/`ic`/`kwargs`/`source`).

---

## Base classes

### `SystemBase` (`families/base.py`)

As before (ParamSet with fixed keys, attribute forwarding, `copy()` /
`with_params()`, `resolve_ic()` priority: arg > self.ic > default_ic > random)
plus:

- `meta` is now a **`MetaStore`** — dict-like, but writes append with history:
  `meta.record(key, value, **context)`, `meta[key]` → latest,
  `meta.history(key)` → all records. `meta == {}` still works.
- `_provenance(**extra)` builds the dict attached to `Trajectory.meta`.
- **Engine-dispatch seam (stream C-FAM):** `_default_backend` ClassVar +
  `_dispatch(backend=, **kwargs)`. Every family's `interp` / `jit` / `reference`
  integration branch funnels through `_dispatch` → `engine.run.integrate`, so the
  FFI marshalling, divergence guards and engine-path provenance live once in
  `run.integrate` instead of being re-implemented per family. `_default_backend`
  is each family's default integrator — `"interp"` (the Rust engine interpreter)
  for every concrete family; the abstract `SystemBase` keeps `"reference"` (the
  wheel-free oracle). Passing `backend=None` to a family's `integrate` /
  `iterate` resolves to it. `run.integrate` also resolves the `method=` string
  through the solver registry (`solvers.resolve`) — canonicalising spellings and
  aliases (`"RK45"`/`"dopri5"` → `"rk45"`) and rejecting v2-only names (`"LSODA"`)
  with a stiff hint — and rebuilds the ODE tape `with_jacobian=True` when an
  implicit kernel needs it. The output grid each
  family samples on is the one hoisted `tsdynamics.utils.grids.make_output_grid`
  (the four byte-identical `_make_t_eval` copies are gone). **SDEs are the
  exception** — they keep the dedicated `run.sde_integrate_dense` /
  `run.sde_ensemble_final` seam (`run.integrate` cannot carry the noise
  seed/step and refuses an SDE).

### `Trajectory` (`data/trajectory.py`)

Lives in `tsdynamics.data` — it is a *data* type the families produce, not a
family itself. Re-exported from `families.base` / `tsdynamics.families` and the
top level, so `from tsdynamics import Trajectory` and `from tsdynamics.data
import Trajectory` are the same object.

- Named components when the class declares `variables`: `traj["x"]`,
  `traj[["x","z"]]`, `traj.component("x")`.
- Point-set ops: `minmax()`, `standardize()`, `neighbors(q, k)` (lazy KD-tree).
- `meta` carries provenance (system, params, solver, dt, tolerances, ic,
  version); preserved through slicing/`after()`.

### The `System` protocol (`families/protocol.py`)

All three families + all derived wrappers implement:
`step(n_or_dt) -> new state`, `state()`, `set_state(u)`, `time()`,
`reinit(u, *, t, params)`, `trajectory(...)`, `is_discrete`.

- First `step()`/`state()` on a cold system does an implicit `reinit()`.
- ODE: `reinit` lowers the system to an engine tape once; each `step(dt)`
  integrates one `dt` chunk through `run.integrate` from the live state.
- **DDE `set_state` raises** (state is a history function); `reinit(u)`
  restarts from a constant past. DDE stepping is forward-only (each `step`
  re-integrates from the constant past via the method of steps).
- Map `step(n)` runs the per-call pure-Python `_step` loop (silencing NumPy FP
  warnings so an overflow surfaces as the explicit divergence `RuntimeError`).
- Param changes after `reinit` need a new `reinit` to reach a live stepper.

### `ContinuousSystem` extras

- **Jacobian autogen**: `jacobian_sym()` (SymEngine `diff` wrt `y(j)`),
  `jacobian(u, t)` numeric via cached `symengine.Lambdify`, `_rhs_numeric()`
  (fast numeric RHS — used by figures, Poincaré Hermite refinement, and the
  reference-backend cross-validation). Hand-written `_jacobian` on ODE systems is
  never used at runtime; it is cross-checked against autogen in tests. `abs`/`sign`
  derivatives are resolved a.e. (`_resolve_derivative_nodes`).
- `integrate(backend=)` defaults to `_default_backend` (`"interp"`). `"interp"`
  / `"jit"` / `"reference"` route through the shared C-FAM seam (`_dispatch` →
  `engine.run.integrate`) to the Rust engine (or its pure-Python oracle).
  `run.integrate` resolves `method=` through the solver registry and lowers the
  tape `with_jacobian=True` for the implicit stiff kernels (`bdf` /`rosenbrock` /
  `trbdf2`); stiff catalogue systems declare `_default_method = "bdf"`.

### `DiscreteMap` extras

- `__init_subclass__` validates that `_step`/`_jacobian` positional parameter
  names match the `params` dict order — mismatches raise `TypeError` at
  import (also catches re-ordered `params` in subclasses).
- `_jacobian_fd_check = False` ClassVar opts a map out of the
  finite-difference Jacobian test (only for orbits living on discontinuities,
  e.g. Baker).
- `iterate(backend=...)` runs the iteration on the Rust engine (`"interp"`
  default / `"jit"` / `"reference"` pure-Python oracle). The engine loop lives in
  `crates/tsdyn-engine/src/map.rs`; all backends lower `_step` to the IR, so
  piecewise/`numpy`-ufunc steps raise `TapeCompileError`. The engine path
  diverges loudly (raises); the random-IC retry still applies when `iterate` is
  called without an explicit `ic`.

### `StochasticSystem` extras

- **Diagonal-Itô SDE** family (`families/stochastic.py`):
  `dX_k = f_k dt + g_k dW_k` with independent `dW_k`. Subclass contract is
  `_drift(y, t, **params)` (like `_equations`) + `_diffusion(y, t, **params)`
  (one noise coefficient per component); both symbolic, both lower via
  `engine.compile.lower_sde` (drift tape + diffusion tape, the latter carrying
  `∂g/∂u` for Milstein).
- `integrate(..., method=, seed=, backend=)` runs a fixed-step scheme — `dt` *is*
  the noise scale `√dt` (so `dt` sets both the discretisation and the output grid).
  `method`: `"euler_maruyama"` (order 0.5, default) or `"milstein"` (order 1.0).
  `seed` makes the noise realisation reproducible (recorded in `traj.meta`).
  `backend`: `"reference"` (default, pure Python) or `"interp"`/`"jit"` — the
  compiled engine via `tsdynamics._rust` (stream E-WIRE).
- `ensemble(ics, ..., backend=)` seeds trajectory `i` from `seed_for(seed, i)` —
  depending only on the index — so a batch is reproducible and mirrors the Rust
  engine's parallel-equals-serial contract; a diverged trajectory becomes a `NaN`
  row. `backend="interp"/"jit"` fans the batch out on the engine's rayon pool.
- The real engine is Rust: kernels in `crates/tsdyn-solvers/src/sde/**`
  (own `SdeKernel` trait, RNG-free — the engine hands them a pre-drawn `dw`),
  loop + seeded RNG in `crates/tsdyn-engine/src/sde.rs`. The two-tape SDE FFI
  (`integrate_sde_dense` / `integrate_sde_ensemble_final` in `tsdyn-core`) is
  wired (stream E-WIRE): `backend="interp"/"jit"` (the default) dispatches the
  drift+diffusion call to the engine (interpreter or Cranelift JIT). The
  pure-Python **reference** integrator (a faithful `SplitMix64` port) reproduces
  it bit-for-bit under a fixed seed and stays available as the wheel-free oracle.
- **Registry-detected (stream C-FAM):** a concrete `StochasticSystem` subclass
  registers with family `sde` (`_drift` is the concrete-rhs marker, and
  `StochasticSystem` is in the family-base table), so it appears in
  `registry.all_systems(family="sde")`. There are no built-in SDE systems in the
  catalogue yet, so `registry.families()` (builtin-only) is unchanged; a future
  built-in SDE needs an entry in `tests/_sampling.py::SDE_SAMPLES` (a guard test
  enforces completeness, mirroring `DDE_HISTORIES`).
- **The SDE engine path does not use `run.integrate`** (it cannot carry the noise
  seed/step): `backend="interp"/"jit"` dispatch through the dedicated
  `run.sde_integrate_dense` / `run.sde_ensemble_final` seam, and `run.integrate`
  /`run.ensemble` *refuse* an SDE problem.

---

## Derived systems & analysis

- Wrappers forward `params`/`meta`; `with_params()` re-parametrizes the inner
  system and rebuilds the wrapper → orbit diagrams over `PoincareMap` /
  `StroboscopicMap` are bifurcation diagrams of flows (ODE control-param
  caching keeps sweeps cheap; DDE sweeps recompile per value).
- `PoincareMap` refines crossings with cubic Hermite using `_rhs_numeric`
  (O(dt⁴)); falls back to linear interpolation for DDEs.
- **`TangentSystem` is the one Lyapunov engine** (stream C-DERIV): the
  variational/QR machinery lives here, and `DiscreteMap.lyapunov_spectrum` /
  `ContinuousSystem.lyapunov_spectrum` are thin delegations to it. Modes:
  - **maps**: NumPy `W ← J(x)·W` + QR, Jacobian at the **pre-image** `x_n` (the
    correct tangent-map convention), with random-IC retry on divergence.
  - **ODEs**: the **backend-neutral** engine path via `backend=`
    (`"interp"`/`"jit"`/`"reference"`): the *extended* variational ODE (state ⊕ k
    tangent vectors, built in `derived/_variational.py` and lowered via the
    public `engine.compile.lower_expressions`) is integrated per dt-chunk through
    `engine.run.integrate` then QR-reorthonormalised. `backend="reference"`
    validates it against analytic spectra without the compiled wheel;
    `backend="jitcode"` (and any other legacy name) raises
    `ValueError("unknown ODE tangent backend")`.
  - **DDEs**: raise — their tangent space is the infinite-dimensional history
    space; use `DelaySystem.lyapunov_spectrum` (NOT routed through
    `TangentSystem`), the engine estimator described below.
  `TangentSystem.lyapunov_spectrum(...)` wraps the streaming `step()`/
  `exponents()` API into the standard burn-in + time-weighted estimate.
- **`DelaySystem.lyapunov_spectrum(backend="interp"/"jit")`** (E-DDE-LYAP) is the
  engine DDE Lyapunov estimator (`families/_dde_lyapunov.py`), the
  infinite-dimensional-history analogue of the ODE variational core: it builds
  the **extended** DDE — base state ⊕ `k` deviation states, the deviation
  equations being the symbolic variational dynamics (a per-current-state Jacobian
  plus one Jacobian per delay slot, so delayed deviations are just extra delay
  slots — **the frozen IR is untouched**) — and integrates it on the Rust DDE
  engine in chunks of one delay window. Benettin renormalisation is over the
  deviation **history segment** (a function-space QR, so `n_exp` may exceed
  `dim`); with chunk `= τ_max` and `dt | τ_max` the base history is reused exactly
  (no reseed-interpolation error) and the deviation directions are recombined
  exactly (the variational dynamics is linear). Validated (reference-free) on all
  5 built-in DDEs (and a 2-D synthetic DDE): descending spectrum that brackets 0,
  Mackey–Glass leading exponent positive (matching its `known_lyapunov`
  `n_positive=1`), `interp`==`jit` bit-for-bit. (The original Rust-vs-`jitcdde`
  parity gate ran before JiTCDDE was removed.) `backend="reference"` raises (no
  pure-Python DDE integrator); `"interp"`/`"jit"` only.
- `max_lyapunov` (Benettin two-trajectory) needs `set_state` → raises for DDEs.
  Its continuous normalization divides by the **measured elapsed `time()`** of
  the reference run (not a guessed step-size attribute), so it is correct for
  any continuous system including `WrappedSystem` stepped with `dt=None`.
- `lyapunov_from_data` (A-LYAP) estimates the maximal exponent from a measured
  series via delay embedding + neighbour divergence (Kantz 1994 default,
  Rosenstein et al. 1993 optional); returns a `LyapunovFromData` carrying the
  stretching curve `S(k)` — fit the linear scaling region (inspect, then pass
  `fit=(lo, hi)`). A private delay-embed helper keeps it independent of the
  delay-embedding stream.
- `fixed_points` (A-FP) finds map fixed points (`f(x)=x`) **and** flow equilibria
  (`f(x)=0`) by multi-start Newton on the analytic Jacobian; `method="sd"`/`"dl"`
  add the Schmelcher–Diakonos/Davidchack–Lai stabilising transformations (maps
  only) to reach unstable points. Map stability is `|λ|<1`, flow stability
  `Re λ<0` (the `FixedPoint.continuous` flag picks the convention).
- `periodic_orbits` (A-FP) finds map period-`p` orbits as fixed points of `fᵖ`
  (Davidchack–Lai by default), with a minimal-period (`prime`) filter and
  cyclic-shift dedup. `periodic_orbit` finds a flow limit cycle by single
  shooting on `(x0, T)` (bordered Newton + monodromy via the RK4 variational
  core; Floquet multipliers for stability, the trivial ≈1 multiplier found by
  eigenvector alignment with `f(x0)`; rejects equilibrium-collapse on a centre).
  `estimate_period` reads a signal's period (autocorrelation/FFT) to seed
  shooting. All A-FP routines are backend-free (fast tier), self-contained in
  `analysis/fixedpoints/` (own `_common.py`), and self-register into
  `registry.analyses`.
- `orbit_diagram` (A-ORBIT) sweeps a parameter of any discrete view (a
  `DiscreteMap`, or a flow wrapped in `PoincareMap` / `StroboscopicMap` → a
  bifurcation diagram) recording the asymptotic orbit; `OrbitDiagram.periods()`
  /`.bifurcation_points()` quantify the cascade (scale-free branch clustering;
  logistic onsets land on `r₁=3`, `r₂=1+√6`). `return_map` builds the
  first-return / next-amplitude map of a recurring observable — successive
  extrema (`kind="max"/"min"`, the Lorenz z-maxima cusp, parabolically sharpened)
  or successive Poincaré crossings (`kind="poincare"`) — from a System,
  `Trajectory`, or bare 1-D series. `poincare_section` gives root-refined
  crossings from a system or interpolated crossings from data. The orbits
  subpackage is backend-free (extrema/sweeps over the standard stepping API) and
  self-registers into `registry.analyses`.
- `find_attractors` / `basins_of_attraction` (A-BASIN) drive any map/flow over a
  `CellGrid` tessellation with a recurrence finite-state machine (the
  `AttractorMapper`, Datseris–Wagemakers 2022): a trajectory that recurrently
  re-visits cells has found an attractor, transient cells become its basin, and a
  near-coincident split is proximity-merged (`merge_tol`). Flows step by `dt` per
  cell check, maps by one iteration; a raised/non-finite step is divergence, a
  finite out-of-box excursion uses the lost-counter. `basins_of_attraction` paints
  a `Grid` (pass a separate `recurrence` box to image a *slice* of a higher-dim
  flow — the magnetic pendulum); `basin_fractions` is Monte-Carlo basin stability
  (Menck 2013). The metrics read a label image (no integration, fast tier):
  `basin_entropy` (Daza 2016 `Sb`/`Sbb`, `Sbb>ln2` ⇒ fractal), `uncertainty_exponent`
  (Grebogi 1983, `D₀=D−α`; `as_label_array` squeezes degenerate slice axes so the
  dimension is right), `wada_property` (Daza 2015 grid test), `resilience`
  (Halekotte–Feudel 2020 distance-to-boundary via EDT). `continuation` re-finds +
  matches attractors across a parameter by `set_distance` (greedy nearest, RAFM
  Datseris 2023; `min_fraction` drops saddle-passage spurious sets), and
  `tipping_points` reads off where a basin annihilates. Validation systems
  (Newton z³ map ⅓-basins, two-well Duffing ½-basins, magnetic pendulum) live in
  `tests/test_basins.py` — they are TEST-LOCAL, not catalogue systems.
  Self-registers into `registry.analyses` (family `basins`).

---

## Code conventions

- **Formatter/Linter:** `ruff format` / `ruff check` (line length 100; D rules on)
- **Docstrings:** NumPy convention; cite original papers, never competitor software
- **Never reference the Julia dynamical-systems ecosystem** in code, docs, or
  comments — ideas may be absorbed, citations go to the original literature.
- **Commits:** Conventional Commits; PRs are squash-merged and the PR title
  becomes the release-deciding commit (enforced by `pr-title.yml`)

Run before pushing:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest -m "not slow" --no-cov   # fast tier
uv run pytest --no-cov                 # + slow tier (compiles)
uv run pytest -m full --no-cov         # exhaustive sweep (nightly in CI)
TSD_DOCS_FIGURES=0 uv run mkdocs build --strict   # docs sanity
```

### Test harness (stream I-QA)

The suite has three layers, all **registry-driven where possible** so new
systems/analyses join the sweeps with zero test edits:

- **Registry sweeps.** `tests/conftest.py` parametrizes fixtures over the
  registries: `ode_entry`/`dde_entry`/`map_entry`/`sde_entry`/`system_entry`
  (built-in systems) and `analysis_entry`/`transform_entry` (the D4
  `registry.analyses`/`registry.transforms` plugin surface — the latter only
  populates once `tsdynamics.transforms` is imported, which `conftest` does).
  `tests/test_analysis_registry.py` runs the meta-QA over every registered
  analysis/transform (callable, documented, round-trips, top-level export
  agreement) plus headline-membership guards.
- **Property tests (Hypothesis).** `tests/test_property_*.py` assert
  *mathematical invariants* of the analysis/transform layer (embed value
  preservation, PSD non-negativity, permutation-entropy monotone-transform
  invariance, surrogate spectrum/amplitude preservation, recurrence symmetry +
  target-rate calibration, dimension-of-a-d-cube ≈ d, …). `hypothesis` is a dev
  dependency; `conftest` registers a profile (deadline off, health checks
  suppressed — required under `filterwarnings=["error"]`). Shared deterministic
  signal builders live in `tests/_strategies.py` (sinusoid/AR(1)/logistic/Hénon,
  compile-free → fast tier); reproducible by seed so a failing example replays.
- **Known-value catalogue.** `tests/test_known_values.py` (literature Lyapunov
  spectra via the `known_lyapunov` ClassVar) and `tests/test_known_quantifiers.py`
  (analytic identities + cross-quantifier "regular vs random" agreement: five
  independent complexity measures must concur). Per-stream literature numbers
  stay in each stream's own test file; `test_known_quantifiers.py` does not
  duplicate them.

When adding an analysis/transform, the registry meta-QA picks it up
automatically (give it a docstring, register it). When adding a property test,
reuse `_strategies` and assert a real invariant — never a tautology.

---

## Versioning & release (python-semantic-release)

- `__version__` lives in `src/tsdynamics/__init__.py` **and** `[project].version`
  in `pyproject.toml`; PSR rewrites both (`version_variables` + `version_toml`).
  The static `[project].version` is required because the build backend is
  `maturin` (it cannot read a Python `__version__`).
- Every push to `main` runs `release.yml`: `test` (builds the engine + runs the
  suite) → PSR computes the bump from conventional commits (feat→minor,
  fix/perf→patch, `!`→major) and tags vX.Y.Z + creates the GitHub Release (PSR
  does **not** build — `build_command=""`) → `wheels`/`sdist` build the
  per-platform abi3 wheels from the new tag → `publish` uploads them to PyPI via
  trusted publishing (`environment: pypi`, bound to the filename `release.yml` —
  don't rename) and attaches them to the Release.
- CHANGELOG.md is maintained by python-semantic-release; release notes also land on GitHub Releases.
- Workflows: `ci.yml` (PR gate — installs Rust + builds the engine via
  `uv sync`, runs the suite), `docs.yml` (build + Pages deploy with figure cache),
  `release.yml`, `pr-title.yml`, `nightly.yml` (`-m full`), `rust-workspace.yml`
  (the pure-Rust tsdyn-* workspace), `engine-bindings.yml` (the focused engine job:
  tsdyn-core fmt/clippy/cargo-test + the **engine-marked** Python tests, including
  the catalogue gate `tests/test_xval_catalogue.py`), `wheels.yml` (cross-platform
  abi3 wheel build smoke on packaging PRs + on-demand full matrix, artifacts only —
  the release build/publish lives in `release.yml`).
- **The `engine` marker (stream I-XVAL):** any test module that imports the
  compiled `tsdynamics._rust` extension is auto-tagged `engine` by a
  `conftest.pytest_collection_modifyitems` hook (detection in
  `tests/_engine_marker.py`), so `engine-bindings.yml` selects them with
  `-m "engine and not full"` instead of a hand-maintained file list — a new
  engine test joins the job with zero CI edits. `tests/test_engine_coverage.py`
  guards the invariant. These engine tests `importorskip("tsdynamics._rust")`, so
  they still skip cleanly anywhere the extension is absent.
- **Packaging shape (ROADMAP §11):** the project ships as **one maturin wheel** —
  the pure-Python `tsdynamics` package (from `src/`, `python-source="src"`) plus
  the compiled `tsdynamics/_rust` abi3 extension (`module-name="tsdynamics._rust"`,
  `manifest-path="crates/tsdyn-core/Cargo.toml"`) in the same wheel. abi3 (cp312)
  means one wheel per (platform, arch) covers every CPython ≥ 3.12. Full rationale
  + recipe: `docs/theory/packaging.md`; invariants guarded by
  `tests/test_packaging.py`.

---

## Adding a new system

1. Drop the class into the right module under `systems/continuous/` or
   `systems/discrete/`, following the family contract (see docs → Start →
   concepts, or any existing system).
2. Add the class name to the module's `__all__` and the category package's
   `__init__.py` (`systems/continuous/__init__.py` or `systems/discrete/__init__.py`).
   `systems/__init__.py` flat-re-exports it automatically (→ `tsdynamics.systems.<Name>`
   and the lazy `tsdynamics.<Name>`); a registry test fails loudly if you forget the
   category `__all__`.
3. **That's it** — the registry picks it up, the bulk test suite sweeps it,
   and the docs build generates its page (equations + figure) automatically.
4. Optional metadata: `variables`, `reference`, `known_lyapunov` ClassVars;
   `default_ic` if random ICs escape the basin; `_structural_params` for
   variable-dim systems; `_jacobian_fd_check = False` for discontinuous maps;
   `_default_method = "bdf"` for a stiff ODE.
5. For a new DDE: also add a non-equilibrium history to
   `tests/_sampling.py::DDE_HISTORIES` (guard test enforces this).  For a new
   *built-in* SDE (`StochasticSystem` subclass): add a `{"seed":…, "ic":…}` entry
   to `tests/_sampling.py::SDE_SAMPLES` (the `sde`-family guard test enforces it).

---

## No compilation cache (zero warmup)

The engine lowers each system to an in-process IR tape on first use — there is
**no on-disk compile cache** and no C-compilation step (the old
`~/.cache/tsdynamics/` JiTCODE/JiTCDDE cache and the `TSDYNAMICS_CACHE` override
are gone). Editing an `_equations`/`_step` body just takes effect on the next
run; nothing to wipe.

- Control parameters are read live from the system on every run, so a
  parameter change never re-lowers. A *delay* value (DDE) or a structural
  parameter is baked into the tape, so changing one re-lowers.
- The docs figure cache is unrelated and still exists: `.cache/docs-figures`,
  keyed by class source hash (CI persists it via actions/cache).

---

## Common pitfalls

| Situation | What happens / what to do |
|---|---|
| `_equations` uses NumPy or `math` | The engine tape can't lower it. Use `symengine.sin`/`cos`/... |
| Variable-dim system without `_structural_params` | Lowering-time `range(N)` fails. Add `_structural_params = frozenset({"N"})`. |
| Map params order ≠ `_step` signature order | **Raises `TypeError` at import**. |
| DDE with constant past at a fixed point | Lyapunov exponents ≈ 0. Provide a non-equilibrium `history`. |
| Tight tolerances on DDE | `rtol=atol=1e-3` is the safe start. |
| `set_state` on a DDE | Raises by design — use `reinit(u)`. |
| Stiff ODE: which method? | `"bdf"` is the **variable-order (1–5) BDF** and the right default for stiff ODEs (far faster than the fixed-order `rosenbrock`/`trbdf2`, which stay selectable). `run.integrate` auto-builds the Jacobian-carrying tape for the implicit kernels, so `integrate(method="bdf")` "just works". The legacy SciPy name `"LSODA"` is no longer a method — declare `_default_method = "bdf"`. |
| Param change ignored by a live stepper | `reinit()` after parameter changes (or use `with_params`). |
| Orbit diagram over a DDE wrapper | Re-lowers the tape per parameter value — slow by design, document it. |
| New DDE fails `test_dde_histories_complete` | Add its history to `tests/_sampling.py`. |

---

## Quick reference

```python
import numpy as np
import tsdynamics as ts

# ODE
lor = ts.Lorenz()
traj = lor.integrate(final_time=100.0, dt=0.01)
traj["x"]                                   # named component
exps = lor.lyapunov_spectrum(final_time=300.0)   # → [0.91, ~0, -14.57]
ts.kaplan_yorke_dimension(exps)             # → ~2.06

# Backends: "interp" (default) / "jit" (Cranelift) / "reference" (pure-Python oracle)
traj = lor.integrate(final_time=100.0, dt=0.01, backend="jit")

# Protocol stepping
lor.reinit([1.0, 1.0, 1.0])
u = lor.step(0.01)

# Derived systems → analysis composition
pmap = ts.PoincareMap(ts.Rossler(), plane=(1, 0.0))
section = pmap.trajectory(500)
od = ts.orbit_diagram(pmap, "c", np.linspace(2, 6, 50), components=0)

# Maps
h = ts.Henon()
h.iterate(steps=5000)
ts.fixed_points(h)                          # analytic saddles
ts.max_lyapunov(h, ic=[0.1, 0.1])           # ≈ 0.42

# DDE (integrate first, then Lyapunov from the end state)
mg = ts.MackeyGlass()
traj = mg.integrate(final_time=500.0, dt=0.5, history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)])
exps = mg.lyapunov_spectrum(n_exp=1, dt=0.5, ic=traj.y[-1])

# Registry
from tsdynamics import registry
registry.families()                         # {'ode': 118, 'dde': 5, 'map': 26}
```
