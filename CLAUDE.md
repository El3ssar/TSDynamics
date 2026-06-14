# CLAUDE.md — TSDynamics

Architecture, conventions and patterns for any code change in this repo.
Keep this file in sync with the code — if you rename a method or attribute,
update this doc in the same PR.

---

## Project overview

**TSDynamics** is a Python library for studying dynamical systems. It provides:

- Compiled ODE integration via JiTCODE (`ContinuousSystem`), with an
  experimental Rust backend via pydiffsol (`backend="diffsol"`)
- Compiled DDE integration via JiTCDDE (`DelaySystem`)
- Discrete-map iteration via Numba (`DiscreteMap`)
- A uniform stepping protocol (`System`) implemented by all families
- Derived-system wrappers (`PoincareMap`, `StroboscopicMap`, `TangentSystem`,
  `EnsembleSystem`, `ProjectedSystem`)
- An analysis toolkit (`orbit_diagram`, `poincare_section`, Lyapunov tools,
  `fixed_points`)
- A runtime registry of all systems powering bulk tests and auto-generated docs

The user defines the math (one symbolic `_equations` method, or `_step` +
`_jacobian` for maps); the library handles compilation, caching, integration,
output grids, and documentation.

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
│   ├── continuous.py         # ContinuousSystem (was ode_base.py; JiTCODE + jacobian autogen)
│   ├── delay.py              # DelaySystem (was dde_base.py; JiTCDDE, forward-only)
│   ├── discrete.py           # DiscreteMap (was map_base.py; Numba + signature validation)
│   └── stochastic.py         # StochasticSystem — diagonal-Itô SDEs (_drift+_diffusion; EM/Milstein)
├── engine/                   # Rust-facing engine layer (was backends/); E7 adds _rust
│   ├── compile.py            # symbolic dynamics → IR Tape (all families) + reference evaluator (E6)
│   ├── problem.py            # per-family Problem builders bundling a tape + runtime context (E6)
│   ├── run.py                # backend select (interp|jit|reference) + integrate/ensemble (E6)
│   ├── rustcore.py           # v2-seed tape emitter + accelerator wrappers (superseded by compile/run; retired at M3)
│   └── diffsol.py            # experimental SymEngine→DiffSL + pydiffsol (v2 backend; retired at M3)
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
│   ├── orbits/               # orbit_diagram + OrbitDiagram (orbit_diagram.py); poincare_section (poincare.py)
│   ├── lyapunov/             # lyapunov_spectrum, max_lyapunov, kaplan_yorke_dimension
│   ├── fixedpoints/          # fixed_points + FixedPoint (maps, multi-start Newton; was fixed_points.py)
│   └── chaos/ basins/ dimensions/ embedding/ entropy/ recurrence/ surrogate/   # empty, owned by A-* streams
├── transforms/               # signal/feature transforms — skeleton (stream T-XFORM)
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

`tsdynamics.__all__` exports:

- Every built-in system class (149 today: 118 ODE + 5 DDE + 26 maps)
- Base classes: `ContinuousSystem`, `DelaySystem`, `DiscreteMap`,
  `StochasticSystem`; result type `Trajectory`
- Derived wrappers: `PoincareMap`, `StroboscopicMap`, `TangentSystem`,
  `EnsembleSystem`, `ProjectedSystem`
- Analysis: `orbit_diagram`, `OrbitDiagram`, `poincare_section`,
  `lyapunov_spectrum`, `max_lyapunov`, `kaplan_yorke_dimension`,
  `fixed_points`, `FixedPoint`
- Derived: `WrappedSystem` (adapt any external stepper to the protocol)
- State-space geometry (`data`): `Box`, `Ball`, `Grid`, `sampler`,
  `grid_points`, `set_distance` — the primitives the basin/attractor layer
  builds on (Monte-Carlo + full-grid sampling, attractor-matching distances)
- Submodules: `analysis`, `data`, `derived`, `families`, `registry`,
  `systems`, `utils`

Reachable but not top-level: `SystemBase`, `ParamSet`, `MetaStore`, `System`
(protocol) via `tsdynamics.families`; `staticjit` via `tsdynamics.utils`.
The engine layer (`tsdynamics.engine`) and the skeleton `transforms`/`viz`
packages are importable but not advertised in `__all__`.

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
  is each family's current default integrator (the **v2** backend — `jitcode` /
  `jitcdde` / `numba` / `reference`); it is the single knob the M3 migration
  flips to a Rust engine backend, and passing `backend=None` to a family's
  `integrate` / `iterate` resolves to it. The output grid each family samples on
  is the one hoisted `tsdynamics.utils.grids.make_output_grid` (the four
  byte-identical `_make_t_eval` copies are gone). **SDEs are the exception** —
  they keep the dedicated `run.sde_integrate_dense` / `run.sde_ensemble_final`
  seam (`run.integrate` cannot carry the noise seed/step and refuses an SDE).

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
- ODE: each instance gets a private jitcode wrapper from the shared cached
  `.so` (`_fresh_stepper`) — two live steppers never clobber each other.
- **DDE `set_state` raises** (state is a history function); `reinit(u)`
  restarts from a constant past. DDE stepping is forward-only.
- Map `step(n)` uses the compiled batch loop for n > 16, per-call `_step`
  otherwise (avoids per-params re-JIT in orbit-diagram sweeps).
- Param changes after `reinit` need a new `reinit` to reach a live stepper.

### `ContinuousSystem` extras

- **Jacobian autogen**: `jacobian_sym()` (SymEngine `diff` wrt `y(j)`),
  `jacobian(u, t)` numeric via cached `symengine.Lambdify`, `_rhs_numeric()`
  (fast numeric RHS — used by figures, Poincaré Hermite refinement, and the
  diffsol cross-validation). Hand-written `_jacobian` on ODE systems is never
  used at runtime; it is cross-checked against autogen in tests. `abs`/`sign`
  derivatives are resolved a.e. (`_resolve_derivative_nodes`).
- `integrate(backend=)` defaults to `_default_backend` (`"jitcode"`, the v2
  compile-to-C path). `"diffsol"` routes to `tsdynamics.engine.diffsol` (optional
  extra; translator: SymEngine → DiffSL with ICs-as-inputs, LLVM JIT, solver map
  RK45→tsit45, LSODA→bdf); `"auto"` picks diffsol-or-jitcode. `"interp"` / `"jit"`
  / `"reference"` route through the shared C-FAM seam (`_dispatch` →
  `engine.run.integrate`) to the Rust sole engine (or its pure-Python oracle) —
  the additive routing C-FAM gives the ODE family (it had no `interp`/`jit` path
  before).

### `DiscreteMap` extras

- `__init_subclass__` validates that `_step`/`_jacobian` positional parameter
  names match the `params` dict order — mismatches raise `TypeError` at
  import (also catches re-ordered `params` in subclasses).
- `_jacobian_fd_check = False` ClassVar opts a map out of the
  finite-difference Jacobian test (only for orbits living on discontinuities,
  e.g. Baker).
- `iterate(backend=...)` selects where iteration runs: `"numba"` (default, the
  v2 in-process loop) or the Rust engine (`"reference"` pure-Python oracle now;
  `"interp"`/`"jit"` once `tsdynamics._rust` ships). The engine loop lives in
  `crates/tsdyn-engine/src/map.rs`; non-Numba backends lower `_step` to the IR,
  so piecewise/`numpy`-ufunc steps raise `TapeCompileError`. The engine path
  diverges loudly (raises, no random-IC retry).

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
  wired (stream E-WIRE): `backend="interp"/"jit"` dispatches the drift+diffusion
  call to the engine (interpreter or Cranelift JIT) and reproduces the Python
  reference under a fixed seed. The pure-Python **reference** integrator (a
  faithful `SplitMix64` port) stays the default until the migration gate (M3).
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
  `ContinuousSystem.lyapunov_spectrum` are thin delegations to it (no more
  triplicated QR/`jitcode_lyap` loops). Modes:
  - **maps**: NumPy `W ← J(x)·W` + QR, Jacobian at the **pre-image** `x_n` (the
    correct tangent-map convention), with random-IC retry on divergence.
  - **ODEs**: two interchangeable backends via `backend=`. `"jitcode"` (default)
    wraps the compiled variational equations (`jitcode_lyap`).
    `"interp"`/`"jit"`/`"reference"` is the **backend-neutral** path: the
    *extended* variational ODE (state ⊕ k tangent vectors, built in
    `derived/_variational.py` and lowered via the public
    `engine.compile.lower_expressions`) is integrated per dt-chunk through
    `engine.run.integrate` then QR-reorthonormalised — the successor that
    survives the JiTCODE removal at M3 (the Rust engine becomes the variational
    integrator). Validated now via `backend="reference"` against analytic
    spectra; `"interp"`/`"jit"` need the `tsdynamics._rust` wheel.
  - **DDEs**: raise — their tangent space is the infinite-dimensional history
    space; use `DelaySystem.lyapunov_spectrum` (still `jitcdde_lyap`, the
    documented exception, NOT routed through `TangentSystem`).
  `TangentSystem.lyapunov_spectrum(...)` wraps the streaming `step()`/
  `exponents()` API into the standard burn-in + time-weighted estimate.
- `max_lyapunov` (Benettin two-trajectory) needs `set_state` → raises for DDEs.
- `fixed_points` is map-only for now.

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

---

## Versioning & release (python-semantic-release)

- `__version__` lives ONLY in `src/tsdynamics/__init__.py`; PSR rewrites it.
- Every push to `main` runs `release.yml`: tests → PSR computes the bump from
  conventional commits (feat→minor, fix/perf→patch, `!`→major) → tag,
  GitHub release, PyPI publish via trusted publishing (`environment: pypi`,
  bound to the filename `release.yml` — don't rename).
- CHANGELOG.md is maintained by python-semantic-release; release notes also land on GitHub Releases.
- Workflows: `ci.yml` (PR gate), `docs.yml` (build + Pages deploy with
  figure cache), `release.yml`, `pr-title.yml`, `nightly.yml` (`-m full`).

---

## Adding a new system

1. Drop the class into the right module under `systems/continuous/` or
   `systems/discrete/`, following the family contract (see docs → Start →
   concepts, or any existing system).
2. Add the class name to the module's `__all__` and the parent package's
   `__init__.py` (a registry test fails loudly if you forget).
3. **That's it** — the registry picks it up, the bulk test suite sweeps it,
   and the docs build generates its page (equations + figure) automatically.
4. Optional metadata: `variables`, `reference`, `known_lyapunov` ClassVars;
   `default_ic` if random ICs escape the basin; `_structural_params` for
   variable-dim systems; `_jacobian_fd_check = False` for discontinuous maps;
   `_compile_simplify = False` for an ODE whose large rational RHS hangs
   JiTCODE's simplify codegen pass.
5. For a new DDE: also add a non-equilibrium history to
   `tests/_sampling.py::DDE_HISTORIES` (guard test enforces this).  For a new
   *built-in* SDE (`StochasticSystem` subclass): add a `{"seed":…, "ic":…}` entry
   to `tests/_sampling.py::SDE_SAMPLES` (the `sde`-family guard test enforces it).

---

## Compilation cache

Compiled JiTCODE/JiTCDDE objects live in `~/.cache/tsdynamics/`
(override: `TSDYNAMICS_CACHE`).

- ODE cache key: `tsdyn_<ClassName>_<dim>[_<hash-of-structural-params>]`;
  non-structural params are runtime control parameters — no recompile on change.
- DDE cache key: `tsdyn_dde_<ClassName>_<hash-of-all-params>`.
- The docs figure cache is separate: `.cache/docs-figures`, keyed by class
  source hash (CI persists it via actions/cache).
- Wipe `~/.cache/tsdynamics/` after editing an `_equations` body.

---

## Common pitfalls

| Situation | What happens / what to do |
|---|---|
| `_equations` uses NumPy or `math` | JiTCODE can't compile it. Use `symengine.sin`/`cos`/... |
| Compile hangs on a large rational RHS (e.g. rotlet flows) | JiTCODE's `simplify(ratio=1)` codegen pass is super-linear. Set `_compile_simplify = False` on the class (see `BlinkingRotlet`). |
| Fractional power / steep `tanh` fails only on `backend="diffsol"` | Solvers probe outside the physical domain: a real `p**q` goes complex for `p<0`, and autodiff of `tanh(k·x)` overflows for huge `k·x`. Guard with `abs(p)` under the power and clamp the `tanh` argument (see `WindmiReduced`). |
| Variable-dim system without `_structural_params` | Compile-time `range(N)` fails. Add `_structural_params = frozenset({"N"})`. |
| Map params order ≠ `_step` signature order | **Raises `TypeError` at import** (since v2). |
| DDE with constant past at a fixed point | Lyapunov exponents ≈ 0. Provide a non-equilibrium `history`. |
| Tight tolerances on DDE | `rtol=atol=1e-3` is the safe start. |
| `set_state` on a DDE | Raises by design — use `reinit(u)`. |
| Implicit method on the Rust engine without a Jacobian | `integrate(method="rosenbrock"/"trbdf2", backend="interp"/"jit")` needs a Jacobian-carrying tape. The engine **raises** (it will not silently degrade to forward Euler); pass a problem built `with_jacobian=True`, or use an explicit method. `tsdynamics.solvers.build_kwargs(method)` returns `{"with_jacobian": True}` for implicit methods (C-SOLV); the family engine-dispatch seam (C-FAM) merges it so the stiff path "just works". |
| Param change ignored by a live stepper | `reinit()` after parameter changes (or use `with_params`). |
| Stale compiled cache after editing `_equations` | Wipe `~/.cache/tsdynamics/`. |
| Orbit diagram over a DDE wrapper | Recompiles per parameter value — slow by design, document it. |
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

# Experimental Rust backend
traj = lor.integrate(final_time=100.0, dt=0.01, backend="diffsol")

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
