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

```
src/tsdynamics/
├── __init__.py               # __version__ (managed by python-semantic-release) + re-exports
├── registry.py               # SystemEntry + all_systems/by_family/get/families
├── base/
│   ├── base.py               # SystemBase, ParamSet, Trajectory, MetaStore
│   ├── protocol.py           # the System runtime Protocol
│   ├── ode_base.py           # ContinuousSystem (JiTCODE + jacobian autogen + protocol)
│   ├── dde_base.py           # DelaySystem (JiTCDDE + protocol, forward-only)
│   └── map_base.py           # DiscreteMap (Numba + signature validation + protocol)
├── derived/
│   ├── _base.py              # DerivedSystem (wrapper base, with_params rebuilds)
│   ├── poincare.py           # PoincareMap (Hermite-refined crossings)
│   ├── stroboscopic.py       # StroboscopicMap
│   ├── tangent.py            # TangentSystem (Lyapunov engine)
│   ├── ensemble.py           # EnsembleSystem
│   └── projected.py          # ProjectedSystem
├── analysis/
│   ├── orbit_diagram.py      # orbit_diagram + OrbitDiagram
│   ├── poincare.py           # poincare_section (system or trajectory input)
│   ├── lyapunov.py           # lyapunov_spectrum, max_lyapunov, kaplan_yorke_dimension
│   └── fixed_points.py       # fixed_points + FixedPoint (maps, multi-start Newton)
├── backends/
│   └── diffsol.py            # experimental: SymEngine→DiffSL translator + pydiffsol
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
- Base classes: `ContinuousSystem`, `DelaySystem`, `DiscreteMap`; result type `Trajectory`
- Derived wrappers: `PoincareMap`, `StroboscopicMap`, `TangentSystem`,
  `EnsembleSystem`, `ProjectedSystem`
- Analysis: `orbit_diagram`, `OrbitDiagram`, `poincare_section`,
  `lyapunov_spectrum`, `max_lyapunov`, `kaplan_yorke_dimension`,
  `fixed_points`, `FixedPoint`
- Submodules: `analysis`, `base`, `derived`, `registry`, `systems`, `utils`

Reachable but not top-level: `SystemBase`, `ParamSet`, `MetaStore`, `System`
(protocol) via `tsdynamics.base`; `staticjit` via `tsdynamics.utils`.

---

## The registry (load-bearing!)

Every concrete `SystemBase` subclass auto-registers at class-definition time
(`SystemBase.__init_subclass__` → `registry.register_class`). `SystemEntry`
records name/cls/family/category/dim/params/reference/known_lyapunov.

- `registry.all_systems(family=, category=, builtin=True)` — iteration default
  is builtin-only (module under `tsdynamics.systems`); user classes register
  as non-builtin.
- **Family detection walks the MRO** (`DiscreteMap` → map, `DelaySystem` → dde,
  `ContinuousSystem` → ode) — NOT the module path; the DDE systems live in
  `systems/continuous/delayed_systems.py`.
- Duplicate builtin class names raise at import.
- Consumers: registry-driven test parametrization (`tests/conftest.py`
  `pytest_generate_tests`), the docs autogen hook, and users.

Optional per-system metadata ClassVars: `variables` (component names →
`traj["x"]`, docs labels), `reference` (literature citation shown in docs),
`known_lyapunov` (drives `tests/test_known_values.py`; keys: `spectrum`+`atol`,
or `n_positive`, plus `params`/`ic`/`kwargs`/`source`).

---

## Base classes

### `SystemBase` (`base/base.py`)

As before (ParamSet with fixed keys, attribute forwarding, `copy()` /
`with_params()`, `resolve_ic()` priority: arg > self.ic > default_ic > random)
plus:

- `meta` is now a **`MetaStore`** — dict-like, but writes append with history:
  `meta.record(key, value, **context)`, `meta[key]` → latest,
  `meta.history(key)` → all records. `meta == {}` still works.
- `_provenance(**extra)` builds the dict attached to `Trajectory.meta`.

### `Trajectory`

- Named components when the class declares `variables`: `traj["x"]`,
  `traj[["x","z"]]`, `traj.component("x")`.
- Point-set ops: `minmax()`, `standardize()`, `neighbors(q, k)` (lazy KD-tree).
- `meta` carries provenance (system, params, solver, dt, tolerances, ic,
  version); preserved through slicing/`after()`.

### The `System` protocol (`base/protocol.py`)

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
- `integrate(..., backend="diffsol")` routes to
  `tsdynamics.backends.diffsol` (optional extra; translator: SymEngine →
  DiffSL with ICs-as-inputs, LLVM JIT, solver map RK45→tsit45, LSODA→bdf).

### `DiscreteMap` extras

- `__init_subclass__` validates that `_step`/`_jacobian` positional parameter
  names match the `params` dict order — mismatches raise `TypeError` at
  import (also catches re-ordered `params` in subclasses).
- `_jacobian_fd_check = False` ClassVar opts a map out of the
  finite-difference Jacobian test (only for orbits living on discontinuities,
  e.g. Baker).

---

## Derived systems & analysis

- Wrappers forward `params`/`meta`; `with_params()` re-parametrizes the inner
  system and rebuilds the wrapper → orbit diagrams over `PoincareMap` /
  `StroboscopicMap` are bifurcation diagrams of flows (ODE control-param
  caching keeps sweeps cheap; DDE sweeps recompile per value).
- `PoincareMap` refines crossings with cubic Hermite using `_rhs_numeric`
  (O(dt⁴)); falls back to linear interpolation for DDEs.
- `TangentSystem`: maps = NumPy QR loop; ODEs = wraps `jitcode_lyap`; DDEs
  raise (use `DelaySystem.lyapunov_spectrum`).
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
- No CHANGELOG file — release notes live on GitHub Releases.
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
   variable-dim systems; `_jacobian_fd_check = False` for discontinuous maps.
5. For a new DDE: also add a non-equilibrium history to
   `tests/_sampling.py::DDE_HISTORIES` (guard test enforces this).

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
| Variable-dim system without `_structural_params` | Compile-time `range(N)` fails. Add `_structural_params = frozenset({"N"})`. |
| Map params order ≠ `_step` signature order | **Raises `TypeError` at import** (since v2). |
| DDE with constant past at a fixed point | Lyapunov exponents ≈ 0. Provide a non-equilibrium `history`. |
| Tight tolerances on DDE | `rtol=atol=1e-3` is the safe start. |
| `set_state` on a DDE | Raises by design — use `reinit(u)`. |
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
