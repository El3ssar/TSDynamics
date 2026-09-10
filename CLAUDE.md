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
  through `backend="jit"` (the Cranelift JIT, **the default since v6**),
  `"interp"` (the SSA-tape interpreter, bit-for-bit identical), or `"reference"`
  (a dependency-light pure-Python SciPy oracle, ODE + maps only — the
  cross-check, not for production use)
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
├── registry.py               # system registry (SystemEntry + all_systems/…) + generic analyses/renderers registries (solvers live in tsdynamics.solvers)
├── families/                 # base classes + the System protocol (was base/)
│   ├── base.py               # SystemBase, ParamSet, MetaStore (re-exports Trajectory from data)
│   ├── protocol.py           # the System runtime Protocol
│   ├── continuous.py         # ContinuousSystem (engine integrate + jacobian autogen)
│   ├── delay.py              # DelaySystem (engine method-of-steps, forward-only)
│   ├── discrete.py           # DiscreteMap (engine iterate + signature validation)
│   ├── stochastic.py         # StochasticSystem — diagonal-Itô SDEs (_drift+_diffusion; EM/Milstein)
│   ├── _accessors.py         # analysis-accessor mixin (the deliberate families→analysis lazy-import layering seam)
│   ├── _plottable.py         # SystemPlottable plotting seam (system.to_plot_spec/plot, splits plot vs integration kwargs)
│   └── wrapped.py            # WrappedSystem (canonical home; adapt an external stepper — re-exported via derived)
├── engine/                   # Rust-facing engine layer; tsdynamics._rust is the sole backend
│   ├── symbols.py            # engine-native symbolic frontend: state_time_symbols() → (Function("y"), Symbol("t"))
│   ├── compile.py            # symbolic dynamics → IR Tape (all families) + reference evaluator + bounded lowered-tape CACHE (lower_*_cached / clear_tape_cache / tape_cache_stats)
│   ├── problem.py            # per-family Problem builders bundling a tape + runtime context
│   ├── run.py                # orchestration: integrate/ensemble + eval_rhs/eval_jac + backend resolve (resolve_backend/_engine/BACKENDS/EngineNotAvailableError) + problem-coercion/naming/provenance; RE-EXPORTS every name the split-out submodules own so tsdynamics.engine.run.<X> keeps working for EVERY X
│   ├── run_methods.py        # method= resolution + auto-stiffness split out of run.py: _resolve_method_for/_recommend_method/_resolve_method_and_prepare (shared by integrate+ensemble)
│   ├── _families.py          # per-family runners split out of run.py: _run_continuous/_step_continuous/_run_dde/_sample_past/_run_map + the low-level _engine_* FFI shims (_engine_integrate_dense/_engine_ensemble_final/_engine_map_ensemble_final)
│   ├── stepper.py            # resumable OdeStepper API split out of run.py: make_ode_stepper/step_advance/step_advance_to_event (WS-STEPPER)
│   ├── sde_run.py            # SDE dense/ensemble seam split out of run.py: sde_integrate_dense/sde_ensemble_final (the seed/step-carrying path run.integrate refuses)
│   ├── events.py             # event subsystem split out of run.py: crossings/Event/EventSolution/integrate_events (WS-CROSSKERNEL + WS-EVENTSAPI)
│   └── reference.py          # the pure-Python reference oracle (backend="reference") split out of run.py: SciPy-stepped ODE + reference-evaluated map
├── solvers/                  # F2 registry mechanism + C-SOLV in-tree specs (explicit/implicit/stochastic) + method= resolution/aliases + auto-stiffness (select.py)
├── derived/
│   ├── _base.py              # DerivedSystem (wrapper base, with_params rebuilds)
│   ├── poincare.py           # PoincareMap (Hermite-refined crossings)
│   ├── _crossings.py         # WS-CROSSKERNEL engine-event crossing collector (section_crossings)
│   ├── stroboscopic.py       # StroboscopicMap
│   ├── tangent.py            # TangentSystem (Lyapunov engine)
│   ├── _variational.py       # backend-neutral extended variational lowering (ODE Lyapunov)
│   ├── ensemble.py           # EnsembleSystem
│   ├── projected.py          # ProjectedSystem
│   └── wrapped.py            # back-compat shim → re-exports WrappedSystem from families.wrapped
├── data/                     # state-space geometry + trajectory lingua franca (was sampling.py)
│   ├── trajectory.py         # Trajectory (canonical home; re-exported via families + top level)
│   └── sampling.py           # Box/Ball/Grid, sampler, grid_points, set_distance
├── analysis/                 # quantifiers, one subpackage per A-* stream (A-LAYOUT reorg)
│   ├── __init__.py           # flat re-exports (public API) + analyses plugin discovery
│   ├── _result.py            # re-exporting FACADE: re-exports the result hierarchy from the _result_* submodules (back-compat import surface)
│   ├── _result_base.py       # AnalysisResult (frozen-dataclass base: meta/repr/summary/to_dict/to_frame/plot seam)
│   ├── _result_scalar.py     # ScalarResult / CountResult (+ _NumericOps)
│   ├── _result_array.py      # ArrayResult
│   ├── _result_collection.py # CollectionResult
│   ├── _result_scaling.py    # ScalingResult (mixes _NumericOps like ScalarResult: full float drop-in — comparisons/arithmetic + value-based ==/hash; subclasses re-apply @dataclass(frozen=True, eq=False))
│   ├── _result_viz.py        # the .plot accessor seam (_PlotAccessor / VisualizationNotInstalled)
│   ├── _result_json.py       # to_dict / repr helpers (_jsonify / _is_frame_scalar)
│   ├── orbits/               # A-ORBIT: orbit_diagram + OrbitDiagram (+ periods/bifurcation_points; orbit_diagram.py); poincare_section (poincare.py); return_map + ReturnMap (first-return/next-amplitude map; return_map.py); self-registers into registry.analyses
│   ├── lyapunov/             # A-LYAP: lyapunov_spectrum, max_lyapunov, kaplan_yorke_dimension + lyapunov_from_data (Kantz/Rosenstein, from_data.py); self-registers into registry.analyses
│   ├── fixedpoints/          # A-FP: fixed_points/FixedPoint (maps+flow equilibria, Newton/SD/DL + rigorous Krawczyk method="interval" in _interval.py — fixed.py), periodic_orbits/periodic_orbit/PeriodicOrbit + estimate_period (periodic.py), shared primitives (_common.py); self-registers
│   ├── dimensions/           # A-DIM: correlation/generalized-Rényi/fixed-mass fractal dims + scaling-region fit
│   ├── chaos/               # A-CHAOS: GALI_k (Skokos) + 0–1 test (Gottwald–Melbourne) + expansion entropy (Hunt–Ott); maps via _jacobian, flows via self-contained RK4 variational core (no engine/compile)
│   ├── recurrence/          # A-RQA: recurrence_matrix (fixed ε / target rate, sparse cKDTree) + rqa (DET/LAM/L_max/ENTR/TT) + windowed_rqa; self-registers into registry.analyses
│   ├── basins/              # A-BASIN: find_attractors/basins_of_attraction (recurrence-FSM AttractorMapper) + basin_fractions (basin stability) + basin_entropy/uncertainty_exponent/wada_property (boundary structure) + continuation/tipping_points + resilience; cell tessellation in _common.py; self-registers into registry.analyses
│   ├── embedding/           # owned by A-EMBED
│   └── sampling/            # sagitta tools: estimate_dt_from_sagitta (output-dt selector) + sagitta_profile (per-point bow, the color_by="sagitta" field). NOT registered into registry.analyses (a sampling tool, not a quantifier); SagittaDt result is hidden (not exported)
├── viz/                      # PlotSpec IR seam + renderers (mpl/plotly/json/threejs) + compose.py (ts.viz.plot)
├── systems/
│   ├── continuous/           # 9 ODE category modules (+ spatial_fields.py 2-D PDEs) + delayed_systems.py (DDEs!)
│   └── discrete/             # 5 map category modules
└── utils/                    # the LEAF package: values both families/ and engine/ must agree on
    ├── grids.py              # make_output_grid (the single hoisted output-grid builder; sagitta tooling moved to analysis/sampling/)
    └── tolerances.py         # the single hoisted rtol/atol defaults (DEFAULT_/DDE_/DDE_LYAPUNOV_/BASIN_); see "Solver tolerances"

hooks/docs_autogen.py          # mkdocs hook: per-system pages + figures/viewers (TSD_DOCS_ONLY= subset preview)
docs/_tooling/equations.py     # symbolic → LaTeX rendering for docs
docs/_tooling/figures.py       # cached static-figure rendering (engine rk4 / scipy for stiff+discontinuous ODEs)
docs/_tooling/threejs_viewer.py # cached interactive three.js attractor viewers (3-D ODE pages; iframe-embedded)
docs/_tooling/field_movies.py  # cached animated spatial-field movies (2-D _field_shape pages: GrayScott/SwiftHohenberg → autoplay <video> hero; mp4 via ffmpeg + gif fallback)
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

**`tsdynamics.__all__` is exactly TWELVE names** (measured — re-measure, never
nudge, if you change it):

```
ContinuousSystem DelaySystem DiscreteMap StochasticSystem WrappedSystem
Trajectory  plot  systems analysis viz errors  __version__
```

The membership rule is written above `__all__` in `src/tsdynamics/__init__.py`
and is the thing to argue against before adding a name:

> A name earns a top-level slot only if a user TYPES it in ordinary work — and no
> user should ever have to construct a library type to make a call, so a name
> that is exported *because a signature demands it* is evidence of a **signature
> bug**, not of a needed export.

Four corollaries, applied mechanically:

- **C1 the toll rule** — plain Python (tuples/lists/strings/numbers/arrays)
  reaches every front door. Fix the signature, then demote the name; never the
  reverse. (`ts.basins_of_attraction(vdp, [(-3, 3), (-3, 3)])` already worked,
  which is why `Box`/`Ball`/`Grid` never needed exporting.)
- **C2 received ≠ typed** — a type you get *back* is not one you type. Exactly
  one exception: `Trajectory`.
- **C3 one concept, one spelling** — two grammars for the same argument is the
  silent-wrong-answer defect, not flexibility.
- **C4 a name must resolve to what it advertises** — no `__all__` entry may be
  shadowed by a submodule of the same name. Gate:
  `tests/test_namespace_curation.py::test_no_all_entry_is_shadowed`, swept over
  every public package.

Everything else is **demoted, never removed** — `ts.<name>` still resolves and
`from tsdynamics import <name>` still imports:

- The 177 built-in systems (142 ODE + 6 DDE + 26 maps + 3 SDE) via
  `tsdynamics.systems`; `ts.Lorenz` resolves lazily through `__getattr__`.
- Every analysis function and result class — `orbit_diagram` (the ONE spelling
  since v6: the `bifurcation_diagram` alias was the same object under a second
  name, and a shared implementation can name only one of its spellings in a
  message, so half of all callers were answered about a function they had never
  typed — `ts.bifurcation_diagram` / `ts.analysis.bifurcation_diagram` now
  redirect; `PlotKind.BIFURCATION_DIAGRAM`, the *picture*, is a different concept
  and is untouched), `OrbitDiagram`, `poincare_section`,
  `return_map`/`ReturnMap`, `lyapunov_spectrum`, `max_lyapunov`,
  `kaplan_yorke_dimension`, `lyapunov_from_data`/`LyapunovFromData`,
  `fixed_points`/`FixedPoint`, `periodic_orbits`/`periodic_orbit`/`PeriodicOrbit`,
  `estimate_period`; A-DIM `correlation_dimension`, `correlation_sum`,
  `generalized_dimension`, `box_counting_dimension`, `information_dimension`,
  `dimension_spectrum`, `fixed_mass_dimension`, `DimensionResult`; A-CHAOS
  `gali`/`GALIResult`, `zero_one_test`, `expansion_entropy`/
  `ExpansionEntropyResult`; A-RQA `recurrence_matrix`/`RecurrenceMatrix`, `rqa`/
  `RQAResult`, `windowed_rqa`/`WindowedRQA`; A-BASIN `find_attractors`,
  `basins_of_attraction`, `basin_fractions`, `basin_entropy`,
  `uncertainty_exponent`, `wada_property`, `continuation`, `tipping_points`,
  `resilience`, `Attractor`, `AttractorSet`, `BasinsResult`, `BasinFractions`,
  `BasinEntropy`, `UncertaintyExponent`, `WadaResult`, `ContinuationResult`.
  Canonical home `ts.analysis.*`; `ts.analysis.__dir__` is **flat** and mirrors
  its 85-name `__all__` (the leaf namespace enumerates the analyses — cf.
  `numpy.linalg`; `discover_plugins` is deliberately not in `__all__`).
- The **derived wrappers** `PoincareMap` / `StroboscopicMap` / `TangentSystem` /
  `EnsembleSystem` / `ProjectedSystem` → `ts.derived.*`. Legal only because each
  has a **verb on the system**, speaking the same vocabulary as the class:
  `sys.poincare("y", 0.0)` / `sys.poincare(("y", 0.0, "up"))`,
  `sys.stroboscope(period)`, `sys.tangent(k=2)`, `sys.project(0, 2)`,
  `sys.copies(states)`.
- **State-space geometry** (`data`): `Box`, `Ball`, `Grid`, `sampler`,
  `grid_points`, `set_distance` → `ts.data.*`. (`Region` and `region` were never
  top-level and stay `ts.data`-only — nothing was demoted, so `ts.Region` /
  `ts.region` do not resolve; the gate `test_demoted_data_primitives_stay_reachable`
  lists exactly the three that were.) Every `region=` door in
  the library reads **one `(lo, hi[, n])` pair per state component**
  (`data/sampling.py::as_region`, the ONE reading); a `Box`/`Ball`/`Grid` is
  still accepted everywhere, just never required.
- `T` (the per-transform option carrier) → `ts.viz.T`. A plain
  `("name", {options})` pair is a transform call now, so nothing requires it.
- Machinery submodules `data` / `derived` / `engine` / `families` / `registry` /
  `solvers` / `utils` (`_INTERNAL_SUBMODULES`) — bound eagerly, off `__all__`.
  The justification is C1: *nothing in them is required to make a call.* (The
  old comment said they were redundant because their contents were already on the
  top level — the exact reasoning this curation exists to correct.)

**DELETED outright** (the name ceases to exist, `_RENAMED_IN_V6` redirects):

- `ts.basins` — a C4 violation: the alias was a *function* while
  `ts.analysis.basins` is the *subpackage*. `ts.basins_of_attraction` is the one
  spelling; `ts.basins` raises an `AttributeError` naming it.

**The `AttributeError` is the discovery mechanism.** Curation hides ~230
reachable names from autocomplete, so a wrong guess at `ts.<name>` is the only
feedback a user gets, and `_attribute_error` answers it in five ordered cases:
removed-in-v6 → renamed-in-v6 → **at a different address** → a near-miss on a
built-in system → the two catalogues worth tab-completing. The third case is
`_home_of`: an **exact** hit in the public `__all__` of `data` / `derived` /
`analysis` / `viz` outranks every fuzzy guess below it. Without it a name that
is merely demoted got answered with nonsense — `ts.region` (real, at
`ts.data.region`) suggested `ts.systems.Oregonator()` and `ts.Region` suggested
`ts.engine`. Dunder probes short-circuit, so protocol lookups import nothing and
`ts.viz` stays lazy.


**Scope boundary (v6, stream SCOPE-SURGERY).** TSDynamics is a *dynamical systems*
library, deliberately **not** a general time-series toolkit. The governing rule is
**phase-space methods stay; generic series statistics go.** In v6 the following
were **deleted outright** (no shims — the owner sanctioned the break) and now live
in a separate, companion time-series library: `analysis/entropy/` (permutation/dispersion/sample/
multiscale entropy, LZ76, the OutcomeSpace core), `analysis/surrogate/`
(shuffle/FT/AAFT/IAAFT, `time_reversal_asymmetry`, `nonlinear_prediction_error`,
`surrogate_test`), and the whole `transforms/` package (PSD, detrend/normalize,
Butterworth filters, `extract_features`/Hjorth) plus its `_result_common.py` leaf.
What **stayed** is what reconstructs or measures *phase space*:
`analysis/embedding` (Takens — the bridge from data back to phase space),
`analysis/recurrence` (recurrence plots are a phase-space method),
`analysis/lyapunov/from_data.py`, and `analysis/sampling`.
Do **not** re-add generic signal processing here. Beware the false friends:
`expansion_entropy` (A-CHAOS), `basin_entropy` (A-BASIN), RQA's `ENTR`, and
Benettin's "Kolmogorov entropy" citation are all SURVIVORS and unrelated to the
deleted entropy package.

Reachable but not top-level: `SystemBase`, `ParamSet`, `MetaStore`, `System`
(protocol) via `tsdynamics.families`.
`errors` is bound eagerly and **is** in `__all__`; `engine`, `solvers`, `data`,
`derived`, `families`, `registry` and `utils` are all **reachable** as
`ts.<name>` but are deliberately **not** in `__all__` (`engine`/`solvers` are
flagged internal in their docstrings). The `viz` package
(`PlotSpec` IR + four self-registering renderers — matplotlib/plotly/json/threejs —
plus the styling/theme system) is bound **lazily** via the module `__getattr__`, so
a plain `import tsdynamics` pulls in no plotting library at import time —
`ts.viz` resolves (and caches) on first access and shows in `__all__`/`dir()`.

---

## The registry (load-bearing!)

`registry.py` hosts the specialised *system* registry (below) plus two
generic name→object `Registry` containers — `registry.analyses` and
`registry.renderers` — for the analysis and visualization-backend streams to
register into.
Out-of-tree plugins are wired in (A-LAYOUT): `tsdynamics.analysis` calls
`plugins.register_entry_points` at import to load the `tsdynamics.analyses`
entry-point group, and `tsdynamics.viz` does the same for
`tsdynamics.renderers`; in-tree analyses self-register from their own
subpackages (the A-* streams).
There is **no `transforms` registry / entry-point group** — it was removed in v6
along with the generic time-series layer (see the scope boundary above). Do not
re-add one; a companion library's integration surface will be designed when that
library exists.  **This prohibition is about the deleted generic time-series
package, and nothing else.** It does NOT cover `registry.plot_transforms` (the
viz layer's transform→geometry registry, entry-point group
`tsdynamics.plot_transforms`), which is a different concept that happens to share
the word: a *plot* transform turns a system or trajectory into plottable
geometry, and never ships a signal-processing estimator.
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
  Three built-in SDEs ship (`OrnsteinUhlenbeck`, `GeometricBrownianMotion`,
  `DoubleWell`, in `systems/continuous/stochastic_systems.py`), so
  `registry.families()` is `{'ode': …, 'dde': …, 'map': …, 'sde': 3}` for
  builtins; a user `StochasticSystem` subclass registers (non-builtin) with
  family `sde`.
- Duplicate builtin class names raise at import.
- Consumers: registry-driven test parametrization (`tests/conftest.py`
  `pytest_generate_tests`), the docs autogen hook, and users.

Optional per-system metadata ClassVars: `variables` (component names →
`traj["x"]`, docs labels), `reference` (literature citation shown in docs),
`doi` (the bare DOI for that citation — e.g. `"10.1175/..."` — sourced from the
GilpinLab/dysts dataset where available, for the docs per-system page),
`known_lyapunov` (drives `tests/test_known_values.py`; keys: `spectrum`+`atol`,
or `n_positive`, plus `params`/`ic`/`kwargs`/`source`), and — for a
**spatially-extended** system whose state vector is a flattened field —
`_field_shape: tuple[int, ...]` (the spatial grid `(Ny, Nx)` / `(N,)`, resolved
onto `traj.meta["field_shape"]` by `SystemBase.__init__`/`_provenance` for the
`kind="field"` spatial-field movie) and `field_labels` (the names of the field
blocks packed into the state, e.g. Gray–Scott's `("u", "v")`).

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
  is each family's default integrator — **`"jit"` (the Cranelift JIT) for every
  concrete family since v6** (it was `"interp"`; see "Which backend is the
  default" below for the measurement that drove the flip); the abstract
  `SystemBase` keeps `"reference"` (the wheel-free oracle). Passing
  `backend=None` to a family's `integrate` / `iterate` resolves to it, and
  `backend="auto"` resolves to the same `"jit"`. `run.integrate` also resolves the `method=` string
  through the shared `_resolve_method_for` contract (so an alias `"RK45"`/
  `"dopri5"` → `"rk45"`, a rejected v2-only name `"LSODA"`, and the auto-stiffness
  `method="auto"` all behave identically in **`integrate` and `ensemble`**) and
  rebuilds the ODE tape `with_jacobian=True` when an implicit kernel needs it.
  Lowering goes through the cached `lower_*_cached` helpers (see the tape cache
  section), so a parameter sweep reuses one tape. The output grid each
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
- **Topical accessors for the data-consuming analyses** — `traj.dims`,
  `traj.recurrence`, `traj.lyap` are the *same* cached accessor classes a system
  carries (`families/_accessors.py`), bound to the trajectory: `traj.dims.correlation()`,
  `traj.recurrence.rqa()`, `traj.lyap.from_data()`. These analyses take a measured
  point set, and a `Trajectory` is what a user holding one actually has — the
  accessors used to exist only on `SystemBase`, i.e. only on the object half of
  them refuse. The routing seam is `_accessors.is_drivable(subject)`: bound to a
  trajectory there is nothing to run, so a **`system`-first** method
  (`lyap.spectrum` / `lyap.maximal`) raises `InvalidParameterError` naming the
  system-bound spelling, and `run_kwargs=` is refused rather than ignored.
  `Trajectory` has `__slots__`, so the cache is an explicit `_accessor_cache`
  slot reset in `__setstate__` (it is not pickled).
- `meta` carries provenance (system, params, solver, dt, tolerances, ic,
  version); preserved through slicing/`after()`.
- **Plotting front door — `to_plot_spec(kind=None, *, components=None, **kind_kw)`:**
  the one entry point for trajectory plots (the parameterised `viz.producers`
  builders stay an internal detail). Auto-dispatches on the number of *selected*
  components — 1 → `TIME_SERIES`, 2 → `PHASE_PORTRAIT_2D`, 3 → `PHASE_PORTRAIT_3D`,
  **4+ → `SPACETIME`** (a field image, never a misleading 3-D portrait of the
  first three coords). `components=` (a name / index / sequence) picks the
  channels; `kind=` forces any `PlotKind`, plus the `"delay"` *recipe* (an `x(t)`
  vs `x(t-τ)` embedding — `"delay"` is not a `PlotKind`, it routes via
  `_KIND_ALIASES` to a producer emitting `PHASE_PORTRAIT_2D`; `tau` is in **time
  units**, converted to a sample lag via `meta["dt"]`). Per-kind options ride on
  `**kind_kw` (deliberately off the signature, since each is valid for one kind
  only): `tau` (delay, required), `color_by` (time series / portraits),
  `transpose` (spacetime). The routing tables — `_KIND_ALIASES`, `_KIND_KW`
  (per-kind allow-list, validated), and `_PLOT_SPEC_KEYS` (derived from them) —
  live at the top of `data/trajectory.py`; extending a kind's options is a
  one-line edit there. `Trajectory.plot()` / `SystemPlottable.plot()` peel the
  spec-shaping kwargs (the closed `_PLOT_SPEC_KEYS` set) and forward them to
  `to_plot_spec`; the remainder are inline tweaks / backend kwargs.
  `SystemPlottable.to_plot_spec` likewise splits plot kwargs from integration
  kwargs (`final_time`/`dt`/`steps`/`ic`/…) on that same closed set.

### The `System` protocol (`families/protocol.py`)

All three families + all derived wrappers implement:
`step(n_or_dt) -> new state`, `state()`, `set_state(u)`, `time()`,
`reinit(u, *, t, params)`, `trajectory(...)`, `is_discrete`.

- First `step()`/`state()` on a cold system does an implicit `reinit()`.
- ODE: `reinit` lowers the system to an engine tape once; each `step(dt)`
  integrates one `dt` chunk through `run.integrate` from the live state.
- **`backend="reference"` is honored through the stepping protocol** (ODE):
  `reinit(backend="reference")` resolves and stores the backend (`resolve_backend`
  raises `InvalidParameterError` for an unknown name), and `step()` then routes to
  `_step_reference` — one `dt` chunk on the pure-Python reference ODE integrator
  (the same path `integrate(backend="reference")` uses), so the wheel-free oracle
  is reachable via `reinit`/`step`/`state`. It is **no longer silently coerced to
  `interp`** (diagnosis #5); reference owns no resumable `OdeStepper`, so it never
  builds the engine fast path.
- **DDE `set_state` raises** (state is a history function); `reinit(u)`
  restarts from a constant past. DDE stepping is forward-only (each `step`
  re-integrates from the constant past via the method of steps). **DDE
  `backend="reference"` is loudly rejected** (there is no pure-Python DDE
  integrator) rather than silently degraded.
- Map `step(n)` runs the per-call pure-Python `_step` loop (silencing NumPy FP
  warnings so an overflow surfaces as the explicit divergence `ConvergenceError`).
- Param changes after `reinit` need a new `reinit` to reach a live stepper.

### `ContinuousSystem` extras

- **Jacobian autogen**: `jacobian_sym()` (SymEngine `diff` wrt `y(j)`),
  `jacobian(u, t)` numeric via cached `symengine.Lambdify`, `_rhs_numeric()`
  (fast numeric RHS — used by figures, Poincaré Hermite refinement, and the
  reference-backend cross-validation). Hand-written `_jacobian` on ODE systems is
  never used at runtime; it is cross-checked against autogen in tests.
  `abs`/`sign`/`min`/`max` derivatives are resolved a.e.
  (`_resolve_derivative_nodes`) — `min`/`max` follow whichever argument is the
  active extremum, so a symengine `Min`/`Max` ODE (or an `np.minimum`/`np.maximum`
  map lowering to `OP_MIN`/`OP_MAX`) lowers `with_jacobian=True` for the stiff
  (`bdf`/`rosenbrock`/`trbdf2`) and map-Lyapunov paths instead of raising.
- `integrate(backend=)` defaults to `_default_backend` (`"jit"`). `"jit"`
  / `"interp"` / `"reference"` route through the shared C-FAM seam (`_dispatch` →
  `engine.run.integrate`) to the Rust engine (or its pure-Python oracle).
  `run.integrate` resolves `method=` through the solver registry and lowers the
  tape `with_jacobian=True` for the implicit stiff kernels (`bdf` /`rosenbrock` /
  `trbdf2`); stiff catalogue systems declare `_default_method = "bdf"`.
  **`method="auto"`** (stream FIX-AUTOSTIFF) is wired into `run.integrate`: it
  lowers the problem, probes the Jacobian spectrum at the start state via
  `solvers.recommend` (the one-point `solvers.is_stiff` heuristic) and selects
  `bdf` on a stiff RHS / `rk45` otherwise — the resolved kernel is recorded in
  `traj.meta["method"]`. It is a *heuristic* (IC-dependent), so a reliably-stiff
  system should still declare `_default_method`; maps (no solver kernel) treat
  `"auto"` as a no-op. `"auto"` is wired through the **shared** `_resolve_method_for`
  contract, so it resolves consistently on **every** `method=` entry point —
  `integrate`, `ensemble`, the resumable stepping protocol
  (`ContinuousSystem.reinit(method="auto")` → `step`), and the events seam
  (`run(events=…, method="auto")`); each probes the start-state Jacobian spectrum
  and records the canonical resolved kernel (e.g. `"rk45"`) in `traj.meta["method"]`
  / the event solution. (Earlier only `integrate`/`ensemble` understood it.)

### Dense output and `max_step` (v6, `eng-no-dense-output` / `perf-no-max-step`)

**`dt` is an output sampling interval, not an accuracy knob.** Before v6
`integrate_grid` had no dense output: the adaptive stepper was *forced to land on
every requested output sample*, so the answer depended on the output grid (Lorenz
to T=10 at `rtol=1e-6` gave error 1.30e-3 at `dt=10` but 4.3e-10 at `dt=0.001` — a
**3.0e6x** spread) and `rtol` was inert on a fine grid (`rtol=1e-4`, `1e-6`, `1e-8`
and `1e-10` returned **bit-identical** arrays at `dt=0.001`). Both are fixed:
after v6 the grid spread is **1.004x** and the delivered error tracks `rtol` across
six decades.

- **`Caps::dense` gates whether dense output happens at all**, not merely which
  interpolant is used. There is deliberately **no universal cubic-Hermite
  fallback**: measured, an endpoint Hermite extension is 13-384x worse than the
  native one for the order-5 kernels and 4.8e3-2.2e7x worse than `contd8` for
  `dop853`, so a Hermite floor would trade a contract bug for an accuracy bug.
- **Blast radius = `{rk45, tsit5, dop853}` on `integrate`/`run` with a >=3-point
  grid.** Every other kernel (`rk4`, `bs3`, `cashkarp`, `rkf45`, `heun_euler`,
  every fixed-step and every implicit/stiff kernel) keeps land-on-every-sample,
  **bit-for-bit**. `bdf`'s stiff over-resolution tax is *not* fixed here — a
  fixed-order extension of a variable-order method is order-inconsistent by
  construction and needs its own design.
- **Two structural rules make the two-node callers provably untouched**: the dense
  march *lands exactly on `t_eval.last()`* (so the final row is always the
  integrated state, bit-identical to `integrate_final`) and *interpolates strictly
  interior points only*. A two-node grid has no interior point, so
  `lyapunov::classify`, the basin cell march and the resumable
  `OdeStepper::advance` (all `[t, tf]`) are unchanged with no opt-out flag.
- **Kernels.** `rk45` uses Shampine's DP5 continuous extension and `tsit5`
  Tsitouras' — both **free** (a linear combination of the seven stages the step
  already computed, zero extra RHS evals), order 4 (local O(h^5)). `dop853` uses
  Hairer's **`contd8`**, order 7 (measured local order 8.00), which needs four
  extra stages — so it is built in the new **defaulted** `Solver::prepare_dense`
  trait method, called at most once per accepted step and only when that step is
  actually sampled inside (12 -> 16 RHS evals, +33%, on interpolated steps only).
  `interpolate` stays `&self` and RHS-free because the event root-finder calls it
  10-30x per step.
- **`max_step`** (`IntegrateConfig::max_step`, and `max_step=` on `integrate` /
  `run` / `reinit` / `ensemble` / `crossings` / `integrate_events`) bounds any
  single internal step; `None`/`f64::INFINITY` (the default) is provably inert
  (`h.min(INFINITY) == h`). It is a step **size**; `max_steps` is a step **count**
  — one character apart, opposite units, so keep them documented adjacently. It is
  the honest replacement for the accidental bound the forced landing used to
  provide: `max_step=dt` reproduces the pre-v6 step regime explicitly.
- **Bypass:** `TSDYNAMICS_NO_DENSE_OUTPUT` (truthy => the pre-v6 forced-landing
  march), mirroring `TSDYNAMICS_NO_TAPE_CACHE`/`TSDYNAMICS_NO_JIT_CACHE`. There is
  deliberately **no public `dense_output=` kwarg**: the library has one output
  semantics, and the legitimate per-call need is served by `max_step`.
  `traj.meta` records `max_step` and `dense_output`.
- **Measured cost** (Lorenz T=100, `rk45`, `rtol=1e-6`): `dt=0.02` 4.63 -> 2.70 ms,
  `dt=0.001` 60.5 -> 8.3 ms (7.3x), `dt=1e-4` 592 -> 72.5 ms (8.2x); RHS
  evaluations on a 1001-point grid 6001 -> 805 and now **independent of the output
  resolution** (pinned by a counting test, which cannot flake). `dop853` at
  `dt=0.001`: 126.7 -> 17.5 ms.

### Solver tolerances (v6, `utils/tolerances.py`)

**Every `rtol=`/`atol=` default in the library is a named constant in the leaf
module `utils/tolerances.py`.** Before v6 the pair `1e-6`/`1e-9` was duplicated
as bare literals across ~16 sites in five subpackages plus the docstrings that
quoted them — which is exactly how two "same" defaults drift apart. The module
is a leaf (imports nothing from `tsdynamics`), so `families/` (which imports the
engine only lazily) and `engine/run.py` both take it at module scope with no
cycle; `engine/run.py` was rejected as the home for precisely that reason. A
polish gate (`test_polish_standards.py::test_no_bare_tolerance_literal_in_the_library`,
AST-based) fails on any bare `rtol`/`atol` numeric literal in a signature, a call
keyword or a `self._rtol = …` assignment; genuine homonyms (FNN's `R_tol`/`A_tol`,
the orbit-diagram branch-clustering `rtol`) are an explicit, liveness-checked
carve-out table.

**The ODE default tightened to `rtol=1e-9` / `atol=1e-12`** (from `1e-6`/`1e-9`).
This is the other half of dense output: the forced landing used to subsidise
accuracy `rtol` never asked for, so removing it would silently cost a user who
never touched `rtol` 2x-9600x of delivered accuracy. Measured at the defaults
(`dt=0.02`, `T=5`, error at final time vs SciPy `DOP853` @ `rtol=1e-13`) over 15
catalogue systems: **median 1459x more accurate for a median 1.74x cost**
(Lorenz `2.2e-4 -> 1.9e-7`, Halvorsen `1.5e-3 -> 2.9e-7`, Chua `1.1e-1 ->
1.6e-5`). Against dense output's own ~1.7x speedup that is roughly cost-neutral
versus pre-v6, at ~100x the pre-v6 accuracy.

**Three drivers keep their own, looser number** — all documented, all measured,
none accidental. The rule that decides is: *the bump is owed only to surfaces
dense output changed* (the adaptive explicit kernels sampling a grid with
interior points).

| Constant | Value | Driver | Measured impact of following the global default |
|---|---|---|---|
| `DEFAULT_RTOL`/`DEFAULT_ATOL` | `1e-9`/`1e-12` | ODE `integrate`/`run`/`ensemble`/`reinit`+`step`/events/ODE Lyapunov/crossings | (the default) |
| `DDE_RTOL`/`DDE_ATOL` | `1e-3`/`1e-3` | `DelaySystem._default_rtol/_atol` | method of steps lands on every sample -> tolerance inert: **5 of 6 built-in DDEs bit-identical** from `1e-3` to `1e-9` at the default `dt`; `IkedaDelay` 1.4x cost |
| `DDE_LYAPUNOV_RTOL`/`_ATOL` | `1e-7`/`1e-9` | `dde_lyapunov_spectrum` | same march; agrees with `1e-9`/`1e-12` inside the estimator's finite-time scatter, same cost |
| `BASIN_RTOL`/`BASIN_ATOL` | `1e-6`/`1e-9` | basin cell march (**both** `_AttractorMapper._reinit` and `run.basin_march`) | 2.27x (Duffing 60x60) / 3.01x (magnetic pendulum 35x35) slower for **0.00% of labels changing** |

- The **basin pair is load-bearing in two places at once**: the Rust march is
  contractually bit-identical to the Python `_AttractorMapper` oracle
  (`tests/test_basin_kernel.py`), so `_reinit` now passes the tolerance
  *explicitly* for a flow instead of inheriting `ContinuousSystem.reinit`'s
  default. If those two sites ever name different constants the equivalence
  breaks silently; `test_basin_march_python_and_rust_name_the_same_tolerance`
  guards it. (A map takes no tolerances — `DiscreteMap.reinit` has none.)
- The **section-crossing march** (`derived/_crossings.py`) is also outside the
  blast radius but keeps *no* private number: it pins fixed-step `rk4`, which has
  no error control, so its tolerances are genuinely **inert** (measured: a
  400-crossing Rössler section is bit-identical at 1.01x cost). Nothing to trade,
  so it follows the global constant.
- The DDE "tight tolerances stall the solver" folklore is **withdrawn**: it was a
  v2 JiTCDDE property. All 6 built-in DDEs complete at `rtol=1e-12`/`atol=1e-15`
  over `T=500` on the Rust engine.

### `DiscreteMap` extras

- `__init_subclass__` validates that `_step`/`_jacobian` positional parameter
  names match the `params` dict order — mismatches raise `TypeError` at
  import (also catches re-ordered `params` in subclasses).
- **`_jacobian` is OPTIONAL (v6).** Only `_step` is `@abstractmethod`. The
  default `DiscreteMap._jacobian` is a `classmethod` that derives the Jacobian
  **symbolically from `_step`** via `engine.compile.map_jacobian_fn` — the same
  trace `lower_map` performs, whose docstring already called the traced
  derivative "the single source of truth". So defining a map is *writing
  `_step`*, exactly as defining a flow is writing `_equations`; the two families
  now answer the same way, and the hand-transcription surface that produced
  catalogue Jacobian bugs is gone. A hand-written `@staticmethod _jacobian`
  still **wins** (every catalogue map keeps one, as the test-oracle side of
  `_jacobian_fd_check`). Write one when the step cannot be traced (a Python `if`
  on the state → `TapeCompileError` naming the class *and* `_jacobian`) or when
  a one-sided slope on a discontinuity is the meaningful answer. Results are
  memoised per `(kernel object, parameter values)` in `_MAP_JACOBIAN_MEMO`, and
  the a.e. `Derivative` nodes (`abs`/`sign`/`floor`) resolve through the same
  `_resolve_derivative_nodes` the ODE Jacobian uses. Gate:
  `tests/test_families_fixes.py::TestMapJacobianIsAutogenerated` (autogen ==
  hand-written to 1e-14 on Hénon; identical Lyapunov spectrum).
- `_jacobian_fd_check = False` ClassVar opts a map out of the
  finite-difference Jacobian test (only for orbits living on discontinuities,
  e.g. Baker).
- `iterate(backend=...)` runs the iteration on the Rust engine (`"jit"`
  default / `"interp"` / `"reference"` pure-Python oracle). The engine loop lives in
  `crates/tsdyn-engine/src/map.rs`; all backends lower `_step` to the IR, so
  piecewise/`numpy`-ufunc steps raise `TapeCompileError`. The engine path
  diverges loudly (raises); the random-IC retry still applies when `iterate` is
  called without an explicit `ic`.

### `run` states its own signature (v6)

`run` is the canonical trajectory verb on all four families, and until v6 it was
a `**kwargs` passthrough: `ContinuousSystem.run` bound `final_time`, `dt`,
`events` and nothing else, so the library's most-typed call showed **2 of ~12**
keywords to `inspect.signature`, tab-completion and every IDE tooltip.
`transient=` — which the quick reference below uses — was discoverable only from
prose. The family-specific aliases (`integrate` / `iterate`) had the full
signature all along, so the *alias* documented itself while the *canonical verb*
did not.

Each family's `run` now binds exactly what its alias binds and forwards
explicitly, keeping `**kwargs` only for the genuine solver-option passthrough:

| family | `run` binds |
|---|---|
| `ContinuousSystem` | `final_time`, `dt`, `t0`, `ic`, `transient`, `method`, `rtol`, `atol`, `max_step`, `backend`, `seed`, `events` |
| `DiscreteMap` | `n` (+ the `steps` alias via `_resolve_iteration_count`), `ic`, `transient`, `backend`, `seed`, `max_retries` |
| `DelaySystem` | `final_time`, `dt`, `ic`, `history`, `transient`, `method`, `rtol`, `atol`, `backend`, `seed` |
| `StochasticSystem` | `final_time`, `dt`, `t0`, `ic`, `transient`, `method`, `seed`, `backend` |

Behaviour is unchanged and verified bit-identical (`run(...)` vs
`integrate(...)`/`iterate(...)` on a pinned IC, all four families) — every bound
default is the *same object* the alias defaults to, so binding it and forwarding
it is indistinguishable from not passing it. The wrong-family horizon words
still raise by name (`hen.run(final_time=…)` / `lor.run(n=…)`), because they
reach `**kwargs` and the alias's guard, not a bound parameter.

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
  `backend`: `"jit"` (the default, like every other family) / `"interp"` — the
  compiled engine via `tsdynamics._rust` (stream E-WIRE) — or `"reference"` (pure
  Python). (This line said `"reference"` was the SDE default; it never was.)
- `ensemble(ics, ..., backend=)` seeds trajectory `i` from `seed_for(seed, i)` —
  depending only on the index — so a batch is reproducible and mirrors the Rust
  engine's parallel-equals-serial contract; a diverged trajectory becomes a `NaN`
  row. `backend="interp"/"jit"` fans the batch out on the engine's rayon pool.
- The real engine is Rust: kernels in `crates/tsdyn-solvers/src/sde/**`
  (own `SdeKernel` trait, RNG-free — the engine hands them a pre-drawn `dw`),
  loop + seeded RNG in `crates/tsdyn-engine/src/sde.rs`. The two-tape SDE FFI
  (`integrate_sde_dense` / `integrate_sde_ensemble_final` in `tsdyn-core`) is
  wired (stream E-WIRE): `backend="jit"` (the default) / `"interp"` dispatches
  the drift+diffusion call to the engine (Cranelift JIT or interpreter). The
  pure-Python **reference** integrator (a faithful `SplitMix64` port, sharing the
  engine's tolerant fixed-step landing) reproduces the engine **to floating-point
  tolerance** under a fixed seed — the integer RNG stream and draw order are
  identical; the only residual difference is the Box–Muller normal (Python libm
  `sin`/`cos` vs Rust `sin_cos`, ≤1 ULP/draw), so it is *not* bit-for-bit — and
  stays available as the wheel-free oracle. **Fixed-step landing (stream
  FIX-SDE-WIENER):** the output grid is uniform in `dt`, but float roundoff makes
  a nominally `dt`-wide segment compute `remaining = t_end − t` a few ULP above
  `dt`; the landing step absorbs that into one canonical `dt` step
  (`_LANDING_REL_TOL` / Rust `LANDING_REL_TOL`, kept in lock-step) instead of
  taking a full `dt` step plus a spurious sub-ULP sliver — so `integrate` and the
  `step()` loop trace the same path and `interp == jit` bit-for-bit.
- **Registry-detected (stream C-FAM):** a concrete `StochasticSystem` subclass
  registers with family `sde` (`_drift` is the concrete-rhs marker, and
  `StochasticSystem` is in the family-base table), so it appears in
  `registry.all_systems(family="sde")`. Three built-in SDEs ship
  (`OrnsteinUhlenbeck`, `GeometricBrownianMotion`, `DoubleWell`), so
  `registry.families()` (builtin-only) includes `'sde': 3`; each built-in SDE has
  an entry in `tests/_sampling.py::SDE_SAMPLES` (a guard test enforces
  completeness, mirroring `DDE_HISTORIES`).
- **The SDE engine path does not use `run.integrate`** (it cannot carry the noise
  seed/step): `backend="interp"/"jit"` dispatch through the dedicated
  `run.sde_integrate_dense` / `run.sde_ensemble_final` seam, and `run.integrate`
  /`run.ensemble` *refuse* an SDE problem.

### Solver-kernel & interpreter performance (Rust engine internals)

The Rust solver kernels (`crates/tsdyn-solvers/`) and the SSA interpreter
(`crates/tsdyn-vm/src/interp.rs`) carry a few work-saving optimisations; all are
answer-preserving (`interp == jit` and equal to the un-optimised path to the
documented tolerance):

- **FSAL (First Same As Last) on the adaptive explicit kernels** — `rk45`,
  `tsit5`, `bs3` reuse the last stage `f` of the accepted step as the first stage
  of the next (a genuine FSAL pair: `c_last = 1`, `a_last = b`). The reuse is
  guarded on the *next* step's first-stage point being bit-for-bit the live state
  (so a rejected trial keeps the reuse valid); the shared implementation lives once
  in the explicit module, and the non-FSAL adaptive kernels (cashkarp/rkf45/
  heun_euler) recompute. (`dop853` does **no** propagation FSAL reuse — it recomputes
  its 12 stages every accepted step; its uncomputed 13th stage is a dense-output stage,
  not a first-stage-reuse hook.)
- **Frozen-Jacobian / LU reuse on the SDIRK / ESDIRK stiff kernels** — `sdirk2`
  and `trbdf2` run a *modified*-Newton iteration (`crates/tsdyn-solvers/src/
  implicit/newton.rs`): freeze the analytic Jacobian `J = ∂f/∂u`, factor the
  iteration matrix `I − coef·h·J` once, and **reuse the frozen `J` + its LU
  factorization across substages and across consecutive steps** while `h` (and so
  the bit-equal shift) holds — `solve_substage_reuse`/`NewtonWork` keep the cache.
  Both stiff kernels' substages share their diagonal coefficient, so stage 2
  reuses stage 1's LU; robustness is held by a refactor-on-degradation trigger.
- **Interpreter dead-register elimination on the RHS-only path** — a
  Jacobian-bearing tape carries registers that feed *only* the Jacobian outputs.
  The RHS-only `Interpreter::eval` runs with a precomputed liveness mask that skips
  every op no RHS output transitively depends on (mirroring the Cranelift JIT's
  `reachable` set), so it never computes Jacobian-only subexpressions; the
  Jacobian path `eval_jac` ignores the mask and runs the full tape.

---

## Derived systems & analysis

- Wrappers forward `params`/`meta`; `with_params()` re-parametrizes the inner
  system and rebuilds the wrapper → orbit diagrams over `PoincareMap` /
  `StroboscopicMap` are bifurcation diagrams of flows (an ODE control-param sweep
  reuses one cached lowered tape — see the tape cache below — so it stays cheap;
  a DDE sweep bakes its delays/params into the tape, so each value is a cache miss
  that re-lowers).
- `PoincareMap` refines crossings with cubic Hermite using `_rhs_numeric`
  (O(dt⁴)); falls back to linear interpolation for DDEs.
- **The section API is named (stream WS-POINCARE-API):** `PoincareMap` and
  `poincare_section` accept the friendly `plane` spelling — `(axis, c)` with
  `axis` a component **name** (resolved against `variables`, e.g. `("y", 0.0)`)
  or an index, `(axis, c, direction)` with a `"up"`/`"down"`/`"both"` word that
  overrides `direction=`, or `(normal, offset)` for an arbitrary normal
  (resolution + direction-word parsing in `derived/poincare.py::_resolve_section_plane`;
  `PoincareMap.plane` is stored as the resolved `(index, offset)`). Both
  `poincare_section` and `PoincareMap.trajectory` return a **`PoincareSection`** —
  a thin `Trajectory` subclass carrying `POINCARE_SECTION` plot intent plus a
  `.summary()`/`.to_dict()`/`.plot` result surface (the WS-WRAP carve-out home).
  The crossing-transient keyword stays the dedicated `skip_crossings` (glossary
  §5: `transient` never counts section crossings).
- **`PoincareMap.trajectory` is engine-native (stream WS-CROSSKERNEL):** the bulk
  crossing collection wires the Rust event engine
  (`integrate_events` → FFI `integrate_events_dense` → `engine.run.crossings` →
  `derived/_crossings.py::section_crossings`), marching the whole attractor and
  refining every crossing in **one engine call** instead of the per-`dt`
  `step()` loop — ~80–100× faster (the named "Poincaré is slow" culprit).
  `poincare_section` and `return_map(kind="poincare")` (which call `trajectory`)
  inherit it; `orbit_diagram` over a `PoincareMap` drives the wrapper with `step()`
  and so is **not** accelerated here (that needs a resumable `step()` — WS-STEPPER —
  or `orbit_diagram` to call `trajectory` — WS-MAPITER). The engine march uses the
  **fixed-step `rk4` kernel at the detection `dt`**. That is a **workaround, not a
  design property**: it was adopted because the engine's adaptive kernels carried
  no step ceiling, so an adaptive march would grow the step, skip crossings and
  degrade the O(h⁴) Hermite refinement. Since v6 the engine *has* a `max_step`
  ceiling (see "Dense output and `max_step`" above), so an adaptive march is now
  possible and the `_EXPLICIT_METHODS` restriction (which excludes the 5
  stiff-default flows from the fast path) could be lifted with `max_step = dt`.
  **Repointing the march is a separate ticket** — it would move every Poincaré
  number and break the answer-identity contract below — so as of v6 nothing here
  changed: `rk4` carries no `Caps::dense`, so the whole
  Poincaré/`return_map(kind="poincare")` surface is byte-identical. It is
  answer-identical to the Python loop driven at the same `rk4`/`dt` discretisation (the engine event refinement reproduces
  `PoincareMap._refine` to ~machine precision per crossing; over many crossings of
  a chaotic flow the two float-distinct computations diverge by roundoff, as any
  two would — the section is the same attractor). DDEs (no `_rhs_numeric`), stiff
  defaults (an implicit `_default_method`), and `backend="reference"` keep the
  Python loop. Divergence / no-crossing-within-`max_time` raises
  `ConvergenceError` (a `RuntimeError` subclass). (The unwired `integrate_events`
  was the dead E-EVENT code.)
- **General events API (stream WS-EVENTSAPI):** `ContinuousSystem.run(events=[...])`
  exposes the same wired event engine generally — a scipy-shaped `events=` surface
  for arbitrary stopping (A-RQA / A-BASIN / custom). An **`Event`**
  (`tsdynamics.engine.run.Event`) is a *symbolic* scalar condition `g(u, t) = 0`
  so one spec drives both paths: a callable `g(y, t)`/`g(y, t, **params)` over the
  engine state accessor (returning one SymEngine expr, optionally with scipy-style
  `.direction`/`.terminal` attributes), or a plane tuple (`("z", 27.0, "up")` /
  `(normal, offset)`, the WS-POINCARE-API spelling). `run(events=)` returns the
  dense `Trajectory` (truncated at the first **terminal** crossing) with each
  event's crossings in `meta["t_events"]`/`meta["y_events"]` (one array per event,
  scipy-named) plus `meta["terminated"]`. The seam is `engine.run.integrate_events`
  (returns an `EventSolution`): the compiled engine runs one `crossings()`
  per event (explicit methods) and coordinates terminal stop; an implicit/stiff
  method or `backend="reference"` routes to `scipy.integrate.solve_ivp(events=)` —
  an independent oracle the tests cross-check (early Lorenz crossings agree to
  ~1e-10; a non-chaotic oscillator to ~2e-13, plus an analytic `cos t`
  zero-crossing check). **Since v6 the engine refines crossings with the kernel's
  own continuous extension** rather than the endpoint cubic-Hermite fallback,
  because `rk45`/`tsit5`/`dop853` now carry `Caps::dense` — that turned on ~40
  lines of engine code which had never executed outside one `#[cfg(test)]` kernel,
  and the SciPy cross-check tightened accordingly (oscillator 1e-6 -> 1e-11,
  Lorenz early crossings 1e-3 -> 1e-8). `PoincareMap` is unaffected: it is pinned
  to `rk4`, which is not dense.
  `PoincareMap.as_events()` returns the section as one `Event`, so the
  section is reproduced through this seam (`PoincareMap` is one consumer).
  Restricted to ODEs (maps have no continuous crossings; DDE/SDE raise).
- **`TangentSystem` is the one Lyapunov engine** (stream C-DERIV): the
  variational/QR machinery lives here, and `DiscreteMap.lyapunov_spectrum` /
  `ContinuousSystem.lyapunov_spectrum` are thin delegations to it. Modes:
  - **maps**: the **engine** QR tangent-map kernel via `backend=`
    (`"interp"`/`"jit"`/`"reference"`, stream `perf/map-lyapunov-kernel`): the
    *whole* QR iteration — propagate `k` deviation vectors `W ← J(x_n)·W` (Jacobian
    at the **pre-image** `x_n`, the correct tangent-map convention), modified
    Gram–Schmidt reortho every `reortho_interval`, accumulate `Σ log|R_ii|`, average
    — runs in **one** Rust call (`engine.run.map_lyapunov` →
    `crates/tsdyn-engine/src/map_lyapunov.rs` → FFI `map_lyapunov_spectrum`) on the
    map tape lowered `with_jacobian=True`, with random-IC retry on divergence. This
    replaces the per-step Python→FFI NumPy QR loop (~6000× faster native than the
    released per-iterate loop). `interp`==`jit` **bit-for-bit** (same lowered tape);
    against the pure-Python `_accumulate_map` oracle it differs only by the
    lowered-IR vs NumPy `_step`/`_jacobian` float order (the WS-MAPITER caveat) —
    the same attractor, the same spectrum to tolerance. `backend="reference"`, a
    non-lowering `_step` (piecewise/ufunc → `TapeCompileError`), and a wheel-free
    environment transparently keep the pure-Python QR loop (`_accumulate_map`, the
    oracle).
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
  deviation **history segment** (a function-space QR, so `k` may exceed
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
  any continuous system including `WrappedSystem` stepped with `dt=None`. **For a
  map (stream `perf/map-lyapunov-kernel`)** it is the **leading exponent of the
  engine QR tangent-map spectrum** (`steps = n·steps_per` from the burnt-in state,
  `k=1`), run in one Rust call — far faster and more robust than the per-iterate
  two-trajectory rescaling (no `d0`/collapse tuning); a non-lowering `_step` or a
  wheel-free env falls back to the two-trajectory loop. The **continuous-system
  path is unchanged** — only the map path moved to the kernel.
- `lyapunov_from_data` (A-LYAP) estimates the maximal exponent from a measured
  series via delay embedding + neighbour divergence (Kantz 1994 default,
  Rosenstein et al. 1993 optional); returns a `LyapunovFromData` carrying the
  stretching curve `S(k)` — fit the linear scaling region (inspect, then pass
  `fit=(lo, hi)`). A private delay-embed helper keeps it independent of the
  delay-embedding stream. **`dt` is read from the data when the data knows it**
  (`_sampling_interval_of`): a `Trajectory` carries its sampling interval — in
  `meta["dt"]`, else in its `t` axis — so `lyapunov_from_data(traj)` /
  `traj.lyap.from_data()` answer **per unit time** and compare directly with
  `lyapunov_spectrum`, while a bare array (no time axis) keeps `dt = 1.0`, i.e.
  per sample / per iteration. Before this the accessor returned the per-sample
  number from an object that was holding the interval — a silent ~60× on a
  Lorenz run at `dt=0.02`. An explicit `dt=` always wins; a non-uniform `t` axis
  raises rather than being averaged into one.
- `fixed_points` (A-FP) finds map fixed points (`f(x)=x`) **and** flow equilibria
  (`f(x)=0`) by multi-start Newton on the analytic Jacobian; `method="sd"`/`"dl"`
  add the Schmelcher–Diakonos/Davidchack–Lai stabilising transformations (maps
  only) to reach unstable points. Map stability is `|λ|<1`, flow stability
  `Re λ<0` (the `FixedPoint.continuous` flag picks the convention).
  **`method="interval"`** (stream `perf/fixedpoints-interval`, engine in
  `_interval.py`) is the *rigorous* alternative: the Krawczyk operator brackets
  **all** roots inside the (required) `region` by interval branch-and-prune with
  an existence+uniqueness certificate per sub-box — so it cannot silently miss a
  root (multi-start can), is deterministic (no `seed`), and is faster on the
  analytic systems it applies to (~2–11× on the catalogue; on Thomas's 27
  equilibria it finds all 27 where `n_seeds=200` Newton finds 23). It builds the
  interval residual+Jacobian by **forward-mode AD over intervals** (`IntervalJet`
  pushed through the map `_step` or the flow's SymEngine RHS tree — no symbolic
  diff), supports `sin/cos/exp/log/sqrt/cosh/tanh/abs`+integer powers (covers
  every catalogue flow + analytic maps), and raises `InvalidInputError` for a
  kernel it can't enclose (a `<`/`%`/non-integer power → use `"newton"`). The
  arithmetic is plain-float (rigorous to round-off, machine-precise roots; a
  *certified* outward-rounded kernel is a future engine-side project). Theory +
  benchmark in `docs/theory/fixed-points-interval.md`. Additive: `newton`/`sd`/`dl`
  unchanged.
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
  subpackage is mostly backend-free (extrema/sweeps over the standard stepping
  API) and self-registers into `registry.analyses`. **Exception (stream
  WS-MAPITER):** when `orbit_diagram` sweeps a genuine `DiscreteMap`, each value's
  `transient + n` run is a single `iterate` call on the Rust engine (one FFI
  round-trip, ~9× faster than the per-step loop) — byte-identical where the engine
  and NumPy agree bit-for-bit (e.g. the logistic map) and same-attractor for a
  chaotic map. Flow wrappers (`PoincareMap`/`StroboscopicMap`), maps whose `_step`
  will not lower, and wheel-free environments transparently keep the stepping API.
- `find_attractors` / `basins_of_attraction` (A-BASIN) drive any map/flow over a
  `CellGrid` tessellation with a recurrence finite-state machine (the
  `AttractorMapper`, Datseris–Wagemakers 2022): a trajectory that recurrently
  re-visits cells has found an attractor, transient cells become its basin, and a
  near-coincident split is proximity-merged (`merge_tol`). Flows step by `dt` per
  cell check, maps by one iteration; a raised/non-finite step is divergence, a
  finite out-of-box excursion uses the lost-counter. **Engine-native march (stream
  `perf/basin-march`):** on a supported run — an ODE flow or a map whose `_step`
  lowers, on the `interp`/`jit` backend — the *whole* per-IC FSM (stepping +
  cell-binning + the shared-label early-out) runs in **one sequential Rust kernel
  call** (`crates/tsdyn-engine/src/basin.rs` → FFI `basin_march_flow`/
  `basin_march_map` → `engine.run.basin_march` → `attractors.classify_seeds`), so
  there is **no** per-`dt` Python→FFI round-trip — fast *without* parallelism (the
  march stays **sequential by design**: the shared, order-dependent labelling is the
  dominant work-saver, and parallelising it costs a measured ~34× over-march). The
  kernel is **bit-identical** to the pure-Python `_AttractorMapper` (it drives the
  same engine stepper per cell check): the basin **label image** and `AttractorSet`
  are byte-for-byte identical **for flows** (both paths advance the same engine
  stepper); for **maps** the label image is *empirically* byte-identical across the
  catalogue but only **same-attractor** guaranteed (a ULP `_step` difference can bin
  a boundary-straddling iterate into a neighbouring cell). For a **map** the located point cloud follows
  the lowered IR vs the pure-Python `_step`, which differ by ULPs, so a chaotic
  map's located cells are the *same attractor a few cells apart* (the WS-MAPITER
  IR-vs-NumPy caveat). `reference`, a non-lowering `_step` (e.g. the complex Newton
  map), and DDE/SDE keep the per-seed **Python loop** (the fallback and the oracle).
  `basins_of_attraction` paints
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

## Visualization (the `viz` seam)

`tsdynamics.viz` is the **backend-agnostic** plotting layer. `import tsdynamics`
pulls in **no** plotting library — `ts.viz` is bound lazily, and every renderer
import is deferred to first render.

### Transforms, primitives and the compatibility matrix (v6)

The plotting layer has **three nouns**, and every new plot is expressed in them.
Nothing else in the library learns a new name when one is added.

- **A transform** turns a *subject* — a `Trajectory`, a system, an analysis
  result, a bare array — into **`Geometry`**: typed channels (`x`/`y`/`z`/`c`/
  `u`/`v`/`frames`/…) in `Part`s, plus a `Frame` (coordinate space + axis names),
  axis labels/limits/**scales**, and provenance. `Geometry` is deliberately *not*
  a second IR: it is never serialized and no renderer ever sees one.
  Transforms live in `viz/transforms/` (`_data.py` = the migrated producers,
  `planar.py`/`fields.py`, `series.py`, `hilbert.py`, `spectra.py`,
  `stability.py`). **A transform owns no new math** — it adapts an estimator
  from `tsdynamics.analysis` (declared in `PlotTransform.analysis`); anything
  needing new numerics gets an analysis function, with its own citation and
  tests, first.
- **Exactly two source categories**, and there is no third. **`data`** =
  computable from a series or a point set (and it *also* accepts a system,
  because a model gives you data for free — it integrates once and records the
  choice in `meta`). **`model`** = must evaluate or integrate the right-hand side
  at points that are **not** in the input (a lattice of ICs, a Jacobian at an
  equilibrium); handing one a bare array raises `InvalidInputError` naming what
  it needs.
- **A primitive** is *how* geometry is drawn — `line`/`line3d`/`points`/
  `points3d`/`steps`/`markers`/`image`/`density`/`contour`/`surface3d`/`quiver`/
  `bars`/`histogram`/`band`/`boundary`/`errorbars` (`viz/transforms/
  _primitives.py`). "Primitive" does not mean "simple": a basin image and a 3-D
  surface are primitives. **Every primitive lowers to the frozen 11-mark
  `PlotKind` vocabulary** — `contour` and `steps` emit plain `LINE` layers,
  `density` emits an `IMAGE` — which is why the matrix can grow without touching
  a renderer or the `PlotKind` enum. A contour's polylines are coloured **by
  level** from the transform's declared colormap (an unstyled layer per level
  would take successive *palette* colours and read as N unrelated series).
- **The compatibility matrix is DECLARED, never implicit.**
  `PlotTransform.primitives` **is** the row, given at the definition site; an
  undeclared pair **raises** `InvalidParameterError` naming the valid set (and,
  when the requested primitive is some other transform's default, naming that
  transform). Never a fallback, never a warning — a renderer's
  `VisualizationDegraded` is a different situation (same plot, different
  backend); here no correct drawing exists.
- **Registry:** `registry.plot_transforms` (record: `PlotTransform`), entry-point
  group `tsdynamics.plot_transforms` for out-of-tree plots. Registering is
  **one decorator call at the definition site and nothing else** — no renderer
  edit, no `PlotKind` edit, no `compose` edit, no test edit, no gallery edit.
  (CLAUDE.md's "no `transforms` registry" note governs the DELETED generic
  time-series layer only. The PSD of a *trajectory* is re-admitted as a plot
  transform under a checked rule — `ADMITTED_SERIES_DIAGNOSTICS` /
  `EXCLUDED_SERIES_TOOLBOX` in `viz/transforms/_registry.py`: the power spectrum
  of a phase-space orbit is a phase-space diagnostic; a PSD toolbox with
  windowing, detrending and filter design is not. `spectrogram` and friends are
  refused *at registration*.)
- **Front doors.** `ts.plot` is the only plotting name in the curated top-level
  `__all__`; `T` was demoted to `ts.viz.T` once a plain `("name", {options})`
  pair became a transform call (nothing requires the type any more).

  ```python
  ts.plot(traj)                                  # no transform named → viz.plot
  ts.plot(traj.y)                                # ...a bare array works too
  ts.plot(traj, color="red", title="Lorenz", theme="dark")   # style AT the door
  ts.plot(traj, "delay_embedding", delay=7)      # a transform by name (7 SAMPLES)
  ts.plot(traj, "phase_portrait", primitive="density")   # …drawn differently
  ts.plot(fhn, "flow_speed", "streamlines", "nullclines")  # overlay, order-free
  ts.plot(vdp, ("flow_speed", {"log": True}), ("streamlines", {"color": "w"}))
  ts.plot(vdp, ts.viz.T("flow_speed", log=True))          # the same thing, typed
  g = ts.viz.geometry(sys, "ftle", grid=201)     # the arrays, and stop there
  ts.viz.draw(g, "contour")                      # hand them back to the library
  ts.viz.list_transforms(); ts.viz.compatibility()   # what can this draw?
  ```

  A positional **string, `("name", {options})` pair, or `T`** is a transform;
  anything else is a subject (a system, a `Trajectory`, a result, **or a bare
  array** — coerced to an index-time `Trajectory` at the door).
  `ts.plot` always returns a `PlotSpec`, so a result feeds straight back in.
  Shared keywords are routed only to the transforms whose `compute` (or chosen
  primitive) accepts them — **plus every canonical style key** (applied to the
  named transform's layers) and `title`/`xlabel`/`ylabel`/`zlabel`/`theme`
  (applied to every panel).
- **The style vocabulary is the same at all THREE plotting doors.**
  `ts.plot(subject, color=…)`, `traj.plot(color=…)` and `system.plot(color=…)`
  peel one derived set — `viz.style.style_names()` (canonical `STYLE_KEYS` +
  aliases) — plus `theme`. They used to disagree in the worst possible way:
  `title=` landed at all three but `color=` raised at the two method doors, and
  on `system.plot` it fell through to the integration and was reported as
  *"color is not a valid integrate()/run() keyword"*, naming a vocabulary the
  caller was not speaking. Style is now peeled **before** the leftovers reach
  `trajectory()`, so an integration typo is still an integration typo. Gate:
  `tests/test_viz_compose.py::test_style_keywords_land_identically_at_all_three_plot_doors`.
- **Writing one is a decorator call and nothing else — for the AUTHOR too.**
  Everything the recipe needs is public (`ts.viz.plot_transform`, `PlotKind`,
  `Geometry`, `Part`, `FrameSpace`, `make_frame`, `Presentation`); no private
  module is imported. `compute` may return a **plain mapping of channels**
  (`{"x": ..., "y": ..., "z": ...}`) and the registry stamps the name, frame,
  ndim and `labels=` from the decorator, which already declared all four —
  available whenever the transform declares a single `(frame, ndim)` pair. The
  declared `ndim` is now **checked** against the returned frame (an ndim mismatch
  used to be accepted silently).
- **Gates.** `tests/test_viz_compatibility.py` renders every declared cell on
  matplotlib and refuses every undeclared one; `tests/test_viz_transforms.py`
  pins the substrate and the PSD admission rule;
  `tests/test_viz_gallery.py` pins the gallery against the registry (and, in the
  slow tier, renders every cell and fails on a blank figure).
- **The gallery (`docs/visualization/gallery.md` + `docs/_tooling/gallery.py`).**
  The user-facing answer to *"what can this draw?"*, **generated from the
  registry at docs-build time**: one entry per transform (grouped by source
  category), one tab per declared primitive, and beside each figure the code that
  produced it — the snippet string is *executed*, so code and picture cannot
  drift. A curated `SHOWCASE` table picks the subject; a transform with no entry
  still appears, drawn on the `example` factory its registration already ships
  (so "one registration and nothing else" holds). Figures are content-addressed
  in `.cache/docs-gallery` (bump `RENDERER_VERSION` when the *rendering* changes)
  and registered as generated files, so nothing is written into `docs/`; a failed
  cell warns through the `mkdocs` logger, which `--strict` turns into a build
  failure. Run it standalone with
  `.venv/bin/python docs/_tooling/gallery.py [--only NAME] [--force]`.

### The rest of the seam

- **`PlotSpec` IR (`viz/spec.py`):** a JSON-serializable description of a plot —
  a semantic `PlotKind`, drawable `Layer`s, typed `Axis`/`Colorbar`/`Legend`,
  and `to_dict`/`from_dict` round-trip. The `PlotKind` enum is a **frozen,
  reviewed contract** (governance gate `tests/test_viz_vocab.py` pins the exact
  membership; adding a kind edits that gate deliberately).
- **Renderers (`viz/render/`):** in-tree backends `mpl` (the universal reference
  renderer — `kinds=None`, draws everything, the fallback), `plotly`
  (interactive 2-D + 3-D + HTML), `json` and `threejs` (data-export). Dispatch
  (`viz/render/__init__.py`) selects by name or capability and falls back to mpl
  with a `VisualizationDegraded` warning when a backend declines a spec.
  **matplotlib is the deterministic default** (`_PREFERRED_DEFAULT_BACKEND`,
  `_seat_preferred_first`): a no-backend `render()` is matplotlib on the first and
  every subsequent render (the old register-order-dependent default-flip is gone);
  a custom/registered backend is reached only by an explicit `backend="name"`
  (`caps`-normalised aliases like `"mpl"` accepted). json never draws — it
  serializes — so it is exempt from the honoring negotiation below.
- **Styling & theming (`viz/style.py`):** the look of every plot is controlled by a
  **canonical, validated, introspectable** vocabulary, honored consistently across
  the three *visual* backends (matplotlib/plotly/threejs; json serializes it). The
  pieces:
  - **`STYLE_KEYS`** — the closed per-layer style vocabulary (`color`, `linewidth`,
    `linestyle`, `marker`, `markersize`, `alpha`, `cmap`, `fill`, `fillalpha`,
    `zorder`), each a `StyleKey(name, aliases, honored_by, validate, doc)`.
    `normalize_style()` is the single choke point: it canonicalises aliases
    (`lw`→`linewidth`, `c`→`color`, `ms`→`markersize`, `"--"`→`"dashed"`,
    `"o"`→`"circle"`, …), validates values (rejects out-of-range / wrong-type,
    incl. bool), and **drops unknown keys with a warning** (no more silent typos).
  - **`honored_by` is an enforced contract.** Each `StyleKey` declares which
    backends genuinely render it; `caps.style_honoring_gaps(spec, backend)`
    collects every per-layer key, `Animation` knob, and `Theme` field the chosen
    backend does **not** honor, and the dispatcher emits **one consolidated
    `VisualizationDegraded`** per render naming them (renderers then run
    `warn=False`). `tests/test_viz_honoring_contract.py` renders every
    honored claim and asserts the artifact reflects it (and that every non-honored
    key warns) — an overclaim cannot ship green. (`fill`/`fillalpha` apply to AREA
    marks only; `cmap`/`linestyle`/marker-shape are not honored by threejs.)
  - **`Theme`** (a **frozen** dataclass: palette, background, foreground, font,
    grid, line/marker defaults) + the **`THEMES`** registry with four built-ins
    (`default`/`dark`/`minimal`/`publication`) and a single mutable global default
    via `set_theme`/`get_theme`/`themes`/`register_theme` (the **only** mutable
    viz global — `tests/conftest.py` has an autouse fixture snapshotting+restoring
    it around every test). A `PlotSpec` carries a private `_theme`; renderers read
    `spec.resolved_theme` (the spec's theme, else the global default) and apply it
    first (palette colours unstyled layers), then per-layer style overrides it.
  - **Fluent tweaks** (all mutate-and-return-self, so they chain and render
    identically on every backend): `.style(**keys)`, `.recolor(*colors)`,
    `.theme(name|Theme, **overrides)` (a setter; `theme` is positional-only),
    `.palette(...)`, `.grid(...)`, `.font(...)`, `.background(...)`, `.size(...)`,
    alongside the existing `.relabel/.rescale/.limits/.ticks/.colorize/.animate/…`.
    Public introspection: `ts.viz.STYLE_KEYS`, `ts.viz.themes()`,
    `ts.viz.get_theme()/set_theme()`. (Full guide: `docs/visualization/styling.md`.)
- **Single-panel front door:** `Trajectory.to_plot_spec(...)` / `.plot(...)` (see
  the `Trajectory` section) builds **one panel**.
- **Composition — `tsdynamics.viz.plot(*things, layout="overlay", **build_kw)`
  (`viz/compose.py`):** the figure-level front door. It converts each thing (a
  `Trajectory` / system / result / `PlotSpec`) to a spec — forwarding `build_kw`
  (`components` / `kind` / per-kind options) to each — and returns a **`PlotSpec`**:
  - `layout="overlay"` (default) merges overlay-compatible single-panel specs
    (`TIME_SERIES` / `PHASE_PORTRAIT_2D` / `PHASE_PORTRAIT_3D`) onto **one** set of
    axes, disambiguating legend labels by source; incompatible kinds (an image vs
    a portrait, or 2-D vs 3-D) raise — use a panelled layout instead.
  - `layout="stack"/"row"/"grid"` builds a `PlotKind.COMPOSITE` spec carrying
    child `panels` (each a single-panel `PlotSpec`) and a `Layout` (`mode` +
    `rows`/`cols`/`share_x`/`share_y`). Composite inputs are **flattened** one
    level, so `plot(plot(...), plot(...), layout="stack")` composes (input type ==
    output type == `PlotSpec` → fully recursive).
  The returned spec **renders itself**: `PlotSpec` has `.plot()` (inline-tweak +
  render), `.render(backend=)`, `.save(path)` (raster/vector → matplotlib, `.html`
  → plotly, by extension), and a notebook `_repr_mimebundle_`.
  - **Composite rendering:** **both** in-tree drawing backends tile `panels` into
    a subplot grid. The **matplotlib** renderer (`_render_composite` / per-panel
    `_draw_2d_panel` / `_threed._draw_3d_panel`; 2-D panels optionally share axes).
    The **plotly** renderer (`viz/render/plotly/_composite.py::render_composite`)
    tiles natively via `plotly.subplots.make_subplots(rows, cols, specs=…)`: each
    panel's cell is typed `"scene"` (a 3-D panel) or `"xy"` (a 2-D panel) so a
    composite **mixing** a time-series panel and a 3-D portrait renders with each
    panel on the right subplot type; per-panel traces are built by the factored
    single-panel cores (`_core.build_2d_traces` / `_threed.build_3d_traces`) and
    added with `add_trace(…, row, col)`, and per-panel colorbars are repositioned
    into each panel's own domain (`_place_colorbars`) so stacked images do not
    collide. `.save("fig.html")` on a composite therefore yields one interactive
    multi-panel page. The capability check recurses into `panels`
    (`RendererCapabilities.can_render_spec`), so plotly still falls back to mpl
    when a panel uses a kind it declines. (Plotly **declines** only `COMPOSITE`
    *animations* for now — `viz/render/plotly/_anim.py` is single-panel.)
- **Animation — an orthogonal modifier (`viz/spec.py::Animation`,
  `PlotSpec.animation`):** any spec of any `PlotKind` (single-panel or composite)
  becomes a movie by carrying an `Animation`; the semantic `kind` is unchanged and
  a backend that cannot animate draws the final frame. Built via
  `to_plot_spec(animate=True | dict | Animation)` / `ts.viz.plot(..., animate=...)`,
  then tuned with the chainable spec methods `.animate(fps/duration/loop/pingpong)`
  / `.trail(length=("time"|"steps", v) | None, fade)` / `.head(show/size/color/symbol)`
  / `.camera(elev/azim/spin)` / `.clock(fmt)` (all mutate-and-return-self, composing
  with the static `.relabel`/`.rescale`/`.limits`/`.style(…)`/… tweaks — including
  `.style(axes=False)`, which records `meta["axes_visible"]=False` and every
  renderer honors it: mpl `ax.set_axis_off()`, plotly axis/scene `visible=False`,
  for a clean "attractor floating in space" still or animation). There are **two
  frame models** (`Animation.mode`). **`reveal`** (the default): the layer keeps
  its full static data and each frame shows a comet — head at the current sample,
  tail reaching back `trail_length` (`None` ⇒ persistent); the frame math lives on
  `Animation` (`head_indices`/`tail_samples`/`frame_count`). Per-kind head default:
  on for portraits / spacetime, off for a plain time series; a composite plays
  panels in **lockstep** on one master clock (each panel keeps its own per-kind
  head). **`frames`** (stream VIZ-SPATIAL-FIELD): a **spatial-field movie** — the
  field of a spatially-extended system (a method-of-lines PDE) *played over time*.
  Each frame is the field's **spatial** state at that instant, and the per-frame
  plot's shape follows the field's spatial dimensionality: a **1-D field** `u(x)`
  is a travelling-wave **line** (the profile — Kuramoto–Sivashinsky), a **2-D
  field** `u(x,y)` an `imshow` **heatmap** movie (Gray–Scott / Swift–Hohenberg).
  ONE semantic kind covers both — the new `SPATIAL_FIELD` `PlotKind` (the
  renderer dispatches on the field's spatial ndim, like `to_plot_spec`
  auto-dispatches on component count). The producer
  (`viz/producers.py::spatial_field`) stacks every per-time snapshot on the
  layer's `"frames"` channel (shape `(T, *spatial)`) and keeps the **final** field
  as the static layer data (`z` for 2-D / `y` for 1-D), so a still save / a backend
  that can't animate draws the final field; the mpl renderer plays the stack frame
  by frame (`viz/render/mpl/_anim.py::_field_movie_driver` → `_field_movie_2d` /
  `_field_movie_1d`), consecutive frames carrying genuinely different data. **Front
  door:** `system.to_plot_spec(kind="field", animate=True)` — the `"field"` recipe
  routes via `_KIND_ALIASES`; the spatial layout comes from the **system**, via the
  optional `_field_shape: tuple[int, ...]` ClassVar (recorded onto
  `traj.meta["field_shape"]` at integration time, so a bare `Trajectory` carries
  it). No `shape` kwarg: a system with no `_field_shape` (or a 1-D one) is a 1-D
  profile (honest — never guesses a 2-D grid). A multi-block field state declares
  `field_labels` (e.g. Gray–Scott's `("u", "v")`); `components="u"|"v"` picks the
  block, defaulting to the **last** (the activator). The field movie is
  **matplotlib-only** (mp4/gif): plotly *declines* an animated `SPATIAL_FIELD` (a
  static field it draws), threejs draws a 1-D profile / declines a 2-D field.
  Rendering: **matplotlib** →
  `viz/render/mpl/_anim.py` builds a `FuncAnimation` (`.save("x.mp4"/"x.gif")` via
  ffmpeg/pillow); **plotly** → `viz/render/plotly/_anim.py`: HTML export
  (`.save("x.html")`, the zero-extra-dep default) is a **real-time** animation — the
  full attractor is drawn once (static, rotatable) and a `requestAnimationFrame`
  loop streams a comet by **mutating its trace buffers in place with
  `Plotly.extendTraces`** (append the next points, trim to a `maxPoints` sliding
  window; the single-point head uses `maxPoints` 1; once per loop the window resets
  to the start via `restyle`). **Rotate-while-playing for 3-D is via pause-on-drag**
  (verified in a real headless-Chrome drag probe): a gl3d (`Scatter3d`) trace update
  has `editType` `calc`/`plot`, so it forces a **full WebGL scene replot** — there is
  NO lightweight position-only gl3d update (the in-place `scattergl` batch update is
  2-D only), and replotting each frame *while* the user orbits **cancels the drag
  gesture** (and pegs the thread). So the rAF loop **fully suspends the comet stream
  for the duration of a drag**: a **capture-phase `pointerdown`** on the graph div
  (fires *before* plotly's own canvas handler, so not one `extendTraces` lands after
  the gesture starts) sets a `dragging` flag and the loop does **zero trace work**
  until `pointerup`/`pointercancel`, leaving the gl3d scene free to orbit as smoothly
  as a static 3-D plot; the stream resumes from where it paused on release. The live
  drag camera is mirrored into `gd.layout.scene.camera` (via `plotly_relayouting`) so
  that first resumed redraw keeps the dragged pose — **no snap-back** — and a constant
  `uirevision` holds the camera. (Earlier attempts that kept *streaming* during the
  drag — a `Plotly.react` rebuild, then plain `extendTraces` — could not be rotated:
  every per-frame gl3d replot ate the orbit gesture. The honest fact is plotly gl3d
  cannot stream data while you orbit; pausing the stream during the drag is the fix.
  2-D plots have no orbit gesture and `scattergl` updates cheaply, so the pause is a
  no-op there.) The full-curve cache is
  read from plotly's **decoded** data (`gd._fullData`, a `Float64Array`), not raw
  `gd.data` — plotly 6 stores a base64 typed-array *spec* (`{dtype, bdata}`, no
  `.slice`) in `gd.data[i].x`, so the loop normalises each axis up front (`asArray`)
  + slices via `Array.prototype.slice.call` (plain arrays, so `extendTraces`'
  type-match is satisfied), or the per-frame read throws on frame 0 and the curve
  stays static (issue #464). The comet/head traces are emitted as **plain Python
  lists** (not numpy → not a `{bdata}` spec) so `extendTraces` can append to them.
  A minimal **play/pause + restart overlay**
  (bottom-left, with a % readout) makes it obviously alive without devtools.
  A returned live figure (notebooks) instead uses a plotly frames + play/slider
  player (`build_animated_figure`). Camera-spin/clock are mpl-only for now.
  **threejs** → the data exporter (`viz/render/threejs/_lower.py`) adds a
  `metadata.animation` block (`fps`/`duration`/`trail_length_samples`/`head`/
  `n_samples`) when `spec.animation` is set — the geometry buffers are untouched
  (byte-identical to the static export). The reference loader
  (`docs/_static/tsdyn-threejs-loader.js`) honors it with a **reveal comet**: a
  faint full-curve backdrop + a bright windowed trail advanced by
  `geometry.setDrawRange(start, count)` + a `THREE.Points` head, on a
  `requestAnimationFrame` clock with a play/pause overlay — and because the
  draw-range update is independent of `OrbitControls`, the camera is held still by
  default yet **orbitable while it plays**. The reveal sweeps a **line** index
  buffer, so the block is emitted only for `LINE`/`LINE3D` specs (`_ANIMATED_MARKS`);
  an animated `points`-only / `surface`-only spec has no comet to play, so the
  exporter drops to a static payload and **warns** (`VisualizationDegraded`) rather
  than silently dropping it (and the loader auto-rotates, never freezing the
  camera). A static payload (no `metadata.animation`) renders exactly as before.
  `PlotSpec.save` picks the
  backend by extension (animated: `.html`→plotly, `.mp4`/`.gif`→matplotlib) and
  takes `fps`/`dpi`/`size`. The `Animation` directive round-trips through
  `to_dict`/`from_dict` and adds no `PlotKind` (the frozen vocab is untouched).

---

## Code conventions

- **Formatter/Linter:** `ruff format` / `ruff check` (line length 100; D rules on)
- **Types:** `mypy --strict src/tsdynamics` is **green and CI-gated** (the
  `typecheck` job in `ci.yml`). The core library is fully strict; the system
  *catalogue* (`tsdynamics.systems.*`) relaxes exactly the three codes inherent to
  its framework-contract kernels — `override`/`no-untyped-def`/`no-untyped-call`
  (the `_equations`/`_step`/`_jacobian`/`_drift`/`_diffusion` math bodies, whose
  parameters arrive positionally by `params`-dict order) — via a documented
  `[tool.mypy.overrides]` block. Map kernels are plain `@staticmethod`, so the
  type checker reads a map's `_step(x, y, a, b)` first argument as state, not
  `self`.
- **Typed errors (stream WS-ERRORS, `tsdynamics.errors`):** the library raises its
  own exception types, each subclassing the builtin a caller would historically
  catch (the hierarchy is purely *additive*) — `InvalidParameterError` (a
  `ValueError`: a bad `dt`/method/backend/section/param value), `InvalidInputError`
  (a `TypeError`: a malformed argument, e.g. an array of the wrong shape),
  `ConvergenceError` (a `RuntimeError`: divergence / non-convergence / no crossing),
  and `BackendError`/`EngineNotAvailableError` (a `RuntimeError`: the compiled engine
  is unavailable). These are wired across the families and `engine/run.py`
  (divergence → `ConvergenceError`; an out-of-range `dt`/unknown backend/bad
  argument → `InvalidParameterError` / `InvalidInputError`) so a `RuntimeError` /
  `ValueError` / `TypeError` `except` keeps catching the same failure.
  **The mapping is raised at the FFI boundary itself** (v6 WP1,
  `crates/tsdyn-core/src/lib.rs::to_py_err`): `EngineError::Diverged` →
  `ConvergenceError` and `EngineError::InvalidParameter` → `InvalidParameterError`
  are constructed by importing `tsdynamics.errors` from the binding, so **every**
  engine surface (integrate / DDE / SDE / map / events / stepper / basin /
  Lyapunov) inherits the typed error and no call site sniffs a message for
  `"diverg"`. The bridge's split is deliberate: `EngineError::BadShape` is call
  *geometry* (lengths / dims / tape structure → plain `ValueError`),
  `EngineError::InvalidParameter` is a scalar *option value* (`rtol`/`atol`, a
  step/cadence/delay, the integration window, the event direction).
  **Tolerances are validated by construction**: `build_solver` takes a
  `marshal::Tolerances`, whose only constructor rejects a non-finite or negative
  `rtol`/`atol` and the unsatisfiable `rtol == atol == 0` (exactly one zero is
  legal — pure relative / pure absolute control), so a new tolerance-taking
  surface cannot forget the guard. `utils/grids.make_output_grid` likewise
  requires a **finite** `dt`/`t0`/`final_time`, not merely `dt > 0`.
  **`StepBudgetError`** (a `ConvergenceError` subclass, v6 WP2) is the *stalled*
  half of "did not reach the final time": hitting the engine's per-segment step
  cap with a **finite** state is a solver-settings problem (looser tolerance,
  an implicit method), not a blow-up, and the engine no longer calls it one
  (`EngineError::StepBudget`). An engine allocation failure raises `MemoryError`;
  an interrupt re-raises the signal handler's own `KeyboardInterrupt`.
- **Process safety (stream v6 WP2, `crates/tsdyn-engine/src/{alloc,pool,interrupt}.rs`):**
  three infrastructure modules exist so that an engine call can never damage the
  *process* hosting it. Keep new engine code inside them.
  - **`alloc::try_zeroed(rows, cols)`** is how every caller-sized **output**
    buffer is allocated (map orbits, dense grids, DDE/SDE grids, the sweep's
    point buffer). `vec![0.0; rows * cols]` either panics `capacity overflow` or
    — worse — calls `handle_alloc_error`, which **aborts the interpreter**;
    `try_zeroed` does `checked_mul` + `try_reserve_exact` and returns an error
    the bridge maps to `MemoryError`. The engine's *working* buffers are sized by
    the (bridge-validated) tape and stay on the infallible `vec![]` form.
  - **`pool::with_pool`** owns the engine's rayon pool, **PID-tagged and rebuilt
    after a `fork()`**. Every parallel loop goes through it; the ambient global
    pool must not be used, because its workers do not survive a fork and the
    first parallel call in a `multiprocessing` child hangs forever on them.
  - **`interrupt::Poller`** makes long calls Ctrl-C-able. Engine calls run with
    the GIL **released** (`tsdyn-core`'s `detached()` wraps `py.detach` *and*
    arms the thread), so CPython only sees the signal when the call returns; the
    loops therefore poll a hook — installed once by the binding, calling
    `py.check_signals()` — every `POLL_STRIDE` (4096) **units of engine work**
    (one solver step / one map iterate), on the calling thread only (never a
    rayon worker). Measured cost on a 1e6-step run: none above noise. Every
    long-running loop polls: `integrate`, `map`, `dde`, `sde`, `event` (both
    marches), `param_sweep`, `map_lyapunov`, `basin`, and — via
    `integrate_grid_polled`, which threads **one** poller through many short
    segments — `lyapunov`. A per-segment poller would reset before reaching a
    stride, which is exactly why that variant exists.
  - **The three ensembles cancel through a flag, not the hook**
    (`interrupt::Cancel` + `pool::with_pool_interruptible`). A fan-out runs every
    trajectory on a *worker*, and workers are never armed, so the per-trajectory
    poller could never fire — and `ThreadPool::install` additionally **parked the
    one armed thread** for the whole batch. So an ensemble ignored Ctrl-C
    entirely, on precisely the calls most likely to run for minutes. The fix has
    two halves, and both are required: the fan-out is handed to a **scoped driver
    thread** (which may park), leaving the calling thread in a `recv_timeout` loop
    that consults the hook every 5 ms and raises a shared `Cancel`; and each
    worker `interrupt::watch`es that flag for the life of one trajectory, reading
    it — a **relaxed atomic load, no GIL, no shared lock** — on its normal
    `Poller` stride. `ensemble_final` / `iterate_ensemble_final` /
    `sde_ensemble_final` return an `interrupted: bool`, which the bridge turns
    into `EngineError::Interrupted`; the flag is authoritative rather than a scan
    of `status`, because a batch cancelled at the very end has already consumed
    the signal and must still report it. **The batch partition is untouched**, so
    parallel == serial stays bit-for-bit (verified: an N-IC batch equals N
    one-at-a-time calls with `max|diff| == 0.0` on interp and jit, ODE and map, at
    1/4/8 threads, and SDE member `i` still depends only on `seed_for(seed, i)`).
    Cost on the headline path (1000 Lorenz ICs, interleaved same-binary A/B):
    **+0.9% / +0.7% / +1.0% / −0.2%** at 1/2/4/8 threads; the driver thread is a
    fixed ~35–55 µs per *call*, so it is only visible on sub-millisecond batches,
    and it is skipped entirely when the caller is not armed.
  - Each family has its own `Interrupted` variant that unwinds to
    `EngineError::Interrupted`. Two loops must **not** fold it into a divergence:
    the Lyapunov chunk loop (`lyapunov::classify`) and the basin march (whose
    `Advance` enum replaced a `bool` so an interrupt cannot be painted into the
    basin image as a diverged IC).
  - **Divergence is caught by magnitude, not by waiting for `inf`:**
    `integrate::OVERFLOW_SCALE = 1e150` (`√f64::MAX`, an *overflow* scale, not a
    tuned physical one) — the same one-comparison-per-component guard runs in the
    event marches. `bdf` shares the implicit kernels' `STEP_FLOOR_REL`; without
    it a blow-up took ~18 s to report where the others took ~0.02 s.
  - Non-test `debug_assert!`s stay debug-only **except** in
    `tsdyn-jit::JitEvaluator`, whose three buffer checks guard raw-pointer writes
    from compiled native code and so are real `assert!`s. Everything else is
    either validated at the bridge (`marshal::validate_grid`, the per-entry-point
    shape checks) or fails as a bounds-checked panic, not memory unsafety.
- **Docstrings:** NumPy convention; cite the **original paper** for each method.
  This is a code-style norm — keep code/docstrings pointing at the source
  literature rather than at whichever library we consulted. It is **not** a veto
  on naming other tools: the docs (in particular References → benchmarks) may name
  and compare against other libraries, including DynamicalSystems.jl, where a
  head-to-head is informative. (There is no citation-lint build gate; that hook
  was removed.)
- **Commits:** Conventional Commits; PRs are squash-merged and the PR title
  becomes the release-deciding commit (enforced by `pr-title.yml`)

Run before pushing:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy --strict src/tsdynamics     # type gate (CI-blocking; must be clean)
make test                              # change-scoped fast tier (the loop — see below)
make test-slow                         # change-scoped slow tier, if you touched anything heavy
TSD_DOCS_FIGURES=0 uv run mkdocs build --strict   # docs sanity (only if docs/ changed)
```

`make test-all` / `make test-full` run the *whole* fast / fast+slow tiers
(parallel) for a final pre-push sanity check; `uv run pytest -m full --no-cov` is
the exhaustive nightly sweep. **Do not reach for the full suite as your routine
inner loop** — see below.

### Change-scoped testing (stream CI-CHANGED) — use this, not the full suite

The bulk suite is registry-driven (every test parametrized over all 177 systems
+ every analysis), so a plain `uv run pytest` is thousands of items and
takes minutes. **To check your work, run only what your diff touches:**

```bash
make test            # = uv run pytest --changed -m "not slow and not full" --no-cov -n auto
make test-slow       # = the slow tier, change-scoped
uv run pytest --changed -m "not slow and not full" --no-cov -n auto   # equivalent
uv run pytest --changed --changed-since=HEAD~3 ...                    # custom diff base
```

`--changed` (implemented in `tests/_changed_select.py`, wired through
`conftest.py`) diffs the working tree vs `origin/main` (override with
`--changed-since=REF` or `$TSD_CHANGED_BASE` / `make test BASE=<ref>`) and selects:

- a touched **system module** → only that module's systems in the per-system
  sweeps (mapped through the live registry by module name), **plus** the
  registry-blind catalogue RHS-correctness gates (`test_equation_reference.py`
  golden snapshot, `test_catalogue_literature.py`, `test_xval_catalogue.py`) via
  the `_CATALOGUE_GATES` table — a kernel-body edit cannot reach those by
  per-system scoping, so they are selected explicitly;
- a touched **analysis area** (`analysis/<area>/`) → that area's test files;
- a touched **test file** → that file; plus cheap registry/layout guard tests.

It is **biased to over-select**: any *foundational* change (the engine / solver /
family / derived / data / utils layers, the registry or package `__init__`, a
shared test fixture, `pyproject`/`uv.lock`, **any Rust crate**, or **any CI
workflow**) — or any path it does not recognise — disables selection and runs the
full tier. The selector prints exactly what it kept and why (`[changed-select] …`).

This is the same mechanism PR CI uses (`ci.yml` / `engine-bindings.yml` pass
`--changed`). The **full** suite is *not* skipped forever: it runs on every push
to `main` (the release gate in `release.yml`, with coverage) and every night
(`nightly.yml`, `-m full`). So a mis-scoped selection can only delay catching a
regression to the merge/nightly — it cannot ship one. Add a new analysis area?
Map it in `_AREA_TESTS` (the guard `tests/test_changed_select.py` fails until you
do). **CI is fast now — agents and humans alike should iterate with `make test`,
never the full `uv run pytest`, for routine work.**

### Test harness (stream I-QA)

The suite has three layers, all **registry-driven where possible** so new
systems/analyses join the sweeps with zero test edits:

- **Registry sweeps.** `tests/conftest.py` parametrizes fixtures over the
  registries: `ode_entry`/`dde_entry`/`map_entry`/`sde_entry`/`system_entry`
  (built-in systems) and `analysis_entry` (the D4 `registry.analyses` plugin
  surface).
  `tests/test_analysis_registry.py` runs the meta-QA over every registered
  analysis (callable, documented, round-trips, top-level export
  agreement) plus headline-membership guards.
- **Property tests (Hypothesis).** `tests/test_property_*.py` assert
  *mathematical invariants* of the analysis layer (embed value
  preservation, recurrence symmetry +
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
- **Catalogue RHS correctness gate.** `tests/test_equation_reference.py` defends
  the `_equations`/`_step`/`_drift` kernels against transcription / operator-
  precedence bugs (the class that produced the `WindmiReduced` `p**1/2` defect) in
  two layers: a curated set of well-known systems whose RHS is checked against an
  independent hand-derivation at a fixed state, plus a **golden snapshot** — the
  lowered-tape hash of every catalogue system pinned in
  `tests/_equation_reference_golden.txt`, so any unintended change to a kernel
  fails the snapshot test and names the system (regenerate only after a reviewed
  change).
- **SDE coverage gate.** `tests/test_sde_coverage.py` exercises what the empty
  registry SDE sweep cannot — a **multi-dimensional** state-dependent diagonal-Itô
  SDE (per-component Wiener substrate + Milstein correction across a vector state)
  and the **analytic moments** of the canonical Ornstein–Uhlenbeck / geometric
  Brownian motion processes.
- **Tape-cache gate.** `tests/test_lowering_cache.py` proves the lowered-tape
  cache is correct: a control-param sweep is served from the cache (one tape),
  a structural / map / DDE param change or a monkeypatched kernel is a deliberate
  miss, and a `TSDYNAMICS_NO_TAPE_CACHE` run is value-identical to a cached one.

When adding an analysis, the registry meta-QA picks it up
automatically (give it a docstring, register it). When adding a property test,
reuse `_strategies` and assert a real invariant — never a tautology.

### Documentation is executable and published BY DEFAULT (v6, `docs-truth`)

Two docs contracts were inverted in v6. Both default to *included*, so the way
to opt out is an explicit, reviewed entry — never silence.

- **The doctest gate discovers its subjects** (`tests/_doctest_select.py`).
  Every module under `src/tsdynamics` containing a `>>>` and every `docs` page
  with a runnable ` ```python ` fence is executed under
  `filterwarnings = error`. It used to be an **allow-list** naming 20 modules and
  22 pages — while **12 of the 25 modules it did not name were failing**, and 3
  of the pages it *did* name had no runnable fence and passed vacuously. Leaving
  the gate now costs a named entry with a written reason in `EXEMPT_MODULES` /
  `EXEMPT_PAGES`, and exemptions are **self-cleaning**: a `full`-tier guard
  re-runs each and **fails when one starts passing**, so the list can only
  shrink. The 12 current exemptions are all *docstring* defects in `src/`
  (a missing expected-output line, a comment on a `want` line, a name the
  docstring never binds) — not defects in the code they document.
  - Gated today: **27 modules + 41 pages** (was 15 + 20).
  - A fence that is a signature listing, a calling pattern, or a deliberate
    demonstration of what *raises* opts out **in place** with `# skip-doctest`
    — visible to the page's reader, unlike a name in a list elsewhere.
  - **Known gap — the gate does not fire on docs-only PRs.**
    `tests/_changed_select.py` lists `docs/`, `*.md`, `README.md` and
    `mkdocs.yml` under `_IGNORE_PREFIXES`/`_IGNORE_SUFFIXES`/`_IGNORE_FILES`
    ("no bearing on the test suite"), so a diff touching only documentation
    selects three cheap registry guards and **not** `test_doctests.py` —
    verified: `classify(['docs/analysis/lyapunov.md'])` does not select it.
    The full suite on merge/nightly still catches it, so a broken example is
    delayed rather than shipped. Fixing it means mapping those paths to
    `test_doctests.py` instead of ignoring them.
- **`mkdocs.yml` publishes every page except a short enumerated list.** The
  blanket tree-drop `theory/` had removed ~1.2k lines of accurate documentation
  from the site — including `theory/fixed-points-interval.md`, which a **shipping
  source docstring** (`analysis/fixedpoints/_interval.py`) links to — and nothing
  failed. Only the three pre-generation `systems/*/index.md` stubs (replaced by
  `hooks/docs_autogen.py`) and the non-page asset dirs are excluded now;
  `tests/test_doctests.py` fails if that set grows or if any content tree is
  dropped wholesale. `validation.unrecognized_links` is **`warn`** (was `ignore`),
  so a broken internal link fails `mkdocs build --strict`.

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
- Workflows: `ci.yml` (PR gate — **change-scoped**: a `changes` filter skips the
  Python jobs on out-of-scope PRs; `build-engine` compiles the abi3 wheel ONCE per
  OS and shares it to the test matrix as an artifact (no per-cell recompile);
  `test-fast` (os×py matrix) and `test-slow` (one cell) install that wheel and run
  `pytest --changed … -n auto`, so only the diff's tests run, in parallel),
  `docs.yml` (build + Pages deploy with figure cache), `release.yml` (the full
  suite + coverage on every push to main — the un-scoped safety net + publish
  gate), `pr-title.yml`, `nightly.yml` (`-m full`), `rust-workspace.yml`
  (the pure-Rust tsdyn-* workspace), `engine-bindings.yml` (the focused engine job:
  tsdyn-core fmt/clippy/cargo-test + the **engine-marked** Python tests run
  `--changed`, including the catalogue gate `tests/test_xval_catalogue.py`),
  `wheels.yml` (cross-platform
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
- **Packaging shape:** the project ships as **one maturin wheel** —
  the pure-Python `tsdynamics` package (from `src/`, `python-source="src"`) plus
  the compiled `tsdynamics/_rust` abi3 extension (`module-name="tsdynamics._rust"`,
  `manifest-path="crates/tsdyn-core/Cargo.toml"`) in the same wheel. abi3 (cp312)
  means one wheel per (platform, arch) covers every CPython ≥ 3.12. Full rationale
  + recipe: `docs/theory/packaging.md`; invariants guarded by
  `tests/test_packaging.py`.
- **The PyO3 bridge is a package (`crates/tsdyn-core/src/bridge/`):** the FFI
  marshalling layer (once a single `bridge.rs`) is now a `bridge/` module —
  `marshal.rs` (the shared tape/array ⇄ FFI conversions), one file per family/
  surface (`ode.rs` / `dde.rs` / `sde.rs` / `map.rs` / `events.rs`), the resumable
  `stepper.rs` (`OdeStepper`), and `mod.rs` (the `#[pymodule]` entry binding every
  exported FFI function). The exported FFI names are unchanged — it is a pure
  Rust-side split.

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

## Tape lowering & the in-process tape cache (no on-disk cache)

The engine lowers each system to an in-process IR tape on first use — there is
**no on-disk compile cache** and no C-compilation step (the old
`~/.cache/tsdynamics/` JiTCODE/JiTCDDE cache and the `TSDYNAMICS_CACHE` override
are gone). Editing an `_equations`/`_step` body just takes effect on the next
run; nothing to wipe.

- Control parameters are read live from the system on every run (through
  `problem.params_vec()`), so a parameter change never bakes into the tape. A
  *delay* value (DDE) or a structural parameter **is** baked into the tape.
- **In-process lowered-tape cache (stream PERF-LOWER-CACHE, `engine/compile.py`):**
  lowering is a pure function of the *math* (kernel body, dimension, structural
  parameters, DDE/map params, the `with_jacobian` flag), so the lowered `Tape`
  (and `LoweredSDE`) is memoised by `lower_ode_cached` / `lower_map_cached` /
  `lower_dde_cached` / `lower_sde_cached`. The key carries the system class,
  `with_jacobian`, dim, the structural (ODE/SDE) or all (map/DDE) parameter
  values, and the **kernel callable object itself** (so a monkeypatched/redefined
  kernel is a deliberate miss — no stale tape); **control-parameter values are
  deliberately absent**, so a control-param sweep (continuation, an orbit diagram
  over a `PoincareMap`, a Lyapunov sweep) reuses one cached tape instead of
  re-lowering a byte-identical one per value — the win is largest for the
  high-dim method-of-lines fields (a Gray–Scott lowers in seconds). The cache is
  a bounded LRU (`_TAPE_CACHE_MAXSIZE = 256`) and thread-safe; correctness rests
  on the `Tape`/`LoweredSDE` being immutable frozen dataclasses (`to_arrays`
  returns fresh FFI copies). Surface: `clear_tape_cache()` (empty + reset
  counters), `tape_cache_stats()` → `{"hits","misses","size","maxsize"}`, and the
  `TSDYNAMICS_NO_TAPE_CACHE` env var (truthy ⇒ always re-lower — the bypass that
  proves WITH-cache == WITHOUT-cache).
- **Compiled-evaluator (JIT) cache — the Rust twin, one layer down (v6 WP3-perf,
  `crates/tsdyn-jit/src/cache.rs`):** the tape cache above stops the *lowering*
  being repeated; this stops the *Cranelift compile* being repeated. Every FFI
  entry point builds its evaluator through `bridge/marshal.rs::build_evaluator`,
  which used to call `JitEvaluator::new` afresh on every call — so a
  `backend="jit"` run paid the whole compile before its first step (~0.13 ms for
  Lorenz, ~296 ms for a Gray–Scott field, making `jit` *slower* than `interp`
  below ~1000 Lorenz steps and burning ~29 s on a 100-value Gray–Scott sweep).
  `tsdyn_jit::cached_evaluator` memoises the compiled evaluator and hands out an
  `Arc` (wrapped back into the engine's `&dyn Evaluator` seam by
  `SharedJitEvaluator`). The key is the **whole `Tape`**: a hash selects a
  candidate and full `Tape` equality accepts it, so a changed immediate / opcode
  / `with_jacobian` flag is always a miss and a hash collision costs a compile,
  never a stale hit. Bounded LRU (`CACHE_MAXSIZE = 64` — smaller than the tape
  cache's 256 because an entry holds executable pages), thread-safe, compiles
  outside the lock. Surface: `tsdynamics.engine.run.jit_cache_stats()` /
  `clear_jit_cache()` (→ the `_rust` functions of the same names) and the
  `TSDYNAMICS_NO_JIT_CACHE` env var (truthy ⇒ always re-compile — the bypass that
  proves WITH-cache == WITHOUT-cache, `tests/test_jit_cache.py`). Answer-preserving:
  the same tape compiles to the same code, so `interp == jit` stays bit-for-bit.
  Measured: Lorenz 10-step 0.237 → 0.064 ms, Gray–Scott 10-step 306 → 4.2 ms
  (73×), a 20-value Gray–Scott sweep 6.23 → 0.10 s (63×); `jit` is now faster
  than `interp` at *every* run length (the old ~1000-step crossover is gone).
- The docs figure cache is unrelated and still exists: `.cache/docs-figures`,
  keyed by class source hash (CI persists it via actions/cache).

### Which backend is the default (and when to revisit)

`_default_backend` is **`"jit"`** on all four concrete families (v6). It was
`"interp"` for the whole v3–v5 line, and the flip was **measurement-driven**, not
a preference — record the numbers here so a future reader can re-litigate it
without re-deriving them.

*Why it used to be `interp`:* the Cranelift JIT recompiled the whole tape on
**every FFI call**, so `jit` was slower than `interp` below ~1000 Lorenz steps.
The v6 compiled-evaluator cache (above) removed that per-call compile, which
invalidated the original reason.

*The re-measurement* (all 136 catalogue ODE systems **as the catalogue stood
then** — it is 142 today; the conclusion is unaffected, but do not read "136" as
a current count — warm process, JIT **and** tape caches emptied per system, so
the "first call" column is a genuine cold cost):

| quantity | median | p90 | max |
|---|---|---|---|
| `jit` first-call cost over `interp` | **+0.65 ms** | +4.57 ms | +639 ms (GrayScott, dim 4608) |

| quantity | median | p10 | min |
|---|---|---|---|
| steady-state speedup (T=5, dt=0.01) | **1.55×** | 1.13× | 0.75× |

That sweep flagged `jit` as **slower on 3 of 136 systems** — `PehlivanWei` 0.99×,
`SprottB` 0.75×, `SprottF` 0.95×. **That result did not survive a controlled
re-measurement.** Re-timed with a *pinned* IC, interleaved A/B and min-of-25, all
three are **faster** on `jit`: 1.55× / 1.19× / 1.19×. The cause is a measurement
trap worth remembering: **those three declare no `default_ic`, so
`resolve_ic(None)` draws a fresh random start on every call** — different ICs take
different numbers of adaptive steps, so an un-pinned A/B is comparing different
amounts of work, not two backends. (The same trap bit the bit-identity check:
`interp` vs `interp` also "differs" on a system with no `default_ic`.) **Always
pin `ic=` when timing or diffing.**

So there is **no per-system `_default_backend = "interp"` override**, and none is
warranted: no catalogue system is known to be genuinely slower on the JIT, and a
per-system override would be a maintenance liability re-validated on every
machine and kernel change. Users who want the interpreter pass `backend="interp"`.

*Memory:* the compiled-evaluator cache is a bounded 64-entry LRU. Measured over
two passes across 97 distinct catalogue ODEs: RSS 59 → 104 MB, cache saturating at
exactly 64/64, and the **second pass added +0.2 MB** — it plateaus, it does not
leak. Note the corollary: a session cycling round-robin through >64 distinct
systems is the LRU worst case and re-compiles on nearly every visit (measured 22
hits / 172 misses over 2×97 systems). Bounded, but not free.

*When to revisit:* if the p90 first-call cost grows materially (a Cranelift
regression, or a jump in the number of catalogue systems with huge tapes), or if a
system turns up that is genuinely slower on `jit` under a *pinned-IC* A/B. The
measurement harness is the one described above — clear both caches per system, pin
the IC, and time the first call plus a T=5/dt=0.01 steady-state run on each
backend.

### Engine performance benches & the regression gate (v6 WP3-perf)

The engine's optimisations are all **answer-preserving**, so the correctness
suite cannot see them — a refactor could undo any one with `cargo test` green.
Two layers now cover them:

- **Deterministic counting tests** (the actual gate, cannot flake):
  `explicit::rk45::tests::fsal_reuse_saves_one_rhs_eval_per_continued_step` pins
  the FSAL stage reuse at the RHS-evaluation count (7 stages on the first step, 6
  on each continued one) and its sibling proves a re-seated state invalidates the
  cache; `implicit::sdirk2::tests::frozen_jacobian_is_reused_across_substages_sharing_a_shift`
  pins the frozen-Jacobian/LU reuse at **two** factorizations per step (one per
  distinct substage shift) where the always-re-form path would do six; the
  interpreter's dead-register elimination was already pinned structurally by
  `interp::tests::rhs_live_mask_is_the_shared_reachable_from_pass`.
- **Criterion benches** (`crates/{tsdyn-vm,tsdyn-solvers,tsdyn-engine,tsdyn-jit}/benches/`,
  run with `cd crates && cargo bench --workspace`) quantify what those savings are
  worth, plus a full `integrate_grid` / `iterate_dense` and the JIT
  compile-vs-cache-hit ratio. `benchmarks/check_engine_bench.py` compares a run
  with the ceilings in `benchmarks/engine_bench_baseline.json` and
  `.github/workflows/perf-engine.yml` enforces it. The ceilings are **10× the
  reference**, so the gate catches order-of-magnitude regressions (a cache that
  stopped caching, an allocation in a hot loop) and *nothing finer* — deliberately,
  because criterion wall-clock on a shared runner varies 2–3× and a flaky blocking
  gate is worse than none. Regenerate with
  `python benchmarks/check_engine_bench.py --update crates/target/criterion`.

---

## Common pitfalls

| Situation | What happens / what to do |
|---|---|
| `_equations` uses NumPy or `math` | The engine tape can't lower it. Use `symengine.sin`/`cos`/... |
| `_equations` writes `y[0]` **or** `x, y, z = u` | The ODE/SDE state is an **accessor**, not a vector: read it by CALLING it (`u(0)`, `u(1)`, …), which is also what makes a DDE's `u(0, t - tau)` expressible. A `DiscreteMap._step` **does** take a plain vector — that is the one place the families differ, and it is why a map author writes the unpack in an ODE. Both spellings are diagnosed by name, with the corrected line echoed back (`_subscripted_accessor_hint`). |
| Variable-dim system without `_structural_params` | Lowering-time `range(N)` fails. Add `_structural_params = frozenset({"N"})`. |
| Map params order ≠ `_step` signature order | **Raises `TypeError` at import**. |
| DDE with constant past at a fixed point | Lyapunov exponents ≈ 0. Provide a non-equilibrium `history`. |
| Tight tolerances on DDE | `rtol=atol=1e-3` is the DDE default and the right start — **not** because tightening stalls the solver (measured: all 6 built-in DDEs complete at `1e-12`/`1e-15`, T=500) but because the method of steps lands on every sample, so `dt` bounds the step and the tolerance is inert (5 of 6 are bit-identical from `1e-3` to `1e-9`). |
| Adding a new `rtol=`/`atol=` default | Don't write a literal — name a constant in `utils/tolerances.py`. A gate (`test_polish_standards.py::test_no_bare_tolerance_literal_in_the_library`) fails on a bare literal in any signature, call keyword or `self._rtol =` assignment. |
| "My results got less accurate in v6" | `dt` is now **sampling only** — it no longer secretly bounds the internal step (see "Dense output and `max_step`"). The default `rtol`/`atol` tightened to `1e-9`/`1e-12` to compensate, so a plain `.integrate()` is *more* accurate than pre-v6, not less. If you pinned `rtol=1e-6` explicitly you kept the old accuracy on a coarser step — tighten it; or pass `max_step=dt` to reproduce the old step regime; or set `TSDYNAMICS_NO_DENSE_OUTPUT=1` to reproduce pre-v6 numbers exactly. |
| An adaptive kernel strides over a narrow feature | Pass `max_step=`. (A step *size* — `max_steps` is a step *count*.) |
| `set_state` on a DDE | Raises by design — use `reinit(u)`. |
| Stiff ODE: which method? | `"bdf"` is the **variable-order (1–5) BDF** and the right default for stiff ODEs (far faster than the fixed-order `rosenbrock`/`trbdf2`, which stay selectable). `run.integrate` auto-builds the Jacobian-carrying tape for the implicit kernels, so `integrate(method="bdf")` "just works". The legacy SciPy name `"LSODA"` is no longer a method — declare `_default_method = "bdf"`. Pass `method="auto"` to let `solvers.recommend` probe stiffness and pick `bdf`/`rk45` — a one-point heuristic, so prefer `_default_method` for a system known to be stiff. |
| Param change ignored by a live stepper | `reinit()` after parameter changes (or use `with_params`). |
| Orbit diagram over a DDE wrapper | Re-lowers the tape per parameter value — slow by design, document it. |
| New DDE fails `test_dde_histories_complete` | Add its history to `tests/_sampling.py`. |

---

## Quick reference

```python
import numpy as np
import tsdynamics as ts

# ODE — `run` is the one canonical verb on every family (`integrate` is its alias)
lor = ts.systems.Lorenz()
traj = lor.run(final_time=100.0, dt=0.01, transient=10.0)
traj["x"]                                   # named component
exps = lor.lyapunov_spectrum(final_time=300.0)   # → LyapunovSpectrum [0.91, ~0, -14.57]
exps.kaplan_yorke                           # → ~2.06   (also ts.kaplan_yorke_dimension(exps))

# Backends: "jit" (Cranelift, default) / "interp" (SSA interpreter, bit-identical)
#           / "reference" (pure-Python oracle — the cross-check, not for production)
traj = lor.run(final_time=100.0, dt=0.01, backend="interp")

# dt is OUTPUT SAMPLING ONLY; rtol/atol set accuracy (default 1e-9/1e-12 since
# v6 — see "Solver tolerances"), max_step bounds the step
traj = lor.run(final_time=100.0, dt=0.001, rtol=1e-10, atol=1e-13)
traj = lor.run(final_time=100.0, dt=0.01, max_step=0.01)   # bound the step

# Protocol stepping
lor.reinit([1.0, 1.0, 1.0])
u = lor.step(0.01)

# Derived systems: a VERB on the system, same vocabulary as the class
ros = ts.systems.Rossler()
pmap = ros.poincare("y", 0.0, direction="up")   # = ts.derived.PoincareMap(ros, ...)
section = pmap.trajectory(500)                  # → PoincareSection
sec = ts.poincare_section(ros, plane=("y", 0.0, "up"), crossings=500, seed=0)
od = ts.orbit_diagram(pmap, "c", np.linspace(2, 6, 50), points_per_value=100)
tang = lor.tangent(k=2)                         # the Lyapunov engine, steppable
finals = lor.ensemble(np.random.rand(100, 3), final_time=10.0)   # → (100, 3)

# Event detection / arbitrary stopping (scipy-shaped events=)
sol = lor.run(final_time=100, dt=0.01, events=[("z", 27.0, "up")])
sol.meta["t_events"][0]                       # times z=27 was crossed upward
stop = lambda y, t: y(0)**2 + y(1)**2 + y(2)**2 - 50.0**2  # leave a ball → stop
stop.terminal = True
lor.run(final_time=1e3, events=[stop])                    # truncates at the crossing

# Maps — the horizon word is `n` (`steps` is accepted on run/iterate)
h = ts.systems.Henon()
h.run(5000, transient=500)
ts.fixed_points(h)                          # analytic saddles
ts.fixed_points(ts.systems.VanDerPol(), region=[(-3, 3), (-3, 3)])   # plain bounds
ts.max_lyapunov(h, ic=[0.1, 0.1])           # ≈ 0.42

# Regions: one (lo, hi[, n]) pair PER STATE COMPONENT, everywhere. No type to build.
ts.basins_of_attraction(h, [(-2, 2, 60), (-2, 2, 60)])

# Plotting: one front door, style + labels at the call site, spec in / spec out
ts.plot(traj, color="crimson", linewidth=2, title="Lorenz", theme="dark").save("l.png")
ts.plot(np.sin(np.linspace(0, 40, 2000)))              # bare arrays plot too
ts.plot(ros, ("flow_speed", {"log": True}), "streamlines")   # ("name", {opts}) pairs
ts.viz.list_transforms(); ts.viz.compatibility()       # what can this draw?

# DDE (integrate first, then Lyapunov from the end state)
mg = ts.systems.MackeyGlass()
traj = mg.run(final_time=500.0, dt=0.5, history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)])
exps = mg.lyapunov_spectrum(k=1, dt=0.5, ic=traj.y[-1])

# Registry
from tsdynamics import registry
registry.families()                         # {'ode': 142, 'dde': 6, 'map': 26, 'sde': 3}
```
