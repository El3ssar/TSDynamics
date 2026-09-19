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
├── registry.py               # system registry (SystemEntry + all_systems/…) + generic analyses/renderers registries (solvers live in tsdynamics._solvers)
├── families/                 # base classes + the System protocol (was base/)
│   ├── base.py               # SystemBase + MetaStore + Absent + the v6 absent-name errors
│   ├── _params.py            # ParamSet — a **dict subclass** since v6 (fixed keys, as_tuple, param_hash)
│   ├── _info.py              # SystemInfo (system.info) + the `variables` descriptor + family_of
│   ├── _derive.py            # DeriveMixin: the two derivation verbs, poincare (absorbs the strobe) + ensemble
│   ├── _kwargs.py            # the closed-run-signature guard + the per-name WHY table
│   ├── _hidden.py            # the @hide decorator: the families/derived half of the §11 visibility ruling (a `__dir__` edit, NEVER a rename — every hidden name stays bound, importable, callable)
│   ├── protocol.py           # the System runtime Protocol (run/step/state/time/reinit + dim/family)
│   ├── continuous.py         # ContinuousSystem (engine run + jacobian autogen)
│   ├── delay.py              # DelaySystem (engine method-of-steps, forward-only)
│   ├── discrete.py           # DiscreteMap (engine iterate + signature validation)
│   ├── stochastic.py         # StochasticSystem — diagonal-Itô SDEs (_drift+_diffusion; EM/Milstein)
│   ├── _plottable.py         # SystemPlottable plotting seam (system.__plot_spec__/plot, splits plot vs run kwargs)
│   └── wrapped.py            # WrappedSystem (canonical home; adapt an external stepper — re-exported via derived)
├── _engine/                   # Rust-facing engine layer; tsdynamics._rust is the sole backend
│   ├── symbols.py            # engine-native symbolic frontend: state_time_symbols() → (Function("y"), Symbol("t"))
│   ├── compile.py            # symbolic dynamics → IR Tape (all families) + reference evaluator + bounded lowered-tape CACHE (lower_*_cached / clear_tape_cache / tape_cache_stats)
│   ├── problem.py            # per-family Problem builders bundling a tape + runtime context
│   ├── run.py                # orchestration: integrate/ensemble + eval_rhs/eval_jac + backend resolve (resolve_backend/_engine/BACKENDS/EngineNotAvailableError) + problem-coercion/naming/provenance; RE-EXPORTS every name the split-out submodules own so tsdynamics._engine.run.<X> keeps working for EVERY X
│   ├── run_methods.py        # method= resolution + auto-stiffness split out of run.py: _resolve_method_for/_recommend_method/_resolve_method_and_prepare (shared by integrate+ensemble)
│   ├── _families.py          # per-family runners split out of run.py: _run_continuous/_step_continuous/_run_dde/_sample_past/_run_map + the low-level _engine_* FFI shims (_engine_integrate_dense/_engine_ensemble_final/_engine_map_ensemble_final)
│   ├── stepper.py            # resumable OdeStepper API split out of run.py: make_ode_stepper/step_advance/step_advance_to_event (WS-STEPPER)
│   ├── sde_run.py            # SDE dense/ensemble seam split out of run.py: sde_integrate_dense/sde_ensemble_final (the seed/step-carrying path run.integrate refuses)
│   ├── events.py             # event subsystem split out of run.py: crossings/Event/EventSolution/integrate_events (WS-CROSSKERNEL + WS-EVENTSAPI)
│   └── reference.py          # the pure-Python reference oracle (backend="reference") split out of run.py: SciPy-stepped ODE + reference-evaluated map
├── _solvers/                  # F2 registry mechanism + C-SOLV in-tree specs (explicit/implicit/stochastic) + method= resolution/aliases + auto-stiffness (select.py)
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
│   ├── results.py            # ts.analysis.results — the 32 result CLASSES behind one dot (namespace only; each still lives in its own subpackage)
│   ├── _result_base.py       # AnalysisResult (frozen-dataclass base: meta / the repr that IS the answer / to_dict / to_frame / plot seam)
│   ├── _result_scalar.py     # ScalarResult / CountResult (+ _NumericOps, now on NDArrayOperatorsMixin)
│   ├── _result_array.py      # ArrayResult (also NDArrayOperatorsMixin — the full operator set)
│   ├── _result_collection.py # CollectionResult (a complete sequence: positional [], by_id, __array__)
│   ├── _result_scaling.py    # ScalingResult (mixes _NumericOps like ScalarResult: full float drop-in — comparisons/arithmetic + value-based ==/hash; subclasses re-apply @dataclass(frozen=True, eq=False)); + n_fit / r_squared fit diagnostics
│   ├── _result_viz.py        # the .plot accessor seam (_PlotAccessor / VisualizationNotInstalled)
│   ├── _result_json.py       # to_dict / repr helpers (_jsonify / _is_frame_scalar / _fmt) + the repr formatters (_sig / _state / _vector / _pct)
│   ├── orbits/               # A-ORBIT: orbit_diagram + OrbitDiagram (+ periods/bifurcation_points; orbit_diagram.py); poincare_section (poincare.py); return_map + ReturnMap (first-return/next-amplitude map; return_map.py); self-registers into registry.analyses
│   ├── lyapunov/             # A-LYAP: lyapunov_spectrum (one door: tangent frame, or two-trajectory when there is no Jacobian), kaplan_yorke_dimension + lyapunov_from_data (Kantz/Rosenstein, from_data.py); self-registers into registry.analyses
│   ├── fixedpoints/          # A-FP: fixed_points/FixedPoint (maps+flow equilibria, Newton/SD/DL + rigorous Krawczyk method="interval" in _interval.py — fixed.py), periodic_orbits/PeriodicOrbit (map orbits AND a flow's limit cycle, one verb → one OrbitSet) + estimate_period (periodic.py), shared primitives (_common.py); self-registers
│   ├── dimensions/           # A-DIM: correlation/generalized-Rényi/fixed-mass fractal dims + scaling-region fit
│   ├── chaos/               # A-CHAOS: GALI_k (Skokos) + 0–1 test (Gottwald–Melbourne) + expansion entropy (Hunt–Ott); maps via _jacobian, flows via self-contained RK4 variational core (no engine/compile)
│   ├── recurrence/          # A-RQA: recurrence_matrix (fixed ε / target rate, sparse cKDTree) + rqa (DET/LAM/L_max/ENTR/TT) + windowed_rqa; self-registers into registry.analyses
│   ├── basins/              # A-BASIN: attractors/basins (recurrence-FSM AttractorMapper) + basin_fractions (basin stability) + basin_entropy/uncertainty_exponent/wada_property (boundary structure) + continuation/tipping_points + resilience; cell tessellation in _common.py; self-registers into registry.analyses
│   ├── embedding/           # owned by A-EMBED
│   └── sampling/            # sagitta tools: estimate_dt_from_sagitta (output-dt selector) + sagitta_profile (per-point bow, the color_by="sagitta" field). NOT registered into registry.analyses (a sampling tool, not a quantifier); SagittaDt result is hidden (not exported)
├── viz/                      # PlotSpec IR seam + renderers (mpl/plotly/json/threejs) + compose.py (ts.viz.plot)
│   ├── export.py             # the JSON envelope: to_json/from_json + to_dict_envelope/from_dict_envelope + SCHEMA_VERSION (all off ts.viz.spec's listing; the round trip a user drives is Plot.to_dict/to_json + ts.viz.load)
│   └── _visibility.py        # the viz half of the §11 visibility ruling: the curated __dir__ tables
├── systems/
│   ├── _declared_params.py   # merge_declared_params: the ONE parameter merge every VARIABLE-DIMENSION system's custom __init__ performs (v6.0.1)
│   ├── continuous/           # 9 ODE category modules (+ spatial_fields.py 2-D PDEs) + delayed_systems.py (DDEs!)
│   └── discrete/             # 5 map category modules
└── _utils/                    # the LEAF package: values both families/ and engine/ must agree on
    ├── escape.py             # Unbounded + detect_unbounded (the runaway-orbit verdict Trajectory.unbounded reports)
    ├── grids.py              # make_output_grid (the single hoisted output-grid builder) + validate_max_step (the one max_step guard, shared by the reference integrator and the event seam); sagitta tooling moved to analysis/sampling/
    ├── lookup.py             # is_hashable — the guard in front of every `value in <set/dict>` over user input; see "A closed vocabulary is checked with `in`"
    ├── plot_namespace.py     # subject.plot as a callable NAMESPACE (bound on Trajectory + SystemBase) + the retired-`to_plot_spec` message
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

Built-in system classes live under `tsdynamics.systems` — the canonical path is
`tsdynamics.systems.<Name>` (e.g. `tsdynamics.systems.Lorenz`, flat across
`continuous`/`discrete`). **`ts.systems` is a registry like the others since v6:
`names()` / `find()` / `get()` sit beside the 177 classes** (`__all__` is
**180**), so the one namespace with 177 members is the one you can search —
`ts.systems.find("delay")`, `ts.systems.find(family="dde")`,
`ts.systems.get("Lorenz")()`. The two category subpackages stay importable and
are off the listing; a wrong guess is an `AttributeError` naming the nearest
catalogue entries and `find`. `systems/__init__.py` flat-re-exports every catalogue
class automatically (driven by each category module's `__all__`), so a new system
needs no manual edit there. **Since v6 `ts.Lorenz` no longer resolves** — see C5
and the redirect ladder below; the `MovedInV6` it raises names
`ts.systems.Lorenz()`.

**`tsdynamics.__all__` is exactly ELEVEN names** (measured — re-measure, never
nudge, if you change it), sorted, one per line:

```
ContinuousSystem       StochasticSystem       plot
DelaySystem            Trajectory             systems
DiscreteMap            WrappedSystem          viz
                       __version__            analysis
```

Five classes you subclass · one type you receive and annotate · one plotting verb
· three registries · `__version__`.  The six typed exceptions were promoted to
the top level and then **demoted again** — see "The six typed exceptions left
`__all__`" below: the hierarchy is additive, so `except ValueError` already
catches them and nothing a user writes requires this library's spelling.  They
live at `ts.errors.<Name>`.

`errors` (the module) **left `__all__` and stays bound forever**: the Rust bridge
does `py.import("tsdynamics.errors")` at `crates/tsdyn-core/src/lib.rs` when it
builds a typed exception at the FFI boundary, so the module path is an **ABI**,
not a curation choice. Gates: `test_errors_the_module_left_the_listing_but_is_bound_forever`
and `test_the_rust_bridge_still_names_the_module_it_imports`.

The membership rule is written above `__all__` in `src/tsdynamics/__init__.py`
and is the thing to argue against before adding a name:

> A name earns a top-level slot only if a user TYPES it in ordinary work — and no
> user should ever have to construct a library type to make a call, so a name
> that is exported *because a signature demands it* is evidence of a **signature
> bug**, not of a needed export.

Four corollaries, applied mechanically:

- **C1 the toll rule** — plain Python (tuples/lists/strings/numbers/arrays)
  reaches every front door. Fix the signature, then demote the name; never the
  reverse. (`ts.analysis.basins(vdp, [(-3, 3), (-3, 3)])` already worked,
  which is why `Box`/`Ball`/`Grid` never needed exporting.)
- **C2 received ≠ typed** — a type you get *back* is not one you type. Exactly
  one exception: `Trajectory`.
- **C3 one concept, one spelling** — two grammars for the same argument is the
  silent-wrong-answer defect, not flexibility.
- **C4 a name must resolve to what it advertises** — no `__all__` entry may be
  shadowed by a submodule of the same name. Gate:
  `tests/test_namespace_curation.py::test_no_all_entry_is_shadowed`, swept over
  every public package. (`tsdynamics.viz: {"spec"}` and
  `tsdynamics.analysis: {"results"}` are *declared* submodule exports: a user who
  types `ts.viz.spec` means the module, and neither name is also a verb.)
- **C5 `dir()` is the truth, for API names** — `dir(M) == sorted(M.__all__)` for
  every public package, every listed name resolves, and **no API name resolves
  unless it is listed**. This is the v6 change: demotion used to mean *drop the
  name from `__all__` and leave it bound*, which bought a tidy `dir()` and nothing
  else — `ts.<TAB>` said seventeen while `getattr` answered to ~260, and the two
  hand-written re-export blocks producing those bindings were already measurably
  stale (`ts.LyapunovSpectrum` resolved; `ts.Embedding` did not). Submodules bound
  by an ordinary `import` are the one exemption and are enumerated in
  `_INTERNAL_SUBMODULES`. Gates: `test_public_package_dir_mirrors_all` and
  `test_no_api_name_resolves_on_the_top_level` (a per-name sweep over the whole
  demoted population, built live from the public homes).

Everything else is **demoted, never removed** — it lives at exactly one address,
one dot down, and the exception a wrong guess raises prints that address:

- The 177 built-in systems (142 ODE + 6 DDE + 26 maps + 3 SDE) via
  `tsdynamics.systems`; `ts.Lorenz` resolves lazily through `__getattr__`.
- Every analysis function and result class — `orbit_diagram` (the ONE spelling
  since v6: the `bifurcation_diagram` alias was the same object under a second
  name, and a shared implementation can name only one of its spellings in a
  message, so half of all callers were answered about a function they had never
  typed — `ts.bifurcation_diagram` / `ts.analysis.bifurcation_diagram` now
  redirect; `PlotKind.BIFURCATION_DIAGRAM`, the *picture*, is a different concept
  and is untouched), `OrbitDiagram`, `poincare_section`,
  `return_map`/`ReturnMap`, `lyapunov_spectrum`,
  `kaplan_yorke_dimension`, `lyapunov_from_data`/`LyapunovFromData`,
  `fixed_points`/`FixedPoint`, `periodic_orbits`/`PeriodicOrbit` (the flow's limit
  cycle was absorbed here in v6 — one verb, one return type: an `OrbitSet`),
  `estimate_period`; A-DIM `correlation_dimension`, `correlation_sum`,
  `generalized_dimension`, `box_counting_dimension`, `information_dimension`,
  `dimension_spectrum`, `fixed_mass_dimension`, `DimensionResult`; A-CHAOS
  `gali`/`GALIResult`, `zero_one_test`, `expansion_entropy`/
  `ExpansionEntropyResult`; A-RQA `recurrence_matrix`/`RecurrenceMatrix`, `rqa`/
  `RQAResult`, `windowed_rqa`/`WindowedRQA`; A-BASIN `attractors`,
  `basins`, `basin_fractions`, `basin_entropy`,
  `uncertainty_exponent`, `wada_property`, `continuation`, `tipping_points`,
  `resilience`, `Attractor`, `AttractorSet`, `BasinsResult`, `BasinFractions`,
  `BasinEntropy`, `UncertaintyExponent`, `WadaResult`, `ContinuationResult`.
  Canonical home `ts.analysis.*` — since v6 the **only** home (see "The analysis
  door" below): `ts.analysis.__dir__` is **flat** and mirrors its **52-name**
  `__all__`, which is *generated from `registry.analyses`*, never hand-written.
- The **derived wrappers** `PoincareMap` / `StroboscopicMap` / `TangentSystem` /
  `Ensemble` / `ProjectedSystem` → `ts.derived.*`. **Since v6 exactly TWO of them
  have a verb on the system** (ruling A3):
  `sys.poincare("y", 0.0)` / `sys.poincare(("y", 0.0, "up"))` / `sys.poincare(period=T)`
  — `poincare` **absorbed `stroboscope`**, because a plane is an affine surface
  `g(u) = n·u - c` and a period samples the phase *circle*: two disjoint
  arguments, one verb, two return types (both at once raises naming the choice;
  neither on a forced flow infers the drive period, on an autonomous one chooses
  a plane) — and `sys.ensemble(states)` → an `Ensemble` *system* whose `.run(...)`
  returns a `TrajectoryBatch` carrying `.final` **and `.diverged`** (the `(n,)`
  boolean mask, new in round 8; the repr prints `⚠ 5 of 12 diverged` when there
  is something to print and stays silent when there is not). A batch used to
  return NaN rows with no count and no attribute of any kind, so
  `np.mean(batch.final, axis=0)` was `nan` and `np.nanmean` a survivor-biased
  mean over an unflagged subsample — on precisely the call where a user
  aggregates and cannot inspect each member. "Diverged" is the **one** escape
  scale (`utils.escape.ESCAPE_SCALE`) the trajectory repr and the Lyapunov
  verdict already use, read off the end state.
  `stroboscope`, `project`, `tangent` and `copies` are **gone from the object**
  (census: no user callers); each `AttributeError` names the replacement —
  `ts.derived.ProjectedSystem(sys, [0, 2])` / `traj["x", "z"]`,
  `ts.derived.TangentSystem(sys, k=2)`, `sys.ensemble(states)`. (Both of those
  lines are *run* by a gate — the old `ProjectedSystem(sys, 0, 2)` spelling
  raised `TypeError`, which is the one thing a remedy line may never do.)
- **State-space geometry** (`data`): `Box`, `Ball`, `Grid`, `Region`, `region`,
  `as_region`, `sampler`, `grid_points`, `set_distance` → `ts.data.*`. Every
  `region=` door in the library reads **one `(lo, hi[, n])` pair per state
  component** (`data/sampling.py::as_region`, the ONE reading); a `Box`/`Ball`/
  `Grid` is accepted everywhere and required nowhere, which is exactly why the
  types are demoted (C1). Verified by running all eleven doors with plain tuples:
  `fixed_points(region=)` (both `newton` and `interval`), `basins`, `attractors`,
  `basin_fractions`, `expansion_entropy(region=)`, `periodic_orbits(region=)`,
  `data.sampler`, `data.grid_points`, `data.as_region`, `data.set_distance`. Gate:
  `test_no_region_argument_requires_a_library_type`.
- `T` (the per-transform option carrier) → `ts.viz.T`. A plain
  `("name", {options})` pair is a transform call now, so nothing requires it.
- Machinery submodules `data` / `derived` / `engine` / `errors` / `families` /
  `plugins` / `registry` / `solvers` / `utils` (`_INTERNAL_SUBMODULES`) — bound
  eagerly, off `__all__`.
  The justification is C1: *nothing in them is required to make a call.* (The
  old comment said they were redundant because their contents were already on the
  top level — the exact reasoning this curation exists to correct.)

**RENAMED** (`_redirects.RENAMED_IN_V6`, one row each, sorted):
`bifurcation_diagram` → `ts.analysis.orbit_diagram` · `basins_of_attraction` →
`ts.analysis.basins` · `find_attractors` → `ts.analysis.attractors` ·
`periodic_orbit` → `ts.analysis.periodic_orbits` · `PlotSpec` → `ts.viz.Plot`.
Note the reversal: in v5 `ts.basins` was deleted *because* it was a function
colliding with the `analysis.basins` subpackage; in v6 the subpackages are
unbound from `ts.analysis` (contract §5.4), so `basins` is free and is now the
canonical spelling.

**The exception IS the migration guide.** Curation hides ~260 reachable names
from autocomplete, so a wrong guess at `ts.<name>` is the only feedback a user
gets. The ordered ladder in `tsdynamics/__init__.py::__getattr__`:

| case | hit | raises |
|---|---|---|
| 0 | a `_`-prefixed or dunder probe | bare `AttributeError`, imports nothing |
| 0 | `viz` / `plot` | resolves (lazily) and caches |
| 1 | exact in `_RENAMED_IN_V6` | **`MovedInV6`** |
| 2 | exact in `_REMOVED_IN_V6` | **`MovedInV6`** |
| 3 | exact in a public home's `__all__` (`_home_of`) | **`MovedInV6`** |
| 4 | a near miss (ranked) | `AttributeError` |
| 5 | nothing matches | `AttributeError` |

**`errors.MovedInV6` is an `ImportError`, not an `AttributeError`, and that is
measured, not stylistic.** On CPython 3.14.2 a module `__getattr__` that raises
anything matching `AttributeError` has its message **discarded** by
`from tsdynamics import X` and replaced with the generic "cannot import name";
raise `ImportError` and the text propagates verbatim. The corpus uses the
from-import spelling 111 times. A class inheriting *both* is impossible
(`TypeError: multiple bases have instance lay-out conflict`), so the import
spelling wins. Cases 4–5 **must** stay `AttributeError` or `hasattr(ts, …)`
breaks for every name in the universe; the cost is confined to the enumerated
dead names, where raising is the point. There are consequently **no tombstone
objects and no tombstone modules**.

The redirect tables are pure data in **`src/tsdynamics/_redirects.py`** — sorted
by key, **append-only**, one row per moved name, so several builders appending
rows merge cleanly while `__init__.py` owns the machinery. *If your change
removes or renames a public name, its row belongs there, in the same commit.*

Case 3 before case 4 is load-bearing: an exact hit in a public `__all__` is a
certainty and must outrank every guess. Without it a merely-demoted name got
answered with nonsense — `ts.region` (real, at `ts.data.region`) suggested
`ts.systems.Oregonator()`.

The **suggestion scorer** needs *two* floors, not one, and the second is the
interesting one: similarity ≥ 0.6 **and** coverage ≥ 0.8, where coverage is how
much of what the user *typed* the candidate accounts for. A plain 0.6 floor
offers `ts.systems.Tent()` for `ts.Event` (ratio 0.667, coverage 0.600); raising
the floor to reject that also rejects `lyapunov` → `lyapunov_spectrum` (ratio
0.640, coverage 1.000), the most useful suggestion in the library. At equal score
a **verb outranks a type** (`fixed_points` over `FixedPoint`), because `difflib`
breaks ties by string order, which encodes nothing.

Dunder and `_`-prefixed probes short-circuit, so protocol lookups import nothing
and `ts.viz` stays lazy — and `from tsdynamics import _rust` keeps working, since
the import machinery asks `hasattr` first and only falls back to importing the
submodule if that answered `False`.


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

**The machinery packages are PRIVATE (v6.1).** `engine`, `solvers` and `utils` are
`_engine`, `_solvers` and `_utils` on disk, so `ts.engine` raises `AttributeError`
rather than resolving to 66 names of FFI shims and tape internals. The rename was
the only mechanism that works: Python auto-binds any imported submodule onto its
parent, so a package the library imports internally is reachable as `ts.<name>`
however carefully `__init__.py` avoids binding it — dropping the import buys
nothing. Measured before the change: `engine` 10 modules / 66 public names,
`solvers` 5 / 33, `utils` 5 / 26, against 2 / 14 / 3 mentions in user-facing docs
and 122 / 20 / 29 in tests. That ratio *is* the definition of machinery.
`families`, `data`, `derived`, `registry`, `plugins` and `errors` stay reachable:
each has real user-facing traffic, and `errors` is additionally an ABI.

**The six typed exceptions left `__all__`** for a measured reason, not a taste:
the hierarchy is purely additive — `InvalidParameterError` IS a `ValueError`,
`ConvergenceError` IS a `RuntimeError`, `InvalidInputError` IS a `TypeError` — so
`except ValueError` already catches a bad `dt` and nothing a user writes requires
this library's spelling. They are a *refinement* (telling one failure from
another), and a refinement lives at `ts.errors.<Name>`. `ts.__all__` is
**ELEVEN**. Gates: `test_the_typed_errors_live_at_their_own_address_because_you_never_need_them`
asserts both halves — the additive hierarchy AND the redirect — because only
together do they make the demotion safe.

Reachable but not top-level: `SystemBase`, `ParamSet`, `MetaStore`, `System`
(protocol) via `tsdynamics.families`.
`errors` is bound eagerly and is **not** in `__all__` (the six exception classes
are exported directly — see above); `engine`, `solvers`, `data`,
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
`plugins.ALL_GROUPS` is the **six** extension doors —
`tsdynamics.systems` / `.solvers` / `.analyses` / `.renderers` /
`.plot_transforms` / `.plot_primitives` (the last added in v6). **A declared group
is a promise, and a promise nothing loads is a lie**: `SYSTEMS_GROUP` has been
advertised since F2 with no consumer, so a third-party system package declaring
against it was silently ignored. `PLOT_PRIMITIVES_GROUP` was the second such
dead door and is **live since v6.0.1** (`viz.discover_plugins` loads it via
`viz/__init__.py::_register_primitive_entry_points`) — it mattered more than it
looks, because `ts.viz.transforms.allow` is the *only* route a custom primitive
has into a shipped transform, so the dead group made the two viz extension doors
unusable together. `SYSTEMS_GROUP` is still open (tracked as a `strict` xfail).
Gate: `tests/test_registry.py::test_every_declared_plugin_group_has_a_consumer`,
which since v6.0.1 requires an actual **loader call** (`load_plugins` /
`register_entry_points`) applied to the group under any local alias — a bare
*mention* of the constant used to satisfy it, so the fix could have been faked —
and never the group string alone (`"tsdynamics.systems"` appears in every module
path under `systems/`, which is how the first dead group passed unnoticed).
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
richer `tsdynamics._solvers` registry (a `name → SolverSpec` table with
capability flags + `solvers/` directory and entry-point discovery via
`plugins.py`, stream F2). Do not re-add a `solvers` registry to `registry.py`.
**Family detection keys off the module
prefix `tsdynamics.families`** (the `_BASE_PREFIX` in `registry.py` and the guard
in `SystemBase.__init_subclass__`) — both moved from `tsdynamics.base` in the F3
reorg; keep them in lock-step if the families package ever moves again.

Every concrete `SystemBase` subclass auto-registers at class-definition time
(`SystemBase.__init_subclass__` → `registry.register_class`). `SystemEntry`
records name/cls/family/category/dim/params plus five catalogue-metadata fields:
`reference`, **`doi`**, **`field_shape`**, **`field_labels`**, `known_lyapunov`.
The middle three are new in v6 and close the **155-dropped-DOI headline bug**: the
values were on the classes the whole time and the record they were copied into had
no slot for them, so the docs tool read `None` for every one.

**Each is read under BOTH spellings, underscored first** (`_METADATA_CLASSVARS` /
`_classvar` in `registry.py`). v6 moves these ClassVars behind an underscore so
they stay off `system.<TAB>` (`system.info` absorbs them) — and a rename like that
is precisely the change that orphans its readers *silently*, because `None` is a
legal value for all five. Reading both spellings is not a shim: it is the only
formulation that cannot go quiet while the declaration side of the rename is in
flight, and it stays correct after. Gate:
`tests/test_registry.py::test_catalogue_metadata_reaches_the_registry`, which
counts arrivals with a floor (≥170 citations, ≥150 DOIs, ≥20 `known_lyapunov`)
rather than merely asserting the field exists.

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

Optional per-system metadata ClassVars. **Since v6 every one of them is
underscored** — they are facts *about* the class, printed by `system.info`, and
they were the loudest half of a `lorenz.<TAB>` that had to shrink to 19 names:
`variables` (component names → `traj["x"]`, docs labels — the one that stays
un-underscored, because an author *declares* it and `dim` now follows it),
`_reference` (literature citation shown in docs), `_doi` (the bare DOI for that
citation — e.g. `"10.1175/..."` — sourced from the GilpinLab/dysts dataset where
available, for the docs per-system page), `_known_lyapunov` (drives
`tests/test_known_values.py`; keys: `spectrum`+`atol`, or `n_positive`, plus
`params`/`ic`/`kwargs`/`source`), `_default_ic`, and — for a
**spatially-extended** system whose state vector is a flattened field —
`_field_shape: tuple[int, ...]` (the spatial grid `(Ny, Nx)` / `(N,)`, resolved
onto `traj.meta["field_shape"]` by `SystemBase.__init__`/`_provenance` for the
`kind="field"` spatial-field movie) and `_field_labels` (the names of the field
blocks packed into the state, e.g. Gray–Scott's `("u", "v")`).

**`register_class` reads the underscored spellings** (`registry.py`), and so must
anything else outside the class. Two live consumers were reading the old public
names and silently getting `None` for all 177 systems — `registry.register_class`
itself (every `SystemEntry.reference` / `.known_lyapunov` was `None`, so the docs
rendered no citation and `test_known_values.py` skipped rather than asserted) and
`analysis/planar.py::window_for`, whose pilot-orbit branch is gated on
`default_ic` and so quietly fell back to a *wrong auto window* for every model
plot. If you add a reader, read `_<name>` — or, outside the package,
`registry._classvar(cls, name)`, which knows both spellings.

**Four more readers were found in `docs/_tooling/` in v6.0.1**, every one silent
for the same reason (`None` is legal for all five fields), and together they are
the largest measured instance of this defect in the repo:

| reader | saw | truth |
|---|---|---|
| `make_bibliography.py` (`cls.reference` / `cls.doi`) | **0** / 177 | 172 refs, 155 DOIs |
| `properties.py` (`entry.cls.known_lyapunov`) | **0** / 177 | 21 |
| `plot_dt.py` (`type(sys).default_ic`) | **0** / 177 | 53 |
| `catalog.py` (`cls.field_labels`) | **0** / 177 | 1 |

So the whole References page was being generated from zero citations. Note the
right fix differs by site: `field_labels` / `doi` are **`SystemEntry` fields**
(the record already resolves both spellings), so read them off `entry`; the rest
go through `registry._classvar`. Gates (`tests/test_analysis_discovery.py::TestNoRemovedNameIsReadAsAString`):
`test_the_docs_tooling_does_not_read_a_removed_name_as_a_string` sweeps the
directory, `test_the_docs_tooling_actually_receives_the_catalogue_metadata`
*counts what arrives* — a spelling can be right and still be wired to the wrong
object — and `test_no_module_reaches_for_a_removed_accessor_by_attribute` covers
the **attribute** shape (`system.resolve_ic(...)`), which the `getattr`-string
sweep cannot see and which hid eight live call sites. `docs/_tooling/**` selects
that file (`tests/_changed_select.py::_DOCS_TOOLING_TESTS`).

---

## Base classes

### `SystemBase` (`families/base.py`)

**The v6 tab surface is exactly 20 names** (ruling A5), per family:

```
ContinuousSystem  20  copy dim ensemble family ic info jacobian jacobian_sym params plot
                      poincare reinit rhs run set_state state step time variables with_params
DiscreteMap       17  (- poincare, - jacobian_sym, - rhs)
StochasticSystem  19  (- jacobian_sym)
DelaySystem       16  (- set_state, - jacobian, - jacobian_sym, - rhs)
WrappedSystem     13  (- jacobian, - jacobian_sym, - rhs, - ic, - info, - params, - with_params)
```

It was **19** through round 10.  **`rhs(u, t=0.0)`** is the twentieth and the
only name ever *added* to the core: `jacobian` is public and is the derivative of
a function the library would hand back only as `_rhs_numeric`, and three of the
plotting layer's transforms (`vector_field` / `flow_speed` / `streamlines`) **are**
that function — so "is this picture right?" had no public answer, and a blind
tester had to finite-difference `reinit`/`step` to check one.  Two builders and
that tester independently typed the private name, which is exactly the evidence
the membership rule asks for.  On a `StochasticSystem` it is the **drift**, named
as such, matching `jacobian` (already the drift Jacobian).  The three families
with no `f(u, t)` declare an `Absent` that states why: a DDE's field needs the
whole history, a map has no vector field (`step(1)` is the analogue), a
`WrappedSystem` holds an opaque stepper.  Gate: `tests/test_families_rhs.py`.

An absent name is bound to a `base.Absent` **descriptor**, so `hasattr` is
`False`, `dir()` omits it, and reading it raises an `AttributeError` stating the
mathematical reason plus a runnable line.  `SystemBase.__getattr__` answers every
*removed* name from `_MOVED_IN_V6` / `_DELETED_ACCESSORS` the same way — the error
**is** the migration guide.

- `ParamSet` is a **`dict` subclass** (`families/_params.py`): `isinstance(p, dict)`
  and `json.dumps(p)` work, `repr` is a plain dict, and all five inherited
  mutators (`pop`/`popitem`/`clear`/`setdefault`/`update`) are overridden so the
  fixed-key contract — and `as_tuple()`, the **ordered** tape contract — survive.
- `dim` is **read-only** on an instance, and **follows `variables`**: declaring
  names is enough, declaring both in disagreement raises at class definition.
- `variables` is a lazy per-instance descriptor (`families/_info.py`): **every**
  system names every component (declared names, a field system's `u0..v0..`, a
  repeated unit's `x0 y0 z0 x1..`, else `y0..y{dim-1}`).  Read off the *class* it
  is still the declared tuple, for the nine class-level readers deferred to v6.1.
- `family` (`"ode"|"dde"|"map"|"sde"`) **replaces `is_discrete`**, which could not
  tell a delay system from a stochastic one.
- `info` is a frozen `SystemInfo` record that **absorbed** `reference`, `doi`,
  `known_lyapunov`, `field_labels` and `default_ic` off the tab surface; the
  ClassVars are now `_reference` / `_doi` / `_known_lyapunov` / `_field_labels` /
  `_default_ic`, and `__init_subclass__` migrates a class still using the public
  spelling.  `system.meta` and its `MetaStore` are gone — a *run* records its own
  provenance on `traj.meta`.
- `copy()` / `with_params()` forward `dim=` and `field_shape=`, so a
  `Sys(dim=2)`-constructed system can be re-parametrised (it used to raise
  "does not declare its state-space dimension", breaking continuation and orbit
  diagrams for exactly the systems the docs teach you to write).
- `resolve_ic` / `ic_generator` moved behind an underscore.
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
  `backend=None` to a family's `run` resolves to it, and
  `backend="auto"` resolves to the same `"jit"`. `run.integrate` also resolves the `method=` string
  through the shared `_resolve_method_for` contract (so an alias `"RK45"`/
  `"dopri5"` → `"rk45"`, a rejected v2-only name `"LSODA"`, and the auto-stiffness
  `method="auto"` all behave identically in **`integrate` and `ensemble`**) and
  rebuilds the ODE tape `with_jacobian=True` when an implicit kernel needs it.
  Lowering goes through the cached `lower_*_cached` helpers (see the tape cache
  section), so a parameter sweep reuses one tape. The output grid each
  family samples on is the one hoisted `tsdynamics._utils.grids.make_output_grid`
  (the four byte-identical `_make_t_eval` copies are gone). **SDEs are the
  exception** — they keep the dedicated `run.sde_integrate_dense` /
  `run.sde_ensemble_final` seam (`run.integrate` cannot carry the noise
  seed/step and refuses an SDE).

### `Trajectory` (`data/trajectory.py`)

Lives in `tsdynamics.data` — it is a *data* type the families produce, not a
family itself. Re-exported from `families.base` / `tsdynamics.families` and the
top level, so `from tsdynamics import Trajectory` and `from tsdynamics.data
import Trajectory` are the same object.

- **One indexing grammar**: *a name selects a column, several names select a
  sub-trajectory, a number selects a row.* `traj["x"]` is the `(T,)` array;
  `traj["x", "z"]` is a `Trajectory` that names its **own** columns (so
  `traj["x", "z"]["z"]` is `z`, which the old `sel` spelling got wrong — it wrote
  the names into `meta`, which `variables` never read); `traj[10:50]` is rows.
  The list spelling `traj[["x","z"]]` raises, naming the one that works.
  `traj.component("x")` survives for the by-index form.
- Point-set ops: `minmax()`, `standardize()`, `neighbors(q, k)` → a `Neighbors`
  record (`.state` / `.distance` / `.index`, one shape for every `k`; it used to
  hand back scipy's raw `(distances, indices)`, rank-polymorphic in `k`).
- **No topical accessors** — `traj.dims` / `traj.recurrence` / `traj.lyap` and
  `families/_accessors.py` are **gone** (ruling A2): an analysis is a free
  function whose first argument is its subject, so it is
  `ts.analysis.correlation_dimension(traj)` / `ts.analysis.rqa(traj)` /
  `ts.analysis.lyapunov_from_data(traj)`. A guess at a removed accessor is
  answered by name, and the answer names the **data-first** free function (the
  table is `data/trajectory.py::_DELETED_TRAJECTORY_ACCESSORS`) — a remedy line
  a library hands back must RESOLVE *and run for the subject that was held*.
- **`subject.plot` is a callable NAMESPACE** (`_utils/plot_namespace.py`, bound on
  both `Trajectory` and `SystemBase`): `traj.plot()` is the verb it always was,
  and `traj.plot.<TAB>` lists every transform that admits **this** subject, so
  `traj.plot.psd()` == `ts.plot(traj, "psd")`. It is the discovery route ruling
  A2 promised in exchange for taking the analyses off the object — a
  module-level registry cannot be tab-completed from the thing in your hand.
  **Since v6 round 11 each entry carries that transform's OWN signature.** It was
  a bare `functools.partial`, so `help(traj.plot.psd)` printed `ts.plot`'s generic
  composition signature `(*things, layout, rows, cols, share_x, …)` — the owner's
  complaint ("I would like to choose the parameter I want, the resolution of the
  sweep") was about knobs that were **real and invisible**: `orbit_diagram` has
  taken `param=` / `values=` / `points=` / `transient=` / `bins=` all along, and a
  *wrong* keyword was already answered well. Each namespace entry is now a real
  function carrying the transform's `__name__` / `__doc__` / `__signature__`,
  generated from its `compute`, so `help(logistic.plot.orbit_diagram)` prints
  `orbit_diagram(*, param='r', values=None, points=100, transient=500,
  components=0, primitive='points', **style)` plus what may draw it. The same
  source feeds the registry record's repr (`ts.viz.transforms.get("orbit_diagram")`
  prints its options) and the DataFrame-able `compatibility().rows()`. The printed
  **matrix** deliberately stays option-free: 39 rows x an option list is not a
  table anyone reads.
- `meta` carries provenance (system, params, solver, dt, tolerances, ic,
  version); preserved through slicing/`after()`.
- **Plotting front door — `traj.plot(...)`, and the seam under it is the dunder
  `__plot_spec__(kind=None, *, components=None, animate=False, primitive=None,
  **kind_kw)`.** `to_plot_spec` is **gone** (v6): it was a second public spelling
  of a picture you already knew how to ask for, and a user who typed it had to
  learn that some plots are made with `plot` and others with `to_plot_spec`. There
  is one verb now — `ts.plot(subject)` / `subject.plot()` — and one seam every
  subject carries, so `ts.plot` classifies with a single predicate. A guess at the
  old name is answered by name, and the line it hands back runs on the object that
  was held. The seam auto-dispatches on the number of *selected*
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
  `__plot_spec__`; the remainder are inline tweaks / backend kwargs.
  `SystemPlottable.__plot_spec__` likewise splits plot kwargs from integration
  kwargs (`final_time`/`dt`/`steps`/`ic`/…) on that same closed set.

### The `System` protocol (`families/protocol.py`)

All four families + all derived wrappers implement:
`run(...) -> Trajectory`, `step(n_or_dt) -> new state`, `state()`, `time()`,
`reinit(u, **kw)`, plus the data members `dim` and `family`.

**`set_state` left the protocol in v6** and became a per-family *capability*: on
Python >= 3.12 `isinstance` checks data members, so keeping it while removing it
from `DelaySystem` (whose state is a history function, not a point) would make
`isinstance(mg, System)` **False**.  Ask for it with `hasattr`.
`trajectory` -> `run` and `is_discrete` -> `family` are hard breaks for
third-party implementers; a `Protocol` cannot supply a default for a data member,
so there is no compatible middle ground.

- First `step()`/`state()` on a cold system does an implicit `reinit()`.
- ODE: `reinit` lowers the system to an engine tape once; each `step(dt)`
  integrates one `dt` chunk through `engine.run.integrate` from the live state.
- **`backend="reference"` is honored through the stepping protocol** (ODE):
  `reinit(backend="reference")` resolves and stores the backend (`resolve_backend`
  raises `InvalidParameterError` for an unknown name), and `step()` then routes to
  `_step_reference` — one `dt` chunk on the pure-Python reference ODE integrator
  (the same path `run(backend="reference")` uses), so the wheel-free oracle
  is reachable via `reinit`/`step`/`state`. It is **no longer silently coerced to
  `interp`** (diagnosis #5); reference owns no resumable `OdeStepper`, so it never
  builds the engine fast path.
- **DDE `set_state` does not exist** (state is a history function, not a point —
  it used to exist and raise `NotImplementedError`, a member that lies); use
  `reinit(u)` to restart from a constant past. DDE stepping is forward-only (each `step`
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
- `run(backend=)` defaults to `_default_backend` (`"jit"`). `"jit"`
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

### A closed vocabulary is checked with `in` — so guard the hash (v6.0.1)

A keyword whose domain is a closed set is validated as `if value not in <set>:
raise InvalidParameterError(...)`, which is correct for every value a caller
might *mean* and wrong for every value they might mistype as a container: **`in`
hashes its left operand first**, so the call died with the interpreter's words
about a private table one line before the typed error would have named the
spellings. Measured on v6.0.0 at **six** doors — `force=` (the one filed),
`theme=`, `primitive=`, `layout=`, `backend=` and `kind=`, the last two being far
more commonly typed than the first. Two of the six hashed *again* further in (the
`_invalid_primitive_message` builder, and `Registry.get`, which is shared by all
six registries), so the guard is needed at the lookup **and** at the message.

`_utils/lookup.py::is_hashable` is the one spelling; ask it before any
membership test over caller input. Two sibling traps live in the same doors and
are guarded the same way: a **numpy** value makes `==` elementwise (so
`mode == "overlay"` raises *"truth value of an array is ambiguous"*), and
`difflib.get_close_matches` requires a `str`. Gate:
`tests/test_viz_frictions.py::test_an_unhashable_value_is_refused_by_name_at_every_closed_vocabulary_door`
(6 doors x 4 value types) plus its `..._is_untouched` twin, which proves the
guard never costs a legal call.

### A variable-dimension system's `__init__` is not a second front door (v6.0.1)

Five catalogue systems resolve their own `dim` from a structural parameter —
`GrayScott` / `SwiftHohenberg` / `KuramotoSivashinsky` from `N`, `Lorenz96` from
`N`, `MultiChua` from `n_circuits` — so each needs a custom `__init__`. Every
hand-rolled copy had drifted into being *laxer* than `SystemBase.__init__`, in
two ways, and **all five** were affected (the review named only the field
systems; a `with_params` sweep over the registry found the other two):

- **a parameter given twice was silently preferred, not refused** — the merge
  happened *before* `super()`, so the base never saw the duplication and
  `KuramotoSivashinsky(L=40.0, params={"L": 22.0})` built at `L = 40.0` in
  silence;
- **`dim=` / `field_shape=` were rejected as unknown parameters** — `with_params`
  and `copy` forward both on every rebuild, a free `**param_kwargs` swallowed
  them, and so re-parametrising any of the five raised, breaking continuation,
  orbit diagrams and every parameter sweep over them.

`systems/_declared_params.py::merge_declared_params` is the one merge; a custom
`__init__` binds `dim`/`field_shape` explicitly and discards them (they are
derived). Gates: `test_families_fixes.py::TestAFieldSystemRefusesAParameterGivenTwice`
(7 cases) and `::test_no_catalogue_system_raises_on_reparametrisation`, which is
registry-driven, so a sixth such system cannot ship broken.

### Solver tolerances (v6, `_utils/tolerances.py`)

**Every `rtol=`/`atol=` default in the library is a named constant in the leaf
module `_utils/tolerances.py`.** Before v6 the pair `1e-6`/`1e-9` was duplicated
as bare literals across ~16 sites in five subpackages plus the docstrings that
quoted them — which is exactly how two "same" defaults drift apart. The module
is a leaf (imports nothing from `tsdynamics`), so `families/` (which imports the
engine only lazily) and `_engine/run.py` both take it at module scope with no
cycle; `_engine/run.py` was rejected as the home for precisely that reason. A
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
- `run(backend=...)` runs the iteration on the Rust engine (`"jit"`
  default / `"interp"` / `"reference"` pure-Python oracle). The engine loop lives in
  `crates/tsdyn-engine/src/map.rs`; all backends lower `_step` to the IR, so
  piecewise/`numpy`-ufunc steps raise `TapeCompileError`. The engine path
  diverges loudly (raises); the random-IC retry still applies when `run` is
  called without an explicit `ic`.

### `run` is THE trajectory verb (v6) — `integrate` / `iterate` / `trajectory` are gone

One verb on all four families and every derived wrapper.  `integrate`, `iterate`
and `trajectory` are **removed, not aliased**: `hasattr(sys, "integrate")` is
`False`, and the `AttributeError` names `run` and hands back a runnable line.
`run` now *owns* the body (the direction reversed — it used to forward to
`integrate`).

**Every signature is CLOSED.**  Before v6 `DelaySystem.run` accepted and
*silently dropped* any keyword — `max_step`, `t0`, `events`, `nonsense_kw` all
returned baseline-identical trajectories.  Each family routes its leftovers
through `families/_kwargs.py::reject_unknown_run_keywords`, driven by a per-name
**why** table: a word that reached the wrong family is not a typo, it is the
right word for a different kind of dynamics, so the message states the
mathematical reason and gives the line to type.

| family | `run` binds |
|---|---|
| `ContinuousSystem` | `final_time`, `dt`, `t0`, `ic`, `transient`, `solver`, `rtol`, `atol`, `max_step`, `backend`, `seed`, `events` |
| `DiscreteMap` | `steps`, `ic`, `transient`, `backend`, `seed`, `max_retries` |
| `DelaySystem` | `final_time`, `dt`, `ic`, `history`, `transient`, `solver`, `rtol`, `atol`, `backend`, `seed` |
| `StochasticSystem` | `final_time`, `dt`, `t0`, `ic`, `transient`, `solver`, `seed`, `backend` |

- **`method=` became `solver=`** on `run` *and* `reinit`: `solver=` selects a
  numerical kernel, `method=` selects an *estimator* on an analysis
  (`lyapunov_from_data(method="kantz")`).  A `method=` at `run()` raises naming
  `solver=`.  (`n` likewise became `steps` — one concept, one spelling.)
  **`traj.meta` records the kernel under BOTH `solver` and `method`**, the same
  value: `meta` is provenance a user reads back, and recording only the word the
  signature *stopped* accepting made the record disagree with the call that
  produced it.  `_default_method` is `"rk45"`, lowercase — the canonical
  spelling `meta` and `system.info` both print (it read `"RK45"`, so the one
  ClassVar and the two displays gave three spellings of one kernel).
- **A map's `run(steps=N)` returns `N + 1` rows**, the initial condition first,
  exactly as a flow returns its `ic` at `t0`.  It used to return `N` rows
  starting at `f(ic)`, so `t[0] = 0` labelled `x_1`, `traj["x"][n]` was
  `x_{n+1}`, and every cobweb started one iterate late — silently, and only for
  the family whose users are counting iterates.
- **`dt=None`** means "the family's `_default_dt`" (0.02 for ODE/DDE/SDE), which
  is what `system.info` prints under `defaults`.
- **`run(ic=…)` no longer mutates `self.ic`.**  Measured at HEAD: `l.ic` was
  `None`, and after `l.run(ic=[3,3,3])` it was `[3. 3. 3.]`, so a later bare
  `run()` silently started elsewhere.  The **auto-resolved** cases (`_default_ic`,
  the random draw) still latch — that is what makes a bare `run()` twice
  reproducible.
- **`run()` is always a fresh integration; `step()` is the one that continues.**
  `pmap.run(steps=5)` twice used to return different data; every wrapper now
  reinitialises first.

### `StochasticSystem` extras

- **Diagonal-Itô SDE** family (`families/stochastic.py`):
  `dX_k = f_k dt + g_k dW_k` with independent `dW_k`. Subclass contract is
  `_drift(y, t, **params)` (like `_equations`) + `_diffusion(y, t, **params)`
  (one noise coefficient per component); both symbolic, both lower via
  `engine.compile.lower_sde` (drift tape + diffusion tape, the latter carrying
  `∂g/∂u` for Milstein).
- `run(..., solver=, seed=, backend=)` runs a fixed-step scheme — `dt` *is*
  the noise scale `√dt` (so `dt` sets both the discretisation and the output grid).
  `solver`: `"euler_maruyama"` (order 0.5, default) or `"milstein"` (order 1.0).
  `seed` makes the noise realisation reproducible (recorded in `traj.meta`).
  `backend`: `"jit"` (the default, like every other family) / `"interp"` — the
  compiled engine via `tsdynamics._rust` (stream E-WIRE) — or `"reference"` (pure
  Python). (This line said `"reference"` was the SDE default; it never was.)
- `ensemble(states).run(..., seed=)` seeds member `i` from `seed_for(seed, i)` —
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

## Analysis results — the repr IS the answer (v6, C3)

Every registered analysis returns an `AnalysisResult` subclass; there are **32**
of them and a user constructs none. Two things changed in v6.

**`summary()` is deleted and `__repr__` became what it printed.** The readable
text already existed inside `summary()`, which nothing advertised and no REPL
calls, while the repr — the thing a console and a notebook actually show — gave a
constructor-shaped one-liner. Every result now renders as

```
<Name>  <THE ANSWER>   <verdict>   (<subject>)
    <up to four supporting lines>
    [0] <item>                       # a collection, 10 items then "... [N total]"
```

built from four hooks on `AnalysisResult`, none of them mandatory: `_answer()`
(the measurement, defaulting to the `_repr_fields` rendering), `_interpretation()`
(the verdict), `_context()` (the trailing parenthetical — the system by default,
or the settings that make the number meaningful), and `_details()` /
`_item_lines()`. `__str__` is the headline, `_repr_html_` is the repr in a
`<pre>` (so the notebook and the console cannot drift), and `__format__` formats
the number. A result commonly listed inside another (a fixed point, an attractor,
an orbit) overrides `_as_item()` for the compact list form — **not** `__str__`.
Number formatting lives in `_result_json.py`: `_sig` (significant figures — `np.round`
is scale-blind), `_state` (a state vector, `max_line_width=10_000`, elided above 8
components), `_vector` (a list of independent quantities, each at its own scale),
`_pct`.

**A verdict must be supported by the data.** `WadaResult.applicable` is the
archetype: `wada_property` early-returns zeros when there are fewer than 3 basins,
so `W = 0` read as a *measured negative* when nothing was measured; the repr now
says `not applicable — the Wada test needs ≥ 3 basins, this image has 2` and
`to_dict(full=True)["W"]` is `None`. The **Lyapunov verdict** follows the same
rule (contract §4.4): the zero floor is the estimator's own realised zero
(`min|λ|` for a flow, which has a structural zero exponent; the relative floor for
a map, which does not), and the regime is named **only when the count is stable
across a 10× tolerance band** — otherwise the repr says `indeterminate at this
horizon`. That fixes `Lorenz` at `final_time=20` (was *hyperchaotic*),
`HenonHeiles` (now honestly hedged) and `LotkaVolterra`, a shipped conservative
system that was called **chaotic** at every horizon. It is a **repr-only** rule
with no estimator change; a σ-carrying estimator is a v6.1 ticket.

**A quantity read off a RUNAWAY ORBIT is untrusted, however clean the fit**
(round 8). `Trajectory.unbounded` detected an escaping orbit from v6 and the
repr printed it in red; what it did *not* do was reach the estimators, so
`correlation_dimension` of a Chua run that peaked at `1.6e9` answered
`D_corr = 0.48954` over `log r ∈ [8.93, 10.4]` with `R² = 0.9976` and
`trusted = True`, and `lyapunov_from_data` called a monotone blow-up
`chaotic (λ > 0)` to four significant figures with a `7e-5` standard error and
zero warnings. Three pieces, one reading each:
`analysis/_common.py::runaway_meta` (the guard: warns `RunawayOrbitWarning` and
returns `{"unbounded": <line>}`), `ScalingResult.runaway` (reads that key), and
`ScalingResult.trusted`, which ANDs it in — so it also reaches the two
subclasses that declare `trusted` as a field. **It reads the COERCED DATA, not
the `Trajectory` flag**, which is what catches `lyapunov_from_data(tr["x"][::10])`:
indexing a runaway hands back a bare array, and an escape a slice can launder is
an escape that gets through. It **outranks every fit diagnostic** — an escape
produces the straightest log–log curve in the library — and it is named first in
both `_details` and the Lyapunov reason clause. The number is still returned
(refusing outright makes the estimator useless on exactly the systems people
reach for). Stamped at four sites: the three `DimensionResult` constructions and
`lyapunov_from_data`. Gate: `tests/test_beta_retest.py::TestAnAnalysisOfARunawayOrbitSaysSo`.

**A result behaves as the plain thing it replaced.** `f"{result:.3f}"` used to
raise on every numeric result. `np.asarray(fixed_points)` was a `(n,)` array of
*objects*. `ArrayResult / 2` raised while `ArrayResult * 2` worked. All fixed:
`_NumericOps` and `ArrayResult` now sit on `np.lib.mixins.NDArrayOperatorsMixin`
**plus** `__array_ufunc__` (both halves are required — `__array_ufunc__` alone
breaks `result * 2`, because `int.__mul__` returns `NotImplemented` without ever
reaching NumPy), and `to_dict(full=True)` adds
the derived answers the repr reports (`kaplan_yorke`, `recurrence_rate`,
`applicable`/`W`, `n_fit`/`r_squared`) — `full` only ever *adds* keys.

**Indexing a collection gives you a `numpy.ndarray`, never a result object.**
This is the owner's D1, applied at the root and to every sibling:
`fixed_points(lor)[0]` is the `(3,)` point, `periodic_orbits(...)[0]` the
`(p, dim)` orbit, `attractors(...)[0]` the `(dim,)` centre,
`windowed_rqa(...)[0]` the measure row. Making the record classes *emulate*
arrays was the half-measure that leaked: `np.asarray(fps[0])` worked while
`fps[0].shape` and `fps[0].tolist()` raised, and on an `AttractorSet`
`np.asarray(aset)[0]` (a centre) was a **different thing** from
`np.asarray(aset[0])` (a whole point cloud).

Nothing was deleted — the diagnostics moved **out of the way, not out of
reach**, under one spelling learned once across all four collections:

- **`.details`** — the records, aligned index-for-index with `[]`
  (`fps.details[0].eigenvalues`);
- the **vectorised per-member arrays** that were already there —
  `fps.points` / `.eigenvalues` / `.is_stable`, `orbits.periods` /
  `.multipliers`, `aset.centers` / `.cells`;
- **`by_id(k)`** — still the by-label door, and still hands back the *record*
  (an id is what you read off a basin image, and the next question is always
  *about* that attractor). `BasinFractions.by_id` correctly stays a float: its
  members **are** floats, so there is no record to hand back.

Collections remain complete sequences indexed **by position** (`AttractorSet[0]`
is the first attractor; it used to raise `KeyError`, because ids start at 1).
All 32 reprs are byte-identical across the change. Two are deliberately *not*
bare arrays, because a 2-tuple of plain Python is not a class to learn:
`OrbitDiagram[i]` is `(value, points)` and `ReturnMap[i]` is `(v_n, v_{n+1})` —
`for value, points in od` is the `dict.items()` of a parameter sweep, and
`np.asarray(od)` is still the real `(N, 2)` scatter.

**`ts.analysis.results`** is the namespace for the 32 classes (contract §2.7):
off the flat `ts.analysis` listing, still importable, still what `isinstance`
sees; each class still lives in the subpackage that produces it.
Gates: `tests/test_result_repr.py` (walks `AnalysisResult.__subclasses__()`, so a
new result class cannot ship without a rendered fixture in
`tests/_result_fixtures.py`), `tests/test_result_plain.py`,
`tests/test_results_namespace.py`, and — since round 9 —
**`tests/test_result_visibility.py`**, which pins what `result.<TAB>` shows: a
literal `LISTED` row per class (**492** names total, re-measured never nudged),
the capability half derived from `_HIDDEN_ATTRIBUTES` / `fields()` / the MRO so
every hidden name is proved to still resolve, and the two reachability rules
above. It is selected by `--changed` for **every** analysis area (the
`_RESULT_SURFACE_GATE` appended to each `_AREA_TESTS` row), because the
declarations it pins live in the area modules, not in the shared `_result*.py`
files that escalate to a full run — without that, an area could add a public
field to its result class, grow the listing, and go green on the PR.

### `result.plot` is the FIFTH plotting door, and it speaks the same words

`result.plot` is a `_PlotAccessor` (`analysis/_result_viz.py`) — callable, and a
**namespace of the transform names that admit this result**, exactly as
`traj.plot` / `system.plot` are (`_utils/plot_namespace.py`). So
`result.plot.<TAB>` means one thing everywhere in the library, and
`dim.plot.scaling_fit()` **is** `ts.plot(dim, "scaling_fit")`.

Two things were wrong and are fixed:

- **The eight kind-forcing methods are gone** — `.scaling()` `.diagnostic()`
  `.time_series()` `.phase()` `.image()` `.bifurcation()` `.return_map()`
  `.section()`. They did not build a different picture; they **relabelled** the
  result's own spec with a different `PlotKind`. Measured over the 30 result
  fixtures: *not one of the eight changed a single byte of layer data*, and each
  was truthful only when the kind it forced already was the result's natural kind
  — a no-op. `.time_series()` / `.image()` / `.bifurcation()` / `.section()` were
  truthful on **zero** results, so `lyapunov_spectrum(lor).plot.section()`
  returned a bar chart of exponents labelled `poincare_section`. The module had
  already deleted `.histogram()` and `.spectrum()` on exactly that reasoning; the
  other eight fail the same test. The replacement is strictly more capable,
  because a transform *computes* where a relabel only renamed:
  `ts.plot(dim, "scaling_fit")` builds **five** layers (curve, fitted line, window
  markers) where `dim.plot.scaling()` built one. A guess at a retired name is
  answered by name and says what it used to do.
- **The style and figure vocabularies used to raise here.** Measured:
  `result.plot(color=…)`, `(title=…)`, `(theme=…)` and `(xscale=…)` all raised
  `InvalidParameterError` while `ts.plot(result, …)` accepted every one. The door
  now splits its keywords four ways — the result's own `__plot_spec__` names, the
  style vocabulary + `theme`, the 17 `FIGURE_KEYS`, then the renderer table — and
  an unknown word names all four sets. `kind=` is refused, naming the transform
  spelling.

Gate: `tests/test_plot_accessor_kinds.py` (inverted from the stream that once
proved each forced kind was a *valid* `PlotKind` — a valid kind is not a correct
picture).

---

## The analysis door (v6 — ruling A2)

**An analysis is a FREE FUNCTION whose first argument is the thing it is about.**
There is no bound method on a system, on a `Trajectory`, or on a result:

```python
ts.analysis.lyapunov_spectrum(lorenz)          # a property of the equations
ts.analysis.correlation_dimension(traj)        # a property of a point set
ts.analysis.kaplan_yorke_dimension(spectrum)   # a property of the answer above
```

The four topical accessors (`.lyap` / `.chaos` / `.dims` / `.recurrence`) and
`families/_accessors.py` are **deleted**. The correctness argument was in that
module's own source: `sys.chaos.zero_one()` pre-ran with the family's `run()`
defaults and reported **K = −0.026** for Lorenz where the free function reports
**0.999** — a convenience that hides a sampling choice is a silent-wrong-answer
generator. Guessing a removed name is answered by
`families/base.py::_absent_name_error`, which names the free function.

**Discovery is therefore a first-class deliverable**, and it is generated:

- **`ts.analysis.<TAB>` is exactly 52 names** — 49 analyses plus `find`,
  `register`, `results` — *computed from `registry.analyses`*
  (`_refresh_surface`), never hand-written, so a registered analysis cannot be
  missing from the tab surface and a listed name cannot be missing from the
  registry. The 32 result classes stay **bound** on `ts.analysis` but live at
  `ts.analysis.results` and appear in no `__all__` (C2).
- **`ts.analysis.__doc__` is the grouped map**, generated at import
  (`_discovery.grouped_map`): grouped by *what you are holding* — **20** analyses
  take a system, **24** take a trajectory/array, **6** take another analysis's
  result — then by area. A flat sort cannot answer "is this chaotic?"; one of the
  five that can (`zero_one_test`) contains no word a newcomer would search for.
- **`ts.analysis.find(what, /)`** takes ONE positional: a **string**
  (free-text over name/area/keywords/summary) or a **subject** (a system, a
  `Trajectory`, an array, a result, or any of their classes) or nothing.
  `find(lorenz)` → 20, `find(henon)` → **13** (a map has no vector field, so the
  seven `flow`-only field analyses drop out), `find(traj)` → 24, `find()` → 49.
  Returns an `AnalysisList` of the **functions** whose repr is the grouped table.
  The scorer's weights are in `_discovery.score`; a **frozen `GOLD` table** in
  `tests/test_analysis_discovery.py` fails a build, not a user's REPL, when a new
  analysis makes an existing question ambiguous.
- **`@ts.analysis.register(...)`** is the one registration door, used by every
  in-tree analysis at its own definition site (§7.7): `subjects=` (`"system"`
  expands to `flow`+`map`; `"trajectory"`/`"array"`; or a result class *name*),
  `area=` (one of `_discovery.AREAS`), `returns=`, `keywords=`, `cite=`, `doi=`.
  **Summaries are derived from the docstring** (`_discovery.summarise`: first
  sentence, RST roles unwrapped, ≤ 72 columns) — never a second hand-maintained
  string. The registry metadata keys are `subjects`/`area` (they were
  `needs`/`family` before v6).
- **The ten capability subpackages stop shadowing** (C4). After they have
  imported and self-registered, `analysis/__init__.py` **deletes them from its own
  module dict** and a teaching `__getattr__` answers the guess:
  `ts.analysis.lyapunov` raises an `AttributeError` naming the three free
  functions. It is deliberately an `AttributeError` and **not** `MovedInV6`: the
  import machinery falls back to `sys.modules` for a submodule only on
  `AttributeError`, which is what keeps `import tsdynamics.analysis.lyapunov` and
  `from tsdynamics.analysis import planar` working. **One exception —
  `ts.analysis.basins` is the ANALYSIS**, which is why
  `import tsdynamics.analysis.basins as bas` binds the *function*; reach that
  package with `importlib.import_module`.
- **A renamed name raises `MovedInV6`** (an `ImportError`, so the text survives
  `from tsdynamics.analysis import X`) naming the one spelling:
  `basins_of_attraction` → `basins`, `find_attractors` → `attractors`,
  `periodic_orbit` → `periodic_orbits`, `bifurcation_diagram` /`bifurcation` →
  `orbit_diagram`. A *guess* stays an `AttributeError`, so
  `hasattr(ts.analysis, "anything")` keeps answering `False`.
- **Four keyword renames a user types** (C3 — one concept, one spelling):
  `return_map(method=)` → `kind=` (`method=` is the *estimator* word, `kind=`
  picks `"max"`/`"min"`/`"poincare"`); `estimate_period(component=)` →
  `components=`; `zero_one_test` **gains** `components=0`; **and since v6 round 4
  every other door agrees** — the nine analysis doors that still said
  `component=` (`autocorrelation` `cao_dimension` `embed` `embedding_dimension`
  `false_nearest_neighbors` `mutual_information` `optimal_delay` `orbit_diagram`
  `return_map`) and all sixteen plot transforms that did (`psd`, `cobweb`,
  `hilbert*`, `spatial_field`, `ensemble_fan`, …) were renamed together, so a
  user meets ONE word at every door; the singular raises, naming the plural, and
  private helpers (`_as_series(component=)`, `_component_index`) keep their own
  name because they are not a door.  Gates:
  `tests/test_api_contract.py::COMPONENT_SINGULAR_DOORS` and
  `tests/test_polish_standards.py::_NAMEGATE_DEFERRED_PARAM`, both now **empty**
  frozensets asserted by set equality, so a new singular door fails the build.
  `ftle_field(time=)` likewise became **`final_time=`** (M49), the word its two
  promoted siblings and `run` already spoke.  And the scaling-window
  factor on every dimension estimator, `tol=` → **`flatness=`** — it is how flat
  the fitted log–log window has to be, one letter from the `rtol`/`atol` the same
  estimators take and unrelated to both. Six doors carry it
  (`correlation_dimension`, `generalized_dimension`, `dimension_spectrum`,
  `fixed_mass_dimension` directly; `box_counting_dimension` and
  `information_dimension` through `**kwargs`, which intercept `tol=` by hand so
  the message cannot name the private `_core_kwargs`). The private
  `_scaling.fit_scaling_region(tol=)` keeps its own spelling: it is slated for
  privatisation, and renaming it reaches into a gated docs page and two other
  slots' files. Gate:
  `tests/test_dimensions.py::TestTheScalingWindowKeywordIsCalledFlatness`.
- **`_discovery.teach(name, *, held, has_system=)`** is the ONE builder for every
  wrong-subject message; `attribute_error()` (object doors) and `wrong_subject()`
  (free-function doors) wrap the same body, wrapped to 88 columns *including* the
  traceback's own `AttributeError: ` prefix. Three small tables carry what the
  registry cannot: `PRODUCER` (which analysis makes a result-first analysis's
  subject), `SIBLING` (the data-first twin), `NO_SIBLING` (why there isn't one).
  **The shipped doors call it**: `analysis/_common.py::reject_system` /
  `reject_data` route every *named, registered* analysis through `wrong_subject`,
  so the text a user sees is the text the gate asserts. (They kept their generic
  wording for the two cases the builder cannot serve: a shared coercion helper
  that calls in with no `analysis=` name, and a `hint=` caller whose input is not
  a trajectory at all.) The three bespoke `hint=`-carrying guards on the
  **result**-first analyses are gone — `kaplan_yorke_dimension`, the basin
  metrics and `tipping_points` each opened with *"expects measured data"*, which
  is the wrong clause for an analysis that reads another analysis's answer.
  `find_line(held)` builds the `ts.analysis.find(...)` line **with the count that
  call will print**, read from the registry.
- **A removed name must not be read as a STRING** (contract §9.4 rules 1 and 3):
  `getattr(system, "is_discrete", False)` and `type(system).default_ic` both have
  *legal* defaults, so the rename orphans them silently. Two live instances were
  fixed here — `orbits/poincare.py::_seeded_ic` (a seeded section overrode the
  declared IC of all **46** systems that declare one) and
  `basins/attractors.py` at three sites (which decides whether the basin FSM
  advances a map by one iterate or by `dt`). The sweep is a gate:
  `tests/test_analysis_discovery.py::TestNoRemovedNameIsReadAsAString`.

**The 8 `analysis/planar.py` field analyses are public in v6**
(`flow_field` `streamlines` `nullclines` `ftle_field` `escape_time_field`
`transient_time_field` `trace_determinant` `invariant_density`), plus
`set_distance` from `tsdynamics.data`. They register from `analysis/__init__.py`
(a single module cannot self-register the way a subpackage can). Seven declare
`subjects=("flow",)` — they evaluate or integrate the RHS at points that are in
no trajectory — which is what makes `find(henon)` answer 14 and not 21.

**Known exception to "every registered analysis returns an `AnalysisResult`"**:
14 of the 50 return a plain value (an array, a float, a tuple, a list of small
records) — the 7 promoted field analyses, `invariant_density`, `correlation_sum`,
`dimension_spectrum`, `autocorrelation`, `sagitta_profile`,
`estimate_dt_from_sagitta`, `set_distance`. They are enumerated in
`tests/test_analysis_registry.py::_PLAIN_VALUE_ANALYSES` and declare no
`returns=`; every other analysis must declare the class it returns, and the gate
checks the declaration against the annotation. Giving those 14 result wrappers is
a v6.1 item (it moves a returned *type*, in two files outside the analysis
subpackages).

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
  (`tsdynamics._engine.run.Event`) is a *symbolic* scalar condition `g(u, t) = 0`
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
  The streaming `step()` / `exponents()` API is wrapped into the standard
  burn-in + time-weighted estimate by `ts.analysis.lyapunov_spectrum` — analyses
  are free functions (A2), so there is no `TangentSystem.lyapunov_spectrum`.
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
- **`max_lyapunov` is GONE; `lyapunov_spectrum(k=1)` is the one spelling.** Two
  doors onto one question answered it with two numbers: measured on Hénon from
  `ic=[0.1, 0.1]`, `max_lyapunov(n=20000)` returned `0.41973…` and
  `lyapunov_spectrum(k=1, n=20000)[0]` returned `0.41599…`, because `n` counted
  *rescaling cycles of ten iterates* at one door and *iterations* at the other —
  ten times the work for one nominal horizon. The better half moved in (the
  burn-in, and the Jacobian-free two-trajectory machine) and the second door
  closed; `ts.analysis.max_lyapunov` raises `MovedInV6` naming this one. That
  takes `registry.analyses` from 50 to **49** and `ts.analysis.__all__` from 53
  to **52** — re-measure, never nudge.
  The **Benettin two-trajectory** estimator survives as the branch
  `lyapunov_spectrum` takes when the system has no right-hand side to
  differentiate (a `WrappedSystem` around an external stepper, a non-smooth map):
  it needs `set_state`, so it still raises for DDEs, and its continuous
  normalization divides by the **measured elapsed `time()`** of the reference run
  (not a guessed step-size attribute), so it is correct for any continuous system
  including a `WrappedSystem` stepped with `dt=None`. Its two knobs `d0=` and
  `steps_per=` are **refused by name** on a system that does have a Jacobian,
  which is the whole point: there is nothing perturbed and nothing rescaled on
  that path.
- **Units are stated at the door, and enforced.** `transient` was TIME at four
  Lyapunov doors and a COUNT at two, for the same subject:
  the two-trajectory door discarded ~10 time units where
  `lyapunov_spectrum(lorenz, transient=500)` discarded 500 — 50× apart, sibling
  functions, one word. It is time-for-a-flow / iterations-for-a-map everywhere
  now, every shipped default is bit-identical, and an `int >= 100` on a flow is
  **named** rather than read as a 100× longer burn-in. `lyapunov_spectrum` also
  took `method=` for a kernel after v6 renamed that word to `solver=` at `run()`
  — a user who obeyed the rule they had just been taught hit
  `unexpected keyword 'solver'`. Both doors say `solver=` now, and the four
  catalogue `_known_lyapunov["kwargs"]` dicts that carried `"method"` were
  renamed with them.
- `lyapunov_from_data` (A-LYAP) estimates the maximal exponent from a measured
  series via delay embedding + neighbour divergence (Kantz 1994 default,
  Rosenstein et al. 1993 optional); returns a `LyapunovFromData` carrying the
  stretching curve `S(k)` — fit the linear scaling region (inspect, then pass
  `fit=(lo, hi)`). A private delay-embed helper keeps it independent of the
  delay-embedding stream. **`dt` is read from the data when the data knows it**
  (`_sampling_interval_of`): a `Trajectory` carries its sampling interval, so
  `lyapunov_from_data(traj)` answers **per unit time** and compares directly with
  `lyapunov_spectrum`, while a bare array (no time axis) keeps `dt = 1.0`, i.e.
  per sample / per iteration. **Since v6 the `t` AXIS is the source, not
  `meta["dt"]`** — `meta` used to record what the *run* asked for while slicing
  carried it verbatim, so a decimated trajectory reported the undecimated step:
  measured, `tr[::5]` of a Lorenz run at `dt=0.01` still said
  `meta["dt"] == 0.01` while `np.diff(t)[0] == 0.05`, and the exponent came back
  **9.809 where the truth is 1.962**, off by exactly the decimation factor with
  no exception. The reader prefers `traj.dt` (the `Trajectory`'s own axis-derived
  reading) and falls back to `meta["dt"]` only when there is no usable axis. An
  explicit `dt=` always wins; a non-uniform `t` axis raises rather than being
  averaged into one. **Round 8 closed the other half**: selecting rows now
  re-derives the three `meta` keys that describe the *rows* rather than the run
  (`data/trajectory.py::_ROW_DERIVED_META` = `dt`/`t0`/`ic`, applied once in
  `Trajectory._rows`, which `__getitem__` and `after` both go through), because
  `meta` is public and a user reading `meta["dt"]` to convert a per-sample
  exponent got a 5x wrong factor handed over on request. A slice with no uniform
  step **drops** `dt` rather than reporting a wrong one; everything else in
  `meta` (system, params, solver, tolerances) is still true of a slice and is
  carried verbatim.
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
  cyclic-shift dedup. **The same verb finds a flow's limit cycle** (v6 absorbed
  the old `periodic_orbit`, singular: one verb, one return type — an `OrbitSet`
  of one) by single shooting on `(x0, T)`; on a flow the `period` positional is
  the period *guess*, since a flow's period is a real unknown (bordered Newton +
  monodromy via the RK4 variational core; Floquet multipliers for stability, the
  trivial ≈1 multiplier found by eigenvector alignment with `f(x0)`; rejects
  equilibrium-collapse on a centre). A map with no `period` named raises, saying
  a map's orbits are the fixed points of `fᵖ` — one root problem per `p`.
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
- `attractors` / `basins` (A-BASIN) drive any map/flow over a
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
  `basins` paints
  a `Grid` (pass a separate `recurrence` box to image a *slice* of a higher-dim
  flow — the magnetic pendulum); `basin_fractions` is Monte-Carlo basin stability
  (Menck 2013). **The seed count is `n_seeds=` at every door in this family**
  (round 8): `basin_fractions` and `continuation` alone spelled it `n`, which
  `basin_fractions`'s own docstring conceded, and `n=` now raises naming
  `n_seeds=`. The four `**fsm` doors (`basins` / `attractors` /
  `basin_fractions` / `continuation`) validate their leftovers through
  `basins/_common.py::reject_unknown_fsm` **before** forwarding, so a typo here
  is answered by name instead of surfacing as
  `TypeError: _AttractorMapper.__init__() got an unexpected keyword argument` —
  a private class, in an exception with no remedy. The flat-recurrence-box error
  builds its runnable line from the **caller's own box** (`_widened_box_lines`);
  it used to end in a hand-written three-axis template, so a 4-D system was
  handed a line that fails on dimension the moment it is pasted.
  The metrics read a label image (no integration, fast tier):
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

**`ts.viz.__all__` is exactly FOURTEEN names** (down from 32; gate
`tests/test_viz_dispatch.py::test_ts_viz_tab_surface_is_the_contract`, exact and
sorted — re-measure, never nudge):

```
Plot  VisualizationDegraded  compatibility  draw  geometry  grid  load
plot  primitives  renderers  spec  styles  themes  transforms
```

Four registries with the **identical four-verb shape** (`register` / `names` /
`find` / `get` — learn one, know all four): `transforms` · `primitives` ·
`renderers` · `themes`. (`transforms` carries a **fifth**, `allow`, which widens
a transform's declared primitive row — see below; it is the only asymmetry and
it exists because a user's custom primitive has no other way in.) Two drawing
doors (`plot`, `draw`), one panel arranger (`grid`), one received type (`Plot`),
the arrays escape hatch (`geometry`), the matrix (`compatibility`), the style
table (`styles`), the round-trip loader (`load`), the IR one dot away (`spec`),
and **the warning you catch** (`VisualizationDegraded`, promoted in v6 round 9:
34 doc mentions and 120 test uses against an address inside
`ts.viz.render`, which is why `docs/visualization/styling.md` was reduced to
comparing `w[0].category.__name__` to a *string*).

Everything else is **demoted, never removed** — the IR nouns, `to_json` /
`from_json` / `to_dict_envelope` / `from_dict_envelope` / `SCHEMA_VERSION`,
`STYLE_KEYS` / `Theme` / `THEMES` / `get_theme` / `set_theme` / `register_theme`
/ `normalize_style`, `Plottable`, `plot_transform`, `make_frame`, and the
`render` subpackage are all still bound and importable (`_INTERNAL_NAMES`), just
off the tab surface. Two names genuinely stop resolving and say so
(`viz/__init__.py::_MOVED`): `ts.viz.PlotSpec` → `ts.viz.Plot`, and
`ts.viz.list_transforms` → `ts.viz.transforms.names()`. A *guess* is still an
ordinary `AttributeError`, so `hasattr(ts.viz, anything)` keeps working.

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
  `stability.py`, and — v6 — `results.py`: `recurrence` / `orbit_diagram` /
  `basins` / `ensemble_fan`, the four whose *result* could draw itself but which
  had no transform, and so could not be overlaid, gridded or re-primitived).
  **A transform owns no new math** — it adapts an estimator
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
  **`density`'s bin count follows the sample count** (`_density_bins`, round 8,
  `sqrt(n/4)` per axis clamped to `[16, 400]`). It was a flat `400 × 400` — right
  for the million-point orbit diagram its docstring cites, and **blank** for
  everything smaller: 200 points over 160 000 bins is 0.125 % occupancy and
  nothing survives the panel downsample. Measured ink at each transform's *own
  shipped example*: `phase_portrait` 0.0000 → 0.193, `delay_embedding`
  0.0008 → 0.218, `poincare_section` 0.0005 → 0.084. The cap still binds at the
  orbit diagram, so that picture is unchanged. Three matrix cells rendered blank
  and `tests/test_viz_compatibility.py` called all three green, because an artist
  can carry finite data and still be invisible — which is the whole argument for
  the pixel layer below.
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
  ts.plot(a, b, "phase_portrait")                # TWO subjects, one transform
  ts.plot(vdp, t1, t2, "vector_field", "nullclines")   # the flagship figure
  g = ts.viz.geometry(sys, "ftle", grid=201)     # the arrays, and stop there
  ts.viz.draw(g, "contour")                      # hand them back to the library
  ts.viz.draw({"x": r, "y": C}, "line", labels=("log r", "log C(r)"))  # no transform
  ts.viz.grid(p1, p2, p3, rows=1, cols=3, share_color=True)   # panels of anything
  ts.viz.transforms.names(); ts.viz.compatibility()   # what can this draw?
  ```

  A positional **string, `("name", {options})` pair, or `T`** is a transform;
  anything else is a subject (a system, a `Trajectory`, a result, a finished
  `Plot` — **closure** — **or a bare array**, coerced to an index-time
  `Trajectory` at the door).
  `ts.plot` always returns a `Plot`, so a result feeds straight back in.
  **Subjects × transforms is a cross product FILTERED BY DECLARED SOURCE (v6):**
  a named transform applies to every subject its `subjects` admit, a subject no
  named transform admits draws its **default view**, and a transform no subject
  admits raises naming what it needs. That is what makes
  `ts.plot(a, b, "phase_portrait")` one figure with two orbits (it used to answer
  *"needs exactly one subject to apply them to, got 2"* — the most obvious
  comparison plot in dynamics) while keeping the flagship
  `ts.plot(vdp, t1, t2, "vector_field", "nullclines")` legal, which a *full*
  cross product would not (it would hand `vector_field` a `Trajectory`).
  Shared keywords are routed only to the transforms whose `compute` (or chosen
  primitive) accepts them — **plus every canonical style key** (applied to the
  named transform's layers) and the **17 figure keywords**
  (`viz.spec.FIGURE_KEYS`: `title`/`xlabel`/`ylabel`/`zlabel`/`xlim`/`ylim`/
  `zlim`/`xscale`/`yscale`/`zscale`/`xticks`/`yticks`/`zticks`/`clim`/`colorbar`/
  `legend`/`theme`), applied to every panel. The composition knobs
  (`layout`/`rows`/`cols`/`share_x`/`share_y`/`share_color`/`primitive`/`force`/
  `animate`/`fps`/`labels`/`ax`) are **on the signature**, so `help(ts.plot)`
  shows them. **There is no `on=`** — this line used to claim one, and the claim
  was measured false in v6 round 9 (`inspect.signature(ts.plot)`). Putting a
  result onto a figure you already have is
  `result.overlay_on(existing_plot)`, which is the *only* route and is therefore
  a permanent KEEP; since round 9 it also **widens** the host's window so what
  you added is inside the frame (a `model` host pins its own sampled box, so an
  equilibrium outside it used to be silently clipped).
- **`ts.viz.draw` is the arrays door** (`viz/transforms/_registry.py::draw`): a
  channel mapping, a **list** of them, or a hand-built `Geometry` → a `Plot`, with
  no transform registered and no IR type imported. `Geometry.transform` became
  **provenance, not a lookup key** (a stamp naming no registered transform gets an
  ad-hoc record allowing every primitive), and a hand-built geometry is stamped
  `FrameSpace.FREE` — *"I did not say what space this is"* — which **overlays
  with anything**: the caller opted out of the frame system, so there is no
  coordinate claim to violate. Because `draw` returns a `Plot`, it composes,
  grids and animates with everything else.
- **`share_color=True` has a real contract (v6).** Measured before, it was a
  complete no-op: three `flow_speed` panels drew three colorbars at clims
  `(5.8e-05, 16.74)` / `(5.2e-04, 35.69)` / `(1.4e-03, 96.86)` — three
  incomparable scales sold as a comparison figure. It now **unifies `clim`** to
  the union across colorbar-bearing panels, keeps **one** colorbar, and
  **refuses** panels whose colorbars label different quantities. Done in the IR
  (`compose._unify_colour`), so every backend inherits it.
- **Legend disambiguation runs over the FINAL layer set**, not per `add`: before,
  `plot(t1, t2, t3)` gave `(1)/(2)/(3)` while `plot(t1).add(t2).add(t3)` gave
  `(1)/(2)/(2)` — a legend naming two different orbits identically, which is a
  wrong answer, not a cosmetic one.
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
- **`labels=` names the curves, and it works at EVERY plotting door.**
  `ts.plot(a, b, labels=["reference", "perturbed"])` matched positionally to the
  subjects, and `traj.plot(labels=…)` / `system.plot(labels=…)` /
  `result.plot(labels=…)` naming the one curve that door draws.  Comparing two
  parameter values is the commonest figure in dynamics and it had no spelling at
  all — `label=` / `labels=` / `legend_labels=` were each refused, two of them
  suggesting `zlabel=`, which names an **axis**.  A word accepted at one door and
  refused at another is the defect the shared vocabulary exists to prevent, so
  the word is listed in `_UNIVERSAL_PLOT_KEYWORDS` (not in
  `_COMPOSITION_KEYWORDS`, which drives the *"that word belongs elsewhere"*
  sentence).  And **`nearest_keyword` never offers back the word that was
  typed** — a shared suggestion pool spanning doors produced
  `labels= — did you mean labels=?`, which reads as a bug in the library.
- **Writing one is ZERO DECLARATIONS (v6 round 11); four is what you write when
  you want CONTROL.** `@ts.viz.transforms.register()` over a function returning a
  channel mapping draws, and `source` / `frame` / `kind` / `primitives` are each
  *inferred from the first geometry* if you leave them out (`_registry.py`:
  `_infer_frame` / `_infer_primitives` / `_resolve_deferred*`). Every default is
  **derived, never guessed**: the compatibility row from the primitives' own
  declared `requires` (the predicate registration already used to reject an
  impossible pair, run forwards), the frame from the channel shapes (2-D `z` = a
  lattice, 1-D `z` = a 3-D curve, else `FREE` — *"I did not say what space this
  is"*, exactly what `draw()` stamps on hand-built arrays), the kind from the
  mark the chosen primitive emits, and `source="data"` because that is the wider
  category. Inference runs **once**, on the first geometry, and the filled-in
  record replaces the deferred one — afterwards it is indistinguishable from a
  declared transform. **An `allow()` made before the first draw WIDENS the
  inferred row** rather than being replaced by it; it used to be thrown away
  silently, which made the two extension doors unusable together (a custom
  primitive reaches a transform *only* through `allow`). And a transform that
  declares no `labels=` now names the axes it can **recognise** —
  `_registry._derived_labels`: a channel that *is* the subject's time column is
  `t`, one that *is* a named state component takes that name, anything else stays
  blank, because `x` is not a better label than none. Declared labels win
  outright and a transform returning its own `Geometry` never reaches it.
  Caveat, and it is real: an **inferred row is permissive where a declared one is
  curated** — the derivation is a channel test, not a taste test, so a lattice
  transform's inferred row admits `bars` as well as `image`. Every cell is
  structurally legal and the *default* is right; declare `primitives=` for a
  curated row.
  Writing all four is the same as it ever was:
  `@ts.viz.transforms.register(source=, frame=, kind=, primitives=)`.
  Everything else is **derived**: `name`
  ← `fn.__name__`, `doc` ← the docstring's first line, **`ndim` ← the frame's
  arity** (`_frames.space_arity`; measured, 33 of the 35 in-tree transforms
  already declared exactly that, and the 2 exceptions were frame
  mis-declarations — `invariant_density` now declares `frame=("scaling",
  "state2")`), `default_primitive` ← `primitives[0]`, `subjects` ← `source`
  (`model` → `("system",)`, `data` → `("trajectory", "array", "system")`).
  `ndim=` survives only for a geometry whose shape varies *within* one space
  (`spatial_field`: a 2-D lattice or a 1-D profile). `kind=` is **required**
  unless several frames are declared — it used to be de jure optional and de
  facto mandatory, so an author's omission failed at *plot* time on a user's
  machine. `source=` is validated at registration (`source="oracle"` used to
  register). `exclusive=` is gone (empty on all 35, driving a `!` marker that
  could never appear); `aliases=` is new (`direction_field` → `vector_field`) and
  an alias is a **spelling, not a second row** — one record, one matrix line.
  `compute` may return a plain mapping **or a list of them** (one `Part` each,
  reading the four reserved keys `label` / `style` / `primitive` / `mark`), so
  **no transform author ever needs an IR type**. Nothing private is imported:
  `ts.viz.transforms.register`, `PlotKind`, `Geometry`, `Part`, `FrameSpace`,
  `make_frame`, `Presentation` are all public. `plot_transform` is the same
  function under its pre-v6 name.
- **A new primitive is one decorator too** —
  `@ts.viz.primitives.register(requires=("x","y"), marks=("line","points"))`
  over a function taking a `Part` (`part["x"]` is the array) and returning the
  **same mapping convention transforms use**, each piece optionally naming its
  `mark`. The name is `fn.__name__` unless you pass one — the *facade* on
  `ts.viz.primitives` re-declared `name` as required long after the function it
  forwards to had stopped requiring it, so the zero-argument spelling raised
  `TypeError: register() missing 1 required positional argument` at the public
  address while working at the private one. Marks are coerced from the words an
  author already uses
  (`_primitives.as_mark`), so **no new `PlotKind` is ever needed to add a way of
  drawing** — the invariant that keeps the matrix growable.
- **`ts.viz.transforms` is FIVE names** — the four shared registry verbs
  `register` / `names` / `find` / `get`, plus **`allow`**. (It was 22 until v6
  round 9; the other seventeen were the IR types and the front doors, every one
  measured `is`-identical to its `ts.viz.<name>` / `ts.viz.spec.<name>` spelling
  — `plot_transform is register` included — so nothing became a second object and
  nothing stopped resolving.) `allow(transform, primitive)` widens a shipped
  transform's declared compatibility row: a user's own
  `@ts.viz.primitives.register("stem")` is refused by every shipped transform
  until it is called, and it is the **only** way in, so the refusal message names
  it. `find(subject=traj)` is the *user's*
  question ("what can I draw from THIS?"), `find("spectrum")` is free text over
  name + summary, `find(source=/frame=/primitive=/available=)` the author's.
  `ts.viz.compatibility()` prints the matrix **grouped by source**, each row with
  its one-line summary, `*` marking the default, a `†` footnote for the
  shape-dependent rows, and the three calls to make next.
- **Gates.** `tests/test_viz_compatibility.py` renders every declared cell on
  matplotlib and refuses every undeclared one; `tests/test_viz_transforms.py`
  pins the substrate, the PSD admission rule and the zero-declaration door;
  `tests/test_viz_gallery.py` pins the gallery against the registry (and, in the
  slow tier, renders every cell and fails on a blank figure).
- **The TRUTH layer (round 11): `tests/test_viz_truth.py`.** The owner could not
  tell whether the plots were *correct*, and no existing gate answered that:
  `test_viz_pixels.py` proves a transform **drew something**,
  `test_viz_compatibility.py` proves it **drew without raising**, and neither can
  tell a correct vector field from one rotated ninety degrees. This one checks the
  **numbers behind the picture** against oracles that never touch the viz layer —
  analytic signals whose answer is written down, the system's own `rhs` and
  `jacobian`, `numpy.linalg`, textbook constants (the logistic onsets at 3 and
  1+√6), and the `tsdynamics.analysis` estimator each adapter claims to adapt.
  **No transform turned out to be wrong**; the deliverable is the proof plus a
  *"What correct looks like"* section in each of those transforms' docstrings,
  naming the test that proves it, printed by `help(traj.plot.<name>)`. The module
  docstring enumerates the 19 of 39 with **no independent oracle** and why (the
  Hilbert family — the curve is its own definition; the four lattice-integration
  field analyses, whose only oracle is the same integration; and the adapters
  whose numerics are tested where they live). Teeth are shown by mutation (a
  transposed `flow_speed` image is caught at 2.01 max error), not by a regression
  it happened to find.
- **The PIXEL layer (round 8): `tests/_pixels.py` + `tests/test_viz_pixels.py`.**
  Every other viz gate inspects the *description* of a figure — the `Plot`, the
  layers, the recorded keyword — and never the figure, so almost every wrong
  picture the beta round produced passed every existing test. This one renders to
  Agg, reads the RGBA array back and **measures** it: `_pixels.py` is the
  instrument (`render` / `ink` / `changed_pixels` / per-region reads),
  `test_viz_pixels.py` the 23 tests over 198 items (13.6 s in the fast tier).
  Seven properties, each named after what a user sees: **not blank** (every
  transform's default primitive in the fast tier, all 101 declared matrix cells
  in the slow one), **ink matches the data** (recurrence at 5 % and 20 %),
  **colour honoured** (2-D, 3-D and images), **limits honoured**, **all 17 figure
  keywords built AND rendered AND shown to CHANGE the picture**, **animation**
  (consecutive frames differ; frame count matches, in memory and in a written
  `.gif`), **composition**. Everything asserted is a robust property — is there
  ink, did it move, is it roughly this fraction — never a hash and never a golden
  image, because fonts and layout solvers move pixels and a flaky pixel test is
  worse than none. Each floor carries its measured margin in its own comment;
  the thinnest is 2.9x. It found the two wrong pictures fixed in round 8: the
  `density` blank above, and `viz/render/mpl/_threed.py::_apply_3d_axes` applying
  the three axis **labels and limits and nothing else**, so `xticks`/`yticks`/
  `zticks`/`xscale`/`yscale`/`zscale` were honoured on a 2-D axes and silently
  dropped on a 3-D one — the common axes, for 106 of the catalogue's ODEs.
  (`_apply_3d_scale_and_ticks` now applies them for all three; `Axes3D` takes
  `set_zscale`/`set_zticks`/`ax.zaxis` exactly as the flat axes takes their x/y
  counterparts.) Scales and ticks are applied **before** the limits, because
  setting a log scale resets the view interval.
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
  `.venv/bin/python docs/_tooling/gallery.py [--only NAME] [--force]`; it prints
  the uncurated transforms at the end, so a newly registered one is visible the
  moment it lands. As of v6 that list is **empty**: 99 cells over 38 transforms,
  every one curated.

### The rest of the seam

- **`Plot` (`viz/spec.py`) — one type, no facade.** What was `PlotSpec` is now
  **`Plot`**: the *same class object* (`PlotSpec` stays bound in `viz/spec.py` as
  an alias, so the 557 in-tree annotations and `isinstance` checks are untouched;
  it is in no `__all__` and no `dir()`, and `ts.viz.PlotSpec` answers with the new
  spelling). It is still a JSON-serializable description of a plot — a semantic
  `PlotKind`, drawable `Layer`s, typed `Axis`/`Colorbar`/`Legend`,
  `to_dict`/`from_dict`/`to_json` round-trip — and the `PlotKind` enum remains a
  **frozen, reviewed contract** (governance gate `tests/test_viz_vocab.py`).
  - **The escape hatch is the point: `p.fig` / `p.ax` / `p.axes`.** Rendered once
    through matplotlib and cached; `.ax` on a composite raises naming `.axes`;
    `.fig` on an animated plot draws the final frame and warns. Escalating from
    easy to expert is **one dot**, never a type change and never a rewrite.
  - **Cache invalidation is attached at the definition site, not tabulated.** The
    `_mutates` decorator wraps every mutate-and-return-self method and drops the
    figure cache first, so a cached figure can never disagree with the plot it
    came from. Gate
    `tests/test_viz_spec.py::test_every_mutating_plot_method_drops_the_figure_cache`
    **derives** the required set from the source (any public method returning
    `Plot`/`PlotSpec`/**`Self`** — `add` is the `Self` one an annotation filter
    would miss) and fails on a new tweak that forgets.
  - **Handing the figure out then mutating warns exactly once**
    (`VisualizationDegraded`): re-rendering would discard hand edits. **Rule:
    library tweaks first, matplotlib last.**
  - **Selection:** `p[0]` / `p["psd"]` return a `Plot` (so they chain);
    `p.panels` stays the *list*. On a single-panel plot `p[0] is p`.
  - **`p.style(*which, **keys)`** addresses layers by producing transform or by
    legend label (`p.style("nullclines", color="white")`) — the provenance stamp
    `Layer.transform` was written by every primitive and read by nothing. An
    unmatched name raises, naming what the plot does have.
  - **Annotation verbs** `.vline/.hline/.span/.text` take plain Python and
    normalise `**style` through `normalize_style`; `Annotation.from_mapping`
    coerces a plain dict (the measured `AttributeError: 'dict' object has no
    attribute 'style'`). `Annotation` is **not** renamed and the JSON envelope is
    unchanged.
  - **`Plot.grid` → `Plot.gridlines`** (the panel arranger is the module-level
    `ts.viz.grid`; one attribute cannot mean two things). `p.grid` raises an
    `AttributeError` naming `gridlines` — the error *is* the migration guide.
  - **`.clock(fmt=)` is validated at the call.** `{t}` / `{i}` / a bare `{}`
    (normalised to `{t}`); a positional field such as `"t={:.1f}"` used to be
    accepted and raise a raw `IndexError` inside `.save()` hundreds of frames later.
  - **The repr says what you hold and what to do next**, names the producing
    transforms of an overlay (≤ 4 distinct), names a composite's arrangement, and
    stays extension-aware (`.save('f.png')` → `.save('f.gif')` when animated).
    `_repr_html_` is the no-backend notebook fallback.
  - **`FIGURE_KEYS` (17 names) is the one figure vocabulary**, derived as
    `frozenset(_INLINE_TWEAKS) | _COLORIZE_TWEAKS | {"theme"}`, with
    `split_figure_keywords` / `apply_figure_keywords` the one peeler and one
    applier. Measured pre-v6: **12 of 17 raised at `ts.plot(...)` and all 17
    worked at `traj.plot(...)`**, because the front door carried its own five-name
    copy. Every door peels these (and `style.style_names()`) **before** the
    remainder is treated as something to compute, so an integration typo is still
    reported as an integration typo.
  - **`ts.viz.spec` is the IR sub-namespace** — exactly 18 nouns
    (`Animation Annotation Axis Colorbar Frame FrameSpace Geometry Layer Layout
    Legend Part PlotKind PlotTransform Presentation RenderResult
    RendererCapabilities T make_frame`). v6 round 9 dropped the three envelope
    names (`SCHEMA_VERSION` / `to_dict_envelope` / `from_dict_envelope` — still
    bound; the round trip a user drives is `Plot.to_dict`/`to_json` +
    `ts.viz.load`) and promoted the two a **third-party renderer** cannot declare
    itself without (`RendererCapabilities` / `RenderResult`, measured
    `AttributeError` at both public addresses before). Ten are owned by sibling
    modules that import `spec.py`, so they are re-exported through a module
    `__getattr__` (`_LAZY_IR_NAMES`) — no cycle, and `import tsdynamics` still
    pulls in no plotting library. `Plot` is deliberately **not** here: it is the
    one type you annotate, and it lives one level up at `ts.viz.Plot`.
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
  - **`ts.viz.renderers`** is the backend registry in the shared four-verb shape
    (`register`/`names`/`find`/`get`), and **every verb registers the in-tree
    backends first**: measured pre-v6, `registry.renderers.names()` was `[]` in a
    fresh session and the full list once something had drawn, so any
    introspection before the first plot lied. `find(writes=".svg")` /
    `find(kind=…)` / `find(supports_3d=True)`.
  - **`writes` splits into `writes_static` / `writes_animated`**, and `Plot.save`
    consults each backend's declaration and **nothing else** (`_writers_for`);
    the hardcoded `_IMAGE_EXT`/`_MOVIE_EXT`/`_WRITABLE_EXT` table in `spec.py` is
    gone. It contradicted the backends in both directions — measured, `.webp` was
    *declared* by matplotlib and refused, `.pgf` was *not* declared and accepted,
    and matplotlib declared 17 extensions of which `savefig` cannot write five
    (`.apng .m4v .mov .webm .mp4`) — while a registered third-party backend could
    be rendered by name and its declared extension **never** saved. Consequences:
    `.mp4` on a **static** plot is now a typed error (*"'.mp4' is a movie format
    and this Plot is not animated. Add animate=True at the door, or call
    .animate() here."*) instead of matplotlib's raw `ValueError`; an unknown
    extension names every writable one and who writes it; a wrong `backend=` names
    the ones that can. `_writes_its_own_file` picks the "hand it `path=`" route
    from `data_export`/`web_export` + an accepted `path` keyword, not from a list
    of backend names.
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
  - **`ts.viz.styles`** is the public listing of that vocabulary (`names()` /
    `get("lw")` → the canonical `StyleKey` / `find(honored_by="threejs")` /
    `repr` prints the table). It is the answer to *"what can I pass, and will
    this backend draw it?"*, and it is the **same** vocabulary at every plotting
    door.
  - **`Theme`** (a **frozen** dataclass: palette, background, foreground, font,
    grid, line/marker defaults) + the **`THEMES`** registry with four built-ins
    (`default`/`dark`/`minimal`/`publication`) and a single mutable global default
    (the **only** mutable viz global — `tests/conftest.py` has an autouse fixture
    snapshotting+restoring it around every test). A `Plot` carries a private
    `_theme`; renderers read `spec.resolved_theme` (the plot's theme, else the
    global default) and apply it first (palette colours unstyled layers), then
    per-layer style overrides it.
  - **`ts.viz.themes` is a registry object**, in the same four-verb shape
    (`register`/`names`/`find`/`get`) plus `use(name)`; it is still **callable**
    (`ts.viz.themes()` → the name list), so the pre-v6 spelling keeps working.
    `themes.register(name, theme=None, /, **fields)` takes **keywords** and
    optionally a base to derive from — which removes the only reason `Theme` was
    ever exported (corollary C1). Measured pre-v6: `register_theme({...})` gave
    `AttributeError: 'dict' object has no attribute 'name'` and `set_theme({...})`
    a raw `TypeError: cannot use 'dict' as a dict key`; `Theme.replace` (taught by
    the design docs) never existed. `themes.use` raises `InvalidParameterError`
    naming `themes.names()`. `register_theme`/`set_theme`/`get_theme` stay bound
    and importable, off the listing.
  - **Fluent tweaks** (all mutate-and-return-self, so they chain and render
    identically on every backend): `.style(*which, **keys)`, `.recolor(*colors)`,
    `.theme(name|Theme, **overrides)` (a setter; `theme` is positional-only),
    `.palette(...)`, `.gridlines(...)`, `.font(...)`, `.background(...)`,
    `.size(...)`, `.vline/.hline/.span/.text`, alongside
    `.relabel/.rescale/.limits/.ticks/.colorize/.animate/…`.
    (Full guide: `docs/visualization/styling.md`.)
- **Single-panel front door:** `traj.plot(...)` / `system.plot(...)` (see the
  `Trajectory` section) builds **one panel**.
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
    `rows`/`cols`/`share_x`/`share_y`). Input type == output type == `PlotSpec`,
    so composition is fully recursive.
  - **The layout ALGEBRA nests (v6 round 11).** `a | b` is a row, `a / b` is a
    column, and **a chain of the same operator stays flat while a group carrying
    the other one nests** — so `a | b | c` is one row of three, and
    `(plot(a) | plot(b)) / plot(c)` is a row of two with `c` spanning the width
    below, which is the picture the parentheses describe. It used to *flatten
    every* nesting: measured, `(a|b)/c` gave `panels=3, mode='stack'` — three
    stacked rows — and `a/(b|c)`, `(a|b)/(c|d)` and `(a/b)|c` all lost their
    structure the same way. Rendering is a **nested matplotlib GridSpec**; the
    JSON round trip already carried nesting. A nested group's untitled state used
    to read as an absence, so `(portrait | series) / psd` was suptitled `"psd"` —
    the whole page labelled with the name of one of its three panels — and the
    repr said "2 panels" when there were three.
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
    when a panel uses a kind it declines. Plotly declines `COMPOSITE`
    *animations* (`viz/render/plotly/_anim.py` is single-panel) **and nested
    composites** (`make_subplots` is one flat grid): a nested tree drew a
    genuinely wrong picture there — measured, `(a|b)/c` came out as a two-cell
    figure carrying **one** of the three curves — so it now declines and falls
    back to matplotlib with the usual `VisualizationDegraded`.
- **Writing a movie blits (v6, `viz/render/mpl/_anim.py`).** The frame drivers
  always *mutated* artists rather than rebuilding them, but **writing** a frame
  still cost a whole figure: matplotlib's `Animation.save` draws through
  `draw_idle` and discards it, `print_figure` draws again to settle the layout
  engine, and `savefig` draws a third time — each re-solving constrained layout
  and re-measuring every tick label. Three changes, all confined to `save`:
  `_FastAnimation._post_draw` skips the discarded draw; **`_LayoutFreeze`**
  converges the layout solver once and then drops it; and **`_FrameCompositor`**
  replaces the figure's `draw` with a cached background raster plus the handful of
  artists the drivers touch.
  - **It is O(n) → O(1) in whole-figure draws**, which is the claim that cannot
    flake: measured with `Figure.draw` counted, the old path did **3.00 draws per
    frame** at every length (20 → 60, 60 → 180) while the new one does a
    **constant 7 (2-D) / 17 (3-D)** for the whole movie, 20 frames or 180. That is
    why the factor *grows* with movie length.
  - **The headline, at the default movie length** (360 frames, `.mp4`, min of 2,
    spread ≤ 1.04×): a Lorenz comet is **30.16 s → 0.91 s**, i.e. **11.9 → 396.0
    fps (33×)**; a time series **45×**, a 3-D comet **23×**, a clock **20×**, a
    spatial field **13×**.
  - **Measured** — `benchmarks/animation_bench.py`, 60-frame `.mp4`, warm-up
    discarded, min of 3, interleaved, spread ≤ 1.05×, **all nine kinds**:

    | kind | before (fps) | after (fps) | speedup |
    |---|---|---|---|
    | time series | 8.8 | 160.4 | **18.3×** |
    | 2-D comet | 12.0 | 184.6 | **15.4×** |
    | clock | 11.2 | 133.9 | **12.0×** |
    | spatial field | 11.0 | 113.9 | **10.4×** |
    | composite (lockstep) | 6.1 | 50.1 | **8.2×** |
    | 3-D comet | 11.2 | 79.9 | **7.2×** |
    | fading comet | 10.6 | 74.2 | **7.0×** |
    | spinning camera *(exempt)* | 16.2 | 27.4 | **1.7×** |
    | `layout="frames"` *(exempt)* | 49.9 | 81.0 | **1.6×** |

    The two exempt kinds still gain from the other two savings. Longer movies
    amortise the fixed calibration further — which is why every 360-frame row
    above is roughly twice its 60-frame one. *Time the benchmark the way it times
    itself*: a first run in a fresh process pays the font cache, the first Agg
    render and the first `ffmpeg` spawn (measured 11.39 s vs a 6.95 s steady
    state), so a single cold-vs-warm pair over-reports by 2-3×. The table prints a
    `spread` column (max/min within an arm) — a row far above 1 was measured on a
    busy machine and is not evidence.
  - **Correctness is measured, not argued.** The compositor renders probe frames
    the unoptimised way and **bit-compares** them; any difference disables it for
    that save. `tests/test_viz_anim_fast.py` pins byte-identity on nine kinds
    through both writer paths, pins the draw count, and pins that **every eligible
    kind actually blits** — falling back safely is the safety net, not the goal.
    `TSDYNAMICS_NO_BLIT` is the bypass that proves WITH == WITHOUT.
  - **Two kinds are exempt by construction.** A **spinning** 3-D camera re-draws
    the axes, so it is never armed — but it still gets the layout freeze, and that
    is a *fix*: a spinning axes re-measures its tick labels every frame, so
    constrained layout never converged and the axes walked through **11 distinct
    rectangles over 20 frames**. A `layout="frames"` composite gets neither (it
    calls `fig.clear()` per frame, so there is no static prefix and no single
    framing).
  - Three defects the blitting work surfaced, all fixed here: **annotations were
    silently dropped from every animated render** (the reveal renderer called
    neither annotation applier, so a `.vline()` on a movie drew nothing while the
    same spec drew it as a still); the **fading-comet driver was not a pure
    function of the frame index** (frame 0 kept the previous frame's comet, which
    is what a `pingpong` loop replays every cycle); and a **composite ignored its
    own clock** (`animate=` stamps a default `Animation` on each panel, and the
    renderer read that in preference to the composite's, so `.animate(duration=2)`
    wrote the 360-frame default — `_lockstep` now takes fps/duration/n_frames/
    loop/pingpong from the master and leaves head/trail/spin to the panel).
- **Every `Animation` knob is validated, and a refusal is a NO-OP (v6 round 11).**
  One `Animation.validate()` serves every door — the dataclass, the
  `animate={...}` dict, `from_dict`, and the four chainable tweaks — and every
  tweak writes through **`Animation.updated(**changes)`**, which builds the
  candidate with `dataclasses.replace` (so `__post_init__` validates the *copy*)
  and assigns only on success. The signatures had DECLARED their vocabularies as
  `Literal`s since v6 and nothing checked them at runtime, so
  `trail(length=('whatever', 3.0))`, `animate(mode='typo')`,
  `head(symbol='banana')`, `fps=-5`, `n_frames=0`, `camera(spin='fast')`,
  `backdrop_alpha=3`, `loop='yes'` and `trail(fade='lots')` were all accepted and
  quietly did something else; a **switch** takes `True`/`False` only (`bool(x)`
  accepts every object, which is what made the truthy string look valid beside
  eight strict knobs) and a **number** is refused by name rather than escaping as
  `ValueError: could not convert string to float: '2s'`. The *ordering* was the
  subtle half: the tweaks assigned first and validated after, so
  `p.animate(fps=-5)` raised **and stored `-5`**, and every later tweak on that
  plot re-raised the old error naming a knob the caller had not touched —
  unusable plot, misleading message, in the interactive session where these knobs
  are tuned. `camera` and `clock` read every argument before writing any.
  Gates: `tests/test_viz_frictions.py::test_a_refused_animation_value_is_never_written`
  and its siblings (15 knobs, parametrized).
- **Animation — an orthogonal modifier (`viz/spec.py::Animation`,
  `PlotSpec.animation`):** any spec of any `PlotKind` (single-panel or composite)
  becomes a movie by carrying an `Animation`; the semantic `kind` is unchanged and
  a backend that cannot animate draws the final frame. Built via
  `ts.plot(subject, animate=True | dict | Animation)` / `ts.viz.plot(..., animate=...)`,
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
  head) — **unless its `Layout.mode` is `"frames"`**, in which case the panels are
  consecutive in *time* rather than in space and are **played one after another**
  (`mpl/_anim.py::_render_frames_movie`): the parameter-sweep movie, expressed
  entirely in the composition grammar and needing no new API
  (`ts.plot(*[ts.plot(sys.with_params(r=r), "cobweb") for r in rs],
  layout="frames", fps=15).save("cascade.mp4")`). **The renderer half is what ships
  today; the front-door spelling in that snippet still raises `unknown layout
  'frames'`** — `Layout.mode`'s literal (`viz/spec.py`) and `compose.py`'s
  `_COMPOSITE_MODES` are the two remaining halves, and until they land the mode is
  reachable only by constructing the `Layout` directly. The two were deliberately
  split: shipping the *spec* half alone would build a composite no backend can
  draw, which is worse than a clear refusal. Each frame re-draws one panel in
  full through the same static panel bodies the tiled renderer uses, so a frame is
  the picture that panel renders on its own; the axis ranges are the **union** over
  the panels (a movie whose axes rescale every frame shows you the axes moving, not
  the dynamics) and `share_color=True` unifies the colour range for the same
  reason; `n_frames`/`duration`/`pingpong` index the **panel list**; the figure is
  left showing the **last** frame so a still save (`.png`) is the final panel and
  not a blank page. Scope is **matplotlib only** (mp4/gif) — plotly declines every
  `COMPOSITE` and falls back to it, three.js exports the last panel and says so.
  **`frames`** (stream VIZ-SPATIAL-FIELD): a **spatial-field movie** — the
  field of a spatially-extended system (a method-of-lines PDE) *played over time*.
  Each frame is the field's **spatial** state at that instant, and the per-frame
  plot's shape follows the field's spatial dimensionality: a **1-D field** `u(x)`
  is a travelling-wave **line** (the profile — Kuramoto–Sivashinsky), a **2-D
  field** `u(x,y)` an `imshow` **heatmap** movie (Gray–Scott / Swift–Hohenberg).
  ONE semantic kind covers both — the new `SPATIAL_FIELD` `PlotKind` (the
  renderer dispatches on the field's spatial ndim, like the `__plot_spec__` seam
  auto-dispatches on component count). The producer
  (`viz/producers.py::spatial_field`) stacks every per-time snapshot on the
  layer's `"frames"` channel (shape `(T, *spatial)`) and keeps the **final** field
  as the static layer data (`z` for 2-D / `y` for 1-D), so a still save / a backend
  that can't animate draws the final field; the mpl renderer plays the stack frame
  by frame (`viz/render/mpl/_anim.py::_field_movie_driver` → `_field_movie_2d` /
  `_field_movie_1d`), consecutive frames carrying genuinely different data. **Front
  door:** `ts.plot(system, "spatial_field", animate=True)` — at `ts.plot` a picture
  is named POSITIONALLY, by its transform; the method doors (`traj.plot` /
  `system.plot`) take `kind="field"`, which routes via `_KIND_ALIASES`.
  The spatial layout comes from the **system**, via the
  optional `_field_shape: tuple[int, ...]` ClassVar (recorded onto
  `traj.meta["field_shape"]` at integration time, so a bare `Trajectory` carries
  it). No `shape` kwarg: a system with no `_field_shape` (or a 1-D one) is a 1-D
  profile (honest — never guesses a 2-D grid). A multi-block field state declares
  `field_labels` (e.g. Gray–Scott's `("u", "v")`); `components="u"|"v"` picks the
  block, defaulting to the **last** (the activator). **`mode="frames"` is resolved
  from the DATA, not from the door** (v6, `mpl/_anim.py::_play_the_field_stack`):
  a `"frames"` channel exists only because the producer stacked the per-time
  snapshots so they could be played, so an animated `SPATIAL_FIELD` carrying one
  plays it whichever door built the spec. Only the `kind="field"`
  recipe forced the mode, so the transform spelling — `ts.plot(traj,
  "spatial_field", animate=True)`, the one the front door and the v6 docs use —
  animated in `reveal` mode and swept a ruler across the **final** field: measured
  on a Swift–Hohenberg lattice whose stack is `(101, 8, 8)`, 8 frames (the lattice
  *width*, read as a sample axis) instead of the requested 12. The field movie is
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
  **`fps` reaches both HTML exports** (v6): neither browser loop has a frame clock
  — each traverses the series in `duration` seconds at the browser's ~60 Hz — and
  both used a hard-coded `12.0` whenever `duration` was unset, so `.animate(fps=60)`
  was honored by matplotlib and **dropped in silence** by the web exports, with no
  `caps` gap declared for it. `Animation` already relates the two
  (`frame_count = round(duration * fps)`), so both now invert it:
  `plotly/_anim.py::playback_seconds` and `threejs/_lower.py::_playback_seconds`
  (deliberate twins in two backend packages, guarded by
  `test_both_html_exporters_derive_the_same_playback_duration`; the single natural
  home is a method on `Animation`). At the defaults — `fps=30`,
  `Animation.DEFAULT_FRAMES=360` — the derived value is **12.0 s exactly**, so no
  existing export changed speed; measured in a browser, `fps=60` vs `fps=10` now
  advance the comet 8x apart (plotly `STRIDE` 8 vs 1).
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
  An **animated COMPOSITE** is the same rule one level up (v6,
  `_degrade_animated_composite`): the loader reveals one draw range in one scene,
  so a multi-panel movie has no three.js form — `_lower_composite` never read the
  top-level `Animation` at all, so `animate=True` used to vanish between the call
  and the file. It now exports the composite **statically** with ONE
  `VisualizationDegraded` naming the drop and the `.mp4`/`.gif` way out; a
  `layout="frames"` composite is the sharper case (its panels are frames, so tiling
  them in space would be a different picture) and exports the **last** panel.
  The page (`_page.py`) is self-contained — payload + loader inlined, the only
  external reference the pinned three.js import map, with a matplotlib poster +
  `<noscript>` fallback — and was verified loading and animating in a real browser
  (WebGL canvas drawn, progress readout advancing, no console errors), for a
  single-panel comet and a composite alike.
  `PlotSpec.save` picks the
  backend by extension (animated: `.html`→plotly, `.mp4`/`.gif`→matplotlib) and
  takes `fps`/`dpi`/`size`. The `Animation` directive round-trips through
  `to_dict`/`from_dict` and adds no `PlotKind` (the frozen vocab is untouched).

---

## Code conventions

- **Visibility (v6 round 9, CONTRACT §11).** *A name is public if and only if a
  user TYPES it in ordinary work, **or** it is the only route to a capability the
  owner has protected* — composition, custom transforms, custom primitives,
  grids, animation, the matplotlib escape hatch, writing your own
  system/analysis/solver/renderer, the six plugin entry points. Two mechanical
  rules follow, and both are gated:
  - **Hiding is a DISCOVERY change, never a REACHABILITY change.** Every hide is
    a `__dir__` / `__all__` edit. The name stays bound, importable, callable and
    tested; only the tab surface shrinks. `families/_hidden.py::hide` and
    `viz/_visibility.py` are the two mechanisms — **never** an underscore rename,
    which is what would cost capability.
  - **Every REACHABLE module declares `__all__` AND defines `__dir__` returning
    `sorted(__all__)`.** `__all__` governs only `import *`; it has zero effect on
    TAB. Before this rule 44 modules leaked 329 non-API names, mostly their own
    imports (symengine's `sin`/`cos`/`exp` looked like library helpers on 38
    catalogue modules). The rule is stated for *reachable* modules — one whose
    own path carries a `_` component is exempt, because nobody tab-completes into
    it, and counting those was how an earlier measurement over-reported the
    remaining work 34/126 when the user-visible figure was 18/50.
    **The 16 catalogue modules were the last holdouts**: 15 of them declared no
    `__all__` at all, so `chaotic_attractors` offered `ClassVar`, `exp` and `np`
    beside its 50 systems. Each now lists the classes it defines (175 in total)
    and carries the `__dir__`; reachable leaks are **0**.
    Gates (`tests/test_namespace_curation.py`):
    `test_no_reachable_module_offers_a_borrowed_name` sweeps every reachable
    module (a `X | Y` union reports `__module__ == "typing"`, so a name the
    module declares in its own `__all__` is exempt — that is what `ts.data.Region`
    and `engine.problem.Problem` are), and
    `test_every_catalogue_module_lists_the_classes_it_defines` keeps `__all__`
    complete, since a `__dir__` over a partial `__all__` would *hide a system*
    from its own module. The older backlog table stays split "no new offender"
    (fast tier) / "delete the stale row" (`full`), so one slot's correct fix
    cannot redden everyone else.
  - **Corollary, learned the hard way:** a name a library *prints back* must be
    a name the reader can then tab-complete. `Geometry.parts` was hidden while
    two of `Geometry`'s own errors ended *"iterate g.parts"*; it is listed again,
    and `tests/test_viz_visibility.py` now derives the requirement from the
    source rather than asserting a literal list.
  - **The corollary cost 40 names back on the result layer, and is now TWO
    GATES** (`tests/test_result_visibility.py::TestAPrintedNameIsACompletableName`).
    The same sweep that hid 78 result names also hid, or never had, names the
    library itself hands the reader. Restoring them took the layer 452 → **492**,
    purely additively — no listing lost a name.
    - *A name the repr prints must be listed* (+32). `RQAResult` printed
      `DET = 0.999 · LAM = 0.993 · L_max = 678 · ENTR = 3.726` and tab-completed
      **none** of the four (all four resolved); `DimensionResult` printed
      `15 fit pts, R² = 0.99992` while listing only `r_squared`, the half you
      cannot judge a fit by alone. The mechanism is
      `AnalysisResult._extra_attribute_names` — names served by a subclass's own
      `__getattr__`, which Python cannot see, so they must be *declared* to be
      listed. It existed already but fed only the wrong-guess suggester, not
      `__dir__`: that half-wiring is what let the defect through. The gate is
      scoped to printed names the result **carries**, because a repr also writes
      the maths symbol for its quantity (`D_corr`, `D_KY`, `H0`, `GALI_2`, `K`)
      and the swept parameter's name (`r`, `delta`) — those are not attributes
      and requiring them would mean naming every quantity twice.
    - *A name `to_dict(full=True)` promises must exist* (+8). `to_dict` is the
      door every result's `AttributeError` recommends, so a key is a claim — and
      three were false, each a **casing** nobody would guess: `BasinEntropy`
      emitted `Sb`/`Sbb` (Daza's, and what its own repr prints) carrying only
      `sb`/`sbb`; `UncertaintyExponent` emitted `D0` (Grebogi's) carrying
      `boundary_dimension`; `LyapunovSpectrum` emitted `zero_tolerance` — the
      threshold its chaos verdict turns on — from a *private* property. This
      gate needs no judgement about which printed symbols are names: `to_dict`
      already decided. One documented exception, `unit` (provenance, reachable
      as `meta["unit"]`), and the table must not grow.

    The nine RQA abbreviations now have **one** source of truth
    (`RQAResult._PRINTED_ABBREVIATIONS`, spelled as the repr prints them); the
    case-insensitive lookup map, the listing, and `WindowedRQA`'s copy are all
    derived from it, so the two RQA results cannot drift into answering to
    different names for one quantity.
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
  is unavailable). These are wired across the families and `_engine/run.py`
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
- a touched **documentation** path → the tests that *execute* it: `docs/**`,
  `README.md`, `CLAUDE.md` and `mkdocs.yml` → `test_doctests.py`, and
  `docs/_tooling/**` → that plus `test_docs_figures_golden.py` /
  `test_viz_gallery.py` / `test_catalogue_dynamics.py` (see the doctest-gate
  section below — documentation is executable, so it is not ignorable);
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

- **The API contract gate — `tests/test_api_contract.py`.** The single file that
  pins the v6 public surface, so `planning/api-v6/CONTRACT.md` §2's "every listing
  is a contract" has teeth. Six invariants, each the inverse of a way the surface
  eroded: the **listings** (`ts.__all__` = 11; `system.<TAB>` = the 20-name core
  minus each family's declared absences; `ts.analysis` = 52 (49 analyses + `find`
  + `register` + `results`); `ts.viz` = 14;
  `dir(M) == sorted(M.__all__)` over the 14 declared public packages); **one
  spelling** (no two exported names are the same object; no submodule shadowed by
  a same-named function — `import tsdynamics.analysis.recurrence.rqa as m` must
  bind the *module*); **plain Python** (every live `region=` / `plane=` /
  `components=` door is *called* with tuples and strings, and a guard fails when a
  new door is not in the call table); **legible results** (all 32 result reprs
  state a number, a count or a verdict); **taught removals** (every redirect and
  every demoted name hands back a spelling that RESOLVES, including through
  `from tsdynamics import X`); and **real signatures** (each family's `run` binds
  exactly §3.1's why-table; no public callable hides behind `(*args, **kwargs)`;
  no return annotation names a deleted type). Contract items another slot still
  owns are tracked either as a `strict` xfail naming that slot or — where the
  defect is a population that must *shrink* — as a declared table asserted by set
  equality, so fixing one fails until its row is deleted.
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
  - Gated today: **32 modules + 42 pages** (was 15 + 20; the page count includes
    `tutorials/seven-tasks.md`, the cold-start tutorial).
  - A fence that is a signature listing, a calling pattern, or a deliberate
    demonstration of what *raises* opts out **in place** with `# skip-doctest`
    — visible to the page's reader, unlike a name in a list elsewhere.
  - **The gate fires on docs-only PRs (v6; the known gap is CLOSED).**
    `tests/_changed_select.py` used to list `docs/`, `*.md`, `README.md` and
    `mkdocs.yml` under `_IGNORE_PREFIXES`/`_IGNORE_SUFFIXES`/`_IGNORE_FILES`
    ("no bearing on the test suite"), so a diff touching only documentation
    selected three cheap registry guards and **not** `test_doctests.py`.
    Documentation is now classified by `_docs_gate_tests`, checked **before**
    the ignore table (which is keyed on the `.md` suffix and the `docs/`
    prefix): `docs/**` selects `test_doctests.py`; `README.md` / `CLAUDE.md` /
    `mkdocs.yml` select it too (their catalogue counts are checked against the
    live registry and the `exclude_docs` block is parsed); and
    `docs/_tooling/**` — *code the suite imports* — additionally selects
    `test_docs_figures_golden.py`, `test_viz_gallery.py` and
    `test_catalogue_dynamics.py`. `planning/`, `.claude/`, `CHANGELOG.md` and
    `CONTRIBUTING.md` stay ignored, because no test reads them. Guarded by
    `tests/test_changed_select.py` (`test_a_docs_page_selects_the_gate_that_executes_it`,
    `test_the_repo_root_files_the_doctest_gate_reads_are_not_ignored`,
    `test_docs_tooling_selects_the_tests_that_import_it`,
    `test_no_file_is_both_docs_gated_and_ignored`).
- **`mkdocs.yml` publishes every page except a short enumerated list.** The
  blanket tree-drop `theory/` had removed ~1.2k lines of accurate documentation
  from the site — including `theory/fixed-points-interval.md`, which a **shipping
  source docstring** (`analysis/fixedpoints/_interval.py`) links to — and nothing
  failed. Only the three pre-generation `systems/*/index.md` stubs (replaced by
  `hooks/docs_autogen.py`) and the non-page asset dirs are excluded now;
  `tests/test_doctests.py` fails if that set grows or if any content tree is
  dropped wholesale. `validation.unrecognized_links` is **`warn`** (was `ignore`),
  so a broken internal link fails `mkdocs build --strict`.

- **The 177 generated system pages are gated at build time** (v6,
  `hooks/docs_autogen.py`). They are the library's largest single body of example
  code and **nothing executes them**, so a renamed verb ships silently on all 177
  at once — which is exactly what happened: the "Define it" block was emitting
  `sys.integrate(...)` / `sys.iterate(steps=…)` / `sys.lyapunov_spectrum()` /
  `ts.kaplan_yorke_dimension(...)` / `n_exp=`, five spellings v6 removed. Two
  build-time assertions close it, and `--strict` turns either into a failure
  naming the system:
  - `_assert_define_block_uses_the_live_api` refuses a block containing any
    entry of `_DEAD_API_SPELLINGS`. The block now reads
    `system.run(...)` + `ts.plot(traj)` + `ts.analysis.<name>(system)`.
  - `_assert_doi_is_rendered` refuses a page whose record carries a `doi` but
    whose reference card has no `doi.org` link. **155 of the 177 built-ins carry
    a DOI**, and until `SystemEntry` gained the field every one of those links
    was dropped with no symptom. Measured after the fix: `155` pages under
    `site/systems/` contain a `doi.org` link, `0` contain a removed verb.

- **The cold-start tutorial is `docs/tutorials/seven-tasks.md`** — simulate ·
  Lyapunov + is-it-chaotic · bifurcation diagram of a map *and* of a flow ·
  fixed points + stability · basins · animated phase portrait over a vector
  field · define your own ODE. Every fence is executed by the doctest gate, so
  it cannot rot. It is the page `docs/start/index.md` and
  `docs/tutorials/index.md` both point at first.

- **The gallery is fully curated**: 99 cells over 38 transforms, every one with
  a hand-written `SHOWCASE` subject and caption (the fallback path — an
  uncurated transform drawn on its `example` factory — is exercised by nothing
  in-tree, deliberately). Regenerate with
  `.venv/bin/python docs/_tooling/gallery.py [--only NAME] [--force]`; it prints
  any uncurated transform so a newly registered one is visible immediately.

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
- **In-process lowered-tape cache (stream PERF-LOWER-CACHE, `_engine/compile.py`):**
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
  outside the lock. Surface: `tsdynamics._engine.run.jit_cache_stats()` /
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
| An estimator answers confidently on an orbit that blew up | It doesn't, since round 8: a dimension or `lyapunov_from_data` taken from data whose magnitude escaped warns `RunawayOrbitWarning`, reports `trusted = False`, and says so in the repr. The number is still returned. The check reads the **coerced data**, so `traj["x"][::10]` is caught too. |
| `batch.final` has NaN rows | `batch.diverged` is the mask, and the repr says `⚠ 5 of 12 diverged`. An average over `.final` is an average over the survivors — `np.nanmean` on an unflagged subsample is the trap this closes. |
| Reading `meta["dt"]` off a sliced trajectory | Correct since round 8 — row selection re-derives `dt`/`t0`/`ic` from the rows in hand, and *drops* `dt` when the slice has no uniform step. `traj.dt` remains the reading the library itself uses. |
| `basin_fractions(n=…)` / `continuation(n=…)` | **`n_seeds=`** since round 8 — the word every sibling already used. A typo in the `**fsm` passthrough is answered by name, never by naming `_AttractorMapper`. |
| `ts.plot(a, b, label=[...])` | The word is `labels=` (one per subject) and every plotting door now suggests it. It used to answer `did you mean zlabel=?` — an *axis* name — at the one door whose pool omitted it. |
| `(plot(a) \| plot(b)) / plot(c)` drew three stacked rows | Fixed in round 11: the algebra **nests**. A chain of one operator stays flat (`a\|b\|c` is one row of three); a group carrying the other one nests, so that expression is a row of two with `c` spanning the width below. Nesting is matplotlib-only — plotly declines and falls back with a warning, because `make_subplots` is one flat grid and it drew a genuinely wrong picture. |
| A rejected animation value keeps coming back | It cannot since round 11: `Animation.updated()` validates a **copy** and assigns only on success, so `p.animate(fps=-5)` raises and leaves the plot untouched. It used to store the bad value, and every later tweak re-raised the *old* error naming a knob you had not touched. |
| `p.animate(loop="yes")` / `p.trail(fade="lots")` were accepted | Refused by name now — a switch is `True`/`False`. `bool(x)` accepts every object, which is what let a truthy string look valid next to eight strict knobs. |
| `help(traj.plot.psd)` shows `(*things, layout, rows, cols, …)` | Fixed: every `subject.plot.<name>` entry carries the transform's own `__name__` / `__doc__` / `__signature__`, so `help` prints that plot's real knobs. The options were always there; nothing surfaced them. |
| `ts.plot(lorenz, "trace_determinant")` labels its axes `tr J` / `det J` | Only for a **planar** flow. Above two coordinates the plane classifies a 2x2 *block*, so the axes read `tr J\|xy` and the call warns once. Measured: Lorenz's origin plots at `(-11, -270)` where the full 3x3 `J` has `(-13.667, +720)` — a different number and a different SIGN on the determinant. The verdict (*saddle*) is right; the numbers are the slice's. `plane=` picks the block; the full spectrum is `ts.analysis.fixed_points(system).eigenvalues`. |
| You want the vector field itself, to check a picture | `system.rhs(u, t=0.0)` — public since round 11, next to the `jacobian` that is its derivative. `_rhs_numeric` is still the fast pre-bound callable the internals use. |
| `for name, array in part.items()` on a `ts.viz.geometry` part | Works — `Part` reads like the mapping it is (`keys` / `values` / `items` / `in` / `len` / `dict(part)`), and its repr names `part['x']` as the way in. |
| `ts.viz.transforms.allow(mine, "stem")` seemed to do nothing | Fixed: on a transform that declares no `primitives=`, the inferred row now **widens** what `allow` wrote instead of replacing it. Since `allow` is the only route from a custom primitive to any transform, that made both extension doors unusable together. |
| A field over a big box with the axes zoomed in | Pass both: `ts.plot(sys, "vector_field", domain=[(-6,6),(-6,6)], xlim=(-3,3))`. `domain=` is the evaluation window, `xlim=` only the axes; naming `domain=` sends a *shared* `xlim=` back to its figure job. Both inside one transform's own option dict is still the genuine clash and still raises. |
| `ts.plot(map, "cobweb")` is a solid block | Fixed: a cobweb draws `_COBWEB_DEFAULT_STEPS` (50) orbit steps, not the map's whole 1000-step default run, so the parabola and the diagonal it is read against are visible. `steps=` picks another count; `steps=0` draws the lot. |
| `_equations` writes `y[0]` **or** `x, y, z = u` | The ODE/SDE state is an **accessor**, not a vector: read it by CALLING it (`u(0)`, `u(1)`, …), which is also what makes a DDE's `u(0, t - tau)` expressible. A `DiscreteMap._step` **does** take a plain vector — that is the one place the families differ, and it is why a map author writes the unpack in an ODE. Both spellings are diagnosed by name, with the corrected line echoed back (`_subscripted_accessor_hint`). |
| Variable-dim system without `_structural_params` | Lowering-time `range(N)` fails. Add `_structural_params = frozenset({"N"})`. |
| Map params order ≠ `_step` signature order | **Raises `TypeError` at import**. |
| DDE with constant past at a fixed point | Lyapunov exponents ≈ 0. Provide a non-equilibrium `history`. |
| Tight tolerances on DDE | `rtol=atol=1e-3` is the DDE default and the right start — **not** because tightening stalls the solver (measured: all 6 built-in DDEs complete at `1e-12`/`1e-15`, T=500) but because the method of steps lands on every sample, so `dt` bounds the step and the tolerance is inert (5 of 6 are bit-identical from `1e-3` to `1e-9`). |
| Adding a new `rtol=`/`atol=` default | Don't write a literal — name a constant in `_utils/tolerances.py`. A gate (`test_polish_standards.py::test_no_bare_tolerance_literal_in_the_library`) fails on a bare literal in any signature, call keyword or `self._rtol =` assignment. |
| "My results got less accurate in v6" | `dt` is now **sampling only** — it no longer secretly bounds the internal step (see "Dense output and `max_step`"). The default `rtol`/`atol` tightened to `1e-9`/`1e-12` to compensate, so a plain `.run()` is *more* accurate than pre-v6, not less. If you pinned `rtol=1e-6` explicitly you kept the old accuracy on a coarser step — tighten it; or pass `max_step=dt` to reproduce the old step regime; or set `TSDYNAMICS_NO_DENSE_OUTPUT=1` to reproduce pre-v6 numbers exactly. |
| An adaptive kernel strides over a narrow feature | Pass `max_step=`. (A step *size* — `max_steps` is a step *count*.) |
| A repr shows `PoincareSection(crossings=…)` / an `OrbitSet` says "period 6" | Fixed in v6: **every** result's repr IS the answer (§4.3) — the section prints `PoincareSection  300 crossings of y = 0 up   ·  3-D states   (Rossler)` and a *flow's* period renders as the real number `T = 6.66329` (a map's stays an integer count). `summary()` exists on nothing. |
| `ts.plot(traj, "psd", components="x")` used to raise | `components=` is the ONE spelling at every analysis and every transform door since v6 (M38). |
| `system.to_plot_spec(...)` | The plotting seam is the dunder `__plot_spec__` — carried by systems, `Trajectory` and all 32 results, so `ts.plot` classifies a subject with ONE predicate. The verb you type is `plot`. |
| `fixed_points(sys)[0].x` / `for fp in fps: fp.stable` | `[]` and iteration give **arrays** now (the owner's D1). Read `fps.points` / `.eigenvalues` / `.is_stable` for the vectorised answer, `fps.details[i]` for the record, `fps.by_id(k)` for a labelled one. Same four spellings on `OrbitSet`, `AttractorSet`, `WindowedRQA`. |
| `traj[["x", "z"]]` | The list spelling raises, naming `traj["x", "z"]` — which returns a `Trajectory` that names its own columns, so a second selection off it is right. |
| `result.plot.scaling()` / `.phase()` / `.section()` | The eight kind-forcing methods are gone — they relabelled the spec without redrawing it. `result.plot()` is this result's own view; `result.plot.<TAB>` lists the transforms that draw it (`result.plot.scaling_fit()`), and it takes the same style/figure vocabulary as every other plotting door. |
| `WrappedSystem(initial=…)` | `ic=`, the word every other family uses. Refused by name. |
| A docs fence calls `.show()` | `show()` warns on a windowless backend, and the doctest gate pins matplotlib to `Agg`, so a fence that displays fails under `filterwarnings=error`. Build the plot in the fence; teach `.show()` in prose or a `# skip-doctest` block. |
| A tool reads `cls.default_ic` / `cls.reference` / `cls.known_lyapunov` | Those ClassVars are underscored since v6. Read them through `registry._classvar(cls, name)`, which knows both spellings — `docs/_tooling/figures.py` named the public one and drew **every** IC-declaring system's figure from the wrong start, and four more readers in `docs/_tooling/` were found in v6.0.1 (the bibliography was generating from **0 of 177** citations). If the value is a `SystemEntry` field (`reference` / `doi` / `field_labels` / `known_lyapunov`), read it off the **entry**. |
| You add a `value in <closed set>` check on a keyword | Guard it: `if not is_hashable(value) or value not in …`. `in` hashes first, so `kind=["x"]` died naming a private table instead of the valid kinds — at six doors on v6.0.0. `_utils/lookup.py` is the one spelling. Watch the two siblings: a numpy value makes `==` elementwise, and `difflib` needs a `str`. |
| `with_params` / `copy` raises on a system that sizes itself from a parameter | Fixed in v6.0.1 for all five (`GrayScott`, `SwiftHohenberg`, `KuramotoSivashinsky`, `Lorenz96`, `MultiChua`). A custom `__init__` must **bind and discard** `dim=` / `field_shape=` — `with_params` forwards them on every rebuild — and merge its parameters through `systems/_declared_params.py::merge_declared_params`, which also refuses a parameter given twice instead of silently preferring one. |
| A map's `run(seed=…)` gave a different orbit each call | Only when the first attempt diverged: `_resolve_ic` reaches the seeded generator on just one of its branches, so a map with a declared `_default_ic` (Hénon) drew its **retries** from OS entropy. Fixed in v6.0.1; a retrying seeded run is reproducible now. Pin `ic=` (or `seed=`) whenever you time or diff — an unpinned draw compares different work, and it is what made `test_map_iterate_shape_and_finiteness[Bogdanov]` a ~1-in-1000 flake. |
| A movie renders differently from its still, or you suspect the blitting | `TSDYNAMICS_NO_BLIT=1` writes every frame the unoptimised way; if that changes the picture it is a bug, not a setting — the compositor bit-compares probe frames and disables itself on any difference. |
| A new animated artist is mutated per frame | Declare it on the `_LayerDriver` that mutates it, at the same site. An undeclared artist freezes at its first frame (it lands in the cached background); a declared one that the drivers do *not* mutate only costs a redraw. And the driver must be a pure function of the frame index — the compositor restores the frame it calibrated on, and `pingpong`/`loop` replay frame 0. |
| `set_state` on a DDE | **Does not exist** (v6) — the state is a history function; use `reinit(u)` for a constant past or `run(history=...)`. |
| `ts.Lorenz` / `ts.correlation_dimension` / `ts.Box` stopped resolving | Deliberate (C5). The top level is 11 names; everything else lives at one address, and the `MovedInV6` prints it: `ts.systems.Lorenz()` / `ts.analysis.correlation_dimension` / `ts.data.Box`. |
| Removing or renaming a public name | Add its row to `src/tsdynamics/_redirects.py` **in the same commit**, sorted by key. The table is the migration guide; a removed name with no row gets the generic near-miss answer. |
| An error message hands back `ts.<something>` | It must **resolve**. `ts.fixed_points(system)` no longer runs, so a message offering it is worse than one offering nothing. Gate: `test_polish_standards.py::test_errgate_remedy_lines_resolve`, whose must-shrink table went 5 rows → 1 in round 9 (the four `analysis/dimensions` messages now say `ts.analysis.…`). |
| A teaching `AttributeError` ends in someone else's guess | Seal it with `errors.taught(err, name)`. CPython augments any `AttributeError` escaping a `__getattr__` and prints `Did you mean: …?` from `dir(obj)` — measured, it answered `lorenz.lyapunov_spectrum` with the **private** `_lyapunov_spectrum` and `traj.dims` with `dim`, an integer. Every message builder (`_absent_name_error`, `_discovery.attribute_error`, `plot_seam_error`, `Trajectory._miss`) seals; gate: `tests/test_visibility_restorations.py`. |
| You hid a name and something still prints it | Then it is not hidden, it is broken. A remedy line, a repr and a docstring are all discovery surfaces; if one names `x.y`, `y` belongs in `dir(x)`. See CONTRACT §11 and the `Geometry.parts` row. |
| A result's repr names a quantity `dir()` will not complete | A gate fails (`test_result_visibility.py::TestAPrintedNameIsACompletableName`). If the name is served by the class's own `__getattr__`, declare it in `_extra_attribute_names` — Python cannot see such a name, so it is invisible to `dir` until declared. Otherwise drop it from `_HIDDEN_ATTRIBUTES`. `r.DET` / `r.L_max` / `w.DET` are the RQA case this closed. |
| A 1-D `DiscreteMap` you wrote crashes inside your own `_step` | Fixed in round 9: the shared tangent helper passed 1-D maps a bare `float`, so the documented `u[0]` spelling raised `'float' object is not subscriptable` at every A-FP/A-CHAOS door (`periodic_orbits`, `fixed_points`, map Lyapunov). A map kernel now always receives the `(dim,)` **vector**, every dimension. Invisible in-house because all seven built-in 1-D maps are scalar-style (`r * x * (1 - x)`), which NumPy broadcasting accepts either way. |
| `ts.EnsembleSystem` | The class is `Ensemble`; `ts.derived.Ensemble`. It has a `_redirects.py` row now — removing a public name without one is what made it the single demoted name that answered with a generic near-miss instead of its address. |
| `system.integrate(...)` / `.iterate(...)` / `.trajectory(...)` | **Gone** (v6) — `run` is the one trajectory verb; the `AttributeError` prints the replacement line. |
| `run(method="rk45")` | **`solver=`** since v6: `solver=` picks a numerical kernel, `method=` picks an *estimator* on an analysis. |
| A keyword your `run` silently ignored | It no longer can — every family's signature is closed (`families/_kwargs.py`) and the message states why that word belongs to a different family. |
| Stiff ODE: which method? | `"bdf"` is the **variable-order (1–5) BDF** and the right default for stiff ODEs (far faster than the fixed-order `rosenbrock`/`trbdf2`, which stay selectable). `run.integrate` auto-builds the Jacobian-carrying tape for the implicit kernels, so `integrate(method="bdf")` "just works". The legacy SciPy name `"LSODA"` is no longer a method — declare `_default_method = "bdf"`. Pass `method="auto"` to let `solvers.recommend` probe stiffness and pick `bdf`/`rk45` — a one-point heuristic, so prefer `_default_method` for a system known to be stiff. |
| Param change ignored by a live stepper | `reinit()` after parameter changes (or use `with_params`). |
| Orbit diagram over a DDE wrapper | Re-lowers the tape per parameter value — slow by design, document it. |
| New DDE fails `test_dde_histories_complete` | Add its history to `tests/_sampling.py`. |

---

## Quick reference

```python
import numpy as np
import tsdynamics as ts

# ODE — `run` is THE trajectory verb (integrate/iterate/trajectory are gone in v6)
lor = ts.systems.Lorenz()
traj = lor.run(final_time=100.0, dt=0.01, transient=10.0)
traj["x"]                                   # one name  → the (T,) column
traj["x", "z"]                              # two names → a Trajectory naming x, z
exps = ts.analysis.lyapunov_spectrum(lor, final_time=300.0)  # [0.91, ~0, -14.57]
exps.kaplan_yorke                           # → ~2.06

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
section = pmap.run(500)                         # → PoincareSection
strobe = ts.systems.Duffing().poincare(period=4.488)   # poincare absorbed stroboscope
sec = ts.analysis.poincare_section(ros, plane=("y", 0.0, "up"), crossings=500, seed=0)
od = ts.analysis.orbit_diagram(pmap, "c", np.linspace(2, 6, 50), points_per_value=100)
tang = ts.derived.TangentSystem(lor, k=2)       # the Lyapunov engine, steppable
band = lor.ensemble(np.random.rand(100, 3))    # → an Ensemble *system*
finals = band.run(final_time=10.0).final       # → (100, 3)

# Event detection / arbitrary stopping (scipy-shaped events=)
sol = lor.run(final_time=100, dt=0.01, events=[("z", 27.0, "up")])
sol.meta["t_events"][0]                       # times z=27 was crossed upward
stop = lambda y, t: y(0)**2 + y(1)**2 + y(2)**2 - 50.0**2  # leave a ball → stop
stop.terminal = True
lor.run(final_time=1e3, events=[stop])                    # truncates at the crossing

# Maps — the horizon word is `steps` (a count; `final_time` is refused by name)
h = ts.systems.Henon()
h.run(5000, transient=500)
fps = ts.analysis.fixed_points(h)           # analytic saddles
fps[0]                                      # the point, as a (dim,) ARRAY
fps.is_stable, fps.eigenvalues              # vectorised, one row per member
fps.details[0].continuous                   # the record, when you want one member
ts.analysis.fixed_points(ts.systems.VanDerPol(), region=[(-3, 3), (-3, 3)])  # plain bounds
ts.analysis.lyapunov_spectrum(h, k=1, ic=[0.1, 0.1])   # ≈ 0.42 — one exponent

# Analyses are FREE FUNCTIONS on their subject (ruling A2) — and they say so
ts.analysis.find(h)                         # what can I measure on THIS?  (13)
ts.analysis.find("is this chaotic")         # who answers THIS question?  (6)
print(ts.analysis.__doc__)                  # the 49, grouped by what you hold

# Regions: one (lo, hi[, n]) pair PER STATE COMPONENT, everywhere. No type to build.
ts.analysis.basins(h, [(-2, 2, 60), (-2, 2, 60)])

# Plotting: one front door, style + labels at the call site, spec in / spec out
ts.plot(traj, color="crimson", linewidth=2, title="Lorenz", theme="dark").save("l.png")
ts.plot(np.sin(np.linspace(0, 40, 2000)))              # bare arrays plot too
ts.plot(ros, ("flow_speed", {"log": True}), "streamlines")   # ("name", {opts}) pairs
ts.plot(traj_a, traj_b, "phase_portrait")              # two orbits, one figure
ts.viz.grid(p1, p2, p3, cols=2, share_color=True)      # a grid of DIFFERENT plots
ts.viz.draw({"x": r, "y": C}, "line")                  # arrays, no transform needed
ts.viz.transforms.names(); ts.viz.compatibility()      # what can this draw?

# DDE (integrate first, then Lyapunov from the end state)
mg = ts.systems.MackeyGlass()
traj = mg.run(final_time=500.0, dt=0.5, history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)])
# ...an analysis is a FREE FUNCTION (A2) — the bound method was removed in v6
exps = ts.analysis.lyapunov_spectrum(mg, k=1, dt=0.5, ic=traj.y[-1])

# Registry
from tsdynamics import registry
registry.families()                         # {'ode': 142, 'dde': 6, 'map': 26, 'sde': 3}
```
