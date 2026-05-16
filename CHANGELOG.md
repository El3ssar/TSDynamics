## Unreleased

### Features (N2.b — Rust ODE stepper, partial)

- **`crates/tsdyn-ode`**: adaptive explicit RK integrators with dense output on
  the uniform Python grid — **DP5**, **DP8** (SciPy-compatible error indicator +
  `Dop853DenseOutput`), **Tsit5** (OrdinaryDiffEq.jl tableau + interpolant),
  **BS3** / RK23 (SciPy tableau + dense `P`), plus fixed-step **RK4**.
  Butcher coefficients remain vendored with upstream attribution in
  `butcher.rs`; DP8 embedded errors `E3`/`E5` match SciPy's
  `dop853_coefficients`.
- **`erk_extra.rs`**: DP8 / Tsit5 / BS3 drivers (FSAL bookkeeping).
- **PyO3** `integrate_ode(...)` unchanged surface.
- **`ContinuousSystem.integrate`** dispatches to Rust for **`DP5`**, **`DP8`**
  (aliases **`DOP853`** / **`dop853`**), **`TSIT5`**, **`BS3`**, **`RK4`** when
  lowering succeeds; JiTCODE fallback elsewhere (**`VERN9`** driver still
  pending in Rust).
- **Golden trajectories** (`tests/native/regression/ode/*.npz`) regenerated with
  Rust DP8 so committed `(t, y)` matches replay;
  **`test_golden_trajectory_matches_rust_dp8`** asserts **`DP8`** integrates back
  to the stored grid (`rtol=1e-9`, `atol=1e-11`).
- **Rust adaptive integrators** no longer enforce a fixed internal step cap;
  pathological runs still fail via non-finite state or step-size collapse.

### Features (N2.a — Pure-Rust ODE stepper suite, sub-stage A)

- **IR (ODE-side)**: extended the symbolic IR with the opcodes the
  continuous catalogue needs that the discrete-map catalogue (N1) did
  not — `OP_TIME` (explicit `t`), `OP_TANH` / `OP_SINH` / `OP_COSH`,
  `OP_POWF` (fractional float exponent), `OP_POW2` (general two-operand
  power for symbolic exponents like `x ** n` in `CircadianRhythm`,
  `x ** (y * z)` in `YuWang`).  Wire format is forward-compatible —
  opcode bytes are stable; map bytecode is unchanged.
- **CompiledOde**: new payload alongside `CompiledMap` that carries
  the RHS program, an optional Jacobian program (skipped when SymEngine
  auto-diff yields unevaluated `Derivative` nodes on `Abs` / `sign` —
  six systems hit this), `param_names`, `dim`, and `n_params`.
- **`_ode_lowering`**: new module (`src/tsdynamics/base/_ode_lowering.py`)
  that walks the SymEngine expression tree returned by `_equations` —
  one-shot walker with var/param/time symbol maps, structural-parameter
  baking, special cases for `sqrt(x) := x ** Rational(1, 2)`, and
  best-effort Jacobian auto-diff with graceful fallback to
  `has_jacobian=False`.
- **PyO3 ODE RHS evaluator** (`crates/tsdyn-native/src/ode.rs`):
  `eval_ode_rhs(bytecode, t, y, params) -> (dim,)` and
  `eval_ode_rhs_batch(bytecode, ts, ys, params) -> (N, dim)`.  Both
  re-exported from `tsdynamics._native`.  The batched form amortises
  the PyO3 trampoline cost over the per-system regression sweep.
- **Golden ODE trajectories**: `scripts/generate_ode_goldens.py` integrates
  each catalogue entry under a 45 s wallclock timeout and writes
  `tests/native/regression/ode/<System>.npz`
  (`t, y, ic, params, param_names, structural_params, structural_values,
  final_time, dt, rtol, atol, method, seed`).  **112** of **115** systems land;
  **3** (`Duffing`, `SprottD`, `SprottI`) skip after blowing up at random ICs —
  hand-pick ICs if goldens are needed for those.
- **Tests**: `tests/test_native_ode.py` — lowering + IR-vs-SymEngine RHS batch on
  every catalogue system (systems whose RHS is entirely NaN on the sample hull
  still skip); golden-file sanity + **Rust DP8 trajectory replay** against the
  committed ``.npz`` grids (`rtol=1e-9`, `atol=1e-11`; **Duffing** / **SprottD**
  / **SprottI** have no golden); `tests/test_ode_methods.py` cross-checks Lorenz
  dense output (**DP5**, **Tsit5**, **BS3** vs **DP8**) plus **RK4**
  repeatability and **DOP853→DP8** alias smoke.

### Breaking changes (analysis API refinement, post-M2)

- **analysis return types are unified**.  Every analysis primitive now
  returns a :class:`Trajectory`.  This deletes three special-case result
  types and makes downstream code (V1 plotters) shape-agnostic.
  - `norm` previously returned a bare 1-D ndarray; now a
    `Trajectory` with `y.shape == (T, 1)`.
  - `local_maxima` / `local_minima` previously returned a
    `(t_peaks, y_peaks)` tuple; now a `Trajectory` of peak times and
    peak heights (`y.shape == (K, 1)`).  Both gained a `refined=False`
    kwarg — pass `refined=True` for sub-sample-accurate times/heights
    via cubic-Hermite refinement.
  - `return_times` previously returned a bare 1-D ndarray; now a
    `Trajectory` of the first-peak time vs ISI.
  - `detect_events` previously returned `EventResult`; now a
    `Trajectory` of crossing times and state at crossings.  The
    `EventResult` class is removed.
  - `return_map` previously returned a `ReturnMap` dataclass; now a
    `Trajectory` of crossing times and observable values
    (`y.shape == (K, 1)`).  The canonical `x_{k+1}` vs `x_k` pair view
    is now computed at plot time via
    `traj.to_dataspec(kind="return_map", step=1)`.
- **Three event-condition classes removed**.
  `Threshold` was the same primitive as `Plane` with a different default
  `direction` — replaced by `Plane(direction="up")` and the new shortcut
  kwargs (`axis=`, `value=`, `direction=`).  `Custom` was a thin
  callable wrapper — bare callables now accepted directly by
  `detect_events` / `poincare_section` / `return_map`.  `LocalExtremum`
  duplicated `local_maxima` / `local_minima` — superseded by
  `local_maxima(refined=True)`.
- **Three call styles** on every event-driven op
  (`detect_events`, `poincare_section`, `return_map`):
  ```python
  traj.detect_events(axis=2, value=27.0, direction="up")   # shortcut kwargs
  traj.detect_events(Plane(axis=2, value=27.0))            # condition object
  traj.detect_events(lambda t, y: np.linalg.norm(y) - 1.0) # bare callable
  ```
- **File layout**: `analysis/events.py`, `analysis/sections.py`,
  `analysis/return_map.py` were merged into `analysis/_events.py`.
  Public surface unchanged — import everything from `tsdynamics.analysis`.
- **Trajectory** gained a `meta: dict[str, Any]` field for future
  per-result metadata.

### Features

- **maps**: discrete-map iteration and Lyapunov spectrum now run through a
  Rust kernel under a symbolic IR (N1). `DiscreteMap._step` /
  `_jacobian` are traced once via NumPy ufunc / operator overloads,
  lowered to a stack-machine bytecode, and evaluated in Rust. The IR op
  set (`Const, Var, Param`, arithmetic, `Pow(i32)`, `Mod`, `Sin/Cos/Exp/
  Log/Abs/Sqrt/Arccos/Sign`, `Where`, comparisons, `And`) is shared with
  the upcoming variational ODE (N3) and cranelift JIT (N4) milestones.
- **native**: Rust extension consolidated into a single cdylib
  `tsdynamics._native._core` (was `_smoke`). Exposes `add_one`,
  `iterate_map`, `lyapunov_spectrum_map`.
- **trajectory**: `Trajectory` gained enrichment methods (M1) — `decimate`,
  `resample`, `project`, `window`, `derivative`, `norm`, `local_maxima`,
  `local_minima`, `return_times`, `to_dataspec`. Every method returns a
  fresh `Trajectory` (or ndarray) and the source arrays are never aliased.
- **analysis**: new `tsdynamics.analysis` subpackage exposing the same
  enrichment operations as pure `(t, y) → (t', y')` functions in
  `analysis.trajectory_ops`. The `Trajectory` methods are thin wrappers
  so the algorithms stay independently unit-testable.
- **events**: `tsdynamics.analysis` now ships event & section detection
  (M2): `EventCondition` protocol + `Plane`, `LinearPlane`, `Threshold`,
  `LocalExtremum`, `Custom` condition classes, `EventResult` container,
  and `detect_events(traj, condition, *, rtol=1e-8)`. Detection uses
  sign-change brackets refined by `scipy.optimize.brentq` on a cubic
  Hermite interpolant built from central-difference slopes;
  `LocalExtremum` instead refines via the Hermite derivative (a closed-
  form quadratic in the local parameter).
- **sections**: `poincare_section(traj, plane, *, direction="up")` returns a
  refined-crossings `Trajectory` carrying the full state at each crossing
  (no axis collapse; the user can `.project()` if they want a
  `(dim - 1)`-dim view).
- **return-map**: `return_map(traj, plane, observable=0, *, step=1)` returns
  a `ReturnMap(x, y, t, step, observable_meta)`; supports both an integer
  component and a callable `fn(t, y) -> float` for the scalar observable,
  with `step >= 1` for N-step return maps. `.to_dataspec(kind="return_map")`
  matches the M1 placeholder dict shape so V2 can swap in `DataSpec`
  without breaking call sites.

### Refactoring

- **maps/geometric**: `Tent` and `Baker` rewritten to use `np.where`
  instead of Python `if`/`else` so they trace into the new IR. Outputs
  are bit-identical to the previous Numba path.

### Internals

- **base**: new `_ir.py`, `_tracer.py`, `_lowering.py` under
  `tsdynamics.base`. `DiscreteMap` keeps the Numba-compiled fallback
  loop for user-defined maps that contain operations the IR can't yet
  represent — they trigger `NotLowerableError` during lowering and the
  dispatcher transparently falls through. `@staticjit` therefore stays
  as `staticmethod(njit(...))`; making it a no-op is deferred until
  the Numba fallback retires (post-N4).

## v1.0.0 (2026-05-15)

### Bug Fixes

- **base**: widen ParamSet.param_hash from 32 to 64 bits
- **dde**: explicit _delay_params with validation; LE zero-weight warning
- **maps**: Jacobian correctness, parameter order, and robust LE loop
- **ode**: variable-dim systems compile cleanly; KS large-L is non-trivial

### Features

- **base**: support class-level default_ic in SystemBase.resolve_ic
- **utils**: Drop deprecated implementation of other approaches to find proper integration dt

### Refactoring

- **systems**: make every ODE/DDE _equations keyword-only
- remove viz submodule and tidy public surface
- **ode_base**: Introduce platform-specific extension handling for compiled ODE objects, enhance caching logic, and improve docstring clarity for better understanding of integration and Lyapunov processes
- **dde_base**: Enhance caching mechanism for compiled DDE objects, improve path handling, and update docstrings for clarity
- **discrete**: Replace DynMap with DiscreteMap in multiple classes, update method names for consistency, and enhance docstring clarity for improved maintainability
- **continous**: Replace DynSys with ContinuousSystem in multiple classes, update method names for consistency, and remove unused imports to enhance code clarity and maintainability
- **base**: Introduce new core classes (ParamSet, Trajectory, SystemBase) and enhance existing classes for improved structure and clarity

## v0.2.0 (2026-04-11)

### Features

- **tests**: feat(tests): Add synthetic KS trajectory fixture and enhance visualization tests for ax handling and plotter consistency
- **chaotic_attractors.py**: feat(chaotic_attractors): Enhance Lorenz96 and KuramotoSivashinsky classes with improved initialization and updated documentation. Adjusted parameter requirements and default integrator settings for stability
- **viz**: Refactor visualization module: enhance public API, add private utilities, and improve function contracts. Remove base.py and update plotter and animator functions for consistency. Introduce transforms for transient removal and spectral analysis

### Refactoring

- **pyproject.toml**: refactor(pyproject.toml): Update linting rules to accommodate math notation conventions and enhance docstring requirements for math classes, improving code clarity and consistency
- **utils-plotters-transforms**: Enhance docstring clarity and improve parameter handling in bifurcation_diagram function
- **utils**: Improve docstring clarity and consistency in curvature_dt, frequency_dt, general, and sagitta_dt modules
- **chaotic_attractors.py,-chem_bio_systems.py,-climate_geophysics.py,-neural_cognitive.py,-oscillatory_systems.py,-physical_systems.py,-exotic_maps.py**: Improve docstring clarity and consistency across multiple classes, enhance initialization logic, and refine mathematical expressions for better readability and accuracy
- **ode_base.py-dde_base.py**: refactor(dde_base.py, ode_base.py): Improve docstring clarity for rhs and jac methods, and streamline local Lyapunov exponents extraction in the integration process
- **map_base.py**: refactor(map_base.py): Enhance docstrings for rhs and jac methods, and improve initial condition handling in iterate method for clarity and robustness
- **oscillatory_systems.py**: refactor(oscillatory_systems.py): Improve docstring formatting in Lissajous3D and Lissajous2D classes by standardizing parameter and return sections for clarity
- **utils**: refactor(utils): Update type hints to use the new syntax and improve clarity in curvature_dt, frequency_dt, general, and sagitta_dt modules
- **neural_cognitive.py**: refactor(neural_cognitive.py): Update initialization in Hopfield and BeerRNN classes to use n_dim for clarity and consistency in neuron dimension handling
- **ode_base.py-map_base.py-dde_base.py-base.py**: refactor: Update type hints in base classes and improve docstring formatting across multiple files

## v0.1.0 (2026-04-09)

### Feat

- add GitHub Actions workflows for publishing and testing
- add comprehensive test suite for TSDynamics

### Fix

- address issues with past_from_function and diagonal zero handling
- use stored initial_conds in lyapunov_spectrum when none passed
