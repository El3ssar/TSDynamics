## Unreleased

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
