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
