---
description: The recipe for contributing a new integration kernel to the Rust engine — the Solver trait, register_solver!, the shared control loops, the one central build_solver table, the Python SolverSpec mirror, and the cross-validation tests.
---

<span class="ts-kicker">Project · Adding a solver</span>

# Adding a solver

A new integration method is a two-sided contribution: the **real kernel lives in
Rust** (in the `tsdyn-solvers` crate), and a small **`SolverSpec` mirrors it** in
the Python registry so users can select it by name with `method="..."`. This
page is the end-to-end recipe at a level a contributor can follow.

!!! note "This is an engine change"
    Unlike [adding a system](adding-a-system.md) — pure Python, no build step —
    a solver touches the Rust engine, so you need a
    [Rust toolchain](https://rustup.rs/) and the editable dev install
    (`uv sync --group dev`). The `reference` backend is a SciPy oracle that does
    **not** run our kernels; test a new kernel on the engine backends —
    `backend="jit"` (the default) and `backend="interp"`, which must agree
    bit-for-bit.

## The mental model

The engine owns the integration **loop**; a kernel owns a single **step**. Every
kernel implements the `Solver` trait (`crates/tsdyn-solvers/src/solver.rs`):

```rust
pub trait Solver {
    fn name(&self) -> &'static str;
    fn caps(&self) -> Caps;
    fn step(&mut self, /* state, rhs evaluator, h, ... */) -> StepResult;
}
```

Kernels **self-register** at link time via the `register_solver!` macro
(`inventory`-based — there is no central kernel table), and both the SSA
interpreter and the Cranelift JIT back the same `Evaluator` trait, so a kernel is
automatically **backend-agnostic** and `interp == jit` bit-for-bit.

## The fast inner loop

Iterate against the pure-Rust crate — no Python, no maturin — while you get the
tableau right:

```bash
cd crates && cargo test -p tsdyn-solvers      # ≈ 10 s
```

!!! warning "The stale-extension gotcha"
    `uv run pytest` does **not** auto-rebuild the Rust extension on a pure-Rust
    edit — it reuses the stale `src/tsdynamics/_rust.abi3.so`, and your change
    silently has no effect. After editing any Rust file, force a rebuild:

    ```bash
    uv run --reinstall-package tsdynamics pytest -m "not slow" --no-cov
    ```

    Look for the `Building tsdynamics` line. Sanity check with
    `stat -c %y crates/tsdyn-core/src/bridge/marshal.rs src/tsdynamics/_rust.abi3.so`
    — the `.so` must be **newer**. (`uv` may also re-sync `uv.lock`; restore it
    with `git checkout uv.lock` if so.)

## Writing the kernel

Two shared control layers do almost all the work — you supply a Butcher tableau
(or a one-step formula) and a thin `Solver` impl.

=== "Explicit Runge–Kutta"

    The shared `explicit/control.rs` implements the step machinery; you provide
    the tableau constants.

    - **Fixed-step**: tableau `C` / `A` / `B` + a thin `Solver` calling
      `fixed_step(...)`. It builds through the registry's default factory — **no
      `build_solver` change**.
    - **Adaptive** (single embedded error vector): tableau `C` / `A` / `B` / `E`
      + an `err_exponent` (`= −1 / (est_order + 1)`) + a `with_tolerances`
      constructor + `adaptive_step(...)`. (`dop853` is the exception — it blends
      its own error estimate in a custom `step()`.)
    - **Wire it up**: add `mod your_kernel;` and `pub use` to
      `crates/tsdyn-solvers/src/explicit/mod.rs`.

    The registration lives in the kernel's own always-compiled module — copy the
    pattern from `explicit/rk45.rs`:

    ```rust
    register_solver!(
        "your_kernel",
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
        || Box::new(YourKernel::new())
    );
    ```

=== "Implicit / stiff"

    The shared `implicit/control.rs` gives you adaptive error control for free
    via step-doubling + Richardson extrapolation: implement the `BaseStep` trait
    (the one-step formula + `order()`), and `DoublingWork::doubled_step` handles
    the rest.

    - Linear algebra is in `implicit/linalg.rs` (`build_shifted` for
      $I - c\,h\,J$, plus `lu_factor` / `lu_solve`); the modified-Newton substage
      solver and `wrms` norm are in `implicit/newton.rs`.
    - Every implicit kernel needs the analytic Jacobian, so `Caps::implicit`
      defaults `needs_jacobian = true` — the engine lowers the tape
      `with_jacobian=true` automatically for it.
    - **Wire it up**: add `mod your_kernel;` and `pub use` to
      `crates/tsdyn-solvers/src/implicit/mod.rs`, and `register_solver!` with
      `Caps::implicit(...)`.

## The one central table

There is exactly **one** hardcoded table, and it is easy to miss. An **adaptive**
kernel that owns its own `rtol` / `atol` must get an arm in
`build_solver` (`crates/tsdyn-core/src/bridge/marshal.rs`), or its tolerances are
**silently ignored** (it falls through to the default-tolerance registry
factory):

```rust
// crates/tsdyn-core/src/bridge/marshal.rs
pub fn build_solver(name: &'static str, rtol: f64, atol: f64) -> Box<dyn Solver> {
    match name {
        "rk45"   => Box::new(Rk45::with_tolerances(rtol, atol)),
        // ... one arm per tolerance-owning adaptive kernel ...
        "your_kernel" => Box::new(YourKernel::with_tolerances(rtol, atol)),
        // Fixed-step + Adams multistep + out-of-tree plugins build through the
        // registry factory (no arm needed):
        other    => tsdyn_solvers::make(other).expect("resolve_solver guarantees this"),
    }
}
```

Also add the concrete type to the `use tsdyn_solvers::explicit::{...}` (or
`implicit::{...}`) import at the top of that file. **Fixed-step kernels need no
arm** — they carry no tolerances.

## Mirroring it in Python

Add a `SolverSpec` — the Python mirror of the Rust kernel — to the matching dict
in `src/tsdynamics/solvers/` (`explicit.py`, `implicit.py`, or `stochastic.py`).
The capability flags **must match the Rust `Caps`**:

```python
# skip-doctest — a template: registering a placeholder kernel would put
# "your_kernel" in the solver table for the rest of the session
from tsdynamics._solvers import SolverSpec, SolverCaps, register

register(
    SolverSpec(
        name="your_kernel",
        caps=SolverCaps(
            kind="explicit",        # or "implicit"
            adaptive=True,
            needs_jacobian=False,   # True for implicit kernels
            supports=frozenset({"ode"}),
        ),
        description="Your Kernel N(M) adaptive — one-line human description",
    )
)
```

Optional friendly aliases go in `select.py::_ALIASES` (keys are normalised —
lowercased, whitespace / `-` → `_` — so `"RK45"`, `"dopri5"`, and `"rk-45"` all
resolve). No change to `run.py` is needed: `run.integrate` flows any registered
method through `solvers.resolve`, so once the spec is registered,
`run(solver="your_kernel")` just works and the tolerances thread through
`build_solver`.

## Tests to update

- **`tests/test_solvers.py`** — the `_EXPECTED_BUILTINS` dict and the
  `test_available_for_filters_by_family` **exact-set** (`==`) assertions break on
  a new solver; update them. `test_python_specs_mirror_rust_register_macros`
  parses the Rust source (a regex over `register_solver!` / `Caps::implicit` /
  `needs_jacobian`) and stays green as long as your Python spec matches the Rust
  `Caps`.
- **Rust cross-validation** — `crates/tsdyn-solvers/tests/explicit_accuracy.rs`
  and `implicit_stiff.rs` are the strong oracle for tableau correctness: they
  assert cross-method agreement on Lorenz / stiff reference problems. A wrong
  tableau constant fails here, not in a subtle downstream number.

## The invariants your kernel must hold

| Invariant | Why |
| --------- | --- |
| `interp == jit` bit-for-bit | Both backends drive the same `&mut dyn Solver` through the `Evaluator` trait |
| Python `Caps` == Rust `Caps` | `test_python_specs_mirror_rust_register_macros` gates it |
| Adaptive kernel → `build_solver` arm | Otherwise `rtol` / `atol` are silently ignored |
| Cites the **original paper** | Never a competing library — a hard rule in docstrings and docs |

## See also

- [Integration & methods](../analysis/integration-and-methods.md) — the solver
  and backend layer from the user's side, with the full capability table.
- [Contributing](contributing.md) — the quality gates and PR flow.
- [Adding a system](adding-a-system.md) — the other, pure-Python extension path.
