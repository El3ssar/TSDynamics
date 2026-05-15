# Rust acceleration (analysis-kernel side)

This doc covers Track C: Rust crates that back *non-stepping* hot loops (sweeps,
basins, recurrence, dimensions, continuation predictor/corrector). The stepping
engines themselves migrate to Rust in Track E ([native-solver-migration.md](native-solver-migration.md)).

## Workspace layout

```
TSDynamics/
├── crates/                      # Rust workspace
│   ├── Cargo.toml               # [workspace] members = [...]
│   ├── tsdyn-core/              # shared types: state buffer, IR scaffolding
│   ├── tsdyn-sweep/             # parameter sweeps (R2)
│   ├── tsdyn-recurrence/        # recurrence matrix + RQA (R4)
│   ├── tsdyn-dim/               # correlation sum, boxcount (R3)
│   ├── tsdyn-basin/             # basin sweep + classification (R5)
│   └── tsdyn-continuation/      # predictor/corrector (R6)
├── pyproject.toml               # maturin as build-backend (replaces hatch)
├── rust-toolchain.toml          # pinned channel = stable
└── src/tsdynamics/
    └── _native/                 # installed extension modules land here
        ├── __init__.py          # thin facade: from ._sweep import sweep_chunked
        ├── _sweep.<ext>.so
        ├── _recurrence.<ext>.so
        └── ...
```

Each crate is `crate-type = ["cdylib"]` and exposes a PyO3 `#[pymodule]` named after
the file (`_sweep`, `_recurrence`, …). The Python facade `_native/__init__.py`
re-exports symbols so the rest of the library imports
`from tsdynamics._native import sweep_chunked` rather than caring which crate.

## Build & packaging

- **Build backend:** maturin. Replaces hatch as `build-backend` in
  `pyproject.toml`. Hatch's vcs-version functionality keeps working via
  `maturin`'s `version-from-file` config.
- **Local development:** `maturin develop --release` installs the extensions into
  the active venv. CI uses `maturin build --release` and uploads to PyPI via
  `cibuildwheel`.
- **Wheels:** built for cp312, cp313 on Linux (manylinux_2_28), macOS (universal2),
  Windows-x64 (Windows added after Linux/macOS prove stable in R1).
- **Source dist:** include the Rust source so users can build from source if no
  wheel matches.

## Boundary principle

> **Python owns the model definition and the public API. Rust owns hot loops.**

Concretely:

- The user writes `_equations` in Python with SymEngine. **Never changes.**
- The symbolic pipeline (SymEngine simplification, Jacobian extraction) stays in
  Python.
- The compiled RHS function — currently from JiTCODE — is callable from Rust via
  cffi (a C function pointer + opaque data pointer). After N4, the JIT-compiled
  RHS lives natively in-process and is callable directly.
- All outer loops (sweep over params, iterate over ICs, time-step the variational
  system, walk a continuation curve) live in Rust.

## No user-facing flag

The Python API never names "Rust." There is no `backend="..."`. The
implementation language is internal. A regression suite (`tests/native/`) asserts
that every Rust-backed analysis returns the same answer (within tolerance) as a
reference Python implementation kept in `tests/reference/`. Reference impls are
slow but trusted; they exist purely to lock the Rust path's correctness.

## Concurrency

- Inside Rust: `rayon` for data parallelism.
- Across Python: PyO3 releases the GIL for the duration of every Rust call. So
  Python callers can drive multiple Rust kernels from threads if they want.
- For Rust kernels that call back into Python (continuation corrector before N4):
  acquire the GIL only for the duration of the callback. Document this in the
  individual kernel's design notes.

## What goes in `tsdyn-core`

- `StateBuffer`: contiguous, cache-friendly storage for `(N_ic, dim)` ensembles.
- IR scaffolding shared with Track E (so analysis kernels and the stepper share a
  single RHS calling convention).
- A `ProblemHandle` opaque struct that wraps either a JiTCODE-cffi function
  pointer (early) or a cranelift-JIT'd native function (after N4). Analysis
  kernels consume `ProblemHandle` so they don't change when the stepping pipeline
  underneath does.

## Acceptance criteria (per Rust milestone)

Every R-milestone lands with:

1. The crate builds clean on Linux, macOS (+ Windows once added).
2. PyO3 bindings have type stubs in `_native/*.pyi`.
3. A reference Python implementation lives in `tests/reference/<feature>.py`.
4. A correctness test asserts Rust output matches reference within tolerance on a
   fixed set of inputs.
5. A benchmark under `bench/<feature>_bench.py` records wall-clock vs the same
   computation in DynamicalSystems.jl (when applicable). Results land in
   `bench/RESULTS.md`.

## Open questions (resolve before R1)

- maturin layout: `crates/` at root vs `python/` + `rust/` split. **Recommended:
  `crates/` at root**, Python package stays at `src/tsdynamics/`.
- MSRV: pin to latest stable, no MSRV promises pre-1.0.
- Windows from R1 or follow-up? **Recommended: Linux + macOS in R1, Windows after
  N2.**
