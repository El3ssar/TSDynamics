# Milestone R1 — Rust toolchain + maturin + CI wheels

Status: TODO
Depends on: M0
Estimated scope: one chat
Design doc: [design/rust-acceleration.md](../design/rust-acceleration.md)

## Motivation

R1 makes Rust possible in this repo. Nothing user-visible changes; no analysis
primitive moves yet. But after R1 we have:

- A Cargo workspace under `crates/`.
- A maturin build that produces `src/tsdynamics/_native/*.so` extension modules.
- CI building wheels on Linux + macOS for cp312 and cp313.
- A smoke crate that proves the round trip works: a Rust function callable from
  Python that returns a value Python can verify.

Every later R-milestone and every N-milestone sits behind this one. Without R1,
nothing in Track C or Track E can move.

## API sketch

The only user-visible thing R1 ships is the import:

```python
>>> from tsdynamics._native._smoke import add_one
>>> add_one(41)
42
```

That's it. All real APIs land in later milestones.

## Design

```
TSDynamics/
├── crates/
│   ├── Cargo.toml               # [workspace] members = ["tsdyn-core", "tsdyn-smoke"]
│   ├── tsdyn-core/              # placeholder, just declares the workspace and shared types
│   │   ├── Cargo.toml
│   │   └── src/lib.rs           # `pub use` of nothing yet; the module is here as a hook for later
│   └── tsdyn-smoke/
│       ├── Cargo.toml           # crate-type = ["cdylib"]; depends on pyo3
│       └── src/lib.rs           # #[pymodule] _smoke { fn add_one(...) }
├── rust-toolchain.toml          # channel = "stable"
├── pyproject.toml               # build-backend = "maturin"; declares the extension modules
└── src/tsdynamics/
    └── _native/
        ├── __init__.py          # `from ._smoke import add_one`
        └── _smoke.pyi           # type stubs
```

### `pyproject.toml` changes

- Switch `[build-system]` to maturin.
- Keep `[project]` metadata (name, version, deps, …) but move version resolution
  to `[tool.maturin]`.
- Configure `[tool.maturin]` to:
  - target Python ≥ 3.12,
  - install Rust extensions into `src/tsdynamics/_native/`,
  - skip auditwheel on macOS,
  - build `tsdyn-smoke` as the only module for R1.

`hatch-vcs` versioning currently lives in `_version.py`. Decide during R1
whether to keep `hatch-vcs` writing `_version.py` (works fine alongside maturin)
or use maturin's git-tag version inference. **Recommendation:** keep
`hatch-vcs`, simpler.

### CI

- `.github/workflows/wheels.yml`: matrix over `(os in [ubuntu-22.04, macos-14],
  python in [3.12, 3.13])`. Uses `messense/maturin-action@v1` for the build,
  `actions/upload-artifact@v4` for the result.
- Existing test workflow (`tests.yml` if present, otherwise create) gets a
  `maturin develop --release` step before pytest.
- Skip Windows from R1. Add in a follow-up after N2.

### `tsdyn-smoke` crate contents

```rust
// crates/tsdyn-smoke/src/lib.rs
use pyo3::prelude::*;

#[pyfunction]
fn add_one(x: i64) -> i64 { x + 1 }

#[pymodule]
fn _smoke(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_one, m)?)?;
    Ok(())
}
```

### `tsdyn-core` crate contents

Empty for R1 except a placeholder type that future crates will import (e.g., a
`ProblemHandle` stub). Exists to lock the workspace dependency graph.

## Files to create / modify

Create:
- `crates/Cargo.toml`
- `crates/tsdyn-core/Cargo.toml`
- `crates/tsdyn-core/src/lib.rs`
- `crates/tsdyn-smoke/Cargo.toml`
- `crates/tsdyn-smoke/src/lib.rs`
- `rust-toolchain.toml`
- `src/tsdynamics/_native/__init__.py`
- `src/tsdynamics/_native/_smoke.pyi`
- `tests/test_native_smoke.py`
- `.github/workflows/wheels.yml`

Modify:
- `pyproject.toml` (build-backend → maturin; `[tool.maturin]` config)
- Existing CI workflow (add `maturin develop` step)
- `.gitignore` (add `target/`, `*.so`, `*.pyd`)

## Acceptance criteria

- [ ] `cargo build --workspace --release` succeeds locally.
- [ ] `maturin develop --release` installs `_smoke.so` into the venv.
- [ ] `python -c "from tsdynamics._native._smoke import add_one; assert add_one(41) == 42"` passes.
- [ ] `tests/test_native_smoke.py` covers the import and the function.
- [ ] CI builds wheels on Ubuntu + macOS for cp312, cp313.
- [ ] `uv run pytest -m "not slow" --no-cov` still passes (no regressions in the
      Python-only test suite).
- [ ] `uv run ruff check src/ tests/` clean.
- [ ] `pip install <built wheel>` on a clean venv works and the smoke import
      succeeds.
- [ ] [design/rust-acceleration.md](../design/rust-acceleration.md) updated if any
      decisions land differently than the doc predicts.

## Out of scope

- Real kernels (R2 onwards).
- Windows wheels (deferred until after N2).
- Replacing Numba (that's N1).
- Any changes to `ContinuousSystem` / `DelaySystem` / `DiscreteMap` (R1 touches
  only build plumbing).

## Open questions for the user

- Confirm `crates/` at repo root (recommended) vs `python/`+`rust/` split.
- Confirm we drop `hatch` as build-backend in favour of `maturin`, but keep
  `hatch-vcs` for version resolution.
- Confirm Linux + macOS only for R1; Windows added after N2.
