---
description: How TSDynamics is packaged and distributed — the pure-Python core, the optional prebuilt Rust engine accelerator, and the path to a single wheel.
---

<span class="ts-kicker">Theory · 04</span>

# Packaging & distribution

TSDynamics ships in two pieces today, converging to one. This page records the
shape, why it is shaped that way, and how the prebuilt wheels are produced.

## What you install

| Distribution | Build backend | Contents | Compiler needed? |
| ------------ | ------------- | -------- | ---------------- |
| `tsdynamics` | pure-Python (wheel + sdist) | the whole library — systems, families, analysis, transforms | **No** for install; the *default* integration backends compile your equations at first call (see the [compilation pipeline](compilation.md)) |
| `tsdynamics-rust-engine` | compiled (maturin) | one file — `tsdynamics/_rust.abi3.so`, the zero-warmup tape interpreter + Cranelift JIT + solver kernels | **No** — prebuilt cross-platform wheels |

The `tsdynamics` distribution is self-sufficient: the default ODE/DDE/map
backends (the v2 engine) run with no extra install. The Rust engine is an
**opt-in accelerator** reached with `backend="interp"` / `backend="jit"`; until
it becomes the default (see *The road to one wheel* below) it is a separate,
optional distribution.

## Why two distributions (for now)

A clean break to a single compiled wheel is the destination, not the current
state: the v2 backends are still the default and the engine is still being
cross-validated against them. Splitting the accelerator out keeps three
properties that matter during the migration:

- **`pip install tsdynamics` keeps needing no wheel for your platform.** It is
  pure-Python, so it installs everywhere, exactly as before.
- **The release pipeline is undisturbed.** `tsdynamics` is still built and
  published by python-semantic-release as a pure-Python wheel + sdist; adding a
  compiled engine does not entangle the version-bump / tag / publish flow.
- **The engine can iterate independently.** Its wheels are built and verified on
  their own schedule while it is gated.

### The namespace, and why it doesn't collide

The engine wheel ships **exactly one file** into the import namespace:

```
tsdynamics/_rust.abi3.so          # from tsdynamics-rust-engine
```

and the pure-Python wheel ships everything else:

```
tsdynamics/__init__.py            # from tsdynamics
tsdynamics/families/…  analysis/…  # from tsdynamics
```

Installed together, both land in the same `site-packages/tsdynamics/` directory.
There is **no shared file path between the two distributions**, so there is no
install conflict, and `import tsdynamics._rust` resolves to the extension sitting
next to `__init__.py`. The engine wheel deliberately does **not** ship a
`tsdynamics/__init__.py`; if it did, the two wheels would fight over ownership of
the package and split `tsdynamics.__path__`.

This is achieved with a maturin *mixed layout* — `crates/tsdyn-core/pyproject.toml`
sets `python-source = "python"` over an otherwise-empty `python/tsdynamics/`
package, which makes maturin place the extension as the `_rust` submodule instead
of emitting a stray top-level `_rust/` package.

!!! note "Editable installs are different"
    An *editable* `tsdynamics` (`pip install -e .`) puts `tsdynamics.__path__` at
    `src/tsdynamics/`, where a separately-installed site-packages engine wheel is
    invisible. For development the engine is instead `cargo build`-ed and the
    `.so` is dropped straight into `src/tsdynamics/_rust.abi3.so` (see
    `.github/workflows/engine-bindings.yml`). Both routes satisfy
    `import tsdynamics._rust`.

## How the wheels are built

`.github/workflows/wheels.yml` builds the engine across platforms with
maturin. Because the extension targets the **stable ABI (abi3, CPython ≥ 3.12)**,
a single wheel per `(platform, architecture)` covers every supported Python — no
per-version matrix.

| Platform | Architectures |
| -------- | ------------- |
| manylinux | x86_64, aarch64 |
| musllinux | x86_64, aarch64 |
| macOS | x86_64, arm64 |
| Windows | x64 |

A source distribution (`sdist`) is also built; it vendors the sibling engine
crates so it is self-contained, but building *from* the sdist still needs a Rust
toolchain — it is the fallback for platforms outside the wheel matrix, not the
no-compiler path.

The workflow runs a one-platform build smoke on packaging-related pull requests
and the full matrix on a manual dispatch or a release tag. It uploads the wheels
as build artifacts and does **not** publish them to an index: pre-migration the
engine is kept off PyPI on purpose.

## The road to one wheel

When the cross-validation gate (stream I-XVAL) retires the v2 backends and the
Rust engine becomes the sole, default engine, the two distributions fold into a
single compiled `tsdynamics` wheel:

1. Switch the root `pyproject.toml` build backend to maturin, with
   `python-source = "src"` and the same `module-name = "tsdynamics._rust"`, so one
   wheel carries both the Python package and the engine.
2. Move the cross-platform matrix from `wheels.yml` into the release flow and
   publish the platform wheels (drop the engine's `Private :: Do Not Upload`
   marker; reconcile the version source with python-semantic-release).
3. Drop the C-compiler requirement and the `~/.cache/tsdynamics/` compile cache
   the v2 backends needed.

After that step, `pip install tsdynamics` delivers the engine prebuilt on every
listed platform, with no compiler and no warmup.
