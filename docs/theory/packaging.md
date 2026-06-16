---
description: How TSDynamics is packaged and distributed — one maturin wheel carrying the Python package and the native Rust engine.
---

<span class="ts-kicker">Theory · 04</span>

# Packaging & distribution

TSDynamics ships as **one [maturin](https://www.maturin.rs/) wheel** — the
pure-Python package and the compiled Rust engine in the same distribution. This
page records the shape, why it is shaped that way, and how the prebuilt wheels
are produced.

## What you install

| Distribution | Build backend | Contents | Compiler needed? |
| ------------ | ------------- | -------- | ---------------- |
| `tsdynamics` | maturin (wheel + sdist) | the whole library — systems, families, analysis, transforms — **plus** `tsdynamics/_rust.abi3.so`, the zero-warmup tape interpreter + Cranelift JIT + solver kernels | **No** for the prebuilt wheel; building *from the sdist* needs a Rust toolchain |

`pip install tsdynamics` pulls a prebuilt `abi3` wheel for your platform, so the
engine arrives compiled — no Rust toolchain and no C compiler. Every family
(ODEs, DDEs, maps) lowers its symbolic equations to the engine in-process and
runs with no warmup, reached through `backend="interp"` (the default) or
`backend="jit"` (see the [compilation pipeline](compilation.md)).

## One wheel, one layout

The wheel bundles the Python package from `src/` and the extension built from
the engine crate into a single `site-packages/tsdynamics/` tree:

```
tsdynamics/__init__.py            # the Python package, from src/
tsdynamics/families/…  analysis/…  # the Python package, from src/
tsdynamics/_rust.abi3.so          # the compiled engine
```

`import tsdynamics._rust` resolves to the extension sitting next to
`__init__.py`. maturin places the extension as the `_rust` submodule of the
package rather than a stray top-level `_rust/` package.

!!! note "Editable installs"
    An *editable* `tsdynamics` (`pip install -e .`) puts `tsdynamics.__path__` at
    `src/tsdynamics/`. For development the engine is `cargo build`-ed and the
    `.so` dropped straight into `src/tsdynamics/_rust.abi3.so` (see
    `.github/workflows/engine-bindings.yml`); the prebuilt-wheel install carries
    it directly. Both routes satisfy `import tsdynamics._rust`.

## How the wheels are built

The root `pyproject.toml` uses maturin as the build backend, with
`python-source = "src"`, `module-name = "tsdynamics._rust"`, and
`manifest-path = "crates/tsdyn-core/Cargo.toml"` so a single wheel carries both
the Python package and the engine.

`.github/workflows/wheels.yml` builds it across platforms. Because the extension
targets the **stable ABI (abi3, CPython ≥ 3.12)**, a single wheel per
`(platform, architecture)` covers every supported Python — no per-version
matrix.

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

The platform wheels are published to PyPI as part of the release flow, so
`pip install tsdynamics` delivers the engine prebuilt on every listed platform,
with no compiler and no warmup.
