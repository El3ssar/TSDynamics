---
description: Install TSDynamics with pip or uv from a prebuilt wheel — the compiled Rust engine ships inside it, so there is no build step and no compiler toolchain.
---

<span class="ts-kicker">Start · 01</span>

# Install

TSDynamics requires **Python ≥ 3.12**. Prebuilt wheels include the compiled
integration engine, so installation on a supported platform needs no compiler.

```bash
pip install tsdynamics
```

or, with [uv](https://docs.astral.sh/uv/):

```bash
uv add tsdynamics
```

## What "no compiler needed" means

`pip install tsdynamics` pulls a **prebuilt `abi3` wheel**. There is one wheel
per platform and architecture — manylinux, musllinux, macOS (Intel and Apple
silicon), and Windows — and each bundles the native Rust engine
(`tsdynamics._rust`) alongside the pure-Python package. Because the wheel is
built against CPython's stable ABI (`abi3`, tagged `cp312`), a *single* wheel
covers supported CPython versions ≥ 3.12.

Concretely, this means:

- **No external toolchain.** Installing a prebuilt wheel needs no compiler.
  The default execution backend uses its bundled JIT compiler in-process.
- **In-process equation compilation.** On first use, the library translates
  equations to an expression tape and the bundled JIT compiles it to native code.
  Both are cached in memory. Startup cost depends on the size of the system;
  repeated runs can reuse the compiled equations. No compiled-equation files
  are written to disk.
- **Interpreter and JIT agreement.** The tape runs on a Cranelift JIT by default,
  or on a Rust interpreter that is bit-for-bit identical — see
  [backends](concepts.md#backends-jit-interp-reference). This parity applies on
  the same machine for the same solver and inputs; floating-point results
  across platforms or numerical-library builds can still differ.

!!! note "Building from source"
    Building from a source distribution or editable checkout needs a Rust
    toolchain because the
    package build backend is [maturin](https://www.maturin.rs/). The normal
    wheel installation path never reaches it. Use
    `pip install --only-binary=:all: tsdynamics` to require a wheel and fail
    instead of building from source.

## Optional extras

The base install pulls in **no plotting library**. It still imports scientific
dependencies, including NumPy and SciPy. The plotting backends are opt-in extras:

| Extra | Installs | When you want it |
| ----- | -------- | ---------------- |
| `tsdynamics[viz]` | `matplotlib` | The reference renderer: static 2-D and 3-D figures, movies (mp4/gif) |
| `tsdynamics[interactive]` | `plotly` | Interactive figures and HTML export; exported pages use CDN JavaScript by default |
| `tsdynamics[plot]` | `matplotlib` | A back-compatible alias of `viz` |

Combine extras in the usual way:

```bash
pip install "tsdynamics[viz,interactive]"
```

The `json` and `threejs` export renderers need no extra at all — they serialize
a plot description over the standard library. See
[Visualization](../visualization/index.md) for what each backend can draw.

## Verify the install

```python
import tsdynamics as ts

print(ts.__version__)
print(ts.registry.families())   # {'ode': 142, 'dde': 6, 'sde': 3, 'map': 26}

traj = ts.systems.Henon().run(steps=100, ic=[0.1, 0.1])
print(traj.y.shape)             # (101, 2): initial state plus 100 iterates
```

`ts.registry.families()` reports the built-in catalogue by family. If

```python
ts.systems.Lorenz().run(final_time=1.0).y.shape   # (51, 3)
```

also returns successfully, the installation can run both maps and ODEs.

## Next

[**02 · First trajectory**](first-trajectory.md) — instantiate the Lorenz
system, run it, read its components, and measure its Lyapunov spectrum.
