---
description: Install TSDynamics with pip or uv from a prebuilt wheel — the compiled Rust engine ships inside it, so there is no build step and no compiler toolchain.
---

<span class="ts-kicker">Start · 01</span>

# Install

TSDynamics runs on **Python ≥ 3.12**. Installing it is a single command, and
the compiled integration engine ships inside the wheel — you need no Rust
toolchain, no C compiler, and no build step.

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
covers every CPython ≥ 3.12.

Concretely, this means:

- **No toolchain.** Installing and running the library never invokes a compiler.
- **No build step, nothing on disk.** Every family — ODEs, delay equations,
  stochastic equations, and maps — lowers its symbolic equations to an
  in-process engine *tape* the first time you run it, in a fraction of a second,
  and the tape is compiled to native code by the built-in JIT. Both steps are
  in-process and memoised per system; there is no ahead-of-time compilation and
  nothing is written to disk. Editing a system's equations simply takes effect on
  the next run; there is no cache to wipe.
- **The same numbers everywhere.** The tape runs on a Cranelift JIT by default,
  or on a Rust interpreter that is bit-for-bit identical — see
  [backends](concepts.md#backends-jit-interp-reference) — so results do not
  depend on which BLAS or Python build you happen to have.

!!! note "Building from source"
    Only building *from the source distribution* — the `sdist`, the fallback for
    a platform outside the wheel matrix — needs a Rust toolchain, because the
    package build backend is [maturin](https://www.maturin.rs/). The normal
    `pip install` path never reaches it.

## Optional extras

The base install pulls in **no plotting library** — `import tsdynamics` stays
lightweight and imports nothing heavy. The plotting backends are opt-in extras:

| Extra | Installs | When you want it |
| ----- | -------- | ---------------- |
| `tsdynamics[viz]` | `matplotlib` | The reference renderer: static 2-D and 3-D figures, movies (mp4/gif) |
| `tsdynamics[interactive]` | `plotly` | The interactive backend: rotatable 3-D and self-contained HTML export |
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
print(traj.y.shape)             # (100, 2)
```

`ts.registry.families()` reports the built-in catalogue by family. If

```python
ts.systems.Lorenz().run(final_time=1.0).y.shape   # (51, 3)
```

also returns cleanly, the engine is wired up and you are ready to go.

## Next

[**02 · First trajectory**](first-trajectory.md) — instantiate the Lorenz
system, integrate it, read its components, and compute its Lyapunov spectrum.
