---
description: Install TSDynamics with pip or uv from a prebuilt wheel, and choose optional extras.
---

<span class="ts-kicker">Start · 01</span>

# Install

TSDynamics requires **Python ≥ 3.12**.

```bash
pip install tsdynamics
```

or, with [uv](https://docs.astral.sh/uv/):

```bash
uv add tsdynamics
```

## No compiler needed

`pip install tsdynamics` pulls a **prebuilt `abi3` wheel** — manylinux,
musllinux, macOS, and Windows — that bundles the native Rust engine. You need
no Rust toolchain and no C compiler to install or run the library; every
family (ODEs, DDEs, maps) lowers its symbolic equations to the engine
in-process and runs with **zero warmup**. Editing a system's equations simply
takes effect on the next run — there is no compile step and no on-disk cache.

!!! note "Building from source"
    Only building *from the source distribution* (the `sdist`, the fallback for
    platforms outside the wheel matrix) needs a Rust toolchain — the package
    build backend is [maturin](https://www.maturin.rs/). See
    [Packaging & distribution](../theory/packaging.md) for the wheel matrix and
    the build recipe.

## Optional extras

| Extra | Installs | When you want it |
| ----- | -------- | ---------------- |
| `tsdynamics[plot]` | `matplotlib` | Plotting trajectories and diagrams in the examples |

```bash
pip install "tsdynamics[plot]"
```

## Verify the install

```python
import tsdynamics as ts

print(ts.__version__)
print(ts.registry.families())   # {'ode': 120, 'dde': 5, 'map': 26}

traj = ts.Henon().iterate(steps=100)
print(traj)                            # Trajectory(n_steps=100, dim=2, ...)
```

If `ts.Lorenz().integrate(final_time=1.0)` also succeeds, the engine is wired
up correctly.

## Next

[**02 · First trajectory**](first-trajectory.md) — a complete worked
example across all three system families.
