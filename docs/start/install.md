---
description: Install TSDynamics with pip or uv, set up a C compiler for JIT compilation, and choose optional extras.
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

## The C compiler requirement

Continuous and delay systems are compiled to native code by JiTCODE /
JiTCDDE, which generate C from your symbolic equations and build it with
your platform's compiler. A working C toolchain must be on the path:

=== "Linux"

    ```bash
    sudo apt-get install build-essential python3-dev
    ```

    (or the equivalent `gcc` + Python headers package for your distribution)

=== "macOS"

    ```bash
    xcode-select --install
    ```

=== "Windows"

    Install the MSVC Build Tools (the "Desktop development with C++"
    workload is sufficient).

Discrete maps do not need the C toolchain — they are JIT-compiled by
Numba, which ships its own backend.

!!! note "First call is slow, every later call is fast"
    The first `integrate()` on a system class compiles a shared library
    and caches it under `~/.cache/tsdynamics/`. Subsequent calls — even in
    new Python sessions — load the cached binary. See the
    [compilation pipeline](../theory/compilation.md) for details.

## Optional extras

| Extra | Installs | When you want it |
| ----- | -------- | ---------------- |
| `tsdynamics[plot]` | `matplotlib` | Plotting trajectories and diagrams in the examples |
| `tsdynamics[diffsol]` | `pydiffsol` | **Experimental** Rust-backed ODE solver backend |

```bash
pip install "tsdynamics[plot]"
```

## The Rust engine accelerator

The zero-warmup Rust engine (`backend="interp"` / `backend="jit"`) ships as a
separate, optional, **prebuilt** distribution — no compiler needed. It is gated
during the v2→Rust migration and is built as a cross-platform wheel rather than
published to the index yet; see [Packaging & distribution](../theory/packaging.md)
for how it is shaped, why, and the path to folding it into a single `tsdynamics`
wheel.

## Verify the install

```python
import tsdynamics as ts

print(ts.__version__)
print(ts.registry.families())   # {'ode': 118, 'dde': 5, 'map': 26}

traj = ts.Henon().iterate(steps=100)   # no C compiler needed for maps
print(traj)                            # Trajectory(n_steps=100, dim=2, ...)
```

If `ts.Lorenz().integrate(final_time=1.0)` also succeeds, the C toolchain
is correctly set up.

## Next

[**02 · First trajectory**](first-trajectory.md) — a complete worked
example across all three system families.
