---
template: home.html
title: TSDynamics — dynamical systems & chaos analysis for Python
description: TSDynamics — 151 built-in dynamical systems, compiled integration, and chaos analysis for Python.
hide:
  - navigation
  - toc
---

# TSDynamics

**151 built-in dynamical systems. Compiled integration. Chaos analysis.**

TSDynamics is a Python library for studying dynamical systems: continuous
flows, delay equations, stochastic systems, and discrete maps, all behind one
interface. You define the math in a single symbolic method; the library lowers
it to a native Rust engine and gives you trajectories,
Lyapunov spectra and chaos indicators, fixed and periodic orbits, bifurcation
diagrams and Poincaré sections, fractal dimensions and delay embeddings,
recurrence/RQA, entropy and complexity, surrogate tests, and the full
attractor / basin / continuation suite.

```python
import tsdynamics as ts

lor = ts.Lorenz()
traj = lor.integrate(final_time=100.0, dt=0.01)

traj["x"]                  # named component access
lor.lyapunov_spectrum()    # ≈ [0.906, 0, -14.57]
```

The right-hand side is lowered to the Rust engine in-process and runs with no
warmup — the first call is as fast as every later one.

---

## Why TSDynamics

<div class="grid cards" markdown>

- **A native engine, zero warmup**

    ---

    ODEs, DDEs and discrete maps all lower to one Rust engine — an SSA-tape
    interpreter (the default) with a Cranelift JIT alongside. There is no
    compile step and no warmup: the first call runs at full speed.

- **151 systems out of the box**

    ---

    120 continuous flows, 5 delay systems, and 26 discrete maps, from
    Lorenz and Rössler to Mackey–Glass and Hénon — each with literature
    defaults, named variables, and known Lyapunov values where published.

- **One protocol, every analysis**

    ---

    Everything that steps — a map, a flow, a Poincaré section of a flow —
    implements the same `System` protocol, so the analysis toolkit composes:
    an orbit diagram over a Poincaré map *is* a bifurcation diagram.

- **Define a system in ten lines**

    ---

    Subclass one of the family base classes, declare a `params` dict and
    `dim`, write one method. Your class auto-registers, the test suite sweeps
    it, and its documentation page is generated from the code.

</div>

---

## Where to go

| You want to | Go to |
| ----------- | ----- |
| Install and run your first trajectory | [Start](start/index.md) |
| Browse the built-in systems | [Systems](systems/index.md) |
| Lyapunov, chaos, dimensions, basins, recurrence, surrogates | [Analysis](analysis/index.md) |
| Follow an end-to-end walkthrough (equations → basins) | [Tutorials](tutorials/index.md) |
| Turn trajectories into features (spectra, filters) | [Transforms](transforms/index.md) |
| Understand the engine pipeline and the math | [Theory](theory/index.md) |
| Look up an exact signature | [Reference](reference/index.md) |
| Contribute a system | [Project](project/contributing.md) |
