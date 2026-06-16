---
description: The catalogue of 149 built-in dynamical systems — 118 continuous flows, 5 delay systems, 26 discrete maps — with auto-generated per-system pages.
---

<span class="ts-kicker">Systems</span>

# Systems

TSDynamics ships **149 built-in systems** across three families:

| Family | Base class | Count | Time | Engine |
| ------ | ---------- | ----- | ---- | ------ |
| [Continuous](continuous/index.md) | `ContinuousSystem` | 118 | continuous | Rust engine |
| [Delay](delay/index.md) | `DelaySystem` | 5 | continuous, with memory | Rust engine |
| [Discrete](discrete/index.md) | `DiscreteMap` | 26 | discrete | Rust engine |

Every built-in is importable from the top level:

```python
import tsdynamics as ts

ts.Lorenz()         # continuous
ts.MackeyGlass()    # delay
ts.Henon()          # discrete
```

or discoverable programmatically through the
[registry](../reference/registry.md):

```python
from tsdynamics import registry

registry.families()                          # {'ode': 118, 'dde': 5, 'map': 26}
[e.name for e in registry.all_systems(family="dde")]
# ['MackeyGlass', 'IkedaDelay', 'SprottDelay', 'ScrollDelay', 'PiecewiseCircuit']
```

## How the pages below are built

The per-system pages under each family section are **generated from the
code at build time** — there are no hand-written equation pages to drift
out of sync. For each registered system, the generator renders:

- the governing equations, derived from the actual symbolic `_equations`
  (or `_step`) definition that the integrator compiles,
- the default parameters, dimension, and named variables,
- a phase-portrait or orbit figure integrated at build time,
- the literature reference and known Lyapunov values, when the class
  declares them.

## Adding a system = writing the class

There is no separate documentation step. Subclass the right base class
(see [the mental model](../start/concepts.md)), and the new class
auto-registers; its documentation page, registry entry, and bulk test
coverage all follow from the class definition itself. The full
checklist lives in [Contributing](../project/contributing.md).
