---
description: API reference for tsdynamics._utils — the shared output-grid helper.
---

<span class="ts-kicker">Reference</span>

# Utilities

## Output grid

The single shared builder of the uniform, endpoint-inclusive output time grid
(`[t0, t0+dt, …, tf]`) that every integrator samples its results onto.

::: tsdynamics._utils.grids.make_output_grid
