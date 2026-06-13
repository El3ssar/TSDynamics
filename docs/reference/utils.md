---
description: API reference for tsdynamics.utils — the staticjit decorator and sagitta-based timestep estimation.
---

<span class="ts-kicker">Reference</span>

# Utilities

## JIT decorator

::: tsdynamics.utils.general.staticjit

## Timestep estimation

Sagitta-based selection of an output `dt` for a sampled trajectory: pick
the largest stride whose mid-point deviation from the chord (the
*sagitta*) stays below a curvature tolerance.

::: tsdynamics.utils.sagitta_dt.estimate_dt_from_sagitta

::: tsdynamics.utils.sagitta_dt.SagittaDt
