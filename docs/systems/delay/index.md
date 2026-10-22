---
description: Built-in delay differential systems — DDEs run on the native Rust engine, with history functions and DDE Lyapunov spectra.
---

<span class="ts-kicker">Systems · Delay</span>

# Delay systems

**Delay differential equations** (DDEs) subclass
[`DelaySystem`](../../reference/base.md): `MackeyGlass`, `IkedaDelay`,
`SprottDelay`, `ScrollDelay`, `PiecewiseCircuit` and `VossDelay`. In a DDE the
derivative depends on the state at *earlier* times, so even a
one-dimensional equation like Mackey–Glass can be chaotic — its effective
state is a whole function over the delay interval.

In `_equations`, delayed access is written `y(i, t - tau)`:

```python
@staticmethod
def _equations(y, t, *, beta, gamma, tau, n):
    return [beta * y(0, t - tau) / (1 + y(0, t - tau) ** n) - gamma * y(0)]
```

The class lists its delay parameters in `_delay_params` (default
`("tau",)`), which sizes the history buffer.

## History functions

The initial condition of a DDE is a function over `s ≤ 0`, not a point.
Pass it as a callable returning a length-`dim` sequence:

```python
import numpy as np
import tsdynamics as ts

mg = ts.systems.MackeyGlass()
hist = lambda s: [1.0 + 0.1 * np.sin(0.2 * s)]
traj = mg.run(final_time=500.0, dt=0.5, history=hist)
```

Without `history`, a constant past equal to the resolved `ic` is used.
Avoid constant pasts at equilibria — the trajectory simply sits there.

## Lyapunov spectra: the two-step pattern

DDE Lyapunov spectra start from a constant
past rather than an arbitrary history function. The supported pattern is
therefore *integrate first, then measure*:

```python
traj = mg.run(final_time=500.0, dt=0.5, history=hist)   # reach the attractor
exps = ts.analysis.lyapunov_spectrum(mg, k=1, dt=0.5, ic=traj.y[-1])  # from its end state
```

`lyapunov_spectrum` uses a constant past built from `ic`; handing it the
end state of a settled run supplies a candidate constant history. It does not
transfer the trajectory's complete history, so use a sufficient transient and
check dependence on the starting history. A DDE has infinitely many exponents — `k`
chooses how many leading ones to estimate (default 1).

## Tolerances

DDE defaults are `rtol=atol=1e-3`. The sampling grid also bounds internal steps,
and history interpolation affects accuracy. Compare smaller `dt` and tighter
tolerances for the regime of interest rather than assuming either setting alone
controls convergence. Lyapunov calculations need their own history-resolution
and averaging-window checks.

## Lowering note

Unlike ODEs, a DDE's lowered tape depends on **all** of its parameters
(delays shape the history buffer), so each parameter set is re-lowered.
Parameter sweeps over DDEs are correspondingly more expensive than over
ODEs — see [the compilation pipeline](../../theory/compilation.md).

## See also

- [First trajectory](../../start/first-trajectory.md) — the Mackey–Glass walkthrough
- [Lyapunov spectra](../../analysis/lyapunov.md) — all three families side by side
