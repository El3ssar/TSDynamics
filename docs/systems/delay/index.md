---
description: The 5 built-in delay differential systems — DDEs run on the native Rust engine, with history functions and DDE Lyapunov spectra.
---

<span class="ts-kicker">Systems · Delay</span>

# Delay systems

Five **delay differential equations** (DDEs), subclasses of
[`DelaySystem`](../../reference/base.md): `MackeyGlass`, `IkedaDelay`,
`SprottDelay`, `ScrollDelay`, and `PiecewiseCircuit`. In a DDE the
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

mg = ts.MackeyGlass()
hist = lambda s: [1.0 + 0.1 * np.sin(0.2 * s)]
traj = mg.integrate(final_time=500.0, dt=0.5, history=hist)
```

Without `history`, a constant past equal to the resolved `ic` is used.
Avoid constant pasts at equilibria — the trajectory simply sits there.

## Lyapunov spectra: the two-step pattern

DDE Lyapunov spectra — a capability few tools offer — start from a constant
past rather than an arbitrary history function. The supported pattern is
therefore *integrate first, then measure*:

```python
traj = mg.integrate(final_time=500.0, dt=0.5, history=hist)   # reach the attractor
exps = mg.lyapunov_spectrum(n_exp=1, dt=0.5, ic=traj.y[-1])   # start from its end state
```

`lyapunov_spectrum` uses a constant past built from `ic`; handing it the
end state of a settled run starts the measurement *on* the attractor and
avoids trivial exponents. A DDE has infinitely many exponents — `n_exp`
chooses how many leading ones to estimate (default 1).

## Tolerances

DDE defaults are `rtol=atol=1e-3`, looser than the ODE defaults — and that
is deliberate:

!!! warning "Do not use ODE-style tight tolerances"
    Values like `rtol=1e-6, atol=1e-9` routinely stall the DDE solver, and
    in Lyapunov runs they can corrupt the variational state before the
    first renormalization, producing `inf`/`nan` exponents. Start at
    `1e-3` and tighten only with evidence.

## Lowering note

Unlike ODEs, a DDE's lowered tape depends on **all** of its parameters
(delays shape the history buffer), so each parameter set is re-lowered.
Parameter sweeps over DDEs are correspondingly more expensive than over
ODEs — see [the compilation pipeline](../../theory/compilation.md).

## See also

- [First trajectory](../../start/first-trajectory.md) — the Mackey–Glass walkthrough
- [Lyapunov spectra](../../analysis/lyapunov.md) — all three families side by side
