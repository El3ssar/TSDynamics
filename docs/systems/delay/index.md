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
exps = mg.lyapunov_spectrum(k=1, dt=0.5, ic=traj.y[-1])   # start from its end state
```

`lyapunov_spectrum` uses a constant past built from `ic`; handing it the
end state of a settled run starts the measurement *on* the attractor and
avoids trivial exponents. A DDE has infinitely many exponents — `k`
chooses how many leading ones to estimate (default 1).

## Tolerances

DDE defaults are `rtol=atol=1e-3`, looser than the ODE defaults (`1e-9`/`1e-12`)
— and that is deliberate.

!!! info "Why the DDE default did not follow the v6 ODE bump"
    The ODE default tightened in v6 to compensate for native dense output: the
    adaptive stepper no longer lands on every output sample, so `rtol` had to
    take over the accuracy the forced landing used to supply. **The DDE method of
    steps never had dense output** — it still lands on every sample — so it lost
    nothing and needs no compensation.

    The tolerance is in fact largely *inert* here, because `dt` already bounds
    the internal step below the natural error. Measured over all six built-in
    DDEs at the default `dt=0.02` to `T=10`, five return a **bit-identical**
    final state at `rtol=1e-3` and at `rtol=1e-9`; only `IkedaDelay` — the one
    system whose step is genuinely tolerance-bound — differs, costing 1.4× for a
    6× accuracy gain.

    Tightening is **safe**: all six complete at `rtol=1e-12, atol=1e-15` over
    `T=500`. (An earlier version of this page warned that tight tolerances
    "routinely stall the DDE solver". That was true of the v2 JiTCDDE backend;
    it is not true of the Rust method-of-steps engine, and the claim has been
    withdrawn.) You simply gain little by default, so `1e-3` stays the starting
    point — tighten with evidence, as always.

## Lowering note

Unlike ODEs, a DDE's lowered tape depends on **all** of its parameters
(delays shape the history buffer), so each parameter set is re-lowered.
Parameter sweeps over DDEs are correspondingly more expensive than over
ODEs — see [the compilation pipeline](../../theory/compilation.md).

## See also

- [First trajectory](../../start/first-trajectory.md) — the Mackey–Glass walkthrough
- [Lyapunov spectra](../../analysis/lyapunov.md) — all three families side by side
