---
description: A complete TSDynamics walkthrough — define a bistable Duffing oscillator, integrate it, then find its attractors, paint their basins, and quantify the boundary.
---

<span class="ts-kicker">Tutorial</span>

# From equations to basins

This is the whole library in one sitting. We start from the
[definition contract](../start/concepts.md) — `params`, `dim`, one symbolic
`_equations` method — and end with the *global* stability picture of a
multistable system: where its attractors are, which initial conditions reach
each one, and how tangled the boundary between them is.

The example is the damped **two-well Duffing oscillator**

$$\ddot{x} = x - x^3 - \delta\,\dot{x},$$

a unit mass in the double-well potential $V(x) = -\tfrac12 x^2 + \tfrac14 x^4$
with linear damping $\delta$. The two wells sit at $x = \pm 1$; with damping
every trajectory eventually rolls into one of them. *Which* one is the
question basins answer.

## 1. Write the equations

A system is a class: the parameters, the dimension, and the right-hand side as
a first-order system $\dot{x}=y,\ \dot{y}=x-x^3-\delta y$. That is the entire
contract — compilation, caching, output grids and provenance are handled for
you.

```python
import numpy as np
import tsdynamics as ts
from tsdynamics import data

class DuffingTwoWell(ts.ContinuousSystem):
    """Damped two-well Duffing oscillator: x'' = x - x**3 - delta * x'."""

    params = {"delta": 0.3}
    dim = 2
    variables = ("x", "y")

    @staticmethod
    def _equations(Y, t, *, delta):
        x, y = Y(0), Y(1)
        return (y, x - x**3 - delta * y)

sys = DuffingTwoWell()
```

!!! note "Symbolic right-hand side"
    `_equations` is read symbolically (SymEngine), so use plain arithmetic and
    `symengine` functions — never NumPy or `math` inside it. `Y(0)`, `Y(1)`
    are the state components. See [the mental model](../start/concepts.md).

## 2. Integrate — see the bistability

Two nearby initial conditions can fall into *different* wells. Integrate from
each and read off the end state:

```python
sys.integrate(final_time=60.0, dt=0.05, ic=[ 1.5, 0.5]).y[-1]   # ≈ [ 1.0, 0.0]
sys.integrate(final_time=60.0, dt=0.05, ic=[-1.5, 0.5]).y[-1]   # ≈ [-1.0, 0.0]
```

Both settle onto a fixed point — the bottom of a well at $x=\pm 1,\ y=0$. That
two stable states coexist is what makes the *basin* question meaningful.

## 3. Find the attractors

[`find_attractors`](../analysis/basins.md) drives the flow from a grid of
seeds over a [`Box`](../reference/data.md) of state space and follows each
until it recurrently revisits the same cells — the signature of having reached
an attractor (Datseris & Wagemakers, 2022):

```python
region = data.Box(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

att = ts.find_attractors(sys, region, resolution=40, n_seeds=200,
                         dt=0.5, max_steps=2000, seed=0)
att                       # AttractorSet(2 attractors, 0/200 diverged)
[a.center for a in att]   # ≈ [[-1.0, 0.0], [1.0, 0.0]]
```

Two attractors, located at the well bottoms, and nothing escaped the box.

## 4. Paint the basins

[`basins_of_attraction`](../analysis/basins.md) labels a
[`Grid`](../reference/data.md) of initial conditions by *which* attractor each
one reaches — the basin map:

```python
grid = data.Grid(np.array([-2.0, -2.0]), np.array([2.0, 2.0]), (60, 60))
basins = ts.basins_of_attraction(sys, grid, dt=0.5, max_steps=2000)

basins.n_attractors     # 2
basins.labels.shape     # (60, 60) — integer attractor id per cell
basins.fractions        # {1: ~0.53, 2: ~0.47}
```

`labels` is a plain integer array, ready to plot:

```python
# import matplotlib.pyplot as plt
# plt.imshow(basins.labels.T, origin="lower", extent=[-2, 2, -2, 2], cmap="coolwarm")
# plt.xlabel("x"); plt.ylabel("y")
```

For a higher-dimensional flow you would image a 2-D *slice* by passing a
separate `recurrence` box — see the [basins reference](../analysis/basins.md).

## 5. Quantify the boundary

How predictable is the outcome near the boundary? Two label-image diagnostics
(no further integration — they read the painted grid):

```python
ts.basin_fractions(sys, region, n=400, dt=0.5, max_steps=2000, seed=0)
# BasinFractions({1: 0.53, 2: 0.47}, diverged=0, n=400)   — basin stability

ts.basin_entropy(basins)
# BasinEntropy(Sb=0.237, Sbb=0.526, fractal_boundary=False)

ts.uncertainty_exponent(basins)
# UncertaintyExponent(alpha=0.88, D0=1.12, R2=0.998)
```

The fractions are close to the symmetric $\tfrac12,\tfrac12$ — basin
*stability* in the sense of Menck et al. (2013). The basin entropy (Daza et
al., 2016) reports `fractal_boundary=False`: at this damping the boundary is a
smooth curve, and the uncertainty exponent (Grebogi et al., 1983)
$\alpha \approx 0.88$ is close to $1$, i.e. boundary dimension
$D_0 = D - \alpha \approx 1.1$ — barely above a line.

!!! tip "When the boundary turns fractal"
    Smoothness is not guaranteed. Drive a system with competing attractors
    hard enough — the textbook case is a magnetic pendulum over three magnets —
    and the boundary becomes fractal and even *Wada* (every boundary point
    touches all basins at once). Then `basin_entropy` reports
    `fractal_boundary=True` (its boundary entropy exceeds $\ln 2$) and
    [`wada_property`](../analysis/basins.md) detects the Wada structure. The
    [Attractors & basins](../analysis/basins.md) page covers those measures and
    the continuation / tipping / resilience tools that track them across a
    parameter.

## What you built

From three lines of math you produced the complete multistability portrait —
attractors, basins, basin stability, and a quantified boundary — every result
carrying its provenance in `traj.meta`. That is the path the whole toolkit is
designed around: define the system once, then compose
[analysis](../analysis/index.md) on top.

## See also

- [Attractors & basins](../analysis/basins.md) — the full basin/continuation API
- [The mental model](../start/concepts.md) — the definition contract in depth
- [Lyapunov spectra](../analysis/lyapunov.md) — is a given attractor chaotic?
