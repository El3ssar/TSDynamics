---
description: Finding fixed points of discrete maps by multi-start Newton iteration, with eigenvalue-based stability classification.
---

<span class="ts-kicker">Analysis · 05</span>

# Fixed points

`fixed_points` finds the solutions of $f(x) = x$ for a discrete map and
classifies their stability from the Jacobian eigenvalues.

```python
import tsdynamics as ts

fps = ts.fixed_points(ts.Henon())
fps
# [FixedPoint([-1.131354  -0.339406], unstable, |λ|max=3.2598),
#  FixedPoint([ 0.631354  0.189406], unstable, |λ|max=1.9237)]
```

## How it works

1. **Seeding** — `n_seeds` random points are drawn from a search `box`
   (default: the orbit's bounding box padded by 50 %), plus points sampled
   from a short orbit of the map itself, which biases the search toward
   the attractor region.
2. **Newton** — each seed runs Newton's method on $g(x) = f(x) - x$,
   using the map's exact `_jacobian` (no finite differences), up to
   `max_iter` iterations or residual `tol`.
3. **Dedup & classify** — converged roots closer than `dedup_tol` are
   merged; each survivor is classified **stable iff every** $|\lambda_i| < 1$
   for the eigenvalues $\lambda_i$ of $J(x^*)$.

Pass `seed=` for reproducible seeding, and a `box=((lo,...), (hi,...))`
to search a specific region (roots outside the box are discarded).

## Checking against the analytic answer: Hénon

For the Hénon map $x' = 1 - a x^2 + y$, $y' = b x$, a fixed point
satisfies $a x^2 + (1 - b)x - 1 = 0$, so

$$
x^* = \frac{-(1 - b) \pm \sqrt{(1 - b)^2 + 4a}}{2a},
\qquad y^* = b\,x^*.
$$

With the classic $a = 1.4$, $b = 0.3$: $x^* \approx 0.6314$ and
$x^* \approx -1.1314$ — matching the numeric output above. Both are
saddles (one eigenvalue inside the unit circle, one outside), which is
exactly what the strange attractor needs: orbits are attracted along the
stable direction and stretched along the unstable one.

```python
import numpy as np

a, b = 1.4, 0.3
x_exact = (-(1 - b) + np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)
assert abs(fps[1].x[0] - x_exact) < 1e-10
```

## The `FixedPoint` record

```python
fp = fps[0]
fp.x              # the fixed point, shape (dim,)
fp.eigenvalues    # eigenvalues of the Jacobian at fp.x
fp.stable         # True iff all |λ| < 1
```

!!! note "Maps only, for now"
    `fixed_points` requires a `DiscreteMap` — it relies on the explicit
    `_jacobian`. For equilibria of flows, the analogous problem is
    $f(x) = 0$ on the right-hand side; a flow-aware version is a natural
    extension and not yet implemented.

## See also

- [Orbit & bifurcation diagrams](orbit-diagrams.md) — where fixed points are born and lost
- [Reference · Analysis](../reference/analysis.md) — `fixed_points` / `FixedPoint` signatures
