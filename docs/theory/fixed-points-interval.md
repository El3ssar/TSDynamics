---
description: The numerical interval-Newton / Krawczyk fixed-point search — deterministic branch-and-prune, comparison with multi-start Newton, and its limits without directed rounding.
---

<span class="ts-kicker">Theory · 08</span>

# Interval fixed-point search (`method="interval"`)

[`fixed_points`](../analysis/fixed-points.md) defaults to **multi-start
Newton**: scatter many seeds across a box, run a local solve from each, keep the
distinct roots. It is fast and general, but it has one structural weakness — a
finite number of seeds can leave a root's basin unsampled, so it can **silently
miss** an equilibrium. There is no certificate that the returned set is complete.

`method="interval"` searches the required region systematically with a
**Krawczyk operator** and interval branch-and-prune. This implementation uses
ordinary floating-point interval arithmetic, without outward rounding, and may
polish unresolved small boxes by residual checks. Its results are numerical
approximations: it provides neither formal certificates nor a guarantee that
every root has been found.

## The Krawczyk operator

For a box $[\mathbf{x}] = [\underline{\mathbf{x}}, \overline{\mathbf{x}}]$ with
midpoint $m$, residual $g$, interval Jacobian enclosure $G([\mathbf{x}])$, and a
preconditioner $Y \approx J(m)^{-1}$, define

$$
K([\mathbf{x}]) \;=\; m \;-\; Y\,g(m) \;+\; \bigl(I - Y\,G([\mathbf{x}])\bigr)\,([\mathbf{x}] - m).
$$

With exact interval enclosures, the mathematical test is (Krawczyk 1969;
Neumaier 1990):

- if $K([\mathbf{x}]) \cap [\mathbf{x}] = \varnothing$, the box holds **no** root → **prune**;
- if $K([\mathbf{x}]) \subset \operatorname{int}[\mathbf{x}]$, the box holds a
  **unique** root (existence *and* uniqueness) → **accept** (then contract to
  machine precision and polish with one Newton step);
- otherwise contract to $K \cap [\mathbf{x}]$, or **bisect** the widest axis and
  recurse.

A certified implementation can use these tests to prove exclusion and
existence/uniqueness. TSDynamics uses the same numerical structure, but its
plain-float arithmetic does not ensure the enclosures required for those proofs.
Finite subdivision limits and residual-based recovery of small boxes also
prevent a completeness claim.

### The interval right-hand side

The operator needs an **interval extension** of the residual and its Jacobian.
TSDynamics builds both in one pass with **forward-mode automatic differentiation
over intervals** (`IntervalJet`): a value-interval carrying a gradient-interval
vector, pushed through the system kernel. The Jacobian is then the AD gradient of
the residual *itself* — never a symbolically differentiated form (which would
explode into reciprocal or variable powers for many flows). For a **map** the
seeded jets go straight through the pure-Python `_step`; for a **flow** they walk
the SymEngine right-hand-side expression tree. The supported elementary
functions (`sin`, `cos`, `exp`, `log`, `sqrt`, `cosh`, `tanh`, `abs`, integer
powers) cover many built-in flows and analytic maps.

A kernel whose operations the interval engine cannot enclose — a comparison
(`<`), a modulo (`%`), a non-integer/variable power — raises
`InvalidInputError` at build time, pointing back at `method="newton"`. The
interval method is purely **additive**: `"newton"`/`"sd"`/`"dl"` are unchanged.

## Historical benchmark — speed and roots recovered

Measured locally (CPython, no warm cache), interval vs multi-start Newton on the
same box. Both recovered the same roots within numerical tolerance; interval
was faster in these measurements. These historical timings are examples,
not a performance or completeness guarantee.

| System (roots in box)  | interval | Newton (`seed=0`) | speed-up |
|------------------------|---------:|------------------:|---------:|
| Hénon map (2)          |  ~3 ms   |  ~33 ms           |  ~11×    |
| Tinkerbell map (2)     | ~17 ms   |  ~62 ms           |  ~3.7×   |
| Rössler flow (2)       | ~22 ms   | ~192 ms           |  ~8.8×   |
| Lorenz flow (3)        | ~138 ms  | ~229 ms           |  ~1.7×   |

Systematic subdivision can recover more roots than a modest random seed count.
For the Thomas flow in $[-6, 6]^3$, the historical run reported:

| Method                   | equilibria found | time     |
|--------------------------|-----------------:|---------:|
| **interval**            | **27**           | ~0.9 s   |
| Newton, `n_seeds=200`    | 23               | ~1.3 s   |
| Newton, `n_seeds=500`    | 26               | ~3.0 s   |
| Newton, `n_seeds=1000`   | 27               | ~6.6 s   |

In this run, multi-start took about 7× longer to recover the same 27 roots.
Neither comparison establishes that an arbitrary search box has been exhausted
with a formal proof.

The interval result is **deterministic** (no `seed`): the box decomposition is a
function of the math alone.

## Where the win comes from — and its limits

The speed-up is real even in pure Python because the work is *targeted*: the
operator prunes empty regions wholesale and contracts quadratically near a root,
visiting only tens to a few thousand boxes, instead of running hundreds of full
Newton trajectories. (A compiled interval kernel would be faster still, but the
Python implementation already beats the multi-start default.)

Limits:

1. **No certified bounds or completeness guarantee.** Ordinary `float`
   operations do not enclose rounding error. Accepted roots are numerical
   approximations checked by their residual, rather than proven enclosures.
   Certification requires outward rounding throughout, validated elementary
   functions, and explicit handling of every unresolved box.

2. **Interior roots are easier to isolate.** The Krawczyk inclusion test
   requires a root strictly inside its box. Choose `region`
   so the equilibria sit in its interior (pad the box slightly past where you
   expect roots).

## References

- R. Krawczyk, *Newton-Algorithmen zur Bestimmung von Nullstellen mit
  Fehlerschranken*, **Computing** 4 (1969), 187–201.
- A. Neumaier, *Interval Methods for Systems of Equations*, Cambridge University
  Press, 1990.
- R. E. Moore, R. B. Kearfott & M. J. Cloud, *Introduction to Interval
  Analysis*, SIAM, 2009.
