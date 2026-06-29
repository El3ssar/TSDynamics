---
description: The rigorous interval-Newton / Krawczyk fixed-point finder — how it brackets every root in a box, the benchmark vs multi-start Newton, and where the win comes from.
---

<span class="ts-kicker">Theory · 08</span>

# Rigorous fixed points (`method="interval"`)

[`fixed_points`](../analysis/fixed-points.md) defaults to **multi-start
Newton**: scatter many seeds across a box, run a local solve from each, keep the
distinct roots. It is fast and general, but it has one structural weakness — a
finite number of seeds can leave a root's basin unsampled, so it can **silently
miss** an equilibrium. There is no certificate that the returned set is complete.

`method="interval"` is the rigorous alternative. The **Krawczyk operator**
brackets *all* roots of the residual inside a search box by interval
branch-and-prune, with an existence-and-uniqueness certificate per sub-box.

## The Krawczyk operator

For a box $[\mathbf{x}] = [\underline{\mathbf{x}}, \overline{\mathbf{x}}]$ with
midpoint $m$, residual $g$, interval Jacobian enclosure $G([\mathbf{x}])$, and a
preconditioner $Y \approx J(m)^{-1}$, define

$$
K([\mathbf{x}]) \;=\; m \;-\; Y\,g(m) \;+\; \bigl(I - Y\,G([\mathbf{x}])\bigr)\,([\mathbf{x}] - m).
$$

Then (Krawczyk 1969; Neumaier 1990):

- if $K([\mathbf{x}]) \cap [\mathbf{x}] = \varnothing$, the box holds **no** root → **prune**;
- if $K([\mathbf{x}]) \subset \operatorname{int}[\mathbf{x}]$, the box holds a
  **unique** root (existence *and* uniqueness) → **accept** (then contract to
  machine precision and polish with one Newton step);
- otherwise contract to $K \cap [\mathbf{x}]$, or **bisect** the widest axis and
  recurse.

Because every sub-box is either pruned with a no-root proof or accepted with a
unique-root proof, the union of accepted boxes is a *complete* enumeration of the
roots in the original box (up to floating-point round-off — see the caveat
below). That is the property multi-start cannot offer.

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
powers) cover the entire built-in catalogue's flows and its analytic maps.

A kernel whose operations the interval engine cannot enclose — a comparison
(`<`), a modulo (`%`), a non-integer/variable power — raises
`InvalidInputError` at build time, pointing back at `method="newton"`. The
interval method is purely **additive**: `"newton"`/`"sd"`/`"dl"` are unchanged.

## Benchmark — speed *and* completeness

Measured locally (CPython, no warm cache), interval vs multi-start Newton on the
same box. Both find the same roots to machine precision; interval is faster on
every case **and** rigorously complete.

| System (roots in box)  | interval | Newton (`seed=0`) | speed-up |
|------------------------|---------:|------------------:|---------:|
| Hénon map (2)          |  ~3 ms   |  ~33 ms           |  ~11×    |
| Tinkerbell map (2)     | ~17 ms   |  ~62 ms           |  ~3.7×   |
| Rössler flow (2)       | ~22 ms   | ~192 ms           |  ~8.8×   |
| Lorenz flow (3)        | ~138 ms  | ~229 ms           |  ~1.7×   |

The completeness gain is starkest where a system has **many** equilibria. The
Thomas flow has **27** equilibria in $[-6, 6]^3$:

| Method                   | equilibria found | time     |
|--------------------------|-----------------:|---------:|
| **interval** (rigorous)  | **27 — all**     | ~0.9 s   |
| Newton, `n_seeds=200`    | 23               | ~1.3 s   |
| Newton, `n_seeds=500`    | 26               | ~3.0 s   |
| Newton, `n_seeds=1000`   | 27               | ~6.6 s   |

Multi-start needs ~7× the time to *match* the interval method's completeness, and
at the default seed count it silently misses four equilibria — exactly the
failure mode the certificate rules out.

The interval result is **deterministic** (no `seed`): the box decomposition is a
function of the math alone.

## Where the win comes from — and its limits

The speed-up is real even in pure Python because the work is *targeted*: the
operator prunes empty regions wholesale and contracts quadratically near a root,
visiting only tens to a few thousand boxes, instead of running hundreds of full
Newton trajectories. (A compiled interval kernel would be faster still, but the
Python implementation already beats the multi-start default.)

Two honest caveats:

1. **Round-off, not certified bounds.** The interval arithmetic uses ordinary
   `float` operations (no outward directed rounding), so the bracketing is
   rigorous only up to floating-point round-off, and roots are *recovered* to
   machine precision rather than returned as proven enclosures. A fully
   *certified* result (provably correct outward-rounded bounds) would need a
   directed-rounding interval kernel — best done engine-side in Rust, a larger
   future project. For the practical task of *finding every fixed point in a
   box*, the current implementation is both complete and machine-precise.

2. **Interior only.** The uniqueness test requires a root strictly inside the
   box, so a root lying exactly on a box face is not certified. Choose `region`
   so the equilibria sit in its interior (pad the box slightly past where you
   expect roots).

## References

- R. Krawczyk, *Newton-Algorithmen zur Bestimmung von Nullstellen mit
  Fehlerschranken*, **Computing** 4 (1969), 187–201.
- A. Neumaier, *Interval Methods for Systems of Equations*, Cambridge University
  Press, 1990.
- R. E. Moore, R. B. Kearfott & M. J. Cloud, *Introduction to Interval
  Analysis*, SIAM, 2009.
