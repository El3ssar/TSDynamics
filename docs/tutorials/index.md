---
description: End-to-end journeys through TSDynamics — each tutorial wires several tools together on one system, from the equations you write to a finished, literature-grounded result.
---

<span class="ts-kicker">Tutorials · Overview</span>

# Tutorials

End-to-end journeys through the library. Where the [analysis pages](../analysis/index.md)
each isolate a single tool and the [system catalogue](../systems/index.md) each
document a single model, a tutorial wires several together on one system — from
the equations you write to a finished result you could put in a paper, every
number reproducible and every claim traceable to the literature.

Each one is self-contained and runnable end to end: paste the blocks in order,
keep the pinned initial conditions, and you get the numbers and figures shown.
Read them in any order — but if you are new, the first three trace the arc most
nonlinear-dynamics courses follow, from *what is chaos* to *how do I find it in
real data*.

## The journeys

### Understanding chaos

- [**Anatomy of a chaotic attractor**](chaotic-attractor.md) — define the Lorenz
  system, integrate it, plot the butterfly, then *certify* the chaos: watch two
  neighbouring states diverge, compute the Lyapunov spectrum, read the
  Kaplan–Yorke dimension, and cross-check it against the correlation dimension.
  The vocabulary the rest of the toolkit is built on.
- [**The road to chaos**](bifurcations.md) — where does chaos come *from*? Sweep
  the logistic map's growth rate and watch a fixed point period-double onto the
  textbook onsets $r_1 = 3$ and $r_2 = 1+\sqrt6$, cascade into chaos, and open a
  period-3 window. Then do the same for a *flow* — a Rössler bifurcation diagram
  through a Poincaré section.

### Working from data

- [**Reconstruction from one signal**](reconstruction.md) — you have a single
  recorded channel $x(t)$ and no equations. Rebuild the attractor by delay
  embedding, choose the delay and dimension with principled heuristics, then
  measure the fractal dimension and a data-driven Lyapunov exponent from the
  recording alone — and check both against the ground truth.
- [**Poincaré sections & return maps**](poincare-return-maps.md) — section a flow
  to turn it into a discrete map, then expose the one-dimensional map hiding
  inside a strange attractor — the famous Lorenz $z$-maxima cusp.

### Global structure & noise

- [**Basins & multistability**](basins-multistability.md) — a bistable
  oscillator has two possible fates. Find both attractors, paint the basins that
  decide which initial conditions reach each one, quantify the boundary, then
  tilt the potential and watch a basin annihilate at a tipping point.
- [**Noise-driven dynamics**](noise-driven.md) — add noise. Integrate a single
  stochastic path, gather ensemble statistics, watch noise drive Kramers escape
  between two wells, and check it against the analytic laws.

## How to read them

Every tutorial follows the same discipline:

- **The system is defined once.** A class with `params`, `dim`, and a symbolic
  right-hand side — the [definition contract](../start/concepts.md) — or a
  built-in from the [catalogue](../systems/index.md). Everything downstream
  composes over that one object.
- **Every number is reproducible.** Snippets pass an explicit `ic=` (or `seed=`
  for anything stochastic), so the printed values are exactly what the code
  produces. Paste and run; you should see the same figures.
- **Every claim cites its source.** The onsets, exponents and dimensions quoted
  are literature values, and the See-also links point to the [analysis
  reference](../analysis/index.md) page where each tool is documented in full.

More journeys are added as the library grows.

## See also

- [Analysis](../analysis/index.md) — the quantifier reference each tutorial draws on
- [Systems](../systems/index.md) — the 177 built-in models the journeys run on
- [The mental model](../start/concepts.md) — the definition contract in depth
- [Visualization](../visualization/index.md) — the plotting layer every figure is built with
