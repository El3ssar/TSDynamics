---
description: The global stability picture of a multistable system — locate attractors, paint their basins, and quantify boundary structure, continuation, tipping and resilience.
---

<span class="ts-kicker">Analysis · 12</span>

# Attractors & basins

A multistable system has several coexisting attractors; which one you reach
depends on where you start. This toolkit gives the **global** picture — it
locates the attractors, paints the basin of initial conditions feeding each,
and quantifies how tangled the boundaries between them are, how those basins
deform as a parameter moves, and how robust an attractor is to a perturbation.

| Function | Answers | Reads |
|---|---|---|
| [`find_attractors`](#locating-attractors) | what attractors exist | a system |
| [`basins_of_attraction`](#painting-basins) | which IC goes where | a system + `Grid` |
| [`basin_fractions`](#painting-basins) | each basin's volume share | a system |
| [`basin_entropy`](#boundary-structure) | is the boundary fractal? | a label image |
| [`uncertainty_exponent`](#boundary-structure) | boundary dimension $D_0$ | a label image |
| [`wada_property`](#boundary-structure) | do ≥3 basins share a boundary? | a label image |
| [`continuation`](#continuation-tipping-resilience) | attractors vs a parameter | a system |
| [`tipping_points`](#continuation-tipping-resilience) | where a basin dies | a `ContinuationResult` |
| [`resilience`](#continuation-tipping-resilience) | distance to the boundary | a `BasinsResult` |

## Locating attractors

`find_attractors` drives any map or flow over a cell tessellation of a search
region with a recurrence finite-state machine (Datseris & Wagemakers 2022): a
trajectory that *recurrently revisits* the same cells has landed on an
attractor; near-coincident attractors are proximity-merged (`merge_tol`).

```python
import tsdynamics as ts
from tsdynamics import Box

aset = ts.find_attractors(system, Box([-2, -2], [2, 2]), resolution=200)
aset.attractors        # list[Attractor], each with .id / .points / .cells
aset.centers           # one representative point per attractor
```

Flows step by `dt` per cell check, maps by one iteration. A raised or
non-finite step is treated as divergence; a finite excursion outside the box is
counted by a lost-counter. The same engine powers the basin and continuation
routines below.

## Painting basins

`basins_of_attraction` runs that finder from **every cell of a `Grid` of
initial conditions** and labels each with the attractor it reaches — a colour
map of state space.

```python
from tsdynamics import Grid

res = ts.basins_of_attraction(system, Grid([-2, -2], [2, 2], (300, 300)))
res.labels             # int array, one attractor id per IC (−1 = diverged)
res.fractions          # {attractor_id: basin volume share}
res.diverged_fraction
```

!!! note "Imaging a slice of a higher-dim flow"
    For a flow whose state space is bigger than the 2-D picture you want, pass a
    separate `recurrence` box: the basin is painted over the 2-D `Grid` while the
    recurrence FSM runs in the full space. This is how the magnetic pendulum's
    famous fractal basins are imaged from its 4-D phase space.

`basin_fractions` skips the image and Monte-Carlo–samples the region instead —
**basin stability** in the sense of Menck et al. (2013), the probability a
random IC lands on each attractor, with a `standard_error`:

```python
bf = ts.basin_fractions(system, Box([-2, -2], [2, 2]), n=10000)
bf.fractions, bf.dominant, bf.standard_error
```

### Expected fractions (validated)

These three test-local systems pin the fractions to exact values:

| System | Basins | Each fraction |
|---|---|---|
| Newton's method on $z^3 = 1$ (a map) | 3 (symmetric) | $\approx 1/3$ |
| Two-well Duffing oscillator | 2 | $\approx 1/2$ |
| Magnetic pendulum | 3, *fractal* boundary | Wada (see below) |

The runnable end-to-end versions live in the
[equations → basins tutorial](../tutorials/equations-to-basins.md).

## Boundary structure

The three metrics below read a **label image** (the `res.labels` array) and do
no integration at all, so they run instantly:

=== "basin_entropy"

    ```python
    import numpy as np
    rng = np.random.default_rng(0)
    riddled = rng.integers(0, 3, size=(200, 200))   # a maximally-mixed boundary

    be = ts.basin_entropy(riddled)
    be.sb, be.sbb              # ≈ 1.0595, 1.0595
    be.fractal_boundary        # True   (Sbb > ln 2 ≈ 0.693)
    ```

    The basin entropy $S_b$ and boundary basin entropy $S_{bb}$ (Daza et al.
    2016) measure label diversity per box. The diagnostic is **$S_{bb} > \ln 2$
    ⇒ a fractal boundary** — exposed directly as `be.fractal_boundary`.

=== "uncertainty_exponent"

    ```python
    ue = ts.uncertainty_exponent(riddled)
    ue.alpha                   # uncertainty exponent α
    ue.boundary_dimension      # D₀ = D − α
    ```

    The fraction of $\epsilon$-uncertain points scales as $f(\epsilon)\sim
    \epsilon^{\alpha}$ (Grebogi et al. 1983); a **small $\alpha$** means a thick,
    fractal boundary where you can rarely predict the outcome. The boundary
    fractal dimension is $D_0 = D - \alpha$.

=== "wada_property"

    ```python
    wr = ts.wada_property(res.labels)
    wr.is_wada, wr.n_basins
    ```

    The grid test of Daza et al. (2015) for the **Wada property** — every point
    on the boundary of one basin lies on the boundary of *all* of them (≥3
    basins sharing one boundary), the hallmark of the magnetic pendulum.

## Continuation, tipping, resilience

`continuation` re-finds the attractors at each value of a swept parameter and
**matches** them across values by state-space distance (greedy nearest;
Datseris et al. 2023), so a single attractor keeps its id along its branch.
`min_fraction` drops spurious saddle-passage sets.

```python
import numpy as np

cont = ts.continuation(system, "F", np.linspace(0.0, 2.0, 41),
                       Box([-2, -2], [2, 2]))
cont.fractions         # basin share of each matched attractor per value
tips = ts.tipping_points(cont)   # where a basin annihilates → a tipping point
```

`tipping_points` reads off the parameter values where a basin's share collapses
to zero. `resilience` measures, for a chosen attractor in a painted
`BasinsResult`, the distance from its attractor to the nearest basin boundary —
the largest perturbation it can absorb without tipping (Halekotte & Feudel
2020):

```python
r = ts.resilience(res, attractor_id=0)   # → distance to the basin boundary
```

## See also

- [Equations → basins tutorial](../tutorials/equations-to-basins.md) — the runnable end-to-end example
- [Lyapunov spectra](lyapunov.md) — the local stretching rate on a single attractor
- [Systems](../systems/index.md) — the built-in catalogue of maps and flows

## References

- Grebogi, McDonald, Ott & Yorke (1983), *Phys. Lett. A* **99**, 415.
- Menck, Heitzig, Marwan & Kurths (2013), *Nat. Phys.* **9**, 89.
- Daza, Wagemakers, Sanjuán & Yorke (2015), *Sci. Rep.* **5**, 16579.
- Daza, Wagemakers, Georgeot, Guéry-Odelin & Sanjuán (2016), *Sci. Rep.* **6**, 31416.
- Halekotte & Feudel (2020), *Sci. Rep.* **10**, 11783.
- Datseris & Wagemakers (2022), *Chaos* **32**, 023104.
- Datseris, Rossi & Wagemakers (2023), *J. Phys. Complex.* **4**, 025008.
