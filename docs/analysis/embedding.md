---
description: State-space reconstruction from a scalar measurement — Takens delay embedding, with delay and dimension selection heuristics.
---

<span class="ts-kicker">Analysis · 09</span>

# Delay embeddings

When you have a single measured signal but no equations, Takens' theorem
(Takens 1981) says the attractor's geometry can be reconstructed from
*delays* of that one observable. The scalar series $x_t$ becomes vectors

$$
\mathbf{y}_t = \big(x_t,\; x_{t-\tau},\; \dots,\; x_{t-(m-1)\tau}\big),
$$

and for a large enough embedding dimension $m$ the reconstruction is
diffeomorphic to the true state space — so invariants like dimension and
the maximal Lyapunov exponent survive. TSDynamics gives you the embedding
map plus the two heuristics that pick its parameters $\tau$ and $m$.

| Function | Picks | Method |
|---|---|---|
| [`embed`](#the-embedding-map) | the vectors $\mathbf{y}_t$ | Takens delay map |
| [`optimal_delay`](#choosing-the-delay) | the delay $\tau$ | MI / autocorrelation |
| [`embedding_dimension`](#choosing-the-dimension) | the dimension $m$ | FNN / Cao |

All routines accept a bare 1-D array, a `Trajectory`, or a multi-component
array (pick the channel with `component=`). They are backend-free — pure
NumPy over the series — so they run in the fast tier.

## The embedding map

`embed` builds the delayed coordinate vectors. Pass scalar `dimension` /
`delay` for a univariate embedding, or sequences for a non-uniform /
multivariate one.

```python
import numpy as np
import tsdynamics as ts

x = np.sin(np.linspace(0, 200, 4000))
emb = ts.embed(x, dimension=3, delay=24)
emb.shape          # (3952, 3)  — (n − (m−1)τ) rows of length m
```

## Choosing the delay

`optimal_delay` returns $\tau$. Too small and the coordinates are nearly
identical (the attractor squashes onto the diagonal); too large and they
decorrelate into noise — so you want the first "knee".

=== "Mutual information (default)"

    ```python
    h = ts.Henon()
    x = h.trajectory(6000, transient=500, ic=[0.1, 0.1]).y[:, 0]
    ts.optimal_delay(x, method="mi")      # = 16
    ```

    The first minimum of the time-delayed mutual information
    $I(x_t; x_{t-\tau})$ (Fraser & Swinney 1986) — the lag at which the
    delayed coordinate is maximally *independent* yet still dynamically
    linked. `mutual_information` returns the full $I(\tau)$ curve.

=== "Autocorrelation"

    ```python
    x = np.sin(np.linspace(0, 200, 4000))
    ts.optimal_delay(x, method="acf")     # = 24
    ```

    The lag where the autocorrelation first falls to $1 - 1/e$ (or its
    first zero) — cheaper and linear, fine for smooth oscillatory signals.
    `autocorrelation` returns the full curve.

## Choosing the dimension

`embedding_dimension` returns the minimal $m$ that unfolds the attractor —
the dimension at which spurious crossings introduced by under-embedding
disappear. Two estimators are available via `method=`, each also exposed
directly. The result behaves as the integer $m$ in arithmetic
(`int(result)`) while carrying the diagnostic curve it was read from.

=== "False nearest neighbours (unambiguous)"

    ```python
    x = ts.Henon().trajectory(6000, transient=500, ic=[0.1, 0.1]).y[:, 0]
    res = ts.false_nearest_neighbors(x, delay=1, max_dim=8)
    int(res)              # = 2   (Hénon)
    res.fnn_fraction      # fraction of false neighbours per dimension
    ```

    `false_nearest_neighbors` (Kennel, Brown & Abarbanel 1992) flags, at
    each $m$, neighbours that fly apart when a coordinate is added — the
    *false* ones. $m$ is where that fraction drops to ~0. This is the
    sharp, unambiguous estimator.

=== "Cao's method (parameter-light)"

    ```python
    x = ts.Henon().trajectory(6000, transient=500, ic=[0.1, 0.1]).y[:, 0]
    res = ts.cao_dimension(x, delay=1, max_dim=8)
    int(res)              # = 2   (Hénon)
    res.afn_e1            # E1(d): saturates at the true dimension
    ```

    `cao_dimension` (Cao 1997) tracks $E_1(d)$, the mean change in nearest-
    neighbour distance as a coordinate is added; $m$ is where $E_1$
    *saturates*. It needs only the delay and avoids FNN's two tolerances,
    but often saturates **one step late** — a safe over-embed.

!!! note "Inspect the curve"
    Both estimators carry the per-dimension diagnostic (`res.dims` with
    `res.fnn_fraction` / `res.afn_e1`). For a borderline attractor, read
    off $m$ from the curve yourself rather than trusting the threshold.

## Known values

| Signal | $m$ | Note |
|---|---|---|
| Sine wave | 2 | a closed loop lives in 2-D (verified, FNN + Cao) |
| Hénon (`x` channel) | 2 | FNN drops to 0 at $m=2$ (verified) |
| Rössler / Lorenz | 3 | FNN unambiguous; Cao saturates one step late |

The headline test: reconstruct an attractor from **one coordinate** and
recover its invariants. Embedding Rössler's $x(t)$ alone and taking the
[correlation dimension](dimensions.md) returns $\approx 2.0$ — the same
value as the full 3-D attractor (expected literature value; the ODE
reconstruction compiles, so it is slow):

```python
ros = ts.Rossler()
x = ros.integrate(final_time=2000.0, dt=0.05).y[:, 0]
tau = ts.optimal_delay(x, method="mi")
m = int(ts.false_nearest_neighbors(x, delay=tau))      # → 3
rec = ts.embed(x, dimension=m, delay=tau)
ts.correlation_dimension(rec)                          # ≈ 2.0
```

## See also

- [Fractal dimensions](dimensions.md) — read invariants off the reconstruction
- [Lyapunov spectra](lyapunov.md) — `lyapunov_from_data` embeds internally
- [Recurrence & RQA](recurrence.md) — recurrence plots of an embedded series

## References

- Takens (1981), *Lecture Notes in Mathematics* **898**, 366.
- Fraser & Swinney (1986), *Phys. Rev. A* **33**, 1134.
- Kennel, Brown & Abarbanel (1992), *Phys. Rev. A* **45**, 3403.
- Cao (1997), *Physica D* **110**, 43.
