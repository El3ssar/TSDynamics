---
description: Takens delay embedding — reconstructing a system's state space from a single scalar measurement, with mutual-information delay selection and Cao / false-nearest-neighbour dimension selection.
---

<span class="ts-kicker">Analysis · Delay embeddings</span>

# Delay embeddings

Most of the analysis toolkit wants a *state-space cloud* — a point per instant
in the system's phase space. But an experiment usually hands you a single scalar
channel: a voltage, a concentration, a brightness. Under suitable smoothness and
genericity assumptions, delay embedding can reconstruct a state space from one
observable $x(t)$ of a deterministic system. It stacks time-delayed copies of
the signal into a vector:

$$
y_i = \big(x_i,\; x_{i+\tau},\; x_{i+2\tau},\; \dots,\; x_{i+(m-1)\tau}\big).
$$

Takens (1981) concerns generic smooth dynamics and observables on a compact
manifold of dimension $d$, with $m \geq 2d+1$ coordinates. Extensions to fractal
attractors use box-counting dimension under additional assumptions. These
theorems motivate the method; they do not certify an arbitrary sensor, delay,
or finite noisy recording, nor the accuracy of numerical invariants estimated
from it. TSDynamics provides heuristics for the delay $\tau$ and dimension $m$,
and diagnostic curves with which to assess them. The functions live under
`tsdynamics.analysis.embedding` and are also available as `ts.analysis.<name>`:

```python
import tsdynamics as ts
from tsdynamics.analysis.embedding import (
    embed, optimal_delay, mutual_information, autocorrelation,
    embedding_dimension, cao_dimension, false_nearest_neighbors,
)
```

## The reconstruction, from one channel

The figure compares the measured $(x, y)$ projection of a Rössler trajectory with
delay coordinates constructed from **$x(t)$ alone**. The reconstruction shows
related folded structure; similarity between two projections does not establish
a topological equivalence or verify the embedding theorem's assumptions.

<div class="ts-ref" markdown>

<div class="ts-item" markdown>
```python
ros = ts.systems.Rossler()
traj = ros.run(final_time=400.0, dt=0.05, ic=[1.0, 0.0, 0.0])
x = traj.y[1000:, 0]        # keep ONLY the x channel

tau = ts.analysis.optimal_delay(x, method="mi", max_delay=120)   # 25 samples
emb = ts.analysis.embed(x, dimension=3, delay=tau)
emb.shape                   # (6951, 3)
emb[:, 0], emb[:, 1]        # (x(t), x(t+tau)) — the reconstructed plane
```

`embed(data, dimension, delay)` builds the
$(N - (m-1)\tau) \times m$ matrix of delay vectors, one reconstructed state per
row in temporal order. The returned `Embedding` behaves as that bare array —
index it, slice it, `np.asarray()` it — so it drops straight into the point-set
analyses.
</div>
<figure class="ts-fig" markdown>
![Rössler true state space beside a delay reconstruction from x(t) alone](../assets/figures/analysis/embedding.svg){ loading=lazy }
<figcaption>Left: a Rössler trajectory projected onto its measured (x, y) coordinates. Right: delayed coordinates from x(t), using a delay selected by mutual information. Both views use a finite record.</figcaption>
</figure>

</div>

The reconstruction feeds the geometric quantifiers directly. Because the rows
keep the original sampling order, an index-based Theiler window still removes
temporally-correlated pairs:

```python
ts.analysis.correlation_dimension(emb, theiler=tau)   # finite-sample, selected-scale estimate
```

`embed` also does **multivariate** embedding: pass a multi-component trajectory
(or a 2-D array) *without* `components=` and it stacks per-channel delay
coordinates into one joint reconstruction, with an optional per-channel
`dimension` / `delay`.

## Choosing the delay

The delay balances two practical limitations. Too small, and successive
coordinates $x_i$ and $x_{i+\tau}$ are nearly identical — the reconstruction
collapses onto the diagonal (redundancy). Too large, and on a chaotic signal
their statistical dependence may become weak and neighbouring reconstructed
states less useful. Inspect the delay diagnostic and reconstruction together.

The recommended nonlinear criterion is the **first local minimum of the
time-delayed mutual information** (Fraser & Swinney 1986): the lag at which
dependence between $x_i$ and $x_{i+\tau}$ first reaches a local minimum. This is a
selection heuristic rather than an optimum guaranteed for every signal.

```python
tau = ts.analysis.optimal_delay(x, method="mi", max_delay=120)   # 25   (first MI minimum)

mi = ts.analysis.mutual_information(x, max_delay=120)
mi.optimal_lag          # 25   — the same lag, read off the curve
mi.plot()               # the I(tau) diagnostic, with the chosen tau marked
```

`optimal_delay` returns a count that behaves as the integer $\tau$ — it prints,
formats and indexes as that integer — so it drops straight into `embed`. The linear alternatives are the classic
autocorrelation rules — `method="acf"` (first lag where the autocorrelation
falls to $1/e$) and `method="acf_zero"` (first zero crossing); the raw curve is
available via `autocorrelation(x)`. On the Rössler $x$ channel the $1/e$ rule
gives $\tau \approx 22$, close to the mutual-information choice.

## Choosing the dimension

The dimension must be large enough to **unfold** the attractor — to remove the
false crossings a low-dimensional projection creates where two distant states
happen to overlap. TSDynamics offers two complementary neighbour-based
estimators, unified behind `embedding_dimension`.

**Cao's averaged false neighbours** (Cao 1997) tracks, for each point, how the
distance to its nearest neighbour changes when one extra delay coordinate is
added. The dimension-averaged ratio $E_1(d) = E(d+1)/E(d)$ rises to and then
**saturates at 1**; the smallest $d$ past which it stops changing is the minimum
embedding dimension under the chosen threshold. A second quantity $E_2(d)$ can
help assess stochastic contamination; a near-unit curve alone does not prove
that a finite recording is random or deterministic.

**Kennel's false nearest neighbours** (Kennel, Brown & Abarbanel 1992) calls a
neighbour *false* when adding a coordinate pushes it far apart — its proximity
was a projection artefact. The false-neighbour fraction decays to zero at the
correct dimension.

```python
m_cao = ts.analysis.embedding_dimension(x, method="cao", delay=tau, max_dim=8)
int(m_cao)              # 4      (drops straight into embed)
m_cao.afn_e1            # the E1(d) saturation curve
m_cao.plot()            # E1 / E2 vs d, with the chosen m marked

m_fnn = ts.analysis.embedding_dimension(x, method="fnn", delay=tau, max_dim=8)
int(m_fnn)              # 3      (a reconstruction heuristic, not the attractor dimension)
m_fnn.saturated        # whether the criterion was met before the search ceiling
```

Both return an `EmbeddingDimension`: `int(result)` is the recommended $m$, and
the per-dimension diagnostic is carried so you can inspect the
saturation/decay rather than trust a single number. The two estimators need not
agree to the integer — Cao's saturation threshold and Kennel's tolerances are
conservative in different directions — so on a marginal case take $m$ from the
*shape* of the curve. If no tested dimension meets the threshold, the integer is
the best candidate within the tested range and `.saturated` is `False`; extend
the search and inspect the curve before using it as a resolved recommendation.

!!! note "Set a Theiler window on densely sampled flows"
    Both dimension estimators find each point's nearest *spatial* neighbour, and
    on a densely sampled flow the true nearest neighbour is often just the next
    sample in time — a temporal artefact, not a geometric one. Pass
    `theiler=w` (a few autocorrelation times) to exclude neighbours closer than
    $w$ samples in time. For the same reason, Cao's $E_1$/$E_2$ curves are
    unreliable at exactly $d = 1$ (the 1-D nearest-neighbour gap can be
    arbitrarily small). This can also bias the recommended $m$ if its threshold
    is crossed there. Inspect the curves and check sensitivity to the delay,
    temporal exclusion and available sample count.

## The full pipeline

Delay, dimension, embed, analyse — from one channel to a dimension estimate:

```python
x = ts.systems.Rossler().run(final_time=400.0, dt=0.05, ic=[1.0, 0.0, 0.0]).y[1000:, 0]

tau = ts.analysis.optimal_delay(x, method="mi", max_delay=120)     # 25
m = int(ts.analysis.embedding_dimension(x, method="fnn", delay=tau, max_dim=8))  # 3
emb = ts.analysis.embed(x, dimension=m, delay=tau)
dimension = ts.analysis.correlation_dimension(emb, theiler=tau)
dimension.plot()   # inspect the fitted scales and local slopes
```

This is the standard route to a fractal dimension or a data-driven Lyapunov
exponent when you have a recording but no equations —
[`lyapunov_from_data`](lyapunov.md) does its own internal embedding for exactly
this reason. A short Rössler record can return a slope near 1.7 over its selected
scales, below its usual asymptotic dimension near 2. Agreement with a similarly
short full-state estimate is a useful cross-check, not proof of convergence.
Vary the record length, temporal exclusion and scaling window.

## See also

- [Fractal dimensions](dimensions.md) — the reconstruction's most common
  consumer; remember the `theiler=tau` window on an embedded flow
- [Lyapunov spectra](lyapunov.md) — `lyapunov_from_data` estimates the maximal
  exponent from a delay embedding of a measured series
- [Recurrence & RQA](recurrence.md) — recurrence plots are built on the same
  $(m, \tau)$ reconstruction

## References

- F. Takens, "Detecting strange attractors in turbulence", in *Dynamical Systems
  and Turbulence*, Lecture Notes in Mathematics **898**, 366 (1981).
- A. M. Fraser and H. L. Swinney, "Independent coordinates for strange attractors
  from mutual information", *Phys. Rev. A* **33**, 1134 (1986).
- L. Cao, "Practical method for determining the minimum embedding dimension of a
  scalar time series", *Physica D* **110**, 43 (1997).
- M. B. Kennel, R. Brown and H. D. I. Abarbanel, "Determining embedding
  dimension for phase-space reconstruction using a geometrical construction",
  *Phys. Rev. A* **45**, 3403 (1992).
