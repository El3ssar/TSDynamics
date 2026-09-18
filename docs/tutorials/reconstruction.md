---
description: You have one recorded signal x(t) and no equations — reconstruct the attractor by Takens delay embedding, choose the delay and dimension with principled heuristics, and measure the correlation dimension and a data-driven Lyapunov exponent from the recording alone.
---

<span class="ts-kicker">Tutorial · Reconstruction from one signal</span>

# Reconstruction from one signal

The first two tutorials had the luxury every textbook has and no experiment ever
does: the equations. Real data arrives the other way round. You have a single
recorded channel — a voltage, a concentration, a light curve — sampled over time,
and *nothing else*. No state vector, no model, no idea whether the wiggle in
front of you is a deterministic system worth analysing or just filtered noise.

This tutorial walks the whole inference the other direction: from one scalar
$x(t)$ back to the geometry and invariants of the system that produced it. We
will **reconstruct** the attractor from the single channel, choose the two
reconstruction parameters with principled heuristics, and measure the
attractor's fractal dimension and its largest Lyapunov exponent *from the
recording* — then check every recovered number against the truth we threw away.

To keep it honest and reproducible we *generate* the mystery signal from a known
system (the Rössler flow) and then immediately throw the model away, keeping only
one component. Everything after that line uses the scalar alone — so we can check
every recovered number against the truth at the end.

```python
import tsdynamics as ts

# generate a trajectory, then keep ONLY the x channel — pretend this is all you recorded
full = ts.systems.Rossler().run(final_time=400.0, dt=0.05, ic=[1.0, 0.0, 0.0])
x = full.y[1000:, 0]        # a single scalar signal, transient dropped
x.shape                     # (7001,)
```

From here on, `x` is the only thing we touch.

## 1. The theorem that makes it possible

You would think one channel throws away most of the state. Takens' embedding
theorem (Takens, 1981) says otherwise: from one *generic* observable of a
deterministic system you can reconstruct a trajectory whose attractor is
diffeomorphic to the true one — same topology, same invariants — simply by
stacking time-delayed copies of the single signal into a vector,

$$
\mathbf{y}_i = \big(x_i,\; x_{i+\tau},\; x_{i+2\tau},\; \dots,\; x_{i+(m-1)\tau}\big).
$$

Two choices make the reconstruction faithful: the **delay** $\tau$ and the
**embedding dimension** $m$. The
[delay-embedding tools](../analysis/embedding.md) give a principled way to pick
each.

## 2. Choose the delay

The delay trades two failures against each other. Too small, and $x_i$ and
$x_{i+\tau}$ are nearly identical — the reconstruction collapses onto the
diagonal. Too large, and on a chaotic signal they become causally unrelated — the
geometry folds noise into itself. The recommended criterion is the **first local
minimum of the time-delayed mutual information** (Fraser & Swinney, 1986): the
lag at which the next coordinate adds the most *new* information while staying
dynamically related to the current one.

```python
tau = ts.analysis.optimal_delay(x, method="mi", max_delay=120)   # 25 samples

mi = ts.analysis.mutual_information(x, max_delay=120)
mi.optimal_lag          # 25   — the same lag, read off the I(tau) curve
# mi.plot()             # inspect the curve, with the chosen tau marked
```

`optimal_delay` returns a count that *is* the integer $\tau = 25$ — it prints,
formats and indexes as that integer — so it drops straight into `embed`. The linear alternative — the autocorrelation $1/e$
rule, `method="acf"` — gives $\tau \approx 22$ here, close to the
mutual-information choice; when they disagree, prefer the mutual-information lag,
which sees nonlinear dependence the autocorrelation misses.

## 3. Choose the dimension

The dimension must be large enough to **unfold** the attractor — to remove the
false crossings a too-flat projection creates where two distant states happen to
overlap. **Cao's method** (Cao, 1997) and **Kennel's false nearest neighbours**
(Kennel et al., 1992) both detect the smallest $m$ at which those artefacts
vanish, unified behind `embedding_dimension`:

```python
m_fnn = ts.analysis.embedding_dimension(x, method="fnn", delay=tau, max_dim=8)
int(m_fnn)              # 3   — Kennel FNN; matches Rössler's true dimension

m_cao = ts.analysis.embedding_dimension(x, method="cao", delay=tau, max_dim=8)
int(m_cao)             # 4   — Cao's conservative saturation threshold
# m_cao.plot()         # the E1/E2 saturation curves, with the chosen m marked
```

The two estimators need not agree to the integer — Cao's saturation threshold and
Kennel's tolerance are conservative in different directions — so on a marginal
case read $m$ off the *shape* of the diagnostic curve rather than trusting a
single number, and when in doubt embed one dimension higher. Here FNN recovers
the true dimension 3 exactly; we take $m = 3$.

## 4. Reconstruct — one channel becomes an attractor

With both parameters chosen, `embed` builds the matrix of delay vectors:

<div class="ts-ref" markdown>

<div class="ts-item" markdown>
```python
emb = ts.analysis.embed(x, dimension=3, delay=tau)
emb.shape               # (6947, 3) — one reconstructed state per row, in time order
```

The returned `Embedding` behaves as that bare array, so it drops straight into
any point-set analysis. The two leading columns $(x_i,\,x_{i+\tau})$ are the
reconstructed plane — plot them and the folded Rössler loop reappears, rebuilt
from the single channel. Because a `Trajectory` can build the same delay view
directly, you can eyeball the reconstruction with the plotting front door:

```python
# skip-doctest — .save() writes a file; needs the optional tsdynamics[viz] backend
# the x(t) vs x(t - tau) delay portrait, straight from the series.
# `delay=` is in SAMPLES (what optimal_delay returned); `delay_time=` is the
# same lag in time units — here delay_time = tau * 0.05 would be equivalent.
ts.plot(full["x"], "delay_embedding", delay=tau).save("delay.png")
```
</div>

<figure class="ts-fig" markdown>
![A delay-coordinate reconstruction of a flow from one channel](../assets/figures/viz/kind-delay.svg){ loading=lazy }
<figcaption><span class="lbl">FIG 1</span> · a delay embedding — x(t) against x(t − τ) — built from a single recorded channel. The <code>delay_embedding</code> transform also takes <code>delay_time=</code> in <em>time units</em> (here τ = 25 samples × dt = 1.25). The loop is the reconstructed attractor: same topology as the true state space, from one signal.</figcaption>
</figure>

</div>

## 5. Measure the attractor — from the reconstruction

Now the payoff. Takens guarantees the reconstruction preserves the *invariants*,
so any quantity that depends only on the attractor's geometry or dynamics can be
computed from the embedded cloud — no equations needed.

**Fractal dimension.** The [correlation dimension](../analysis/dimensions.md)
(Grassberger & Procaccia, 1983) reads off the point cloud directly. On a
reconstructed *flow* you must set a Theiler window, because temporally adjacent
samples sit spuriously close and bias the estimate downward — a few delays is a
safe choice:

```python
ts.analysis.correlation_dimension(emb, theiler=tau)     # ≈ 1.74   (Rössler D2, from x alone)
```

**Largest Lyapunov exponent.** [`lyapunov_from_data`](../analysis/lyapunov.md)
estimates the top exponent from how fast embedded neighbours diverge (Kantz,
1994). Pass the sampling interval `dt=` so the rate comes out per unit time, and
— crucially — **inspect the stretching curve before quoting a number**:

```python
res = ts.analysis.lyapunov_from_data(x, dimension=3, delay=tau, dt=0.05, k_max=80)
res.times, res.divergence      # the S(k) curve — plot it, find the linear stretch
```

The curve $S(k)$ rises, flattens, then climbs into a clean linear region roughly
between look-ahead times $0.3$ and $1.3$ (sample indices 6–26). The exponent is
the slope of *that* stretch, not of the whole curve — fit it explicitly:

```python
res = ts.analysis.lyapunov_from_data(x, dimension=3, delay=tau, dt=0.05, k_max=80, fit=(6, 26))
float(res)             # ≈ 0.074   per unit time — positive, so chaotic
```

!!! warning "Never quote the automatic fit blind"
    On a flow the divergence curve typically overshoots before it settles, so the
    default automatic fit **overestimates** (here it returns $\approx 0.13$, well
    above the truth). The estimate is only as good as the scaling region you
    choose — always look at `res.times` vs `res.divergence` and pass an explicit
    `fit=(lo, hi)` before quoting a publishable value. This is *the* place
    from-data exponents go wrong.

**How did we do?** We can cheat now and check against the model we threw away:

```python
# the truth, from the full state and the equations
ts.analysis.correlation_dimension(full.y[1000:], theiler=tau)                     # ≈ 1.75
ts.analysis.lyapunov_spectrum(ts.systems.Rossler(), k=1, ic=[1.0, 0.0, 0.0])  # ≈ 0.07
```

$D_2 = 1.74$ from one channel versus $1.75$ from the full state; $\lambda \approx
0.074$ from the recording versus the true $\approx 0.06$–$0.07$ (literature:
$0.071$). One scalar signal, and we recovered both invariants to within their
estimation error.

## What you built

Starting from a single scalar channel with the model discarded, you rebuilt the
attractor by delay embedding, chose $\tau$ and $m$ from mutual information and
false-nearest-neighbours, measured the correlation dimension ($1.74$) and the
largest Lyapunov exponent ($0.074$ per unit time) *from the recording*, and
checked both against the ground truth we had thrown away. This is the standard
route from an experimental time series to a defensible statement about the
system behind it.

## See also

- [Delay embeddings](../analysis/embedding.md) — the full `embed` / `optimal_delay` / `embedding_dimension` API and multivariate embedding
- [Fractal dimensions](../analysis/dimensions.md) — correlation and the rest of the fractal-geometry estimators (and the Theiler window)
- [Lyapunov spectra](../analysis/lyapunov.md) — `lyapunov_from_data`, the Kantz / Rosenstein estimators, and reading the stretching curve
- [Recurrence & RQA](../analysis/recurrence.md) — another quantifier built straight on the reconstruction
- [Anatomy of a chaotic attractor](chaotic-attractor.md) — the same invariants, computed from the equations instead

## References

- Takens, F. (1981). Detecting strange attractors in turbulence. *LNM* **898**, 366.
- Fraser, A. M. & Swinney, H. L. (1986). Independent coordinates for strange attractors from mutual information. *Phys. Rev. A* **33**, 1134.
- Cao, L. (1997). Practical method for determining the minimum embedding dimension of a scalar time series. *Physica D* **110**, 43.
- Kennel, M. B., Brown, R. & Abarbanel, H. D. I. (1992). Determining embedding dimension for phase-space reconstruction … *Phys. Rev. A* **45**, 3403.
- Grassberger, P. & Procaccia, I. (1983). Measuring the strangeness of strange attractors. *Physica D* **9**, 189.
- Kantz, H. (1994). A robust method to estimate the maximal Lyapunov exponent of a time series. *Phys. Lett. A* **185**, 77.
