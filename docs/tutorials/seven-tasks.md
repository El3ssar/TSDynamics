---
description: Seven tasks in one sitting — simulate a system, test it for chaos, draw a bifurcation diagram of a map and of a flow, find fixed points, paint basins, animate a phase portrait over a vector field, and define a system of your own. Every line runs.
---

<span class="ts-kicker">Tutorials · Start here</span>

# Seven tasks in one sitting

This is the cold-start page: seven things people actually come to a
dynamical-systems library to do, each in a handful of lines, on one page, in
order. Paste them in sequence into a fresh session — every line here is executed
by the documentation gate, so what you see is what runs.

If you only read one page, read this one. The rest of the site is the same
material at depth.

Three facts carry the whole API:

1. **`run` is the one verb that produces data**, on every family.
2. **Every analysis is a free function whose first argument is its subject** —
   `ts.analysis.lyapunov_spectrum(lorenz)`. There are no analysis methods.
3. **`ts.plot` draws everything you hand it**: a positional *string* says how to
   draw, everything else says what to draw.

```python
import numpy as np
import tsdynamics as ts
```

---

## 1 · Simulate a system

Pick a system, `run` it, get a `Trajectory`.

```python
lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
traj = lor.run(final_time=100.0, dt=0.01)

traj.y.shape          # (10001, 3) — one row per sample, one column per component
traj["x"][:3]         # the x channel, by name
len(traj)             # 10001 samples
traj.meta["method"]   # the solver kernel that ran — 'rk45'
```

`dt` is the **output sampling interval**, not an accuracy knob: the solver is
adaptive and uses genuine dense output, so a coarse `dt` costs resolution and
nothing else. Accuracy is `rtol`/`atol` (`1e-9`/`1e-12` by default) and
`max_step=` bounds the internal step when a feature is narrow.

A map is the same verb with the horizon its mathematics actually has — a count:

```python
hen = ts.systems.Henon()
orbit = hen.run(steps=5000, ic=[0.1, 0.1], transient=500)
orbit.t[:4]           # array([0, 1, 2, 3]) — integer iterate indices
```

Ask a map for `final_time=` and the error explains the mathematics rather than
just the signature:

```
InvalidParameterError: final_time is not a valid Henon.run() keyword, got 10.0.
final_time is a *flow* keyword: a map has no continuous time — its horizon is a
count of iterations.
    Henon().run(steps=1000)
```

Draw it:

```python
ts.plot(traj, color="#4B3F9E", title="Lorenz")
```

---

## 2 · Is it chaotic?

The Lyapunov spectrum is the defining measurement. It comes from the variational
equations — exact, not finite-differenced — and **the result prints the answer**:

```python
spec = ts.analysis.lyapunov_spectrum(lor, final_time=300.0, dt=0.05, transient=40.0)
print(spec)
# LyapunovSpectrum  λ = [0.9081, -0.001005, -14.57]   chaotic · D_KY = 2.062   (Lorenz)
#     (3 exponents · flow (realised zero) · λ > 0.0146)
```

One positive exponent, one at zero (the direction *along* the flow), one strongly
negative: the positive·zero·negative signature of a chaotic dissipative flow. The
verdict word is only printed when it survives a **tenfold change of the zero
tolerance** — otherwise the result says `indeterminate at this horizon` instead of
guessing, and the supporting line always shows the floor it decided at.

The result behaves like the array it reports, so it drops into ordinary code:

```python
float(spec[0])          # 0.908…  the leading exponent
np.asarray(spec)        # the bare (3,) array
spec.kaplan_yorke       # 2.06 — the fractal dimension implied by the spectrum
```

Two cheaper verdicts, useful as cross-checks:

```python
ts.analysis.max_lyapunov(hen, ic=[0.1, 0.1])      # ≈ 0.42, no Jacobian needed
print(ts.analysis.zero_one_test(lor, final_time=300.0, dt=0.1))
# ZeroOneResult  K = 0.998306   chaotic (K ≈ 1)   (Lorenz)
```

Not sure which tool answers your question? Ask:

```python
ts.analysis.find("is this chaotic")
```

---

## 3 · A bifurcation diagram — of a map, and of a flow

Sweep one parameter and record where the long-term motion settles.

```python
logistic = ts.systems.Logistic()
od = ts.analysis.orbit_diagram(logistic, "r", np.linspace(2.8, 4.0, 400),
                               points_per_value=80, transient=300)
print(od)
# OrbitDiagram  r ∈ [2.8, 4] · 400 values × 80 points   (Logistic)
#     periods seen: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12 · 132 aperiodic values
#     bifurcations at r = 3.003, 3.451, 3.544, 3.55 …

od.bifurcation_points()[:2]   # ≈ [3.00, 3.45] — the textbook r₁ = 3, r₂ = 1 + √6
ts.plot(od, color="k", markersize=0.3, alpha=0.5)
```

A **flow** has no discrete view of its own — so derive one. `sys.poincare(...)`
presents the flow as the map of its crossings through a section plane, and a
parameter sweep over *that* is the bifurcation diagram of the flow:

```python
ros = ts.systems.Rossler(ic=[1.0, 1.0, 1.0])
section = ros.poincare("y", 0.0, direction="up")      # a discrete system

odf = ts.analysis.orbit_diagram(section, "c", np.linspace(2.0, 6.0, 60),
                                points_per_value=40, transient=40)
print(odf)
# OrbitDiagram  c ∈ [2, 6] · 60 values × 40 points   (PoincareMap)
#     periods seen: 1, 2, 3, 4, 6 · 28 aperiodic values
```

1 → 2 → 4 → chaos, the identical period-doubling cascade, one dimension up. For a
*periodically forced* oscillator the natural section is a strobe instead of a
plane — the same verb, given a period:

```python
strobe = ts.systems.Duffing().poincare(period=2 * np.pi / 1.4)
```

---

## 4 · Fixed points and their stability

Equilibria of a flow ($f(x) = 0$) and fixed points of a map ($f(x) = x$) come out
of the same call, found by multi-start Newton on the analytic Jacobian:

```python
print(ts.analysis.fixed_points(lor))
# FixedPointSet  3 points · 0 stable, 3 unstable   (Lorenz)
#     [0] x* = [-8.4853 -8.4853 27.    ]  unstable  Re(λ)max = +0.09396
#     [1] x* = [ 3.1536e-23  3.1536e-23 -6.4468e-23]  unstable  Re(λ)max = +11.83
#     [2] x* = [ 8.4853  8.4853 27.    ]  unstable  Re(λ)max = +0.09396
```

The two nontrivial equilibria sit at the centre of each wing of the butterfly,
and their complex eigenvalue pair is why the orbit *spirals* out of a wing before
switching. Indexing gives you the point; the linearisation of every point comes
back as one array:

```python
fps = ts.analysis.fixed_points(lor)
fps[0]                 # the first equilibrium, as a (dim,) array
fps.eigenvalues[0]     # the Jacobian spectrum at that point
fps.is_stable[0]       # False
```

**Regions are plain bounds** — one `(lo, hi)` pair per state component, and never
a library type:

```python
vdp = ts.systems.VanDerPol(params={"mu": 1.0})
print(ts.analysis.fixed_points(vdp, region=[(-3, 3), (-3, 3)]))
# FixedPointSet  1 point · 0 stable, 1 unstable   (VanDerPol)
#     [0] x* = [0. 0.]  unstable  Re(λ)max = +0.5
```

A single unstable equilibrium in a bounded 2-D flow means the orbits have to go
*somewhere*: that is the limit cycle you will see in §6.

For a map, `periodic_orbits` finds period-$p$ orbits the same way, and
`method="interval"` swaps multi-start Newton for a rigorous interval enclosure
that cannot silently miss a root.

---

## 5 · Basins of attraction

When a system has more than one possible fate, the interesting question is
*which* — as a function of where you start. The catalogue has no unforced
two-well oscillator, so here is one; §7 explains the contract in full.

```python
class DuffingTwoWell(ts.ContinuousSystem):
    """Unforced damped Duffing oscillator: two wells at x = ±1."""

    variables = ("x", "v")
    params = {"delta": 0.25}

    def _equations(u, t, delta):
        x, v = u(0), u(1)
        return [v, x - x**3 - delta * v]
```

```python
duff = DuffingTwoWell()
basins = ts.analysis.basins(duff, [(-2.0, 2.0, 40), (-2.0, 2.0, 40)],
                            dt=0.5, max_steps=2000)
print(basins)
# BasinsResult  40×40 grid · 2 basins: #1 50.7% · #2 49.3% · 0.0% diverged
#     (DuffingTwoWell)

basins.labels.shape       # (40, 40) — one attractor id per cell, -1 = diverged
ts.plot(basins)           # the basin image
```

`(lo, hi, n)` per component: the third entry is how finely to grid that axis. The
two basins split the plane along the stable manifold of the saddle at the origin.

Two follow-ups measure the *structure* of that partition, and each takes the
label image another analysis returned:

```python
print(ts.analysis.basin_entropy(basins.labels))
# BasinEntropy  Sb = 0.3757 · Sbb = 0.5465
#     (44/64 boxes on the boundary, box size 5, log base 2.718)
```

`Sbb > ln 2` would certify a **fractal** boundary — here it does not, because a
smooth manifold separates the wells. And if you want the shares without the
image, Monte-Carlo them instead (basin stability, Menck et al. 2013):

```python
print(ts.analysis.basin_fractions(duff, [(-2.0, 2.0), (-2.0, 2.0)],
                                  n=400, dt=0.5, max_steps=2000))
# BasinFractions  #1 51.5% ± 2.5% · #2 48.5% ± 2.5% · 0.0% diverged
#     (DuffingTwoWell, 400 samples)
```

---

## 6 · An animated phase portrait over a vector field

One call, two kinds of subject. The **system** supplies the field — which needs
the equations, not the data — and the **trajectory** supplies the orbit. Each
named transform is matched to the subject it declares it needs, so the call is
unambiguous and order-free:

```python
orbit = vdp.run(final_time=30.0, dt=0.02, ic=[0.5, 0.0])

p = ts.plot(vdp, orbit, "vector_field", "nullclines",
            title="Van der Pol", xlabel="x", ylabel="v")
p.style("vector_field", alpha=0.35, color="0.6")
p.style("nullclines", color="#E8912D", linewidth=1.4)
```

The orbit spirals out of the unstable origin from §4 and locks onto the limit
cycle; the nullclines are the curves where each component of the velocity
vanishes, and they cross exactly at the equilibrium.

Now make it move. **Animation is an orthogonal modifier** — the plot means the
same thing, it just plays:

```python
movie = (ts.plot(vdp, orbit, "vector_field", animate=True)
           .animate(n_frames=60, fps=20)
           .trail(("time", 6.0), fade=True)     # a 6-time-unit fading comet tail
           .head(size=9, color="#E8912D")       # the "current state" marker
           .style("vector_field", alpha=0.35, color="0.6"))
```

```python
# skip-doctest — writing the movie takes a few seconds and needs ffmpeg or pillow
movie.save("vdp.gif")      # .mp4 and .gif via matplotlib
movie.save("vdp.html")     # ...or a real-time comet you can pan while it plays
```

!!! tip "Animate on a solid background"
    A GIF has no alpha channel, so a transparent animation flattens to one fill
    colour. Give a movie a solid stage: `.background("#0B0F14")`.

---

## 7 · Define your own system

The catalogue is a starting point, not a ceiling. Name the components, declare
the parameters, write the math once:

```python
class RosenzweigMacArthur(ts.ContinuousSystem):
    """Predator–prey with a Holling type-II functional response."""

    variables = ("prey", "predator")            # dim inferred = 2
    params = {"r": 1.0, "K": 3.0, "a": 1.0, "h": 1.0, "e": 0.6, "m": 0.2}
    _reference = "Rosenzweig & MacArthur (1963), Am. Nat. 97, 209-223"
    _doi = "10.1086/282272"

    def _equations(u, t, r, K, a, h, e, m):
        x, y = u(0), u(1)
        grazing = a * x * y / (1.0 + a * h * x)
        return [r * x * (1.0 - x / K) - grazing,
                e * grazing - m * y]
```

That is the whole contract. `variables` names the components and **`dim` follows
from it**; `params` are runtime values, so changing one is free. The body builds
**symbolic** expressions — plain arithmetic plus `symengine` functions (`sin`,
`exp`, `sqrt`, …) — and `u(i)` reads component `i`. It is lowered to the Rust
engine once, not called sample by sample, so NumPy and a Python `if` on the state
do not belong here.

`system.info` prints everything the library read back to you:

```python
rma = RosenzweigMacArthur()
print(rma.info)
# RosenzweigMacArthur — 2-D continuous flow              __main__.RosenzweigMacArthur
#   parameters  r = 1   K = 3   a = 1   h = 1   e = 0.6   m = 0.2
#   variables   prey, predator
#   reference   Rosenzweig & MacArthur (1963), Am. Nat. 97, 209-223   doi:10.1086/282272
#   defaults    solver=rk45  rtol=1e-09  atol=1e-12  dt=0.02  backend=jit
#   analyses    ts.analysis.find(system) → 50
```

There is no registration call and no build step — and **everything in tasks 1–6
now works on it**, because every tool in the library is written against the same
protocol:

```python
traj = rma.run(final_time=200.0, dt=0.05, ic=[1.0, 0.5])

print(ts.analysis.fixed_points(rma, region=[(0.01, 3.0), (0.01, 3.0)]))
# FixedPointSet  1 point · 0 stable, 1 unstable   (RosenzweigMacArthur)
#     [0] x* = [0.5  1.25]  unstable  Re(λ)max = +0.05556

print(ts.analysis.lyapunov_spectrum(rma, final_time=2000.0, dt=0.05,
                                    ic=[1.0, 0.5], transient=200.0))
# LyapunovSpectrum  λ = [-0.0004678, -0.1094]   regular   (RosenzweigMacArthur)

ts.plot(rma, traj, "flow_speed", "nullclines", title="predator–prey")
```

The coexistence equilibrium is unstable and the spectrum has no positive
exponent: the population settles onto a **limit cycle**, the predator–prey
oscillation. That is the paradox of enrichment, measured in five lines.

Your class also joined the registry the moment it was defined:

```python
from tsdynamics import registry

registry.get("RosenzweigMacArthur").dim          # 2
registry.get("RosenzweigMacArthur").is_builtin   # False — yours, not shipped
```

For a system contributed to the library itself, that same registration is what
generates its documentation page and sweeps it into the test suite.

---

## Where to go next

| You want | Go to |
| -------- | ----- |
| The mental model — four families, one protocol | [The mental model](../start/concepts.md) |
| Every quantifier, with its literature | [Analysis](../analysis/index.md) |
| The full plotting grammar — grids, primitives, themes | [Visualization](../visualization/index.md) |
| The 177 built-in systems | [Systems](../systems/index.md) |
| Chaos, certified end to end | [Anatomy of a chaotic attractor](chaotic-attractor.md) |
| Where chaos comes from | [The road to chaos](bifurcations.md) |
| Only data, no equations | [Reconstruction from one signal](reconstruction.md) |
| Multistability in depth | [Basins & multistability](basins-multistability.md) |
| Noise | [Noise-driven dynamics](noise-driven.md) |
