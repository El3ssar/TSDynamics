---
description: Simulate Lorenz and Hénon, select trajectory components, make a plot and estimate a Lyapunov spectrum.
---

<span class="ts-kicker">Start · 02</span>

# First trajectory

This example simulates the Lorenz system, selects components from its trajectory,
plots the orbit and estimates its Lyapunov spectrum. A Hénon-map example then
shows which parts of the workflow change for discrete time.

## Instantiate a system

Built-in systems live under `tsdynamics.systems`. Each is a class you
instantiate with its parameters. Its catalogue page lists the equations,
defaults and references; Lorenz uses the familiar convection parameters below.

```python
import tsdynamics as ts

lor = ts.systems.Lorenz(ic=[1, 1, 1])    # sigma=10, rho=28, beta=8/3
```

Override any parameter by keyword — `ts.systems.Lorenz(rho=35.0)` — and set the
starting state with `ic=[...]`. An explicit starting state makes your example
reproducible. If you leave it unset, the model uses its declared default or
draws a random initial condition; a random start need not lie in an attractor's basin.

## Run it

Use `run` to simulate a flow, delay system, map or SDE. It returns a
[`Trajectory`](../analysis/integration-and-methods.md).

<div class="ts-ref" markdown>

<div class="ts-item" markdown>
```python
traj = lor.run(final_time=100.0, dt=0.01)

traj.t.shape      # (10001,)   — the time grid
traj.y.shape      # (10001, 3) — state at each time
```

`dt` is only the **output grid** — where the solution is sampled into the
returned arrays. The internal solver is adaptive, choosing its own steps to meet
the error tolerances. A coarse `dt` stores fewer samples without relaxing those
tolerances, but it can hide fast oscillations or spikes in a plot. The
right-hand side is lowered to the Rust engine in-process and JIT-compiled to
native code on first use. Later calls reuse the in-memory cache while the model's
structure is unchanged; no manual compilation step is needed.

Prefer explicit control? Pass `solver=`, `rtol=`/`atol=`, `ic=`, `max_step=` or a
`backend=`. All of that is covered in
[Integration & methods](../analysis/integration-and-methods.md).

!!! note "`solver=` picks a numerical kernel; `method=` picks an estimator"
    `lor.run(solver="dop853")` chooses the
    integration kernel; `ts.analysis.optimal_delay(x, method="mi")` chooses an
    estimation algorithm. Passing `method=` to `run` raises and says so.
</div>

<figure class="ts-fig" markdown>
![Lorenz attractor and its component time series](../assets/figures/analysis/integrate.svg){ loading=lazy }
<figcaption>The Lorenz attractor (left) and the same trajectory as its stacked <code>x(t)</code>, <code>y(t)</code>, <code>z(t)</code> component series (right) — two views of one <code>Trajectory</code>.</figcaption>
</figure>

</div>

## Read named components

Because `Lorenz` declares `variables = ("x", "y", "z")`, the trajectory knows
its components by name. Index it like a dictionary of channels, slice off a
transient, or unpack it into `(t, y)` arrays:

```python
x    = traj["x"]              # (10001,)   — one named component
xz   = traj["x", "z"]         # several, in order → a Trajectory naming x and z
tail = traj.after(20.0)       # drop the transient before t = 20
t, y = traj.t, traj.y         # the two arrays: (10001,) and (10001, 3)
```

Strings select named components; integer indices, slices and masks select rows.
`traj[10:50]`, `traj[::5]` and `traj[mask]` all give you a trajectory back.

!!! warning "`traj.y` is the whole state array; `traj[\"y\"]` is the component named `y`"
    `traj.y` has shape `(T, dim)`; `traj["y"]` has shape `(T,)`.
    (Note also that `traj.y` is the **transpose** of SciPy's
    `solve_ivp(...).y`, which is `(dim, T)`. Every comparison in these docs
    writes `sol.y.T`.)

Iterating a trajectory yields one `(t, state)` pair per sample, and `len(traj)` is
the number of samples — so `len()` and iteration agree, as they should for any
Python container.

The trajectory also carries its full provenance in `traj.meta` — the system, its
parameters, the solver, `dt`, tolerances, the initial condition, and the library
version — and that metadata survives slicing.

## Plot it

With a plotting backend installed (`pip install "tsdynamics[viz]"`), `ts.plot`
chooses a default view from the number of components you
select — one is a time series, two a 2-D phase portrait, three a 3-D one — and
returns a `Plot` you can save, style or compose:

```python
# skip-doctest — .save() writes a file and needs the optional tsdynamics[viz] backend
ts.plot(traj).save("lorenz.png")                     # 3-D phase portrait
ts.plot(traj, components="x").save("x.png")          # a single-component series
ts.plot(traj, color="crimson", title="Lorenz")       # styled at the door
```

A positional string names a transform: `ts.plot(traj, "psd")` plots its power
spectrum, and
`ts.plot(traj_a, traj_b, "phase_portrait")` is both orbits on one pair of axes.
The whole plotting layer — grids, primitives, themes, animation — is documented
under [Visualization](../visualization/index.md).

## Quantify: the Lyapunov spectrum

Analysis functions take the system or data as their first argument. The
Lyapunov spectrum estimates the average exponential rates at which
nearby trajectories separate. For this model, TSDynamics differentiates the
symbolic equations and integrates the resulting variational equations. The
derivatives are symbolic; the reported exponents are numerical, finite-time estimates.

```python
ts.analysis.lyapunov_spectrum(lor)      # ≈ [0.9, 0.0, -14.6]
```

For a well-converged estimate, run for longer. The result prints its own answer:

```python
print(ts.analysis.lyapunov_spectrum(lor, final_time=500.0, dt=0.05))
# LyapunovSpectrum  λ = [0.9137, -0.0001274, -14.58]   chaotic · D_KY = 2.063   (Lorenz)
```

One positive exponent, one near-zero (the direction along the flow), one
strongly negative — numerical evidence consistent with a chaotic attractor.
A finite-time estimate is not a proof. The verdict word is only printed when it survives a tenfold
change of the zero tolerance; otherwise the result says `indeterminate at this
horizon`.

Feed the spectrum to the Kaplan–Yorke formula for a fractal-dimension estimate —
an analysis whose subject is *another analysis's answer*:

```python
spectrum = ts.analysis.lyapunov_spectrum(lor, final_time=500.0, dt=0.05)
ts.analysis.kaplan_yorke_dimension(spectrum)
# ≈ 2.06
```

The [Lyapunov page](../analysis/lyapunov.md) covers the full spectrum, the
Jacobian-free maximal estimator, and estimation from a bare measured series.

## The same loop for a map

A map has no time: its independent variable is an integer index, so the same
`run` verb takes a number of `steps` rather than a `final_time`, and returns the
same `Trajectory`.

```python
h = ts.systems.Henon()                        # a=1.4, b=0.3
orbit = h.run(steps=5000, ic=[0.1, 0.1])
orbit.t                                       # array([0, 1, 2, ...]) — step indices
orbit.y.shape                                 # (5001, 2)

ts.analysis.lyapunov_spectrum(h, n=5000, ic=[0.1, 0.1])   # ≈ [0.42, -1.62]
```

Use `steps=` for maps. Passing `final_time=` gives an error explaining the
required argument:

```
InvalidParameterError: final_time is not a valid Henon.run() keyword, got 10.0.
final_time is a *flow* keyword: a map has no continuous time — its horizon is a
count of iterations.
    Henon().run(steps=1000)
```

The map spectrum is computed by a QR decomposition of the Jacobian product in a
single forward pass on the engine. Map trajectories support the same component
selection and plotting methods as flow trajectories. Each analysis documents
which families it supports.

!!! tip "Give a map an explicit initial condition"
    A random start can land in a divergent basin; the library retries from a new
    random point if that happens, but passing a known-good `ic=[...]` (as above)
    keeps a worked example reproducible and warning-free.

## Next

[**03 · The mental model**](concepts.md) — the four families, the one stepping
protocol every system shares, and the derived wrappers that turn a flow into a
map.
