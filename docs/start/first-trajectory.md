---
description: Instantiate the Lorenz system, integrate it, read its named components, plot the attractor, and compute its Lyapunov spectrum — the full loop in a dozen lines.
---

<span class="ts-kicker">Start · 02</span>

# First trajectory

Every workflow in TSDynamics starts the same way: pick a system, integrate it,
and work with the result. This page walks the whole loop for the Lorenz
attractor — instantiate, integrate, read components, plot, and quantify — then
does the same for a discrete map. A dozen lines end to end.

## Instantiate a system

Built-in systems live under `tsdynamics.systems`. Each is a class you
instantiate with its parameters; the defaults reproduce the textbook attractor.

```python
import tsdynamics as ts

lor = ts.systems.Lorenz()          # sigma=10, rho=28, beta=8/3
```

Override any parameter by keyword — `ts.systems.Lorenz(rho=35.0)` — or set an
initial condition with `ic=[...]`. Leaving `ic` unset lets the library pick a
sensible starting point. (For convenience `ts.Lorenz` also resolves to the same
class, but the explicit `ts.systems.Lorenz` path is the one that autocompletes.)

## Integrate

A continuous system **integrates** over a time span. `integrate` returns a single
[`Trajectory`](../analysis/integration-and-methods.md).

<div class="ts-ref" markdown>

<div class="ts-item" markdown>
```python
traj = lor.integrate(final_time=100.0, dt=0.01)

traj.t.shape      # (10001,)   — the time grid
traj.y.shape      # (10001, 3) — state at each time
```

`dt` is only the **output grid** — where the solution is sampled into the
returned arrays. The internal solver is adaptive, choosing its own steps to meet
the error tolerances, so a coarse `dt` costs resolution but never accuracy. The
right-hand side is lowered to the Rust engine in-process and runs with no warmup;
the very first call is already at full speed.

Prefer explicit control? Pass `method=`, `rtol=`/`atol=`, `ic=`, or a
`backend=`. All of that is covered in
[Integration & methods](../analysis/integration-and-methods.md).
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
x   = traj["x"]              # (10001,)   — one named component
xz  = traj[["x", "z"]]       # (10001, 2) — several, in order
tail = traj.after(20.0)      # drop the transient before t = 20
t, y = traj                  # tuple-unpacking: t is (10001,), y is (10001, 3)
```

The trajectory also carries its full provenance in `traj.meta` — the system, its
parameters, the solver, `dt`, tolerances, the initial condition, and the library
version — and that metadata survives slicing.

## Plot it

With a plotting backend installed (`pip install "tsdynamics[viz]"`) a trajectory
draws itself. `to_plot_spec()` builds a plot description that auto-dispatches on
how many components you select — one is a time series, two a 2-D phase portrait,
three a 3-D one — and `.save()` renders it:

```python
# skip-doctest — .save() renders to a file, needs the optional tsdynamics[viz] backend
traj.to_plot_spec().save("lorenz.png")            # 3-D phase portrait (3 components)
traj.to_plot_spec(components="x").save("x.png")   # a single-component time series
```

For figures that overlay or panel several things, reach for the composition
front door `ts.viz.plot(...)`. The whole plotting layer — backends, themes, and
animation — is documented under [Visualization](../visualization/index.md).

## Quantify: the Lyapunov spectrum

The payoff of having the symbolic equations is that analysis is exact. The
Lyapunov spectrum — the average exponential rates at which nearby trajectories
separate — comes straight from the variational equations, no finite differences:

```python
lor.lyapunov_spectrum()                 # ≈ [0.9, 0.0, -14.6]
```

For a well-converged estimate, integrate for longer:

```python
lor.lyapunov_spectrum(final_time=500.0, dt=0.05)   # ≈ [0.91, 0.0, -14.57]
```

One positive exponent, one near-zero (the direction along the flow), one
strongly negative — the positive/zero/negative signature that certifies a
chaotic attractor. Feed the spectrum to the Kaplan–Yorke formula for a
fractal-dimension estimate:

```python
ts.kaplan_yorke_dimension(lor.lyapunov_spectrum(final_time=500.0, dt=0.05))
# ≈ 2.06
```

The [Lyapunov page](../analysis/lyapunov.md) covers the full spectrum, the
Jacobian-free maximal estimator, and estimation from a bare measured series.

## The same loop for a map

Discrete maps **iterate** instead of integrate — the discrete analogue of the
same `Trajectory`. A map takes a number of `steps` rather than a `final_time`:

```python
h = ts.systems.Henon()                        # a=1.4, b=0.3
orbit = h.iterate(steps=5000, ic=[0.1, 0.1])
orbit.t                                        # array([0, 1, 2, ...]) — step indices
orbit.y.shape                                  # (5000, 2)

h.lyapunov_spectrum(steps=5000, ic=[0.1, 0.1])   # ≈ [0.42, -1.62]
```

The map spectrum is computed by a QR decomposition of the Jacobian product in a
single forward pass on the engine. Everything downstream — the trajectory
object, named components, the analysis toolkit — behaves exactly as it does for
a flow.

!!! tip "Give a map an explicit initial condition"
    A random start can land in a divergent basin; the library retries from a new
    random point if that happens, but passing a known-good `ic=[...]` (as above)
    keeps a worked example reproducible and warning-free.

## Next

[**03 · The mental model**](concepts.md) — the four families, the one stepping
protocol every system shares, and the derived wrappers that turn a flow into a
map.
