# TSDynamics v6 — release notes

v6 is a deliberate, un-shimmed break. **Nothing was deprecated; things moved, and
every removed name raises an error that prints the line to type instead.** That
error *is* the migration guide — you should never have to read this file to fix a
script. It is here for the overview.

Two sentences describe the whole release:

> You will understand **systems** (ODE, DDE, SDE, maps) and **trajectories**.
> Everything else stopped shouting.

---

## The five things you will notice first

**1. `run` is the only trajectory verb.**

```python
traj = lorenz.run(final_time=100.0, dt=0.01)      # was .integrate(...)
orbit = henon.run(steps=5000, transient=500)      # was .iterate(...)
```

`integrate` / `iterate` / `trajectory` are gone on every family and every derived
wrapper. Every `run` signature is now **closed** — a keyword that belongs to a
different family is refused *by name*, with the mathematical reason
(`final_time` on a map: "a map has no continuous time — its horizon is a count of
iterations").

`method=` became **`solver=`**: `solver=` picks a numerical kernel, `method=`
picks an *estimator* on an analysis (`max_lyapunov(method="kantz")`).

**2. Analyses are free functions.**

```python
ts.analysis.lyapunov_spectrum(lorenz)          # a property of the equations
ts.analysis.correlation_dimension(traj)        # a property of a point set
ts.analysis.kaplan_yorke_dimension(spectrum)   # a property of the answer above
```

There is no bound method on a system, a `Trajectory` or a result. The four
topical accessors (`.lyap` / `.chaos` / `.dims` / `.recurrence`) are deleted: a
convenience that silently pre-ran with the family's defaults reported
**K = −0.026** for Lorenz's 0–1 test where the honest answer is **0.999**.

Discovery replaces them, and it is generated, not hand-written:
`ts.analysis.<TAB>` is 53 names, `print(ts.analysis.__doc__)` groups all 50 by
*what you are holding*, and `ts.analysis.find(lorenz)` / `find("is this chaotic")`
answer the two questions people actually ask.

**3. The top level is seventeen names.**

Five classes you subclass, `Trajectory`, `plot`, three registries
(`systems` / `analysis` / `viz`), six exception classes, `__version__`.
Everything else lives at exactly one address one dot down, and a wrong guess
prints it. `ts.systems` is now searchable too — `names()` / `find()` / `get()`.

**4. The result repr *is* the answer.**

```
>>> ts.analysis.lyapunov_spectrum(lorenz)
LyapunovSpectrum  λ = [0.916, 0.0001893, -14.58]   chaotic · D_KY = 2.063   (Lorenz)
    (3 exponents · flow (realised zero) · λ > 0.0146)
```

`summary()` is deleted on every result — the repr became what it printed. The
verdict is also programmatic (`spectrum.chaotic`, `to_dict(full=True)`), and a
verdict is only stated when the data supports it: a horizon too short to decide
says *indeterminate at this horizon* rather than inventing a regime.

**5. Plotting is one front door with closure.**

```python
ts.plot(traj)                                    # the default view
ts.plot(traj, "phase_portrait", primitive="density")
ts.plot(a, b, "phase_portrait")                  # two orbits, one figure
ts.plot(vdp, t1, t2, "vector_field", "nullclines")
ts.viz.grid(p1, p2, p3, cols=2, share_color=True)
ts.plot(traj, animate=True, fps=30).save("comet.mp4")
```

A positional **string** says *how* to draw; anything else says *what* to draw;
and a finished `Plot` handed back in is just another thing to draw. That closure
is why grids-of-different-plots, movies-of-anything and escape-and-return all
work with no extra API. `p.fig` / `p.ax` / `p.axes` is the matplotlib escape
hatch — one dot, never a rewrite.

---

## Numbers moved (deliberately) in exactly one place

`dt` is an **output sampling interval**, not an accuracy knob. The adaptive
solvers now use real dense output instead of being forced to land on every
requested sample, so the answer no longer depends on how finely you sampled it.
To keep that from costing accuracy silently, the ODE defaults tightened to
`rtol=1e-9` / `atol=1e-12`.

Measured over 15 catalogue systems at the defaults: **median 1459× more accurate
for a median 1.74× cost**. A plain `.run()` is *more* accurate than it was in v5,
not less.

If you pinned `rtol=1e-6` explicitly, tighten it; or pass `max_step=dt` to
reproduce the old step regime exactly; or set `TSDYNAMICS_NO_DENSE_OUTPUT=1` to
reproduce pre-v6 numbers bit-for-bit.

`backend="jit"` (the Cranelift JIT) is now the default on every family. The
compiled evaluator is cached, so it is faster than the interpreter at *every* run
length — Gray–Scott's 20-value sweep went from 6.23 s to 0.10 s.

---

## Renames you may hit

| v5 | v6 |
|---|---|
| `system.integrate(...)` / `.iterate(...)` / `.trajectory(...)` | `system.run(...)` |
| `run(method=...)` / `reinit(method=...)` | `solver=` |
| `run(n=...)` on a map | `run(steps=...)` |
| `system.lyapunov_spectrum()` and every other bound analysis | `ts.analysis.<name>(system)` |
| `system.stroboscope(period=T)` | `system.poincare(period=T)` |
| `system.project(...)` / `.tangent(...)` / `.copies(...)` | `traj[["x","z"]]` · `ts.derived.TangentSystem(...)` · `system.ensemble(states)` |
| `system.is_discrete` | `system.family` (`"ode"` / `"dde"` / `"map"` / `"sde"`) |
| `system.meta` | a run records its own provenance on `traj.meta` |
| `ts.Lorenz` · `ts.correlation_dimension` · `ts.Box` · `ts.T` | `ts.systems.Lorenz` · `ts.analysis.correlation_dimension` · `ts.data.Box` · `ts.viz.T` |
| `bifurcation_diagram` · `basins_of_attraction` · `find_attractors` · `periodic_orbit` | `orbit_diagram` · `basins` · `attractors` · `periodic_orbits` |
| `component=` on any analysis or plot transform | `components=` (one spelling everywhere) |
| `ftle_field(time=)` | `ftle_field(final_time=)` |
| `return_map(method=)` | `return_map(kind=)` |
| the 7 dimension estimators' `tol=` | `flatness=` |
| `result.summary()` | `repr(result)` / `print(result)` |
| `PlotSpec` | `Plot` (the same class; `ts.viz.Plot`) |
| `p.grid(...)` on a plot | `p.gridlines(...)` (`ts.viz.grid` is the panel arranger) |

**Deleted outright, with no replacement here.** The generic time-series layer —
`analysis/entropy/`, `analysis/surrogate/` and the whole `transforms/` package
(PSD toolbox, detrend/normalize, Butterworth filters, feature extraction). The
governing rule is *phase-space methods stay; generic series statistics go*, and
they live in a companion library now. What stayed is what reconstructs or
measures phase space: delay embedding, recurrence/RQA, `lyapunov_from_data`, the
sagitta sampling tools.

---

## Writing your own system is unchanged, and better diagnosed

```python
class ShimizuMorioka(ts.ContinuousSystem):
    params = {"a": 0.75, "b": 0.45}
    variables = ("x", "y", "z")          # dim follows from this now

    @staticmethod
    def _equations(u, t, a, b):
        x, y, z = u(0), u(1), u(2)
        return [y, x - a * y - x * z, -b * z + x**2]
```

Every beginner mistake — `y[0]`, `x, y, z = u`, a missing `@staticmethod`,
`np.sin`, a missing `dim`, a structural parameter that is not a parameter — is
refused with your own source line echoed back and the corrected line printed.
