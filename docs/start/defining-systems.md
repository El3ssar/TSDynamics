---
description: Subclass a family base and write the dynamics as one symbolic method — a worked FitzHugh–Nagumo example end to end, plus maps and SDEs, auto-registration, and auto-generated docs.
---

<span class="ts-kicker">Start · 04</span>

# Defining systems

The built-in catalogue is a starting point, not a ceiling. Defining your own
system is the same short contract every built-in follows: name the components,
declare the parameters, and write one method holding the math. Subclass the right
family base and everything else — lowering to the engine, integration, the whole
[analysis toolkit](../analysis/index.md), and even a documentation page — comes
for free.

This page builds one ODE end to end, then shows the map and SDE variants.

## A worked example: FitzHugh–Nagumo

The [FitzHugh–Nagumo model](https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model)
is a two-variable caricature of an excitable neuron — a fast voltage-like
variable $v$ and a slow recovery variable $w$:

$$
\dot v = v - \tfrac{1}{3}v^3 - w + I, \qquad
\dot w = \tfrac{1}{\tau}\,(v + a - b\,w).
$$

It is not in the catalogue, so we can build it from scratch. Subclass
`ContinuousSystem`, name the components, declare the parameters, and translate
the equations into `_equations`:

```python
import tsdynamics as ts

class FitzHughNagumo(ts.ContinuousSystem):
    """FitzHugh–Nagumo excitable-neuron model."""

    variables = ("v", "w")                   # dim inferred = 2
    params = {"a": 0.7, "b": 0.8, "tau": 12.5, "I": 0.5}
    _reference = "FitzHugh (1961), Biophys. J. 1, 445–466"
    _doi = "10.1016/S0006-3495(61)86902-6"

    def _equations(y, t, a, b, tau, I):
        v, w = y(0), y(1)
        return (
            v - v**3 / 3 - w + I,
            (v + a - b * w) / tau,
        )
```

Four things are worth reading carefully.

**`variables`** names the components, and **`dim` follows from it** — declaring
both is no longer required, and declaring both while disagreeing raises at class
definition. The names let the trajectory be indexed by name (`traj["v"]`) and
give the docs and plots meaningful axis labels. (A system with no natural names
declares `dim = N` instead and gets `y0 … y{N-1}`.)

**`params`** is an ordinary dict of named control parameters with their defaults.
They become attributes (`fhn.tau`), and `_equations` receives them by name — the
parameter names must match the dict keys.

**`_equations`** returns one symbolic expression per component, in state order.
`y(i)` is the symbolic accessor for component `i`; `t` is the time symbol. The
body must build **symbolic** expressions — plain arithmetic plus
`symengine` functions (`sin`, `cos`, `exp`, `sqrt`, …). It is lowered to an
engine tape once, not called sample by sample, so NumPy, Python's `math`, and a
Python `if` on the state do not belong here (they cannot be traced).

**`@staticmethod` is optional.** The first slot of `_equations` is the state
accessor whatever you call it; only naming it `self` is a mistake, and the error
says so rather than demanding a decorator.

!!! tip "`system.info` is the frozen record of everything above"
    ```python
    print(FitzHughNagumo().info)
    ```
    prints the family and dimension, the rendered equations, the parameters,
    the component names, the citation and DOI, the solver defaults, and how many
    analyses accept this subject — one place to check that what you declared is
    what the library read.

!!! tip "Nonlinear functions come from `symengine`"
    For anything beyond `+ - * / **`, import from `symengine`:
    ```python
    from symengine import sin, cos, exp, sqrt
    # ... inside _equations:
    return (om, -(g / L) * sin(theta) - b * om)   # a damped pendulum
    ```

### It just works

There is no registration call and no build step. The class is a fully
functional system the moment it is defined:

```python
fhn = FitzHughNagumo()
traj = fhn.run(final_time=200.0, dt=0.05, ic=[-1.0, 1.0])

traj.y.shape          # (4001, 2)
traj["v"].min(), traj["v"].max()   # ≈ (-2.0, 1.9) — the relaxation spikes
```

Because a system *is* its equations, the analysis toolkit applies immediately.
The lone equilibrium is unstable, which is exactly why the state settles into a
limit cycle around it:

```python
fps = ts.analysis.fixed_points(fhn)
fps[0].x        # ≈ [-0.805, -0.131]  — the single equilibrium
fps[0].stable   # False → the orbit spirals out onto a limit cycle

ts.analysis.estimate_period(fhn.run(final_time=400, dt=0.05, ic=[-1.0, 1.0])["v"], dt=0.05)
# ≈ 39  — the relaxation-oscillation period
```

That is the entire loop: define the math once, and every quantifier — Lyapunov
spectra, Poincaré sections, bifurcation diagrams, dimensions, basins — composes
over it without another line of glue.

## Variable-dimension systems

If the *shape* of the equations depends on a parameter — a lattice size, a number
of coupled oscillators — declare that parameter in `_structural_params` so the
engine re-lowers when it changes (ordinary parameters are read live and never
bake into the tape):

```python
class Ring(ts.ContinuousSystem):
    params = {"N": 8, "k": 1.0}
    dim = 8
    _structural_params = frozenset({"N"})

    def _equations(y, t, N, k):
        return tuple(
            k * (y((i - 1) % N) - 2 * y(i) + y((i + 1) % N))
            for i in range(N)
        )
```

Set `dim` to match the default `N`; when a caller changes `N` and `dim` together,
the loop bound follows.

## Maps

A discrete map $\mathbf{x}_{n+1} = \mathbf{F}(\mathbf{x}_n)$ subclasses
`DiscreteMap`. Fill in `_step`, and — for the exact Lyapunov spectrum — the
`_jacobian`. Maps are the one family whose body uses **NumPy** functions:

```python
import numpy as np

class StandardMap(ts.DiscreteMap):
    """Chirikov standard (kicked-rotor) map."""

    params = {"k": 0.97}
    variables = ("theta", "p")

    @staticmethod
    def _step(X, k):
        theta, p = X
        p_new = p + k * np.sin(theta)
        theta_new = theta + p_new
        return (theta_new, p_new)
```

```python
sm = StandardMap()
sm.run(steps=3000, ic=[0.1, 0.1]).y.shape     # (3000, 2)
```

**`_jacobian` is optional.** Defining a map is *writing `_step`*, exactly as
defining a flow is writing `_equations`: the Jacobian is derived symbolically
from the step by the same trace the engine already performs. Write one by hand
only when the step cannot be traced (a Python `if` on the state), or when a
one-sided slope on a discontinuity is the meaningful answer:

```python
# skip-doctest — a class-body fragment, shown in place
    @staticmethod
    def _jacobian(X, k):                  # optional — this is what gets derived
        theta, p = X
        return ((1 + k * np.cos(theta), 1.0), (k * np.cos(theta), 1.0))
```

The state vector `X` is a genuine array here — a map `_step` is the one family
whose body unpacks its state — because the map kernel is traced through NumPy
ufuncs rather than built as a symbolic tree.

!!! warning "Map parameters are positional"
    `_step` and `_jacobian` take their parameters **positionally**, in the
    insertion order of `params`. The state vector `X` is the *first* argument
    (unpack it inside), and the parameter names must match the `params` order. A
    signature that disagrees is rejected with a `TypeError` at import, so a
    re-ordered `params` cannot fail silently.

## Stochastic equations

A diagonal-Itô SDE $dX_k = f_k\,dt + g_k\,dW_k$ subclasses `StochasticSystem`
and provides a symbolic **drift** and a symbolic **diffusion** (one noise
coefficient per component). Here is a bistable double-well diffusion, whose noise
lets the state hop between the two wells at $x = \pm 1$:

```python
class DoubleWellSDE(ts.StochasticSystem):
    """Overdamped gradient flow in a symmetric double well."""

    params = {"a": 1.0, "b": 1.0, "sigma": 0.4}
    variables = ("x",)
    _default_ic = [1.0]

    @staticmethod
    def _drift(y, t, a, b, sigma):
        return [a * y(0) - b * y(0) ** 3]     # −U'(x), the deterministic skeleton

    @staticmethod
    def _diffusion(y, t, a, b, sigma):
        return [sigma]                        # constant (additive) noise
```

```python
dw = DoubleWellSDE()
traj = dw.run(final_time=100.0, dt=0.01, seed=1)   # seed → reproducible path
```

Like maps, `_drift`/`_diffusion` take parameters positionally. Running an SDE
uses a fixed-step scheme (`solver="euler_maruyama"` by default, `"milstein"` for
strong order 1); `dt` is the noise scale $\sqrt{dt}$, and `seed=` fixes the
realisation.

Delay equations follow the same shape with `DelaySystem` and a delayed accessor
`y(i, t - tau)` — see [the mental model](concepts.md#dde-delaysystem).

## Code you already have: `WrappedSystem`

The four families all ask you for the *equations*, so the library can lower them
to the engine. When you already have a **stepper** — a foreign ODE solver, an
agent-based model, a hardware-in-the-loop rig, anything that advances a state
vector — `WrappedSystem` adapts it to the same protocol, and the whole analysis
toolkit applies unchanged. It is the one place where subclassing (well,
instantiating) a library type is genuinely the only path: there is no equation to
lower and so nothing to infer.

```python
import numpy as np

def step(u, n):                      # your stepper, whatever it is inside
    x = u[0]
    for _ in range(int(n)):
        x = 3.9 * x * (1.0 - x)
    return [x]

sysm = ts.WrappedSystem(step, dim=1, family="map",
                        initial=[0.5], variables=("x",))

traj = sysm.run(500)                       # a Trajectory, like any other family
float(ts.analysis.max_lyapunov(sysm, ic=[0.3]))     # ≈ 0.5 — chaotic
ts.plot(traj)                              # and it plots like any other
```

`step_fn(state, n_or_dt) -> new_state` is the whole contract. Pass
`family="ode"` and `n_or_dt` becomes a time increment instead of an iteration
count; `default_dt=` is what a bare `.step()` advances by. Anything that reads a
system through the protocol — orbit diagrams, Poincaré sections, ensembles,
basins — works on it.

## Auto-registration and auto-docs

Every concrete family subclass **registers itself at class-definition time** —
no decorator, no manual list. The moment `FitzHughNagumo` is defined it is in the
registry:

```python
from tsdynamics import registry

registry.get("FitzHughNagumo")     # SystemEntry(name='FitzHughNagumo', family='ode', dim=2, ...)
registry.get("FitzHughNagumo").is_builtin    # False — it is yours, not shipped
```

User classes register as **non-builtin**, so the built-in catalogue and its test
sweeps are unaffected, while your own tooling can still discover your systems
through the same registry.

For a system contributed to the library itself, registration is what wires it
into everything at once: the bulk test suite sweeps it, and the docs build
generates its page — equations rendered to LaTeX, a static figure, an
interactive 3-D viewer for 3-D flows — directly from the class. Adding a system
to the catalogue *is* adding its tests and its documentation. The optional
metadata that enriches that page — `_reference` and `_doi` for the citation,
`_known_lyapunov` for a literature-value test, `_default_ic` for a robust
starting point — is declared as class attributes alongside `params` and
`variables`. They are underscored because they are *facts the library reads*, not
part of the tab surface of the system you get back: `system.info` is where you
read them again.

---

## See also

- [The mental model](concepts.md) — the four families, the protocol, and the backends
- [Systems](../systems/index.md) — the 177 built-ins, each defined exactly this way
- [Analysis](../analysis/index.md) — the quantifiers that compose over any system you define
- [Fixed & periodic points](../analysis/fixed-points.md) — the equilibria and limit cycles used above
