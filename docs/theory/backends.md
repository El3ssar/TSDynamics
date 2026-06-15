---
description: Choosing an integration backend — JiTCODE (C) vs the diffsol Rust/LLVM backend.
---

# Backends

ODE integration runs through one of two backends, selected per call with the
`backend=` argument of [`integrate`](../analysis/integrate.md).

| | `jitcode` (default) | `diffsol` | `auto` |
|---|---|---|---|
| Engine | symbolic → C → compiled `.so` | symbolic → DiffSL → LLVM JIT (Rust solvers) | picks `diffsol` if installed, else `jitcode` |
| Toolchain | needs a **C compiler** | **no compiler** — prebuilt wheels | — |
| Install | core | `pip install "tsdynamics[diffsol]"` | — |
| Cold start | seconds (C compile) | sub-second (LLVM JIT) | — |
| Families | ODE · DDE · maps | **ODE only** | ODE only |

```python
import tsdynamics as ts

ts.Lorenz().integrate(final_time=100, dt=0.01)                   # jitcode (default)
ts.Lorenz().integrate(final_time=100, dt=0.01, backend="diffsol")  # explicit Rust path
ts.Lorenz().integrate(final_time=100, dt=0.01, backend="auto")     # diffsol if installed
```

## Choosing a solver

`integrate(method=...)` selects the solver. Names are SciPy-style and map onto
each backend's kernels:

| `method` | JiTCODE | diffsol |
|---|---|---|
| `RK45` / `dopri5` | dopri5 (explicit, default) | tsit45 |
| `DOP853` | dop853 | tsit45 |
| `LSODA` / `VODE` | lsoda / vode (implicit, stiff) | bdf |
| `tr_bdf2`, `esdirk34` | — | (implicit) |

**Stiff systems need an implicit solver.** An explicit method (the default
`RK45`) can fail outright on a stiff right-hand side. Systems known to need an
implicit solver declare it themselves — e.g. `Oregonator`, `Duffing`,
`SprottL`, `SprottP`, `SprottJerk` set ``_default_method = "LSODA"`` — so
`sys.integrate()` "just works" without the caller having to know. When you
define a stiff system of your own, set ``_default_method`` on the class.

> Broadening solver coverage (more methods, automatic stiffness detection,
> per-system hints) is an ongoing Phase-2 goal — see the project roadmap.

## The diffsol backend

`backend="diffsol"` translates the *same* symbolic `_equations` to
[DiffSL](https://martinjrobins.github.io/diffsol/) — a small DSL that the
Rust [diffsol](https://github.com/martinjrobins/diffsol) crate JIT-compiles
through LLVM — and integrates with Rust solver kernels. Initial conditions
and parameters are solve-time inputs, so one compiled module serves every IC
and every parameter value (the same caching economics as the JiTCODE path,
without the C-compiler dependency). SciPy-style method names map onto the
Rust solvers: `RK45`/`dopri5`/`DOP853` → `tsit45`; `LSODA`/`VODE` → `bdf`;
plus `tr_bdf2` and `esdirk34`.

**Coverage.** The translator handles **all 118 built-in ODE systems**
(guarded on every CI run by `tests/test_diffsol_translation.py`), and the
trajectories are cross-validated against JiTCODE — the curated sample on
every `slow` run, the full catalogue nightly.

**Speed** (Lorenz/Rössler/Chen, 1000 time units, `rtol=1e-9`, 100k output
points, warm caches; regenerate with `benches/bench_backends.py`):

| system | jitcode | diffsol | scipy | diffsol speedup |
|---|---|---|---|---|
| Lorenz | 1.20 s | **0.10 s** | 18.5 s | 11.9× |
| Rössler | 0.89 s | **0.03 s** | 5.6 s | 28.7× |
| Chen | 1.40 s | **0.15 s** | 28.8 s | 9.2× |

## Limitations

- **ODE only.** Delay systems and maps keep their own backends. (A Rust DDE
  solver and an SDE family are a future milestone — see the project roadmap.)
- The RHS must use functions DiffSL provides (`sin`/`cos`/`tan`/`exp`/`log`/
  `sqrt`/`abs`/`sign`/`tanh`/…); an unsupported construct raises a clear
  `DiffSLTranslationError` (none occur in the built-in catalogue).
- A couple of pathological built-ins integrate on the default JiTCODE backend
  but not on diffsol's solvers: **BlinkingRotlet** (near-discontinuous blinking
  protocol stalls every adaptive solver) and **WindmiReduced** (a
  `tanh(2200·…)` near-step plus fractional powers that go complex for negative
  arguments). They're excluded from the diffsol cross-validation sweep; use the
  default backend for them.
- Experimental: `jitcode` remains the default. Once the nightly full-catalogue
  cross-validation has a green track record, `auto` becomes the default so a
  plain `pip install "tsdynamics[diffsol]"` gives a zero-compiler install.

## The Rust core (experimental accelerator)

`tsdynamics` is pure-Python and installs with no compiler. The optional
`tsdynamics-core` package (a PyO3/maturin Rust crate) adds GIL-free numeric
kernels the Python layer offloads to when it is present.

Its keystone is an **expression tape VM**: the *same* symbolic `_equations`
that feed the DiffSL backend are lowered to a flat list of SSA instructions,
which a small Rust stack machine evaluates with no Python callbacks and no
runtime compiler. That makes **ensemble integration** — thousands of
independent trajectories from a grid of initial conditions — embarrassingly
parallel via [rayon](https://github.com/rayon-rs/rayon), the primitive that
makes basin-of-attraction and Monte-Carlo sweeps tractable.

The tape lowers and reproduces the RHS of **all 118 built-in ODE systems** to
machine precision (guarded on every CI run by
`tests/test_rustcore_translation.py`). Three solver kernels are available via
`method=`:

| `method` | kernel |
|---|---|
| `RK45` / `dopri5` (default) | adaptive Dormand-Prince 5(4), Hermite dense output |
| `RK4` | fixed-step classic RK4 |
| `stiff` / `Rosenbrock` / `LSODA` / `BDF` | L-stable linearly-implicit Euler + Richardson, with the **analytic Jacobian** |

The stiff kernel lowers the system's symbolic Jacobian into the *same* tape
(abs/sign derivatives resolved a.e.), so it needs no finite differences and no
Newton iteration — cross-validated against SciPy's `Radau`/`LSODA` on stiff Van
der Pol, Robertson, and Oregonator. The rayon ensemble is checked bit-identical
to a serial reference; a diverging trajectory raises (single integration) or
returns `NaN` (ensemble — so escaped initial conditions are flagged, not faked).
It is the foundation the planned Rust **SDE** and **DDE** solvers build on — see
the project roadmap.

> Experimental and not yet a user-facing `backend=`, and the crate is not yet on
> PyPI. Build it locally with `maturin build -m crates/tsdynamics-core/Cargo.toml`.

!!! info "Where this is heading — the v3 tiered engine"
    The accelerator above is the *seed* of a single Rust engine that will back
    **every** family (ODE, DDE, map, SDE). It exposes two numerically-identical
    evaluators behind one interface — a zero-warmup **interpreter** and a
    pure-Rust **Cranelift JIT** (no LLVM, so wheels stay trivial) — selectable
    as `integrate(..., backend="interp")` or `backend="jit"`, with
    `backend="reference"` a pure-Python oracle for cross-validation. These
    options already exist in the API surface. They remain **opt-in and under
    cross-validation**: the C-compiled `jitcode` path stays the default until
    the migration is gated complete, after which it (and the diffsol bridge) is
    retired in favour of the one engine.

## See also

- [Compilation pipeline](compilation.md) — the JiTCODE path and the shared symbolic core
- API: `tsdynamics.engine.diffsol`
