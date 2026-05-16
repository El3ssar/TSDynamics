# Milestone N1 — Rust map stepper (drops Numba dispatch)

Status: DONE — landed 2026-05-16
Depends on: R1
Estimated scope: one chat (came in at one)
Design doc: [design/native-solver-migration.md](../design/native-solver-migration.md)

## Outcome

Shipped as planned, with two deviations:

1. **`@staticjit` stays as `staticmethod(njit(...))`** rather than
   becoming a no-op. The fallback path for non-lowerable user maps
   needs Numba to compile. Revisit when the fallback retires post-N4.
2. **Performance gate missed.** Tree-walking IR interpreter is 0.36×
   (Henon) to 2.34× (Logistic) of Numba's LLVM path — expected for an
   interpreter against JIT'd native code. The IR's strategic value is
   what carries the milestone; cranelift JIT (N4) closes the gap.

Trajectories match the Numba goldens **bit-exactly**; Lyapunov spectra
to **5e-15** (vs. 1e-12 / 1e-6 targets). See
`bench/RESULTS.md` for benchmarks and `tests/test_native_maps.py` for
the 81 regression cases.

## Motivation

N1 is the first end-to-end Rust replacement of an existing compute path. Discrete
maps are the easiest target: no adaptive stepping, no continuous extension, no
event detection. Just a tight loop over `X_{n+1} = f(X_n)`. The work in N1 mostly
goes into the symbolic-IR plumbing that every later N-milestone reuses.

After N1, `DiscreteMap.iterate` and `DiscreteMap.lyapunov_spectrum` run in Rust.
Numba is no longer in the dispatch path, though the `@staticjit` decorator stays
(as a no-op wrapper) so user subclasses keep working unchanged.

## API sketch

User-facing API is unchanged:

```python
h = ts.Henon()
traj = h.iterate(steps=10000)              # now runs in Rust internally
exps = h.lyapunov_spectrum(steps=10000)    # now runs in Rust internally
```

The only visible differences:

- Faster on large step counts (target: 2-5× over the Numba path on Henon at
  10^7 steps).
- Fewer first-call warmup penalties (no Numba JIT compile on first iterate).

New internal API:

```python
# src/tsdynamics/_native/__init__.py
def iterate_map(handle: ProblemHandle, ic: np.ndarray, steps: int) -> np.ndarray: ...
def lyapunov_spectrum_map(
    handle: ProblemHandle, ic: np.ndarray, steps: int,
    n_exp: int, reortho_interval: int,
) -> np.ndarray: ...
```

`ProblemHandle` is the opaque type from `tsdyn-core`. For N1 it wraps a
Python-side dispatch back into `cls._step` and `cls._jacobian` — i.e., **the
hottest inner kernel is still Python during N1's loop**, but the loop structure,
QR, divergence detection, and reortho machinery move to Rust. The full
symbolic→IR pipeline lands in N4.

Wait — that's the easy path. Better N1: do the symbolic-IR lowering for maps
specifically, since most map `_step` methods are simple polynomial expressions
that lower cleanly. Plan accordingly: N1 implements a minimal IR (just enough for
the 26 built-in maps to compile) and a small interpreter on the Rust side. The
full cranelift JIT remains for N4.

## Design

### IR for maps

```rust
// crates/tsdyn-core/src/ir.rs (new)
pub enum Expr {
    Const(f64),
    Var(usize),                   // X[i]
    Param(usize),                 // params[i]
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, i32),          // integer power for now
    Neg(Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Exp(Box<Expr>),
    Log(Box<Expr>),
    Abs(Box<Expr>),
    // …extend as built-in maps demand
}

pub struct CompiledMap {
    pub step: Vec<Expr>,          // length = dim
    pub jacobian: Vec<Vec<Expr>>, // dim × dim
    pub dim: usize,
    pub n_params: usize,
}
```

The interpreter is a plain `match`-based eval. Slow per op but fine: total work
is dominated by the loop count, not the per-op overhead, and we'll have cranelift
JIT for the ops by N4.

### Python lowering

```python
# src/tsdynamics/base/_lowering.py (new)
def lower_to_ir(step_fn, jacobian_fn, params: ParamSet, dim: int) -> CompiledMap:
    """Evaluate step_fn and jacobian_fn with SymEngine symbols, walk the result,
    emit IR. Raises NotLowerableError if the SymEngine tree contains nodes we
    don't yet handle (caller can fall back to the Python path during the
    transition period)."""
```

`@staticjit` becomes a no-op `staticmethod` wrapper that doesn't call Numba. We
keep the decorator name so user subclasses stay compatible.

### Rust kernels

`crates/tsdyn-maps/` (new):

- `pub fn iterate(map: &CompiledMap, ic: &[f64], params: &[f64], steps: usize) -> Vec<f64>`
- `pub fn lyapunov_spectrum(map: &CompiledMap, ic: &[f64], params: &[f64],
   steps: usize, n_exp: usize, reortho_interval: usize) -> Vec<f64>`

QR via `nalgebra`. Divergence detection inline (NaN check on state).

### `DiscreteMap.iterate` becomes

```python
def iterate(self, steps=1000, ic=None, max_retries=10):
    ic = self.resolve_ic(ic)
    compiled = self._compile()                  # caches per-class
    arr = _native.iterate_map(compiled, ic, steps)
    return Trajectory(t=np.arange(steps + 1), y=arr, system=self)
```

`_compile` lives on `DiscreteMap` (or a mixin), evaluates the step/jacobian
symbolically once, caches the result on the class.

## Files to create / modify

Create:
- `crates/tsdyn-core/src/ir.rs`
- `crates/tsdyn-maps/Cargo.toml`
- `crates/tsdyn-maps/src/lib.rs`
- `src/tsdynamics/base/_lowering.py`
- `src/tsdynamics/_native/_maps.pyi`
- `tests/native/regression/<26 built-in maps>.npz` (golden files, generated from
  the current Numba path before any code changes land)
- `tests/test_native_maps.py`

Modify:
- `crates/Cargo.toml` (add `tsdyn-maps` to workspace members)
- `pyproject.toml` (add `tsdyn-maps` to maturin's module list)
- `src/tsdynamics/base/map_base.py` (dispatch to Rust)
- `src/tsdynamics/utils/general.py` (`@staticjit` → no-op)
- `src/tsdynamics/_native/__init__.py` (re-export the new functions)

## Acceptance criteria

- [x] Golden files exist under `tests/native/regression/` for every built-in
      map, generated from the *current* Numba path before any other change is
      committed in this milestone.
- [x] All 26 built-in maps lower cleanly to IR (`lower_to_ir` doesn't raise on
      any of them).
- [x] `Henon().iterate(steps=10_000)` matches its golden file to within
      `1e-12` in float64. (Achieved: **bit-exact**.)
- [x] `Henon().lyapunov_spectrum(steps=10_000)` matches the golden Lyapunov
      spectrum to within `1e-6`. (Achieved: 4.44e-16.)
- [x] Equivalent checks pass for every other built-in map.
- [ ] `@staticjit` is a `staticmethod` no-op; user `_step` methods still callable
      from pure Python. **Deferred** — see Outcome above.
- [x] Wall-clock benchmark recorded in `bench/maps_bench.py`. Target ≥ 2×
      speedup **missed** on tight maps (interpreter overhead). Documented in
      `bench/RESULTS.md`; closed by N4 cranelift.
- [x] `uv run pytest --no-cov` passes (full suite, including slow tests).
      748 passed, 56 skipped.
- [x] `uv run ruff check src/ tests/` clean.

## Out of scope

- ODE / DDE migration (N2 / N5).
- Cranelift JIT (N4). The IR is interpreted in N1; JIT compilation comes in N4.
- Removing Numba from `pyproject.toml` (deferred to N7). Numba stays a
  transitive dev dep so the golden-file generation script keeps working.
- Trigonometric / transcendental IR ops not used by any built-in map (we extend
  the IR opportunistically as user maps demand).

## Open questions for the user

- Should `_compile` cache hit disk (so the IR is rebuilt only on `_step` source
  change) or stay process-local? Recommended: process-local cache for N1, disk
  cache lands in N4 alongside cranelift.
- Is there value in keeping a Python-fallback path during N1's lifetime in case
  IR lowering fails on some user-defined map? Recommended: yes — `lower_to_ir`
  raises `NotLowerableError`, caller catches it and uses the legacy Numba path
  for that one map. After N4 we revisit.
- Should we benchmark against DynamicalSystems.jl's `iterate` on Henon as part
  of N1? Recommended: yes, record it in `bench/RESULTS.md` even if we're
  slower at this stage. Track the gap explicitly.
