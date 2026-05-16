# Milestone N3 — Variational ODE Lyapunov in Rust

Status: WIP
Depends on: N2 (stepper + IR + dispatch)
Estimated scope: one chat
Design doc: [design/native-solver-migration.md](../design/native-solver-migration.md)

## Read before writing code

1. This file.
2. [`design/native-solver-migration.md`](../design/native-solver-migration.md)
   — **N3** section.
3. [`milestones/N2-rust-ode-stepper.md`](N2-rust-ode-stepper.md) — N3
   plugs into the trait shape designed there.
4. The current `ContinuousSystem.lyapunov_spectrum` implementation in
   `src/tsdynamics/base/ode_base.py` — uses `jitcode_lyap`.  N3 replaces
   it entirely.

## Mission

After N2 the ODE *integrate* path is pure Rust but
`lyapunov_spectrum` still goes through `jitcode_lyap` (which itself
auto-derives the variational system).  N3 ports that final pipeline to
Rust:

1. Symbolically derive `J(x)` from `_equations` once per system
   (SymEngine already does this — N1's `_lowering` reuses the trees;
   N3 adds the "augmented state" build-out).
2. Lower the variational RHS `(f(x), J(x) Δ_1, …, J(x) Δ_k)` to the
   same IR.  No new opcodes — `J(x)` and `J(x) Δ` are matrix-vector
   products built from existing arithmetic / `Mul` / `Sum` nodes.
3. Integrate the augmented system with N2's stepper (the trait was
   designed so this is a pure plug-in: `Rhs::dim() = dim * (n_exp + 1)`
   and an `IrInterpreterRhs::variational` impl).
4. Reorthogonalise tangent vectors with `nalgebra` QR every
   `reortho_interval` steps.  Accumulate `log(|R[i, i]|)` weighted by
   wall time; divide by total integration time at the end.

End state: `jitcode_lyap` is no longer imported.  JiTCODE is no longer
needed for *any* ODE compute path — it remains only as a SymEngine
convenience inside the lowering pass, and that goes away too once
SymEngine is used directly (N4-prep).

## Public API

Unchanged:

```python
exps = lor.lyapunov_spectrum(
    final_time=300.0, dt=0.1, burn_in=50.0,
    n_exp=None, method="DP8", rtol=1e-8, atol=1e-10,
)
```

`n_exp=None` defaults to `dim`.  `method=` reuses N2's catalogue.

## Internal API

```python
# src/tsdynamics/_native/__init__.py
def lyapunov_spectrum_ode(
    handle: OdeProblemHandle,
    t_span: tuple[float, float],
    ic: np.ndarray,
    params: np.ndarray,
    *,
    n_exp: int,
    reortho_interval: float,    # in time units, not steps
    method: str,
    rtol: float,
    atol: float,
    burn_in: float,
) -> np.ndarray:                 # shape (n_exp,)
    ...
```

The interval is in **time** to make the result solver-step-count
independent.

## Rust side

- New module `crates/tsdyn-ode/src/variational.rs`:
  - `struct VariationalRhs { rhs: IrInterpreterRhs, n_exp: usize }`
    impl-ing `Rhs` over the augmented state.
  - QR reortho via `nalgebra::Matrix::qr`.
  - Time-weighted accumulator.

- Extends `tsdyn-pyo3` with one new binding.

## Acceptance criteria

- [ ] Goldens: Lyapunov spectra for every continuous system stored
      under `tests/native/regression/ode/<system>.lyap.npz`, generated
      from the *current* `jitcode_lyap` path before N3 lands.
- [ ] Rust spectra match goldens to `rtol=1e-3` (Lyapunov spectra are
      inherently noisy; this is the loose-but-conventional tolerance).
- [ ] `Lorenz().lyapunov_spectrum(...)` → `[≈0.91, ≈0, ≈-14.57]` within
      `rtol=1e-2`.
- [ ] `jitcode_lyap` no longer imported by `src/tsdynamics/`.  Verified
      by grep.
- [ ] Wall-clock benchmark in `bench/RESULTS.md`.  Target: at parity
      with `jitcode_lyap` (within 1.5×).  N4 closes the rest.
- [ ] `uv run pytest --no-cov` passes.  `uv run ruff check` clean.

## Out of scope

- DDE Lyapunov — N5 follow-on.
- Covariant Lyapunov vectors (CLVs).  Future milestone.
- Floquet exponents for periodic orbits — M13.

## Open questions for the user

- The augmented-state ordering: store as `[x_0, …, x_{d-1}, Δ_{1,0}, …,
  Δ_{1,d-1}, Δ_{2,0}, …]` (state-first, then tangents) or interleaved?
  Recommendation: state-first.  Matches `jitcode_lyap`'s convention and
  is the format every reference uses.

- Allow Gram-Schmidt as a faster alternative to QR for low `n_exp`?
  Recommendation: no — QR is `n_exp²` work, fast enough that the
  reortho cost is dominated by integration.  Don't open the option
  unless a benchmark proves otherwise.
