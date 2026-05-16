# Status — updated 2026-05-17 (N2 ✅ — N3 🔧 variational Lyapunov begins)

Track **N2** (Pure-Rust ODE suite) is **feature-complete enough to close**: IR dispatch, Rosenbrocks,
golden regressions/benches from **N2.d**, and the reorganised **`crates/tsdyn-solver-base`** /
**`methods/`** tree for growing multi-solver backends ahead of **N5 DDE**.

**What just landed**

- **`tsdyn-solver-base`**: `uniform_time_grid` for ODE + future DDE output sampling.
- **`tsdyn-ode/src/methods/`**: tableau + explicit + implicit timestep clusters; **`driver`** as sole catalogue **`match`**.
- Design doc / **CLAUDE** updated with Rust layering; **CHANGELOG** rewritten “Changed” block.

**Where we are**

- **N3 — Variational ODE Lyapunov in Rust**: **ACTIVE** (`milestones/N3-rust-variational-lyapunov.md`). Next jobs: augmented IR/lowering sketch, **`lyapunov_spectrum_ode`** PyO3 surface, **`jitcode_lyap`** removal checklist, Lyap goldens.

**Defer / parallel**

| Item | Bucket |
|------|--------|
| **R2** (rayon sweep) | Separate Track C milestone (no authored spec yet in-repo). Safe parallel once picked up independent of N3. |
| **Vern6–8** timesteppers | Deferred N2 stretch goal. |

## Next action for a solver chat

Continue **N3** Lyapunov port (see **`N3-rust-variational-lyapunov.md`** § Mission + acceptance).

## Archived N2 housekeeping (carry if useful)

Jacobian gaps on **`Abs`**/**`sign`** systems; two golden NPZ gaps (**ExcitableCell**, **BlinkingRotlet**);
tightening `tests/test_ode_methods.py` vs DP8 tolerances (`5e-7` interim).

## How to resume

1. `STATUS.md` + `milestones/N3-rust-variational-lyapunov.md`
2. `design/native-solver-migration.md` § N3
3. `CLAUDE.md` § Rust crates
