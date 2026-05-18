# Status — updated 2026-05-17 (N3 landed)

Track **N2** — **`integrate`** path is **functionally DONE** enough to unblock **N3**:
Rust catalogue + regressions/benches landed. Stretch items (calibrated chaotic-vs-JIT statistics, genesis goldens wording) live under **`milestones/N2-rust-ode-stepper.md` § Acceptance criteria / N2.x backlog**.

---

**N3 — Variational ODE Lyapunov — DONE**

- `ContinuousSystem.lyapunov_spectrum` runs the **Rust** variational integrator + **QR** (`lyapunov_spectrum_ode` in `tsdyn-native`).
- **112** Lyapunov golden files under `tests/native/regression/ode/*.lyap.npz`; **3** catalogue systems **waived** (non-finite augmented RHS — see `tests/test_ode_lyapunov_goldens.py`).
- Piecewise-smooth catalogue ODEs gained explicit `_jacobian` where autodiff left `Derivative(sign)` (CellularNeuralNetwork, AnishchenkoAstakhov, StickSlipOscillator, Colpitts, FluidTrampoline).
- `jitcode_lyap` is **not** imported under `src/tsdynamics/`.

---

**Active milestone (suggested):** **N4 — Cranelift JIT for the IR** *or* continue **N2.x** backlog / bench polish — pick in the next chat.

Details: **`milestones/N3-rust-variational-lyapunov.md`** (acceptance checklist), **`design/native-solver-migration.md`** § N3.

---

## Next action

1. Run **`uv run pytest --no-cov`** on the branch before merge if not already.
2. Start **N4** scoping (or **R2** / **V1** if parallelising tracks) using **`ROADMAP.md`**.
