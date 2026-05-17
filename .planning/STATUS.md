# Status — updated 2026-05-17 (Track E clarification)

Track **N2** — **`integrate`** path is **functionally DONE** enough to unblock **N3**:
Rust catalogue + regressions/benches landed. Stretch items (calibrated chaotic-vs-JIT statistics, genesis goldens wording) live under **`milestones/N2-rust-ode-stepper.md` § Acceptance criteria / N2.x backlog**.

---

### Can **N3** run in parallel?

- **Depends on Track E prerequisites only:** yes — **N3** needs today's Rust `integrate` + IR Jacobian plumbing. It **does not** wait on the chaotic-stat harness.
- **Parallel with Track C** (`R2`, …) or **V**/other rows: yes — orthogonal files (watch merge conflicts near `Cargo.toml`, `CHANGELOG`, `continuous` docs).
- **Parallel with someone still grinding N2.x backlog**: usually yes on **distinct branches**, but coordinate if both touch **`ode_base.py`** / crates layout.

---

**Active milestone:** **N3 — Variational ODE Lyapunov in Rust**
(`milestones/N3-rust-variational-lyapunov.md`, `design/native-solver-migration.md` § N3).

Recent N2-completion additions this session:

- `tests/test_ode_ir_cache.py` — verifies **`lower_ode_to_ir` called once per cache slice** (+ structural-split).
- **`crates/tsdyn-ode/src/events.rs`** skeleton for future M2 dense-output retrofit.
- **`tests/test_ode_methods.py`** — Lorenz-vs-DP8 dense alignment for **VERN6/7/8/9**, **DP5**, **TSIT5**, **BS3** with method-specific atol bands.

---

## Next action

Resume **N3** implementation (Rust variational LHS + QR + PyO3 + goldens workflow).

Consult **`milestones/N3-rust-variational-lyapunov.md`** + **`CLAUDE.md`** § Rust crates.
