# Status — updated 2026-05-16 after N1 landed

Current milestone: **none — N1 closed. Pick the next from ROADMAP.md.**
Phase: pick. Track A (M1, M2), Track B (V1), Track C (R2), and Track E
(N2) are all unblocked and can run in parallel chats.
Last-touched files: `crates/**`, `src/tsdynamics/base/_ir.py`,
`src/tsdynamics/base/_tracer.py`, `src/tsdynamics/base/_lowering.py`,
`src/tsdynamics/base/map_base.py`,
`src/tsdynamics/systems/discrete/geometric_maps.py`,
`src/tsdynamics/_native/__init__.py`,
`tests/test_native_maps.py`, `tests/native/regression/*.npz`,
`scripts/generate_map_goldens.py`, `bench/maps_bench.py`,
`bench/RESULTS.md`, `pyproject.toml`, `.planning/**`.

## What's done

- **M0**: planning framework bootstrapped.
- **R1**: Rust toolchain + maturin + CI wheels — landed 2026-05-16.
- **N1**: Rust map stepper — landed 2026-05-16.
  - **Symbolic IR shipped** in `crates/tsdyn-core/src/ir.rs`. Op set
    sized for all 26 built-in maps:
    `Const, Var, Param, Add, Sub, Mul, Div, Neg, Pow(i32), Mod, Sin,
    Cos, Exp, Log, Abs, Sqrt, Arccos, Sign, Where, Lt, Le, Gt, Ge, And`.
    Stack-machine bytecode wire format with stable opcode bytes;
    `Expr` enum reused by N3 / N4 / future kernels.
  - **Python tracer** in `src/tsdynamics/base/_tracer.py` —
    `__array_ufunc__` + `__array_function__` + operator overloads.
    Maps continue to use `np.cos` / `np.sin` / `np.where` as today;
    tracer intercepts and emits IR. `__bool__` and float `__pow__`
    raise `NotLowerableError` (caught by caller → falls back to Numba).
  - **Lowering** in `src/tsdynamics/base/_lowering.py` — calls
    `cls._step.py_func` and `cls._jacobian.py_func` to bypass the
    `@staticjit` njit wrapper at trace time, walks returned tuples /
    lists into IR.
  - **Rust kernels** in `crates/tsdyn-maps/src/lib.rs` — `iterate`
    (tight loop, inline NaN check) and `lyapunov_spectrum` (modified
    Gram-Schmidt QR on `dim × n_exp` tangent bundle, log-diag accumulator).
  - **Native module consolidation**: `tsdyn-smoke` retired, replaced by
    `tsdyn-native` (cdylib `tsdynamics._native._core`) housing `add_one`
    plus the two N1 map functions. One maturin manifest going forward.
  - **`Tent` / `Baker` rewritten** to `np.where` so they trace cleanly.
    `if`/`else` replaced; outputs match the Numba goldens bit-exactly.
  - **Goldens**: `scripts/generate_map_goldens.py` produced one `.npz`
    per built-in map under `tests/native/regression/` *before* any
    other N1 change landed.
  - **Tests**: `tests/test_native_maps.py` — 81 cases, all green.
    Trajectories match goldens to **0.0 (bit-exact)**; Lyapunov spectra
    to **5e-15** (vs. 1e-12 / 1e-6 target). Fallback path also tested
    via a deliberately non-lowerable map.
  - **Benchmarks** in `bench/RESULTS.md`. Tree-walking interpreter
    misses the 2× target on tight maps (0.36× on Henon at 1e7 steps;
    2.3× on Logistic — the simplest map). Documented as expected and
    closed in N4 (cranelift JIT consumes the same IR).
  - **Deviations from milestone doc**:
    - `@staticjit` was left as `staticmethod(njit(...))` rather than
      becoming a no-op — the fallback path needs Numba to compile. The
      "make it a no-op in N1" line in the milestone doc is wrong;
      revisit after N4 when Numba retires.
  - **Acceptance criteria** (from milestone): all green except the 2×
    benchmark target, which is a documented gap closed by N4.

## What's in progress

- Nothing. N1 is closed.

## Next action

Pick one of:

- **M1** (Trajectory enrichment): slice/decimate/derivative/project/window.
  Self-contained Track A work, no Rust dependency.
- **M2** (Event & section detection): powers Poincaré, return maps. Track A.
- **R2** (Rust parameter-sweep kernel): rayon-backed. Unlocks M3+.
- **V1** (Viz skeleton): DataSpec/Transform/Plotter live here. Track B.
- **N2** (Rust ODE stepper): biggest single Track E milestone but the
  N1 IR is in place, so this is now multi-chat but tractable.

Default recommendation: **M1**, since it's the smallest and most
self-contained. Then M2. By the time those land, N2 has a clearer scope.

## R1 follow-ups (still parked)

- SCM-driven versioning via `update_version.py`.
- Collapse `publish.yml` + `release.yml` + the wheel-emitting half of
  `wheels.yml` into a single tag-triggered publish workflow.
- Windows wheels: add to `wheels.yml` matrix after N2 proves no blockers.

## N1 follow-ups (parked)

- **Performance**: add a constant-folding pass on the IR (lift
  `1.4 * x**2` to `Mul(Const(1.4), Pow(Var(0), 2))` evaluated with
  fewer ops; the win is small per step but compounds). Optional —
  N4 cranelift makes it moot.
- **Disk-backed IR cache**: process-local for N1; revisit in N4 when
  cranelift codegen makes recompilation per-process expensive.
- **DynamicalSystems.jl comparison**: not yet recorded. Add when P1's
  benchmark harness lands.

## How to resume

A future chat should:

1. Read this file.
2. Pick a milestone from the "Next action" list (or whatever the
   user asks for).
3. Open `.planning/milestones/<chosen>.md` (or create it from the
   template if it doesn't yet exist).
4. After landing the milestone: tick it in `ROADMAP.md`, rewrite this
   `STATUS.md`, commit `.planning/` together with the code.
