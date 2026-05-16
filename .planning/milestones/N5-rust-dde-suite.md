# Milestone N5 — Pure-Rust DDE solver suite

Status: TODO (blocked by N2 + N4)
Depends on: N2 (stepper, IR, dispatch), N4 (cranelift JIT — for perf;
            correctness-wise N5 can land on the N2 interpreter)
Estimated scope: **multi-chat**.  Probably the hardest milestone of
            Track E.
Design doc: [design/native-solver-migration.md](../design/native-solver-migration.md)

## Read before writing code

1. This file.
2. [`design/native-solver-migration.md`](../design/native-solver-migration.md)
   — **N5** section.
3. [`milestones/N2-rust-ode-stepper.md`](N2-rust-ode-stepper.md) — the
   `Stepper` / `Rhs` traits N5 reuses.
4. The current DDE path:
   - `src/tsdynamics/base/dde_base.py` — `DelaySystem`, `integrate`,
     `lyapunov_spectrum`.
   - JiTCDDE's source as the reference implementation.  Read
     specifically how it tracks breakpoints and how the discontinuity
     surfaces propagate through delays.
5. **Before any Rust is written**: check whether the Rust ecosystem has
   acquired a usable DDE crate since this milestone was specced.  If a
   maintained crate exists with the right shape, fork the milestone to
   use it.

## Mission

DDE integration runs end-to-end in Rust.  JiTCDDE goes away.

The hard parts:

1. **History buffer**.  Ring buffer of past `(t, y, slope_for_Hermite)`
   triples sized from the longest delay × headroom.  Random-access
   query via binary search + cubic Hermite interpolation.  Same Hermite
   that the ODE stepper's dense output uses.
2. **Breakpoint propagation**.  Each delay τ_i seeds a sequence of
   discontinuity surfaces at `t = t_0 + k · τ_i`.  These propagate
   through subsequent delays: a discontinuity at time `s` in the state
   produces a discontinuity at `s + τ_j` in `y(t - τ_j)`, of one order
   lower.  Track this as a priority queue; step controller refines `dt`
   when a breakpoint falls inside the proposed step.
3. **State-dependent delays**.  τ = τ(x(t)) — re-evaluate at every
   step.  Breakpoint set is recomputed lazily; some delays may need
   iterative resolution if the step crosses a breakpoint.
4. **Variational DDE for Lyapunov**.  Each tangent vector needs its
   *own* history buffer.  Memory grows linearly with `n_exp`.

## What stays

The Rust ODE stepper trait from N2 is reused unchanged.  The new piece
is a `DdeRhs` that wraps the IR-evaluated RHS with the history-buffer
lookup machinery:

```rust
struct DdeRhs {
    ir_rhs: IrInterpreterRhs,
    history: HistoryBuffer,
    delays: Vec<Delay>,    // either Const(f64) or Stateful(...)
}

impl Rhs for DdeRhs {
    fn eval(&self, t: f64, y: &[f64], params: &[f64], dy: &mut [f64]) {
        // 1. Resolve each delay τ_i (constant or τ_i(x(t))).
        // 2. Look up y(t - τ_i) via Hermite interpolation in self.history.
        // 3. Call ir_rhs.eval with (t, y, y_delayed_1, …, params).
    }
    ...
}
```

The IR opcode space grows by one: `DelayedVar(i, delay_idx)` —
"component i of the (delay_idx)-th historical lookup at the current
step's resolved delay".  Lowering tracks delays declared in
`DelaySystem._delay_params` and `_delays()`.

## Public API

Unchanged:

```python
mg = ts.MackeyGlass()
hist = lambda s: [1.0 + 0.1 * np.sin(0.2 * s)]
traj = mg.integrate(final_time=500.0, dt=0.5, history=hist,
                    method="DP5", rtol=1e-6, atol=1e-9)
exps = mg.lyapunov_spectrum(n_exp=1, dt=0.5, ic=traj.y[-1])
```

Same `method=` catalogue as N2 (explicit RK methods).  Stiff Rosenbrock
for DDEs is research-grade; defer to a follow-up.

## Acceptance criteria

- [ ] Goldens for every built-in DDE (Mackey-Glass, Ikeda DDE, etc.)
      from the *current* JiTCDDE path before N5 code lands.
- [ ] Rust trajectories match goldens within `rtol=1e-6` for the
      first 10 Lyapunov times.  Beyond that, statistical observables
      match.
- [ ] Lyapunov spectra match JiTCDDE_lyap goldens within `rtol=1e-2`.
- [ ] State-dependent delays exercised by at least one regression test
      (build a synthetic `τ(x)` system with known periodic solution).
- [ ] Discontinuity propagation correctness: a system with a known
      discontinuity surface (e.g. piecewise-linear `f` triggered at a
      breakpoint) integrates without solver-order degradation across
      the discontinuity.  Compare to an analytical solution.
- [ ] `jitcdde` no longer imported by `src/tsdynamics/`.  Verified by
      grep.
- [ ] Wall-clock benchmark in `bench/RESULTS.md` against JiTCDDE on the
      built-in DDE suite.  No hard speed target — DDEs are bound by
      memory bandwidth on the history lookup, and we'll learn the gap.
- [ ] `uv run pytest --no-cov` passes (full suite, slow included).

## Out of scope

- **Implicit DDE methods** (BDF-DDE, Radau-DDE).  Research-grade; no
  built-in needs them.
- **Neutral DDEs** (where `y'` appears on the RHS with a delay).  Not
  used by any built-in.
- **Distributed delays** (integral over a window).  Not used by any
  built-in.

## Suggested chat split

- **N5.a** — History buffer + Hermite interpolation + breakpoint queue.
  Pure Rust, no Python touch.  Unit-test in Rust against analytical
  Hermite values.
- **N5.b** — Constant-delay path: hook into N2's stepper for the
  built-in DDEs with constant delays.  Mackey-Glass / Ikeda / Lang-
  Kobayashi pass goldens.
- **N5.c** — State-dependent delays + variational DDE.
- **N5.d** — Polish, benchmarks, JiTCDDE removal.

## Open questions for the user (revisit when N5 starts)

1. Does the Rust DDE crate landscape have a maintainable starting
   point by then?  If so, fork accordingly.

2. History buffer sizing: fixed allocation at integration start
   (overshoot a bit, panic on overflow) or auto-grow?  Recommendation:
   fixed, panic on overflow with a clear error message pointing at the
   delay × `t_final` calculation the user should set up.  Auto-grow is
   nice but obscures memory cost in research-scale runs.

3. Per-tangent history buffers for Lyapunov: store as `n_exp` separate
   ring buffers or one fat buffer?  Recommendation: separate.  Cache
   locality per integration step is what matters, and tangent vectors
   are touched together with their own histories.
