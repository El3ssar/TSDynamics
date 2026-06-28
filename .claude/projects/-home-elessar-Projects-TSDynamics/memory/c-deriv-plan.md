---
name: c-deriv-plan
description: C-DERIV (#37) stream scope/design — unify Lyapunov into TangentSystem, backend-neutral variational core
metadata:
  type: project
---

Claimed C-DERIV (#37, Closes) on 2026-06-15; worktree `.claude/worktrees/tsd-C-DERIV`, branch `stream/C-DERIV-derived-engine`, owns `src/tsdynamics/derived/**`. Deps C-FAM #34 + C-DATA #36 both CLOSED.

**Hard constraint:** `tsdynamics._rust` is NOT built in dev OR in CI's pytest job (ci.yml only installs build-essential for jitcode). So `backend="interp"/"jit"` raise `EngineNotAvailableError` everywhere in CI, and the slow `test_known_values.py` Lyapunov sweep relies on jitcode (ODE) / numba (map) / jitcdde (DDE). → Do NOT flip default Lyapunov mechanism to the engine; reference(scipy-over-tape) is too slow for the slow tier.

**Design (this PR):**
- `derived/_variational.py` (new): builds the extended variational ODE [state ⊕ k tangent vectors] symbolically (mirrors lower_ode: jitcode y/t syms, control p-syms, `_resolve_derivative_nodes` for a.e. Jacobian) and lowers via public `engine.compile.lower_expressions` → a `dim*(k+1)`-output Tape. Layout z[0:dim]=x, z[dim+i*dim:]=tangent col i.
- `TangentSystem` = the one Lyapunov engine. Modes: map (NumPy J·W+QR, **pre-step J** = correct convention), ode `backend="jitcode"` (default, existing jitcode_lyap path), ode `backend="interp"/"jit"/"reference"` (new: integrate ext tape via `run.integrate` per dt chunk + QR). New `TangentSystem.lyapunov_spectrum(...)` (burn-in + time-weighted average + meta).
- Family delegation (de-triplication): `DiscreteMap.lyapunov_spectrum` → `TangentSystem(self,k).lyapunov_spectrum(...)`; `ContinuousSystem.lyapunov_spectrum` → `TangentSystem(self,k,backend="jitcode").lyapunov_spectrum(...)`. **DDE stays jitcdde** (infinite-dim tangent space — documented exception, NOT via TangentSystem).
- jitcode delegation is behavior-preserving (jitcode local-exps * dt, weighted avg = identical). Touching families/{continuous,discrete}.py is cross-lane but §13b explicitly assigns this to C-DERIV and C-FAM is merged (no active owner).

PR title `build: [C-DERIV] ...` (build/ to avoid release). See [[v3-consolidation-pr]] §13b, [[c-fam-engine-seam]], [[reap-merged-worktrees]].
