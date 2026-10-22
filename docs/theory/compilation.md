---
description: How symbolic equations become an expression tape, which parameters trigger recompilation, and how Jacobians are derived.
---

<span class="ts-kicker">Theory · 01</span>

# Lowering pipeline

## From symbolic equations to an expression tape {#symbolic-ir-tape-no-toolchain-no-on-disk-cache}

When you first integrate or iterate a system, its one symbolic method is
**lowered** to an instruction-tape the Rust engine evaluates directly:

```text
_equations(y, t, **params)          your one symbolic method
        │  evaluated with SymEngine symbols
        ▼
SymEngine expression list           dim expressions, params as symbols
        │  tsdynamics._engine.compile.lower_ode (SSA lowering)
        ▼
IR tape                             a flat list of SSA instructions (+ the
        │                           analytic Jacobian when a stiff method needs it)
        ▼
the engine                          Cranelift-JITed (jit, default) or interpreted (interp)
```

Lowering and JIT compilation happen in the Python process using the installed
Rust extension. They need no external C/LLVM toolchain and create no disk cache.
Compiled evaluators are cached in memory; startup cost depends on equation count
and complexity. The map (`_step`) and DDE/SDE paths use the same representation
(`lower_map` / `lower_dde` / `lower_sde`).

## Structural vs. control parameters

Ordinary parameters are **not** baked into the tape. The engine reads their
values live from the system on every run, so a parameter sweep re-lowers
nothing:

```python
import numpy as np
import tsdynamics as ts

lor = ts.systems.Lorenz()
for rho in np.linspace(0, 50, 200):
    lor.with_params(rho=rho).run(final_time=50.0)   # one tape, 200 runs
```

Two kinds of parameter *do* change the tape's shape and therefore re-lower:

- **Structural parameters** — integer loop bounds like Lorenz-96's `N`, which
  determine how many expressions `_equations` returns. Declare them in
  `_structural_params`:

  ```python
  from tsdynamics import ContinuousSystem

  class Lorenz96(ContinuousSystem):
      params = {"f": 8.0, "N": 20}
      _structural_params = frozenset({"N"})
  ```

- **DDE delay values** — a delay shapes the engine's history buffer, so it is
  baked into the tape rather than read live. This is why
  [DDE parameter sweeps](../analysis/orbit-diagrams.md#cost-notes) re-lower per
  value.

## Automatically derived Jacobians {#jacobians-for-free}

Because the right-hand side exists symbolically, the Jacobian does too — no
hand-derivation, no finite differences:

```python
lor = ts.systems.Lorenz()
lor.rhs([1.0, 1.0, 1.0])        # ndarray (3,), the field itself: du/dt
lor.jacobian_sym()              # dim × dim SymEngine expressions, ∂f_i/∂y_j
lor.jacobian([1.0, 1.0, 1.0])   # ndarray (3, 3), evaluated at a state
```

`jacobian_sym` differentiates `_equations` with SymEngine; `jacobian` evaluates
a cached numeric form. The same symbolic differentiation lowers the Jacobian
into the *same* tape for the implicit (stiff) solver kernels and for the
extended variational system of [Lyapunov runs](lyapunov.md) (abs/sign
derivatives resolved a.e.). Hand-written `_jacobian` methods on ODE system
classes are not used by the numerical solver; the test suite compares them with
the automatically differentiated form.

`rhs(u, t=0.0)` is the function `jacobian` is the derivative *of* — the vector
field, evaluated with the system's current parameters. It is what the
`vector_field` / `flow_speed` / `streamlines` plots draw, so it is also how you
check one of those pictures against the equations. (The internal
`_rhs_numeric()` returns a pre-bound callable instead of a value, which is what
the [Poincaré crossing refinement](../analysis/poincare.md) and the
`reference`-backend cross-validation want in a tight loop.)

## See also

- [Backends](backends.md) — how the lowered tape is evaluated (interp / jit / reference)
- [Conventions](conventions.md) — what else is keyed on parameter hashes
