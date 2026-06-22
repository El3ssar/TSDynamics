"""Backend-neutral variational (tangent-flow) lowering for ODE Lyapunov spectra.

The Lyapunov spectrum of a flow is the time-averaged logarithmic stretch of a
set of deviation (tangent) vectors carried along a trajectory.  The deviation
vectors obey the *variational equation* ``dW/dt = J(x, t) · W`` alongside the
base flow ``dx/dt = f(x, t)``.

This module is the engine path for ODE Lyapunov: it constructs the **extended**
ODE — base state stacked with ``k`` tangent vectors — symbolically and lowers it
to an engine :class:`Tape` through the public
:func:`tsdynamics.engine.compile.lower_expressions`.  That tape runs on any
evaluator behind the frozen ``Evaluator`` seam (the zero-warmup interpreter, the
Cranelift JIT, or the pure-Python reference oracle), so the Rust engine is the
variational integrator.

The renormalisation loop that consumes the extended flow (QR every step,
accumulate ``log|diag R|``) lives in :class:`~tsdynamics.derived.tangent.TangentSystem`,
the single home of the variational machinery for every family.

Extended-state layout (length ``dim·(k+1)``)::

    z[0 : dim]                                  base state  x
    z[dim + i*dim : dim + (i+1)*dim]            i-th tangent vector  (i = 0 … k-1)
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["build_variational_tape", "embed_extended", "split_extended"]


def build_variational_tape(system: Any, k: int) -> Any:
    """Lower the extended variational ODE of ``system`` with ``k`` tangents to a Tape.

    Mirrors :func:`tsdynamics.engine.compile.lower_ode` for the base flow, then
    appends ``k`` blocks of ``dim`` tangent equations ``dw_i/dt = J · w_i`` whose
    Jacobian entries reuse the same a.e.-resolved symbolic derivatives the stiff
    solver path uses (``abs``/``sign`` resolved a.e. via
    :func:`tsdynamics.families.continuous._resolve_derivative_nodes`).

    Structural parameters are folded to constants; control parameters become
    ``Param`` inputs in ``system._control_params()`` order, so the resulting tape
    reads its parameter vector from the (base) system with no recompile on a
    control-parameter change — exactly like the base ODE tape.

    Parameters
    ----------
    system : ContinuousSystem
        The flow whose variational dynamics to lower.
    k : int
        Number of tangent vectors (``1 ≤ k ≤ system.dim``).

    Returns
    -------
    Tape
        A lowered, validated tape with ``dim·(k+1)`` state inputs and outputs.
    """
    import symengine

    from tsdynamics.engine.compile import lower_expressions
    from tsdynamics.engine.symbols import state_time_symbols
    from tsdynamics.families.continuous import _resolve_derivative_nodes

    y, t_sym = state_time_symbols()

    dim = system.dim
    if not 1 <= int(k) <= dim:
        raise ValueError(f"k must be in [1, {dim}], got {k}")
    k = int(k)

    struct_vals = system._structural_vals()
    control_names = list(system._control_params())
    control_syms = {name: symengine.Symbol(f"p{i}") for i, name in enumerate(control_names)}

    f_raw = list(type(system)._equations(y, t_sym, **{**struct_vals, **control_syms}))
    if len(f_raw) != dim:
        raise ValueError(f"_equations must return {dim} expressions, got {len(f_raw)}")

    # Canonical state/time symbols (the y(i)/t function-application leaves are
    # not plain Symbols, so substitute them out before lowering).
    u_syms = [symengine.Symbol(f"u{j}") for j in range(dim)]
    t_canon = symengine.Symbol("t")
    subs = {y(j): u_syms[j] for j in range(dim)}
    subs[t_sym] = t_canon
    f = [symengine.sympify(e).subs(subs) for e in f_raw]

    # Jacobian jac[j][c] = ∂f_j/∂u_c, derivatives of abs/sign resolved a.e.
    jac = [
        [_resolve_derivative_nodes(symengine.sympify(f[j]).diff(u_syms[c])) for c in range(dim)]
        for j in range(dim)
    ]

    # Tangent inputs w_{i*dim + c}: component c of the i-th deviation vector.
    w_syms = [symengine.Symbol(f"w{m}") for m in range(dim * k)]

    exprs: list[Any] = list(f)
    for i in range(k):
        base = i * dim
        for j in range(dim):
            acc = symengine.Integer(0)
            for c in range(dim):
                acc = acc + jac[j][c] * w_syms[base + c]
            exprs.append(acc)

    return lower_expressions(
        exprs,
        [*u_syms, *w_syms],
        param_syms=[control_syms[name] for name in control_names],
        time_sym=t_canon,
        jacobian=False,
        control_names=control_names,
    )


def embed_extended(x: Any, w: np.ndarray) -> np.ndarray:
    """Pack a base state ``x`` (dim,) and tangent matrix ``W`` (dim, k) into ``z``.

    The tangent block is stored column-major: vector 0 first, then vector 1, …
    so it round-trips with :func:`split_extended`.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    w_arr = np.asarray(w, dtype=np.float64)
    return np.concatenate([x_arr, w_arr.T.reshape(-1)])


def split_extended(z: Any, dim: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack an extended state ``z`` into ``(x, W)`` with ``W`` of shape (dim, k)."""
    z_arr = np.asarray(z, dtype=np.float64).ravel()
    x = z_arr[:dim].copy()
    w = z_arr[dim : dim + dim * k].reshape(k, dim).T.copy()
    return x, w


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
