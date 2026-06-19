r"""Forward parameter sensitivity of ODE trajectories (stream E-SENS).

The *forward sensitivity* of a flow is the propagated derivative of the state
with respect to a parameter, ``S_{k,i}(t) = ∂u_k(t) / ∂p_i``.  Differentiating
the ODE ``du/dt = f(u, t, p)`` with respect to ``p_i`` gives the **forward
sensitivity equation**

.. math::

    \\frac{d}{dt}\\, S_{k,i} = \\sum_j \\frac{∂f_k}{∂u_j} S_{j,i}
                              + \\frac{∂f_k}{∂p_i},

a linear, inhomogeneous ODE driven by the exact symbolic state Jacobian
``∂f/∂u`` and the exact symbolic parameter Jacobian ``∂f/∂p`` (stream E-SENS —
the engine moat the finite-difference competition cannot match).

This module is the engine path for forward sensitivity: it constructs the
**extended** ODE — base state stacked with the ``dim × n_param`` sensitivity
columns — symbolically and lowers it to an engine :class:`~tsdynamics.engine.compile.Tape`
through the public :func:`tsdynamics.engine.compile.lower_expressions`.  That
single augmented tape runs on any evaluator behind the frozen ``Evaluator`` seam
(the zero-warmup interpreter, the Cranelift JIT, or the pure-Python reference
oracle) in **one engine pass**, so the Rust engine is the sensitivity
integrator — no new Rust, no FFI change (it reuses the same augmented-state
pattern as the Lyapunov variational core,
:mod:`tsdynamics.derived._variational`).

Extended-state layout (length ``dim·(1 + n_param)``)::

    z[0 : dim]                              base state  u
    z[dim + k*n_param + i]                  sensitivity  S_{k,i} = ∂u_k/∂p_i

Initial conditions are ``u(0) = u0`` and ``S(0) = ∂u0/∂p = 0`` (the standard
parameter-independent initial state).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Sensitivity", "build_sensitivity_tape", "forward_sensitivity", "split_sensitivity"]


# ``eq=False`` keeps the default identity-based equality/hash: the ndarray fields
# make an auto-generated ``__eq__`` raise on ``==`` and an auto-hash raise on
# ``hash`` (same trap :class:`Tape` guards explicitly).
@dataclass(frozen=True, eq=False)
class Sensitivity:
    """Forward parameter sensitivity of an ODE trajectory (stream E-SENS).

    The result of :func:`forward_sensitivity` /
    :meth:`tsdynamics.families.ContinuousSystem.sensitivity`: the base
    trajectory and its sensitivity to each control parameter, sampled on a
    shared time grid.  The sensitivity assumes a **parameter-independent initial
    state** (``S(t0) = ∂u0/∂p = 0``); a caller whose initial condition depends on
    a parameter would need that ``∂u0/∂p`` added to the start of ``S``.

    Attributes
    ----------
    t : ndarray, shape (n_t,)
        The output time grid.
    y : ndarray, shape (n_t, dim)
        The base trajectory ``u(t)``.
    S : ndarray, shape (n_t, dim, n_param)
        The forward sensitivities ``S[n, k, i] = ∂u_k(t_n) / ∂p_i``.
    param_names : list[str]
        The control-parameter names, in the order of the ``S`` last axis.
    system : object
        Back-reference to the source system.
    """

    t: np.ndarray
    y: np.ndarray
    S: np.ndarray
    param_names: list[str]
    system: Any = None

    @property
    def final(self) -> np.ndarray:
        """The final-time sensitivity matrix ``∂u(T)/∂p`` of shape ``(dim, n_param)``."""
        return self.S[-1]

    def __getitem__(self, name: str) -> np.ndarray:
        """Sensitivity wrt one named parameter: ``sens['rho']`` → ``(n_t, dim)``."""
        try:
            i = self.param_names.index(name)
        except ValueError:
            raise KeyError(
                f"{name!r} is not a control parameter; choose from {self.param_names}"
            ) from None
        return self.S[:, :, i]

    def __repr__(self) -> str:
        name = type(self.system).__name__ if self.system is not None else "?"
        return (
            f"Sensitivity({name}, n_t={self.t.size}, dim={self.y.shape[1]}, "
            f"params={self.param_names})"
        )


def split_sensitivity(z: Any, dim: int, n_param: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack an extended state ``z`` into ``(u, S)`` with ``S`` of shape ``(dim, n_param)``.

    The inverse of the extended layout this module integrates: ``z[:dim]`` is the
    base state and ``z[dim:]`` reshapes row-major to ``S[k, i] = ∂u_k/∂p_i``.
    """
    z_arr = np.asarray(z, dtype=np.float64).ravel()
    u = z_arr[:dim].copy()
    s = z_arr[dim : dim + dim * n_param].reshape(dim, n_param).copy()
    return u, s


def build_sensitivity_tape(system: Any, *, with_jacobian: bool = False) -> Any:
    """Lower the extended forward-sensitivity ODE of ``system`` to a Tape.

    Mirrors :func:`tsdynamics.engine.compile.lower_ode` for the base flow, then
    appends the ``dim × n_param`` sensitivity equations

    ``dS_{k,i}/dt = Σ_c (∂f_k/∂u_c) S_{c,i} + ∂f_k/∂p_i``

    whose Jacobian entries reuse the same a.e.-resolved symbolic derivatives the
    stiff-solver and variational paths use (``abs``/``sign`` resolved a.e. via
    :func:`tsdynamics.families.continuous._resolve_derivative_nodes`).

    Structural parameters are folded to constants; control parameters stay
    ``Param`` inputs in ``system._control_params()`` order, so the augmented tape
    reads its parameter vector from the (base) system with no recompile on a
    control-parameter change — exactly like the base ODE tape.  Sensitivity is
    defined with respect to the **control** parameters only.

    Parameters
    ----------
    system : ContinuousSystem
        The flow whose forward-sensitivity dynamics to lower.
    with_jacobian : bool, default False
        Emit the analytic Jacobian of the *augmented* system into the tape (the
        ``dim·(1 + n_param)`` square ``∂z'/∂z``), required when the sensitivity is
        integrated with an implicit/stiff kernel.

    Returns
    -------
    Tape
        A lowered, validated tape with ``dim·(1 + n_param)`` state inputs and
        outputs (``n_param`` is the number of control parameters; the tape is the
        plain base ODE when there are none).
    """
    import symengine

    from tsdynamics.engine.compile import lower_expressions
    from tsdynamics.engine.symbols import state_time_symbols
    from tsdynamics.families.continuous import _resolve_derivative_nodes

    y, t_sym = state_time_symbols()

    dim = system.dim
    struct_vals = system._structural_vals()
    control_names = list(system._control_params())
    n_param = len(control_names)
    control_syms = {name: symengine.Symbol(f"p{i}") for i, name in enumerate(control_names)}

    f_raw = list(type(system)._equations(y, t_sym, **{**struct_vals, **control_syms}))
    if len(f_raw) != dim:
        raise ValueError(f"_equations must return {dim} expressions, got {len(f_raw)}")

    # Canonical state/time symbols (the y(i)/t function-application leaves are not
    # plain Symbols, so substitute them out before lowering).
    u_syms = [symengine.Symbol(f"u{j}") for j in range(dim)]
    t_canon = symengine.Symbol("t")
    subs = {y(j): u_syms[j] for j in range(dim)}
    subs[t_sym] = t_canon
    f = [symengine.sympify(e).subs(subs) for e in f_raw]
    param_syms = [control_syms[name] for name in control_names]

    # Exact symbolic Jacobians: jac[k][c] = ∂f_k/∂u_c, dfdp[k][i] = ∂f_k/∂p_i.
    jac = [
        [_resolve_derivative_nodes(f[k].diff(u_syms[c])) for c in range(dim)] for k in range(dim)
    ]
    dfdp = [
        [_resolve_derivative_nodes(f[k].diff(param_syms[i])) for i in range(n_param)]
        for k in range(dim)
    ]

    # Sensitivity inputs s_{c*n_param + i} = S_{c,i} = ∂u_c/∂p_i.
    s_syms = [symengine.Symbol(f"s{m}") for m in range(dim * n_param)]

    exprs: list[Any] = list(f)
    for k in range(dim):
        for i in range(n_param):
            acc = dfdp[k][i]
            for c in range(dim):
                acc = acc + jac[k][c] * s_syms[c * n_param + i]
            exprs.append(acc)

    return lower_expressions(
        exprs,
        [*u_syms, *s_syms],
        param_syms=param_syms,
        time_sym=t_canon,
        jacobian=with_jacobian,
        control_names=control_names,
    )


def forward_sensitivity(
    system: Any,
    *,
    final_time: float = 100.0,
    dt: float = 0.02,
    t0: float = 0.0,
    ic: Any = None,
    method: str | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    backend: str = "interp",
) -> Sensitivity:
    """Integrate the forward parameter sensitivity of ``system`` in one engine pass.

    Builds the extended ODE (base state ⊕ sensitivity columns,
    :func:`build_sensitivity_tape`) and integrates it through the shared engine
    seam (:func:`tsdynamics.engine.run.integrate`).  The sensitivity is exact in
    the derivatives (analytic ``∂f/∂u`` and ``∂f/∂p``) and as accurate as the
    integrator in time — no finite differencing of the flow.

    Parameters
    ----------
    system : ContinuousSystem
        The flow to differentiate.
    final_time : float, default 100.0
        End of the integration window.
    dt : float, default 0.02
        Output sampling interval (the internal stepper is adaptive).
    t0 : float, default 0.0
        Start time.
    ic : array-like, optional
        Initial state at ``t0`` (resolved via ``system.resolve_ic``).  The
        sensitivity starts at ``S(t0) = 0`` (a parameter-independent IC).
    method : str, optional
        Integrator name; defaults to the system's ``_default_method``.  Use an
        explicit high-order method (e.g. ``"dop853"``) with tight tolerances for
        the sharpest sensitivities.
    rtol, atol : float
        Solver tolerances (default ``1e-8`` / ``1e-10``).
    backend : {"interp", "jit", "reference"}, default "interp"
        Where the augmented ODE is integrated.

    Returns
    -------
    Sensitivity
        The base trajectory and its sensitivity to each control parameter.
    """
    from tsdynamics import solvers
    from tsdynamics.engine.problem import ODEProblem
    from tsdynamics.engine.run import integrate

    dim = system.dim
    control_names = list(system._control_params())
    n_param = len(control_names)
    ic_arr = np.asarray(system.resolve_ic(ic), dtype=np.float64).ravel()

    # An implicit/stiff kernel needs ∂z'/∂z of the *augmented* system on the tape,
    # so resolve the method first and lower the augmented Jacobian when required —
    # mirroring run.integrate's auto-Jacobian for the base ODE path (which it does
    # *not* do for a pre-built Problem, so it must be done here).
    resolved = method if method is not None else system._default_method
    needs_jac = bool(solvers.resolve(resolved).build_kwargs.get("with_jacobian"))

    aug_tape = build_sensitivity_tape(system, with_jacobian=needs_jac)
    # z0 = [u0; vec(S0)] with S0 = 0 (the IC does not depend on the parameters).
    z0 = np.concatenate([ic_arr, np.zeros(dim * n_param, dtype=np.float64)])
    prob = ODEProblem(tape=aug_tape, ic=z0, t0=float(t0), system=system)

    # The pre-built ODEProblem fully determines the initial state (z0) and start
    # time, so only the grid/solver knobs are passed here (integrate ignores `ic`
    # for a pre-built Problem); `t0` is forwarded so the output grid starts there.
    traj = integrate(
        prob,
        final_time=final_time,
        dt=dt,
        t0=t0,
        method=resolved,
        rtol=rtol,
        atol=atol,
        backend=backend,
    )
    z = np.asarray(traj.y, dtype=np.float64)
    n_t = z.shape[0]
    y = z[:, :dim].copy()
    s = z[:, dim : dim + dim * n_param].reshape(n_t, dim, n_param).copy()
    return Sensitivity(
        t=np.asarray(traj.t, dtype=np.float64), y=y, S=s, param_names=control_names, system=system
    )
