"""Engine parameter sensitivity ∂f/∂p, Hessian ∂²f/∂u², forward ∂u(t)/∂p (E-SENS).

Validates the exact-symbolic derivative moat the bifurcation/continuation and
gradient-fitting tracks build on:

- ``∂f/∂p`` (parameter Jacobian) matches a hand value AND a finite difference of
  the RHS to ``1e-7``, in pure Python (reference evaluator + Lambdify) and
  round-tripped through the compiled engine (interp);
- ``∂²f/∂u²`` (state Hessian) matches a hand value for a polynomial system and a
  finite difference of the state Jacobian for a transcendental one;
- forward sensitivity ``∂u(t)/∂p`` reproduces a closed-form analytic value
  (``dx/da`` of ``x'=ax`` is ``t·x(t)``) and a central finite difference of the
  Lorenz flow to ``1e-6`` in one engine pass, ``interp == jit`` bit-for-bit.

References
----------
The forward-sensitivity equation ``Ṡ = (∂f/∂u) S + ∂f/∂p`` is standard; see e.g.
Dickinson & Gelinas, "Sensitivity analysis of ordinary differential equation
systems", J. Comput. Phys. 21 (1976) 123-143.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.engine import run
from tsdynamics.engine.compile import (
    TapeCompileError,
    eval_tape_param_jac,
    lower_ode,
)
from tsdynamics.engine.problem import ODEProblem
from tsdynamics.engine.sensitivity import (
    Sensitivity,
    build_sensitivity_tape,
    forward_sensitivity,
    split_sensitivity,
)
from tsdynamics.families import ContinuousSystem

# The forward-sensitivity and engine round-trip tests run on the compiled engine
# (interp/jit); skip cleanly where the extension is absent (and auto-tag this
# module ``engine`` so the engine CI job selects it, stream I-XVAL).
pytest.importorskip("tsdynamics._rust")


# ---------------------------------------------------------------------------
# Test-local systems with closed-form derivatives
# ---------------------------------------------------------------------------


class LinDecay(ContinuousSystem):
    """``x' = a x`` — solution ``x(t) = x0 e^{a t}``, so ``∂x/∂a = t·x(t)`` exactly."""

    params = {"a": -0.7}
    dim = 1

    @staticmethod
    def _equations(y, t, a):
        return [a * y(0)]


class Brusselator(ContinuousSystem):
    """Brusselator ``x' = a - (b+1)x + x²w,  w' = b x - x²w`` (Prigogine-Lefever 1968).

    Exact ``∂f/∂a = [1, 0]`` and ``∂f/∂b = [-x, x]`` — a two-parameter hand value.
    """

    params = {"a": 1.0, "b": 3.0}
    dim = 2

    @staticmethod
    def _equations(y, t, a, b):
        x, w = y(0), y(1)
        return [a - (b + 1.0) * x + x**2 * w, b * x - x**2 * w]


class Pendulum(ContinuousSystem):
    """Damped driven pendulum ``θ'=ω, ω'=-(g/L)sinθ - c ω`` — a transcendental RHS.

    Used to exercise the Hessian on ``sin`` (``∂²ω'/∂θ² = (g/L) sinθ``) against a
    finite difference of the analytic Jacobian.
    """

    params = {"g": 9.81, "L": 1.0, "c": 0.2}
    dim = 2

    @staticmethod
    def _equations(y, t, g, L, c):
        import symengine

        return [y(1), -(g / L) * symengine.sin(y(0)) - c * y(1)]


class NoControl(ContinuousSystem):
    """``x' = -c x`` with ``c`` *structural* — no control parameters (``n_param = 0``)."""

    params = {"c": 1.0}
    dim = 1
    _structural_params = frozenset({"c"})

    @staticmethod
    def _equations(y, t, c):
        return [-c * y(0)]


# ---------------------------------------------------------------------------
# ∂f/∂p — exact parameter Jacobian (hand value + finite difference)
# ---------------------------------------------------------------------------


def _lorenz_dfdp_hand(u: np.ndarray) -> np.ndarray:
    """Hand-computed ``∂f/∂p`` of Lorenz, columns ``(sigma, rho, beta)``."""
    x, y, z = u
    return np.array([[y - x, 0.0, 0.0], [0.0, x, 0.0], [0.0, 0.0, -z]])


def _fd_param_jacobian(system, u, t=0.0, eps=1e-6) -> np.ndarray:
    """Central finite difference of the (reference-lowered) RHS wrt each control param."""
    names = list(system._control_params())
    cols = []
    for name in names:
        hi = system.copy()
        hi.params[name] = hi.params[name] + eps
        lo = system.copy()
        lo.params[name] = lo.params[name] - eps
        f_hi = run.eval_rhs(hi, u, t, backend="reference")
        f_lo = run.eval_rhs(lo, u, t, backend="reference")
        cols.append((f_hi - f_lo) / (2.0 * eps))
    return np.array(cols).T if cols else np.zeros((system.dim, 0))


def test_param_jacobian_lorenz_matches_hand_value() -> None:
    lor = ts.Lorenz()
    u = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(lor.parameter_jacobian(u), _lorenz_dfdp_hand(u), atol=1e-12)


def test_param_jacobian_lorenz_matches_finite_difference() -> None:
    lor = ts.Lorenz()
    for u in ([1.0, 2.0, 3.0], [-4.0, 0.5, 10.0]):
        u = np.asarray(u)
        np.testing.assert_allclose(lor.parameter_jacobian(u), _fd_param_jacobian(lor, u), atol=1e-7)


def test_param_jacobian_brusselator_matches_hand_and_fd() -> None:
    sys = Brusselator()
    u = np.array([0.7, 2.3])
    hand = np.array([[1.0, -u[0]], [0.0, u[0]]])  # cols (a, b)
    pj = sys.parameter_jacobian(u)
    np.testing.assert_allclose(pj, hand, atol=1e-12)
    np.testing.assert_allclose(pj, _fd_param_jacobian(sys, u), atol=1e-7)
    # Pin the (row = equation, col = parameter) orientation loudly: a transposed
    # ∂f/∂p would put -x at [1,0] instead of [0,1] (Lorenz is diagonal and cannot
    # catch this — Brusselator's off-diagonal ∂(x')/∂b = -x is the orientation key).
    assert pj[0, 1] == pytest.approx(-u[0]) and pj[1, 0] == 0.0


def test_param_jacobian_no_control_params_is_empty() -> None:
    sys = NoControl()
    pj = sys.parameter_jacobian([2.0])
    assert pj.shape == (1, 0)


def test_param_jacobian_sym_shape_and_order() -> None:
    rows = ts.Lorenz().parameter_jacobian_sym()
    assert len(rows) == 3 and all(len(r) == 3 for r in rows)  # dim × n_param


# ---------------------------------------------------------------------------
# ∂f/∂p reference evaluator + engine round-trip through tsdynamics._rust
# ---------------------------------------------------------------------------


def test_lowered_param_jacobian_reference_evaluator() -> None:
    lor = ts.Lorenz()
    u = np.array([1.0, 2.0, 3.0])
    tape = lower_ode(lor, with_param_jacobian=True)
    assert tape.has_param_jacobian and tape.n_param_jac == 3
    f_ref = run.eval_rhs(lor, u, backend="reference")
    deriv, pj = eval_tape_param_jac(tape, u, [10.0, 28.0, 8 / 3], 0.0)
    np.testing.assert_allclose(pj, _lorenz_dfdp_hand(u), atol=1e-12)
    np.testing.assert_allclose(deriv, f_ref, atol=1e-12)  # RHS output is unchanged


def test_param_jacobian_round_trips_through_engine() -> None:
    """``∂f/∂p`` exposed as RHS outputs evaluates identically on the compiled engine."""
    lor = ts.Lorenz()
    u = np.array([1.0, 2.0, 3.0])
    tape = lower_ode(lor, with_param_jacobian=True)
    pj_tape = tape.with_param_jac_as_outputs()
    prob = ODEProblem(tape=pj_tape, ic=u, system=lor)
    flat = run.eval_rhs(prob, u, 0.0, backend="interp")
    np.testing.assert_allclose(flat.reshape(3, 3), _lorenz_dfdp_hand(u), atol=1e-12)


def test_param_jac_outputs_not_in_wire_payload() -> None:
    """The parameter Jacobian is Python-side only — the FFI wire shape is unchanged."""
    tape = lower_ode(ts.Lorenz(), with_param_jacobian=True)
    plain = lower_ode(ts.Lorenz())
    assert tape.has_param_jacobian and not plain.has_param_jacobian
    # to_arrays() is the frozen 8-tuple either way (no param-Jacobian leaks across).
    assert len(tape.to_arrays()) == 8 == len(plain.to_arrays())


def test_with_param_jac_as_outputs_requires_param_jacobian() -> None:
    with pytest.raises(TapeCompileError, match="parameter Jacobian"):
        lower_ode(ts.Lorenz()).with_param_jac_as_outputs()


def test_eval_tape_param_jac_requires_param_jacobian() -> None:
    with pytest.raises(ValueError, match="parameter Jacobian"):
        eval_tape_param_jac(lower_ode(ts.Lorenz()), [1.0, 2.0, 3.0], [10.0, 28.0, 8 / 3])


def test_param_jac_outputs_length_validated() -> None:
    from tsdynamics.engine.compile import Tape

    good = lower_ode(ts.Lorenz(), with_param_jacobian=True)
    with pytest.raises(TapeCompileError, match="param_jac_outputs length"):
        Tape(
            ops=good.ops,
            a=good.a,
            b=good.b,
            imm=good.imm,
            outputs=good.outputs,
            n_state=good.n_state,
            n_param=good.n_param,
            param_jac_outputs=good.param_jac_outputs[:-1],  # wrong length (not dim*n_param)
        ).validate()


# ---------------------------------------------------------------------------
# ∂²f/∂u² — exact state Hessian (hand value + finite difference of the Jacobian)
# ---------------------------------------------------------------------------


def test_hessian_lorenz_matches_hand_value() -> None:
    """Lorenz is quadratic: the only nonzero second derivatives are the cross terms."""
    lor = ts.Lorenz()
    expected = np.zeros((3, 3, 3))
    expected[1, 0, 2] = expected[1, 2, 0] = -1.0  # ∂²(x(ρ-z)-y)/∂x∂z
    expected[2, 0, 1] = expected[2, 1, 0] = 1.0  # ∂²(xy-βz)/∂x∂y
    for u in ([1.0, 2.0, 3.0], [-5.0, 0.0, 7.0]):
        np.testing.assert_allclose(lor.hessian(u), expected, atol=1e-12)


def test_hessian_symmetric_in_last_two_axes() -> None:
    H = ts.Rossler().hessian([1.0, -2.0, 0.5])
    np.testing.assert_allclose(H, np.swapaxes(H, 1, 2), atol=1e-12)


def test_hessian_transcendental_matches_fd_of_jacobian() -> None:
    """``∂²f/∂u²`` of the pendulum matches a central FD of the analytic Jacobian."""
    sys = Pendulum()
    u = np.array([0.6, -0.3])
    H = sys.hessian(u)
    eps = 1e-6
    fd = np.empty((sys.dim, sys.dim, sys.dim))
    for j in range(sys.dim):
        du = np.zeros(sys.dim)
        du[j] = eps
        fd[:, :, j] = (sys.jacobian(u + du) - sys.jacobian(u - du)) / (2.0 * eps)
    np.testing.assert_allclose(H, fd, atol=1e-6)


def test_hessian_sym_shape() -> None:
    blocks = ts.Lorenz().hessian_sym()
    assert len(blocks) == 3
    assert all(len(b) == 3 and all(len(r) == 3 for r in b) for b in blocks)


# ---------------------------------------------------------------------------
# Forward sensitivity ∂u(t)/∂p — analytic, finite-difference, interp == jit
# ---------------------------------------------------------------------------


def test_forward_sensitivity_linear_analytic() -> None:
    """``dx/da`` of ``x' = a x`` is exactly ``t·x(t)`` (anchored to the closed form)."""
    s = LinDecay()
    sens = s.sensitivity(final_time=3.0, dt=0.25, ic=[2.0], method="dop853", rtol=1e-11, atol=1e-13)
    # Anchor the base trajectory to the true closed form first, so t·x(t) is a
    # genuinely external oracle (not the engine's own x(t) on both sides).
    np.testing.assert_allclose(sens.y[:, 0], 2.0 * np.exp(-0.7 * sens.t), atol=1e-9)
    np.testing.assert_allclose(sens.S[:, 0, 0], sens.t * sens.y[:, 0], atol=1e-9)


def test_forward_sensitivity_layout_dim_ne_nparam() -> None:
    """A ``dim=2, n_param=1`` system pins the ``S`` layout: a transpose can't even reshape.

    ``x' = -a x,  w' = -2 a w`` ⇒ ``∂x/∂a = -t·x(t)``, ``∂w/∂a = -2t·w(t)`` exactly.
    """

    class TwoDecayOneParam(ContinuousSystem):
        params = {"a": 0.5}
        dim = 2

        @staticmethod
        def _equations(y, t, a):
            return [-a * y(0), -2.0 * a * y(1)]

    s = TwoDecayOneParam()
    sens = s.sensitivity(
        final_time=2.0, dt=0.2, ic=[3.0, 1.0], method="dop853", rtol=1e-11, atol=1e-13
    )
    assert sens.S.shape == (sens.t.size, 2, 1)
    np.testing.assert_allclose(sens.S[:, 0, 0], -sens.t * sens.y[:, 0], atol=1e-9)
    np.testing.assert_allclose(sens.S[:, 1, 0], -2.0 * sens.t * sens.y[:, 1], atol=1e-9)


def test_forward_sensitivity_lorenz_matches_finite_difference() -> None:
    """``∂u(T)/∂p`` of Lorenz matches a central FD of the flow to 1e-6 in one pass.

    Every parameter column (sigma, rho, beta) is checked, so a transposed or
    column-swapped sensitivity matrix would fail (column ``i`` must reproduce the
    FD of perturbing parameter ``i`` specifically).
    """
    ic = [1.0, 1.0, 1.0]
    T, dt = 2.0, 0.01
    sens = ts.Lorenz(ic=ic).sensitivity(
        final_time=T, dt=dt, ic=ic, method="dop853", rtol=1e-11, atol=1e-13
    )
    assert sens.param_names == ["sigma", "rho", "beta"]

    def state_at_horizon(name: str, value: float) -> np.ndarray:
        m = ts.Lorenz(ic=ic)
        m.params[name] = value
        return m.integrate(final_time=T, dt=dt, ic=ic, method="dop853", rtol=1e-12, atol=1e-14).y[
            -1
        ]

    eps = 3e-4
    for i, name in enumerate(sens.param_names):
        base = ts.Lorenz().params[name]
        fd = (state_at_horizon(name, base + eps) - state_at_horizon(name, base - eps)) / (2.0 * eps)
        np.testing.assert_allclose(sens.final[:, i], fd, atol=1e-6, err_msg=f"column {name}")


def test_forward_sensitivity_interp_equals_jit_bit_for_bit() -> None:
    ic = [1.0, 1.0, 1.0]
    kw = dict(final_time=2.0, dt=0.02, ic=ic, method="dop853")
    si = ts.Lorenz(ic=ic).sensitivity(backend="interp", **kw)
    sj = ts.Lorenz(ic=ic).sensitivity(backend="jit", **kw)
    assert np.array_equal(si.y, sj.y)
    assert np.array_equal(si.S, sj.S)


def test_forward_sensitivity_reference_agrees_with_engine() -> None:
    ic = [1.0, 1.0, 1.0]
    kw = dict(final_time=1.5, dt=0.05, ic=ic, method="dop853", rtol=1e-10, atol=1e-12)
    ref = ts.Lorenz(ic=ic).sensitivity(backend="reference", **kw)
    eng = ts.Lorenz(ic=ic).sensitivity(backend="interp", **kw)
    np.testing.assert_allclose(ref.S, eng.S, rtol=1e-5, atol=1e-7)


def test_forward_sensitivity_stiff_method_agrees_with_explicit() -> None:
    """A stiff (``bdf``) sensitivity run lowers the augmented Jacobian and matches dop853."""
    sys = Brusselator()
    ic = [1.5, 3.0]
    kw = dict(final_time=4.0, dt=0.1, ic=ic, rtol=1e-9, atol=1e-11)
    explicit = sys.sensitivity(method="dop853", **kw)
    stiff = sys.sensitivity(method="bdf", **kw)
    np.testing.assert_allclose(stiff.S, explicit.S, rtol=1e-4, atol=1e-6)


def test_build_sensitivity_tape_with_jacobian_is_square() -> None:
    tape = build_sensitivity_tape(ts.Lorenz(), with_jacobian=True)
    assert tape.has_jacobian
    assert tape.jac_outputs.size == tape.dim * tape.dim  # 12 × 12 augmented Jacobian


def test_forward_sensitivity_no_control_params() -> None:
    sens = NoControl().sensitivity(final_time=1.0, dt=0.25, ic=[3.0], backend="reference")
    assert sens.S.shape == (sens.t.size, 1, 0)
    assert sens.param_names == []


# ---------------------------------------------------------------------------
# Sensitivity result object + extended-tape plumbing
# ---------------------------------------------------------------------------


def test_sensitivity_result_indexing_and_final() -> None:
    ic = [1.0, 1.0, 1.0]
    sens = ts.Lorenz(ic=ic).sensitivity(final_time=1.0, dt=0.1, ic=ic, backend="reference")
    assert isinstance(sens, Sensitivity)
    assert sens["rho"].shape == (sens.t.size, 3)
    np.testing.assert_array_equal(sens["rho"], sens.S[:, :, 1])
    assert sens.final.shape == (3, 3)
    with pytest.raises(KeyError, match="not a control parameter"):
        sens["nope"]
    assert "Lorenz" in repr(sens)


def test_build_sensitivity_tape_shape() -> None:
    lor = ts.Lorenz()
    tape = build_sensitivity_tape(lor)
    # dim·(1 + n_param) = 3·(1 + 3) = 12 augmented state inputs/outputs.
    assert tape.dim == 12
    assert tape.n_param == 3


def test_split_sensitivity_roundtrip() -> None:
    dim, n_param = 3, 2
    u = np.array([1.0, 2.0, 3.0])
    s = np.arange(6.0).reshape(dim, n_param)
    z = np.concatenate([u, s.reshape(-1)])
    u2, s2 = split_sensitivity(z, dim, n_param)
    np.testing.assert_array_equal(u2, u)
    np.testing.assert_array_equal(s2, s)


def test_forward_sensitivity_function_matches_method() -> None:
    ic = [1.0, 1.0, 1.0]
    kw = dict(final_time=1.0, dt=0.1, ic=ic, backend="reference")
    via_fn = forward_sensitivity(ts.Lorenz(ic=ic), **kw)
    via_method = ts.Lorenz(ic=ic).sensitivity(**kw)
    np.testing.assert_array_equal(via_fn.S, via_method.S)
