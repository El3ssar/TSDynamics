"""Coverage for the symbolic→IR compiler (:mod:`tsdynamics.engine.compile`).

Runs without the compiled Rust engine: lowering needs only SymEngine + the
jitcode/jitcdde symbols, and the reference evaluator is pure Python.  This is the
fast-tier guard that the whole ODE catalogue lowers to a well-formed tape whose
reference evaluation reproduces the symbolic RHS (and analytic Jacobian) to
machine precision — the F1/E6 correctness contract — plus the map/DDE/SDE
lowering paths and the tape-validation invariants that mirror ``tsdyn-ir``.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.engine import compile as compile_ir
from tsdynamics.engine.compile import (
    DelaySlot,
    Tape,
    TapeCompileError,
    eval_tape,
    eval_tape_jac,
    lower_dde,
    lower_map,
    lower_ode,
    lower_sde,
    run_tape,
)
from tsdynamics.families.discrete import _unwrap_static

# ---------------------------------------------------------------------------
# Opcode wire values — pinned to the frozen IR (crates/tsdyn-ir/src/op.rs)
# ---------------------------------------------------------------------------


def test_opcode_wire_values_match_frozen_ir() -> None:
    """The Python emitter's opcode integers are the FFI contract; pin every one."""
    assert (compile_ir.OP_CONST, compile_ir.OP_STATE, compile_ir.OP_PARAM, compile_ir.OP_TIME) == (
        0,
        1,
        2,
        3,
    )
    assert (
        compile_ir.OP_ADD,
        compile_ir.OP_SUB,
        compile_ir.OP_MUL,
        compile_ir.OP_DIV,
        compile_ir.OP_POW,
        compile_ir.OP_POWI,
    ) == (10, 11, 12, 13, 14, 15)
    assert (compile_ir.OP_NEG, compile_ir.OP_RECIP) == (20, 21)
    assert compile_ir._FUNC_OPS["sin"] == 30 and compile_ir._FUNC_OPS["atanh"] == 46
    assert compile_ir._FUNC_OPS["Abs"] == 36 and compile_ir._FUNC_OPS["sign"] == 37
    # Non-smooth / piecewise block (stream E-OPS) — reserved wire range 50-69.
    assert (
        compile_ir.OP_LT,
        compile_ir.OP_LE,
        compile_ir.OP_GT,
        compile_ir.OP_GE,
        compile_ir.OP_EQ,
        compile_ir.OP_NE,
    ) == (50, 51, 52, 53, 54, 55)
    assert (compile_ir.OP_MIN, compile_ir.OP_MAX) == (56, 57)
    assert (compile_ir.OP_FLOOR, compile_ir.OP_CEIL) == (58, 59)
    assert (compile_ir.OP_MOD, compile_ir.OP_REM) == (60, 61)
    assert compile_ir._FUNC_OPS["floor"] == 58 and compile_ir._FUNC_OPS["ceiling"] == 59


# ---------------------------------------------------------------------------
# ODE lowering — the catalogue-wide correctness sweep
# ---------------------------------------------------------------------------


def test_every_ode_lowers_to_valid_tape(ode_entry) -> None:
    """Every built-in ODE lowers to a well-formed tape (RHS + Jacobian)."""
    system = ode_entry.cls()
    tape = lower_ode(system, with_jacobian=True)
    tape.validate()  # mirrors tsdyn-ir Tape::validate
    assert tape.dim == system.dim == tape.n_state
    assert tape.has_jacobian
    assert tape.jac_outputs.size == system.dim**2
    # control inputs = non-structural params, in order.
    structural = getattr(type(system), "_structural_params", frozenset())
    assert tape.control_names == [k for k in system.params if k not in structural]


def test_every_ode_rhs_matches_symbolic(ode_entry, rng) -> None:
    """The lowered RHS reproduces the symbolic RHS to machine precision."""
    system = ode_entry.cls()
    tape = lower_ode(system)
    f = system._rhs_numeric()
    p = np.array([float(system.params[k]) for k in tape.control_names])
    for _ in range(12):
        u = rng.standard_normal(system.dim)
        t = float(rng.uniform(0.0, 3.0))
        got = eval_tape(tape, u, p, t)
        ref = f(u, t)
        mask = np.isfinite(got) & np.isfinite(ref)
        assert np.allclose(got[mask], ref[mask], rtol=1e-9, atol=1e-11), (
            f"{ode_entry.name}: RHS lowering mismatch at u={u}, t={t}"
        )


def test_every_ode_jacobian_matches_symbolic(ode_entry, rng) -> None:
    """The lowered analytic Jacobian reproduces ``system.jacobian`` everywhere."""
    system = ode_entry.cls()
    tape = lower_ode(system, with_jacobian=True)
    p = np.array([float(system.params[k]) for k in tape.control_names])
    for _ in range(8):
        # Modest scale keeps exp/tanh systems in a comparable-magnitude regime
        # (the lowering is exact; only fp magnitude differs).
        u = rng.standard_normal(system.dim) * 0.6
        _, jac = eval_tape_jac(tape, u, p, 0.0)
        ref = system.jacobian(u, 0.0)
        mask = np.isfinite(jac) & np.isfinite(ref)
        assert np.allclose(jac[mask], ref[mask], rtol=1e-7, atol=1e-8), (
            f"{ode_entry.name}: Jacobian lowering mismatch at u={u}"
        )


def test_ode_catalogue_lowers_completely() -> None:
    """Aggregate guard: the whole ODE catalogue lowers (no straggler)."""
    failed = []
    for e in registry.all_systems(family="ode"):
        try:
            lower_ode(e.cls(), with_jacobian=True).validate()
        except Exception as exc:  # noqa: BLE001 - record, don't abort the sweep
            failed.append((e.name, str(exc).splitlines()[0][:70]))
    assert not failed, f"{len(failed)} ODE systems failed to lower: {failed}"


def test_abs_sign_jacobian_resolved_a_e() -> None:
    """A system with ``abs`` lowers its Jacobian a.e. (``d|u|/du = sign u``)."""
    # Build a tiny system using abs so we exercise the a.e. derivative path.
    import symengine

    class AbsSys(ts.ContinuousSystem):
        params = {"k": 2.0}
        dim = 1

        @staticmethod
        def _equations(y, t, k):
            return [-k * symengine.Abs(y(0))]

    s = AbsSys()
    tape = lower_ode(s, with_jacobian=True)
    p = np.array([2.0])
    # f = -2|u|, ∂f/∂u = -2 sign(u): +2 for u<0, -2 for u>0, 0 at u=0.
    _, j_pos = eval_tape_jac(tape, [3.0], p, 0.0)
    _, j_neg = eval_tape_jac(tape, [-3.0], p, 0.0)
    _, j_zero = eval_tape_jac(tape, [0.0], p, 0.0)
    assert j_pos[0, 0] == pytest.approx(-2.0)
    assert j_neg[0, 0] == pytest.approx(2.0)
    assert j_zero[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Reference evaluator semantics (mirror tsdyn-ir/src/reference.rs)
# ---------------------------------------------------------------------------


def test_reference_sign_is_zero_at_zero() -> None:
    """``sign(0) = 0`` — the a.e. convention shared with the Rust evaluator."""
    import symengine

    class S(ts.ContinuousSystem):
        params: dict = {}
        dim = 1

        @staticmethod
        def _equations(y, t):
            return [symengine.sign(y(0))]

    tape = lower_ode(S())
    assert eval_tape(tape, [0.0])[0] == 0.0
    assert eval_tape(tape, [3.5])[0] == 1.0
    assert eval_tape(tape, [-3.5])[0] == -1.0


def test_reference_pow_forms() -> None:
    """Integer ``Powi``, reciprocal, sqrt and runtime ``Pow`` all evaluate right."""
    import symengine

    class P(ts.ContinuousSystem):
        params = {"q": 2.5}
        dim = 1

        @staticmethod
        def _equations(y, t, q):
            u = y(0)
            return [u**3 + 1 / u + symengine.sqrt(symengine.Abs(u)) + symengine.Abs(u) ** q]

    tape = lower_ode(P())
    u = 1.7
    got = eval_tape(tape, [u], [2.5])[0]
    want = u**3 + 1 / u + abs(u) ** 0.5 + abs(u) ** 2.5
    assert got == pytest.approx(want, rel=1e-13)


def test_powi_uses_square_and_multiply() -> None:
    """``OP_POWI`` reduces by square-and-multiply (matching Rust ``f64::powi``),
    not NumPy's exp·log ``pow`` — the integer-power path of the reference oracle.
    Cross-language bit-exactness is asserted by the I-XVAL gate; here we check the
    reduction is mathematically correct across signs and the edge exponents."""
    from tsdynamics.engine.compile import _powi

    for base, exp in [(0.9, 50), (1.1, 13), (2.0, 10), (0.5, -4), (7.0, 1)]:
        assert float(_powi(np.float64(base), exp)) == pytest.approx(base**exp, rel=1e-12)
    assert float(_powi(np.float64(4.0), 0)) == 1.0
    assert float(_powi(np.float64(-3.0), 1)) == -3.0
    assert float(_powi(np.float64(2.0), -3)) == pytest.approx(0.125, rel=1e-15)


def test_run_tape_coerces_scalar_params() -> None:
    """``run_tape`` accepts a scalar/empty parameter argument without error."""
    import symengine

    class Q(ts.ContinuousSystem):
        params: dict = {}
        dim = 1

        @staticmethod
        def _equations(y, t):
            return [symengine.sin(y(0))]

    tape = lower_ode(Q())
    assert eval_tape(tape, [0.5])[0] == pytest.approx(np.sin(0.5))
    # explicit empty params, and a stray scalar, must both be tolerated
    assert run_tape(tape, [0.5], ()).size == tape.n_reg


def test_reference_evaluator_is_ieee754_on_singular_states() -> None:
    """Singular states yield inf/NaN silently (like the Rust evaluator), never raise.

    Runs under the project's strict ``filterwarnings = error`` policy, so a
    NumPy scalar warning escaping :func:`run_tape` would fail this test.
    """

    class Recip(ts.ContinuousSystem):
        params: dict = {}
        dim = 1

        @staticmethod
        def _equations(y, t):
            return [1 / y(0)]  # singular at u = 0

    tape = lower_ode(Recip())
    assert np.isinf(eval_tape(tape, [0.0])[0])  # 1/0 → +inf, not an exception

    class FracPow(ts.ContinuousSystem):
        params: dict = {}
        dim = 1

        @staticmethod
        def _equations(y, t):
            return [y(0) ** 1.5]  # a negative base to a fractional power → NaN in reals

    tape2 = lower_ode(FracPow())
    assert np.isnan(eval_tape(tape2, [-2.0])[0])


# ---------------------------------------------------------------------------
# Tape validation (mirror tsdyn-ir Tape::validate)
# ---------------------------------------------------------------------------


def _ok_tape() -> Tape:
    # f(u) = u0 * p0
    return Tape(
        ops=np.array([compile_ir.OP_STATE, compile_ir.OP_PARAM, compile_ir.OP_MUL], dtype=np.int32),
        a=np.array([0, 0, 0], dtype=np.int32),
        b=np.array([0, 0, 1], dtype=np.int32),
        imm=np.zeros(3),
        outputs=np.array([2], dtype=np.int32),
        n_state=1,
        n_param=1,
    )


def test_valid_tape_passes_validation() -> None:
    _ok_tape().validate()


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda t: t.b.__setitem__(2, 3), "strictly earlier"),  # forward reference
        (lambda t: t.a.__setitem__(2, 2), "strictly earlier"),  # self reference
        (lambda t: t.a.__setitem__(0, 5), "state index"),  # state index out of range
        (lambda t: t.a.__setitem__(1, 3), "param index"),  # param index out of range
        (lambda t: t.outputs.__setitem__(0, 9), r"outputs\[0\]"),  # output index out of range
    ],
)
def test_validation_rejects_malformed_tapes(mutate, needle) -> None:
    """Every structural invariant of the frozen IR is enforced on the Python side.

    The tape is a frozen dataclass, but its arrays are mutable, so we corrupt a
    single entry in place — the same way a buggy emitter would.
    """
    t = _ok_tape()
    mutate(t)
    with pytest.raises(TapeCompileError, match=needle):
        t.validate()


def test_validation_rejects_length_mismatch() -> None:
    """``ops``/``a``/``b``/``imm`` must all share a length."""
    t = Tape(
        ops=np.array([compile_ir.OP_STATE, compile_ir.OP_STATE], dtype=np.int32),
        a=np.array([0], dtype=np.int32),  # too short
        b=np.array([0, 0], dtype=np.int32),
        imm=np.zeros(2),
        outputs=np.array([1], dtype=np.int32),
        n_state=1,
        n_param=0,
    )
    with pytest.raises(TapeCompileError, match="mismatched"):
        t.validate()


def test_validation_rejects_bad_jacobian_shape() -> None:
    """A non-empty ``jac_outputs`` whose length is not ``dim*dim`` is rejected."""
    t = Tape(
        ops=np.array([compile_ir.OP_STATE, compile_ir.OP_PARAM, compile_ir.OP_MUL], dtype=np.int32),
        a=np.array([0, 0, 0], dtype=np.int32),
        b=np.array([0, 0, 1], dtype=np.int32),
        imm=np.zeros(3),
        outputs=np.array([2], dtype=np.int32),
        n_state=1,
        n_param=1,
        jac_outputs=np.array([2, 2], dtype=np.int32),  # dim=1 → needs len 1
    )
    with pytest.raises(TapeCompileError, match="dim"):
        t.validate()


def test_to_arrays_round_trips_shapes_and_dtypes() -> None:
    """``to_arrays`` yields the contiguous wire tuple the Rust FFI ingests."""
    tape = lower_ode(ts.Lorenz(), with_jacobian=True)
    ops, a, b, imm, outputs, jac, n_state, n_param = tape.to_arrays()
    assert ops.dtype == np.int32 and imm.dtype == np.float64
    assert ops.size == a.size == b.size == imm.size == tape.n_reg
    assert outputs.size == tape.dim and jac.size == tape.dim**2
    assert (n_state, n_param) == (3, 3)
    assert all(arr.flags["C_CONTIGUOUS"] for arr in (ops, a, b, imm, outputs, jac))


# ---------------------------------------------------------------------------
# Map lowering (symbolic trace)
# ---------------------------------------------------------------------------


def test_map_lowers_and_matches_step(map_entry, rng) -> None:
    """A traceable map lowers to a tape matching ``_step``; the rest raise cleanly."""
    system = map_entry.cls()
    try:
        tape = lower_map(system, with_jacobian=True)
    except TapeCompileError:
        pytest.skip(f"{map_entry.name}: _step not symbolically traceable (branching/ufunc)")
    step = _unwrap_static(type(system)._step)
    jac = _unwrap_static(type(system)._jacobian)
    params = system.params.as_tuple()
    for _ in range(10):
        u = rng.standard_normal(system.dim) * 0.3
        d, J = eval_tape_jac(tape, u)
        assert np.allclose(d, np.asarray(step(u, *params), dtype=float), rtol=1e-10, atol=1e-12)
        assert np.allclose(J, np.asarray(jac(u, *params), dtype=float), rtol=1e-7, atol=1e-9)


def test_baker_branchless_step_lowers() -> None:
    """Baker's piecewise step lowers since E-OPS (branch rewritten as ``np.where``).

    Its modular reductions lower via ``floor`` and the branch via a ``Piecewise``
    comparison-blend, so the formerly-unrepresentable map is now a straight-line
    tape — with a Jacobian that survives the floor/Piecewise derivative.
    """
    tape = lower_map(ts.Baker(), with_jacobian=True)
    assert tape.dim == 2
    assert tape.has_jacobian


def test_python_branch_on_state_raises_tape_compile_error() -> None:
    """A *Python* ``if`` on the state still cannot trace → a clear error.

    E-OPS lowers branchless selection (``np.where`` → ``Piecewise``), but a
    data-dependent Python branch is fundamentally untraceable: evaluating it
    forces a Boolean truth value of a symbolic Relational.
    """
    from tsdynamics.families import DiscreteMap
    from tsdynamics.utils import staticjit

    class _PyBranchMap(DiscreteMap):
        params = {"a": 0.5}
        dim = 1

        @staticjit
        def _step(X, a):
            x = X
            if x < a:  # Python branch on the state — unrepresentable
                return a * x
            return a * (1.0 - x)

        @staticjit
        def _jacobian(X, a):
            return [a]

    with pytest.raises(TapeCompileError, match="trace"):
        lower_map(_PyBranchMap())


def test_whole_map_catalogue_lowers() -> None:
    """Every built-in map lowers to a straight-line tape (the E-OPS acceptance)."""
    failed = []
    for e in registry.all_systems(family="map"):
        try:
            lower_map(e.cls())
        except TapeCompileError:
            failed.append(e.name)
    assert not failed, f"maps that no longer lower: {failed}"


# ---------------------------------------------------------------------------
# DDE lowering (delayed accesses → delay slots)
# ---------------------------------------------------------------------------


def test_every_dde_lowers_with_delay_slots(dde_entry) -> None:
    """Every built-in DDE lowers; delayed accesses become extra delay-slot inputs."""
    system = dde_entry.cls()
    tape, slots = lower_dde(system)
    tape.validate()
    assert tape.dim == system.dim
    assert tape.n_state == system.dim + len(slots)
    assert all(isinstance(s, DelaySlot) for s in slots)
    for k, s in enumerate(slots):
        assert s.input_index == system.dim + k
        assert 0 <= s.component < system.dim
        assert s.delay > 0.0


def test_dde_rhs_matches_manual_mackey_glass() -> None:
    """MackeyGlass lowers to the right RHS over (current, delayed) inputs."""
    mg = ts.MackeyGlass()
    tape, slots = lower_dde(mg)
    assert len(slots) == 1 and slots[0].component == 0
    assert slots[0].delay == pytest.approx(float(mg.tau))
    beta, gamma, n = float(mg.beta), float(mg.gamma), float(mg.n)
    y0, y_tau = 0.8, 1.3
    got = eval_tape(tape, [y0, y_tau])[0]
    want = beta * y_tau / (1.0 + y_tau**n) - gamma * y0
    assert got == pytest.approx(want, rel=1e-12)


def test_dde_out_of_range_delay_component_raises() -> None:
    """A delayed access to a non-existent component is rejected at lower time."""

    class BadDelay(ts.DelaySystem):
        params = {"tau": 1.0}
        dim = 1

        @staticmethod
        def _equations(y, t, tau):
            return [-y(0) + y(2, t - tau)]  # component 2 invalid for dim 1

    with pytest.raises(TapeCompileError, match="component 2"):
        lower_dde(BadDelay())


# ---------------------------------------------------------------------------
# SDE lowering (diagonal-Itô drift + diffusion)
# ---------------------------------------------------------------------------


class _OU(ts.ContinuousSystem):
    """An Ornstein–Uhlenbeck SDE stand-in (the E-SDE base class is a stub)."""

    params = {"theta": 1.5, "mu": 0.2, "sigma": 0.3}
    dim = 1

    @staticmethod
    def _equations(y, t, theta, mu, sigma):  # ODE drift, for resolve_ic etc.
        return [theta * (mu - y(0))]

    @staticmethod
    def _drift(y, t, theta, mu, sigma):
        return [theta * (mu - y(0))]

    @staticmethod
    def _diffusion(y, t, theta, mu, sigma):
        return [sigma + 0.0 * y(0)]  # additive (constant) diagonal noise


def test_sde_lowers_drift_and_diffusion() -> None:
    """A diagonal-Itô SDE lowers to a drift tape and a diffusion tape."""
    ou = _OU()
    lowered = lower_sde(ou, with_diffusion_jacobian=True)
    lowered.drift.validate()
    lowered.diffusion.validate()
    assert lowered.drift.control_names == ["theta", "mu", "sigma"]
    assert lowered.diffusion.has_jacobian  # ∂g/∂u for Milstein
    p = np.array([1.5, 0.2, 0.3])
    assert eval_tape(lowered.drift, [0.7], p)[0] == pytest.approx(1.5 * (0.2 - 0.7))
    assert eval_tape(lowered.diffusion, [0.7], p)[0] == pytest.approx(0.3)


def test_sde_multiplicative_noise_jacobian() -> None:
    """Multiplicative diffusion ``g = sigma*u`` lowers ``∂g/∂u = sigma`` for Milstein."""

    class GBM(ts.ContinuousSystem):
        params = {"mu": 0.1, "sigma": 0.4}
        dim = 1

        @staticmethod
        def _equations(y, t, mu, sigma):
            return [mu * y(0)]

        @staticmethod
        def _drift(y, t, mu, sigma):
            return [mu * y(0)]

        @staticmethod
        def _diffusion(y, t, mu, sigma):
            return [sigma * y(0)]

    lowered = lower_sde(GBM(), with_diffusion_jacobian=True)
    p = np.array([0.1, 0.4])
    _, jac = eval_tape_jac(lowered.diffusion, [2.0], p)
    assert jac[0, 0] == pytest.approx(0.4)  # ∂(sigma*u)/∂u
