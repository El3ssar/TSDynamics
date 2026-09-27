"""Prepared raw evaluation preserves numerical components, shape and ownership."""

import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from tsdynamics._engine.compile import (
    OP_ADD,
    OP_CONST,
    OP_DIV,
    OP_MUL,
    OP_PARAM,
    OP_SELECT,
    OP_SIGNBIT,
    OP_STATE,
    OP_TIME,
    Tape,
)

native = pytest.importorskip("tsdynamics._rust")


def _tape(jacobian=True):
    # f=[a*x+t,sqrt(y)], J=[[a,0],[0,.5/sqrt(y)]].
    return Tape(
        ops=np.array(
            [OP_STATE, OP_STATE, OP_PARAM, OP_TIME, OP_MUL, OP_ADD, 35, OP_CONST, OP_CONST, OP_DIV]
        ),
        a=np.array([0, 1, 0, 0, 2, 4, 1, 0, 0, 8]),
        b=np.array([0, 0, 0, 0, 0, 3, 0, 0, 0, 6]),
        imm=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]),
        outputs=np.array([5, 6]),
        jac_outputs=np.array([2, 7, 7, 9]) if jacobian else np.empty(0, dtype=int),
        n_state=2,
        n_param=1,
    )


def _expected(arguments, jacobian):
    x, y, t, a = np.moveaxis(arguments, -1, 0)
    with np.errstate(all="ignore"):
        root = np.sqrt(y)
        values = [a * x + t, root]
        if jacobian:
            values.extend([a, np.zeros_like(a), np.zeros_like(a), 0.5 / root])
    return np.stack(values, axis=-1)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("jacobian", [False, True])
@pytest.mark.parametrize("leading", [(), (1,), (7,), (2, 3), (0,), (2, 0, 3)])
def test_point_and_arbitrary_batch_shapes(jit, jacobian, leading):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), jit)
    arguments = np.broadcast_to(np.array([0.5, 4.0, 0.25, 2.0]), (*leading, 4)).copy()
    fn = evaluator.eval_jac if jacobian else evaluator.eval_rhs
    before = arguments.copy()
    actual = fn(arguments)
    np.testing.assert_array_equal(arguments, before)
    expected = _expected(arguments, jacobian)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert not np.shares_memory(actual, arguments)
    other = fn(arguments)
    assert not np.shares_memory(actual, other)
    other[...] = 123
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("layout", ["fortran", "strided", "reversed"])
def test_controls_and_clocks_follow_logical_row_order(jit, layout):
    arguments = np.arange(1.0, 49.0).reshape(12, 4)
    if layout == "fortran":
        arguments = np.asfortranarray(arguments)
    elif layout == "strided":
        arguments = np.repeat(arguments, 2, axis=-1)[:, ::2]
    else:
        arguments = arguments[::-1, ::-1]
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), jit)
    np.testing.assert_allclose(
        evaluator.eval_jac(arguments), _expected(arguments, True), rtol=0, atol=0
    )


@pytest.mark.parametrize("jit", [False, True])
def test_partial_nonfinite_rows_are_retained(jit):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), jit)
    arguments = np.array(
        [
            [2.0, -1.0, 0.5, 3.0],
            [1.0, -0.0, 0.25, 2.0],
            [np.nan, 4.0, 0.125, 1.0],
            [1.0, np.inf, 0.5, 2.0],
        ]
    )
    for jacobian in [False, True]:
        actual = (evaluator.eval_jac if jacobian else evaluator.eval_rhs)(arguments)
        expected = _expected(arguments, jacobian)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        mask = ~np.isnan(expected)
        np.testing.assert_array_equal(actual[mask].view(np.uint64), expected[mask].view(np.uint64))
    assert evaluator.eval_rhs(arguments)[0, 0] == 6.5


@pytest.mark.parametrize("jit", [False, True])
def test_large_finite_values_have_no_integration_magnitude_screen(jit):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), jit)
    arguments = np.array([1e200, 4.0, 0.0, 0.5])
    np.testing.assert_array_equal(evaluator.eval_rhs(arguments), [5e199, 2.0])


@pytest.mark.parametrize("jit", [False, True])
def test_prepared_lifetime_owns_code_and_tape(jit):
    tape = _tape()
    arrays = tape.to_arrays()
    evaluator = native.PreparedEvaluator(*arrays, jit)
    tape.imm[:] = 19
    arrays[4][:] = 0
    del tape, arrays
    gc.collect()
    native.clear_jit_cache()
    arguments = np.array([0.5, 4.0, 0.25, 2.0])
    np.testing.assert_array_equal(evaluator.eval_jac(arguments), _expected(arguments, True))


@pytest.mark.parametrize("jit", [False, True])
def test_one_prepared_object_has_independent_concurrent_scratch(jit):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), jit)
    matrices = [
        np.broadcast_to([i + 0.5, i + 1.0, 0.25, i + 2.0], (256, 4)).copy() for i in range(8)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(evaluator.eval_jac, matrices))
    for arguments, actual in zip(matrices, outputs, strict=True):
        np.testing.assert_array_equal(actual, _expected(arguments, True))


@pytest.mark.parametrize("arguments", [np.array(1.0), np.ones(3), np.ones((2, 5))])
def test_wrong_argument_width_is_refused(arguments):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), False)
    with pytest.raises(ValueError, match="width"):
        evaluator.eval_rhs(arguments)


def test_jacobian_is_not_invented_or_computed_implicitly():
    evaluator = native.PreparedEvaluator(*_tape(False).to_arrays(), False)
    with pytest.raises(ValueError, match="Jacobian"):
        evaluator.eval_jac(np.array([0.5, 4.0, 0.25, 2.0]))


def test_real_float64_boundary_rejects_complex_values():
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), False)
    with pytest.raises(TypeError):
        evaluator.eval_rhs(np.array([1 + 1j, 4.0, 0.25, 2.0]))


def test_declared_input_width_overflow_is_refused_before_compilation():
    tape = _tape(False)
    arrays = list(tape.to_arrays())
    arrays[-2:] = [np.iinfo(np.uintp).max, 1]
    with pytest.raises(ValueError, match="width"):
        native.PreparedEvaluator(*arrays, False)


@pytest.mark.parametrize("jit", [False, True])
def test_primal_output_count_can_differ_from_state_width(jit):
    tape = replace(_tape(False), outputs=np.array([5]))
    evaluator = native.PreparedEvaluator(*tape.to_arrays(), jit)
    arguments = np.array([[0.5, 4.0, 0.25, 2.0], [2.0, 9.0, -1.0, 3.0]])
    assert evaluator.input_width == 4 and evaluator.output_width == 1
    np.testing.assert_array_equal(evaluator.eval_rhs(arguments), [[1.25], [5.0]])


def test_non_square_jacobian_declaration_is_not_presented_as_a_complete_gradient():
    # Valid 1x1 wire shape, but the primal reads two state slots.
    tape = replace(_tape(False), outputs=np.array([5]), jac_outputs=np.array([2]))
    evaluator = native.PreparedEvaluator(*tape.to_arrays(), False)
    with pytest.raises(ValueError, match="equal state and output widths"):
        evaluator.eval_jac(np.array([0.5, 4.0, 0.25, 2.0]))


def test_malformed_tape_is_refused_at_preparation():
    arrays = list(_tape(False).to_arrays())
    arrays[0][0] = 999
    with pytest.raises(ValueError):
        native.PreparedEvaluator(*arrays, False)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("source", [OP_STATE, OP_PARAM, OP_TIME, OP_CONST])
@pytest.mark.parametrize("jacobian_only", [False, True])
def test_signbit_observes_exact_bits_through_all_input_sources(jit, source, jacobian_only):
    magnitudes = np.array(
        [
            0x0000000000000000,
            0x0000000000000001,
            0x000FFFFFFFFFFFFF,
            0x0010000000000000,
            0x3FF0000000000000,
            0x7FEFFFFFFFFFFFFF,
            0x7FF0000000000000,
            0x7FF0000000000001,
            0x7FF8000000000000,
            0x7FFABCDEF1234567,
            0x7FFFFFFFFFFFFFFF,
        ],
        dtype=np.uint64,
    )
    bits = np.concatenate([magnitudes, magnitudes | np.uint64(1 << 63)])
    expected = np.concatenate([np.zeros(len(magnitudes)), np.ones(len(magnitudes))])
    for value, sign in zip(bits.view(np.float64), expected, strict=True):
        # Use selected Jacobian wire outputs to exercise its separate native
        # body; this is not an automatic derivative claim about signbit.
        tape = Tape(
            ops=np.array([source, OP_SIGNBIT, OP_CONST]),
            a=np.array([0, 0, 0]),
            b=np.zeros(3, dtype=int),
            imm=np.array([value if source == OP_CONST else 0.0, 0.0, 0.0]),
            outputs=np.array([2 if jacobian_only else 1]),
            jac_outputs=np.array([1 if jacobian_only else 2]),
            n_state=1,
            n_param=1,
        )
        evaluator = native.PreparedEvaluator(*tape.to_arrays(), jit)
        arguments = np.array([value, value, value])
        original_bits = arguments.view(np.uint64).copy()
        wanted = np.array([0.0, sign] if jacobian_only else [sign, 0.0])
        np.testing.assert_array_equal(
            evaluator.eval_rhs(arguments).view(np.uint64), wanted[:1].view(np.uint64)
        )
        np.testing.assert_array_equal(
            evaluator.eval_jac(arguments).view(np.uint64), wanted.view(np.uint64)
        )
        np.testing.assert_array_equal(arguments.view(np.uint64), original_bits)


@pytest.mark.parametrize("jit", [False, True])
def test_computed_nan_sign_reaches_finite_composition(jit):
    # sqrt(x), signbit(sqrt(x)), 10*signbit(sqrt(x))-3, with finite x=-1.
    tape = Tape(
        ops=np.array([OP_STATE, 35, OP_SIGNBIT, OP_CONST, OP_MUL, OP_CONST, 11]),
        a=np.array([0, 0, 1, 0, 2, 0, 4]),
        b=np.array([0, 0, 0, 0, 3, 0, 5]),
        imm=np.array([0.0, 0.0, 0.0, 10.0, 0.0, 3.0, 0.0]),
        outputs=np.array([1, 2, 6]),
        n_state=1,
        n_param=0,
    )
    evaluator = native.PreparedEvaluator(*tape.to_arrays(), jit)
    actual = evaluator.eval_rhs(np.array([-1.0, 0.0]))
    assert np.isnan(actual[0])
    sign = float(actual[:1].view(np.uint64)[0] >> np.uint64(63))
    np.testing.assert_array_equal(actual[1:], [sign, 10 * sign - 3])


def _select_tape(jacobian_only=False):
    return Tape(
        ops=np.array([OP_STATE, OP_PARAM, OP_PARAM, OP_SELECT, OP_CONST]),
        a=np.array([0, 0, 1, 0, 0]),
        b=np.array([0, 0, 0, 1, 0]),
        imm=np.array([0.0, 0.0, 0.0, 2.0, 0.0]),
        outputs=np.array([4 if jacobian_only else 3]),
        jac_outputs=np.array([3 if jacobian_only else 4]),
        n_state=1,
        n_param=2,
    )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("jacobian_only", [False, True])
@pytest.mark.parametrize("layout", ["point", "nested", "fortran", "strided", "reversed"])
def test_select_copies_chosen_bits_with_nonfinite_unused_arms(jit, jacobian_only, layout):
    conditions = np.array([0.0, -0.0, 1.0, -2.0, np.inf, -np.inf, np.nan])
    arm_bits = np.array(
        [
            0,
            1 << 63,
            0x3FF0000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000001234,
            0xFFF8000000005678,
            0x7FF0000000000001,
            0xFFF0000000000001,
        ],
        dtype=np.uint64,
    )
    arms = arm_bits.view(np.float64)
    rows = np.array([[c, 0.0, left, right] for c in conditions for left in arms for right in arms])
    if layout == "point":
        rows = rows[-1]
    elif layout == "nested":
        rows = rows.reshape(len(conditions), len(arms), len(arms), 4)
    elif layout == "fortran":
        rows = np.asfortranarray(rows)
    elif layout == "strided":
        rows = np.repeat(rows, 2, axis=-1)[..., ::2]
    else:
        rows = rows[::-1]
    source_bits = rows.view(np.uint64).copy()
    wanted = np.where(rows[..., 0] != 0, source_bits[..., 2], source_bits[..., 3])
    evaluator = native.PreparedEvaluator(*_select_tape(jacobian_only).to_arrays(), jit)
    result = evaluator.eval_jac(rows)
    chosen_slot = 1 if jacobian_only else 0
    np.testing.assert_array_equal(result[..., chosen_slot].view(np.uint64), wanted)
    np.testing.assert_array_equal(result[..., 1 - chosen_slot], 0.0)
    np.testing.assert_array_equal(rows.view(np.uint64), source_bits)
    assert not np.shares_memory(result, rows)
    expected_rhs = np.zeros_like(wanted) if jacobian_only else wanted
    np.testing.assert_array_equal(evaluator.eval_rhs(rows)[..., 0].view(np.uint64), expected_rhs)
    assert evaluator.eval_rhs(np.empty((2, 0, 4))).shape == (2, 0, 1)


@pytest.mark.parametrize("encoded", [np.nan, np.inf, -np.inf, -1.0, 0.5, 2147483648.0, 3.0, 4.0])
def test_select_false_register_is_validated_at_native_boundary(encoded):
    arrays = list(_select_tape().to_arrays())
    arrays[3] = arrays[3].copy()
    arrays[3][3] = encoded
    with pytest.raises(ValueError, match="register"):
        native.PreparedEvaluator(*arrays, False)


@pytest.mark.parametrize("jit", [False, True])
def test_select_unused_domain_error_is_not_masked_into_a_wrong_finite_value(jit):
    # fmin(where(x>0, 1, sqrt(-x)), 2), evaluated directly over the wire.
    tape = Tape(
        ops=np.array([OP_STATE, OP_CONST, OP_CONST, OP_CONST, 52, 20, 35, OP_SELECT, 56]),
        a=np.array([0, 0, 0, 0, 0, 0, 5, 4, 7]),
        b=np.array([0, 0, 0, 0, 1, 0, 0, 2, 3]),
        imm=np.array([0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 6.0, 0.0]),
        outputs=np.array([7, 8]),
        n_state=1,
        n_param=0,
    )
    evaluator = native.PreparedEvaluator(*tape.to_arrays(), jit)
    np.testing.assert_array_equal(
        evaluator.eval_rhs(np.array([[1.0, 0.0], [-9.0, 0.0]])), [[1.0, 1.0], [3.0, 2.0]]
    )
