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
