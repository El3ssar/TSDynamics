"""Checked prepared evaluation keeps raw IEEE behavior and validation order."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from test_prepared_evaluator import _expected, _tape

from tsdynamics._engine.compile import OP_CONST, OP_DIV, OP_SELECT, OP_STATE, Tape
from tsdynamics.errors import InvalidParameterError

native = pytest.importorskip("tsdynamics._rust")


def _function(evaluator, jacobian, checked=True):
    name = "eval_jac" if jacobian else "eval_rhs"
    return getattr(evaluator, name + ("_checked" if checked else ""))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("jacobian", [False, True])
@pytest.mark.parametrize("leading", [(), (1,), (7,), (2, 3), (0,), (2, 0, 3)])
def test_checked_shape_bits_and_owned_outputs(jit, jacobian, leading):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), jit)
    arguments = np.broadcast_to(np.array([0.5, 4.0, 0.25, 2.0]), (*leading, 4)).copy()
    before = arguments.copy()
    actual = _function(evaluator, jacobian)(arguments)
    raw = _function(evaluator, jacobian, False)(arguments)
    expected = _expected(arguments, jacobian)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(actual.view(np.uint64), raw.view(np.uint64))
    np.testing.assert_array_equal(arguments, before)
    assert not np.shares_memory(actual, arguments)
    other = _function(evaluator, jacobian)(arguments)
    other[...] = 123
    assert not np.shares_memory(actual, other)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("jacobian", [False, True])
@pytest.mark.parametrize("layout", ["fortran", "strided", "reversed", "broadcast", "readonly"])
def test_checked_batches_follow_logical_input_layout(jit, jacobian, layout):
    arguments = np.arange(1.0, 49.0).reshape(12, 4)
    if layout == "fortran":
        arguments = np.asfortranarray(arguments)
    elif layout == "strided":
        arguments = np.repeat(arguments, 2, axis=-1)[:, ::2]
    elif layout == "reversed":
        arguments = arguments[::-1, ::-1]
    elif layout == "broadcast":
        arguments = np.broadcast_to(arguments[0], (2, 3, 4))
    else:
        arguments.setflags(write=False)
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), jit)
    actual = _function(evaluator, jacobian)(arguments)
    np.testing.assert_array_equal(actual, _expected(arguments, jacobian))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("jacobian", [False, True])
@pytest.mark.parametrize("column", [0, 1, 2, 3])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_entire_input_scan_precedes_first_row_output_domain_error(jit, jacobian, column, value):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), jit)
    arguments = np.array([[1.0, -1.0, 0.0, 1.0], [1.0, 4.0, 0.0, 1.0]])
    arguments[1, column] = value
    with pytest.raises(
        InvalidParameterError, match="^state, time and control parameters must be finite$"
    ):
        _function(evaluator, jacobian)(arguments)


@pytest.mark.parametrize("jit", [False, True])
def test_fused_jacobian_checks_primal_even_when_derivative_is_finite(jit):
    # log(x), d/dx=1/x: the implemented primal is undefined at negative x.
    tape = Tape(
        ops=np.array([OP_STATE, 34, OP_CONST, OP_DIV]),
        a=np.array([0, 0, 0, 2]),
        b=np.array([0, 0, 0, 0]),
        imm=np.array([0.0, 0.0, 1.0, 0.0]),
        outputs=np.array([1]),
        jac_outputs=np.array([3]),
        n_state=1,
        n_param=0,
    )
    evaluator = native.PreparedEvaluator(*tape.to_arrays(), jit)
    raw = evaluator.eval_jac(np.array([-1.0, 0.0]))
    assert np.isnan(raw[0]) and raw[1] == -1
    with pytest.raises(InvalidParameterError, match="^Jacobian is undefined"):
        evaluator.eval_jac_checked(np.array([-1.0, 0.0]))
    np.testing.assert_array_equal(evaluator.eval_jac_checked(np.array([1.0, 0.0])), [0, 1])


@pytest.mark.parametrize("jit", [False, True])
def test_rhs_can_be_finite_when_jacobian_is_unresolved_and_raw_rows_survive(jit):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), jit)
    at_zero = np.array([1.0, -0.0, 0.0, 1.0])
    assert evaluator.eval_rhs_checked(at_zero)[1].view(np.uint64) == np.float64(-0.0).view(
        np.uint64
    )
    with pytest.raises(InvalidParameterError, match="^Jacobian is undefined"):
        evaluator.eval_jac_checked(at_zero)
    bad = np.array([2.0, -1.0, 0.5, 3.0])
    raw = evaluator.eval_rhs(bad)
    assert raw[0] == 6.5 and np.isnan(raw[1])
    with pytest.raises(InvalidParameterError, match="^right-hand side is undefined"):
        evaluator.eval_rhs_checked(bad)
    again = evaluator.eval_rhs(bad)
    assert again[0] == raw[0] and np.isnan(again[1])


@pytest.mark.parametrize("jit", [False, True])
def test_unused_nonfinite_registers_and_unselected_nan_do_not_fail_checked_outputs(jit):
    tape = Tape(
        ops=np.array([OP_STATE, 35, OP_CONST, OP_CONST, OP_SELECT]),
        a=np.array([0, 0, 0, 0, 2]),
        b=np.array([0, 0, 0, 0, 3]),
        imm=np.array([0.0, 0.0, 1.0, -0.0, 1.0]),
        outputs=np.array([4]),
        jac_outputs=np.array([], dtype=int),
        n_state=2,
        n_param=1,
    )
    evaluator = native.PreparedEvaluator(*tape.to_arrays(), jit)
    arguments = np.array([-1.0, 2.0, 3.0, 4.0])
    assert np.signbit(evaluator.eval_rhs_checked(arguments)[0])
    for column in (1, 2, 3):
        changed = arguments.copy()
        changed[column] = np.nan
        assert np.signbit(evaluator.eval_rhs(changed)[0])
        with pytest.raises(InvalidParameterError, match="must be finite"):
            evaluator.eval_rhs_checked(changed)


@pytest.mark.parametrize("jit", [False, True])
def test_checked_nonsquare_rhs_keeps_declared_state_clock_and_control_offsets(jit):
    tape = replace(_tape(False), outputs=np.array([5]), n_state=3)
    evaluator = native.PreparedEvaluator(*tape.to_arrays(), jit)
    args = np.array([0.5, 4.0, 123.0, 0.25, 2.0])
    np.testing.assert_array_equal(evaluator.eval_rhs_checked(args), [1.25])
    with pytest.raises(ValueError, match="without a Jacobian"):
        evaluator.eval_jac_checked(np.empty((0, 5)))


@pytest.mark.parametrize("jit", [False, True])
def test_checked_large_and_small_finite_values_keep_bits(jit):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), jit)
    arguments = np.array([[1e200, 4.0, 0.0, 0.5], [1e-200, 9.0, -0.0, 0.5], [-0.0, 4.0, -0.0, 1.0]])
    for jacobian in (False, True):
        actual = _function(evaluator, jacobian)(arguments)
        raw = _function(evaluator, jacobian, False)(arguments)
        np.testing.assert_array_equal(actual.view(np.uint64), raw.view(np.uint64))


@pytest.mark.parametrize("jacobian", [False, True])
@pytest.mark.parametrize("arguments", [np.array(np.nan), np.full(3, np.nan), np.empty((0, 5))])
def test_checked_geometry_errors_precede_finiteness(jacobian, arguments):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), False)
    with pytest.raises(ValueError, match="width"):
        _function(evaluator, jacobian)(arguments)


@pytest.mark.parametrize("dtype", [np.float32, np.complex128, object])
def test_checked_binding_keeps_strict_float64_boundary(dtype):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), False)
    with pytest.raises(TypeError):
        evaluator.eval_rhs_checked(np.array([0.5, 4.0, 0.25, 2.0], dtype=dtype))


@pytest.mark.parametrize("jit", [False, True])
def test_checked_handle_reuse_and_concurrency_after_refusals(jit):
    evaluator = native.PreparedEvaluator(*_tape().to_arrays(), jit)
    native.clear_jit_cache()

    def evaluate(index):
        with pytest.raises(InvalidParameterError):
            evaluator.eval_rhs_checked(np.array([1.0, -1.0, 0.0, 1.0]))
        args = np.broadcast_to([index + 0.5, 4.0, 0.25, 2.0], (128, 4))
        return args, evaluator.eval_jac_checked(args)

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(evaluate, range(8)))
    for arguments, actual in outputs:
        np.testing.assert_array_equal(actual, _expected(arguments, True))
    for index in range(1, len(outputs)):
        assert not np.shares_memory(outputs[0][1], outputs[index][1])
