"""The native componentwise correction must reject unproved cross-noise terms."""

import numpy as np
import pytest

from tsdynamics._engine.compile import OP_CONST, OP_PARAM, OP_STATE, Tape

engine = pytest.importorskip("tsdynamics._rust")


def tapes(driver=1.0):
    parameterized = driver is None
    fields = dict(
        ops=np.array([OP_CONST, OP_PARAM if parameterized else OP_CONST, OP_STATE]),
        a=np.zeros(3, dtype=np.int32),
        b=np.zeros(3, dtype=np.int32),
        imm=np.array([0.0, 0.0 if parameterized else driver, 0.0]),
        n_state=2,
        n_param=int(parameterized),
    )
    drift = Tape(**fields, outputs=np.array([0, 0]))
    # g=(driver,x0), with an intentionally false all-zero Jacobian. A claimed
    # derivative cannot erase the dependency of the primal second coefficient.
    diffusion = Tape(**fields, outputs=np.array([1, 2]), jac_outputs=np.zeros(4, dtype=np.int32))
    return drift, diffusion, np.zeros(int(parameterized))


def run_dense(driver, method, jit):
    drift, diffusion, controls = tapes(driver)
    return engine.integrate_sde_dense(
        *drift.to_arrays(),
        *diffusion.to_arrays(),
        np.array([0.25, 0.0]),
        controls,
        np.array([0.0, 0.1, 0.2]),
        method,
        0.1,
        42,
        jit,
    )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("driver", [1.0, None])
def test_manual_dense_tapes_cannot_certify_coupling_with_zero_jacobian(jit, driver):
    with pytest.raises(NotImplementedError, match="cross-noise.*euler_maruyama"):
        run_dense(driver, "milstein", jit)


@pytest.mark.parametrize("jit", [False, True])
def test_manual_ensemble_tapes_use_the_same_structural_guard(jit):
    drift, diffusion, controls = tapes()
    with pytest.raises(NotImplementedError, match="cross-noise.*euler_maruyama"):
        engine.integrate_sde_ensemble_final(
            *drift.to_arrays(),
            *diffusion.to_arrays(),
            np.array([[0.0, 0.0], [0.25, 0.0]]),
            controls,
            0.0,
            0.2,
            "milstein",
            0.1,
            42,
            jit,
        )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("zero", [0.0, -0.0])
def test_literal_zero_driver_admits_dependence_on_the_deterministic_coordinate(jit, zero):
    result = run_dense(zero, "milstein", jit)
    expected = run_dense(zero, "euler_maruyama", jit)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.asarray(result)[:, 0], [0.25, 0.25, 0.25])


@pytest.mark.parametrize("jit", [False, True])
def test_euler_maruyama_remains_available_for_cross_state_diffusion(jit):
    result = np.asarray(run_dense(1.0, "euler_maruyama", jit))
    assert result.shape == (3, 2)
    assert np.isfinite(result).all()
