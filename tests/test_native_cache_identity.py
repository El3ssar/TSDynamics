"""Native code identity and truthful numerical-screen diagnostics."""

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics._engine import run
from tsdynamics._engine.compile import OP_ADD, OP_CONST, OP_STATE, Tape
from tsdynamics._engine.problem import MapProblem


def _offset_tape(first, second):
    # The exact operation sequence is intentional: +0 and -0 are different
    # instructions even though ordinary floating-point equality equates them.
    return Tape(
        ops=np.array([OP_STATE, OP_CONST, OP_ADD, OP_STATE, OP_CONST, OP_ADD]),
        a=np.array([0, 0, 0, 1, 0, 3]),
        b=np.array([0, 0, 1, 0, 0, 4]),
        imm=np.array([0.0, first, 0.0, 0.0, second, 0.0]),
        outputs=np.array([2, 5]),
        n_state=2,
        n_param=0,
    )


def test_jit_hash_collision_preserves_distinct_signed_zero_code(monkeypatch):
    monkeypatch.delenv("TSDYNAMICS_NO_JIT_CACHE", raising=False)
    run.clear_jit_cache()
    try:
        for first, second in [(0.0, -0.0), (-0.0, 0.0), (0.0, -0.0)]:
            problem = MapProblem(_offset_tape(first, second), np.array([-0.0, -0.0]))
            expected = np.array([-0.0 + first, -0.0 + second])
            for backend in ["reference", "interp", "jit"]:
                actual = run.integrate(problem, final_time=1, backend=backend).y[-1]
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        stats = run.jit_cache_stats()
        assert (stats["misses"], stats["hits"], stats["size"]) == (2, 1, 2)
    finally:
        run.clear_jit_cache()


@pytest.mark.parametrize("backend", ["jit", "interp"])
@pytest.mark.parametrize("solver", ["rk4", "dop853"])
@pytest.mark.parametrize("sample_dt", [0.125, 0.03125])
def test_bounded_large_equilibrium_reports_a_numerical_screen(backend, solver, sample_dt):
    class Constant(ts.ContinuousSystem):
        variables = ("x",)
        params = {}

        @staticmethod
        def _equations(y, t):
            return [0.0]

    # This patch deliberately preserves the legacy limit. A constant finite
    # solution must not be described as a proved physical divergence.
    with pytest.raises(ts.errors.ConvergenceError) as caught:
        Constant().run(
            final_time=0.125, dt=sample_dt, ic=[1e200], backend=backend, solver=solver
        )
    message = str(caught.value)
    assert "numerical integration failed" in message
    assert "magnitude-screen limit" in message
    assert "does not establish divergence" in message
    assert "Rescale" in message
    assert "integration diverged" not in message
    assert "RHS is diverging" not in message
