"""An explicit native allowance counts trials across the complete requested run."""

import numpy as np
import pytest

from tsdynamics.errors import InvalidParameterError, StepBudgetError

native = pytest.importorskip("tsdynamics._rust")


def _arguments(jit=False):
    # Base x'=0, tangent w'=0. Each RK4 quarter-window costs exactly one trial.
    return (
        np.array([0], np.int32),
        np.array([0], np.int32),
        np.array([0], np.int32),
        np.array([0.0]),
        np.array([0, 0], np.int32),
        np.empty(0, np.int32),
        2,
        0,
        np.empty(0),
        "rk4",
        1e-9,
        1e-12,
        1,
        1,
        np.array([5.0, 1.0]),
        0.0,
        0.25,
        0.5,
        0.5,
        jit,
    )


@pytest.mark.parametrize("jit", [False, True])
def test_final_trial_at_the_allowance_completes_and_legacy_abi_is_unchanged(jit):
    args = _arguments(jit)
    before = args[14].copy()
    legacy = native.lyapunov_spectrum_ode(*args)
    assert len(legacy) == 3
    result = native.lyapunov_spectrum_ode_budgeted(*args, max_steps=4)
    assert len(result) == 4
    for old, new in zip(legacy, result[:3], strict=True):
        np.testing.assert_array_equal(old.view(np.uint64), new.view(np.uint64))
    work = result[3]
    assert work["complete"] is True
    assert work["scope"] == "whole_analysis" and work["unit"] == "solver_trial_steps"
    assert work["attempted_steps"] == work["accepted_steps"] == 4
    assert work["rejected_steps"] == work["failed_steps"] == 0
    assert work["completed_burn_chunks"] == work["completed_average_chunks"] == 2
    assert work["reached_time"] == work["last_qr_time"] == work["requested_end"] == 1.0
    assert work["solver"] == "rk4"
    assert work["backend"] == ("jit" if jit else "interp")
    np.testing.assert_array_equal(args[14], before)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    ("limit", "phase", "clock", "burn_chunks", "average_chunks"),
    [(1, "burn_in", 0.25, 1, 0), (2, "averaging", 0.5, 2, 0), (3, "averaging", 0.75, 2, 1)],
)
def test_exhaustion_retains_phase_clock_settings_and_no_completed_spectrum(
    jit, limit, phase, clock, burn_chunks, average_chunks
):
    args = _arguments(jit)
    before = args[14].copy()
    with pytest.raises(StepBudgetError, match="calculation is incomplete") as caught:
        native.lyapunov_spectrum_ode_budgeted(*args, max_steps=limit)
    work = caught.value.work
    assert work["complete"] is False
    assert work["phase"] == phase
    assert work["reached_time"] == work["last_qr_time"] == clock
    assert work["requested_end"] == 1.0
    assert work["attempted_steps"] == work["accepted_steps"] == limit
    assert work["rejected_steps"] == work["failed_steps"] == 0
    assert work["completed_burn_chunks"] == burn_chunks
    assert work["completed_average_chunks"] == average_chunks
    assert (work["dt"], work["rtol"], work["atol"]) == (0.25, 1e-9, 1e-12)
    assert "spectrum" not in work and "diverg" not in str(caught.value)
    np.testing.assert_array_equal(args[14], before)


@pytest.mark.parametrize(
    "limit", [0, -1, True, False, np.bool_(True), np.bool_(False), 1.5, np.nan, 1 + 0j]
)
def test_native_allowance_refuses_invalid_counts_before_running(limit):
    with pytest.raises(InvalidParameterError, match="max_steps.*positive integer"):
        native.lyapunov_spectrum_ode_budgeted(*_arguments(), max_steps=limit)


def test_index_callback_interrupt_is_not_rewritten_as_an_invalid_count():
    class Interrupted:
        def __index__(self):
            raise KeyboardInterrupt("stopped while reading the requested allowance")

    with pytest.raises(KeyboardInterrupt, match="stopped while reading"):
        native.lyapunov_spectrum_ode_budgeted(*_arguments(), max_steps=Interrupted())


@pytest.mark.parametrize("budgeted", [False, True])
@pytest.mark.parametrize("overflow", ["product", "sum"])
def test_huge_declared_dimension_cannot_wrap_into_a_tiny_valid_tape(budgeted, overflow):
    args = list(_arguments())
    largest = int(np.iinfo(np.uintp).max)
    args[12] = largest // 2 + 2 if overflow == "product" else largest
    args[13] = 1 if overflow == "product" else largest
    function = native.lyapunov_spectrum_ode_budgeted if budgeted else native.lyapunov_spectrum_ode
    options = {"max_steps": 4} if budgeted else {}
    with pytest.raises(ValueError, match="extended dimension.*overflows"):
        function(*args, **options)
