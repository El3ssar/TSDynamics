"""The wheel-free oracle rejects invalid settings just like the native engine."""

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import ConvergenceError, InvalidParameterError


@pytest.mark.parametrize("events", [None, [("z", 20.0)]])
@pytest.mark.parametrize(
    "tolerances",
    [
        {"rtol": -1.0},
        {"atol": -1.0},
        {"rtol": np.nan},
        {"atol": np.nan},
        {"rtol": np.inf},
        {"atol": np.inf},
        {"rtol": 0.0, "atol": 0.0},
    ],
)
def test_reference_rejects_invalid_tolerances(tolerances, events):
    with pytest.raises(InvalidParameterError, match="rtol|atol"):
        ts.systems.Lorenz().run(
            final_time=0.1,
            dt=0.01,
            ic=[1.0, 1.0, 1.0],
            backend="reference",
            events=events,
            **tolerances,
        )


@pytest.mark.parametrize("tolerances", [{"rtol": 0.0}, {"atol": 0.0}])
def test_reference_preserves_one_sided_tolerance_control(tolerances):
    traj = ts.systems.Lorenz().run(
        final_time=0.1,
        dt=0.01,
        ic=[1.0, 1.0, 1.0],
        backend="reference",
        **tolerances,
    )
    assert np.isfinite(traj.y).all()


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_delay_step_rejects_invalid_increment_without_moving_clock(dt):
    sys = ts.systems.MackeyGlass()
    sys.reinit([1.1])
    with pytest.raises(InvalidParameterError, match="dt"):
        sys.step(dt)
    assert sys.time() == 0.0
    np.testing.assert_array_equal(sys.state(), [1.1])


def test_failed_delay_step_keeps_previous_clock_and_state(monkeypatch):
    sys = ts.systems.MackeyGlass()
    sys.reinit([1.1])
    sys.step(0.1)
    prior_t, prior_u = sys.time(), sys.state()

    def fail(self, **kwargs):
        raise ConvergenceError("simulated integration failure")

    monkeypatch.setattr(type(sys), "run", fail)
    with pytest.raises(ConvergenceError):
        sys.step(0.2)
    assert sys.time() == prior_t
    np.testing.assert_array_equal(sys.state(), prior_u)
