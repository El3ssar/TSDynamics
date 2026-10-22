"""A typo in a restart's parameter mapping must not partially change the model."""

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.derived import TangentSystem


@pytest.mark.parametrize(
    "factory,parameter,value,initial",
    [
        (ts.systems.Lorenz, "sigma", 12.0, [1.0, 1.0, 1.0]),
        (ts.systems.Henon, "a", 1.2, [0.0, 0.0]),
        (ts.systems.MackeyGlass, "beta", 0.3, [1.2]),
        (ts.systems.OrnsteinUhlenbeck, "theta", 2.0, [0.0]),
        (lambda **kw: TangentSystem(ts.systems.Lorenz(**kw)), "sigma", 12.0, [1.0, 1.0, 1.0]),
    ],
)
def test_restart_validates_all_parameter_names_before_changing_any(
    factory, parameter, value, initial
):
    system = factory(ic=initial)
    system.reinit(initial)
    before = dict(system.params)
    state = system.state()
    with pytest.raises(KeyError, match="misspelled_parameter"):
        system.reinit(params={parameter: value, "misspelled_parameter": 1.0})
    assert system.params == before
    np.testing.assert_array_equal(system.state(), state)
