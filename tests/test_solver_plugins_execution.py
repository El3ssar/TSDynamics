"""A solver plugin's declared kernel must execute through every solver door."""

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import _solvers as solvers
from tsdynamics.derived import TangentSystem
from tsdynamics.errors import InvalidParameterError


@pytest.fixture
def kernel_alias():
    registered = []

    def alias(kernel):
        name = f"plugin_{kernel}"
        solvers.register(
            solvers.SolverSpec(
                name=name, kernel=kernel, caps=solvers.get(kernel).caps, origin="plugin"
            )
        )
        registered.append(name)
        return name

    yield alias
    for name in registered:
        solvers.unregister(name)


@pytest.mark.parametrize("backend", ["interp", "jit", "reference"])
@pytest.mark.parametrize("kernel", ["rk4", "bdf"])
def test_ode_plugin_kernel_matches_direct_run(kernel_alias, backend, kernel):
    alias = kernel_alias(kernel)
    system = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
    expected = system.run(0.03, dt=0.01, solver=kernel, backend=backend)
    actual = system.run(0.03, dt=0.01, solver=alias, backend=backend)
    np.testing.assert_array_equal(actual.y, expected.y)
    assert actual.meta["solver"] == kernel


@pytest.mark.parametrize("backend", ["interp", "jit"])
def test_ode_plugin_kernel_drives_live_steps_and_ensembles(kernel_alias, backend):
    alias = kernel_alias("rk4")
    ic = [1.0, 1.0, 1.0]
    system = ts.systems.Lorenz(ic=ic)
    expected = ts.systems.Lorenz(ic=ic)
    system.reinit(ic, solver=alias, backend=backend)
    expected.reinit(ic, solver="rk4", backend=backend)
    np.testing.assert_array_equal(system.step(0.01), expected.step(0.01))
    states = [[1.0, 1.0, 1.0], [2.0, 1.0, 3.0]]
    actual_batch = system.ensemble(states).run(0.03, dt=0.01, solver=alias, backend=backend)
    expected_batch = expected.ensemble(states).run(0.03, dt=0.01, solver="rk4", backend=backend)
    np.testing.assert_array_equal(actual_batch.y, expected_batch.y)


@pytest.mark.parametrize("backend", ["interp", "jit"])
def test_dde_plugin_kernel_matches_direct_run(kernel_alias, backend):
    alias = kernel_alias("rk4")
    system = ts.systems.MackeyGlass(ic=[1.2])
    expected = system.run(0.03, dt=0.01, solver="rk4", backend=backend)
    actual = system.run(0.03, dt=0.01, solver=alias, backend=backend)
    np.testing.assert_array_equal(actual.y, expected.y)


@pytest.mark.parametrize("backend", ["interp", "jit", "reference"])
@pytest.mark.parametrize("kernel", ["euler_maruyama", "milstein"])
def test_sde_plugin_kernel_matches_seeded_run_and_step(kernel_alias, backend, kernel):
    alias = kernel_alias(kernel)
    system = ts.systems.OrnsteinUhlenbeck(ic=[0.25])
    expected = system.run(0.03, dt=0.01, solver=kernel, backend=backend, seed=13)
    actual = system.run(0.03, dt=0.01, solver=alias, backend=backend, seed=13)
    np.testing.assert_array_equal(actual.y, expected.y)
    system.reinit([0.25], solver=alias, seed=13)
    other = ts.systems.OrnsteinUhlenbeck(ic=[0.25])
    other.reinit([0.25], solver=kernel, seed=13)
    np.testing.assert_array_equal(system.step(0.01), other.step(0.01))


@pytest.mark.parametrize("backend", ["interp", "jit"])
def test_tangent_stepper_uses_plugin_kernel(kernel_alias, backend):
    alias = kernel_alias("rk4")
    actual = TangentSystem(ts.systems.Lorenz(), k=1, backend=backend)
    expected = TangentSystem(ts.systems.Lorenz(), k=1, backend=backend)
    actual.reinit([1.0, 1.0, 1.0], method=alias)
    expected.reinit([1.0, 1.0, 1.0], method="rk4")
    actual.step(0.01)
    expected.step(0.01)
    np.testing.assert_array_equal(actual.state(), expected.state())
    assert actual._ode_stepper is not None


def test_sde_default_solver_can_name_a_registered_kernel_alias(kernel_alias):
    alias = kernel_alias("milstein")

    class PluginDefault(ts.systems.OrnsteinUhlenbeck):
        _default_method = alias

    system = PluginDefault(ic=[0.25])
    expected = system.run(0.03, dt=0.01, solver="milstein", seed=13)
    actual = system.run(0.03, dt=0.01, seed=13)
    np.testing.assert_array_equal(actual.y, expected.y)


def test_sde_metadata_does_not_install_an_unimplemented_scheme():
    name = "plugin_unimplemented_sde"
    solvers.register(
        solvers.SolverSpec(
            name=name,
            kernel="unimplemented_sde",
            caps=solvers.get("euler_maruyama").caps,
            origin="plugin",
        )
    )
    try:
        with pytest.raises(InvalidParameterError, match="does not install"):
            ts.systems.OrnsteinUhlenbeck().run(0.03, dt=0.01, solver=name)
    finally:
        solvers.unregister(name)
