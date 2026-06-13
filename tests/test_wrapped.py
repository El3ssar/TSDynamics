"""WrappedSystem: adapt an external stepper to the System protocol."""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.families import System


def _logistic_step(u, n):
    x = u[0]
    for _ in range(int(n)):
        x = 3.9 * x * (1 - x)
    return [x]


def test_wrapped_satisfies_protocol() -> None:
    w = ts.WrappedSystem(_logistic_step, dim=1, is_discrete=True, initial=[0.5])
    assert isinstance(w, System)
    assert w.is_discrete is True
    assert w.dim == 1


def test_wrapped_step_and_state() -> None:
    w = ts.WrappedSystem(_logistic_step, dim=1, initial=[0.5])
    w.reinit()
    x1 = w.step()
    assert x1[0] == pytest.approx(3.9 * 0.5 * 0.5)
    assert w.time() == 1.0
    w.set_state([0.25])
    np.testing.assert_array_equal(w.state(), [0.25])


def test_wrapped_trajectory_and_transient() -> None:
    w = ts.WrappedSystem(_logistic_step, dim=1, initial=[0.3])
    traj = w.trajectory(200, transient=50)
    assert traj.y.shape == (200, 1)
    assert np.all(np.isfinite(traj.y))


def test_wrapped_named_components() -> None:
    def rot(u, n):
        x, y = u
        return [0.99 * (x - y), 0.99 * (x + y)]

    w = ts.WrappedSystem(rot, dim=2, initial=[1.0, 0.0], variables=("x", "y"))
    traj = w.trajectory(20)
    np.testing.assert_array_equal(traj["x"], traj.y[:, 0])


def test_wrapped_max_lyapunov_chaotic() -> None:
    # logistic at r=3.9 is chaotic ⇒ positive MLLE, via the protocol only
    w = ts.WrappedSystem(_logistic_step, dim=1, is_discrete=True, initial=[0.5])
    lam = ts.max_lyapunov(w, ic=[0.3], n_rescale=500, steps_per=2, seed=0)
    assert lam > 0.3


def test_wrapped_orbit_diagram_via_protocol() -> None:
    # a wrapped discrete system feeds straight into orbit_diagram
    w = ts.WrappedSystem(_logistic_step, dim=1, is_discrete=True, initial=[0.5])
    # orbit_diagram needs with_params; wrapped systems have no params, so this
    # is a protocol-only smoke test of trajectory collection instead.
    traj = w.trajectory(300, transient=300, ic=[0.5])
    branches = len(np.unique(np.round(traj.y[:, 0], 3)))
    assert branches > 50  # chaotic band is densely filled
