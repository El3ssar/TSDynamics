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
    lam = ts.max_lyapunov(w, ic=[0.3], n=500, steps_per=2, seed=0)
    assert lam > 0.3


def _expanding_flow(u, dt):
    # exact flow map of dx/dt = 0.5 x over a time increment dt
    return [u[0] * np.exp(0.5 * dt)]


def test_wrapped_continuous_max_lyapunov_dt_normalization() -> None:
    # Regression: a *continuous* WrappedSystem stepped with dt=None must
    # normalize by the real per-step time advance (its ``default_dt``), not a
    # hardcoded 0.01.  dx/dt = 0.5 x has maximal exponent 0.5; the dt=None call
    # must (a) recover 0.5 and (b) equal the explicit-dt call.  The previous
    # code read the missing ``_default_step_dt`` attribute, fell back to 0.01,
    # and returned the exponent scaled by default_dt/0.01 = 5x too large.
    # d0 is large (1e-4) on purpose: the flow is exactly linear, so the growth
    # rate is independent of the perturbation size, and a small d0 against a
    # growing reference would lose it to floating-point cancellation.
    kw = {"n": 100, "steps_per": 2, "transient": 10, "d0": 1e-4, "seed": 0}
    w = ts.WrappedSystem(_expanding_flow, dim=1, is_discrete=False, initial=[1.0], default_dt=0.05)
    lam_default = ts.max_lyapunov(w, ic=[1.0], **kw)  # dt=None → steps by 0.05
    lam_explicit = ts.max_lyapunov(w, ic=[1.0], dt=0.05, **kw)
    assert lam_default == pytest.approx(0.5, abs=1e-6)
    assert lam_default == pytest.approx(lam_explicit, rel=1e-9)


def test_wrapped_trajectory_final_time_dt_continuous() -> None:
    # Regression: a CONTINUOUS wrapper must accept the family-uniform
    # trajectory(final_time=, dt=) spelling (a generic protocol caller uses it).
    # Pre-fix, trajectory(n, *, transient, ic) had no final_time/dt and raised
    # TypeError under such callers.  final_time=1.0, dt=0.1 -> 10 samples, each
    # advancing the exact flow by 0.1.
    w = ts.WrappedSystem(_expanding_flow, dim=1, is_discrete=False, default_dt=0.1)
    traj = w.trajectory(final_time=1.0, dt=0.1, ic=[1.0])
    assert traj.y.shape == (10, 1)
    # exact flow map of dx/dt = 0.5 x over 10 steps of 0.1 each → exp(0.5 * 1.0)
    np.testing.assert_allclose(traj.y[-1, 0], np.exp(0.5 * 1.0), rtol=1e-9)
    np.testing.assert_allclose(traj.t, np.arange(1, 11) * 0.1, rtol=1e-12)


def test_wrapped_trajectory_final_time_uses_default_dt() -> None:
    # final_time with no dt derives the step from default_dt.
    w = ts.WrappedSystem(_expanding_flow, dim=1, is_discrete=False, default_dt=0.25)
    traj = w.trajectory(final_time=1.0, ic=[1.0])  # 1.0 / 0.25 = 4 samples
    assert traj.y.shape == (4, 1)
    np.testing.assert_allclose(traj.y[-1, 0], np.exp(0.5 * 1.0), rtol=1e-9)


def test_wrapped_trajectory_positional_n_still_works() -> None:
    # Backward compatibility: the n-positional form is unchanged.
    w = ts.WrappedSystem(_logistic_step, dim=1, initial=[0.3])
    traj = w.trajectory(200, transient=50)
    assert traj.y.shape == (200, 1)
    assert np.all(np.isfinite(traj.y))


def test_wrapped_trajectory_count_source_validation() -> None:
    from tsdynamics.errors import InvalidInputError

    w = ts.WrappedSystem(_logistic_step, dim=1, initial=[0.5])
    with pytest.raises(InvalidInputError):
        w.trajectory()  # neither n nor final_time
    with pytest.raises(InvalidInputError):
        w.trajectory(10, final_time=5.0)  # both


def test_wrapped_orbit_diagram_via_protocol() -> None:
    # a wrapped discrete system feeds straight into orbit_diagram
    w = ts.WrappedSystem(_logistic_step, dim=1, is_discrete=True, initial=[0.5])
    # orbit_diagram needs with_params; wrapped systems have no params, so this
    # is a protocol-only smoke test of trajectory collection instead.
    traj = w.trajectory(300, transient=300, ic=[0.5])
    branches = len(np.unique(np.round(traj.y[:, 0], 3)))
    assert branches > 50  # chaotic band is densely filled
