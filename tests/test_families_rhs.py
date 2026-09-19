"""``system.rhs(u, t)`` — the vector field ``jacobian`` is the derivative of.

The gap, in the blind tester's words:

    *"if ``system.jacobian(u, t)`` is public, ``system.rhs(u, t)`` is too.
    ``dir(vdp)`` has ``jacobian`` and ``jacobian_sym`` but no RHS; the only route
    is ``_rhs_numeric``. This is exactly the gap that made task 5 hard:
    ``flow_speed``, ``streamlines`` and ``vector_field`` ARE the RHS, and a user
    has no sanctioned way to check them."*

Two other readers independently reached for the same private name while writing
transforms.  The evidence a name needs before it earns a slot on
``system.<TAB>`` is that users type it; three did, and the only spelling
available was private.

What is asserted here:

1. the door exists, is listed, and returns the field — verified against a
   **finite difference of the public** ``jacobian``, so the two public answers
   are proved consistent with each other rather than with an internal;
2. it is the same numbers the engine integrates (a one-step Euler comparison);
3. it follows the system's *current* parameters, so ``with_params`` moves it;
4. the three families that have no ``f(u, t)`` say so by name, with a runnable
   line — a delay system (the field needs the whole history), a map (no vector
   field at all) and a ``WrappedSystem`` (an opaque stepper).
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts

_FLOWS = ("Lorenz", "Rossler", "VanDerPol", "Thomas", "Chua")


class TestTheRightHandSideHasAPublicDoor:
    def test_it_is_on_the_tab_surface_of_a_flow(self):
        lor = ts.systems.Lorenz()
        assert "rhs" in dir(lor)
        assert callable(lor.rhs)

    @pytest.mark.parametrize("name", _FLOWS)
    def test_the_jacobian_is_the_derivative_of_what_rhs_returns(self, name):
        """The two public doors, checked against each other and nothing else."""
        system = getattr(ts.systems, name)()
        dim = int(system.dim)
        rng = np.random.default_rng(0)
        u = rng.normal(scale=0.7, size=dim)
        jac = np.asarray(system.jacobian(u, 0.0), dtype=float)
        h = 1e-6
        for col in range(dim):
            step = np.zeros(dim)
            step[col] = h
            numeric = (system.rhs(u + step) - system.rhs(u - step)) / (2 * h)
            assert numeric == pytest.approx(jac[:, col], abs=2e-5), (name, col)

    def test_the_shape_is_the_state_shape(self):
        lor = ts.systems.Lorenz()
        assert np.asarray(lor.rhs([1.0, 2.0, 3.0])).shape == (3,)

    def test_it_is_the_field_the_engine_integrates(self):
        """One tiny step of the real integrator must agree with ``ic + dt * rhs``."""
        lor = ts.systems.Lorenz()
        ic = np.array([1.0, 2.0, 3.0])
        dt = 1e-6
        traj = lor.run(final_time=dt, dt=dt, ic=ic)
        moved = (np.asarray(traj.y[-1], dtype=float) - ic) / dt
        assert moved == pytest.approx(lor.rhs(ic), rel=1e-4, abs=1e-4)

    def test_it_follows_the_systems_current_parameters(self):
        lor = ts.systems.Lorenz()
        u = np.array([1.0, 2.0, 3.0])
        before = lor.rhs(u).copy()
        after = lor.with_params(sigma=42.0).rhs(u)
        assert not np.allclose(before, after)
        # dx/dt = sigma (y - x) — the one component sigma touches.
        assert after[0] == pytest.approx(42.0 * (u[1] - u[0]))

    def test_time_reaches_a_non_autonomous_field(self):
        """Every shipped "forced" flow is autonomised with a phase coordinate, so
        the ``t`` argument is exercised on a system written for this test."""
        import symengine

        class _Driven(ts.ContinuousSystem):
            params = {"a": 1.0}  # noqa: RUF012

            @staticmethod
            def _equations(u, t, a):
                return [symengine.sin(a * t) - u(0)]

        driven = _Driven(dim=1)
        u = np.array([0.3])
        assert driven.rhs(u, 0.0) == pytest.approx([-0.3])
        assert driven.rhs(u, np.pi / 2) == pytest.approx([0.7])

    def test_an_sde_answers_with_its_drift_named_as_such(self):
        """The twin of ``jacobian``, which is already the DRIFT Jacobian."""
        ou = ts.systems.OrnsteinUhlenbeck()
        theta = float(ou.params["theta"])
        mu = float(ou.params["mu"])
        assert ou.rhs([2.0]) == pytest.approx([theta * (mu - 2.0)])
        assert "drift" in (ts.systems.OrnsteinUhlenbeck.rhs.__doc__ or "").lower()


class TestAFamilyWithNoVectorFieldSaysSoByName:
    @pytest.mark.parametrize(
        ("factory", "because", "remedy"),
        [
            (ts.systems.MackeyGlass, "past times", "system.run("),
            (ts.systems.Henon, "no vector field", "system.step(1)"),
        ],
    )
    def test_the_absence_states_the_reason_and_hands_back_a_line(self, factory, because, remedy):
        system = factory()
        assert "rhs" not in dir(system)
        assert not hasattr(system, "rhs")
        with pytest.raises(AttributeError) as excinfo:
            system.rhs  # noqa: B018
        message = str(excinfo.value)
        assert "rhs" in message
        assert because in message
        assert remedy in message

    def test_a_wrapped_stepper_has_none_either(self):
        wrapped = ts.WrappedSystem(lambda u, dt: np.asarray(u), dim=2, family="map")
        assert "rhs" not in dir(wrapped)
        with pytest.raises(AttributeError, match="opaque stepper"):
            wrapped.rhs  # noqa: B018
