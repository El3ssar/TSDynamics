"""A run that cannot finish is refused *quickly* — the beta round's last engine gap.

Two blind testers independently hit the same wall: a diverging initial condition
was refused with a correct, actionable message, **24 seconds later**.  One of
them, porting from DynamicalSystems.jl, called it "the difference between a
usable sweep and a broken one" — in a parameter sweep the latency is multiplied
by every bad value.

The cause was that the only guard against a march that goes nowhere was a step
*count* (``DEFAULT_MAX_STEPS``, 1e8 per output segment) with no relation to the
span the caller asked for, so exhausting it *was* the cost of the diagnosis.
:data:`tsdyn_engine::HOPELESS_STEPS` turns the span into a step floor and reads
the same condition off the step size in O(1) instead.

What these tests pin is the pair of properties that make that safe:

* the refusal is fast **and says the same thing it always said**, and
* nothing that used to succeed now fails, and nothing that used to be reported
  as a *divergence* is now reported as a stall.

The second is the one with teeth.  A guard that fails fast by also failing
*wrongly* would be a far worse bug than the latency it fixes.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import symengine as se

import tsdynamics as ts

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

# The reproduction, verbatim.  `LorenzBounded` carries a `1 - |u|^2/r^2` factor,
# so a start well outside the bounding sphere runs away cubically: the state
# climbs to ~2e5 and the step collapses to ~1e-14, which is *below double
# precision resolution of t* — so `t` freezes while the state never gets near the
# 1e150 escape scale.  Measured before the fix: 33.0 s.  After: 0.41 s.
_BAD_START = [-44.529592, -22.733539, -49.582328]

# Generous enough to survive a loaded CI box (the fixed path measures ~0.4 s and
# the unfixed one 33 s, so anything in between separates them unambiguously).
_PATIENCE_SECONDS = 8.0


def _timed(fn):
    """Run ``fn``, returning ``(elapsed_seconds, raised_exception_or_None)``."""
    t0 = time.perf_counter()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the exception *is* the measurement
        return time.perf_counter() - t0, exc
    return time.perf_counter() - t0, None


class _Blowup(ts.ContinuousSystem):
    """``x' = x²`` — a finite-time blow-up whose state reaches the escape scale."""

    params = {"a": 1.0}
    variables = ("x",)

    @staticmethod
    def _equations(u, t, a):
        return [a * u(0) ** 2]


class _LogBlowup(ts.ContinuousSystem):
    """``x' = exp(x)`` — a blow-up whose state grows only logarithmically."""

    params = {"a": 1.0}
    variables = ("x",)

    @staticmethod
    def _equations(u, t, a):
        return [a * se.exp(u(0))]


class _VeryStiff(ts.ContinuousSystem):
    """``x' = -k x`` with ``k = 1e7`` — legitimately needs millions of steps."""

    params = {"k": 1.0e7}
    variables = ("x",)

    @staticmethod
    def _equations(u, t, k):
        return [-k * u(0)]


class TestABadStartIsRefusedQuickly:
    """The headline: the same refusal, in about a second instead of half a minute."""

    def test_a_diverging_start_is_refused_in_about_a_second(self):
        elapsed, exc = _timed(
            lambda: ts.systems.LorenzBounded().run(final_time=100.0, dt=0.01, ic=_BAD_START)
        )
        assert exc is not None, "a start that cannot be integrated must be refused"
        assert elapsed < _PATIENCE_SECONDS, (
            f"the refusal took {elapsed:.1f}s; it used to take 33s and the whole "
            "point of the span-relative step floor is that it no longer does"
        )

    def test_the_refusal_still_says_what_it_always_said(self):
        """Fail fast, not differently: same class, same advice, same words."""
        _, exc = _timed(
            lambda: ts.systems.LorenzBounded().run(final_time=100.0, dt=0.01, ic=_BAD_START)
        )
        assert isinstance(exc, ts.errors.StepBudgetError)
        # `StepBudgetError` is a `ConvergenceError` is a `RuntimeError`: every
        # handler that caught this before still catches it.
        assert isinstance(exc, ts.errors.ConvergenceError)
        assert isinstance(exc, RuntimeError)
        message = str(exc)
        assert "did not reach the final time" in message
        # The remedy the testers called correct, word for word.
        assert "looser rtol/atol" in message
        assert "method='bdf'" in message
        assert "shorter integration span" in message
        # ...and it now names the step size that made it impossible.
        assert "step size collapsed" in message

    def test_the_refusal_does_not_depend_on_the_output_resolution(self):
        """``dt`` is a sampling interval: it must not move the guard.

        The floor is relative to the *span*, so a caller who samples the same
        integration ten thousand times more finely gets the same verdict — and
        gets it just as fast.  A per-segment guard would have refused at one
        ``dt`` and ground at another.
        """
        verdicts = {}
        for dt in (0.01, 1.0):
            elapsed, exc = _timed(
                lambda dt=dt: ts.systems.LorenzBounded().run(final_time=100.0, dt=dt, ic=_BAD_START)
            )
            assert elapsed < _PATIENCE_SECONDS, f"dt={dt} took {elapsed:.1f}s"
            verdicts[dt] = type(exc).__name__
        assert verdicts[0.01] == verdicts[1.0] == "StepBudgetError", verdicts


class TestNothingThatWorkedStoppedWorking:
    """The guard may only ever refuse a run that was never going to return."""

    def test_a_run_needing_millions_of_steps_still_completes(self):
        """``k = 1e7`` on an explicit kernel is ~3.6e6 steps in ONE segment.

        This is the shape a naive "scale the cap to the span" fix breaks: the
        whole integration is a single output segment, so any per-segment budget
        derived from ``dt`` would have refused it.  The floor is a statement
        about the *step size*, not the step count, so it does not.
        """
        traj = _VeryStiff().run(final_time=1.0, dt=1.0, ic=[1.0], solver="rk45")
        # x(1) = e^{-1e7} underflows to 0; what matters is that it converged.
        assert np.isfinite(traj.y).all()
        assert abs(traj.y[-1, 0]) < 1e-9

    @pytest.mark.parametrize("solver", ["rk45", "dop853", "tsit5", "rk4", "bdf"])
    def test_an_ordinary_integration_is_untouched(self, solver):
        lor = ts.systems.Lorenz()
        traj = lor.run(final_time=5.0, dt=0.01, ic=[1.0, 1.0, 1.0], solver=solver)
        assert np.isfinite(traj.y).all()
        assert traj.y.shape[0] == 501

    def test_every_delay_system_still_integrates(self):
        """DDEs march on a delay-capped step — a forced-short step, not a stall."""
        from tsdynamics import registry

        entries = list(registry.all_systems(family="dde"))
        assert entries, "the DDE catalogue should not be empty"
        for entry in entries:
            system = entry.cls()
            ic = list(np.full(system.dim, 0.7))
            traj = system.run(final_time=20.0, dt=0.1, ic=ic)
            assert np.isfinite(traj.y).all(), entry.name

    def test_a_single_step_window_is_not_a_stall(self):
        """The two-node grid every in-engine driver marches on.

        The basin cell march, the Lyapunov chunk loop and the resumable stepper
        all call the engine with ``[t, t + dt]``.  Their span is one ``dt``, so
        their floor is ``dt / 1e12`` — the tightest floor in the library, and the
        one a false positive would show up in first.
        """
        lor = ts.systems.Lorenz()
        lor.reinit([1.0, 1.0, 1.0])
        for _ in range(50):
            state = lor.step(0.01)
        assert np.isfinite(state).all()


class TestABlowUpStillSaysItDiverged:
    """A collapsed step is the shared symptom of two different diagnoses.

    Both a stall and a finite-time blow-up end with the step size in the dust;
    the only difference is whether the state reaches the escape scale before
    ``t`` runs out of resolution.  The guard therefore grants a bounded grace
    (``STALL_GRACE_STEPS``) so the escape guard keeps first refusal — otherwise
    every overflow blow-up would be relabelled a solver-settings problem, which
    is the wrong diagnosis *and* the wrong remedy.
    """

    def test_a_polynomial_blow_up_is_reported_as_a_divergence(self):
        elapsed, exc = _timed(lambda: _Blowup().run(final_time=10.0, dt=0.01, ic=[1.0]))
        assert isinstance(exc, ts.errors.ConvergenceError)
        assert not isinstance(exc, ts.errors.StepBudgetError), (
            "x' = x² escapes to 1e150; calling that a stalled run would send the "
            "user to tune rtol when their trajectory left the building"
        )
        assert "diverged" in str(exc)
        assert elapsed < _PATIENCE_SECONDS

    def test_an_exponential_blow_up_is_reported_as_a_divergence(self):
        _, exc = _timed(lambda: _LogBlowup().run(final_time=5.0, dt=0.01, ic=[0.0]))
        assert isinstance(exc, ts.errors.ConvergenceError)
        assert not isinstance(exc, ts.errors.StepBudgetError)
        assert "diverged" in str(exc)
