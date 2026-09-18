"""What a blind tester re-running the worst paths was still given wrongly.

Every test here is named after the **behaviour a user sees**, and every one of
them failed before the change it guards.  The findings, worst first:

1. An estimator handed a **runaway orbit** answered ``trusted = True`` with a
   four-significant-figure exponent and no warning, on data the library had
   itself flagged as *"not on an attractor"*.
2. A ``TrajectoryBatch`` with five NaN members printed ``12 trajectories`` and
   exposed no way to find out.
3. ``tr[::5].meta["dt"]`` stayed at the *run's* ``dt`` — a 5x wrong conversion
   factor, handed over on request.
4. ``label=`` at the front door was answered with ``zlabel=``, an **axis** name.
5. ``basin_fractions(..., n_seeds=...)`` leaked ``_AttractorMapper`` in a
   ``TypeError``.
6. A remedy line for a 4-D system handed back a 3-axis template.
7. The vector-field warning taught ``domain=`` + ``xlim=`` as separable, and the
   pair then raised.
8. The cobweb's default orbit buried the map it is drawn to teach.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.analysis._common import RunawayOrbitWarning


@pytest.fixture(scope="module")
def runaway() -> ts.Trajectory:
    """A Chua run that leaves the building — finite, returned, and meaningless."""
    return ts.systems.Chua().run(final_time=50.0, dt=0.01, ic=[500.0, 0.0, 0.0])


class TestAnAnalysisOfARunawayOrbitSaysSo:
    """Finding 1 — the flag the trajectory already carried reached nothing."""

    def test_the_fixture_really_is_a_runaway(self, runaway: ts.Trajectory) -> None:
        """Guard the premise: if Chua stops escaping, the rest is vacuous."""
        assert runaway.unbounded is not None
        assert runaway.unbounded.peak > 1e8

    def test_correlation_dimension_of_a_runaway_is_untrusted(self, runaway: ts.Trajectory) -> None:
        """It answered ``D_corr = 0.48954 ... R2 = 0.9976``, trusted, silently."""
        with pytest.warns(RunawayOrbitWarning, match="not on an attractor"):
            result = ts.analysis.correlation_dimension(runaway)
        assert result.trusted is False
        assert "not on an attractor" in repr(result)

    def test_a_runaway_beats_a_clean_looking_fit(self, runaway: ts.Trajectory) -> None:
        """The escape outranks R^2: the fit was *excellent* and meaningless."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RunawayOrbitWarning)
            result = ts.analysis.correlation_dimension(runaway)
        assert result.r_squared > 0.99  # the fit that fooled the reader
        assert result.trusted is False

    def test_lyapunov_from_data_of_a_runaway_refuses_the_chaos_verdict(
        self, runaway: ts.Trajectory
    ) -> None:
        """It called a monotone blow-up ``chaotic`` to four figures."""
        with pytest.warns(RunawayOrbitWarning):
            result = ts.analysis.lyapunov_from_data(runaway["x"][::10], dt=0.1)
        assert result.trusted is False
        assert result.chaotic is None
        assert "not on an attractor" in repr(result)

    def test_a_bare_array_that_ran_away_is_caught_too(self, runaway: ts.Trajectory) -> None:
        """The escape is read off the DATA, so indexing cannot launder it."""
        series = np.asarray(runaway["x"])[::10]
        assert not hasattr(series, "unbounded")
        with pytest.warns(RunawayOrbitWarning):
            ts.analysis.lyapunov_from_data(series, dt=0.1)

    def test_an_honest_attractor_is_left_alone(self) -> None:
        """The guard must not fire on the orbits people actually measure."""
        traj = ts.systems.Lorenz().run(final_time=60.0, dt=0.02, ic=[1.0, 1.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RunawayOrbitWarning)
            result = ts.analysis.correlation_dimension(traj)
        assert result.trusted is True

    def test_the_number_is_still_returned(self, runaway: ts.Trajectory) -> None:
        """Refusing outright would make the estimator useless; it hedges instead."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RunawayOrbitWarning)
            result = ts.analysis.correlation_dimension(runaway)
        assert np.isfinite(float(result))
        assert result.to_dict(full=True)["trusted"] is False


class TestAnEnsembleSaysHowManyMembersDiverged:
    """Finding 2 — 5 of 12 members were NaN and the batch printed nothing."""

    @pytest.fixture(scope="class")
    def batch(self) -> object:
        class Blowup(ts.ContinuousSystem):
            """dx/dt = x^2 - a: everything right of sqrt(a) escapes in finite time."""

            params = {"a": 1.0}
            variables = ("x", "y")
            dim = 2

            @staticmethod
            def _equations(u, t, a):  # noqa: ANN001, ANN205, D102
                return [u(0) ** 2 - a, -u(1)]

        ics = np.column_stack([np.linspace(-0.5, 2.0, 12), np.ones(12)])
        return Blowup().ensemble(ics).run(final_time=20.0, dt=0.1)

    def test_the_batch_exposes_which_members_diverged(self, batch: object) -> None:
        """``dir(batch)`` held no ``diverged``, no ``status``, no count."""
        mask = batch.diverged  # type: ignore[attr-defined]
        assert mask.dtype == bool
        assert mask.shape == (12,)
        assert mask.sum() == 5

    def test_the_repr_says_how_many_diverged(self, batch: object) -> None:
        """It read ``12 trajectories - 201 samples - 2-D states`` and stopped."""
        assert "5 of 12 diverged" in repr(batch)

    def test_a_clean_batch_says_nothing_about_divergence(self) -> None:
        """The clause is a warning, not decoration: it must not always appear."""
        ics = np.array([[1.0, 1.0, 1.0], [1.01, 1.0, 1.0]])
        clean = ts.systems.Lorenz().ensemble(ics).run(final_time=2.0, dt=0.1)
        assert clean.diverged.sum() == 0
        assert "diverged" not in repr(clean)


class TestASlicedTrajectoryDescribesItself:
    """Finding 3 — ``meta`` kept describing the run after the object changed."""

    @pytest.fixture(scope="class")
    def traj(self) -> ts.Trajectory:
        return ts.systems.Lorenz().run(final_time=20.0, dt=0.01, ic=[1.0, 1.0, 1.0])

    def test_decimating_updates_the_recorded_step(self, traj: ts.Trajectory) -> None:
        """``tr[::5].meta['dt']`` was 0.01 while ``tr[::5].dt`` was 0.05."""
        sub = traj[::5]
        assert sub.meta["dt"] == pytest.approx(0.05)
        assert sub.meta["dt"] == pytest.approx(sub.dt)

    def test_a_tail_slice_updates_its_start_and_its_first_state(self, traj: ts.Trajectory) -> None:
        """``meta['t0']`` said 0.0 and ``meta['ic']`` said [1, 1, 1] for a t=5 tail."""
        tail = traj[500:]
        assert tail.meta["t0"] == pytest.approx(float(tail.t[0]))
        np.testing.assert_allclose(tail.meta["ic"], tail.y[0])

    def test_a_non_uniform_slice_drops_the_step_rather_than_lying(
        self, traj: ts.Trajectory
    ) -> None:
        """No uniform step exists, so none is reported."""
        ragged = traj[[0, 1, 5, 400]]
        assert ragged.meta.get("dt") is None
        assert ragged.dt is None

    def test_the_run_provenance_survives(self, traj: ts.Trajectory) -> None:
        """Only the three keys that describe the ROWS are re-derived."""
        sub = traj[::5]
        assert sub.meta["system"] == "Lorenz"
        assert sub.meta["solver"] == traj.meta["solver"]
        assert sub.meta["rtol"] == traj.meta["rtol"]


class TestTheWordForNamingCurvesIsDiscoverableFromItsOwnError:
    """Finding 4 — ``label=`` was answered with ``zlabel=``, an axis name."""

    @staticmethod
    def _two_orbits() -> tuple[ts.Trajectory, ts.Trajectory]:
        lor = ts.systems.Lorenz()
        return (
            lor.run(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0]),
            lor.run(final_time=2.0, dt=0.05, ic=[1.1, 1.0, 1.0]),
        )

    def test_the_singular_names_the_plural(self) -> None:
        a, b = self._two_orbits()
        with pytest.raises(ts.errors.InvalidParameterError) as excinfo:
            ts.plot(a, b, components="x", label=["A", "B"])
        message = str(excinfo.value)
        assert "labels=" in message
        assert "zlabel" not in message

    def test_the_accepted_listing_names_the_word_that_works(self) -> None:
        """``labels=`` was on the signature and in no listing any error printed."""
        a, b = self._two_orbits()
        with pytest.raises(ts.errors.InvalidParameterError) as excinfo:
            ts.plot(a, b, components="x", nonsense_kw=1)
        assert "labels=" in str(excinfo.value)


class TestBasinFractionsSpeaksTheSameWordAsItsSiblings:
    """Finding 5 — the odd-one-out spelling, and a private class in a TypeError."""

    def test_n_seeds_is_the_spelling(self) -> None:
        henon = ts.systems.Henon()
        result = ts.analysis.basin_fractions(henon, [(-2, 2), (-2, 2)], n_seeds=40, seed=0)
        assert result.n == 40

    def test_the_retired_spelling_names_the_live_one(self) -> None:
        henon = ts.systems.Henon()
        with pytest.raises(ts.errors.InvalidParameterError, match="n_seeds"):
            ts.analysis.basin_fractions(henon, [(-2, 2), (-2, 2)], n=40, seed=0)

    def test_a_typo_never_names_a_private_class(self) -> None:
        """It read ``_AttractorMapper.__init__() got an unexpected keyword``."""
        henon = ts.systems.Henon()
        with pytest.raises(ts.errors.InvalidParameterError) as excinfo:
            ts.analysis.basin_fractions(henon, [(-2, 2), (-2, 2)], lost_step=3, seed=0)
        message = str(excinfo.value)
        assert "_AttractorMapper" not in message
        assert "lost_steps" in message


class TestARemedyLineMatchesTheCallItAnswers:
    """Finding 6 — a 4-D system was handed a 3-axis template to copy."""

    def test_the_flat_recurrence_box_line_has_one_entry_per_component(self) -> None:
        class Four(ts.ContinuousSystem):
            """A 4-D flow, so a 3-axis remedy cannot be pasted."""

            params = {"a": 0.2}
            variables = ("x", "y", "z", "w")
            dim = 4

            @staticmethod
            def _equations(u, t, a):  # noqa: ANN001, ANN205, D102
                return [-u(1) - u(2), u(0) + a * u(1), 0.2 + u(2) * (u(0) - 5.7), -0.05 * u(3)]

        with pytest.raises(ts.errors.InvalidInputError) as excinfo:
            ts.analysis.basins(Four(), [(-2, 2, 6), (-2, 2, 6), (0, 0, 1), (0, 0, 1)])
        line = next(ln for ln in str(excinfo.value).splitlines() if "recurrence=[" in ln)
        assert line.count("(") - line.count("()") == 4


class TestAFieldCanBeDrawnOverOneBoxAndFramedOverAnother:
    """Finding 7 — the warning taught a combination the next call refused."""

    def test_domain_and_xlim_may_be_given_together(self) -> None:
        import matplotlib.pyplot as plt

        vdp = ts.systems.VanDerPol()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot = ts.plot(
                vdp, "vector_field", domain=[(-6, 6), (-6, 6)], xlim=(-3, 3), ylim=(-3, 3)
            )
        # The field was evaluated over the big box...
        xs = np.concatenate([np.asarray(layer.data["x"]).ravel() for layer in plot.layers])
        assert xs.max() > 3.5
        # ...and the axes were framed on the small one.
        assert plot.x.limits == (-3, 3)
        plt.close("all")

    def test_two_spellings_inside_one_transform_call_still_raise(self) -> None:
        vdp = ts.systems.VanDerPol()
        with pytest.raises(ts.errors.InvalidParameterError, match="two spellings"):
            ts.plot(vdp, ("vector_field", {"domain": [(-6, 6), (-6, 6)], "xlim": (-3, 3)}))


class TestTheCobwebShowsTheMapItIsDrawnToTeach:
    """Finding 8 — the staircase covered the parabola and the diagonal solid."""

    def test_the_default_staircase_is_short_enough_to_see_through(self) -> None:
        import matplotlib.pyplot as plt

        plot = ts.plot(ts.systems.Logistic(), "cobweb")
        stair = next(layer for layer in plot.layers if layer.label == "orbit")
        assert np.asarray(stair.data["x"]).size <= 201
        plt.close("all")
