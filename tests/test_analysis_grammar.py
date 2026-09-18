"""The analysis layer's *grammar*: one word per concept, and the unit is stated.

Every test here is the inverse of a defect the v6 API audit found by hand.  They
are grouped by the class of defect rather than by area, because the point of each
is that the rule holds **across** areas -- a keyword that means two things in two
sibling functions is the defect, and one function in isolation cannot show it.

- **Redundancy** -- two machines computing one quantity, which is how they came
  to disagree.  ``max_lyapunov`` is retired: the maximal exponent is
  ``lyapunov_spectrum(k=1)`` and nothing else, so the two cannot drift.
- **Units** -- ``transient`` is time for a flow and iterations for a map, at
  every door, and the refusal says so when the wrong one arrives.
- **Wrong subject** -- handing data to a model analysis (or the reverse) is
  answered by name, with a line that runs, and with the typed error the rest of
  the layer raises.
- **Silent wrong answers** -- a constant series is not "deterministic", a
  three-sample lag curve has no lag on it, and a period in time units is not
  quoted in samples.
- **One spelling** -- ``region`` is positional everywhere, ``plane`` is refused
  when it carries the *other* door's meaning, and one call has one return shape.
"""

from __future__ import annotations

import inspect
import re
import warnings

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import ConvergenceError, InvalidInputError, InvalidParameterError

A = ts.analysis


@pytest.fixture(scope="module")
def henon() -> object:
    return ts.systems.Henon()


@pytest.fixture(scope="module")
def lorenz() -> object:
    return ts.systems.Lorenz()


# ---------------------------------------------------------------------------
# Redundancy retired: one quantity, one machine
# ---------------------------------------------------------------------------


class TestTheMaximalExponentHasExactlyOneDoor:
    """``max_lyapunov`` is gone; ``lyapunov_spectrum(k=1)`` is the one door.

    Measured at HEAD before the retirement, on Hénon from ``ic=[0.1, 0.1]``:
    ``max_lyapunov(n=20000)`` returned ``0.4232673343379148`` and
    ``lyapunov_spectrum(k=1, n=20000)[0]`` returned ``0.4159988587314602`` --
    one nominal question, two public functions, two answers, and nothing told
    the reader which to believe.  The difference was the burn-in, which only one
    of them did; it is now this door's default.
    """

    def test_the_second_door_is_closed_and_names_the_first(self) -> None:
        with pytest.raises(ImportError, match=r"lyapunov_spectrum\(system, k=1\)"):
            A.max_lyapunov  # noqa: B018 - the attribute access IS the test

    def test_the_surviving_number_is_the_better_one(self, henon: object) -> None:
        """The burnt-in value, not the cold one -- bit for bit."""
        got = float(A.lyapunov_spectrum(henon, k=1, n=20_000, ic=[0.1, 0.1])[0])
        assert got == 0.4232673343379148

    def test_a_map_burn_in_can_be_switched_off_by_name(self, henon: object) -> None:
        cold = float(A.lyapunov_spectrum(henon, k=1, n=20_000, ic=[0.1, 0.1], transient=0)[0])
        assert cold == 0.4159988587314602

    def test_n_counts_iterations_on_a_map(self, henon: object) -> None:
        """Doubling ``n`` doubles the work -- ``n`` is iterations, not cycles."""
        short = float(A.lyapunov_spectrum(henon, k=1, n=2_000, ic=[0.1, 0.1], transient=0)[0])
        long = float(A.lyapunov_spectrum(henon, k=1, n=20_000, ic=[0.1, 0.1], transient=0)[0])
        assert short != long

    def test_a_flow_keeps_the_jacobian_free_two_trajectory_estimator(self, lorenz: object) -> None:
        """The Jacobian-free algorithm survived the retirement, behind this door."""
        doc = A.lyapunov_spectrum.__doc__ or ""
        assert "two-trajectory" in doc, "the docstring must say the machine is still here"
        assert "no right-hand side to differentiate" in doc

    def test_the_two_trajectory_knobs_came_with_the_machine(self) -> None:
        """``d0`` and ``steps_per`` are not formalities — one of them is load-bearing.

        On an exactly-linear expanding flow the growth rate does not depend on
        the perturbation size, so the default ``d0 = 1e-9`` against a reference
        that grows without bound loses the separation to floating-point
        cancellation and the loop reports a collapse.  ``d0=1e-4`` measures
        ``dx/dt = 0.5 x`` to 1e-6, at ``dt=None`` and at an explicit ``dt``
        alike — the normalisation regression that keyword exists to pin.
        """

        def expanding(u: object, dt: float) -> list[float]:
            return [float(np.asarray(u).ravel()[0]) * float(np.exp(0.5 * dt))]

        w = ts.WrappedSystem(expanding, dim=1, family="ode", ic=[1.0], default_dt=0.05)
        kw = {"final_time": 10.0, "steps_per": 2, "transient": 10.0, "d0": 1e-4}
        implicit = float(A.lyapunov_spectrum(w, k=1, ic=[1.0], **kw)[0])
        explicit = float(A.lyapunov_spectrum(w, k=1, ic=[1.0], dt=0.05, **kw)[0])
        assert implicit == pytest.approx(0.5, abs=1e-5)
        assert implicit == explicit, "dt=None must normalise by the real per-step advance"

    @pytest.mark.parametrize("word", ["d0", "steps_per"])
    def test_a_perturbation_knob_at_a_frame_machine_is_refused_by_name(
        self, lorenz: object, word: str
    ) -> None:
        """Nothing is perturbed when a tangent frame is carried — say so."""
        with pytest.raises(InvalidParameterError, match="right-hand side to differentiate"):
            A.lyapunov_spectrum(lorenz, final_time=10.0, **{word: 2})

    def test_a_stepper_only_map_still_has_a_machine(self) -> None:
        """A ``WrappedSystem`` map has no tangent tape, so it rescales two orbits.

        Before the fold this raised ``lyapunov_spectrum() needs a system that
        implements it`` and the only way through was the second door, so
        retiring that door without folding the machine in would have deleted the
        capability outright.
        """

        def logistic(x: object, _n: object) -> list[float]:
            v = float(np.asarray(x).ravel()[0])
            return [3.9 * v * (1.0 - v)]

        w = ts.WrappedSystem(logistic, dim=1, family="map", ic=[0.5])
        spec = A.lyapunov_spectrum(w, k=1, ic=[0.3], n=5_000)
        assert float(spec[0]) > 0.3
        assert spec.meta["estimator"] == "two-trajectory"


class TestARedundantNameSaysWhyItExists:
    """A kept alias must state its identity; a kept *estimator* must state its edge."""

    @pytest.mark.parametrize(
        ("name", "phrase"),
        [
            ("box_counting_dimension", "identical"),
            ("information_dimension", "identical"),
            ("generalized_dimension", "different estimator"),
            ("correlation_dimension", "Not the same estimator"),
            ("correlation_sum", "the *curve*"),
            ("kaplan_yorke_dimension", "spectrum.kaplan_yorke"),
            ("cao_dimension", "no rejection threshold to pick"),
            ("false_nearest_neighbors", "explicit, interpretable rejection criterion"),
            ("embedding_dimension", "identical"),
        ],
    )
    def test_the_docstring_says_which_to_choose(self, name: str, phrase: str) -> None:
        # Normalised, because a docstring wraps: the sentence is what matters,
        # not where the line break landed.
        doc = re.sub(r"\s+", " ", getattr(A, name).__doc__ or "")
        assert phrase in doc, f"{name} does not say why you would choose it"

    def test_the_two_d0_spellings_are_the_same_number(self) -> None:
        pts = ts.systems.Lorenz().run(final_time=40.0, dt=0.02, ic=[1.0, 1.0, 1.0]).y
        with warnings.catch_warnings():  # the D_q monotonicity report is orthogonal here
            warnings.simplefilter("ignore")
            assert float(A.box_counting_dimension(pts)) == float(A.generalized_dimension(pts, 0.0))
            assert float(A.information_dimension(pts)) == float(A.generalized_dimension(pts, 1.0))


# ---------------------------------------------------------------------------
# Units stated at the door -- and in the refusal
# ---------------------------------------------------------------------------


class TestTransientHasOneUnitPerFamily:
    """Time for a flow, iterations for a map, at every door that takes it.

    Measured before the fix: ``lyapunov_spectrum(henon, transient=500)``
    REFUSED the word outright, so the one family whose estimator needed a
    burn-in most was the one that could not ask for it -- while the sibling door
    did it silently and got a different number for the same question.
    """

    #: Every public analysis that takes ``transient``, and the unit word its
    #: docstring must contain.  A new door joins the sweep by existing.
    DOORS = (
        "lyapunov_spectrum",
        "gali",
        "zero_one_test",
        "return_map",
        "orbit_diagram",
        "periodic_orbits",
    )

    @pytest.mark.parametrize("name", DOORS)
    def test_the_docstring_states_the_unit(self, name: str) -> None:
        fn = getattr(A, name)
        assert "transient" in inspect.signature(fn).parameters
        doc = fn.__doc__ or ""
        start = doc.index("transient :")
        entry = re.sub(r"\s+", " ", doc[start : start + 700]).lower()
        assert "time units" in entry or "iterations" in entry, entry[:200]

    def test_a_fractional_iteration_count_on_a_map_is_refused(self, henon: object) -> None:
        with pytest.raises(InvalidParameterError, match="ITERATIONS"):
            A.lyapunov_spectrum(henon, transient=2.5)

    def test_a_negative_burn_in_is_refused(self, henon: object) -> None:
        with pytest.raises(InvalidParameterError, match="ITERATIONS"):
            A.lyapunov_spectrum(henon, transient=-1)

    def test_an_off_basin_random_draw_is_retried_not_raised(self) -> None:
        """The burn-in runs BEFORE the family's retry, so it needs its own.

        ``FoldedTowel`` declares no default initial condition and a good
        fraction of its box is outside the basin.  Measured when the burn-in
        first landed here with no retry, it failed roughly half its calls with
        ``ConvergenceError: map diverged at iteration 12`` — a draw the estimator
        downstream would have simply re-drawn.

        (The FIRST draw is still the family's own unseeded one — that is the
        documented behaviour of a system with no ``_default_ic`` — so this pins
        that every call *answers*, not that two calls agree bit for bit.)
        """
        towel = ts.systems.FoldedTowel()
        for _ in range(6):
            exps = np.asarray(A.lyapunov_spectrum(towel, n=5_000))
            assert np.all(np.isfinite(exps)), exps
            # the hyperchaotic folded-towel spectrum, ~(0.43, 0.38, -3.30)
            assert exps[0] == pytest.approx(0.43, abs=0.05), exps

    def test_an_explicit_ic_that_diverges_is_never_silently_replaced(self) -> None:
        """A start the caller typed is theirs; re-drawing it would answer elsewhere."""
        towel = ts.systems.FoldedTowel()
        with pytest.raises(ConvergenceError):
            A.lyapunov_spectrum(towel, n=5_000, ic=[50.0, 50.0, 50.0])

    def test_a_map_burns_in_by_default(self, henon: object) -> None:
        """``transient=None`` is 500 iterations, not nothing.

        The word used to be REFUSED here -- "a map spectrum has nothing to
        discard" -- which reads the QR frame's reorthonormalisation as if it
        landed the base orbit on the attractor.  It does not.
        """
        assert float(A.lyapunov_spectrum(henon, k=1, n=5_000, ic=[0.1, 0.1])[0]) != float(
            A.lyapunov_spectrum(henon, k=1, n=5_000, ic=[0.1, 0.1], transient=0)[0]
        )


class TestTheKernelWordIsSolver:
    """v6 split ``solver=`` (a kernel) from ``method=`` (an estimator) at ``run``.

    ``lyapunov_spectrum`` had not followed, so a user who obeyed the rule they
    had just been taught by ``run()``'s own refusal hit
    ``unexpected keyword 'solver'`` and had nowhere to go.
    """

    def test_solver_is_the_word(self, lorenz: object) -> None:
        exps = A.lyapunov_spectrum(lorenz, final_time=20.0, ic=[1.0, 1.0, 1.0], solver="rk4")
        assert np.asarray(exps).shape == (3,)

    def test_method_raises_naming_solver(self, lorenz: object) -> None:
        with pytest.raises(InvalidParameterError) as excinfo:
            A.lyapunov_spectrum(lorenz, final_time=20.0, solver=None, method="rk4")
        assert 'solver="bdf"' in str(excinfo.value)

    @pytest.mark.parametrize(
        ("kwargs", "phrase"),
        [
            ({"final_time": 10.0}, "TIME UNITS"),
            ({"dt": 0.1}, "TIME UNITS"),
            ({"solver": "rk4"}, "numerical kernel"),
        ],
    )
    def test_a_flow_keyword_at_a_map_is_refused_with_its_unit(
        self, henon: object, kwargs: dict[str, object], phrase: str
    ) -> None:
        with pytest.raises(InvalidParameterError, match=phrase):
            A.lyapunov_spectrum(henon, **kwargs)  # type: ignore[arg-type]

    def test_a_map_keyword_at_a_flow_is_refused_with_its_unit(self, lorenz: object) -> None:
        with pytest.raises(InvalidParameterError, match="ITERATIONS"):
            A.lyapunov_spectrum(lorenz, n=10)


# ---------------------------------------------------------------------------
# Wrong subject: answered by name, with a line that runs
# ---------------------------------------------------------------------------


class TestEveryWrongSubjectDoorIsTyped:
    """A wrong subject is a ``TypeError`` (``InvalidInputError``) everywhere.

    ``gali`` and ``expansion_entropy`` raised ``NotImplementedError`` -- a
    ``RuntimeError`` -- so ``except TypeError`` caught this mistake at eighteen
    doors and missed it at those two.  ``orbit_diagram`` leaked
    ``'Trajectory' object has no attribute 'with_params'``;
    ``invariant_density`` leaked numpy's ``float() argument must be ... not
    'Lorenz'``.
    """

    def test_data_handed_to_a_model_analysis(self, lorenz: object) -> None:
        traj = lorenz.run(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0])  # type: ignore[attr-defined]
        for call in (
            lambda: A.gali(traj),
            lambda: A.expansion_entropy(traj),
            lambda: A.orbit_diagram(traj, "rho", [1.0, 2.0]),
        ):
            with pytest.raises(TypeError) as excinfo:
                call()
            message = str(excinfo.value)
            assert "needs a system" in message
            assert "traj.system" in message, "the remedy must run on what was held"

    def test_a_system_handed_to_a_data_analysis(self, lorenz: object) -> None:
        with pytest.raises(TypeError) as excinfo:
            A.invariant_density(lorenz)
        assert "Lorenz" in str(excinfo.value)
        assert "float()" not in str(excinfo.value), "no raw numpy text"

    def test_a_result_first_analysis_names_its_producer(self, lorenz: object) -> None:
        traj = lorenz.run(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0])  # type: ignore[attr-defined]
        with pytest.raises(TypeError) as excinfo:
            A.basin_entropy(traj)
        message = str(excinfo.value)
        assert "reads what another analysis returns" in message
        assert "ts.analysis.basins(system, region)" in message

    def test_a_dde_at_a_tangent_analysis_says_why(self) -> None:
        mg = ts.systems.MackeyGlass()
        for call in (lambda: A.gali(mg, k=2), lambda: A.expansion_entropy(mg)):
            with pytest.raises(InvalidInputError, match="infinite-dimensional history"):
                call()

    def test_a_field_analysis_on_a_map_hands_back_a_line(self, henon: object) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            A.flow_field(henon)
        message = str(excinfo.value)
        assert "no vector field" in message
        assert "ts.analysis.find(system)" in message


# ---------------------------------------------------------------------------
# Silent wrong answers
# ---------------------------------------------------------------------------


class TestADegenerateInputIsRefusedNotAnswered:
    def test_a_constant_series_is_not_deterministic(self) -> None:
        """``rqa(np.ones(2000))`` reported ``DET = 1.000 ... deterministic``.

        Every pair of points recurs, so the plot is all-ones: there are no
        *lines* to count, and DET / LAM / ENTR are vacuous.  Every sibling
        estimator (``mutual_information``, ``correlation_dimension``,
        ``estimate_period``) already refused a constant series by name.
        """
        flat = np.ones(2000)
        for call in (
            lambda: A.rqa(flat, recurrence_rate=0.05),
            lambda: A.recurrence_matrix(flat, recurrence_rate=0.05),
            lambda: A.windowed_rqa(flat[:600], window=100, recurrence_rate=0.05),
        ):
            with pytest.raises(ValueError, match="series is constant"):
                call()

    def test_a_lag_curve_needs_more_samples_than_lags(self) -> None:
        """``optimal_delay(np.arange(3.), max_delay=50)`` returned a confident ``1``."""
        with pytest.raises(ValueError, match="series too short"):
            A.optimal_delay(np.arange(3.0), max_delay=50)

    def test_non_finite_data_is_diagnosed_as_non_finite(self) -> None:
        """It was reported as "no autocorrelation zero-crossing -- may be aperiodic".

        An accurate description of the symptom and a false diagnosis of the
        cause, which sent the caller to ``method='fft'`` or to collect more data.
        """
        with pytest.raises(ValueError, match="non-finite"):
            A.estimate_period(np.array([1.0, np.nan, 3.0] * 20))


class TestThePeriodIsQuotedInTheUnitItWasMeasuredIn:
    """``estimate_period`` returns ``lag * step``; the unit follows ``step``.

    The repr hard-coded "samples", so the library's own docstring example --
    ``estimate_period(VanDerPol().run(final_time=200, dt=0.01))`` -- printed
    ``6.65372 samples`` for 6.65372 TIME UNITS (665 samples), wrong by ``1/dt``
    for every ``Trajectory`` caller.
    """

    def test_a_trajectory_answers_in_time_units(self) -> None:
        traj = ts.systems.VanDerPol().run(final_time=200.0, dt=0.01, ic=[2.0, 0.0])
        result = A.estimate_period(traj)
        assert "time units" in str(result)
        assert float(result) == pytest.approx(6.65, abs=0.2)

    def test_a_bare_array_answers_in_samples(self) -> None:
        x = np.sin(2.0 * np.pi * np.arange(4000) / 500.0)
        result = A.estimate_period(x)
        assert "samples" in str(result)
        assert float(result) == pytest.approx(500.0, rel=0.05)

    def test_an_explicit_dt_switches_the_unit(self) -> None:
        x = np.sin(2.0 * np.pi * np.arange(4000) / 500.0)
        assert "time units" in str(A.estimate_period(x, dt=0.01))


# ---------------------------------------------------------------------------
# One spelling per concept
# ---------------------------------------------------------------------------


class TestRegionIsPositionalAtEveryDoor:
    """Five doors took it positionally and two did not, for the same argument."""

    DOORS = (
        "attractors",
        "basins",
        "basin_fractions",
        "continuation",
        "expansion_entropy",
        "fixed_points",
        "periodic_orbits",
    )

    @pytest.mark.parametrize("name", DOORS)
    def test_region_is_not_keyword_only(self, name: str) -> None:
        parameter = inspect.signature(getattr(A, name)).parameters["region"]
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, name

    def test_it_is_actually_callable_that_way(self) -> None:
        vdp = ts.systems.VanDerPol()
        assert len(A.fixed_points(vdp, [(-4.0, 4.0), (-4.0, 4.0)])) >= 1
        assert len(A.periodic_orbits(ts.systems.Henon(), 2, [(-2.0, 2.0), (-2.0, 2.0)])) >= 1


class TestPlaneNeverMeansTheOtherDoorsThing:
    """``plane=`` is a cutting SECTION at one door and VIEW AXES at eight others.

    Each used to silently eat the other's spelling:
    ``flow_field(lorenz, plane=("y", 0.0))`` drew the x-y plane and said nothing.
    """

    def test_a_section_spelling_at_a_field_analysis(self, lorenz: object) -> None:
        with pytest.raises(InvalidParameterError) as excinfo:
            A.flow_field(lorenz, plane=("y", 0.0), grid=3)
        message = str(excinfo.value)
        assert "poincare_section" in message
        assert 'plane=("x", "z")' in message

    def test_an_axes_spelling_at_the_section_door(self, lorenz: object) -> None:
        with pytest.raises(InvalidParameterError) as excinfo:
            A.poincare_section(lorenz, ("x", "z"), crossings=3)
        assert "flow_field" in str(excinfo.value)

    @pytest.mark.parametrize("plane", [(0, 1), ("x", "z"), (0, 2)])
    def test_the_real_axes_spellings_still_work(self, lorenz: object, plane: tuple) -> None:
        assert A.flow_field(lorenz, plane=plane, grid=3) is not None


class TestOneCallHasOneReturnShape:
    def test_zero_one_test_always_returns_the_result(self) -> None:
        """``return_distribution=True`` made one call return a 2-tuple instead."""
        x = ts.systems.Logistic(params={"r": 4.0}).run(steps=2500, ic=[0.3]).component("x")
        result = A.zero_one_test(x, n_c=20, seed=0)
        assert not isinstance(result, tuple)
        assert result.distribution.shape == (20,)
        assert float(result) == pytest.approx(float(np.median(result.distribution)))
        assert "return_distribution" not in inspect.signature(A.zero_one_test).parameters

    def test_correlation_sum_names_its_two_arrays(self) -> None:
        pts = ts.systems.Lorenz().run(final_time=30.0, dt=0.02, ic=[1.0, 1.0, 1.0]).y
        curve = A.correlation_sum(pts)
        radii, sums = curve  # still unpacks like the bare tuple it replaced
        assert curve.radii is radii and curve.sums is sums
        assert radii.ndim == sums.ndim == 1


class TestAnAnsweringDefaultIsReproducible:
    """Half the namespace answered differently on every call.

    Measured: three identical ``fixed_points(Thomas(), region=[(-5, 5)] * 3,
    n_seeds=60)`` calls found 16, 19 and 17 equilibria.
    """

    SAMPLING_DOORS = (
        "fixed_points",
        "periodic_orbits",
        "orbit_diagram",
        "poincare_section",
        "return_map",
        "attractors",
        "basins",
        "basin_fractions",
        "continuation",
        "expansion_entropy",
        "gali",
        "zero_one_test",
    )

    @pytest.mark.parametrize("name", SAMPLING_DOORS)
    def test_seed_defaults_to_zero(self, name: str) -> None:
        assert inspect.signature(getattr(A, name)).parameters["seed"].default == 0, name

    def test_repeated_calls_agree(self) -> None:
        thomas = ts.systems.Thomas()
        counts = {
            len(A.fixed_points(thomas, region=[(-5.0, 5.0)] * 3, n_seeds=40)) for _ in range(3)
        }
        assert len(counts) == 1, counts
