"""The frictions six blind testers hit, pinned by the behaviour they saw.

Each test is named after what a *user* experienced, not after the function that
changed, and each docstring carries the measurement that motivated it.  They are
deliberately in one file: they are a regression suite for a class of defect —
*the library knows something and does not say it* — rather than for one module.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import symengine as se

import tsdynamics as ts
from tsdynamics.analysis.dimensions import UnembeddedSeriesWarning
from tsdynamics.errors import InvalidInputError, InvalidParameterError

# ---------------------------------------------------------------------------
# 1. A diverged trajectory must never look like an answer
# ---------------------------------------------------------------------------


def test_an_escaping_run_says_so_instead_of_returning_a_clean_trajectory():
    """``Chua().run(ic=[500, 0, 0])`` returns ``max|y| = 1.6e9`` and no exception.

    It is finite, ``isnan`` is all ``False``, and the repr called it an ordinary
    trajectory — so an orbit that left the building was typographically
    identical to one on the attractor.
    """
    escaped = ts.systems.Chua().run(final_time=50.0, dt=0.01, ic=[500.0, 0.0, 0.0])
    assert np.isfinite(escaped.y).all()  # the premise: nothing raised
    record = escaped.unbounded
    assert record is not None
    assert record.peak > 1e8
    assert "unbounded" in repr(escaped)

    bounded = ts.systems.Lorenz().run(final_time=20.0, dt=0.01, ic=[1.0, 1.0, 1.0])
    assert bounded.unbounded is None
    assert "unbounded" not in repr(bounded)


class _Spiral(ts.ContinuousSystem):
    """An unstable focus — it crosses ``y = 0`` forever while growing forever."""

    params = {"a": 0.9}
    dim = 2
    variables = ("x", "y")

    @staticmethod
    def _equations(Y, t, *, a):
        return [a * Y(0) - Y(1), Y(0) + a * Y(1)]


def test_a_poincare_section_of_an_escaping_orbit_is_not_reported_as_crossings():
    """300 crossings of an orbit at ``max|x| = 9e55`` read exactly like 300 honest ones."""
    section = ts.analysis.poincare_section(
        _Spiral(), plane=("y", 0.0), crossings=40, ic=[1.0, 0.0], max_time=1e4
    )
    assert section.unbounded is not None
    assert "not a section of an attractor" in repr(section)


def test_a_lyapunov_spectrum_of_an_escaping_orbit_refuses_a_verdict():
    """It reported ``λ = [0.3057, 0.3054, -6.068]`` and hedged only about the horizon."""
    spectrum = ts.analysis.lyapunov_spectrum(
        ts.systems.Chua(), final_time=60.0, ic=[500.0, 0.0, 0.0]
    )
    assert spectrum.unbounded is True
    assert spectrum.chaotic is None
    assert spectrum.regime == "unbounded"
    assert "UNBOUNDED" in repr(spectrum)


def test_a_catalogue_system_with_a_finite_basin_starts_inside_it():
    """9 of 20 random draws in U[0,1]^3 escaped, so the first line anyone types

    returned garbage about a quarter of the time, differently in every process.
    """
    for name in ("Chua", "MultiChua", "SprottJerk", "SprottL", "SprottG"):
        system = ts.systems.get(name)()
        assert system._resolved_default_ic() is not None, name
        assert system.run(final_time=120.0, dt=0.5).unbounded is None, name


# ---------------------------------------------------------------------------
# 2. A dimension of one coordinate is not a dimension
# ---------------------------------------------------------------------------


def test_a_fractal_dimension_of_a_scalar_series_warns_and_is_untrusted():
    """``correlation_dimension(lorenz_x)`` answered ``0.997 ± 0.00015, R² = 1``.

    The truth is 2.06.  Its sibling ``lyapunov_from_data`` auto-embeds the same
    input and says so, so two neighbours in one documented group made opposite
    assumptions in silence.
    """
    traj = ts.systems.Lorenz().run(final_time=120.0, dt=0.01, transient=20.0, ic=[1.0, 1.0, 1.0])
    x = np.asarray(traj["x"])
    with pytest.warns(UnembeddedSeriesWarning, match="embed"):
        flat = ts.analysis.correlation_dimension(x)
    assert flat.trusted is False and flat.unembedded is True
    embedded = ts.analysis.correlation_dimension(ts.analysis.embed(x))
    assert embedded.trusted is True and 1.8 < float(embedded) < 2.4


# ---------------------------------------------------------------------------
# 3. A bifurcation diagram must distinguish an equilibrium from a period-1 cycle
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_an_orbit_diagram_names_the_equilibrium_branch():
    """Chua over alpha in [6, 11] drew a flat line summarised as ``periods seen: 1``.

    Four fifths of the sweep is the fixed-point branch, and nothing said so — a
    reader who cannot do a Routh-Hurwitz by hand reads that as a broken sweep.
    """
    diagram = ts.analysis.orbit_diagram(
        ts.systems.Chua(),
        "alpha",
        np.linspace(6.0, 11.0, 30),
        points_per_value=40,
        transient=80,
        dt=0.02,
    )
    assert diagram.equilibria.any()
    assert "EQUILIBRIUM" in repr(diagram)


def test_bifurcation_points_can_say_which_are_period_doublings():
    """20 values came back, of which the FIRST was the cascade and 19 were band splits.

    Successive gaps gave ratios like 11.7 — nothing near Feigenbaum's 4.669 — and
    no flag distinguished them.
    """
    diagram = ts.analysis.orbit_diagram(
        ts.systems.Logistic(), "r", np.linspace(2.8, 3.6, 120), points_per_value=80, transient=300
    )
    rows = diagram.bifurcation_points(labelled=True)
    doublings = rows["value"][rows["kind"] == "period-doubling"]
    assert doublings.size >= 2
    assert abs(float(doublings[0]) - 3.0) < 0.02
    assert abs(float(doublings[1]) - (1.0 + np.sqrt(6.0))) < 0.02
    # the plain array is unchanged
    np.testing.assert_allclose(diagram.bifurcation_points(), rows["value"])


# ---------------------------------------------------------------------------
# 4. A verdict must be a property of the system, not of the grid
# ---------------------------------------------------------------------------


class _Tank(ts.ContinuousSystem):
    """A damped oscillator in a double well — a provably SMOOTH basin boundary."""

    params = {"delta": 0.3, "load": 0.0}
    dim = 2
    variables = ("x", "v")

    @staticmethod
    def _equations(Y, t, *, delta, load):
        return [Y(1), -delta * Y(1) + Y(0) - Y(0) ** 3 + load]


@pytest.mark.slow
def test_the_uncertainty_exponent_verdict_does_not_flip_with_the_grid():
    """40x40 said "fractal boundary", 60x60 said "smooth", with nothing in between.

    The boundary is a saddle's stable manifold: smooth, D0 = 1.  An engineer who
    reads "fractal boundary" concludes the safety margin is meaningless and
    stops, so a false positive here costs a project.
    """
    system = _Tank()
    coarse = ts.analysis.uncertainty_exponent(
        ts.analysis.basins(system, [(-2.0, 2.0, 40), (-2.0, 2.0, 40)])
    )
    assert coarse.resolved is False
    assert coarse.final_state_sensitive is None
    assert "inconclusive at this resolution" in repr(coarse)

    fine = ts.analysis.uncertainty_exponent(
        ts.analysis.basins(system, [(-2.0, 2.0, 100), (-2.0, 2.0, 100)])
    )
    assert fine.resolved is True
    assert fine.final_state_sensitive is False  # smooth, which is the truth


@pytest.mark.slow
def test_resilience_reports_the_grid_it_was_measured_on():
    """It printed ``0.615385`` for a distance that is exactly six cells, biased +7.8%.

    Its sibling in the same module prints ``49.4% ± 0.5%``; six significant
    figures on a safety margin is an invitation to quote it.
    """
    image = ts.analysis.basins(_Tank(), [(-2.0, 2.0, 40), (-2.0, 2.0, 40)])
    margin = ts.analysis.resilience(image, attractor_id=1)
    text = repr(margin)
    assert "±" in text and "cells on a 40 × 40 grid" in text
    assert float(margin) > 0.0


# ---------------------------------------------------------------------------
# 5. The result surface: what the repr shows is what you can reach
# ---------------------------------------------------------------------------


def test_the_repr_labels_an_item_with_the_accessor_that_returns_it():
    """The repr printed ``[0] x* = ... unstable``, and ``fp[0]`` is a bare ndarray.

    Three readers guessed ``fp[0].stable`` and got an ``AttributeError``.  ``[]``
    still gives numbers (contract §4.2 rule 6); the printed token is now the
    expression that gives the record.
    """
    points = ts.analysis.fixed_points(ts.systems.VanDerPol(), region=[(-3, 3), (-3, 3)], seed=0)
    assert "details[0]" in repr(points)
    assert points.details[0].stable is not None


def test_printing_a_result_shows_everything_the_repl_shows():
    """``print(attractor_set)`` gave the count and dropped the locations."""
    spectrum = ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), final_time=40.0, ic=[1.0] * 3)
    assert str(spectrum) == repr(spectrum)
    assert "\n" in str(spectrum)
    assert f"{spectrum}" == spectrum.headline  # an f-string embeds one line


def test_headline_is_a_value_not_a_bound_method():
    """``print(result.headline)`` rendered ``<bound method AnalysisResult.headline …>``."""
    spectrum = ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), final_time=20.0, ic=[1.0] * 3)
    assert isinstance(spectrum.headline, str)


def test_to_frame_uses_the_declared_variable_names():
    """A pendulum declaring ``("theta", "omega")`` tabulated as ``x0`` / ``x1``."""
    # pandas is not a dependency — `to_frame()` raises a message naming it.
    pytest.importorskip("pandas")

    class Pendulum(ts.ContinuousSystem):
        params = {"b": 0.2}
        dim = 2
        variables = ("theta", "omega")

        @staticmethod
        def _equations(Y, t, *, b):
            return [Y(1), -b * Y(1) - se.sin(Y(0))]

    frame = ts.analysis.fixed_points(Pendulum(), region=[(-4, 4), (-3, 3)], seed=0).to_frame()
    assert "x_theta" in frame.columns and "x_omega" in frame.columns


def test_to_dict_does_not_carry_the_raw_distributions_by_default():
    """One ``RQAResult.to_dict()`` printed 236 KB, almost all of it a histogram."""
    traj = ts.systems.Lorenz().run(final_time=20.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    result = ts.analysis.rqa(traj, recurrence_rate=0.05)
    compact = result.to_dict()
    assert len(json.dumps(compact)) < 4000
    assert "omitted" in compact["diagonal_lengths"]
    assert len(json.dumps(result.to_dict(full=True))) > len(json.dumps(compact))
    # the verdict the repr prints is a key, not a keyword argument away
    assert compact["verdict"] == result.verdict


def test_the_literature_abbreviations_the_repr_prints_are_reachable():
    """The repr says ``L_max`` / ``DET`` / ``LAM`` / ``ENTR``; only the long names resolved."""
    traj = ts.systems.Lorenz().run(final_time=20.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    result = ts.analysis.rqa(traj, recurrence_rate=0.05)
    assert result.L_max == result.max_diagonal_length
    assert result.determinism == result.DET
    assert result.laminarity == result.LAM
    assert result.diagonal_entropy == result.ENTR


def test_the_labels_the_repr_prints_are_dict_keys():
    """The repr said ``Sbb`` and ``D0``; the keys were ``sbb`` and

    ``boundary_dimension``.  Three ``KeyError``s, all from names the library had
    just printed at the reader.
    """
    image = ts.analysis.basins(_Tank(), [(-2.0, 2.0, 30), (-2.0, 2.0, 30)])
    entropy = ts.analysis.basin_entropy(image)
    assert "Sbb" in repr(entropy)
    assert entropy.to_dict()["Sbb"] == entropy.sbb
    exponent = ts.analysis.uncertainty_exponent(image)
    assert "D0" in repr(exponent)
    assert exponent.to_dict()["D0"] == exponent.boundary_dimension


def test_a_scalar_field_reprs_its_answer():
    """Seven of the fifty analyses return one, and it dumped three arrays."""
    field = ts.analysis.ftle_field(ts.systems.VanDerPol(), [(-2, 2, 15), (-2, 2, 15)], final_time=1)
    text = repr(field)
    assert text.startswith("ScalarField") and "15×15" in text
    assert str(field) == text


def test_an_analysis_reached_as_a_method_on_a_result_is_taught_by_name():
    """``cont.tipping_points()`` was a SECOND spelling against the library's own rule."""
    spectrum = ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), final_time=20.0, ic=[1.0] * 3)
    with pytest.raises(AttributeError, match=r"ts\.analysis\.kaplan_yorke_dimension\(result\)"):
        spectrum.kaplan_yorke_dimension()
    assert hasattr(spectrum, "nope") is False


# ---------------------------------------------------------------------------
# 6. The doors: plain Python, and a refusal that teaches
# ---------------------------------------------------------------------------


def test_a_state_handed_to_a_constructor_is_answered_by_name():
    """``Lorenz([1., 1., 1.])`` gave ``TypeError: object is not iterable`` from dict()."""
    with pytest.raises(InvalidInputError, match="takes PARAMETERS, not a state"):
        ts.systems.Lorenz([1.0, 1.0, 1.0])
    assert ts.systems.Lorenz(sigma=12.0).sigma == 12.0


@pytest.mark.parametrize(
    "base,needle",
    [
        (ts.ContinuousSystem, "_equations"),
        (ts.DiscreteMap, "_step"),
        (ts.StochasticSystem, "_drift"),
        (ts.DelaySystem, "y(0, t - tau)"),
    ],
)
def test_an_abstract_family_teaches_its_subclass_contract(base, needle):
    """CPython's own message names the methods and stops — at the step a new

    author has least to go on.
    """
    with pytest.raises(TypeError) as err:
        base()
    text = str(err.value)
    assert needle in text
    assert "help(ts." in text


def test_an_sde_written_as_equations_is_told_which_half_is_missing():
    """``abstract methods '_diffusion', '_drift'`` never noticed ``_equations`` was there."""

    class Noisy(ts.StochasticSystem):
        params = {"a": 1.0}
        dim = 1

        @staticmethod
        def _equations(Y, t, *, a):
            return [-a * Y(0)]

    with pytest.raises(TypeError, match="that is the drift"):
        Noisy()


def test_an_unknown_solver_names_the_nearest_and_only_its_own_family():
    """``solver='LSODA'`` dumped 26 kernels alphabetically, including SDE-only ones."""
    with pytest.raises(InvalidParameterError) as err:
        ts.systems.Lorenz().run(final_time=1.0, solver="LSODA", ic=[1.0, 1.0, 1.0])
    text = str(err.value)
    assert "'bdf'" in text
    assert "euler_maruyama" not in text and "milstein" not in text


def test_poincare_section_takes_an_initial_condition_like_its_siblings():
    """It raised the one bare ``TypeError`` in the library, leaving only ``seed=``."""
    system = ts.systems.Rossler()
    first = ts.analysis.poincare_section(system, plane=("y", 0.0), crossings=20, ic=[1.0, 1.0, 1.0])
    again = ts.analysis.poincare_section(system, plane=("y", 0.0), crossings=20, ic=[1.0, 1.0, 1.0])
    np.testing.assert_array_equal(first.y, again.y)


def test_a_measured_trajectory_can_name_its_channel_at_the_constructor():
    """``variables=`` raised a bare ``TypeError`` on the door data users are pointed at."""
    traj = ts.Trajectory(np.arange(5.0), np.arange(5.0), variables=("voltage",))
    assert traj.variables == ("voltage",)
    np.testing.assert_array_equal(traj["voltage"], np.arange(5.0))
    with pytest.raises(InvalidInputError, match="every state component"):
        ts.Trajectory(np.arange(5.0), np.arange(5.0), variables=("a", "b"))


def test_every_field_analysis_reads_the_one_region_grammar():
    """Crossing the two grammars gave ``takes 1 positional argument but 2 were given``."""
    system = ts.systems.VanDerPol()
    region = [(-2.0, 2.0, 12), (-2.0, 2.0, 12)]
    assert ts.analysis.ftle_field(system, region, final_time=1.0).values.shape == (12, 12)
    assert ts.analysis.flow_field(system, region) is not None
    assert ts.analysis.nullclines(system, region) is not None
    with pytest.raises(InvalidParameterError, match="second answer to the same question"):
        ts.analysis.ftle_field(system, region, xlim=(-1.0, 1.0))


def test_a_map_run_returns_the_state_it_started_from():
    """``Logistic(r=2.8).run(steps=4, ic=[0.1]).y[0]`` was ``0.252`` — ``f(0.1)``.

    So ``t[0] = 0`` labelled ``x_1`` and ``traj["x"][n]`` was ``x_{n+1}``, while
    the flow family returned its initial condition as row 0.
    """
    run = ts.systems.Logistic(r=2.8).run(steps=4, ic=[0.1])
    assert run.y.shape == (5, 1)
    assert float(run.y[0, 0]) == pytest.approx(0.1)
    assert float(run.y[1, 0]) == pytest.approx(2.8 * 0.1 * 0.9)
    np.testing.assert_array_equal(run.t, np.arange(5))


# ---------------------------------------------------------------------------
# 7. Discovery: the same verb answers the same way
# ---------------------------------------------------------------------------


def test_a_wrong_case_name_is_answered_with_the_right_case():
    """``ts.Systems`` suggested three unrelated wrapper classes and never ``ts.systems``."""
    with pytest.raises(AttributeError, match=r"ts\.systems"):
        ts.Systems  # noqa: B018


def test_the_systems_registry_prints_a_table_like_the_analysis_one():
    """It returned ``[<class 'tsdynamics.systems.continuous....Chua'>, ...]``."""
    text = repr(ts.systems.find("chua"))
    assert "ts.systems.Chua" in text
    assert "Matsumoto (1984)" in text
    assert "class '" not in text
    assert ts.systems.find("chua")[0] is ts.systems.Chua


def test_a_removed_capability_says_it_was_removed_rather_than_guessing():
    """``permutation_entropy`` was answered "Did you mean expansion_entropy?"."""
    from tsdynamics.errors import MovedInV6

    with pytest.raises(MovedInV6, match="entropy estimators"):
        ts.analysis.permutation_entropy  # noqa: B018


def test_no_runtime_message_claims_a_version_the_user_may_not_have():
    """``MovedInV6: ... moved in v6`` on a 5.4.0 install made a reader stop and check."""
    from tsdynamics.errors import MovedInV6

    with pytest.raises(MovedInV6) as err:
        ts.Box  # noqa: B018
    assert "in v6" not in str(err.value)


def test_an_sde_info_card_shows_the_noise():
    """``OrnsteinUhlenbeck.info`` printed only the drift and listed ``sigma`` as unused."""
    text = str(ts.systems.OrnsteinUhlenbeck().info)
    assert "dW" in text and "sigma" in text
    assert "dx/dt" not in text  # an SDE has no derivative


def test_lyapunov_from_data_commits_to_a_verdict_when_it_can():
    """``verdict`` was ``None`` on the estimator billed as the chaos test."""
    traj = ts.systems.Lorenz().run(final_time=150.0, dt=0.01, transient=20.0, ic=[1.0, 1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ts.analysis.lyapunov_from_data(traj["x"], dt=0.01)
    assert result.chaotic is True
    assert result.verdict and "chaotic" in result.verdict


def test_a_short_record_is_not_trusted_however_straight_the_fit():
    """2000 samples gave λ = 1.209 (truth 0.906) with R² = 0.9997 and ``trusted=True``."""
    traj = ts.systems.Lorenz().run(final_time=150.0, dt=0.01, transient=20.0, ic=[1.0, 1.0, 1.0])
    x = np.asarray(traj["x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        short = ts.analysis.lyapunov_from_data(ts.Trajectory(np.arange(2000) * 0.01, x[:2000]))
        long = ts.analysis.lyapunov_from_data(ts.Trajectory(np.arange(12000) * 0.01, x[:12000]))
    assert short.trusted is False and short.independent_windows < 10.0
    assert "too short" in str(short.verdict)
    assert long.trusted is True and long.independent_windows >= 10.0


def test_the_embedding_doors_agree_about_the_delay():
    """``optimal_delay(x)`` said 16 while ``embedding_dimension(x)`` reconstructed at 1."""
    traj = ts.systems.Lorenz().run(final_time=80.0, dt=0.01, transient=20.0, ic=[1.0, 1.0, 1.0])
    x = np.asarray(traj["x"])[:6000]
    tau = int(ts.analysis.optimal_delay(x))
    assert int(ts.analysis.embedding_dimension(x).meta["delay"]) == tau
    assert int(ts.analysis.embedding_dimension(x, method="fnn").meta["delay"]) == tau


def test_an_analysis_result_can_be_an_argument_to_the_next_analysis():
    """``embed(x, dimension=fnn_result)`` raised about "multivariate channels"."""
    traj = ts.systems.Lorenz().run(final_time=60.0, dt=0.01, transient=20.0, ic=[1.0, 1.0, 1.0])
    x = np.asarray(traj["x"])[:5000]
    dimension = ts.analysis.embedding_dimension(x, method="fnn")
    delay = ts.analysis.optimal_delay(x)
    embedded = ts.analysis.embed(x, dimension=dimension, delay=delay)
    assert embedded.values.shape[1] == int(dimension)


def test_find_answers_the_words_a_reader_actually_types():
    """``find("robust")`` and ``find("safety margin")`` both returned nothing."""
    for question in ("robust", "safety margin", "will it tip over"):
        names = {f.__name__ for f in ts.analysis.find(question)}
        assert names & {"resilience", "tipping_points"}, question


def test_the_zero_one_test_is_findable_from_the_data_it_accepts():
    """Its own summary says "a system or a measured observable"; find(array) hid it."""
    names = {f.__name__ for f in ts.analysis.find(np.zeros((64, 2)))}
    assert "zero_one_test" in names


def test_the_solver_keyword_and_the_record_use_one_word():
    """``run(solver=)`` was recorded as ``meta["method"]`` and documented as ``"RK45"``."""
    traj = ts.systems.Lorenz().run(final_time=1.0, ic=[1.0, 1.0, 1.0], solver="dop853")
    assert traj.meta["solver"] == "dop853"
    assert ts.ContinuousSystem._default_method == "rk45"
