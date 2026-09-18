"""First contact: what the library says in the first five minutes.

Six blind testers reached the library with no documentation beyond ``help()``
and whatever it told them.  Every test here is named after something one of them
*typed*, and asserts what they should have been told back.

Two things separate this file from the per-feature suites:

* it sweeps where the beta pass sampled — **every** catalogue map, **every**
  built-in SDE, **every** result class — because a first-contact message that is
  right for ``Logistic`` and wrong for the other twenty-five is still a
  first-contact defect;
* it asserts the *remedy line* resolves and runs, not merely that something was
  raised.  A library that hands back a line which does not run is worse than one
  that hands back nothing.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.errors import InvalidInputError, InvalidParameterError

# ---------------------------------------------------------------------------
# 1. A state handed to the constructor's parameter slot
# ---------------------------------------------------------------------------


def test_a_state_where_parameters_go_names_both_doors() -> None:
    """``Lorenz([1,1,1])`` raised ``TypeError: object is not iterable`` from ``dict()``.

    The commonest first line from anyone arriving with a ``u0``-first habit
    (DynamicalSystems.jl, ``solve_ivp``), five frames inside ``families/base.py``.
    """
    with pytest.raises(InvalidInputError) as err:
        ts.systems.Lorenz([1, 1, 1])
    text = str(err.value)
    assert "takes PARAMETERS, not a state" in text
    # It must name a parameter this class really declares...
    assert "sigma=10.0" in text
    # ...and the run that does what was meant.
    assert "run(ic=[1, 1, 1])" in text


def test_the_two_lines_a_state_in_the_wrong_slot_hands_back_both_run() -> None:
    """A remedy line that does not run is worse than no remedy line at all."""
    ic = [1.0, 1.0, 1.0]
    assert ts.systems.Lorenz().run(final_time=0.5, ic=ic).y.shape[1] == 3
    assert ts.systems.Lorenz(sigma=10.0).run(final_time=0.5, ic=ic).y.shape[1] == 3
    assert ts.systems.Lorenz(ic=ic).run(final_time=0.5).y.shape[1] == 3


@pytest.mark.parametrize(
    ("family", "base"),
    [
        ("ode", ts.ContinuousSystem),
        ("map", ts.DiscreteMap),
        ("dde", ts.DelaySystem),
        ("sde", ts.StochasticSystem),
    ],
)
def test_instantiating_a_family_base_points_at_the_subclass_contract(
    family: str, base: type
) -> None:
    """CPython's own message names the missing methods and stops there.

    Each family must instead say what that family *is*, show the skeleton, and
    point at ``help(ts.<Base>)`` — the door that keeps working when the skeleton
    is not what this author needs.
    """
    with pytest.raises(TypeError) as err:
        base()
    text = str(err.value)
    assert f"help(ts.{base.__name__})" in text, family
    assert "# the subclass contract" in text, family
    assert "cannot be instantiated" in text, family


def test_the_help_the_contract_error_points_at_resolves() -> None:
    """``help(ts.X)`` is only a remedy if ``ts.X`` is a name that resolves."""
    for name in ("ContinuousSystem", "DiscreteMap", "DelaySystem", "StochasticSystem"):
        assert getattr(ts, name).__doc__, name


# ---------------------------------------------------------------------------
# 3. Naming the channels of measured data
# ---------------------------------------------------------------------------


def test_measured_data_can_name_its_channels_at_the_constructor() -> None:
    """``variables=`` is what everyone types; it used to be a bare ``TypeError``.

    It is accepted rather than redirected to ``meta={"variables": ...}``: the
    names are *data about the data*, the constructor is where a user has them,
    and every downstream door (``traj["v"]``, plots, ``to_frame``) already reads
    them from one place.
    """
    t = np.linspace(0.0, 1.0, 10)
    y = np.column_stack([t, 2 * t, 3 * t])
    traj = ts.Trajectory(t, y, variables=("x", "y", "z"))
    assert traj.variables == ("x", "y", "z")
    np.testing.assert_array_equal(traj["y"], 2 * t)
    # and the names survive a column selection, which is what makes them useful
    assert traj["x", "z"].variables == ("x", "z")


def test_naming_the_wrong_number_of_channels_says_how_many_there_are() -> None:
    """Silently accepting two names for three columns is a wrong answer later."""
    t = np.linspace(0.0, 1.0, 10)
    y = np.column_stack([t, 2 * t, 3 * t])
    with pytest.raises(InvalidInputError, match="every state component"):
        ts.Trajectory(t, y, variables=("a", "b"))


# ---------------------------------------------------------------------------
# 4. An unknown solver, answered by the family that was asked
# ---------------------------------------------------------------------------

#: One system per family, with the horizon keyword that family's ``run`` binds.
_ONE_PER_FAMILY: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("ode", "Lorenz", {"final_time": 1.0, "ic": [1.0, 1.0, 1.0]}),
    ("dde", "MackeyGlass", {"final_time": 1.0}),
    ("sde", "OrnsteinUhlenbeck", {"final_time": 1.0, "ic": [0.1]}),
)

#: Kernels that belong to exactly one family.  Offering one of these to another
#: family is the defect: ``solver='LSODA'`` on a *deterministic* flow used to
#: dump 26 kernels alphabetically, ``milstein`` and ``euler_maruyama`` included.
_FOREIGN_KERNELS = {
    "ode": ("milstein", "euler_maruyama"),
    "dde": ("milstein", "euler_maruyama", "bdf", "rosenbrock", "trbdf2"),
    "sde": ("rk45", "dop853", "bdf"),
}


@pytest.mark.parametrize(("family", "system", "run_kw"), _ONE_PER_FAMILY)
def test_an_unknown_solver_is_refused_by_name_and_by_family(
    family: str, system: str, run_kw: dict[str, Any]
) -> None:
    """``solver='LSODA'`` is a SciPy/v2 name with no engine kernel, on any family."""
    with pytest.raises(InvalidParameterError) as err:
        getattr(ts.systems, system)().run(solver="LSODA", **run_kw)
    text = str(err.value)
    assert "unknown solver 'LSODA'" in text, family
    for foreign in _FOREIGN_KERNELS[family]:
        assert f"'{foreign}'" not in text, (family, foreign, text)


def test_an_unknown_solver_suggests_a_kernel_this_family_can_actually_drive() -> None:
    """A suggestion is a promise: the offered name has to run on the caller's system."""
    with pytest.raises(InvalidParameterError) as err:
        ts.systems.Lorenz().run(final_time=1.0, ic=[1.0, 1.0, 1.0], solver="LSODA")
    assert "'bdf'" in str(err.value)
    # ...and 'bdf' is a kernel this system really drives.
    assert ts.systems.Lorenz().run(final_time=1.0, ic=[1.0, 1.0, 1.0], solver="bdf").y.shape[1] == 3


def test_a_transposed_sde_scheme_gets_the_nearest_match_like_every_other_door() -> None:
    """``solver='milstien'`` got a bare listing where every sibling said "did you mean"."""
    with pytest.raises(InvalidParameterError) as err:
        ts.systems.OrnsteinUhlenbeck().run(final_time=1.0, ic=[0.1], solver="milstien")
    assert "'milstein'" in str(err.value)


def test_a_deterministic_kernel_on_an_sde_is_answered_with_the_mathematics() -> None:
    """``rk45`` on an SDE is not a typo — it is the right word for other dynamics."""
    with pytest.raises(InvalidParameterError) as err:
        ts.systems.OrnsteinUhlenbeck().run(final_time=1.0, ic=[0.1], solver="rk45")
    text = str(err.value)
    assert "dW" in text and "euler_maruyama" in text


def test_the_stepping_door_refuses_the_same_names_run_does() -> None:
    """``reinit(solver=)`` accepted an SDE kernel and failed one call later, in ``step``."""
    with pytest.raises(InvalidParameterError):
        ts.systems.Lorenz().reinit([1.0, 1.0, 1.0], solver="milstein")


# ---------------------------------------------------------------------------
# 5. A map's run starts where it was told to start
# ---------------------------------------------------------------------------


def _map_entries() -> list[Any]:
    return sorted(registry.all_systems(family="map"), key=lambda e: e.name)


@pytest.mark.parametrize("entry", _map_entries(), ids=lambda e: e.name)
@pytest.mark.parametrize("backend", ["jit", "interp"])
def test_a_map_run_returns_the_state_it_started_from(entry: Any, backend: str) -> None:
    """``t[n]`` must label ``x_n``, on every catalogue map and every backend.

    Asked for 4 steps from 0.1, a map returned 4 numbers starting at the
    *second*, so ``t = 0`` was labelled with the first iterate and every cobweb
    started one step late.  Flows never did this.

    The oracle is the class's own pure-Python ``_step`` applied n times — an
    independent computation, not the same engine call under another name.
    """
    system = entry.cls()
    ic = np.asarray(system._resolve_ic(None), dtype=float)
    expected = [ic.copy()]
    state = ic.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(4):
            state = np.asarray(system._step(state, *system.params.as_tuple()), dtype=float)
            expected.append(state.copy())
    if not np.all(np.isfinite(expected[-1])):
        pytest.skip(f"{entry.name} leaves the attractor from its resolved ic")

    run = system.run(steps=4, ic=ic, backend=backend)
    assert run.y.shape[0] == 5, "n steps from an ic is n+1 states"
    np.testing.assert_array_equal(run.t, np.arange(5.0))
    for n, want in enumerate(expected):
        np.testing.assert_allclose(
            run.y[n], want, rtol=1e-8, atol=1e-10, err_msg=f"{entry.name}: y[{n}] is not x_{n}"
        )


def test_a_user_defined_map_starts_where_it_was_told_to_start() -> None:
    """The catalogue is not the contract — a map someone writes gets the same one."""

    class Doubler(ts.DiscreteMap):
        params = {"a": 2.0}
        dim = 1
        variables = ("x",)

        @staticmethod
        def _step(X, a):  # type: ignore[no-untyped-def]
            return np.array([a * X[0]])

    run = Doubler().run(steps=4, ic=[1.0])
    np.testing.assert_array_equal(run.y.ravel(), [1.0, 2.0, 4.0, 8.0, 16.0])
    np.testing.assert_array_equal(run.t, np.arange(5.0))


def test_a_maps_time_axis_keeps_counting_through_a_transient() -> None:
    """``t[n]`` labels the true iterate index, so a discarded transient shows in it."""

    class Doubler(ts.DiscreteMap):
        params = {"a": 2.0}
        dim = 1
        variables = ("x",)

        @staticmethod
        def _step(X, a):  # type: ignore[no-untyped-def]
            return np.array([a * X[0]])

    run = Doubler().run(steps=3, ic=[1.0], transient=2)
    np.testing.assert_array_equal(run.t, [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(run.y.ravel(), [4.0, 8.0, 16.0, 32.0])


# ---------------------------------------------------------------------------
# 6. The info card of an SDE shows the noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry", sorted(registry.all_systems(family="sde"), key=lambda e: e.name), ids=lambda e: e.name
)
def test_an_sde_info_card_shows_the_diffusion(entry: Any) -> None:
    """Without it ``sigma`` reads as an unused parameter — on the one family where
    the noise *is* the point."""
    card = str(entry.cls().info)
    assert "dW" in card, card
    assert " dt + " in card, card


def test_a_multi_channel_sde_names_which_wiener_drives_which_component() -> None:
    """Diagonal Itô means one independent ``dW`` per component; the card must say so."""

    class TwoNoise(ts.StochasticSystem):
        params = {"k": 1.5, "s1": 0.2, "s2": 0.7}
        dim = 2
        variables = ("x", "y")

        @staticmethod
        def _drift(u, t, *, k, s1, s2):  # type: ignore[no-untyped-def]
            return [-k * u(0), u(0) - u(1)]

        @staticmethod
        def _diffusion(u, t, *, k, s1, s2):  # type: ignore[no-untyped-def]
            return [s1, s2 * u(1)]

    card = str(TwoNoise().info)
    assert "dW_0" in card and "dW_1" in card, card
    assert "s1" in card and "s2" in card, card


def test_a_deterministic_flow_grows_no_noise_term() -> None:
    """The card reads the equations back; an ODE has no ``dW`` to read."""
    assert "dW" not in str(ts.systems.Lorenz().info)


# ---------------------------------------------------------------------------
# 7. A result is usable wherever its number is
# ---------------------------------------------------------------------------


def test_an_estimated_dimension_can_be_a_length_a_range_and_a_slice() -> None:
    """``embed`` survived only because it calls ``int()``; Python asks ``__index__``.

    Measured: ``m`` printed ``m = 4``, ``int(m)`` gave ``4``, and ``range(m)``
    raised *"'EmbeddingDimension' object cannot be interpreted as an integer"*.
    """
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0.0, 400.0, 4000)) + 0.05 * rng.normal(size=4000)
    m = ts.analysis.embedding_dimension(x, delay=8)
    assert list(range(m)) == list(range(int(m)))
    assert np.zeros(m).shape == (int(m),)
    assert np.arange(10)[:m].size == int(m)


def test_the_two_spellings_of_printing_a_count_agree() -> None:
    """``"%d" % m`` printed ``4`` while ``f"{m:d}"`` raised, naming ``float``.

    ``%d`` asks ``__index__``; the f-string spec went through the float the
    result reports, and ``format(4.0, 'd')`` is a ``ValueError`` about a type
    the caller never typed.
    """
    tau = ts.analysis.optimal_delay(np.sin(np.linspace(0.0, 400.0, 4000)))
    assert f"{tau:d}" == "%d" % tau == str(int(tau))  # noqa: UP031 — the two spellings ARE the test

    rng = np.random.default_rng(3)
    x = np.sin(np.linspace(0.0, 400.0, 4000)) + 0.05 * rng.normal(size=4000)
    m = ts.analysis.embedding_dimension(x, delay=8)
    assert f"{m:d}" == "%d" % m == str(int(m))  # noqa: UP031 — ditto
    assert f"{m:.2f}" == f"{float(m):.2f}"  # the float spellings still float


def test_the_chaining_idiom_the_docstrings_teach_works_on_both_keywords() -> None:
    """``embed(x, dimension=…, delay=…)`` — both fed by another analysis's answer."""
    rng = np.random.default_rng(1)
    x = np.sin(np.linspace(0.0, 400.0, 4000)) + 0.05 * rng.normal(size=4000)
    dimension = ts.analysis.embedding_dimension(x, delay=8)
    delay = ts.analysis.optimal_delay(x)
    embedded = ts.analysis.embed(x, dimension=dimension, delay=delay)
    assert embedded.values.shape[1] == int(dimension)


def test_a_measurement_that_is_not_a_whole_number_refuses_to_be_an_index() -> None:
    """``range(D)`` on :math:`D = 2.06` is a units confusion, not a request for 2.

    Python's ``__index__`` contract is *lossless* conversion, so refusing is the
    specification here — and truncating a fractal dimension into a count is the
    wrong-answer-with-a-straight-face this library exists to refuse.
    """
    from tsdynamics.analysis.results import ScalarResult

    with pytest.raises(TypeError) as err:
        range(ScalarResult(2.5))  # noqa: B305
    text = str(err.value)
    assert "not a whole number" in text
    assert "round(result)   # 2" in text
    assert "float(result)   # 2.5" in text


def test_a_whole_number_indexes_however_it_is_spelt() -> None:
    """2.0 is a whole number; the guard is on the value, not on the class."""
    from tsdynamics.analysis.results import CountResult, ScalarResult

    assert list(range(ScalarResult(3.0))) == [0, 1, 2]
    assert list(range(CountResult(3))) == [0, 1, 2]


def test_a_result_that_is_not_a_number_says_what_it_carries_instead() -> None:
    """The refusal names the next thing to type, like ``__array__``'s does."""
    signal = np.sin(np.linspace(0.0, 50.0, 400))[:, None]
    rqa = ts.analysis.rqa(signal, recurrence_rate=0.05)
    with pytest.raises(TypeError) as err:
        range(rqa)  # noqa: B305
    text = str(err.value)
    assert "RQAResult is not a number" in text
    assert "recurrence_rate" in text


def test_a_collection_asked_for_a_count_is_pointed_at_its_length() -> None:
    """``range(result)`` on a set is almost certainly reaching for ``len``."""
    points = ts.analysis.fixed_points(ts.systems.Henon())
    with pytest.raises(TypeError) as err:
        range(points)  # noqa: B305
    text = str(err.value)
    assert "len(result)" in text
    assert str(len(points)) in text


# ---------------------------------------------------------------------------
# 8. Every result's table carries the result's answer
# ---------------------------------------------------------------------------


def _fixtures() -> dict[str, Any]:
    from _result_fixtures import build

    return build()


@pytest.mark.parametrize("name", sorted(_fixtures()))
def test_a_result_that_is_a_number_puts_that_number_in_its_table(name: str) -> None:
    """``optimal_delay(x).to_frame()`` was one column reading ``unit / samples``.

    The unit of a number that was not in the table — an export door that dropped
    the answer.  ``CountResult`` subclasses ``int`` and carries its count as a
    *property*, so its inherited dataclass fields are ``('meta',)`` and the
    field-driven row builder found nothing to tabulate.
    """
    pytest.importorskip("pandas")
    result = _fixtures()[name]
    answer = result._as_number()
    if answer is None:
        pytest.skip(f"{name} is not a number")
    frame = result.to_frame()
    cells = [
        float(v)
        for column in frame.columns
        for v in frame[column].to_numpy()
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
    ]
    assert any(np.isclose(v, answer, rtol=1e-9, atol=0.0) for v in cells), (
        f"{name}.to_frame() has columns {list(frame.columns)} and none of them "
        f"holds the answer it prints ({answer!r})"
    )


def test_a_measured_delay_exports_the_delay_and_not_only_its_unit() -> None:
    """The live door, not a fixture: the shape of the defect the engineer reported."""
    pytest.importorskip("pandas")
    rng = np.random.default_rng(2)
    x = np.sin(np.linspace(0.0, 400.0, 4000)) + 0.05 * rng.normal(size=4000)
    tau = ts.analysis.optimal_delay(x)
    frame = tau.to_frame()
    assert "value" in frame.columns, list(frame.columns)
    assert int(frame["value"].iloc[0]) == int(tau)


@pytest.mark.parametrize("name", sorted(_fixtures()))
def test_no_result_exports_a_table_of_labels_with_no_measurement(name: str) -> None:
    """A frame that has columns must have a measurement in at least one of them.

    An empty frame is honest ("nothing was found"); a frame of column headings
    over nothing is the export-door failure — it reads as a bug in the caller's
    own code.
    """
    pytest.importorskip("pandas")
    frame = _fixtures()[name].to_frame()
    if len(frame.columns) == 0:
        return
    assert not all(frame[c].isna().all() for c in frame.columns), (
        f"{name}.to_frame() is {list(frame.columns)} and every column is empty"
    )
