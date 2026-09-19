"""One spelling, no leaks, units at the door — the v6 round-6 gates.

Each test here is the inverse of a defect the round-6 audit measured by hand.
The comment above each names what it was, so a future reader can tell a
*contract* from a *preference*.
"""

from __future__ import annotations

import copy as copy_module

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.data.trajectory import MinMax, Neighbors, Trajectory, as_trajectory
from tsdynamics.errors import InvalidInputError, InvalidParameterError

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _lorenz_traj() -> Trajectory:
    return ts.systems.Lorenz().run(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# 1. The bracket: names select columns, numbers select rows — one rule each
# ---------------------------------------------------------------------------


class TestTheBracketHasOneGrammar:
    """``traj["x", "z"]`` is THE column spelling and it returns a Trajectory."""

    def test_one_name_is_the_bare_series(self) -> None:
        tr = _lorenz_traj()
        assert isinstance(tr["x"], np.ndarray)
        assert tr["x"].shape == (tr.n_steps,)

    def test_several_names_give_back_a_trajectory(self) -> None:
        tr = _lorenz_traj()
        xz = tr["x", "z"]
        assert isinstance(xz, Trajectory)
        assert xz.shape == (tr.n_steps, 2)
        np.testing.assert_array_equal(xz.t, tr.t)
        np.testing.assert_array_equal(np.asarray(xz), tr.y[:, [0, 2]])

    def test_the_selection_names_its_own_columns(self) -> None:
        """The measured silent-wrong-answer: ``sel('x','z')['y']`` returned ``z``."""
        # pandas is not a dependency; `to_frame()` names it when absent.
        pytest.importorskip("pandas")
        tr = _lorenz_traj()
        xz = tr["x", "z"]
        assert xz.variables == ("x", "z")
        np.testing.assert_array_equal(xz["z"], tr["z"])
        with pytest.raises(KeyError, match="Declared variables"):
            xz["y"]  # the inner system's name must NOT resolve here
        assert list(xz.to_frame().columns) == ["x", "z"]

    def test_the_selection_composes(self) -> None:
        tr = _lorenz_traj()
        assert tr["x", "z"][2:].shape == (tr.n_steps - 2, 2)

    def test_the_list_spelling_is_gone_and_names_the_one_that_works(self) -> None:
        tr = _lorenz_traj()
        with pytest.raises(InvalidInputError) as err:
            tr[["x", "z"]]
        assert "traj['x', 'z']" in str(err.value)

    def test_numbers_still_select_rows(self) -> None:
        tr = _lorenz_traj()
        for key in (0, slice(2, 5), [0, 2], np.array([True] + [False] * (tr.n_steps - 1))):
            picked = tr[key]
            assert isinstance(picked, Trajectory), key
        assert tr[[0, 2]].shape == (2, 3)

    def test_a_joint_index_is_refused_not_leaked_to_numpy(self) -> None:
        """``traj[:, 0]`` used to surface a raw numpy ``IndexError``."""
        tr = _lorenz_traj()
        for key in ((slice(None), 0), ("x", 0)):
            with pytest.raises(InvalidInputError) as err:
                tr[key]
            assert "traj[10:50]" in str(err.value)

    def test_sel_is_gone_and_the_error_names_the_bracket(self) -> None:
        tr = _lorenz_traj()
        assert not hasattr(tr, "sel")
        with pytest.raises(AttributeError, match=r"traj\['x', 'z'\]"):
            tr.sel  # noqa: B018


class TestEveryTrajectoryNamesEveryComponent:
    """``variables`` is never ``None`` — measured data named its columns in
    ``to_frame`` and on its plot, then refused ``traj["y0"]``."""

    def test_measured_data_answers_its_generated_names(self) -> None:
        # pandas is not a dependency; `to_frame()` names it when absent.
        pytest.importorskip("pandas")
        bare = as_trajectory(np.random.default_rng(0).random((20, 2)))
        assert bare.variables == ("y0", "y1")
        assert list(bare.to_frame().columns) == ["y0", "y1"]
        assert bare["y0"].shape == (20,)
        assert bare["y0", "y1"].shape == (20, 2)

    def test_a_declared_system_still_wins(self) -> None:
        assert _lorenz_traj().variables == ("x", "y", "z")

    def test_a_generated_name_tuple_reaches_projection(self) -> None:
        """``ProjectedSystem`` read ``type(system).variables`` — the CLASS slot —
        so the 5 built-ins that generate their names were refused by name."""
        l96 = ts.systems.Lorenz96(N=5)
        proj = ts.derived.ProjectedSystem(l96, ["y0", "y2"])
        assert proj.variables == ("y0", "y2")


# ---------------------------------------------------------------------------
# 2. No internal leaks through a return value
# ---------------------------------------------------------------------------


class TestNeighborsAnswersWithStates:
    """It handed back ``cKDTree.query``'s raw pair, rank-polymorphic in ``k``."""

    def test_the_answer_is_the_neighbouring_states(self) -> None:
        tr = _lorenz_traj()
        near = tr.neighbors([1.0, 1.0, 1.0], k=3)
        assert isinstance(near, Neighbors)
        assert near.state.shape == (3, tr.dim)
        np.testing.assert_array_equal(near.state, tr.y[near.index])
        np.testing.assert_array_equal(np.asarray(near), near.state)

    @pytest.mark.parametrize("k", [1, 2, 5])
    def test_the_shape_does_not_depend_on_k(self, k: int) -> None:
        tr = _lorenz_traj()
        near = tr.neighbors([1.0, 1.0, 1.0], k=k)
        assert near.distance.shape == (k,)
        assert near.index.shape == (k,)
        assert near.state.shape == (k, tr.dim)
        assert near.index[0] == near.index[0]  # indexable at every k

    def test_many_queries_keep_the_query_axis(self) -> None:
        tr = _lorenz_traj()
        near = tr.neighbors(tr.y[:4], k=2)
        assert near.distance.shape == (4, 2)
        assert near.state.shape == (4, 2, tr.dim)

    def test_it_still_unpacks_as_the_old_pair(self) -> None:
        tr = _lorenz_traj()
        distance, index = tr.neighbors(tr.y[0], k=2)
        np.testing.assert_array_equal(distance, tr.neighbors(tr.y[0], k=2).distance)
        np.testing.assert_array_equal(index, tr.neighbors(tr.y[0], k=2).index)

    def test_more_neighbours_than_points_reports_nan_not_an_indexerror(self) -> None:
        tr = Trajectory(np.arange(2.0), np.zeros((2, 2)))
        near = tr.neighbors([0.0, 0.0], k=4)
        assert np.isnan(near.state[2:]).all()

    def test_the_repr_states_the_answer(self) -> None:
        text = repr(_lorenz_traj().neighbors([1.0, 1.0, 1.0], k=2))
        assert "2 nearest states" in text
        assert "near.state" in text


def test_minmax_names_its_halves_and_still_unpacks() -> None:
    tr = _lorenz_traj()
    extent = tr.minmax()
    assert isinstance(extent, MinMax)
    lo, hi = extent
    np.testing.assert_array_equal(lo, extent.lo)
    np.testing.assert_array_equal(hi, extent.hi)


class TestTheBatchIsAMeasurementNotAWorkspace:
    """``TrajectoryBatch`` was a ``list`` subclass: ten mutation verbs on the
    tab surface of a result, and ``batch.append("x")`` corrupted ``.final``."""

    @staticmethod
    def _batch() -> object:
        band = ts.systems.Lorenz().ensemble([[1.0, 1.0, 1.0], [1.0, 1.0, 1.001]])
        return band.run(final_time=1.0, dt=0.5)

    def test_no_mutator_is_reachable(self) -> None:
        batch = self._batch()
        for verb in ("append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse"):
            assert not hasattr(batch, verb), verb

    def test_it_is_still_a_complete_sequence(self) -> None:
        batch = self._batch()
        assert len(batch) == 2  # type: ignore[arg-type]
        assert isinstance(list(batch)[0], Trajectory)  # type: ignore[call-overload]
        assert batch[0] is list(batch)[0]  # type: ignore[index,call-overload]

    def test_the_batch_view_is_numeric(self) -> None:
        batch = self._batch()
        assert batch.final.shape == (2, 3)  # type: ignore[attr-defined]
        assert batch.y.shape[0] == 2  # type: ignore[attr-defined]
        assert np.asarray(batch).ndim == 3


# ---------------------------------------------------------------------------
# 3. Every member tells the truth
# ---------------------------------------------------------------------------


def test_a_tangent_systems_run_records_the_exponents_converging() -> None:
    """It inherited ``run`` and raised ``NotImplementedError('')`` while
    ``hasattr`` said yes."""
    tang = ts.derived.TangentSystem(ts.systems.Henon(), k=2)
    conv = tang.run(steps=300, ic=[0.1, 0.1])
    assert isinstance(conv, Trajectory)
    assert conv.variables == ("lambda1", "lambda2")
    assert conv["lambda1"][-1] > 0.0  # Hénon is chaotic


def test_the_map_ic_error_hands_back_a_line_that_runs() -> None:
    """``_ic_example`` probed ``hasattr(self, "iterate")`` — a name v6 REMOVED —
    so all 26 maps were told to call ``run(final_time=...)``, which they refuse."""
    with pytest.raises(InvalidInputError) as err:
        ts.systems.Henon().run(steps=10, ic=[0.1])
    line = str(err.value).splitlines()[-1].strip()
    assert line.startswith("Henon().run(steps=")
    assert ts.systems.Henon().run(steps=1000, ic=[1.0, 1.0]).shape == (1001, 2)


def test_a_wrapper_answers_for_its_own_keywords() -> None:
    """``pmap.run(nonsense_kw=1)`` was reported as ``Rossler.reinit()``."""
    pmap = ts.systems.Rossler().poincare("y", 0.0)
    with pytest.raises(InvalidParameterError, match=r"PoincareMap\.run\(\)"):
        pmap.run(steps=3, nonsense_kw=1)


def test_a_wrapper_teaches_a_retired_verb_like_a_system_does() -> None:
    pmap = ts.systems.Rossler().poincare("y", 0.0)
    with pytest.raises(AttributeError, match="run is the one trajectory verb"):
        pmap.integrate  # noqa: B018


def test_the_poincare_signature_prints_no_memory_address() -> None:
    import inspect

    text = str(inspect.signature(ts.systems.Lorenz().poincare))
    assert "object at 0x" not in text
    assert "= <unset>" in text


def test_a_map_reports_its_clock_as_a_count() -> None:
    henon = ts.systems.Henon()
    henon.reinit([0.1, 0.1])
    henon.step(7)
    assert isinstance(henon.time(), int)
    assert henon.time() == 7


# ---------------------------------------------------------------------------
# 4. The families agree where no mathematics separates them
# ---------------------------------------------------------------------------


def test_every_continuous_family_steps_by_the_same_bare_amount() -> None:
    """The DDE stepped by 0.1 where the ODE and the SDE stepped by 0.01."""
    defaults = {
        cls.__name__: cls._default_step_dt
        for cls in (
            ts.ContinuousSystem,
            ts.DelaySystem,
            ts.StochasticSystem,
        )
    }
    assert set(defaults.values()) == {0.01}, defaults


@pytest.mark.parametrize(
    ("factory", "extra"),
    [
        (ts.systems.Lorenz, {}),
        (ts.systems.MackeyGlass, {}),
    ],
)
def test_reinit_takes_backend_where_a_stepper_can_honour_it(factory, extra) -> None:
    system = factory()
    system.reinit(backend="interp", **extra)
    assert np.all(np.isfinite(system.step()))


@pytest.mark.parametrize("factory", [ts.systems.Henon, ts.systems.OrnsteinUhlenbeck])
def test_reinit_refuses_backend_by_name_where_it_cannot(factory) -> None:
    """Accepting an inert keyword is the silent-wrong-answer defect; refusing
    with the mathematical reason is the library's own rule."""
    with pytest.raises(InvalidParameterError) as err:
        factory().reinit(backend="interp")
    assert "no engine to choose" in str(err.value)


@pytest.mark.parametrize(
    "factory",
    [ts.systems.Lorenz, ts.systems.Henon, ts.systems.MackeyGlass, ts.systems.OrnsteinUhlenbeck],
)
def test_every_family_refuses_an_unknown_reinit_keyword_by_name(factory) -> None:
    """A map's ``reinit`` raised a bare ``TypeError``, so ``except
    InvalidParameterError`` caught the typo on a flow and missed it on a map."""
    with pytest.raises(InvalidParameterError, match="nonsense_kw"):
        factory().reinit(nonsense_kw=1)


@pytest.mark.parametrize(
    ("factory", "word"),
    [
        (ts.systems.Lorenz, "final_time"),
        (ts.systems.Henon, "steps"),
        (ts.systems.MackeyGlass, "final_time"),
        (ts.systems.OrnsteinUhlenbeck, "final_time"),
    ],
)
def test_run_states_the_unit_of_its_horizon_and_its_transient(factory, word) -> None:
    import inspect

    doc = inspect.getdoc(type(factory()).run) or ""
    assert f"{word} : " in doc
    unit = "iterations" if word == "steps" else "time units"
    horizon_block = doc.split(f"{word} : ", 1)[1].split("\n        ", 1)[0]
    assert unit in horizon_block or unit in doc.split(f"{word} : ", 1)[1][:400]
    transient_block = doc.split("transient : ", 1)[1][:400]
    assert unit in transient_block


def test_the_dde_history_door_states_the_sign_of_s() -> None:
    import inspect

    doc = inspect.getdoc(ts.DelaySystem.run) or ""
    history = doc.split("history : ", 1)[1][:600]
    assert "s ≤ 0" in history
    assert "never called with ``s > 0``" in history


def test_no_removed_verb_survives_in_a_family_docstring() -> None:
    """A rename that leaves its readers behind goes quiet; these are read by a
    user through ``help()``, so they cannot."""
    import inspect

    dead = (":meth:`integrate`", ":meth:`iterate`", ":meth:`trajectory`", "ic_generator`")
    for cls in (ts.ContinuousSystem, ts.DiscreteMap, ts.DelaySystem, ts.StochasticSystem):
        for verb in ("run", "step", "reinit"):
            doc = inspect.getdoc(getattr(cls, verb, None)) or ""
            for name in dead:
                assert name not in doc, f"{cls.__name__}.{verb} names {name}"


# ---------------------------------------------------------------------------
# 5. Copying must not change what the copy knows about its IC
# ---------------------------------------------------------------------------


def test_copying_does_not_promote_an_auto_drawn_ic_to_user_chosen() -> None:
    """``_ic_explicit`` is what disables the random-IC divergence retry, so a
    ``with_params`` sweep silently turned the retry off for every value."""
    system = ts.systems.LorenzBounded()
    system.run(final_time=0.2, dt=0.1)  # auto-draws an IC
    assert system.__dict__["_ic_explicit"] is False
    for clone in (
        system.copy(),
        system.with_params(),
        copy_module.copy(system),
        copy_module.deepcopy(system),
    ):
        assert clone.__dict__["_ic_explicit"] is False

    chosen = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
    for clone in (chosen.copy(), chosen.with_params(), copy_module.copy(chosen)):
        assert clone.__dict__["_ic_explicit"] is True


# ---------------------------------------------------------------------------
# 6. WrappedSystem speaks the family-uniform vocabulary
# ---------------------------------------------------------------------------


class TestWrappedSystemSpeaksTheSharedWords:
    @staticmethod
    def _flow() -> ts.WrappedSystem:
        return ts.WrappedSystem(
            lambda u, dt: [u[0] * np.exp(0.5 * dt)], dim=1, family="ode", default_dt=0.1
        )

    def test_the_initial_state_is_spelled_ic(self) -> None:
        w = ts.WrappedSystem(lambda u, n: u, dim=1, ic=[0.5])
        np.testing.assert_array_equal(w.state(), [0.5])
        # ``initial=`` was this constructor's private word for the one thing every
        # other family calls ``ic``; it is refused BY NAME, not quietly accepted.
        with pytest.raises(InvalidParameterError, match="spelled ic="):
            ts.WrappedSystem(lambda u, n: u, dim=1, initial=[0.5])

    def test_transient_is_time_on_a_continuous_wrapper(self) -> None:
        """It counted SAMPLES, and refused a float outright."""
        traj = self._flow().run(final_time=1.0, dt=0.1, transient=0.5)
        assert traj.t[0] == pytest.approx(0.6)

    def test_transient_is_iterations_on_a_discrete_wrapper(self) -> None:
        w = ts.WrappedSystem(lambda u, n: u * 0.9, dim=1, family="map", ic=[1.0])
        assert w.run(5, transient=2).t[0] == pytest.approx(3.0)
        with pytest.raises(InvalidParameterError, match="whole number of iterations"):
            w.run(5, transient=0.5)

    def test_dim_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            self._flow().dim = 7  # type: ignore[misc]

    def test_it_records_the_same_provenance_every_family_does(self) -> None:
        meta = self._flow().run(final_time=1.0, dt=0.1, ic=[2.0]).meta
        assert {"system", "family", "dt", "ic", "transient", "tsdynamics"} <= set(meta)
        assert meta["ic"] == [2.0]
