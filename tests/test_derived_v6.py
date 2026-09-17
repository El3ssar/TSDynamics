"""The v6 derived-wrapper contract, pinned (``CONTRACT.md`` §3.2, §3.5).

Five defects are reproduced-and-fixed here, every one measured on the v5 tree:

* ``hasattr(pm, "plot")`` was ``False`` on all five wrappers while
  ``__plot_spec__`` was ``True`` — the seam existed and the verb did not;
* ``_DerivedSystem.params`` blindly forwarded ``self.system.params``, so
  ``PoincareMap(wrapped).run(5)`` marched correctly and then died with a raw
  ``AttributeError``;
* ``isinstance(Ensemble(...), System)`` was ``False`` while the other four were
  ``True``;
* four of five reprs printed only the inner system, so two sections on
  *different planes* reprred identically;
* ``pmap.run(steps=5)`` twice returned different data.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.derived import (
    Ensemble,
    PoincareMap,
    ProjectedSystem,
    StroboscopicMap,
    TangentSystem,
)
from tsdynamics.errors import InvalidParameterError
from tsdynamics.families.protocol import System

#: Everything a derived wrapper must answer, because a system answers it.
EXPECTED_CORE = {"dim", "params", "variables", "run", "step", "state", "time", "reinit", "plot"}


def wrappers() -> dict[str, object]:
    """One live instance of each of the five, on a pinned initial condition."""
    lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
    ros = ts.systems.Rossler(ic=[1.0, 1.0, 1.0])
    duf = ts.systems.Duffing(ic=[0.5, 0.0, 0.0])
    return {
        "PoincareMap": ros.poincare("y", 0.0),
        "StroboscopicMap": duf.poincare(period=4.487989505128276),
        "TangentSystem": TangentSystem(lor, 2),
        "ProjectedSystem": ProjectedSystem(lor, [0, 2]),
        "Ensemble": lor.ensemble([[1.0, 1.0, 1.0], [1.0, 1.0, 1.001]]),
    }


@pytest.mark.parametrize("name", sorted(wrappers()))
def test_every_wrapper_answers_the_core(name):
    wrapper = wrappers()[name]
    missing = sorted(n for n in EXPECTED_CORE if not hasattr(wrapper, n))
    assert missing == []


@pytest.mark.parametrize("name", sorted(wrappers()))
def test_every_wrapper_is_a_system(name):
    assert isinstance(wrappers()[name], System)


@pytest.mark.parametrize("name", sorted(wrappers()))
def test_no_wrapper_still_answers_the_old_verbs(name):
    wrapper = wrappers()[name]
    assert not hasattr(wrapper, "trajectory")
    assert not hasattr(wrapper, "to_plot_spec")


def test_the_five_reprs_name_their_defining_datum():
    got = {name: repr(w) for name, w in wrappers().items()}
    assert got["PoincareMap"] == "PoincareMap(Rossler, plane=y = 0 up)"
    assert got["StroboscopicMap"] == "StroboscopicMap(Duffing, period=4.48799)"
    assert got["TangentSystem"] == "TangentSystem(Lorenz, k=2)"
    assert got["ProjectedSystem"] == "ProjectedSystem(Lorenz, x, z)"
    assert got["Ensemble"] == "Ensemble(Lorenz, m=2)"


def test_two_sections_on_different_planes_do_not_repr_identically():
    ros = ts.systems.Rossler(ic=[1.0, 1.0, 1.0])
    assert repr(ros.poincare("y", 0.0)) != repr(ros.poincare("x", 1.0))


def test_a_wrapped_system_does_not_die_on_params():
    # PoincareMap(wrapped).params used to raise a raw AttributeError from
    # derived/_base.py, *after* the march had already succeeded.
    inner = ts.WrappedSystem(
        lambda u, dt: u + dt * np.array([-u[1], u[0]]), dim=2, family="ode", initial=[1.0, 0.0]
    )
    pmap = PoincareMap(inner, ("y0", 0.0))
    assert dict(pmap.params) == {}


class TestPoincareAbsorbsStroboscope:
    """§3.2 — one verb, two return types, selected by which keyword you gave."""

    def test_a_plane_gives_a_poincare_map(self):
        assert isinstance(ts.systems.Rossler().poincare("y", 0.0), PoincareMap)

    def test_a_period_gives_a_stroboscopic_map(self):
        smap = ts.systems.Duffing().poincare(period=4.488)
        assert isinstance(smap, StroboscopicMap)
        assert smap.period == pytest.approx(4.488)

    def test_a_forced_system_infers_its_own_period(self):
        smap = ts.systems.Duffing().poincare()
        assert isinstance(smap, StroboscopicMap)
        assert smap.period == pytest.approx(2 * np.pi / ts.systems.Duffing().omega)

    def test_an_autonomous_flow_falls_back_to_choosing_a_plane(self):
        assert isinstance(ts.systems.Rossler().poincare(), PoincareMap)

    def test_both_at_once_is_refused_and_names_the_choice(self):
        with pytest.raises(InvalidParameterError) as err:
            ts.systems.Duffing().poincare("z", 0.0, period=4.488)
        text = str(err.value)
        assert "not both" in text
        assert "poincare(period=4.488)" in text

    def test_the_strobe_samples_the_forcing_period(self):
        smap = ts.systems.Duffing(ic=[0.5, 0.0, 0.0]).poincare(period=4.488)
        times = smap.run(4).t
        assert np.allclose(np.diff(times), 4.488)


class TestEnsembleIsOneNoun:
    """§3.2 — ``sys.ensemble(states)`` returns a system; ``.run()`` a batch."""

    def test_ensemble_returns_a_system_not_an_array(self):
        band = ts.systems.Lorenz().ensemble([[1.0, 1.0, 1.0], [1.0, 1.0, 1.001]])
        assert isinstance(band, Ensemble)
        assert band.size == 2

    def test_run_returns_a_batch_whose_final_is_the_old_return_value(self):
        band = ts.systems.Lorenz().ensemble([[1.0, 1.0, 1.0], [1.0, 1.0, 1.001]])
        batch = band.run(final_time=1.0, dt=0.5)
        assert len(batch) == 2
        assert batch.final.shape == (2, 3)
        assert np.allclose(batch.final[0], batch[0].y[-1])

    def test_the_batch_is_still_a_list_of_trajectories(self):
        band = ts.systems.Lorenz().ensemble([[1.0, 1.0, 1.0]])
        batch = band.run(final_time=1.0, dt=0.5)
        assert isinstance(batch, list)
        assert isinstance(batch[0], ts.Trajectory)

    def test_an_sde_batch_seeds_each_member_by_index(self):
        # The engine's parallel-equals-serial contract: member i depends only on
        # seed_for(seed, i).  One shared seed would give every member the SAME path.
        band = ts.systems.OrnsteinUhlenbeck().ensemble([[1.0], [1.0]])
        batch = band.run(final_time=1.0, dt=0.05, seed=7)
        assert not np.allclose(batch[0].y, batch[1].y)
        again = band.run(final_time=1.0, dt=0.05, seed=7)
        assert np.allclose(batch[0].y, again[0].y)

    def test_copies_is_gone_and_the_error_names_ensemble(self):
        with pytest.raises(AttributeError, match="ensemble returns the lazy wrapper"):
            ts.systems.Lorenz().copies([[1.0, 1.0, 1.0]])


class TestProjectAndTangentLeftTheObjectButNotTheLibrary:
    """§3.2 — their *capability* must survive at the new address."""

    def test_project_is_gone_from_the_object(self):
        with pytest.raises(AttributeError) as err:
            ts.systems.Lorenz().project(0, 2)
        text = str(err.value)
        assert 'traj[["x", "z"]]' in text
        assert "ts.derived.ProjectedSystem" in text

    def test_the_trajectory_slice_genuinely_replaces_project(self):
        lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
        full = lor.run(final_time=2.0, dt=0.1)
        sliced = np.asarray(np.asarray(full.y)[:, [0, 2]], dtype=float)
        projected = ProjectedSystem(lor, [0, 2]).run(final_time=2.0, dt=0.1)
        assert np.max(np.abs(sliced - np.asarray(projected.y, dtype=float))) == 0.0

    def test_tangent_is_gone_from_the_object(self):
        with pytest.raises(AttributeError, match="ts.derived.TangentSystem"):
            ts.systems.Lorenz().tangent(k=2)

    def test_tangent_system_still_drives_the_lyapunov_path(self):
        lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
        exps = np.asarray(
            TangentSystem(lor, 3)._lyapunov_spectrum(final_time=60.0, dt=0.05, transient=10.0),
            dtype=float,
        )
        assert exps.shape == (3,)
        assert exps[0] > 0.5
        assert exps[-1] < -10.0
        assert abs(exps[1]) < 0.3
