"""The v6 system-object contract, pinned.

Every assertion here is a *user-facing* promise from ``CONTRACT.md`` §2.2, §3.1
and §3.6 — the exact tab listing, the one trajectory verb, the closed ``run``
signatures, and the two behaviour fixes.  Each one fails on the v5 tree: the
listings were 37-39 names, ``integrate``/``iterate``/``trajectory`` all resolved,
``DelaySystem.run`` swallowed any keyword, and ``run(ic=...)`` latched.
"""

from __future__ import annotations

import inspect
import operator

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidParameterError
from tsdynamics.families.protocol import System

# --- CONTRACT.md §2.2 — the exact listings -------------------------------- #

CORE_19 = (
    "copy",
    "dim",
    "ensemble",
    "family",
    "ic",
    "info",
    "jacobian",
    "jacobian_sym",
    "params",
    "plot",
    "poincare",
    "reinit",
    "run",
    "set_state",
    "state",
    "step",
    "time",
    "variables",
    "with_params",
)


def touch(obj: object, name: str) -> object:
    """Read one attribute — the thing a user types, and the thing v6 must answer."""
    return operator.attrgetter(name)(obj)


def public(obj: object) -> list[str]:
    """The names ``obj.<TAB>`` offers."""
    return sorted(n for n in dir(obj) if not n.startswith("_"))


class TestTheTabListing:
    """``system.<TAB>`` is 19 names, every one a verb or a fact (ruling A5)."""

    def test_continuous_system_is_exactly_the_core_nineteen(self):
        assert public(ts.systems.Lorenz()) == sorted(CORE_19)

    def test_a_map_drops_poincare_and_jacobian_sym(self):
        # A section is the crossing of a CONTINUOUS trajectory; a map's kernel is
        # traced numerically rather than held as a symbolic tree.
        assert public(ts.systems.Henon()) == sorted(set(CORE_19) - {"poincare", "jacobian_sym"})

    def test_a_stochastic_system_drops_jacobian_sym_only(self):
        assert public(ts.systems.OrnsteinUhlenbeck()) == sorted(set(CORE_19) - {"jacobian_sym"})

    def test_a_delay_system_drops_set_state_and_both_jacobians(self):
        assert public(ts.systems.MackeyGlass()) == sorted(
            set(CORE_19) - {"set_state", "jacobian", "jacobian_sym"}
        )

    def test_a_wrapped_system_is_thirteen(self):
        w = ts.WrappedSystem(lambda u, dt: u * 0.9, dim=1, family="map")
        assert public(w) == [
            "copy",
            "dim",
            "ensemble",
            "family",
            "plot",
            "poincare",
            "reinit",
            "run",
            "set_state",
            "state",
            "step",
            "time",
            "variables",
        ]

    @pytest.mark.parametrize("name", ["Lorenz", "Henon", "MackeyGlass", "OrnsteinUhlenbeck"])
    def test_nothing_the_listing_denies_still_resolves(self, name):
        system = getattr(ts.systems, name)()
        gone = {
            "integrate",
            "iterate",
            "trajectory",
            "copies",
            "stroboscope",
            "project",
            "tangent",
            "meta",
            "to_plot_spec",
            "is_discrete",
            "default_ic",
            "reference",
            "doi",
            "known_lyapunov",
            "field_labels",
            "lyap",
            "chaos",
            "dims",
            "recurrence",
            "lyapunov_spectrum",
            "fixed_points",
        }
        assert gone.isdisjoint(public(system))


class TestAbsentNamesTeach:
    """§3.6 — a name that cannot work does not exist, and says why."""

    def test_integrate_is_gone_and_the_error_names_run(self):
        lor = ts.systems.Lorenz()
        assert not hasattr(lor, "integrate")
        with pytest.raises(AttributeError) as err:
            touch(lor, "integrate")
        text = str(err.value)
        assert "run is the one trajectory verb" in text
        assert "system.run(" in text

    def test_a_map_has_no_poincare_and_the_reason_is_mathematical(self):
        with pytest.raises(AttributeError, match="no in-between to cross"):
            touch(ts.systems.Henon(), "poincare")

    def test_a_delay_system_has_no_set_state_and_offers_reinit(self):
        mg = ts.systems.MackeyGlass()
        assert not hasattr(mg, "set_state")
        with pytest.raises(AttributeError) as err:
            touch(mg, "set_state")
        assert "history function" in str(err.value)
        assert "reinit(history=" in str(err.value)

    def test_a_deleted_accessor_namespace_lists_free_functions(self):
        with pytest.raises(AttributeError) as err:
            touch(ts.systems.Lorenz(), "lyap")
        text = str(err.value)
        assert "ts.analysis.lyapunov_spectrum(system)" in text
        assert "ts.analysis.find(system)" in text

    def test_a_parameter_typo_is_still_answered_as_a_parameter_typo(self):
        with pytest.raises(AttributeError, match="Did you mean 'sigma'"):
            touch(ts.systems.Lorenz(), "sigmaa")

    def test_hasattr_on_an_unknown_name_is_false_not_an_exception(self):
        assert hasattr(ts.systems.Lorenz(), "no_such_name_at_all") is False


# --- CONTRACT.md §3.1 — one verb, closed signatures ----------------------- #


class TestRunIsTheOneVerb:
    def test_every_family_answers_run(self):
        assert ts.systems.Lorenz().run(final_time=1.0, dt=0.1).y.shape == (11, 3)
        # ``steps + 1`` rows: the state it started from, then one per iteration
        # — the same ``N + 1`` a flow returns for ``final_time / dt == N``.
        assert ts.systems.Henon().run(steps=50).y.shape == (51, 2)
        assert ts.systems.OrnsteinUhlenbeck().run(final_time=1.0, dt=0.1, seed=0).y.shape == (11, 1)
        traj = ts.systems.MackeyGlass().run(final_time=10.0, dt=0.5)
        assert traj.y.shape[0] == 21

    def test_the_horizon_word_follows_the_family(self):
        assert inspect.signature(ts.systems.Lorenz().run).parameters["final_time"]
        assert inspect.signature(ts.systems.Henon().run).parameters["steps"]

    def test_a_map_refuses_final_time_by_name(self):
        with pytest.raises(InvalidParameterError) as err:
            ts.systems.Henon().run(final_time=10.0)
        text = str(err.value)
        assert "a map has no continuous time" in text
        assert "Henon().run(steps=1000)" in text

    def test_a_flow_refuses_steps_by_name(self):
        with pytest.raises(InvalidParameterError, match="steps is a .map. keyword"):
            ts.systems.Lorenz().run(steps=1000)

    def test_a_delay_system_no_longer_swallows_unknown_keywords(self):
        # Measured on v5: every one of these was accepted and silently dropped.
        mg = ts.systems.MackeyGlass()
        for bad in ({"max_step": 0.1}, {"t0": 1.0}, {"events": []}, {"nonsense_kw": 1}):
            with pytest.raises(InvalidParameterError):
                mg.run(final_time=1.0, dt=0.5, **bad)

    def test_a_near_miss_keyword_is_answered_with_the_accepted_set(self):
        with pytest.raises(InvalidParameterError) as err:
            ts.systems.MackeyGlass().run(final_time=1.0, histry=1)
        text = str(err.value)
        assert "Did you mean 'history'?" in text
        assert "final_time, dt, ic, history, transient, solver, rtol, atol, backend, seed" in text

    def test_events_on_a_delay_system_names_the_engine_limit(self):
        with pytest.raises(InvalidParameterError) as err:
            ts.systems.MackeyGlass().run(final_time=1.0, events=[("x", 1.0)])
        assert "engine limitation, not a property of delay systems" in str(err.value)


class TestSolverReplacesMethod:
    """C3 — ``solver=`` is the numerical kernel; ``method=`` is an estimator."""

    @pytest.mark.parametrize("name", ["Lorenz", "MackeyGlass", "OrnsteinUhlenbeck", "Henon"])
    def test_method_is_refused_by_name_on_every_family(self, name):
        with pytest.raises(InvalidParameterError, match="solver"):
            getattr(ts.systems, name)().run(method="rk45")

    def test_solver_selects_the_kernel(self):
        traj = ts.systems.Lorenz().run(final_time=1.0, dt=0.5, solver="dop853", ic=[1.0, 1.0, 1.0])
        assert traj.meta["method"] == "dop853"


# --- CONTRACT.md §3.1 — the two behaviour fixes --------------------------- #


class TestRunDoesNotMutateTheSystem:
    def test_an_explicit_ic_is_not_latched(self):
        lor = ts.systems.Lorenz()
        assert lor.ic is None
        lor.run(final_time=0.5, dt=0.5, ic=[3.0, 3.0, 3.0])
        assert lor.ic is None, "run(ic=...) must not latch the initial condition"

    def test_a_constructor_ic_survives_a_run_with_another_ic(self):
        lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
        lor.run(final_time=0.5, dt=0.5, ic=[9.0, 9.0, 9.0])
        assert np.array_equal(lor.ic, [1.0, 1.0, 1.0])

    def test_an_auto_resolved_ic_still_latches_so_a_bare_run_repeats(self):
        lor = ts.systems.Lorenz()
        a = lor.run(final_time=0.5, dt=0.5)
        b = lor.run(final_time=0.5, dt=0.5)
        assert lor.ic is not None
        assert np.array_equal(a.y, b.y)

    def test_a_map_run_with_an_ic_does_not_latch_either(self):
        hen = ts.systems.Henon()
        hen.run(steps=10, ic=[0.1, 0.1])
        assert hen.ic is None


class TestRunIsAlwaysFresh:
    """§3.1 fix 2 — ``run()`` restarts; ``step()`` is the one that continues."""

    def test_a_poincare_map_run_twice_returns_the_same_data(self):
        pmap = ts.systems.Rossler(ic=[1.0, 1.0, 1.0]).poincare("y", 0.0)
        first = pmap.run(steps=5)
        second = pmap.run(steps=5)
        assert np.allclose(first.y, second.y)

    def test_a_stroboscopic_map_run_twice_returns_the_same_data(self):
        smap = ts.systems.Duffing(ic=[0.5, 0.0, 0.0]).poincare(period=1.0)
        assert np.allclose(smap.run(5).y, smap.run(5).y)

    def test_step_still_continues_after_a_run(self):
        pmap = ts.systems.Rossler(ic=[1.0, 1.0, 1.0]).poincare("y", 0.0)
        section = pmap.run(steps=3)
        nxt = pmap.step()
        assert not np.allclose(nxt, section.y[0])


# --- CONTRACT.md §3.4 / §3.7 — identity ----------------------------------- #


class TestIdentity:
    def test_dim_is_read_only(self):
        lor = ts.systems.Lorenz()
        with pytest.raises(InvalidParameterError, match="read-only"):
            lor.dim = 5

    def test_family_replaces_is_discrete_and_separates_dde_from_sde(self):
        assert ts.systems.Lorenz().family == "ode"
        assert ts.systems.Henon().family == "map"
        assert ts.systems.MackeyGlass().family == "dde"
        assert ts.systems.OrnsteinUhlenbeck().family == "sde"

    def test_params_is_a_real_dict(self):
        import json

        params = ts.systems.Lorenz().params
        assert isinstance(params, dict)
        assert repr(params) == "{'sigma': 10.0, 'rho': 28.0, 'beta': 2.6666666666666665}"
        assert json.loads(json.dumps(params))["rho"] == 28.0

    def test_params_stays_fixed_key_through_every_dict_mutator(self):
        params = ts.systems.Lorenz().params
        for call in (
            lambda: params.__setitem__("nope", 1),
            lambda: params.update(nope=1),
            lambda: params.setdefault("nope", 1),
        ):
            with pytest.raises(KeyError):
                call()
        for call in (params.clear, params.popitem, lambda: params.pop("rho")):
            with pytest.raises(TypeError):
                call()

    def test_the_ordered_tape_contract_survives_the_dict_subclass(self):
        assert ts.systems.Lorenz().params.as_tuple() == (10.0, 28.0, 8 / 3)

    def test_every_catalogue_system_names_every_component(self):
        """``variables`` is never ``None`` and always has exactly ``dim`` names.

        This is the whole point of the v6 rule: before it, 5 of 177 systems named
        nothing and ``traj["x"]`` on them was a guess.
        """
        from tsdynamics import registry

        missing = []
        for entry in registry.all_systems():
            system = entry.cls()
            names = system.variables
            if not isinstance(names, tuple) or len(names) != system.dim:
                missing.append(entry.name)
            elif len(set(names)) != len(names):
                missing.append(f"{entry.name} (duplicate names)")
        assert missing == []

    @pytest.mark.parametrize(
        ("name", "head"),
        [
            ("Lorenz", ("x", "y", "z")),
            ("GrayScott", ("u0", "u1", "u2")),
        ],
    )
    def test_declared_and_field_names_win(self, name, head):
        system = getattr(ts.systems, name)()
        assert system.variables[: len(head)] == head

    def test_a_field_system_names_are_block_major(self):
        gs = ts.systems.GrayScott()
        names = gs.variables
        assert names[0] == "u0"
        assert names[gs.dim // 2] == "v0"
        assert names[-1] == f"v{gs.dim // 2 - 1}"

    def test_dim_follows_variables(self):
        class Declared(ts.ContinuousSystem):
            """A system that declares names and no dim."""

            variables = ("a", "b")
            params = {"k": 1.0}

            @staticmethod
            def _equations(u, t, k):
                return [-k * u(0), -k * u(1)]

        assert Declared().dim == 2

    def test_declaring_dim_and_variables_in_disagreement_is_refused(self):
        with pytest.raises(TypeError, match="same number"):

            class Disagreeing(ts.ContinuousSystem):
                """dim says three, variables names two."""

                dim = 3
                variables = ("a", "b")
                params = {"k": 1.0}

                @staticmethod
                def _equations(u, t, k):
                    return [-k * u(0)]

    def test_info_is_the_record_the_classvars_moved_into(self):
        info = ts.systems.Lorenz().info
        assert info.name == "Lorenz"
        assert info.family == "ode"
        assert info.doi and info.doi.startswith("10.1175/")
        assert info.reference and "Lorenz" in info.reference
        assert info.known_lyapunov is not None
        text = repr(info)
        assert "dx/dt" in text
        assert "reference" in text
        assert "doi:10.1175/" in text

    def test_with_params_survives_a_dim_constructed_system(self):
        class DimCtor(ts.ContinuousSystem):
            """Declares no class-level dim; the constructor supplies it."""

            params = {"k": 1.0}

            @staticmethod
            def _equations(u, t, k):
                return [-k * u(0), -k * u(1)]

        system = DimCtor(dim=2)
        assert system.with_params(k=2.0).dim == 2
        assert system.copy().dim == 2


# --- CONTRACT.md §3.6 — the shrunken protocol ----------------------------- #


class TestTheSystemProtocol:
    @pytest.mark.parametrize("name", ["Lorenz", "Henon", "MackeyGlass", "OrnsteinUhlenbeck"])
    def test_every_family_satisfies_it(self, name):
        assert isinstance(getattr(ts.systems, name)(), System)

    def test_a_delay_system_satisfies_it_despite_having_no_set_state(self):
        # The reason ``set_state`` left the protocol: on 3.12+ ``isinstance``
        # checks data members, so keeping it would make this False.
        assert isinstance(ts.systems.MackeyGlass(), System)

    def test_set_state_is_not_a_protocol_member(self):
        assert "set_state" not in System.__protocol_attrs__

    def test_run_and_family_are_protocol_members(self):
        assert {"run", "family", "dim", "step", "state", "time", "reinit"} <= (
            System.__protocol_attrs__
        )


# --- CONTRACT.md §11 — the visibility ruling ------------------------------ #
#
# Round 9 (`hide-what-is-not-typed`).  The rule under every assertion below:
#
#     Hiding is a DISCOVERY change, never a REACHABILITY change.
#
# So each listing test has a twin that reads the hidden name back.  A surface
# cannot silently regrow (the listing tests fail), and a "simplification" cannot
# silently cost capability (the reachability tests fail) — guardrails G1 and G2
# are both mechanical here, not editorial.


def _wrapped_system() -> object:
    """A live ``WrappedSystem`` — the adapter door, built the way a user builds it."""
    return ts.families.WrappedSystem(lambda u, dt: np.asarray(u, dtype=float) * 0.99, dim=2)


def _every_object_that_has_a_tab_surface() -> dict[str, object]:
    """One live instance of everything ``families/`` · ``data/`` · ``derived/`` hands back.

    Built once per test rather than at import: a module-level fixture that
    integrates would make collection cost a run.
    """
    from tsdynamics.derived import ProjectedSystem, TangentSystem

    lor = ts.systems.Lorenz()
    ros = ts.systems.Rossler()
    band = lor.ensemble([[1.0, 1.0, 1.0], [1.001, 1.0, 1.0]])
    pmap = ros.poincare("y", 0.0)
    return {
        "ContinuousSystem": lor,
        "DiscreteMap": ts.systems.Henon(),
        "DelaySystem": ts.systems.MackeyGlass(),
        "StochasticSystem": ts.systems.OrnsteinUhlenbeck(),
        "WrappedSystem": _wrapped_system(),
        "Trajectory": lor.run(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0]),
        "PoincareMap": pmap,
        "StroboscopicMap": ts.systems.Duffing().poincare(period=4.488),
        "TangentSystem": TangentSystem(lor, k=2),
        "Ensemble": band,
        "ProjectedSystem": ProjectedSystem(lor, ["x", "z"]),
        "TrajectoryBatch": band.run(final_time=0.3, dt=0.1),
        "EnsembleSamples": band.collect(2),
        "ParamSet": lor.params,
        "SystemInfo": lor.info,
    }


#: Every tab surface this slot owns, measured.  Re-measure, never nudge.
#:
#: The five family rows and ``Trajectory`` are UNCHANGED by the visibility
#: ruling — §11 found nothing to hide on a system or a trajectory, which is the
#: finding, not an omission: the overwhelm was one dot down, on the wrappers and
#: the records.  The four that moved are marked.
TAB_SURFACES: dict[str, tuple[str, ...]] = {
    # --- unchanged by §11 (pinned so a future round cannot drift them) ---
    "ContinuousSystem": CORE_19,
    "DiscreteMap": tuple(sorted(set(CORE_19) - {"poincare", "jacobian_sym"})),
    "DelaySystem": tuple(sorted(set(CORE_19) - {"set_state", "jacobian", "jacobian_sym"})),
    "StochasticSystem": tuple(sorted(set(CORE_19) - {"jacobian_sym"})),
    "WrappedSystem": (
        "copy",
        "dim",
        "ensemble",
        "family",
        "plot",
        "poincare",
        "reinit",
        "run",
        "set_state",
        "state",
        "step",
        "time",
        "variables",
    ),
    "Trajectory": (
        "after",
        "component",
        "dim",
        "dt",
        "meta",
        "minmax",
        "n_steps",
        "neighbors",
        "plot",
        "set_distance",
        "shape",
        "standardize",
        "system",
        "t",
        "to_frame",
        "unbounded",
        "unpack",
        "variables",
        "y",
    ),
    "StroboscopicMap": (
        "copy",
        "dim",
        "family",
        "params",
        "period",
        "plot",
        "reinit",
        "run",
        "set_state",
        "state",
        "step",
        "system",
        "time",
        "variables",
        "with_params",
    ),
    # --- moved by §11 T2 ---
    "PoincareMap": (  # 21 -> 20   (- plane_auto)
        "as_events",
        "copy",
        "crossing_count",
        "dim",
        "direction",
        "dt",
        "family",
        "max_time",
        "params",
        "plane",
        "plot",
        "reinit",
        "run",
        "set_state",
        "state",
        "step",
        "system",
        "time",
        "variables",
        "with_params",
    ),
    "TangentSystem": (  # 19 -> 17   (- convergence, - growths)
        "copy",
        "deviations",
        "dim",
        "exponents",
        "family",
        "k",
        "params",
        "plot",
        "reinit",
        "run",
        "set_state",
        "state",
        "step",
        "system",
        "time",
        "variables",
        "with_params",
    ),
    "Ensemble": (  # 16 -> 14   (- states, - set_states; template -> system)
        "collect",
        "dim",
        "family",
        "members",
        "params",
        "plot",
        "reinit",
        "run",
        "size",
        "state",
        "step",
        "system",
        "time",
        "variables",
    ),
    "ProjectedSystem": (  # 16 -> 15   (- complete)
        "components",
        "copy",
        "dim",
        "family",
        "params",
        "plot",
        "reinit",
        "run",
        "set_state",
        "state",
        "step",
        "system",
        "time",
        "variables",
        "with_params",
    ),
    "TrajectoryBatch": ("diverged", "final", "plot", "t", "to_frame", "y"),  # 8 -> 6
    "EnsembleSamples": ("states", "times"),  # 4 -> 2
    "ParamSet": (  # 14 -> 10   (- clear, - fromkeys, - pop, - popitem)
        "as_dict",
        "as_tuple",
        "copy",
        "get",
        "items",
        "keys",
        "param_hash",
        "setdefault",
        "update",
        "values",
    ),
    "SystemInfo": (  # 15 -> 14   (- of)
        "default_ic",
        "defaults",
        "dim",
        "doi",
        "equations",
        "family",
        "field_labels",
        "field_shape",
        "known_lyapunov",
        "name",
        "parameters",
        "qualname",
        "reference",
        "variables",
    ),
}


class TestEveryTabSurfaceIsPinned:
    """``<TAB>`` on anything this slot hands back is an exact, measured listing."""

    def test_every_listing_is_exactly_what_was_measured(self):
        live = _every_object_that_has_a_tab_surface()
        assert set(live) == set(TAB_SURFACES), "a new returned type needs a pinned listing"
        actual = {name: tuple(public(obj)) for name, obj in live.items()}
        expected = {name: tuple(sorted(cols)) for name, cols in TAB_SURFACES.items()}
        assert actual == expected

    def test_the_family_core_and_the_trajectory_did_not_shrink(self):
        # The §11 finding, asserted as a finding: hiding cost the five family
        # rows and the Trajectory nothing at all.
        live = _every_object_that_has_a_tab_surface()
        assert len(public(live["ContinuousSystem"])) == 19
        assert len(public(live["Trajectory"])) == 19


class TestHidingNeverUnbinds:
    """G1, mechanically: every name §11 withheld is still reachable and still works."""

    def test_the_registry_covers_this_slot(self):
        from tsdynamics.families._hidden import HIDDEN_NAMES

        owners = {key.rsplit(".", 1)[-1] for key in HIDDEN_NAMES}
        assert {
            "ParamSet",
            "SystemInfo",
            "PoincareMap",
            "TangentSystem",
            "Ensemble",
            "ProjectedSystem",
            "TrajectoryBatch",
            "EnsembleSamples",
        } <= owners

    def test_every_hidden_name_still_resolves_on_a_live_object(self):
        from tsdynamics.families._hidden import HIDDEN_NAMES

        by_class = {key.rsplit(".", 1)[-1]: names for key, names in HIDDEN_NAMES.items()}
        live = _every_object_that_has_a_tab_surface()
        checked = 0
        for cls_name, obj in live.items():
            for name in by_class.get(cls_name, ()):
                assert name not in public(obj), f"{cls_name}.{name} is listed after all"
                assert hasattr(obj, name), f"{cls_name}.{name} was UNBOUND, not hidden"
                touch(obj, name)  # the read a user would do
                checked += 1
        assert checked >= 12, "the sweep stopped finding the hidden population"

    def test_the_hidden_callables_still_compute_the_same_answers(self):
        # Not merely bound — still *correct*.  These four are the ones with
        # live callers outside this slot (viz.transforms.spectra drives
        # `convergence`; docs/analysis/lyapunov.md runs `growths()`), which is
        # why renaming them to `_name` was the wrong mechanism.
        from tsdynamics.derived import TangentSystem

        hen = ts.systems.Henon()
        times, est = TangentSystem(hen, k=2).convergence(steps=100, ic=[0.1, 0.1])
        fresh = TangentSystem(hen, k=2).run(steps=100, ic=[0.1, 0.1]).unpack()
        np.testing.assert_array_equal(times, fresh[0])
        np.testing.assert_array_equal(est, fresh[1])

        tang = TangentSystem(hen, k=2)
        tang.reinit([0.1, 0.1])
        tang.step()
        assert tang.growths().shape == (2,)

        band = ts.systems.Lorenz().ensemble([[1.0, 1.0, 1.0], [1.001, 1.0, 1.0]])
        np.testing.assert_array_equal(band.states(), band.state())
        band.set_states([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(band.states()[0], [0.0, 0.0, 0.0])

        assert ts.systems.Rossler().poincare("y", 0.0).plane_auto is False

    def test_a_fixed_key_mapping_still_refuses_the_hidden_mutators(self):
        # `dir()` is not the lookup path — the teaching error still fires.
        params = ts.systems.Lorenz().params
        for call in (lambda: params.pop("sigma"), params.popitem, params.clear):
            with pytest.raises(ts.InvalidInputError, match="fixed-key"):
                call()

    def test_hiding_composes_with_a_base_that_already_curates_its_listing(self):
        # Load-bearing: ``SystemBase.__dir__`` drops the ``Absent`` slots, so a
        # decorated subclass that jumped straight to ``object.__dir__`` would
        # silently re-advertise names that *cannot work* on that family — the
        # exact lie ``Absent`` exists to prevent.  The decorator therefore
        # delegates along the MRO, which this pins.
        from tsdynamics.families._hidden import hide

        class Curated:
            def __dir__(self):
                return ["kept", "curated_only"]

        @hide("kept")
        class Narrowed(Curated):
            pass

        assert dir(Narrowed()) == ["curated_only"]


class TestTheRenamedAndTheReadOnly:
    """§11 T2's one rename and one write-guard, both with the message that teaches."""

    def test_an_ensemble_names_its_inner_system_the_way_every_view_does(self):
        band = ts.systems.Lorenz().ensemble([[1.0, 1.0, 1.0]])
        assert isinstance(band.system, ts.ContinuousSystem)
        with pytest.raises(AttributeError, match=r"band\.system"):
            touch(band, "template")

    def test_the_inner_system_has_one_spelling_across_every_wrapper(self):
        live = _every_object_that_has_a_tab_surface()
        wrappers = [
            "PoincareMap",
            "StroboscopicMap",
            "TangentSystem",
            "Ensemble",
            "ProjectedSystem",
        ]
        assert all("system" in public(live[name]) for name in wrappers)

    def test_a_section_plane_cannot_be_relabelled_after_the_march(self):
        pmap = ts.systems.Rossler().poincare("y", 0.0)
        with pytest.raises(InvalidParameterError, match="read-only"):
            pmap.plane = ("z", 27.0)
        assert pmap.plane == (1, 0.0)  # and the section is untouched

    def test_the_refusal_hands_back_a_line_that_builds_the_section_meant(self):
        pmap = ts.systems.Rossler().poincare("y", 0.0)
        with pytest.raises(InvalidParameterError) as excinfo:
            pmap.plane = ("z", 27.0)
        assert "system.poincare(" in str(excinfo.value)

    def test_the_v5_class_alias_is_bound_but_unlisted(self):
        import tsdynamics.derived as derived

        assert derived.EnsembleSystem is derived.Ensemble
        assert "EnsembleSystem" not in derived.__all__
        assert "EnsembleSystem" not in dir(derived)


class TestEveryModuleDeclaresItsOwnSurface:
    """§11's enforcement clause, swept over this slot's three packages.

    ``__all__`` alone buys nothing a user sees: it governs ``import *`` and has
    **zero** effect on ``<TAB>``.  A module that declares one must therefore also
    define ``__dir__``, or the listing it curated is not the listing anyone gets.
    """

    @staticmethod
    def _modules() -> list[object]:
        import importlib
        import pkgutil

        mods = []
        for pkg_name in ("tsdynamics.families", "tsdynamics.data", "tsdynamics.derived"):
            pkg = importlib.import_module(pkg_name)
            mods.append(pkg)
            mods += [
                importlib.import_module(f"{pkg_name}.{sub.name}")
                for sub in pkgutil.iter_modules(pkg.__path__)
            ]
        return mods

    def test_every_module_declares_all_and_mirrors_it_in_dir(self):
        missing = [
            m.__name__
            for m in self._modules()
            if not hasattr(m, "__all__") or "__dir__" not in vars(m)
        ]
        assert missing == []

    def test_no_module_offers_a_name_from_outside_the_library(self):
        # 55 names leaked across 9 modules before §11 T4 — `np`, `math`,
        # `weakref`, `OrderedDict`, `abstractmethod`, `annotations`, `Any`,
        # `cast`, ... — none of which says anything about the module offering it.
        #
        # This is deliberately phrased over `dir()` *and* resolved ownership
        # rather than over `vars()`: `from typing import Any` at the top of a
        # module is ordinary Python and is not the defect.  The defect is a
        # module whose **listing** hands those names to a reader, which is what
        # happens the moment `__all__`/`__dir__` is missing or wrong.
        import types
        import typing

        def is_ours(value: object) -> bool:
            """Was *value* defined inside this library?

            A ``X | Y`` alias (``data.Region``) reports ``__module__ ==
            'typing'`` however library-owned its members are, so a union is
            judged by its arguments.
            """
            args = typing.get_args(value)
            if args:
                return all(is_ours(arg) for arg in args)
            owner = (
                value.__name__
                if isinstance(value, types.ModuleType)
                else getattr(value, "__module__", None)
            )
            return owner is None or owner.startswith("tsdynamics")

        foreign = {}
        for m in self._modules():
            offered = sorted(
                name
                for name in dir(m)
                if not name.startswith("_") and not is_ours(getattr(m, name))
            )
            if offered:
                foreign[m.__name__] = offered
        assert foreign == {}

    def test_dir_is_exactly_the_sorted_listing(self):
        assert all(dir(m) == sorted(m.__all__) for m in self._modules())


class TestNothingPublicIsUndocumented:
    """§11.6 defect 2, swept: ``help()`` on a public member must answer.

    ``help(DerivedSystem.dim)`` printed three blank lines and ``help(pmap.dim)``
    printed ``int([x]) -> integer`` — the builtin's reference, about a different
    subject.  ``help(traj.y)`` printed NumPy's, ``help(traj.meta)`` ``dict()``'s.
    20 members, all in this slot; the sweep now finds none.
    """

    def test_every_public_member_of_every_returned_type_has_a_docstring(self):
        import inspect

        undocumented = []
        for cls_name, obj in _every_object_that_has_a_tab_surface().items():
            cls = type(obj)
            for name in public(obj):
                member = inspect.getattr_static(cls, name, None)
                if member is None:  # a plain instance attribute carries no doc slot
                    continue
                if isinstance(member, property):
                    doc = member.__doc__
                elif isinstance(member, staticmethod | classmethod):
                    doc = member.__func__.__doc__
                else:
                    doc = inspect.getdoc(member)
                if not (doc or "").strip():
                    undocumented.append(f"{cls_name}.{name}")
        assert undocumented == []

    def test_the_four_trajectory_channels_document_themselves(self):
        import inspect

        from tsdynamics.data import Trajectory

        for name in ("t", "y", "system", "meta"):
            doc = inspect.getdoc(getattr(Trajectory, name)) or ""
            assert len(doc) > 40, f"Trajectory.{name} still has no docstring of its own"


# --- CONTRACT.md §11.3 T2 + §11.6 defect 1 — a write cannot corrupt an answer  #
#
# The visibility ruling hid ~40 slots across this territory, but it also found
# eight public attributes that were *writable* and, once written, made the object
# answer wrongly rather than loudly.  The chair ruled all of them **KEEP** — a
# guarded setter, never a hidden name — because renaming a system's components or
# re-aiming a projection is legitimate customization (G1).  What follows pins the
# guard AND the capability it protects: for every refusal there is a paired test
# that the legitimate write still works.


def _lorenz():
    return ts.systems.Lorenz()


def _tangent():
    from tsdynamics.derived import TangentSystem

    return TangentSystem(_lorenz(), k=2)


def _projected():
    from tsdynamics.derived import ProjectedSystem

    return ProjectedSystem(_lorenz(), [0, 2])


def _section():
    return _lorenz().poincare("z", 20.0)


def _traj():
    return _lorenz().run(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0])


#: ``(id, factory, attribute, rejected value, message must name)``.  One row per
#: attribute §11.6 defect 1 lists that this slot owns; ``Plot.kind``/``.layers``
#: are the viz slot's two.
STRUCTURAL_WRITES = [
    ("system-params", _lorenz, "params", {"sigma": 10.0}, "with_params"),
    ("system-plot", _lorenz, "plot", 42, "plot()"),
    ("system-dim", _lorenz, "dim", 5, "read-only"),
    ("system-variables-arity", _lorenz, "variables", ("a", "b"), "but this system has 3"),
    ("system-variables-type", _lorenz, "variables", 42, "sequence of 3 component names"),
    ("system-variables-dupes", _lorenz, "variables", ("a", "a", "b"), "repeats"),
    ("tangent-k-range", _tangent, "k", 7, "must be in [1, 3]"),
    ("tangent-k-whole", _tangent, "k", 2.5, "whole number"),
    ("projected-components-range", _projected, "components", [0, 99], "does not exist"),
    ("projected-components-name", _projected, "components", ["x", "q"], "no component 'q'"),
    ("projected-components-empty", _projected, "components", [], "at least one"),
    ("section-plane", _section, "plane", (0, 1.0), "read-only"),
    ("traj-y-rows", _traj, "y", np.zeros((3, 3)), "rows"),
    ("traj-t-rows", _traj, "t", np.zeros(3), "rows"),
    ("traj-meta-type", _traj, "meta", 42, "provenance mapping"),
]


class TestAWriteCannotCorruptTheAnswer:
    """§11.6 defect 1: the write is refused where the mistake is, not five frames on."""

    @pytest.mark.parametrize(
        ("factory", "attr", "value", "names"),
        [row[1:] for row in STRUCTURAL_WRITES],
        ids=[row[0] for row in STRUCTURAL_WRITES],
    )
    def test_a_structural_write_is_refused_and_the_message_says_why(
        self, factory, attr, value, names
    ):
        subject = factory()
        with pytest.raises((ts.InvalidParameterError, ts.InvalidInputError)) as excinfo:
            setattr(subject, attr, value)
        assert names in str(excinfo.value)

    @pytest.mark.parametrize(
        ("factory", "attr", "value", "names"),
        [row[1:] for row in STRUCTURAL_WRITES],
        ids=[row[0] for row in STRUCTURAL_WRITES],
    )
    def test_the_refusal_is_catchable_as_the_builtin_it_subclasses(
        self, factory, attr, value, names
    ):
        # The typed hierarchy is purely additive, so code that predates it and
        # catches ValueError/TypeError keeps catching these.
        subject = factory()
        with pytest.raises((ValueError, TypeError)):
            setattr(subject, attr, value)

    def test_every_remedy_line_the_refusal_prints_actually_parses(self):
        # The errgate rule: a line the library hands back must be one the user can
        # type.  13 catalogue systems declare NO parameters (every ``Sprott*``),
        # and a template filled from an empty parameter list printed
        # ``sprotta.<param> = ...``.  Those lines are dropped, not faked.
        import ast

        from tsdynamics import registry

        unparseable = []
        for entry in registry.all_systems():
            try:
                entry.cls().params = {"x": 1}
            except (ts.InvalidParameterError, ts.InvalidInputError) as err:
                for line in str(err).splitlines()[1:]:
                    if not line.strip():
                        continue
                    try:
                        ast.parse(line.strip())
                    except SyntaxError:
                        unparseable.append((entry.name, line.strip()))
        assert unparseable == []

    def test_a_system_with_no_parameters_says_so_instead_of_inventing_one(self):
        bare = ts.systems.SprottA()
        assert bare.params == {}
        with pytest.raises(ts.InvalidParameterError, match="declares no parameters"):
            bare.params = {"a": 1.0}

    def test_a_refused_write_leaves_the_object_usable(self):
        # The whole point: before the guard, `lor.variables = ("a", "b")` was
        # accepted and `lor.info` then raised a raw `IndexError: tuple index out
        # of range` from inside the equation renderer.
        lor = _lorenz()
        with pytest.raises(ts.InvalidInputError):
            lor.variables = ("a", "b")
        assert lor.variables == ("x", "y", "z")
        assert lor.info.dim == 3
        assert lor.run(final_time=0.3, dt=0.1, ic=[1.0, 1.0, 1.0]).shape == (4, 3)


class TestTheGuardedWritesStillWork:
    """G1, paired with every refusal above: not one capability was cut."""

    def test_a_parameter_value_is_as_writable_as_it_ever_was(self):
        lor = _lorenz()
        lor.sigma = 12.0
        lor.params["rho"] = 29.0
        lor.params.update({"beta": 2.0})
        assert (lor.sigma, lor.rho, lor.beta) == (12.0, 29.0, 2.0)
        assert lor.run(final_time=0.3, dt=0.1, ic=[1.0, 1.0, 1.0]).shape == (4, 3)
        assert _lorenz().with_params(sigma=12.0).sigma == 12.0

    def test_renaming_the_components_works_end_to_end(self):
        lor = _lorenz()
        lor.variables = ("a", "b", "c")
        traj = lor.run(final_time=0.3, dt=0.1, ic=[1.0, 1.0, 1.0])
        assert lor.variables == ("a", "b", "c")
        assert traj.variables == ("a", "b", "c")
        assert traj["a"].shape == (4,)
        assert lor.info.variables == ("a", "b", "c")

    def test_a_projection_can_be_re_aimed_by_name_or_by_index(self):
        proj = _projected()
        assert (proj.components, proj.dim, proj.variables) == ((0, 2), 2, ("x", "z"))
        proj.components = ["y"]
        proj.reinit([1.0, 1.0, 1.0])
        assert (proj.components, proj.dim, proj.variables) == ((1,), 1, ("y",))
        assert proj.step(0.01).shape == (1,)

    @pytest.mark.parametrize(("system", "ic", "new_k"), [("Lorenz", [1.0, 1.0, 1.0], 3)])
    def test_a_tangent_frame_can_be_widened_after_construction(self, system, ic, new_k):
        tang = _tangent()
        tang.reinit(ic)
        tang.step()
        tang.k = new_k
        tang.reinit(ic)
        tang.step()
        # The extended variational tape is shaped by k and was cached on the
        # structural parameters ALONE, so this used to reuse the k=2 tape and die
        # with `cannot reshape array of size 6 into shape (3,3)`.
        assert tang.deviations().shape == (3, new_k)
        assert len(tang.exponents()) == new_k

    def test_a_map_tangent_frame_narrows_too(self):
        from tsdynamics.derived import TangentSystem

        tang = TangentSystem(ts.systems.Henon(), k=2)
        tang.reinit([0.1, 0.1])
        tang.step()
        tang.k = 1
        tang.reinit([0.1, 0.1])
        tang.step()
        assert tang.deviations().shape == (2, 1)

    def test_writing_the_same_k_keeps_a_running_average(self):
        tang = _tangent()
        tang.reinit([1.0, 1.0, 1.0])
        for _ in range(5):
            tang.step()
        before = tang.exponents().copy()
        tang.k = 2  # the value it already has — a no-op, not a reset
        np.testing.assert_array_equal(tang.exponents(), before)

    def test_measured_data_can_still_be_patched_in_place(self):
        traj = _traj()
        traj.y = traj.y + 100.0
        assert float(traj.y[0, 0]) == pytest.approx(101.0)
        traj.y = traj.y[:, 0]  # a 1-D write normalises, exactly as the constructor does
        assert traj.y.shape == (11, 1)
        traj.meta = {"system": "measured"}
        assert traj.meta == {"system": "measured"}


class TestAStateWriteDropsWhatWasDerivedFromIt:
    """The defect verbatim: two caches are computed from ``y`` and were kept."""

    def test_the_escape_verdict_is_recomputed(self):
        traj = _traj()
        assert traj.unbounded is None  # read it, so the verdict is cached
        traj.y = np.outer(np.linspace(1.0, 1e12, traj.n_steps), np.ones(3))
        assert traj.unbounded is not None, "unbounded reported a stale verdict"
        assert "unbounded" in str(traj.unbounded)

    def test_the_neighbour_index_is_rebuilt(self):
        traj = _traj()
        traj.neighbors(traj.y[0], 2)  # build the KD-tree over the original states
        traj.y = traj.y + 100.0
        moved = traj.neighbors(traj.y[0], 2)
        # Queried at a state that IS in the new cloud, so distance 0 — a stale
        # tree would answer from the pre-write coordinates.
        assert float(moved.distance[0]) == pytest.approx(0.0, abs=1e-9)

    def test_a_write_keeps_the_two_axes_describing_the_same_rows(self):
        traj = _traj()
        with pytest.raises(ts.InvalidInputError, match="rows"):
            traj.y = np.zeros((3, 3))
        assert traj.shape == (11, 3)
        assert traj.y.shape[0] == traj.t.shape[0]


class TestTheFixedKeyMappingAnswersInEverySpelling:
    """§11.3 T2: ``fromkeys`` was hidden but not overridden, so it raised about
    ``ParamSet.__init__`` — a constructor the caller never wrote."""

    def test_fromkeys_is_refused_by_name_like_its_three_siblings(self):
        from tsdynamics.families import ParamSet

        params = _lorenz().params
        for call in (
            lambda: ParamSet.fromkeys(["a"]),
            lambda: params.fromkeys(["a"]),
            lambda: ParamSet.fromkeys(["a"], 0.0),
        ):
            with pytest.raises(ts.InvalidInputError, match="fixed-key"):
                call()

    def test_the_four_hidden_mutators_are_off_the_listing_but_answer(self):
        params = _lorenz().params
        for name in ("clear", "fromkeys", "pop", "popitem"):
            assert name not in public(params), f"ParamSet.{name} is listed after all"
            assert hasattr(params, name), f"ParamSet.{name} was UNBOUND, not hidden"

    def test_the_values_are_still_writable_through_every_listed_door(self):
        params = _lorenz().params
        params["sigma"] = 11.0
        params.sigma = 12.0
        params.update({"rho": 29.0})
        assert params.setdefault("sigma") == 12.0
        assert params.as_tuple() == (12.0, 29.0, params["beta"])
