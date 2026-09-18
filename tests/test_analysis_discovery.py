"""The analysis layer's discovery surface — the gate for owner ruling A2.

A2 took every analysis off every object.  What is left is a *namespace* and a
*search*, and if either is bad the capability is unreachable — so the surface
itself is the contract:

* ``ts.analysis.<TAB>`` is exactly 52 names, generated from the registry;
* ``ts.analysis.__doc__`` groups the 49 analyses by what you are holding;
* ``find()`` answers a question OR a subject, with a frozen ``GOLD`` table that
  fails a build rather than a user's REPL;
* the ten implementation subpackages stop shadowing, and a guess at a renamed
  name answers with the spelling that replaced it.

Every count here is measured against CONTRACT §2.4 / §5.2 / §5.3 / §5.4.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.analysis import _discovery
from tsdynamics.errors import MovedInV6

# The contract's §2.4 listing, verbatim.  52 names: 49 analyses + find/register/results.
# ``max_lyapunov`` left in v6 round 6: it was a SECOND door onto the maximal
# exponent and answered with a different number, so it folded into
# ``lyapunov_spectrum(system, k=1)``.
CONTRACT_NAMES = sorted(
    [
        "attractors",
        "autocorrelation",
        "basin_entropy",
        "basin_fractions",
        "basins",
        "box_counting_dimension",
        "cao_dimension",
        "continuation",
        "correlation_dimension",
        "correlation_sum",
        "dimension_spectrum",
        "embed",
        "embedding_dimension",
        "escape_time_field",
        "estimate_dt_from_sagitta",
        "estimate_period",
        "expansion_entropy",
        "false_nearest_neighbors",
        "find",
        "fixed_mass_dimension",
        "fixed_points",
        "flow_field",
        "ftle_field",
        "gali",
        "generalized_dimension",
        "information_dimension",
        "invariant_density",
        "kaplan_yorke_dimension",
        "lyapunov_from_data",
        "lyapunov_spectrum",
        "mutual_information",
        "nullclines",
        "optimal_delay",
        "orbit_diagram",
        "periodic_orbits",
        "poincare_section",
        "recurrence_matrix",
        "register",
        "resilience",
        "results",
        "return_map",
        "rqa",
        "sagitta_profile",
        "set_distance",
        "streamlines",
        "tipping_points",
        "trace_determinant",
        "transient_time_field",
        "uncertainty_exponent",
        "wada_property",
        "windowed_rqa",
        "zero_one_test",
    ]
)


class TestTheTabSurface:
    """``ts.analysis.<TAB>`` is the contract's 52 names, generated."""

    def test_all_is_exactly_the_contract_listing(self):
        assert sorted(ts.analysis.__all__) == CONTRACT_NAMES
        assert len(CONTRACT_NAMES) == 52

    def test_dir_mirrors_all(self):
        assert dir(ts.analysis) == sorted(ts.analysis.__all__)

    def test_every_listed_name_resolves(self):
        missing = [n for n in ts.analysis.__all__ if not hasattr(ts.analysis, n)]
        assert missing == []

    def test_all_is_generated_from_the_registry_not_hand_written(self):
        """Every registered analysis is listed, and every listed analysis is registered."""
        registered = set(registry.analyses.names())
        listed = set(ts.analysis.__all__) - {"find", "register", "results"}
        assert listed == registered
        assert len(registered) == 49

    def test_the_32_result_classes_are_bound_but_off_the_tab_surface(self):
        """C2 — a type you only ever get *back* lives at one address."""
        from tsdynamics.analysis import results

        assert len(results.__all__) == 32
        for name in results.__all__:
            assert hasattr(ts.analysis, name), f"{name} must stay reachable"
            assert name not in ts.analysis.__all__, f"{name} must not be on the tab surface"


class TestTheGroupedMap:
    """``ts.analysis.__doc__`` is generated and grouped by what you are holding."""

    def test_the_three_groups_have_the_contract_counts(self):
        entries = registry.analyses.all()
        groups: dict[str, int] = {}
        for e in entries:
            groups[_discovery.group_of(e.metadata["subjects"])] = (
                groups.get(_discovery.group_of(e.metadata["subjects"]), 0) + 1
            )
        assert groups == {"system": 20, "data": 23, "result": 6}

    def test_doc_names_every_analysis_and_the_two_ways_in(self):
        doc = ts.analysis.__doc__
        assert doc is not None
        for name in registry.analyses.names():
            assert f"{name:<25s}" in doc or f"{name} " in doc, name
        assert "ts.analysis.find(traj)" in doc
        assert 'ts.analysis.find("chaotic")' in doc
        assert "You have a SYSTEM" in doc and "You have a RESULT" in doc

    def test_every_summary_is_derived_and_fits(self):
        """Summaries come from the docstring, are <= 72 columns, and are not empty."""
        for e in registry.analyses.all():
            summary = e.metadata["summary"]
            assert summary, e.name
            assert len(summary) <= _discovery.SUMMARY_WIDTH, (e.name, len(summary), summary)
            assert summary == _discovery.summarise(e.obj.__doc__), e.name

    def test_a_summary_is_not_a_restatement_of_the_name(self):
        """A summary whose words are only the name's words teaches nothing."""
        for e in registry.analyses.all():
            words = {w for w in e.metadata["summary"].casefold().split() if len(w) > 2}
            name_words = set(e.name.casefold().split("_"))
            assert words - name_words, f"{e.name}: {e.metadata['summary']!r} restates the name"

    def test_no_rst_or_latex_markup_leaks_into_a_summary(self):
        for e in registry.analyses.all():
            s = e.metadata["summary"]
            assert "``" not in s and ":math:" not in s and ":func:" not in s, (e.name, s)
            assert "\\" not in s, (e.name, s)


#: query -> at least one of these must be in the top 3.  Frozen: adding an
#: analysis that makes an existing question ambiguous fails HERE, not in a
#: user's REPL.
GOLD: dict[str, set[str]] = {
    "is this chaotic": {
        "lyapunov_spectrum",
        "zero_one_test",
        "gali",
        "expansion_entropy",
        "lyapunov_from_data",
    },
    "how do I know if it is chaotic": {
        "lyapunov_spectrum",
        "zero_one_test",
        "gali",
        "expansion_entropy",
        "lyapunov_from_data",
    },
    "chaos": {"lyapunov_spectrum", "zero_one_test", "gali", "expansion_entropy"},
    "chaotic": {"lyapunov_spectrum", "zero_one_test", "gali", "expansion_entropy"},
    "fractal dimension": {
        "correlation_dimension",
        "box_counting_dimension",
        "generalized_dimension",
        "information_dimension",
        "fixed_mass_dimension",
        "kaplan_yorke_dimension",
    },
    "bifurcation": {"orbit_diagram", "continuation", "tipping_points"},
    "equilibria": {"fixed_points"},
    "limit cycle": {"periodic_orbits"},
    "takens": {
        "embed",
        "optimal_delay",
        "embedding_dimension",
        "cao_dimension",
        "false_nearest_neighbors",
        "mutual_information",
    },
    "recurrence plot": {"recurrence_matrix", "rqa", "windowed_rqa"},
    "multistability": {"basins", "attractors", "basin_fractions"},
    "predictability": {"uncertainty_exponent", "lyapunov_spectrum"},
    "surface of section": {"poincare_section"},
}


class TestFind:
    """``find`` answers a question or a subject — the only route for a wrong guess."""

    @pytest.mark.parametrize("query", sorted(GOLD))
    def test_gold_query_lands_in_the_top_three(self, query):
        top3 = [f.__name__ for f in ts.analysis.find(query)][:3]
        assert set(top3) & GOLD[query], f"{query!r} -> {top3}"

    def test_a_short_stopword_cannot_match_a_name_it_is_buried_in(self):
        """Without the 3-character floor, "is" in "is this chaotic" hits set_dIStance."""
        names = [f.__name__ for f in ts.analysis.find("is this chaotic")]
        assert "set_distance" not in names

    def test_a_miss_says_what_to_type_instead(self):
        out = repr(ts.analysis.find("wavelet"))
        assert "nothing matches 'wavelet'" in out
        assert "ts.analysis.__doc__" in out and "ts.analysis.find(subject)" in out
        assert "lyapunov" in out  # the areas to browse
        assert ts.analysis.find("wavelet") == []

    def test_a_miss_on_a_removed_capability_says_it_was_removed(self):
        """A word from the deleted time-series layer must not read as "cannot do that".

        ``find("surrogate")`` answered *nothing matches*, which a reader takes as
        a statement about the library's ability rather than about its scope — and
        goes off to reimplement an FT surrogate test by hand.
        """
        out = repr(ts.analysis.find("surrogate"))
        assert "REMOVED" in out and "surrogate-data tests" in out
        assert "ts.analysis.lyapunov_from_data(traj)" in out

    def test_find_understands_the_words_a_reader_brings(self):
        """``find`` is advertised as plain English, so it answers plain English.

        Measured before: ``find("robust")`` and ``find("safety margin")`` both
        returned nothing, while ``resilience``'s own summary line reads
        "Minimal-fatal-shock resilience".
        """
        for question in ("robust", "safety margin", "will it tip over"):
            names = {f.__name__ for f in ts.analysis.find(question)}
            assert names, question
            assert names & {"resilience", "tipping_points"}, (question, names)

    def test_find_returns_the_callables(self):
        hits = ts.analysis.find("multistability")
        assert all(callable(f) for f in hits)
        assert {"attractors", "basin_fractions", "basins"} <= {f.__name__ for f in hits}

    def test_a_flow_gets_20_and_a_map_gets_13(self):
        """A map has no vector field, so the seven field analyses drop out."""
        assert len(ts.analysis.find(ts.systems.Lorenz())) == 20
        assert len(ts.analysis.find(ts.systems.Henon())) == 13
        field = {
            "flow_field",
            "streamlines",
            "nullclines",
            "ftle_field",
            "escape_time_field",
            "transient_time_field",
            "trace_determinant",
        }
        got = {f.__name__ for f in ts.analysis.find(ts.systems.Henon())}
        assert not (got & field)

    def test_a_class_answers_like_its_instance(self):
        assert len(ts.analysis.find(ts.systems.Lorenz)) == 20
        assert len(ts.analysis.find(ts.systems.Henon)) == 13

    def test_a_trajectory_gets_24_and_a_bare_array_the_same(self):
        """24 since ``zero_one_test`` was registered for data as well as systems.

        Its own summary line says it runs "on a system **or a measured
        observable**", it works on a bare ndarray, and it was registered
        system-only — so the library's best discovery route hid the one analysis
        that answers the question a data-first user arrives with.
        """
        traj = ts.systems.Lorenz().run(final_time=2.0, dt=0.05)
        assert len(ts.analysis.find(traj)) == 24
        assert len(ts.analysis.find(np.zeros((50, 3)))) == 24
        assert "zero_one_test" in {f.__name__ for f in ts.analysis.find(np.zeros((50, 3)))}

    def test_a_result_gets_the_analyses_that_read_it(self):
        spectrum = ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), final_time=10.0)
        assert [f.__name__ for f in ts.analysis.find(spectrum)] == ["kaplan_yorke_dimension"]

    def test_no_argument_lists_everything(self):
        assert len(ts.analysis.find()) == 49

    def test_the_repr_is_the_grouped_table(self):
        out = repr(ts.analysis.find(ts.systems.Henon()))
        assert out.startswith("13 analyses take a Henon")
        assert "You have a SYSTEM" in out

    def test_a_subject_that_is_neither_says_so(self):
        with pytest.raises(ts.InvalidInputError, match="takes a question or a subject"):
            ts.analysis.find(object())


class TestSubpackagesStopShadowing:
    """CONTRACT §5.4 — ``ts.analysis.lyapunov`` is the first thing a refugee types."""

    AREAS = (
        "basins",
        "chaos",
        "dimensions",
        "embedding",
        "fixedpoints",
        "lyapunov",
        "orbits",
        "planar",
        "recurrence",
        "sampling",
    )

    @pytest.mark.parametrize("name", [a for a in AREAS if a != "basins"])
    def test_no_capability_subpackage_resolves_on_ts_analysis(self, name):
        with pytest.raises(AttributeError) as err:
            getattr(ts.analysis, name)
        text = str(err.value)
        assert "is the implementation package, not a verb" in text
        assert "ts.analysis." in text and "(...)" in text
        assert "ts.analysis.find(" in text

    def test_basins_resolves_to_the_analysis_not_the_package(self):
        """The one name where the function WINS: ``basins`` is a verb a user types."""
        assert callable(ts.analysis.basins)
        assert ts.analysis.basins.__name__ == "basins"

    @pytest.mark.parametrize("name", AREAS)
    def test_the_package_is_still_importable(self, name):
        import importlib

        mod = importlib.import_module(f"tsdynamics.analysis.{name}")
        assert mod.__name__ == f"tsdynamics.analysis.{name}"
        # ...and the `from X import submodule` spelling still reads sys.modules
        ns: dict[str, object] = {}
        exec(f"from tsdynamics.analysis import {name}", ns)  # noqa: S102
        assert ns[name] is mod or callable(ns[name])


#: renamed name -> the substring the redirect must contain.
RENAMES = {
    "basins_of_attraction": "ts.analysis.basins(system, region)",
    "find_attractors": "ts.analysis.attractors(system, region)",
    "periodic_orbit": "ts.analysis.periodic_orbits(",
    "bifurcation_diagram": "ts.analysis.orbit_diagram(",
    "bifurcation": "ts.analysis.orbit_diagram(",
}


class TestRenamesTeachTheirReplacement:
    """A removed name raises an error that prints its replacement — that IS the guide."""

    @pytest.mark.parametrize("old,line", sorted(RENAMES.items()))
    def test_attribute_access_answers_with_the_new_spelling(self, old, line):
        with pytest.raises(MovedInV6) as err:
            getattr(ts.analysis, old)
        assert line in str(err.value)
        assert old in str(err.value)

    @pytest.mark.parametrize("old,line", sorted(RENAMES.items()))
    def test_the_import_spelling_keeps_the_text(self, old, line):
        """C6 — ``from X import Y`` discards an ``AttributeError``'s message, not an
        ``ImportError``'s.  111 corpus call sites use this spelling."""
        with pytest.raises(ImportError) as err:
            exec(f"from tsdynamics.analysis import {old}", {})  # noqa: S102
        assert line in str(err.value), str(err.value)

    def test_a_near_miss_is_an_attribute_error_not_an_import_error(self):
        """``hasattr(ts.analysis, <anything>)`` must keep answering False."""
        with pytest.raises(AttributeError) as err:
            _ = ts.analysis.correlation_dimensio
        assert "Did you mean 'correlation_dimension'" in str(err.value)
        assert not isinstance(err.value, ImportError)
        assert hasattr(ts.analysis, "random_typo_xyz") is False

    def test_a_genuine_miss_points_at_find(self):
        with pytest.raises(AttributeError) as err:
            _ = ts.analysis.wavelet_transform
        assert "ts.analysis.find(" in str(err.value)


class TestTheRegistrationDoor:
    """CONTRACT §7.2 — the definition site is the registration site."""

    def test_register_is_a_decorator_and_a_function(self):
        @ts.analysis.register(subjects=("trajectory",), area="dimensions", keywords="testonly")
        def _probe_dimension(subject):
            """Probe dimension for the gate."""

        try:
            assert registry.analyses.get("_probe_dimension") is _probe_dimension
            entry = registry.analyses.entry("_probe_dimension")
            assert entry.metadata["area"] == "dimensions"
            assert entry.metadata["subjects"] == ("trajectory",)
            assert entry.metadata["summary"] == "Probe dimension for the gate."
        finally:
            registry.analyses._entries.pop("_probe_dimension", None)

    def test_system_expands_to_flow_and_map(self):
        assert registry.analyses.entry("fixed_points").metadata["subjects"] == ("flow", "map")

    def test_an_unknown_area_is_refused_at_registration(self):
        with pytest.raises(ts.InvalidParameterError, match="must be one of"):
            ts.analysis.register(subjects=("trajectory",), area="nonsense")(lambda d: None)

    def test_every_analysis_declares_its_subjects_and_area(self):
        for e in registry.analyses.all():
            assert e.metadata.get("subjects"), e.name
            assert e.metadata.get("area") in _discovery.AREAS, e.name

    def test_an_unknown_keyword_names_this_function(self):
        """§7.2's 'wrapper that re-raises an unknown keyword' — CPython already does it."""
        with pytest.raises(TypeError) as err:
            ts.analysis.correlation_dimension(np.zeros((20, 2)), nonsense=1)
        assert "correlation_dimension" in str(err.value)


class TestTheSharedMessageBuilder:
    """CONTRACT §5.6 — one builder, so the object door and the free-function door
    cannot drift.  ``teach`` is that builder; both ``attribute_error`` and
    ``wrong_subject`` wrap the same body."""

    def test_the_two_doors_share_one_body(self):
        """Both doors render the SAME ``teach`` output; only the opening clause and
        the wrap column differ, because the traceback prefixes differ in width."""
        for e in registry.analyses.all():
            for held in ("flow", "map", "data"):
                clause, *lines = _discovery.teach(e.name, held=held)
                attr = str(_discovery.attribute_error(e.name, "X", held))
                free = str(_discovery.wrong_subject(e.name, "X", held))
                for door in (attr, free):
                    # the clause, un-wrapped
                    assert clause in " ".join(door.split("\n    ")[0].split()), (e.name, held)
                    # ...and every runnable line, verbatim and indented
                    for line in lines:
                        assert f"\n    {line}" in door, (e.name, held, line)

    def test_a_data_analysis_reached_with_a_model_hands_back_the_run_line(self):
        lines = _discovery.teach("correlation_dimension", held="flow")
        assert lines[0].startswith("it measures a point set")
        assert lines[1] == "traj = system.run(200.0, dt=0.02)"
        assert lines[2] == "ts.analysis.correlation_dimension(traj)"

    def test_the_run_line_uses_the_families_own_horizon_word(self):
        assert "20000" in _discovery.teach("rqa", held="map")[1]
        assert "dt=" in _discovery.teach("rqa", held="flow")[1]

    def test_a_model_analysis_reached_with_data_offers_its_data_sibling(self):
        lines = _discovery.teach("lyapunov_spectrum", held="data")
        assert lines[0] == "it is a property of the equations, not of a point set."
        assert lines[1] == "ts.analysis.lyapunov_spectrum(traj.system)"
        assert lines[2] == "ts.analysis.lyapunov_from_data(traj)"

    def test_no_sibling_says_why_there_is_none(self):
        lines = _discovery.teach("fixed_points", held="data")
        assert "they are roots of the equations" in lines[0]
        assert len(lines) == 2

    def test_a_trajectory_with_no_system_is_sent_to_find(self):
        lines = _discovery.teach("fixed_points", held="data", has_system=False)
        assert lines[-1].startswith("ts.analysis.find(traj)")

    def test_a_result_analysis_names_its_producer(self):
        """The live defect this closes: ``kaplan_yorke_dimension(lor)`` used to say
        "expects measured data" when what it wants is a Lyapunov spectrum."""
        lines = _discovery.teach("kaplan_yorke_dimension", held="flow")
        assert lines[0] == "it reads what another analysis returns. Compute that first:"
        assert lines[1] == "exps = ts.analysis.lyapunov_spectrum(system)"
        assert lines[2] == "ts.analysis.kaplan_yorke_dimension(exps)"
        assert "measured data" not in "\n".join(lines)

    def test_the_free_function_door_names_what_it_needs_and_what_it_got(self):
        err = _discovery.wrong_subject("correlation_dimension", "Lorenz", "flow")
        assert isinstance(err, ts.InvalidInputError)
        head = str(err).splitlines()[0]
        assert head.startswith("correlation_dimension() needs data, and got a system (Lorenz)")

    def test_every_message_fits_the_gated_standard(self):
        """<= 6 lines, <= 88 columns INCLUDING the traceback's own prefix, and at
        least one runnable line."""
        prefix = "AttributeError: "
        for e in registry.analyses.all():
            for held in ("flow", "map", "data"):
                text = str(_discovery.attribute_error(e.name, "Lorenz", held))
                lines = text.splitlines()
                assert 2 <= len(lines) <= 6, (e.name, held, len(lines))
                assert len(prefix) + len(lines[0]) <= 88, (e.name, held, lines[0])
                assert all(len(line) <= 88 for line in lines[1:]), (e.name, held)
                assert any(line.strip().startswith(("ts.", "traj")) for line in lines[1:])
                assert e.name in text


class TestTheLiveDoorsUseTheBuilder:
    """The builder is only worth having if the SHIPPED functions call it.

    ``_discovery.teach`` existed and was correct before this gate, while the
    front doors kept their own hand-written strings — so the contract's §5.6
    text was reachable from a test and from nowhere a user types.  Measured on
    the pre-gate tree::

        >>> ts.analysis.kaplan_yorke_dimension(ts.systems.Lorenz())
        InvalidInputError: kaplan_yorke_dimension() expects measured data, not a
        System (got Lorenz). It takes an already computed Lyapunov spectrum:

    — the exact wrong clause §5.6 names (this analysis does not take measured
    data), from a door whose builder already knew better.
    """

    #: ``(call, name, class name, held)`` — one live door per wanted-subject
    #: group, both held kinds.  Each is callable with a single wrong argument,
    #: so the guard fires before any binding error can.
    LIVE_DOORS = (
        ("correlation_dimension", "Lorenz", "flow"),
        ("rqa", "Henon", "map"),
        ("optimal_delay", "Lorenz", "flow"),
        ("estimate_period", "Lorenz", "flow"),
        ("recurrence_matrix", "Lorenz", "flow"),
        ("kaplan_yorke_dimension", "Lorenz", "flow"),
        ("basin_entropy", "Lorenz", "flow"),
        ("wada_property", "Lorenz", "flow"),
        ("uncertainty_exponent", "Lorenz", "flow"),
        ("resilience", "Lorenz", "flow"),
    )

    @pytest.mark.parametrize(("name", "cls", "held"), LIVE_DOORS)
    def test_a_live_door_raises_exactly_what_the_builder_renders(self, name, cls, held):
        subject = getattr(ts.systems, cls)()
        with pytest.raises(ts.InvalidInputError) as excinfo:
            getattr(ts.analysis, name)(subject)
        assert str(excinfo.value) == str(_discovery.wrong_subject(name, cls, held))

    @pytest.mark.parametrize("name", ["lyapunov_spectrum", "fixed_points", "periodic_orbits"])
    def test_a_system_first_door_reached_with_a_trajectory(self, name):
        traj = ts.systems.Lorenz().run(final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0])
        with pytest.raises(ts.InvalidInputError) as excinfo:
            getattr(ts.analysis, name)(traj)
        assert str(excinfo.value) == str(_discovery.wrong_subject(name, "Trajectory", "data"))

    def test_a_result_first_door_never_claims_to_want_measured_data(self):
        """The named live bug, asserted on the shipped call rather than the builder."""
        for name in ("kaplan_yorke_dimension", "basin_entropy", "resilience", "wada_property"):
            with pytest.raises(ts.InvalidInputError) as excinfo:
                getattr(ts.analysis, name)(ts.systems.Lorenz())
            text = str(excinfo.value)
            assert "needs a result from another analysis" in " ".join(text.split())
            assert "expects measured data" not in text

    def test_a_bare_array_still_gets_the_data_driven_sibling(self):
        """``has_system=False`` must not swallow the twin that answers the caller.

        A user holding only a measurement is exactly who needs
        ``lyapunov_from_data``; sending them to a listing instead is the one
        case where the shortest message is the least useful one.
        """
        with pytest.raises(ts.InvalidInputError) as excinfo:
            ts.analysis.lyapunov_spectrum(np.zeros(64))
        text = str(excinfo.value)
        assert "ts.analysis.lyapunov_from_data(traj)" in text
        assert "ts.analysis.find(traj)" in text

    def test_the_find_line_states_the_count_that_call_will_print(self):
        """A message that quotes a number must agree with the call it recommends."""
        for held, subject in (("flow", ts.systems.Lorenz()), ("map", ts.systems.Henon())):
            line = _discovery.find_line(held)
            quoted = int(line.split("# all ")[1].split()[0])
            assert quoted == len(ts.analysis.find(subject))
        traj = ts.systems.Lorenz().run(final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0])
        line = _discovery.find_line("data")
        assert int(line.split("# all ")[1].split()[0]) == len(ts.analysis.find(traj))


class TestNoRemovedNameIsReadAsAString:
    """CONTRACT §9.4 rules 1 and 3, as an executable sweep over this package.

    Renaming a ClassVar or removing a method orphans every ``getattr(x, "old",
    default)`` that read it — and because the default is a *legal* value for all
    of them, nothing raises and the answer quietly changes.  Two live instances
    were found by this sweep and fixed:

    * ``analysis/orbits/poincare.py`` read ``type(system).default_ic`` (renamed
      ``_default_ic`` in round 1), so a seeded section overrode every system's
      declared initial state with a random draw;
    * ``analysis/basins/attractors.py`` read ``getattr(system, "is_discrete",
      False)`` at three sites, which decides whether the basin FSM advances a
      map by one iterate or by ``dt``.
    """

    #: Public names v6 removed from systems / trajectories, with the v6 reading.
    REMOVED = {
        "is_discrete": 'family == "map" (or the private _is_discrete on a wrapper)',
        "default_ic": "_default_ic",
        "resolve_ic": "_resolve_ic",
        "ic_generator": "_ic_generator",
        "integrate": "run",
        "iterate": "run",
        "trajectory": "run",
    }

    def test_no_analysis_module_reaches_for_a_removed_name(self):
        import ast
        import pathlib

        root = pathlib.Path(ts.analysis.__file__).parent
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                # ``getattr(obj, "<removed>", ...)`` — the silent shape.
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in self.REMOVED
                ):
                    offenders.append(
                        f"{path.name}:{node.lineno} getattr(…, {node.args[1].value!r})"
                    )
                # ``obj.<removed>`` / ``obj.<removed>(...)`` — the loud shape.
                if isinstance(node, ast.Attribute) and node.attr in self.REMOVED:
                    value = node.value
                    # A result's OWN field (``self.is_discrete`` on GALIResult) is
                    # not a system read, and is not what this gate is about.
                    if isinstance(value, ast.Name) and value.id in ("self", "cls"):
                        continue
                    offenders.append(f"{path.name}:{node.lineno} .{node.attr}")
        assert not offenders, "removed names still read in tsdynamics.analysis:\n  " + "\n  ".join(
            offenders
        )


class TestFindMatchesIntentNotOnlyWords:
    """A question asked in the reader's words must still land.

    ``find`` scores against the registry's own vocabulary, so a field that has
    two names for everything — and a reader who is an engineer, not a
    dynamicist — falls through it.  Measured before this widening over 128
    plausible questions, **20** returned nothing while the analysis that answers
    them was registered the whole time.
    """

    #: question -> the analysis that must appear in the top 3.  Grouped by whose
    #: vocabulary it is.  A row here is a claim that the library HAS an answer.
    INTENT = {
        # the engineer's words
        "robust": "resilience",
        "robustness": "resilience",
        "resilient": "resilience",
        "safety margin": "resilience",
        "design margin": "resilience",
        "how much disturbance can it take": "resilience",
        "failure": "resilience",
        "will it withstand a shock": "resilience",
        "buffer": "resilience",
        "tolerance to shocks": "resilience",
        # the field's words
        "critical transition": "tipping_points",
        "early warning": "tipping_points",
        "regime shift": "tipping_points",
        "hysteresis": "continuation",
        "crisis": "tipping_points",
        "irreversible": "tipping_points",
        "multistability": "attractors",
        "bistability": "attractors",
        "alternative stable states": "attractors",
        "sensitive dependence": "lyapunov_spectrum",
        "predictability horizon": "lyapunov_spectrum",
        "forecast": "lyapunov_spectrum",
        "intermittency": "rqa",
        "laminar": "rqa",
        "quasiperiodic": "gali",
        "torus": "gali",
        "separatrix": "basins",
        "watershed": "basins",
        "riddled": "wada_property",
        "reconstruct": "embed",
        "state space reconstruction": "embed",
        "takens": "embed",
    }

    @pytest.mark.parametrize(("question", "wanted"), sorted(INTENT.items()))
    def test_the_question_reaches_the_analysis_that_answers_it(self, question, wanted):
        top = [f.__name__ for f in ts.analysis.find(question)][:3]
        assert wanted in top, f"{question!r} -> {top}"

    #: Questions the library genuinely cannot answer.  They must stay EMPTY: a
    #: synonym row invented for one of these would answer a question nobody here
    #: can, which is the failure mode the whole curation exists to prevent.
    NO_ANSWER = ("synchronisation", "resonance", "entrainment", "noise floor", "stiffness")

    @pytest.mark.parametrize("question", NO_ANSWER)
    def test_a_question_with_no_answer_here_stays_empty(self, question):
        assert list(ts.analysis.find(question)) == []

    def test_a_dead_end_says_where_to_look_instead(self):
        out = repr(ts.analysis.find("synchronisation"))
        assert "nothing matches" in out
        assert "ts.analysis.find(subject)" in out
