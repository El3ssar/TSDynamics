"""The visibility ruling, pinned on the **result** layer (v6 round 9, contract §11).

The owner's order was one sentence: sweep every class and every returned object
and hide what a user never types, "so we dont get people overwhelmed with stuff
they wont use directly anyways."  On the result layer that took the listing from
**530 names to 492** across the 32 classes — the extremes being ``CountResult``
20 → 8 (it is an ``int``, and eleven of its twenty names were ``bit_length`` and
friends), ``LyapunovFromData`` 33 → 28 and ``DimensionResult`` 27 → 23.

The sweep first reached 452, and then gave 40 names back, because hiding has a
second edge the first pass cut itself on: **a name the library hands the reader
must be a name the reader can then use.**  Two rules, both now gated below, and
every restoration additive — no listing lost a name when they landed:

*A name the repr prints must be listed* (+32).  ``RQAResult`` printed
``DET = 0.999 · LAM = 0.993 · L_max = 678 · ENTR = 3.726`` and tab-completed none
of the four; ``WindowedRQA`` printed ``DET`` and did not even *resolve* it; and
``DimensionResult`` printed ``15 fit pts, R² = 0.99992`` while listing only the
second of the pair a reader judges a fit by.

*A name ``to_dict`` promises must exist* (+8).  ``to_dict(full=True)`` is the
door every result's ``AttributeError`` recommends, so a key there is a claim.
Three were false, each a casing nobody would guess: ``BasinEntropy`` emitted
``Sb``/``Sbb`` carrying only ``sb``/``sbb``, ``UncertaintyExponent`` emitted
``D0`` carrying ``boundary_dimension``, and ``LyapunovSpectrum`` emitted
``zero_tolerance`` — the threshold its chaos verdict turns on — from a private
property.

The whole sweep rests on one rule, and this file is what keeps it honest:

    **Hiding is a DISCOVERY change, never a REACHABILITY change.**

Every name the ruling took off a result's listing is still bound, still
resolves, still exports through :meth:`~tsdynamics.analysis.results.AnalysisResult.to_dict`,
and every existing caller keeps working.  That is what lets the sweep satisfy the
owner's other standing order — *"simple must NOT mean that it hinders the
customizability of the library"* — at a measured capability cost of zero.

So the gates come in two halves, and both are load-bearing:

**The pin** (:data:`LISTED`) is a literal table, one row per result class, of
exactly what ``result.<TAB>`` shows.  It has to be literal: a listing derived
from the code cannot pin the code.  A new public name on a result, or a hide
that is removed, fails here and has to be argued for in the diff rather than
appearing in a user's REPL unannounced.

**The capability half** is *derived* — from ``_HIDDEN_ATTRIBUTES``, from
:func:`dataclasses.fields`, from the MRO — never from a second literal list.
That is the round-9 lesson (``Geometry.parts`` was hidden while two of
``Geometry``'s own errors handed it back): a requirement asserted as a frozen
list goes stale silently, so anything that must stay *true of every name* is
computed from the source.

These gates fail on the pre-ruling code: ``LISTED`` rejects all 32 rows (each
carried ``build_meta``), and the three retired-spelling cases each resolved to a
live attribute instead of raising.
"""

from __future__ import annotations

import re
from dataclasses import fields
from typing import Any

import pytest
from _result_fixtures import build

from tsdynamics.analysis._result import AnalysisResult
from tsdynamics.analysis.sampling.sagitta import SagittaDt

RESULTS = build()

#: What ``result.<TAB>`` lists, per class — the contract §11 surface, measured.
#:
#: **Do not nudge a row to make a test pass.**  Re-measure it, and justify the
#: move: a name appearing here is a name every reader of that result now has to
#: read past, and a name leaving is one a reader can no longer discover.
LISTED: dict[str, str] = {
    "AnalysisResult": "headline meta overlay_on plot to_dict to_frame verdict",
    "ArrayResult": "headline meta overlay_on plot to_dict to_frame values verdict",
    "Attractor": "cells center dim headline id meta overlay_on plot points to_dict to_frame verdict",
    "AttractorSet": "attractors by_id cells centers details diverged headline ids match meta "
    "overlay_on plot to_dict to_frame verdict",
    "BasinEntropy": "Sb Sbb fractal_boundary headline meta n_boundary_boxes n_boxes overlay_on "
    "plot sb sbb to_dict to_frame verdict",
    "BasinFractions": "attractors by_id diverged dominant fractions headline ids meta n overlay_on "
    "plot standard_error to_dict to_frame verdict",
    "BasinsResult": "attractors diverged_fraction fractions grid headline labels meta n_attractors "
    "overlay_on plot shape to_dict to_frame verdict",
    "CollectionResult": "by_id details headline meta overlay_on plot to_dict to_frame verdict",
    "ContinuationResult": "attractors centers diverged fractions headline ids meta overlay_on "
    "param plot to_dict to_frame values verdict",
    "CountResult": "headline meta overlay_on plot to_dict to_frame value verdict",
    "DimensionResult": "abscissa dimension fit_region headline intercept kind local_slopes meta "
    "n_fit ordinate overlay_on plot q r_squared runaway scaling_window stderr to_dict to_frame "
    "trusted unembedded value verdict",
    "Embedding": "headline meta overlay_on plot to_dict to_frame values verdict",
    "EmbeddingDimension": "afn_e1 afn_e2 delay dimension dims fnn_fraction headline meta method "
    "overlay_on plot saturated to_dict to_frame verdict",
    "ExpansionEntropyResult": "applicable chaotic entropy fit_region headline intercept "
    "local_slopes meta n_fit n_samples n_survivors ordinate overlay_on plot r_squared runaway "
    "scaling_window stderr times to_dict to_frame trusted value verdict",
    "FixedPoint": "continuous eigenvalue_plane eigenvalues headline meta overlay_on plot stable "
    "to_dict to_frame verdict x",
    "FixedPointSet": "by_id details eigenvalue_plane eigenvalues headline is_stable meta n_stable "
    "n_unstable overlay_on plot points stable to_dict to_frame unstable verdict",
    "GALIResult": "applicable chaotic decay_rate final headline is_chaotic is_discrete k meta "
    "overlay_on plot times to_dict to_frame values verdict",
    "LyapunovFromData": "chaotic delay divergence embedding_dim fit_efoldings fit_region "
    "headline independent_windows intercept local_slopes lyapunov meta method n_fit n_reference "
    "overlay_on plot r_squared runaway scaling_window stderr theiler times to_dict to_frame "
    "trusted value verdict",
    "LyapunovSpectrum": "chaotic exponents headline kaplan_yorke meta n_positive overlay_on plot "
    "regime to_dict to_frame unbounded values verdict zero_tolerance",
    "MutualInformation": "headline meta optimal_lag overlay_on plot to_dict to_frame values verdict",
    "OrbitDiagram": "bifurcation_points components equilibria flat headline meta overlay_on param "
    "periods plot points to_dict to_frame values verdict",
    "OrbitSet": "by_id details headline is_stable meta multipliers n_stable n_unstable overlay_on "
    "periods plot points stable to_dict to_frame unstable verdict",
    "PeriodicOrbit": "continuous eigenvalue_plane headline meta multipliers overlay_on period plot "
    "points residual stable to_dict to_frame verdict",
    "RQAResult": "DET DIV ENTR L LAM L_max RR TT V_max applicable avg_diagonal_length "
    "determinism deterministic diagonal_entropy diagonal_lengths divergence epsilon headline "
    "laminarity max_diagonal_length max_vertical_length meta overlay_on plot recurrence_rate "
    "size theiler_window to_dict to_frame trapping_time verdict vertical_lengths",
    "RecurrenceMatrix": "epsilon headline matrix meta overlay_on plot recurrence_rate size "
    "theiler_window to_dict to_frame toarray verdict",
    "ReturnMap": "cobweb current flat headline kind meta overlay_on plot successor times to_dict "
    "to_frame values verdict",
    "ScalarResult": "chaotic headline meta overlay_on plot to_dict to_frame value verdict",
    "ScalingResult": "abscissa fit_region headline intercept local_slopes meta n_fit ordinate "
    "overlay_on plot r_squared runaway scaling_window stderr to_dict to_frame trusted value "
    "verdict",
    "UncertaintyExponent": "D0 alpha applicable boundary_dimension contradicted_by_basin_entropy "
    "epsilons final_state_sensitive headline meta overlay_on plot r_squared resolved "
    "state_dimension to_dict to_frame uncertain_fractions verdict",
    "WadaResult": "W applicable fractions headline meta n_basins n_boundary_cells overlay_on plot "
    "radii to_dict to_frame verdict wada",
    "WindowedRQA": "DET DIV ENTR L LAM L_max RR TT V_max avg_diagonal_length centers details "
    "determinism diagonal_entropy divergence headline laminarity max_diagonal_length "
    "max_vertical_length measure measures meta overlay_on plot recurrence_rate results step "
    "to_dict to_frame trapping_time verdict window",
    "ZeroOneResult": "chaotic distribution headline meta overlay_on p plot q to_dict to_frame "
    "value verdict",
}

#: The four names **every** result carries, whatever it measured.  A user learns
#: them once and they work on all 32, which is the only reason a reduced listing
#: is navigable at all.
UNIVERSAL = frozenset({"plot", "to_dict", "to_frame", "meta"})

#: The spellings round 9 **retired** — the only names in the result layer that
#: stopped resolving — mapped to the replacement the ``AttributeError`` must name.
#: Two were measured-identical duplicates of a field; ``f`` was a one-letter
#: field renamed to say what it holds.
RETIRED = {
    "UncertaintyExponent": ("f", "result.uncertain_fractions"),
    "ScalingResult": ("fit_slice", "result.fit_region"),
    "ExpansionEntropyResult": ("log_growth", "result.ordinate"),
}


def _library_subclasses(cls: type) -> set[type]:
    """Every transitive **library** subclass of ``cls``.

    Restricted to ``tsdynamics.*``: other test modules define throwaway
    subclasses to probe the machinery, and ``__subclasses__()`` sees every one of
    them as soon as its module is imported into the same worker.
    """
    found: set[type] = set()
    for sub in cls.__subclasses__():
        if sub.__module__.startswith("tsdynamics."):
            found.add(sub)
        found |= _library_subclasses(sub)
    return found


def _listed(obj: Any) -> set[str]:
    """The public names ``dir(obj)`` shows — what a reader actually tab-completes."""
    return {n for n in dir(obj) if not n.startswith("_")}


def _declared_names(obj: Any) -> set[str]:
    """Every name this result's class or instance genuinely carries.

    Used to skip hides a *sibling* class declared: ``_HIDDEN_ATTRIBUTES`` is
    unioned down the MRO, so ``ScalingResult``'s hides ride along on classes that
    never had an ``x`` to hide.
    """
    return set(dir(type(obj))) | {f.name for f in fields(obj)}


RESULT_IDS = sorted(RESULTS)


@pytest.fixture(params=RESULT_IDS)
def result(request: pytest.FixtureRequest) -> Any:
    """One realistic instance of each of the 32 result classes."""
    return RESULTS[request.param]


# --------------------------------------------------------------------------
# The pin: what a reader sees
# --------------------------------------------------------------------------


class TestTheListingIsTheContract:
    """``result.<TAB>`` is a reviewed surface, not whatever the fields happen to be."""

    def test_a_result_lists_exactly_the_names_the_ruling_left_it(self, result: Any) -> None:
        """Exact set equality — the surface can neither regrow nor silently shrink."""
        name = type(result).__name__
        assert _listed(result) == set(LISTED[name].split()), (
            f"{name}.<TAB> moved.  Re-measure the row in LISTED and justify the change; "
            f"a name added here is one every reader of this result must now read past."
        )

    def test_the_table_covers_every_result_class(self) -> None:
        """A new result class cannot ship without a reviewed row.

        Derived from ``AnalysisResult.__subclasses__()``, so the table cannot
        quietly fall behind the code it pins.
        """
        live = {c.__name__ for c in _library_subclasses(AnalysisResult)} | {"AnalysisResult"}
        assert live == set(LISTED), (
            f"missing rows: {sorted(live - set(LISTED))}; stale rows: {sorted(set(LISTED) - live)}"
        )

    def test_the_whole_result_layer_lists_four_hundred_and_ninety_two_names(self) -> None:
        """The headline number, pinned once.

        530 before the ruling, 492 after: 78 names hidden and 40 given back — 32 to
        the printed-name rule and 8 to the rule that ``to_dict`` may not promise a
        name the object lacks.  Every one of the 78 is still bound.  A one-line
        reminder that this file is about a *total*, not a taste.
        """
        assert sum(len(_listed(r)) for r in RESULTS.values()) == 492

    def test_every_result_carries_the_four_names_learned_once(self, result: Any) -> None:
        """``plot`` / ``to_dict`` / ``to_frame`` / ``meta`` are on all 32 (KEEP)."""
        assert _listed(result) >= UNIVERSAL

    def test_the_listing_is_sorted_and_free_of_duplicates(self, result: Any) -> None:
        """``dir()`` is consumed by a human and by autocompletion; both want order."""
        listing = dir(result)
        assert listing == sorted(listing)
        assert len(listing) == len(set(listing))


# --------------------------------------------------------------------------
# The capability half: hiding cost nothing (G1)
# --------------------------------------------------------------------------


class TestHidingIsADiscoveryChangeOnly:
    """Every hidden name is still bound, still readable, still exported."""

    def test_every_hidden_name_still_resolves(self, result: Any) -> None:
        """The whole ruling's premise, checked per class over ``_HIDDEN_ATTRIBUTES``."""
        carried = _declared_names(result)
        for name in sorted(type(result)._HIDDEN_ATTRIBUTES & carried):
            getattr(result, name)  # must not raise

    def test_every_hidden_name_is_actually_off_the_listing(self, result: Any) -> None:
        """A hide that did not take is worse than none — it reads as done."""
        assert not (type(result)._HIDDEN_ATTRIBUTES & _listed(result))

    def test_every_hidden_field_still_exports_through_to_dict(self, result: Any) -> None:
        """``to_dict`` is the escape hatch that makes a hidden field recoverable.

        A user who cannot see ``box_size`` on the listing must still find it in
        the export, or the hide became a deletion.
        """
        exported = result.to_dict(full=True)
        hidden_fields = {f.name for f in fields(result)} & type(result)._HIDDEN_ATTRIBUTES
        assert hidden_fields <= set(exported)

    def test_a_subclass_inherits_every_hide_its_parents_declared(self, result: Any) -> None:
        """Declaring ``_HIDDEN_ATTRIBUTES`` in a class body must not un-hide the parents'.

        A plain class-body assignment *shadows*, so without the union in
        ``__init_subclass__`` a subclass hiding one name of its own would restore
        every name its ancestors hid — silently, since nothing raises.
        """
        for base in type(result).__mro__[1:]:
            declared = frozenset(base.__dict__.get("_HIDDEN_ATTRIBUTES", ()))
            assert declared <= type(result)._HIDDEN_ATTRIBUTES

    def test_hasattr_still_answers_false_for_a_name_that_is_not_there(self, result: Any) -> None:
        """Curation must not turn a miss into an exception."""
        assert not hasattr(result, "definitely_not_a_result_attribute")

    def test_the_sagitta_selector_hides_its_scratch_and_keeps_it_reachable(self) -> None:
        """``SagittaDt`` is not an ``AnalysisResult``, so it carries its own copy.

        The rule still binds: ``searched_ms`` (every candidate stride the search
        tried) is off the listing and still readable.
        """
        assert "searched_ms" in {f.name for f in fields(SagittaDt)}
        assert "searched_ms" in SagittaDt._HIDDEN_ATTRIBUTES
        assert "searched_ms" not in dir(SagittaDt(1.0, 1, 0.0, (), 0.5, 1e-3, (), ""))
        assert SagittaDt(1.0, 1, 0.0, (), 0.5, 1e-3, (7,), "").searched_ms == (7,)


# --------------------------------------------------------------------------
# KEEP + DOC: a listed name must say what it is
# --------------------------------------------------------------------------


class TestEveryListedNameSaysWhatItIs:
    """A reduced listing is only an improvement if what survives is legible."""

    def test_every_listed_callable_or_property_is_documented(self, result: Any) -> None:
        """Derived from the source, never from a second literal list.

        A dataclass *field* is exempt here — it has no ``__doc__`` slot of its
        own, and is documented in its class's ``Attributes`` section instead
        (pinned by the next test for ``values``, the one name that means three
        different things across the layer).
        """
        import inspect

        cls = type(result)
        undocumented = []
        for name in sorted(_listed(result)):
            attr = inspect.getattr_static(cls, name, None)
            if attr is None:  # a dataclass field, not a class attribute
                continue
            doc = attr.__doc__ if isinstance(attr, property) else getattr(attr, "__doc__", None)
            if not (doc or "").strip():
                undocumented.append(name)
        assert not undocumented, f"{cls.__name__}: undocumented listed names {undocumented}"

    def test_values_is_documented_wherever_it_is_listed(self, result: Any) -> None:
        """``.values`` means three different things, so each class must say which.

        It is the measurement on a spectrum, the *swept parameter axis* on an
        orbit diagram and a continuation, and the map's *input* on a return map —
        a reader who learned one of those on one result would be wrong on the
        other two.
        """
        import inspect

        if "values" not in _listed(result):
            pytest.skip("this result carries no `values`")
        doc = inspect.getdoc(type(result)) or ""
        assert "values" in doc, f"{type(result).__name__} lists `values` and never documents it"


# --------------------------------------------------------------------------
# The three names that genuinely stopped resolving
# --------------------------------------------------------------------------


class TestARetiredSpellingIsAnsweredByName:
    """A removed name is answered with the replacement, not with a ranked guess."""

    @pytest.mark.parametrize("cls_name", sorted(RETIRED))
    def test_the_message_names_the_line_that_works(self, cls_name: str) -> None:
        result = RESULTS[cls_name]
        old, replacement = RETIRED[cls_name]
        with pytest.raises(AttributeError) as excinfo:
            getattr(result, old)
        message = str(excinfo.value)
        assert old in message and replacement in message

    @pytest.mark.parametrize("cls_name", sorted(RETIRED))
    def test_the_replacement_the_message_names_actually_resolves(self, cls_name: str) -> None:
        """A remedy line a library hands back must run on the object that was held."""
        result = RESULTS[cls_name]
        _, replacement = RETIRED[cls_name]
        getattr(result, replacement.removeprefix("result."))

    @pytest.mark.parametrize("cls_name", sorted(RETIRED))
    def test_the_retired_name_is_off_the_listing_too(self, cls_name: str) -> None:
        old, _ = RETIRED[cls_name]
        assert old not in _listed(RESULTS[cls_name])


class TestAPrintedNameIsACompletableName:
    """Whatever a result's repr names, ``dir()`` must list — contract §11.

    The corollary the visibility sweep cut itself on twice.  ``Geometry.parts``
    was the viz instance (hidden, while two of ``Geometry``'s own errors ended
    *"iterate g.parts"*); on this layer it was RQA, whose headline is written in
    the abbreviations every RQA paper uses::

        RQAResult  DET = 0.999 · LAM = 0.993 · L_max = 678 · ENTR = 3.726

    All four resolved.  None was in ``dir()``.  A reader who types what they just
    read succeeds; a reader who reaches for TAB is told the quantity does not
    exist — and ``ENTR`` → ``diagonal_entropy`` is not a guess anyone should have
    to make.  So the rule is mechanical: **if the repr prints ``NAME = value``,
    ``NAME`` is in the listing.**
    """

    #: ``NAME = value`` / ``NAME ∈ [...]`` / ``NAME ≈ value`` as a repr writes them.
    _ASSIGNMENT = re.compile(r"(?:^|[\s·(])([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|∈|≈)")

    def test_every_name_the_repr_assigns_to_is_listed(self, result: Any) -> None:
        """Sweeps all 32 reprs; no allow-list, so a new result is covered on arrival.

        Scoped to names the result **actually carries**, because a repr also
        writes mathematical symbols for its quantity (``D_corr``, ``D_KY``,
        ``H0``, ``GALI_2``, ``K``), the swept parameter's own name (``r``,
        ``delta``), and settings (``m``, ``τ``, ``l_min``).  Those are not
        attribute names and never were; requiring them to resolve would be
        requiring the library to name a quantity twice.

        What this catches is the opposite and realer case: a name that *is* an
        attribute, printed by the repr, and kept off the listing — ``RQAResult``
        printing ``DET = 0.999`` while ``dir()`` offered only ``determinism``.
        The companion case, a name ``to_dict`` promises but the object does not
        have, is :meth:`test_every_to_dict_key_resolves_as_an_attribute` below.
        """
        listed, text = _listed(result), repr(result)
        printed = {m.group(1) for m in self._ASSIGNMENT.finditer(text)}
        carried = {n for n in printed if hasattr(result, n)}
        missing = sorted(carried - listed)
        assert not missing, (
            f"{type(result).__name__} prints {missing} and resolves "
            f"{'it' if len(missing) == 1 else 'them'}, but lists "
            f"{'it' if len(missing) == 1 else 'them'} in neither dir() nor "
            f"_extra_attribute_names.  Declare there (for a name served by the "
            f"class's own __getattr__, which dir cannot see) or drop it from "
            f"_HIDDEN_ATTRIBUTES — a name the library teaches it must complete."
        )

    #: ``to_dict`` keys that are deliberately not attributes, with the reason.
    #: **This table must not grow.**  ``unit`` is provenance: it is read from
    #: ``meta["unit"]``, describes how to *read* the number rather than being a
    #: measurement of the subject, and is already reachable as ``r.meta["unit"]``.
    _TO_DICT_ONLY: dict[str, str] = {"unit": 'provenance, reachable as meta["unit"]'}

    def test_every_to_dict_key_resolves_as_an_attribute(self, result: Any) -> None:
        """``to_dict(full=True)`` may not promise a name the object does not have.

        ``to_dict`` is the documented "everything it knows" door — every result's
        ``AttributeError`` ends by recommending it — so a key there is a claim
        that the result carries that quantity.  Three claims were false, and all
        three were a **casing** difference nobody would guess: ``BasinEntropy``
        emitted ``Sb``/``Sbb`` (Daza's casing, and what its own repr prints) while
        carrying only ``sb``/``sbb``; ``UncertaintyExponent`` emitted ``D0``
        (Grebogi's) while carrying ``boundary_dimension``; ``LyapunovSpectrum``
        emitted ``zero_tolerance``, the threshold its chaos verdict turns on,
        from a **private** property.

        This is the gate that would have caught them, and it needs no judgement
        call about which printed symbols are names — ``to_dict`` already decided.
        """
        try:
            keys = set(result.to_dict(full=True))
        except Exception as exc:  # pragma: no cover - a result that cannot serialize
            pytest.fail(f"{type(result).__name__}.to_dict(full=True) raised {exc!r}")
        missing = sorted(k for k in keys if not hasattr(result, k) and k not in self._TO_DICT_ONLY)
        assert not missing, (
            f"{type(result).__name__}.to_dict(full=True) emits {missing}, which "
            f"the object does not carry.  Add the property (usually an alias of "
            f"a differently-cased field), or stop emitting the key."
        )

    def test_the_two_rqa_results_answer_to_the_same_nine_names(self) -> None:
        """``RQAResult`` and ``WindowedRQA`` measure the same nine quantities.

        They are produced by the same estimator over one window and many, so a
        name that works on one must work on the other.  ``WindowedRQA`` used to
        print ``DET ∈ [0.392, 0.722]`` while ``w.DET`` raised ``AttributeError``
        outright — the same nine measures, reachable under one spelling on one
        class and the other spelling on the other.
        """
        single, windowed = RESULTS["RQAResult"], RESULTS["WindowedRQA"]
        shared = set(type(single)._PRINTED_ABBREVIATIONS) - {"DIV", "L"} | {"DIV", "L"}
        for short in shared:
            assert hasattr(single, short), f"RQAResult lost {short}"
            assert short in dir(single), f"RQAResult hides {short}"
            assert hasattr(windowed, short), f"WindowedRQA cannot resolve {short}"
            assert short in dir(windowed), f"WindowedRQA hides {short}"

    def test_the_abbreviations_have_exactly_one_source_of_truth(self) -> None:
        """The lookup map, the listing and ``WindowedRQA`` all derive from one table.

        Three hand-maintained copies of the same nine pairs is how two classes
        drift into answering to different names for one quantity.
        """
        from tsdynamics.analysis.recurrence import windowed as w_mod
        from tsdynamics.analysis.recurrence.rqa import RQAResult

        printed = RQAResult._PRINTED_ABBREVIATIONS
        assert {
            s.replace("_", "").upper(): long for s, long in printed.items()
        } == RQAResult._ABBREVIATIONS
        assert set(RQAResult._extra_attribute_names) == set(printed)
        assert set(w_mod._ABBREVIATIONS_FOR_MEASURES) <= set(printed)
