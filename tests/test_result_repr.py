"""The repr of an analysis result **is** the answer (v6, contract §4.2–§4.4).

Before v6 the readable text lived in ``summary()`` — which nothing advertised,
no docstring pointed at, and no REPL ever calls — while ``__repr__``, the thing a
console and a notebook actually show, gave a constructor-shaped one-liner.  v6
deletes ``summary()`` and makes the repr what it printed.

These gates pin the user-facing consequences over **every** result class, not a
sample: the class list is walked from ``AnalysisResult.__subclasses__()``, so a
new result cannot ship without an entry in ``tests/_result_fixtures.py`` and
without satisfying the shape below.

They fail on the pre-v6 code: every ``assert "(" not in first_word``-style shape
check rejects ``LyapunovSpectrum([0.916, ...])``, ``summary`` is asserted absent
while it existed, and the three Lyapunov-verdict cases each returned the wrong
word (``Lorenz`` T=20 → *hyperchaotic*, ``LotkaVolterra`` → *chaotic*,
``HenonHeiles`` → *hyperchaotic*).
"""

from __future__ import annotations

import numpy as np
import pytest
from _result_fixtures import build, wada_applicable

from tsdynamics.analysis._result import AnalysisResult
from tsdynamics.analysis._result_base import _MAX_DETAILS, _MAX_ITEMS
from tsdynamics.analysis.lyapunov import LyapunovSpectrum

RESULTS = build()
NAMES = sorted(RESULTS)


def _all_subclasses(cls: type) -> set[type]:
    """Every (transitive) **library** subclass of ``cls``.

    Restricted to ``tsdynamics.*`` because other test modules define throwaway
    subclasses to probe the machinery, and ``__subclasses__()`` sees them all as
    soon as their module has been imported into the same worker.
    """
    found: set[type] = set()
    for sub in cls.__subclasses__():
        if sub.__module__.startswith("tsdynamics."):
            found.add(sub)
        found |= _all_subclasses(sub)
    return found


# ---------------------------------------------------------------------------
# Coverage: the fixture table IS the class list
# ---------------------------------------------------------------------------


def test_every_result_class_has_a_fixture():
    """A new result class cannot ship without a rendered example here."""
    live = {c.__name__ for c in _all_subclasses(AnalysisResult)} | {"AnalysisResult"}
    assert live == set(NAMES), (
        f"missing fixtures: {sorted(live - set(NAMES))}; "
        f"stale fixtures: {sorted(set(NAMES) - live)}"
    )


# ---------------------------------------------------------------------------
# The repr shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_repr_is_prose_not_a_constructor_call(name):
    """No result reprs as ``Name(field=..., field=...)`` any more."""
    text = repr(RESULTS[name])
    head = text.splitlines()[0]
    assert head, f"{name} reprs as an empty line"
    assert not head.rstrip().endswith(")") or " " in head, (
        f"{name} still reprs as a constructor call: {head!r}"
    )
    assert "=None" not in head and "field=" not in head


@pytest.mark.parametrize("name", NAMES)
def test_repr_first_line_is_str(name):
    """``str(result)`` is the headline — the repr's first line, verbatim."""
    result = RESULTS[name]
    if name == "CountResult":
        # The one documented exception: a count IS its integer, because
        # ``f"tau={c}"`` must print ``tau=9``.
        assert str(result) == "9"
        return
    assert str(result) == repr(result).splitlines()[0]


@pytest.mark.parametrize("name", NAMES)
def test_repr_never_wraps_and_never_dumps(name):
    """Every line is one line, indented four spaces, and the body is bounded."""
    lines = repr(RESULTS[name]).splitlines()
    assert lines, name
    assert not lines[0].startswith(" "), f"{name}: headline is indented"
    for line in lines[1:]:
        assert line.startswith("    "), f"{name}: body line is not indented: {line!r}"
    assert len(lines) - 1 <= _MAX_DETAILS + _MAX_ITEMS + 1, f"{name}: repr dumps {len(lines)} lines"


@pytest.mark.parametrize("name", NAMES)
def test_repr_names_what_was_measured(name):
    """The headline opens with the class name, or the analysis for a generic wrapper."""
    result = RESULTS[name]
    head = repr(result).splitlines()[0]
    anonymous = type(result).__name__ in AnalysisResult._ANONYMOUS_RESULT_TYPES
    expected = str(result.meta["analysis"]) if anonymous else type(result).__name__
    assert head.startswith(expected), f"{name}: headline {head!r} does not open with {expected!r}"


@pytest.mark.parametrize("name", NAMES)
def test_notebook_and_console_show_the_same_text(name):
    """``_repr_html_`` is the repr, escaped — the two renderings cannot drift."""
    import html

    result = RESULTS[name]
    assert html.escape(repr(result)) in result._repr_html_()


@pytest.mark.parametrize("name", NAMES)
def test_summary_is_gone(name):
    """``summary()`` is deleted; the repr replaced it (contract §4.2 rule 3)."""
    assert not hasattr(RESULTS[name], "summary")


@pytest.mark.parametrize("name", ["FixedPointSet", "OrbitSet", "AttractorSet"])
def test_item_lines_use_the_compact_form_not_the_member_headline(name):
    """A member's row must not repeat the class name and subject the header gave.

    Members override ``_as_item()``; a collection that renders ``str(item)``
    instead prints the member's full headline on every row — the class name and
    the system, twice per line, for the whole list.

    The rows are built from the **records** (``.details``), not from what ``[]``
    hands back — indexing gives numbers, and an array has no readout.
    """
    result = RESULTS[name]
    member = result.details[0]
    for line in repr(result).splitlines()[1:]:
        assert type(member).__name__ not in line, f"{name}: item row repeats the class name"
    assert member._as_item() in repr(result)


@pytest.mark.parametrize("name", NAMES)
def test_a_default_constructed_result_still_reprs(name):
    """A repr must never be the thing that raises in a console.

    Every field on every result has a default (except ``RQAResult``'s, which are
    all required), so a half-built result is constructible — and was reachable:
    ``RecurrenceMatrix()`` carries ``matrix=None`` and its ``size`` raised
    ``AttributeError``, while a ``ScalingResult`` whose ``fit_region`` does not
    fit its curve raised ``IndexError``.
    """
    cls = type(RESULTS[name])
    try:
        blank = cls()
    except TypeError:
        pytest.skip(f"{name} has required fields")
    assert isinstance(repr(blank), str)


def test_a_scaling_fit_region_outside_its_curve_still_reprs():
    """The repr clips the window; ``scaling_window`` still raises, as an accessor should."""
    from tsdynamics.analysis._result import ScalingResult

    r = ScalingResult(abscissa=np.arange(3.0), ordinate=np.arange(3.0), fit_region=(0, 10))
    assert "x ∈ [0, 2]" in repr(r)
    with pytest.raises(IndexError):
        _ = r.scaling_window


def test_collection_item_list_is_truncated():
    """A long collection lists ten items and then says how many there are."""
    from tsdynamics.analysis._result import CollectionResult

    many = CollectionResult(items=tuple(float(i) for i in range(25)), meta={"analysis": "probe"})
    lines = repr(many).splitlines()
    assert len(lines) == 1 + _MAX_ITEMS + 1
    assert lines[-1].strip() == "... [25 total]"


# ---------------------------------------------------------------------------
# Verdicts must be supported by the data (contract §4.2 rule 9)
# ---------------------------------------------------------------------------


def test_wada_says_not_applicable_instead_of_printing_a_measured_zero():
    """``W = 0`` on a two-basin image is *no measurement*, and now says so."""
    result = RESULTS["WadaResult"]
    assert result.applicable is False
    assert result.W is None
    text = repr(result)
    assert "not applicable" in text
    assert "W = 0" not in text and "W=0" not in text
    assert result.to_dict(full=True)["W"] is None


def test_wada_reports_the_number_when_the_test_does_apply():
    """Three basins and a real boundary: the fraction is a measurement again."""
    result = wada_applicable()
    assert result.applicable is True
    assert pytest.approx(0.98) == result.W
    assert "W = 0.98" in repr(result)
    assert "Wada basins" in repr(result)


# ---------------------------------------------------------------------------
# The Lyapunov verdict (contract §4.4) — the rule, on the cases it was chosen for
# ---------------------------------------------------------------------------


def _spectrum(values, **meta) -> LyapunovSpectrum:
    return LyapunovSpectrum(
        values=np.asarray(values, dtype=float), meta={"analysis": "lyapunov_spectrum", **meta}
    )


@pytest.mark.parametrize(
    ("label", "values", "meta", "expected"),
    [
        # 00 §11.5 defect 1: a flow whose zero exponent has not converged at a
        # short horizon was called HYPERchaotic by the 1e-3-of-the-max rule.
        ("Lorenz T=20", [0.897512, 0.0153046, -14.5795], {"system": "Lorenz"}, "chaotic"),
        ("Lorenz converged", [0.9160, 0.000189, -14.583], {"system": "Lorenz"}, "chaotic"),
        # A map has no structural zero, so it keeps the relative floor.
        ("Henon", [0.41922, -1.62319], {"system": "Henon"}, "chaotic"),
        # 00 §11.5 defect 3 (unnoticed by the design): a shipped CONSERVATIVE
        # system was reported chaotic at every horizon.
        ("LotkaVolterra", [1e-5, -2e-5, -3e-4], {"system": "LotkaVolterra"}, "regular"),
        # 00 §11.5 defect 2: a Hamiltonian system has TWO structural zeros, so
        # the honest answer is that the horizon does not resolve them.
        (
            "HenonHeiles",
            [0.0011, 0.0002, -0.0003, -0.0012],
            {"system": "HenonHeiles"},
            "indeterminate",
        ),
    ],
)
def test_lyapunov_verdict(label, values, meta, expected):
    """The verdict names the regime only when a 10x tolerance band agrees."""
    verdict = _spectrum(values, **meta)._interpretation()
    assert verdict is not None
    assert verdict.startswith(expected), f"{label}: got {verdict!r}, expected {expected!r}"


def test_lyapunov_verdict_says_hyperchaotic_when_two_exponents_clear_the_band():
    """The word is still reachable — the rule tightened, it did not delete a case."""
    verdict = _spectrum([0.5, 0.3, 0.0001, -1.0], system="Lorenz")._interpretation()
    assert verdict is not None and verdict.startswith("hyperchaotic")


def test_lyapunov_verdict_is_calibrated_on_the_realised_zero_for_a_flow():
    """A flow's floor is its own ``min|lambda|``; a map's is the relative one."""
    flow = _spectrum([0.9, 0.02, -14.0], system="Lorenz")
    a_map = _spectrum([0.9, 0.02, -14.0], system="Henon")
    assert flow._is_flow and not a_map._is_flow
    assert flow._zero_tolerance == pytest.approx(0.02)
    assert a_map._zero_tolerance == pytest.approx(0.014)


def test_lyapunov_repr_shows_the_dimension_only_when_something_expands():
    """``D_KY`` on a spectrum with no positive exponent would look like a measurement."""
    assert "D_KY" in repr(_spectrum([0.916, 0.000189, -14.583], system="Lorenz"))
    assert "D_KY" not in repr(_spectrum([1e-5, -2e-5, -3e-4], system="LotkaVolterra"))


# ---------------------------------------------------------------------------
# A verdict must be SUPPORTED by the data (contract §4.2 rule 9)
#
# ``WadaResult.applicable`` was the archetype and the only one: five other
# results printed a confident regime off a degenerate measurement.  Each case
# below was reproduced on the shipped code before this change; the comment on
# each records what it printed.
# ---------------------------------------------------------------------------


def _degenerate_cases() -> dict[str, object]:
    """One deliberately-degenerate instance per hedging rule."""
    from tsdynamics.analysis.results import (
        DimensionResult,
        EmbeddingDimension,
        ExpansionEntropyResult,
        GALIResult,
        RQAResult,
        UncertaintyExponent,
    )

    return {
        # was: "DET = 0.000 · LAM = 0.000 · L_max = 0 · ENTR = 0.000   stochastic
        #       (few diagonal lines)" — from a matrix with ZERO recurrence points.
        "rqa-with-no-recurrence-points": RQAResult(
            recurrence_rate=0.0,
            determinism=0.0,
            laminarity=0.0,
            avg_diagonal_length=0.0,
            max_diagonal_length=0,
            divergence=float("inf"),
            diagonal_entropy=0.0,
            trapping_time=0.0,
            max_vertical_length=0,
            size=300,
            epsilon=1e-12,
            theiler_window=0,
            min_diagonal=2,
            min_vertical=2,
            diagonal_lengths=np.empty(0, dtype=int),
            vertical_lengths=np.empty(0, dtype=int),
            meta={"analysis": "rqa"},
        ),
        # was: "H0 = 0 ± 0   non-chaotic (H0 <= 0)   (0/60 survivors)" — the
        # ``se > 0.0`` clause let a 0 +/- 0 fit through to the CONFIDENT branch.
        "expansion-entropy-with-no-survivors": ExpansionEntropyResult(
            estimate=0.0,
            stderr=0.0,
            abscissa=np.arange(9.0),
            ordinate=np.zeros(9),
            fit_region=(0, 8),
            intercept=0.0,
            n_samples=60,
            n_survivors=0,
            meta={"analysis": "expansion_entropy"},
        ),
        # was: "D_corr = 1.8183 ± 0   (correlation, q=2, 2 fit pts, R² = 1)" and
        # ``trusted = True`` — a line through two points always scores R² = 1.
        "dimension-from-a-two-point-fit": DimensionResult(
            estimate=1.8183,
            stderr=0.0,
            abscissa=np.array([-0.691, -0.23]),
            ordinate=np.array([-1.0, -0.162]),
            fit_region=(0, 1),
            intercept=0.0,
            kind="correlation",
            q=2.0,
            meta={"analysis": "correlation_dimension"},
        ),
        # was: "GALI_2 = 1 at the end   regular (GALI bounded)   (1 samples ...)"
        # — a verdict read off the curve's own starting value.
        "gali-from-one-sample": GALIResult(
            k=2,
            times=np.zeros(1),
            values=np.ones(1),
            is_discrete=True,
            meta={"analysis": "gali", "system": "Henon"},
        ),
        # was: "m = 10" with E1 = 0.6 and no flag — the ceiling the search failed
        # to escape, presented as the answer.
        "embedding-dimension-at-the-ceiling": EmbeddingDimension(
            dimension=10,
            dims=np.arange(1, 11),
            method="cao",
            delay=9,
            afn_e1=np.linspace(0.2, 0.6, 10),
            meta={"analysis": "cao_dimension"},
        ),
        # was: "final-state sensitive (fractal boundary)" off TWO radii — the R²
        # acceptance gate cannot fail on a two-point fit.
        "uncertainty-exponent-from-two-radii": UncertaintyExponent(
            alpha=0.66,
            boundary_dimension=1.34,
            state_dimension=2,
            epsilons=np.array([0.01, 0.02]),
            f=np.array([0.10, 0.16]),
            r_squared=1.0,
            meta={"analysis": "uncertainty_exponent"},
        ),
        # was: "not Wada (W < 0.95)" for an image with 2 basins — the archetype,
        # already fixed; kept here so the whole family is checked in one place.
        "wada-with-two-basins": RESULTS["WadaResult"],
        # A dense two-point recurrence plot is not degenerate; the sparse one is.
        "recurrence-matrix-is-not-a-verdict": RESULTS["RecurrenceMatrix"],
    }


#: The words a hedged repr may use.  ``not applicable`` / ``UNTRUSTED`` /
#: ``did not saturate`` are the three shapes; a bare regime word is the defect.
_HEDGES = ("not applicable", "UNTRUSTED", "did not saturate", "too few")

#: A confident regime word must NOT appear in a hedged repr.
_REGIME_WORDS = ("chaotic", "regular", "deterministic", "stochastic", "final-state sensitive")


@pytest.mark.parametrize("case", sorted(_degenerate_cases()))
def test_a_degenerate_result_hedges(case):
    """A confident number where nothing was measured is a wrong answer."""
    result = _degenerate_cases()[case]
    text = repr(result)
    if case == "recurrence-matrix-is-not-a-verdict":
        assert result._interpretation() is None  # it classifies nothing at all
        return
    assert any(h in text for h in _HEDGES), f"{case}: the repr states a verdict it cannot support"
    if case != "embedding-dimension-at-the-ceiling":
        assert not any(w in text for w in _REGIME_WORDS), f"{case}: named a regime anyway"


def test_a_degenerate_result_reports_none_not_false_for_its_verdict():
    """``None`` means *not measured*; ``False`` would mean *measured negative*."""
    cases = _degenerate_cases()
    assert cases["rqa-with-no-recurrence-points"].deterministic is None
    assert cases["expansion-entropy-with-no-survivors"].chaotic is None
    assert cases["gali-from-one-sample"].chaotic is None
    assert cases["uncertainty-exponent-from-two-radii"].final_state_sensitive is None
    assert cases["wada-with-two-basins"].wada is None
    assert cases["dimension-from-a-two-point-fit"].trusted is False
    assert cases["embedding-dimension-at-the-ceiling"].saturated is False


def test_a_degenerate_result_exports_none_for_the_number_it_did_not_measure():
    """``to_dict(full=True)`` must not hand back a vacuous ``0.0`` as a reading."""
    cases = _degenerate_cases()
    assert cases["rqa-with-no-recurrence-points"].to_dict(full=True)["determinism"] is None
    assert cases["wada-with-two-basins"].to_dict(full=True)["W"] is None


def test_r_squared_is_not_printed_for_a_fit_that_cannot_fail_it():
    """``R² = 1`` off two points is an identity, not a diagnostic."""
    text = repr(_degenerate_cases()["dimension-from-a-two-point-fit"])
    assert "R² undefined (2 fit pts)" in text
    assert "R² = 1" not in text
