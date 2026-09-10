"""A result behaves as the plain thing it replaced (v6, contract §4.2 rules 4–7).

Every analysis returns a result *object* instead of a bare float / array / list.
That is only defensible if the object is a complete drop-in for what it replaced
— otherwise the wrapper is a tax, paid at every call site, for provenance the
caller did not ask for.

Each gate below fails on the pre-v6 code.  Measured at HEAD before the change::

    >>> f"{ScalarResult(value=0.42):.3f}"
    TypeError: unsupported format string passed to ScalarResult.__format__
    >>> f"{ScalingResult(estimate=2.0):.3f}"
    TypeError: unsupported format string passed to ScalingResult.__format__
    >>> f"{ArrayResult(values=np.arange(3.0)):.3f}"
    TypeError: unsupported format string passed to ArrayResult.__format__
    >>> np.asarray(fixed_point_set).dtype, .shape
    dtype('O'), (2,)
    >>> ArrayResult(values=np.arange(1.0, 4.0)) / 2
    TypeError: unsupported operand type(s) for /: 'ArrayResult' and 'int'
    >>> ArrayResult(values=np.arange(1.0, 4.0)) ** 2
    TypeError: unsupported operand type(s) for ** or pow(): 'ArrayResult' and 'int'
    >>> -ArrayResult(values=np.arange(1.0, 4.0))
    TypeError: bad operand type for unary -: 'ArrayResult'
    >>> attractor_set[0]
    KeyError: 0                       # ids start at 1, so [] was NOT positional
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _result_fixtures import build

from tsdynamics.analysis._result import (
    AnalysisResult,
    ArrayResult,
    CollectionResult,
    CountResult,
    ScalarResult,
    ScalingResult,
)

RESULTS = build()


# ---------------------------------------------------------------------------
# __format__ — f"{result:.3f}" used to raise on EVERY numeric result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ScalarResult", "0.423"),
        ("ScalingResult", "2.001"),
        ("CountResult", "9.000"),
        ("DimensionResult", "2.001"),
        ("LyapunovFromData", "0.006"),
        ("ExpansionEntropyResult", "0.430"),
        ("ZeroOneResult", "0.998"),
        ("GALIResult", "0.000"),
    ],
)
def test_numeric_result_formats_as_its_number(name, expected):
    """``f"{result:.3f}"`` formats the number, not the wrapper."""
    assert f"{RESULTS[name]:.3f}" == expected


def test_empty_format_spec_is_the_headline():
    """``f"{result}"`` is ``str(result)`` — the headline (contract §4.2 rule 12)."""
    result = RESULTS["ScalarResult"]
    assert f"{result}" == str(result) == repr(result).splitlines()[0]


def test_count_result_prints_as_its_integer():
    """``f"tau={c}"`` prints ``tau=9`` — the committed figure depends on it.

    ``int.__str__ is object.__str__``, so without an explicit override a
    ``CountResult`` renders through ``__repr__`` and the caption reads
    ``tau=CountResult(28)``.  That output is committed to the repository, in
    ``docs/assets/figures/analysis/embedding.svg``.
    """
    tau = RESULTS["CountResult"]
    assert f"tau={tau}" == "tau=9"
    assert f"{tau:d}" == "9" and f"{tau:>4}" == "   9"


def test_a_non_numeric_result_formats_its_headline_rather_than_raising():
    """``f"{result:>60}"`` aligns the headline instead of erroring."""
    result = RESULTS["RQAResult"]
    assert f"{result:>200}".strip() == str(result)


# ---------------------------------------------------------------------------
# The numeric protocol stays COMPLETE (contract §4.2 rule 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["ScalarResult", "ScalingResult", "DimensionResult"])
def test_python_operators_still_return_plain_numbers(name):
    """``result * 2`` and ``result + 1`` must not regress into NumPy dispatch.

    ``__array_ufunc__`` alone would break them: NumPy is consulted only when
    NumPy dispatches, and ``int.__mul__(result)`` returns ``NotImplemented``
    without ever reaching it.
    """
    r = RESULTS[name]
    v = float(r)
    assert r * 2 == pytest.approx(2 * v)
    assert r + 1 == pytest.approx(v + 1)
    assert 2 * r == pytest.approx(2 * v)
    assert 1 - r == pytest.approx(1 - v)
    assert abs(-r) == pytest.approx(abs(v))
    assert (r > v - 1) and (r < v + 1)


@pytest.mark.parametrize("name", ["ScalarResult", "ScalingResult"])
def test_the_operators_the_mixin_fills_in_now_work(name):
    """``**``, ``//``, ``%`` and ``np.exp`` route through ``__array_ufunc__``."""
    r = RESULTS[name]
    v = float(r)
    assert float(r**2) == pytest.approx(v**2)
    assert float(r // 1) == pytest.approx(v // 1)
    assert float(np.exp(r)) == pytest.approx(math.exp(v))


def test_array_result_has_the_full_operator_set():
    """``/``, ``**``, unary ``-`` and ``//`` raised ``TypeError`` before v6."""
    r = ArrayResult(values=np.arange(1.0, 4.0))
    assert np.allclose(r / 2, [0.5, 1.0, 1.5])
    assert np.allclose(r**2, [1.0, 4.0, 9.0])
    assert np.allclose(-r, [-1.0, -2.0, -3.0])
    assert np.allclose(r // 2, [0.0, 1.0, 1.0])
    assert np.allclose(r + r, [2.0, 4.0, 6.0])
    assert np.allclose(np.sqrt(r), np.sqrt([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# np.asarray gives the NATURAL array (contract §4.2 rule 5)
# ---------------------------------------------------------------------------


def test_asarray_of_a_fixed_point_set_is_a_real_float_matrix():
    """``np.asarray(fixed_points)`` used to be a ``(n,)`` array of *objects*."""
    arr = np.asarray(RESULTS["FixedPointSet"])
    assert arr.dtype == np.float64
    assert arr.shape == (2, 2)
    assert np.allclose(arr[0], [-1.131354, -0.339406])


def test_asarray_of_an_orbit_set_is_a_real_float_matrix():
    """The same holds for the periodic-orbit collection."""
    arr = np.asarray(RESULTS["OrbitSet"])
    assert arr.dtype == np.float64 and arr.ndim == 2


def test_asarray_of_an_attractor_set_is_the_representatives():
    """An attractor set arrays as its ``(n_attractors, dim)`` representatives."""
    arr = np.asarray(RESULTS["AttractorSet"])
    assert arr.shape == (2, 2)
    assert np.allclose(np.sort(arr[:, 0]), [-1.0, 1.0])


def test_asarray_of_an_array_result_is_its_array():
    """A wrapped array arrays as itself, unchanged."""
    assert np.allclose(np.asarray(RESULTS["LyapunovSpectrum"]), [0.9160, 0.000189, -14.583])


def test_asarray_of_a_ragged_collection_falls_back_rather_than_lying():
    """Items of differing length keep the object array — never padded or dropped."""
    ragged = CollectionResult(items=(np.zeros(2), np.zeros(3)))
    assert np.asarray(ragged).dtype == object


# ---------------------------------------------------------------------------
# Sequences are COMPLETE and positional (contract §4.2 rule 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["FixedPointSet", "OrbitSet", "AttractorSet", "WindowedRQA"])
def test_collections_are_sequences(name):
    """``len`` / ``[]`` / iteration all work and agree with each other."""
    c = RESULTS[name]
    n = len(c)
    assert n > 0
    listed = list(c)
    assert len(listed) == n
    assert c[0] is listed[0]
    assert c[n - 1] is listed[-1]
    assert isinstance(c[0:1], list)


def test_attractor_set_indexes_by_position_and_looks_up_by_id():
    """``aset[0]`` is the FIRST attractor; ``aset.by_id(1)`` is the one labelled 1.

    Ids start at ``1``, so before v6 ``aset[0]`` raised ``KeyError`` while
    ``aset[1]`` returned the first attractor — an index that reads positional and
    is not.
    """
    aset = RESULTS["AttractorSet"]
    assert aset[0].id == 1
    assert aset[1].id == 2
    assert aset.by_id(2) is aset[1]
    with pytest.raises(KeyError):
        aset.by_id(99)


def test_collection_by_id_finds_the_labelled_item():
    """``by_id`` is the explicit id lookup on any collection whose items carry one."""
    aset = RESULTS["AttractorSet"]
    items = CollectionResult(items=tuple(aset))
    assert items.by_id(2).id == 2
    with pytest.raises(KeyError):
        items.by_id(7)


def test_basin_fractions_stays_an_id_mapping():
    """A mapping keeps id lookup; a positional ``[]`` there would be a wrong answer."""
    bf = RESULTS["BasinFractions"]
    assert bf[1] == pytest.approx(0.55)
    assert bf.by_id(2) == pytest.approx(0.45)


def test_collection_supports_the_rest_of_the_sequence_protocol():
    """``in``, ``reversed``, ``.index`` and ``.count`` behave like a list's."""
    c = CollectionResult(items=(1.0, 2.0, 2.0))
    assert 2.0 in c and 5.0 not in c
    assert list(reversed(c)) == [2.0, 2.0, 1.0]
    assert c.index(2.0) == 1
    assert c.count(2.0) == 2


# ---------------------------------------------------------------------------
# to_dict stays JSON-safe, and `full=` only ever ADDS keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_to_dict_is_json_serializable(name):
    """Every result exports as plain JSON, with and without ``full``."""
    import json

    result = RESULTS[name]
    json.dumps(result.to_dict())
    json.dumps(result.to_dict(full=True))


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_full_only_adds_keys(name):
    """``to_dict(full=True)`` is a superset — nothing an export relied on is dropped."""
    result = RESULTS[name]
    plain, full = result.to_dict(), result.to_dict(full=True)
    assert set(plain) <= set(full), f"{name}: full dropped {sorted(set(plain) - set(full))}"


def test_full_carries_the_numbers_the_repr_shows():
    """The derived answers the headline reports become exportable."""
    spectrum = RESULTS["LyapunovSpectrum"].to_dict(full=True)
    assert spectrum["kaplan_yorke"] == pytest.approx(2.063, abs=1e-3)
    assert spectrum["n_positive"] == 1
    matrix = RESULTS["RecurrenceMatrix"].to_dict(full=True)
    assert matrix["size"] == 251 and matrix["recurrence_rate"] == pytest.approx(0.05, abs=1e-3)


# ---------------------------------------------------------------------------
# The subclassing machinery the whole layer rests on (contract §4.2 rule 11)
# ---------------------------------------------------------------------------


def test_the_init_subclass_hook_still_claims_the_repr():
    """Removing it silently reverts every subclass to a dataclass field dump."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Probe(AnalysisResult):
        x: float = 1.0

    assert "__repr__" in Probe.__dict__
    assert repr(Probe()) == "Probe  x = 1"


def test_a_result_with_no_fields_still_reprs():
    """The base is renderable, so a bare subclass never crashes a console."""
    assert repr(AnalysisResult()) == "AnalysisResult"


@pytest.mark.parametrize("cls", [ScalarResult, ScalingResult, CountResult])
def test_numeric_results_are_hashable_and_compare_by_value(cls):
    """They stand in for a number in a set / dict key / equality test."""
    a, b = cls(), cls()
    assert a == b and hash(a) == hash(b)
    assert float(a) == 0.0
