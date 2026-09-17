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

#: Results that carry ``__getitem__`` and hold at least one item, so indexing is
#: exercisable.  Built from the fixtures rather than listed, so a new sized
#: result joins the sweep with no edit here.
SIZED = sorted(
    name
    for name, r in RESULTS.items()
    if hasattr(type(r), "__getitem__") and hasattr(type(r), "__len__") and len(r) > 0
)


def _same_item(a, b) -> bool:
    """Whether two items yielded by ``[]`` and by iteration are the same answer."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    return a is b or a == b


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
    assert _same_item(c[0], listed[0])
    assert _same_item(c[n - 1], listed[-1])


def test_attractor_set_indexes_by_position_and_looks_up_by_id():
    """``aset[0]`` is the FIRST attractor; ``aset.by_id(1)`` is the one labelled 1.

    Ids start at ``1``, so before v6 ``aset[0]`` raised ``KeyError`` while
    ``aset[1]`` returned the first attractor — an index that reads positional and
    is not.  Position gives the **numbers** (the centre); the *label* is the door
    you reach for after reading an id off a basin image, so it gives the record.
    """
    aset = RESULTS["AttractorSet"]
    assert aset.details[0].id == 1
    assert aset.details[1].id == 2
    assert np.allclose(aset[0], aset.details[0].center)
    assert aset.by_id(2) is aset.details[1]
    with pytest.raises(KeyError):
        aset.by_id(99)


def test_collection_by_id_finds_the_labelled_item():
    """``by_id`` is the explicit id lookup on any collection whose items carry one."""
    aset = RESULTS["AttractorSet"]
    items = CollectionResult(items=aset.details)
    assert items.by_id(2).id == 2
    with pytest.raises(KeyError):
        items.by_id(7)


def test_basin_fractions_is_a_complete_positional_sequence():
    """``bf[0]`` is the FIRST share; ``bf.by_id(1)`` is the one labelled 1.

    The exact defect v6 fixed for ``AttractorSet`` ("ids start at 1, so ``[0]``
    raised ``KeyError``") and left in place on its sibling one file away.
    Measured at HEAD before this change::

        bf[0]     -> KeyError: 0
        list(bf)  -> KeyError: 0       # no __iter__, so the legacy protocol
        for x in bf -> KeyError: 0     # ...called __getitem__(0)
        len(bf)   -> TypeError: object of type 'BasinFractions' has no len()
    """
    bf = RESULTS["BasinFractions"]
    assert len(bf) == 2
    assert bf[0] == pytest.approx(0.55)
    assert bf[1] == pytest.approx(0.45)
    assert list(bf) == pytest.approx([0.55, 0.45])
    assert [x for x in bf] == pytest.approx([0.55, 0.45])  # noqa: C416 - exercises __iter__
    assert np.asarray(bf).dtype == np.float64
    assert bf.by_id(1) == pytest.approx(0.55)
    assert bf.by_id(2) == pytest.approx(0.45)
    with pytest.raises(KeyError):
        bf.by_id(99)


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


# ---------------------------------------------------------------------------
# np.asarray is NUMERIC, or it REFUSES by name (contract §4.2 rule 5)
#
# Measured at HEAD before this change: 17 of the 32 classes silently produced a
# **0-d object array** — ``array(RQAResult  DET = 0.794 …, dtype=object)`` — which
# does not raise, plots as nothing, and arithmetics into a ``TypeError`` far from
# the call site.  Every one of them is now either a real numeric array or a
# ``TypeError`` naming the field to ask for.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_asarray_is_numeric_or_refuses(name):
    """No result ever yields an *object* array: it is numbers, or it says why not."""
    result = RESULTS[name]
    try:
        arr = np.asarray(result)
    except TypeError as exc:
        message = str(exc)
        assert type(result).__name__ in message, f"{name}: the refusal must name the class"
        carried = result._numeric_field_names()
        if carried:
            assert carried[0] in message, f"{name}: the refusal must name what to ask for"
        return
    assert arr.dtype != object, f"{name}: np.asarray gave a silent object array"
    assert np.issubdtype(arr.dtype, np.number) or arr.dtype == bool, (
        f"{name}: np.asarray gave {arr.dtype}, which is not numbers"
    )


def _is_plain_numbers(part) -> bool:
    """Whether ``part`` is numbers rather than an object a caller must unwrap.

    A result class is never numbers, however completely it emulates an array:
    ``fps[0].shape`` and ``fps[0].tolist()`` still raise ``AttributeError`` on
    one, which is the class making itself visible at exactly the moment the
    caller believed it was holding an array.
    """
    if isinstance(part, AnalysisResult):
        return False
    arr = np.asarray(part)
    return bool(arr.dtype != object and arr.size and np.issubdtype(arr.dtype, np.number))


@pytest.mark.parametrize("name", SIZED)
def test_indexing_a_collection_gives_numbers(name):
    """``result[0]`` is numbers — never an object you must learn to unwrap.

    The owner's D1: ``ts.analysis.fixed_points(lor)[0]`` handed back a
    ``FixedPoint``, ``periodic_orbits(...)[0]`` a ``PeriodicOrbit``,
    ``attractors(...)[0]`` an ``Attractor``, ``windowed_rqa(...)[0]`` an
    ``RQAResult`` — four classes and twenty attribute names on the learning path.

    Making those records *behave* like arrays was not enough: the class still
    showed through the moment a caller reached for the array API it had just
    been shown worked.  Measured on the array-backed ``FixedPoint``::

        np.asarray(fps[0])  ->  array([-1.13, -0.34])     # looks like an array
        fps[0].shape        ->  AttributeError: 'FixedPoint' object has no attribute 'shape'
        fps[0].tolist()     ->  AttributeError: 'FixedPoint' object has no attribute 'tolist'

    So ``[]`` hands back the :class:`numpy.ndarray` itself.  An item that is
    naturally a *pair* (an orbit diagram's ``(parameter, orbit)``, a return map's
    ``(v_n, v_{n+1})``) is plain Python and is checked part by part.
    """
    item = RESULTS[name][0]
    parts = item if isinstance(item, tuple) else (item,)
    for k, part in enumerate(parts):
        assert _is_plain_numbers(part), (
            f"{name}[0]"
            + (f"[{k}]" if len(parts) > 1 else "")
            + f" is a {type(part).__name__}, not numbers"
        )


@pytest.mark.parametrize("name", SIZED)
def test_iterating_a_collection_gives_what_indexing_does(name):
    """One grammar: ``for x in r`` yields exactly ``r[0], r[1], …``."""
    r = RESULTS[name]
    listed = list(r)
    assert len(listed) == len(r)
    for i, value in enumerate(listed):
        assert _same_item(value, r[i]), f"{name}: iteration and [] disagree at {i}"
        parts = value if isinstance(value, tuple) else (value,)
        assert all(_is_plain_numbers(p) for p in parts), (
            f"{name}: iteration yielded a {type(value).__name__}, not numbers"
        )


#: The collections that wrap a per-member RECORD, and the class that record is.
#: Every one of them must reach it under the SAME name (contract §4.2 rule 6):
#: ``[]`` gives numbers, ``.details`` gives the records, one lesson for all four.
RECORD_COLLECTIONS = [
    ("FixedPointSet", "FixedPoint"),
    ("OrbitSet", "PeriodicOrbit"),
    ("AttractorSet", "Attractor"),
    ("WindowedRQA", "RQAResult"),
]


@pytest.mark.parametrize(("name", "record"), RECORD_COLLECTIONS)
def test_every_collection_offers_its_records_under_one_name(name, record):
    """``.details`` is the one spelling for "the objects ``[]`` no longer hands back".

    Four collections, four record classes, **one** accessor — so the diagnostics
    are out of the way without being out of reach, and the name is learned once.
    """
    r = RESULTS[name]
    details = r.details
    assert isinstance(details, tuple), f"{name}.details must be a tuple, got {type(details)}"
    assert len(details) == len(r), f"{name}.details is not aligned with the collection"
    assert type(details[0]).__name__ == record
    assert isinstance(details[0], AnalysisResult)
    # ...and the record is the one that member's numbers came from.
    assert _same_item(r[0], r._item_value(details[0]) if hasattr(r, "_item_value") else r[0])


def test_a_fixed_point_set_indexes_to_its_points():
    """The owner's headline case, end to end.

    ``fixed_points(lor)[0]`` is the point, as an ordinary array — and every
    diagnostic is still reachable by name, vectorised, with no loop and no
    second class in the way.
    """
    fps = RESULTS["FixedPointSet"]
    point = fps[0]
    assert type(point) is np.ndarray
    assert point.shape == (2,) and point.dtype == np.float64
    assert point.tolist() == pytest.approx([-1.131354, -0.339406])
    assert np.allclose(point, np.asarray(fps)[0])  # [] and asarray agree, row for row
    assert np.allclose(fps[0] - fps[1], np.asarray(fps)[0] - np.asarray(fps)[1])
    assert np.linalg.norm(point) == pytest.approx(1.181168, rel=1e-5)
    # The diagnostics: by name on the SET (vectorised) ...
    assert fps.is_stable.tolist() == [False, False]
    assert fps.eigenvalues.shape == (2, 2)
    # ... and the full record one dot out of the way, with its repr intact.
    assert fps.details[0].stable is False
    assert np.allclose(fps.details[0].eigenvalues, [3.2598, -0.09202])
    assert "unstable" in repr(fps.details[0])


def test_an_orbit_set_indexes_to_the_orbit_points():
    """An orbit *is* its ``(p, dim)`` points, not the centroid of them."""
    orbits = RESULTS["OrbitSet"]
    orbit = orbits[0]
    assert type(orbit) is np.ndarray
    assert orbit.shape == (2, 2)
    assert np.allclose(orbit, orbits.details[0].points)
    assert np.allclose(orbits.points[0], orbit)
    # The set still arrays rectangularly (one representative per orbit), because
    # orbits of differing period cannot stack.
    assert np.asarray(orbits).shape == (1, 2)


def test_an_attractor_set_indexes_to_the_centres():
    """``aset[0]`` and ``np.asarray(aset)[0]`` are the same ``(dim,)`` centre."""
    aset = RESULTS["AttractorSet"]
    assert type(aset[0]) is np.ndarray
    assert np.allclose(aset[0], np.asarray(aset)[0])
    assert np.allclose(aset[0], aset.details[0].center)
    assert [np.asarray(c).tolist() for c in aset] == np.asarray(aset).tolist()


def test_filtering_a_collection_keeps_the_result_surface():
    """``fps.stable`` is a ``FixedPointSet``, not a bare ``list``.

    Narrowing used to throw the whole surface away — no repr, no ``to_frame``,
    no ``.plot`` — which is the one thing a result object is for.
    """
    fps = RESULTS["FixedPointSet"]
    assert type(fps.stable) is type(fps)
    assert type(fps.unstable) is type(fps)
    assert len(fps.unstable) == 2 and len(fps.stable) == 0
    assert "point" in repr(fps.unstable)
    orbits = RESULTS["OrbitSet"]
    assert type(orbits.unstable) is type(orbits)


def test_vectorised_accessors_need_no_loop():
    """Every per-member diagnostic is reachable as one array."""
    fps = RESULTS["FixedPointSet"]
    assert fps.points.shape == (2, 2)
    assert fps.eigenvalues.shape == (2, 2)
    assert fps.is_stable.tolist() == [False, False]
    orbits = RESULTS["OrbitSet"]
    assert orbits.periods.tolist() == [2]
    assert orbits.multipliers.shape == (1, 2)
    aset = RESULTS["AttractorSet"]
    assert aset.centers.shape == (2, 2) and aset.cells.tolist() == [3, 3]


def test_a_continuation_reaches_its_attractor_positions_without_three_containers():
    """``cont.centers`` is the ``(n_values, n_ids, dim)`` array behind ``.attractors``.

    ``.attractors`` is a *list of dicts of records* — three containers deep
    before a coordinate.  The numbers are one rectangular array, aligned index
    for index with ``np.asarray(cont)`` and with ``.ids``, ``nan`` where an
    attractor is absent at that parameter value.
    """
    from tsdynamics.analysis.results import ContinuationResult

    cont = RESULTS["ContinuationResult"]
    centers = cont.centers
    assert centers.dtype == np.float64
    assert centers.shape[:2] == np.asarray(cont).shape == (len(cont.values), len(cont.ids))
    assert centers.ndim == 3
    present = cont.attractors[0][cont.ids[0]]
    assert np.allclose(centers[0, 0], np.asarray(present.center, dtype=float))
    assert ContinuationResult().centers.shape == (0, 0, 0)  # an empty sweep still arrays


def test_windowed_rqa_is_a_table_of_numbers():
    """``w[i]`` is a measure row and ``np.asarray(w)`` the whole table.

    It used to be a ``(4,)`` array of result *wrappers*.  The per-window readouts
    moved to ``.details`` — reachable by name, out of the way.
    """
    w = RESULTS["WindowedRQA"]
    table = np.asarray(w)
    assert table.shape == (len(w), len(w.measures)) and table.dtype == np.float64
    assert np.allclose(w[0], table[0])
    assert w.measures[1] == "determinism"
    assert np.allclose(w.determinism, table[:, 1])
    from tsdynamics.analysis.results import RQAResult

    assert isinstance(w.details[0], RQAResult)


def test_a_wrong_guess_on_a_windowed_rqa_names_the_class():
    """It was the only one of the 32 whose ``AttributeError`` named nothing."""
    with pytest.raises(AttributeError) as excinfo:
        RESULTS["WindowedRQA"].determinsm  # noqa: B018
    message = str(excinfo.value)
    assert "WindowedRQA" in message and "determinism" in message


@pytest.mark.parametrize("name", ["OrbitDiagram", "ReturnMap"])
def test_sized_and_iterable_implies_subscriptable(name):
    """Both were ``len``-able and iterable but ``r[0]`` raised ``TypeError``."""
    r = RESULTS[name]
    assert _same_item(r[0], next(iter(r)))
    assert len(list(r)) == len(r)
    arr = np.asarray(r)
    assert arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype == np.float64


# ---------------------------------------------------------------------------
# f"{result:.3f}" on a NON-number names this class (contract §4.2 rule 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["RQAResult", "WadaResult", "BasinEntropy"])
def test_a_numeric_format_code_on_a_non_number_names_the_class(name):
    """It used to raise ``ValueError: Unknown format code 'f' … of type 'str'``.

    That message names ``str`` — a type the caller never typed — because the
    fallback formatted ``str(self)``.
    """
    with pytest.raises(TypeError) as excinfo:
        f"{RESULTS[name]:.3f}"  # noqa: B028
    assert name in str(excinfo.value)


# ---------------------------------------------------------------------------
# Truthiness is refused or meaningful (contract §4.2 rule 7)
# ---------------------------------------------------------------------------


def test_truthiness_is_refused_or_meaningful():
    """``bool(result)`` was ``True`` for 31 of 32 — a coin flip that always won.

    ``if ts.analysis.wada_property(...):`` fired whether or not the boundary was
    Wada.  A sized result keeps Python's convention, a number keeps the number's,
    and everything else refuses by name.
    """
    for name, result in sorted(RESULTS.items()):
        try:
            truth = bool(result)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TypeError):
                assert type(result).__name__ in str(exc), f"{name}: the refusal must name itself"
            continue
        if hasattr(type(result), "__len__"):
            assert truth == (len(result) > 0), f"{name}: sized truthiness must be non-emptiness"
        else:
            assert truth == bool(result._as_number()), f"{name}: numeric truthiness must be its own"


def test_a_wada_result_refuses_the_coin_flip():
    """The archetype, both ways round."""
    from _result_fixtures import wada_applicable

    for result in (RESULTS["WadaResult"], wada_applicable()):
        with pytest.raises(TypeError, match="ambiguous"):
            bool(result)
    assert RESULTS["WadaResult"].wada is None  # 2 basins: the test did not apply
    assert wada_applicable().wada is True


# ---------------------------------------------------------------------------
# Every printed verdict is reachable BY NAME, and exported (rules 9, 10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_every_printed_verdict_is_reachable_by_name(name):
    """``result.verdict`` is the clause the repr prints, and ``full=True`` emits it."""
    result = RESULTS[name]
    printed = result._interpretation()
    assert result.verdict == printed
    assert result.to_dict(full=True)["verdict"] == printed


@pytest.mark.parametrize(
    ("name", "accessor", "expected"),
    [
        ("ZeroOneResult", "chaotic", True),
        ("ExpansionEntropyResult", "chaotic", True),
        ("ScalarResult", "chaotic", True),  # max_lyapunov
        ("RQAResult", "deterministic", True),
        ("UncertaintyExponent", "final_state_sensitive", True),
        ("GALIResult", "chaotic", True),
        ("BasinEntropy", "fractal_boundary", False),
        ("LyapunovSpectrum", "chaotic", True),
    ],
)
def test_one_adjective_named_boolean_per_classifying_result(name, accessor, expected):
    """The repr said ``chaotic``; the instance surface did not.  Now it does.

    Six spellings for one concept before v6 (``chaotic`` / ``is_chaotic(...)`` /
    ``is_wada`` / ``fractal_boundary`` / ``trusted`` / ``stable``).
    """
    assert getattr(RESULTS[name], accessor) is expected


# ---------------------------------------------------------------------------
# to_frame contains THE ANSWER (contract §4.2 rule 8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_to_frame_contains_the_answer(name):
    """No cell holds a ``dict`` or a nested result, and vectors become columns."""
    pd = pytest.importorskip("pandas")
    frame = RESULTS[name].to_frame()
    assert isinstance(frame, pd.DataFrame)
    for column in frame.columns:
        for cell in frame[column]:
            assert not isinstance(cell, (dict, AnalysisResult)), (
                f"{name}.to_frame()['{column}'] holds a {type(cell).__name__}, not a value"
            )


def test_a_member_tabulates_like_one_row_of_its_set():
    """``FixedPoint.to_frame()`` dropped the coordinates and kept the booleans.

    Measured at HEAD before this change::

        fps[0].to_frame().columns -> ['stable', 'continuous']
        fps.to_frame().columns    -> ['x0','x1','eigenvalues0','eigenvalues1',
                                      'stable','continuous']

    The same information, two answers — because the base ``to_frame`` kept only
    ``_display_fields`` *scalars* while the collection spread its vectors.
    """
    pytest.importorskip("pandas")
    point = RESULTS["FixedPoint"].to_frame()
    assert list(point.columns) == [
        "x0",
        "x1",
        "eigenvalues0",
        "eigenvalues1",
        "stable",
        "continuous",
    ]
    assert list(RESULTS["FixedPointSet"].to_frame().columns) == list(point.columns)
    # An attractor's ``points`` is an (m, dim) cloud, so the row carries its
    # representative instead — the answer to *where*, which ``['id','cells']``
    # (all the old frame held) does not contain.
    assert {"center0", "center1"} <= set(RESULTS["Attractor"].to_frame().columns)
    # A recurrence matrix tabulated its SETTINGS and dropped the matrix.
    assert list(RESULTS["RecurrenceMatrix"].to_frame().columns) == ["i", "j"]


# ---------------------------------------------------------------------------
# meta is PROVENANCE, not payload (contract §4.2 rule 8)
# ---------------------------------------------------------------------------

#: Above this many elements, a value on ``meta`` is payload.  ``estimate_period``
#: parked its whole autocorrelation curve there, so ``to_dict()`` for ONE float
#: was 30 824 characters of JSON.
_MAX_META_ELEMENTS = 32


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_meta_is_provenance_not_payload(name):
    """No ``meta`` value is a bulk array."""
    for key, value in (RESULTS[name].meta or {}).items():
        size = np.asarray(value).size if isinstance(value, (np.ndarray, list, tuple)) else 1
        assert size <= _MAX_META_ELEMENTS, f"{name}: meta[{key!r}] carries {size} elements"


def test_estimate_period_keeps_its_curve_off_meta():
    """The live estimator, not a fixture: ``to_dict()`` for one float stays small."""
    import json

    import tsdynamics as ts

    t = np.linspace(0.0, 100.0, 4001)
    result = ts.analysis.estimate_period(np.sin(2 * np.pi * t / 5.0), dt=float(t[1] - t[0]))
    assert float(result) == pytest.approx(5.0, rel=1e-3)
    assert len(json.dumps(result.to_dict())) < 500
    for value in result.meta.values():
        assert np.asarray(value).size <= _MAX_META_ELEMENTS


def test_estimate_period_states_the_unit_it_measured_in():
    """It printed ``samples`` for a value in TIME UNITS — wrong by ``1/dt``."""
    import tsdynamics as ts

    t = np.linspace(0.0, 100.0, 4001)
    signal = np.sin(2 * np.pi * t / 5.0)
    timed = ts.analysis.estimate_period(signal, dt=float(t[1] - t[0]))
    bare = ts.analysis.estimate_period(signal)
    assert "time units" in repr(timed) and float(timed) == pytest.approx(5.0, rel=1e-3)
    assert "samples" in repr(bare) and float(bare) == pytest.approx(200.0, rel=1e-3)
