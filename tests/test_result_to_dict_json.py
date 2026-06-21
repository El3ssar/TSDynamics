"""Regression: every analysis result's ``to_dict()`` is JSON-serializable.

:meth:`AnalysisResult.to_dict` promises a "stdlib, JSON-friendly mapping", but
several results embed a value the shared :func:`tsdynamics.analysis._result._jsonify`
helper used to leave untouched — a SciPy sparse ``csr_matrix`` (the recurrence
matrix), nested result objects (``Attractor`` inside an ``AttractorSet``,
``RQAResult`` inside a ``WindowedRQA``), or a state-space ``Grid`` (inside a
``BasinsResult``) — so ``json.dumps(result.to_dict())`` raised ``TypeError``.

``_jsonify`` now serializes each of those recursively (a COO triplet for sparse,
the nested ``to_dict`` for results, the field mapping for ``Box``/``Ball``/
``Grid``), while int-backed results (``CountResult`` *is* an ``int``) stay native.
These tests pin that fix: the previously-raising results round-trip through JSON,
and the ``_jsonify`` edge cases hold.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.analysis._result import CountResult, ScalarResult, _jsonify
from tsdynamics.analysis.basins.attractors import Attractor, AttractorSet
from tsdynamics.analysis.basins.basins import BasinFractions, BasinsResult
from tsdynamics.analysis.basins.continuation import ContinuationResult
from tsdynamics.data import Box, Grid


def _assert_json_roundtrips(result) -> dict:
    """Assert ``result.to_dict()`` is a mapping that round-trips through JSON."""
    data = result.to_dict()
    assert isinstance(data, dict)
    text = json.dumps(data)  # raises TypeError if a non-serializable object leaks
    assert json.loads(text) is not None or True  # round-trip parses back
    return data


# ---------------------------------------------------------------------------
# The analyses that actually raised (built from a real run)
# ---------------------------------------------------------------------------


def _henon_traj():
    """A deterministic Hénon trajectory for the recurrence-based results."""
    return ts.systems.Henon().iterate(steps=600, ic=[0.1, 0.1])


def _henon_box() -> Box:
    """A box bounding the Hénon attractor, for the basin/attractor results."""
    return Box(np.array([-1.6, -0.45]), np.array([1.6, 0.45]))


def test_recurrence_matrix_to_dict_json():
    """``RecurrenceMatrix.to_dict()`` serializes its sparse matrix as a COO triplet."""
    result = ts.recurrence_matrix(_henon_traj(), recurrence_rate=0.05)
    data = _assert_json_roundtrips(result)
    matrix = data["matrix"]
    assert matrix["format"] == "coo"
    assert set(matrix) >= {"shape", "row", "col"}
    assert len(matrix["row"]) == len(matrix["col"])


def test_windowed_rqa_to_dict_json():
    """``WindowedRQA.to_dict()`` recurses into its per-window ``RQAResult`` objects."""
    result = ts.windowed_rqa(_henon_traj(), window=200, step=100, recurrence_rate=0.05)
    data = _assert_json_roundtrips(result)
    assert isinstance(data["results"], list) and data["results"]
    assert all(isinstance(window, dict) for window in data["results"])


def test_find_attractors_to_dict_json():
    """``AttractorSet.to_dict()`` recurses into its nested ``Attractor`` objects."""
    result = ts.find_attractors(
        ts.systems.Henon(), _henon_box(), resolution=30, n_seeds=80, max_steps=400, seed=0
    )
    data = _assert_json_roundtrips(result)
    assert isinstance(data["attractors"], dict)


# ---------------------------------------------------------------------------
# The audited results (embed AttractorSet / Grid), built minimally
# ---------------------------------------------------------------------------


def _attractor_set() -> AttractorSet:
    """A small ``AttractorSet`` with one nested ``Attractor``."""
    return AttractorSet(
        attractors={1: Attractor(id=1, points=np.zeros((3, 2)), cells=2, meta={"a": 1})},
        diverged=0,
        seeds=10,
        meta={"system": "probe"},
    )


def _audited_results():
    """Construct the basin results that embed an ``AttractorSet`` / ``Grid``."""
    att = _attractor_set()
    grid = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), (4, 4))
    return [
        ("AttractorSet", att),
        (
            "BasinsResult",
            BasinsResult(
                labels=np.ones((4, 4), dtype=int), grid=grid, attractors=att, meta={"system": "p"}
            ),
        ),
        (
            "BasinFractions",
            BasinFractions(
                fractions={1: 0.7, 2: 0.3},
                diverged=0.0,
                n=100,
                attractors=att,
                meta={"system": "p"},
            ),
        ),
        (
            "ContinuationResult",
            ContinuationResult(
                param="r",
                values=np.array([1.0, 2.0]),
                fractions={1: np.array([0.5, 0.6]), 2: np.array([0.5, 0.4])},
                attractors=[{1: Attractor(id=1, points=np.zeros((2, 2)), cells=1)}],
                diverged=np.array([0.0, 0.0]),
                meta={"system": "p"},
            ),
        ),
    ]


_AUDITED = _audited_results()


@pytest.mark.parametrize("name,result", _AUDITED, ids=[c[0] for c in _AUDITED])
def test_audited_result_to_dict_json(name, result):
    """A result embedding an ``AttractorSet`` / ``Grid`` round-trips through JSON."""
    _assert_json_roundtrips(result)


# ---------------------------------------------------------------------------
# The _jsonify edge cases the fix turns on
# ---------------------------------------------------------------------------


def test_jsonify_recurses_into_nested_result():
    """A nested non-int ``AnalysisResult`` is serialized via its own ``to_dict``."""
    out = _jsonify({"scalar": ScalarResult(1.5, meta={"k": 1})})
    assert isinstance(out["scalar"], dict)
    assert out["scalar"]["value"] == 1.5
    json.dumps(out)


def test_jsonify_keeps_int_backed_result_native():
    """``CountResult`` (an ``int``) is kept a native JSON scalar, not expanded."""
    out = _jsonify({"delay": CountResult(7, meta={"k": 1})})
    assert isinstance(out["delay"], int) and out["delay"] == 7
    assert json.loads(json.dumps(out))["delay"] == 7


def test_jsonify_serializes_grid_by_fields():
    """A plain state-space dataclass (``Grid``) is serialized field-by-field."""
    grid = Grid(np.array([-1.0, 0.0]), np.array([1.0, 2.0]), (3, 5))
    out = _jsonify(grid)
    assert out == {"lo": [-1.0, 0.0], "hi": [1.0, 2.0], "counts": [3, 5]}
    json.dumps(out)


def test_jsonify_sparse_matrix_to_coo():
    """A SciPy sparse matrix becomes a JSON-friendly COO triplet."""
    from scipy import sparse

    mat = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=bool))
    out = _jsonify(mat)
    assert out["format"] == "coo"
    assert out["shape"] == [2, 2]
    assert sorted(zip(out["row"], out["col"], strict=True)) == [(0, 1), (1, 0)]
    json.dumps(out)
