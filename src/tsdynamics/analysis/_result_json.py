"""JSON / repr coercion helpers shared by the analysis result classes.

These three pure helpers (``_jsonify`` for :meth:`AnalysisResult.to_dict`,
``_is_frame_scalar`` for :meth:`AnalysisResult.to_frame`, and ``_fmt`` for the
compact ``__repr__`` / :meth:`summary`) were originally inline in
``analysis/_result.py``; they live here so the split result modules can share
them without a cycle.  ``_jsonify`` recognises nested :class:`AnalysisResult`
objects, so it is imported lazily (inside the function) to avoid a circular
import with :mod:`tsdynamics.analysis._result_base`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


def _jsonify(obj: Any) -> Any:
    """Coerce ``obj`` into a JSON-serializable form (arrays → lists).

    Handles the value shapes analysis results actually carry: NumPy arrays and
    scalars, mappings, ``list``/``tuple``/``set`` containers (recursively), nested
    result objects (an ``Attractor`` inside an ``AttractorSet``, an ``RQAResult``
    inside a ``WindowedRQA``), SciPy sparse matrices (a recurrence matrix), and the
    plain state-space dataclasses (``Box``/``Ball``/``Grid``) a result may embed.
    Plain Python scalars pass through unchanged; anything else is returned as-is
    (the caller owns exotic payloads).
    """
    from tsdynamics.analysis._result_base import AnalysisResult

    # Complex scalars/arrays (e.g. a spiral fixed point's stability eigenvalues):
    # JSON has no complex type, so emit the real part when the imaginary part is
    # zero (a real-valued-but-complex-dtype result stays a plain float) and a
    # [real, imag] pair otherwise. Checked before the ndarray/generic branches,
    # whose ``tolist()``/``item()`` would otherwise hand back a Python ``complex``
    # that ``json.dumps`` rejects.
    if isinstance(obj, (complex, np.complexfloating)):
        z = complex(obj)
        return z.real if z.imag == 0.0 else [z.real, z.imag]
    if isinstance(obj, np.ndarray) and np.iscomplexobj(obj):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonify(v) for v in obj]
    # A nested result object (e.g. an ``Attractor`` inside an ``AttractorSet``):
    # serialize it through its own JSON-friendly ``to_dict`` rather than leaving a
    # non-serializable object behind.  ``int``-backed results (``CountResult`` *is*
    # an ``int``) are excluded so they stay native JSON scalars.
    if isinstance(obj, AnalysisResult) and not isinstance(obj, (int, float, str)):
        return obj.to_dict()
    # A SciPy sparse matrix (the recurrence matrix): emit a COO triplet so it
    # round-trips without densifying to ``O(N^2)``.
    if hasattr(obj, "tocoo") and hasattr(obj, "shape") and hasattr(obj, "nnz"):
        coo = obj.tocoo()
        return {
            "format": "coo",
            "shape": list(coo.shape),
            "row": coo.row.tolist(),
            "col": coo.col.tolist(),
            "data": _jsonify(coo.data),
        }
    # A plain data-carrying dataclass embedded in a result (``Box``/``Ball``/
    # ``Grid``): serialize its fields recursively.  ``AnalysisResult`` is handled
    # above (its curated ``to_dict``); the scalar/class guard keeps int-backed
    # results native and never expands a dataclass *type*.
    if is_dataclass(obj) and not isinstance(obj, (int, float, str, type)):
        return {f.name: _jsonify(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def _is_frame_scalar(value: Any) -> bool:
    """Return whether ``value`` belongs in a single DataFrame cell.

    True for plain scalars and 0-d arrays; False for containers and n-d arrays.
    Tests the type directly rather than via ``numpy.ndim`` so a ragged/mixed
    container (e.g. ``(1, [2, 3])``) is excluded rather than raising when NumPy
    tries to coerce it to an array.
    """
    if isinstance(value, (list, tuple, set, frozenset, Mapping)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return True


def _fmt(value: Any) -> str:
    """Format a single value for the compact :meth:`AnalysisResult.__repr__`."""
    if value is None:
        return repr(value)
    if isinstance(value, (bool, np.bool_)):
        return repr(bool(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _fmt(value.item())
        if value.size <= 4:
            return "[" + ", ".join(_fmt(v) for v in value.ravel().tolist()) + "]"
        return f"array(shape={tuple(value.shape)})"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        if len(value) <= 4:
            inner = ", ".join(_fmt(v) for v in value)
            return f"[{inner}]" if isinstance(value, list) else f"({inner})"
        return f"{type(value).__name__}(len={len(value)})"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."
