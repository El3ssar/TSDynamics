"""JSON / repr coercion helpers shared by the analysis result classes.

These pure helpers (``_jsonify`` for :meth:`AnalysisResult.to_dict`,
``_is_frame_scalar`` for :meth:`AnalysisResult.to_frame`, and the formatting
family ``_fmt`` / ``_sig`` / ``_state`` / ``_pct`` for ``__repr__``) were
originally inline in ``analysis/_result.py``; they live here so the split result
modules can share them without a cycle.  ``_jsonify`` recognises nested
:class:`AnalysisResult` objects, so it is imported lazily (inside the function)
to avoid a circular import with :mod:`tsdynamics.analysis._result_base`.

Formatting rules (v6, contract §4.2 rule 13)
--------------------------------------------
A result's repr **is** the answer, so the number formatting has to be readable
at every scale a dynamical system produces:

- :func:`_sig` is **significant-figure** based.  ``np.round(x, 4)`` is
  scale-blind — a Lyapunov exponent of ``2.5e-5`` rounds to ``0.`` and the repr
  then asserts something false.
- :func:`_state` renders a state vector with ``max_line_width=10_000`` and
  elides above :data:`_MAX_STATE_COMPONENTS` components.  ``numpy``'s default
  wraps at 75 columns, embedding a newline that would break the indent contract
  of every multi-line repr on any state with more than ~8 components (every
  method-of-lines field in the catalogue).
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
    # An ``int``-backed result (``CountResult`` *is* an ``int``): collapse it to a
    # plain ``int`` so the jsonified tree carries only native types — never a
    # result-object identity, and never a stale ``meta`` riding on the subclass.
    # Checked before the general-result branch (which would expand it via
    # ``to_dict``).
    if isinstance(obj, AnalysisResult) and isinstance(obj, int):
        return int(obj)
    # A nested result object (e.g. an ``Attractor`` inside an ``AttractorSet``):
    # serialize it through its own JSON-friendly ``to_dict`` rather than leaving a
    # non-serializable object behind.  ``float``/``str``-backed results stay native.
    if isinstance(obj, AnalysisResult) and not isinstance(obj, (float, str)):
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


#: Above this many components a state vector is elided in a repr (contract §4.2
#: rule 13).  Three components at each end plus ``...`` stays inside one terminal
#: line for a 3-D flow, a 9-variable climate model and a 4608-cell Gray--Scott
#: field alike.
_MAX_STATE_COMPONENTS = 8

#: Never wrap a state vector: a newline inside a repr line destroys the
#: indentation contract of the multi-line reprs (contract §4.3).
_NO_WRAP = 10_000


def _sig(value: Any, digits: int = 5) -> str:
    """Format one number to ``digits`` significant figures.

    Significant figures, not decimals: ``2.5e-05`` must not print as ``0``.
    Non-finite values print as themselves (``nan`` / ``inf`` / ``-inf``).

    Parameters
    ----------
    value : float
        The number to render.
    digits : int, default 5
        Significant figures to keep.

    Returns
    -------
    str
    """
    x = float(value)
    if not np.isfinite(x):
        return str(x)
    return f"{x:.{digits}g}"


def _pct(fraction: Any, digits: int = 1) -> str:
    """Format a 0--1 fraction as a percentage string (``0.496`` → ``49.6%``)."""
    return f"{100.0 * float(fraction):.{digits}f}%"


def _state(values: Any, *, precision: int = 4, max_components: int | None = None) -> str:
    """Render a state vector on **one** line, elided above ``max_components``.

    ``numpy.array2string`` with ``max_line_width`` pinned so nothing wraps, and
    ``threshold`` set so a long state summarises as ``[a b c ... x y z]`` rather
    than dumping a whole field.  Scale-aware: numpy switches to exponential
    notation on its own when the values need it.

    Parameters
    ----------
    values : array_like
        The state (flattened before rendering).
    precision : int, default 4
        Digits after the decimal point in positional notation.
    max_components : int, optional
        Elision threshold; defaults to :data:`_MAX_STATE_COMPONENTS`.

    Returns
    -------
    str
        Including the enclosing brackets, e.g. ``[-1.1314 -0.3394]``.
    """
    threshold = _MAX_STATE_COMPONENTS if max_components is None else max_components
    arr = np.asarray(values, dtype=float).ravel()
    return np.array2string(
        arr, precision=precision, max_line_width=_NO_WRAP, threshold=threshold, edgeitems=3
    )


def _vector(values: Any, digits: int = 4, *, max_components: int | None = None) -> str:
    """Render a list of **quantities** — significant figures, comma separated.

    The companion of :func:`_state`.  A state vector is one point and reads best
    space separated the way NumPy prints it; a list of independent measurements
    (a Lyapunov spectrum, a set of multipliers) reads as a list, and each entry
    needs its *own* scale — ``[0.916, 0.000189, -14.58]`` says three different
    things that a single shared exponent (NumPy's all-or-nothing mode:
    ``[9.1600e-01 1.8900e-04 -1.4583e+01]``) hides.

    Parameters
    ----------
    values : array_like
        The quantities (flattened before rendering).
    digits : int, default 4
        Significant figures per entry.
    max_components : int, optional
        Elision threshold; defaults to :data:`_MAX_STATE_COMPONENTS`.

    Returns
    -------
    str
        Including the enclosing brackets, e.g. ``[0.916, 0.000189, -14.58]``.
    """
    threshold = _MAX_STATE_COMPONENTS if max_components is None else max_components
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size > threshold:
        head = ", ".join(_sig(v, digits) for v in arr[:3])
        tail = ", ".join(_sig(v, digits) for v in arr[-3:])
        return f"[{head}, ..., {tail}]"
    return "[" + ", ".join(_sig(v, digits) for v in arr) + "]"


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
