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

    True for plain scalars and 0-d arrays; False for containers, n-d arrays, and
    the structured payloads a result embeds (a nested
    :class:`~tsdynamics.analysis._result_base.AnalysisResult`, a state-space
    ``Box``/``Ball``/``Grid``).  Tests the type directly rather than via
    ``numpy.ndim`` so a ragged/mixed container (e.g. ``(1, [2, 3])``) is excluded
    rather than raising when NumPy tries to coerce it to an array.

    The structured exclusion is v6: ``BasinsResult.to_frame()`` used to hand back
    a cell holding ``{'lo': [-2.0, -2.0], 'hi': ...}`` and ``BasinFractions`` a
    cell holding a whole ``AttractorSet`` — neither is a table.  Those payloads
    ride on ``frame.attrs`` instead.
    """
    from dataclasses import is_dataclass

    from tsdynamics.analysis._result_base import AnalysisResult

    if isinstance(value, (list, tuple, set, frozenset, Mapping)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    # An int-backed result (``CountResult``) *is* its number and stays a cell.
    if isinstance(value, AnalysisResult):
        return isinstance(value, (int, float, str))
    if is_dataclass(value) and not isinstance(value, (int, float, str, type)):
        return False
    # A SciPy sparse matrix (a recurrence plot) is a whole image, not a cell.
    return not (hasattr(value, "tocoo") and hasattr(value, "nnz"))


#: Widest vector field :func:`_spread` will turn into columns.
_MAX_SPREAD = 32


def _content_fields(item: Any) -> tuple[str, ...] | None:
    """Return the fields a TABLE should carry — the content, not the repr selection.

    ``_display_fields`` answers *what the one-line repr shows*, and a field
    marked ``repr=False`` (a fixed point's ``eigenvalues``, an attractor's
    ``points``) is deliberately absent from it.  A table is the other question,
    so it reads the dataclass fields and drops only ``meta``.

    Lives here rather than in ``_result_collection`` because **every**
    :meth:`~tsdynamics.analysis._result_base.AnalysisResult.to_frame` uses it
    since v6: a singular member used to tabulate its booleans and drop its
    coordinates while its plural tabulated them properly — the same information,
    two answers.
    """
    import dataclasses

    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return tuple(f.name for f in dataclasses.fields(item) if f.name != "meta")
    display = getattr(item, "_display_fields", None)
    return tuple(display()) if callable(display) else None


#: Fields whose components are STATE components, and therefore carry the
#: system's declared variable names.  A Lyapunov spectrum's ``values`` and a
#: fixed point's ``eigenvalues`` are indexed by *mode*, not by variable, so they
#: stay numbered; a point, a centre, a state is indexed by variable.
_STATE_VALUED_FIELDS = frozenset({"center", "centre", "point", "points", "state", "x"})


def _spread(name: str, value: Any, labels: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Expand a vector display field into one column per component.

    Returns ``{}`` for anything that is not a short 1-D numeric sequence, so a
    nested structure is dropped rather than rendered as a repr string.

    ``labels`` names the components when the field is a **state** vector and the
    producing system declared names, so a pendulum's fixed point tabulates as
    ``x_theta`` / ``x_omega`` rather than ``x0`` / ``x1``.  The names already
    reached ``traj["theta"]``, ``system.info`` and every plot axis; this is the
    one place they used to be dropped.
    """
    try:
        arr = np.asarray(value)
    except Exception:  # pragma: no cover - defensive
        return {}
    if arr.ndim != 1 or arr.size == 0 or arr.size > _MAX_SPREAD:
        return {}
    if not np.issubdtype(arr.dtype, np.number):
        return {}
    if labels is not None and len(labels) == arr.size:
        return {f"{name}_{labels[i]}": _jsonify(v) for i, v in enumerate(arr.tolist())}
    return {f"{name}{i}": _jsonify(v) for i, v in enumerate(arr.tolist())}


#: Above this many entries an array is a **raw distribution**, not a value, and
#: ``to_dict()`` summarises it instead of inlining it.  Measured: printing one
#: ``RQAResult.to_dict()`` produced 236 KB, almost all of it the
#: ``diagonal_lengths`` histogram — on the call that is the natural way to put a
#: result in a report.  ``to_dict(full=True)`` inlines everything.
_BULK_ARRAY_ELEMENTS = 256


def _jsonify_bounded(value: Any, name: str) -> Any:
    """JSON-coerce ``value``, summarising a bulk array rather than inlining it.

    The key is always present — a summarised field yields
    ``{"shape": [...], "dtype": "...", "omitted": "pass full=True"}`` — so no
    consumer gains a ``KeyError``, only a smaller payload.
    """
    arr = value if isinstance(value, np.ndarray) else None
    if arr is None or arr.size <= _BULK_ARRAY_ELEMENTS:
        return _jsonify(value)
    return {
        "shape": [int(n) for n in arr.shape],
        "dtype": str(arr.dtype),
        "omitted": f"{arr.size} values — pass to_dict(full=True) for {name!r}",
    }


def _row_for(item: Any, *, variables: Any = None) -> dict[str, Any]:
    """Return one tidy DataFrame row for ``item`` — scalars kept, vectors spread.

    The row is the item's **content**: its dataclass fields, plus the derived
    quantities its repr reports (``_derived``), which are the answer as often as
    the fields are — a Lyapunov spectrum's ``kaplan_yorke``, an attractor's
    ``center``.  A value that is itself a result / a mapping / a grid / a sparse
    matrix / a higher-dimensional array is **dropped** rather than parked in a
    cell: a table cell holding a ``dict`` or an ``AnalysisResult`` is not a table
    (contract §4.2 rule 8).
    """
    names = _content_fields(item)
    if names is None:
        return {"value": _jsonify(item)}
    row: dict[str, Any] = {}
    meta = getattr(item, "meta", None)
    declared = variables
    if declared is None and isinstance(meta, Mapping):
        declared = meta.get("variables")
    state_labels = tuple(str(v) for v in declared) if declared else None

    def _put(name: str, value: Any) -> None:
        if _is_frame_scalar(value):
            row[name] = _jsonify(value)
        else:
            labels = state_labels if name in _STATE_VALUED_FIELDS else None
            row.update(_spread(name, value, labels))

    for name in names:
        try:
            _put(name, getattr(item, name))
        except AttributeError:
            continue
    derived = getattr(item, "_derived", None)
    if callable(derived):
        for name, value in derived().items():
            if name not in row:
                _put(name, value)
    return row


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
