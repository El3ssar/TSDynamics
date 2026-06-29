"""Scalar-valued result classes: :class:`ScalarResult` and :class:`CountResult`.

Split out of ``analysis/_result.py``; carries the numeric-protocol mixin
(:class:`_NumericOps`) the float-backed result wraps itself in, plus the int
subclass :class:`CountResult` that genuinely *is* an ``int``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from tsdynamics.analysis._result_base import AnalysisResult
from tsdynamics.analysis._result_json import _jsonify


def _coerce_float(value: Any) -> float:
    """Return ``float(value)`` or raise — the gate for numeric dunder forwarding."""
    return float(value)


class _NumericOps:
    """Mixin giving a result the full numeric protocol of ``float(self)``.

    A :class:`ScalarResult` wraps a bare ``float``/``int`` return so it can carry
    ``.meta`` and the result surface, but it must stay a *drop-in* for the number
    it replaces: ``result > 0.9``, ``result == pytest.approx(x)``, ``abs(result)``
    and ``2 * result`` all have to keep working without callers unwrapping it.

    Dunder methods are resolved on the *type*, never via ``__getattr__``, so the
    operators are spelled out here.  Each forwards to ``float(self)`` and coerces
    the other operand with :func:`_coerce_float`; an operand that is not
    float-convertible (an ``ndarray``, a ``pytest.approx`` sentinel) yields
    :data:`NotImplemented` so Python falls back to *its* reflected operator — which
    is exactly how ``result == pytest.approx(x)`` resolves (approx then calls
    ``float(self)`` itself).
    """

    def __float__(self) -> float:  # noqa: D105
        return _coerce_float(self.value)  # type: ignore[attr-defined]

    def __int__(self) -> int:  # noqa: D105
        return int(_coerce_float(self.value))  # type: ignore[attr-defined]

    def __bool__(self) -> bool:  # noqa: D105
        return bool(_coerce_float(self.value))  # type: ignore[attr-defined]

    def __round__(self, ndigits: int | None = None) -> float | int:  # noqa: D105
        return round(float(self), ndigits) if ndigits is not None else round(float(self))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:  # noqa: D105
        # NumPy 2.0 passes ``copy`` into ``__array__``; honor it (a 0-d array of a
        # Python float is always a fresh buffer, so ``copy=False`` is also safe).
        arr = np.asarray(float(self))
        if dtype is not None:
            arr = arr.astype(dtype, copy=bool(copy))
        elif copy:
            arr = arr.copy()
        return arr

    # -- comparisons (NotImplemented → reflected op, e.g. pytest.approx) ------

    def __eq__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) == _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __ne__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) != _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __lt__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) < _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __le__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) <= _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __gt__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) > _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __ge__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) >= _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __hash__(self) -> int:  # noqa: D105
        return hash(float(self))

    # -- arithmetic ----------------------------------------------------------

    def __add__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) + _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) - _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __rsub__(self, other: Any) -> Any:  # noqa: D105
        try:
            return _coerce_float(other) - float(self)
        except (TypeError, ValueError):
            return NotImplemented

    def __mul__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) * _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:  # noqa: D105
        try:
            return float(self) / _coerce_float(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __rtruediv__(self, other: Any) -> Any:  # noqa: D105
        try:
            return _coerce_float(other) / float(self)
        except (TypeError, ValueError):
            return NotImplemented

    def __neg__(self) -> float:  # noqa: D105
        return -float(self)

    def __pos__(self) -> float:  # noqa: D105
        return +float(self)

    def __abs__(self) -> float:  # noqa: D105
        return abs(float(self))


@dataclass(frozen=True, eq=False)
class ScalarResult(_NumericOps, AnalysisResult):
    """A single scalar measurement that still behaves like its number.

    Wraps a bare ``float`` return (a maximal Lyapunov exponent, an entropy, a
    0--1-test ``K``, …) so it carries the :class:`AnalysisResult` surface —
    ``.meta``, ``.summary()``, ``.to_dict()``, the ``.plot`` seam — while
    ``float(result)`` and every comparison / arithmetic operator keep working via
    :class:`_NumericOps`, so it is a drop-in for the value it replaces.

    Subclasses may add domain context fields (e.g. ``base``, ``normalized``);
    they must re-apply ``@dataclass(frozen=True, eq=False)`` so the numeric
    ``__eq__`` / ``__hash__`` are not regenerated by the dataclass machinery.

    Attributes
    ----------
    value : float
        The measured number.  ``float(result)`` returns it.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("value",)

    value: float = 0.0

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe the scalar as a one-point :class:`PlotSpec` (rarely plotted).

        A lone number has no natural figure; this emits a minimal
        ``DIAGNOSTIC_CURVE`` carrying the value so the ``.plot`` seam resolves
        uniformly.  The :mod:`tsdynamics.viz.spec` import is lazy.
        """
        from . import _plotbuilder as pb

        return pb.spec(
            kind,
            "diagnostic_curve",
            layers=[pb.markers(np.array([0.0]), np.array([float(self)]))],
            xlabel="index",
            ylabel="value",
            title=type(self).__name__,
            meta=self.meta,
        )


class CountResult(int, AnalysisResult):
    """A scalar *integer* result that genuinely **is** an ``int``.

    Subclasses ``int`` (rather than wrapping one) so a count read off the data —
    an estimated delay from ``optimal_delay``, a dimension — is a complete drop-in
    for the bare integer it replaces: ``isinstance(result, int)`` holds, it indexes
    and slices arrays, it survives ``delay=result`` round-trips into estimators
    that type-check their arguments, and all integer arithmetic / comparisons work
    natively.  It *also* carries the :class:`AnalysisResult` surface — ``.meta`` /
    ``.summary()`` / ``.to_dict()`` / the ``.plot`` seam.

    Attributes
    ----------
    value : int
        The measured count (an alias for the integer itself).
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("value",)

    def __new__(cls, value: Any = 0, *, meta: Mapping[str, Any] | None = None) -> CountResult:
        """Construct the integer (``int.__new__``); ``meta`` is set in ``__init__``."""
        return super().__new__(cls, int(value))

    def __init__(self, value: Any = 0, *, meta: Mapping[str, Any] | None = None) -> None:
        """Attach provenance; ``AnalysisResult`` is frozen, so set it via ``object``."""
        object.__setattr__(self, "meta", dict(meta) if meta else {})

    @property
    def value(self) -> int:
        """The measured count (the integer value itself)."""
        return int(self)

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}({int(self)})"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the value and provenance."""
        return {"value": int(self), "meta": _jsonify(self.meta)}

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe the count as a one-point :class:`PlotSpec` (rarely plotted)."""
        from . import _plotbuilder as pb

        return pb.spec(
            kind,
            "diagnostic_curve",
            layers=[pb.markers(np.array([0.0]), np.array([float(self)]))],
            xlabel="index",
            ylabel="value",
            title=type(self).__name__,
            meta=self.meta,
        )
