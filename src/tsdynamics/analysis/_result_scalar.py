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
from tsdynamics.analysis._result_json import _jsonify, _sig

#: Analyses whose answer is a **rate**, so its unit depends on whether the
#: subject's time is continuous: per unit time for a flow, per iteration for a
#: map.  Printing the wrong one is a wrong answer, not a cosmetic slip — the two
#: differ by the sampling interval (a factor of ~60 on a Lorenz run at dt=0.02).
_RATE_ANALYSES = frozenset({"max_lyapunov"})

#: Fixed units for the analyses that return a bare number.  A result may always
#: override by recording ``meta["unit"]`` at the call site; this table exists so
#: the ones that ship today read correctly without touching their estimators.
_UNITS: dict[str, str] = {
    "optimal_delay": "samples",
    "estimate_period": "samples",
    "embedding_dimension": "",
    "cao_dimension": "",
    "false_nearest_neighbors": "",
    "transient_time_field": "time units",
}


def _coerce_float(value: Any) -> float:
    """Return ``float(value)`` or raise — the gate for numeric dunder forwarding."""
    return float(value)


class _NumericOps(np.lib.mixins.NDArrayOperatorsMixin):
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

    **Both halves are required (contract §4.2 rule 4).**  The spelled-out
    operators above cover what a *Python* number does and are kept because
    ``__array_ufunc__`` alone would break them: NumPy consults it only when NumPy
    dispatches, and ``int.__mul__(result)`` returns :data:`NotImplemented`
    without ever reaching NumPy, so ``result * 2`` and ``result + 1`` — which
    work today — would start raising.  :class:`numpy.lib.mixins.NDArrayOperatorsMixin`
    then fills in every operator *not* spelled out (``**``, ``//``, ``%``,
    ``divmod``, the in-place forms) by routing it through
    :meth:`__array_ufunc__`, which unwraps the result to its float and hands the
    call back to NumPy.
    """

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Unwrap every result operand to its float and defer to NumPy.

        The single entry point :class:`numpy.lib.mixins.NDArrayOperatorsMixin`
        routes its generated operators through, so ``result ** 2`` and
        ``np.exp(result)`` both work and both return a plain NumPy value.
        """
        args = [float(x) if isinstance(x, _NumericOps) else x for x in inputs]
        out = kwargs.get("out")
        if out is not None:
            kwargs["out"] = tuple(np.asarray(o) if isinstance(o, _NumericOps) else o for o in out)
        return getattr(ufunc, method)(*args, **kwargs)

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
    ``.meta``, the readout ``repr``, ``.to_dict()``, the ``.plot`` seam — while
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

    # -- the readout ------------------------------------------------------

    def _unit(self) -> str:
        """Return the unit the number is quoted in (``""`` when dimensionless).

        ``meta["unit"]`` wins when an estimator records one.  Otherwise a rate
        (:data:`_RATE_ANALYSES`) is resolved against the subject's family — per
        iteration for a map, per unit time for a flow — and everything else
        reads :data:`_UNITS`.
        """
        unit = self.meta.get("unit") if self.meta else None
        if unit:
            return str(unit)
        analysis = str(self.meta.get("analysis") or "") if self.meta else ""
        if analysis in _RATE_ANALYSES:
            family = self._subject_family()
            if family == "map":
                return "per iteration"
            if family is None:
                return ""
            return "per unit time"
        return _UNITS.get(analysis, "")

    def _answer(self) -> str:
        """Return ``= <value> <unit>`` — the number, in the reader's units."""
        unit = self._unit()
        return f"= {_sig(float(self), 6)}" + (f" {unit}" if unit else "")

    def _interpretation(self) -> str | None:
        r"""Name the dynamics when the number is a Lyapunov exponent.

        Only ``max_lyapunov`` gets a verdict, and only against the same realised
        floor :class:`~tsdynamics.analysis.LyapunovSpectrum` uses: a bare
        ``lambda > 0`` test would call floating-point noise chaos.
        """
        if (self.meta.get("analysis") if self.meta else None) != "max_lyapunov":
            return None
        value = float(self)
        if not np.isfinite(value):
            return None
        return "→ chaotic (λ > 0)" if value > 0.0 else "→ regular (λ ≤ 0)"

    def __plot_spec__(self, kind: str | None = None) -> Any:
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
    the readout ``repr`` / ``.to_dict()`` / the ``.plot`` seam.

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

    def _unit(self) -> str:
        """Return the unit the count is quoted in (``"samples"`` for a delay)."""
        unit = self.meta.get("unit") if self.meta else None
        if unit:
            return str(unit)
        return _UNITS.get(str(self.meta.get("analysis") or "") if self.meta else "", "")

    def _answer(self) -> str:
        """Return ``= <count> <unit>``."""
        unit = self._unit()
        return f"= {int(self)}" + (f" {unit}" if unit else "")

    #: ``int.__repr__`` sits ahead of :class:`AnalysisResult` in the MRO, so the
    #: shared headline repr has to be claimed explicitly — otherwise a count
    #: reprs as a bare ``9`` and the console never says *what* was measured.
    __repr__ = AnalysisResult.__repr__

    def __str__(self) -> str:
        """Return the plain integer text — a count **is** its number.

        The one deliberate exception to "``str`` is the headline".  A
        :class:`CountResult` is an ``int`` subclass whose whole point is being a
        drop-in, and ``int.__str__ is object.__str__``, so without this override
        ``f"tau={c}"`` renders ``tau=CountResult(28)``.  That output is committed
        to the repository, inside
        ``docs/assets/figures/analysis/embedding.svg``.
        """
        return repr(int(self))

    def __format__(self, spec: str) -> str:
        """Format as an ``int``, falling back to ``float`` for float codes.

        ``f"{tau:d}"`` and ``f"{tau:>4}"`` keep integer semantics; ``f"{tau:.3f}"``
        formats the same number as a float instead of raising.
        """
        if not spec:
            return str(self)
        try:
            return int.__format__(self, spec)
        except (TypeError, ValueError):
            return format(float(self), spec)

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the value and provenance.

        Parameters
        ----------
        full : bool, default False
            Also emit the derived quantities (here: the ``unit`` the count is
            quoted in).
        """
        data = {"value": int(self), "meta": _jsonify(self.meta)}
        if full:
            data.update(self._derived())
        return data

    def _derived(self) -> dict[str, Any]:
        """Return the unit the repr quotes, so an export can carry it too."""
        return {"unit": self._unit()}

    def __plot_spec__(self, kind: str | None = None) -> Any:
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
