r"""
Windowed recurrence quantification.

Running RQA in a sliding window turns each scalar measure into a time series that
tracks how the dynamics change along the trajectory — a drop in determinism or a
rise in laminarity flags a transition between regimes (Marwan, Romano, Thiel &
Kurths, *Phys. Rep.* **438**, 237, 2007).  Each window is quantified
independently with :func:`~tsdynamics.analysis.rqa`.

Hold the threshold fixed (``threshold=``) to compare the *absolute* level of
recurrence across windows, or fix the recurrence rate (``recurrence_rate=``) to
compare line *structure* at a constant density (the threshold is then re-derived
per window).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from tsdynamics.errors import InvalidParameterError, remedy

from .._result import AnalysisResult
from .._result_base import _unknown_result_attribute
from .._result_json import _sig
from ._common import _as_points
from .matrix import DEFAULT_RECURRENCE_RATE
from .rqa import RQAResult, rqa

__all__ = ["WindowedRQA", "windowed_rqa"]

# RQAResult scalar measures exposed as per-window arrays on WindowedRQA.
_MEASURES = (
    "recurrence_rate",
    "determinism",
    "laminarity",
    "avg_diagonal_length",
    "max_diagonal_length",
    "divergence",
    "diagonal_entropy",
    "trapping_time",
    "max_vertical_length",
)


@dataclass(frozen=True)
class WindowedRQA(AnalysisResult):
    r"""RQA measures over a sliding window.

    Each scalar measure of :class:`~tsdynamics.analysis.RQAResult` is available as
    a per-window array of the same length as :attr:`centers` (the window-centre
    sample indices), e.g. ``windowed.determinism`` or
    ``windowed.measure("laminarity")``.

    Attributes
    ----------
    centers : ndarray
        Window-centre positions in sample units (``start + (window-1)/2``).
    results : tuple[RQAResult, ...]
        The per-window results, in order.
    window : int
        Window length in samples.
    step : int
        Stride between consecutive windows in samples.
    """

    #: Served through this class's own ``__getattr__``, so ``dir()`` cannot see
    #: them; declared here so a wrong guess can still be corrected to one.
    _extra_attribute_names: ClassVar[tuple[str, ...]] = _MEASURES

    centers: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)
    results: tuple[RQAResult, ...] = field(default=(), repr=False, compare=False)
    window: int = 0
    step: int = 0

    def __len__(self) -> int:  # noqa: D105
        return len(self.results)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate the per-window measure ROWS — numbers, not wrappers."""
        return iter(self.table())

    def __getitem__(self, key: Any) -> Any:
        """Return window ``key``'s measure row, shape ``(9,)`` (:attr:`measures` names it).

        Numbers, not a wrapper (contract §4.2 rule 6): ``w[0]`` used to hand back
        an :class:`~tsdynamics.analysis.results.RQAResult`, so ``np.asarray(w)``
        was a ``(n_windows,)`` array of *objects* that arithmetics into a
        ``TypeError``.  The per-window readouts are still there, by name, at
        :attr:`details`.
        """
        return self.table()[key]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return the ``(n_windows, 9)`` table — one row per window, one column per measure."""
        arr = self.table()
        return arr.astype(dtype, copy=bool(copy)) if dtype is not None else arr

    @property
    def details(self) -> tuple[RQAResult, ...]:
        """The per-window readouts, in order — the objects ``[]`` no longer hands back.

        ``w.details[3]`` is window 3's full
        :class:`~tsdynamics.analysis.results.RQAResult`, repr, verdict and all.
        """
        return self.results

    @property
    def measures(self) -> tuple[str, ...]:
        """Names of the columns of :meth:`table` / ``np.asarray(self)`` / ``self[i]``."""
        return _MEASURES

    def table(self) -> np.ndarray:
        """Return every measure over every window as one ``(n_windows, 9)`` array.

        Columns are :attr:`measures`, in that order.

        Returns
        -------
        numpy.ndarray
        """
        return np.array(
            [[float(getattr(r, name)) for name in _MEASURES] for r in self.results], dtype=float
        ).reshape(len(self.results), len(_MEASURES))

    def measure(self, name: str) -> np.ndarray:
        """Return one RQA measure as an array over windows.

        Parameters
        ----------
        name : str
            Any scalar attribute of :class:`~tsdynamics.analysis.RQAResult`
            (e.g. ``"determinism"``, ``"laminarity"``, ``"recurrence_rate"``).
        """
        if name not in _MEASURES:
            raise ValueError(f"unknown RQA measure {name!r}; choose from {_MEASURES}.")
        return np.array([getattr(r, name) for r in self.results], dtype=float)

    def __plot_spec__(self, kind: str | None = None) -> Any:
        r"""Describe the windowed RQA as a :class:`PlotSpec`.

        Builds a ``DIAGNOSTIC_CURVE`` carrying a ``LINE`` of the **determinism**
        measure against the window-centre index — the canonical sliding-RQA view
        for spotting dynamical regime transitions (a drop in determinism flags a
        shift away from deterministic/periodic behaviour).  Read any other measure
        off :meth:`measure`.  The :mod:`tsdynamics.viz.spec` import is lazy.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` uses ``DIAGNOSTIC_CURVE``.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        centers = np.asarray(self.centers, dtype=float)
        det = np.asarray(self.measure("determinism"), dtype=float)
        return pb.spec(
            kind,
            "diagnostic_curve",
            layers=[pb.line(centers, det, label="DET")],
            xlabel="window centre",
            ylabel="determinism",
            title="Windowed RQA (determinism)",
        )

    def __getattr__(self, name: str) -> np.ndarray:
        # Expose each measure as an attribute without storing nine arrays. Only
        # consulted for names missing on the instance, so the dataclass fields are
        # untouched and unknown names (incl. copy/pickle dunders) raise plainly.
        if name in _MEASURES:
            return self.measure(name)
        if name.startswith("_"):
            raise AttributeError(name)
        # A wrong guess must name THIS class and the nearest measure: this was the
        # only one of the 32 results whose ``AttributeError`` was a bare
        # ``AttributeError("determinsm")``, naming nothing and suggesting nothing.
        raise _unknown_result_attribute(self, name)

    def _answer(self) -> str:
        """Return how many windows were measured, and their geometry."""
        return f"{len(self)} windows of {self.window} samples · step {self.step}"

    def _details(self) -> tuple[str, ...]:
        """Return the span of determinism across the windows — the regime signal."""
        if not self.results:
            return ()
        det = self.measure("determinism")
        centres = np.asarray(self.centers, dtype=float)
        span = f" · centres {_sig(centres[0], 4)}..{_sig(centres[-1], 4)}" if centres.size else ""
        return (f"DET ∈ [{_sig(det.min(), 3)}, {_sig(det.max(), 3)}]{span}",)


def windowed_rqa(
    data: Any,
    *,
    window: int | None = None,
    step: int | None = None,
    threshold: float | None = None,
    recurrence_rate: float | None = None,
    metric: str | float = "euclidean",
    theiler: int = 0,
    min_diagonal: int = 2,
    min_vertical: int = 2,
) -> WindowedRQA:
    r"""RQA in a sliding window: nonstationarity in time.

    Runs :func:`~tsdynamics.analysis.rqa` over overlapping windows.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The state points (or a 1-D series).
    window : int
        Window length in samples (``>= 2``).  Required: it is the time scale the
        analysis is *about* — long enough to hold several recurrences, short
        enough that the dynamics is stationary across it — so there is no honest
        default, and omitting it is answered with a concrete suggestion sized
        from this series.
    step : int, optional
        Stride between windows (default: ``window`` — non-overlapping).
    threshold, recurrence_rate : float, optional
        Exactly one; passed to each window's matrix (see
        :func:`~tsdynamics.analysis.recurrence_matrix`).
    metric : str or float, default "euclidean"
        Distance metric.
    theiler : int, default 0
        Excluded near-diagonal band, applied within each window.  Same caveats as
        :func:`~tsdynamics.analysis.rqa`: ``0`` is required for textbook
        ``LAM``/``TT``, a nonzero window is required for a meaningful ``L_max``
        on a densely sampled flow (each window warns when its ``L_max``
        saturates).
    min_diagonal, min_vertical : int
        Minimum line lengths (see :func:`~tsdynamics.analysis.rqa`).

    Returns
    -------
    WindowedRQA

    Raises
    ------
    InvalidInputError
        If ``data`` is a ``System``: this is a *data-first* analysis, so run the
        system and pass its trajectory.
    InvalidParameterError
        If ``window`` is missing or out of range, or ``step < 1``.  A
        ``ValueError`` subclass, so ``except ValueError`` keeps catching it.
    """
    points = _as_points(data, analysis="windowed_rqa")
    n = points.shape[0]
    if window is None:
        raise InvalidParameterError(
            "windowed_rqa needs a window: the number of samples each RQA is "
            "measured over. It is the time scale the result is about — long enough "
            "to hold several recurrences, short enough that the dynamics does not "
            "change across it — so there is no default that could be right."
            + remedy(
                f"ts.analysis.windowed_rqa(data, window={max(2, n // 20)}, "
                f"recurrence_rate={DEFAULT_RECURRENCE_RATE})",
                lead=f"A twentieth of this {n}-sample series is a place to start:",
            )
        )
    window = int(window)
    if window < 2:
        raise InvalidParameterError(
            f"window must be >= 2 — it is a count of samples, and one sample has no "
            f"recurrence structure; got {window}."
            + remedy(f"ts.analysis.windowed_rqa(data, window={max(2, n // 20)})")
        )
    if window > n:
        raise InvalidParameterError(
            f"window={window} exceeds the series length N={n}, so not one window "
            f"fits." + remedy(f"ts.analysis.windowed_rqa(data, window={max(2, n // 20)})")
        )
    step = window if step is None else int(step)
    if step < 1:
        raise InvalidParameterError(
            f"step must be >= 1 — it is the stride between windows in samples; got {step}."
            + remedy(f"ts.analysis.windowed_rqa(data, window={window}, step={max(1, window // 2)})")
        )

    starts = range(0, n - window + 1, step)
    results = tuple(
        rqa(
            points[s : s + window],
            threshold=threshold,
            recurrence_rate=recurrence_rate,
            metric=metric,
            theiler=theiler,
            min_diagonal=min_diagonal,
            min_vertical=min_vertical,
        )
        for s in starts
    )
    centers = np.array([s + (window - 1) / 2.0 for s in starts], dtype=float)
    return WindowedRQA(
        centers=centers,
        results=results,
        window=window,
        step=step,
        meta={"analysis": "windowed_rqa", "window": int(window), "step": int(step)},
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
