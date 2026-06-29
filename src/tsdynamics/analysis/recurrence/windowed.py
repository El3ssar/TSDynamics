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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._result import AnalysisResult
from ._common import _as_points
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

    centers: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)
    results: tuple[RQAResult, ...] = field(default=(), repr=False, compare=False)
    window: int = 0
    step: int = 0

    def __len__(self) -> int:  # noqa: D105
        return len(self.results)

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

    def to_plot_spec(self, kind: str | None = None) -> Any:
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
        raise AttributeError(name)

    def __repr__(self) -> str:  # noqa: D105
        return f"WindowedRQA(n_windows={len(self)}, window={self.window}, step={self.step})"


def windowed_rqa(
    data: Any,
    *,
    window: int,
    step: int | None = None,
    threshold: float | None = None,
    recurrence_rate: float | None = None,
    metric: str | float = "euclidean",
    theiler: int = 0,
    min_diagonal: int = 2,
    min_vertical: int = 2,
) -> WindowedRQA:
    r"""Run :func:`~tsdynamics.analysis.rqa` in a sliding window.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The state points (or a 1-D series).
    window : int
        Window length in samples (``>= 2``).
    step : int, optional
        Stride between windows (default: ``window`` — non-overlapping).
    threshold, recurrence_rate : float, optional
        Exactly one; passed to each window's matrix (see
        :func:`~tsdynamics.analysis.recurrence_matrix`).
    metric : str or float, default "euclidean"
        Distance metric.
    theiler : int, default 0
        Excluded near-diagonal band, applied within each window.
    min_diagonal, min_vertical : int
        Minimum line lengths (see :func:`~tsdynamics.analysis.rqa`).

    Returns
    -------
    WindowedRQA

    Raises
    ------
    ValueError
        If ``window`` is out of range or ``step < 1``.
    """
    points = _as_points(data)
    n = points.shape[0]
    window = int(window)
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}.")
    if window > n:
        raise ValueError(f"window={window} exceeds the series length N={n}.")
    step = window if step is None else int(step)
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}.")

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
