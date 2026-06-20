r"""
Recurrence quantification analysis (RQA).

RQA reduces a recurrence matrix to scalar measures of the small-scale structure
its diagonal and vertical lines encode (Zbilut & Webber, *Phys. Lett. A* **171**,
199, 1992; Marwan, Romano, Thiel & Kurths, *Phys. Rep.* **438**, 237, 2007):

- **Diagonal** lines mark stretches where two trajectory segments evolve in
  parallel — the signature of *deterministic* dynamics.  Their statistics give
  the **recurrence rate** (RR), **determinism** (DET), the mean and maximum line
  length (L, L_max), the **divergence** (DIV = 1/L_max, related to the largest
  Lyapunov exponent) and the line-length **entropy** (ENTR).
- **Vertical** lines mark states the system is trapped near for a while — the
  signature of *laminar* / intermittent phases.  They give the **laminarity**
  (LAM) and **trapping time** (TT).

Line lengths are read straight off the sparse matrix — diagonals densify one at a
time, vertical lines come from the consecutive runs of each column's stored row
indices — so no dense :math:`N \times N` array is formed.  The line of identity
is excluded (see :func:`~tsdynamics.analysis.recurrence_matrix`), and by symmetry
the diagonal statistics are gathered from the upper triangle alone (every ratio
and length is unchanged by the doubling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._common import _run_lengths, _runs_from_sorted
from .matrix import RecurrenceMatrix, recurrence_matrix

__all__ = ["RQAResult", "rqa"]


@dataclass(frozen=True)
class RQAResult:
    r"""Recurrence-quantification measures of one recurrence matrix.

    Attributes
    ----------
    recurrence_rate : float
        ``RR`` — density of recurrence points, :math:`\#\{R_{ij}=1\}/N^2`.
    determinism : float
        ``DET`` — fraction of recurrence points that lie on diagonal lines of
        length :math:`\ge` ``min_diagonal``.
    laminarity : float
        ``LAM`` — fraction of recurrence points that lie on vertical lines of
        length :math:`\ge` ``min_vertical``.
    avg_diagonal_length : float
        ``L`` — mean length of the diagonal lines counted by ``DET``.
    max_diagonal_length : int
        ``L_max`` — longest diagonal line (excluding the line of identity).
    divergence : float
        ``DIV`` :math:`= 1/L_{\max}` (``inf`` when there are no diagonal lines).
    diagonal_entropy : float
        ``ENTR`` — Shannon entropy (nats) of the diagonal line-length
        distribution.
    trapping_time : float
        ``TT`` — mean length of the vertical lines counted by ``LAM``.
    max_vertical_length : int
        ``V_max`` — longest vertical line.
    size : int
        Number of states ``N``.
    epsilon : float
        Threshold the matrix was built with.
    theiler_window : int
        Excluded near-diagonal band.
    min_diagonal, min_vertical : int
        Minimum line lengths counted as diagonal / vertical lines.
    diagonal_lengths, vertical_lengths : ndarray
        Raw line-length histograms (every run, before the ``min_*`` cut), kept
        for inspection / plotting.
    """

    recurrence_rate: float
    determinism: float
    laminarity: float
    avg_diagonal_length: float
    max_diagonal_length: int
    divergence: float
    diagonal_entropy: float
    trapping_time: float
    max_vertical_length: int
    size: int
    epsilon: float
    theiler_window: int
    min_diagonal: int
    min_vertical: int
    diagonal_lengths: np.ndarray = field(repr=False)
    vertical_lengths: np.ndarray = field(repr=False)

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe the diagonal line-length distribution as a :class:`PlotSpec`.

        Builds a ``DIAGNOSTIC_CURVE`` spec carrying a ``HISTOGRAM`` layer of the
        diagonal line lengths — the distribution ``DET``, ``L``, ``L_max`` and
        ``ENTR`` are all read off — with the length on ``x`` and the count on
        ``y``.  The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec
        never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"diagnostic_curve"``).  ``None``
            uses ``DIAGNOSTIC_CURVE``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.DIAGNOSTIC_CURVE
        lengths = np.asarray(self.diagonal_lengths, dtype=float)
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=f"RQA  DET = {self.determinism:.3g}, ENTR = {self.diagonal_entropy:.3g}",
            x=Axis(label="diagonal line length"),
            y=Axis(label="count"),
            layers=[Layer(PlotKind.HISTOGRAM, {"x": lengths}, label="diagonal lengths")],
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RQAResult(N={self.size}, RR={self.recurrence_rate:.3g}, "
            f"DET={self.determinism:.3g}, LAM={self.laminarity:.3g}, "
            f"L_max={self.max_diagonal_length}, ENTR={self.diagonal_entropy:.3g})"
        )


def _diagonal_lengths(mat: Any) -> np.ndarray:
    """Lengths of every diagonal recurrence line in the upper triangle."""
    n = mat.shape[0]
    out: list[np.ndarray] = []
    for k in range(1, n):
        runs = _run_lengths(np.asarray(mat.diagonal(k), dtype=bool))
        if runs.size:
            out.append(runs)
    return np.concatenate(out) if out else np.empty(0, dtype=np.intp)


def _vertical_lengths(mat: Any) -> np.ndarray:
    """Lengths of every vertical recurrence line (consecutive rows per column)."""
    csc = mat.tocsc()
    csc.sort_indices()  # ascending row indices within each column
    indptr, indices = csc.indptr, csc.indices
    out: list[np.ndarray] = []
    for j in range(csc.shape[1]):
        runs = _runs_from_sorted(indices[indptr[j] : indptr[j + 1]])
        if runs.size:
            out.append(runs)
    return np.concatenate(out) if out else np.empty(0, dtype=np.intp)


def _line_stats(lengths: np.ndarray, min_length: int) -> tuple[float, float, int]:
    """``(fraction_in_long_lines, mean_long_length, max_long_length)``.

    ``fraction`` is over *all* recurrence points on lines of this orientation
    (the denominator is ``lengths.sum()``); the mean and max are over lines no
    shorter than ``min_length``.
    """
    total = float(lengths.sum())
    long = lengths[lengths >= min_length]
    frac = float(long.sum()) / total if total > 0.0 else 0.0
    if long.size == 0:
        return frac, 0.0, 0
    return frac, float(long.mean()), int(long.max())


def _diagonal_entropy(lengths: np.ndarray, min_length: int) -> float:
    """Shannon entropy (nats) of the length distribution of lines >= ``min_length``."""
    long = lengths[lengths >= min_length]
    if long.size == 0:
        return 0.0
    counts = np.bincount(long)
    probs = counts[counts > 0] / float(long.size)
    return float(-(probs * np.log(probs)).sum())


def rqa(
    data: Any,
    *,
    threshold: float | None = None,
    recurrence_rate: float | None = None,
    metric: str | float = "euclidean",
    theiler_window: int = 0,
    min_diagonal: int = 2,
    min_vertical: int = 2,
) -> RQAResult:
    r"""Recurrence quantification of a trajectory, series, or recurrence matrix.

    Parameters
    ----------
    data : RecurrenceMatrix, Trajectory, or array-like
        A prebuilt :class:`~tsdynamics.analysis.RecurrenceMatrix`, or a point set
        / series from which one is built with the parameters below.
    threshold, recurrence_rate : float, optional
        Threshold or target recurrence rate when ``data`` is not already a
        recurrence matrix — exactly one, as in
        :func:`~tsdynamics.analysis.recurrence_matrix`.  Both must be omitted when
        ``data`` is a :class:`~tsdynamics.analysis.RecurrenceMatrix`.
    metric : str or float, default "euclidean"
        Distance metric (ignored when ``data`` is a recurrence matrix).
    theiler_window : int, default 0
        Excluded near-diagonal band (ignored when ``data`` is a recurrence
        matrix).
    min_diagonal : int, default 2
        Shortest diagonal line counted toward ``DET`` / ``L`` / ``ENTR``.
    min_vertical : int, default 2
        Shortest vertical line counted toward ``LAM`` / ``TT``.

    Returns
    -------
    RQAResult

    Raises
    ------
    ValueError
        On conflicting matrix-building arguments or ``min_* < 1``.

    References
    ----------
    N. Marwan, M. C. Romano, M. Thiel and J. Kurths, "Recurrence plots for the
    analysis of complex systems", *Phys. Rep.* **438**, 237 (2007).
    """
    if int(min_diagonal) < 1 or int(min_vertical) < 1:
        raise ValueError("min_diagonal and min_vertical must be >= 1.")
    lmin, vmin = int(min_diagonal), int(min_vertical)

    if isinstance(data, RecurrenceMatrix):
        if threshold is not None or recurrence_rate is not None:
            raise ValueError(
                "threshold=/recurrence_rate= do not apply when data is a RecurrenceMatrix; "
                "build the matrix with the desired parameters instead."
            )
        rm = data
    else:
        rm = recurrence_matrix(
            data,
            threshold=threshold,
            recurrence_rate=recurrence_rate,
            metric=metric,
            theiler_window=theiler_window,
        )

    diag = _diagonal_lengths(rm.matrix)
    vert = _vertical_lengths(rm.matrix)

    det, avg_diag, l_max = _line_stats(diag, lmin)
    lam, tt, v_max = _line_stats(vert, vmin)
    entr = _diagonal_entropy(diag, lmin)
    divergence = 1.0 / l_max if l_max > 0 else float("inf")

    return RQAResult(
        recurrence_rate=rm.recurrence_rate,
        determinism=det,
        laminarity=lam,
        avg_diagonal_length=avg_diag,
        max_diagonal_length=l_max,
        divergence=divergence,
        diagonal_entropy=entr,
        trapping_time=tt,
        max_vertical_length=v_max,
        size=rm.size,
        epsilon=rm.epsilon,
        theiler_window=rm.theiler_window,
        min_diagonal=lmin,
        min_vertical=vmin,
        diagonal_lengths=diag,
        vertical_lengths=vert,
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
