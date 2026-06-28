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

Line lengths are read straight off the sparse matrix in one vectorised pass —
diagonal lines are the consecutive runs of the upper-triangle entries grouped by
diagonal index :math:`k = j - i`, vertical lines the consecutive runs of each
column's stored row indices — so no dense :math:`N \times N` array (nor any
per-diagonal / per-column Python loop) is formed.  The line of identity is
excluded (see :func:`~tsdynamics.analysis.recurrence_matrix`), and by symmetry
the diagonal statistics are gathered from the upper triangle alone (every ratio
and length is unchanged by the doubling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .._result import AnalysisResult
from ._common import _diagonal_run_lengths, _vertical_run_lengths
from .matrix import RecurrenceMatrix, recurrence_matrix

__all__ = ["RQAResult", "rqa"]


@dataclass(frozen=True)
class RQAResult(AnalysisResult):
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
        ``L_max`` — longest diagonal line, excluding the line of identity and
        with **no** ``min_diagonal`` filter (Marwan et al. 2007); ``0`` only when
        there are no diagonal recurrence lines at all.
    divergence : float
        ``DIV`` :math:`= 1/L_{\max}` (``inf`` only when there are no diagonal
        lines, i.e. ``L_max == 0``).
    diagonal_entropy : float
        ``ENTR`` — Shannon entropy (nats) of the diagonal line-length
        distribution.
    trapping_time : float
        ``TT`` — mean length of the vertical lines counted by ``LAM``.
    max_vertical_length : int
        ``V_max`` — longest vertical line, with **no** ``min_vertical`` filter
        (the vertical analogue of ``L_max``).
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
    diagonal_lengths: np.ndarray = field(repr=False, compare=False)
    vertical_lengths: np.ndarray = field(repr=False, compare=False)

    #: The scalar RQA measures shown on the ``CATEGORICAL_BAR`` readout, as
    #: ``(short label, attribute name)`` pairs.  These are the headline
    #: structure quantifiers that share the ``[0, 1]`` / small-number range a
    #: bar chart reads cleanly (the unbounded ``L_max`` / ``V_max`` / ``DIV``
    #: stay off the bars — they are in :meth:`summary` and ``to_dict``).
    _BAR_MEASURES: ClassVar[tuple[tuple[str, str], ...]] = (
        ("RR", "recurrence_rate"),
        ("DET", "determinism"),
        ("LAM", "laminarity"),
        ("ENTR", "diagonal_entropy"),
    )

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe the scalar RQA measures as a :class:`PlotSpec` bar readout.

        Builds a ``CATEGORICAL_BAR`` whose bars are the headline structure
        quantifiers — ``RR`` (recurrence rate), ``DET`` (determinism), ``LAM``
        (laminarity) and ``ENTR`` (diagonal-line entropy) — one bar per measure,
        the category axis carrying the measure labels.  This is the at-a-glance
        readout of "how deterministic / laminar is this trajectory"; the
        unbounded measures (``L_max`` / ``V_max`` / ``DIV``) stay in
        :meth:`summary` rather than crushing the bar scale.  No line-length
        histogram is walked — the values are the already-computed scalar fields.
        The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never
        pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"categorical_bar"``).  ``None``
            uses ``CATEGORICAL_BAR``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.CATEGORICAL_BAR
        labels = [lbl for lbl, _ in self._BAR_MEASURES]
        cat = np.arange(len(labels), dtype=float)
        values = np.array([float(getattr(self, attr)) for _, attr in self._BAR_MEASURES])
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=f"RQA  DET = {self.determinism:.3g}, LAM = {self.laminarity:.3g}",
            x=Axis(label="measure", scale="categorical", categories=labels),
            y=Axis(label="value", limits=(0.0, 1.0)),
            layers=[Layer(PlotKind.BAR, {"cat": cat, "y": values}, label="RQA measures")],
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RQAResult(N={self.size}, RR={self.recurrence_rate:.3g}, "
            f"DET={self.determinism:.3g}, LAM={self.laminarity:.3g}, "
            f"L_max={self.max_diagonal_length}, ENTR={self.diagonal_entropy:.3g})"
        )


def _line_stats(lengths: np.ndarray, min_length: int) -> tuple[float, float, int]:
    """``(fraction_in_long_lines, mean_long_length, max_line_length)``.

    ``fraction`` and the mean are over the recurrence-point statistics counted by
    the ratio measures (DET / LAM and L / TT): the denominator is the total
    recurrence points on lines of this orientation (``lengths.sum()``), and the
    mean is over lines no shorter than ``min_length``.  The **maximum** is the
    longest line in the *full* unfiltered histogram (no ``min_length`` cut) — the
    Marwan et al. (2007) definition of ``L_max`` / ``V_max`` (the line of identity
    is already absent from the recurrence matrix), which feeds ``DIV = 1/L_max``.
    """
    total = float(lengths.sum())
    long = lengths[lengths >= min_length]
    frac = float(long.sum()) / total if total > 0.0 else 0.0
    # L_max / V_max are the longest line overall, *not* filtered by min_length
    # (so a near-random series with only short lines still reports L_max >= 1 and
    # a finite DIV instead of L_max = 0 / DIV = inf).
    max_len = int(lengths.max()) if lengths.size else 0
    mean_long = float(long.mean()) if long.size else 0.0
    return frac, mean_long, max_len


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
    theiler: int = 0,
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
    theiler : int, default 0
        Excluded near-diagonal band (ignored when ``data`` is a recurrence
        matrix).
    min_diagonal : int, default 2
        Shortest diagonal line counted toward ``DET`` / ``L`` / ``ENTR``.
    min_vertical : int, default 2
        Shortest vertical line counted toward ``LAM`` / ``TT``.

    Returns
    -------
    RQAResult
        The recurrence-quantification measures (RR, DET, LAM, L, L_max, DIV,
        ENTR, TT, V_max) plus the raw diagonal / vertical line-length histograms.

    Raises
    ------
    ValueError
        If ``min_diagonal`` or ``min_vertical`` is ``< 1``, or if ``data`` is a
        :class:`~tsdynamics.analysis.RecurrenceMatrix` and a ``threshold`` /
        ``recurrence_rate`` is also given (build the matrix with those instead).
        Matrix-building errors (no threshold/rate, out-of-range values, an
        over-wide Theiler window) propagate from
        :func:`~tsdynamics.analysis.recurrence_matrix`.

    Notes
    -----
    ``L_max`` and ``V_max`` are the longest diagonal / vertical lines over the
    *full* line-length histograms (no ``min_*`` filter), matching Marwan et al.
    (2007); ``DIV = 1 / L_max``.  The ``min_diagonal`` / ``min_vertical`` cut
    applies only to the ratio and mean measures (DET / LAM / L / TT / ENTR).

    References
    ----------
    N. Marwan, M. C. Romano, M. Thiel and J. Kurths, "Recurrence plots for the
    analysis of complex systems", *Phys. Rep.* **438**, 237 (2007).

    Examples
    --------
    >>> import numpy as np
    >>> import tsdynamics as ts
    >>> t = np.linspace(0.0, 100.0, 1000)
    >>> emb = ts.embed(np.sin(t), dimension=2, delay=5)
    >>> res = ts.rqa(emb, recurrence_rate=0.05)
    >>> 0.0 <= res.determinism <= 1.0
    True
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
            theiler=theiler,
        )

    diag = _diagonal_run_lengths(rm.matrix)
    vert = _vertical_run_lengths(rm.matrix)

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
        meta={
            "analysis": "rqa",
            "size": int(rm.size),
            "epsilon": float(rm.epsilon),
            "theiler": int(rm.theiler_window),
        },
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
