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
per-diagonal / per-column Python loop) is formed.  By symmetry the diagonal
statistics are gathered from the upper triangle alone (every ratio and length is
unchanged by the doubling).

**The line of identity is treated per orientation, as the definitions require.**
:math:`R_{ii}=1` holds by definition (a state is within :math:`\varepsilon` of
itself), and the matrices built here drop it because it is one trivial infinite
*diagonal* line.  The **vertical** structures, however, are defined on the matrix
that carries it (Marwan et al. 2007, §3.5): the sojourn of a laminar state is one
vertical line *through* the diagonal.  So at ``theiler == 0`` — where the
excluded band :math:`|i-j| \le 0` *is* the line of identity — the vertical pass
restores :math:`R_{ii}=1` before reading runs; without that every vertical line
is bisected by the missing diagonal point, halving ``TT`` and under-reporting
``V_max``.  The diagonal pass keeps it excluded.

At ``theiler > 0`` the line of identity lies **inside** the band the caller
excluded, so it is *not* restored: re-inserting it would fabricate ``N`` isolated
recurrence points in a band that was explicitly removed, and the resulting
matrix would be neither the textbook one nor the Theiler-filtered one.  All three
vertical measures (``LAM``, ``TT``, ``V_max``) are therefore read off one
consistent matrix — the filtered one — and are not comparable to published
values; see the ``theiler`` argument.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .._common import reject_system
from .._result import AnalysisResult
from .._result_base import _unknown_result_attribute
from .._result_json import _sig
from ._common import _diagonal_run_lengths, _longest_run_on_diagonal, _vertical_run_lengths
from .matrix import RecurrenceMatrix, recurrence_matrix

__all__ = ["RQAResult", "rqa"]

#: DET at or above which the repr names the structure deterministic, and at or
#: below which it names it stochastic.  DET is the fraction of recurrence points
#: on diagonal lines: uncorrelated noise leaves isolated points (DET -> 0) and a
#: deterministic signal repeats stretches of trajectory (DET -> 1).  The band
#: between them is left unnamed; a mid-range DET is a real answer that no single
#: word describes.
_DETERMINISTIC_DET = 0.6
_STOCHASTIC_DET = 0.2


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
        length :math:`\ge` ``min_vertical``.  At ``theiler == 0`` the vertical
        statistics are read off the matrix **with** the line of identity (Marwan
        et al. 2007), so the denominator is ``nnz + N``; at ``theiler > 0`` the
        line of identity is inside the excluded band and the denominator is
        ``nnz``.
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
        ``TT`` — mean length of the vertical lines counted by ``LAM``, on the
        same matrix ``LAM`` is read from (at ``theiler == 0`` that matrix carries
        the line of identity, so a state trapped for ``v`` consecutive samples
        contributes one line of length ``v``, not two of length ``~v/2``).
    max_vertical_length : int
        ``V_max`` — longest vertical line, with **no** ``min_vertical`` filter
        (the vertical analogue of ``L_max``).  At ``theiler == 0`` it is ``1``
        when nothing recurs, since :math:`R_{ii}=1` is itself a length-1 vertical
        line; at ``theiler > 0`` (no line of identity) it is ``0``.
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

    #: The literature abbreviations the repr prints, mapped to the attributes
    #: that hold them.  The abbreviations are *right* — they are what every RQA
    #: paper calls these quantities — but the repr showing ``L_max`` while only
    #: ``max_diagonal_length`` resolves is a name the library taught and then
    #: refused, in a library where nearly every other wrong guess is translated.
    _ABBREVIATIONS: ClassVar[dict[str, str]] = {
        "DET": "determinism",
        "DIV": "divergence",
        "ENTR": "diagonal_entropy",
        "L": "avg_diagonal_length",
        "L_MAX": "max_diagonal_length",
        "LAM": "laminarity",
        "RR": "recurrence_rate",
        "TT": "trapping_time",
        "V_MAX": "max_vertical_length",
    }

    #: What ``dir()`` and the wrong-guess message advertise beyond the fields.
    _extra_attribute_names: ClassVar[tuple[str, ...]] = tuple(_ABBREVIATIONS)

    def _printed_names(self) -> dict[str, str]:
        """Map the literature abbreviations the repr prints to the attributes."""
        return {
            "DET": "determinism",
            "ENTR": "diagonal_entropy",
            "LAM": "laminarity",
            "L_max": "max_diagonal_length",
            "RR": "recurrence_rate",
        }

    def __getattr__(self, name: str) -> Any:
        """Serve the literature abbreviation the repr prints (``q.L_max``).

        Case-insensitively, because the repr writes ``L_max`` and the papers
        write ``Lmax`` and ``LMAX``.  Anything else falls through to the shared
        wrong-guess message.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        attribute = type(self)._ABBREVIATIONS.get(name.replace("_", "").upper()) or type(
            self
        )._ABBREVIATIONS.get(name.upper())
        if attribute is not None:
            return getattr(self, attribute)
        raise _unknown_result_attribute(self, name)

    def __plot_spec__(self, kind: str | None = None) -> Any:
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
        from .. import _plotbuilder as pb

        labels = [lbl for lbl, _ in self._BAR_MEASURES]
        cat = np.arange(len(labels), dtype=float)
        values = np.array([float(getattr(self, attr)) for _, attr in self._BAR_MEASURES])
        return pb.spec(
            kind,
            "categorical_bar",
            layers=[pb.bar(values, cat=cat, label="RQA measures")],
            xlabel="measure",
            xscale="categorical",
            xcategories=labels,
            ylabel="value",
            ylimits=(0.0, 1.0),
            title=f"RQA  DET = {self.determinism:.3g}, LAM = {self.laminarity:.3g}",
        )

    def _answer(self) -> str:
        """Return the four headline RQA measures."""
        return (
            f"DET = {self.determinism:.3f} · LAM = {self.laminarity:.3f} · "
            f"L_max = {self.max_diagonal_length} · ENTR = {self.diagonal_entropy:.3f}"
        )

    @property
    def applicable(self) -> bool:
        r"""Whether there were any recurrence points to quantify at all.

        ``False`` when the matrix holds no recurrence point (``RR = 0``) or no
        diagonal line was counted: ``DET``/``LAM``/``ENTR`` are then ``0`` by
        vacuity, not by measurement.  Before v6 that vacuum printed the full
        verdict — ``rqa(np.random.normal(size=300), threshold=1e-12)`` returned
        ``DET = 0.000 … stochastic (few diagonal lines)``, a confident reading of
        a matrix with zero points — the same defect
        :attr:`~tsdynamics.analysis.results.WadaResult.applicable` was introduced
        for, one subpackage away.

        Returns
        -------
        bool
        """
        rr = float(self.recurrence_rate)
        return bool(np.isfinite(rr) and rr > 0.0 and np.asarray(self.diagonal_lengths).size > 0)

    @property
    def deterministic(self) -> bool | None:
        """Whether the diagonal statistics say the signal is deterministic.

        ``None`` — never ``False`` — in the unnamed middle of the ``DET`` range,
        and whenever the readout is not :attr:`applicable`.

        Returns
        -------
        bool or None
        """
        if not self.applicable:
            return None
        det = float(self.determinism)
        if not np.isfinite(det):
            return None
        if det >= _DETERMINISTIC_DET:
            return True
        if det <= _STOCHASTIC_DET:
            return False
        return None

    def _interpretation(self) -> str | None:
        """Name the structure the diagonal statistics show, when it is clear-cut.

        DET is the fraction of recurrence points lying on diagonal lines: a
        deterministic signal repeats stretches of its trajectory and drives it
        toward 1, while uncorrelated noise leaves only isolated points and drives
        it toward 0 (Marwan et al., 2007).  Both regimes are named only at the
        ends of the range; the middle is left unnamed rather than rounded — and
        an **empty** matrix is named as such rather than read as "stochastic".
        """
        if not self.applicable:
            return (
                f"not applicable — no recurrence points at ε = {_sig(self.epsilon, 4)}, "
                "so there are no lines to count"
            )
        verdict = self.deterministic
        if verdict is None:
            return None
        return "deterministic" if verdict else "stochastic (few diagonal lines)"

    def _details(self) -> tuple[str, ...]:
        """Return the settings the measures were computed under."""
        return (
            f"(N={self.size}, RR={self.recurrence_rate:.3f}, ε={_sig(self.epsilon, 4)}, "
            f"l_min={self.min_diagonal}, v_min={self.min_vertical})",
        )

    def _derived(self) -> dict[str, Any]:
        """Export the applicability flag and the verdict the repr reports.

        ``determinism`` is re-emitted as ``None`` when the readout does not
        apply, matching
        :meth:`~tsdynamics.analysis.results.WadaResult._derived`: a number that
        was not measured must not export as ``0.0``.
        """
        data: dict[str, Any] = {"applicable": self.applicable, "deterministic": self.deterministic}
        if not self.applicable:
            data["determinism"] = None
        return data


def _line_stats(lengths: np.ndarray, min_length: int) -> tuple[float, float, int]:
    """``(fraction_in_long_lines, mean_long_length, max_line_length)``.

    ``fraction`` and the mean are over the recurrence-point statistics counted by
    the ratio measures (DET / LAM and L / TT): the denominator is the total
    recurrence points on lines of this orientation (``lengths.sum()``), and the
    mean is over lines no shorter than ``min_length``.  The **maximum** is the
    longest line in the *full* unfiltered histogram (no ``min_length`` cut) — the
    Marwan et al. (2007) definition of ``L_max`` / ``V_max`` — which feeds
    ``DIV = 1/L_max``.  Whether the line of identity is present in ``lengths`` is
    the caller's business: it is excluded from the diagonal histogram and
    restored in the vertical one (see the module docstring).
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


#: Fraction of the longest possible line at which a **tangential-motion**
#: ``L_max`` is called saturated.  Measured over flows, maps, noise and periodic
#: signals, the two populations are far apart: a tangential ``L_max`` (the run
#: lives on the diagonal next to the excluded band) covers 0.44-1.00 of the
#: series — Lorenz ``dt=0.01`` 0.987, Lorenz ``dt=0.005`` 1.000, Halvorsen 1.000,
#: a constant series 1.000 — while every non-tangential case measured sits at
#: 0.02 or below (Henon 0.005, white noise 0.004, an embedded sinusoid at
#: ``theiler=12`` exactly 0.000, since its ``L_max`` lives on the *period*
#: diagonal, which is legitimate).  ``0.5`` is the wide gap between them.
_SATURATION_FRACTION = 0.5


def _warn_if_lmax_saturated(mat: Any, l_max: int, n: int, theiler: int) -> None:
    r"""Warn when ``L_max`` is set by tangential motion rather than by dynamics.

    With a Theiler window ``w`` the shortest surviving diagonal index is
    :math:`k = w + 1`, whose diagonal holds :math:`N - w - 1` entries.  When a
    flow is sampled finely enough that consecutive states lie within
    :math:`\varepsilon`, *that* diagonal is recurrent essentially end to end and
    supplies ``L_max`` — so ``L_max`` reports the series length, not the
    dynamics, and :math:`DIV = 1/L_{\max}` (advertised as related to the largest
    Lyapunov exponent) degenerates with it.  Measured on Lorenz at ``RR=0.05``:
    ``dt=0.01``, ``N=3001`` gives ``L_max = 2962`` and ``DIV = 3.38e-4``, against
    ``L_max = 1162``, ``DIV = 8.61e-4`` for the same run at ``theiler=17``.

    The test is **causal, not a threshold on ``L_max`` itself**: it fires only
    when the longest line in the whole matrix is the one on the first admissible
    diagonal, and that line covers at least :data:`_SATURATION_FRACTION` of the
    series.  Testing ``L_max`` alone is wrong in both directions — a clean
    periodic signal legitimately reaches ``L_max = N - p`` on its *period*
    diagonal (an embedded sinusoid measures 0.98 of the ceiling, and warning
    there would be a false alarm), while requiring exact equality
    ``L_max == N - w - 1`` means one non-recurrent link anywhere along the band
    diagonal silences the guard entirely — which is what let the ``N = 3001``
    Lorenz run above through at 98.7 % saturation.

    Silently returning the saturated number is the failure this guard exists to
    prevent; the fix is a Theiler window of a few autocorrelation times.
    """
    ceiling = n - theiler - 1
    if l_max < 2 or ceiling < 2 or l_max < _SATURATION_FRACTION * ceiling:
        return
    if _longest_run_on_diagonal(mat, theiler + 1) != l_max:
        return  # the longest line lives away from the band: genuine recurrence
    warnings.warn(
        f"L_max = {l_max} of a possible {ceiling} (N={n}, theiler={theiler}), and that "
        f"line lies on diagonal k={theiler + 1}, the one next to the excluded band: "
        "consecutive samples are recurrent with each other, so L_max and "
        f"DIV = 1/L_max = {1.0 / l_max:.3g} are saturated by tangential motion and carry "
        "no dynamical information. This is expected for a densely sampled flow at "
        "theiler=0 — pass a Theiler window of a few autocorrelation times "
        "(e.g. theiler=optimal_delay(x)).",
        UserWarning,
        stacklevel=3,
    )


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
        matrix).  ``0`` is the standard convention and the only value for which
        the **vertical** measures (``LAM`` / ``TT`` / ``V_max``) are the textbook
        ones: the Theiler window is a *diagonal*-line correction, and a nonzero
        band cuts a hole through every vertical line (see Notes).  For the
        diagonal measures on a densely sampled flow a nonzero window is
        essential — ``theiler=0`` there saturates ``L_max`` and ``rqa`` warns.
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

    Warns
    -----
    UserWarning
        When ``L_max`` is set by **tangential motion** — the longest line in the
        matrix is the one on diagonal ``k = theiler + 1`` (the one next to the
        excluded band) and it spans at least half the series, so ``L_max`` and
        ``DIV`` report the sampling, not the dynamics.  Raise ``theiler``.  A
        genuinely periodic signal, whose longest line sits on its *period*
        diagonal, does not warn however long that line is.

    Notes
    -----
    ``L_max`` and ``V_max`` are the longest diagonal / vertical lines over the
    *full* line-length histograms (no ``min_*`` filter), matching Marwan et al.
    (2007); ``DIV = 1 / L_max``.  The ``min_diagonal`` / ``min_vertical`` cut
    applies only to the ratio and mean measures (DET / LAM / L / TT / ENTR).

    **Line of identity.**  The diagonal measures are read off the matrix without
    :math:`R_{ii}` (one trivial infinite line).  At ``theiler == 0`` the vertical
    measures are read off the matrix **with** it, as Marwan et al. (2007) §3.5
    define them — a laminar sojourn is one vertical line *through* the diagonal,
    and dropping the diagonal point bisects it (measured on 60 identical states
    followed by 140 separated ones: ``TT`` 30.5 and ``V_max`` 59 without the
    restoration, against the closed-form 60 and 60).

    **Theiler window and the vertical measures.**  ``theiler > 0`` removes the
    band :math:`|i - j| \le w` from the matrix, which is what the diagonal
    statistics need but punches a hole through every vertical line: ``LAM`` /
    ``TT`` / ``V_max`` are then computed on a truncated matrix and are *not*
    comparable to published values.  Quantify laminarity at ``theiler=0`` (the
    default) and use a nonzero window only for the diagonal measures.  The
    excluded band *contains* the line of identity, so :math:`R_{ii}` is **not**
    restored in that case — fabricating those ``N`` points inside the removed
    band would make ``LAM`` / ``TT`` / ``V_max`` describe a matrix that is
    neither the textbook one nor the one the caller asked for.

    References
    ----------
    N. Marwan, M. C. Romano, M. Thiel and J. Kurths, "Recurrence plots for the
    analysis of complex systems", *Phys. Rep.* **438**, 237 (2007).

    Examples
    --------
    >>> import numpy as np
    >>> import tsdynamics as ts
    >>> t = np.linspace(0.0, 100.0, 1000)
    >>> emb = ts.analysis.embed(np.sin(t), dimension=2, delay=5)
    >>> res = ts.analysis.rqa(emb, recurrence_rate=0.05)
    >>> 0.0 <= res.determinism <= 1.0
    True
    """
    # Before any keyword validation: a System handed to this data-first analysis
    # must be named as such, not reported as a missing threshold.
    reject_system(data, analysis="rqa")
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
    # Vertical structures are defined on the matrix carrying R_ii = 1 (Marwan et
    # al. 2007 §3.5) -- but only when R_ii is not itself inside the excluded
    # Theiler band.  `_vertical_run_lengths` owns that decision, so LAM, TT and
    # V_max are always read off one and the same matrix.
    vert = _vertical_run_lengths(rm.matrix, theiler=rm.theiler_window)

    det, avg_diag, l_max = _line_stats(diag, lmin)
    lam, tt, v_max = _line_stats(vert, vmin)
    entr = _diagonal_entropy(diag, lmin)
    divergence = 1.0 / l_max if l_max > 0 else float("inf")
    _warn_if_lmax_saturated(rm.matrix, l_max, rm.size, rm.theiler_window)

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
