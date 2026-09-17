r"""
Recurrence matrices.

A recurrence plot (Eckmann, Kamphorst & Ruelle, *Europhys. Lett.* **4**, 973,
1987) records, for every pair of states on a trajectory, whether they are
mutually close:

.. math::

    R_{ij} = \Theta\!\big(\varepsilon - \lVert x_i - x_j \rVert\big),

i.e. :math:`R_{ij}=1` when :math:`x_j` lies within :math:`\varepsilon` of
:math:`x_i`.  The threshold is set either directly (``threshold``) or implicitly
through a target recurrence rate (``recurrence_rate``), the matrix density that
fixes :math:`\varepsilon` from the distribution of pairwise distances.

The matrix is symmetric and is stored **sparse** (``scipy.sparse``): the
recurrent pairs are found with a k-d tree range search
(:meth:`scipy.spatial.cKDTree.query_pairs`), so construction scales to long
series without forming the dense :math:`N \times N` array.  The **line of
identity** (:math:`i=j`) and the **Theiler band** (:math:`|i-j| \le w`) carry no
recurrences here — the diagonal is trivially recurrent and a few neighbouring
samples of a densely sampled flow are spuriously close, biasing every line-based
statistic (Theiler, *Phys. Rev. A* **34**, 2427, 1986); ``theiler`` sets
``w`` (default ``0`` keeps the off-diagonal recurrences and drops only the line
of identity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from tsdynamics.errors import InvalidParameterError, remedy

from .._common import reject_system
from .._result import AnalysisResult
from .._result_json import _sig
from ._common import _as_points, _metric_p, _threshold_for_rate

__all__ = ["DEFAULT_RECURRENCE_RATE", "RecurrenceMatrix", "recurrence_matrix"]

#: The recurrence rate every "I don't know the scale of my data" error message
#: recommends.  A recurrence plot needs *some* notion of "close", and of the two
#: ways to say it only the density can be chosen without already knowing the
#: scale of the data — which is exactly what someone reaching for
#: ``recurrence_matrix(data)`` does not yet know.  5 % is the density RQA
#: practice recommends (Marwan et al., *Phys. Rep.* **438**, 237, 2007, §3.2.1;
#: Webber & Zbilut's "a few percent"): dense enough that diagonal lines form,
#: sparse enough that the plot is not saturated.  It is deliberately *not* a
#: silent default for ``recurrence_rate=`` — the threshold is the one modelling
#: choice a recurrence plot cannot make for you, so the call asks rather than
#: guesses, and the error names the number to paste.
DEFAULT_RECURRENCE_RATE = 0.05

#: The default rendered for an error message's runnable line.
_RATE_TEXT = repr(DEFAULT_RECURRENCE_RATE)

#: Largest lattice side :meth:`RecurrenceMatrix._display_field` will build.  A
#: recurrence plot bigger than the figure has pixels cannot be drawn honestly
#: marker-by-marker, so the field is binned to at most this side and each pixel
#: carries the local recurrence density.  1000 is comfortably above any figure a
#: screen or a journal will show (a 10 in x 300 dpi page is 3000 px across a
#: *whole* figure, and the axes are a fraction of it) and bounds the field at
#: 8 MB, so the cap is invisible in the picture and decisive in the memory.
MAX_DISPLAY_SIDE = 1000


@dataclass(frozen=True)
class RecurrenceMatrix(AnalysisResult):
    r"""A binary recurrence matrix with the parameters it was built from.

    The matrix is symmetric, sparse and excludes the line of identity (plus the
    Theiler band when ``theiler_window > 0``).  Pass it straight to
    :func:`~tsdynamics.analysis.rqa` for quantification, or read
    :attr:`recurrence_rate` / densify with :meth:`toarray` for inspection.

    Attributes
    ----------
    matrix : scipy.sparse.csr_matrix
        The :math:`N \times N` boolean recurrence matrix.
    epsilon : float
        The distance threshold actually used.
    metric : str or float
        The metric the threshold is measured in.
    theiler_window : int
        Excluded near-diagonal band :math:`|i-j| \le w`.
    """

    matrix: Any = field(default=None, repr=False, compare=False)
    epsilon: float = 0.0
    metric: str | float = "euclidean"
    theiler_window: int = 0

    @property
    def size(self) -> int:
        """Number of states ``N`` (the matrix is ``N x N``); ``0`` when unpopulated.

        The field defaults to ``None`` (it carries no matrix until an estimator
        fills it in), so this reports ``0`` rather than raising — the repr reads
        it, and a repr must never be the thing that raises in a console.
        """
        return 0 if self.matrix is None else int(self.matrix.shape[0])

    @property
    def recurrence_rate(self) -> float:
        r"""Matrix density :math:`RR = \#\{R_{ij}=1\}/N^2` (``0`` when unpopulated)."""
        n = self.size
        return float(self.matrix.nnz) / float(n * n) if n else 0.0

    def toarray(self) -> np.ndarray:
        """Return the dense boolean ``(N, N)`` matrix (materialises ``O(N^2)``)."""
        return cast(np.ndarray, self.matrix.toarray().astype(bool))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:  # noqa: D105
        # NumPy 2.0 passes ``copy`` into ``__array__``; ``toarray`` always
        # materialises a fresh buffer, so honoring it costs nothing.
        arr = self.toarray()
        return arr.astype(dtype, copy=bool(copy)) if dtype is not None else arr

    def to_frame(self) -> Any:
        """Return the recurrent pairs as a tidy ``(nnz, 2)`` ``i`` / ``j`` table.

        The base :meth:`~tsdynamics.analysis._result.AnalysisResult.to_frame`
        would drop the matrix and tabulate ``['epsilon', 'metric',
        'theiler_window']`` — the settings, with **the answer** missing (contract
        §4.2 rule 8).  The matrix is never densified: the stored COO coordinates
        are exactly the recurrences, so the frame is ``nnz`` rows, not
        :math:`N^2`.  The settings ride on ``frame.attrs``.

        Returns
        -------
        pandas.DataFrame
            Columns ``i`` / ``j``, one row per recurrent pair.

        Raises
        ------
        ImportError
            If :mod:`pandas` is not installed.
        """
        pd = self._require_pandas()
        if self.matrix is None:
            frame = pd.DataFrame({"i": [], "j": []})
        else:
            coo = self.matrix.tocoo()
            frame = pd.DataFrame({"i": coo.row, "j": coo.col})
        frame.attrs["meta"] = dict(self.meta) if self.meta else {}
        frame.attrs["epsilon"] = float(self.epsilon)
        frame.attrs["metric"] = self.metric
        frame.attrs["theiler_window"] = int(self.theiler_window)
        return frame

    def _display_field(self) -> tuple[NDArray[np.float64], int]:
        r"""Return the recurrence field at a bounded display resolution, and its side.

        Built from the stored COO ``(i, j)`` indices by binning them onto an
        ``n × n`` lattice with ``n = min(size, MAX_DISPLAY_SIDE)`` and dividing
        by the exact number of matrix cells each bin covers — so every pixel is
        the **local recurrence density** in :math:`[0, 1]`.  When the matrix
        already fits (``n == size``) each bin is one cell and the field is the
        exact 0/1 matrix, bit-for-bit.

        The work is :math:`O(\#\text{recurrences})` and the memory
        :math:`O(n^2)` with ``n`` capped, so the :math:`O(N^2)` densification a
        plain ``todense()`` would cost is never paid — which was the whole reason
        this route drew a sparse scatter in the first place.
        """
        size = int(self.size)
        side = min(size, MAX_DISPLAY_SIDE)
        coo = self.matrix.tocoo()
        rows = (np.asarray(coo.row, dtype=np.int64) * side) // size
        cols = (np.asarray(coo.col, dtype=np.int64) * side) // size
        hist = np.bincount(rows * side + cols, minlength=side * side).astype(float)
        # Exact cells-per-bin along each axis: the bins are not all the same
        # width when `side` does not divide `size`, and dividing by the average
        # would tint whole rows and columns of the picture.
        per_axis = np.bincount((np.arange(size) * side) // size, minlength=side).astype(float)
        cells = np.outer(per_axis, per_axis).ravel()
        field: NDArray[np.float64] = (hist / np.where(cells > 0.0, cells, 1.0)).reshape(side, side)
        return field, side

    def __plot_spec__(self, kind: str | None = None) -> Any:
        r"""Describe this recurrence matrix as a backend-agnostic :class:`PlotSpec`.

        Builds a ``RECURRENCE_PLOT`` as an ``IMAGE`` of the recurrence field at a
        bounded display resolution (see :meth:`_display_field`), on a square
        (``aspect="equal"``) canvas with both axes labelled by the state index.

        It used to be a **sparse scatter**, one marker per recurrent pair, and
        that is a wrong picture at any realistic record length: measured, a
        1494×1494 matrix at a verified 5% density draws 112 060 markers into
        about 137 000 device pixels, so they overlap and the result is a
        near-solid black square — destroying exactly the diagonal-line structure
        ``DET`` / ``L_max`` / ``ENTR`` measure, on the one picture that *is* the
        deliverable in RQA work.  Binning to the display resolution keeps the
        density faithful (a pixel is the local recurrence rate) **and** keeps the
        cost :math:`O(\#\text{recurrences})`, so nothing is traded away for it.
        The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never
        pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"recurrence_plot"``).  ``None``
            uses ``RECURRENCE_PLOT``.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        field, side = self._display_field()
        n = int(self.size)
        binned = "" if side == n else f", binned to {side}×{side}"
        return pb.spec(
            kind,
            "recurrence_plot",
            layers=[
                pb.image(
                    field,
                    x=np.linspace(0.0, float(n), side),
                    y=np.linspace(0.0, float(n), side),
                    label="recurrence",
                    style={"cmap": "binary"},
                )
            ],
            aspect="equal",
            xlabel="$i$",
            ylabel="$j$",
            title=f"recurrence plot (RR = {self.recurrence_rate:.3g}{binned})",
        )

    def _answer(self) -> str:
        """Return the plot's size and its two defining numbers."""
        n = int(self.size)
        return f"{n}×{n} · RR = {self.recurrence_rate:.3f} · ε = {_sig(self.epsilon, 4)}"

    def _context(self) -> str | None:
        """Return the metric and the Theiler window the matrix was built with."""
        bits = [str(self.metric), f"theiler={self.theiler_window}"]
        system = self._system_label()
        if system:
            bits.insert(0, system)
        return ", ".join(bits)

    def _derived(self) -> dict[str, Any]:
        """Export the size and recurrence rate the repr reports."""
        return {"size": int(self.size), "recurrence_rate": float(self.recurrence_rate)}


def recurrence_matrix(
    data: Any,
    *,
    threshold: float | None = None,
    recurrence_rate: float | None = None,
    metric: str | float = "euclidean",
    theiler: int = 0,
) -> RecurrenceMatrix:
    r"""Build a recurrence matrix from a trajectory or point set.

    Parameters
    ----------
    data : Trajectory or array-like, shape (N, dim)
        The state points (a :class:`~tsdynamics.data.Trajectory` or a raw array;
        a 1-D series is treated as a single scalar component).  For phase-space
        recurrence of a scalar measurement, embed it first
        (:func:`tsdynamics.analysis.embed`).
    threshold : float, optional
        Fixed distance threshold :math:`\varepsilon`, in the units of the data.
        Give this *or* ``recurrence_rate``, never both.
    recurrence_rate : float, optional
        Target matrix density in ``(0, 1)``; :math:`\varepsilon` is chosen from
        the distribution of pairwise distances so the realised
        :attr:`~RecurrenceMatrix.recurrence_rate` is close to it.  The realised
        rate can differ slightly because distances are discrete (and sampled for
        very long series).  This is the one to reach for when the scale of the
        data is not already known; :data:`DEFAULT_RECURRENCE_RATE` is the value
        RQA practice recommends.
    metric : str or float, default "euclidean"
        Distance metric (``"euclidean"``, ``"manhattan"``, ``"chebyshev"``, or a
        numeric Minkowski exponent).  ``"chebyshev"`` (the maximum norm) is the
        common RQA choice.
    theiler : int, default 0
        Exclude the near-diagonal band :math:`|i-j| \le w`.  ``0`` keeps every
        off-diagonal recurrence and drops only the line of identity; raise it to
        a few autocorrelation times for densely sampled flows.

    Returns
    -------
    RecurrenceMatrix

    Raises
    ------
    InvalidInputError
        If ``data`` is a ``System``: this is a *data-first* analysis, so run the
        system and pass its trajectory.
    InvalidParameterError
        If neither or both of ``threshold`` / ``recurrence_rate`` are given (they
        set the same quantity two different ways), or if either is out of range.
        A ``ValueError`` subclass, so ``except ValueError`` keeps catching it.
    ValueError
        If the Theiler window leaves no valid pairs.

    References
    ----------
    J.-P. Eckmann, S. O. Kamphorst and D. Ruelle, "Recurrence plots of dynamical
    systems", *Europhys. Lett.* **4**, 973 (1987).
    """
    from scipy import sparse
    from scipy.spatial import cKDTree

    # Before any keyword validation: handing a System to a data-first analysis is
    # the mistake to name, and a message about threshold=/recurrence_rate= would
    # send the caller down the wrong path entirely.
    reject_system(data, analysis="recurrence_matrix")
    if threshold is not None and recurrence_rate is not None:
        raise InvalidParameterError(
            f"pass exactly one of threshold= or recurrence_rate=: they set the same "
            f"quantity two different ways (the threshold is what a target rate is "
            f"solved for), and you gave threshold={threshold!r} *and* "
            f"recurrence_rate={recurrence_rate!r}."
            + remedy(
                f"ts.analysis.recurrence_matrix(data, recurrence_rate={recurrence_rate!r})",
                f"ts.analysis.recurrence_matrix(data, threshold={threshold!r})",
                lead="Either fix the density, or fix the distance:",
            )
        )
    if threshold is None and recurrence_rate is None:
        raise InvalidParameterError(
            "a recurrence plot needs a scale, so pass exactly one of threshold= (a "
            "distance, in the units of the data) or recurrence_rate= (the fraction "
            "of point pairs to count as recurrent)."
            + remedy(
                f"ts.analysis.recurrence_matrix(data, recurrence_rate={_RATE_TEXT})",
                lead=(
                    "If you do not yet know the scale of your data, ask for the "
                    f"density RQA practice recommends ({_RATE_TEXT}):"
                ),
            )
        )
    points = _as_points(data, analysis="recurrence_matrix")
    n = points.shape[0]
    p = _metric_p(metric)
    w = int(theiler)
    if w < 0:
        raise ValueError("theiler must be non-negative.")
    if w >= n - 1:
        raise ValueError(f"theiler={w} excludes every pair for N={n}; reduce it.")

    if threshold is not None:
        eps = float(threshold)
        if not (eps > 0.0):
            raise InvalidParameterError(
                f"threshold is a distance in the units of the data, so it must be "
                f"positive; got {threshold!r}. If you do not know the scale of the "
                f"data, ask for a density instead."
                + remedy(f"ts.analysis.recurrence_matrix(data, recurrence_rate={_RATE_TEXT})")
            )
    else:
        assert recurrence_rate is not None  # guaranteed by the defaulting above
        rate = float(recurrence_rate)
        if not (0.0 < rate < 1.0):
            raise InvalidParameterError(
                f"recurrence_rate is the fraction of point pairs counted as "
                f"recurrent, so it must lie in (0, 1); got {recurrence_rate!r}"
                + (" — that looks like a percentage." if rate > 1.0 else ".")
                + remedy(
                    f"ts.analysis.recurrence_matrix(data, recurrence_rate="
                    f"{min(0.99, rate / 100) if rate > 1.0 else _RATE_TEXT})"
                )
            )
        eps = _threshold_for_rate(points, rate, p, w)

    tree = cKDTree(points)
    pairs = tree.query_pairs(r=eps, p=p, output_type="ndarray")  # (M, 2), i < j
    if pairs.size:
        i, j = pairs[:, 0], pairs[:, 1]
        keep = (j - i) > w
        i, j = i[keep], j[keep]
    else:
        i = j = np.empty(0, dtype=np.intp)

    # Symmetrise: store both (i, j) and (j, i); the diagonal is never recurrent here.
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    ones = np.ones(rows.size, dtype=bool)
    mat = sparse.csr_matrix((ones, (rows, cols)), shape=(n, n))
    return RecurrenceMatrix(
        matrix=mat,
        epsilon=eps,
        metric=metric,
        theiler_window=w,
        meta={"analysis": "recurrence_matrix", "epsilon": float(eps), "theiler": int(w)},
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
