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
from typing import Any

import numpy as np

from .._result import AnalysisResult
from ._common import _as_points, _metric_p, _threshold_for_rate

__all__ = ["RecurrenceMatrix", "recurrence_matrix"]


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
        """Number of states ``N`` (the matrix is ``N x N``)."""
        return self.matrix.shape[0]

    @property
    def recurrence_rate(self) -> float:
        r"""Matrix density :math:`RR = \#\{R_{ij}=1\}/N^2`."""
        n = self.size
        return float(self.matrix.nnz) / float(n * n) if n else 0.0

    def toarray(self) -> np.ndarray:
        """Return the dense boolean ``(N, N)`` matrix (materialises ``O(N^2)``)."""
        return self.matrix.toarray().astype(bool)

    def __array__(self, dtype: Any = None) -> np.ndarray:  # noqa: D105
        arr = self.toarray()
        return arr.astype(dtype) if dtype is not None else arr

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe this recurrence matrix as a backend-agnostic :class:`PlotSpec`.

        Builds a ``RECURRENCE_PLOT`` image layer of the dense boolean
        :math:`N \times N` matrix on a square (``aspect="equal"``) canvas, with
        both axes labelled by the state index :math:`i`.  The matrix is
        densified with :meth:`toarray` (``O(N^2)``).  The
        :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls a
        plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"recurrence_plot"``).  ``None``
            uses ``RECURRENCE_PLOT``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.RECURRENCE_PLOT
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            aspect="equal",
            title=f"recurrence plot (RR = {self.recurrence_rate:.3g})",
            x=Axis(label="$i$"),
            y=Axis(label="$j$"),
            layers=[Layer(PlotKind.IMAGE, {"c": self.toarray()}, style={"cmap": "binary"})],
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RecurrenceMatrix(N={self.size}, eps={self.epsilon:.4g}, "
            f"metric={self.metric!r}, theiler={self.theiler_window}, "
            f"RR={self.recurrence_rate:.4g})"
        )


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
        Fixed distance threshold :math:`\varepsilon`.  Exactly one of
        ``threshold`` or ``recurrence_rate`` must be given.
    recurrence_rate : float, optional
        Target matrix density in ``(0, 1)``; :math:`\varepsilon` is chosen from
        the distribution of pairwise distances so the realised
        :attr:`~RecurrenceMatrix.recurrence_rate` is close to it.  The realised
        rate can differ slightly because distances are discrete (and sampled for
        very long series).
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
    ValueError
        If neither or both of ``threshold`` / ``recurrence_rate`` are given, if
        either is out of range, or if the Theiler window leaves no valid pairs.

    References
    ----------
    J.-P. Eckmann, S. O. Kamphorst and D. Ruelle, "Recurrence plots of dynamical
    systems", *Europhys. Lett.* **4**, 973 (1987).
    """
    from scipy import sparse
    from scipy.spatial import cKDTree

    if (threshold is None) == (recurrence_rate is None):
        raise ValueError("pass exactly one of threshold= or recurrence_rate=.")
    points = _as_points(data)
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
            raise ValueError(f"threshold must be positive, got {threshold!r}.")
    else:
        rate = float(recurrence_rate)
        if not (0.0 < rate < 1.0):
            raise ValueError(f"recurrence_rate must be in (0, 1), got {recurrence_rate!r}.")
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
