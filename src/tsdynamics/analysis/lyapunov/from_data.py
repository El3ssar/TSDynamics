"""Maximal Lyapunov exponent from a measured time series.

Two closely-related neighbour-divergence estimators that need only a scalar (or
multivariate) recording — no equations of motion:

- **Kantz (1994)** — average, over reference points, of the log of the *mean*
  distance between the forward images of all neighbours found within a ball of
  radius ``eps``.  Robust to noise because it averages over a neighbourhood.
- **Rosenstein, Collins & De Luca (1993)** — tracks the single nearest
  neighbour of each reference point; cheaper, well-suited to short records.

Both build the ``S(k)`` *stretching curve* — the mean log divergence after ``k``
samples — whose slope over the linear scaling region is the maximal Lyapunov
exponent.  The signal is first reconstructed in an ``m``-dimensional delay
embedding (Takens 1981).

References
----------
H. Kantz, "A robust method to estimate the maximal Lyapunov exponent of a time
series", *Physics Letters A* **185** (1994) 77-87.

M. T. Rosenstein, J. J. Collins & C. J. De Luca, "A practical method for
calculating largest Lyapunov exponents from small data sets", *Physica D*
**65** (1993) 117-134.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["LyapunovFromData", "lyapunov_from_data"]

_TINY = float(np.finfo(float).tiny)


@dataclass(frozen=True)
class LyapunovFromData:
    """Outcome of :func:`lyapunov_from_data`: the divergence curve and its slope.

    Attributes
    ----------
    lyapunov : float
        Estimated maximal Lyapunov exponent (per unit time), the slope of
        ``divergence`` against ``times`` over ``fit_region``.
    times : numpy.ndarray
        Relative times ``k * dt`` for ``k = 0 … k_max``.
    divergence : numpy.ndarray
        The stretching curve ``S(k)`` — mean log divergence after ``k`` samples.
        Inspect ``times`` vs ``divergence`` to choose a scaling region and refine
        the estimate with an explicit ``fit=(lo, hi)``.
    fit_region : tuple[int, int]
        Inclusive index range into ``times``/``divergence`` used for the slope.
    embedding_dim, delay, theiler : int
        Reconstruction parameters actually used.
    n_reference : int
        Number of reference points that contributed (had a usable neighbour).
    method : str
        ``"kantz"`` or ``"rosenstein"``.

    A :class:`LyapunovFromData` casts to ``float`` as its ``lyapunov`` value, so
    ``float(lyapunov_from_data(x))`` returns the exponent directly.
    """

    lyapunov: float
    times: np.ndarray
    divergence: np.ndarray
    fit_region: tuple[int, int]
    embedding_dim: int
    delay: int
    theiler: int
    n_reference: int
    method: str

    def __float__(self) -> float:
        """Return the estimated maximal Lyapunov exponent."""
        return float(self.lyapunov)

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe the divergence curve as a backend-agnostic :class:`PlotSpec`.

        Builds a ``SCALING_FIT`` spec — the stretching curve :math:`S(k)` (mean
        log-divergence) against time as a scatter, the fitted scaling region
        highlighted, and the line of slope :attr:`lyapunov` drawn over it — the
        same schema the fractal-dimension estimators emit, so a single
        ``result.plot.scaling()`` renders it.  The :mod:`tsdynamics.viz.spec`
        import is lazy, so building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"scaling_fit"``).  ``None`` uses
            ``SCALING_FIT``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.SCALING_FIT
        t = np.asarray(self.times, dtype=float)
        s = np.asarray(self.divergence, dtype=float)
        lo, hi = self.fit_region
        layers = [Layer(PlotKind.SCATTER, {"x": t, "y": s}, label="$S(k)$")]
        if t.size and hi >= lo:
            layers.append(
                Layer(
                    PlotKind.MARKERS, {"x": t[lo : hi + 1], "y": s[lo : hi + 1]}, label="fit region"
                )
            )
            # The line of slope `lyapunov` anchored to the fit-region centroid.
            tc = float(np.mean(t[lo : hi + 1]))
            sc = float(np.mean(s[lo : hi + 1]))
            fit_x = np.array([t[lo], t[hi]], dtype=float)
            fit_y = sc + self.lyapunov * (fit_x - tc)
            layers.append(
                Layer(PlotKind.LINE, {"x": fit_x, "y": fit_y}, label=f"slope = {self.lyapunov:.3g}")
            )
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=f"max. Lyapunov ({self.method}) = {self.lyapunov:.3g}",
            x=Axis(label="time"),
            y=Axis(label="mean log divergence $S(k)$"),
            layers=layers,
        )

    def __repr__(self) -> str:  # noqa: D105
        lo, hi = self.fit_region
        return (
            f"LyapunovFromData(lyapunov={self.lyapunov:.4g}, method={self.method!r}, "
            f"m={self.embedding_dim}, tau={self.delay}, fit_region=({lo}, {hi}), "
            f"n_reference={self.n_reference})"
        )


def _delay_embed(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Delay-coordinate embedding of ``series`` into ``m`` blocks spaced by ``tau``.

    Row ``n`` is ``[x[n], x[n+tau], …, x[n+(m-1)tau]]`` (each ``x`` a channel
    vector for multivariate input), so the forward image of row ``n`` after
    ``k`` samples is simply row ``n + k``.
    """
    x = np.asarray(series, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError("series must be 1D (scalar) or 2D (n_samples, n_channels).")
    n, d = x.shape
    span = (m - 1) * tau
    rows = n - span
    if rows <= 0:
        raise ValueError(
            f"series too short: {n} samples cannot fill an m={m}, tau={tau} embedding "
            f"(needs more than {span})."
        )
    emb = np.empty((rows, m * d), dtype=float)
    for j in range(m):
        emb[:, j * d : (j + 1) * d] = x[j * tau : j * tau + rows]
    return emb


def _auto_fit_region(divergence: np.ndarray, *, min_len: int) -> tuple[int, int]:
    """Pick ``[0, knee]`` — the initial rise before the curve saturates.

    The stretching curve rises roughly linearly while neighbours separate
    exponentially, then bends over as they reach the attractor's size.  The knee
    is the first sample whose local slope drops below half the initial slope.
    """
    n = divergence.size
    if n <= min_len:
        return (0, n - 1)
    k = np.arange(n, dtype=float)
    s0 = float(np.polyfit(k[:min_len], divergence[:min_len], 1)[0])
    if not np.isfinite(s0) or s0 <= 0.0:
        return (0, n - 1)
    diffs = np.diff(divergence)
    hi = n - 1
    for i in range(min_len, diffs.size + 1):
        if diffs[i - 1] < 0.5 * s0:
            hi = i - 1
            break
    return (0, max(hi, min_len - 1))


def lyapunov_from_data(
    data: np.ndarray,
    *,
    dt: float = 1.0,
    dimension: int = 3,
    delay: int = 1,
    theiler: int | None = None,
    k_max: int = 20,
    eps: float | None = None,
    n_neighbors: int = 1,
    method: str = "kantz",
    fit: tuple[int, int] | None = None,
) -> LyapunovFromData:
    r"""Estimate the maximal Lyapunov exponent from a time series.

    Reconstructs an ``m``-dimensional delay embedding of ``series``, measures how
    fast nearby trajectories diverge as a function of the look-ahead ``k``, and
    reads the exponent off the slope of the resulting log-divergence curve.

    Parameters
    ----------
    data : array_like
        1-D scalar series, or 2-D ``(n_samples, n_channels)`` for a multivariate
        recording.
    dt : float, default 1.0
        Sampling interval (time between consecutive samples).  Use ``1.0`` for a
        map (the exponent is then per iteration).
    dimension : int, default 3
        Embedding dimension.  Choose it from the data — large enough to unfold
        the attractor (e.g. a false-nearest-neighbour estimate); too small
        underestimates the exponent.
    delay : int, default 1
        Embedding delay, in samples.  For oversampled flows pick it near the
        first minimum of the mutual information / first zero of the
        autocorrelation.
    theiler : int, optional
        Theiler window (Theiler 1986): neighbours with ``|n - j| <= theiler`` are
        rejected so temporally-correlated points are not mistaken for dynamical
        neighbours.  Defaults to ``(dimension - 1) * delay`` (the embedding span).
    k_max : int, default 20
        Number of forward samples over which divergence is tracked; the curve
        spans ``k = 0 … k_max``.
    eps : float, optional
        Neighbour-ball radius for ``method="kantz"``.  Defaults to ``0.1`` times
        the standard deviation of the embedded coordinates.  Ignored by
        ``"rosenstein"`` (which uses the single nearest neighbour).
    n_neighbors : int, default 1
        Minimum neighbours a reference point needs to contribute (Kantz only).
    method : {"kantz", "rosenstein"}, default "kantz"
        Divergence estimator (see module docstring).
    fit : tuple[int, int], optional
        Inclusive sample range ``(lo, hi)`` over which the slope is fit.  Defaults
        to an automatic scaling region (the initial linear rise).  Inspect the
        returned curve and set this explicitly for a reliable estimate.

    Returns
    -------
    LyapunovFromData
        The estimated exponent, the full divergence curve, and the parameters
        used.  Casts to ``float`` as the exponent.

    Notes
    -----
    The estimate is only as good as the embedding and the chosen scaling region.
    Always look at ``result.times`` vs ``result.divergence``: a trustworthy
    estimate comes from a clear straight segment before the curve saturates.

    Examples
    --------
    >>> import tsdynamics as ts
    >>> traj = ts.Henon().trajectory(6000, transient=500, ic=[0.1, 0.1])
    >>> res = ts.lyapunov_from_data(traj.y[:, 0], dimension=4, k_max=12, fit=(0, 6))
    >>> 0.30 < float(res) < 0.55      # ≈ 0.42
    True
    """
    dimension = int(dimension)
    delay = int(delay)
    k_max = int(k_max)
    n_neighbors = int(n_neighbors)
    dt = float(dt)
    method = method.lower()
    if dimension < 1:
        raise ValueError("dimension (embedding dimension) must be >= 1.")
    if delay < 1:
        raise ValueError("delay (embedding delay) must be >= 1.")
    if k_max < 2:
        raise ValueError("k_max must be >= 2 to fit a slope.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if method not in {"kantz", "rosenstein"}:
        raise ValueError(f"method must be 'kantz' or 'rosenstein', got {method!r}.")
    theiler = (dimension - 1) * delay if theiler is None else int(theiler)
    if theiler < 0:
        raise ValueError("theiler must be >= 0.")

    from scipy.spatial import cKDTree

    emb = _delay_embed(data, dimension, delay)
    n_rows = emb.shape[0]
    last = n_rows - 1 - k_max  # references/neighbours need their k-ahead image to exist
    if last < 1:
        raise ValueError(
            "k_max is too large for the embedded series: no forward images remain. "
            "Use a longer series or reduce k_max, dimension, or delay."
        )
    tree = cKDTree(emb)

    if method == "kantz":
        if eps is None:
            eps = 0.1 * float(np.std(emb))
        eps = float(eps)
        if eps <= 0.0:
            raise ValueError("eps must be positive (series may be constant).")
        ref_idx: list[int] = []
        neigh_of: list[np.ndarray] = []
        for n in range(last + 1):
            cand = tree.query_ball_point(emb[n], eps)
            neigh = [j for j in cand if j <= last and abs(j - n) > theiler]
            if len(neigh) >= n_neighbors:
                ref_idx.append(n)
                neigh_of.append(np.asarray(neigh, dtype=int))
        if not ref_idx:
            raise ValueError(
                "no reference point has a neighbour within eps outside the Theiler "
                "window; increase eps, lower dimension, or shorten the Theiler window."
            )
        divergence = np.empty(k_max + 1)
        for k in range(k_max + 1):
            acc = 0.0
            for n, neigh in zip(ref_idx, neigh_of, strict=True):
                d = np.linalg.norm(emb[n + k] - emb[neigh + k], axis=1)
                acc += np.log(max(float(d.mean()), _TINY))
            divergence[k] = acc / len(ref_idx)
        n_reference = len(ref_idx)
    else:  # rosenstein
        # Examine the n_query nearest candidates per point; the cap is a
        # heuristic sized to clear the Theiler window. A reference point whose
        # candidates are *all* rejected (inside the window or past `last`) is
        # skipped rather than matched to a wrong neighbour, lowering n_reference.
        n_query = min(n_rows, 4 * theiler + 20)
        _, idx_all = tree.query(emb, k=n_query)
        idx_all = np.atleast_2d(idx_all)
        refs: list[int] = []
        nn: list[int] = []
        for n in range(last + 1):
            for col in range(1, n_query):  # column 0 is the point itself
                j = int(idx_all[n, col])
                if j <= last and abs(j - n) > theiler:
                    refs.append(n)
                    nn.append(j)
                    break
        if not refs:
            raise ValueError(
                "no nearest neighbour outside the Theiler window was found; "
                "use a longer series or shorten the Theiler window."
            )
        ref_arr = np.asarray(refs, dtype=int)
        nn_arr = np.asarray(nn, dtype=int)
        divergence = np.empty(k_max + 1)
        for k in range(k_max + 1):
            d = np.linalg.norm(emb[ref_arr + k] - emb[nn_arr + k], axis=1)
            divergence[k] = float(np.mean(np.log(np.maximum(d, _TINY))))
        n_reference = ref_arr.size

    times = np.arange(k_max + 1, dtype=float) * dt
    if fit is None:
        lo, hi = _auto_fit_region(divergence, min_len=max(3, (k_max + 1) // 3))
    else:
        lo, hi = int(fit[0]), int(fit[1])
        if not (0 <= lo < hi <= k_max):
            raise ValueError(f"fit region {fit!r} must satisfy 0 <= lo < hi <= k_max ({k_max}).")
    slope = float(np.polyfit(times[lo : hi + 1], divergence[lo : hi + 1], 1)[0])

    return LyapunovFromData(
        lyapunov=slope,
        times=times,
        divergence=divergence,
        fit_region=(lo, hi),
        embedding_dim=dimension,
        delay=delay,
        theiler=theiler,
        n_reference=n_reference,
        method=method,
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
