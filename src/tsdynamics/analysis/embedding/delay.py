r"""
Choosing the embedding delay :math:`\tau`.

The delay trades redundancy against irrelevance: too small and successive
coordinates are nearly identical (the reconstruction collapses onto the
diagonal); too large and they become causally unrelated.  Two standard criteria
are provided:

- **Autocorrelation** — the linear measure.  A common rule takes :math:`\tau` as
  the first lag at which the autocorrelation drops to :math:`1/e`, or its first
  zero crossing.
- **Time-delayed mutual information** (Fraser & Swinney, 1986) — the nonlinear
  measure, and the more widely recommended one.  The first local *minimum* of
  :math:`I(\tau)` marks the delay at which :math:`x_{i+\tau}` adds the most new
  information about the state while still being dynamically related to
  :math:`x_i`.

:func:`optimal_delay` returns a single recommended :math:`\tau`;
:func:`mutual_information` and :func:`autocorrelation` expose the underlying
curves for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .._common import reject_system
from .._result import ArrayResult, CountResult
from ._common import _as_series

__all__ = ["MutualInformation", "autocorrelation", "mutual_information", "optimal_delay"]


@dataclass(frozen=True, eq=False)
class MutualInformation(ArrayResult):
    r"""A time-delayed mutual-information curve :math:`I(\tau)` with its first minimum.

    An :class:`~tsdynamics.analysis._result.ArrayResult`, so it is a drop-in for
    the bare ``(max_delay + 1,)`` curve array (``np.asarray(result)``, indexing
    and iteration defer to it) while it also carries ``.meta`` / the readout ``repr`` /
    the ``.plot`` seam.  The curve's **first local minimum** is the recommended
    embedding delay (Fraser & Swinney 1986); :attr:`optimal_lag` reads it off and
    :meth:`__plot_spec__` annotates it, so the delay-selection diagnostic plots as
    one figure.

    Attributes
    ----------
    values : numpy.ndarray
        The mutual information :math:`I(\tau)` at lags :math:`\tau = 0, 1, \dots`.
        ``np.asarray(result)`` returns it.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("optimal_lag",)

    def _answer(self) -> str:
        r"""Return the recommended delay — the number the curve is computed for."""
        return f"τ = {self.optimal_lag} samples"

    def _context(self) -> str | None:
        """Return the lag range the curve spans."""
        n = int(np.asarray(self.values).size)
        return f"I(τ) over τ = 0..{max(n - 1, 0)}"

    def _derived(self) -> dict[str, Any]:
        """Export the recommended delay the repr reports."""
        return {"optimal_lag": self.optimal_lag}

    @property
    def optimal_lag(self) -> int:
        r"""The recommended delay — the first *significant* local minimum of :math:`I(\tau)`.

        The first interior lag that is strictly below its left neighbour, no
        greater than its right (the Fraser--Swinney rule, taken at the *onset* of
        the dip so a flat-bottomed valley is not over-estimated), **and** that
        still carries information — see :func:`_select_delay` for the noise floor
        and for what happens when no such lag exists.

        The floor needs the estimator's bin count and sample size; both are
        recorded in ``meta`` by :func:`mutual_information`.  A curve built by hand
        (no such ``meta``) is treated as noise-free, which is the pre-v6
        behaviour.
        """
        curve = np.asarray(self.values, dtype=float)
        bins = self.meta.get("bins")
        n = self.meta.get("n_samples")
        floor = 0.0 if bins is None or n is None else _mi_noise_floor(int(bins), int(n))
        return _select_delay(curve, floor=floor)

    def __plot_spec__(self, kind: str | None = None) -> Any:
        r"""Describe the mutual-information diagnostic as a :class:`PlotSpec`.

        Builds a ``DIAGNOSTIC_CURVE``: the curve :math:`I(\tau)` as a ``LINE``
        against the lag :math:`\tau`, with the chosen delay :attr:`optimal_lag`
        marked by a vertical reference line (the first-minimum criterion).  The
        :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls a
        plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"diagnostic_curve"``).  ``None``
            uses ``DIAGNOSTIC_CURVE``.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        curve = np.asarray(self.values, dtype=float)
        lags = np.arange(curve.size, dtype=float)
        lag = self.optimal_lag if curve.size else 0
        return pb.spec(
            kind,
            "diagnostic_curve",
            layers=[pb.line(lags, curve, label=r"$I(\tau)$")],
            xlabel=r"delay $\tau$",
            ylabel=r"$I(\tau)$",
            title=f"mutual information (first min at $\\tau$ = {lag})"
            if curve.size
            else "mutual information",
            annotations=[pb.vline(float(lag), text=rf"$\tau$ = {lag}")] if curve.size else [],
            meta=self.meta,
        )


def autocorrelation(
    data: Any, *, max_delay: int = 50, components: int | str | None = None
) -> np.ndarray:
    r"""Normalised autocorrelation function up to ``max_delay``.

    Parameters
    ----------
    data : array-like or Trajectory
        The scalar series (or a selected ``components``).
    max_delay : int, default 50
        Largest lag returned.  Clamped to ``N - 1``.
    components : int or str, optional
        Component selector for a multi-component input.

    Returns
    -------
    ndarray, shape (max_delay + 1,)
        ``acf[k]`` is the autocorrelation at lag ``k`` (``acf[0] == 1``).

    Raises
    ------
    ValueError
        If ``max_delay`` is negative, or the series is constant (zero variance,
        so the autocorrelation is undefined).

    Notes
    -----
    Computed via FFT (Wiener--Khinchin) on the mean-subtracted series.  This is
    the standard **biased** autocorrelation estimator — each lag is normalised by
    the full zero-lag variance (not by the shrinking overlap count at that lag) —
    which is positive-definite and the conventional choice for delay selection.
    """
    x = _as_series(data, component=components, analysis="autocorrelation")
    n = x.size
    max_delay = int(max_delay)
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative.")
    max_delay = min(max_delay, n - 1)

    x = x - x.mean()
    var = float(x @ x)
    if var == 0.0:
        raise ValueError("series is constant; autocorrelation is undefined.")

    # Linear (non-circular) autocorrelation via zero-padded FFT.
    size = int(2 ** np.ceil(np.log2(2 * n - 1)))
    f = np.fft.rfft(x, size)
    acf_full = np.fft.irfft(f * np.conj(f), size)[: max_delay + 1]
    return acf_full / var


def _auto_bins(n: int) -> int:
    """Default histogram bin count for the mutual-information estimate."""
    return int(np.clip(np.sqrt(n / 5.0), 16, 128))


def mutual_information(
    data: Any,
    *,
    max_delay: int = 50,
    bins: int | None = None,
    base: float = np.e,
    components: int | str | None = None,
) -> MutualInformation:
    r"""Time-delayed mutual information :math:`I(\tau)` up to ``max_delay``.

    The histogram estimator of

    .. math::

        I(\tau) = \sum_{a,b} p_{ab}(\tau)\,
                  \log\frac{p_{ab}(\tau)}{p_a\, p_b},

    where :math:`p_{ab}` is the joint distribution of :math:`(x_i, x_{i+\tau})`
    over an equal-width 2-D histogram and :math:`p_a, p_b` its marginals.

    Parameters
    ----------
    data : array-like or Trajectory
        The scalar series (or a selected ``components``).
    max_delay : int, default 50
        Largest lag returned.  Clamped to ``N - 2``.
    bins : int, optional
        Number of histogram bins per axis.  Default: a sample-size-dependent
        rule, ``clip(sqrt(N/5), 16, 128)``.
    base : float, default ``e``
        Logarithm base — ``e`` for nats, ``2`` for bits.  Only rescales the
        curve; the location of the first minimum is unaffected.
    components : int or str, optional
        Component selector for a multi-component input.

    Returns
    -------
    MutualInformation
        Behaves as an ``(max_delay + 1,)`` ``ndarray``: ``mi[k]`` is
        :math:`I(k)`; ``mi[0]`` is the entropy of the (binned) series itself
        (its self-information).  ``result.optimal_lag`` reads off the
        first-minimum delay and ``result.plot()`` renders the diagnostic.

    Raises
    ------
    ValueError
        If ``max_delay`` is negative, ``bins`` is less than ``2``, or the series
        is constant (its range collapses, leaving the histogram undefined).

    References
    ----------
    A. M. Fraser and H. L. Swinney, "Independent coordinates for strange
    attractors from mutual information", *Phys. Rev. A* **33**, 1134 (1986).

    Examples
    --------
    >>> import numpy as np
    >>> from tsdynamics.analysis.embedding import mutual_information
    >>> t = np.linspace(0.0, 100.0, 2000)
    >>> x = np.sin(2.0 * np.pi * 0.1 * t)
    >>> mi = mutual_information(x, max_delay=40)
    >>> int(mi.optimal_lag) >= 1
    True
    """
    x = _as_series(data, component=components, analysis="mutual_information")
    n = x.size
    max_delay = int(max_delay)
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative.")
    max_delay = min(max_delay, n - 2)
    nbins = int(bins) if bins is not None else _auto_bins(n)
    if nbins < 2:
        raise ValueError("bins must be >= 2.")
    log = np.log if base == np.e else (lambda v: np.log(v) / np.log(base))

    # A shared, fixed bin grid over the series range keeps marginals consistent
    # across lags (Fraser--Swinney use one partition of the data range).
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        raise ValueError("series is constant; mutual information is undefined.")
    edges = np.linspace(lo, hi, nbins + 1)
    # Pre-bin every sample once; the lagged pair (a, b) just indexes shifted views.
    codes = np.clip(np.digitize(x, edges[1:-1]), 0, nbins - 1)

    mi = np.empty(max_delay + 1, dtype=float)
    for tau in range(max_delay + 1):
        a = codes[: n - tau]
        b = codes[tau:] if tau > 0 else codes
        # Flatten the (a, b) integer bin pair to a single linear index and count
        # with bincount — the joint 2-D histogram, no per-element scatter.  a and
        # b are already integer bin codes in [0, nbins), so a*nbins + b is the
        # row-major flat index; this is bit-equivalent to np.add.at but vectorised.
        flat = np.bincount(a * nbins + b, minlength=nbins * nbins)
        joint = flat.reshape(nbins, nbins).astype(float)
        total = joint.sum()
        joint /= total
        p_a = joint.sum(axis=1)
        p_b = joint.sum(axis=0)
        mask = joint > 0.0
        outer = p_a[:, None] * p_b[None, :]
        mi[tau] = float(np.sum(joint[mask] * log(joint[mask] / outer[mask])))
    return MutualInformation(
        values=mi,
        meta={
            "analysis": "mutual_information",
            "max_delay": max_delay,
            # Recorded so ``optimal_lag`` can rebuild the estimator's independence
            # floor (``_mi_noise_floor``) without re-deriving the binning rule.
            "bins": nbins,
            "n_samples": n,
        },
    )


def _first_local_min(curve: np.ndarray) -> int | None:
    """Index of the first interior local minimum (the *onset* of the valley).

    Fraser & Swinney's first local minimum marks the onset of the dip.  The
    predicate is strict on the descending side and non-strict on the rising side
    (``curve[k] < curve[k-1] and curve[k] <= curve[k+1]``), so a flat-bottomed
    valley returns the *first* lag of its plateau (the onset) rather than its far
    edge.  On strictly shaped curves this is identical to the strict-on-both-sides
    rule; on a quantised / flat curve it no longer over-estimates the delay.

    Shape only — no significance test.  :func:`_select_delay` is the caller that
    adds the noise floor.
    """
    for k in range(1, curve.size - 1):
        if curve[k] < curve[k - 1] and curve[k] <= curve[k + 1]:
            return k
    return None


#: How many times the independence bias a minimum's mutual information must
#: exceed to count as a real dip rather than a fluctuation of the noise floor.
#: Measured margins on the reference series (20k-point Hénon, 8k-point Lorenz at
#: ``dt = 0.02``): the *spurious* Hénon minimum sits at 1.09 floors and the
#: *genuine* Lorenz one at 9.97, so ``2`` separates them with an order of
#: magnitude of headroom on both sides.
_MI_FLOOR_FACTOR = 2.0


def _mi_noise_floor(bins: int, n: int) -> float:
    r"""Mutual information the histogram estimator reports for *independent* data.

    The plug-in (maximum-likelihood) estimate of :math:`I` on a ``bins x bins``
    contingency table is biased upward: for genuinely independent variables
    :math:`2 N \hat{I}` is asymptotically :math:`\chi^2` with
    :math:`(\text{bins}-1)^2` degrees of freedom, so
    :math:`\langle\hat I\rangle \approx (\text{bins}-1)^2 / 2N` — the Miller--Madow
    bias.  Below a few times this value the curve is flat noise and *any* wiggle
    in it is an artefact of the binning, not a feature of the dynamics.

    References
    ----------
    G. A. Miller, "Note on the bias of information estimates", in *Information
    Theory in Psychology* (1955), pp. 95-100.
    """
    return (bins - 1.0) ** 2 / (2.0 * max(n, 1))


def _select_delay(curve: np.ndarray, *, floor: float) -> int:
    r"""Fraser--Swinney delay from a mutual-information curve, with a noise guard.

    A local minimum of :math:`I(\tau)` is only meaningful while :math:`I` is still
    above the level the estimator would report for *independent* variables
    (:func:`_mi_noise_floor`).  Past that point the curve is a flat noise floor
    whose wiggles are binning artefacts, and taking the first of them is how a
    **map** used to get a wildly wrong delay: on a 20 000-point Hénon orbit
    :math:`I(\tau)` decays monotonically to the floor by :math:`\tau \approx 20`,
    a fluctuation at :math:`\tau = 23` was read as "the" first minimum, and
    ``embedding_dimension`` then returned 8 (its ``max_dim``) instead of 2.

    When no significant minimum exists the fallback distinguishes the two ways
    that can happen, because they want opposite answers:

    * **The curve has reached the floor** (a map, or any fully decorrelated
      series).  Every lag beyond the decay is equally uninformative, so the
      smallest lag is the most informative one: :math:`\tau = 1`.  For Hénon that
      restores ``embedding_dimension`` :math:`= 2`.
    * **The curve is still well above the floor at the largest lag probed**
      (a monotone decay truncated by ``max_delay`` — an oversampled flow).  The
      series has *not* decorrelated within the window, so the longest available
      lag is the best available answer: :math:`\tau = \text{max\_delay}`.

    Parameters
    ----------
    curve : ndarray
        :math:`I(\tau)` at lags :math:`\tau = 0, 1, \dots`.
    floor : float
        The independence bias of the estimator.  ``0.0`` disables the guard
        (every local minimum is accepted), which is the pre-v6 behaviour.

    Returns
    -------
    int
        The recommended delay, always ``>= 1``.
    """
    if curve.size < 2:
        return 1
    significant = curve > _MI_FLOOR_FACTOR * floor
    for k in range(1, curve.size - 1):
        if significant[k] and curve[k] < curve[k - 1] and curve[k] <= curve[k + 1]:
            return k
    # No significant dip: informative-but-truncated -> longest lag; decayed -> 1.
    return int(curve.size - 1) if significant[-1] else 1


def optimal_delay(
    data: Any,
    *,
    method: str = "mi",
    max_delay: int = 50,
    bins: int | None = None,
    components: int | str | None = None,
) -> CountResult:
    r"""Recommend an embedding delay :math:`\tau` (in samples).

    Parameters
    ----------
    data : array-like or Trajectory
        The scalar series (or a selected ``components``).
    method : {"mi", "acf", "acf_zero"}, default "mi"
        - ``"mi"`` — first local minimum of the time-delayed mutual information
          (Fraser & Swinney); the recommended nonlinear criterion.
        - ``"acf"`` — first lag where the autocorrelation falls to ``1/e``.
        - ``"acf_zero"`` — first lag where the autocorrelation crosses zero.
    max_delay : int, default 50
        Largest lag considered.
    bins : int, optional
        Histogram bins for the mutual-information estimate (``method="mi"``).
    components : int or str, optional
        Component selector for a multi-component input.

    Returns
    -------
    CountResult
        The recommended delay (behaves as an ``int``), always ``>= 1``.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``"mi"`` / ``"acf"`` / ``"acf_zero"``, or if
        the underlying curve estimator rejects the series (constant input, bad
        ``max_delay`` / ``bins``).

    Notes
    -----
    If no first minimum / crossing is found within ``max_delay`` (e.g. a
    slowly-decaying curve), the criterion's global fallback is used — the
    location of the smallest mutual information, or ``max_delay`` for the
    autocorrelation rules — so a usable delay is always returned.

    The mutual-information criterion is the more widely recommended nonlinear
    measure (Fraser & Swinney 1986); the autocorrelation rules are the classic
    linear alternatives.

    The first local minimum is taken at the **onset** of the dip: on a
    flat-bottomed (quantised) mutual-information valley the recommended
    :math:`\tau` is the first lag of the plateau, not its far edge — so the
    delay is not over-estimated on coarsely sampled curves.

    References
    ----------
    A. M. Fraser and H. L. Swinney, "Independent coordinates for strange
    attractors from mutual information", *Phys. Rev. A* **33**, 1134 (1986).

    Examples
    --------
    >>> import numpy as np
    >>> from tsdynamics.analysis.embedding import optimal_delay
    >>> t = np.linspace(0.0, 100.0, 4000)
    >>> x = np.sin(2.0 * np.pi * 0.05 * t)
    >>> int(optimal_delay(x, method="mi", max_delay=60)) >= 1
    True
    """
    reject_system(data, analysis="optimal_delay")
    method = method.lower()
    if method == "mi":
        curve = mutual_information(data, max_delay=max_delay, bins=bins, components=components)
        tau = curve.optimal_lag
    elif method in ("acf", "acf_zero"):
        acf = autocorrelation(data, max_delay=max_delay, components=components)
        if method == "acf":
            below = np.flatnonzero(acf[1:] <= 1.0 / np.e)
        else:
            below = np.flatnonzero(acf[1:] <= 0.0)
        # never crosses → longest lag available
        tau = int(below[0]) + 1 if below.size else max(int(acf.size) - 1, 1)
    else:
        raise ValueError(f"unknown method {method!r}; use 'mi', 'acf', or 'acf_zero'.")

    return CountResult(value=int(tau), meta={"analysis": "optimal_delay", "method": method})


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
