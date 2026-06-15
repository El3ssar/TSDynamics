r"""
Shared plumbing for the surrogate-data toolkit.

Holds the scalar-series coercion (:func:`as_series`, duck-typed against a
:class:`~tsdynamics.data.Trajectory` to avoid an import cycle), the half-spectrum
phase randomiser the Fourier-family generators share (:func:`phase_randomize`),
and the hypothesis-test arithmetic that turns a data statistic plus a surrogate
ensemble into a rank p-value and a Gaussian significance (:func:`empirical_pvalue`,
:func:`gaussian_significance`).  The generators live in :mod:`.generators`, the
discriminating statistics in :mod:`.statistics`, and the test wrapper in
:mod:`.hypothesis`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["as_series", "empirical_pvalue", "gaussian_significance", "phase_randomize"]


def as_series(x: Any, component: int | str | None = None) -> np.ndarray:
    """Coerce ``x`` into a finite 1-D ``float64`` array (a single scalar series).

    Accepts a 1-D array-like, a 2-D array (a column is selected by ``component``),
    or a :class:`~tsdynamics.data.Trajectory` (a component is selected by index,
    or by name when the system declares ``variables``).  Surrogate methods are
    univariate, so multi-component input must name the column to test.

    Parameters
    ----------
    x : array-like or Trajectory
        The data.  A 1-D sequence is used directly.
    component : int or str, optional
        Which column/component to extract from 2-D input or a trajectory.
        Required when the input has more than one component; for a 1-D input it
        must be left ``None`` (or ``0``).

    Returns
    -------
    numpy.ndarray
        A contiguous 1-D ``float64`` array.

    Raises
    ------
    ValueError
        If the input is not 1-D/2-D, has fewer than three samples, or holds
        non-finite values.
    """
    # Trajectory: defer to its named-component machinery without importing it
    # (duck-typed to avoid a hard dependency cycle through families / data).
    if hasattr(x, "y") and hasattr(x, "component") and not isinstance(x, np.ndarray):
        if component is None:
            if x.y.ndim == 1 or x.y.shape[1] == 1:
                arr = np.asarray(x.y, dtype=float).ravel()
            else:
                raise ValueError(
                    "trajectory has multiple components; pass component= to select one "
                    f"(e.g. component=0 or a name from {getattr(x, 'variables', None)})."
                )
        else:
            arr = np.asarray(x.component(component), dtype=float).ravel()
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2:
            if component is None:
                if arr.shape[1] == 1:
                    arr = arr[:, 0]
                else:
                    raise ValueError(
                        "2-D input has multiple columns; pass component= to select one."
                    )
            elif not isinstance(component, (int, np.integer)):
                raise TypeError("component must be an integer index for a plain 2-D array.")
            else:
                arr = arr[:, int(component)]
        elif arr.ndim == 1:
            if component not in (None, 0):
                raise ValueError("component= is meaningless for a 1-D series.")
        else:
            raise ValueError(f"expected a 1-D or 2-D series, got {arr.ndim}-D input.")

    arr = np.ascontiguousarray(arr, dtype=float)
    if arr.size < 3:
        raise ValueError(f"need at least three samples for a surrogate, got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("series contains non-finite values (nan/inf).")
    return arr


def phase_randomize(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    r"""Return ``n`` phase-randomised (Fourier) surrogates of ``x``.

    Each surrogate keeps the magnitude spectrum :math:`|X_k|` of ``x`` exactly —
    hence the periodogram and the (circular) autocorrelation — while replacing the
    Fourier phases with i.i.d. :math:`U[0, 2\pi)` draws.  The zero-frequency (DC)
    component, and the Nyquist component when ``len(x)`` is even, are kept real so
    the inverse transform is real (Theiler et al., 1992).

    Parameters
    ----------
    x : numpy.ndarray
        The 1-D source series.
    n : int
        How many independent surrogates to draw.
    rng : numpy.random.Generator
        The random source (seed it upstream for reproducibility).

    Returns
    -------
    numpy.ndarray, shape (n, len(x))
        The surrogate ensemble.
    """
    N = x.size
    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum)
    n_freq = magnitude.size
    # Random phases per surrogate; columns 0 (DC) and -1 (Nyquist, N even) stay real.
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n, n_freq))
    phases[:, 0] = 0.0
    if N % 2 == 0:
        phases[:, -1] = 0.0
    surrogate_spectrum = magnitude[None, :] * np.exp(1j * phases)
    return np.fft.irfft(surrogate_spectrum, n=N, axis=1)


def empirical_pvalue(t_data: float, t_surr: np.ndarray, tail: str = "two") -> float:
    r"""Rank-based surrogate p-value of a data statistic against an ensemble.

    With ``M`` surrogate statistics the rank estimator counts how many fall in the
    rejection tail and adds the data point itself to both numerator and the ``M+1``
    denominator, so the smallest attainable one-sided value is :math:`1/(M+1)` and
    the two-sided is :math:`2/(M+1)` — the basis of Theiler et al.'s
    :math:`M = 2/\alpha - 1` rule (e.g. ``M = 39`` for a two-sided ``α = 0.05``).
    Ties count toward the tail, so the test is conservative.

    Parameters
    ----------
    t_data : float
        The statistic evaluated on the data.
    t_surr : numpy.ndarray
        The statistic evaluated on each surrogate.
    tail : {"two", "greater", "less"}, default "two"
        ``"greater"``/``"less"`` reject when the data statistic is large/small
        relative to the surrogates; ``"two"`` rejects either extreme.

    Returns
    -------
    float
        The p-value in ``(0, 1]``.
    """
    t_surr = np.asarray(t_surr, dtype=float)
    M = t_surr.size
    if M == 0:
        raise ValueError("need at least one surrogate statistic.")
    n_ge = int(np.count_nonzero(t_surr >= t_data))
    n_le = int(np.count_nonzero(t_surr <= t_data))
    p_greater = (1 + n_ge) / (M + 1)
    p_less = (1 + n_le) / (M + 1)
    if tail == "greater":
        return p_greater
    if tail == "less":
        return p_less
    if tail == "two":
        return min(1.0, 2.0 * min(p_greater, p_less))
    raise ValueError(f"tail must be 'two', 'greater' or 'less', got {tail!r}.")


def gaussian_significance(t_data: float, t_surr: np.ndarray) -> float:
    r"""Significance of ``t_data`` in standard deviations of the surrogate ensemble.

    The classic surrogate "number of sigmas"
    :math:`S = (T_\text{data} - \langle T\rangle) / \sigma_T` (Theiler et al.,
    1992), a parametric companion to the rank p-value that assumes the surrogate
    statistics are roughly Gaussian.  Returns ``0.0`` when the ensemble has no
    spread (a degenerate statistic).

    Parameters
    ----------
    t_data : float
        The statistic evaluated on the data.
    t_surr : numpy.ndarray
        The statistic evaluated on each surrogate.

    Returns
    -------
    float
        The signed significance (positive when the data lies above the mean).
    """
    t_surr = np.asarray(t_surr, dtype=float)
    if t_surr.size < 2:
        return 0.0
    sigma = float(t_surr.std(ddof=1))
    if sigma == 0.0 or not math.isfinite(sigma):
        return 0.0
    return float((t_data - t_surr.mean()) / sigma)
