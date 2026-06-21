r"""
Discriminating statistics for surrogate nonlinearity tests (stream **A-SURR**).

A surrogate test is only as good as the statistic that separates the data from
its linear surrogates.  Two complementary, literature-standard discriminators
live here, both blind to the linear properties the Fourier-family surrogates
preserve, so a non-zero contrast is evidence of nonlinear (or non-Gaussian,
time-irreversible) structure:

- :func:`time_reversal_asymmetry` — the normalised third moment of the series
  increments (Schreiber & Schmitz, *Phys. Rev. Lett.* **77**, 635, 1996;
  Diks et al., *Phys. Lett. A* **201**, 221, 1995).  Linear Gaussian processes,
  and static nonlinear re-scalings of them, are time-reversible, so this vanishes
  on FT/AAFT/IAAFT surrogates while a dissipative flow such as Lorenz is strongly
  irreversible.  A cheap, robust default.
- :func:`nonlinear_prediction_error` — the out-of-sample error of a
  locally-constant phase-space predictor (Sugihara & May, *Nature* **344**, 734,
  1990; Kantz & Schreiber, *Nonlinear Time Series Analysis*, 2004).  Determinism
  makes the data more predictable than its linear surrogates.

Both reduce a 1-D series to a single float; pass either (by value or by the names
``"time_reversal"`` / ``"prediction_error"``) to
:func:`~tsdynamics.analysis.surrogate.hypothesis.surrogate_test`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["nonlinear_prediction_error", "time_reversal_asymmetry"]


def time_reversal_asymmetry(data: np.ndarray, delay: int = 1) -> float:
    r"""Time-reversal asymmetry statistic of a scalar series.

    Computes the dimensionless third-moment ratio of the lagged increments,

    .. math::

        T_\text{rev} = \frac{\langle (x_{t} - x_{t-\ell})^3 \rangle}
                            {\langle (x_{t} - x_{t-\ell})^2 \rangle^{3/2}},

    which changes sign under time reversal and is therefore identically zero (in
    expectation) for any time-reversible process — including a linear Gaussian
    process and any static monotonic transform of one, the nulls the Fourier-family
    surrogates realise.  A dissipative deterministic flow breaks that symmetry, so
    a large :math:`|T_\text{rev}|` relative to the surrogates flags nonlinearity.

    Parameters
    ----------
    data : numpy.ndarray
        The 1-D series.
    delay : int, default 1
        The increment lag :math:`\ell` (in samples), ``>= 1``.

    Returns
    -------
    float
        The asymmetry ratio (``0.0`` for a constant series).
    """
    data = np.asarray(data, dtype=float)
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}.")
    if data.size <= delay:
        raise ValueError(f"series too short: need > {delay} samples, got {data.size}.")
    increments = data[delay:] - data[:-delay]
    second = float(np.mean(increments**2))
    if second == 0.0:
        return 0.0
    third = float(np.mean(increments**3))
    return float(third / second**1.5)


def nonlinear_prediction_error(
    data: np.ndarray,
    dimension: int = 3,
    delay: int = 1,
    *,
    horizon: int = 1,
    n_neighbors: int = 4,
    theiler: int = 1,
) -> float:
    r"""Out-of-sample error of a locally-constant phase-space predictor.

    Reconstructs the series in ``dimension`` delay coordinates (delay ``delay``)
    and, for each point, predicts the value ``horizon`` steps ahead as the mean of
    the futures of its ``n_neighbors`` nearest phase-space neighbours, excluding
    temporal neighbours within a Theiler window.  The returned error is the
    root-mean-square prediction residual normalised by the standard deviation of the
    series, so a perfectly unpredictable series scores :math:`\approx 1` and a
    deterministic one scores well below it (Sugihara & May, 1990; Kantz &
    Schreiber, 2004).

    Because determinism is exactly what the linear surrogates lack, a *small* error
    relative to the surrogate ensemble is evidence of nonlinear determinism (use a
    one-sided ``tail="less"`` test).

    Parameters
    ----------
    data : numpy.ndarray
        The 1-D series.
    dimension : int, default 3
        Embedding dimension, ``>= 1``.
    delay : int, default 1
        Embedding delay in samples, ``>= 1``.
    horizon : int, default 1
        Prediction horizon in samples, ``>= 1``.
    n_neighbors : int, default 4
        Number of nearest neighbours averaged per prediction.
    theiler : int, default 1
        Half-width of the temporal-exclusion (Theiler) window; neighbours with
        ``|i - j| <= theiler`` are skipped so the predictor cannot cheat with
        trivially-adjacent points.

    Returns
    -------
    float
        The normalised RMS prediction error.

    Raises
    ------
    ValueError
        If the parameters are out of range or the series is too short to embed and
        find the requested neighbours.
    """
    data = np.asarray(data, dtype=float)
    if dimension < 1 or delay < 1 or horizon < 1:
        raise ValueError("dimension, delay and horizon must all be >= 1.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")
    if theiler < 0:
        raise ValueError("theiler must be >= 0.")
    span = (dimension - 1) * delay
    n_points = data.size - span - horizon
    if n_points <= n_neighbors + 2 * theiler + 1:
        raise ValueError(
            f"series too short: need more than {n_neighbors + 2 * theiler + 1 + span + horizon} "
            f"samples for dimension={dimension}, delay={delay}, horizon={horizon}, "
            f"got {data.size}."
        )

    # Delay-coordinate vectors that all have a valid `horizon`-ahead target.
    base = np.arange(n_points)
    embed = data[base[:, None] + np.arange(dimension)[None, :] * delay]
    targets = data[base + span + horizon]

    std = float(data.std())
    if std == 0.0:
        return 0.0

    tree = cKDTree(embed)
    # Over-query so the Theiler band can be filtered and still leave n_neighbors.
    k_query = min(n_points, n_neighbors + 2 * theiler + 1)
    _, idx = tree.query(embed, k=k_query)

    predictions = np.empty(n_points, dtype=float)
    valid = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        neighbours = [j for j in idx[i] if abs(int(j) - i) > theiler]
        if len(neighbours) < n_neighbors:
            continue
        chosen = neighbours[:n_neighbors]
        predictions[i] = targets[chosen].mean()
        valid[i] = True

    if not np.any(valid):
        raise ValueError(
            "no point retained enough non-Theiler neighbours; relax theiler/dimension/delay."
        )
    residual = predictions[valid] - targets[valid]
    return float(np.sqrt(np.mean(residual**2)) / std)


#: Statistic name → callable, resolved by ``surrogate_test``.
STATISTICS: dict[str, Callable[..., float]] = {
    "time_reversal": time_reversal_asymmetry,
    "prediction_error": nonlinear_prediction_error,
}


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
