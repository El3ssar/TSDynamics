from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TimeStepEstimate:
    """
    Result container for curvature-based time step estimation.

    Attributes
    ----------
    delta_t : float
        Recommended coarse time step Δt (seconds).
    stride : int
        Integer subsampling stride k such that k * Δt0 <= Δt.
    kappa_percentile : float
        Robust curvature level κ used (percentile over κ_i).
    predicted_error : float
        Predicted interpolation error at the chosen stride: 0.5 * κ * (stride*Δt0)^2.
    indices : np.ndarray
        Subsample indices (0-based), including the last index.
    t_mid : Optional[np.ndarray]
        Time grid for the interior points where derivatives/κ_i are computed (shape (T-2,)).
    kappa_series : Optional[np.ndarray]
        Per-sample curvature series κ_i for interior points (shape (T-2,)).
    """
    delta_t: float
    stride: int
    kappa_percentile: float
    predicted_error: float
    indices: np.ndarray
    t_mid: Optional[np.ndarray] = None
    kappa_series: Optional[np.ndarray] = None


def estimate_curvature_timestep(
    y: np.ndarray,
    dt0: float,
    epsilon: float,
    *,
    percentile: float = 95.0,
    nu: float = 1.0,               # safety factor in (0,1]
    return_details: bool = True,
) -> TimeStepEstimate:
    """
    Compute the curvature-based *continuous* Δt bound for integration, and a suggested
    integer stride for subsampling the *current* data. `delta_t` in the result is the
    continuous Δt_hat; `stride` is optional for decimating the given series.

    Implements:
        κ_i = || a_i - proj_{v_i}(a_i) ||  (central differences),
        Δt_hat = ν * sqrt(2 * ε / κ_p),     κ_p = p-th percentile of {κ_i}.
    """
    # ----------- validation -----------
    if not isinstance(y, np.ndarray) or y.ndim != 2:
        raise ValueError(f"y must be 2D (T, n_dim); got {type(y)} with shape {getattr(y, 'shape', None)}.")
    T, d = y.shape
    if T < 3:
        raise ValueError(f"Need at least T >= 3 to form central differences; got T={T}.")
    if not np.isfinite(dt0) or dt0 <= 0:
        raise ValueError("dt0 must be a finite positive float.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive float.")
    if not (0.0 < percentile < 100.0):
        raise ValueError("percentile must be in (0, 100).")
    if not (0.0 < nu <= 1.0):
        raise ValueError("nu must be in (0, 1].")

    y = np.asarray(y, dtype=np.float64)

    # ----------- per-feature std normalization (no centering) -----------
    with np.errstate(invalid="ignore"):
        std = y.std(axis=0, ddof=1)
    scale = std.copy()
    scale[~np.isfinite(scale) | (scale == 0.0)] = 1.0
    y_scaled = y / scale

    # ----------- central differences on interior points -----------
    y_next = y_scaled[2:]
    y_mid  = y_scaled[1:-1]
    y_prev = y_scaled[:-2]

    v = (y_next - y_prev) / (2.0 * dt0)           # velocity
    a = (y_next - 2.0 * y_mid + y_prev) / (dt0**2)  # acceleration

    v2 = np.einsum("ij,ij->i", v, v)
    av = np.einsum("ij,ij->i", a, v)

    v2_med = np.nanmedian(v2) if np.isfinite(v2).any() else 0.0
    tiny = 1e-12 * max(1.0, v2_med)
    moving_mask = v2 > tiny

    kappa = np.empty_like(v2)
    if moving_mask.any():
        alpha_proj = np.zeros_like(av)
        alpha_proj[moving_mask] = av[moving_mask] / v2[moving_mask]
        a_par = (alpha_proj[:, None]) * v
        a_perp = a - a_par
        kappa[moving_mask] = np.linalg.norm(a_perp[moving_mask], axis=1)
    if (~moving_mask).any():
        kappa[~moving_mask] = np.linalg.norm(a[~moving_mask], axis=1)

    # ----------- robust curvature and Δt bound -----------
    kappa_p = float(np.nanpercentile(kappa, percentile))
    if not np.isfinite(kappa_p) or kappa_p < 0.0:
        raise RuntimeError("Computed invalid κ percentile; check input data.")

    if kappa_p == 0.0:
        # Straight line (in normalized space): no curvature → no geometric sagitta.
        delta_t_hat = np.inf
        stride = max(1, T - 1)
        predicted_error = 0.0
    else:
        delta_t_hat = nu * np.sqrt(2.0 * epsilon / kappa_p)   # <-- CONTINUOUS bound for *integration*
        # Optional: stride only for subsampling current data
        stride = max(1, int(np.floor(delta_t_hat / dt0)))
        predicted_error = 0.5 * kappa_p * (delta_t_hat ** 2)  # = ν^2 * ε

    # ----------- subsample indices for the current series -----------
    idx = np.arange(0, T, stride, dtype=int)
    if idx[-1] != (T - 1):
        idx = np.concatenate([idx, np.array([T - 1], dtype=int)])

    t_mid = kappa_series = None
    if return_details:
        t_mid = dt0 * np.arange(1, T - 1, dtype=np.float64)
        kappa_series = kappa

    return TimeStepEstimate(
        delta_t=float(delta_t_hat),         # <-- integration step to USE going forward
        stride=int(stride),                 # optional decimation of the existing data
        kappa_percentile=float(kappa_p),
        predicted_error=float(predicted_error),  # ~ ν^2 * ε at the bound
        indices=idx,
        t_mid=t_mid,
        kappa_series=kappa_series,
    )





__all__ = ["TimeStepEstimate", "estimate_curvature_timestep"]
