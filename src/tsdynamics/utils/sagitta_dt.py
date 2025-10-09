from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

__all__ = ["SagittaDt", "estimate_dt_from_sagitta"]

@dataclass(frozen=True)
class SagittaDt:
    delta_t: float          # Δt* = m*dt0
    stride: int             # m*
    percentile_value: float # percentile_p(s_i(m*)) actually achieved
    indices: np.ndarray     # decimation indices for current data (include last)
    p: float                # percentile used
    epsilon: float          # tolerance used (in normalized units)
    searched_ms: np.ndarray # candidate m's that were evaluated
    notes: str = ""


def _sagitta_percentile(y: np.ndarray, m: int, p: float) -> float:
    """
    Compute p-th percentile of sagitta distances using triples (i-m, i, i+m).
    Works for y shape (T, d). Assumes T >= 2m+1.
    """
    A = y[:-2*m, :]        # y_{i-m}
    B = y[m:-m, :]         # y_i
    C = y[2*m:, :]         # y_{i+m}

    AC = C - A
    L = np.linalg.norm(AC, axis=1)
    # Avoid division by zero: where L==0, sagitta is just ||B-A|| (degenerate chord)
    ok = L > 0
    u = np.zeros_like(AC)
    u[ok] = (AC[ok].T / L[ok]).T

    BA = B - A
    # projection length onto AC
    proj_len = np.einsum("ij,ij->i", BA, u)
    proj = (proj_len[:, None]) * u
    perp = BA - proj
    s = np.linalg.norm(perp, axis=1)

    # robust percentile
    return float(np.nanpercentile(s, p))


def estimate_dt_from_sagitta(
    y: np.ndarray,
    dt0: float,
    *,
    epsilon: float,
    percentile: float = 95.0,
    coarsen_only: bool = True,
    min_points_per_segment: int = 3,
    search_growth: float = 1.5,
) -> SagittaDt:
    """
    Choose Δt* = m*dt0 by bounding the percentile of sagitta (orthogonal geometric deviation)
    below a tolerance ε on σ-normalized features.

    Parameters
    ----------
    y : np.ndarray, shape (T, d)
        Time-ordered samples.
    dt0 : float
        Current sampling step.
    epsilon : float
        Geometric tolerance (same units as σ-normalized y).
    percentile : float
        Percentile p for robustness (e.g., 95.0).
    coarsen_only : bool
        If True, never suggest Δt* < dt0 (i.e., m >= 1).
    min_points_per_segment : int
        Require at least this many interior triples to evaluate a candidate m.
    search_growth : float
        Multiplicative step to grow m in the coarse search (e.g., 1.5 or 2.0).

    Returns
    -------
    SagittaDt
    """
    if not isinstance(y, np.ndarray) or y.ndim != 2:
        raise ValueError("y must be a (T, d) array.")
    T, d = y.shape
    if T < 5:
        raise ValueError("Need T >= 5.")
    if not (dt0 > 0):
        raise ValueError("dt0 must be > 0.")
    if not (epsilon > 0):
        raise ValueError("epsilon must be > 0.")
    if not (0.0 < percentile <= 100.0):
        raise ValueError("percentile must be in (0,100).")

    # per-feature std normalization (scale-invariance)
    std = y.std(axis=0, ddof=1)
    std[~np.isfinite(std) | (std == 0.0)] = 1.0
    y_norm = y / std

    # maximum m such that we have enough triples
    m_max = (T - 1) // 2
    # also ensure enough statistics:
    while m_max > 1 and (T - 2*m_max) < min_points_per_segment:
        m_max -= 1

    if m_max < 1:
        # cannot coarsen; return dt0
        return SagittaDt(
            delta_t=dt0,
            stride=1,
            percentile_value=0.0,
            indices=np.arange(T, dtype=int),
            p=percentile,
            epsilon=epsilon,
            searched_ms=np.array([1], dtype=int),
            notes="Too few samples to evaluate coarsening; returning dt0."
        )

    # -------- coarse exponential search to bracket m* --------
    ms_eval = []
    def ok_m(m: int) -> Tuple[bool, float]:
        if (T - 2*m) < min_points_per_segment:
            return False, np.nan
        val = _sagitta_percentile(y_norm, m, percentile)
        return (val <= epsilon), val

    m = 1
    best_m = 1
    best_val = _sagitta_percentile(y_norm, 1, percentile)
    ms_eval.append((1, best_val))
    # If even m=1 violates epsilon and coarsen_only, clamp at 1.
    if (best_val > epsilon) and coarsen_only:
        indices = np.arange(T, dtype=int)
        return SagittaDt(
            delta_t=dt0,
            stride=1,
            percentile_value=best_val,
            indices=indices,
            p=percentile,
            epsilon=epsilon,
            searched_ms=np.array([1], dtype=int),
            notes="m=1 already exceeds ε; coarsen_only=True so Δt*=dt0."
        )

    # grow m until it fails or we hit m_max
    m_list = [1]
    val_list = [best_val]
    while True:
        m_next = int(np.floor(m * search_growth))
        if m_next <= m:
            m_next = m + 1
        if m_next > m_max:
            break
        ok, val = ok_m(m_next)
        ms_eval.append((m_next, val))
        m_list.append(m_next)
        val_list.append(val)
        if ok:
            best_m, best_val = m_next, val
            m = m_next
        else:
            # bracket found: (best_m ok) < (m_next fail)
            break

    # -------- binary search between last ok and first fail (if any) --------
    lo = best_m
    hi = m_next if (len(m_list) > 1 and val_list[-1] > epsilon) else m_max + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, val = ok_m(mid)
        ms_eval.append((mid, val))
        if ok:
            lo, best_m, best_val = mid, mid, val
        else:
            hi = mid

    # build indices for decimation at best_m
    stride = int(best_m)
    idx = np.arange(0, T, stride, dtype=int)
    if idx[-1] != (T - 1):
        idx = np.concatenate([idx, np.array([T - 1], dtype=int)])

    ms_eval_sorted = np.array(sorted(ms_eval, key=lambda x: x[0]), dtype=float)
    searched_ms = ms_eval_sorted[:, 0].astype(int)

    return SagittaDt(
        delta_t=stride * dt0,
        stride=stride,
        percentile_value=float(best_val),
        indices=idx,
        p=percentile,
        epsilon=epsilon,
        searched_ms=searched_ms,
        notes=""
    )
