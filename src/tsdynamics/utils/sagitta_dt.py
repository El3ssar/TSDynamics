from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

__all__ = ["SagittaDt", "estimate_dt_from_sagitta"]

@dataclass(frozen=True)
class SagittaDt:
    delta_t: float                 # Δt* = stride * base_dt
    stride: int                    # stride m*
    percentile_value: float        # achieved S_p(m*) at the chosen stride
    indices: np.ndarray            # decimation indices for current data (include last)
    p: float                       # percentile used (kept for backward-compat)
    epsilon: float                 # tolerance used (kept for backward-compat)
    searched_ms: np.ndarray        # candidate strides evaluated
    notes: str = ""                # info / warnings


def _compute_sagitta_percentile(samples: np.ndarray, span: int, percentile_p: float) -> float:
    """
    Compute p-th percentile of sagitta over triples (i-span, i, i+span) on 'samples' (shape: (n_samples, n_features)).
    """
    A = samples[:-2 * span, :]             # y_{i-span}
    B = samples[span:-span, :]             # y_i
    C = samples[2 * span:, :]              # y_{i+span}

    AC = C - A
    chord_length = np.linalg.norm(AC, axis=1)
    unit_chord = np.zeros_like(AC)
    valid = chord_length > 0
    if np.any(valid):
        unit_chord[valid] = (AC[valid].T / chord_length[valid]).T

    BA = B - A
    proj_len = np.einsum("ij,ij->i", BA, unit_chord)  # scalar projection of BA onto AC
    proj_vec = (proj_len[:, None]) * unit_chord
    perp_vec = BA - proj_vec
    sagitta = np.linalg.norm(perp_vec, axis=1)

    return float(np.nanpercentile(sagitta, percentile_p))


def _estimate_lag_ami(y: np.ndarray, max_lag: int | None = None, n_bins: int = 16) -> int:
    """
    Estimate optimal time lag via first local minimum of AMI.
    Fast cap: max_lag <= 2000 to avoid O(n * max_lag) blowups on long series.
    """
    y = np.asarray(y, float).ravel()
    n = y.size
    if n < 10:
        return 1
    if max_lag is None:
        max_lag = max(2, min(n // 10, 2000))
    else:
        max_lag = min(max_lag, n // 2)

    y_min, y_max = np.min(y), np.max(y)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or (y_max - y_min) < 1e-12:
        return 1
    y_norm = (y - y_min) / (y_max - y_min)

    ami_values = []
    for lag in range(1, max_lag + 1):
        y1 = y_norm[:-lag]
        y2 = y_norm[lag:]
        hist_2d, _, _ = np.histogram2d(y1, y2, bins=n_bins, range=[[0, 1], [0, 1]])
        p_joint = hist_2d / np.sum(hist_2d)
        p_y1 = np.sum(p_joint, axis=1)
        p_y2 = np.sum(p_joint, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.log(p_joint) - np.log(p_y1[:, None]) - np.log(p_y2[None, :])
            mi = np.nansum(p_joint * log_term)
        ami_values.append(max(mi, 0.0))

    ami_values = np.asarray(ami_values, float)
    for i in range(1, len(ami_values) - 1):
        if ami_values[i - 1] > ami_values[i] < ami_values[i + 1]:
            return i + 1
    return int(np.argmin(ami_values)) + 1


def _estimate_embedding_dim_fnn(
    y: np.ndarray,
    lag: int,
    max_dim: int = 15,
    rtol: float = 15.0,
    atol: float = 2.0,
    fnn_threshold: float = 0.01,
) -> int:
    """
    Fast Kennel FNN with subset neighbor pool + Theiler window + early stop.
    Internal constants (kept simple): pool_size=5000, n_test=500, theiler=2*lag.
    """
    x = np.asarray(y, float).ravel()
    n = x.size
    if n < 5 or lag < 1:
        return 2

    POOL_SIZE = 5000
    N_TEST = 500
    theiler = max(1, 2 * lag)

    sigma_x = x.std(ddof=1) if x.std(ddof=1) > 0 else 1.0

    chosen_m = 2
    below_twice = 0  # early stop once we’re below threshold twice in a row

    for m in range(2, max_dim + 1):
        # embedding lengths
        Tm = n - (m - 1) * lag
        Tm1 = n - m * lag
        if Tm1 < 10:
            break

        # build embeddings (contiguous views)
        Xm = np.column_stack([x[i * lag: i * lag + Tm] for i in range(m)])
        Xm1 = np.column_stack([x[i * lag: i * lag + Tm1] for i in range(m + 1)])

        # pool + test indices (uniform subsampling)
        pool_size = min(POOL_SIZE, Tm)
        pool_idx = np.linspace(0, Tm - 1, pool_size, dtype=int)
        if pool_size <= 2:
            break
        step = max(1, pool_size // N_TEST)
        test_idx = pool_idx[::step][:N_TEST]
        if test_idx.size == 0:
            break

        # distances in m-dim between test and pool
        from scipy.spatial.distance import cdist
        Dm = cdist(Xm[test_idx], Xm[pool_idx], metric="euclidean")

        # apply Theiler window per row (mask temporal neighbors)
        for r, ti in enumerate(test_idx):
            mask_bad = np.abs(pool_idx - ti) <= theiler
            Dm[r, mask_bad] = np.inf

        # nearest neighbor indices in the pool
        nn_col = np.argmin(Dm, axis=1)
        nn_dist_m = Dm[np.arange(nn_col.size), nn_col]
        valid_nn = np.isfinite(nn_dist_m)
        if not np.any(valid_nn):
            continue

        # map to absolute indices
        nn_idx = pool_idx[nn_col[valid_nn]]
        ti_valid = test_idx[valid_nn]
        Rm = nn_dist_m[valid_nn]

        # Only pairs where both indices < Tm1 can be used for E1/E2
        mask_range = (ti_valid < Tm1) & (nn_idx < Tm1) & (Rm > 0)
        if not np.any(mask_range):
            continue

        ti2 = ti_valid[mask_range]
        nj2 = nn_idx[mask_range]
        Rm2 = Rm[mask_range]

        # E1: unfolding along the new axis
        num = np.abs(x[ti2 + m * lag] - x[nj2 + m * lag])
        E1 = (num / Rm2) > rtol

        # E2: absolute growth in (m+1)-dim
        diff_m1 = Xm1[ti2] - Xm1[nj2]
        Rm1 = np.linalg.norm(diff_m1, axis=1)
        E2 = (Rm1 / sigma_x) > atol

        fnn_fraction = np.mean(E1 | E2)

        if fnn_fraction < fnn_threshold:
            below_twice += 1
            chosen_m = m
            if below_twice >= 2:
                break
        else:
            below_twice = 0
            chosen_m = m

    return int(max(2, chosen_m))


def _compute_sagitta_stats(samples: np.ndarray, span: int, percentile_p: float) -> tuple[float, float]:
    """
    Return (sagitta_percentile, chord_median) for triples at a given span.
    """
    A = samples[:-2 * span, :]
    B = samples[span:-span, :]
    C = samples[2 * span:, :]

    AC = C - A
    chord_length = np.linalg.norm(AC, axis=1)
    unit_chord = np.zeros_like(AC)
    valid = chord_length > 0
    if np.any(valid):
        unit_chord[valid] = (AC[valid].T / chord_length[valid]).T

    BA = B - A
    proj_len = np.einsum("ij,ij->i", BA, unit_chord)
    proj_vec = (proj_len[:, None]) * unit_chord
    perp_vec = BA - proj_vec
    sagitta = np.linalg.norm(perp_vec, axis=1)

    s_p = float(np.nanpercentile(sagitta, percentile_p))
    # robust chord scale
    chord_med = float(np.nanmedian(chord_length[valid])) if np.any(valid) else 0.0
    return s_p, chord_med


def _takens_embedding(y: np.ndarray, lag: int, embed_dim: int) -> np.ndarray:
    """
    Create Takens (delay coordinate) embedding of a 1D time series.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        1D time series.
    lag : int
        Time lag between coordinates.
    embed_dim : int
        Embedding dimension.

    Returns
    -------
    np.ndarray, shape (n_embedded, embed_dim)
        Embedded time series where each row is [y(t), y(t+lag), ..., y(t+(embed_dim-1)*lag)].
    """
    n_samples = len(y)
    n_embedded = n_samples - (embed_dim - 1) * lag

    if n_embedded <= 0:
        raise ValueError(f"Not enough samples for embedding: need > {(embed_dim - 1) * lag}, got {n_samples}")

    embedded = np.zeros((n_embedded, embed_dim))
    for i in range(embed_dim):
        embedded[:, i] = y[i * lag : i * lag + n_embedded]

    return embedded


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
    Sagitta-based Δt selector (derivative-free, robust, idempotent).

    - For d = 1: automatically applies Takens embedding after estimating optimal lag (AMI)
      and embedding dimension (FNN). The embedding is transparent to the user.
    - For d > 1: computes sagitta in state space as before.
      y_std is per-feature σ-normalized (ddof=1).

    Parameters
    ----------
    y : np.ndarray, shape (n_samples, n_dim) or (n_samples,)
        Time-ordered samples y_i. If 1D, Takens embedding will be applied automatically.
    dt0 : float
        Base sampling step Δt0 > 0.
    epsilon : float
        Geometric tolerance (in σ-normalized units).
    percentile : float
        Robust percentile p (e.g., 95.0).
    coarsen_only : bool
        If True, never suggest Δt* < Δt0 (i.e., m >= 1).
    min_points_per_segment : int
        Require at least this many triples (i-span, i, i+span) to evaluate a candidate span.
    search_growth : float
        Multiplicative growth for coarse search over spans.

    Returns
    -------
    SagittaDt
        Result with chosen Δt*, stride, achieved percentile value, indices, and bookkeeping.
        For 1D input, the 'notes' field will contain embedding parameters (lag, dimension).
    """
    # -------- input validation and dimensionality handling --------
    if not isinstance(y, np.ndarray):
        raise ValueError("y must be a numpy array.")

    # Convert to 2D if needed and detect if we need embedding
    needs_embedding = False
    embedding_lag = None
    embedding_dim = None

    if y.ndim == 1:
        # 1D time series - will need Takens embedding
        needs_embedding = True
        y_original = y.copy()
        n_samples_original = len(y)
    elif y.ndim == 2:
        n_samples, n_dim = y.shape
        if n_dim == 1:
            # Shape is (n, 1) - treat as 1D
            needs_embedding = True
            y_original = y.squeeze()
            n_samples_original = len(y_original)
        # else: n_dim > 1, proceed normally
    else:
        raise ValueError("y must be 1D or 2D array.")

    # Apply Takens embedding if needed
    embedding_notes = ""
    if needs_embedding:
        if n_samples_original < 50:
            raise ValueError(f"Need at least 50 samples for 1D embedding, got {n_samples_original}.")

        print("Need to apply Takens embedding")

        # Estimate optimal lag using AMI
        print("Estimating lag...")
        embedding_lag = _estimate_lag_ami(y_original)

        # Estimate optimal embedding dimension using FNN
        print("Estimating embedding dimension...")
        embedding_dim = _estimate_embedding_dim_fnn(y_original, embedding_lag)

        # Apply Takens embedding
        y = _takens_embedding(y_original, embedding_lag, embedding_dim)
        n_samples, n_dim = y.shape

        embedding_notes = f"Applied Takens embedding: lag={embedding_lag}, dim={embedding_dim}. "
    else:
        # Already 2D with n_dim > 1
        n_samples, n_dim = y.shape

    # Standard validation
    if n_samples < 5:
        raise ValueError("Need at least n_samples >= 5 after embedding.")
    if not (dt0 > 0):
        raise ValueError("dt0 must be > 0.")
    if not (epsilon > 0):
        raise ValueError("epsilon must be > 0.")
    if not (0.0 < percentile <= 100.0):
        raise ValueError("percentile must be in (0, 100].")
    if search_growth <= 1.0:
        raise ValueError("search_growth must be > 1.0.")
    if min_points_per_segment < 1:
        raise ValueError("min_points_per_segment must be >= 1.")

    # -------- per-feature σ-normalization (scale invariance) --------
    y = np.asarray(y, dtype=float)
    feature_std = y.std(axis=0, ddof=1)
    feature_std[~np.isfinite(feature_std) | (feature_std == 0.0)] = 1.0 # when not finite or zero, set to 1
    normalized_values = y / feature_std  # shape (n_samples, n_dim)

    samples_for_sagitta = normalized_values
    notes = embedding_notes + "Used state-space sagitta."

    # -------- determine max span with enough triples --------
    max_span = (n_samples - 1) // 2
    while max_span > 1 and (n_samples - 2 * max_span) < min_points_per_segment:
        max_span -= 1
    if max_span < 1:
        # Not enough data to evaluate any span > 0; fall back to Δt0
        chosen_stride = 1
        achieved_percentile = 0.0
        searched_spans = np.array([1], dtype=int)
        indices = np.arange(n_samples, dtype=int)
        delta_t_star = chosen_stride * dt0
        return SagittaDt(
            delta_t=float(delta_t_star),
            stride=int(chosen_stride),
            percentile_value=float(achieved_percentile),
            indices=indices,
            p=float(percentile),
            epsilon=float(epsilon),
            searched_ms=searched_spans,
            notes=notes + " Too few samples; returning Δt0."
        )

    use_relative = needs_embedding  # relative sagitta only for 1D (embedded); d>1 unchanged

    # -------- helper to test a span --------
    def span_is_ok(span: int) -> Tuple[bool, float]:
        if (n_samples - 2 * span) < min_points_per_segment:
            return False, np.nan
        if use_relative:
            s_p, chord_med = _compute_sagitta_stats(samples_for_sagitta, span, percentile)
            denom = chord_med if (chord_med > 1e-12 and np.isfinite(chord_med)) else 1.0
            rel = s_p / denom
            return (rel <= epsilon), rel   # interpret epsilon as RELATIVE tolerance in 1D case
        else:
            val = _compute_sagitta_percentile(samples_for_sagitta, span, percentile)
            return (val <= epsilon), val   # keep absolute criterion for d>1 (backward-compatible)


    # -------- coarse search (exponential growth) --------
    evaluated = []
    current_span = 1
    ok, best_val = span_is_ok(1)
    evaluated.append((1, best_val))

    chosen_stride = 1 if ok else 0
    achieved_percentile = best_val

    # If even span=1 violates epsilon and coarsen_only, clamp to Δt0
    if (not ok) and coarsen_only:
        chosen_stride = 1
        achieved_percentile = best_val
        searched_spans = np.array([1], dtype=int)
        indices = np.arange(n_samples, dtype=int)
        delta_t_star = chosen_stride * dt0
        return SagittaDt(
            delta_t=float(delta_t_star),
            stride=int(chosen_stride),
            percentile_value=float(achieved_percentile),
            indices=indices,
            p=float(percentile),
            epsilon=float(epsilon),
            searched_ms=searched_spans,
            notes=notes + " span=1 exceeds ε; coarsen_only=True so Δt*=Δt0."
        )

    # grow span until it fails or we hit max_span
    first_fail_span = None
    while True:
        next_span = int(np.floor(current_span * search_growth))
        if next_span <= current_span:
            next_span = current_span + 1
        if next_span > max_span:
            break
        ok_next, val_next = span_is_ok(next_span)
        evaluated.append((next_span, val_next))
        if ok_next:
            chosen_stride = next_span
            achieved_percentile = val_next
            current_span = next_span
        else:
            first_fail_span = next_span
            break

    # -------- binary search between last ok and first fail (if any) --------
    low_ok = max(1, chosen_stride if chosen_stride else 1)
    high_fail = first_fail_span if first_fail_span is not None else (max_span + 1)

    while high_fail - low_ok > 1:
        mid = (low_ok + high_fail) // 2
        ok_mid, val_mid = span_is_ok(mid)
        evaluated.append((mid, val_mid))
        if ok_mid:
            low_ok = mid
            chosen_stride = mid
            achieved_percentile = val_mid
        else:
            high_fail = mid

    # -------- build decimation indices for chosen stride --------
    if chosen_stride < 1:
        chosen_stride = 1
    indices = np.arange(0, n_samples, chosen_stride, dtype=int)
    if indices[-1] != (n_samples - 1):
        indices = np.concatenate([indices, np.array([n_samples - 1], dtype=int)])

    # -------- finalize single return --------
    delta_t_star = chosen_stride * dt0
    evaluated_sorted = np.array(sorted(evaluated, key=lambda x: x[0]), dtype=float)
    searched_spans = evaluated_sorted[:, 0].astype(int) if evaluated_sorted.size else np.array([1], dtype=int)

    return SagittaDt(
        delta_t=float(delta_t_star),
        stride=int(chosen_stride),
        percentile_value=float(achieved_percentile),
        indices=indices,
        p=float(percentile),
        epsilon=float(epsilon),
        searched_ms=searched_spans,
        notes=notes
    )
