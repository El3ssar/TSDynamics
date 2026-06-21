from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist  # fix #3: module-level import

__all__ = ["SagittaDt", "estimate_dt_from_sagitta"]


@dataclass(frozen=True)
class SagittaDt:
    """Result container for sagitta-based Δt selection."""

    delta_t: float  # Δt* = stride * base_dt
    stride: int  # stride m*
    percentile_value: float  # achieved S_p(m*) at the chosen stride
    indices: np.ndarray  # decimation indices for current data (include last)
    p: float  # percentile used (kept for backward-compat)
    epsilon: float  # tolerance used (kept for backward-compat)
    searched_ms: np.ndarray  # candidate strides evaluated
    notes: str = ""  # info / warnings


def _compute_sagitta_stats(
    samples: np.ndarray, span: int, percentile_p: float
) -> tuple[float, float]:
    """Return (sagitta_percentile, chord_median) for triples at a given span."""
    # fix #5: removed _compute_sagitta_percentile duplicate; all callers use this
    n = samples.shape[0]
    centers = np.arange(span, n - span, span)

    A = samples[centers - span, :]
    B = samples[centers, :]
    C = samples[centers + span, :]

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
    chord_med = float(np.nanmedian(chord_length[valid])) if np.any(valid) else 0.0
    return s_p, chord_med


def _estimate_lag_ami(y: np.ndarray, max_lag: int | None = None, n_bins: int = 16) -> int:
    """
    Estimate optimal time lag via first local minimum of AMI.

    Fast cap: ``max_lag <= 2000`` to avoid O(n × max_lag) blowups on long series.
    """
    y = np.asarray(y, float).ravel()
    n = y.size
    if n < 10:
        return 1
    max_lag = max(2, min(n // 10, 2000)) if max_lag is None else min(max_lag, n // 2)

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

    ami_arr = np.asarray(ami_values, float)
    for i in range(1, len(ami_arr) - 1):
        if ami_arr[i - 1] > ami_arr[i] < ami_arr[i + 1]:
            return i + 1
    return int(np.argmin(ami_arr)) + 1


def _estimate_embedding_dim_fnn(
    y: np.ndarray,
    lag: int,
    max_dim: int = 15,
    rtol: float = 15.0,
    atol: float = 2.0,
    fnn_threshold: float = 0.01,
) -> int:
    """
    Estimate embedding dimension via False Nearest Neighbours (Kennel algorithm).

    Uses a subset neighbor pool + Theiler window + early stop.
    Internal constants: pool_size=5000, n_test=500, theiler=2*lag.
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
    below_twice = 0

    for m in range(2, max_dim + 1):
        Tm = n - (m - 1) * lag
        Tm1 = n - m * lag
        if Tm1 < 10:
            break

        Xm = np.column_stack([x[i * lag : i * lag + Tm] for i in range(m)])
        Xm1 = np.column_stack([x[i * lag : i * lag + Tm1] for i in range(m + 1)])

        pool_size = min(POOL_SIZE, Tm)
        pool_idx = np.linspace(0, Tm - 1, pool_size, dtype=int)
        if pool_size <= 2:
            break
        step = max(1, pool_size // N_TEST)
        test_idx = pool_idx[::step][:N_TEST]
        if test_idx.size == 0:
            break

        Dm = cdist(Xm[test_idx], Xm[pool_idx], metric="euclidean")  # fix #3: no local import

        for r, ti in enumerate(test_idx):
            mask_bad = np.abs(pool_idx - ti) <= theiler
            Dm[r, mask_bad] = np.inf

        nn_col = np.argmin(Dm, axis=1)
        nn_dist_m = Dm[np.arange(nn_col.size), nn_col]
        valid_nn = np.isfinite(nn_dist_m)
        if not np.any(valid_nn):
            continue

        nn_idx = pool_idx[nn_col[valid_nn]]
        ti_valid = test_idx[valid_nn]
        Rm = nn_dist_m[valid_nn]

        mask_range = (ti_valid < Tm1) & (nn_idx < Tm1) & (Rm > 0)
        if not np.any(mask_range):
            continue

        ti2 = ti_valid[mask_range]
        nj2 = nn_idx[mask_range]
        Rm2 = Rm[mask_range]

        num = np.abs(x[ti2 + m * lag] - x[nj2 + m * lag])
        E1 = (num / Rm2) > rtol

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
        raise ValueError(
            f"Not enough samples for embedding: need > {(embed_dim - 1) * lag}, got {n_samples}"
        )

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
    use_relative: bool | None = None,
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
        Geometric tolerance. Absolute (σ-normalized units) when use_relative=False,
        relative (sagitta / chord_median) when use_relative=True.
    percentile : float
        Robust percentile p (e.g., 95.0).
    coarsen_only : bool
        If True, never suggest Δt* < Δt0 (i.e., m >= 1). Does not affect behaviour
        when span=1 already exceeds ε — that case is flagged via 'notes' and
        stride=1 is returned regardless.
    use_relative : bool or None
        Whether to use relative sagitta criterion (sagitta / chord_median <= ε).
        If None (default), automatically set to True for 1D input (post-embedding)
        and False for multivariate input. Pass explicitly to override.
        NOTE: relative criterion is not strictly monotone in stride; binary search
        is approximate for that case.
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

    needs_embedding = False

    if y.ndim == 1:
        needs_embedding = True
        y_original = y.copy()
        n_samples_original = len(y)
    elif y.ndim == 2:
        n_samples_original_unused, n_dim = y.shape
        if n_dim == 1:
            needs_embedding = True
            y_original = y.squeeze()
            n_samples_original = len(y_original)
    else:
        raise ValueError("y must be 1D or 2D array.")

    embedding_notes = ""
    if needs_embedding:
        if n_samples_original < 50:
            raise ValueError(
                f"Need at least 50 samples for 1D embedding, got {n_samples_original}."
            )

        print("Estimating lag...")
        embedding_lag = _estimate_lag_ami(y_original)

        print("Estimating embedding dimension...")
        embedding_dim = _estimate_embedding_dim_fnn(y_original, embedding_lag)

        y = _takens_embedding(y_original, embedding_lag, embedding_dim)
        n_samples, n_dim = y.shape

        embedding_notes = f"Applied Takens embedding: lag={embedding_lag}, dim={embedding_dim}. "
    else:
        n_samples, n_dim = y.shape

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

    # fix #2: use_relative is now an explicit parameter, not silently tied to needs_embedding
    if use_relative is None:
        use_relative = needs_embedding

    # -------- per-feature σ-normalization (scale invariance) --------
    y = np.asarray(y, dtype=float)
    feature_std = y.std(axis=0, ddof=1)
    feature_std[~np.isfinite(feature_std) | (feature_std == 0.0)] = 1.0
    samples_for_sagitta = y / feature_std

    criterion_note = "relative sagitta" if use_relative else "absolute sagitta"
    notes = embedding_notes + f"Used state-space {criterion_note}."
    if use_relative:
        notes += " NOTE: relative criterion is not strictly monotone; binary search is approximate."

    # -------- determine max span with enough triples --------
    max_span = (n_samples - 1) // 2
    while max_span > 1 and (n_samples - 2 * max_span) < min_points_per_segment:
        max_span -= 1

    def _make_result(
        stride: int, achieved: float, searched: list[tuple[int, float]], extra_note: str
    ) -> SagittaDt:
        idx = np.arange(0, n_samples, stride, dtype=int)
        if idx[-1] != (n_samples - 1):
            idx = np.concatenate([idx, np.array([n_samples - 1], dtype=int)])
        evs = np.array(sorted(searched, key=lambda x: x[0]), dtype=float)
        spans = evs[:, 0].astype(int) if evs.size else np.array([1], dtype=int)
        return SagittaDt(
            delta_t=float(stride * dt0),
            stride=int(stride),
            percentile_value=float(achieved),
            indices=idx,
            p=float(percentile),
            epsilon=float(epsilon),
            searched_ms=spans,
            notes=notes + extra_note,
        )

    if max_span < 1:
        return _make_result(1, 0.0, [(1, 0.0)], " Too few samples; returning Δt0.")

    # -------- helper to test a span --------
    def span_is_ok(span: int) -> tuple[bool, float]:
        if (n_samples - 2 * span) < min_points_per_segment:
            return False, np.nan
        s_p, chord_med = _compute_sagitta_stats(samples_for_sagitta, span, percentile)
        if use_relative:
            denom = chord_med if (chord_med > 1e-12 and np.isfinite(chord_med)) else 1.0
            val = s_p / denom
        else:
            val = s_p
        return (val <= epsilon), val

    # -------- coarse search (exponential growth) --------
    evaluated: list[tuple[int, float]] = []

    ok_1, val_1 = span_is_ok(1)
    evaluated.append((1, val_1))

    if not ok_1:
        # fix #1: this is not a coarsen_only decision — span=1 failing means the data
        # is already under-sampled relative to ε. Return stride=1 with a clear note.
        # coarsen_only is irrelevant here since we cannot go finer.
        return _make_result(
            1, val_1, evaluated, " span=1 exceeds ε; data may be under-sampled at dt0."
        )

    # fix #4: chosen_stride always valid from here; no sentinel 0 needed
    chosen_stride = 1
    achieved_percentile = val_1
    current_span = 1
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
    # fix #6: low_ok is now always a confirmed-ok stride (no sentinel ambiguity)
    low_ok = chosen_stride
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

    return _make_result(chosen_stride, achieved_percentile, evaluated, "")


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
