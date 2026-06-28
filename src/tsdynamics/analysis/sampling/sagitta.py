"""Sagitta-based Δt selector for time-ordered samples.

A derivative-free, scale-invariant, idempotent heuristic for choosing an output
sampling step Δt* from a sampled trajectory.  It decimates the series and, for a
range of candidate strides, measures the *sagitta* — the perpendicular bow of the
midpoint of each (i−span, i, i+span) triple off its chord — at a robust
percentile, then picks the largest stride whose sagitta stays within a geometric
tolerance.  One-dimensional input is first delay-embedded (lag via AMI, dimension
via FNN) so the geometric criterion lives in a reconstructed state space.

The public surface is :func:`estimate_dt_from_sagitta` (the Δt selector) and
:func:`sagitta_profile` (the per-point sagitta along a curve — its bow off the
local chord, used e.g. as a ``color_by`` field).  The result container
``SagittaDt`` is intentionally not exported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsdynamics.analysis.embedding import embedding_dimension, optimal_delay
from tsdynamics.errors import invalid_value

__all__ = ["estimate_dt_from_sagitta", "sagitta_profile"]


@dataclass(frozen=True)
class SagittaDt:
    r"""Result of a sagitta-based :math:`\Delta t` selection.

    Returned by :func:`estimate_dt_from_sagitta`; not part of the public export
    surface (reach it via the return value).

    Attributes
    ----------
    delta_t : float
        The recommended output step :math:`\Delta t^\ast = \text{stride} \times
        \Delta t_0`.
    stride : int
        The chosen decimation stride :math:`m^\ast`.
    percentile_value : float
        The achieved sagitta percentile :math:`S_p(m^\ast)` at that stride.
    indices : numpy.ndarray
        Decimation indices for the input data (always including the last sample).
    p : float
        The percentile used (kept for inspection).
    epsilon : float
        The geometric tolerance used (kept for inspection).
    searched_ms : numpy.ndarray
        The candidate strides evaluated during the search.
    notes : str
        Informational / warning text (e.g. embedding parameters for 1-D input).
    """

    delta_t: float  # Δt* = stride * base_dt
    stride: int  # stride m*
    percentile_value: float  # achieved S_p(m*) at the chosen stride
    indices: np.ndarray  # decimation indices for current data (include last)
    p: float  # percentile used (kept for backward-compat)
    epsilon: float  # tolerance used (kept for backward-compat)
    searched_ms: np.ndarray  # candidate strides evaluated
    notes: str = ""  # info / warnings


def _sagitta_chord(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-triple ``(sagitta, chord_length)`` for points ``a, b, c``.

    The *sagitta* is the perpendicular bow of the middle point ``b`` off the chord
    ``a -> c``; the *chord length* is ``|c - a|``.  This is the one geometric
    kernel shared by the Δt selector (:func:`_compute_sagitta_stats`) and the
    per-point profile (:func:`sagitta_profile`).

    A *degenerate* triple — one whose chord ``|c - a|`` is zero — has no defined
    perpendicular and its sagitta is set to ``0`` (rather than the spurious full
    ``|b - a|`` a zero chord direction would otherwise produce); such triples are
    also dropped from the chord-median denominator downstream.
    """
    ac = c - a
    chord = np.linalg.norm(ac, axis=1)
    unit = np.zeros_like(ac)
    valid = chord > 0
    if np.any(valid):
        unit[valid] = (ac[valid].T / chord[valid]).T
    ba = b - a
    proj = np.einsum("ij,ij->i", ba, unit)[:, None] * unit
    sagitta = np.linalg.norm(ba - proj, axis=1)
    # Degenerate (zero-length) chords have no defined bow: score them 0 instead of
    # the full |b - a| that the zero unit vector would yield.
    sagitta = np.where(valid, sagitta, 0.0)
    return sagitta, chord


def _compute_sagitta_stats(
    samples: np.ndarray, span: int, percentile_p: float
) -> tuple[float, float]:
    """Return ``(sagitta_percentile, chord_median)`` for triples at a given span.

    Degenerate triples (zero-length chord) contribute a sagitta of ``0`` and are
    excluded from the chord median.
    """
    n = samples.shape[0]
    centers = np.arange(span, n - span, span)
    sagitta, chord_length = _sagitta_chord(
        samples[centers - span, :], samples[centers, :], samples[centers + span, :]
    )
    valid = chord_length > 0
    s_p = float(np.nanpercentile(sagitta, percentile_p))
    chord_med = float(np.nanmedian(chord_length[valid])) if np.any(valid) else 0.0
    return s_p, chord_med


def sagitta_profile(
    samples: np.ndarray,
    *,
    span: int = 1,
    relative: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Per-point *sagitta* along a sampled curve — its local bow off the chord.

    For each interior point ``i`` the sagitta is the perpendicular distance of
    ``samples[i]`` from the chord through its neighbours ``(i-span, i+span)`` (the
    same geometry the sagitta-based Δt selector uses, here evaluated at *every*
    point rather than a strided percentile).  Large where the trajectory bends
    sharply, ~0 on straight runs; the first/last ``span`` points are 0.

    Parameters
    ----------
    samples : np.ndarray, shape (n,) or (n, d)
        Time-ordered points (a 1-D series is treated as a column).
    span : int, optional
        Neighbour offset of the chord ``(i-span, i, i+span)``.  Default 1.
    relative : bool, optional
        Divide each bow by its chord length, giving a dimensionless *bend* ratio
        that is (largely) independent of the local speed/step.  Default ``True``.
    normalize : bool, optional
        σ-normalize each feature first (per-feature std, ``ddof=1``) so no single
        coordinate dominates the geometry.  Default ``True``.

    Returns
    -------
    np.ndarray, shape (n,)
        The per-point sagitta, aligned to ``samples``.
    """
    s = np.asarray(samples, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    n = s.shape[0]
    out = np.zeros(n, dtype=float)
    if span < 1 or n < 2 * span + 1:
        return out
    if normalize:
        std = s.std(axis=0, ddof=1)
        std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
        s = s / std
    i = np.arange(span, n - span)
    sagitta, chord = _sagitta_chord(s[i - span], s[i], s[i + span])
    if relative:
        sagitta = sagitta / np.where(chord > 0.0, chord, 1.0)
    out[i] = sagitta
    return out


def _takens_embedding(y: np.ndarray, lag: int, embed_dim: int) -> np.ndarray:
    """Build a Takens (delay-coordinate) embedding of a 1-D series.

    Each row is ``[y(t), y(t+lag), ..., y(t+(embed_dim-1)*lag)]``.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        The scalar time series.
    lag : int
        Delay (in samples) between successive coordinates.
    embed_dim : int
        Embedding dimension.

    Returns
    -------
    np.ndarray, shape (n_embedded, embed_dim)
        The delay-coordinate vectors.

    Raises
    ------
    InvalidParameterError
        If the series is too short for the requested ``lag``/``embed_dim``.
    """
    n_samples = len(y)
    n_embedded = n_samples - (embed_dim - 1) * lag

    if n_embedded <= 0:
        raise invalid_value(
            "y",
            n_samples,
            rule=f"too few samples for embedding (need > {(embed_dim - 1) * lag})",
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
    r"""Sagitta-based output-step :math:`\Delta t^\ast` selector.

    A derivative-free, scale-invariant, idempotent heuristic for the output
    sampling step.  For a range of candidate strides it measures the *sagitta* —
    the perpendicular bow of the midpoint of each ``(i-span, i, i+span)`` triple
    off its chord — at a robust percentile, then picks the largest stride whose
    sagitta still satisfies the geometric tolerance ``epsilon``.

    A one-dimensional input is first delay-embedded so the geometric criterion
    lives in a reconstructed state space: the delay is the first minimum of the
    time-delayed mutual information (Fraser & Swinney 1986) and the dimension is
    Kennel's false-nearest-neighbour estimate (Kennel, Brown & Abarbanel 1992),
    both via the public :mod:`tsdynamics.analysis.embedding` estimators
    (:func:`~tsdynamics.analysis.embedding.optimal_delay` and
    :func:`~tsdynamics.analysis.embedding.embedding_dimension` with
    ``method="fnn"``).  Multivariate input is used directly, per-feature
    σ-normalised (``ddof=1``).

    Parameters
    ----------
    y : numpy.ndarray, shape (n_samples,) or (n_samples, n_dim)
        Time-ordered samples.  A 1-D (or ``(n, 1)``) series is delay-embedded
        automatically.  Must be finite.
    dt0 : float
        Base sampling step :math:`\Delta t_0 > 0`.
    epsilon : float
        Geometric tolerance :math:`\varepsilon > 0`.  Absolute (σ-normalised
        units) when ``use_relative=False``; relative (``sagitta / chord_median``)
        when ``use_relative=True``.
    percentile : float, optional
        Robust percentile :math:`p \in (0, 100]`.  Default ``95.0``.
    coarsen_only : bool, optional
        If ``True``, never suggest :math:`\Delta t^\ast < \Delta t_0`
        (``stride >= 1``).  Does not affect behaviour when ``span=1`` already
        exceeds :math:`\varepsilon` — that case is flagged via ``notes`` and
        ``stride=1`` is returned regardless.  Default ``True``.
    use_relative : bool or None, optional
        Whether to use the relative sagitta criterion
        (``sagitta / chord_median <= epsilon``).  ``None`` (default) selects
        ``True`` for 1-D input (post-embedding) and ``False`` for multivariate
        input.  The relative criterion is not strictly monotone in stride, so the
        binary search is approximate for that case.
    min_points_per_segment : int, optional
        Minimum number of triples ``(i-span, i, i+span)`` required to evaluate a
        candidate span.  Default ``3``.
    search_growth : float, optional
        Multiplicative growth factor (``> 1``) for the coarse stride search.
        Default ``1.5``.

    Returns
    -------
    SagittaDt
        The chosen :math:`\Delta t^\ast`, stride, achieved percentile value,
        decimation ``indices``, and search bookkeeping.  For 1-D input the
        ``notes`` field records the embedding parameters (lag, dimension).

    Raises
    ------
    InvalidParameterError
        If ``y`` is not a NumPy array, is not 1-D/2-D, contains non-finite
        values, has too few samples, or if ``dt0`` / ``epsilon`` / ``percentile``
        / ``search_growth`` / ``min_points_per_segment`` violate their bounds.

    References
    ----------
    A. M. Fraser and H. L. Swinney, "Independent coordinates for strange
    attractors from mutual information", *Phys. Rev. A* **33**, 1134 (1986).

    M. B. Kennel, R. Brown and H. D. I. Abarbanel, "Determining embedding
    dimension for phase-space reconstruction using a geometrical construction",
    *Phys. Rev. A* **45**, 3403 (1992).
    """
    # -------- input validation and dimensionality handling --------
    if not isinstance(y, np.ndarray):
        raise invalid_value("y", type(y).__name__, rule="must be a numpy array")
    if not np.all(np.isfinite(y)):
        raise invalid_value("y", "non-finite", rule="must be finite (no nan/inf)")

    needs_embedding = False

    if y.ndim == 1:
        needs_embedding = True
        y_original = y.copy()
        n_samples_original = len(y)
    elif y.ndim == 2:
        _, n_dim = y.shape
        if n_dim == 1:
            needs_embedding = True
            y_original = y.squeeze()
            n_samples_original = len(y_original)
    else:
        raise invalid_value("y", y.ndim, rule="must be a 1D or 2D array (got y.ndim)")

    embedding_notes = ""
    if needs_embedding:
        if n_samples_original < 50:
            raise invalid_value(
                "y",
                n_samples_original,
                rule="must have at least 50 samples for 1D embedding (got len(y))",
            )

        # Delegate delay/dimension selection to the public embedding estimators
        # (first-minimum AMI delay; Kennel FNN dimension) — one implementation,
        # no drift.
        embedding_lag = int(optimal_delay(y_original, method="mi"))
        embedding_dim = int(embedding_dimension(y_original, method="fnn", delay=embedding_lag))

        y = _takens_embedding(y_original, embedding_lag, embedding_dim)
        n_samples, n_dim = y.shape

        embedding_notes = f"Applied Takens embedding: lag={embedding_lag}, dim={embedding_dim}. "
    else:
        n_samples, n_dim = y.shape

    if n_samples < 5:
        raise invalid_value(
            "y", n_samples, rule="must have at least 5 samples after embedding (got n_samples)"
        )
    if not (dt0 > 0):
        raise invalid_value("dt0", dt0, rule="must be > 0")
    if not (epsilon > 0):
        raise invalid_value("epsilon", epsilon, rule="must be > 0")
    if not (0.0 < percentile <= 100.0):
        raise invalid_value("percentile", percentile, rule="must be in (0, 100]")
    if search_growth <= 1.0:
        raise invalid_value("search_growth", search_growth, rule="must be > 1.0")
    if min_points_per_segment < 1:
        raise invalid_value("min_points_per_segment", min_points_per_segment, rule="must be >= 1")

    # use_relative is an explicit parameter, not silently tied to needs_embedding
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
