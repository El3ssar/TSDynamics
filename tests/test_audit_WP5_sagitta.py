"""Regression tests for WP5_sagitta — min-points guard + degenerate-triple consistency.

Covers two audit findings in ``tsdynamics.analysis.sampling.sagitta``:

* A11-1: ``min_points_per_segment`` must be compared against the *actual* number
  of strided triples ``len(np.arange(span, n-span, span))``, not the admissible
  centre span ``n - 2*span``.  Pre-fix, a span evaluated from a single triple
  passed the guard.
* A11-3: degenerate (zero-chord) triples must be excluded from the sagitta
  percentile too, not just the chord median — otherwise injected zeros bias the
  numerator of the relative criterion downward.
"""

from __future__ import annotations

import numpy as np

from tsdynamics.analysis.sampling.sagitta import (
    _compute_sagitta_stats,
    _triple_count,
    estimate_dt_from_sagitta,
)


def test_triple_count_matches_arange_length() -> None:
    """``_triple_count`` equals the strided centre count, not ``n - 2*span``."""
    for n in (50, 100, 137):
        for span in range(1, (n - 1) // 2 + 1):
            expected = len(np.arange(span, n - span, span))
            assert _triple_count(n, span) == expected


def test_min_points_guard_rejects_single_triple_spans() -> None:
    """A span with too few real triples is rejected by the min-points guard.

    Pre-fix the guard compared ``n - 2*span`` (e.g. 20 for n=100, span=40) to
    ``min_points_per_segment``, so a span yielding a single triple passed and the
    percentile was a one-sample estimate.  With the real triple count, no chosen
    stride may rest on fewer than ``min_points_per_segment`` triples.
    """
    # Multivariate input avoids the embedding path; smoothly varying so span=1 is ok.
    t = np.linspace(0.0, 8.0, 100)
    y = np.column_stack([np.sin(t), np.cos(t), 0.3 * np.sin(2.0 * t)])

    res = estimate_dt_from_sagitta(
        y, dt0=0.1, epsilon=10.0, min_points_per_segment=3, use_relative=False
    )

    # Every span actually evaluated must carry at least min_points_per_segment triples.
    for span in res.searched_ms.astype(int):
        assert _triple_count(len(y), int(span)) >= 3, (
            f"span={span} evaluated with only {_triple_count(len(y), int(span))} triples"
        )
    # The chosen stride likewise.
    assert _triple_count(len(y), res.stride) >= 3


def test_degenerate_triples_excluded_from_percentile() -> None:
    """Zero-chord triples do not dilute the sagitta percentile (consistent population).

    Build samples where every strided triple at ``span=1`` is a genuine bow EXCEPT
    a block of repeated (stalled) points whose chord is zero.  Pre-fix the
    percentile ran over all triples including injected zeros, dragging ``s_p``
    below the true bow of the valid triples; post-fix it equals the percentile of
    the valid triples alone.
    """
    span = 1
    # Valid stretch: a zig-zag with a small bow, so its 95th percentile is positive.
    valid_part = np.array([[float(i), 1.0 if i % 2 == 0 else -1.0] for i in range(20)], dtype=float)
    # Degenerate stretch: dominate with repeated (stalled) points → zero chords,
    # so an all-triples percentile collapses to ~0 while the valid-only one stays
    # at the true bow.
    stalled = np.tile(valid_part[-1], (600, 1))
    samples = np.vstack([valid_part, stalled])

    s_p, chord_med = _compute_sagitta_stats(samples, span, 95.0)

    # Reference: percentile over the valid triples only.
    centers = np.arange(span, samples.shape[0] - span, span)
    ac = samples[centers + span] - samples[centers - span]
    chord = np.linalg.norm(ac, axis=1)
    valid = chord > 0
    ba = samples[centers] - samples[centers - span]
    unit = np.zeros_like(ac)
    unit[valid] = (ac[valid].T / chord[valid]).T
    proj = np.einsum("ij,ij->i", ba, unit)[:, None] * unit
    sag = np.linalg.norm(ba - proj, axis=1)
    expected_s_p = float(np.nanpercentile(sag[valid], 95.0))

    assert s_p == expected_s_p
    # Pre-fix value (percentile over ALL triples, zeros included) is strictly smaller.
    s_p_with_zeros = float(np.nanpercentile(np.where(valid, sag, 0.0), 95.0))
    assert s_p_with_zeros < expected_s_p
    assert s_p > 0.0
    assert chord_med > 0.0
