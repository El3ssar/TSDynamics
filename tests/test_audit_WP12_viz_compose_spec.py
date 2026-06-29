"""Regression tests for WP12 — viz compose input-aliasing + Animation frames.

Covers three audit findings:

- ``X4a-1``: :func:`tsdynamics.viz.plot` (overlay) must not share its presentation
  objects (axes / colorbar / legend) with the first input spec, so an in-place
  tweak on the composed result cannot rewrite an input.
- ``X4a-3``: :meth:`Animation.frame_count` must honor the ``n_samples`` cap at the
  degenerate single-sample boundary (a single-sample layer is a still — one frame,
  never two duplicate frames), and :meth:`Animation.head_indices` must stay safe.
- ``X4a-4``: passing an explicit :class:`Animation` to ``plot(..., animate=anim)``
  must honor that Animation's own ``head`` verbatim (no silent per-kind override),
  with no dead ``replace(x, head=x.head)`` no-op.
"""

from __future__ import annotations

import numpy as np

from tsdynamics.viz import plot
from tsdynamics.viz.spec import (
    Animation,
    Colorbar,
    Layer,
    Legend,
    PlotKind,
    PlotSpec,
)


def _series_spec(title: str) -> PlotSpec:
    """Build a minimal overlay-compatible time-series spec (no integration)."""
    t = np.linspace(0.0, 1.0, 8)
    layer = Layer(PlotKind.TIME_SERIES, {"x": t, "y": np.sin(t)}, label="x")
    return PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[layer],
        title=title,
        colorbar=Colorbar(),
        legend=Legend(),
    )


# ---------------------------------------------------------------------------
# X4a-1 — overlay must not alias the first input's presentation objects
# ---------------------------------------------------------------------------


def test_overlay_does_not_alias_input_axes() -> None:
    """The merged spec's axes/colorbar/legend are fresh copies, not input refs."""
    s1, s2 = _series_spec("a"), _series_spec("b")
    merged = plot(s1, s2)

    # Distinct object identities — the spooky-action-at-a-distance precondition.
    assert merged.x is not s1.x
    assert merged.y is not s1.y
    assert merged.colorbar is not s1.colorbar

    # And the contract that matters: an in-place tweak on the result leaves the
    # input untouched (pre-fix this rewrote s1.x.label).
    before = s1.x.label
    merged.relabel(x="MUTATED-TIME").rescale(y="log")
    assert s1.x.label == before
    assert s1.y.scale == "linear"
    assert merged.x.label == "MUTATED-TIME"


# ---------------------------------------------------------------------------
# X4a-3 — single-sample frame_count / head_indices boundary
# ---------------------------------------------------------------------------


def test_frame_count_single_sample_is_a_still() -> None:
    """A 1-sample layer yields exactly one frame, never two duplicates."""
    anim = Animation()
    assert anim.frame_count(1) == 1
    assert anim.frame_count(0) == 1
    # head_indices points at the only sample, once — pre-fix it returned [0, 0].
    assert anim.head_indices(1) == [0]
    # The normal (multi-sample) path is unchanged: still floored at 2.
    assert anim.frame_count(2) == 2
    assert anim.frame_count(5) == 5
    assert anim.head_indices(3)[0] == 0
    assert anim.head_indices(3)[-1] == 2


def test_frame_count_n_frames_capped_at_one_sample() -> None:
    """An explicit n_frames cannot fabricate frames beyond a single sample."""
    assert Animation(n_frames=10).frame_count(1) == 1


# ---------------------------------------------------------------------------
# X4a-4 — explicit Animation head is honored verbatim
# ---------------------------------------------------------------------------


def test_explicit_animation_head_honored_on_overlay() -> None:
    """plot(..., animate=Animation(head=False)) keeps head=False on a portrait."""
    # Build a single phase-portrait input so the per-kind head default would be
    # True — the bug would let that default leak in for a dict/bare-True spelling,
    # but an explicit Animation(head=False) must win.
    xy = np.linspace(0.0, 1.0, 8)
    layer = Layer(PlotKind.PHASE_PORTRAIT_2D, {"x": xy, "y": xy**2}, label="orbit")
    base = PlotSpec(kind=PlotKind.PHASE_PORTRAIT_2D, layers=[layer], title="p")

    user_anim = Animation(head=False, fps=12.0)
    result = plot(base, animate=user_anim)
    assert result.animation is not None
    assert result.animation.head is False  # not overridden to the portrait default
    assert result.animation.fps == 12.0
