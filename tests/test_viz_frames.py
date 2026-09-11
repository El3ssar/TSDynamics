"""What a set of axes *means* — the frame vocabulary, its arity, and the refusals.

Three claims, all of them user-facing:

1. **A ``free`` frame overlays with anything.**  It is what
   :func:`tsdynamics.viz.draw` stamps on hand-built arrays, and a caller who
   opted out of the frame system made no coordinate claim to violate — refusing
   would punish precisely the person who used the escape hatch.
2. **A space knows its own arity**, so a transform declares *where* it draws and
   never *how many axes that is*.
3. **A refusal names the part that differs.**  Before v6 the space branch printed
   only ``space.value``, so a 2-D lattice refusing a 1-D profile read ``cannot
   overlay frame 'grid2' on frame 'grid2'`` — the same words on both sides of a
   "these differ" sentence.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidParameterError
from tsdynamics.viz._frames import (
    FRAME_SPACES,
    Frame,
    FrameSpace,
    check_overlay,
    space_arity,
)
from tsdynamics.viz.transforms import make_frame


def _traj(dim: int = 2, n: int = 60):
    from tsdynamics.data import Trajectory

    t = np.linspace(0.0, 10.0, n)
    y = np.column_stack([np.sin(t + k) for k in range(dim)])
    return Trajectory(t, y, system=None, meta={"system": "demo", "dt": float(t[1])})


# ---------------------------------------------------------------------------
# FrameSpace.FREE
# ---------------------------------------------------------------------------


def test_free_is_in_the_vocabulary_and_knows_it_is_free():
    assert FrameSpace.FREE in FRAME_SPACES
    assert FrameSpace.FREE == "free"
    assert Frame(FrameSpace.FREE, 2, ("", "")).is_free
    assert not Frame(FrameSpace.STATE2, 2, ("x", "y")).is_free


@pytest.mark.parametrize(
    "other",
    [
        Frame(FrameSpace.STATE2, 2, ("x", "y")),
        Frame(FrameSpace.TIME, 1, ("t",)),
        Frame(FrameSpace.STATE3, 3, ("x", "y", "z")),
        Frame(FrameSpace.GRID2, 2, ("i", "j")),
    ],
)
def test_a_free_frame_is_compatible_with_every_other_frame(other):
    """The escape hatch composes — in both directions, and at any arity."""
    free = Frame(FrameSpace.FREE, 2, ("", ""))
    assert free.compatible_with(other)
    assert other.compatible_with(free)


def test_a_free_frame_inherits_the_host_and_contributes_nothing():
    """A ``free`` part takes the host's labels; it has no coordinates to donate."""
    free = Frame(FrameSpace.FREE, 2, ("", ""))
    host = Frame(FrameSpace.STATE2, 2, ("x", "v"))
    assert host.merge(free) == host
    assert free.merge(host) == host
    assert free.merge(free).is_free


def test_hand_built_arrays_overlay_onto_a_real_plot():
    """The end-to-end reason ``FREE`` exists: *add my curve to your plot*."""
    orbit = ts.plot(_traj(), "phase_portrait")
    guide = ts.viz.draw({"x": np.linspace(-1, 1, 9), "y": np.zeros(9), "label": "y = 0"}, "line")
    merged = ts.plot(orbit, guide)
    assert len(merged.layers) == len(orbit.layers) + 1
    # ...and the merged figure keeps the host's coordinate meaning, not "free"
    assert merged.frame.space is FrameSpace.STATE2
    assert "y = 0" in [layer.label for layer in merged.layers]


# ---------------------------------------------------------------------------
# arity
# ---------------------------------------------------------------------------


def test_every_space_declares_its_arity():
    """A space with no arity would make ``ndim`` underivable for some transform."""
    for space in FRAME_SPACES:
        assert space_arity(space) >= 1


def test_make_frame_derives_the_axis_count_from_the_space():
    """``make_frame(space, labels)`` — the second argument is the labels, not a number."""
    assert make_frame(FrameSpace.STATE2, ("x", "v")).ndim == 2
    assert make_frame(FrameSpace.TIME, ("t", "x")).axes == ("t",)
    assert make_frame(FrameSpace.STATE3, ("x",)).axes == ("x", "", "")
    # the one case ``ndim=`` survives for: a shape that varies *within* one space
    assert make_frame(FrameSpace.GRID2, ("x",), 1).ndim == 1


def test_the_in_tree_transforms_agree_with_the_derived_arity():
    """33 of 35 already did; the two that did not were mis-declared frames."""
    for record in ts.viz.transforms.transforms():
        if len(record.frame) == len(record.ndim):
            derived = tuple(space_arity(space) for space in record.frame)
            assert record.ndim == derived, record.name


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------


def _spec(kind, labels):
    from tsdynamics.viz.spec import Axis, Plot

    axes = list(labels) + [None, None, None]
    return Plot(
        kind=kind,
        ndim=2,
        x=Axis(label=axes[0] or ""),
        y=Axis(label=axes[1] or ""),
    )


def test_a_different_space_refusal_names_both_spaces():
    a = ts.plot(_traj(), "phase_portrait")
    b = ts.plot(_traj(), "time_series")
    with pytest.raises(InvalidParameterError) as err:
        check_overlay([a, b])
    text = str(err.value)
    assert "state2" in text and "time" in text
    assert "layout='stack'" in text


def test_a_different_axis_count_refusal_says_so_rather_than_repeating_one_word():
    """The measured defect: ``cannot overlay frame 'grid2' on frame 'grid2'``."""
    lattice = Frame(FrameSpace.GRID2, 2, ("x", "y"))
    profile = Frame(FrameSpace.GRID2, 1, ("x",))
    a, b = _spec("image", ("x", "y")), _spec("image", ("x", "y"))
    a.frame, b.frame = lattice, profile
    with pytest.raises(InvalidParameterError) as err:
        check_overlay([a, b])
    text = str(err.value)
    assert "coordinate axes" in text
    # the two sides of a "these differ" sentence must not read identically
    left, _, right = text.partition(" vs ")
    assert left.rsplit(" ", 1)[-1] != right.split(" ")[0]


def test_an_axis_mismatch_refusal_names_the_offending_axis():
    a, b = _spec("phase_portrait_2d", ("x", "y")), _spec("phase_portrait_2d", ("x", "z"))
    with pytest.raises(InvalidParameterError) as err:
        check_overlay([a, b])
    text = str(err.value)
    assert "axis 2: z vs y" in text
    assert "components=" in text  # ...and the next move
