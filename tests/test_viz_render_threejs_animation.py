"""The three.js export as a *playable* artifact (contract §6.6, ruling A4).

``tests/test_viz_threejs.py`` pins the payload schema.  This file pins the part a
reader actually experiences: that the exported page **plays**, that the knobs the
user turned reach it, and that anything it cannot play is named out loud.

Verified in a real browser while these were written (headless Chrome, the page
served over ``http://127.0.0.1``): the module loads from the pinned three.js CDN,
a WebGL canvas is drawn (28 000 lit pixels), the progress readout advances
(7 % → 11 % over 3 s with ``requestAnimationFrame`` throttled by the hidden tab),
and ``fps=60`` versus ``fps=10`` advance the comet at 0.200 versus 0.038 % per
frame.  The browser cannot run in CI, so what is gated here is the *contract*
those runs confirmed: the exporter writes what the shipped loader reads.
"""

from __future__ import annotations

import re
import warnings

import pytest

import tsdynamics as ts
from tsdynamics.viz.render.caps import VisualizationDegraded
from tsdynamics.viz.render.threejs import loader_source
from tsdynamics.viz.spec import Animation


@pytest.fixture
def traj():
    return ts.systems.Lorenz().run(final_time=8.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(1.0)


def _animation_block(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VisualizationDegraded)
        payload = spec.render(backend="threejs").payload
    return payload["metadata"].get("animation")


# ---------------------------------------------------------------------------
# the exporter writes what the loader reads — derived from the loader, not listed
# ---------------------------------------------------------------------------


def test_the_exporter_writes_every_field_the_shipped_loader_reads(traj):
    """No ``metadata.animation`` field the reference loader consults may be missing.

    The expected set is **read out of the loader source**, so a loader that starts
    consulting a new field fails here instead of silently reading ``undefined``.
    """
    fields = set(re.findall(r"\banim\.([A-Za-z_][A-Za-z0-9_]*)", loader_source()))
    assert fields, "the loader no longer reads the animation block — check the regex"

    block = _animation_block(ts.plot(traj, animate=True))
    assert block is not None
    missing = sorted(fields - set(block))
    assert missing == [], f"the loader reads {missing}, which the exporter does not write"


def test_fps_reaches_the_page_as_a_playback_duration(traj):
    """``.animate(fps=...)`` changes how fast the browser plays the comet.

    The loader has no frame clock — it traverses the series in
    ``animation.duration`` seconds and reads ``fps`` nowhere — so the exporter
    inverts ``Animation``'s own relation (``frame_count = duration * fps``).
    Measured in a browser: ``fps=60`` advanced the head 0.200 % per frame against
    ``fps=10``'s 0.038 %.
    """
    fast = _animation_block(ts.plot(traj, animate=True).animate(fps=60.0))
    slow = _animation_block(ts.plot(traj, animate=True).animate(fps=10.0))
    assert fast["duration"] == pytest.approx(slow["duration"] / 6.0)
    assert fast["duration"] < slow["duration"]


def test_the_default_playback_speed_is_unchanged(traj):
    """At the defaults the derived duration IS the loader's own 12 s fallback.

    ``DEFAULT_FRAMES`` (360) at ``fps`` (30) is 12.0 s exactly, so deriving the
    duration cannot have changed the speed of any export that already existed.
    """
    block = _animation_block(ts.plot(traj, animate=True))
    assert block["fps"] == 30.0
    assert block["duration"] == pytest.approx(12.0)
    assert Animation.DEFAULT_FRAMES / Animation().fps == pytest.approx(12.0)


def test_an_explicit_duration_still_wins(traj):
    """``duration`` is the same quantity stated directly, so it is never overridden."""
    assert _animation_block(ts.plot(traj, animate=True).animate(duration=4.0))["duration"] == 4.0


def test_both_html_exporters_derive_the_same_playback_duration():
    """The two web exports are twins on purpose; this is the guard that keeps them so.

    three.js (``_lower._playback_seconds``) and plotly's real-time comet
    (``_anim.playback_seconds``) each pick a traversal time for a browser loop that
    has no frame clock.  They are duplicated rather than imported across backend
    packages, so the agreement is asserted here.  The single natural home is a
    method on ``Animation`` (``viz/spec.py``, another slot's file).
    """
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._anim import playback_seconds as plotly_seconds
    from tsdynamics.viz.render.threejs._lower import _playback_seconds as threejs_seconds

    for n in (2, 50, 101, 2701, 100_000):
        for anim in (
            Animation(),
            Animation(fps=60.0),
            Animation(fps=10.0),
            Animation(duration=4.0),
            Animation(n_frames=12),
        ):
            assert plotly_seconds(anim, n) == threejs_seconds(anim, n), (n, anim)


def test_fps_reaches_the_plotly_realtime_html(tmp_path, traj):
    """``.animate(fps=)`` changes the stride of the ``requestAnimationFrame`` comet.

    The loop advances ``STRIDE`` samples per browser frame, chosen so the whole
    series takes the playback duration — so a faster ``fps`` is a bigger stride.
    """
    pytest.importorskip("plotly")
    strides = {}
    for tag, fps in (("fast", 60.0), ("slow", 10.0)):
        out = tmp_path / f"{tag}.html"
        ts.plot(traj, animate=True).animate(fps=fps).save(str(out))
        strides[tag] = int(re.search(r"STRIDE\s*=\s*(\d+)", out.read_text(encoding="utf-8"))[1])
    assert strides["fast"] > strides["slow"], strides


def test_a_short_curve_is_no_longer_stretched_over_twelve_seconds():
    """A 101-point orbit plays in its own time, not padded to the fallback."""
    short = ts.systems.Lorenz().run(final_time=1.0, dt=0.01, ic=[1.0, 1.0, 1.0])
    block = _animation_block(ts.plot(short, animate=True))
    assert block["n_samples"] == 101
    assert block["duration"] == pytest.approx(101 / 30.0, rel=1e-6)


# ---------------------------------------------------------------------------
# what it cannot play, it names — ONE warning, never silence
# ---------------------------------------------------------------------------


def test_an_animated_composite_exports_statically_and_says_so(traj):
    """One scene, one draw range — a multi-panel movie has no three.js form at all.

    Before this, the top-level ``Animation`` on a composite was simply not read by
    ``_lower_composite``: the payload came out tiled and static with **no** warning,
    so ``animate=True`` vanished between the call and the file.
    """
    comp = ts.plot(ts.plot(traj, "phase_portrait"), ts.plot(traj, "psd"), layout="row")
    assert _animation_block(comp) is None  # a static composite: nothing to say

    animated = ts.plot(
        ts.plot(traj, "phase_portrait"), ts.plot(traj, "psd"), layout="row", animate=True
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = animated.render(backend="threejs").payload
    degraded = [w for w in caught if issubclass(w.category, VisualizationDegraded)]

    assert len(degraded) == 1, [str(w.message) for w in degraded]
    text = str(degraded[0].message)
    assert "animation" in text and ".mp4" in text  # what was dropped, and the way out
    assert payload["kind"] == "composite"  # ...and the layout itself still exports
    assert len(payload["panels"]) == 2
    assert "animation" not in payload["metadata"]


def test_a_points_only_animation_still_reveals_rather_than_warning(traj):
    """The pre-existing "nothing to reveal" degrade is unchanged by the composite one."""
    spec = ts.plot(traj, "phase_portrait", primitive="points3d")
    spec.animate(n_frames=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = spec.render(backend="threejs").payload
    # a points cloud IS revealable, so this must NOT warn — the guard is specific
    assert [w for w in caught if issubclass(w.category, VisualizationDegraded)] == []
    assert payload["metadata"]["animation"] is not None


# ---------------------------------------------------------------------------
# the page itself: self-contained, and carrying what the browser needs
# ---------------------------------------------------------------------------


def test_the_written_page_is_self_contained_and_playable(tmp_path, traj):
    """Everything the browser needs is in the file: payload, loader, import map, fallback.

    This is the artifact that was loaded in a real browser: it drew a WebGL canvas
    and advanced its progress readout with no same-origin request.
    """
    out = tmp_path / "attractor.html"
    ts.plot(traj, animate=True).trail(("time", 2.0)).head(size=8).save(str(out), backend="threejs")
    text = out.read_text(encoding="utf-8")

    assert out.stat().st_size > 100_000  # the payload and the loader are both inline
    assert "importmap" in text and "three@" in text  # the pinned three.js build
    assert "renderThreejsPayload" in text  # the loader, inlined
    assert '"animation"' in text  # the directive the loader plays
    assert "<noscript>" in text  # ...and a reader with no JS still sees the attractor
    # The loader is INLINED, not linked: its private machinery is in the page, and
    # the page loads no sibling script.  (A ``fetch``-a-sibling page needs a web
    # server, which is the one thing a portable artifact cannot assume.)
    assert "function installAnimation" in text
    assert "<script src=" not in text
