"""Regression lock for the plotly real-time animated-HTML export (issue #464).

The bug: ``spec.save("x.html")`` on an animated 3-D (and 2-D) spec rendered a
**static** curve — the ``requestAnimationFrame`` comet loop threw on frame 0 and
died.  Root cause: the driver cached the full curve from ``gd.data[i].x``, but
plotly 6 serialises a numpy array into a **base64 typed-array spec** object
(``{dtype, bdata}``) and stores *that* in ``gd.data[i].x`` — which has no
``.slice`` — so the per-frame ``f.x.slice(lo, hi + 1)`` raised
``TypeError: f.x.slice is not a function`` and the loop stopped.

The fix reads the curve from plotly's **decoded** data (``gd._fullData[i].x``,
a ``Float64Array``) with a fallback to ``gd.data[i].x``, and normalises each axis
array up front (``asArray``) so the per-frame slice / head-index reads always work
regardless of plotly's storage format.

These tests are **browser-free** (CI has no browser): they assert on the
generated HTML/JS *structure*.  The in-browser behaviour (``VERDICT: ANIMATING``
in a headless-Chrome screenshot-diff harness) is verified out of band; here we
lock the JS contract so a future refactor reverting to the raw ``gd.data`` read —
or dropping the rAF/comet wiring — fails fast.
"""

from __future__ import annotations

import pytest

import tsdynamics as ts


def _html(*, components=None) -> str:
    """Render the plotly real-time animated HTML for a Lorenz orbit as a string."""
    from tsdynamics.viz.render.plotly._anim import animated_html

    tr = ts.Lorenz().integrate(final_time=20.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(3.0)
    spec = tr.to_plot_spec(components=components, animate=True)
    return animated_html(spec, html=True, full_html=False)


# ---------------------------------------------------------------------------
# The fix: the full curve is read from plotly's DECODED data, not raw gd.data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("components", [None, ["x", "z"], "x"])
def test_realtime_js_reads_the_full_curve_from_fulldata(components):
    """The driver caches the curve from ``gd._fullData`` (decoded Float64Array).

    This is the #464 regression lock: plotly stashes a base64 ``{dtype, bdata}``
    typed-array *spec* (no ``.slice``) in ``gd.data[i].x``, while
    ``gd._fullData[i].x`` is the decoded array.  Reverting to the raw ``gd.data``
    read reintroduces the frame-0 ``.slice is not a function`` crash.
    """
    pytest.importorskip("plotly")
    js = _html(components=components)
    assert "_fullData" in js  # read the decoded curve, not the {bdata} spec
    # The exact buggy cache line must NOT come back.
    assert "var full = LAYERS.map(function(m) { var c = gd.data[m.ctx]; return {x:c.x" not in js


@pytest.mark.parametrize("components", [None, ["x", "z"], "x"])
def test_realtime_js_normalises_axis_arrays_before_slicing(components):
    """The curve axes are realised once (``asArray``) so per-frame slices never throw.

    The guard normalises each axis (Float64Array / Array / a stray ``{bdata}``
    spec) into a real array up front, so ``f.x.slice(lo, hi + 1)`` and the head
    index reads work regardless of plotly's storage format.
    """
    pytest.importorskip("plotly")
    js = _html(components=components)
    assert "asArray" in js  # the up-front normalisation guard
    assert "bdata" in js  # the guard explicitly handles plotly's base64 typed-array spec
    assert "Array.prototype.slice.call" in js  # plain-array per-frame slice (extendTraces-safe)


# ---------------------------------------------------------------------------
# The rAF / comet streaming wiring (Plotly.extendTraces — rotatable while playing)
# ---------------------------------------------------------------------------


def test_realtime_js_streams_the_comet_via_extendtraces():
    """The smooth driver advances the comet with ``Plotly.extendTraces``.

    ``extendTraces`` mutates the comet's trace buffers in place (a sliding window via
    ``maxPoints``) — the playback engine.  ``Plotly.react`` (which rebuilt the moving
    gl3d traces every frame and ate the orbit drag) must NOT be our per-frame driver.
    Rotation *while* playing is handled separately — by pausing the stream during a
    drag (see :func:`test_realtime_js_pauses_the_comet_stream_during_a_drag`), because
    a gl3d data update always replots the whole scene and cannot be done mid-orbit.
    Inspect the driver TEMPLATE (not the full HTML, whose bundled plotly.js naturally
    contains the string ``Plotly.react``).
    """
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._anim import _REALTIME_JS

    js = _REALTIME_JS
    assert "requestAnimationFrame" in js  # the frame-rate driver
    assert "tsd-anim" in js  # the diagnostic marker / starting log
    assert "Plotly.extendTraces" in js  # in-place buffer streaming (no scene rebuild)
    assert "Plotly.react(" not in js  # our driver must not CALL react (it rebuilds gl3d)


def test_realtime_js_pauses_the_comet_stream_during_a_drag():
    """Rotate-while-playing (3-D) is done by PAUSING the comet stream during a drag.

    A gl3d (``Scatter3d``) trace update forces plotly to replot the whole WebGL scene
    (its data ``editType`` is ``calc``/``plot``; there is no lightweight position-only
    update for gl3d), and replotting each frame *while* the user orbits cancels the
    drag gesture — which is why the 3-D animation could not be rotated.  The driver
    therefore suspends the comet stream for the duration of a drag: a capture-phase
    ``pointerdown`` sets a ``dragging`` flag and the rAF loop does zero trace work
    until ``pointerup``, leaving the gl3d scene free to orbit as smoothly as a static
    3-D plot; the stream resumes on release.  The live drag camera is still mirrored
    into ``gd.layout.scene.camera`` (via ``plotly_relayouting``) so there is no
    snap-back when the stream resumes.
    """
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._anim import _REALTIME_JS

    js = _REALTIME_JS
    assert "pointerdown" in js  # the capture-phase drag-start hook (fires before plotly)
    assert "dragging" in js  # the flag that suspends the comet stream mid-drag
    assert "pointerup" in js  # the drag-end hook that resumes the stream
    assert "plotly_relayouting" in js  # the live-drag camera is tracked …
    assert "scene.camera" in js  # … and mirrored into gd.layout.scene.camera (no snap-back)


def test_realtime_html_3d_and_2d_both_carry_the_fix(tmp_path):
    """The saved ``.html`` (the user-facing path) carries the fix for 3-D and 2-D."""
    pytest.importorskip("plotly")
    tr = ts.Lorenz().integrate(final_time=20.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(3.0)
    for name, comps in [("orbit3d.html", None), ("orbit2d.html", ["x", "z"])]:
        out = tmp_path / name
        assert tr.to_plot_spec(components=comps, animate=True).save(str(out)) == str(out)
        text = out.read_text()
        assert "_fullData" in text  # the fix shipped in the written artifact
        assert "requestAnimationFrame" in text and "Plotly.extendTraces" in text
