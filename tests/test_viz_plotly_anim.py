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
    ``maxPoints``) and never re-creates the gl3d traces or rebuilds the WebGL scene —
    so the attractor is rotatable WHILE it plays, with no click-to-rotate lag.
    ``Plotly.react`` (which rebuilt the moving gl3d traces every frame and ate the
    orbit drag) must NOT be our per-frame driver.  Inspect the driver TEMPLATE (not
    the full HTML, whose bundled plotly.js naturally contains the string
    ``Plotly.react``).
    """
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._anim import _REALTIME_JS

    js = _REALTIME_JS
    assert "requestAnimationFrame" in js  # the frame-rate driver
    assert "tsd-anim" in js  # the diagnostic marker / starting log
    assert "Plotly.extendTraces" in js  # in-place buffer streaming (no scene rebuild)
    assert "Plotly.react(" not in js  # our driver must not CALL react (it rebuilds gl3d)


def test_realtime_js_mirrors_the_live_camera_during_a_drag():
    """Rotate-while-it-plays WITHOUT snap-back: the live drag camera is mirrored.

    A gl3d comet redraw restores the camera from ``gd.layout`` (uirevision); a mouse
    orbit-drag only commits the new camera on release, so mid-drag each redraw used to
    snap the view back to the drag's start.  The driver mirrors the live
    ``plotly_relayouting`` camera into ``gd.layout.scene.camera`` so the next redraw
    keeps the in-progress rotation — the animation never stops and there is no pause
    hack (guards against regressing to the react-era ``!interacting`` workaround).
    """
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.plotly._anim import _REALTIME_JS

    js = _REALTIME_JS
    assert "plotly_relayouting" in js  # the live-drag event we track
    assert "scene.camera" in js  # mirrored into gd.layout.scene.camera
    assert "interacting" not in js  # no pause hack — the comet streams uninterrupted


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
