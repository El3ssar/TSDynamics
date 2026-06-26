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
    assert ".slice(lo, hi + 1)" in js  # the per-frame comet slice the guard protects


# ---------------------------------------------------------------------------
# The rAF / comet / react wiring (the correct machinery) stays intact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("components", [None, ["x", "z"], "x"])
def test_realtime_js_keeps_the_raf_comet_react_wiring(components):
    """The smooth driver: a rAF loop advancing a comet via ``Plotly.react``."""
    pytest.importorskip("plotly")
    js = _html(components=components)
    assert "requestAnimationFrame" in js  # the frame-rate driver
    assert "tsd-anim" in js  # the diagnostic marker / starting log
    assert "Plotly.react" in js  # re-renders gl3d under constant uirevision
    assert "Plotly.restyle" not in js  # restyle does NOT redraw gl3d — must never appear
    assert "cometProto" in js  # the comet trace template
    assert "gd.data.slice()" in js  # fresh array each tick → react's immutable diff fires
    assert "uirevision" in js  # camera/zoom preserved across react calls


def test_realtime_html_3d_and_2d_both_carry_the_fix(tmp_path):
    """The saved ``.html`` (the user-facing path) carries the fix for 3-D and 2-D."""
    pytest.importorskip("plotly")
    tr = ts.Lorenz().integrate(final_time=20.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(3.0)
    for name, comps in [("orbit3d.html", None), ("orbit2d.html", ["x", "z"])]:
        out = tmp_path / name
        assert tr.to_plot_spec(components=comps, animate=True).save(str(out)) == str(out)
        text = out.read_text()
        assert "_fullData" in text  # the fix shipped in the written artifact
        assert "requestAnimationFrame" in text and "Plotly.react" in text
