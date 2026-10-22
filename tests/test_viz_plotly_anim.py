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

import json
import shutil
import subprocess

import numpy as np
import pytest

import tsdynamics as ts


def _html(*, components=None) -> str:
    """Render the plotly real-time animated HTML for a Lorenz orbit as a string."""
    from tsdynamics.viz.render.plotly._anim import animated_html

    tr = ts.systems.Lorenz().run(final_time=20.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(3.0)
    spec = tr.__plot_spec__(components=components, animate=True)
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
    # The executable clock tests below also feed decoded typed arrays through
    # actual updates, including heterogeneous curve lengths and scalar colours.


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
    tr = ts.systems.Lorenz().run(final_time=20.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(3.0)
    for name, comps in [("orbit3d.html", None), ("orbit2d.html", ["x", "z"])]:
        out = tmp_path / name
        assert tr.__plot_spec__(components=comps, animate=True).save(str(out)) == str(out)
        text = out.read_text()
        assert "_fullData" in text  # the fix shipped in the written artifact
        assert "requestAnimationFrame" in text and "Plotly.extendTraces" in text


@pytest.fixture
def plotly_clock_probe():
    from tsdynamics.viz.render.plotly._anim import _REALTIME_JS

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    harness = r"""
    function mount({lengths = [101], duration = 10, loop = false,
                    pingpong = false, reverseDuration = null, reduced = false, delayed = false} = {}) {
      class Element {
        constructor() { this.children = []; this.style = {}; this.attributes = {}; }
        appendChild(child) { this.children.push(child); }
        setAttribute(k, v) { this.attributes[k] = v; }
        contains(el) { return this === el || this.children.includes(el); }
      }
      const frames = new Map(), listeners = new Map(), pending = [];
      let nextFrame = 0, calls = 0;
      const gd = new Element(); gd.data = []; gd._fullData = []; gd.layout = {}; gd.isConnected = true;
      const layers = lengths.map(n => {
        const ctx = gd.data.length;
        const values = Array.from({length:n}, (_, i) => i);
        gd.data.push({x:{dtype:'f8',bdata:'encoded'}, y:{dtype:'f8',bdata:'encoded'},
                      z:{dtype:'f8',bdata:'encoded'}});
        gd._fullData[ctx] = {x:Float64Array.from(values), y:Float64Array.from(values),
                            z:Float64Array.from(values),
                            marker:{color:Float64Array.from(values)}};
        gd.data.push({x:[0],y:[0],z:[0],marker:{color:[0]}});
        gd.data.push({x:[0],y:[0],z:[0],marker:{color:0}});
        return {ctx,comet:ctx+1,head:ctx+2,n,c:true,size:false};
      });
      function change(update, indices, append, maxPoints) {
        calls++;
        indices.forEach((index, offset) => Object.entries(update).forEach(([key, values]) => {
          const path = key.split('.'), trace = gd.data[index];
          const parent = path.length === 2 ? (trace[path[0]] ||= {}) : trace;
          const prop = path[path.length-1];
          const value = Array.isArray(values) ? values[offset] : values;
          if (['x','y','z'].includes(key) && value.some(v => !Number.isFinite(v))) {
            throw Error('undefined/nonfinite animation coordinate');
          }
          parent[prop] = append ? [...parent[prop], ...value].slice(-maxPoints) : value;
        }));
        return delayed ? new Promise(resolve => pending.push(resolve)) : Promise.resolve();
      }
      const Plotly = {extendTraces:(...a) => change(a[1],a[2],true,a[3]),
                      restyle:(...a) => change(a[1],a[2],false)};
      const document = {hidden:false, getElementById:() => gd, createElement:() => new Element()};
      const window = {Plotly, matchMedia:() => ({matches:reduced}),
        addEventListener:(k, fn) => listeners.set('window:'+k, fn),
        removeEventListener:k => listeners.delete('window:'+k)};
      gd.addEventListener = (k, fn) => listeners.set('gd:'+k, fn);
      gd.removeEventListener = k => listeners.delete('gd:'+k);
      gd.on = (k, fn) => listeners.set(k, fn); gd.removeListener = k => listeners.delete(k);
      const requestFrame = fn => { frames.set(++nextFrame, fn); return nextFrame; };
      let script = CLOCK.replace('__LAYERS__', JSON.stringify(layers)).replace('__WINDOW__','10')
        .replace('__N__', String(Math.max(...lengths))).replace('__DURATION_MS__',String(duration*1000))
        .replace('__LOOP__',String(loop))
        .replace('__REVERSE_DURATION_MS__',String(pingpong ? (reverseDuration ?? duration)*1000 : 0))
        .replace('__IS3D__','false').replace('__PERSISTENT__','false');
      new Function('document','window','Plotly','requestAnimationFrame','cancelAnimationFrame',
        'getComputedStyle',script)(document,window,Plotly,requestFrame,id => frames.delete(id),
                                 () => ({position:'static'}));
      const [play,restart,readout] = gd.children[0].children;
      async function flush() { for(let i=0;i<4;i++) await Promise.resolve(); }
      return {play,restart,readout,
        heads:() => layers.map(m => gd.data[m.head].x[0]),
        colors:() => layers.map(m => gd.data[m.head].marker.color), calls:() => calls,
        resources:() => [frames.size,listeners.size],
        hidden:value => { document.hidden=value; }, detached:() => { gd.isConnected=false; },
        async settle() { while(pending.length) pending.shift()(); await flush(); },
        async frame(now) {
          const callbacks=[...frames.values()];frames.clear();
          for (const fn of callbacks) fn(now);
          await flush();
        }
      };
    }
    """

    def run(scenario, *, clock=None):
        script = (
            "const CLOCK = "
            + json.dumps(_REALTIME_JS if clock is None else clock)
            + ";\n"
            + harness
        )
        script += (
            "\n(async () => {\n"
            + scenario
            + "\n})().catch(e => {console.error(e);process.exitCode=1});"
        )
        result = subprocess.run(
            [node, "-"], input=script, text=True, capture_output=True, timeout=10, check=False
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    return run


def test_html_playback_uses_elapsed_time_with_unequal_lengths_at_every_refresh_rate(
    plotly_clock_probe,
):
    results = plotly_clock_probe("""
      const results=[];
      for (const hz of [30,60,120]) {
        const view=mount({lengths:[101,11,1]});await view.frame(0);
        for(let i=1;i<=5*hz;i++) await view.frame(i*1000/hz);
        const half=view.heads();
        for(let i=5*hz+1;i<=10*hz;i++) await view.frame(i*1000/hz);
        results.push([half,view.heads(),view.colors(),view.play.attributes['aria-label']]);
      }
      console.log(JSON.stringify(results));
    """)
    assert results == [[[50, 5, 0], [100, 10, 0], [100, 10, 0], "Play animation"]] * 3


def test_html_pingpong_uses_the_exported_reverse_frame_duration(plotly_clock_probe):
    result = plotly_clock_probe("""
      const v=mount({duration:1,pingpong:true,reverseDuration:0.5}), values=[];
      for(const t of [0,500,1000,1250,1500]) {await v.frame(t);values.push(v.heads()[0]);}
      console.log(JSON.stringify([values,v.play.attributes['aria-label']]));
    """)
    assert result == [[0, 50, 100, 50, 0], "Play animation"]


def test_html_pause_hidden_view_restart_and_cleanup_are_deterministic(plotly_clock_probe):
    result = plotly_clock_probe("""
      const v=mount({reduced:true});await v.frame(0);await v.frame(5000);
      const paused=v.heads();v.play.onclick();await v.frame(9000);await v.frame(14000);
      const half=v.heads();v.hidden(true);await v.frame(90000);v.hidden(false);
      await v.frame(100000);const resumed=v.heads();await v.frame(105000);
      const end=v.heads();v.restart.onclick();await v.frame(110000);const restart=v.heads();
      v.detached();await v.frame(110100);
      console.log(JSON.stringify([paused,half,resumed,end,restart,v.resources()]));
    """)
    assert result == [[0], [50], [50], [100], [0], [0, 0]]


def test_html_waits_for_slow_plotly_updates_instead_of_queueing_requests(plotly_clock_probe):
    result = plotly_clock_probe("""
      const v=mount({delayed:true});await v.frame(0);await v.frame(1000);
      const first=v.calls();for(let i=2;i<=8;i++) await v.frame(i*1000);
      const waiting=v.calls();await v.settle();await v.frame(9000);
      console.log(JSON.stringify([first,waiting,v.calls(),v.heads()]));
    """)
    assert result == [2, 2, 4, [90]]


def test_plotly_animation_preserves_styles_static_layers_and_axis_presentation():
    pytest.importorskip("plotly")
    line = ts.viz.draw(
        {"x": np.arange(5), "y": np.arange(5)},
        "line",
        color="crimson",
        alpha=0.4,
        linewidth=7,
        linestyle="dashed",
    )
    bars = ts.viz.draw({"x": np.array([1]), "y": np.array([2])}, "bars")
    plot = (line + bars).animate(n_frames=5).trail(("steps", 2), backdrop=False).head(False)
    plot.vline(2).ticks(x=[0, 2, 4]).limits(x=(0, 4))
    fig = plot.render(backend="plotly")
    assert any(trace.type == "bar" for trace in fig.data)
    assert fig.frames[-1].data[0].line.dash == "dash"
    assert fig.frames[-1].data[0].line.width == 7
    assert fig.frames[-1].data[0].opacity == 0.4
    assert fig.layout.xaxis.tickvals == (0, 2, 4)
    assert fig.layout.xaxis.range == (0, 4)
    assert len(fig.layout.shapes) == 1


def test_animated_point_cloud_remains_unconnected_and_colored():
    pytest.importorskip("plotly")
    plot = (
        ts.viz.draw(
            {"x": np.arange(5), "y": np.arange(5), "c": np.arange(5)},
            "points",
        )
        .animate(n_frames=3)
        .head(False)
    )
    fig = plot.render(backend="plotly")
    assert fig.frames[-1].data[0].mode == "markers"
    assert list(fig.frames[-1].data[0].marker.color) == [0, 1, 2, 3, 4]
    assert fig.frames[-1].data[0].marker.cmin == 0
    assert fig.frames[-1].data[0].marker.cmax == 4
