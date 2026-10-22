"""plotly animation rendering (stream VIZ-ANIM).

Two paths, because plotly's built-in *frame* animation re-renders the whole WebGL
scene per frame (janky, and it fights the camera even with ``uirevision``):

- **HTML export — the real one (:func:`animated_html`):** the full attractor is
  drawn **once** (a faint, static, rotatable trace) and a ``requestAnimationFrame``
  loop (:data:`_REALTIME_JS`) advances a **comet** (a windowed trail + a head) by
  **streaming the comet's trace buffers in place with ``Plotly.extendTraces``**
  (append the next points, trim to a ``maxPoints`` window).  **Rotate-while-playing
  via pause-on-drag:** updating a 3-D (gl3d) trace forces plotly to replot the whole
  WebGL scene (no lightweight position-only update exists for gl3d — that is a
  ``scattergl`` 2-D-only optimisation), and replotting *while* the user is orbiting
  cancels the drag gesture.  So the loop **suspends the comet stream for the duration
  of a drag** — a capture-phase ``pointerdown`` (before plotly's own handler) sets a
  ``dragging`` flag and the loop does zero trace work until ``pointerup`` — leaving
  the gl3d scene free to orbit as smoothly as a static 3-D plot; on release the
  stream resumes.  A constant ``uirevision`` (plus a live-camera mirror during the
  drag) keeps the camera the user sets, with no snap-back.  It is also tiny (the
  curve is embedded once, in the static context trace).  Zero-extra-dependency
  "share the animation" path (no ffmpeg/kaleido).
- **Live figure (:func:`build_animated_figure`, notebooks):** a plotly ``frames``
  + play-button + slider figure (the only option for a returned ``Figure`` with no
  HTML wrapper to host a script).  Functional, but the HTML export is the smooth one.

Scope (this release): curve layers (``LINE`` / ``LINE3D`` / ``SCATTER``).
Annotation glyphs (``MARKERS``) remain static. The 3-D camera *spin* and the time clock are matplotlib-only for now
(here the user drives the camera directly); an animated spacetime image renders
through matplotlib.  A spec with no animatable curve layer falls back to a static
figure.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._visibility import listing_dir
from ...spec import PlotKind, PlotSpec

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["animated_html", "build_animated_figure", "playback_seconds"]

__dir__ = listing_dir(__all__)


def playback_seconds(anim: Any, n_samples: int) -> float:
    """Delegate to :meth:`tsdynamics.viz.spec.Animation.playback_seconds` [M41].

    Kept as a one-line adapter so this backend's call sites stay short; the
    algebra lives in exactly one place now, shared with the three.js exporter.
    """
    return float(anim.playback_seconds(int(n_samples)))


#: The real-time driver: a ``requestAnimationFrame`` loop that advances a comet by
#: **streaming its trace buffers in place with ``Plotly.extendTraces``** (append the
#: next points, trim to a ``maxPoints`` sliding window; the single-point head uses
#: ``maxPoints`` 1; the window resets to the start once per loop via ``restyle``).
#:
#: **Pause-on-drag — the rotate-while-playing fix.**  Any update to a *3-D* (gl3d /
#: ``Scatter3d``) trace forces plotly to **replot the whole WebGL scene** (its
#: ``editType`` for trace data is ``calc``/``plot``; there is no lightweight
#: position-only update for gl3d — the in-place ``scattergl`` batch update is 2-D
#: only).  Doing that every frame *while* the user is mid-orbit cancels the drag
#: gesture (and pegs the thread), which is why the 3-D animation could not be
#: rotated.  So the loop **fully suspends the comet stream for the duration of a
#: drag**: a capture-phase ``pointerdown`` (fired *before* plotly's own handler, so
#: not one ``extendTraces`` lands after the gesture starts) sets ``dragging`` and the
#: loop does **zero** trace work until the matching ``pointerup`` — leaving the gl3d
#: scene free to orbit as smoothly as a static 3-D plot.  On release the stream
#: resumes from where it paused.  To avoid a snap-back on that first resumed redraw
#: (a gl3d redraw restores the camera from ``gd.layout`` under ``uirevision``, but a
#: drag only *commits* its camera on release), the live camera is mirrored into
#: ``gd.layout.scene.camera`` continuously via ``plotly_relayouting``.  2-D plots
#: have no orbit gesture and ``scattergl`` updates cheaply, so the pause is a no-op
#: there.  The full curve lives in the static context trace (never touched).
#: ``{plot_id}`` is substituted by plotly with the graph-div id; ``__…__`` tokens are
#: filled below.
_REALTIME_JS = """
(function() {
  var gd = document.getElementById('{plot_id}');
  if (!gd || !window.Plotly) { console.warn('tsd-anim: no graph div / Plotly'); return; }
  var LAYERS = __LAYERS__, WINDOW = __WINDOW__, N = __N__;
  var DURATION = __DURATION_MS__, REVERSE_DURATION = __REVERSE_DURATION_MS__, LOOP = __LOOP__;
  var IS3D = __IS3D__, PERSISTENT = __PERSISTENT__;
  function asArray(v) {
    if (v == null) { return []; }
    if (typeof v.slice === 'function' && typeof v.length === 'number') { return v; }
    if (v.bdata != null && typeof Plotly.dataArray === 'function') {
      try { return Plotly.dataArray(v); } catch (e) { /* decoded fallback below */ }
    }
    return Array.from(v);
  }
  var decoded = gd._fullData || gd.data;
  var full = LAYERS.map(function(m) {
    var d = decoded[m.ctx] || gd.data[m.ctx], raw = gd.data[m.ctx];
    var f = {x: asArray(d.x || raw.x), y: asArray(d.y || raw.y)};
    if (IS3D) { f.z = asArray(d.z || raw.z); }
    if (m.c) { f.c = asArray((d.marker || raw.marker).color); }
    if (m.size) { f.size = asArray((d.marker || raw.marker).size); }
    m.n = Math.min(m.n, f.x.length, f.y.length, IS3D ? f.z.length : m.n);
    m.last = 0;
    m.window = PERSISTENT ? m.n : Math.max(1, Math.round(WINDOW * (m.n - 1) / Math.max(1, N - 1))) + 1;
    return f;
  });
  if (getComputedStyle(gd).position === 'static') { gd.style.position = 'relative'; }
  var bar = document.createElement('div');
  bar.setAttribute('role', 'group'); bar.setAttribute('aria-label', 'Animation controls');
  bar.style.cssText = 'position:absolute;left:10px;bottom:10px;z-index:10;display:flex;'
    + 'gap:6px;align-items:center;font:12px system-ui,sans-serif;color:#888;user-select:none;';
  function mkBtn(txt, label) {
    var b = document.createElement('button');
    b.textContent = txt; b.type = 'button'; b.setAttribute('aria-label', label); b.title = label;
    b.style.cssText = 'cursor:pointer;border:1px solid #8888;border-radius:6px;'
      + 'background:rgba(127,127,127,.12);color:inherit;width:30px;height:26px;'
      + 'font-size:13px;line-height:1;padding:0;';
    return b;
  }
  var playBtn = mkBtn('❚❚', 'Pause animation'), restartBtn = mkBtn('↺', 'Restart animation');
  var readout = document.createElement('span'); readout.textContent = '0%';
  bar.appendChild(playBtn); bar.appendChild(restartBtn); bar.appendChild(readout); gd.appendChild(bar);
  var dragging = false, dead = false, busy = false, wheelTimer = null, rafId = null;
  var userPaused = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var position = 0, direction = 1, lastTime = null, reset = false, generation = 0;
  function showPlayState() {
    playBtn.textContent = userPaused ? '▶' : '❚❚';
    playBtn.setAttribute('aria-label', userPaused ? 'Play animation' : 'Pause animation');
  }
  showPlayState();
  function endDrag() { dragging = false; lastTime = null; }
  function beginDrag(e) { if (!bar.contains(e.target)) { dragging = true; lastTime = null; } }
  function wheel() {
    dragging = true; lastTime = null;
    if (wheelTimer) { clearTimeout(wheelTimer); }
    wheelTimer = setTimeout(endDrag, 250);
  }
  function mirrorCam(e) {
    if (e && e['scene.camera']) {
      if (!gd.layout.scene) { gd.layout.scene = {}; }
      gd.layout.scene.camera = e['scene.camera'];
    }
  }
  function relayout(e) { mirrorCam(e); endDrag(); }
  gd.addEventListener('pointerdown', beginDrag, true); gd.addEventListener('wheel', wheel, true);
  window.addEventListener('pointerup', endDrag, true); window.addEventListener('pointercancel', endDrag, true);
  if (typeof gd.on === 'function') {
    gd.on('plotly_relayouting', mirrorCam); gd.on('plotly_relayout', relayout);
  }
  function cleanup() {
    gd.removeEventListener('pointerdown', beginDrag, true); gd.removeEventListener('wheel', wheel, true);
    window.removeEventListener('pointerup', endDrag, true); window.removeEventListener('pointercancel', endDrag, true);
    if (typeof gd.removeListener === 'function') {
      gd.removeListener('plotly_relayouting', mirrorCam); gd.removeListener('plotly_relayout', relayout);
    }
    if (wheelTimer) { clearTimeout(wheelTimer); }
  }
  function schedule() { if (rafId === null && !dead) { rafId = requestAnimationFrame(frame); } }
  function stop() {
    dead = true; userPaused = true; showPlayState();
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
  }
  function failed(e) { console.error('tsd-anim: trace update failed', e); stop(); }
  function endpoint() {
    if (direction > 0 && REVERSE_DURATION > 0) { direction = -1; return; }
    else if (LOOP) { direction = 1; position = 0; reset = true; }
    else { stop(); }
    lastTime = null;
  }
  function indices(lo, hi, reverse) {
    var out = [];
    if (reverse) { for (var j = lo; j >= hi; j--) { out.push(j); } }
    else { for (var k = lo; k <= hi; k++) { out.push(k); } }
    return out;
  }
  function paint(target, atEnd) {
    var tasks = [], ticket = generation, resetting = reset;
    reset = false; busy = true;
    try {
      LAYERS.forEach(function(m, l) {
        var f = full[l], j = Math.min(m.n - 1, Math.max(0, Math.round(target * (m.n - 1))));
        if (m.n < 1 || (!resetting && j === m.last)) { return; }
        var reverse = j < m.last;
        var lo = reverse ? Math.min(m.last - 1, j + m.window - 1) : Math.max(m.last + 1, j - m.window + 1);
        var ix = resetting ? [j] : indices(lo, j, reverse);
        var update = {};
        ['x', 'y', 'z'].forEach(function(key) {
          if (f[key]) { update[key] = [ix.map(function(k) { return f[key][k]; })]; }
        });
        if (m.c) { update['marker.color'] = [ix.map(function(k) { return f.c[k]; })]; }
        if (m.size) { update['marker.size'] = [ix.map(function(k) { return f.size[k]; })]; }
        // One in-flight batch at a time; a slow WebGL redraw cannot queue a request flood.
        tasks.push(resetting ? Plotly.restyle(gd, update, [m.comet])
                             : Plotly.extendTraces(gd, update, [m.comet], m.window));
        if (m.head >= 0) {
          var point = {};
          ['x', 'y', 'z'].forEach(function(key) { if (f[key]) { point[key] = [[f[key][j]]]; } });
          if (m.c) { point['marker.color'] = f.c[j]; }
          tasks.push(Plotly.restyle(gd, point, [m.head]));
        }
        m.last = j;
      });
    } catch (e) { busy = false; failed(e); return; }
    Promise.all(tasks).then(function() {
      busy = false;
      if (ticket !== generation) { return; }
      readout.textContent = Math.round(100 * target) + '%';
      if (atEnd) { endpoint(); }
    }).catch(function(e) { busy = false; failed(e); });
  }
  function frame(now) {
    rafId = null;
    if (dead) { return; }
    if (gd.isConnected === false) { stop(); cleanup(); return; }
    schedule();
    if (userPaused || dragging || document.hidden) { lastTime = null; return; }
    if (lastTime === null) { lastTime = now; }
    var delta = Math.max(0, now - lastTime); lastTime = now;
    var legDuration = direction > 0 ? DURATION : REVERSE_DURATION;
    position = Math.min(1, Math.max(0, position + direction * delta / legDuration));
    var atEnd = direction > 0 ? position >= 1 : position <= 0;
    if (!busy) { paint(position, atEnd); }
  }
  function restart() {
    generation++; direction = 1; position = 0; lastTime = null; reset = true;
    if (dead) { dead = false; userPaused = false; }
    showPlayState(); schedule();
  }
  playBtn.onclick = function() {
    if (dead) { restart(); return; }
    userPaused = !userPaused; lastTime = null; showPlayState();
  };
  restartBtn.onclick = restart;
  schedule();
})();
"""

#: Layer marks the plotly reveal animator drives (curves).
_CURVE_MARKS = frozenset({PlotKind.LINE, PlotKind.LINE3D, PlotKind.SCATTER})

#: A constant ``uirevision`` token — pinning it preserves the camera across frames.
_UIREVISION = "tsd-anim"


def _data_range(spec: PlotSpec, channel: str, *, logarithmic: bool = False) -> list[float] | None:
    """Return the padded min/max of ``channel`` across layers, for a fixed range."""
    lo, hi = np.inf, -np.inf
    for layer in spec.layers:
        arr = layer.data.get(channel)
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if logarithmic:
            a = np.log10(a[a > 0])
        if a.size:
            lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    if hi <= lo:
        return [lo - 1.0, hi + 1.0]
    pad = 0.05 * (hi - lo)
    return [lo - pad, hi + pad]


def _curve_layers(spec: PlotSpec) -> list[Any]:
    """Return the animatable curve layers of ``spec`` (in draw order)."""
    return [
        lyr
        for lyr in spec.layers
        if PlotKind(lyr.kind) in _CURVE_MARKS and np.asarray(lyr.data.get("y", [])).size
    ]


def _sample_count(layers: list[Any]) -> int:
    """Return the longest curve length among ``layers``."""
    n = 0
    for lyr in layers:
        arr = lyr.data.get("x", lyr.data.get("y"))
        if arr is not None:
            n = max(n, int(np.asarray(arr).shape[0]))
    return max(n, 1)


def _traces(spec: PlotSpec, layers: list[Any], *, three_d: bool) -> list[Any]:
    """Use the static renderer's mark/style rules for animation as well."""
    subset = dataclasses.replace(spec, layers=layers, animation=None)
    if three_d:
        from ._threed import build_3d_traces

        return build_3d_traces(subset)
    from ._core import build_2d_traces

    return build_2d_traces(subset)


def _curve_templates(spec: PlotSpec, curves: list[Any], *, three_d: bool) -> list[Any]:
    templates = []
    palette = spec.resolved_theme.palette
    for i, layer in enumerate(curves):
        template = _traces(spec, [layer], three_d=three_d)[0].to_plotly_json()
        if palette:
            color = palette[i % len(palette)]
            for key in ("line", "marker"):
                template.setdefault(key, {}).setdefault("color", color)
        c = layer.data.get("c")
        if c is not None:
            values = np.asarray(c, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                marker = template.setdefault("marker", {})
                marker.setdefault("cmin", float(finite.min()))
                marker.setdefault("cmax", float(finite.max()))
        templates.append(template)
    return templates


def _static_traces(spec: PlotSpec, curves: list[Any], *, three_d: bool) -> list[Any]:
    ids = {id(layer) for layer in curves}
    return _traces(spec, [layer for layer in spec.layers if id(layer) not in ids], three_d=three_d)


def build_animated_figure(spec: PlotSpec) -> go.Figure:
    """Build a camera-locked plotly figure with ``frames`` + play / slider."""
    import plotly.graph_objects as go

    anim = spec.animation
    assert anim is not None
    three_d = spec.is_three_d
    curves = _curve_layers(spec)
    if not curves:
        # Nothing to reveal — hand back a static figure (the final frame).
        from ._core import render as _static_render

        # ``dataclasses.replace`` clones the spec by reference; the old
        # ``from_dict(to_dict())`` idiom round-tripped every float array through a
        # dict just to drop one field.
        static = dataclasses.replace(spec, animation=None)
        return _static_render(static)

    n = _sample_count(curves)
    heads = anim.head_indices(n)
    dt = spec.meta.get("dt") if isinstance(spec.meta, dict) else None
    try:
        dt_f = float(dt) if dt is not None and float(dt) > 0 else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        dt_f = None
    tail = anim.tail_samples(dt_f)
    arrays = [_curve_arrays(lyr) for lyr in curves]
    if three_d:
        for arr in arrays:
            arr.setdefault("z", np.zeros_like(arr["x"]))
    templates = _curve_templates(spec, curves, three_d=three_d)

    # Base traces: a faint full-curve context (only when windowed) + the comet at
    # frame 0.  Record the comet trace indices so frames update *only* those (the
    # context stays put → small frames, and the camera/uirevision is undisturbed).
    base: list[Any] = _static_traces(spec, curves, three_d=three_d)
    comet_idx: list[int] = []
    for arr, layer, template in zip(arrays, curves, templates, strict=True):
        if tail is not None and anim.backdrop and anim.backdrop_alpha > 0:
            base.append(_context_trace(template, anim, three_d=three_d))
        for trace in _comet_traces(
            arr, layer, heads[0], tail, anim, three_d=three_d, template=template, n_clock=n
        ):
            comet_idx.append(len(base))
            base.append(trace)

    frames = [
        go.Frame(
            data=[
                t
                for arr, layer, template in zip(arrays, curves, templates, strict=True)
                for t in _comet_traces(
                    arr, layer, idx, tail, anim, three_d=three_d, template=template, n_clock=n
                )
            ],
            traces=comet_idx,
            name=str(k),
        )
        for k, idx in enumerate(heads)
    ]

    fig = go.Figure(data=base, frames=frames)
    _apply_layout(fig, spec, anim, three_d=three_d)
    return fig


def _curve_arrays(layer: Any) -> dict[str, np.ndarray]:
    """Extract the ``x``/``y``/``z`` arrays of a curve layer as floats."""
    out = {
        "x": np.asarray(layer.data["x"], dtype=float),
        "y": np.asarray(layer.data["y"], dtype=float),
    }
    if "z" in layer.data:
        out["z"] = np.asarray(layer.data["z"], dtype=float)
    for key in ("c", "size"):
        if key in layer.data:
            out[key] = np.asarray(layer.data[key], dtype=float)
    return out


def _context_trace(template: dict[str, Any], anim: Any, *, three_d: bool) -> Any:
    """Build the full curve drawn once, faintly — the static rotatable backdrop."""
    import plotly.graph_objects as go

    payload = dict(template)
    payload.update(
        opacity=anim.backdrop_alpha if anim.backdrop and anim.trail_kind is not None else 0.0,
        showlegend=False,
        hoverinfo="skip",
    )
    payload["marker"] = {**payload.get("marker", {}), "showscale": False}
    return (go.Scatter3d if three_d else go.Scatter)(**payload)


def _comet_traces(
    arr: dict[str, np.ndarray],
    layer: Any,
    head: int,
    tail: int | None,
    anim: Any,
    *,
    three_d: bool,
    template: dict[str, Any],
    n_clock: int,
) -> list[Any]:
    """Build the animated comet for one layer at sample ``head`` (windowed trail + head)."""
    import plotly.graph_objects as go

    n_local = len(arr["x"])
    ratio = (n_local - 1) / max(1, n_clock - 1)
    head = min(n_local - 1, max(0, round(head * ratio)))
    tail = None if tail is None else max(1, round(tail * ratio))
    lo = 0 if tail is None else max(0, head - tail)
    sl = slice(lo, head + 1)
    trace_type = go.Scatter3d if three_d else go.Scatter
    payload = dict(template)
    for key in ("x", "y", "z"):
        if key in arr:
            payload[key] = arr[key][sl].tolist()
    marker = dict(payload.get("marker", {}))
    if "c" in arr:
        marker["color"] = arr["c"][sl].tolist()
    if "size" in arr:
        marker["size"] = arr["size"][sl].tolist()
    payload["marker"] = marker
    out = [trace_type(**payload)]
    if anim.head:
        from ._core import _plotly_colour

        color = _plotly_colour(
            anim.head_color or layer.style.get("color") or template.get("line", {}).get("color")
        )
        head_marker = {**marker, "size": anim.head_size, "color": color, "showscale": False}
        if anim.head_color is None and "c" in arr:
            head_marker["color"] = float(arr["c"][head])
        point = {key: [arr[key][head]] for key in ("x", "y", "z") if key in arr}
        out.append(trace_type(**point, mode="markers", marker=head_marker, showlegend=False))
    return out


def _apply_layout(fig: go.Figure, spec: PlotSpec, anim: Any, *, three_d: bool) -> None:
    """Camera-locked layout: fixed ranges, play / pause, slider, real transitions."""
    frame_ms = 1000.0 / float(anim.fps) if anim.fps > 0 else 40.0
    # 2-D tweens the head glide between frames; 3-D has no trace tween (rely on the
    # dense frame schedule), so its transition stays instantaneous.
    trans_ms = 0.0 if three_d else frame_ms
    transition = {"duration": trans_ms, "easing": "linear"}
    play_args = {
        "frame": {"duration": frame_ms, "redraw": three_d},
        "fromcurrent": True,
        "mode": "immediate",
        "transition": transition,
    }
    play = {
        "type": "buttons",
        "showactive": False,
        "x": 0.05,
        "y": 0.05,
        "xanchor": "right",
        "yanchor": "top",
        "buttons": [
            {"label": "▶ play", "method": "animate", "args": [None, play_args]},
            {
                "label": "❚❚",
                "method": "animate",
                "args": [
                    [None],
                    {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                ],
            },
        ],
    }
    slider = {
        "active": 0,
        "x": 0.1,
        "len": 0.85,
        "steps": [
            {
                "label": "",
                "method": "animate",
                "args": [
                    [f.name],
                    {
                        "frame": {"duration": 0, "redraw": three_d},
                        "mode": "immediate",
                        "transition": transition,
                    },
                ],
            }
            for f in fig.frames
        ],
    }
    layout: dict[str, Any] = {
        "updatemenus": [play],
        "sliders": [slider],
        "uirevision": _UIREVISION,  # preserve camera / zoom across frames
        "margin": {"t": 40, "r": 10, "b": 10, "l": 10},
    }

    layout = {**_fixed_layout(spec, three_d=three_d), **layout}
    fig.update_layout(**layout)


def animated_html(
    spec: PlotSpec,
    *,
    path: str | os.PathLike[str] | None = None,
    html: bool = False,
    full_html: bool | None = None,
    include_plotlyjs: str | bool = "cdn",
) -> str | os.PathLike[str]:
    """Export a smooth, **rotatable** real-time animation as HTML.

    Draws the full attractor once (faint, static) + a comet (windowed trail + head)
    per curve layer, then attaches a ``requestAnimationFrame`` loop
    (:data:`_REALTIME_JS`) that streams the comet in place with
    ``Plotly.extendTraces``.  For a 3-D plot the comet stream is **suspended for the
    duration of a mouse drag** (a gl3d trace update forces a full WebGL replot that
    would otherwise cancel the orbit gesture), so the attractor rotates as smoothly
    as a static 3-D plot and the animation resumes on release; ``uirevision`` plus a
    live-camera mirror keep the pose with no snap-back.  A minimal play/pause +
    restart overlay makes it obviously alive and controllable.  Returns the HTML
    string (``html=True``) or the written path (``path=``).
    """
    import plotly.graph_objects as go

    from ._html import to_html as _to_html
    from ._html import write_html as _write_html

    anim = spec.animation
    assert anim is not None
    three_d = spec.is_three_d
    curves = _curve_layers(spec)
    if not curves:  # nothing to animate — write the static figure
        # ``dataclasses.replace`` clones the spec by reference; the old
        # ``from_dict(to_dict())`` idiom round-tripped every float array through a
        # dict just to drop one field.
        static = dataclasses.replace(spec, animation=None)
        if path is not None:
            return _write_html(
                static, path, full_html=full_html is not False, include_plotlyjs=include_plotlyjs
            )
        return _to_html(static, full_html=bool(full_html), include_plotlyjs=include_plotlyjs)

    arrays = [_curve_arrays(lyr) for lyr in curves]
    if three_d:
        for arr in arrays:
            arr.setdefault("z", np.zeros_like(arr["x"]))
    templates = _curve_templates(spec, curves, three_d=three_d)
    n = _sample_count(curves)
    dt = spec.meta.get("dt") if isinstance(spec.meta, dict) else None
    try:
        dt_f = float(dt) if dt is not None and float(dt) > 0 else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        dt_f = None
    tail = anim.tail_samples(dt_f)
    window = tail if tail is not None else n

    # One faint static context trace (the full curve, read by the JS loop) + a comet
    # (trail + head) per layer.  ``layers_meta`` tells the loop which trace is which.
    base: list[Any] = _static_traces(spec, curves, three_d=three_d)
    layers_meta: list[dict[str, Any]] = []
    for arr, layer, template in zip(arrays, curves, templates, strict=True):
        meta: dict[str, Any] = {
            "ctx": len(base),
            "n": len(arr["x"]),
            "c": "c" in arr,
            "size": "size" in arr,
        }
        base.append(_context_trace(template, anim, three_d=three_d))
        comet = _comet_traces(
            arr, layer, 0, window, anim, three_d=three_d, template=template, n_clock=n
        )
        meta["comet"] = len(base)
        base.append(comet[0])
        meta["head"] = -1
        if anim.head and len(comet) > 1:
            meta["head"] = len(base)
            base.append(comet[1])
        layers_meta.append(meta)

    fig = go.Figure(data=base)
    _realtime_layout(fig, spec, three_d=three_d)

    # Elapsed active time determines progress independently of display refresh.
    duration = playback_seconds(anim, n)
    js = (
        _REALTIME_JS.replace("__LAYERS__", json.dumps(layers_meta))
        .replace("__WINDOW__", str(int(window)))
        .replace("__DURATION_MS__", repr(float(duration * 1000.0)))
        .replace("__REVERSE_DURATION_MS__", repr(float(anim._reverse_seconds(n) * 1000.0)))
        .replace("__PERSISTENT__", "true" if tail is None else "false")
        .replace("__IS3D__", "true" if three_d else "false")
        .replace("__N__", str(int(n)))
        .replace("__LOOP__", "true" if anim.loop else "false")
    )
    if path is not None:
        return _write_html(
            fig,
            path,
            full_html=full_html is not False,
            include_plotlyjs=include_plotlyjs,
            post_script=js,
        )
    return _to_html(
        fig, full_html=bool(full_html), include_plotlyjs=include_plotlyjs, post_script=js
    )


def _fixed_layout(spec: PlotSpec, *, three_d: bool) -> dict[str, Any]:
    """Keep the static plot's axes, annotations, legend and theme during playback."""
    if three_d:
        from ._threed import render_3d

        static = dataclasses.replace(spec, layers=[], animation=None)
        layout = dict(render_3d(static).layout.to_plotly_json())
        layout.pop("template", None)
        axes = layout["scene"]
        axes["uirevision"] = _UIREVISION
    else:
        from ._core import _build_2d_layout

        layout = _build_2d_layout(spec)
        axes = layout
    for channel in ("x", "y", "z") if three_d else ("x", "y"):
        key = channel + "axis"
        axis = axes.setdefault(key, {})
        if "range" not in axis:
            rng = _data_range(spec, channel, logarithmic=axis.get("type") == "log")
            if rng is not None:
                axis["range"] = rng
    layout.update(uirevision=_UIREVISION, margin={"t": 40, "r": 10, "b": 10, "l": 10})
    return layout


def _realtime_layout(fig: go.Figure, spec: PlotSpec, *, three_d: bool) -> None:
    """Apply the same fixed presentation to HTML and returned-figure animations."""
    fig.update_layout(**_fixed_layout(spec, three_d=three_d))
