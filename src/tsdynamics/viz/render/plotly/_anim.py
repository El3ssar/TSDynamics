"""plotly animation rendering (stream VIZ-ANIM).

Two paths, because plotly's built-in *frame* animation re-renders the whole WebGL
scene per frame (janky, and it fights the camera even with ``uirevision``):

- **HTML export — the real one (:func:`animated_html`):** the full attractor is
  drawn **once** (a faint, static, rotatable trace) and a ``requestAnimationFrame``
  loop (:data:`_REALTIME_JS`) advances a **comet** (a windowed trail + a head) by
  **streaming the comet's trace buffers in place with ``Plotly.extendTraces``**
  (append the next points, trim to a ``maxPoints`` window).  Crucially
  ``extendTraces`` does **not** re-create the gl3d traces or rebuild the WebGL scene
  (it only pokes the existing GPU buffers), so — unlike ``Plotly.react``, which tears
  the moving traces down every frame and thereby ate an in-progress orbit drag — the
  attractor is **rotatable with the mouse WHILE it plays, with no click lag and
  without the animation ever stopping**.  A constant ``uirevision`` keeps the camera
  the user sets.  It is also tiny (the curve is embedded once, in the static context
  trace).  Zero-extra-dependency "share the animation" path (no ffmpeg/kaleido).
- **Live figure (:func:`build_animated_figure`, notebooks):** a plotly ``frames``
  + play-button + slider figure (the only option for a returned ``Figure`` with no
  HTML wrapper to host a script).  Functional, but the HTML export is the smooth one.

Scope (this release): curve layers (``LINE`` / ``LINE3D`` / ``SCATTER`` /
``MARKERS``).  The 3-D camera *spin* and the time clock are matplotlib-only for now
(here the user drives the camera directly); an animated spacetime image renders
through matplotlib.  A spec with no animatable curve layer falls back to a static
figure.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from ...spec import Axis, PlotKind, PlotSpec

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["animated_html", "build_animated_figure"]

#: The real-time driver: a ``requestAnimationFrame`` loop that advances a comet by
#: **streaming its trace buffers in place with ``Plotly.extendTraces``** (append the
#: next points, trim to a ``maxPoints`` sliding window; the single-point head uses
#: ``maxPoints`` 1; the window resets to the start once per loop via ``restyle``).
#: ``extendTraces`` redraws the gl3d comet WITHOUT re-creating the traces or
#: rebuilding the WebGL scene — so an in-progress mouse orbit-drag is never
#: interrupted (rotate WHILE it plays, no click lag, the stream never stops), and a
#: constant ``uirevision`` keeps the user's camera.  (``Plotly.react`` also redraws
#: gl3d but tears the moving traces down every frame; that per-frame scene rebuild
#: ate the drag and lagged the first rotation — so it is deliberately NOT used here.)
#: The full curve lives in the static context trace (never touched).  ``{plot_id}``
#: is substituted by plotly with the graph-div id; ``__…__`` tokens are filled below.
_REALTIME_JS = """
(function() {
  var gd = document.getElementById('{plot_id}');
  if (!gd || !window.Plotly) { console.warn('tsd-anim: no graph div / Plotly'); return; }
  var LAYERS = __LAYERS__, WINDOW = __WINDOW__, STRIDE = __STRIDE__;
  var IS3D = __IS3D__, N = __N__;
  // Cache the full curve arrays from the initial figure.
  // Read the curve from plotly's DECODED data (gd._fullData): plotly serialises a
  // numpy array into a base64 typed-array *spec* object ({dtype, bdata}) and stores
  // THAT in gd.data[i].x — which has no .slice — while gd._fullData[i].x is the
  // decoded Float64Array.  Normalise each axis once into a real array so the
  // per-frame .slice(lo, hi+1) (and the head index read) always works regardless of
  // plotly's storage format.  (Without this the loop throws on frame 0 and the
  // curve stays static — issue #464.)
  function asArray(v) {
    if (v == null) { return []; }
    // A real Array / typed array (Float64Array …) already slices.
    if (typeof v.slice === 'function' && typeof v.length === 'number') { return v; }
    // A plotly base64 typed-array spec ({dtype, bdata}) or any iterable → realise it.
    if (v.bdata != null && typeof Plotly.dataArray === 'function') {
      try { return Plotly.dataArray(v); } catch (e) { /* fall through */ }
    }
    var a = Array.from(v);
    if (a.length) { return a; }
    // Never hand back an unsliceable {bdata} spec — an empty realisation of one
    // becomes [] so no downstream .slice / index can throw.
    return v.bdata != null ? [] : v;
  }
  var decoded = gd._fullData || gd.data;
  var full = LAYERS.map(function(m) {
    // Prefer plotly's decoded data (Float64Array); fall back to the raw figure data.
    var d = (decoded && decoded[m.ctx] != null) ? decoded[m.ctx] : gd.data[m.ctx];
    var raw = gd.data[m.ctx];
    return {
      x: asArray(d.x != null ? d.x : raw.x),
      y: asArray(d.y != null ? d.y : raw.y),
      z: asArray(d.z != null ? d.z : raw.z),
    };
  });
  console.info('tsd-anim: starting', {N: N, window: WINDOW, stride: STRIDE, is3d: IS3D});

  // --- A minimal, always-visible play/pause + restart control overlay --------
  // (so the animation is obviously alive and controllable without devtools).
  if (getComputedStyle(gd).position === 'static') { gd.style.position = 'relative'; }
  var bar = document.createElement('div');
  bar.style.cssText = 'position:absolute;left:10px;bottom:10px;z-index:10;display:flex;'
    + 'gap:6px;align-items:center;font:12px system-ui,sans-serif;color:#888;'
    + 'user-select:none;';
  function mkBtn(txt) {
    var b = document.createElement('button');
    b.textContent = txt;
    b.style.cssText = 'cursor:pointer;border:1px solid #8888;border-radius:6px;'
      + 'background:rgba(127,127,127,.12);color:inherit;width:30px;height:26px;'
      + 'font-size:13px;line-height:1;padding:0;';
    return b;
  }
  var playBtn = mkBtn('❚❚'), restartBtn = mkBtn('↺');
  var readout = document.createElement('span');
  bar.appendChild(playBtn); bar.appendChild(restartBtn); bar.appendChild(readout);
  gd.appendChild(bar);

  // Keep the comet animating WHILE the user orbits, with NO snap-back.  A gl3d comet
  // redraw restores the camera from gd.layout (uirevision); a mouse orbit-drag only
  // *commits* the new camera to gd.layout on release, so mid-drag each extendTraces
  // redraw was snapping the view back to where the drag began.  plotly reports the
  // live camera continuously through 'plotly_relayouting', so we mirror it into
  // gd.layout.scene.camera as the drag happens — the next redraw then keeps the live
  // pose.  Pure data mirror (we issue no relayout/redraw → no recursion, no flicker,
  // the animation never stops); on release uirevision keeps the committed pose.
  if (typeof gd.on === 'function') {
    var mirrorCam = function(e) {
      if (e && e['scene.camera']) {
        if (!gd.layout.scene) { gd.layout.scene = {}; }
        gd.layout.scene.camera = e['scene.camera'];
      }
    };
    gd.on('plotly_relayouting', mirrorCam);
    gd.on('plotly_relayout', mirrorCam);
  }

  // Stream the comet by MUTATING the trace buffers in place.  Plotly.extendTraces
  // appends the next points to the comet line (trimmed to a WINDOW-length sliding
  // window via maxPoints) and advances the single-point head (maxPoints 1).  Unlike
  // Plotly.react it does NOT re-create the gl3d traces or rebuild the WebGL scene —
  // it only pokes the existing trace GPU buffers — so an in-progress mouse
  // orbit-drag is never interrupted: you rotate WHILE it plays, with no click lag,
  // and the camera is left untouched (constant uirevision).  (react redraws gl3d
  // too, but tears the moving traces down every frame, and that per-frame scene
  // rebuild is what ate the drag and lagged the first rotation.)
  var i = 1, playing = true, raf = null;
  // A plain-Array slice (not a typed-array slice) so the appended chunk matches the
  // comet trace's own plain-array data — extendTraces rejects a type mismatch.
  function slc(a, lo, hi) { return Array.prototype.slice.call(a, lo, hi + 1); }
  function pushRange(lo, hi) {
    for (var l = 0; l < LAYERS.length; l++) {
      var m = LAYERS[l], f = full[l];
      var ext = IS3D
        ? {x: [slc(f.x, lo, hi)], y: [slc(f.y, lo, hi)], z: [slc(f.z, lo, hi)]}
        : {x: [slc(f.x, lo, hi)], y: [slc(f.y, lo, hi)]};
      try {
        Plotly.extendTraces(gd, ext, [m.comet], WINDOW);
        if (m.head >= 0) {
          var eh = IS3D
            ? {x: [[f.x[hi]]], y: [[f.y[hi]]], z: [[f.z[hi]]]}
            : {x: [[f.x[hi]]], y: [[f.y[hi]]]};
          Plotly.extendTraces(gd, eh, [m.head], 1);
        }
      } catch (e) {
        console.error('tsd-anim: extendTraces failed', e);
        playing = false; playBtn.textContent = '▶'; return false;
      }
    }
    return true;
  }
  function seedStart() {
    // Reset the comet + head to the first sample (one in-place redraw, once per
    // loop) so the windowed comet does not draw a line back across the attractor.
    for (var l = 0; l < LAYERS.length; l++) {
      var m = LAYERS[l], f = full[l];
      var s = IS3D ? {x: [[f.x[0]]], y: [[f.y[0]]], z: [[f.z[0]]]}
                   : {x: [[f.x[0]]], y: [[f.y[0]]]};
      Plotly.restyle(gd, s, [m.comet]);
      if (m.head >= 0) { Plotly.restyle(gd, s, [m.head]); }
    }
    i = 0;
  }
  function tick() {
    if (!playing) { raf = null; return; }
    var lo = i + 1, hi = Math.min(i + STRIDE, N - 1);
    if (lo <= hi && !pushRange(lo, hi)) { return; }
    readout.textContent = Math.round(100 * hi / (N - 1)) + '%';
    i = hi;
    if (i >= N - 1) { seedStart(); }
    raf = requestAnimationFrame(tick);
  }
  function start() { if (!raf) { raf = requestAnimationFrame(tick); } }
  playBtn.onclick = function() {
    playing = !playing;
    playBtn.textContent = playing ? '❚❚' : '▶';
    if (playing) { start(); }
  };
  restartBtn.onclick = function() { seedStart(); readout.textContent = '0%'; };
  start();
})();
"""

#: Layer marks the plotly reveal animator drives (curves).
_CURVE_MARKS = frozenset({PlotKind.LINE, PlotKind.LINE3D, PlotKind.SCATTER, PlotKind.MARKERS})

#: A constant ``uirevision`` token — pinning it preserves the camera across frames.
_UIREVISION = "tsd-anim"


def _data_range(spec: PlotSpec, channel: str) -> list[float] | None:
    """Return the padded min/max of ``channel`` across layers, for a fixed range."""
    lo, hi = np.inf, -np.inf
    for layer in spec.layers:
        arr = layer.data.get(channel)
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
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
    return [lyr for lyr in spec.layers if PlotKind(lyr.kind) in _CURVE_MARKS]


def _sample_count(layers: list[Any]) -> int:
    """Return the longest curve length among ``layers``."""
    n = 0
    for lyr in layers:
        arr = lyr.data.get("x", lyr.data.get("y"))
        if arr is not None:
            n = max(n, int(np.asarray(arr).shape[0]))
    return max(n, 2)


def build_animated_figure(spec: PlotSpec) -> go.Figure:
    """Build a camera-locked plotly figure with ``frames`` + play / slider."""
    import plotly.graph_objects as go

    anim = spec.animation
    assert anim is not None
    three_d = spec.z is not None or any(
        PlotKind(lyr.kind) in (PlotKind.LINE3D, PlotKind.SURFACE3D) for lyr in spec.layers
    )
    curves = _curve_layers(spec)
    if not curves:
        # Nothing to reveal — hand back a static figure (the final frame).
        from ._core import render as _static_render

        static = PlotSpec.from_dict({**spec.to_dict(), "animation": None})
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

    # Base traces: a faint full-curve context (only when windowed) + the comet at
    # frame 0.  Record the comet trace indices so frames update *only* those (the
    # context stays put → small frames, and the camera/uirevision is undisturbed).
    base: list[Any] = []
    comet_idx: list[int] = []
    for arr, layer in zip(arrays, curves, strict=True):
        if tail is not None:
            base.append(_context_trace(arr, layer, three_d=three_d))
        for trace in _comet_traces(arr, layer, heads[0], tail, anim, three_d=three_d):
            comet_idx.append(len(base))
            base.append(trace)

    frames = [
        go.Frame(
            data=[
                t
                for arr, layer in zip(arrays, curves, strict=True)
                for t in _comet_traces(arr, layer, idx, tail, anim, three_d=three_d)
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
    return out


def _context_trace(arr: dict[str, np.ndarray], layer: Any, *, three_d: bool) -> Any:
    """Build the full curve drawn once, faintly — the static rotatable backdrop."""
    import plotly.graph_objects as go

    color = layer.style.get("color")
    line = {"color": color, "width": 1.5} if color else {"width": 1.5}
    common = dict(mode="lines", line=line, opacity=0.25, showlegend=False, hoverinfo="skip")
    if three_d:
        return go.Scatter3d(x=arr["x"], y=arr["y"], z=arr["z"], **common)
    return go.Scatter(x=arr["x"], y=arr["y"], **common)


def _comet_traces(
    arr: dict[str, np.ndarray], layer: Any, head: int, tail: int | None, anim: Any, *, three_d: bool
) -> list[Any]:
    """Build the animated comet for one layer at sample ``head`` (windowed trail + head)."""
    import plotly.graph_objects as go

    lo = 0 if tail is None else max(0, head - tail)
    sl = slice(lo, head + 1)
    color = layer.style.get("color")
    width = layer.style.get("lw", layer.style.get("linewidth", 3))
    out: list[Any] = []
    if three_d:
        out.append(
            go.Scatter3d(
                # Plain lists (not numpy → not a base64 typed-array spec) so the
                # real-time driver's Plotly.extendTraces can append to the comet.
                x=arr["x"][sl].tolist(),
                y=arr["y"][sl].tolist(),
                z=arr["z"][sl].tolist(),
                mode="lines",
                line={"color": color, "width": width} if color else {"width": width},
                name=layer.label or "",
                showlegend=layer.label is not None,
            )
        )
        if anim.head:
            out.append(
                go.Scatter3d(
                    x=[arr["x"][head]],
                    y=[arr["y"][head]],
                    z=[arr["z"][head]],
                    mode="markers",
                    marker={"size": anim.head_size, "color": anim.head_color or color},
                    showlegend=False,
                )
            )
    else:
        out.append(
            go.Scatter(
                # Plain lists so the real-time driver's Plotly.extendTraces can append.
                x=arr["x"][sl].tolist(),
                y=arr["y"][sl].tolist(),
                mode="lines",
                line={"color": color, "width": width} if color else {"width": width},
                name=layer.label or "",
                showlegend=layer.label is not None,
            )
        )
        if anim.head:
            out.append(
                go.Scatter(
                    x=[arr["x"][head]],
                    y=[arr["y"][head]],
                    mode="markers",
                    marker={"size": anim.head_size, "color": anim.head_color or color},
                    showlegend=False,
                )
            )
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
    if spec.title:
        layout["title"] = {"text": spec.title}

    if three_d:
        layout["scene"] = {
            "uirevision": _UIREVISION,  # the load-bearing bit: keep the camera while playing
            "xaxis": {"title": {"text": spec.x.label}, "range": _data_range(spec, "x")},
            "yaxis": {"title": {"text": spec.y.label}, "range": _data_range(spec, "y")},
            "zaxis": {
                "title": {"text": spec.z.label if spec.z is not None else ""},
                "range": _data_range(spec, "z"),
            },
        }
    else:
        layout["xaxis"] = _axis(spec.x, _data_range(spec, "x"))
        layout["yaxis"] = _axis(spec.y, _data_range(spec, "y"))
        if spec.aspect == "equal":
            layout["yaxis"]["scaleanchor"] = "x"
            layout["yaxis"]["scaleratio"] = 1
    _maybe_hide_layout_axes(layout, spec, three_d=three_d)
    fig.update_layout(**layout)


def _maybe_hide_layout_axes(layout: dict[str, Any], spec: PlotSpec, *, three_d: bool) -> None:
    """Honor ``style(axes=False)`` on a plotly animation layout dict (in place)."""
    if not spec._axes_hidden():
        return
    if three_d:
        for k in ("xaxis", "yaxis", "zaxis"):
            if k in layout["scene"]:
                layout["scene"][k]["visible"] = False
    else:
        layout["xaxis"]["visible"] = False
        layout["yaxis"]["visible"] = False


def _axis(axis: Axis, data_range: list[float] | None) -> dict[str, Any]:
    """Return a 2-D plotly axis dict (label + fixed range from the spec or data)."""
    rng = list(axis.limits) if axis.limits is not None else data_range
    out: dict[str, Any] = {"title": {"text": axis.label}}
    if rng is not None:
        out["range"] = rng
    if axis.scale in ("log", "symlog"):
        out["type"] = "log"
    return out


def animated_html(
    spec: PlotSpec,
    *,
    path: str | os.PathLike[str] | None = None,
    html: bool = False,
    full_html: bool | None = None,
    include_plotlyjs: str | bool = "cdn",
) -> str | os.PathLike[str]:
    """Export a smooth, **rotatable-while-playing** real-time animation as HTML.

    Draws the full attractor once (faint, static) + a comet (windowed trail + head)
    per curve layer, then attaches a ``requestAnimationFrame`` loop
    (:data:`_REALTIME_JS`) that advances the comet via ``Plotly.react`` — handing
    back fresh comet trace objects while the layout (hence the camera) and the
    static context trace are left untouched, so the camera the user sets is
    **never reset** and motion runs at the browser's frame rate.  A minimal
    play/pause + restart overlay makes it obviously alive and controllable.
    Returns the HTML string (``html=True``) or the written path (``path=``).
    """
    import plotly.graph_objects as go

    from ._html import to_html as _to_html
    from ._html import write_html as _write_html

    anim = spec.animation
    assert anim is not None
    three_d = spec.z is not None or any(
        PlotKind(lyr.kind) in (PlotKind.LINE3D, PlotKind.SURFACE3D) for lyr in spec.layers
    )
    curves = _curve_layers(spec)
    if not curves:  # nothing to animate — write the static figure
        static = PlotSpec.from_dict({**spec.to_dict(), "animation": None})
        if path is not None:
            return _write_html(
                static, path, full_html=full_html is not False, include_plotlyjs=include_plotlyjs
            )
        return _to_html(static, full_html=bool(full_html), include_plotlyjs=include_plotlyjs)

    arrays = [_curve_arrays(lyr) for lyr in curves]
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
    base: list[Any] = []
    layers_meta: list[dict[str, int]] = []
    for arr, layer in zip(arrays, curves, strict=True):
        meta: dict[str, int] = {"ctx": len(base)}
        base.append(_context_trace(arr, layer, three_d=three_d))
        comet = _comet_traces(arr, layer, 1, window, anim, three_d=three_d)
        meta["comet"] = len(base)
        base.append(comet[0])
        meta["head"] = -1
        if anim.head and len(comet) > 1:
            meta["head"] = len(base)
            base.append(comet[1])
        layers_meta.append(meta)

    fig = go.Figure(data=base)
    _realtime_layout(fig, spec, three_d=three_d)

    # Speed: traverse the whole series in ~duration seconds at the browser's 60 fps.
    duration = float(anim.duration) if anim.duration else 12.0
    stride = max(1, round(n / (duration * 60.0)))
    js = (
        _REALTIME_JS.replace("__LAYERS__", json.dumps(layers_meta))
        .replace("__WINDOW__", str(int(window)))
        .replace("__STRIDE__", str(int(stride)))
        .replace("__IS3D__", "true" if three_d else "false")
        .replace("__N__", str(int(n)))
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


def _realtime_layout(fig: go.Figure, spec: PlotSpec, *, three_d: bool) -> None:
    """Apply fixed ranges + title for the real-time animation (no frames / play button)."""
    layout: dict[str, Any] = {
        "margin": {"t": 40, "r": 10, "b": 10, "l": 10},
        "uirevision": _UIREVISION,
    }
    if spec.title:
        layout["title"] = {"text": spec.title}
    if three_d:
        layout["scene"] = {
            "uirevision": _UIREVISION,
            "xaxis": {"title": {"text": spec.x.label}, "range": _data_range(spec, "x")},
            "yaxis": {"title": {"text": spec.y.label}, "range": _data_range(spec, "y")},
            "zaxis": {
                "title": {"text": spec.z.label if spec.z is not None else ""},
                "range": _data_range(spec, "z"),
            },
        }
    else:
        layout["xaxis"] = _axis(spec.x, _data_range(spec, "x"))
        layout["yaxis"] = _axis(spec.y, _data_range(spec, "y"))
        if spec.aspect == "equal":
            layout["yaxis"]["scaleanchor"] = "x"
            layout["yaxis"]["scaleratio"] = 1
    _maybe_hide_layout_axes(layout, spec, three_d=three_d)
    fig.update_layout(**layout)
