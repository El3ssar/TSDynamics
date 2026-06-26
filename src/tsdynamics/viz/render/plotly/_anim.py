"""plotly animation rendering (stream VIZ-ANIM).

Two paths, because plotly's built-in *frame* animation re-renders the whole WebGL
scene per frame (janky, and it fights the camera even with ``uirevision``):

- **HTML export — the real one (:func:`animated_html`):** the full attractor is
  drawn **once** (a faint, static, rotatable trace) and a ``requestAnimationFrame``
  loop (:data:`_REALTIME_JS`) advances a **comet** (a windowed trail + a head) by
  re-rendering with ``Plotly.react`` (which, unlike ``restyle``, actually redraws a
  gl3d/WebGL trace) under a constant ``uirevision`` (which makes plotly preserve the
  user's camera across the updates).  So the user can **orbit the attractor with the
  mouse while it plays**, at 60 fps (genuinely smooth).  It is also tiny (the curve
  is embedded once, in the static context trace).  Zero-extra-dependency "share the
  animation" path (no ffmpeg/kaleido).
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
#: mutating only the comet/head trace data and re-rendering with ``Plotly.react``.
#: ``react`` re-renders the WebGL (gl3d) scene — which ``Plotly.restyle`` does *not*
#: do for 3-D traces (the data updates but nothing redraws) — while a constant
#: ``uirevision`` makes plotly **preserve the user's camera / zoom / pan across the
#: updates** (the documented "uirevision persist" pattern), so a 3-D attractor stays
#: rotatable with the mouse while it animates.  The full curve lives in the static
#: context trace (unchanged each tick, so ``react`` diffs it away — only the small
#: comet re-renders).  ``{plot_id}`` is substituted by plotly with the graph-div id;
#: ``__…__`` tokens are filled below.
_REALTIME_JS = """
(function() {
  var gd = document.getElementById('{plot_id}');
  if (!gd || !window.Plotly) { console.warn('tsd-anim: no graph div / Plotly'); return; }
  var LAYERS = __LAYERS__, WINDOW = __WINDOW__, STRIDE = __STRIDE__;
  var IS3D = __IS3D__, N = __N__;
  var layout = gd.layout;
  // Cache the full curve + the comet/head trace templates from the initial figure.
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
  var cometProto = LAYERS.map(function(m) { return gd.data[m.comet]; });
  var headProto = LAYERS.map(function(m) { return m.head >= 0 ? gd.data[m.head] : null; });
  var ctxTrace = LAYERS.map(function(m) { return gd.data[m.ctx]; });
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

  var i = 1, playing = true, raf = null;
  function render() {
    var hi = i % N, lo = Math.max(0, hi - WINDOW);
    // A FRESH array with FRESH comet/head trace objects each tick — plotly's
    // react() treats an unchanged array *reference* as unchanged data (its
    // immutable diff), so the comet must be a new object with freshly sliced
    // arrays; the static context curve keeps its original object, so react
    // diffs it away and only the small comet/head re-render (camera untouched).
    var arr = gd.data.slice();
    for (var l = 0; l < LAYERS.length; l++) {
      var m = LAYERS[l], f = full[l];
      var comet = Object.assign({}, cometProto[l],
        {x: f.x.slice(lo, hi + 1), y: f.y.slice(lo, hi + 1)});
      if (IS3D) { comet.z = f.z.slice(lo, hi + 1); }
      arr[m.comet] = comet;
      arr[m.ctx] = ctxTrace[l];
      if (m.head >= 0) {
        var head = Object.assign({}, headProto[l], {x: [f.x[hi]], y: [f.y[hi]]});
        if (IS3D) { head.z = [f.z[hi]]; }
        arr[m.head] = head;
      }
    }
    try {
      Plotly.react(gd, arr, layout);
    } catch (e) {
      console.error('tsd-anim: react failed', e);
      playing = false; playBtn.textContent = '▶'; return;
    }
    readout.textContent = Math.round(100 * hi / (N - 1)) + '%';
  }
  function tick() {
    if (!playing) { raf = null; return; }
    render();
    i += STRIDE; if (i >= N) { i = 1; }
    raf = requestAnimationFrame(tick);
  }
  function start() { if (!raf) { raf = requestAnimationFrame(tick); } }
  playBtn.onclick = function() {
    playing = !playing;
    playBtn.textContent = playing ? '❚❚' : '▶';
    if (playing) { start(); }
  };
  restartBtn.onclick = function() { i = 1; render(); };
  render();   // paint frame 0 immediately so the comet is visible at rest
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
                x=arr["x"][sl],
                y=arr["y"][sl],
                z=arr["z"][sl],
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
                x=arr["x"][sl],
                y=arr["y"][sl],
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
