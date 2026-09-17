/**
 * Reference three.js loader for the TSDynamics `threejs` export payload
 * (stream VIZ-THREEJS-DOC).
 *
 * A `PlotSpec` rendered with `spec.render("threejs")` produces a
 * BufferGeometry-ready JSON payload (see docs/visualization/backends.md for the
 * schema).  This module turns that payload into a live, orbitable three.js
 * scene with one call — it is the canonical reference a browser front-end can copy.
 *
 * Usage (ES module, with an import map mapping "three" to a CDN build):
 *
 *     import { renderThreejsPayload } from "./tsdyn-threejs-loader.js";
 *     const payload = await (await fetch("lorenz-threejs.json")).json();
 *     const handle = renderThreejsPayload(document.querySelector("#viewer"), payload);
 *
 * `handle` carries { scene, camera, renderer, controls, dispose() }.
 *
 * Animation (reveal comet)
 * ------------------------
 * When the payload's `metadata.animation` block is present (an *animated*
 * `PlotSpec`, e.g. `ts.plot(traj, animate=True)`), each line geometry plays a
 * **reveal comet**: a faint full-curve backdrop is drawn once, and a bright comet
 * (a windowed trail + a `THREE.Points` head) sweeps the curve by advancing
 * `geometry.setDrawRange(start, count)` per `requestAnimationFrame` — no buffer is
 * re-uploaded.  `OrbitControls` keeps running, so the camera is independent of the
 * geometry update: **orbit the attractor with the mouse while it plays.**  A
 * minimal play/pause + restart overlay (mirroring the plotly export) sits
 * bottom-left.  A *static* payload (no `metadata.animation`) renders exactly as
 * before.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// --- Brand palette (matches the home hero) ---------------------------------
// A thin teal trajectory that fades to the dark stage, an indigo state head.
// The reveal comet inks its trail teal (fading head→tail) and its head indigo,
// regardless of any per-vertex colormap the payload carries — so every viewer
// reads as the elegant hero attractor, not a rainbow tube.
const BRAND_TEAL = new THREE.Color(0x2cc5ae); // bright teal trail head
const BRAND_TEAL_DIM = new THREE.Color(0x11857a); // deep teal backdrop curve
const BRAND_INDIGO = new THREE.Color(0x8c85f2); // indigo state head
const BRAND_STAGE = new THREE.Color(0x0b0f14); // dark stage (trail fades to this)

/** Ceiling on `renderer.setPixelRatio` — see the note where it is applied. */
const MAX_PIXEL_RATIO = 2;

/**
 * A soft round sprite for the state head, built once from a radial-gradient
 * canvas.  A bare `THREE.PointsMaterial` with `sizeAttenuation:false` draws a
 * **hard square** — the blocky white pip visible before this fix; mapping this
 * texture (with `alphaTest`/additive blending) turns the head into a glowing
 * circular dot that reads as the indigo state marker, not a pixel.  Lazily
 * created and cached so every comet shares one GPU texture.
 */
let _headSpriteTexture = null;
function headSprite() {
  if (_headSpriteTexture) return _headSpriteTexture;
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const r = size / 2;
  const grad = ctx.createRadialGradient(r, r, 0, r, r, r);
  // Bright opaque core → soft transparent halo (a bloom-friendly falloff).
  grad.addColorStop(0.0, "rgba(255,255,255,1)");
  grad.addColorStop(0.35, "rgba(255,255,255,0.9)");
  grad.addColorStop(0.7, "rgba(255,255,255,0.25)");
  grad.addColorStop(1.0, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  _headSpriteTexture = tex;
  return tex;
}

/**
 * A small viridis-like colour ramp, as (r, g, b) control points in [0, 1].
 *
 * Schema v3 payloads carry a per-vertex **scalar** `c` channel rather than a
 * pre-expanded RGB `colors` array — three floats per vertex to encode one scalar
 * was 38.6% of the payload weight, and the ramp is a *presentation* choice that
 * belongs to the renderer, not the data.  This is that ramp; `scalarColors()`
 * expands it in the browser, where the expansion is free.
 */
const COLORMAP_STOPS = [
  [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
  [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
];

/** Map t in [0, 1] to an [r, g, b] triple by linear interpolation of the ramp. */
function colormap(t) {
  const u = Math.min(1, Math.max(0, t));
  const last = COLORMAP_STOPS.length - 1;
  const pos = u * last;
  const i = Math.min(last - 1, Math.floor(pos));
  if (pos >= last) return COLORMAP_STOPS[last];
  const f = pos - i;
  const a = COLORMAP_STOPS[i], b = COLORMAP_STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Expand a scalar per-vertex field into a flat Float32Array of RGB triples. */
function scalarColors(values) {
  const n = values.length;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const v = values[i];
    const t = !Number.isFinite(v) ? 0 : span > 0 ? (v - lo) / span : 0.5;
    const rgb = colormap(t);
    out[3 * i] = rgb[0];
    out[3 * i + 1] = rgb[1];
    out[3 * i + 2] = rgb[2];
  }
  return out;
}

/**
 * Build a THREE.BufferGeometry from one payload geometry.
 *
 * Accepts both payload generations: schema v3's scalar `c` channel (expanded
 * here through `colormap`) and a legacy v2 pre-expanded `colors` array.  A
 * `"line"` geometry carries no index buffer in v3 — its vertex order *is* its
 * draw order — so the caller draws a contiguous `THREE.Line`.
 */
function buildGeometry(geom) {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(geom.positions);
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (geom.c && geom.c.length * 3 === positions.length) {
    geometry.setAttribute("color", new THREE.BufferAttribute(scalarColors(geom.c), 3));
  } else if (geom.colors && geom.colors.length === geom.positions.length) {
    const colors = new Float32Array(geom.colors);
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }
  if (geom.indices && geom.indices.length) {
    geometry.setIndex(geom.indices);
  }
  return geometry;
}

/**
 * Resolve a material color from a geometry's `material` block, a palette, and
 * an index — producing a THREE.Color-compatible value or a fallback hex number.
 *
 * Priority: geom.material.color (explicit) → palette[index % len] → fallback.
 *
 * @param {object|null} mat - the geometry's `material` block (may be null).
 * @param {string[]} palette - the theme palette color cycle (may be empty).
 * @param {number} index - the geometry's index in the layer list (for cycling).
 * @param {number} fallback - a hex number used when nothing else resolves.
 * @returns {string|number} a CSS color string or a hex integer for THREE.Color.
 */
function resolveColor(mat, palette, index, fallback) {
  if (mat && mat.color != null) return mat.color;
  if (palette && palette.length > 0) return palette[index % palette.length];
  return fallback;
}

/**
 * Screen-space dot diameter, in device pixels, for a static cloud of `n` points.
 *
 * A point *size* must be a screen quantity, never a world one.  Before this the
 * static branch built its `THREE.PointsMaterial` with `size: 0.6` and
 * `sizeAttenuation` left at its `true` default — i.e. 0.6 **world units**.  On a
 * map's iterate cloud that is catastrophic: the Hénon attractor is 2.56 wide, so
 * every one of the 40 000 dots was drawn 23% as wide as the whole attractor and the
 * export rendered as a **solid opaque slab** with the Cantor-set banding — the
 * entire visual content of the picture — buried under it.  The same constant is
 * benign on a 10-marker fixed-point overlay and lethal on a 2e5-iterate cloud,
 * which is exactly why it cannot be a world unit.
 *
 * The ladder is by point *count*, because the density of the cloud is what decides
 * how much ink each sample may spend: a dense attractor needs ~1 px dots for its
 * fine structure to survive, while a handful of markers needs a glyph big enough to
 * click on.
 */
function pointPixelSize(n) {
  if (n >= 50000) return 1.2;
  if (n >= 10000) return 1.6;
  if (n >= 2000) return 2.4;
  if (n >= 200) return 4.0;
  return 7.0;
}

/**
 * Default opacity for a static cloud of `n` points.
 *
 * Semi-transparent dots turn overlap into *tone*: a fold the orbit visits often
 * accumulates many hits and reads darker/brighter than one it grazes, so the
 * invariant density becomes visible instead of being clipped to a flat silhouette.
 * A sparse marker set has no overlap to encode, so it stays fully opaque.
 */
function pointOpacity(n) {
  if (n >= 20000) return 0.55;
  if (n >= 5000) return 0.75;
  if (n >= 500) return 0.9;
  return 1.0;
}

/** Build the drawable Object3D for one geometry, dispatching on its `type`. */
function buildObject(geom, index, palette) {
  const geometry = buildGeometry(geom);
  const hasVertexColor = geometry.getAttribute("color") !== undefined;
  const mat = geom.material || null;
  // When per-vertex colors are present they take precedence over the material
  // color; when absent, resolve the color from the material block / palette.
  const baseColor = hasVertexColor
    ? 0xffffff
    : resolveColor(mat, palette, index, 0x4f9dff);
  const opacity = mat && mat.alpha != null ? mat.alpha : 1.0;
  const transparent = opacity < 1.0;

  if (geom.type === "points") {
    const nPts = geometry.getAttribute("position").count;
    // `markersize` is a *point* size in the spec's matplotlib-flavoured vocabulary,
    // so it maps to screen pixels here — never to world units (see pointPixelSize).
    const size = mat && mat.markersize != null ? mat.markersize : pointPixelSize(nPts);
    const alpha = mat && mat.alpha != null ? mat.alpha : pointOpacity(nPts);
    const pointMat = new THREE.PointsMaterial({
      size,
      // Constant screen size: a scientific point cloud is a *set* of samples, and a
      // sample must not grow as the camera dollies toward it.
      sizeAttenuation: false,
      vertexColors: hasVertexColor,
      color: baseColor,
      opacity: alpha,
      transparent: alpha < 1.0,
      // With translucent dots, depth writes would let whichever sample happened to
      // be drawn first mask every later one behind it — killing the very overlap
      // the alpha is there to accumulate.
      depthWrite: alpha >= 1.0,
    });
    const obj = new THREE.Points(geometry, pointMat);
    if (mat && mat.zorder != null) obj.renderOrder = mat.zorder | 0;
    return obj;
  }
  if (geom.type === "surface") {
    geometry.computeVertexNormals();
    const meshMat = new THREE.MeshStandardMaterial({
      vertexColors: hasVertexColor,
      color: baseColor,
      metalness: 0.1,
      roughness: 0.8,
      side: THREE.DoubleSide,
      flatShading: false,
      opacity,
      transparent,
    });
    const obj = new THREE.Mesh(geometry, meshMat);
    if (mat && mat.zorder != null) obj.renderOrder = mat.zorder | 0;
    return obj;
  }
  // "line" — an indexed list of segment endpoints (0,1,1,2,2,3,...).
  const lineMat = new THREE.LineBasicMaterial({
    vertexColors: hasVertexColor,
    color: baseColor,
    linewidth: mat && mat.linewidth != null ? mat.linewidth : 1,
    opacity,
    transparent,
  });
  const obj =
    geom.indices && geom.indices.length
      ? new THREE.LineSegments(geometry, lineMat)
      : new THREE.Line(geometry, lineMat);
  if (mat && mat.zorder != null) obj.renderOrder = mat.zorder | 0;
  return obj;
}

/** Centre of the payload bounds, used as the default orbit target. */
function boundsCentre(bounds) {
  const mid = (b) => (b ? 0.5 * (b[0] + b[1]) : 0);
  return [mid(bounds?.x), mid(bounds?.y), mid(bounds?.z)];
}

// ---------------------------------------------------------------------------
// Axes: the scale reference
// ---------------------------------------------------------------------------
// A WebGL scene has no axes of its own, and for a long time this viewer shipped
// none: an exported attractor was a coloured shape floating in a void, with no way
// to tell whether it spanned 3 units or 3000, and no name on any direction.  That is
// a picture, not a plot.  What follows draws the payload's own `metadata.bounds` as a
// labelled frame — box edges, nice-numbered ticks, and axis names taken from
// `metadata.labels` — so the geometry acquires a scale and an orientation.

/** Angular height of a tick number (~2.5% of the viewport height at fov 50). */
const AXIS_TICK_HEIGHT = 0.023;

/** Angular height of an axis name. */
const AXIS_NAME_HEIGHT = 0.03;

/** Relative luminance of a THREE.Color-compatible value, in [0, 1]. */
function relativeLuminance(colorLike) {
  const c = new THREE.Color(colorLike);
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

/**
 * Choose axis ink that is legible against `background`.
 *
 * The viewer's stage is themeable (the payload may carry a light `theme.background`),
 * so a hard-coded grey would vanish on one of the two. Returns `{ line, text }` CSS
 * colours — the frame is deliberately dimmer than the labels so it reads as a
 * reference grid rather than as data.
 */
function axisInk(background) {
  return relativeLuminance(background) < 0.5
    ? { line: "#5a6673", text: "#aebac7" }
    : { line: "#b3bcc6", text: "#4a5561" };
}

/**
 * "Nice" tick positions covering `[lo, hi]` with roughly `target` intervals.
 *
 * The classic 1/2/5 x 10^k ladder: a tick must land on a number a reader can hold in
 * their head, which is the whole reason a plot has ticks rather than a ruler.
 * Returns `{ values, step }` — the step is handed to `formatTick` so every label on
 * one axis carries the same number of decimals.
 */
function niceTicks(lo, hi, target = 5) {
  if (!(hi > lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) {
    return { values: [], step: 1 };
  }
  const raw = (hi - lo) / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const values = [];
  const first = Math.ceil(lo / step - 1e-9) * step;
  for (let v = first; v <= hi + 1e-9 * step; v += step) {
    // Snap the accumulated float drift so a step of 0.2 does not print "0.6000000001".
    values.push(Math.abs(v) < 1e-9 * step ? 0 : Math.round(v / step) * step);
  }
  return { values, step };
}

/** Format one tick value at a precision implied by the axis `step`. */
function formatTick(v, step) {
  if (v === 0) return "0";
  if (Math.abs(v) >= 1e5 || Math.abs(v) < 1e-4) return v.toExponential(1);
  const decimals = Math.max(0, Math.min(6, -Math.floor(Math.log10(Math.abs(step)) + 1e-9)));
  return v.toFixed(decimals);
}

/**
 * A camera-facing text label, drawn once into a canvas texture.
 *
 * `sizeAttenuation: false` is what makes this a *label* rather than a billboard: the
 * sprite keeps a constant angular size, so a tick number stays the same number of
 * pixels tall whether the reader has orbited close in or pulled far out.  With that
 * flag three.js interprets `scale.y` as an angular height, which works out to
 * roughly `scale.y / (2 tan(fov/2))` of the viewport height — hence the small
 * `heightFrac` values used below.
 *
 * @param {string} text - the label.
 * @param {string} cssColor - fill colour.
 * @param {number} heightFrac - angular height (~3% of the viewport at 0.028).
 * @param {[number, number]} anchor - sprite centre in [0,1]^2 (0.5,0.5 = centred).
 */
function makeTextSprite(text, cssColor, heightFrac, anchor = [0.5, 0.5]) {
  const px = 64; // texture cell height; the sprite is scaled, not the glyphs
  const font = `${Math.round(px * 0.68)}px ui-sans-serif, system-ui, "Segoe UI", sans-serif`;
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = font;
  const pad = Math.round(px * 0.18);
  const width = Math.max(px, Math.ceil(probe.measureText(text).width) + 2 * pad);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = px;
  const ctx = canvas.getContext("2d");
  ctx.font = font;
  ctx.fillStyle = cssColor;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, width / 2, px / 2 + 1);

  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: tex,
      transparent: true,
      depthWrite: false,
      sizeAttenuation: false,
    })
  );
  sprite.center.set(anchor[0], anchor[1]);
  sprite.scale.set((heightFrac * width) / px, heightFrac, 1);
  return sprite;
}

/** Push the 12 edges of the box `[x0,x1] x [y0,y1] x [z0,z1]` into `verts`. */
function pushBoxEdges(verts, x0, x1, y0, y1, z0, z1) {
  const c = [
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
  ];
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  for (const [a, b] of edges) verts.push(...c[a], ...c[b]);
}

/**
 * Build the labelled axes frame for one payload's `metadata`.
 *
 * Two shapes, chosen by the data itself:
 *
 * - **planar** (a 2-D portrait / a map's iterate cloud, exported flat at `z = 0`):
 *   a rectangle in the `z = 0` plane with ticks stepping outward below and to the
 *   left, exactly the L-frame a 2-D plot wears.
 * - **volumetric** (a 3-D flow): the full 12-edge bounding box — which doubles as a
 *   perspective cue while orbiting — with ticks on the three edges meeting at the
 *   near-minimum corner.
 *
 * Returns a `THREE.Group` (empty when the bounds are degenerate, so the caller never
 * has to special-case an unbounded payload).
 *
 * Tick lengths and label offsets are quoted in **screen pixels**, converted to world
 * units through `unit`. They cannot be a fraction of the box diagonal: on Hénon
 * (2.56 wide by 0.77 tall) a gap of 13% of the diagonal is 45% of the box *height*,
 * which threw the axis name half a plot-height clear of the frame. A tick gap is a
 * typographic quantity, so it is measured in the units type is measured in.
 *
 * @param {object} meta - the payload `metadata` (uses `bounds` and `labels`).
 * @param {string|number} background - the stage colour, for ink contrast.
 * @param {{unit: number, textPx: number}} view - world units per screen pixel, and
 *   the on-screen height of a tick label (used to clear the axis name past them).
 */
function buildAxes(meta, background, view) {
  const group = new THREE.Group();
  group.name = "tsdyn-axes";
  const b = (meta && meta.bounds) || null;
  if (!b || !b.x || !b.y) return group;

  const [x0, x1] = b.x;
  const [y0, y1] = b.y;
  const [z0, z1] = b.z || [0, 0];
  const spanX = x1 - x0;
  const spanY = y1 - y0;
  const spanZ = z1 - z0;
  const diag = Math.sqrt(spanX * spanX + spanY * spanY + spanZ * spanZ);
  if (!Number.isFinite(diag) || diag <= 0) return group;

  const planar = spanZ <= 1e-6 * Math.max(spanX, spanY);
  const ink = axisInk(background);
  // World units per screen pixel; falls back to a diagonal-relative guess only when
  // the caller cannot supply a camera scale (never in this file).
  const unit = view && view.unit > 0 ? view.unit : diag / 600;
  const textPx = view && view.textPx > 0 ? view.textPx : 16;
  const tickLen = 6 * unit; // the tick mark itself
  const labelGap = 9 * unit; // tick number, clear of the mark

  const lo = [x0, y0, planar ? 0 : z0];
  const hi = [x1, y1, planar ? 0 : z1];
  const centre = [(x0 + x1) / 2, (y0 + y1) / 2, planar ? 0 : (z0 + z1) / 2];

  // --- the static frame ------------------------------------------------------
  const frameVerts = [];
  if (planar) {
    // A flat rectangle rather than a zero-depth box: 12 edges collapsed onto 4 would
    // draw every line twice, which shows up as a visibly heavier frame.
    frameVerts.push(
      x0, y0, 0, x1, y0, 0,
      x1, y0, 0, x1, y1, 0,
      x1, y1, 0, x0, y1, 0,
      x0, y1, 0, x0, y0, 0
    );
  } else {
    pushBoxEdges(frameVerts, x0, x1, y0, y1, z0, z1);
  }
  const frameGeom = new THREE.BufferGeometry();
  frameGeom.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(frameVerts), 3)
  );
  group.add(
    new THREE.LineSegments(
      frameGeom,
      new THREE.LineBasicMaterial({ color: ink.line, transparent: true, opacity: 0.75 })
    )
  );

  // --- ticks and labels, re-seated from the camera ---------------------------
  // A box has *four* edges parallel to each axis, and which of them may carry that
  // axis's ticks depends entirely on where the eye is: a fixed choice (the edges
  // meeting at the minimum corner) put Lorenz's z ticks and its "z" label straight
  // through the middle of the attractor, unreadable and wrong. So the ticks are
  // re-seated every frame onto the parallel edge that currently projects furthest
  // from the box centre — the silhouette edge, which is always outside the data.
  const labels = (meta && meta.labels) || {};
  const keys = ["x", "y", "z"];
  const built = [];
  const tickVerts = [];
  for (const a of planar ? [0, 1] : [0, 1, 2]) {
    const { values, step } = niceTicks(lo[a], hi[a]);
    if (!values.length) continue;
    const entry = { a, values, sprites: [], name: null, base: tickVerts.length / 3 };
    let widestPx = 0;
    for (const v of values) {
      tickVerts.push(0, 0, 0, 0, 0, 0); // seated by update()
      const sprite = makeTextSprite(formatTick(v, step), ink.text, AXIS_TICK_HEIGHT);
      widestPx = Math.max(widestPx, (textPx * sprite.scale.x) / sprite.scale.y);
      entry.sprites.push(sprite);
      group.add(sprite);
    }
    entry.widestPx = widestPx;
    const name = labels[keys[a]];
    if (name) {
      entry.name = makeTextSprite(String(name), ink.text, AXIS_NAME_HEIGHT);
      group.add(entry.name);
    }
    built.push(entry);
  }

  const tickGeom = new THREE.BufferGeometry();
  tickGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(tickVerts), 3));
  const tickLines = new THREE.LineSegments(
    tickGeom,
    new THREE.LineBasicMaterial({ color: ink.line, transparent: true, opacity: 0.9 })
  );
  group.add(tickLines);

  const _p = new THREE.Vector3();

  /**
   * Project a *group-local* point to NDC, aspect-corrected so screen "distance" is
   * isotropic.  The local→world hop matters for a composite: each panel's axes live
   * under a mount translated to its layout cell, and projecting the untranslated
   * coordinates would seat panel 2's ticks using panel 1's screen geometry.
   */
  function ndc(x, y, z, camera) {
    _p.set(x, y, z).applyMatrix4(group.matrixWorld).project(camera);
    return [_p.x * (camera.aspect || 1), _p.y];
  }

  /**
   * Re-seat every tick, tick label and axis name for the current camera.
   *
   * Cheap enough to run per frame: ~20 sprite positions and one small vertex buffer,
   * against a scene that is already re-rasterising tens of thousands of vertices.
   */
  group.userData.update = function update(camera) {
    group.updateWorldMatrix(true, false);
    const pos = tickGeom.getAttribute("position");
    const centreNdc = ndc(centre[0], centre[1], centre[2], camera);
    for (const entry of built) {
      const a = entry.a;
      const [b, c] = [0, 1, 2].filter((i) => i !== a);
      // Pick the parallel edge whose midpoint projects furthest from the centre.
      let best = null;
      let bestScore = -1;
      for (const bi of [0, 1]) {
        for (const ci of [0, 1]) {
          const mid = [0, 0, 0];
          mid[a] = centre[a];
          mid[b] = bi ? hi[b] : lo[b];
          mid[c] = ci ? hi[c] : lo[c];
          const s = ndc(mid[0], mid[1], mid[2], camera);
          const score = Math.hypot(s[0] - centreNdc[0], s[1] - centreNdc[1]);
          if (score > bestScore) {
            bestScore = score;
            best = mid;
          }
        }
      }
      // Outward = away from the box centre, in the plane perpendicular to the axis
      // (`best[a] === centre[a]`, so the axis component is already zero).
      const d = [best[0] - centre[0], best[1] - centre[1], best[2] - centre[2]];
      const dn = Math.hypot(d[0], d[1], d[2]) || 1;
      const o = [d[0] / dn, d[1] / dn, d[2] / dn];

      // Which way does "outward" run on screen?  That decides where the glyph box
      // hangs, so a label never grows back over the frame it is annotating.
      const s0 = ndc(best[0], best[1], best[2], camera);
      const s1 = ndc(best[0] + o[0] * tickLen, best[1] + o[1] * tickLen, best[2] + o[2] * tickLen, camera);
      const sdx = s1[0] - s0[0];
      const sdy = s1[1] - s0[1];
      const horizontal = Math.abs(sdx) > Math.abs(sdy);
      const anchor = horizontal ? [sdx < 0 ? 1 : 0, 0.5] : [0.5, sdy < 0 ? 1 : 0];

      for (let k = 0; k < entry.values.length; k++) {
        const p = [best[0], best[1], best[2]];
        p[a] = entry.values[k];
        const i = entry.base + 2 * k;
        pos.setXYZ(i, p[0], p[1], p[2]);
        pos.setXYZ(
          i + 1,
          p[0] + o[0] * tickLen,
          p[1] + o[1] * tickLen,
          p[2] + o[2] * tickLen
        );
        const sprite = entry.sprites[k];
        sprite.center.set(anchor[0], anchor[1]);
        sprite.position.set(
          p[0] + o[0] * labelGap,
          p[1] + o[1] * labelGap,
          p[2] + o[2] * labelGap
        );
      }
      if (entry.name) {
        // Clear whatever the tick numbers actually occupy in the offset direction —
        // their width when the labels run off to the side, their height when they
        // hang below.
        const occupied = horizontal ? entry.widestPx : textPx;
        const gap = (9 + occupied + 12) * unit;
        entry.name.center.set(anchor[0], anchor[1]);
        // Off the *edge's* midpoint, not the box centre: `best` already sits at the
        // axis midpoint along `a`, and anchoring the name to the box centre instead
        // parked "y" and "z" in the middle of the attractor.
        entry.name.position.set(
          best[0] + o[0] * gap,
          best[1] + o[1] * gap,
          best[2] + o[2] * gap
        );
      }
    }
    pos.needsUpdate = true;
  };

  return group;
}

/** The diagonal of the axis-aligned bounding box of a flat xyz `positions` array. */
function boundsDiagonal(positions) {
  const n = positions.length;
  if (n < 3) return 1;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
  return Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
}

/**
 * Build a **seam-aware** `LineSegments` index buffer for a polyline `positions`.
 *
 * A *wrapped* flow (a torus / angle attractor whose coordinates are folded mod 2π
 * — Arnold web, Arnold–Beltrami–Childress) has consecutive samples that jump the
 * full box width whenever an angle crosses ±π.  A plain `THREE.Line` connects those
 * two samples with a long chord straight across the cube — a spray of spurious
 * "seam" lines that swamp the real structure (the near-vertical smear that made the
 * Arnold web read as a single bar).  This returns an index buffer of only the
 * *short* segments (consecutive pairs closer than `maxStep`), so the seam chords are
 * simply not drawn — the folded torus reads as its true web.  Returns `null` when no
 * seam is present (the caller then draws a cheaper contiguous `THREE.Line`).
 *
 * @param {ArrayLike<number>} positions - flat xyz triples.
 * @param {number} maxStep - the seam threshold (a fraction of the bounds diagonal).
 */
function segmentIndexSkippingSeams(positions, maxStep) {
  const n = positions.length / 3;
  const idx = [];
  const t2 = maxStep * maxStep;
  let seams = 0;
  for (let i = 0; i + 1 < n; i++) {
    const dx = positions[3 * i + 3] - positions[3 * i];
    const dy = positions[3 * i + 4] - positions[3 * i + 1];
    const dz = positions[3 * i + 5] - positions[3 * i + 2];
    if (dx * dx + dy * dy + dz * dz > t2) {
      seams++;
      continue; // a wrap seam — drop this segment
    }
    idx.push(i, i + 1);
  }
  return seams > 0 ? idx : null;
}

/**
 * Build the reveal comet for one line geometry, matching the home hero: a faint
 * deep-teal full-curve backdrop, a bright teal trail that **fades to the dark
 * stage** from head → tail, and an indigo state head.
 *
 * Unlike a `setDrawRange` window (a flat-colour slice), the trail is a
 * fixed-length `THREE.Line` whose positions *and* per-vertex colours are rewritten
 * every `seek()` — the head vertex is bright teal, older samples lerp toward the
 * stage colour (`pow(f, 1.25)`), so the comet is a glowing tapering streak rather
 * than a uniform tube.  Additive blending makes the head bloom.  The payload's own
 * per-vertex colormap (a viridis `c` channel) is deliberately ignored for the
 * comet — the brand look is one teal trajectory, not a rainbow.
 *
 * Returns a comet object exposing `seek(headVertex, trailVertices)` with the same
 * contract `installAnimation` drives for every comet type.
 *
 * @param {object} geom - the geometry block (positions/colors/indices/material).
 * @param {object} anim - the metadata.animation block.
 * @param {string[]} palette - the theme palette color cycle (unused for the brand comet).
 * @param {number} index - this geometry's index in the scene layer list.
 */
function buildLineComet(geom, anim, palette, index) {
  const positions = geom.positions;
  const nVerts = positions.length / 3;
  const group = new THREE.Group();

  // Faint deep-teal full-curve backdrop (the static context the comet sweeps).
  // A single flat teal colour — no vertex colours — so the whole attractor reads
  // as one thin teal line at rest.  A *wrapped* torus flow (an angle folded mod 2π)
  // is drawn as seam-skipping `LineSegments` so the fold chords don't smear across
  // the cube; an ordinary attractor keeps the cheaper contiguous `THREE.Line`.
  const backdropGeom = new THREE.BufferGeometry();
  backdropGeom.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(positions), 3)
  );
  const seamStep = 0.4 * boundsDiagonal(positions);
  const seamIdx = segmentIndexSkippingSeams(positions, seamStep);
  const backdropMat = new THREE.LineBasicMaterial({
    color: BRAND_TEAL_DIM,
    transparent: true,
    opacity: 0.22,
    depthWrite: false,
  });
  let backdrop;
  if (seamIdx) {
    backdropGeom.setIndex(seamIdx);
    backdrop = new THREE.LineSegments(backdropGeom, backdropMat);
  } else {
    backdrop = new THREE.Line(backdropGeom, backdropMat);
  }
  group.add(backdrop);

  // Bright fading trail — a fixed-length windowed line, positions + colours
  // rewritten per seek().  Additive blending so the head end blooms.
  const trailLen = Math.max(2, (anim.trail_length_samples | 0) || 600);
  const tpos = new Float32Array(trailLen * 3);
  const tcol = new Float32Array(trailLen * 3);
  const trailGeom = new THREE.BufferGeometry();
  trailGeom.setAttribute("position", new THREE.BufferAttribute(tpos, 3));
  trailGeom.setAttribute("color", new THREE.BufferAttribute(tcol, 3));
  const trail = new THREE.Line(
    trailGeom,
    new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
  );
  group.add(trail);

  // The indigo state head — a glowing point at the current sample.
  let head = null;
  if (anim.head !== false) {
    const headColor =
      anim.head_color != null
        ? new THREE.Color(anim.head_color[0], anim.head_color[1], anim.head_color[2])
        : BRAND_INDIGO;
    const headGeom = new THREE.BufferGeometry();
    headGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
    head = new THREE.Points(
      headGeom,
      new THREE.PointsMaterial({
        size: Math.max(9.0, (anim.head_size || 8.0) * 1.6),
        color: headColor,
        map: headSprite(),
        alphaTest: 0.02,
        sizeAttenuation: false,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      })
    );
    head.renderOrder = 3;
    group.add(head);
  }

  // Seam threshold for the trail: a live sample that jumps more than this from the
  // previous one is a wrap fold, and its connecting chord is collapsed onto the
  // previous vertex (a zero-length, invisible segment) so the comet doesn't streak
  // across the cube as it crosses a torus seam.
  const seamStep2 = Math.pow(0.4 * boundsDiagonal(positions), 2);

  function seek(headVertex, trailVertices) {
    const hv = Math.max(0, Math.min(nVerts - 1, headVertex | 0));
    const win = trailVertices == null ? trailLen : Math.min(trailLen, trailVertices);
    const from = Math.max(0, hv - win + 1);
    const m = hv - from + 1; // live samples in the window
    // Rewrite the whole fixed-length buffer: live samples from `from`..`hv`
    // (colour fading tail→head), the rest pinned at the head so the extra
    // vertices collapse onto it (zero-length, invisible segments).
    for (let k = 0; k < trailLen; k++) {
      const idx = from + k;
      const live = k < m && idx < nVerts;
      const src = live ? idx : hv;
      let px = positions[3 * src];
      let py = positions[3 * src + 1];
      let pz = positions[3 * src + 2];
      // Collapse a wrap-seam chord: if this live sample leapt from the previous
      // drawn one, pin it onto the previous vertex so no chord crosses the cube.
      if (live && k > 0) {
        const dx = px - tpos[3 * k - 3];
        const dy = py - tpos[3 * k - 2];
        const dz = pz - tpos[3 * k - 1];
        if (dx * dx + dy * dy + dz * dz > seamStep2) {
          px = tpos[3 * k - 3];
          py = tpos[3 * k - 2];
          pz = tpos[3 * k - 1];
        }
      }
      tpos[3 * k] = px;
      tpos[3 * k + 1] = py;
      tpos[3 * k + 2] = pz;
      // f: 0 at the oldest live sample → 1 at the head.
      let f = m > 1 ? k / (m - 1) : 1;
      if (!live) f = 1;
      f = Math.pow(f, 1.25);
      tcol[3 * k] = BRAND_STAGE.r + (BRAND_TEAL.r - BRAND_STAGE.r) * f;
      tcol[3 * k + 1] = BRAND_STAGE.g + (BRAND_TEAL.g - BRAND_STAGE.g) * f;
      tcol[3 * k + 2] = BRAND_STAGE.b + (BRAND_TEAL.b - BRAND_STAGE.b) * f;
    }
    trailGeom.attributes.position.needsUpdate = true;
    trailGeom.attributes.color.needsUpdate = true;
    if (head) {
      const p = head.geometry.getAttribute("position");
      p.setXYZ(0, positions[3 * hv], positions[3 * hv + 1], positions[3 * hv + 2]);
      p.needsUpdate = true;
    }
  }
  seek(0, anim.trail_length_samples);

  return { group, seek, nVerts };
}

/**
 * Build the reveal comet for one **points** geometry (a map's iterate cloud): a
 * faint full point-cloud backdrop, a bright windowed cloud of the most-recent
 * samples (animated via `setDrawRange` in *vertex* units — a Points geometry is
 * unindexed, so `setDrawRange(start, count)` counts vertices directly), and a
 * `THREE.Points` head at the current sample.
 *
 * Mirrors `buildLineComet`'s `{ group, seek, nVerts }` contract so the shared
 * `installAnimation` clock drives line and points comets identically.  A scatter
 * attractor has no chord to sweep, so the comet is a *trailing swarm* rather than a
 * drawn curve — the recognizable shape (Hénon's banana, the logistic parabola)
 * accumulates as the head wanders it.
 *
 * @param {object} geom - the geometry block (positions/colors/material, type "points").
 * @param {object} anim - the metadata.animation block.
 * @param {string[]} palette - the theme palette color cycle for auto-coloring.
 * @param {number} index - this geometry's index in the scene layer list.
 */
function buildPointsComet(geom, anim, palette, index) {
  const nVerts = geom.positions.length / 3;
  const group = new THREE.Group();
  const mat = geom.material || null;
  const size = mat && mat.markersize != null ? mat.markersize : 1.4;

  // Faint teal full-cloud backdrop (the static attractor the swarm sweeps over) —
  // one flat teal colour (the payload viridis colormap is ignored for the brand
  // look), matching the line comet's deep-teal backdrop.
  const backdropGeom = buildGeometry(geom);
  const backdrop = new THREE.Points(
    backdropGeom,
    new THREE.PointsMaterial({
      color: BRAND_TEAL_DIM,
      size: size,
      sizeAttenuation: false,
      transparent: true,
      opacity: 0.22,
    })
  );
  group.add(backdrop);

  // Bright teal trailing swarm — its own geometry so the draw-range does not touch
  // the backdrop.  Points are unindexed, so setDrawRange is in vertex units.
  const swarmGeom = buildGeometry(geom);
  const swarm = new THREE.Points(
    swarmGeom,
    new THREE.PointsMaterial({
      color: BRAND_TEAL,
      size: size * 1.4,
      sizeAttenuation: false,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
  );
  group.add(swarm);

  // The indigo state head at the current sample.
  let head = null;
  if (anim.head !== false) {
    const headPos = new Float32Array([0, 0, 0]);
    const headGeom = new THREE.BufferGeometry();
    headGeom.setAttribute("position", new THREE.BufferAttribute(headPos, 3));
    const headColor =
      anim.head_color != null
        ? new THREE.Color(anim.head_color[0], anim.head_color[1], anim.head_color[2])
        : BRAND_INDIGO;
    head = new THREE.Points(
      headGeom,
      new THREE.PointsMaterial({
        size: Math.max(9.0, (anim.head_size || 8.0) * 1.6),
        color: headColor,
        map: headSprite(),
        alphaTest: 0.02,
        sizeAttenuation: false,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      })
    );
    head.renderOrder = 3;
    group.add(head);
  }

  const positions = geom.positions;
  function seek(headVertex, trailVertices) {
    const hv = Math.max(0, Math.min(nVerts - 1, headVertex | 0));
    // Swarm window: [lo, hv] in vertex units. trailVertices == null ⇒ persistent.
    const lo = trailVertices == null ? 0 : Math.max(0, hv - trailVertices);
    swarm.geometry.setDrawRange(lo, Math.max(0, hv - lo + 1));
    if (head) {
      const p = head.geometry.getAttribute("position");
      p.setXYZ(0, positions[3 * hv], positions[3 * hv + 1], positions[3 * hv + 2]);
      p.needsUpdate = true;
    }
  }
  seek(0, anim.trail_length_samples);

  return { group, seek, nVerts };
}

/**
 * Install the reveal-comet animation: build a comet per line geometry, a master
 * `requestAnimationFrame` clock advancing the head over all comets in lockstep,
 * and a minimal play/pause + restart overlay.  Returns `{ objects, stop() }`.
 */
function installAnimation(container, comets, anim, visibility) {
  // The reveal length is the longest comet (others clamp to their last vertex).
  let nSamples = anim.n_samples | 0;
  for (const c of comets) {
    nSamples = Math.max(nSamples, c.nVerts);
  }
  nSamples = Math.max(2, nSamples);
  const trail = anim.trail_length_samples != null ? anim.trail_length_samples | 0 : null;

  // Speed: traverse the whole series in ~duration seconds at the browser's ~60fps.
  const duration = anim.duration && anim.duration > 0 ? anim.duration : 12.0;
  const stride = Math.max(1, Math.round(nSamples / (duration * 60.0)));

  // --- play/pause + restart overlay (mirrors the plotly export) --------------
  if (getComputedStyle(container).position === "static") {
    container.style.position = "relative";
  }
  const bar = document.createElement("div");
  bar.style.cssText =
    "position:absolute;left:10px;bottom:10px;z-index:10;display:flex;gap:6px;" +
    "align-items:center;font:12px system-ui,sans-serif;color:#aaa;user-select:none;";
  const mkBtn = (txt) => {
    const b = document.createElement("button");
    b.textContent = txt;
    b.style.cssText =
      "cursor:pointer;border:1px solid #8888;border-radius:6px;" +
      "background:rgba(127,127,127,.18);color:inherit;width:30px;height:26px;" +
      "font-size:13px;line-height:1;padding:0;";
    return b;
  };
  const playBtn = mkBtn("❚❚");
  const restartBtn = mkBtn("↺");
  const readout = document.createElement("span");
  bar.appendChild(playBtn);
  bar.appendChild(restartBtn);
  bar.appendChild(readout);
  container.appendChild(bar);

  let i = 1;
  let playing = true;
  let stopped = false;
  let rafId = 0;

  function paint() {
    const hi = i % nSamples;
    for (const c of comets) {
      c.seek(Math.min(hi, c.nVerts - 1), trail);
    }
    readout.textContent = Math.round((100 * hi) / (nSamples - 1)) + "%";
  }

  // The comet clock is decoupled from OrbitControls' own rAF loop (the camera
  // keeps updating regardless), so only advance the head index here.  When the
  // viewer has scrolled out of view (`visibility.visible === false`) the clock
  // idles: a catalogue page with six viewers otherwise runs six comets — and six
  // WebGL draws — forever, for nobody.
  function tick() {
    if (stopped) return;
    if (playing && (!visibility || visibility.visible)) {
      paint();
      i += stride;
      if (i >= nSamples) {
        if (anim.loop === false) {
          playing = false;
          playBtn.textContent = "▶";
        } else {
          i = 1;
        }
      }
    }
    rafId = requestAnimationFrame(tick);
  }

  playBtn.onclick = () => {
    playing = !playing;
    playBtn.textContent = playing ? "❚❚" : "▶";
  };
  restartBtn.onclick = () => {
    i = 1;
    paint();
  };

  paint(); // paint frame 0 immediately so the comet is visible at rest
  rafId = requestAnimationFrame(tick);
  console.info("tsd-anim(threejs): starting", {
    n: nSamples,
    trail: trail,
    stride: stride,
  });

  return {
    objects: comets.map((c) => c.group),
    stop() {
      stopped = true;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = 0;
      if (bar.parentNode === container) container.removeChild(bar);
    },
  };
}

/**
 * Recursively dispose every GPU resource under `root` (geometries, materials and
 * any texture a material holds).
 *
 * `renderer.dispose()` releases the *context*, not the buffers uploaded through
 * it: without this walk, mounting and unmounting a viewer (a tabbed docs page, a
 * notebook cell re-run, an SPA route change) leaks one full attractor's vertex
 * buffers per mount.
 */
function disposeObject(root) {
  root.traverse((obj) => {
    if (obj.geometry) obj.geometry.dispose();
    const mats = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
    for (const m of mats) {
      for (const key of ["map", "alphaMap", "envMap"]) {
        const tex = m[key];
        // `_headSpriteTexture` is a module-level singleton shared by every comet
        // in the document — disposing it here would blank the head of any other
        // live viewer on the page.  Every other texture is per-object.
        if (tex && tex !== _headSpriteTexture && typeof tex.dispose === "function") {
          tex.dispose();
        }
      }
      m.dispose();
    }
  });
}

/**
 * Render a TSDynamics `threejs` payload into `container` and return a handle.
 *
 * Applies the payload's `metadata.theme` block when present. The threejs
 * backend honors exactly two theme fields (a WebGL scene has no axes / text /
 * grid to ink), so the exporter emits only these and the loader reads only
 * these:
 * - `theme.background` sets the scene background color (overridden by
 *   `opts.background`).
 * - `theme.palette` provides the auto-color cycle for geometries that carry no
 *   per-vertex colors and no explicit `material.color`.
 *
 * No `theme.foreground` / font / grid / title-size fields are present in the
 * payload (the exporter drops them rather than ship a dead field); the
 * renderer's capability layer warns that they are not honored.
 *
 * A **labelled axes frame** (see `buildAxes`) is drawn by default for a static
 * payload — a plot without a scale is a picture, not a plot — and suppressed by
 * default when a reveal comet is installed, where the motion is the subject and tick
 * numbers are clutter. `opts.axes` forces either way.
 *
 * A **composite** payload (`spec` built with `ts.viz.plot(..., layout="row")`)
 * carries its drawables under `payload.panels` rather than `payload.geometries`;
 * each panel is mounted in its own group at the panel's layout `offset`, with its
 * own axes frame built from its own local bounds.
 *
 * @param {HTMLElement} container - a sized element to mount the WebGL canvas in.
 * @param {object} payload - the parsed `spec.render("threejs")` JSON payload.
 * @param {object} [opts] - { background?, autoRotate?: boolean, axes?: boolean }.
 */
export function renderThreejsPayload(container, payload, opts = {}) {
  const width = container.clientWidth || 640;
  const height = container.clientHeight || 420;

  const scene = new THREE.Scene();
  const meta = payload.metadata || {};

  // Apply the theme block: background color + palette.
  const theme = meta.theme || null;
  // Priority: opts.background > theme.background > built-in default.
  const bgColor = opts.background ?? (theme && theme.background) ?? 0x0b1020;
  scene.background = new THREE.Color(bgColor);
  // The palette drives auto-coloring of layers that carry no per-vertex color.
  const palette = theme && Array.isArray(theme.palette) ? theme.palette : [];

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1e5);
  const cam = meta.camera || {};
  // A copy: the planar branch nudges the target to make room for the axis labels,
  // and mutating `payload.metadata.camera.target` in place would corrupt a payload
  // the caller may well mount a second time.
  const target = (cam.target || boundsCentre(meta.bounds)).slice();

  // Axes are on for a static payload (a plot without a scale is a picture) and off
  // for an animated reveal (there the comet is the subject and tick numbers are
  // clutter).  `opts.axes` forces either way.  Resolved here, before the camera,
  // because the framing must leave a gutter for the labels.
  const drawAxes = opts.axes ?? !meta.animation;

  // A **2-D phase portrait** is exported as a flat curve in the z = 0 plane, but the
  // payload camera is a 3-D oblique eye — so a planar loop is viewed edge-on and
  // collapses to a diagonal band (the "sheet of parallel lines" a limit cycle read
  // as).  Detect the planar case (a degenerate z extent) and look **straight down**
  // the z axis instead, so the portrait shows face-on; up becomes +y.  A genuine
  // 3-D attractor keeps its exported oblique camera.
  const bz = meta.bounds && meta.bounds.z;
  const bx = meta.bounds && meta.bounds.x;
  const by = meta.bounds && meta.bounds.y;
  const zExtent = bz ? Math.abs(bz[1] - bz[0]) : 0;
  const xyExtent = Math.max(
    bx ? Math.abs(bx[1] - bx[0]) : 0,
    by ? Math.abs(by[1] - by[0]) : 0
  );
  const isPlanar = xyExtent > 0 && zExtent <= 1e-6 * xyExtent;
  if (isPlanar) {
    // Fit the data box to the viewport rather than backing off by a fixed multiple
    // of the *larger* extent.  The old `1.6 * max(dx, dy)` framed a square attractor
    // acceptably and a wide flat one — Hénon is 2.56 x 0.77 — terribly: it filled a
    // fifth of the canvas height and left the rest black.  Solve the two fits (the
    // vertical one from the fov, the horizontal one from the fov *and* the aspect)
    // and take whichever needs more distance.
    const halfW = 0.5 * (bx ? Math.abs(bx[1] - bx[0]) : xyExtent);
    const halfH = 0.5 * (by ? Math.abs(by[1] - by[0]) : xyExtent);
    const tanHalfFov = Math.tan((camera.fov * Math.PI) / 360);
    const aspect = Math.max(1e-6, width / height);
    const pad = 1.06;
    const fit = (gx, gy) =>
      Math.max((halfH + gy / 2) / tanHalfFov, (halfW + gx / 2) / (tanHalfFov * aspect)) * pad;
    // The axis ticks and names hang outside the data box, so the frame needs a
    // gutter on the -x / -y sides.  That gutter is a *pixel* quantity, and pixels
    // only become world units once the distance is known — so fit once without it,
    // convert, and fit again.  (One pass is plenty: the correction is ~10%.)
    let dist = fit(0, 0);
    let gutter = 0;
    if (drawAxes) {
      const unit0 = (2 * dist * tanHalfFov) / Math.max(1, height);
      gutter = 62 * unit0; // ticks + numbers + the axis name
      dist = fit(gutter, gutter);
    }
    camera.up.set(0, 1, 0);
    camera.position.set(target[0] - gutter / 2, target[1] - gutter / 2, target[2] + dist);
    camera.lookAt(target[0] - gutter / 2, target[1] - gutter / 2, target[2]);
    target[0] -= gutter / 2;
    target[1] -= gutter / 2;
  } else {
    camera.up.set(...(cam.up || [0, 0, 1]));
    camera.position.set(...(cam.position || [1, 1, 1]));
    if (drawAxes) {
      // The exported camera sits exactly one bounding-box diagonal back, which frames
      // the *box* and nothing else — so the tick labels, which hang outside it, fell
      // off the canvas (Lorenz lost its "x" name and its outermost x tick). Dolly
      // back along the same view direction to open a margin for them.
      const eye = new THREE.Vector3(...target);
      camera.position.sub(eye).multiplyScalar(1.22).add(eye);
    }
    camera.lookAt(...target);
  }

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  // Cap the device pixel ratio: a 3x-DPR phone would otherwise rasterise ~9x the
  // fragments of a 1x display for a thin-line scene that gains nothing past 2x,
  // and that is exactly where the battery and the frame budget are tightest.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
  renderer.setSize(width, height);
  container.appendChild(renderer.domElement);

  // Lights (only the surface mesh needs them; harmless for lines/points).
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 0.8);
  key.position.set(1, 1, 1);
  scene.add(key);

  // When the payload is animated, build a reveal comet for every line geometry
  // (a swept curve) and every points geometry (a trailing swarm — a map's iterate
  // cloud); surface geometries are drawn static.  Otherwise draw everything static.
  // `revealing` is true only when at least one comet was actually installed — a
  // (defensive) animation block with no animatable geometry must NOT freeze the camera.
  const anim = meta.animation || null;
  let animationHandle = null;
  let revealing = false;
  // Shared visibility gate: both the render loop and the comet clock idle while
  // the viewer is scrolled out of view (see the IntersectionObserver below).
  const visibility = { visible: true };

  // A composite payload keeps its drawables per panel; a single-panel one keeps them
  // at the top level.  Normalise to a list of `{ geometries, metadata, offset }` so
  // the draw loop below is written once.  Without this a composite export mounted a
  // scene with **nothing in it** — `payload.geometries` is `[]` for a composite.
  const panels =
    Array.isArray(payload.panels) && payload.panels.length
      ? payload.panels.map((p) => ({
          geometries: p.geometries || [],
          metadata: p.metadata || {},
          offset: p.offset || [0, 0, 0],
        }))
      : [{ geometries: payload.geometries || [], metadata: meta, offset: [0, 0, 0] }];

  const comets = [];
  /** Axes groups needing a per-frame re-seat (see `buildAxes`'s `update`). */
  const axesFrames = [];
  for (const panel of panels) {
    const mount = new THREE.Group();
    mount.position.set(panel.offset[0] || 0, panel.offset[1] || 0, panel.offset[2] || 0);
    scene.add(mount);
    for (let i = 0; i < panel.geometries.length; i++) {
      const geom = panel.geometries[i];
      if (anim && geom.type === "line") {
        const comet = buildLineComet(geom, anim, palette, i);
        comets.push(comet);
        mount.add(comet.group);
      } else if (anim && geom.type === "points") {
        const comet = buildPointsComet(geom, anim, palette, i);
        comets.push(comet);
        mount.add(comet.group);
      } else {
        mount.add(buildObject(geom, i, palette)); // static, or an unanimatable surface
      }
    }
    panel.mount = mount;
  }
  if (anim) {
    if (comets.length) {
      animationHandle = installAnimation(container, comets, anim, visibility);
      revealing = true;
    } else {
      console.warn("tsd-anim(threejs): animation block but no line/points geometry to reveal");
    }
  }

  // The axes frame is per *panel*, built from that panel's own local bounds, so a
  // composite row does not get one giant frame straddling both plots.
  if (drawAxes) {
    // World units per screen pixel at the opening camera distance, and the on-screen
    // height of a tick label — the two quantities buildAxes needs to lay type out in
    // typographic rather than data units.
    const eyeDist = camera.position.distanceTo(new THREE.Vector3(...target)) || 1;
    const tanHalfFov = Math.tan((camera.fov * Math.PI) / 360);
    const unit = (2 * eyeDist * tanHalfFov) / Math.max(1, height);
    const textPx = (AXIS_TICK_HEIGHT / (2 * tanHalfFov)) * Math.max(1, height);
    for (const panel of panels) {
      const axesGroup = buildAxes(panel.metadata, bgColor, { unit, textPx });
      panel.mount.add(axesGroup);
      if (axesGroup.userData.update) axesFrames.push(axesGroup);
    }
  }

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(...target);
  controls.enableDamping = true;
  // A reveal comet holds the camera still by default (the geometry reveal is the
  // motion — auto-rotation would mask it); the user can still orbit by mouse.
  // Without an installed comet the camera auto-rotates as in the static export.
  controls.autoRotate = opts.autoRotate ?? !revealing;
  controls.autoRotateSpeed = 0.6;
  controls.update();

  let running = true;
  let frameId = 0;
  function animate() {
    if (!running) return;
    if (visibility.visible) {
      controls.update();
      for (const ax of axesFrames) ax.userData.update(camera);
      renderer.render(scene, camera);
    }
    frameId = requestAnimationFrame(animate);
  }
  animate();

  // --- sizing -----------------------------------------------------------------
  // A `window.resize` listener only fires when the *window* changes size, so a
  // viewer whose container is hidden at boot — a tab panel, an accordion, a
  // `display:none` details block, which is THE common embedding context — measures
  // 0x0, falls back to the 640x420 default, and then stays 640x420 forever once
  // revealed.  A ResizeObserver on the container fires on that reveal (and on any
  // later layout change) and is what actually makes the embed responsive.
  function resizeTo(w, h) {
    if (w <= 0 || h <= 0) return; // still hidden — wait for the next observation
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  let resizeObserver = null;
  if (typeof ResizeObserver !== "undefined") {
    resizeObserver = new ResizeObserver((entries) => {
      const box = entries[0] && entries[0].contentRect;
      resizeTo(
        Math.round((box ? box.width : container.clientWidth) || 0),
        Math.round((box ? box.height : container.clientHeight) || 0)
      );
    });
    resizeObserver.observe(container);
  }
  // Kept as a fallback for environments without ResizeObserver, and because a
  // window resize that does not change the container's box still changes the
  // device pixel ratio on some displays.
  function onResize() {
    resizeTo(container.clientWidth || width, container.clientHeight || height);
  }
  window.addEventListener("resize", onResize);

  // --- off-screen idling ------------------------------------------------------
  // A catalogue page embeds many viewers; without this every one of them keeps a
  // WebGL draw + a comet clock running while it sits far below the fold.
  let intersectionObserver = null;
  if (typeof IntersectionObserver !== "undefined") {
    intersectionObserver = new IntersectionObserver(
      (entries) => {
        visibility.visible = entries.some((e) => e.isIntersecting);
      },
      { threshold: 0 }
    );
    intersectionObserver.observe(container);
  }

  function dispose() {
    running = false;
    if (frameId) cancelAnimationFrame(frameId);
    frameId = 0;
    if (animationHandle) animationHandle.stop();
    if (resizeObserver) resizeObserver.disconnect();
    if (intersectionObserver) intersectionObserver.disconnect();
    window.removeEventListener("resize", onResize);
    controls.dispose();
    disposeObject(scene);
    scene.clear();
    renderer.dispose();
    if (renderer.domElement.parentNode === container) {
      container.removeChild(renderer.domElement);
    }
  }

  return { scene, camera, renderer, controls, dispose };
}
