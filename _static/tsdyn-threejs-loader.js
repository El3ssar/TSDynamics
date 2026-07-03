/**
 * Reference three.js loader for the TSDynamics `threejs` export payload
 * (stream VIZ-THREEJS-DOC).
 *
 * A `PlotSpec` rendered with `spec.render("threejs")` produces a
 * BufferGeometry-ready JSON payload (see docs/visualization/threejs-export.md for
 * the schema).  This module turns that payload into a live, orbitable three.js
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
 * `PlotSpec`, e.g. `to_plot_spec(animate=True)`), each line geometry plays a
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

/** Build a THREE.BufferGeometry from one payload geometry (positions/colors/indices). */
function buildGeometry(geom) {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(geom.positions);
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (geom.colors && geom.colors.length === geom.positions.length) {
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
    const pointMat = new THREE.PointsMaterial({
      size: mat && mat.markersize != null ? mat.markersize : 0.6,
      vertexColors: hasVertexColor,
      color: baseColor,
      opacity,
      transparent,
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
function installAnimation(container, comets, anim) {
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

  function paint() {
    const hi = i % nSamples;
    for (const c of comets) {
      c.seek(Math.min(hi, c.nVerts - 1), trail);
    }
    readout.textContent = Math.round((100 * hi) / (nSamples - 1)) + "%";
  }

  // The comet clock is decoupled from OrbitControls' own rAF loop (the camera
  // keeps updating regardless), so only advance the head index here.
  function tick() {
    if (stopped) return;
    if (playing) {
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
    requestAnimationFrame(tick);
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
  requestAnimationFrame(tick);
  console.info("tsd-anim(threejs): starting", {
    n: nSamples,
    trail: trail,
    stride: stride,
  });

  return {
    objects: comets.map((c) => c.group),
    stop() {
      stopped = true;
      if (bar.parentNode === container) container.removeChild(bar);
    },
  };
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
 * @param {HTMLElement} container - a sized element to mount the WebGL canvas in.
 * @param {object} payload - the parsed `spec.render("threejs")` JSON payload.
 * @param {object} [opts] - { background?: string|number, autoRotate?: boolean }.
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
  const target = cam.target || boundsCentre(meta.bounds);

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
    const dist = 1.6 * (xyExtent || 1);
    camera.up.set(0, 1, 0);
    camera.position.set(target[0], target[1], target[2] + dist);
    camera.lookAt(...target);
  } else {
    camera.up.set(...(cam.up || [0, 0, 1]));
    camera.position.set(...(cam.position || [1, 1, 1]));
    camera.lookAt(...target);
  }

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
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
  const geometries = payload.geometries || [];
  if (anim) {
    const comets = [];
    for (let i = 0; i < geometries.length; i++) {
      const geom = geometries[i];
      if (geom.type === "line") {
        const comet = buildLineComet(geom, anim, palette, i);
        comets.push(comet);
        scene.add(comet.group);
      } else if (geom.type === "points") {
        const comet = buildPointsComet(geom, anim, palette, i);
        comets.push(comet);
        scene.add(comet.group);
      } else {
        scene.add(buildObject(geom, i, palette)); // surface: drawn whole
      }
    }
    if (comets.length) {
      animationHandle = installAnimation(container, comets, anim);
      revealing = true;
    } else {
      console.warn("tsd-anim(threejs): animation block but no line/points geometry to reveal");
    }
  } else {
    for (let i = 0; i < geometries.length; i++) {
      scene.add(buildObject(geometries[i], i, palette));
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
  function animate() {
    if (!running) return;
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();

  function onResize() {
    const w = container.clientWidth || width;
    const h = container.clientHeight || height;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener("resize", onResize);

  function dispose() {
    running = false;
    if (animationHandle) animationHandle.stop();
    window.removeEventListener("resize", onResize);
    controls.dispose();
    renderer.dispose();
    if (renderer.domElement.parentNode === container) {
      container.removeChild(renderer.domElement);
    }
  }

  return { scene, camera, renderer, controls, dispose };
}
