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

/** Build the drawable Object3D for one geometry, dispatching on its `type`. */
function buildObject(geom) {
  const geometry = buildGeometry(geom);
  const hasColor = geometry.getAttribute("color") !== undefined;
  if (geom.type === "points") {
    const material = new THREE.PointsMaterial({
      size: 0.6,
      vertexColors: hasColor,
      color: hasColor ? 0xffffff : 0x4f9dff,
    });
    return new THREE.Points(geometry, material);
  }
  if (geom.type === "surface") {
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      vertexColors: hasColor,
      color: hasColor ? 0xffffff : 0x4f9dff,
      metalness: 0.1,
      roughness: 0.8,
      side: THREE.DoubleSide,
      flatShading: false,
    });
    return new THREE.Mesh(geometry, material);
  }
  // "line" — an indexed list of segment endpoints (0,1,1,2,2,3,...).
  const material = new THREE.LineBasicMaterial({
    vertexColors: hasColor,
    color: hasColor ? 0xffffff : 0x4f9dff,
  });
  return geom.indices && geom.indices.length
    ? new THREE.LineSegments(geometry, material)
    : new THREE.Line(geometry, material);
}

/** Centre of the payload bounds, used as the default orbit target. */
function boundsCentre(bounds) {
  const mid = (b) => (b ? 0.5 * (b[0] + b[1]) : 0);
  return [mid(bounds?.x), mid(bounds?.y), mid(bounds?.z)];
}

/**
 * Build the reveal comet for one line geometry: a faint full-curve backdrop, a
 * bright windowed trail (animated via `setDrawRange`), and a `THREE.Points` head.
 *
 * The `LineSegments` index buffer is `0,1,1,2,2,3,...` (two indices per segment),
 * so `setDrawRange(start, count)` works in *index* units — `count = 2 * segments`.
 * Returns a comet object exposing `seek(headVertex, trailVertices)`.
 */
function buildLineComet(geom, anim) {
  const nVerts = geom.positions.length / 3;
  const hasColor = Boolean(geom.colors && geom.colors.length === geom.positions.length);
  const group = new THREE.Group();

  // Faint full-curve backdrop (the static context the comet sweeps over).
  const backdropGeom = buildGeometry(geom);
  const backdrop = new THREE.LineSegments(
    backdropGeom,
    new THREE.LineBasicMaterial({
      vertexColors: hasColor,
      color: hasColor ? 0xffffff : 0x4f9dff,
      transparent: true,
      opacity: 0.18,
    })
  );
  group.add(backdrop);

  // Bright comet trail — its own geometry so the draw-range does not touch the
  // backdrop.  Drawn as an indexed LineSegments over the same vertices.
  const trailGeom = buildGeometry(geom);
  const trail = new THREE.LineSegments(
    trailGeom,
    new THREE.LineBasicMaterial({
      vertexColors: hasColor,
      color: hasColor ? 0xffffff : 0x6fd6ff,
    })
  );
  group.add(trail);

  // The head marker — a single THREE.Points at the current sample.
  let head = null;
  if (anim.head) {
    const headPos = new Float32Array([0, 0, 0]);
    const headGeom = new THREE.BufferGeometry();
    headGeom.setAttribute("position", new THREE.BufferAttribute(headPos, 3));
    const headColor =
      anim.head_color != null
        ? new THREE.Color(anim.head_color[0], anim.head_color[1], anim.head_color[2])
        : new THREE.Color(0xffe066);
    head = new THREE.Points(
      headGeom,
      new THREE.PointsMaterial({
        size: Math.max(2.0, anim.head_size || 6.0),
        color: headColor,
        sizeAttenuation: false,
      })
    );
    group.add(head);
  }

  const positions = geom.positions;
  function seek(headVertex, trailVertices) {
    const hv = Math.max(0, Math.min(nVerts - 1, headVertex | 0));
    // Trail window: [lo, hv].  trailVertices == null ⇒ persistent (lo = 0).
    const lo = trailVertices == null ? 0 : Math.max(0, hv - trailVertices);
    // Index units: each segment is 2 indices; the window spans (hv - lo) segments.
    const start = 2 * lo;
    const count = 2 * Math.max(0, hv - lo);
    trail.geometry.setDrawRange(start, count);
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
 * @param {HTMLElement} container - a sized element to mount the WebGL canvas in.
 * @param {object} payload - the parsed `spec.render("threejs")` JSON payload.
 * @param {object} [opts] - { background?: number, autoRotate?: boolean }.
 */
export function renderThreejsPayload(container, payload, opts = {}) {
  const width = container.clientWidth || 640;
  const height = container.clientHeight || 420;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(opts.background ?? 0x0b1020);

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1e5);
  const meta = payload.metadata || {};
  const cam = meta.camera || {};
  const target = cam.target || boundsCentre(meta.bounds);
  camera.up.set(...(cam.up || [0, 0, 1]));
  camera.position.set(...(cam.position || [1, 1, 1]));
  camera.lookAt(...target);

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
  // (other geometry types are drawn static); otherwise draw every geometry static.
  const anim = meta.animation || null;
  let animationHandle = null;
  if (anim) {
    const comets = [];
    for (const geom of payload.geometries || []) {
      if (geom.type === "line") {
        const comet = buildLineComet(geom, anim);
        comets.push(comet);
        scene.add(comet.group);
      } else {
        scene.add(buildObject(geom)); // points / surface: drawn whole
      }
    }
    if (comets.length) {
      animationHandle = installAnimation(container, comets, anim);
    }
  } else {
    for (const geom of payload.geometries || []) {
      scene.add(buildObject(geom));
    }
  }

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(...target);
  controls.enableDamping = true;
  // An animated payload holds the camera still by default (the geometry reveal is
  // the motion — auto-rotation would mask it); the user can still orbit by mouse.
  controls.autoRotate = opts.autoRotate ?? !anim;
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
