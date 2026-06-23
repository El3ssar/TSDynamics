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

  for (const geom of payload.geometries || []) {
    scene.add(buildObject(geom));
  }

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(...target);
  controls.enableDamping = true;
  controls.autoRotate = opts.autoRotate ?? true;
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
    window.removeEventListener("resize", onResize);
    controls.dispose();
    renderer.dispose();
    if (renderer.domElement.parentNode === container) {
      container.removeChild(renderer.domElement);
    }
  }

  return { scene, camera, renderer, controls, dispose };
}
