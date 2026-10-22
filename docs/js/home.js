/* The homepage preview uses wall-clock time, independent of display refresh rate. */
(function () {
  "use strict";
  if (window.tsHomeLoaded) return;
  window.tsHomeLoaded = true;

  var cleanup = null, activeStage = null, threePromise = null;

  function loadThree() {
    if (window.THREE) return Promise.resolve(window.THREE);
    if (threePromise) return threePromise;
    threePromise = new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      var timeout = setTimeout(function () { reject(new Error("Preview timed out")); }, 10000);
      script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
      script.async = true;
      script.onload = function () { clearTimeout(timeout); resolve(window.THREE); };
      script.onerror = function () { clearTimeout(timeout); reject(new Error("Preview unavailable")); };
      document.head.appendChild(script);
    });
    threePromise = threePromise.catch(function (error) { threePromise = null; throw error; });
    return threePromise;
  }

  function initHero(stage) {
    var signal = new AbortController(), alive = true, renderer = null, resizeObserver = null;
    var intersection = null, frameId = null, tourTimer = null, previousTime = null;
    var group, scene, camera, current = null, loading = false, visible = true;
    var systems = [], systemIndex = -1, dragging = false, pointerX = 0, pointerY = 0;
    var orbitAfter = 0, speed = 1, duration = 60, tail = 560, scale = 24;
    var motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    var paused = motion.matches;
    var pauseButton = document.getElementById("ts-pause");
    var nextButton = document.getElementById("ts-next");
    var speedSelect = document.getElementById("ts-speed");
    speed = Number(speedSelect.value) || 1;
    var nameLabel = document.getElementById("ts-sysname");
    var status = document.getElementById("ts-boot");
    var controls = document.getElementById("ts-controls");
    var fallback = stage.querySelector("img");
    var base = new URL(stage.dataset.assets, document.baseURI);
    fallback.hidden = false;
    status.hidden = false;
    status.textContent = "Loading interactive preview…";
    controls.hidden = false;
    pauseButton.disabled = nextButton.disabled = speedSelect.disabled = true;

    function listen(target, name, fn) {
      target.addEventListener(name, fn, { signal: signal.signal });
    }

    function cancelFrame() {
      if (frameId !== null) cancelAnimationFrame(frameId);
      frameId = null;
      previousTime = null;
    }

    function disposeGeometry() {
      if (!group) return;
      while (group.children.length) {
        var object = group.children[0];
        group.remove(object);
        if (object.geometry) object.geometry.dispose();
        if (object.material) object.material.dispose();
      }
    }

    function stop() {
      alive = false;
      signal.abort();
      cancelFrame();
      clearTimeout(tourTimer);
      if (intersection) intersection.disconnect();
      if (resizeObserver) resizeObserver.disconnect();
      disposeGeometry();
      if (renderer) { renderer.dispose(); renderer.domElement.remove(); }
    }

    function canAnimate() {
      return alive && stage.isConnected && visible && !document.hidden && !paused && !dragging;
    }

    function requestFrame() {
      if (canAnimate() && frameId === null) frameId = requestAnimationFrame(frame);
    }

    function updateControls() {
      pauseButton.textContent = paused ? "Play" : "Pause";
      pauseButton.setAttribute("aria-label", paused ? "Play animation" : "Pause animation");
    }

    function draw() {
      if (!alive || !current) return;
      var end = Math.min(current.points.length - 1, Math.floor(current.position));
      var start = Math.max(0, end - tail + 1);
      var count = end - start + 1;
      var positions = current.trail.geometry.attributes.position;
      var colors = current.trail.geometry.attributes.color;
      for (var k = 0; k < count; k++) {
        var point = current.points[start + k];
        positions.setXYZ(k, point.x, point.y, point.z);
        var intensity = Math.pow((k + 1) / count, 1.6);
        colors.setXYZ(k, 0.173 * intensity, 0.773 * intensity, 0.682 * intensity);
      }
      current.trail.geometry.setDrawRange(0, count);
      positions.needsUpdate = true;
      colors.needsUpdate = true;
      current.head.position.copy(current.points[end]);
      current.backdrop.material.opacity = paused ? 0.48 : 0.24;
      renderer.render(scene, camera);
    }

    function frame(timestamp) {
      frameId = null;
      if (!canAnimate() || !current) { previousTime = null; return; }
      var elapsed = previousTime === null ? 0 : Math.min((timestamp - previousTime) / 1000, 0.05);
      previousTime = timestamp;
      if (timestamp >= orbitAfter && !motion.matches) group.rotation.y += 0.025 * elapsed;
      if (current.position < current.points.length - 1) {
        current.position = Math.min(current.points.length - 1,
          current.position + current.points.length * elapsed * speed / duration);
        if (current.position === current.points.length - 1 && tourTimer === null) {
          tourTimer = setTimeout(function () {
            tourTimer = null;
            if (canAnimate()) loadNext();
          }, 2000);
        }
      } else if (!loading && tourTimer === null) {
        tourTimer = setTimeout(function () {
          tourTimer = null;
          if (canAnimate()) loadNext();
        }, 2000);
      }
      draw();
      requestFrame();
    }

    function fitCamera() {
      if (!renderer || !current) return;
      var width = stage.clientWidth, height = stage.clientHeight;
      if (!width || !height) return;
      camera.aspect = width / height;
      var halfAngle = Math.atan(Math.tan(Math.PI / 9) * Math.min(1, camera.aspect));
      camera.position.set(2, 1.5, 84).normalize().multiplyScalar(current.radius * 1.2 / Math.sin(halfAngle));
      camera.lookAt(0, 0, 0);
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      draw();
    }

    function build(data, THREE) {
      if (!Number.isInteger(data.n) || data.n < 2 || !Array.isArray(data.xyz) ||
          data.xyz.length !== 3 * data.n || !data.xyz.every(Number.isFinite)) {
        throw new Error("Invalid preview data");
      }
      var points = [], radius = 0;
      for (var i = 0; i < data.n; i++) {
        var point = new THREE.Vector3(data.xyz[3*i] * scale,
          data.xyz[3*i+2] * scale, data.xyz[3*i+1] * scale);
        radius = Math.max(radius, point.length());
        points.push(point);
      }
      if (!radius) throw new Error("Empty preview trajectory");
      disposeGeometry();
      var backdrop = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineBasicMaterial({ color: 0x2cc5ae, transparent: true, opacity: 0.24 }));
      var geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(tail * 3), 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(tail * 3), 3));
      var trail = new THREE.Line(geometry,
        new THREE.LineBasicMaterial({ vertexColors: true, depthWrite: false }));
      // The trail moves each frame; its initially empty bounds cannot cull it.
      trail.frustumCulled = false;
      var head = new THREE.Mesh(new THREE.SphereGeometry(0.42, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0x8c85f2 }));
      group.add(backdrop, trail, head);
      current = { points: points, radius: radius, position: Math.min(tail, data.n - 1),
        backdrop: backdrop, trail: trail, head: head };
      nameLabel.textContent = data.name;
      renderer.domElement.setAttribute("aria-label", data.name + " attractor. Use arrow keys to rotate, or space to pause.");
      fallback.hidden = true;
      status.hidden = true;
      pauseButton.disabled = false;
      speedSelect.disabled = false;
      previousTime = null;
      fitCamera();
      requestFrame();
    }

    function readJSON(url) {
      return fetch(url, { signal: signal.signal }).then(function (response) {
        if (!response.ok) throw new Error("Preview unavailable");
        return response.json();
      });
    }

    function loadNext() {
      if (!alive || loading || !systems.length) return;
      clearTimeout(tourTimer);
      tourTimer = null;
      loading = true;
      nextButton.disabled = true;
      systemIndex = (systemIndex + 1) % systems.length;
      readJSON(new URL(encodeURIComponent(systems[systemIndex]) + ".json", base))
        .then(function (data) { if (alive) build(data, window.THREE); })
        .catch(function (error) {
          if (!alive || error.name === "AbortError") return;
          // Keep the last usable curve (or the static fallback), without an
          // automatic retry loop or one request for every animation frame.
          paused = true;
          updateControls();
          status.textContent = "Preview unavailable. Choose Next system to try another.";
          status.hidden = false;
        })
        .finally(function () { loading = false; if (alive) nextButton.disabled = false; });
    }

    updateControls();
    loadThree().then(function (THREE) {
      if (!alive) return;
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      var canvas = renderer.domElement;
      canvas.tabIndex = 0;
      canvas.setAttribute("role", "img");
      stage.appendChild(canvas);
      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(40, 1, 0.1, 4000);
      group = new THREE.Group();
      group.rotation.set(-0.34, 0.6, 0);
      scene.add(group);

      listen(pauseButton, "click", function () {
        paused = !paused;
        cancelFrame();
        updateControls();
        draw();
        requestFrame();
      });
      listen(speedSelect, "change", function () { speed = Number(speedSelect.value); });
      listen(nextButton, "click", loadNext);
      listen(canvas, "pointerdown", function (event) {
        if (event.button !== 0) return;
        dragging = true;
        pointerX = event.clientX; pointerY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
        cancelFrame();
      });
      listen(canvas, "pointermove", function (event) {
        if (!dragging) return;
        group.rotation.y += (event.clientX - pointerX) * 0.006;
        group.rotation.x += (event.clientY - pointerY) * 0.006;
        pointerX = event.clientX; pointerY = event.clientY;
        draw();
      });
      function releasePointer() {
        dragging = false;
        orbitAfter = performance.now() + 4000;
        requestFrame();
      }
      listen(canvas, "pointerup", releasePointer);
      listen(canvas, "pointercancel", releasePointer);
      listen(canvas, "keydown", function (event) {
        if (event.key === " ") { event.preventDefault(); pauseButton.click(); return; }
        if (!/^Arrow(Left|Right|Up|Down)$/.test(event.key)) return;
        event.preventDefault();
        group.rotation.y += event.key === "ArrowLeft" ? -0.12 : event.key === "ArrowRight" ? 0.12 : 0;
        group.rotation.x += event.key === "ArrowUp" ? -0.12 : event.key === "ArrowDown" ? 0.12 : 0;
        orbitAfter = performance.now() + 4000;
        draw();
      });
      listen(document, "visibilitychange", function () { cancelFrame(); requestFrame(); });
      listen(motion, "change", function () {
        if (motion.matches) { paused = true; cancelFrame(); updateControls(); draw(); }
      });
      resizeObserver = new ResizeObserver(fitCamera);
      resizeObserver.observe(stage);
      intersection = new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        cancelFrame();
        requestFrame();
      });
      intersection.observe(stage);
      return readJSON(new URL("manifest.json", base)).then(function (manifest) {
        if (!alive) return;
        systems = manifest.systems.filter(function (name) { return typeof name === "string"; });
        if (!systems.length) throw new Error("No preview systems");
        loadNext();
      });
    }).catch(function (error) {
      if (!alive || error.name === "AbortError") return;
      status.textContent = "Static preview · explore the systems catalogue below";
      controls.hidden = true;
      if (renderer) renderer.domElement.hidden = true;
    });
    return stop;
  }

  function initCopy() {
    var button = document.getElementById("ts-install-copy");
    if (!button || button.dataset.ready) return;
    button.dataset.ready = "true";
    button.addEventListener("click", function () {
      var code = document.getElementById("ts-install-cmd");
      var command = code.textContent.trim();
      function selectCommand() {
        var range = document.createRange();
        range.selectNodeContents(code);
        var selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        button.textContent = "Press Ctrl/Cmd+C";
      }
      if (!navigator.clipboard) { selectCommand(); return; }
      navigator.clipboard.writeText(command).then(function () {
        button.textContent = "Copied";
      }).catch(selectCommand);
      setTimeout(function () { button.textContent = "Copy"; }, 2000);
    });
  }

  function init() {
    initCopy();
    var stage = document.getElementById("ts-stage");
    if (stage === activeStage) return;
    if (cleanup) cleanup();
    activeStage = stage;
    cleanup = stage ? initHero(stage) : null;
  }
  if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
    document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
  window.addEventListener("pagehide", function () {
    if (cleanup) cleanup();
    activeStage = null;
    cleanup = null;
  });
  window.addEventListener("pageshow", init);
}());
