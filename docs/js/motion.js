/* Play scientific movies in view, retaining the reader's playback choices. */
(function () {
  "use strict";
  if (window.tsdMotionLoaded) return;
  window.tsdMotionLoaded = true;

  var motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var players = new Map();
  var observed = typeof window.IntersectionObserver === "function";

  function player(video) {
    var signal = new AbortController();
    var visible = !observed, disposed = false, userPaused = false;
    var userStarted = false, pending = false, blocked = false, automaticPauses = 0;
    var interrupted = false, retryAfterPending = false;
    var observer = null;
    video.muted = true;
    video.loop = true;

    function listen(target, event, handler) {
      target.addEventListener(event, handler, { signal: signal.signal });
    }
    function allowed() {
      return !disposed && visible && !document.hidden && !userPaused &&
        (!motion.matches || userStarted) && (observed || userStarted);
    }
    function suspend() {
      if (pending) interrupted = true;
      if (!video.paused) {
        automaticPauses++;
        video.pause();
      }
    }
    function refresh() {
      if (!allowed()) {
        suspend(); return;
      }
      if (pending) {
        if (interrupted) retryAfterPending = true;
        return;
      }
      if (!video.paused || blocked) return;
      pending = true;
      interrupted = false;
      retryAfterPending = false;
      try {
        Promise.resolve(video.play()).catch(function (error) {
          // Leaving the viewport can abort an in-flight play request.
          if (!error || error.name !== "AbortError") blocked = true;
        }).finally(function () {
          pending = false;
          var retry = retryAfterPending;
          interrupted = false;
          retryAfterPending = false;
          if (!allowed()) suspend();
          else if (retry && video.paused && !blocked) refresh();
        });
      } catch (error) {
        pending = false;
        blocked = true;
      }
    }
    listen(video, "play", function () {
      if (!pending) {
        userPaused = false;
        userStarted = true;
        blocked = false;
      }
      if (!allowed()) suspend();
    });
    listen(video, "pause", function () {
      if (automaticPauses) { automaticPauses--; return; }
      userPaused = true;
    });
    listen(video, "ended", function () { userPaused = true; });
    if (observed) {
      observer = new window.IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting && entries[0].intersectionRatio >= 0.25;
        refresh();
      }, { threshold: [0, 0.25] });
      observer.observe(video);
    }
    return {
      refresh: refresh,
      preferenceChanged: function () { userStarted = false; refresh(); },
      suspend: suspend,
      dispose: function () {
        disposed = true;
        suspend();
        signal.abort();
        if (observer) observer.disconnect();
      }
    };
  }

  function each(method) { players.forEach(function (control) { control[method](); }); }
  function init() {
    players.forEach(function (control, video) {
      if (!video.isConnected) { control.dispose(); players.delete(video); }
    });
    document.querySelectorAll(".md-content video").forEach(function (video) {
      if (!players.has(video)) players.set(video, player(video));
    });
    each("refresh");
  }
  document.addEventListener("visibilitychange", function () { each("refresh"); });
  motion.addEventListener("change", function () { each("preferenceChanged"); });
  window.addEventListener("pagehide", function () { each("suspend"); });
  window.addEventListener("pageshow", function () { each("refresh"); });
  if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
    document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}());
