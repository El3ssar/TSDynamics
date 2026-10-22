"""Documentation motion behavior and showcase build integration."""

import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_movie_visibility_preferences_and_manual_controls():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the documentation player controller")
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
class Video extends EventTarget {
  constructor() {
    super(); this.paused = true; this.isConnected = true; this.calls = 0;
  }
  play() {
    this.calls++;
    if (this.deferred) {
      this.paused = false;
      queueMicrotask(() => this.dispatchEvent(new Event('play')));
      return this.deferred;
    }
    if (this.reject) return Promise.reject(this.reject);
    this.paused = false;
    queueMicrotask(() => this.dispatchEvent(new Event('play')));
    return Promise.resolve();
  }
  pause() {
    if (this.paused) return;
    this.paused = true;
    queueMicrotask(() => this.dispatchEvent(new Event('pause')));
  }
  manualPlay() { this.paused = false; this.dispatchEvent(new Event('play')); }
  manualPause() { this.paused = true; this.dispatchEvent(new Event('pause')); }
}
function setup(options = {}) {
  const video = new Video();
  if (options.reject) video.reject = {name: options.reject};
  const document = new EventTarget();
  document.hidden = false;
  let videos = [video], navigate;
  document.querySelectorAll = () => videos;
  const window = new EventTarget();
  const motion = new EventTarget(); motion.matches = !!options.reduced;
  window.matchMedia = () => motion;
  const observers = [];
  if (!options.noObserver) window.IntersectionObserver = class {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe(target) { this.target = target; }
    disconnect() { this.disconnected = true; }
    enter(ratio) { this.callback([{isIntersecting: ratio > 0, intersectionRatio: ratio}]); }
  };
  const document$ = {subscribe(fn) { navigate = fn; fn(); }};
  vm.runInNewContext(source, {window, document, document$, AbortController, Promise, Map});
  return {video, document, window, motion, observers,
    navigate(next) { videos = next; navigate(); },
    hidden(value) { document.hidden = value; document.dispatchEvent(new Event('visibilitychange')); },
    reduced(value) { motion.matches = value; motion.dispatchEvent(new Event('change')); }
  };
}
(async () => {
  const a = setup();
  assert.equal(a.video.calls, 0, 'offscreen movies must not load or start');
  assert.ok(a.video.muted && a.video.loop, 'previews are silent, repeating movies');
  a.observers[0].enter(.6); await tick();
  assert.equal(a.video.paused, false, 'visible movies should start');
  a.observers[0].enter(0); await tick();
  assert.equal(a.video.paused, true, 'offscreen movies should suspend');
  a.observers[0].enter(.6); await tick();
  assert.equal(a.video.paused, false, 'automatic suspension should resume in view');
  a.hidden(true); await tick();
  assert.equal(a.video.paused, true, 'hidden browser tabs should suspend');
  a.hidden(false); await tick();
  assert.equal(a.video.paused, false, 'returning to the tab should resume');
  a.video.manualPause();
  a.observers[0].enter(0); a.observers[0].enter(1); a.hidden(true); a.hidden(false);
  await tick();
  assert.equal(a.video.paused, true, 'scrolling and tab changes must preserve user pause');
  a.video.manualPlay(); await tick();
  assert.equal(a.video.paused, false, 'native play must work after native pause');
  a.window.dispatchEvent(new Event('pagehide')); await tick();
  assert.equal(a.video.paused, true, 'pagehide should stop background media');
  a.window.dispatchEvent(new Event('pageshow')); await tick();
  assert.equal(a.video.paused, false, 'history restoration should resume automatic media');

  const b = setup({reduced: true});
  b.observers[0].enter(1); await tick();
  assert.equal(b.video.calls, 0, 'reduced motion should initially leave a poster');
  b.video.manualPlay(); await tick();
  assert.equal(b.video.paused, false, 'reduced motion must still allow explicit play');
  b.observers[0].enter(0); await tick(); b.observers[0].enter(1); await tick();
  assert.equal(b.video.paused, false, 'explicitly requested playback can resume in view');
  b.reduced(false); await tick(); b.reduced(true); await tick();
  assert.equal(b.video.paused, true, 'a new reduced-motion preference should stop playback');

  const c = setup({reject: 'NotAllowedError'});
  c.observers[0].enter(1); await tick();
  c.observers[0].enter(0); c.observers[0].enter(1); await tick();
  assert.equal(c.video.calls, 1, 'autoplay denial must not produce repeated play attempts');
  c.video.manualPlay(); await tick();
  assert.equal(c.video.paused, false, 'native controls remain usable after autoplay denial');

  const d = setup({reject: 'AbortError'});
  d.observers[0].enter(1); await tick();
  d.video.reject = null;
  d.observers[0].enter(0); d.observers[0].enter(1); await tick();
  assert.equal(d.video.paused, false, 'an aborted play request must not permanently block playback');
  const replacement = new Video();
  d.video.isConnected = false; d.navigate([replacement]); await tick();
  assert.equal(d.video.paused, true, 'instant navigation must suspend the previous page');
  assert.equal(d.observers[0].disconnected, true, 'detached movies must release their observer');
  d.observers[1].enter(1); await tick();
  assert.equal(replacement.paused, false, 'new page movies must be initialized');

  const e = setup({noObserver: true});
  assert.equal(e.video.calls, 0, 'older browsers should retain manual controls');
  e.video.manualPlay(); await tick();
  assert.equal(e.video.paused, false, 'manual fallback must remain playable');

  const f = setup();
  let abortPlay;
  f.video.deferred = new Promise((resolve, reject) => { abortPlay = reject; });
  f.observers[0].enter(1); await tick();
  f.observers[0].enter(0); f.observers[0].enter(1);
  f.video.deferred = null;
  abortPlay({name: 'AbortError'}); await tick();
  assert.equal(f.video.calls, 2, 'returning during a pending play should retry once after abort');
  assert.equal(f.video.paused, false, 'a fast scroll out and back must resume the visible movie');

  const g = setup({reject: 'AbortError'});
  g.observers[0].enter(1); await tick(); await tick();
  assert.equal(g.video.calls, 1, 'persistent AbortError must not produce an automatic retry loop');

  const h = setup();
  let abortHistoryPlay;
  h.video.deferred = new Promise((resolve, reject) => { abortHistoryPlay = reject; });
  h.observers[0].enter(1); await tick();
  h.window.dispatchEvent(new Event('pagehide'));
  h.window.dispatchEvent(new Event('pageshow'));
  h.video.deferred = null;
  abortHistoryPlay({name: 'AbortError'}); await tick();
  assert.equal(h.video.calls, 2, 'fast history restoration should retry an interrupted pending play');
  assert.equal(h.video.paused, false, 'restored visible pages must resume their movies');
  console.log('Documentation movie lifecycle passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "docs/js/motion.js")],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_hook_checks_showcases_without_leaking_its_import_path():
    pytest.importorskip("mkdocs")
    previous_path = sys.path.copy()
    hook = runpy.run_path(str(ROOT / "hooks/docs_design.py"))
    hook["on_pre_build"]({})
    assert sys.path == previous_path
