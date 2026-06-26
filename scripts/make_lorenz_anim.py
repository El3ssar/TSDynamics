#!/usr/bin/env python
"""Produce a Lorenz animation for plotly + three.js, to eyeball by hand.

Run it from a checkout of the repo (so the three.js reference loader is found)::

    uv run python scripts/make_lorenz_anim.py

It writes everything into ``./lorenz_anim/`` and prints how to view each:

- ``lorenz_plotly.html`` — open directly in a browser (interactive, animated).
- ``viewer.html`` + ``lorenz_threejs.json`` + a copy of the reference loader —
  serve the folder over http and open ``viewer.html`` (three.js needs a server;
  ES-module imports do not work over ``file://``).

No extra dependencies (just ``tsdynamics``); the headless screenshot-diff gate is
the separate ``scripts/verify_animation.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import tsdynamics as ts

_VIEWER_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Lorenz (three.js)</title>
<style>html,body{margin:0;height:100%;background:#0b1020}#viewer{width:100vw;height:100vh}</style>
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
}}
</script></head><body><div id="viewer"></div>
<script type="module">
  import { renderThreejsPayload } from "./tsdyn-threejs-loader.js";
  const payload = await (await fetch("./lorenz_threejs.json")).json();
  // An animated payload reveals a comet (the camera is mouse-orbitable while it plays).
  renderThreejsPayload(document.querySelector("#viewer"), payload);
</script></body></html>
"""


def _find_loader() -> Path:
    """Locate the three.js reference loader shipped in docs/_static."""
    rel = Path("docs/_static/tsdyn-threejs-loader.js")
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / rel).exists():
            return base / rel
    # Fallback: relative to an editable tsdynamics install (src/tsdynamics → repo root).
    repo = Path(ts.__file__).resolve().parents[2]
    if (repo / rel).exists():
        return repo / rel
    raise SystemExit(f"could not find {rel} — run this from the repo root.")


def main() -> None:
    """Write the Lorenz plotly + three.js animations into ./lorenz_anim/."""
    out = Path("lorenz_anim")
    out.mkdir(exist_ok=True)

    tr = ts.Lorenz().integrate(final_time=80, dt=0.01).after(5)

    # --- plotly: a self-contained interactive HTML (open directly) -----------
    tr.to_plot_spec(animate=True).style(color="teal").save(out / "lorenz_plotly.html")

    # --- three.js: payload + viewer page + a copy of the reference loader ----
    tr.to_plot_spec(animate=True).render("threejs", path=str(out / "lorenz_threejs.json"))
    shutil.copy(_find_loader(), out / "tsdyn-threejs-loader.js")
    (out / "viewer.html").write_text(_VIEWER_HTML)

    print(f"""
Wrote ./{out}/ — view by hand:

  PLOTLY (open directly):
      xdg-open {out}/lorenz_plotly.html
    Expect: an orange comet (trail + head) sweeping the attractor over a faint
    full curve, a play/pause + % readout bottom-left, and you can orbit with the
    mouse WHILE it plays. DevTools console (F12) should show 'tsd-anim: starting'
    and must NOT show 'f.x.slice is not a function'.

  THREE.JS (needs a local server — ES modules can't load over file://):
      (cd {out} && python -m http.server 8000)
    then open  http://localhost:8000/viewer.html
    Expect: the comet reveal over the faint attractor, orbitable while it plays.

  BONUS — matplotlib video (no browser/server needed):
      python -c "import tsdynamics as ts; ts.Lorenz().integrate(final_time=80, dt=0.01).after(5).to_plot_spec(animate=True).save('lorenz.mp4')"
""")


if __name__ == "__main__":
    main()
