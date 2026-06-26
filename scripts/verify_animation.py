#!/usr/bin/env python
"""Headless-browser verifier for the viz animation exports.

Loads an exported animation in headless Chrome (software WebGL), captures the
canvas at several time points, and reports whether the pixels CHANGE between
consecutive frames (the animation plays) or stay identical (static). Works for
both the **plotly** HTML export and the **three.js** payload + reference loader.

It distinguishes genuine motion from a one-shot redraw (it requires change
*between consecutive* frames, not just first-vs-last) and dumps the browser
console so a JS exception surfaces.

Requirements (already in the dev env): ``playwright`` (``uv pip install
playwright``) and a system Chrome/Chromium (``google-chrome-stable``). No
chromium download is needed — it launches with ``channel="chrome"``.

Examples
--------
Plotly HTML (3-D / 2-D / time-series animation)::

    python -c "import tsdynamics as ts; \
        ts.Lorenz().integrate(final_time=80, dt=0.01).after(5)\
          .to_plot_spec(animate=True).save('orbit.html')"
    python scripts/verify_animation.py orbit.html --shots /tmp/shots

three.js payload (served with the reference loader + an import-map page)::

    python -c "import tsdynamics as ts; \
        ts.Lorenz().integrate(final_time=50, dt=0.01).after(5)\
          .to_plot_spec(animate=True).render('threejs', path='lorenz.json')"
    python scripts/verify_animation.py lorenz.json --shots /tmp/shots3d

Exit code is 0 when ANIMATING, 2 when STATIC, so it doubles as a CI-style gate
(on a machine with a browser).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import socket
import sys
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

#: Chrome flags that enable software WebGL in headless mode (gl3d / three.js).
CHROME_ARGS = [
    "--no-sandbox",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swapchains",
    "--ignore-gpu-blocklist",
    "--enable-webgl",
    "--disable-gpu-sandbox",
]

#: The import-map preview page used to render a three.js payload with the
#: reference loader. ``{loader}`` is the loader URL, ``{payload}`` the JSON URL.
_THREEJS_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{{margin:0;height:100%}}#viewer{{width:900px;height:700px}}</style>
<script type="importmap">
{{ "imports": {{
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
}} }}
</script></head><body><div id="viewer"></div>
<script type="module">
  import {{ renderThreejsPayload }} from "{loader}";
  const payload = await (await fetch("{payload}")).json();
  // autoRotate OFF: any motion then proves the GEOMETRY reveal, not camera spin.
  renderThreejsPayload(document.querySelector("#viewer"), payload, {{ autoRotate: false }});
  console.info("verify: rendered threejs payload");
</script></body></html>
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def _serve(directory: pathlib.Path):
    """Serve ``directory`` over http on a free port for the lifetime of the block."""
    port = _free_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()


def _shoot(url: str, *, settle_ms: int, gap_ms: int, n_shots: int) -> tuple[list, list[str]]:
    console: list[str] = []
    shots: list = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=CHROME_ARGS)
        page = browser.new_page(viewport={"width": 900, "height": 700})
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: console.append(f"PAGEERROR: {e}"))
        page.goto(url, wait_until="load")
        page.wait_for_timeout(settle_ms)
        per = max(1, gap_ms // max(1, n_shots))
        for _ in range(n_shots):
            arr = np.asarray(
                Image.open(io.BytesIO(page.screenshot())).convert("RGB"), dtype=np.int16
            )
            shots.append(arr)
            page.wait_for_timeout(per)
        browser.close()
    return shots, console


def _url_for(target: pathlib.Path) -> tuple[str, object]:
    """Return (url, server-context) for a plotly HTML file or a threejs payload."""
    if target.suffix == ".html":
        return target.resolve().as_uri(), contextlib.nullcontext()
    if target.suffix == ".json":
        loader = pathlib.Path("docs/_static/tsdyn-threejs-loader.js").resolve()
        if not loader.exists():
            sys.exit(f"three.js loader not found at {loader} — run from the repo root.")
        # Serve a throwaway temp dir (loader + preview page + a copy of the
        # payload) so the user's directory is never littered.
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="tsd-threejs-"))
        (tmp / "loader.js").write_text(loader.read_text())
        (tmp / "payload.json").write_text(target.resolve().read_text())
        (tmp / "preview.html").write_text(
            _THREEJS_PAGE.format(loader="./loader.js", payload="./payload.json")
        )
        ctx = _serve(tmp)
        base = ctx.__enter__()
        return f"{base}/preview.html", ctx
    sys.exit(f"unsupported target {target} — pass a .html (plotly) or .json (threejs payload).")


def main() -> int:
    """Parse args, render the export in headless Chrome, and print the verdict."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="plotly .html export OR three.js .json payload")
    ap.add_argument("--gap-ms", type=int, default=3000, help="total window sampled (ms)")
    ap.add_argument("--settle-ms", type=int, default=1800, help="wait before the first shot")
    ap.add_argument("--n-shots", type=int, default=5)
    ap.add_argument("--shots", default=None, help="dir to save the PNG frames")
    ap.add_argument(
        "--threshold", type=int, default=150, help="min changed px/frame to call it animating"
    )
    args = ap.parse_args()

    target = pathlib.Path(args.target)
    url, ctx = _url_for(target)
    try:
        shots, console = _shoot(
            url, settle_ms=args.settle_ms, gap_ms=args.gap_ms, n_shots=args.n_shots
        )
    finally:
        if hasattr(ctx, "__exit__"):
            ctx.__exit__(None, None, None)

    deltas = [
        int((np.abs(a - b).max(axis=2) > 12).sum()) for a, b in zip(shots, shots[1:], strict=False)
    ]
    if args.shots:
        d = pathlib.Path(args.shots)
        d.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(shots):
            Image.fromarray(s.astype("uint8")).save(d / f"shot_{i}.png")

    print("=== browser console ===")
    for line in console[:25]:
        print("  ", line)
    print("=== inter-frame changed pixels ===", deltas)
    errs = [c for c in console if c.startswith("PAGEERROR") or "react failed" in c]
    if errs:
        print("!! JS errors detected:", *errs, sep="\n   ")
    animating = any(d > args.threshold for d in deltas)
    print(f"\nVERDICT: {'ANIMATING ✅' if animating else 'STATIC ❌'}")
    return 0 if animating and not errs else 2


if __name__ == "__main__":
    sys.exit(main())
