"""
Build-time **interactive three.js viewers** for the per-system documentation pages.

Where :mod:`figures` renders a *static* PNG of each attractor, this module emits a
self-contained, **live** WebGL viewer: a ``PlotSpec`` of the attractor is lowered
through the in-tree ``threejs`` data-export backend to a BufferGeometry payload, the
payload is inlined into a tiny HTML document that boots the canonical reference
loader (:file:`docs/_static/tsdyn-threejs-loader.js`), and the system page embeds
that document in an ``<iframe>``.  The result is the "animated attractor you can
orbit while it plays" — a ``setDrawRange`` reveal comet over a faint full-curve
backdrop, with ``OrbitControls`` running in its own loop so the mouse can rotate the
scene *while the comet sweeps* (the deferred half of the plotly pause-on-drag work).

Eligibility
-----------
Only **3-D ODE attractors** get a viewer — a flow with ``dim >= 3`` rendered as a
``PHASE_PORTRAIT_3D`` (a ``LINE3D`` mark the reveal comet can sweep).  Spacetime /
spatial-field systems (a ``"spacetime"`` / ``"field"`` :data:`figures.FIG_OVERRIDES`
kind), skipped systems, maps (point clouds — no comet) and DDEs keep their static
figure.  :func:`eligible` is the single gate; everything else falls through to
:mod:`figures`.

Self-containment & dependencies
-------------------------------
The emitted HTML pulls **three.js from a CDN** (jsDelivr, the same build the
``threejs-export`` demo and the plotly export use) via an ES-module import map, so it
ships with no vendored JS and ``import tsdynamics`` stays plotting-free.  The payload
is **inlined** (no second fetch); the loader is **referenced** at its canonical
``_static`` URL (one copy, so a loader fix updates every viewer).  A ``<noscript>`` /
WebGL-failure path falls back to the static PNG, so the page degrades gracefully.

Caching
-------
A content-addressed cache under ``.cache/docs-threejs`` keyed by
``sha256(class source ‖ overrides ‖ this module's knobs)`` means only new or changed
systems ever re-integrate; CI persists the directory between builds (mirroring
:mod:`figures`).  Per-system failures soft-fail (the page falls back to the PNG).
"""

from __future__ import annotations

import hashlib
import html
import inspect
import json
import pathlib

import figures  # docs/_tooling sibling — reuse its robust trajectory acquisition
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "docs-threejs"

#: Bump when the emitted HTML or payload shaping materially changes (cache buster).
VIEWER_VERSION = "2"

#: CDN three.js build (pinned) — matches docs/visualization/threejs-export.md.
_THREE_VERSION = "0.160.0"
_THREE_CDN = f"https://cdn.jsdelivr.net/npm/three@{_THREE_VERSION}"

#: Cap the line vertex count: enough for a smooth comet, small enough to keep the
#: inlined payload light (≈ 4 k points × 3 floats × 2 buffers per system).
MAX_POINTS = 4000

#: Float precision in the inlined payload (positions / colors).  Sub-screen-pixel
#: at any sane zoom, and the sweeping comet masks the quantization — but it roughly
#: halves the JSON size versus full ``repr`` floats.
_POS_DECIMALS = 4
_COL_DECIMALS = 3

#: Reveal timing: traverse the whole attractor in ~14 s, trailing a 600-sample comet.
_DURATION_S = 14.0
_TRAIL_SAMPLES = 600

#: Integration window for the viewer trajectory (a generous attractor, then trimmed).
_FINAL_TIME = 100.0
_DT = 0.01
#: Off-basin random starts retried before the system soft-fails to its static PNG.
_IC_RETRIES = 8
#: Drop this leading fraction as transient before drawing the attractor.
_TRANSIENT_FRAC = 0.15


def eligible(entry) -> bool:
    """Whether ``entry`` gets an interactive 3-D viewer (else it keeps its PNG).

    Eligible == a **3-D ODE attractor the shipped explicit engine can integrate**:
    an ``"ode"`` family system with a fixed ``dim >= 3`` whose figure is a phase
    portrait — i.e. **not** a spacetime / spatial-field override, not explicitly
    skipped, and **not** one of the stiff / discontinuous systems :mod:`figures`
    routes to its SciPy fallback (an explicit fixed-step march would blow up on a
    stiff RHS or stumble across a ``sign``/``abs`` jump).  Variable-dim systems
    (``dim is None``), maps and DDEs all fall through to the static figure.
    """
    if entry.family != "ode":
        return False
    opts = figures.FIG_OVERRIDES.get(entry.name, {})
    if opts.get("skip") or opts.get("kind") in ("spacetime", "field"):
        return False
    if entry.dim is None or entry.dim < 3:
        return False
    # Stiff (implicit _default_method) / discontinuous (a "method" override) systems
    # render their figure via SciPy, not the engine — keep their PNG, no viewer.
    return figures._use_engine_for_ode(entry, opts)


def _viewer_trajectory(entry) -> np.ndarray | None:
    """Integrate a bounded attractor for the viewer (``(n, dim)`` array) or ``None``.

    Uses the shipped engine with the **fixed-step** ``rk4`` kernel rather than the
    adaptive default: a divergent off-basin random start then raises promptly
    (a handful of cheap steps) and is retried, instead of sending the adaptive
    step-controller into a minutes-long step-shrinking spiral as it chases a
    solution racing to infinity.  Mirrors :mod:`figures`' IC-retry + transient-trim
    contract; honours a class ``default_ic`` / a :data:`figures.FIG_OVERRIDES` ``ic``.
    """
    opts = figures.FIG_OVERRIDES.get(entry.name, {})
    final_time = opts.get("final_time", _FINAL_TIME)
    dt = opts.get("dt", _DT)
    rng = np.random.default_rng(42)

    sys_obj = entry.cls()
    ic = figures._resolve_ic(sys_obj, opts.get("ic"))
    for attempt in range(_IC_RETRIES):
        if ic is None or attempt > 0:
            ic = sys_obj.resolve_ic(rng.uniform(0.0, 1.0, sys_obj.dim))
        try:
            traj = sys_obj.integrate(
                final_time=final_time,
                dt=dt,
                ic=np.asarray(ic, dtype=float),
                backend="interp",
                method="rk4",
            )
        except (RuntimeError, ValueError):  # divergence / off-basin start
            ic = None
            continue
        y = traj.y
        if len(y) > 50 and np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e6:
            drop = int(_TRANSIENT_FRAC * len(y))
            return y[drop:]
        ic = None
    return None


def cache_key(entry) -> str:
    """Content hash: class source + figure overrides + this module's knobs."""
    cls_src = inspect.getsource(entry.cls)
    opts = repr(sorted(figures.FIG_OVERRIDES.get(entry.name, {}).items()))
    knobs = "|".join(
        str(k)
        for k in (
            VIEWER_VERSION,
            MAX_POINTS,
            _POS_DECIMALS,
            _COL_DECIMALS,
            _DURATION_S,
            _TRAIL_SAMPLES,
            _FINAL_TIME,
            _DT,
            _IC_RETRIES,
            _TRANSIENT_FRAC,
        )
    )
    return hashlib.sha256((cls_src + opts + knobs).encode()).hexdigest()[:20]


def _axis_labels(entry, n: int = 3) -> list[str]:
    """Return the first ``n`` component names (``variables`` ClassVar) or ``x/y/z``."""
    names = list(getattr(entry.cls, "variables", None) or [])
    default = ["x", "y", "z"]
    return [names[i] if i < len(names) else default[i] for i in range(n)]


def _build_payload(entry) -> dict | None:
    """Integrate ``entry`` and lower a colored, animated 3-D attractor to a payload.

    Integrates a bounded attractor via :func:`_viewer_trajectory` (the fixed-step
    engine march with IC-retry + transient trimming), downsamples to
    :data:`MAX_POINTS`, builds a ``PHASE_PORTRAIT_3D`` spec coloured by time, stamps
    a reveal :class:`Animation`, and renders the in-tree ``threejs`` payload.
    Returns ``None`` on any failure (the page then keeps its static figure).
    """
    from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

    try:
        y = _viewer_trajectory(entry)
    except Exception:  # noqa: BLE001 — soft-fail to the static figure
        return None
    if y is None or y.ndim != 2 or y.shape[1] < 3 or len(y) < 8:
        return None

    pts = y[:, :3]
    stride = max(1, len(pts) // MAX_POINTS)
    pts = pts[::stride]
    color = np.linspace(0.0, 1.0, len(pts))

    labels = _axis_labels(entry)
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        ndim=3,
        aspect="equal",
        title=entry.name,
        x=Axis(label=labels[0]),
        y=Axis(label=labels[1]),
        z=Axis(label=labels[2]),
        layers=[
            Layer(
                PlotKind.LINE3D,
                {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "c": color},
            )
        ],
    )
    spec.animate(duration=_DURATION_S, loop=True)
    spec.trail(("steps", _TRAIL_SAMPLES))
    spec.head(True, size=7.0)
    try:
        payload = spec.render("threejs", raw=True)
    except Exception:  # noqa: BLE001
        return None
    return _round_payload(payload)


def _round_payload(payload: dict) -> dict:
    """Round the bulky float buffers in place to shrink the inlined JSON."""
    for geom in payload.get("geometries", []):
        if "positions" in geom:
            geom["positions"] = [round(float(v), _POS_DECIMALS) for v in geom["positions"]]
        if "colors" in geom:
            geom["colors"] = [round(float(v), _COL_DECIMALS) for v in geom["colors"]]
    return payload


def _html(entry, payload: dict) -> str:
    """Wrap ``payload`` in a self-contained viewer document for an ``<iframe>``.

    The document carries an ES-module import map (three + OrbitControls from the
    CDN), the inlined payload, and a boot script that calls the canonical reference
    loader (referenced at ``../../_static/`` — one shared copy).  A WebGL/JS failure
    or ``<noscript>`` falls back to the static PNG at ``../figures/systems/<name>.png``
    (both paths are fixed relative to the viewer's own ``assets/threejs/`` location).
    """
    payload_json = json.dumps(payload, separators=(",", ":"))
    png = f"../figures/systems/{entry.name}.png"
    alt = html.escape(f"{entry.name} attractor")
    title = html.escape(f"{entry.name} — interactive attractor")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #0b1020; overflow: hidden; }}
  #viewer {{ position: absolute; inset: 0; }}
  #fallback {{
    position: absolute; inset: 0; display: none; object-fit: contain;
    width: 100%; height: 100%; background: #0b1020;
  }}
</style>
<script type="importmap">
{{
  "imports": {{
    "three": "{_THREE_CDN}/build/three.module.js",
    "three/addons/": "{_THREE_CDN}/examples/jsm/"
  }}
}}
</script>
</head>
<body>
<div id="viewer"></div>
<img id="fallback" src="{png}" alt="{alt}" />
<noscript>
  <img src="{png}" alt="{alt}" style="width:100%;height:100%;object-fit:contain" />
</noscript>
<script type="module">
  const payload = {payload_json};
  const viewer = document.getElementById("viewer");
  const fallback = document.getElementById("fallback");
  function degrade(err) {{
    if (viewer) viewer.style.display = "none";
    if (fallback) fallback.style.display = "block";
    console.warn("tsd-threejs: viewer unavailable, showing static figure:", err);
  }}
  try {{
    const {{ renderThreejsPayload }} = await import("../../_static/tsdyn-threejs-loader.js");
    renderThreejsPayload(viewer, payload);
  }} catch (err) {{
    degrade(err);
  }}
</script>
</body>
</html>
"""


def render_html(entry) -> str | None:
    """Return the viewer HTML for ``entry`` (cached on disk), or ``None``.

    Cache hit → read the cached HTML; miss → integrate, lower, wrap, and cache.
    Returns ``None`` for an ineligible system or a soft failure (the page then
    falls back to the static PNG via :mod:`figures`).
    """
    if not eligible(entry):
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{entry.name}-{cache_key(entry)}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    payload = _build_payload(entry)
    if payload is None:
        return None
    doc = _html(entry, payload)
    cached.write_text(doc, encoding="utf-8")
    return doc
