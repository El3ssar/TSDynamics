r"""
Build-time **interactive three.js viewers** for the per-system documentation pages.

Where :mod:`figures` renders a *static* PNG of each attractor, this module emits a
self-contained, **live** WebGL viewer: a ``PlotSpec`` of the attractor is lowered
through the in-tree ``threejs`` data-export backend to a BufferGeometry payload, the
payload is inlined into a tiny HTML document that boots the canonical reference
loader (:file:`docs/_static/tsdyn-threejs-loader.js`), and the system page embeds
that document in an ``<iframe>``.  The result is the "animated attractor you can
orbit while it plays" — a reveal comet over a faint full-curve / full-cloud
backdrop, with ``OrbitControls`` running in its own loop so the mouse can rotate the
scene *while the comet sweeps*.

Dispatch table
--------------
:func:`render_html` dispatches on **family + dimension + ``_field_shape``** and
returns the viewer HTML, or ``None`` so the caller falls back to the static figure:

==========================  ============================================================
System                      Viewer
==========================  ============================================================
ODE, ``dim == 3``           3-D reveal comet (the original viewer).
ODE, ``dim == 2``           2-D reveal comet (planar curve, no axes).
ODE, ``dim >= 4``           **None** — a spatial / high-dim system; a 3-component
  (or any ``_field_shape``)   projection is not an honest attractor, so the page keeps
                            the static field / spacetime / projection figure.
DDE (``dim == 1``)          2-D **delay embedding** :math:`x(t)` vs :math:`x(t-\\tau)`
                            using the system's own ``\\tau``, animated as a comet.
Map, ``dim == 3``           3-D point-scatter reveal comet (a trailing swarm).
Map, ``dim == 2``           2-D point-scatter reveal comet.
Map, ``dim == 1``           2-D **return-map** scatter :math:`x_n` vs :math:`x_{n+1}`
                            (the recognizable parabola / tent), animated.
==========================  ============================================================

A LINE comet sweeps a drawn curve (ODE / DDE); a POINTS comet is a trailing swarm
over the static attractor cloud (maps) — both via the shared loader (the loader
animates ``line`` *and* ``points`` geometries identically off one master clock).

Self-containment & dependencies
-------------------------------
The emitted HTML pulls **three.js from a CDN** (jsDelivr) via an ES-module import
map, so it ships with no vendored JS and ``import tsdynamics`` stays plotting-free.
The payload is **inlined** (no second fetch); the loader is **referenced** at its
canonical ``_static`` URL (one copy, so a loader fix updates every viewer).  A
``<noscript>`` / WebGL-failure path falls back to the static PNG, so the page
degrades gracefully.

Caching
-------
A content-addressed cache under ``.cache/docs-threejs`` keyed by
``sha256(class source ‖ overrides ‖ this module's knobs)`` means only new or changed
systems ever re-integrate; CI persists the directory between builds (mirroring
:mod:`figures`).  Per-system failures soft-fail (the page falls back to the PNG).

Environment
-----------
``TSD_DOCS_FIGURES=0`` skips every heavy render (returns ``None``); ``TSD_DOCS_ONLY``
is honoured by the autogen hook upstream (only the named systems reach this module).
Every render path is wrapped — an unavailable ``threejs`` renderer, a failed
integration, or a divergent map all soft-fail to ``None`` rather than raise.
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import inspect
import json
import os
import pathlib

import figures  # docs/_tooling sibling — reuse its robust trajectory acquisition
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "docs-threejs"

#: Bump when the emitted HTML or payload shaping materially changes (cache buster).
VIEWER_VERSION = "3"

#: CDN three.js build (pinned) — matches docs/visualization/threejs-export.md.
_THREE_VERSION = "0.160.0"
_THREE_CDN = f"https://cdn.jsdelivr.net/npm/three@{_THREE_VERSION}"

#: Cap the vertex count: enough for a smooth comet, small enough to keep the
#: inlined payload light (≈ 4 k points × 3 floats × 2 buffers per system).
MAX_POINTS = 4000

#: Float precision in the inlined payload (positions / colors).  Sub-screen-pixel
#: at any sane zoom, and the sweeping comet masks the quantization — but it roughly
#: halves the JSON size versus full ``repr`` floats.
_POS_DECIMALS = 4
_COL_DECIMALS = 3

#: Reveal timing: traverse the whole attractor in ~14 s, trailing a comet of this
#: many *samples* (a line trail or a points swarm window).
_DURATION_S = 14.0
_TRAIL_SAMPLES = 600
#: Map swarms accumulate the cloud over a longer window (the shape needs the points).
_MAP_TRAIL_SAMPLES = 1200

#: Integration window for the continuous viewer trajectory (trimmed by transient).
_FINAL_TIME = 100.0
_DT = 0.01
#: Off-basin random starts retried before the system soft-fails to its static PNG.
_IC_RETRIES = 8
#: Drop this leading fraction as transient before drawing the attractor.
_TRANSIENT_FRAC = 0.15

#: DDE viewer integration window (longer, coarser dt — the 5 DDEs are all 1-D).
_DDE_FINAL_TIME = 320.0
_DDE_DT = 0.2

#: Map viewer iteration count + burn-in (mirrors :func:`figures._render_map`).
_MAP_STEPS = 12_000
_MAP_BURN = 200

# --- Brand colours (TSDynamics visual identity) ------------------------------
#: Teal attractor tube / cloud and indigo comet head — the per-system page accent.
_TEAL = "#2CC5AE"  # bright teal (the swept curve / static cloud)
_TEAL_DEEP = "#11857A"  # deep teal (kept for the brand; head/tube reference)
_INDIGO_HEAD = (0.549, 0.522, 0.949)  # #8C85F2 as an RGB triple for the comet head
#: Dark canvas background (the design's deep surface).
_BG = "#11151A"


# ---------------------------------------------------------------------------
# Eligibility / dispatch
# ---------------------------------------------------------------------------
def _figures_disabled() -> bool:
    """Whether ``TSD_DOCS_FIGURES=0`` asked us to skip every heavy render."""
    return os.environ.get("TSD_DOCS_FIGURES", "1") == "0"


def _is_field_or_spatial(entry) -> bool:
    """Whether ``entry`` is a spatial-field / high-dim system (→ static figure).

    A system is "spatial / not-an-attractor-comet" when it declares a
    ``_field_shape`` (a flattened field — Gray–Scott, Swift–Hohenberg), when its
    figure is overridden to ``"spacetime"`` / ``"field"`` (Lorenz96, KS), or when it
    carries ``dim >= 4`` (a hyperchaotic / many-body flow whose first three
    components are not an honest portrait).  All of these keep their static image —
    the per-system page picks a 3-component projection or a field/spacetime movie
    elsewhere.
    """
    opts = figures.FIG_OVERRIDES.get(entry.name, {})
    if opts.get("kind") in ("spacetime", "field"):
        return True
    if getattr(entry.cls, "_field_shape", None) is not None:
        return True
    return entry.dim is not None and entry.dim >= 4


def eligible(entry) -> bool:
    """Whether ``entry`` gets an interactive viewer (else it keeps its PNG).

    Eligible families and dims (see the module dispatch table):

    - **ODE** with a fixed ``dim`` in ``{2, 3}`` whose figure is a phase portrait
      (not a spacetime / field override) and which the shipped explicit engine can
      integrate (not stiff / discontinuous — those keep their SciPy-rendered PNG).
    - **DDE** (all 1-D today) → a 2-D delay embedding.
    - **Map** with a fixed ``dim`` in ``{1, 2, 3}`` (1-D → return-map scatter).

    Field / spatial / ``dim >= 4`` systems and variable-dim (``dim is None``)
    systems return ``False`` → the caller falls back to the static figure.
    ``TSD_DOCS_FIGURES=0`` disables every viewer.
    """
    if _figures_disabled():
        return False
    if entry.dim is None:
        return False
    if _is_field_or_spatial(entry):
        return False

    opts = figures.FIG_OVERRIDES.get(entry.name, {})
    if opts.get("skip"):
        return False

    if entry.family == "ode":
        if entry.dim not in (2, 3):
            return False
        # Stiff (implicit _default_method) / discontinuous (a "method" override)
        # systems render via SciPy, not the explicit engine — keep their PNG.
        return figures._use_engine_for_ode(entry, opts)
    if entry.family == "dde":
        return entry.dim == 1  # the catalogue's DDEs are all scalar
    if entry.family == "map":
        return entry.dim in (1, 2, 3)
    return False  # sde or anything else → static


# ---------------------------------------------------------------------------
# Trajectory / cloud acquisition (per family)
# ---------------------------------------------------------------------------
def _smooth_dt(y: np.ndarray, dt0: float, default: float) -> float:
    """Pick a smooth (non-pixelated) output dt via the sagitta heuristic.

    A coarse integration dt can leave the comet visibly faceted; a too-fine one
    bloats the payload.  :func:`estimate_dt_from_sagitta` reads the trajectory's
    own curvature and returns the largest stride whose chordal bow stays under the
    geometric tolerance.  Soft-fails to ``default`` (the heuristic needs a clean,
    finite, sufficiently long series).
    """
    try:
        from tsdynamics.analysis.sampling import estimate_dt_from_sagitta

        if y.ndim == 1:
            y = y[:, None]
        res = estimate_dt_from_sagitta(y, dt0, epsilon=0.02)
        dt = float(res.delta_t)
        if np.isfinite(dt) and dt > 0:
            return dt
    except Exception:  # noqa: BLE001 — heuristic is best-effort
        pass
    return default


def _ode_cloud(entry) -> np.ndarray | None:
    """Integrate a bounded ODE attractor for the viewer (``(n, dim)``) or ``None``.

    Uses the shipped engine with the **fixed-step** ``rk4`` kernel (a divergent
    off-basin random start raises promptly and is retried, instead of sending the
    adaptive controller into a step-shrinking spiral).  Mirrors :mod:`figures`'
    IC-retry + transient-trim contract; honours a class ``default_ic`` / a
    :data:`figures.FIG_OVERRIDES` ``ic``.  Resamples onto a sagitta-smooth output
    grid so the comet is not pixelated.
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
            y = y[drop:]
            # Re-integrate-free smoothing: pick a coarser output stride so the comet
            # is not faceted, by sub-sampling the dense march at the sagitta dt.
            smooth = _smooth_dt(y[:, : min(3, y.shape[1])], dt, dt)
            stride = max(1, int(round(smooth / dt)))
            return y[::stride]
        ic = None
    return None


def _dde_delay_embedding(entry) -> np.ndarray | None:
    """Build the 2-D delay embedding ``[x(t), x(t-τ)]`` for a (scalar) DDE.

    Integrates the system from a non-equilibrium constant-ish history (mirrors
    :func:`figures._render_dde`), reads ``τ`` from the live instance (``_delays()``),
    converts it to a sample lag at the integration ``dt``, and stacks the embedding.
    Returns ``(n, 2)`` or ``None`` on failure.
    """
    sys_obj = entry.cls()
    final_time = figures.FIG_OVERRIDES.get(entry.name, {}).get("final_time", _DDE_FINAL_TIME)
    dt = figures.FIG_OVERRIDES.get(entry.name, {}).get("dt", _DDE_DT)

    def history(s):
        return [0.8 + 0.2 * np.sin(0.2 * s)] * sys_obj.dim

    try:
        traj = sys_obj.integrate(final_time=final_time, dt=dt, history=history)
    except (RuntimeError, ValueError):
        return None
    x = np.asarray(traj.y[:, 0], dtype=float)
    if x.size < 64 or not np.all(np.isfinite(x)):
        return None
    tau = float(sys_obj._delays()[0])
    lag = max(1, int(round(tau / dt)))
    if lag >= x.size - 8:
        return None
    drop = int(_TRANSIENT_FRAC * x.size)
    emb = np.column_stack([x[lag:], x[:-lag]])[drop:]
    # Smooth + cap.
    smooth = _smooth_dt(emb, dt, dt)
    stride = max(1, int(round(smooth / dt)))
    return emb[::stride]


def _map_cloud(entry) -> np.ndarray | None:
    """Iterate a map attractor for the viewer; shape it to a drawable cloud.

    - ``dim == 1`` → the **return map** ``[x_n, x_{n+1}]`` (the recognizable
      parabola / tent), a 2-D scatter.
    - ``dim == 2`` → the iterate cloud as-is.
    - ``dim == 3`` → the 3-D iterate cloud.

    Mirrors :func:`figures._render_map`'s iterate + burn-in contract.  Returns the
    cloud array or ``None`` on a non-finite / too-short run.
    """
    sys_obj = entry.cls()
    try:
        traj = sys_obj.iterate(steps=_MAP_STEPS, max_retries=15)
    except (RuntimeError, ValueError):
        return None
    y = np.asarray(traj.y[_MAP_BURN:], dtype=float)
    if y.shape[0] < 64 or not np.all(np.isfinite(y)):
        return None
    if entry.dim == 1:
        col = y[:, 0]
        return np.column_stack([col[:-1], col[1:]])
    return y


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------
def _axis_labels(entry, n: int) -> list[str]:
    """First ``n`` component names (``variables`` ClassVar) or ``x/y/z`` defaults."""
    names = list(getattr(entry.cls, "variables", None) or [])
    default = ["x", "y", "z"]
    return [names[i] if i < len(names) else default[i] for i in range(n)]


def _kind_for(entry):
    """Return the ``(spec_kind, mark_kind, ndim, is_line)`` tuple for ``entry``.

    ODE / DDE attractors are **lines** (a swept curve); maps are **scatter**
    point-clouds (a trailing swarm).  The viewer ndim is 3 only for a 3-D ODE /
    3-D map; everything else (2-D ODE, DDE embedding, 1-D/2-D map) is 2-D.
    """
    from tsdynamics.viz.spec import PlotKind

    if entry.family == "map":
        # 3-D iterate cloud → 3-D scatter; else 2-D scatter.
        if entry.dim == 3:
            return PlotKind.PHASE_PORTRAIT_3D, PlotKind.SCATTER, 3, False
        return PlotKind.PHASE_PORTRAIT_2D, PlotKind.SCATTER, 2, False
    if entry.family == "dde":
        return PlotKind.PHASE_PORTRAIT_2D, PlotKind.LINE, 2, True
    # ode
    if entry.dim == 3:
        return PlotKind.PHASE_PORTRAIT_3D, PlotKind.LINE3D, 3, True
    return PlotKind.PHASE_PORTRAIT_2D, PlotKind.LINE, 2, True


def _cloud_for(entry) -> np.ndarray | None:
    """Acquire the drawable cloud for ``entry`` (per family), or ``None``."""
    if entry.family == "ode":
        return _ode_cloud(entry)
    if entry.family == "dde":
        return _dde_delay_embedding(entry)
    if entry.family == "map":
        return _map_cloud(entry)
    return None


def _build_payload(entry) -> dict | None:
    """Integrate / iterate ``entry`` and lower an animated attractor to a payload.

    Acquires the cloud (per family), downsamples to :data:`MAX_POINTS`, builds the
    family-appropriate spec (a LINE/LINE3D comet for flows/DDEs, a SCATTER swarm for
    maps), stamps a reveal :class:`Animation` with the brand head, and renders the
    in-tree ``threejs`` payload.  Returns ``None`` on any failure (the page then
    keeps its static figure).
    """
    from tsdynamics.viz.spec import Axis, Layer, PlotSpec

    try:
        cloud = _cloud_for(entry)
    except Exception:  # noqa: BLE001 — soft-fail to the static figure
        return None
    if cloud is None or cloud.ndim != 2 or len(cloud) < 8:
        return None

    spec_kind, mark_kind, ndim, is_line = _kind_for(entry)
    pts = cloud[:, :ndim]
    if pts.shape[1] < ndim:
        return None

    # Ceil-division stride so the kept vertex count never exceeds MAX_POINTS
    # (a floor stride leaves up to 2× the cap; the inlined JSON must stay light).
    stride = max(1, -(-len(pts) // MAX_POINTS))
    pts = pts[::stride]
    if len(pts) < 8:
        return None
    color = np.linspace(0.0, 1.0, len(pts))

    labels = _axis_labels(entry, ndim)
    data = {"x": pts[:, 0], "y": pts[:, 1], "c": color}
    axes = {"x": Axis(label=labels[0]), "y": Axis(label=labels[1])}
    if ndim == 3:
        data["z"] = pts[:, 2]
        axes["z"] = Axis(label=labels[2])

    spec = PlotSpec(
        kind=spec_kind,
        ndim=ndim,
        aspect="equal",
        title=entry.name,
        layers=[Layer(mark_kind, data)],
        **axes,
    )
    # Brand colours: teal swept curve / cloud, indigo comet head.  ``recolor`` sets
    # the layer's base colour; the per-vertex time colour still rides when present,
    # but the swarm/curve material colour falls back to teal where it does not.
    with contextlib.suppress(Exception):  # recolor is a convenience, not load-bearing
        spec.recolor(_TEAL)

    trail = _MAP_TRAIL_SAMPLES if entry.family == "map" else _TRAIL_SAMPLES
    spec.animate(duration=_DURATION_S, loop=True)
    spec.trail(("steps", trail))
    spec.head(True, size=8.0, color="#8C85F2")
    try:
        payload = spec.render("threejs", raw=True)
    except Exception:  # noqa: BLE001 — renderer unavailable / declined
        return None

    # The threejs exporter only emits ``metadata.animation`` for *line* geometries
    # (its reveal comet sweeps a curve).  A map's SCATTER lowers to a ``points``
    # geometry, so the exporter drops the animation and ships a static cloud.  Our
    # loader, however, *does* animate a points geometry (a trailing swarm), so for a
    # points-only payload we synthesise the same animation block the loader reads —
    # from the spec's own directive — to drive that swarm.
    if not is_line:
        payload = _synthesize_points_animation(
            payload, duration=_DURATION_S, loop=True, trail=trail
        )
    payload = _ensure_head_color(payload)
    return _round_payload(payload)


def _synthesize_points_animation(payload: dict, *, duration: float, loop: bool, trail: int) -> dict:
    """Stamp a reveal-animation block onto a points-only payload (a map swarm).

    Mirrors the schema the threejs exporter emits for a line comet — the loader's
    ``installAnimation`` reads ``n_samples`` / ``duration`` / ``loop`` /
    ``trail_length_samples`` / ``head`` / ``head_size`` / ``head_color`` and drives a
    points geometry as a trailing swarm (``buildPointsComet``).  ``n_samples`` is the
    largest geometry's vertex count (the loader clamps each comet to its own length).
    No-op (returns the payload unchanged) if an animation block already exists.
    """
    meta = payload.setdefault("metadata", {})
    if isinstance(meta.get("animation"), dict):
        return payload
    geoms = payload.get("geometries", [])
    n_samples = 0
    for g in geoms:
        if g.get("type") == "points":
            n_samples = max(n_samples, len(g.get("positions", [])) // 3)
    if n_samples < 2:
        return payload  # nothing to reveal — leave it static
    meta["animation"] = {
        "fps": 30.0,
        "duration": float(duration),
        "n_frames": None,
        "loop": bool(loop),
        "pingpong": False,
        "trail_length_samples": int(trail),
        "head": True,
        "head_size": 8.0,
        "head_color": list(_INDIGO_HEAD),
        "n_samples": int(n_samples),
    }
    return payload


def _ensure_head_color(payload: dict) -> dict:
    """Force the indigo brand head colour into the animation metadata.

    The threejs loader reads ``metadata.animation.head_color`` (an RGB triple) for
    the comet head; if the spec did not surface it, stamp the brand indigo so every
    viewer's head is on-brand.
    """
    meta = payload.get("metadata")
    if (
        isinstance(meta, dict)
        and isinstance(meta.get("animation"), dict)
        and meta["animation"].get("head_color") is None
    ):
        meta["animation"]["head_color"] = list(_INDIGO_HEAD)
    return payload


def _round_payload(payload: dict) -> dict:
    """Round the bulky float buffers in place to shrink the inlined JSON."""
    for geom in payload.get("geometries", []):
        if "positions" in geom:
            geom["positions"] = [round(float(v), _POS_DECIMALS) for v in geom["positions"]]
        if "colors" in geom:
            geom["colors"] = [round(float(v), _COL_DECIMALS) for v in geom["colors"]]
    return payload


# ---------------------------------------------------------------------------
# HTML wrapping + caching
# ---------------------------------------------------------------------------
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
            _MAP_TRAIL_SAMPLES,
            _FINAL_TIME,
            _DT,
            _DDE_FINAL_TIME,
            _DDE_DT,
            _MAP_STEPS,
            _MAP_BURN,
            _IC_RETRIES,
            _TRANSIENT_FRAC,
            _TEAL,
            _BG,
            entry.family,
            entry.dim,
        )
    )
    return hashlib.sha256((cls_src + opts + knobs).encode()).hexdigest()[:20]


def _html(entry, payload: dict) -> str:
    """Wrap ``payload`` in a self-contained viewer document for an ``<iframe>``.

    The document carries an ES-module import map (three + OrbitControls from the
    CDN), the inlined payload, and a boot script that calls the canonical reference
    loader (referenced at ``../../_static/`` — one shared copy).  A WebGL/JS failure
    or ``<noscript>`` falls back to the static PNG at ``../figures/systems/<name>.png``
    (both paths are fixed relative to the viewer's own ``assets/threejs/`` location).
    The dark brand canvas (``#11151A``) is passed to the loader as the scene
    background so every family's viewer matches the design surface.
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
  html, body {{ margin: 0; height: 100%; background: {_BG}; overflow: hidden; }}
  #viewer {{ position: absolute; inset: 0; }}
  #fallback {{
    position: absolute; inset: 0; display: none; object-fit: contain;
    width: 100%; height: 100%; background: {_BG};
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
    renderThreejsPayload(viewer, payload, {{ background: "{_BG}" }});
  }} catch (err) {{
    degrade(err);
  }}
</script>
</body>
</html>
"""


def render_html(entry) -> str | None:
    """Return the viewer HTML for ``entry`` (cached on disk), or ``None``.

    Cache hit → read the cached HTML; miss → integrate / iterate, lower, wrap, and
    cache.  Returns ``None`` for an ineligible system (field / spatial / ``dim>=4``,
    a map/flow the engine cannot render, ``TSD_DOCS_FIGURES=0``) or a soft failure —
    the page then falls back to the static PNG via :mod:`figures`.  Never raises.
    """
    try:
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
    except Exception:  # noqa: BLE001 — a viewer must never break the docs build
        return None
