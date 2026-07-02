r"""
Build-time **interactive three.js viewers** for the per-system documentation pages.

Where :mod:`figures` renders a *static* PNG of each attractor, this module emits a
self-contained, **live** WebGL viewer: a ``PlotSpec`` of the attractor is lowered
through the in-tree ``threejs`` data-export backend to a BufferGeometry payload, the
payload is inlined into a tiny HTML document that boots the canonical reference
loader (:file:`docs/_static/tsdyn-threejs-loader.js`), and the system page embeds
that document in an ``<iframe>``.  The result is the "animated attractor you can
orbit while it plays" — a reveal comet over a faint full-curve backdrop, with
``OrbitControls`` running in its own loop so the mouse can rotate the scene *while
the comet sweeps*.

Dispatch (who gets an animated viewer)
--------------------------------------
:func:`render_html` dispatches on **family + dimension + editorial viewer config**:

======================================  ====================================================
System                                  Viewer
======================================  ====================================================
ODE ``dim in {2, 3}`` (non-forced)      3-D / 2-D reveal comet of the state itself.
ODE ``dim >= 4`` **with a projection**  3-D reveal comet of the editorial 3-component
                                        ``projection``; a ``projection2`` yields a SECOND
                                        viewer (``render_html(rec, second=True)``) so the
                                        page shows two faces of a high-dim flow.
ODE with a ``viewer`` editorial block   Honoured verbatim — ``components`` (a 2-/3-index
  (Group B "weird" systems)             projection of the honest attractor), ``wrap``
                                        (wrap listed coords mod 2π for a torus flow),
                                        ``ic`` / ``final_time`` / ``method`` overrides, or
                                        ``mode: "static"`` / ``"drop"`` to defer to a
                                        curated static figure.
DDE (``dim == 1``)                      2-D **delay embedding** :math:`x(t)` vs
                                        :math:`x(t-\tau)`, animated as a comet.
Map (any dim)                           **None** — a static scatter reads better than a
                                        trailing swarm (see :mod:`figures`).
SDE / field / spatial                   **None** — static sample path / field image.
======================================  ====================================================

Every attractor integration samples at the **sagitta-``dt``** returned by
:func:`plot_dt.choose_plot_dt` (``epsilon=0.1``, the maintainer's universal
"no pixelated attractor" rule).  The selector is called with its per-family
*fine-pilot* default step (``FINE_PILOT_DT``: 0.002 for ODEs) — **never** a coarse
``dt0`` override, which was the pixelation bug: a sagitta search can only report a
step no finer than the pilot it ran, so a fast attractor (Chen, DequanLi, QiChen)
integrated at 0.01 just echoed 0.01 back and stayed faceted, whereas the fine pilot
discovers the true 0.002 / 0.006 output step.  We then integrate the viewer
trajectory at that fine step and sub-sample to the sagitta step — and, because even
a *geometrically* smooth coarse dt can leave a slow attractor with too few vertices
to read as a swept line, a :data:`_MIN_DRAW_SAMPLES` floor shrinks the stride
(drawing *finer* than sagitta, never coarser) so every comet has enough points.

Two views for a high-dim flow
-----------------------------
:func:`viewer_payloads` is the generator's front door: it returns *every* view of a
system in one call (``[{"suffix": "", "html": …}, {"suffix": "-b", "html": …}]``),
the ``-b`` present only for a 4-D-plus flow with an editorial ``projection2`` whose
primary rendered.  :func:`render_html` stays as the single-view back-compat entry.

Self-containment, caching & environment mirror the previous design: three.js is
pulled from a pinned CDN via an ES-module import map, the payload is inlined, the
shared loader is referenced at its ``_static`` URL, results are content-addressed
under ``.cache/docs-threejs``, and ``TSD_DOCS_FIGURES=0`` skips every heavy render.
Every render path soft-fails to ``None`` / ``[]`` (the page falls back to its
static PNG).
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import inspect
import json
import os
import pathlib

import figures  # docs/_tooling sibling — reuse its robust IC / trajectory acquisition
import numpy as np
import plot_dt as _plot_dt  # the ONE sagitta-dt selector both renderers call

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "docs-threejs"

#: Canonical reference loader (source of truth) + the site URI the viewer iframes
#: import it from.
LOADER_SRC = ROOT / "docs" / "_static" / "tsdyn-threejs-loader.js"
LOADER_URI = "_static/tsdyn-threejs-loader.js"

#: Bump when the emitted HTML or payload shaping materially changes (cache buster).
VIEWER_VERSION = "6"

#: CDN three.js build (pinned) — matches docs/visualization/threejs-export.md.
_THREE_VERSION = "0.160.0"
_THREE_CDN = f"https://cdn.jsdelivr.net/npm/three@{_THREE_VERSION}"

#: Cap the vertex count: enough for a smooth comet, small enough to keep the
#: inlined payload light.
MAX_POINTS = 5000

#: Float precision in the inlined payload (positions / colors).
_POS_DECIMALS = 4
_COL_DECIMALS = 3

#: Reveal timing: traverse the whole attractor in ~14 s, trailing a comet of this
#: many *samples*.
_DURATION_S = 14.0
_TRAIL_SAMPLES = 600

#: Integration windows (the pilot / sagitta-dt refine the *output* step within these).
_FINAL_TIME = 90.0
_DDE_FINAL_TIME = 320.0
#: Nominal coarse output dt (kept only for legacy DDE plumbing / cache knobs).
_DT = 0.01
_DDE_DT = 0.2
#: The **fine** integration step for the viewer march — the step we actually
#: integrate at before sub-sampling to the sagitta ``smooth_dt``.  It mirrors
#: :data:`plot_dt.FINE_PILOT_DT` (0.002 for ODEs) so a fast attractor is traced at
#: a step fine enough to represent its tight curvature; the stride then coarsens to
#: the sagitta output dt.  Never coarser than ``smooth_dt`` (that is a no-op stride).
_FINE_DT = float(_plot_dt.FINE_PILOT_DT.get("ode", 0.002))
_DDE_FINE_DT = float(_plot_dt.FINE_PILOT_DT.get("dde", 0.02))
#: Off-basin random starts retried before the system soft-fails to its static PNG.
_IC_RETRIES = 8
#: Drop this leading fraction as transient before drawing the attractor.
_TRANSIENT_FRAC = 0.2
#: A comet needs a *floor* of vertices to interpolate smoothly, independent of the
#: sagitta dt.  A slow, gently-curving attractor (Colpitts, the Sprott minimal
#: flows, Rössler) can satisfy the ε=0.1 chord criterion at a coarse dt that leaves
#: only ~70–350 samples — enough to be *geometrically* smooth but visually sparse as
#: a swept line.  Drawing **finer than** the sagitta dt is always at least as smooth
#: (sagitta is an upper bound on the step), so if the sagitta-strided cloud falls
#: below this floor we shrink the stride to reach it — never coarser, only finer.
_MIN_DRAW_SAMPLES = 1600

# --- Brand colours (TSDynamics visual identity) ------------------------------
_TEAL = "#2CC5AE"
_INDIGO_HEAD = (0.549, 0.522, 0.949)  # #8C85F2
_BG = "#0B0F14"


# ---------------------------------------------------------------------------
# Small record accessors (works for a catalogue SystemRecord *or* a bare entry)
# ---------------------------------------------------------------------------
def _viewer_cfg(entry) -> dict:
    """Return the per-system ``viewer`` editorial directive (Group B), else ``{}``."""
    cfg = getattr(entry, "viewer", None)
    return cfg if isinstance(cfg, dict) else {}


def _projection(entry, *, second: bool) -> tuple[int, ...] | None:
    """Return the chosen 3-component projection index tuple (or ``None``).

    ``second=True`` returns ``projection2`` when present (the page's second view).
    """
    proj = getattr(entry, "projection2" if second else "projection", None)
    if not proj:
        return None
    try:
        return tuple(int(i) for i in proj)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Eligibility / dispatch
# ---------------------------------------------------------------------------
def _figures_disabled() -> bool:
    """Whether ``TSD_DOCS_FIGURES=0`` asked us to skip every heavy render."""
    return os.environ.get("TSD_DOCS_FIGURES", "1") == "0"


def _is_field(entry) -> bool:
    """Whether ``entry`` is a spatial field (``_field_shape`` / field figure)."""
    opts = figures.FIG_OVERRIDES.get(entry.name, {})
    if opts.get("kind") in ("spacetime", "field"):
        return True
    return getattr(entry.cls, "_field_shape", None) is not None


def eligible(entry, *, second: bool = False) -> bool:
    """Whether ``entry`` gets an interactive viewer (else it keeps its static PNG).

    - **Maps / SDEs / spatial fields** → never (static reads better; a map is a
      static scatter, an SDE a sample path, a field an image).
    - A ``viewer`` editorial block with ``mode`` ``"static"`` / ``"drop"`` → never
      (the system is drawn as a curated static figure instead).
    - **ODE ``dim in {2, 3}``** (no growing-phase ``components`` override) →
      animate the state directly.
    - **ODE ``dim >= 4``** → animate **only** when an editorial 3-component
      ``projection`` (or ``projection2`` for the ``second`` view) is available.
    - **DDE (1-D)** → animate the delay embedding.

    ``TSD_DOCS_FIGURES=0`` disables every viewer.  A stiff ``_default_method`` no
    longer blocks a flow: the viewer marches it with the adaptive ``rk45`` kernel
    for the thumbnail (still the shipped engine).
    """
    if _figures_disabled():
        return False
    if entry.dim is None and not _is_field(entry):
        # Variable-dim non-field flows (e.g. LorenzCoupled/MultiChua) resolve an
        # effective dim in the catalogue record; a bare entry may not — treat a
        # None dim conservatively as ineligible unless a projection says otherwise.
        return _projection(entry, second=second) is not None

    cfg = _viewer_cfg(entry)
    if cfg.get("mode") in ("static", "drop"):
        return False
    opts = figures.FIG_OVERRIDES.get(entry.name, {})
    if opts.get("skip"):
        return False

    if entry.family == "ode":
        if _is_field(entry):
            return False
        proj = _projection(entry, second=second)
        if second:
            # A second viewer exists only when a projection2 was supplied.
            return proj is not None
        # Primary viewer: 2-/3-D states animate directly; 4-D+ need a projection.
        if entry.dim in (2, 3):
            return True
        return proj is not None
    if entry.family == "dde":
        return not second and entry.dim == 1
    # map / sde / anything else → static
    return False


# ---------------------------------------------------------------------------
# Trajectory / cloud acquisition
# ---------------------------------------------------------------------------
def _pilot_method(entry) -> str:
    """Resolve the engine kernel for the viewer march.

    Honours a figure ``engine_method`` / discontinuous ``method`` override, else
    ``rk45`` for a stiff-default flow and ``rk4`` for the rest (both shipped-engine
    kernels).
    """
    opts = figures.FIG_OVERRIDES.get(entry.name, {})
    if opts.get("engine_method"):
        return str(opts["engine_method"])
    if opts.get("method"):
        return str(opts["method"]).lower()
    default = str(getattr(entry.cls, "_default_method", "rk4") or "rk4").lower()
    # An implicit default (bdf/rosenbrock/trbdf2) can't drive the explicit viewer
    # march — use the adaptive rk45 (stays bounded on these non-stiff-for-plotting
    # flows) instead of raising.
    if default in ("bdf", "rosenbrock", "trbdf2", "sdirk2"):
        return "rk45"
    if default in ("rk4",):
        return "rk4"
    return "rk45"


def _wrap_components(y: np.ndarray, wrap: list[int] | None) -> np.ndarray:
    """Wrap the listed component indices onto ``[-π, π)`` (a torus / angle flow).

    A streamline whose angle grows unbounded (Arnold–Beltrami–Childress, a rotor)
    drifts off-screen; wrapping the angular coordinate mod 2π keeps the honest
    covering-space structure on screen.
    """
    if not wrap:
        return y
    y = y.copy()
    for idx in wrap:
        if 0 <= idx < y.shape[1]:
            y[:, idx] = (y[:, idx] + np.pi) % (2 * np.pi) - np.pi
    return y


def _ode_cloud(entry, *, second: bool) -> np.ndarray | None:
    """Integrate a bounded ODE attractor and shape it to a drawable cloud.

    Honours (in priority order) the ``viewer`` editorial block, then an editorial
    ``projection`` / ``projection2``, then the raw state.  Samples at the
    sagitta-``dt`` (``epsilon=0.1``) so the comet is never faceted.  Returns
    ``(n, k)`` with ``k in {2, 3}`` or ``None`` on a divergent / off-basin run.
    """
    cfg = _viewer_cfg(entry)
    opts = figures.FIG_OVERRIDES.get(entry.name, {})
    final_time = cfg.get("final_time", opts.get("final_time", _FINAL_TIME))
    method = cfg.get("method", _pilot_method(entry))
    transient = float(cfg.get("transient", _TRANSIENT_FRAC))
    wrap = cfg.get("wrap")

    # The smooth output dt (the maintainer's universal sagitta rule, ε = 0.1).
    #
    # CRITICAL: call ``choose_plot_dt`` with its per-family *fine-pilot* default
    # (``dt0=None`` → ``FINE_PILOT_DT["ode"] = 0.002``), **not** the coarse ``_DT``.
    # The sagitta search can only report a step no finer than the pilot it ran; a
    # pilot integrated at 0.01 can never discover that a fast attractor (DequanLi,
    # QiChen, …) actually needs 0.002 / 0.006 to read smooth — it just echoes 0.01
    # back and the comet stays faceted.  Feeding ``dt0=_DT`` here was the pixelation
    # bug: the whole point of the fine pilot is to find the true output dt *from
    # below*.  We then integrate the viewer trajectory at that same fine step and
    # sub-sample to ``smooth_dt`` so the drawn curve is never straightened.
    ic_override = cfg.get("ic", opts.get("ic"))
    smooth_dt = _plot_dt.choose_plot_dt(entry, epsilon=0.1)
    fine_dt = min(smooth_dt, _FINE_DT)
    stride = max(1, int(round(smooth_dt / fine_dt)))

    rng = np.random.default_rng(42)
    sys_obj = entry.cls()
    ic = figures._resolve_ic(sys_obj, ic_override)
    for attempt in range(_IC_RETRIES):
        if ic is None or attempt > 0:
            ic = sys_obj.resolve_ic(rng.uniform(0.0, 1.0, sys_obj.dim))
        try:
            traj = sys_obj.integrate(
                final_time=final_time,
                dt=fine_dt,
                ic=np.asarray(ic, dtype=float),
                backend="interp",
                method=method,
            )
        except (RuntimeError, ValueError):  # divergence / off-basin start
            ic = None
            continue
        y = traj.y
        if len(y) > 50 and np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e7:
            drop = int(transient * len(y))
            y = y[drop:]
            # Sagitta stride keeps the coarsest smooth step; shrink it if that would
            # leave too few vertices for a smooth comet (drawing finer is never worse).
            eff_stride = stride
            if len(y) // max(1, eff_stride) < _MIN_DRAW_SAMPLES:
                eff_stride = max(1, len(y) // _MIN_DRAW_SAMPLES)
            y = y[::eff_stride]
            y = _wrap_components(y, wrap)
            return _select_components(entry, y, second=second)
        ic = None
    return None


def _view_components(entry, dim: int, *, second: bool) -> list[int]:
    """Resolve the component indices for a view (the ONE priority table).

    Priority differs by view so a Group-B "weird" system that pins its **primary**
    projection with ``viewer.components`` can *still* show a genuinely different
    **second** face:

    - **primary** (``second=False``): ``viewer.components`` → editorial
      ``projection`` → the first ``min(3, dim)`` state components;
    - **second** (``second=True``): editorial ``projection2`` wins outright (so it
      is never shadowed by the primary's ``viewer.components``) → ``viewer.components``
      → the first ``min(3, dim)``.

    Always returns a valid, in-range index list of length 2 or 3.
    """
    cfg = _viewer_cfg(entry)
    comps: list | tuple | None = None
    if second:
        proj2 = _projection(entry, second=True)
        comps = list(proj2) if proj2 is not None else cfg.get("components")
    else:
        comps = cfg.get("components")
        if comps is None:
            proj = _projection(entry, second=False)
            comps = list(proj) if proj is not None else None
    if comps is None:
        return list(range(min(3, dim)))
    try:
        idx = [int(c) for c in comps]
    except (TypeError, ValueError):
        return list(range(min(3, dim)))
    idx = [i for i in idx if 0 <= i < dim]
    if len(idx) < 2:
        return list(range(min(3, dim)))
    return idx[:3]


def _select_components(entry, y: np.ndarray, *, second: bool) -> np.ndarray | None:
    """Pick the 2-/3-component view of a full-dim trajectory ``y`` ``(n, dim)``."""
    idx = _view_components(entry, y.shape[1], second=second)
    return y[:, idx]


def _dde_delay_embedding(entry) -> np.ndarray | None:
    """Build the 2-D delay embedding ``[x(t), x(t-τ)]`` for a (scalar) DDE."""
    sys_obj = entry.cls()
    opts = figures.FIG_OVERRIDES.get(entry.name, {})
    final_time = opts.get("final_time", _DDE_FINAL_TIME)
    # Same sagitta rule as the ODE path: let ``choose_plot_dt`` run its fine DDE
    # pilot (``dt0=None`` → ``FINE_PILOT_DT["dde"] = 0.02``) and integrate the
    # embedding at that fine step, sub-sampling to the smooth output dt.
    smooth_dt = _plot_dt.choose_plot_dt(entry, epsilon=0.1)
    fine_dt = min(smooth_dt, _DDE_FINE_DT)
    stride = max(1, int(round(smooth_dt / fine_dt)))

    def history(s):
        return [0.8 + 0.2 * np.sin(0.2 * s)] * sys_obj.dim

    try:
        traj = sys_obj.integrate(final_time=final_time, dt=fine_dt, history=history)
    except (RuntimeError, ValueError):
        return None
    x = np.asarray(traj.y[:, 0], dtype=float)
    if x.size < 64 or not np.all(np.isfinite(x)):
        return None
    tau = float(sys_obj._delays()[0])
    lag = max(1, int(round(tau / fine_dt)))
    if lag >= x.size - 8:
        return None
    drop = int(_TRANSIENT_FRAC * x.size)
    emb = np.column_stack([x[lag:], x[:-lag]])[drop:]
    return emb[::stride]


def _cloud_for(entry, *, second: bool) -> np.ndarray | None:
    """Acquire the drawable cloud for ``entry`` (per family), or ``None``."""
    if entry.family == "ode":
        return _ode_cloud(entry, second=second)
    if entry.family == "dde":
        return _dde_delay_embedding(entry)
    return None


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------
def _axis_labels(entry, n: int, comps: tuple[int, ...] | None) -> list[str]:
    """Component names for the chosen ``comps`` (``variables`` ClassVar) or x/y/z."""
    names = list(getattr(entry.cls, "variables", None) or [])
    default = ["x", "y", "z", "w", "v", "u"]

    def name(i: int) -> str:
        if 0 <= i < len(names):
            return names[i]
        return default[i] if i < len(default) else f"y{i}"

    idx = list(comps) if comps is not None else list(range(n))
    # Always return exactly ``n`` labels (pad with positional fallbacks) so a
    # projection shorter than the drawn ndim can never index past the label list.
    while len(idx) < n:
        idx.append(len(idx))
    return [name(int(c)) for c in idx[:n]]


def _build_payload(entry, *, second: bool) -> dict | None:
    """Integrate ``entry`` and lower an animated attractor comet to a payload."""
    from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

    try:
        cloud = _cloud_for(entry, second=second)
    except Exception:  # noqa: BLE001 — soft-fail to the static figure
        return None
    if cloud is None or cloud.ndim != 2 or len(cloud) < 8:
        return None

    ndim = min(3, cloud.shape[1])
    pts = cloud[:, :ndim]
    if pts.shape[1] < 2:
        return None

    # Downsample to MAX_POINTS (ceil-division stride so the kept count never
    # exceeds the cap; the inlined JSON must stay light).
    dstride = max(1, -(-len(pts) // MAX_POINTS))
    pts = pts[::dstride]
    if len(pts) < 8:
        return None
    color = np.linspace(0.0, 1.0, len(pts))

    spec_kind = PlotKind.PHASE_PORTRAIT_3D if ndim == 3 else PlotKind.PHASE_PORTRAIT_2D
    mark_kind = PlotKind.LINE3D if ndim == 3 else PlotKind.LINE

    # Label from the *true* system dim (the projection indices reference the full
    # state), not the already-projected cloud width — otherwise a projection like
    # (1, 2, 3) would be filtered against a 3-column cloud and lose an index.
    label_dim = entry.dim if isinstance(getattr(entry, "dim", None), int) else cloud.shape[1]
    comps = _selected_comps(entry, label_dim, second=second)
    labels = _axis_labels(entry, ndim, comps)
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
    with contextlib.suppress(Exception):
        spec.recolor(_TEAL)

    spec.animate(duration=_DURATION_S, loop=True)
    spec.trail(("steps", _TRAIL_SAMPLES))
    spec.head(True, size=8.0, color="#8C85F2")
    try:
        payload = spec.render("threejs", raw=True)
    except Exception:  # noqa: BLE001 — renderer unavailable / declined
        return None

    payload = _ensure_head_color(payload)
    return _round_payload(payload)


def _selected_comps(entry, dim: int, *, second: bool) -> tuple[int, ...] | None:
    """Return the component indices actually drawn (for axis labelling).

    Delegates to the shared :func:`_view_components` priority table so the axis
    labels always name the coordinates the cloud actually carries.
    """
    return tuple(_view_components(entry, dim, second=second))


def _ensure_head_color(payload: dict) -> dict:
    """Force the indigo brand head colour into the animation metadata."""
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
def cache_key(entry, *, second: bool) -> str:
    """Content hash: class source + editorial viewer config + this module's knobs."""
    cls_src = inspect.getsource(entry.cls)
    opts = repr(sorted(figures.FIG_OVERRIDES.get(entry.name, {}).items()))
    ed = repr(
        (
            _viewer_cfg(entry),
            _projection(entry, second=False),
            _projection(entry, second=True),
            bool(second),
        )
    )
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
            _FINE_DT,
            _DDE_FINAL_TIME,
            _DDE_DT,
            _DDE_FINE_DT,
            _IC_RETRIES,
            _TRANSIENT_FRAC,
            _TEAL,
            _BG,
            entry.family,
            entry.dim,
        )
    )
    return hashlib.sha256((cls_src + opts + ed + knobs).encode()).hexdigest()[:20]


def _html(entry, payload: dict, *, second: bool) -> str:
    """Wrap ``payload`` in a self-contained viewer document for an ``<iframe>``."""
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


def loader_asset() -> tuple[str, str] | None:
    """Return ``(site_uri, source)`` for the shared three.js loader, or ``None``."""
    try:
        return LOADER_URI, LOADER_SRC.read_text(encoding="utf-8")
    except OSError:
        return None


def has_second_view(entry) -> bool:
    """Whether ``entry`` also gets a *second* animated viewer (``projection2``)."""
    return eligible(entry, second=True)


def render_html(entry, *, second: bool = False) -> str | None:
    """Return the viewer HTML for ``entry`` (cached on disk), or ``None``.

    ``second=True`` renders the ``projection2`` view (the page's second attractor
    animation for a 4-D-plus flow).  Returns ``None`` for an ineligible system, a
    disabled build (``TSD_DOCS_FIGURES=0``), or any soft failure — the page then
    falls back to the static PNG.  Never raises.

    This is the **back-compat** single-view entry point; the generator's preferred
    surface is :func:`viewer_payloads`, which returns every view of a system in one
    call (and never emits a second view whose primary declined).
    """
    try:
        if not eligible(entry, second=second):
            return None

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "-b" if second else ""
        cached = CACHE_DIR / f"{entry.name}{suffix}-{cache_key(entry, second=second)}.html"
        if cached.exists():
            return cached.read_text(encoding="utf-8")

        payload = _build_payload(entry, second=second)
        if payload is None:
            return None
        doc = _html(entry, payload, second=second)
        cached.write_text(doc, encoding="utf-8")
        return doc
    except Exception:  # noqa: BLE001 — a viewer must never break the docs build
        return None


def viewer_payloads(entry) -> list[dict[str, str]]:
    """Return every interactive-viewer document for ``entry``, primary first.

    The docs generator's preferred surface.  Each element is
    ``{"suffix": <str>, "html": <str>}``:

    - ``suffix == ""`` — the **primary** viewer (the state itself for a low-dim
      flow, the editorial ``projection`` for a 4-D-plus flow, the delay embedding
      for a DDE, or the Group-B ``viewer.components`` view).
    - ``suffix == "-b"`` — a **second** projection (``projection2``), present only
      for a 4-D-plus flow that declares one *and* whose primary rendered.

    The generator registers each element at ``assets/threejs/<Name><suffix>.html``
    and embeds one ``<iframe>`` per view.  Returns ``[]`` for an ineligible /
    disabled / soft-failing system (the page then falls back to its static PNG) —
    a second view is **never** emitted without its primary, so the page can't show a
    lone "-b" attractor.  Never raises.
    """
    try:
        primary = render_html(entry, second=False)
        if primary is None:
            return []
        views = [{"suffix": "", "html": primary}]
        if eligible(entry, second=True):
            secondary = render_html(entry, second=True)
            if secondary is not None:
                views.append({"suffix": "-b", "html": secondary})
        return views
    except Exception:  # noqa: BLE001 — a viewer must never break the docs build
        return []
