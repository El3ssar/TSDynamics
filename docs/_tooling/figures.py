"""
Build-time figure rendering for the per-system documentation pages.

Strategy
--------
- **ODE** figures integrate with the **shipped Rust engine** (the same
  ``integrate(backend="interp")`` path the library exposes) for every
  non-stiff, non-discontinuous system — so the docs picture is rendered by the
  code that ships, not an out-of-band SciPy reimplementation.  The handful of
  **stiff** systems (those declaring a ``_default_method``, e.g. ``"bdf"``) and
  **discontinuous** systems (a ``sign``/``abs`` right-hand side, flagged with a
  ``"method"`` override in :data:`FIG_OVERRIDES`) fall back to the local
  ``scipy.solve_ivp`` path over the SymEngine-lambdified numeric RHS
  (``_rhs_numeric``) — the implicit/event handling the explicit engine kernels
  used for figures do not cover.
- **DDE** figures use the real ``integrate`` (the Rust engine; only 5 systems).
- **Map** figures iterate via the family API (the Rust engine).

A content-addressed cache under ``.cache/docs-figures`` keyed by
``sha256(class source ‖ this module's source)`` means only new or changed
systems ever re-render; CI persists the cache directory between builds.
Per-system failures soft-fail (the page ships without a figure).
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import pathlib
import shutil
import warnings

import numpy as np
import plot_dt as _plot_dt  # the ONE sagitta-dt selector both renderers call


def _quiet_numerics(fn):
    """Silence the expected FP / divergence ``RuntimeWarning``s of build-time work.

    Rendering explores each system with random initial conditions and wide
    sampling boxes, so some trajectories legitimately overflow or go non-finite;
    the callers already retry or drop those (finiteness checks, IC re-rolls). This
    keeps the ``--strict`` docs-build log clean without changing any result — the
    control flow that handles divergence is untouched, only the noisy warning is
    suppressed for the duration of the render.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            return fn(*args, **kwargs)

    return wrapper


ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "docs-figures"
OUT_DIR = ROOT / "docs" / "assets" / "figures" / "systems"

_ACCENT = "#4f46e5"  # indigo — flow/system (matches the docs identity)
_ACCENT_2 = "#0d9488"  # teal — secondary / delay-embedding view
_TEAL = "#2CC5AE"  # bright brand teal — SDE sample paths / primary time series

#: Brand-palette line colours for a multi-component time series (Oregonator's
#: three scaled concentrations): teal, indigo, deep teal — the docs identity.
_SERIES_PALETTE = ("#2CC5AE", "#4f46e5", "#11857A")

#: Per-system rendering overrides: final_time, dt, ic, kind, transient_frac.
FIG_OVERRIDES: dict[str, dict] = {
    "Lorenz96": {"kind": "spacetime", "final_time": 60.0, "dt": 0.1},
    "KuramotoSivashinsky": {"kind": "spacetime", "final_time": 150.0, "dt": 0.5},
    # 2-D spatial fields → the final field reshaped to its grid (a heatmap).
    # NO "method" override: that routes to the SciPy fallback, and solve_ivp over a
    # 1k-5k-state Python RHS is intractable.  They are non-stiff + continuous, so the
    # fast Rust engine's explicit kernel renders them (a couple of seconds).
    "GrayScott": {"kind": "field", "final_time": 1500.0, "dt": 3.0},
    "SwiftHohenberg": {"kind": "field", "final_time": 50.0, "dt": 0.1},
    "MultiChua": {"ic": "0.1*ones"},
    "DoubleGyre": {"final_time": 40.0},
    # Stiff relaxation oscillator (Belousov–Zhabotinsky).  An explicit kernel
    # diverges, so integrate through the shipped engine's variable-order BDF via
    # the ``engine_method`` override (the code that ships, not SciPy).  The scaled
    # concentrations span several decades over one relaxation cycle, so a
    # time-series of the three species reads far better than a Z-dominated phase
    # portrait.
    "Oregonator": {
        "final_time": 40.0,
        "dt": 0.005,
        "engine_method": "bdf",
        "kind": "timeseries",
        "series_labels": ("X", "Y", "Z"),  # scaled HBrO₂ / Br⁻ / Ce⁴⁺ (docstring)
    },
    # Györgyi–Field BZ model: very fast relaxation spikes (t0 rescales time), and
    # the ``sqrt(x)``/``max(0,x)`` terms make BDF's Jacobian singular at x=0, so the
    # engine's *adaptive* rk45 renders the oscillation over a short window.  A
    # time-series of the three species reads better than a v-thin phase portrait.
    "BelousovZhabotinsky": {
        "final_time": 0.3,
        "dt": 1e-4,
        "engine_method": "rk45",
        "kind": "timeseries",
        "series_labels": ("x", "z", "v"),  # HBrO₂ / oxidised catalyst / BrMA
    },
    # Finite-basin systems (Blasius, RabinovichFabrikant, Sprott*, Hyper*,
    # HenonHeiles) carry their on-attractor IC as a class ``default_ic`` —
    # the renderer picks it up via ``_resolve_ic``. Only longer integration
    # windows for a fuller attractor live here:
    "SprottD": {"final_time": 60.0},
    "SprottI": {"final_time": 60.0},
    "SprottM": {"final_time": 60.0},
    "SprottO": {"final_time": 60.0},
    "HyperRossler": {"final_time": 60.0},
    # 4-D hyperchaotic flow: fixed-step rk4 diverges at this scale, but the
    # engine's *adaptive* rk45 stays bounded — force it via ``engine_method`` so
    # the static 3-component projection (first three of four coords) renders.  The
    # flow is fast (large excursions per unit time), so a fine output ``dt`` keeps
    # the swept curve smooth; a generous transient trim drops the lead-in lines.
    "HyperQi": {
        "final_time": 30.0,
        "dt": 0.001,
        "engine_method": "rk45",
        "transient_frac": 0.3,
    },
    # Isothermal autocatalytic chemistry: sigma≈0.013 makes the ``beta`` equation
    # fast (mildly stiff), so fixed-step rk4 blows up while the engine's *adaptive*
    # rk45 stays on the bounded oscillation.  The editorial ``viewer`` block sets the
    # on-attractor ic/window; the kernel override lives here (it drives both the
    # static figure and the interactive viewer's ``_pilot_method``).
    "IsothermalChemical": {"engine_method": "rk45"},
    # Discontinuous (sign) right-hand sides — RK45 steps across the jumps:
    "StickSlipOscillator": {"ic": [0.1, 0.1, 0.1], "final_time": 60.0, "method": "RK45"},
    "Colpitts": {"ic": [0.1, 0.1, 0.1], "final_time": 40.0, "method": "RK45"},
    # SDE sample paths (seeded → reproducible/cacheable).  A longer window for the
    # double well so several barrier hops (its signature switching) are visible; the
    # ``±sqrt(a/b) = ±1`` wells are drawn as faint guide lines.  OU and GBM read well
    # over the default window (mean reversion / a positive multiplicative-noise path).
    "DoubleWell": {"final_time": 200.0, "seed": 0, "guides": (-1.0, 1.0)},
    "GeometricBrownianMotion": {"final_time": 100.0, "seed": 0},
    "OrnsteinUhlenbeck": {"final_time": 100.0, "seed": 0},
    # Anticipating-synchronization DDE whose attractor sits near the origin — the
    # default 0.8-centred history escapes its basin and diverges, so start small.
    "VossDelay": {"final_time": 500.0, "dt": 0.2, "history_center": 0.15, "history_amp": 0.1},
}


#: Per-map static-figure curation.  Maps render as **static** scatter plots (a
#: screenshot reads better than an animated trailing swarm), and a few need a
#: curated initial condition / parameter override / view angle to look right:
#:
#: - ``ensemble``: iterate this many short orbits (``ensemble_steps`` each) from
#:   random ICs and pool the points — the honest way to fill a mixing map (Baker)
#:   whose single orbit collapses to a fixed point under binary-doubling round-off.
#: - ``ic`` / ``params``: a curated on-attractor start / parameter set for a map
#:   whose registry default collapses to a point (GumowskiMira).
#: - ``steps`` / ``burn``: iterate count + burn-in.
#: - ``view``: ``(elev, azim)`` for a 3-D map whose thin dimension needs an angle
#:   to reveal its structure (FoldedTowel).
#: - ``bifurcation``: ``(param, lo, hi)`` — a 1-D map also gets a library-generated
#:   bifurcation diagram (``ts.orbit_diagram``) beside its return map.
MAP_OVERRIDES: dict[str, dict] = {
    # Baker's map: 2·x mod 1 exhausts the mantissa and any single orbit collapses
    # to (0,0) after ~52 iterations.  Pool many short independent orbits so the
    # points fill the unit square (the true attractor) without the collapse.
    "Baker": {"ensemble": 500, "ensemble_steps": 40, "burn": 0},
    # GumowskiMira's registry defaults collapse to a tiny region; a curated
    # (a, b, ic) gives its signature spread ornamental attractor.
    "GumowskiMira": {
        "params": {"a": -0.48, "b": 0.93},
        "ic": [0.1, 4.0],
        "steps": 40000,
        "burn": 100,
    },
    # Zaslavskii: the registry defaults (eps=5, nu=0.2, r=2) collapse to a period-2
    # orbit (the "only ~2 points visible" defect), and the milder (eps=9, nu=0.2,
    # r=3) folds to a single thin loop.  The classic dissipative-standard-map
    # parameters (eps=9, nu=0.3, r=2) stretch-and-fold the web onto its signature
    # multi-band fractal strange attractor; a long orbit fills the bands.
    "Zaslavskii": {
        "params": {"eps": 9.0, "nu": 0.3, "r": 2.0},
        "ic": [0.1, 0.1],
        "steps": 200000,
        "burn": 1000,
        "point_size": 0.12,
        "aspect": "auto",  # phase x∈[0,1) vs action y∈[-1.3,1.3]: fill the frame
    },
    # Chirikov standard map (k ≈ 0.97, the critical value): a single orbit only
    # traces one KAM torus / one chaotic filament, so the figure looked like a lone
    # line.  Pool many short orbits from ICs spread over the (p, x) 2π-torus and wrap
    # both coordinates mod 2π — that is the classic mixed phase-space portrait
    # (nested tori threaded by the chaotic sea).
    "Chirikov": {
        "ensemble": 300,
        "ensemble_steps": 250,
        "ensemble_span": (0.0, 6.283185307179586),
        "wrap": (0, 1),
        "swap_axes": True,  # plot x (angle) horizontal, p (action) vertical
        "point_size": 0.06,
        "burn": 0,
    },
    # Gingerbreadman: a random U[0,1)² start can land on a periodic island (the
    # figure showed only a handful of points).  Seed the known chaotic sea explicitly
    # and pool a spray of extra orbits so both the signature "gingerbread man" body
    # and its surrounding period-6 islands fill in.
    "Gingerbreadman": {
        "ensemble": 120,
        "ensemble_steps": 1500,
        "ensemble_span": (-4.0, 7.0),
        "seeds": [[-0.1, 0.0], [0.5, 3.7], [3.7, 0.5], [-2.0, -2.0]],
        "seed_steps": 8000,
        "point_size": 0.05,
        "burn": 0,
    },
    # Folded-towel: a thin (0.85 × 0.075 × 0.75) 3-D cloud — view from an angle
    # that reveals the fold rather than the flat face, and iterate plenty.
    "FoldedTowel": {"steps": 40000, "burn": 500, "view": (22.0, -60.0), "point_size": 0.12},
    "GeneralizedHenon": {"steps": 40000, "burn": 500, "view": (20.0, -70.0), "point_size": 0.12},
    # --- 1-D maps: a return map is dull; add a recognizable bifurcation diagram. ---
    "Logistic": {"bifurcation": ("r", 2.5, 4.0)},
    "Ricker": {"bifurcation": ("a", 1.0, 16.0), "bif_clip": 20.0},
    "Tent": {"bifurcation": ("mu", 0.4, 1.0)},
    "Gauss": {"bifurcation": ("b", -1.0, 1.0)},
    "Chebyshev": {"bifurcation": ("a", 2.0, 8.0)},
    "Circle": {"bifurcation": ("k", 0.0, 8.0)},
    "Ulam": {"bifurcation": ("a", 0.0, 2.0)},
}


def _viewer_cfg(entry) -> dict:
    """Return the per-system editorial ``viewer`` directive (Group B), or ``{}``.

    Works for a catalogue ``SystemRecord`` (carries ``.viewer``) *or* a bare
    registry entry (looked up in ``editorial.json`` by name).
    """
    cfg = getattr(entry, "viewer", None)
    if isinstance(cfg, dict):
        return cfg
    try:
        import catalog  # docs/_tooling sibling

        rec = catalog.load_catalog().by_name(getattr(entry, "name", ""))
        return dict(rec.viewer) if rec is not None else {}
    except Exception:  # noqa: BLE001 — editorial is decoration, never load-bearing
        return {}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.transparent": True,
            "axes.edgecolor": "#88888855",
            "axes.labelcolor": "#888888",
            "xtick.color": "#888888",
            "ytick.color": "#888888",
            "axes.grid": False,
            "font.size": 9,
        }
    )
    return plt


RENDERER_VERSION = "10"  # bump manually when rendering output materially changes


def cache_key(entry) -> str:
    """Content hash: class source + this system's overrides + renderer version.

    Incorporates the editorial ``viewer`` directive and the per-map
    :data:`MAP_OVERRIDES` so a curated IC / projection / bifurcation change
    re-renders the cached figure.
    """
    cls_src = inspect.getsource(entry.cls)
    opts = repr(sorted(FIG_OVERRIDES.get(entry.name, {}).items()))
    mcfg = repr(sorted(MAP_OVERRIDES.get(entry.name, {}).items()))
    vcfg = repr(sorted(_viewer_cfg(entry).items()))
    return hashlib.sha256((cls_src + opts + mcfg + vcfg + RENDERER_VERSION).encode()).hexdigest()[
        :20
    ]


def _resolve_ic(sys_obj, override):
    if override == "0.1*ones":
        return 0.1 * np.ones(sys_obj.dim)
    if override is not None:
        return np.asarray(override, dtype=float)
    if type(sys_obj).default_ic is not None:
        # Honor a class-level basin IC (single source of truth) before
        # falling back to random; the retry loop still re-rolls on failure.
        return np.asarray(type(sys_obj).default_ic, dtype=float).reshape(sys_obj.dim)
    return None  # family default resolution (random U[0,1)^dim, with retries)


def _use_engine_for_ode(entry, opts) -> bool:
    """Whether ``entry`` renders through the shipped engine vs the SciPy fallback.

    The engine ODE path used here drives the explicit, fixed/adaptive RK
    kernels.  It does **not** cover the two cases the docs build needs SciPy
    for, which therefore route to the commented :func:`_ode_trajectory_scipy`
    fallback below:

    - **stiff** systems — their ``_default_method`` resolves to an implicit
      (needs-Jacobian) kernel (e.g. ``"bdf"``), i.e. an explicit integration
      blows up;
    - **discontinuous** right-hand sides (``sign``/``abs``) — flagged by a
      ``"method"`` (``"RK45"``) override in :data:`FIG_OVERRIDES`, where the
      step controller must walk carefully across the jumps.
    """
    if opts.get("method") is not None:  # discontinuous (sign/abs) RHS → SciPy
        return False
    # Stiff systems declare an implicit ``_default_method`` (the base default is
    # the explicit "RK45").  Ask the solver registry whether that kernel needs a
    # Jacobian — robust to any implicit name (bdf / rosenbrock / trbdf2).
    method = getattr(entry.cls, "_default_method", "RK45")
    try:
        from tsdynamics.solvers import resolve

        if resolve(method).spec.caps.needs_jacobian:
            return False
    except Exception:  # noqa: BLE001 — unknown name → be conservative, use SciPy
        return False
    return True


def _ode_trajectory_engine(entry, opts) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a non-stiff ODE through the shipped engine (``backend="interp"``).

    Mirrors the renderer's IC-retry contract: the engine raises on divergence
    (it does not re-roll the IC itself), so off-basin random starts are caught
    and retried here, exactly as the SciPy fallback does.

    Marches with the **fixed-step** ``rk4`` kernel by default rather than the
    adaptive default on purpose: an off-basin random start that races to infinity
    then raises after a handful of cheap steps and is retried, instead of sending
    the *adaptive* step-controller into a minutes-long step-shrinking spiral as it
    chases the diverging solution down to the minimum step size (a cold build of the
    conservative / chaotic catalogue otherwise appears to hang).  ``rk4`` at the fine
    sagitta step (~0.002) is more than accurate enough for a non-stiff attractor
    thumbnail.

    A system that cannot be marched with fixed-step ``rk4`` (a stiff relaxation
    oscillator that needs the implicit ``bdf``; a fast hyperchaotic flow that needs
    the adaptive ``rk45``) sets an ``engine_method`` override in
    :data:`FIG_OVERRIDES`, selecting that shipped-engine kernel here — still the code
    that ships, and still on the IC-retry contract (every engine kernel raises on
    divergence).
    """
    final_time = opts.get("final_time", 100.0)
    method = opts.get("engine_method", "rk4")
    rng = np.random.default_rng(42)

    # The maintainer's sagitta rule (ε = 0.01, "redo with error 0.01"): pick the
    # smooth output dt, integrate at a *fine* step, then sub-sample at that dt so the
    # static curve is never faceted — identical to the three.js viewer's march, so a
    # phase portrait and its interactive twin trace the same smooth curve.
    #
    # CRITICAL: ``choose_plot_dt`` must run its own **fine pilot** to discover the
    # true output dt *from below*.  Call it with ``dt0=None`` so the selector uses
    # its per-family fine-pilot step (``FINE_PILOT_DT["ode"] = 0.002``) — NOT a
    # coarse ``nominal_dt``.  A sagitta search can only ever report a step no finer
    # than the pilot it ran, so feeding a 0.01 pilot could never find that a fast
    # attractor (DequanLi, QiChen, Chen, YuWang, …) needs ~0.002–0.006 to read
    # smooth; it would just echo 0.01 back and stay pixelated.  A curated per-system
    # ``dt`` override is still honoured verbatim (``choose_plot_dt`` short-circuits on
    # a figure ``dt``), so those systems keep their editorial step.
    # A slow, smooth flow whose meaningful figure needs a *long* window (BickleyJet's
    # tracer transport) would otherwise integrate 10M+ steps at the 0.002 fine pilot —
    # and the sagitta pilot inside ``choose_plot_dt`` would itself march the whole
    # window at that step.  Such a system sets an explicit coarse ``integrate_dt``
    # (still far finer than its natural step), which bypasses the pilot entirely.
    if opts.get("integrate_dt"):
        fine_dt = float(opts["integrate_dt"])
        smooth_dt = float(opts.get("dt") or fine_dt)
    else:
        fine_pilot_dt = float(_plot_dt.FINE_PILOT_DT.get("ode", 0.002))
        smooth_dt = _plot_dt.choose_plot_dt(entry, final_time=final_time, dt0=None, epsilon=0.01)
        fine_dt = min(smooth_dt, fine_pilot_dt)
    stride = max(1, int(round(smooth_dt / fine_dt)))
    # Safety floor on the point count: a pathological sagitta dt (a pilot that
    # decayed to a near-fixed manifold) must never sub-sample the drawn attractor
    # down to a handful of segments.  Cap the stride so at least ~2000 samples of the
    # post-transient curve survive — the smooth dt still wins for every well-behaved
    # system (their stride is far below this ceiling), and a static PNG line carries
    # plenty of vertices cheaply, so we keep a generous floor.
    n_fine = max(1, int(final_time / fine_dt))
    keep_frac = 1.0 - float(opts.get("transient_frac", 0.15))
    stride = min(stride, max(1, int(n_fine * keep_frac / 2000)))

    sys_obj = entry.cls()
    ic = _resolve_ic(sys_obj, opts.get("ic"))
    for attempt in range(4):
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
        t, y = traj.t, traj.y
        if len(y) > 50 and np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e6:
            drop = int(opts.get("transient_frac", 0.15) * len(y))
            return t[drop:][::stride], y[drop:][::stride]
        ic = None
    raise RuntimeError("no bounded trajectory found")


def _ode_trajectory_scipy(entry, opts) -> tuple[np.ndarray, np.ndarray]:
    """SciPy ``solve_ivp`` fallback for stiff / discontinuous ODE figures.

    Retained as the renderer fallback for the systems the explicit engine
    kernels used for figures do not cover (see :func:`_use_engine_for_ode`):
    LSODA auto-switches for stiffness, and RK45 walks discontinuous (sign/abs)
    right-hand sides.  Integrates the system's SymEngine-lambdified numeric RHS
    (``_rhs_numeric``).
    """
    final_time = opts.get("final_time", 100.0)
    dt = opts.get("dt", 0.01)
    rng = np.random.default_rng(42)

    sys_obj = entry.cls()
    rhs = sys_obj._rhs_numeric()
    from scipy.integrate import solve_ivp

    def blowup(t, u):  # terminal event: stop divergent runs immediately
        return float(np.max(np.abs(u)) - 1e6)

    blowup.terminal = True

    class _BudgetError(Exception):
        pass

    ic = _resolve_ic(sys_obj, opts.get("ic"))
    for attempt in range(4):
        if ic is None or attempt > 0:
            ic = sys_obj.resolve_ic(rng.uniform(0.0, 1.0, sys_obj.dim))

        # Hard wall-time guard: a stiff or pathological system must not stall
        # the whole docs build — cap the RHS evaluation budget per attempt.
        calls = 0

        def rhs_capped(t, u):
            nonlocal calls
            calls += 1
            if calls > 300_000:
                raise _BudgetError
            return rhs(u, t)

        try:
            sol = solve_ivp(
                rhs_capped,
                (0.0, final_time),
                np.asarray(ic, dtype=float),
                t_eval=np.arange(0.0, final_time, dt),
                # LSODA auto-switches for stiffness; RK45 for discontinuous
                # right-hand sides (sign/abs), where LSODA churns.
                method=opts.get("method", "LSODA"),
                rtol=1e-7,
                atol=1e-9,
                events=blowup,
            )
        except _BudgetError:
            ic = None
            continue
        y = sol.y.T
        diverged = sol.status == 1 or not np.all(np.isfinite(y))  # event fired
        if sol.success and not diverged and len(y) > 50 and np.max(np.abs(y)) < 1e6:
            drop = int(opts.get("transient_frac", 0.15) * len(y))
            return sol.t[drop:], y[drop:]
        ic = None
    raise RuntimeError("no bounded trajectory found")


def _ode_trajectory(entry, opts) -> tuple[np.ndarray, np.ndarray]:
    """Render-time ODE trajectory: shipped engine for the common case, else SciPy.

    Non-stiff, non-discontinuous systems integrate through the shipped Rust
    engine (``integrate(backend="interp")``) so the docs figure is produced by
    the code that ships.  Stiff / discontinuous systems use the commented
    SciPy ``solve_ivp`` fallback (:func:`_ode_trajectory_scipy`).

    An explicit ``engine_method`` override (a stiff system that wants the engine's
    ``bdf``, a fast flow that wants the adaptive ``rk45``) always takes the engine
    path.  This is a **figures-only** override: it deliberately does *not* flip the
    shared :func:`_use_engine_for_ode` predicate that :mod:`threejs_viewer` reads
    for viewer eligibility, so such a system keeps its curated static figure (a
    time-series / projection) rather than an ill-suited 3-D comet.
    """
    if opts.get("engine_method") is not None or _use_engine_for_ode(entry, opts):
        return _ode_trajectory_engine(entry, opts)
    return _ode_trajectory_scipy(entry, opts)


def _field_trajectory(entry, opts) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a spatial-field / spacetime ODE with the system's *own* IC.

    A method-of-lines field (Gray–Scott, Kuramoto–Sivashinsky, Lorenz-96) seeds
    a structured initial field through the library's native ``resolve_ic`` — a
    flat random ``U[0,1)^dim`` start (the generic ``_ode_trajectory_engine``
    retry contract) never develops a pattern and trips "no bounded trajectory
    found".  So we let the system resolve its own IC (no ``ic=`` override) and
    integrate once through the shipped engine.
    """
    final_time = opts.get("final_time", 100.0)
    dt = opts.get("dt", 0.01)
    sys_obj = entry.cls()
    ic = _resolve_ic(sys_obj, opts.get("ic"))
    kwargs = {"final_time": final_time, "dt": dt, "backend": "interp"}
    if ic is not None:
        kwargs["ic"] = np.asarray(ic, dtype=float)
    traj = sys_obj.integrate(**kwargs)
    return traj.t, traj.y


def _wrap_components(y: np.ndarray, wrap) -> np.ndarray:
    """Wrap the listed component indices onto ``[-π, π)`` (a torus / angle flow)."""
    if not wrap:
        return y
    y = y.copy()
    for idx in wrap:
        if 0 <= int(idx) < y.shape[1]:
            y[:, int(idx)] = (y[:, int(idx)] + np.pi) % (2 * np.pi) - np.pi
    return y


def _render_ode(entry, plt, opts):
    if opts.get("kind") in ("field", "spacetime"):
        t, y = _field_trajectory(entry, opts)
        if opts.get("kind") == "field":
            return _render_field(entry, plt, y)
        return _render_spacetime(entry, plt, t, y)

    # Editorial ``viewer`` directive (Group B): a curated static view (a 2-D/3-D
    # projection, a wrapped torus flow, a time series, or a Cartesian polar plot).
    vcfg = _viewer_cfg(entry)
    # Merge the viewer's pilot overrides (ic / final_time / dt / method / transient).
    merged = dict(opts)
    for k_src, k_dst in (
        ("final_time", "final_time"),
        ("dt", "dt"),
        ("integrate_dt", "integrate_dt"),
        ("ic", "ic"),
        ("transient", "transient_frac"),
    ):
        if vcfg.get(k_src) is not None:
            merged[k_dst] = vcfg[k_src]
    if vcfg.get("method"):
        merged["engine_method"] = vcfg["method"]

    static_kind = vcfg.get("static_kind")
    if static_kind == "polar":
        return _render_polar(entry, plt, merged, vcfg)

    t, y = _ode_trajectory(entry, merged)
    y = _wrap_components(y, vcfg.get("wrap"))

    if opts.get("kind") == "timeseries" or static_kind == "timeseries":
        return _render_timeseries(entry, plt, t, y, {**opts, **vcfg})
    if entry.cls().dim is None:
        return _render_spacetime(entry, plt, t, y)

    # A ``components`` projection (Group B forced/kinematic systems) picks the
    # honest 2-/3-D view; otherwise the first min(3, dim) components.
    comps = vcfg.get("components")
    if comps is not None:
        idx = [int(c) for c in comps if 0 <= int(c) < y.shape[1]]
    else:
        idx = list(range(min(3, y.shape[1])))
    view = y[:, idx]
    dim = view.shape[1]

    if dim >= 3:
        fig = plt.figure(figsize=(5.4, 4.2))
        ax = fig.add_subplot(projection="3d")
        ax.plot(view[:, 0], view[:, 1], view[:, 2], lw=0.35, color=_ACCENT)
        ax.set_axis_off()
    elif dim == 2:
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        ax.plot(view[:, 0], view[:, 1], lw=0.4, color=_ACCENT)
        ax.set_xticks([]), ax.set_yticks([])
    else:
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        ax.plot(t, view[:, 0], lw=0.8, color=_ACCENT)
        ax.set_xlabel("t")
    return fig


def _render_polar(entry, plt, opts, vcfg):
    """Render a polar-coordinate flow (r, θ, …) in Cartesian ``(r cosθ, r sinθ)``.

    A tracer stirred in a circular cell (Blinking Rotlet) lives in ``(r, θ)`` with
    ``θ`` winding unbounded; the honest picture is the Cartesian streak line, so we
    map ``(r, θ) → (r cosθ, r sinθ)`` and draw that.
    """
    t, y = _ode_trajectory(entry, opts)
    r = y[:, 0]
    theta = y[:, 1]
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot(x, z, lw=0.3, color=_ACCENT)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    return fig


def _render_spacetime(entry, plt, t, y):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.imshow(
        y.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=(float(t[0]), float(t[-1]), 0, y.shape[1]),
    )
    ax.set_xlabel("t")
    ax.set_ylabel("cell")
    return fig


def _render_field(entry, plt, y):
    """Render a 2-D spatial-field system: the final field reshaped to its grid.

    The system's ``_field_shape`` ``(Ny, Nx)`` gives the spatial grid; for a
    multi-block state (Gray–Scott's ``[u, v]``) the **last** block (the activator)
    is shown — the convention the ``kind="field"`` plot recipe uses.
    """
    sys_obj = entry.cls()
    shape = getattr(sys_obj, "_field_shape", None) or (y.shape[1],)
    cells = int(np.prod(shape))
    block = y[-1, -cells:]  # the last field block at the final time
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(block.reshape(shape), origin="lower", cmap="viridis")
    ax.set_xticks([]), ax.set_yticks([])
    return fig


def _render_timeseries(entry, plt, t, y, opts):
    """Render every component of a low-dim flow as a brand-palette time series.

    Used for a system whose phase portrait would be dominated by one
    wildly-different-scaled coordinate — the stiff Oregonator relaxation
    oscillator, whose three scaled concentrations span several decades over one
    cycle.  All-positive series get a log y-axis (the honest view of a
    multi-decade concentration); a component that dips non-positive gets a linear
    axis.  Component names come from the ``variables`` ClassVar when present.
    """
    names = list(opts.get("series_labels") or getattr(entry.cls, "variables", None) or [])
    # A ``components`` filter (viewer directive) restricts a high-dim system to the
    # few informative channels (ExcitableCell's spiking V; WINDMI's i, v — dropping
    # the runaway pressure integral p).
    comps = opts.get("components")
    if comps is not None:
        idx = [int(c) for c in comps if 0 <= int(c) < y.shape[1]]
    else:
        idx = list(range(y.shape[1]))
    sub = y[:, idx]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for k, i in enumerate(idx):
        label = names[i] if i < len(names) else f"y{i}"
        ax.plot(t, y[:, i], lw=0.8, color=_SERIES_PALETTE[k % len(_SERIES_PALETTE)], label=label)
    if np.all(sub > 0):
        ax.set_yscale("log")
    ax.set_xlabel("t")
    if len(idx) > 1:
        ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor="#888888")
    ax.set_yticks([]) if len(idx) == 1 else None
    fig.tight_layout()
    return fig


def _sde_sample_path(entry, opts) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one **seeded** sample path of a (scalar) SDE for the figure.

    Runs the shipped SDE integrator with a fixed ``seed`` so the rendered path is
    reproducible (hence cacheable).  The default ``reference`` backend (pure
    Python) needs no compiled wheel and reproduces the engine to float tolerance —
    the right choice for a deterministic, portable docs figure.  Honours a
    per-system ``final_time`` / ``dt`` override (the switching double well wants a
    longer window than a mean-reverting OU path).
    """
    final_time = opts.get("final_time", 100.0)
    dt = opts.get("dt", 0.01)
    seed = int(opts.get("seed", 0))
    sys_obj = entry.cls()
    traj = sys_obj.integrate(final_time=final_time, dt=dt, seed=seed, backend="reference")
    return traj.t, traj.y


def _render_sde(entry, plt, opts):
    """Render a seeded SDE sample path ``x(t)`` as a brand-teal time series.

    A one-dimensional diffusion (OU mean reversion, GBM's positive path, the
    double well's noise-driven switching) is shown as its realised trajectory over
    time — the natural depiction of an SDE, and reproducible under the fixed seed.
    The double well's ``±sqrt(a/b)`` wells are drawn as faint guide lines so the
    barrier-hopping reads at a glance.
    """
    t, y = _sde_sample_path(entry, opts)
    x = y[:, 0]
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    for line in opts.get("guides", ()):  # faint horizontal reference levels
        ax.axhline(float(line), lw=0.6, color="#88888855", ls="--")
    ax.plot(t, x, lw=0.7, color=_TEAL)
    names = list(getattr(entry.cls, "variables", None) or [])
    ax.set_xlabel("t")
    ax.set_ylabel(names[0] if names else "x")
    fig.tight_layout()
    return fig


def _render_dde(entry, plt, opts):
    sys_obj = entry.cls()
    final_time = opts.get("final_time", 300.0)
    dt = opts.get("dt", 0.25)
    # Constant-amplitude sinusoidal history; a system whose attractor basin is not
    # near 0.8 (VossDelay sits near the origin) overrides the centre/amplitude.
    center = opts.get("history_center", 0.8)
    amp = opts.get("history_amp", 0.2)

    def history(s):
        return [center + amp * np.sin(0.2 * s)] * sys_obj.dim

    traj = sys_obj.integrate(final_time=final_time, dt=dt, history=history)
    x = traj.y[:, 0]
    tau = float(sys_obj._delays()[0])
    lag = max(1, int(round(tau / dt)))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.4, 2.8), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )
    ax1.plot(traj.t, x, lw=0.7, color=_ACCENT)
    ax1.set_xlabel("t")
    ax2.plot(x[lag:], x[:-lag], lw=0.5, color=_ACCENT_2)
    ax2.set_xlabel("x(t)")
    ax2.set_ylabel("x(t-τ)")
    for ax in (ax1, ax2):
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def _map_cloud(entry, mcfg) -> np.ndarray:
    """Iterate a map into a drawable point cloud, honouring :data:`MAP_OVERRIDES`.

    Handles the special cases the plain single-orbit iterate cannot:
    - ``ensemble`` — pool many short independent orbits (Baker, whose single orbit
      collapses to a fixed point under binary-doubling round-off);
    - ``params`` / ``ic`` — a curated on-attractor start / parameters for a map
      whose registry default collapses (GumowskiMira);
    - ``steps`` / ``burn`` — iterate count + burn-in.
    """
    sys_obj = entry.cls()
    for key, val in (mcfg.get("params") or {}).items():
        if key in sys_obj.params:
            sys_obj.params[key] = val

    if mcfg.get("ensemble"):
        n_orbits = int(mcfg["ensemble"])
        per = int(mcfg.get("ensemble_steps", 40))
        lo, hi = mcfg.get("ensemble_span", (0.001, 0.999))
        rng = np.random.default_rng(0)
        pieces = []
        # Explicit chaotic-sea seeds first (a map whose default random start can land
        # on a periodic island — Gingerbreadman — pins its signature orbit here).
        for seed_ic in mcfg.get("seeds", ()):
            try:
                tr = sys_obj.iterate(
                    steps=int(mcfg.get("seed_steps", per)),
                    ic=np.asarray(seed_ic, dtype=float),
                )
            except (RuntimeError, ValueError):
                continue
            yy = tr.y
            if np.all(np.isfinite(yy)) and np.max(np.abs(yy)) < 1e4:
                pieces.append(yy[20:])
        for _ in range(n_orbits):
            ic = rng.uniform(lo, hi, sys_obj.dim)
            try:
                tr = sys_obj.iterate(steps=per, ic=ic)
            except (RuntimeError, ValueError):
                continue
            yy = tr.y
            if np.all(np.isfinite(yy)) and np.max(np.abs(yy)) < 1e4:
                pieces.append(yy[20:] if len(yy) > 20 else yy)
        if pieces:
            cloud = np.vstack(pieces)
            wrap = mcfg.get("wrap")
            if wrap:
                cloud = cloud.copy()
                for idx in wrap:
                    if 0 <= int(idx) < cloud.shape[1]:
                        cloud[:, int(idx)] = np.mod(cloud[:, int(idx)], 2 * np.pi)
            return cloud
        # fall through to a single orbit on total failure

    steps = int(mcfg.get("steps", 20_000))
    burn = int(mcfg.get("burn", 100))
    ic = np.asarray(mcfg["ic"], dtype=float) if mcfg.get("ic") is not None else None
    kwargs = {"steps": steps, "max_retries": 15}
    if ic is not None:
        kwargs["ic"] = ic
        kwargs.pop("max_retries")  # a curated IC must be honoured, not re-rolled
    tr = sys_obj.iterate(**kwargs)
    return tr.y[burn:]


def _render_bifurcation(entry, plt, mcfg, ax):
    """Draw a library-generated bifurcation diagram (``ts.orbit_diagram``) on ``ax``.

    For a 1-D map, the parameter sweep + asymptotic-orbit scatter is the picture
    people recognise (the logistic period-doubling cascade).  Sweeps the editorial
    ``bifurcation = (param, lo, hi)`` and scatters the resulting orbit.
    """
    import tsdynamics as ts

    param, lo, hi = mcfg["bifurcation"]
    sys_obj = entry.cls()
    od = ts.orbit_diagram(
        sys_obj, param, np.linspace(lo, hi, 700), component=0, transient=400, n=180
    )
    xr, yr = od.flat()
    xr = np.asarray(xr, dtype=float)
    yr = np.asarray(yr, dtype=float)
    clip = mcfg.get("bif_clip")
    if clip is not None:
        keep = np.isfinite(yr) & (np.abs(yr) <= float(clip))
        xr, yr = xr[keep], yr[keep]
    ax.scatter(xr, yr, s=0.12, color=_ACCENT, linewidths=0, alpha=0.6)
    ax.set_xlabel(param)
    ax.set_ylabel(r"$x_\infty$")
    ax.set_xlim(lo, hi)


def _render_map(entry, plt, opts):
    """Render a **static** map figure (a scatter — reads better than an animation).

    - **1-D maps** → the first-return map ``x_n`` vs ``x_{n+1}`` *and* a
      recognizable **bifurcation diagram** (``ts.orbit_diagram``) side by side.
      These stay a **static PNG** on the page (no interactive viewer).
    - **2-D maps** → the iterate cloud (a curated IC / ensemble for the maps whose
      default orbit collapses).  Also a **static PNG** on the page.
    - **3-D maps** (FoldedTowel, GeneralizedHenon) → the iterate cloud as a 3-D
      scatter at a curated view angle.  A dim-3 map now also gets an *interactive*
      (orbitable, **non-animated**) three.js point-cloud viewer (see
      :mod:`threejs_viewer`); this static scatter is its **poster / WebGL-off
      fallback** (the viewer iframe references it as its ``<img>``).  So we keep
      rendering it verbatim — it does not fight the viewer (the docs hook shows the
      PNG inline only for a system with *no* viewer, and hands it to the viewer as
      the poster otherwise).  A good camera + plenty of iterates keep the fold
      legible in the still.
    """
    mcfg = MAP_OVERRIDES.get(entry.name, {})
    sys_obj = entry.cls()
    dim = sys_obj.dim

    if dim == 1:
        y = _map_cloud(entry, mcfg)
        if mcfg.get("bifurcation"):
            # Return map (left) + bifurcation diagram (right).
            fig, (axr, axb) = plt.subplots(
                1, 2, figsize=(8.4, 3.8), gridspec_kw={"width_ratios": [1.0, 1.5]}
            )
            axr.scatter(y[:-1, 0], y[1:, 0], s=0.5, color=_ACCENT_2, linewidths=0)
            axr.set_xlabel(r"$x_n$")
            axr.set_ylabel(r"$x_{n+1}$")
            axr.set_title("return map", fontsize=8, color="#888888")
            _render_bifurcation(entry, plt, mcfg, axb)
            axb.set_title("bifurcation diagram", fontsize=8, color="#888888")
            fig.tight_layout()
            return fig
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.scatter(y[:-1, 0], y[1:, 0], s=0.4, color=_ACCENT, linewidths=0)
        ax.set_xlabel(r"$x_n$")
        ax.set_ylabel(r"$x_{n+1}$")
        return fig

    y = _map_cloud(entry, mcfg)
    ps = float(mcfg.get("point_size", 0.25))
    if dim == 2:
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        # A map whose natural portrait reads better with the second coordinate on the
        # horizontal axis (Chirikov: angle x across, action p up) opts in via
        # ``swap_axes`` — otherwise the raw (col-0, col-1) ordering.
        cx, cy = (1, 0) if mcfg.get("swap_axes") else (0, 1)
        ax.scatter(y[:, cx], y[:, cy], s=ps, color=_ACCENT, linewidths=0)
        # Equal aspect is the honest default for a phase-space cloud, but a map
        # whose two coordinates live on very different scales (Zaslavskii's phase
        # ``x`` on [0,1) vs its action ``y`` spanning several units) reads better on
        # a free (auto) aspect that fills the frame — opt in via ``aspect``.
        if mcfg.get("aspect") != "auto":
            ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]), ax.set_yticks([])
    else:
        fig = plt.figure(figsize=(5.4, 4.6))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=ps, color=_ACCENT, linewidths=0)
        view = mcfg.get("view")
        if view:
            ax.view_init(elev=float(view[0]), azim=float(view[1]))
        ax.set_axis_off()
    return fig


@_quiet_numerics
def render(entry) -> pathlib.Path | None:
    """
    Ensure the figure for ``entry`` exists in ``OUT_DIR``; return its path.

    Cache hit → copy; miss → render + cache.  Returns None on soft failure.
    """
    opts = FIG_OVERRIDES.get(entry.name, {})
    if opts.get("skip"):
        return None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{entry.name}.png"
    cached = CACHE_DIR / f"{entry.name}-{cache_key(entry)}.png"

    if cached.exists():
        shutil.copy(cached, out)
        return out

    plt = _style()
    try:
        if entry.family == "ode":
            fig = _render_ode(entry, plt, opts)
        elif entry.family == "dde":
            fig = _render_dde(entry, plt, opts)
        elif entry.family == "sde":
            fig = _render_sde(entry, plt, opts)
        else:
            fig = _render_map(entry, plt, opts)
        fig.savefig(cached, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:  # noqa: BLE001 — soft-fail, page ships without figure
        plt.close("all")
        return None

    shutil.copy(cached, out)
    return out
