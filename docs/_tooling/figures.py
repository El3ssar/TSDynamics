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

import hashlib
import inspect
import pathlib
import shutil

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "docs-figures"
OUT_DIR = ROOT / "docs" / "assets" / "figures" / "systems"

_ACCENT = "#4f46e5"  # indigo — flow/system (matches the docs identity)
_ACCENT_2 = "#0d9488"  # teal — secondary / delay-embedding view

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
    "Oregonator": {"skip": True},  # stiff — solve_ivp needs special handling
    # Finite-basin systems (Blasius, RabinovichFabrikant, Sprott*, Hyper*,
    # HenonHeiles) carry their on-attractor IC as a class ``default_ic`` —
    # the renderer picks it up via ``_resolve_ic``. Only longer integration
    # windows for a fuller attractor live here:
    "SprottD": {"final_time": 60.0},
    "SprottI": {"final_time": 60.0},
    "SprottM": {"final_time": 60.0},
    "SprottO": {"final_time": 60.0},
    "HyperRossler": {"final_time": 60.0},
    "HyperQi": {"final_time": 30.0},
    # Discontinuous (sign) right-hand sides — RK45 steps across the jumps:
    "StickSlipOscillator": {"ic": [0.1, 0.1, 0.1], "final_time": 60.0, "method": "RK45"},
    "Colpitts": {"ic": [0.1, 0.1, 0.1], "final_time": 40.0, "method": "RK45"},
}


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


RENDERER_VERSION = "4"  # bump manually when rendering output materially changes


def cache_key(entry) -> str:
    """Content hash: class source + this system's overrides + renderer version."""
    cls_src = inspect.getsource(entry.cls)
    opts = repr(sorted(FIG_OVERRIDES.get(entry.name, {}).items()))
    return hashlib.sha256((cls_src + opts + RENDERER_VERSION).encode()).hexdigest()[:20]


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
    if opts.get("method") is not None:  # discontinuous (sign/abs) RHS
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
    """
    final_time = opts.get("final_time", 100.0)
    dt = opts.get("dt", 0.01)
    rng = np.random.default_rng(42)

    sys_obj = entry.cls()
    ic = _resolve_ic(sys_obj, opts.get("ic"))
    for attempt in range(4):
        if ic is None or attempt > 0:
            ic = sys_obj.resolve_ic(rng.uniform(0.0, 1.0, sys_obj.dim))
        try:
            traj = sys_obj.integrate(
                final_time=final_time,
                dt=dt,
                ic=np.asarray(ic, dtype=float),
                backend="interp",
            )
        except (RuntimeError, ValueError):  # divergence / off-basin start
            ic = None
            continue
        t, y = traj.t, traj.y
        if len(y) > 50 and np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e6:
            drop = int(opts.get("transient_frac", 0.15) * len(y))
            return t[drop:], y[drop:]
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
    """
    if _use_engine_for_ode(entry, opts):
        return _ode_trajectory_engine(entry, opts)
    return _ode_trajectory_scipy(entry, opts)


def _render_ode(entry, plt, opts):
    t, y = _ode_trajectory(entry, opts)
    if opts.get("kind") == "field":
        return _render_field(entry, plt, y)
    if opts.get("kind") == "spacetime" or entry.cls().dim is None:
        return _render_spacetime(entry, plt, t, y)
    dim = y.shape[1]
    if dim >= 3:
        fig = plt.figure(figsize=(5.4, 4.2))
        ax = fig.add_subplot(projection="3d")
        ax.plot(y[:, 0], y[:, 1], y[:, 2], lw=0.35, color=_ACCENT)
        ax.set_axis_off()
    elif dim == 2:
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        ax.plot(y[:, 0], y[:, 1], lw=0.4, color=_ACCENT)
        ax.set_xticks([]), ax.set_yticks([])
    else:
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        ax.plot(t, y[:, 0], lw=0.8, color=_ACCENT)
        ax.set_xlabel("t")
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


def _render_dde(entry, plt, opts):
    sys_obj = entry.cls()
    final_time = opts.get("final_time", 300.0)
    dt = opts.get("dt", 0.25)

    def history(s):
        return [0.8 + 0.2 * np.sin(0.2 * s)] * sys_obj.dim

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


def _render_map(entry, plt, opts):
    sys_obj = entry.cls()
    steps = opts.get("steps", 20_000)
    traj = sys_obj.iterate(steps=steps, max_retries=15)
    y = traj.y[100:]
    if sys_obj.dim == 1:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.scatter(y[:-1, 0], y[1:, 0], s=0.4, color=_ACCENT, linewidths=0)
        ax.set_xlabel(r"$x_n$")
        ax.set_ylabel(r"$x_{n+1}$")
    elif sys_obj.dim == 2:
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        ax.scatter(y[:, 0], y[:, 1], s=0.25, color=_ACCENT, linewidths=0)
        ax.set_xticks([]), ax.set_yticks([])
    else:
        fig = plt.figure(figsize=(5.4, 4.2))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=0.25, color=_ACCENT, linewidths=0)
        ax.set_axis_off()
    return fig


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
        else:
            fig = _render_map(entry, plt, opts)
        fig.savefig(cached, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:  # noqa: BLE001 — soft-fail, page ships without figure
        plt.close("all")
        return None

    shutil.copy(cached, out)
    return out
