"""
Build-time figure rendering for the per-system documentation pages.

Strategy
--------
- **ODE** figures integrate with ``scipy.solve_ivp`` over the system's
  SymEngine-lambdified numeric RHS (``_rhs_numeric``) — no engine needed, no
  warmup, ~0.1–1 s per system.
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

_ACCENT = "#5e35b1"
_ACCENT_2 = "#00897b"

#: Per-system rendering overrides: final_time, dt, ic, kind, transient_frac.
FIG_OVERRIDES: dict[str, dict] = {
    "Lorenz96": {"kind": "spacetime", "final_time": 60.0, "dt": 0.1},
    "KuramotoSivashinsky": {"kind": "spacetime", "final_time": 150.0, "dt": 0.5},
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


RENDERER_VERSION = "2"  # bump manually when rendering output materially changes


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


def _ode_trajectory(entry, opts) -> tuple[np.ndarray, np.ndarray]:
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


def _render_ode(entry, plt, opts):
    t, y = _ode_trajectory(entry, opts)
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
