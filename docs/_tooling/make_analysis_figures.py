"""
Generate the analysis-page showcase figures (one per analysis tool).

Each render function uses the real tsdynamics library to produce a didactic,
on-brand (indigo->teal) figure for its docs page. The PNGs are committed static
assets under docs/assets/figures/analysis/; re-run after changing a generator.

    uv run python docs/_tooling/make_analysis_figures.py
"""

from __future__ import annotations

import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets" / "figures" / "analysis"

#: House style — transparent, hairline, quiet (matches docs/_tooling/figures.py).
INDIGO, TEAL, AMBER, ROSE = "#4f46e5", "#0d9488", "#f59e0b", "#e11d48"


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
            "axes.titlesize": 10,
            "legend.frameon": False,
        }
    )
    return plt



def fig_lyapunov(plt, out_path):
    import numpy as np
    import tsdynamics as ts
    from tsdynamics import TangentSystem

    INDIGO = "#4f46e5"
    TEAL = "#0d9488"
    AMBER = "#f59e0b"
    ROSE = "#e11d48"

    known = np.array([0.906, 0.0, -14.57])

    # Settle onto the Lorenz attractor, then track the running spectrum.
    lor = ts.systems.Lorenz()
    settle = lor.integrate(final_time=40.0, dt=0.01)
    ic = settle.y[-1]

    tang = TangentSystem(lor, k=3)
    tang.reinit(ic)

    dt = 0.05
    record_every = 50          # ~ every 2.5 time units
    n_steps = 12000            # final time ~ 600
    times, exps = [], []
    for i in range(1, n_steps + 1):
        tang.step(dt)
        if i % record_every == 0:
            times.append(tang.time())
            exps.append(tang.exponents())
    times = np.asarray(times)
    exps = np.asarray(exps)    # (n_record, 3)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    cols = [INDIGO, TEAL, AMBER]
    for j in range(3):
        ax.plot(times, exps[:, j], color=cols[j], lw=1.3, label=labels[j], zorder=3)

    # Known literature values (dashed ROSE reference lines).
    for val in known:
        ax.axhline(val, color=ROSE, lw=0.9, ls="--", alpha=0.8, zorder=2)
    xr = times[-1]
    for val, txt in zip(known, ["0.906", "0.0", "-14.57"]):
        ax.annotate(txt, xy=(xr, val), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color=ROSE, clip_on=False)

    ax.set_xscale("log")
    ax.set_xlim(times[0], xr)
    ax.set_xlabel("elapsed time  $t$")
    ax.set_ylabel("running Lyapunov exponents")
    ax.margins(x=0)
    ax.legend(loc="center right", fontsize=8, handlelength=1.4)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_orbit_diagram(plt, out_path):
    import numpy as np
    import tsdynamics as ts

    INDIGO = "#4f46e5"
    ROSE = "#e11d48"

    rs = np.linspace(2.5, 4.0, 1400)
    od = ts.orbit_diagram(
        ts.systems.Logistic(), "r", rs,
        n=180, transient=600, component=0,
    )
    x, y = od.flat()

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(x, y, s=0.05, c=INDIGO, alpha=0.35, edgecolors="none", rasterized=True)

    # textbook period-doubling onsets
    for r1 in (3.0, 1.0 + np.sqrt(6.0)):
        ax.axvline(r1, color=ROSE, lw=0.8, ls=(0, (4, 3)), alpha=0.7, zorder=5)
    ax.text(3.0, 1.04, r"$r_1=3$", color=ROSE, fontsize=8, ha="center", va="bottom")
    ax.text(1.0 + np.sqrt(6.0), 1.04, r"$r_2=1+\sqrt{6}$", color=ROSE,
            fontsize=8, ha="center", va="bottom")

    ax.set_xlim(2.5, 4.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("growth rate  $r$")
    ax.set_ylabel("asymptotic  $x$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_recurrence(plt, out_path):
    import numpy as np
    from matplotlib.colors import ListedColormap
    import tsdynamics as ts

    INDIGO = "#4f46e5"
    N = 220
    burn = 200  # discard the transient so both signals sit on their attractor
    rate = 0.06  # comparable density for both panels (scale-free)

    # Periodic: logistic at r=3.5 (a period-4 cycle) -> diagonal-line texture.
    per = ts.Logistic(params={"r": 3.5}).iterate(steps=N + burn, ic=[0.31]).y[burn:, 0]
    rm_per = ts.recurrence_matrix(per, recurrence_rate=rate, theiler=1)

    # Chaotic: logistic at r=4.0 -> broken-up, speckled structure.
    cha = ts.Logistic(params={"r": 4.0}).iterate(steps=N + burn, ic=[0.4]).y[burn:, 0]
    rm_cha = ts.recurrence_matrix(cha, recurrence_rate=rate, theiler=1)

    # Binary colormap: transparent for 0, INDIGO for a recurrent point.
    cmap = ListedColormap([(0, 0, 0, 0), INDIGO])

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.5))
    for ax, rm, label in (
        (axes[0], rm_per, "periodic   $r = 3.5$"),
        (axes[1], rm_cha, "chaotic   $r = 4.0$"),
    ):
        ax.imshow(rm.toarray(), cmap=cmap, origin="lower", interpolation="none",
                  vmin=0, vmax=1, aspect="equal")
        ax.set_title(label)
        ax.set_xlabel("$i$")
        ax.set_xticks([0, N // 2, N])
        ax.set_yticks([0, N // 2, N])
        for s in ax.spines.values():
            s.set_visible(True)
    axes[0].set_ylabel("$j$")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_dimensions(plt, out_path):
    import numpy as np
    import tsdynamics as ts

    INDIGO = "#4f46e5"
    ROSE = "#e11d48"

    # Hénon attractor point cloud (already decorrelated -> no Theiler window needed)
    traj = ts.Henon().trajectory(8000, transient=500, ic=[0.1, 0.1])

    # D2 estimate + the log-log scaling curve it was read from
    res = ts.correlation_dimension(traj, n_radii=32, min_window=8)
    x, y = res.x, res.y                 # log r , log C(r)
    lo, hi = res.fit_slice              # inclusive indices of the fitted region
    D2 = float(res)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # all correlation-sum points
    ax.scatter(x, y, s=14, color=INDIGO, zorder=3, label="$\\log C(r)$ vs $\\log r$",
               edgecolors="none")
    # highlight the points that fall inside the fitted scaling region
    ax.scatter(x[lo:hi + 1], y[lo:hi + 1], s=34, facecolors="none",
               edgecolors=INDIGO, linewidths=1.1, zorder=4)

    # fitted scaling-region line, extended a touch beyond the window
    xpad = 0.12 * (x[hi] - x[lo])
    xline = np.array([x[lo] - xpad, x[hi] + xpad])
    yline = res.intercept + D2 * xline
    ax.plot(xline, yline, color=ROSE, lw=1.6, zorder=5,
            label=f"fit: slope $= D_2 = {D2:.2f}$")

    # slope annotation near the line midpoint
    xm = 0.5 * (x[lo] + x[hi])
    ym = res.intercept + D2 * xm
    ax.annotate(f"$D_2 \\approx {D2:.2f}$", xy=(xm, ym),
                xytext=(8, -22), textcoords="offset points",
                color=ROSE, fontsize=10, fontweight="bold")

    ax.set_xlabel(r"$\log r$")
    ax.set_ylabel(r"$\log C(r)$")
    ax.legend(loc="upper left", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_embedding(plt, out_path):
    import numpy as np
    import tsdynamics as ts

    INDIGO = "#4f46e5"
    TEAL = "#0d9488"

    # Integrate Rossler; we will keep ONLY x(t) for the reconstruction.
    ros = ts.Rossler()
    traj = ros.integrate(final_time=400.0, dt=0.05)
    # drop transient
    n0 = int(traj.y.shape[0] * 0.15)
    xt = traj.y[n0:, 0]
    yt = traj.y[n0:, 1]

    # Pick a delay from x(t) ALONE via mutual information.
    tau = ts.optimal_delay(xt, method="mi", max_delay=120)

    # Takens embedding of the single observable into 3D.
    emb = ts.embed(xt, dimension=3, delay=tau)
    rx, ry = emb[:, 0], emb[:, 1]  # (x(t), x(t-tau))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.4, 3.4))

    # LEFT: the TRUE (x, y) projection of the real attractor.
    axL.plot(xt, yt, color=INDIGO, lw=0.45, alpha=0.85)
    axL.set_xlabel("$x(t)$")
    axL.set_ylabel("$y(t)$")
    axL.set_title("true state space", color="#888888")
    axL.set_aspect("equal", adjustable="datalim")

    # RIGHT: the reconstruction from one coordinate.
    axR.plot(rx, ry, color=TEAL, lw=0.45, alpha=0.85)
    axR.set_xlabel("$x(t)$")
    axR.set_ylabel(rf"$x(t-{tau}\,\Delta t)$")
    axR.set_title("delay reconstruction", color="#888888")
    axR.set_aspect("equal", adjustable="datalim")

    fig.text(0.5, -0.01, rf"one observable $x(t)$ · $\tau={tau}$ samples (MI)",
             ha="center", color="#888888", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_chaos(plt, out_path):
    import numpy as np
    import tsdynamics as ts

    INDIGO = "#4f46e5"
    TEAL   = "#0d9488"
    AMBER  = "#f59e0b"
    ROSE   = "#e11d48"

    lor = ts.Lorenz()
    g2 = ts.gali(lor, 2, final_time=40.0, dt=0.05, seed=0)
    g3 = ts.gali(lor, 3, final_time=40.0, dt=0.05, seed=0)

    # `times` carries the absolute clock (transient burn-in included). Re-zero to
    # elapsed-since-tracking so the decay fills the panel.
    t2 = g2.times - g2.times[0]
    t3 = g3.times - g3.times[0]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    # Lyapunov spectrum for Lorenz: lambda1 ~ 0.906, lambda2 ~ 0, lambda3 ~ -14.57
    l1, l2, l3 = 0.9056, 0.0, -14.572
    slope2 = (l1 - l2)                      # GALI_2 ~ e^{-(l1-l2) t}
    slope3 = (l1 - l2) + (l1 - l3)          # GALI_3 ~ e^{-((l1-l2)+(l1-l3)) t}

    floor = 1e-17
    ax.semilogy(t2, np.clip(g2.values, floor, None), color=INDIGO, lw=1.4,
                label=r"$\mathrm{GALI}_2$  (measured)")
    ax.semilogy(t3, np.clip(g3.values, floor, None), color=TEAL, lw=1.4,
                label=r"$\mathrm{GALI}_3$  (measured)")

    # Theoretical Skokos-law slopes, anchored at the start of each measured curve.
    ref2 = g2.values[0] * np.exp(-slope2 * t2)
    ref3 = g3.values[0] * np.exp(-slope3 * t3)
    m2 = ref2 > floor
    m3 = ref3 > floor
    ax.semilogy(t2[m2], ref2[m2], color=ROSE, lw=0.9, ls="--",
                label=rf"$e^{{-(\lambda_1-\lambda_2)\,t}}$,  slope $\approx {slope2:.2f}$")
    ax.semilogy(t3[m3], ref3[m3], color=AMBER, lw=0.9, ls="--",
                label=rf"$e^{{-[(\lambda_1-\lambda_2)+(\lambda_1-\lambda_3)]\,t}}$,  slope $\approx {slope3:.1f}$")

    ax.set_xlabel("elapsed time  $t$")
    ax.set_ylabel(r"$\mathrm{GALI}_k$")
    ax.set_ylim(floor, 5.0)
    ax.set_xlim(0, float(t2[-1]))
    ax.legend(loc="upper right", fontsize=7.5, handlelength=1.8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_surrogate(plt, out_path):
    import numpy as np
    import tsdynamics as ts

    INDIGO = "#4f46e5"
    ROSE = "#e11d48"

    # Lorenz z-component — the discriminating observable for a time-reversal test.
    lor = ts.Lorenz()
    z = lor.trajectory(final_time=150.0, dt=0.05, transient=20.0)["z"]

    # Surrogate hypothesis test: IAAFT null preserving distribution + spectrum.
    res = ts.surrogate_test(
        z, statistic="time_reversal", method="iaaft", n=200, seed=0
    )
    null = res.surrogate_statistics
    data_stat = res.data_statistic
    p = res.p_value
    zscore = res.z_score

    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    # Histogram of the surrogate (null) statistic.
    lo = min(null.min(), 0.0)
    hi = max(null.max(), data_stat)
    pad = 0.06 * (hi - lo)
    bins = np.linspace(null.min() - 0.01, null.max() + 0.01, 26)
    ax.hist(
        null, bins=bins, color=INDIGO, alpha=0.32, edgecolor=INDIGO,
        linewidth=0.5, label=f"null ({res.n_surrogates} IAAFT surrogates)",
    )

    # Reference line at the original-data statistic, well outside the null.
    ax.axvline(data_stat, color=ROSE, lw=1.6, label="Lorenz z (data)")
    ymax = ax.get_ylim()[1]
    ax.annotate(
        f"data: {data_stat:.2f}\n$z = {zscore:.0f}\\,\\sigma$",
        xy=(data_stat, ymax * 0.62),
        xytext=(data_stat - 0.30, ymax * 0.72),
        color=ROSE, fontsize=9, ha="right", va="center",
        arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.0),
    )

    # p-value verdict.
    verdict = "reject" if res.rejected else "fail to reject"
    ax.text(
        0.03, 0.96,
        f"$p = {p:.3f} \\leq \\alpha = {res.alpha:g}$\n→ {verdict} linear null",
        transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#555555",
    )

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("time-reversal asymmetry statistic")
    ax.set_ylabel("surrogate count")
    ax.legend(loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_basins(plt, out_path):
    import numpy as np
    import tsdynamics as ts
    from tsdynamics import Grid
    from tsdynamics.analysis import basins as bas
    from matplotlib.colors import ListedColormap

    INDIGO = "#4f46e5"
    TEAL = "#0d9488"
    AMBER = "#f59e0b"

    class NewtonMap(ts.DiscreteMap):
        """Newton's method for z**3 - 1 = 0 -- three roots, Wada basins."""

        params: dict = {}
        dim = 2
        variables = ("re", "im")
        _jacobian_fd_check = False

        @staticmethod
        def _step(X):
            z = complex(X[0], X[1])
            z2 = z * z
            z3 = z2 * z
            z = (2.0 * z3 + 1.0) / (3.0 * z2)
            return (z.real, z.imag)

        @staticmethod
        def _jacobian(X):
            return ((0.0, 0.0), (0.0, 0.0))

    # Paint the basins on a square of the complex plane.  Newton on z^3 = 1 has
    # three roots (the cube roots of unity); the boundaries between their basins
    # form the fractal Julia set.
    lo, hi, n = -1.0, 1.0, 200
    nm = NewtonMap()
    res = bas.basins_of_attraction(
        nm,
        Grid([lo, lo], [hi, hi], (n, n)),
        consecutive_recurrences=8,
        attractor_locate_steps=5,
        max_steps=200,
    )

    labels = res.labels  # (n_re, n_im); -1 = diverged
    ids = res.attractors.ids
    centers = {k: res.attractors[k].center for k in ids}

    palette = [INDIGO, TEAL, AMBER]
    img = np.full(labels.shape, np.nan)
    for idx, k in enumerate(ids[:3]):
        img[labels == k] = idx

    cmap = ListedColormap(palette[: max(1, min(3, len(ids)))])
    cmap.set_bad(color="none")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    # labels[i, j] -> (re_i, im_j); transpose so re is x and im is y.
    ax.imshow(
        img.T,
        origin="lower",
        extent=[lo, hi, lo, hi],
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
    )

    for k in ids[:3]:
        c = centers[k]
        ax.plot(
            c[0], c[1],
            marker="o", markersize=7,
            markerfacecolor="white", markeredgecolor="#22222288",
            markeredgewidth=0.8, linestyle="none", zorder=5,
        )

    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_fixed_points(plt, out_path):
    import numpy as np
    import tsdynamics as ts

    INDIGO = "#4f46e5"
    TEAL = "#0d9488"
    ROSE = "#e11d48"

    # Autonomous van der Pol oscillator (the doc page's running example).
    class VanDerPol(ts.ContinuousSystem):
        params = {"mu": 1.0}
        dim = 2
        variables = ("x", "v")

        @staticmethod
        def _equations(y, t, mu):
            return [y(1), mu * (1.0 - y(0) * y(0)) * y(1) - y(0)]

    sys = VanDerPol(params={"mu": 1.0})

    # The limit cycle, found by single shooting (thick TEAL closed curve).
    orb = ts.periodic_orbit(sys, ic=[2.0, 0.0], period_guess=6.0, transient=20.0)
    cyc = np.asarray(orb.points)

    # The unstable equilibrium at the origin (ROSE x) via multi-start Newton.
    fps = ts.fixed_points(sys, region=([-3.0, -4.0], [3.0, 4.0]))

    # A few trajectories spiralling onto the cycle (thin INDIGO): from near the
    # unstable origin outward, and from far outside inward.
    seeds = [
        [0.15, 0.0], [-0.1, 0.1], [0.05, -0.2],
        [3.3, 0.0], [-3.3, 0.0], [0.0, 4.2], [0.0, -4.2],
    ]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    for s in seeds:
        traj = sys.integrate(ic=s, final_time=22.0, dt=0.01)
        ax.plot(traj["x"], traj["v"], color=INDIGO, lw=0.5, alpha=0.55, zorder=1)

    ax.plot(cyc[:, 0], cyc[:, 1], color=TEAL, lw=2.4, zorder=3,
            label=f"limit cycle (T = {orb.period:.3f})")

    for fp in fps:
        if not fp.stable:
            ax.scatter([fp.x[0]], [fp.x[1]], marker="X", s=85, color=ROSE,
                       zorder=5, linewidths=0, label="unstable equilibrium")
        else:
            ax.scatter([fp.x[0]], [fp.x[1]], marker="o", s=55, color=ROSE,
                       zorder=5, linewidths=0, label="stable equilibrium")

    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-4.6, 4.6)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_poincare(plt, out_path):
    import numpy as np
    import tsdynamics as ts

    INDIGO = "#4f46e5"
    TEAL = "#0d9488"

    sys = ts.systems.Rossler()

    # Faint full attractor for context (x-z projection).
    traj = sys.integrate(final_time=300.0, dt=0.02, ic=[1.0, 1.0, 0.0])
    bx = traj["x"]
    bz = traj["z"]

    # Poincare section: plane y = 0, ascending crossings -> one wing of the
    # attractor sampled stroboscopically; the crossings collapse onto a thin,
    # near-one-dimensional return set.
    pmap = ts.PoincareMap(sys, plane=(1, 0.0), direction=+1, dt=0.01)
    sec = pmap.trajectory(600, transient=50, ic=[1.0, 1.0, 0.0])
    sx = sec.y[:, 0]
    sz = sec.y[:, 2]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.plot(bx, bz, color=INDIGO, lw=0.4, alpha=0.15, zorder=1)
    ax.scatter(sx, sz, s=3.0, color=TEAL, alpha=0.9, linewidths=0, zorder=3,
               label="crossings of  $y=0$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.legend(loc="upper left", handletextpad=0.4, markerscale=2.2)

    # Annotate the collapse: full flow vs. the thin return set.
    ax.text(0.03, 0.80, "flow shown faint;\nsection collapses to\na thin 1-D return set",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#888888")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_entropy(plt, out_path):
    import numpy as np
    import tsdynamics as ts

    INDIGO = "#4f46e5"
    TEAL = "#0d9488"
    AMBER = "#f59e0b"

    rng = np.random.default_rng(0)
    n = 2500
    scales = 20

    # Chaotic: logistic map at r=4 (deterministic chaos)
    log = ts.Logistic(params={"r": 4.0})
    chaotic = log.trajectory(n, transient=500, ic=[0.1234]).y[:, 0]

    # Periodic: smooth sine
    t = np.linspace(0, 50 * np.pi, n)
    periodic = np.sin(t)

    # White noise, seeded
    noise = rng.standard_normal(n)

    mse_chaotic = ts.multiscale_entropy(chaotic, scales=scales)
    mse_periodic = ts.multiscale_entropy(periodic, scales=scales)
    mse_noise = ts.multiscale_entropy(noise, scales=scales)

    xs = np.arange(1, scales + 1)

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(xs, mse_chaotic, color=INDIGO, lw=1.5, marker="o", ms=3.5,
            label="logistic $r=4$ (chaotic)")
    ax.plot(xs, mse_noise, color=AMBER, lw=1.5, marker="s", ms=3.0,
            label="white noise")
    ax.plot(xs, mse_periodic, color=TEAL, lw=1.5, marker="^", ms=3.5,
            label="sine (periodic)")

    ax.set_xlabel("scale factor  τ")
    ax.set_ylabel("sample entropy")
    ax.set_xlim(0.5, scales + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    lo = float(np.nanmin([mse_chaotic.min(), mse_periodic.min(), mse_noise.min()]))
    ax.set_ylim(min(0.0, lo) - 0.05, None)
    ax.legend(loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_integrate(plt, out_path):
    import numpy as np
    import tsdynamics as ts
    from matplotlib.gridspec import GridSpec

    INDIGO = "#4f46e5"
    TEAL = "#0d9488"
    AMBER = "#f59e0b"

    lor = ts.Lorenz()
    traj = lor.integrate(final_time=60.0, dt=0.005, ic=[1.0, 1.0, 1.0])
    traj = traj.after(5.0)  # drop the transient onto the attractor

    t = traj.t
    x, y, z = traj["x"], traj["y"], traj["z"]

    fig = plt.figure(figsize=(6.4, 3.8))
    gs = GridSpec(3, 2, width_ratios=[1.05, 1.0], hspace=0.18, wspace=0.12, figure=fig)

    # Left: the 3D attractor
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.plot(x, y, z, color=INDIGO, lw=0.35, alpha=0.85)
    ax3d.set_axis_off()
    ax3d.view_init(elev=22, azim=-60)
    try:
        ax3d.set_box_aspect((1, 1, 0.9))
    except Exception:
        pass

    # Right: stacked x(t), y(t), z(t)
    series = [(x, INDIGO, "x"), (y, TEAL, "y"), (z, AMBER, "z")]
    axes = []
    for i, (s, c, lab) in enumerate(series):
        ax = fig.add_subplot(gs[i, 1], sharex=axes[0] if axes else None)
        ax.plot(t, s, color=c, lw=0.6)
        ax.set_ylabel(lab, color=c, rotation=0, labelpad=10, va="center")
        ax.margins(x=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i < 2:
            ax.tick_params(labelbottom=False)
        axes.append(ax)
    axes[-1].set_xlabel("time")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_solvers(plt, out_path):
    """Work-precision (explicit family) + stiff-problem wall-time comparison."""
    import time

    import numpy as np
    import tsdynamics as ts

    INDIGO = "#4f46e5"
    TEAL = "#0d9488"
    AMBER = "#f59e0b"
    ROSE = "#e11d48"

    # ----- Panel A: work-precision (achieved error vs requested tolerance) -----
    class _Decay(ts.ContinuousSystem):
        params = {"k": 1.0}
        dim = 1

        @staticmethod
        def _equations(y, t, *, k):
            return [-k * y(0)]

    decay = _Decay()
    T = 6.0
    exact = np.exp(-T)
    tols = np.array([1e-3, 1e-5, 1e-7, 1e-9, 1e-11])
    explicit = [("rk45", INDIGO, "o"), ("tsit5", TEAL, "s"), ("dop853", ROSE, "^")]

    ax1 = plt.subplot(1, 2, 1)
    for method, color, marker in explicit:
        errs = []
        for rtol in tols:
            tr = decay.integrate(
                final_time=T, dt=T, ic=[1.0],
                method=method, rtol=float(rtol), atol=float(rtol) * 1e-3,
            )
            errs.append(abs(tr.y[-1, 0] - exact))
        ax1.loglog(tols, errs, marker=marker, color=color, lw=1.6, ms=5, label=method)
    ax1.set_xlabel("requested tolerance  rtol")
    ax1.set_ylabel("achieved global error")
    ax1.set_title("Explicit work-precision", color="#888888", fontsize=9.5)
    ax1.invert_xaxis()
    ax1.grid(True, which="both", color="#88888822", lw=0.5)
    ax1.legend(loc="upper left", fontsize=8)

    # ----- Panel B: stiffness — explicit cost vs implicit cost -----------------
    class _StiffVDP(ts.ContinuousSystem):
        params = {"mu": 1000.0}
        dim = 2

        @staticmethod
        def _equations(y, t, *, mu):
            x, v = y(0), y(1)
            return [v, mu * (1 - x * x) * v - x]

    vdp = _StiffVDP()
    Tv = 3000.0

    def best_ms(method):
        best = np.inf
        xend = None
        for _ in range(3):
            t0 = time.perf_counter()
            tr = vdp.integrate(
                final_time=Tv, dt=Tv / 50, ic=[2.0, 0.0],
                method=method, rtol=1e-6, atol=1e-8,
            )
            best = min(best, (time.perf_counter() - t0) * 1e3)
            xend = tr.y[-1, 0]
        return best, xend

    bars = [
        ("rk45", INDIGO), ("tsit5", TEAL), ("dop853", ROSE),
        ("bdf", AMBER), ("rosenbrock", AMBER), ("trbdf2", AMBER),
    ]
    names = [b[0] for b in bars]
    colors = [b[1] for b in bars]
    times, xends = [], []
    for method, _ in bars:
        ms, xe = best_ms(method)
        times.append(ms)
        xends.append(xe)

    ax2 = plt.subplot(1, 2, 2)
    ypos = np.arange(len(names))
    ax2.barh(ypos, times, color=colors, alpha=0.85, height=0.62)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlabel("wall-time per integration  (ms, log)")
    ax2.set_title("Stiff van der Pol (mu=1000)", color="#888888", fontsize=9.5)
    ax2.grid(True, axis="x", which="both", color="#88888822", lw=0.5)
    for i, ms in enumerate(times):
        ax2.text(ms * 1.15, i, f"{ms:.0f}", va="center", fontsize=7.5, color="#888888")
    assert max(xends) - min(xends) < 1e-2, xends
    ax2.text(
        0.97, 0.04, "all land on x(T)={:.2f}".format(np.mean(xends)),
        transform=ax2.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="#888888", style="italic",
    )

    fig = plt.gcf()
    fig.set_size_inches(6.2, 3.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


FIGURES = {
    "lyapunov": fig_lyapunov,
    "orbit-diagram": fig_orbit_diagram,
    "recurrence": fig_recurrence,
    "dimensions": fig_dimensions,
    "embedding": fig_embedding,
    "chaos": fig_chaos,
    "surrogate": fig_surrogate,
    "basins": fig_basins,
    "fixed-points": fig_fixed_points,
    "poincare": fig_poincare,
    "entropy": fig_entropy,
    "integrate": fig_integrate,
    "solvers": fig_solvers,
}


def main():
    plt = _style()
    OUT.mkdir(parents=True, exist_ok=True)
    for slug, fn in FIGURES.items():
        plt.close("all")
        out = OUT / f"{slug}.png"
        fn(plt, str(out))
        print(f"  ok {slug:16} {out.stat().st_size:>7} bytes")


if __name__ == "__main__":
    main()
