"""Generate the tutorial-page showcase figures (one or more per tutorial).

Each render function uses the real tsdynamics library to produce a didactic,
on-brand figure for a tutorial page: transparent background, the brand teal
(``#11857A``) / indigo (``#574FCF``) accents, viridis/twilight where a colour
map is called for, and IBM Plex label text (the SVG carries the font-family so
the site's IBM Plex webfont applies when the page renders). The SVGs are
committed static assets under ``docs/assets/figures/tutorials/``; re-run after
changing a generator::

    .venv/bin/python docs/_tooling/make_tutorial_figures.py

Every figure here mirrors the numbers used in the tutorial prose, so it is
reproducible: the same explicit initial conditions / seeds appear in both.
"""

from __future__ import annotations

import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets" / "figures" / "tutorials"

#: Brand palette (docs/assets/brand/tokens.css) — teal primary, indigo accent,
#: amber/rose as warm secondaries.
TEAL, INDIGO, AMBER, ROSE = "#11857A", "#574FCF", "#E8912D", "#D64562"

#: Label font — IBM Plex Sans, with DejaVu as the layout fallback.
_FONT_STACK = ["IBM Plex Sans", "DejaVu Sans", "sans-serif"]


def _style():
    """Configure matplotlib for the on-brand, transparent, SVG-friendly house style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": _FONT_STACK,
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


# --------------------------------------------------------------------------- #
# Tutorial 1 — Basins & multistability (tilted two-well Duffing)
# --------------------------------------------------------------------------- #


class _TiltedDuffing:
    """Factory for the tilted two-well Duffing used in the basins tutorial."""

    @staticmethod
    def make():
        import tsdynamics as ts

        class TiltedDuffing(ts.ContinuousSystem):
            params = {"delta": 0.3, "F": 0.0}
            dim = 2
            variables = ("x", "y")

            @staticmethod
            def _equations(Y, t, *, delta, F):
                x, y = Y(0), Y(1)
                return (y, x - x**3 - delta * y + F)

        return TiltedDuffing


def fig_basins_image(plt, out_path):
    """Two-well Duffing basin image (F=0): interleaved wells, smooth boundary."""
    import numpy as np

    import tsdynamics as ts
    from tsdynamics import data

    sys = _TiltedDuffing.make()()  # F = 0

    grid = data.Grid(np.array([-2.0, -2.0]), np.array([2.0, 2.0]), (300, 300))
    basins = ts.basins_of_attraction(sys, grid, dt=0.5, max_steps=2000)
    labels = np.asarray(basins.labels)

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([TEAL, INDIGO])

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.imshow(
        labels.T,
        origin="lower",
        extent=[-2, 2, -2, 2],
        cmap=cmap,
        interpolation="nearest",
        alpha=0.92,
    )
    # Mark the two attractors (well bottoms).
    for a in basins.attractors:
        ax.plot(a.center[0], a.center[1], "o", ms=6, mfc="white", mec="#222", mew=1.2, zorder=5)
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$\\dot{x}_0$")
    ax.set_title("Basins of the two wells  ($F = 0$)")
    ax.text(
        -1.0,
        0.0,
        "left well\n$x=-1$",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
    )
    ax.text(
        1.0,
        0.0,
        "right well\n$x=+1$",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_basins_continuation(plt, out_path):
    """Tilt continuation: left-well basin fraction shrinks and annihilates at F≈0.4."""
    import numpy as np

    import tsdynamics as ts
    from tsdynamics import data

    sys = _TiltedDuffing.make()()
    region = data.Box(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

    values = np.linspace(0.0, 0.5, 11)
    cont = ts.continuation(
        sys, "F", values, region, n=300, resolution=40, dt=0.5, max_steps=2000, seed=0
    )
    fr = cont.fractions
    # attractor 1 = left well, 2 = right well (ids from the F=0 sort order)
    left = np.nan_to_num(np.asarray(fr[1], float), nan=0.0)
    right = np.nan_to_num(np.asarray(fr[2], float), nan=0.0)

    tp = ts.tipping_points(cont)
    tip_F = tp[0]["value"] if len(tp) else None

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.plot(values, left, "-o", color=TEAL, lw=1.6, ms=4, label="left well  ($x<0$)")
    ax.plot(values, right, "-o", color=INDIGO, lw=1.6, ms=4, label="right well  ($x>0$)")
    if tip_F is not None:
        ax.axvline(tip_F, color=ROSE, lw=1.0, ls="--", alpha=0.85)
        ax.annotate(
            "left basin\nannihilates\n(saddle-node)",
            xy=(tip_F, 0.02),
            xytext=(tip_F - 0.02, 0.28),
            ha="right",
            va="center",
            fontsize=8,
            color=ROSE,
            arrowprops=dict(arrowstyle="->", color=ROSE, lw=0.9),
        )
    ax.set_xlabel("tilt  $F$")
    ax.set_ylabel("basin fraction")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center left")
    ax.set_title("Basin stability as the potential is tilted")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tutorial 2 — Poincaré sections & return maps
# --------------------------------------------------------------------------- #


def fig_poincare_section(plt, out_path):
    """Rössler flow with the y=0 section and its crossings (x, z)."""
    import numpy as np

    import tsdynamics as ts

    ros = ts.systems.Rossler()
    ic = ros.integrate(final_time=200.0, dt=0.02, ic=[1.0, 1.0, 1.0]).y[-1]

    # A short window of the flow for context.
    traj = ros.integrate(final_time=120.0, dt=0.01, ic=ic)
    sec = ts.poincare_section(ros, plane=("y", 0.0, "up"), crossings=500, dt=0.02, seed=0)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(traj["x"], traj["z"], color=INDIGO, lw=0.4, alpha=0.16, zorder=1)
    ax.scatter(
        sec.y[:, 0],
        sec.y[:, 2],
        s=4.0,
        color=TEAL,
        alpha=0.9,
        linewidths=0,
        zorder=3,
        label="crossings of  $y=0$  (upward)",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.legend(loc="upper left", markerscale=2.5, handletextpad=0.4)
    ax.set_title("Rössler section: the flow collapses onto a thin sheet")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_return_map(plt, out_path):
    """Two return maps: Rössler section x_{n+1}(x_n) and the Lorenz z-maxima cusp."""
    import numpy as np

    import tsdynamics as ts

    # Rössler first-return in x at the section.
    ros = ts.systems.Rossler()
    rm = ts.return_map(
        ros, "x", method="poincare", plane=("y", 0.0), direction=+1, n=500, dt=0.02, seed=0
    )

    # Lorenz z-maxima cusp.
    lor = ts.systems.Lorenz()
    ic_l = lor.integrate(final_time=40.0, dt=0.01, ic=[1.0, 1.0, 1.0]).y[-1]
    zc = ts.return_map(lor, "z", method="max", n=2000, final_time=400.0, dt=0.01, ic=ic_l)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.4, 3.7))

    # -- Rössler return map --
    lo, hi = rm.current.min(), rm.current.max()
    a0.plot([lo, hi], [lo, hi], color="#888888", lw=0.8, ls="--", alpha=0.7, zorder=1)
    a0.scatter(rm.current, rm.successor, s=6, color=TEAL, alpha=0.85, linewidths=0, zorder=3)
    a0.set_xlabel("$x_n$")
    a0.set_ylabel("$x_{n+1}$")
    a0.set_title("Rössler section return map")
    a0.set_aspect("equal", "box")

    # -- Lorenz z-maxima cusp --
    lo2, hi2 = zc.current.min(), zc.current.max()
    a1.plot([lo2, hi2], [lo2, hi2], color="#888888", lw=0.8, ls="--", alpha=0.7, zorder=1)
    a1.scatter(zc.current, zc.successor, s=6, color=INDIGO, alpha=0.85, linewidths=0, zorder=3)
    a1.set_xlabel("$z_n$")
    a1.set_ylabel("$z_{n+1}$")
    a1.set_title("Lorenz $z$-maxima cusp map")
    a1.set_aspect("equal", "box")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tutorial 3 — Noise-driven dynamics (SDEs)
# --------------------------------------------------------------------------- #


def fig_ou_ensemble(plt, out_path):
    """OU: a few sample paths + the ensemble mean relaxing to mu with a ±sd band."""
    import numpy as np

    import tsdynamics as ts

    ou = ts.systems.OrnsteinUhlenbeck(params={"theta": 1.0, "mu": 0.0, "sigma": 0.3})

    fig, ax = plt.subplots(figsize=(6.4, 3.9))

    # A handful of sample paths.
    for i, seed in enumerate((0, 3, 7, 11, 19)):
        p = ou.integrate(final_time=8.0, dt=0.01, ic=[2.0], seed=seed)
        ax.plot(p.t, p.y[:, 0], color=TEAL, lw=0.7, alpha=0.5, zorder=2)

    # Ensemble mean ± sd on a grid of horizons.
    ics = np.full((3000, 1), 2.0)
    ts_grid = np.linspace(0.2, 8.0, 40)
    means, sds = [], []
    for T in ts_grid:
        f = ou.ensemble(ics, final_time=float(T), dt=0.01, seed=0)
        means.append(f.mean())
        sds.append(f.std())
    means = np.asarray(means)
    sds = np.asarray(sds)

    # Analytic transient mean and stationary sd.
    ax.plot(ts_grid, means, color=INDIGO, lw=2.0, zorder=4, label="ensemble mean")
    ax.fill_between(
        ts_grid,
        means - sds,
        means + sds,
        color=INDIGO,
        alpha=0.15,
        zorder=1,
        label="ensemble  $\\pm\\,$sd",
    )
    ax.plot(
        ts_grid,
        2.0 * np.exp(-ts_grid),
        color=ROSE,
        lw=1.0,
        ls="--",
        zorder=5,
        label=r"$\mu + (x_0-\mu)e^{-\theta t}$",
    )
    sd_inf = 0.3 / np.sqrt(2.0)  # sqrt(sigma^2 / 2 theta)
    ax.axhline(sd_inf, color="#888888", lw=0.7, ls=":", alpha=0.8)
    ax.axhline(-sd_inf, color="#888888", lw=0.7, ls=":", alpha=0.8)
    ax.set_xlabel("time  $t$")
    ax.set_ylabel("$x$")
    ax.set_title("Ornstein–Uhlenbeck: relaxation to the stationary law")
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_double_well_switching(plt, out_path):
    """DoubleWell path hopping between wells + the deterministic skeleton."""
    import numpy as np

    import tsdynamics as ts

    dw = ts.systems.DoubleWell(params={"a": 1.0, "b": 1.0, "sigma": 0.5})
    path = dw.integrate(final_time=200.0, dt=0.01, ic=[-1.0], seed=1)
    t, x = path.t, path.y[:, 0]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [2.6, 1.0]})

    # -- The telegraph path --
    a0.axhline(1.0, color="#888888", lw=0.7, ls=":", alpha=0.7)
    a0.axhline(-1.0, color="#888888", lw=0.7, ls=":", alpha=0.7)
    a0.axhline(0.0, color=ROSE, lw=0.7, ls="--", alpha=0.6)
    a0.plot(t, x, color=TEAL, lw=0.6, alpha=0.9)
    a0.set_xlabel("time  $t$")
    a0.set_ylabel("$x$")
    a0.set_title("Kramers hopping  ($\\sigma = 0.5$)")
    a0.text(6, 1.35, "barrier at $x=0$", color=ROSE, fontsize=8)

    # -- The potential U(x) = -x^2/2 + x^4/4 --
    xs = np.linspace(-1.8, 1.8, 400)
    U = -(xs**2) / 2 + (xs**4) / 4
    a1.plot(U, xs, color=INDIGO, lw=1.6)
    a1.plot([U[np.argmin(np.abs(xs + 1))]], [-1.0], "o", color=TEAL, ms=6)
    a1.plot([U[np.argmin(np.abs(xs - 1))]], [1.0], "o", color=TEAL, ms=6)
    a1.plot([U[np.argmin(np.abs(xs))]], [0.0], "o", color=ROSE, ms=5)
    a1.set_xlabel("$U(x)$")
    a1.set_title("potential")
    a1.tick_params(labelleft=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


FIGURES = {
    "basins-image": fig_basins_image,
    "basins-continuation": fig_basins_continuation,
    "poincare-section": fig_poincare_section,
    "return-map": fig_return_map,
    "ou-ensemble": fig_ou_ensemble,
    "double-well-switching": fig_double_well_switching,
}


def main():
    """Render every tutorial figure to docs/assets/figures/tutorials/<slug>.svg."""
    plt = _style()
    OUT.mkdir(parents=True, exist_ok=True)
    for slug, fn in FIGURES.items():
        plt.close("all")
        out = OUT / f"{slug}.svg"
        fn(plt, str(out))
        print(f"  ok {slug:24} {out.stat().st_size:>8} bytes")


if __name__ == "__main__":
    main()
