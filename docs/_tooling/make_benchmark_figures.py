"""Generate the References/Benchmarks figures from the recorded benchmark data.

These charts are built from the numbers already committed in
``benchmarks/RESULTS.md`` (best-of-N wall time, one full run) — this script does
NOT re-run the benchmark, it only renders the recorded results on-brand. Two
charts: integration and the analysis toolkit against the **Python** ecosystem,
where TSDynamics leads by 1–2 orders of magnitude. (The DynamicalSystems.jl
comparison is a table in the page, not a chart — see ``benchmarks.md``.)

House style matches ``make_analysis_figures.py``: transparent background, the
brand teal (``#11857A``) / indigo (``#574FCF``) accents, IBM Plex label text
(the SVG carries the font-family so the site webfont applies), text kept as
``<text>`` so it themes with the page. Re-run after editing the recorded
numbers::

    .venv/bin/python docs/_tooling/make_benchmark_figures.py
"""

from __future__ import annotations

import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets" / "figures" / "references"

#: Brand palette (docs/assets/brand/tokens.css).
TEAL, INDIGO, AMBER, ROSE = "#11857A", "#574FCF", "#E8912D", "#D64562"
_FONT_STACK = ["IBM Plex Sans", "DejaVu Sans", "sans-serif"]


def _style():
    """Configure matplotlib for the on-brand, transparent, SVG-friendly house style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "none",  # keep text as <text> so IBM Plex applies in-browser
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


def fig_integration_speedup(plt, out_path):
    """How many times faster TSDynamics integrates than each Python baseline.

    Numbers are the recorded best-of-N wall times from ``benchmarks/RESULTS.md``
    (Lorenz, DOP853, rtol=atol=1e-9). Speedup = baseline_time / tsdynamics_time.
    """
    import numpy as np

    # (label, TSDynamics interp ms, jit ms, baseline ms) — from RESULTS.md.
    rows = [
        ("Integration\nshort (T=100)", 7.78, 3.58, [("SciPy", 480.47), ("dysts", 1544.0)]),
        ("Integration\nlong (T=10000)", 780.08, 351.24, [("SciPy", 54921.0), ("dysts", 156802.0)]),
        ("Poincaré section\n(Rössler)", 202.93, 198.11, [("SciPy", 5306.0)]),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 3.9))

    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]  # top-to-bottom
    bar_h = 0.34

    # Two bars per row: interp-vs-slowest-baseline and jit-vs-slowest-baseline.
    interp_speed, jit_speed, annot = [], [], []
    for _, ti, tj, bases in rows:
        slow = max(b for _, b in bases)  # headline against the slowest baseline
        interp_speed.append(slow / ti)
        jit_speed.append(slow / tj)
        base_name = max(bases, key=lambda kv: kv[1])[0]
        annot.append(base_name)

    ax.barh(y + bar_h / 2, interp_speed, height=bar_h, color=TEAL, label="interp", zorder=3)
    ax.barh(y - bar_h / 2, jit_speed, height=bar_h, color=INDIGO, label="jit", zorder=3)

    for yi, s in zip(y + bar_h / 2, interp_speed, strict=True):
        ax.annotate(
            f"{s:.0f}×",
            xy=(s, yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            color=TEAL,
            fontweight="bold",
            clip_on=False,
        )
    for yi, s in zip(y - bar_h / 2, jit_speed, strict=True):
        ax.annotate(
            f"{s:.0f}×",
            xy=(s, yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            color=INDIGO,
            fontweight="bold",
            clip_on=False,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{lbl}\nvs {a}" for lbl, a in zip(labels, annot, strict=True)], fontsize=8.5
    )
    ax.set_xlabel("speedup  (baseline wall time ÷ TSDynamics wall time)")
    ax.set_xlim(0, max(jit_speed) * 1.18)
    ax.axvline(1.0, color="#888888", lw=0.8, ls=":", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Integration speed — TSDynamics vs the Python baselines", loc="left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_analysis_speedup(plt, out_path):
    """Analysis toolkit speed on a shared time series vs the from-data libraries.

    Every library is fed the *same* generated series (see the benchmark
    methodology), so this isolates the estimator. Speedup = fastest competitor
    on that row ÷ TSDynamics. Bars > 1 mean TSDynamics is faster; the two rows
    where a competitor wins are drawn below the 1× line and coloured amber.
    """
    import numpy as np

    # (label, tsdynamics ms, [(competitor, ms), ...]) — from RESULTS.md.
    rows = [
        ("Embedding dim\n(Cao / FNN)", 26.84, [("nolitsa", 1678.0), ("neurokit2", 215.65)]),
        ("Sample entropy", 21.09, [("antropy", 18.09), ("neurokit2", 16.57), ("nolds", 475.90)]),
        (
            "Corr. dimension\n(embedded)",
            210.66,
            [("nolitsa", 375.71), ("nolds", 1864.0), ("dysts", 1220.0), ("neurokit2", 1550.0)],
        ),
        ("Max. Lyapunov\nfrom data", 33.68, [("nolds", 283.17), ("nolitsa", 202.98)]),
        ("RQA determinism", 19.29, [("pyunicorn", 34.91), ("neurokit2", 151.11)]),
        ("Multiscale entropy", 30.54, [("neurokit2", 186.36)]),
        ("IAAFT surrogate", 25.01, [("nolitsa", 21.76), ("neurokit2", 14.90)]),
    ]

    labels, speeds, wins = [], [], []
    for lbl, ts_ms, comps in rows:
        best = min(c for _, c in comps)  # fastest competitor is the honest bar
        speeds.append(best / ts_ms)  # >1 ⇒ TSDynamics faster
        labels.append(lbl)
        wins.append(best / ts_ms >= 1.0)

    order = np.argsort(speeds)  # slowest→fastest, so biggest win at top
    labels = [labels[i] for i in order]
    speeds = [speeds[i] for i in order]
    wins = [wins[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(labels))
    colors = [TEAL if w else AMBER for w in wins]
    ax.barh(y, speeds, color=colors, zorder=3, height=0.62)

    for yi, s, w in zip(y, speeds, wins, strict=True):
        txt = f"{s:.1f}× faster" if w else f"{1 / s:.1f}× slower"
        ax.annotate(
            txt,
            xy=(s, yi),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            color=TEAL if w else AMBER,
            fontweight="bold",
            clip_on=False,
        )

    ax.axvline(1.0, color="#888888", lw=0.9, ls=":", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.2, max(speeds) * 2.6)
    ax.set_xticks([0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.3×", "1×", "3×", "10×", "30×"])
    ax.set_xlabel("TSDynamics speed relative to the fastest competitor  (log scale)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Analysis toolkit — same series, fastest competitor per task", loc="left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



FIGURES = {
    "integration-speedup": fig_integration_speedup,
    "analysis-speedup": fig_analysis_speedup,
}


def main():
    """Render every benchmark figure to docs/assets/figures/references/<slug>.svg."""
    plt = _style()
    OUT.mkdir(parents=True, exist_ok=True)
    for slug, fn in FIGURES.items():
        plt.close("all")
        out = OUT / f"{slug}.svg"
        fn(plt, str(out))
        print(f"  ok {slug:20} {out.stat().st_size:>7} bytes")


if __name__ == "__main__":
    main()
