"""Generate the Visualization-page showcase figures by dogfooding ``ts.viz``.

Every figure on this page is built through the **real** visualization API — the
``Trajectory.to_plot_spec`` / ``system.to_plot_spec`` front door, the
``ts.viz.plot`` composition seam, the fluent ``PlotSpec`` tweaks, the ``Theme``
system, and the ``Animation`` modifier — never raw matplotlib.  That keeps the
docs honest: the pictures are produced by the code the page teaches.

House style
-----------
A brand :class:`~tsdynamics.viz.style.Theme` (``"tsdynamics"``) is registered and
made the active default, so every spec inherits the brand palette (teal
``#11857A`` / indigo ``#574FCF`` + warm accents), IBM Plex label text, and a
transparent background.  Sequential colour data uses ``viridis``; cyclic data
uses ``twilight``.  The specs are rendered through the matplotlib backend and
saved as transparent SVGs (``svg.fonttype=none`` keeps the text as ``<text>`` so
the site's IBM Plex webfont applies); the one animation is a small looping GIF.

Every trajectory is integrated from an **explicit initial condition** so the
committed figure is exactly reproducible.  Outputs are static assets under
``docs/assets/figures/viz/``; re-run after changing a generator::

    .venv/bin/python docs/_tooling/make_viz_figures.py
"""

from __future__ import annotations

import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets" / "figures" / "viz"

#: Brand palette (docs/assets/brand/tokens.css) — teal primary, indigo accent,
#: amber / rose warm secondaries, bright teal for a fifth series.
TEAL, INDIGO, AMBER, ROSE, TEAL2 = "#11857A", "#574FCF", "#E8912D", "#D64562", "#2CC5AE"

#: Label font — IBM Plex Sans, with DejaVu as the layout fallback. Emitted
#: verbatim into the SVG so the page's IBM Plex webfont renders it in-browser.
_FONT_FAMILY = "IBM Plex Sans"


def _prepare():
    """Configure matplotlib for SVG-friendly transparent output and register the brand theme.

    Returns the ``matplotlib.pyplot`` handle (so ``plt.close("all")`` can run
    between figures) — but the *drawing* goes entirely through ``ts.viz``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "none",  # keep text as <text> so IBM Plex applies in-browser
            "savefig.transparent": True,
        }
    )

    from tsdynamics.viz import register_theme, set_theme
    from tsdynamics.viz.style import Theme

    brand = Theme(
        name="tsdynamics",
        palette=(TEAL, INDIGO, AMBER, ROSE, TEAL2),
        background=None,  # transparent — the SVG carries no facecolor
        foreground="#888888",
        font_family=_FONT_FAMILY,
        font_size=9.0,
        title_size=10.0,
        grid=False,
        line_width=1.3,
        marker_size=5.0,
    )
    register_theme(brand)
    set_theme("tsdynamics")
    return plt


def _save_svg(spec, out_path, *, transparent=True):
    """Render a ``PlotSpec`` through matplotlib and write a tight SVG.

    Transparent by default (the brand look); pass ``transparent=False`` when a
    theme's own background (e.g. the dark theme) must show through — a transparent
    save would drop the theme facecolor and leave dark-theme ink illegible.
    """
    fig = spec.render("matplotlib")
    figure = getattr(fig, "figure", fig)
    save_kw = {"bbox_inches": "tight"}
    if transparent:
        save_kw["transparent"] = True
    else:
        # Opaque page WITHOUT passing savefig(facecolor=) — that would override every
        # patch and drop a per-panel theme background (e.g. the dark theme). Instead
        # paint only the figure patch white and keep the axes' own facecolors.
        figure.patch.set_facecolor("white")
        save_kw["transparent"] = False
    figure.savefig(out_path, **save_kw)
    import matplotlib.pyplot as plt

    plt.close(figure)


# ---------------------------------------------------------------------------
# (1) The plot kinds — one real trajectory each
# ---------------------------------------------------------------------------


def fig_kind_time_series(plt, out_path):
    """Rössler x(t), y(t), z(t) as an overlaid TIME_SERIES (auto-dispatch on 3 components)."""
    import tsdynamics as ts
    from tsdynamics.viz.producers import time_series

    ros = ts.systems.Rossler()
    traj = ros.integrate(final_time=120.0, dt=0.02, ic=[1.0, 1.0, 0.0]).after(20.0)

    # The front door `to_plot_spec` auto-dispatches on component COUNT, so three
    # components would give a 3-D portrait. To overlay the three components as lines
    # over t, name the TIME_SERIES kind explicitly via the producer (one LINE per
    # component, palette-coloured, with a legend).
    spec = time_series(traj, components=["x", "y", "z"]).style(lw=0.9).size(6.4, 3.4)
    spec.relabel(title="")
    _save_svg(spec, out_path)


def fig_kind_phase_2d(plt, out_path):
    """Rössler (x, y) phase portrait coloured by time (2 components -> PHASE_PORTRAIT_2D)."""
    import tsdynamics as ts
    from tsdynamics.viz.producers import phase_portrait

    ros = ts.systems.Rossler()
    # A coarser dt keeps the colour-per-segment SVG light while tracing the orbit.
    traj = ros.integrate(final_time=180.0, dt=0.06, ic=[1.0, 1.0, 0.0]).after(20.0)

    # The parameterised producer takes `color_by=` (which the fixed to_plot_spec
    # signature cannot) — colour the curve by elapsed time along the orbit.
    spec = phase_portrait(traj, components=["x", "y"], color_by="time")
    spec.style(lw=0.7).colorize(colorbar=True).size(5.6, 4.6)
    spec.colorbar.cmap = "viridis"
    spec.colorbar.label = "time"
    spec.relabel(title="")
    _save_svg(spec, out_path)


def fig_kind_phase_3d(plt, out_path):
    """Lorenz butterfly as a 3-D PHASE_PORTRAIT_3D (auto-dispatch on 3 components)."""
    import tsdynamics as ts

    lor = ts.systems.Lorenz()
    traj = lor.integrate(final_time=60.0, dt=0.005, ic=[1.0, 1.0, 1.0]).after(5.0)

    # 3 components -> PHASE_PORTRAIT_3D (a LINE3D). Hide the axes for a clean
    # "attractor floating in space" look via the figure-level style(axes=False).
    spec = (
        traj.to_plot_spec(components=["x", "y", "z"])
        .style(lw=0.35, alpha=0.85, axes=False)
        .recolor(INDIGO)
        .camera(elev=22, azim=-60)
        .size(5.6, 4.8)
    )
    spec.relabel(title="")
    _save_svg(spec, out_path)


def fig_kind_spacetime(plt, out_path):
    """Lorenz-96 (20 sites) as a component-index-vs-time SPACETIME image (4+ components)."""
    import numpy as np

    import tsdynamics as ts

    l96 = ts.systems.Lorenz96()
    # Break the symmetric fixed point with a small bump on site 0 (pinned IC).
    ic = 0.01 * np.ones(l96.dim)
    ic[0] += 1.0
    traj = l96.integrate(final_time=30.0, dt=0.05, ic=ic).after(5.0)

    # 4+ components -> SPACETIME (an IMAGE), never a misleading 3-D portrait.
    spec = traj.to_plot_spec().style(cmap="viridis").size(6.6, 3.4)
    spec.colorbar.label = "$x_i$"
    spec.relabel(x="time", y="site index $i$", title="")
    _save_svg(spec, out_path)


def fig_kind_delay(plt, out_path):
    """Mackey–Glass delay embedding x(t) vs x(t - tau) — the natural view of a DDE."""
    import numpy as np

    import tsdynamics as ts

    mg = ts.systems.MackeyGlass()
    traj = mg.integrate(
        final_time=600.0,
        dt=0.5,
        history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)],
    ).after(100.0)

    # kind="delay" is a recipe: x(t) vs x(t - tau); tau is in TIME units (converted
    # to a sample lag via meta["dt"]). It routes to a PHASE_PORTRAIT_2D.
    spec = traj.to_plot_spec(kind="delay", components="x", tau=17.0)
    spec.style(lw=0.5, alpha=0.85).recolor(TEAL).size(4.8, 4.6)
    spec.relabel(title="")
    _save_svg(spec, out_path)


# ---------------------------------------------------------------------------
# (2) The four themes on one representative plot (a 2x2 grid)
# ---------------------------------------------------------------------------


def fig_themes(plt, out_path):
    """Show the four built-in themes (default / dark / minimal / publication) on one portrait."""
    import tsdynamics as ts
    from tsdynamics.viz import get_theme, plot

    lor = ts.systems.Lorenz()
    traj = lor.integrate(final_time=40.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(5.0)

    # Build the SAME (x, z) portrait four times, each pinned to a built-in theme,
    # then tile them into a 2x2 grid with the composition seam. Each panel keeps
    # its own theme (panel theme wins over the composite / global default).
    panels = []
    for name in ("default", "dark", "minimal", "publication"):
        p = traj.to_plot_spec(components=["x", "z"]).theme(get_theme(name))
        p.style(lw=0.5).relabel(title=name)
        panels.append(p)

    grid = plot(*panels, layout="grid").size(6.8, 6.2)
    # Opaque page so the dark theme's near-black panel background shows (a
    # transparent save would drop it and leave the dark-theme ink illegible).
    _save_svg(grid, out_path, transparent=False)


# ---------------------------------------------------------------------------
# (3) Styling — before / after fluent tweaks
# ---------------------------------------------------------------------------


def fig_styling(plt, out_path):
    """One portrait plain, then restyled with .recolor / .style / .grid / .background."""
    import tsdynamics as ts
    from tsdynamics.viz import plot

    lor = ts.systems.Lorenz()
    traj = lor.integrate(final_time=40.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(5.0)

    # LEFT: the bare spec (brand default look).
    before = traj.to_plot_spec(components=["x", "z"]).style(lw=0.5)
    before.relabel(title="before")

    # RIGHT: the same data, restyled with the chainable tweaks — a heavier indigo
    # line, a soft grid, and a light panel background. Each tweak mutates + returns
    # self, so they compose in one expression.
    after = (
        traj.to_plot_spec(components=["x", "z"])
        .recolor(INDIGO)
        .style(lw=1.1, alpha=0.9)
        .grid(True, color="#b9b4ec", alpha=0.5)
        .background("#f4f3fb")
    )
    after.relabel(title="after")

    row = plot(before, after, layout="row").size(7.2, 3.8)
    _save_svg(row, out_path)


# ---------------------------------------------------------------------------
# (4) Composition — overlay of two systems, and a 2x2 panel grid
# ---------------------------------------------------------------------------


def fig_compose_overlay(plt, out_path):
    """Overlay two Rössler orbits (different c) on ONE set of (x, y) axes."""
    import tsdynamics as ts
    from tsdynamics.viz import plot

    # Two Rössler variants — a small limit cycle (c=2.3) and the wide chaotic band
    # (c=5.7): clearly separated so the overlay reads at a glance.
    cycle = ts.Rossler(params={"c": 2.3})
    chaos = ts.Rossler(params={"c": 5.7})
    t1 = cycle.integrate(final_time=300.0, dt=0.02, ic=[1.0, 1.0, 0.0]).after(60.0)
    t2 = chaos.integrate(final_time=300.0, dt=0.02, ic=[1.0, 1.0, 0.0]).after(60.0)

    # layout="overlay" merges compatible single-panel specs onto one axes and
    # disambiguates the legend by source title. Titles seed the legend labels.
    t1.meta["system"] = "c = 2.3 (limit cycle)"
    t2.meta["system"] = "c = 5.7 (chaotic)"
    spec = plot(t1, t2, components=["x", "y"], layout="overlay")
    spec.recolor(TEAL, INDIGO).style(lw=0.7, alpha=0.85).size(5.8, 4.8)
    spec.relabel(title="")
    _save_svg(spec, out_path)


def fig_compose_grid(plt, out_path):
    """Tile four classic attractors into a 2x2 grid, each its own panel."""
    import tsdynamics as ts
    from tsdynamics.viz import plot

    # Lorenz / Rössler declare named variables; Halvorsen / Thomas do not, so those
    # select components by integer index (both spellings work through to_plot_spec).
    specs = []
    for name, ic, comps, color in (
        ("Lorenz", [1.0, 1.0, 1.0], ["x", "z"], INDIGO),
        ("Rossler", [1.0, 1.0, 0.0], ["x", "y"], TEAL),
        ("Halvorsen", [-5.0, 0.0, 0.0], [0, 1], AMBER),
        ("Thomas", [1.1, 1.1, -0.01], [0, 1], ROSE),
    ):
        sys = getattr(ts.systems, name)()
        # A coarser dt keeps each panel's SVG light while still tracing the attractor.
        traj = sys.integrate(final_time=160.0, dt=0.02, ic=ic).after(30.0)
        p = traj.to_plot_spec(components=comps).recolor(color).style(lw=0.4, alpha=0.85)
        p.relabel(title=name)
        specs.append(p)

    grid = plot(*specs, layout="grid").size(6.8, 6.4)
    _save_svg(grid, out_path)


# ---------------------------------------------------------------------------
# (5) Animation — a small looping reveal-comet GIF (Lorenz)
# ---------------------------------------------------------------------------


def fig_animation(plt, out_path):
    """Write a looping reveal-comet GIF of the Lorenz attractor drawing itself in."""
    import tsdynamics as ts

    lor = ts.systems.Lorenz()
    traj = lor.integrate(final_time=45.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(3.0)

    # Animation is an orthogonal modifier: any spec + an Animation becomes a movie.
    # A reveal comet — head at the current sample, a fading tail reaching back 6
    # time units — over the 3-D butterfly, axes hidden for a clean look.
    spec = (
        traj.to_plot_spec(components=["x", "y", "z"], animate=True)
        .animate(n_frames=120, fps=30)
        .trail(("time", 6.0), fade=True)
        .head(size=7.0, color=AMBER)
        .style(lw=0.6, axes=False)
        .recolor(INDIGO)
        .camera(elev=22, azim=-60)
        .size(4.8, 4.4)
    )
    # .save picks the writer from the extension: .gif -> matplotlib FuncAnimation.
    spec.save(str(out_path), dpi=80)


# ---------------------------------------------------------------------------
# (6) Spatial field — Gray–Scott 2-D field snapshot + Kuramoto–Sivashinsky 1-D
# ---------------------------------------------------------------------------


def fig_spatial_field(plt, out_path):
    """Gray–Scott 2-D activator field (heatmap) beside a Kuramoto–Sivashinsky space-time."""
    import numpy as np

    import tsdynamics as ts
    from tsdynamics.viz import plot
    from tsdynamics.viz.producers import spacetime

    # LEFT: Gray–Scott — a 48x48 reaction–diffusion field. kind="field" reshapes the
    # final-time state to its (Ny, Nx) grid (an IMAGE heatmap of the activator v).
    gs = ts.systems.GrayScott()
    gtr = gs.integrate(final_time=1500.0, dt=5.0)
    left = gtr.to_plot_spec(kind="field").style(cmap="viridis")
    left.relabel(title="Gray–Scott  (2-D field)")

    # RIGHT: Kuramoto–Sivashinsky — a 1-D PDE; its space-time diagram is the spatial
    # profile stacked over time. The `spacetime` producer with transpose=True puts
    # space on x and time on y (the canonical KS view).
    ks = ts.systems.KuramotoSivashinsky()
    ic = 0.1 * np.cos(np.linspace(0.0, 2.0 * np.pi, ks.dim, endpoint=False))
    ktr = ks.integrate(final_time=150.0, dt=0.5, ic=ic).after(20.0)
    right = spacetime(ktr, transpose=True).style(cmap="twilight")  # cyclic field -> twilight
    right.colorbar.label = "$u$"
    right.relabel(x="site index", y="time", title="Kuramoto–Sivashinsky  (1-D field, space–time)")

    row = plot(left, right, layout="row").size(7.4, 3.8)
    _save_svg(row, out_path)


# ---------------------------------------------------------------------------
# Registry + driver
# ---------------------------------------------------------------------------

FIGURES = {
    "kind-time-series": fig_kind_time_series,
    "kind-phase-2d": fig_kind_phase_2d,
    "kind-phase-3d": fig_kind_phase_3d,
    "kind-spacetime": fig_kind_spacetime,
    "kind-delay": fig_kind_delay,
    "themes": fig_themes,
    "styling": fig_styling,
    "compose-overlay": fig_compose_overlay,
    "compose-grid": fig_compose_grid,
    "spatial-field": fig_spatial_field,
}

#: Animated outputs (written as GIF rather than SVG).
GIF_FIGURES = {
    "animation-lorenz-reveal": fig_animation,
}


def main():
    """Render every Visualization figure to docs/assets/figures/viz/<slug>.{svg,gif}."""
    plt = _prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    for slug, fn in FIGURES.items():
        plt.close("all")
        out = OUT / f"{slug}.svg"
        fn(plt, str(out))
        print(f"  ok {slug:26} {out.stat().st_size:>8} bytes  (svg)")
    for slug, fn in GIF_FIGURES.items():
        plt.close("all")
        out = OUT / f"{slug}.gif"
        fn(plt, out)
        print(f"  ok {slug:26} {out.stat().st_size:>8} bytes  (gif)")


if __name__ == "__main__":
    main()
