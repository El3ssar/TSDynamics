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
import subprocess

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets" / "figures" / "viz"

#: The self-contained three.js demo payload the Backends page embeds live.
THREEJS_DEMO = (
    pathlib.Path(__file__).resolve().parents[1] / "assets" / "threejs-demo" / "lorenz-threejs.json"
)

#: Brand palette (docs/assets/brand/tokens.css) — teal primary, indigo accent,
#: amber / rose warm secondaries, bright teal for a fifth series.
TEAL, INDIGO, AMBER, ROSE, TEAL2 = "#11857A", "#574FCF", "#E8912D", "#D64562", "#2CC5AE"

#: Label font — IBM Plex Sans, with DejaVu as the layout fallback. Emitted
#: verbatim into the SVG so the page's IBM Plex webfont renders it in-browser.
_FONT_FAMILY = "IBM Plex Sans"

#: The brand **dark stage** every animation is rendered on (matches the three.js
#: viewer's ``#0B0F14`` background and the hero).  A GIF cannot carry a transparent
#: background cleanly — the pillow writer flattens transparency to a jarring green in
#: most viewers — so an animation is saved OPAQUE on this stage rather than
#: transparent, giving the same "attractor floating in a dark room" look as the live
#: WebGL viewers.
_STAGE = "#0B0F14"


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


def _optimise_gif(path):
    """Shrink a GIF in place with an ffmpeg palette pass (falls back to a no-op).

    The matplotlib ``pillow`` writer emits a full-colour GIF; a two-stage ffmpeg
    ``palettegen`` / ``paletteuse`` re-encodes it against a per-clip 128-colour
    palette with Bayer dithering, which roughly halves the file size at the same
    pixel dimensions while keeping the dark stage clean (no visible banding on the
    teal trail).  ffmpeg is optional — if it is absent (or errors) the original,
    correct GIF is kept untouched.
    """
    path = pathlib.Path(path)
    palette = path.with_suffix(".palette.png")
    tmp = path.with_suffix(".opt.gif")
    try:
        # Stage 1: derive a 128-colour palette from the whole clip.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                "palettegen=max_colors=128:stats_mode=full",
                str(palette),
            ],
            check=True,
        )
        # Stage 2: re-encode the GIF against that palette with Bayer dithering.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(path),
                "-i",
                str(palette),
                "-lavfi",
                "paletteuse=dither=bayer:bayer_scale=3",
                str(tmp),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        palette.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
        return  # ffmpeg unavailable — keep the pillow GIF as-is
    palette.unlink(missing_ok=True)
    # Only adopt the re-encode if it actually shrank the file — for a smooth teal
    # gradient the pillow GIF can already be near-optimal, and a dither pass can add
    # bytes; never regress the committed size.
    if tmp.exists() and 0 < tmp.stat().st_size < path.stat().st_size:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)


def _save_anim(spec, out_path, *, dpi, fps=None):
    """Render an animated ``PlotSpec`` and write an OPAQUE, dark-stage looping GIF.

    Goes through the real ``ts.viz`` matplotlib backend (``render_spec`` returns a
    ``FuncAnimation``), then writes it with the pillow writer and an explicit
    ``facecolor=_STAGE`` / ``transparent=False`` so the frames land on the brand dark
    stage.  This is the fix for the "green background" defect: with ``axes=False`` the
    3-D axes patch is turned *off* (transparent), and a transparent GIF flattens to a
    jarring green in most viewers — forcing the figure facecolor and disabling
    transparency keeps the clean dark room.  Finally an ffmpeg palette pass shrinks the
    file when it can.
    """
    from tsdynamics.viz.render import render_spec

    # Ensure the stage colour is on the spec's theme too (fig + axes facecolor), so the
    # 3-D background panes (when present) match; the savefig facecolor is the backstop.
    spec.background(_STAGE)
    fa = render_spec(spec, "matplotlib")
    save_kw = {
        "writer": "pillow",
        "dpi": float(dpi),
        "savefig_kwargs": {"facecolor": _STAGE, "transparent": False},
    }
    if fps is not None:
        save_kw["fps"] = float(fps)
    fa.save(str(out_path), **save_kw)
    import matplotlib.pyplot as plt

    plt.close("all")
    _optimise_gif(out_path)


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
    """Write a looping reveal-comet GIF of the Lorenz attractor drawing itself in.

    Rendered on the brand dark stage at a larger figure size (6.4x6.0 in) and higher
    dpi (85) / fps (25) than the original low-res GIF — so the amber head and fading
    indigo tail read crisply while the file stays a couple hundred KB.
    """
    import tsdynamics as ts

    lor = ts.systems.Lorenz()
    traj = lor.integrate(final_time=45.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(3.0)

    # Animation is an orthogonal modifier: any spec + an Animation becomes a movie.
    # A reveal comet — head at the current sample, a fading tail reaching back 6
    # time units — over the 3-D butterfly, axes hidden for a clean look.
    spec = (
        traj.to_plot_spec(components=["x", "y", "z"], animate=True)
        .animate(n_frames=100, fps=25)
        .trail(("time", 6.0), fade=True)
        .head(size=9.0, color=AMBER)
        .style(lw=0.8, axes=False)
        .recolor(INDIGO)
        .camera(elev=22, azim=-60)
        .size(6.4, 6.0)
    )
    spec.relabel(title="")
    _save_anim(spec, out_path, dpi=85)


def fig_animation_spin(plt, out_path):
    """Render a rotating 3-D Aizawa attractor reveal — the camera spins one full turn.

    Camera *spin* is the matplotlib-only animation knob: the azimuth sweeps ``spin``
    full turns over the whole loop while the comet reveals the orbit, so the whole
    attractor turns in space as it draws itself in.  A persistent trail (``.trail(None)``)
    means nothing ever erases — the elegant "watch the whole thing accrete" hero shot
    — inked in the brand teal with an indigo head, axes hidden.
    """
    import tsdynamics as ts

    # Aizawa declares no named variables, so its components are selected by index.
    aiz = ts.systems.Aizawa()
    traj = aiz.integrate(final_time=95.0, dt=0.01, ic=[0.1, 0.0, 0.0]).after(15.0)

    spec = (
        traj.to_plot_spec(components=[0, 1, 2], animate=True)
        .animate(n_frames=100, fps=25)
        .trail(None)  # persistent — the orbit accretes and stays
        .head(size=9.0, color=INDIGO)
        .camera(elev=18, azim=-70, spin=1.0)  # one full revolution over the loop
        .style(lw=0.7, axes=False)
        .recolor(TEAL)
        .size(6.0, 6.0)
    )
    spec.relabel(title="")
    _save_anim(spec, out_path, dpi=74)


def fig_animation_field(plt, out_path):
    """Render a Gray–Scott 2-D pattern-formation movie — the reaction–diffusion field over time.

    ``kind="field"`` on a spatially-extended system forces the ``frames`` animation
    model: each frame is the activator field's spatial state at that instant, so the
    self-replicating spots genuinely *grow and divide* across the movie (not a comet).
    A striking, honest visual of a PDE forming a Turing pattern — viridis on the dark
    activator field.
    """
    import tsdynamics as ts

    # A 48x48 reaction–diffusion field. GrayScott has a deterministic seeded IC, so
    # the pattern is reproducible; dt=5.0 samples the slow pattern formation.
    gs = ts.systems.GrayScott()
    gtr = gs.integrate(final_time=4000.0, dt=85.0)

    # kind="field" + animate=True → SPATIAL_FIELD, mode="frames": the activator field
    # replayed frame by frame as an imshow heatmap movie.
    spec = (
        gtr.to_plot_spec(kind="field", animate=True)
        .animate(fps=15)
        .style(cmap="viridis")
        .size(4.6, 4.6)
    )
    spec.relabel(title="")
    _save_anim(spec, out_path, dpi=78, fps=14)


def fig_animation_delay(plt, out_path):
    """Render a Mackey–Glass delay-embedding reveal — the DDE attractor reconstructed by a comet.

    The natural view of a scalar delay system is the delay embedding ``x(t)`` vs
    ``x(t-τ)``; animated, a teal comet with an indigo head sweeps it, so the folded
    chaotic band of the infinite-dimensional history draws itself in from a 1-D signal.
    """
    import numpy as np

    import tsdynamics as ts

    mg = ts.systems.MackeyGlass()
    traj = mg.integrate(
        final_time=900.0,
        dt=0.5,
        history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)],
    ).after(150.0)

    spec = (
        traj.to_plot_spec(kind="delay", components="x", tau=17.0, animate=True)
        .animate(n_frames=100, fps=25)
        .trail(("time", 120.0), fade=True)
        .head(size=8.0, color=INDIGO)
        .style(lw=0.6, axes=False)
        .recolor(TEAL)
        .size(5.6, 5.6)
    )
    spec.relabel(title="")
    _save_anim(spec, out_path, dpi=88)


def fig_animation_composite(plt, out_path):
    """Render a two-panel lockstep composite movie — a 3-D portrait beside its own time series.

    A COMPOSITE animation plays every panel on ONE master clock, so the state head on
    the Lorenz butterfly (left) and the sweep on the x(t) trace (right) advance
    together — the same instant shown two ways.  Dogfoods ``ts.viz.plot(..., animate=True)``.
    """
    import tsdynamics as ts
    from tsdynamics.viz import get_theme, plot
    from tsdynamics.viz.producers import time_series
    from tsdynamics.viz.spec import Animation

    lor = ts.systems.Lorenz()
    traj = lor.integrate(final_time=42.0, dt=0.01, ic=[1.0, 1.0, 1.0]).after(3.0)

    # LEFT: the 3-D butterfly (a reveal comet, indigo, axes hidden).
    portrait = (
        traj.to_plot_spec(components=["x", "y", "z"])
        .style(lw=0.7, axes=False)
        .recolor(INDIGO)
        .camera(elev=22, azim=-60)
    )
    portrait.relabel(title="")
    # RIGHT: the x(t) trace (a growing line, its sweep head on the master clock).  Its
    # axes stay visible, so give it a light-on-dark foreground that reads on the stage.
    trace = time_series(traj, components=["x"]).style(lw=1.0).recolor(TEAL)
    trace.relabel(title="x(t)", x="time", y="x")
    trace.theme(get_theme("dark"))  # light-on-dark axes that read on the stage

    # Pass a fully-built master Animation at compose time so every panel inherits the
    # whole timeline — frame count AND the fading 6-time-unit trail — with each panel's
    # per-kind head default (a head on the 3-D portrait, none on the plain time series).
    # A later chained .animate()/.trail() would set only the composite's own clock and
    # leave the per-panel drivers on their heavy default frame count.
    master = Animation(n_frames=100, fps=25, trail_kind="time", trail_length=6.0, trail_fade=True)
    comp = plot(portrait, trace, layout="row", animate=master).size(8.0, 4.1)
    _save_anim(comp, out_path, dpi=80)


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
    "animation-aizawa-spin": fig_animation_spin,
    "animation-grayscott-field": fig_animation_field,
    "animation-mackeyglass-delay": fig_animation_delay,
    "animation-composite": fig_animation_composite,
}


def make_threejs_demo():
    """Regenerate the live three.js demo payload the Backends page embeds.

    The Backends-page "Live demo" fetches ``docs/assets/threejs-demo/lorenz-threejs.json``
    and renders it with the reference loader.  Rather than a hand-committed payload,
    build it through the **exact same pipeline the 3-D catalogue pages use** — the
    ``threejs_viewer`` generator's arc-length-resampled, brand-teal, indigo-headed,
    *animated* reveal-comet payload — so the demo is byte-for-byte the viewer readers
    meet on every attractor page (schema 2, ``metadata.animation`` + ``metadata.theme``).
    Reproducible: a fixed IC via the viewer's pinned seed path.
    """
    import json
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import threejs_viewer  # docs/_tooling sibling

    from tsdynamics import registry

    entry = next(e for e in registry.all_systems() if e.name == "Lorenz")
    payload = threejs_viewer._build_payload(entry, second=False)
    if payload is None:  # pragma: no cover - the engine is required to build docs
        raise RuntimeError("threejs demo payload build returned None (engine unavailable?)")
    THREEJS_DEMO.parent.mkdir(parents=True, exist_ok=True)
    THREEJS_DEMO.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return THREEJS_DEMO


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
    demo = make_threejs_demo()
    print(f"  ok {'threejs-demo':26} {demo.stat().st_size:>8} bytes  (json)")


if __name__ == "__main__":
    main()
