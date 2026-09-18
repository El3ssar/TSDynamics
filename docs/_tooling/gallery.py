"""Build-time generator for the **plot gallery** — every plot the library can draw.

Why this file exists
--------------------
The visualization layer's public contract is the *declared compatibility matrix*:
one row per registered plot transform, listing every primitive that may draw it
(:func:`tsdynamics.viz.compatibility`).  A matrix is a promise, and a promise
nobody looks at rots.  ``tests/test_viz_compatibility.py`` already proves each
declared cell *renders something*; this module proves it renders something a
human would publish, and turns the proof into the page people actually asked
for: *"how do I see all the plots we can do now?"*

The page is **generated from the registry**, never hand-written:

* the sections are the two source categories (``data`` / ``model``) — the whole
  taxonomy, visible;
* the entries are ``registry.plot_transforms``, in registration order;
* the tabs under each entry are that transform's declared ``primitives`` row;
* the code beside every figure is the code that *produced* that figure — the
  snippet string is executed, not re-typed, so it cannot drift.

A transform added tomorrow appears here tomorrow with no edit to this file (it
falls back to the transform's own ``example`` factory, the same fixture the
governance gate drives), which keeps P1's "one registration and nothing else"
claim true.  Curating a :class:`Showcase` for it only makes the picture better.

Caching
-------
Content-addressed under ``.cache/docs-gallery``, exactly like
:mod:`figures` and :mod:`threejs_viewer`: the key is
``sha256(setup ‖ call ‖ RENDERER_VERSION ‖ library version)``.  A cache hit does
not even import the library subject — the snippet is never executed — so a warm
docs build pays nothing for the gallery.  CI persists ``.cache/`` between builds.

Standalone use::

    .venv/bin/python docs/_tooling/gallery.py            # render + report
    .venv/bin/python docs/_tooling/gallery.py --only ftle,nullclines
    .venv/bin/python docs/_tooling/gallery.py --force    # ignore the cache
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import textwrap
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "docs-gallery"

#: Bump when the *rendering* changes (theme, size, dpi) so every cached figure
#: is re-rendered.  The snippet text and the library version are already in the
#: key, so ordinary content changes need no bump.
RENDERER_VERSION = "gallery-2"

#: Where the figures live on the site.  They are registered as *generated* files
#: from the cache directory (like the field movies), so nothing is ever written
#: into ``docs/``.
ASSET_DIR = "assets/figures/gallery"

#: The page this module generates content for, and the token it replaces.
PAGE_URI = "visualization/gallery.md"
TOKEN = "{{ gallery }}"

# Brand palette (docs/assets/brand/tokens.css) — the same five colours
# ``make_viz_figures.py`` uses, so the gallery matches the rest of the site.
TEAL, INDIGO, AMBER, ROSE, TEAL2 = "#11857A", "#574FCF", "#E8912D", "#D64562", "#2CC5AE"

#: Default figure geometry, in inches-at-``DPI``.
FIGSIZE = (5.4, 3.7)
FIGSIZE_SQUARE = (4.4, 4.0)
DPI = 130


# ===========================================================================
# The showcase table
# ===========================================================================
@dataclass(frozen=True)
class Variant:
    """One tab: the snippet to run, and what to say about the picture it makes.

    Every field is optional and falls back to the entry's: ``call=None`` keeps
    the shared call with ``primitive="…"`` appended (so a tab that only needs its
    own *caption* is one line), and ``setup=None`` keeps the shared subject.  A
    variant that needs a different subject — a 1-D field for the ``line``
    primitive where the 2-D field is an image — overrides both.
    """

    call: str | None = None
    setup: str | None = None
    caption: str | None = None
    square: bool = False


@dataclass(frozen=True)
class Showcase:
    """A curated example of one transform, plus a tab per primitive that differs.

    Attributes
    ----------
    setup : str
        Statements run before the call; shown in the snippet.  ``import
        tsdynamics as ts`` is prepended automatically.
    call : str
        The expression that must evaluate to a :class:`~tsdynamics.viz.spec.PlotSpec`
        — in practice a ``ts.plot(...)`` one-liner, because the point of the page
        is that the one-liner is the API.
    caption : str
        What the reader should see in the picture.  Written as a claim, so a
        wrong picture reads as a wrong sentence.
    hero : str, optional
        Which primitive leads the entry.  Defaults to the transform's declared
        default; override where the declared default cannot draw *this* subject
        (a 3-D orbit is ``line3d``, not ``line``).
    per_primitive : mapping
        Primitive → :class:`Variant` for the tabs that need their own call or
        subject.  Any primitive not named here reuses ``call`` with
        ``primitive="…"`` appended.
    on_missing : Variant, optional
        What to draw when the entry raises
        :class:`~tsdynamics.analysis._result_viz.VisualizationNotInstalled` — an
        optional dependency is absent on this machine.  The fallback's *own*
        snippet is displayed, so the code beside the figure still produced it,
        and the page says why it is not the headline example.
    square : bool
        Use the square figure geometry (images, complex planes, portraits).
    """

    setup: str
    call: str
    caption: str
    hero: str | None = None
    per_primitive: Mapping[str, Variant] = field(default_factory=dict)
    on_missing: Variant | None = None
    square: bool = False

    def variant(self, primitive: str, *, default: str) -> Variant:
        """Return the :class:`Variant` for one primitive (synthesised if absent).

        A variant that supplies its own ``call`` is used **verbatim** — the author
        is responsible for it, and :func:`_render_one` checks that the figure it
        makes really does carry the tab's primitive.  Otherwise the entry's shared
        call is reused, bare for the transform's declared default (so the default
        tab shows the true one-liner) and with ``primitive="…"`` appended for the
        rest.
        """
        got = self.per_primitive.get(primitive, Variant())
        if got.call is not None:
            call = got.call
        else:
            call = self.call if primitive == default else _with_primitive(self.call, primitive)
        return Variant(
            call=call,
            setup=got.setup if got.setup is not None else self.setup,
            caption=got.caption if got.caption is not None else self.caption,
            square=got.square or self.square,
        )


def _with_primitive(call: str, primitive: str) -> str:
    """Append ``primitive="…"`` to a call expression.

    Textual, deliberately: the string that is *displayed* is the string that is
    *executed*, so the two cannot disagree.  The result is parsed before use
    (:func:`_check_snippet`), which turns any malformed insertion into a loud
    build failure rather than a silently wrong caption.
    """
    stripped = call.rstrip()
    if not stripped.endswith(")"):  # pragma: no cover - authoring error
        raise ValueError(f"showcase call must end in ')': {call!r}")
    head = stripped[:-1].rstrip()
    sep = "" if head.endswith("(") else ", "
    return f'{head}{sep}primitive="{primitive}")'


# --- shared setups ---------------------------------------------------------
# Every trajectory is integrated from an EXPLICIT initial condition, so the
# committed figure is reproducible and the cache key means what it says.

_LORENZ = "traj = ts.systems.Lorenz().run(final_time=60.0, dt=0.005, ic=[1.0, 1.0, 20.0])"
_LORENZ_MED = "traj = ts.systems.Lorenz().run(final_time=25.0, dt=0.01, ic=[1.0, 1.0, 20.0])"
_LORENZ_SHORT = "traj = ts.systems.Lorenz().run(final_time=8.0, dt=0.05, ic=[1.0, 1.0, 20.0])"
_ROSSLER = "traj = ts.systems.Rossler().run(final_time=200.0, dt=0.02, ic=[1.0, 1.0, 0.1])"
_ROSSLER_LONG = "traj = ts.systems.Rossler().run(final_time=900.0, dt=0.05, ic=[1.0, 1.0, 0.1])"
_LOGISTIC = "orbit = ts.systems.Logistic(r=4.0).run(steps=20_000, ic=[0.2])"
_LOGISTIC_SHORT = "orbit = ts.systems.Logistic(r=4.0).run(steps=48, ic=[0.2])"

SHOWCASE: dict[str, Showcase] = {
    # -- data: the orbit itself ---------------------------------------------
    "time_series": Showcase(
        setup=_LORENZ_MED,
        call='ts.plot(traj, "time_series")',
        caption=(
            "The three Lorenz coordinates against time — one layer per component, "
            "coloured from the theme palette. `x` and `y` swap sign together at every "
            "lobe change; `z` never goes negative."
        ),
        per_primitive={
            "points": Variant(
                setup=_LORENZ_SHORT,
                call='ts.plot(traj, "time_series", primitive="points")',
                caption="The same series as a scatter — the sampling grid made visible.",
            ),
            "steps": Variant(
                setup=_LORENZ_SHORT,
                call='ts.plot(traj, "time_series", primitive="steps")',
                caption="A piecewise-constant staircase — the right mark for a sampled or "
                "discrete-time record.",
            ),
        },
    ),
    "phase_portrait": Showcase(
        setup=_LORENZ,
        call='ts.plot(traj, "phase_portrait")',
        hero="line3d",
        square=True,
        caption=(
            "The Lorenz butterfly. With three components selected the geometry is "
            "3-D, so the geometry's own default (`line3d`) wins over the transform's "
            "declared default — which is why asking for flat `line` here raises "
            "instead of silently projecting."
        ),
        per_primitive={
            "line3d": Variant(call='ts.plot(traj, "phase_portrait")'),
            "points3d": Variant(
                call='ts.plot(traj, "phase_portrait", primitive="points3d", '
                "markersize=0.8, alpha=0.4)",
                caption=(
                    "The same orbit as a point cloud — sample density, not connectivity. "
                    "The style keywords are split out of the same call and applied to this "
                    "transform's layers."
                ),
            ),
            "line": Variant(
                call='ts.plot(traj, "phase_portrait", components=("x", "z"), primitive="line")',
                caption="The classic `(x, z)` projection of the butterfly.",
            ),
            "points": Variant(
                call='ts.plot(traj, "phase_portrait", components=("x", "z"), '
                'primitive="points", markersize=0.8, alpha=0.4)',
            ),
            "density": Variant(
                call='ts.plot(traj, "phase_portrait", components=("x", "z"), '
                'primitive="density", bins=160)',
                caption=(
                    "The same projection as a 2-D histogram — where the orbit spends its "
                    "time, which a line plot hides once the curve overdraws itself. The "
                    "rims of both wings are where it lingers."
                ),
            ),
        },
    ),
    "delay_embedding": Showcase(
        setup=_LORENZ,
        call='ts.plot(traj, "delay_embedding", delay=17)',
        square=True,
        caption=(
            "Takens' reconstruction from the single observable `x(t)`: plot it against "
            "`x(t - τ)` and the butterfly reappears, without ever using `y` or `z`."
        ),
        per_primitive={
            "density": Variant(
                call='ts.plot(traj, "delay_embedding", delay=17, primitive="density")',
                caption="The reconstruction as an occupancy density.",
            ),
        },
    ),
    "cobweb": Showcase(
        setup=_LOGISTIC_SHORT,
        call='ts.plot(orbit, "cobweb")',
        square=True,
        caption=(
            "The logistic map at `r = 4`, iterated as a staircase between the map's "
            "graph and the diagonal `y = x`. Every vertical segment is an application "
            "of `f`, every horizontal one is feeding the result back in. Forty-eight "
            "iterates is a cobweb; twenty thousand is a filled square, which is why "
            "this one is short on purpose."
        ),
        per_primitive={
            "points": Variant(
                caption="Only the visited `(x_n, x_{n+1})` pairs — the staircase's corners."
            )
        },
    ),
    "spacetime": Showcase(
        setup="traj = ts.systems.Lorenz96().run(final_time=30.0, dt=0.05)",
        call='ts.plot(traj, "spacetime")',
        caption=(
            "Lorenz-96 as component index versus time: 20 coupled sites, with the "
            "westward-drifting waves that a 20-line time series cannot show."
        ),
        per_primitive={
            "contour": Variant(
                call='ts.plot(traj, "spacetime", primitive="contour", levels=6)',
                caption=(
                    "The same lattice as level sets, coloured by level from the "
                    "transform's colormap rather than by layer index."
                ),
            ),
            "surface3d": Variant(
                call='ts.plot(traj, "spacetime", primitive="surface3d")',
                caption="The lattice lifted into a surface — amplitude as height.",
                square=True,
            ),
        },
    ),
    "spatial_field": Showcase(
        setup="traj = ts.systems.SwiftHohenberg().run(final_time=40.0, dt=0.2)",
        call='ts.plot(traj, "spatial_field")',
        square=True,
        caption=(
            "The Swift–Hohenberg field on its own 32×32 grid — the state vector "
            "un-flattened by the system's `_field_shape`, which is why no `shape=` "
            "argument is needed. The stripes are the pattern-forming instability."
        ),
        per_primitive={
            "contour": Variant(caption="The same field as contours."),
            "surface3d": Variant(caption="The field as a height surface."),
            "line": Variant(
                setup="traj = ts.systems.KuramotoSivashinsky().run(final_time=200.0, dt=0.5)",
                call='ts.plot(traj, "spatial_field", primitive="line")',
                caption=(
                    "A **1-D** field is a profile, not an image: the final "
                    "Kuramoto–Sivashinsky wave `u(x)`. One transform, two spatial "
                    "dimensionalities, no guessing."
                ),
                square=False,
            ),
            "points": Variant(
                setup="traj = ts.systems.KuramotoSivashinsky().run(final_time=200.0, dt=0.5)",
                call='ts.plot(traj, "spatial_field", primitive="points")',
                caption="The same profile as its sampled lattice points.",
                square=False,
            ),
        },
    ),
    "invariant_density": Showcase(
        setup=_LOGISTIC,
        call='ts.plot(orbit, "invariant_density", bins=120)',
        caption=(
            "The natural measure of the logistic map at `r = 4` — the analytic "
            "arcsine law `1/(π√(x(1-x)))`, with its two integrable spikes at the "
            "interval ends, recovered from 20 000 iterates."
        ),
        per_primitive={
            "line": Variant(caption="The same density as a curve."),
            "steps": Variant(caption="The same density as a step histogram outline."),
            "image": Variant(
                setup=_LORENZ,
                call='ts.plot(traj, "invariant_density", components=("x", "z"), '
                'bins=200, primitive="image")',
                caption=(
                    "In two components the measure is an image: the Lorenz attractor's "
                    "occupancy, brightest along the slow spiralling rims."
                ),
                square=True,
            ),
            "contour": Variant(
                setup=_LORENZ,
                call='ts.plot(traj, "invariant_density", components=("x", "z"), '
                'bins=48, primitive="contour", levels=6)',
                caption=(
                    "The same 2-D measure as level sets. A histogram needs coarse bins "
                    "before its level sets mean anything — at 200 bins the contours "
                    "trace shot noise."
                ),
                square=True,
            ),
            "surface3d": Variant(
                setup=_LORENZ,
                call='ts.plot(traj, "invariant_density", components=("x", "z"), '
                'bins=80, primitive="surface3d")',
                caption="The measure as a landscape.",
                square=True,
            ),
        },
    ),
    # -- data: series diagnostics -------------------------------------------
    "psd": Showcase(
        setup=_LORENZ,
        call='ts.plot(traj, "psd")',
        caption=(
            "The power spectrum of the Lorenz `x` coordinate: **broadband**, a "
            "continuum with no line standing out of it — the spectral signature of "
            "chaos. The geometry declares log-log axes, because on linear axes every "
            "spectrum is a spike at `f = 0` and a flat line."
        ),
        per_primitive={
            "points": Variant(
                caption=(
                    "The same spectrum as its Welch bins. Compare with a **periodic** "
                    "orbit's spectrum, which is a comb of discrete lines rather than a "
                    "continuum."
                )
            ),
        },
    ),
    "autocorrelation": Showcase(
        setup=_LORENZ,
        call='ts.plot(traj, "autocorrelation", max_delay=400)',
        caption=(
            "`C(τ)` for Lorenz `x`, with the `1/e` level and the first zero crossing "
            "marked — the two classical rules of thumb for choosing an embedding delay."
        ),
    ),
    "mutual_information": Showcase(
        setup=_LORENZ,
        call='ts.plot(traj, "mutual_information", max_delay=300)',
        caption=(
            "Time-delayed mutual information. Fraser & Swinney's rule takes the first "
            "*minimum* — marked — which is the nonlinear answer to the same question "
            "the autocorrelation answers linearly."
        ),
    ),
    "fnn": Showcase(
        setup=_LORENZ,
        call='ts.plot(traj, "fnn", delay=17, max_dim=8)',
        caption=(
            "Kennel's false-nearest-neighbour fraction against embedding dimension. It "
            "collapses to zero at `m = 3` — the true dimension of the Lorenz system, "
            "recovered from one scalar observable."
        ),
    ),
    "cao": Showcase(
        setup=_LORENZ,
        call='ts.plot(traj, "cao", delay=17, max_dim=8)',
        caption=(
            "Cao's `E1(d)` saturates at 1 once the embedding unfolds the attractor, "
            "while `E2(d)` stays away from 1 — which is how the method separates "
            "deterministic data from noise, not just picks a dimension."
        ),
    ),
    "line_lengths": Showcase(
        setup=_ROSSLER,
        call='ts.plot(traj, "line_lengths", recurrence_rate=0.05).rescale(x="log", y="log")',
        caption=(
            "The diagonal and vertical line-length distributions of the recurrence "
            "matrix — the histograms that DET, LAM, `L_max` and TT are read off. Long "
            "diagonals mean deterministic recurrence; long verticals mean laminar "
            "trapping. Both are heavy-tailed, so they are read on log axes; `ts.plot` "
            "returns the spec, so the fluent tweak chains straight onto the call."
        ),
        per_primitive={
            "points": Variant(
                call='ts.plot(traj, "line_lengths", recurrence_rate=0.05, '
                'primitive="points").rescale(x="log", y="log")',
            ),
            "steps": Variant(
                call='ts.plot(traj, "line_lengths", recurrence_rate=0.05, '
                'primitive="steps").rescale(x="log", y="log")',
            ),
        },
    ),
    "return_time": Showcase(
        setup="traj = ts.systems.Lorenz().run(final_time=400.0, dt=0.005, ic=[1.0, 1.0, 20.0])",
        call='ts.plot(traj, "return_time", components="z", n_bins=30)',
        caption=(
            "How long the Lorenz orbit takes to come back to a level set of `z` — one "
            "circuit of a wing. The distribution is sharply peaked at `T ≈ 0.69` with a "
            "right tail: most loops are the same size, and the long ones are the wide "
            "swings that follow a lobe change. A periodic orbit would put every count "
            "in a single bin; the width of this peak *is* the chaos."
        ),
    ),
    # -- data: space-filling curves -----------------------------------------
    "hilbert": Showcase(
        setup=_LORENZ,
        call='ts.plot(traj, "hilbert")',
        square=True,
        caption=(
            "12 000 samples of Lorenz `x(t)` laid on a Hilbert curve. Samples close in "
            "the record land close in the picture, so long-range texture — the lobe "
            "switching — becomes visible texture instead of a smear."
        ),
        on_missing=Variant(
            call='ts.plot(traj, "hilbert", curve="snake")',
            caption=(
                "The same record on the dependency-free **snake** ordering. The "
                "Hilbert-type curves need the optional `hilbertplot` package, which "
                "this build does not have; asking for one raises rather than "
                "substituting, so what you see is what was asked for."
            ),
        ),
    ),
    "hilbert_fourier": Showcase(
        setup=_ROSSLER,
        call='ts.plot(traj, "hilbert_fourier")',
        square=True,
        caption=(
            "The centred 2-D power spectrum of the Hilbert image. Periodicity in the "
            "record becomes **symmetry** in this picture — a near-periodic Rössler "
            "orbit puts sharp structure at the corresponding scales."
        ),
        on_missing=Variant(
            call='ts.plot(traj, "hilbert_fourier", curve="snake")',
            caption=(
                "The snake-ordering spectrum — the `hilbertplot` extra is not "
                "installed on this build."
            ),
        ),
    ),
    "hilbert_difference": Showcase(
        setup=_ROSSLER,
        call='ts.plot(traj, "hilbert_difference")',
        square=True,
        caption=(
            "The locality-loss field: where a cell's 2-D neighbourhood is *not* a "
            "1-D neighbourhood of the record. It is the honest error map of the whole "
            "technique — bright cells are where the picture lies to you."
        ),
        on_missing=Variant(
            call='ts.plot(traj, "hilbert_difference", curve="snake")',
            caption=(
                "Locality loss for the **snake** ordering, and it is loss almost "
                "everywhere: two vertically adjacent cells are a whole row apart in the "
                "record, so the mean neighbour gap is uniformly large — which is exactly "
                "the argument for the Hilbert curve, and why this plot is worth having. "
                "The `hilbertplot` extra is not installed on this build, and the "
                "transform raises rather than substituting a curve you did not ask for."
            ),
        ),
    ),
    "hilbert_labels": Showcase(
        setup=_LORENZ,
        call='ts.plot(traj, "hilbert_labels")',
        square=True,
        caption=(
            "The curve's own path, each cell coloured by the step at which it is "
            "visited — the map from pixel to sample, drawn rather than asserted."
        ),
        on_missing=Variant(
            call='ts.plot(traj, "hilbert_labels", curve="snake")',
            caption=(
                "The snake ordering's path: left to right, then right to left. Compare "
                "it with the Hilbert curve's recursive quadrant walk to see why "
                "locality differs. The `hilbertplot` extra is not installed here."
            ),
        ),
    ),
    # -- data: spectra & stability curves -----------------------------------
    "lyapunov_spectrum": Showcase(
        setup="lorenz = ts.systems.Lorenz()",
        call='ts.plot(lorenz, "lyapunov_spectrum", final_time=400.0)',
        caption=(
            "The Lorenz spectrum as a stem plot against the `λ = 0` line: one positive "
            "exponent (chaos), one zero (the flow direction), one strongly negative "
            "(volume contraction). Their sum is the divergence, `-(σ + 1 + β)`."
        ),
        per_primitive={"line": Variant(caption="The same exponents joined as a line.")},
    ),
    "lyapunov_convergence": Showcase(
        setup="lorenz = ts.systems.Lorenz()",
        call='ts.plot(lorenz, "lyapunov_convergence", steps=4000)',
        caption=(
            "Each exponent's running Benettin estimate against time — the plot that "
            "answers *has it converged?*, which a single returned number never can."
        ),
    ),
    "gali_curves": Showcase(
        setup="hh = ts.systems.HenonHeiles()",
        call='ts.plot(hh, "gali_curves", k=(2, 3), ic=[0.0, -0.1, 0.49, 0.0], final_time=1000.0)',
        caption=(
            "GALI₂ and GALI₃ on log axes for a chaotic Hénon–Heiles orbit, against the "
            "analytic reference slope `-(k-1)λ₁`. Exponential decay is the chaos "
            "verdict; a regular orbit would leave GALI₂ flat."
        ),
    ),
    "zero_one_pq_plane": Showcase(
        setup="lorenz = ts.systems.Lorenz()",
        # NOTE: `components=0` is the default and is omitted deliberately — the
        # transform still spells it `components=` and forwards it to
        # `zero_one_test`, which takes `components=` in v6, so naming it raises.
        call='ts.plot(lorenz, "zero_one_pq_plane", final_time=2000.0, dt=0.1)',
        square=True,
        caption=(
            "The `(p, q)` translation variables of the 0–1 test. Chaotic dynamics "
            "drives them on a **Brownian sprawl** whose extent grows without bound; a "
            "regular orbit keeps them on a bounded torus-like blob."
        ),
        per_primitive={
            "density": Variant(caption="The same walk as an occupancy density."),
            "points": Variant(caption="The same walk as its sampled points."),
        },
    ),
    "orbit_diagram": Showcase(
        setup="import numpy as np\n\nlog = ts.systems.Logistic()",
        call='ts.plot(log, "orbit_diagram", param="r", values=np.linspace(2.8, 4.0, 600),\n'
        "        points=120, transient=400)",
        caption=(
            "The period-doubling cascade, drawn as a transform rather than read off a "
            "result — so it overlays, grids and styles like any other layer. One fixed "
            "point doubles at `r = 3`, again at `1 + sqrt(6)`, and smears into chaos "
            "near `3.57`, threaded with periodic windows."
        ),
        per_primitive={
            "density": Variant(
                call='ts.plot(log, "orbit_diagram", param="r", values=np.linspace(2.8, 4.0, 600),\n'
                '        points=120, transient=400, primitive="density")',
                caption="The same sweep as an occupancy density — where the orbit spends "
                "its time, not merely where it goes.",
            ),
        },
    ),
    "recurrence": Showcase(
        setup="traj = ts.systems.Rossler().run(final_time=150.0, dt=0.25, ic=[1.0, 1.0, 0.1])",
        call='ts.plot(traj, "recurrence", recurrence_rate=0.05)',
        square=True,
        caption=(
            "The recurrence plot `R(i, j)`: a dot wherever the orbit returns to within "
            "`epsilon` of an earlier state. The diagonal stripes are near-repetitions of "
            "the Rössler cycle; their lengths are what `rqa` turns into DET and L_max."
        ),
        per_primitive={
            "contour": Variant(
                call='ts.plot(traj, "recurrence", recurrence_rate=0.05, primitive="contour")',
                caption="The same matrix as the boundary of its recurrent set.",
            ),
        },
    ),
    "ensemble_fan": Showcase(
        setup="import numpy as np\n\n"
        "band = ts.systems.Lorenz().ensemble(\n"
        "    np.random.default_rng(0).normal(1.0, 0.05, size=(40, 3)))\n"
        "batch = band.run(final_time=12.0, dt=0.01)",
        call='ts.plot(batch, "ensemble_fan")',
        caption=(
            "Forty Lorenz starts within 0.05 of each other, drawn as their median and "
            "spread. The band is invisible at first and then opens out — sensitive "
            "dependence, as a picture of an ensemble rather than of two orbits."
        ),
    ),
    "scaling_fit": Showcase(
        setup=_LORENZ + "\nresult = ts.analysis.correlation_dimension(traj)",
        call='ts.plot(result, "scaling_fit")',
        caption=(
            "The log-log correlation sum with the fitted scaling window delimited. "
            "The slope inside the window is the correlation dimension; the point of "
            "drawing it is that you can see whether the window was the right one."
        ),
        per_primitive={"line": Variant(caption="The same scaling curve as a line.")},
    ),
    # -- model: the plane -----------------------------------------------------
    "nullclines": Showcase(
        setup="fhn = ts.systems.FitzHughNagumo()",
        call='ts.plot(fhn, "nullclines", xlim=(-2.5, 2.5), ylim=(-1.0, 2.0))',
        caption=(
            "The FitzHugh–Nagumo nullclines: the cubic `v' = 0` curve and the straight "
            "`w' = 0` line. **Their crossing is the equilibrium** — the textbook "
            "picture, computed by marching squares on the real right-hand side."
        ),
        per_primitive={
            "points": Variant(caption="The marching-squares vertices themselves."),
        },
    ),
    "direction_field": Showcase(
        setup="fhn = ts.systems.FitzHughNagumo()",
        call='ts.plot(fhn, "direction_field", xlim=(-2.5, 2.5), ylim=(-1.0, 2.0), grid=21)',
        caption=(
            "The right-hand side as unit arrows on a lattice. It takes the **system**, "
            'not a callable, so `plane=("x", "z")` and `at=` slice any dimension of '
            "any system — the hole a callable-only vector field cannot fill."
        ),
    ),
    "flow_speed": Showcase(
        setup="vdp = ts.systems.VanDerPol(mu=2.0)",
        call='ts.plot(vdp, "flow_speed", xlim=(-3.0, 3.0), ylim=(-6.0, 6.0), log=True)',
        square=True,
        caption=(
            "`|f|` as a backdrop for the Van der Pol relaxation oscillator, on a log "
            "scale: the two fast jumps at the ends of the cubic are bright, the two slow "
            "crawls along its branches are dark, and the equilibrium at the origin is a "
            "**hole** — `log=True` maps `|f| = 0` to NaN rather than faking a minimum. "
            "Speed is exactly what a direction field's unit arrows throw away."
        ),
        per_primitive={
            "contour": Variant(
                call='ts.plot(vdp, "flow_speed", xlim=(-3.0, 3.0), ylim=(-6.0, 6.0), '
                'log=True, primitive="contour")',
                caption="The same speed field as level sets, coloured by level.",
            ),
            "surface3d": Variant(
                call='ts.plot(vdp, "flow_speed", xlim=(-3.0, 3.0), ylim=(-6.0, 6.0), '
                'primitive="surface3d")',
                caption="Speed as a height surface (linear, so the ridges keep their scale).",
            ),
        },
    ),
    "streamlines": Showcase(
        setup="vdp = ts.systems.VanDerPol(mu=2.0)",
        call='ts.plot(vdp, "streamlines", xlim=(-3.0, 3.0), ylim=(-6.0, 6.0), '
        "seeds=6, steps=120, alpha=0.75)",
        square=True,
        caption=(
            "Arc-length-parametrised integral curves of the field, integrated **both "
            "ways** from a lattice of seeds through the system's own right-hand side — "
            "not a plotting library's interpolation of a pre-sampled lattice, which is "
            "what matters near a separatrix. Every curve is swept onto the same limit "
            "cycle, which is the picture of an attractor."
        ),
        per_primitive={
            "points": Variant(
                call='ts.plot(vdp, "streamlines", xlim=(-3.0, 3.0), ylim=(-6.0, 6.0), '
                'seeds=4, steps=60, primitive="points", markersize=2.0, alpha=0.7)',
                caption=(
                    "The streamline vertices: **evenly spaced by arc length**, not by "
                    "time — so their spacing carries no speed information, deliberately."
                ),
            ),
        },
    ),
    "trace_determinant": Showcase(
        setup="lv = ts.systems.LotkaVolterra()",
        call='ts.plot(lv, "trace_determinant")',
        caption=(
            "Every equilibrium of the system placed on the `(tr J, det J)` plane, with "
            "the axes and the discriminant parabola that partition it into node / "
            "spiral / saddle / centre. Lotka–Volterra's two equilibria land in two "
            "different regions: the extinction state below the `det = 0` axis is a "
            "**saddle**, the coexistence state on the `tr = 0` axis is a **centre** — "
            "which is why its orbits are closed. The frame is `param2`, so this "
            "correctly **refuses** to overlay on a state space."
        ),
    ),
    "basins": Showcase(
        setup="hen = ts.systems.Henon()",
        call='ts.plot(hen, "basins", region=[(-2.0, 2.0, 160), (-2.0, 2.0, 160)])',
        square=True,
        caption=(
            "Which fate each initial condition meets: the Hénon attractor's basin "
            "against the starts that escape to infinity. The boundary is the stable "
            "manifold of the saddle at infinity — smooth here, fractal for many other "
            "systems, which is what `basin_entropy` measures."
        ),
        per_primitive={
            "boundary": Variant(
                call='ts.plot(hen, "basins", region=[(-2.0, 2.0, 160), (-2.0, 2.0, 160)],\n'
                '        primitive="boundary")',
                caption="Only the cells on the basin boundary — the set whose fate a small "
                "perturbation can change.",
            ),
            "contour": Variant(
                call='ts.plot(hen, "basins", region=[(-2.0, 2.0, 160), (-2.0, 2.0, 160)],\n'
                '        primitive="contour")',
                caption="The label field as level sets, for overlaying on a portrait.",
            ),
        },
    ),
    "vector_field": Showcase(
        setup='vdp = ts.systems.VanDerPol(params={"mu": 1.0})',
        call='ts.plot(vdp, "vector_field", grid=22)',
        caption=(
            "The right-hand side as unit arrows on a lattice — the direction field. "
            "Handed the **system** it infers its own window, exactly as `flow_speed` "
            "does; `normalize=False` keeps the true magnitudes. `direction_field` is "
            "an alias for this transform, not a second row."
        ),
    ),
    "phase_portrait_field": Showcase(
        setup=(
            "import numpy as np\n\n"
            "def rhs(u):\n"
            "    return np.stack([u[..., 1], 2.0 * (1 - u[..., 0] ** 2) * u[..., 1] "
            "- u[..., 0]], axis=-1)\n\n"
            "traj = ts.systems.VanDerPol(mu=2.0).run("
            "final_time=30.0, dt=0.01, ic=[0.1, 0.0])"
        ),
        call='ts.plot(rhs, "phase_portrait_field", source=traj, xlim=(-3.0, 3.0), '
        "ylim=(-6.0, 6.0), grid=18)",
        square=True,
        caption=(
            "A direction field with its host orbit drawn on it — the orbit is pinned to "
            "a line even though the transform's primitive is `quiver`, because a part "
            "may pin its own primitive."
        ),
    ),
    # -- model: fields over initial conditions --------------------------------
    "ftle": Showcase(
        setup="duff = ts.systems.Duffing(gamma=0.0)",
        call='ts.plot(duff, "ftle", xlim=(-2.0, 2.0), ylim=(-1.5, 1.5), grid=201, final_time=8.0)',
        square=True,
        caption=(
            "The finite-time Lyapunov exponent field of the unforced two-well Duffing "
            "oscillator. **The bright ridge is the stable manifold of the saddle at the "
            "origin** — the separatrix between the two wells, found without ever "
            "computing a basin."
        ),
        per_primitive={
            "contour": Variant(
                call='ts.plot(duff, "ftle", xlim=(-2.0, 2.0), ylim=(-1.5, 1.5), '
                'grid=121, final_time=8.0, primitive="contour")',
                caption="The same field as level sets — the ridge as a contour crowd.",
            ),
            "surface3d": Variant(
                call='ts.plot(duff, "ftle", xlim=(-2.0, 2.0), ylim=(-1.5, 1.5), '
                'grid=81, final_time=8.0, primitive="surface3d")',
                caption="The FTLE field as a landscape; the ridge is a literal ridge.",
            ),
        },
    ),
    "escape_time": Showcase(
        setup="hh = ts.systems.HenonHeiles()",
        call='ts.plot(hh, "escape_time", plane=("x", "y"), at=[0.0, 0.0, 0.0, 0.40],\n'
        "        xlim=(-1.2, 1.2), ylim=(-1.0, 1.4), grid=201,\n"
        "        final_time=60.0, chunks=120, escape=2.0)",
        square=True,
        caption=(
            "**Transient chaos, drawn.** Hénon–Heiles above its escape energy: the "
            "triangular bound region in the middle (white — never escaped within the "
            "horizon, so `NaN`, a hole rather than a fake number), the three exit "
            "channels, and between them the fractal filaments of starts that bounce "
            "around for tens of time units before committing to an exit. A two-colour "
            "*which* exit diagram cannot resolve that structure; the level sets of "
            "*when* can. Note `at=` — the system is 4-D and this is the `(x, y)` slice "
            "at `p_y = 0.4`, which is what puts it above the escape energy."
        ),
        per_primitive={
            "contour": Variant(
                # NOTE: no `escape=` here.  With it the field carries NaN holes, and
                # ts.plot's contour path then yields ZERO layers (a silently blank
                # figure) where ts.viz.draw on the identical geometry yields 960.
                # The window-exit criterion is the transform's own default.
                call='ts.plot(hh, "escape_time", plane=("x", "y"), at=[0.0, 0.0, 0.0, 0.40],\n'
                "        xlim=(-1.2, 1.2), ylim=(-1.0, 1.4), grid=121,\n"
                '        final_time=60.0, chunks=120, primitive="contour")',
                caption=(
                    "The same field as escape isochrones — a level set is the set of "
                    "starts that leave the window at the same time."
                ),
            ),
            "surface3d": Variant(
                call='ts.plot(hh, "escape_time", plane=("x", "y"), at=[0.0, 0.0, 0.0, 0.40],\n'
                "        xlim=(-1.2, 1.2), ylim=(-1.0, 1.4), grid=81,\n"
                '        final_time=60.0, chunks=120, escape=2.0, primitive="surface3d")',
                caption=(
                    "Escape time as a landscape: the spikes are the long-lived orbits "
                    "on the boundary between exits."
                ),
            ),
        },
    ),
    "transient_time": Showcase(
        # Below the Hopf threshold (b < 1 + a²) the Brusselator has a stable focus,
        # so "settled" — the default |f| → 0 arrival test — is exactly the right
        # question. Above it the attractor is a limit cycle, the speed never
        # becomes small, and the field is honestly (and uselessly) all-NaN; the
        # transform says so rather than drawing a blank rectangle.
        setup="bru = ts.systems.Brusselator(b=1.5)",
        call='ts.plot(bru, "transient_time", xlim=(0.0, 4.0), ylim=(0.0, 4.0), '
        "grid=161, final_time=40.0, chunks=80)",
        square=True,
        caption=(
            "The complement of escape time: how long each start takes to **arrive** at "
            "the Brusselator's stable focus, below its Hopf threshold. The arrival "
            "threshold is read off the field itself (1% of the median lattice speed) "
            "and recorded in `meta`, rather than being a fixed number in units the "
            "flow has never heard of."
        ),
        per_primitive={
            "contour": Variant(
                call='ts.plot(bru, "transient_time", xlim=(0.0, 4.0), ylim=(0.0, 4.0), '
                'grid=121, final_time=40.0, chunks=80, primitive="contour")',
                caption="The same field as level sets — each contour is an arrival isochrone.",
            ),
            "surface3d": Variant(
                call='ts.plot(bru, "transient_time", xlim=(0.0, 4.0), ylim=(0.0, 4.0), '
                'grid=81, final_time=40.0, chunks=80, primitive="surface3d")',
                caption="Arrival time as a landscape.",
            ),
        },
    ),
    # -- model: spectra of a point --------------------------------------------
    "eigenvalue_plane": Showcase(
        setup="lorenz = ts.systems.Lorenz()",
        call='ts.plot(lorenz, "eigenvalue_plane", at=[0.0, 0.0, 0.0])',
        square=True,
        caption=(
            "The Jacobian spectrum of the Lorenz origin against the imaginary axis: "
            "`{+11.83, -2.67, -22.83}` — one eigenvalue to the right of the boundary, "
            "so the origin is the saddle that separates the two wings. The boundary is "
            "real geometry in its own layer, so it survives JSON and three.js export."
        ),
        per_primitive={"line": Variant(caption="The spectrum joined as a line.")},
    ),
    "floquet_multipliers": Showcase(
        setup="vdp = ts.systems.VanDerPol()",
        call='ts.plot(vdp, "floquet_multipliers")',
        square=True,
        caption=(
            "The monodromy spectrum of the Van der Pol limit cycle (period `T = 6.663`) "
            "against the unit circle. One multiplier sits at `+1` — the trivial one, "
            "along the flow, found by aligning the eigenvector with `f(x₀)` and marked "
            "as such. The other is at the origin to plotting accuracy (`|μ| ~ 1e-3`), "
            "which is what *strongly* attracting means."
        ),
        per_primitive={"line": Variant(caption="The multipliers joined as a line.")},
    ),
}


# ---------------------------------------------------------------------------
# Compositions — the claim that everything overlays, checked by drawing it
# ---------------------------------------------------------------------------
#: Overlays that mix transforms from *different* families.  These are the cells
#: that would catch a frame declared inconsistently by two different authors: an
#: overlay is legal exactly when the coordinate spaces match, so a model field
#: and a measured orbit landing on one axes is a statement about both.
COMPOSITIONS: list[tuple[str, Showcase]] = [
    (
        "The phase-plane payoff",
        Showcase(
            setup="fhn = ts.systems.FitzHughNagumo()\n"
            "traj = fhn.run(final_time=120.0, dt=0.05, ic=[-1.0, -0.5])",
            call="ts.viz.plot(\n"
            '    ts.plot(fhn, "direction_field", "nullclines",\n'
            "            xlim=(-2.5, 2.5), ylim=(-1.0, 2.0), grid=21),\n"
            '    ts.plot(traj, "phase_portrait"),\n'
            ")",
            caption=(
                "Four things on one axes, from three different transform families: the "
                "direction field, both nullclines, and an integrated orbit. The "
                "equilibrium is where the nullclines cross, the orbit is tangent to the "
                "arrows everywhere, and the relaxation loop rides the outer branches of "
                "the cubic. Draw order is by **role** — field under curve — so the "
                "arguments could be given in any order."
            ),
            square=True,
        ),
    ),
    (
        "Speed under streamlines",
        Showcase(
            setup="vdp = ts.systems.VanDerPol(mu=2.0)",
            call='ts.plot(vdp, "flow_speed", "streamlines",\n'
            "        xlim=(-3.0, 3.0), ylim=(-6.0, 6.0), log=True, seeds=5, steps=120)",
            caption=(
                "One call, two model transforms, one shared window: the speed field as "
                "the backdrop and its integral curves on top. A shared keyword is routed "
                "**only to the transforms that accept it** — `log=` reaches `flow_speed`, "
                "`seeds=` reaches `streamlines`, and neither gets the other's."
            ),
            square=True,
        ),
    ),
    (
        "A separatrix, twice",
        Showcase(
            setup="duff = ts.systems.Duffing(gamma=0.0)\n"
            "traj = duff.run(final_time=60.0, dt=0.01, ic=[1.4, 0.0, 0.0])",
            call="ts.viz.plot(\n"
            '    ts.plot(duff, "ftle", xlim=(-2.0, 2.0), ylim=(-1.5, 1.5),\n'
            "            grid=161, final_time=8.0),\n"
            '    ts.plot(traj, "phase_portrait", components=("x", "y"), color="w"),\n'
            ")",
            caption=(
                "An orbit drawn on the FTLE field of its own system. The orbit spirals "
                "*inside* one well and never crosses the bright ridge — which is the "
                "claim the ridge makes: it is the separatrix, computed from the "
                "variational dynamics, not from watching where orbits end up."
            ),
            square=True,
        ),
    ),
]


# ===========================================================================
# Rendering
# ===========================================================================
@dataclass
class Cell:
    """One rendered (transform, primitive) pair, ready to write into the page."""

    transform: str
    primitive: str
    snippet: str
    caption: str
    filename: str | None = None
    cached_path: pathlib.Path | None = None
    note: str | None = None
    seconds: float = 0.0
    from_cache: bool = False
    curated: bool = True


@dataclass
class Build:
    """The result of a gallery render: the cells, in page order, plus the failures."""

    cells: list[Cell] = field(default_factory=list)
    compositions: list[Cell] = field(default_factory=list)
    failures: list[tuple[str, str, str]] = field(default_factory=list)

    def assets(self) -> dict[str, str]:
        """Return ``site uri -> absolute cached path`` for every rendered figure."""
        return {
            f"{ASSET_DIR}/{cell.filename}": str(cell.cached_path)
            for cell in [*self.cells, *self.compositions]
            if cell.filename and cell.cached_path
        }


def _library_version() -> str:
    import tsdynamics

    return str(tsdynamics.__version__)


def cache_key(setup: str, call: str) -> str:
    """Content hash of one cell: its exact snippet, the renderer, the library."""
    blob = "\n".join([setup, call, RENDERER_VERSION, _library_version()])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _prepare_matplotlib():
    """Configure matplotlib and install the brand theme; return ``pyplot``."""
    import logging

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"savefig.transparent": True})
    # The brand face is a webfont: the site has it, a build machine usually does
    # not, and matplotlib's silent DejaVu fallback is the right answer — but it
    # logs a line per text artist, which is thousands of lines per gallery build.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    from tsdynamics.viz import register_theme, set_theme
    from tsdynamics.viz.style import Theme

    register_theme(
        Theme(
            name="tsdynamics-gallery",
            palette=(TEAL, INDIGO, AMBER, ROSE, TEAL2),
            background=None,  # transparent — readable in both docs colour schemes
            foreground="#888888",
            font_family="IBM Plex Sans",
            font_size=8.5,
            title_size=9.5,
            grid=False,
            line_width=1.2,
            marker_size=4.0,
        )
    )
    set_theme("tsdynamics-gallery")
    return plt


def _check_snippet(setup: str, call: str) -> None:
    """Parse the snippet, so an authoring slip fails loudly instead of silently."""
    import ast

    ast.parse(setup)
    node = ast.parse(call, mode="eval")
    del node


def _namespace(setup: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Execute ``setup`` once per distinct source and memoise its namespace.

    Several primitives of one transform share a subject; integrating the Lorenz
    attractor once for all five phase-portrait tabs is the difference between a
    gallery build and a coffee break.
    """
    if setup in cache:
        return cache[setup]
    import tsdynamics as ts

    ns: dict[str, Any] = {"ts": ts}
    exec(compile(setup, "<gallery-setup>", "exec"), ns)  # noqa: S102 - authored here
    cache[setup] = ns
    return ns


def _render_one(
    variant: Variant, out_path: pathlib.Path, *, primitive: str | None, ns_cache
) -> None:
    """Execute one snippet, check it drew what the tab claims, write the figure."""
    import matplotlib.pyplot as plt

    setup = variant.setup or ""
    call = variant.call or ""
    _check_snippet(setup, call)
    ns = _namespace(setup, ns_cache)
    spec = eval(compile(call, "<gallery-call>", "eval"), dict(ns))  # noqa: S307
    from tsdynamics.viz.spec import PlotSpec

    if not isinstance(spec, PlotSpec):
        raise TypeError(f"snippet {call!r} returned {type(spec).__name__}, not a PlotSpec")
    if primitive is not None:
        _check_primitive(spec, primitive, call)
    size = FIGSIZE_SQUARE if variant.square else FIGSIZE
    spec.size(*size, dpi=DPI)
    fig = spec.render("matplotlib")
    figure = getattr(fig, "figure", fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.close(figure)


def _check_primitive(spec, primitive: str, call: str) -> None:
    """Fail unless the spec actually carries a mark the named primitive builds.

    The one way a curated snippet can lie: a tab labelled ``contour`` whose call
    forgot ``primitive="contour"`` would show the *default* picture under the
    wrong heading, and nothing else would notice.  A primitive declares the
    :class:`~tsdynamics.viz.spec.PlotKind` marks it lowers to, so the check is
    exact for every pair whose marks differ (it cannot separate ``line`` from
    ``steps``, which are both ``LINE`` by construction).
    """
    from tsdynamics.viz.transforms import PRIMITIVES

    marks = PRIMITIVES[primitive].marks
    drawn = {layer.kind for layer in spec.layers}
    if not (drawn & marks):
        raise ValueError(
            f"snippet {call!r} is shown under the {primitive!r} tab but drew "
            f"{sorted(k.value for k in drawn)}, none of which that primitive builds "
            f"({sorted(m.value for m in marks)})."
        )


def _fallback_showcase(record) -> Showcase | None:
    """Build a Showcase from a transform's own ``example`` factory.

    The safety net that keeps "adding a transform is one registration and nothing
    else" true: a transform with no curated entry still appears in the gallery,
    drawn on the small fixture its registration already ships for the
    compatibility gate.  It is labelled as such, so an uncurated entry is visible
    to whoever reads the page — including the author of the next transform.
    """
    if record.example is None:
        return None
    return Showcase(
        setup=f"# subject and options from the transform's own example factory\n"
        f"subject, options = ts.viz.transforms.get({record.name!r}).example("
        f"{record.default_primitive!r})",
        call=f"ts.plot(subject, {record.name!r}, **options)",
        caption=(
            f"*(No curated example yet — drawn on the small fixture "
            f"`{record.name}` registers for the compatibility gate. "
            f"Add one to `docs/_tooling/gallery.py`.)*"
        ),
    )


def _tab_order(record, show: Showcase) -> list[str]:
    """Primitives for one entry: the hero first, then the rest alphabetically."""
    hero = show.hero or record.default_primitive
    rest = sorted(p for p in record.primitives if p != hero)
    return [hero, *rest] if hero in record.primitives else sorted(record.primitives)


def render_all(*, only: set[str] | None = None, force: bool = False, figures: bool = True) -> Build:
    """Render every declared cell of the matrix; return the :class:`Build`.

    Parameters
    ----------
    only : set of str, optional
        Restrict to these transform names (a fast preview).
    force : bool, optional
        Ignore the cache and re-render.
    figures : bool, optional
        ``False`` builds the page structure and snippets with no figures at all —
        what ``TSD_DOCS_FIGURES=0`` asks for.
    """
    from tsdynamics.analysis._result_viz import VisualizationNotInstalled
    from tsdynamics.viz.transforms import transforms

    build = Build()
    ns_cache: dict[str, dict[str, Any]] = {}
    if figures:
        _prepare_matplotlib()

    for record in transforms():
        if only is not None and record.name not in only:
            continue
        show = SHOWCASE.get(record.name) or _fallback_showcase(record)
        if show is None:  # pragma: no cover - registration requires an example in-tree
            build.failures.append((record.name, "-", "no curated Showcase and no example factory"))
            continue
        curated = record.name in SHOWCASE
        for primitive in _tab_order(record, show):
            variant = show.variant(primitive, default=record.default_primitive)
            cell = Cell(
                transform=record.name,
                primitive=primitive,
                snippet=_snippet_text(variant),
                caption=variant.caption or "",
                curated=curated,
            )
            if not figures:
                build.cells.append(cell)
                continue
            key = cache_key(variant.setup or "", variant.call)
            cached = CACHE_DIR / f"{record.name}-{primitive}-{key}.png"
            if cached.exists() and not force:
                cell.filename, cell.cached_path, cell.from_cache = cached.name, cached, True
                build.cells.append(cell)
                continue
            started = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    # A gallery render must not be defeated by a backend's
                    # "I cannot honor this style key" notice; the honoring
                    # contract has its own gate.
                    warnings.simplefilter("ignore")
                    _render_one(variant, cached, primitive=primitive, ns_cache=ns_cache)
            except VisualizationNotInstalled as exc:
                fallback = show.on_missing
                if fallback is None:
                    build.failures.append((record.name, primitive, f"{type(exc).__name__}: {exc}"))
                    continue
                alt = _retarget(fallback, show, primitive, default=record.default_primitive)
                key = cache_key(alt.setup or "", alt.call or "")
                cached = CACHE_DIR / f"{record.name}-{primitive}-{key}.png"
                cell.snippet = _snippet_text(alt)
                cell.caption = alt.caption or cell.caption
                cell.note = "optional dependency missing — fallback example"
                if cached.exists() and not force:
                    cell.from_cache = True
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            _render_one(alt, cached, primitive=primitive, ns_cache=ns_cache)
                    except Exception as exc2:  # noqa: BLE001
                        build.failures.append(
                            (record.name, primitive, f"fallback {type(exc2).__name__}: {exc2}")
                        )
                        continue
            except Exception as exc:  # noqa: BLE001
                build.failures.append((record.name, primitive, f"{type(exc).__name__}: {exc}"))
                continue
            cell.seconds = time.perf_counter() - started
            cell.filename, cell.cached_path = cached.name, cached
            build.cells.append(cell)

    if only is None:
        _render_compositions(build, ns_cache=ns_cache, force=force, figures=figures)
    return build


def _render_compositions(build: Build, *, ns_cache, force: bool, figures: bool) -> None:
    """Render the cross-family overlays that make the composition claim checkable."""
    for title, show in COMPOSITIONS:
        variant = Variant(
            call=show.call, setup=show.setup, caption=show.caption, square=show.square
        )
        cell = Cell(
            transform=title,
            primitive="overlay",
            snippet=_snippet_text(variant),
            caption=variant.caption or "",
        )
        if not figures:
            build.compositions.append(cell)
            continue
        key = cache_key(variant.setup or "", variant.call or "")
        slug = title.lower().replace(" ", "-").replace(",", "")
        cached = CACHE_DIR / f"composition-{slug}-{key}.png"
        if not (cached.exists() and not force):
            started = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _render_one(variant, cached, primitive=None, ns_cache=ns_cache)
            except Exception as exc:  # noqa: BLE001
                build.failures.append((title, "overlay", f"{type(exc).__name__}: {exc}"))
                continue
            cell.seconds = time.perf_counter() - started
        else:
            cell.from_cache = True
        cell.filename, cell.cached_path = cached.name, cached
        build.compositions.append(cell)


def _retarget(fallback: Variant, show: Showcase, primitive: str, *, default: str) -> Variant:
    """Point a fallback variant at one primitive, reusing the entry's defaults."""
    call = fallback.call if fallback.call is not None else show.call
    if 'primitive="' not in call and primitive != default:
        call = _with_primitive(call, primitive)
    return Variant(
        call=call,
        setup=fallback.setup if fallback.setup is not None else show.setup,
        caption=fallback.caption if fallback.caption is not None else show.caption,
        square=fallback.square or show.square,
    )


def _snippet_text(variant: Variant) -> str:
    """Return the displayed code block: the import, the setup, the call — as executed."""
    setup = (variant.setup or "").strip("\n")
    parts = ["import tsdynamics as ts"]
    if setup:
        parts.append(setup)
    parts.append(variant.call or "")
    return "\n\n".join(parts)


# ===========================================================================
# The page
# ===========================================================================
_SOURCE_SECTIONS = (
    (
        "data",
        "Data-capable transforms",
        "Computable from a series or a point set — a `Trajectory`, a bare NumPy "
        "array, an analysis result. They **also** accept a system, because a model "
        "gives you data for free (it is integrated once, and the choice is recorded "
        "in `meta`).",
    ),
    (
        "model",
        "Model-only transforms",
        "These must evaluate or integrate the right-hand side at points that are "
        "**not** in the input, so they need the system itself. Handing one a bare "
        "array cannot work, and it says so rather than guessing.",
    ),
)


def _rel(target: str) -> str:
    """Site-root-relative path as seen from the gallery page."""
    return "../" * PAGE_URI.count("/") + target


def _alt_text(record, cell: Cell) -> str:
    """Return accessible alt text: what the picture shows, not what produced it."""
    doc = record.doc.rstrip(".") if record.doc else record.name.replace("_", " ")
    return f"{doc}, drawn with the {cell.primitive} primitive"


def _matrix_table(records) -> str:
    """Render the whole compatibility matrix as one readable table."""
    lines = [
        "| transform | source | space | primitives (**bold** = default) | what it draws |",
        "|---|---|---|---|---|",
    ]
    for rec in records:
        prims = ", ".join(
            f"**`{p}`**" if p == rec.default_primitive else f"`{p}`" for p in sorted(rec.primitives)
        )
        space = "/".join(s.value for s in rec.frame)
        flag = "" if rec.available else f" *(needs `{rec.requires}`)*"
        lines.append(
            f"| [`{rec.name}`](#{rec.name}) | {rec.source} | `{space}` | {prims} | "
            f"{rec.doc}{flag} |"
        )
    return "\n".join(lines)


def page(build: Build) -> str:
    """Render the gallery body markdown (what replaces :data:`TOKEN`)."""
    from tsdynamics.viz.transforms import transforms

    records = transforms()
    by_transform: dict[str, list[Cell]] = {}
    for cell in build.cells:
        by_transform.setdefault(cell.transform, []).append(cell)

    n_cells = len(build.cells)
    out: list[str] = []
    out.append(
        f"There are **{len(records)} plot transforms** and **{n_cells} declared "
        f"(transform, primitive) pairs** below. Every figure on this page was produced "
        f"by the code printed beside it, and the page itself is generated from "
        f"`registry.plot_transforms` at build time — so a transform that stopped "
        f"drawing would take the docs build down with it."
    )
    out.append("")
    out.append(_matrix_table(records))
    out.append("")

    for source, title, blurb in _SOURCE_SECTIONS:
        chosen = [r for r in records if r.source == source]
        if not chosen:
            continue
        out.append(f"## {title}")
        out.append("")
        out.append(blurb)
        out.append("")
        for rec in chosen:
            cells = by_transform.get(rec.name, [])
            out.extend(_entry(rec, cells))

    if build.compositions:
        out.append("## Compositions")
        out.append("")
        out.append(
            "Nothing above is a special case: a transform's geometry declares the "
            "**coordinate space** it lives in, and any two that agree can share one set "
            "of axes. That is what makes the overlays below one call rather than a "
            "figure-assembly exercise — and it is also why an `(x, z)` portrait "
            "*refuses* `(x, y)` equilibria, and why the `(tr J, det J)` plane refuses a "
            "state space:"
        )
        out.append("")
        out.append("```")
        out.append(
            "InvalidParameterError: cannot overlay frame 'param2' (diagnostic_curve) on "
            "frame 'state2'\n(phase_portrait_2d): they are drawings of different spaces. "
            "Use layout='stack' / 'row' / 'grid'\nto give each its own panel."
        )
        out.append("```")
        out.append("")
        for cell in build.compositions:
            out.append(f"### {cell.transform}")
            out.append("")
            if cell.filename:
                src = _rel(f"{ASSET_DIR}/{cell.filename}")
                out.append(f"![{cell.transform}]({src}){{ loading=lazy }}")
                out.append("")
            if cell.caption:
                out.append(cell.caption)
                out.append("")
            out.append("```python")
            out.append(cell.snippet)
            out.append("```")
            out.append("")

    if build.failures:  # pragma: no cover - a failure fails the build first
        out.append('!!! failure "Some cells did not render"')
        for name, primitive, message in build.failures:
            out.append(f"    - `{name}` / `{primitive}`: {message}")
        out.append("")
    return "\n".join(out)


def _entry(record, cells: list[Cell]) -> list[str]:
    """One transform's section: heading, blurb, one tab per declared primitive."""
    out: list[str] = []
    out.append(f"### `{record.name}` {{ #{record.name} }}")
    out.append("")
    space = "/".join(s.value for s in record.frame)
    out.append(
        f'<span class="ts-kicker">{record.source} · {space} · '
        f"{len(record.primitives)} primitive{'s' if len(record.primitives) != 1 else ''}</span>"
    )
    out.append("")
    if record.doc:
        out.append(record.doc)
        out.append("")
    if record.analysis:
        out.append(f"Adapts `{record.analysis}` — the transform itself owns no new math.")
        out.append("")
    if not record.available:
        out.append(
            f'!!! info "Needs the optional `{record.requires}` package"\n\n'
            f"    `pip install tsdynamics[{record.requires}]`. The row is listed here "
            f"rather than hidden, because a missing row reads as *this plot does not "
            f"exist*, which is a different and less useful statement."
        )
        out.append("")
    if not cells:  # pragma: no cover - only when a render failed
        out.append(f'!!! warning "No figure was rendered for `{record.name}`."')
        out.append("")
        return out
    for cell in cells:
        # ``PlotTransform.exclusive`` was deleted in v6 (measured ``frozenset()``
        # on all 39 rows, and not a parameter of ``register``), so the
        # " · exclusive" suffix this line used to compute could never appear.
        mark = " (default)" if cell.primitive == record.default_primitive else ""
        out.append(f'=== "`{cell.primitive}`{mark}"')
        out.append("")
        if cell.filename:
            alt = _alt_text(record, cell)
            src = _rel(f"{ASSET_DIR}/{cell.filename}")
            out.append(f"    ![{alt}]({src}){{ loading=lazy }}")
            out.append("")
        if cell.caption:
            out.append(textwrap.indent(cell.caption, "    "))
            out.append("")
        if cell.note:
            out.append(f'    !!! info "{cell.note}"')
            out.append("")
        out.append("    ```python")
        out.append(textwrap.indent(cell.snippet, "    "))
        out.append("    ```")
        out.append("")
    return out


# ===========================================================================
# Standalone entry point
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    """Render the gallery from the command line and report what happened."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=None, help="comma-separated transform names")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    parser.add_argument("--no-figures", action="store_true", help="structure only")
    args = parser.parse_args(argv)

    only = {n.strip() for n in args.only.split(",")} if args.only else None
    started = time.perf_counter()
    build = render_all(only=only, force=args.force, figures=not args.no_figures)
    rendered = [c for c in build.cells if not c.from_cache and c.filename]
    slow = sorted(rendered, key=lambda c: -c.seconds)[:8]
    print(
        f"gallery: {len(build.cells)} cells "
        f"({len(rendered)} rendered, {sum(c.from_cache for c in build.cells)} cached) "
        f"in {time.perf_counter() - started:.1f}s"
    )
    uncurated = sorted({c.transform for c in build.cells if not c.curated})
    if uncurated:
        print(f"gallery: uncurated (example-factory fallback): {uncurated}")
    if slow:
        print(
            "gallery: slowest — "
            + ", ".join(f"{c.transform}.{c.primitive} {c.seconds:.1f}s" for c in slow)
        )
    for name, primitive, message in build.failures:
        print(f"gallery: FAILED {name}.{primitive}: {message}")
    return 1 if build.failures else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
