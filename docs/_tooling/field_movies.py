r"""
Build-time **animated spatial-field movies** for the per-system documentation pages.

Where :mod:`figures` renders a *static* PNG of a spatial-field system's final field,
this module emits the **evolving field movie** the library already supports — the
same ``system.__plot_spec__(kind="field", animate=True)`` →
:data:`~tsdynamics.viz.spec.PlotKind.SPATIAL_FIELD` path a user reaches — rendered
to a small H.264 ``.mp4`` (with a poster PNG) and embedded like the interactive
three.js attractor viewers: a controlled, on-demand ``<video>`` on the
system page.  So a Gray–Scott / Swift–Hohenberg page opens with its pattern
*growing on camera* instead of a frozen final still.

Dispatch (who gets a movie)
---------------------------
:func:`eligible` matches declared one- and two-dimensional spatial fields, plus
the curated Lorenz-96 and Kuramoto–Sivashinsky profiles. A one-dimensional field
plays as an evolving line profile; a two-dimensional field plays as a heatmap.
The static spatial or spacetime illustration remains available as a fallback.

Showcase recipe (per system)
-----------------------
Each field's movie is tuned in :data:`MOVIE_RECIPES` for a striking, *clearly
evolving* result — a higher-resolution grid than the fast test default, a vivid
regime, a long-enough horizon that the pattern visibly develops and fills, a
perceptually-uniform (or diverging, for a signed field) colormap, smooth image
interpolation, and no chrome (title / axes / frame).  The knobs:

- ``grid`` — the grid size ``N`` the movie integrates at (finer than the coarse
  registry default so the pattern reads crisp);
- ``params`` — the regime (Gray–Scott feed/kill; Swift–Hohenberg growth control);
- ``warmup`` — an optional burn-in horizon (time units) integrated **before** the
  animated window, so the movie starts past a fast initial transient (e.g.
  Swift–Hohenberg's noise→pattern selection burst, which otherwise front-loads all
  the motion); the burn-in's end state seeds the window's ``ic``;
- ``final_time`` / ``dt`` — the animated-window horizon and the **output cadence**
  (``final_time / dt`` frames; the adaptive engine sub-steps internally, so ``dt``
  only sets how many field snapshots the movie plays);
- ``cmap`` / ``interpolation`` — the look (``magma`` for the one-sided activator,
  ``RdBu_r`` for the signed Swift–Hohenberg field, ``bilinear`` for a smooth image);
- ``fps`` — playback rate (the movie is ``n_frames / fps`` seconds long);
- ``pingpong`` — play the window forward then reverse (reuses the footage on the
  return leg and hides the hard loop seam, so evolution reads across the whole loop).

Self-containment, caching & environment mirror :mod:`threejs_viewer`
--------------------------------------------------------------------
Results are content-addressed under ``.cache/docs-field-movies`` keyed by the
system's class source plus this module's recipe/version knobs, so an unchanged
system is a cheap file copy and CI persists the directory between builds (mirroring
:mod:`figures` / :mod:`threejs_viewer`).  ``TSD_DOCS_FIGURES=0`` skips every render
(the page falls back to its static field figure).  The movie is written with
**ffmpeg** (H.264, ``yuv420p``, ``+faststart`` for instant web playback) when it is
available, else with **pillow** to an animated GIF (so a runner without ffmpeg still
ships a moving hero) — :func:`render` returns the produced asset's extension so the
caller emits the right ``<video>`` / ``<img>`` embed.  Every render path soft-fails
to ``None`` (the page falls back to its static PNG).  Nothing is committed to git —
the movie is a build artifact exactly like a three.js viewer.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import html
import os
import pathlib
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "docs-field-movies"

#: Bump when the movie recipe / encoding / poster shaping materially changes
#: (cache buster — invalidates every on-disk movie).
MOVIE_VERSION = "3"

#: Output pixel size (square) for the rendered field movie.  720 px is a crisp,
#: web-light hero; the H.264 mp4 for a 96² field stays well under a couple hundred
#: KB.
_PX = 720
_DPI = 100

# --- GIF-fallback caps (ffmpeg-absent path only) ------------------------------
#: The pillow-GIF fallback (only when ffmpeg is unavailable) is an uncompressed,
#: per-frame-palettised format, so the full 600-frame / 720-px movie balloons to
#: ~170 MB — production never hits this (CI installs ffmpeg → the ~0.4 MB mp4), but
#: a local no-ffmpeg build would silently emit a giant file.  So the GIF branch is
#: capped: subsample to at most :data:`_GIF_MAX_FRAMES` (evenly, via the animation's
#: own frame schedule) and downscale to :data:`_GIF_PX`, targeting a GIF well under
#: ~8 MB.  The mp4 path is untouched (full frames, full resolution).
_GIF_MAX_FRAMES = 150
_GIF_PX = 360

# --- Brand background (matches the three.js viewer's dark canvas) -------------
_BG = "#0B0F14"


#: Per-system movie recipes.  Each entry tunes one ``_field_shape`` system for a
#: striking, clearly-evolving hero (see the module docstring for the knobs).
#:
#: ``final_time / dt`` is the **frame count** the movie plays; ``dt`` is only the
#: output cadence (the adaptive engine sub-steps for accuracy), so a coarse ``dt``
#: keeps the stack light while the pattern still integrates faithfully.
MOVIE_RECIPES: dict[str, dict] = {
    "Lorenz96": {
        "grid": 20,
        "profile": True,
        "final_time": 60.0,
        "dt": 0.2,
        "fps": 25,
        "size": (1000, 480),
    },
    "KuramotoSivashinsky": {
        "grid": 32,
        "profile": True,
        "final_time": 150.0,
        "dt": 0.5,
        "fps": 25,
        "size": (1000, 480),
    },
    # Gray–Scott: nucleate from the seeded central square and let worms grow and
    # fill the frame on camera.  N=96 (finer than the coarse 48 default) reads
    # crisp; F=0.06/k=0.062 is the vivid worms/spots regime; ~9000 time units lets
    # the pattern spread across the whole periodic grid.  600 frames at 28 fps →
    # a ~21 s loop.  ``magma`` on the one-sided activator ``v``; bilinear for a
    # smooth, non-blocky image; the activator block is the plotted default.
    "GrayScott": {
        "grid": 96,
        "params": {"F": 0.06, "k": 0.062},
        "final_time": 9000.0,
        "dt": 15.0,
        "cmap": "magma",
        "interpolation": "bilinear",
        "fps": 28,
        "component": "v",
    },
    # Swift–Hohenberg: watch stripes/labyrinth wander and coarsen — evenly, across
    # the WHOLE loop.  N=64 (finer than the 32 default) resolves the ~2π-wavelength
    # rolls; the signed field uses a diverging ``RdBu_r`` cmap so crest/trough read
    # as warm/cool.
    #
    # WHY the warm-up + pingpong (issue #512 polish): from a small random field, SH's
    # noise→pattern SELECTION burst packs >half of all motion into the first few time
    # units, then the field coarsens ever more slowly — so a naive t=0..120 movie is a
    # 3-second flurry followed by ~18 s of near-frozen coarsening (a viewer landing
    # mid-loop saw an almost-static field).  Instead: burn in ``warmup=30`` time units
    # (past the burst), animate the later ``final_time=70`` window (dt=0.35 → 201
    # snapshots), and ``pingpong`` it (forward then reverse) — so the loop shows the
    # ongoing, evenly-paced coarsening + drift, reuses the vivid footage on the return
    # leg, and hides the hard loop seam.  Measured: motion in the first 15% of frames
    # drops 60.5% → 22.5% and the per-decile share flattens (max/min 2.5→3.0, no dead
    # midpoint).  201 fwd → 400 played frames at 26 fps → a ~15 s loop.
    "SwiftHohenberg": {
        "grid": 64,
        "params": {"r": 0.4},
        "warmup": 30.0,
        "final_time": 70.0,
        "dt": 0.35,
        "cmap": "RdBu_r",
        "interpolation": "bilinear",
        "fps": 26,
        "pingpong": True,
        # Symmetric clim at ~0.72·(peak amplitude) — the p95 of the established
        # stripe pattern — so the mature labyrinth renders at full RdBu_r saturation
        # (warm crest / cool trough) rather than washing out against the rare extreme.
        "clim": ("symmetric", 0.72),
    },
}


def _figures_disabled() -> bool:
    """Whether ``TSD_DOCS_FIGURES=0`` asked us to skip every heavy render."""
    return os.environ.get("TSD_DOCS_FIGURES", "1") == "0"


def _field_shape(entry) -> tuple[int, ...] | None:
    """Return a system's ``_field_shape`` ClassVar (the spatial grid), or ``None``."""
    shape = getattr(entry, "field_shape", None)
    if shape is None:
        shape = getattr(entry.cls, "_field_shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(n) for n in shape)
    except (TypeError, ValueError):
        return None


def eligible(entry) -> bool:
    """Whether ``entry`` gets an animated field movie (else it keeps its static PNG).

    Declared 1-D/2-D field metadata handles new systems automatically. The two
    older profile models have explicit recipes because their classes predate
    that metadata; a large arbitrary state vector is never assumed spatial.
    """
    if _figures_disabled():
        return False
    shape = _field_shape(entry)
    profile = bool(MOVIE_RECIPES.get(entry.name, {}).get("profile"))
    if not profile and (shape is None or len(shape) not in (1, 2)):
        return False
    # A field movie is an ODE flow (the method-of-lines PDE); guard the family.
    return getattr(entry, "family", None) == "ode"


def _recipe(entry) -> dict:
    """Return the movie recipe for ``entry``.

    An explicit :data:`MOVIE_RECIPES` entry, or a sensible default derived from the
    system's own grid / defaults for a field with no curated recipe.
    """
    if entry.name in MOVIE_RECIPES:
        return MOVIE_RECIPES[entry.name]
    # A generic fallback for a new field with no curated recipe: use the system's own
    # grid and a modest horizon so the movie still plays.
    shape = _field_shape(entry) or (32, 32)
    return {
        "grid": None,
        "profile": len(shape) == 1,
        "params": {},
        "final_time": 100.0,
        "dt": 0.5,
        "cmap": "viridis",
        "interpolation": "bilinear",
        "fps": 25,
    }


@functools.lru_cache(maxsize=1)
def _renderer_fingerprint() -> str:
    """Track drawing hooks as well as numerical and rendering implementations."""
    import figures

    # Numerical fingerprints intentionally omit __plot_spec__. Field movies
    # actually call that protocol, so retain its semantics here, including the
    # family implementation and model overrides. Reuse the gallery's portable
    # whole-library digest, which ignores prose and release stamps.
    return figures._library_render_fingerprint()


def cache_key(entry) -> str:
    """Content hash: class source + this system's recipe + module knobs + renderer."""
    import diagnostics

    recipe = repr(sorted(_recipe(entry).items()))
    knobs = "|".join(str(k) for k in (MOVIE_VERSION, _PX, _DPI, _BG))
    renderer = _renderer_fingerprint()
    # Numerical repairs must also invalidate the footage; checking only the
    # class body would retain a movie produced by an older integration engine.
    implementation = diagnostics.fingerprint(entry)
    generator = diagnostics._python_semantics(pathlib.Path(__file__).read_text())
    digest = hashlib.sha256((recipe + knobs + renderer + implementation).encode())
    digest.update(generator)
    return digest.hexdigest()[:20]


def _ffmpeg_available() -> bool:
    """Whether matplotlib can write an mp4 (the ffmpeg writer is available)."""
    try:
        from matplotlib.animation import writers

        return bool(writers.is_available("ffmpeg"))
    except Exception:  # noqa: BLE001 — no matplotlib / no writer registry
        return False


def _build_spec(entry, recipe: dict):
    """Integrate ``entry`` and build the animated ``SPATIAL_FIELD`` spec (or raise).

    Uses the library's own ``system.__plot_spec__(kind="field", animate=True)`` front
    door — the exact path a user reaches — with the recipe's grid / params / horizon,
    then applies the vivid colormap + smooth interpolation + no-chrome styling.
    """
    import tsdynamics as ts

    cls = entry.cls
    grid = recipe.get("grid")
    params = dict(recipe.get("params", {}))
    # Build the system at the movie grid (N is structural → its own instance).
    sys_obj = cls(N=int(grid)) if grid is not None and "N" in entry.params else cls()
    for key, val in params.items():
        if key in sys_obj.params:
            sys_obj.params[key] = val

    spec_kw: dict = {
        "kind": "field",
        "final_time": float(recipe["final_time"]),
        "dt": float(recipe["dt"]),
        "animate": True,
        "seed": 0,
    }
    component = recipe.get("component")
    if component is not None:
        spec_kw["components"] = component

    # Optional warm-up (``warmup`` time units): integrate past the fast initial
    # transient (e.g. Swift–Hohenberg's noise→pattern SELECTION burst, which packs
    # >half of all motion into the first few time units) and animate a **later**
    # window from the warmed field.  So the played loop shows the ongoing, evenly-
    # paced coarsening / drift instead of a front-loaded burst followed by a frozen
    # tail.  The burn-in is one integrate to ``warmup`` whose end state seeds the
    # animated window's ``ic``.
    warmup = float(recipe.get("warmup", 0.0) or 0.0)
    if warmup > 0.0:
        burn = sys_obj.run(final_time=warmup, dt=warmup, seed=0)
        spec_kw["ic"] = np.asarray(burn.y[-1], dtype=float)

    spec = sys_obj.__plot_spec__(**spec_kw)
    # Vivid, smooth, chrome-free hero: perceptually-uniform / diverging cmap,
    # image interpolation, and no title / axes / frame so the field fills the panel;
    # the dark brand background matches the three.js viewer's canvas (and the
    # ``.ts-field-movie`` CSS) so there is no white margin around the square.
    profile = bool(recipe.get("profile"))
    spec.theme("dark").font(family="sans-serif")
    if profile:
        spec.style(color="#2CC5AE", linewidth=1.8)
        spec.relabel(x="grid point", y="state")
    else:
        spec.style(
            cmap=recipe.get("cmap", "viridis"),
            interpolation=recipe.get("interpolation", "bilinear"),
            axes=False,
        )
    spec.background(_BG)
    # Optional colour-range override.  For a signed field (Swift–Hohenberg) a tight
    # SYMMETRIC clim scaled below the saturation amplitude makes the mature pattern
    # render at full diverging-colormap saturation instead of washing out against the
    # rare extreme; ``"symmetric"`` derives ``±q·max|field|`` from the stack.
    _apply_clim(spec, recipe.get("clim"))
    spec.relabel(title="")  # the page already has the system-name heading
    # Drop the colorbar so the poster (the movie's static final frame) is the same
    # chrome-free centred square as the movie itself — the animated field driver
    # draws no colorbar, so this only affects the still poster, keeping them matched.
    spec.colorbar = None

    # Play EVERY integrated snapshot (not the capped default 360) so the movie is a
    # full, smooth, long-enough loop — the field stack has ``final_time/dt`` frames.
    # ``pingpong`` (recipe opt-in) plays the window forward then reverse: it reuses
    # the vivid footage on the return leg AND hides the hard loop seam (a cut from
    # the mature field back to the window's start), so a viewer landing anywhere in
    # the loop always sees motion.
    n_frames = _stack_frames(spec)
    spec.animate(
        fps=int(recipe.get("fps", 25)),
        loop=True,
        n_frames=n_frames,
        pingpong=bool(recipe.get("pingpong", False)),
    )
    # Reference the ts symbol so a bare import is never flagged unused.
    _ = ts
    return spec


def _apply_clim(spec, clim) -> None:
    """Apply an optional colour-range override to ``spec`` (in place).

    ``clim`` may be an explicit ``(vmin, vmax)`` pair, or the recipe token
    ``("symmetric", q)`` → a symmetric range ``±q·max|field|`` derived from the
    layer's field stack (so a diverging colormap saturates at a fraction of the peak
    amplitude, boosting the mature-pattern contrast).  A ``None`` / unrecognised
    value leaves the producer's full-range clim untouched.
    """
    if clim is None:
        return
    with contextlib.suppress(Exception):
        if isinstance(clim, (tuple, list)) and len(clim) == 2 and clim[0] == "symmetric":
            q = float(clim[1])
            peak = 0.0
            for layer in spec.layers:
                frames = layer.data.get("frames")
                if frames is not None:
                    arr = np.asarray(frames, dtype=float)
                    finite = arr[np.isfinite(arr)]
                    if finite.size:
                        peak = max(peak, float(np.abs(finite).max()))
            if peak > 0.0:
                lim = q * peak
                spec.colorize(clim=(-lim, lim))
        else:
            spec.colorize(clim=(float(clim[0]), float(clim[1])))


def _stack_frames(spec) -> int | None:
    """Return the per-time field-stack length (the number of movie frames), or ``None``.

    The ``SPATIAL_FIELD`` producer stacks every integrated snapshot on the layer's
    ``"frames"`` channel (shape ``(T, *spatial)``); ``T`` is the natural movie frame
    count.  ``None`` when no such stack is present (the animator then picks its own).
    """
    for layer in spec.layers:
        frames = layer.data.get("frames")
        if frames is not None:
            arr = np.asarray(frames)
            if arr.ndim >= 2:
                return int(arr.shape[0])
    return None


def _write_poster(spec, poster_path: pathlib.Path, size: tuple[int, int]) -> bool:
    """Write the final simulated field as the ``<video poster>`` still.

    A still of the animated spec is its fully-evolved final field — the same image
    :mod:`figures` would render, but on the movie's grid / regime / colormap, so the
    poster shows the end of the forward pass. A ping-pong movie subsequently
    returns toward the initial field. Returns ``True`` on success.
    """
    try:
        spec.save(str(poster_path), dpi=_DPI, size=size)
        return poster_path.exists()
    except Exception:  # noqa: BLE001 — a missing poster is non-fatal (video still plays)
        return False


def _cap_for_gif(spec) -> None:
    """Cap the animated ``spec`` for the pillow-GIF fallback (in place).

    A GIF is an uncompressed, per-frame-palettised format, so the full-length,
    full-resolution field movie extrapolates to ~170 MB — production never emits it
    (CI installs ffmpeg → the ~0.4 MB mp4), but a local no-ffmpeg build would.  So
    lower the animation's frame count to at most :data:`_GIF_MAX_FRAMES`; the
    animation's own :meth:`~tsdynamics.viz.spec.Animation.head_indices` then
    subsamples the field stack **evenly** across the loop (the whole evolution is
    still shown, just at a coarser cadence).  The pixel downscale rides on the
    ``size=`` argument at :func:`~._render_to_cache` (:data:`_GIF_PX`).  A no-op when
    the spec carries no animation or is already under the cap.
    """
    anim = getattr(spec, "animation", None)
    if anim is None:
        return
    with contextlib.suppress(Exception):
        # Cap the *played* frame count (pingpong doubles the forward sequence), so
        # the forward n_frames target halves when pingpong is on.
        target = _GIF_MAX_FRAMES // 2 if getattr(anim, "pingpong", False) else _GIF_MAX_FRAMES
        current = _stack_frames(spec)
        if current is None:
            current = anim.n_frames
        if current is None or int(current) > target:
            spec.animate(n_frames=target)


def _render_to_cache(entry, recipe: dict) -> tuple[pathlib.Path, pathlib.Path] | None:
    """Render ``entry``'s movie + poster into the cache; return ``(movie, poster)``.

    Prefers an H.264 ``.mp4`` (small, web-optimised) via ffmpeg; falls back to an
    animated ``.gif`` via pillow when ffmpeg is unavailable, so a runner without it
    still ships a moving hero.  Returns ``None`` on any soft failure.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(entry)
    use_mp4 = _ffmpeg_available()
    ext = "mp4" if use_mp4 else "gif"
    movie = CACHE_DIR / f"{entry.name}-{key}.{ext}"
    poster = CACHE_DIR / f"{entry.name}-{key}.png"
    size = tuple(int(n) for n in recipe.get("size", (_PX, _PX)))
    if movie.exists() and movie.stat().st_size and poster.exists() and poster.stat().st_size:
        return movie, poster

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            spec = _build_spec(entry, recipe)
        except Exception:  # noqa: BLE001 — integration / spec build failed
            return None
        fps = int(recipe.get("fps", 25))
        # The temp path MUST keep the real ``.mp4`` / ``.gif`` extension — ``spec.save``
        # picks the encoder from the extension, so a ``.tmp`` suffix would be treated
        # as a still-image request and fail.  Write beside the target, then rename.
        tmp = movie.with_name(f"{movie.stem}.partial{movie.suffix}")
        try:
            if use_mp4:
                # H.264 with web-friendly flags: yuv420p (broad browser support) +
                # +faststart (moov atom up front → instant streaming playback).
                import matplotlib as mpl

                extra = ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2"]
                with mpl.rc_context({"animation.ffmpeg_args": extra}):
                    spec.save(str(tmp), fps=fps, dpi=_DPI, size=size)
            else:
                # Pillow-GIF fallback (ffmpeg absent): cap frames + downscale so the
                # bulky per-frame-palettised GIF stays a few MB, not ~170 MB.
                _cap_for_gif(spec)
                gif_size = (_GIF_PX, max(2, round(_GIF_PX * size[1] / size[0])))
                spec.save(str(tmp), fps=fps, dpi=_DPI, size=gif_size)
        except Exception:  # noqa: BLE001 — encoder failure → soft-fail to static PNG
            with contextlib.suppress(OSError):
                tmp.unlink()
            return None
        if not tmp.exists() or not tmp.stat().st_size:
            return None
        tmp.replace(movie)
        # The poster is the movie's fully-evolved final frame (a still of the spec).
        _write_poster(spec, poster, size)

    return movie, (poster if poster.exists() else movie)


def render(entry) -> dict | None:
    """Return the field-movie embed assets for ``entry`` (cached on disk), or ``None``.

    On success returns ``{"movie": <abs path>, "movie_uri_name": <basename>,
    "poster": <abs path or None>, "ext": "mp4"|"gif"}`` — the caller registers the
    movie (and poster) as generated site files and emits the ``<video>`` embed.
    Returns ``None`` for an ineligible / disabled / soft-failing system (the page
    then falls back to its static field PNG).  **Never raises** — a movie must not
    break the docs build.
    """
    try:
        if not eligible(entry):
            return None
        recipe = _recipe(entry)
        result = _render_to_cache(entry, recipe)
        if result is None:
            return None
        movie, poster = result
        ext = movie.suffix.lstrip(".")
        has_poster = poster.suffix == ".png" and poster.exists()
        return {
            "movie": movie,
            "ext": ext,
            "poster": poster if has_poster else None,
        }
    except Exception:  # noqa: BLE001 — a movie must never break the docs build
        return None


def embed_html(entry, uri: str, *, movie_uri: str, poster_uri: str | None, ext: str) -> str:
    """Return the ``<video>`` (mp4) / ``<img>`` (gif) embed HTML for a field movie.

    The asset ``src`` is resolved against the *output* (directory-URL) location the
    same way :mod:`threejs_viewer` resolves its iframe ``src``: a page
    ``systems/<t>/<c>/<Name>.md`` serves at ``systems/<t>/<c>/<Name>/`` (one deeper
    than the source), so reaching a site-root asset needs one extra ``../`` over the
    source depth.  MkDocs does **not** rewrite ``<video>`` / ``<img>`` ``src``.

    Videos have native controls and start on request. The GIF fallback lives in
    a closed disclosure so readers choose when a non-pausable animation appears.
    """
    depth = uri.count("/") + 1
    prefix = "../" * depth
    src = prefix + movie_uri
    alt = html.escape(f"{entry.name} spatial-field movie")
    profile_class = " ts-field-movie--profile" if _recipe(entry).get("profile") else ""
    classes = f"ts-field-movie{profile_class}"
    recipe = _recipe(entry)
    settings = []
    if recipe.get("grid") is not None:
        settings.append(f"N={recipe['grid']}")
    settings.extend(f"{key}={value:g}" for key, value in recipe.get("params", {}).items())
    warmup = float(recipe.get("warmup", 0))
    settings.append(f"simulation window: {warmup:g}–{warmup + recipe['final_time']:g}")
    settings.append("forward/reverse playback" if recipe.get("pingpong") else "forward playback")
    note = (
        '<p class="ts-field-settings">Illustration: ' + html.escape(" · ".join(settings)) + "</p>"
    )
    if ext == "gif":
        return (
            f'<details class="ts-field-preview"><summary>Open animated field preview</summary>'
            f'<img class="{classes}" src="{src}" alt="{alt}" loading="lazy" />{note}</details>'
        )
    poster_attr = ""
    if poster_uri:
        poster_attr = f' poster="{prefix + poster_uri}"'
    return (
        f'<video class="{classes}" src="{src}"{poster_attr} '
        f'controls muted playsinline preload="none" '
        f'aria-label="{alt}"></video>{note}'
    )
