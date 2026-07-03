r"""
Build-time **animated spatial-field movies** for the per-system documentation pages.

Where :mod:`figures` renders a *static* PNG of a spatial-field system's final field,
this module emits the **evolving field movie** the library already supports — the
same ``system.to_plot_spec(kind="field", animate=True)`` →
:data:`~tsdynamics.viz.spec.PlotKind.SPATIAL_FIELD` path a user reaches — rendered
to a small H.264 ``.mp4`` (with a poster PNG) and embedded like the interactive
three.js attractor viewers: a lazy-loading, autoplaying, looping ``<video>`` on the
system page.  So a Gray–Scott / Swift–Hohenberg page opens with its pattern
*growing on camera* instead of a frozen final still.

Dispatch (who gets a movie)
---------------------------
:func:`eligible` matches any system that declares an ``_field_shape`` ClassVar (the
spatial grid) — today the two 2-D reaction-diffusion fields (:class:`GrayScott`,
:class:`SwiftHohenberg`) and any future one.  The 1-D
:class:`~tsdynamics.systems.continuous.chaotic_attractors.KuramotoSivashinsky`
already has a curated spacetime-carpet figure and no ``_field_shape``, so it is not
swept here (its hero stays the carpet); a movie is *additive* — an ineligible /
disabled / soft-failing system keeps its static figure.

WOW recipe (per system)
-----------------------
Each field's movie is tuned in :data:`MOVIE_RECIPES` for a striking, *clearly
evolving* result — a higher-resolution grid than the fast test default, a vivid
regime, a long-enough horizon that the pattern visibly develops and fills, a
perceptually-uniform (or diverging, for a signed field) colormap, smooth image
interpolation, and no chrome (title / axes / frame).  The knobs:

- ``grid`` — the grid size ``N`` the movie integrates at (finer than the coarse
  registry default so the pattern reads crisp);
- ``params`` — the regime (Gray–Scott feed/kill; Swift–Hohenberg growth control);
- ``final_time`` / ``dt`` — the integration horizon and the **output cadence**
  (``final_time / dt`` frames; the adaptive engine sub-steps internally, so ``dt``
  only sets how many field snapshots the movie plays);
- ``cmap`` / ``interpolation`` — the look (``magma`` for the one-sided activator,
  ``RdBu_r`` for the signed Swift–Hohenberg field, ``bilinear`` for a smooth image);
- ``fps`` — playback rate (the movie is ``n_frames / fps`` seconds long).

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
import hashlib
import html
import inspect
import os
import pathlib
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "docs-field-movies"

#: Bump when the movie recipe / encoding / poster shaping materially changes
#: (cache buster — invalidates every on-disk movie).
MOVIE_VERSION = "1"

#: Output pixel size (square) for the rendered field movie.  720 px is a crisp,
#: web-light hero; the H.264 mp4 for a 96² field stays well under a couple hundred
#: KB (a GIF fallback is larger but still bounded).
_PX = 720
_DPI = 100

# --- Brand background (matches the three.js viewer's dark canvas) -------------
_BG = "#0B0F14"


#: Per-system movie recipes.  Each entry tunes one ``_field_shape`` system for a
#: striking, clearly-evolving hero (see the module docstring for the knobs).
#:
#: ``final_time / dt`` is the **frame count** the movie plays; ``dt`` is only the
#: output cadence (the adaptive engine sub-steps for accuracy), so a coarse ``dt``
#: keeps the stack light while the pattern still integrates faithfully.
MOVIE_RECIPES: dict[str, dict] = {
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
    # Swift–Hohenberg: from a small random field, watch stripes/labyrinth nucleate
    # and coarsen.  N=64 (finer than the 32 default) resolves the ~2π-wavelength
    # rolls; the signed field uses a diverging ``RdBu_r`` cmap so crest/trough read
    # as warm/cool.  ~120 time units captures the full pattern-selection transient;
    # 600 frames at 28 fps → a ~21 s loop.
    "SwiftHohenberg": {
        "grid": 64,
        "params": {"r": 0.4},
        "final_time": 120.0,
        "dt": 0.2,
        "cmap": "RdBu_r",
        "interpolation": "bilinear",
        "fps": 28,
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
    shape = getattr(entry.cls, "_field_shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(n) for n in shape)
    except (TypeError, ValueError):
        return None


def eligible(entry) -> bool:
    """Whether ``entry`` gets an animated field movie (else it keeps its static PNG).

    Eligible when the system declares a **2-D** ``_field_shape`` (a method-of-lines
    reaction-diffusion / pattern-forming field) and figures are enabled.  A 1-D
    field (or none) is not swept here — its curated spacetime figure reads better as
    the hero.
    """
    if _figures_disabled():
        return False
    shape = _field_shape(entry)
    if shape is None or len(shape) != 2:
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
        "grid": int(shape[0]),
        "params": {},
        "final_time": 100.0,
        "dt": 0.5,
        "cmap": "viridis",
        "interpolation": "bilinear",
        "fps": 25,
    }


def cache_key(entry) -> str:
    """Content hash: class source + this system's recipe + module knobs."""
    cls_src = inspect.getsource(entry.cls)
    recipe = repr(sorted(_recipe(entry).items()))
    knobs = "|".join(str(k) for k in (MOVIE_VERSION, _PX, _DPI, _BG))
    return hashlib.sha256((cls_src + recipe + knobs).encode()).hexdigest()[:20]


def _ffmpeg_available() -> bool:
    """Whether matplotlib can write an mp4 (the ffmpeg writer is available)."""
    try:
        from matplotlib.animation import writers

        return bool(writers.is_available("ffmpeg"))
    except Exception:  # noqa: BLE001 — no matplotlib / no writer registry
        return False


def _build_spec(entry, recipe: dict):
    """Integrate ``entry`` and build the animated ``SPATIAL_FIELD`` spec (or raise).

    Uses the library's own ``system.to_plot_spec(kind="field", animate=True)`` front
    door — the exact path a user reaches — with the recipe's grid / params / horizon,
    then applies the vivid colormap + smooth interpolation + no-chrome styling.
    """
    import tsdynamics as ts

    cls = entry.cls
    grid = int(recipe.get("grid", (_field_shape(entry) or (32,))[0]))
    params = dict(recipe.get("params", {}))
    # Build the system at the movie grid (N is structural → its own instance).
    sys_obj = cls(N=grid) if grid else cls()
    for key, val in params.items():
        if key in sys_obj.params:
            sys_obj.params[key] = val

    spec_kw: dict = {
        "kind": "field",
        "final_time": float(recipe["final_time"]),
        "dt": float(recipe["dt"]),
        "animate": True,
    }
    component = recipe.get("component")
    if component is not None:
        spec_kw["components"] = component

    spec = sys_obj.to_plot_spec(**spec_kw)
    # Vivid, smooth, chrome-free hero: perceptually-uniform / diverging cmap,
    # image interpolation, and no title / axes / frame so the field fills the panel;
    # the dark brand background matches the three.js viewer's canvas (and the
    # ``.ts-field-movie`` CSS) so there is no white margin around the square.
    with contextlib.suppress(Exception):
        spec.style(
            cmap=recipe.get("cmap", "viridis"),
            interpolation=recipe.get("interpolation", "bilinear"),
            axes=False,
        )
    with contextlib.suppress(Exception):
        spec.background(_BG)
    # Optional colour-range override.  For a signed field (Swift–Hohenberg) a tight
    # SYMMETRIC clim scaled below the saturation amplitude makes the mature pattern
    # render at full diverging-colormap saturation instead of washing out against the
    # rare extreme; ``"symmetric"`` derives ``±q·max|field|`` from the stack.
    _apply_clim(spec, recipe.get("clim"))
    spec.title = None  # drop the system-name title (the page already has the heading)

    # Play EVERY integrated snapshot (not the capped default 360) so the movie is a
    # full, smooth, long-enough loop — the field stack has ``final_time/dt`` frames.
    n_frames = _stack_frames(spec)
    spec.animate(fps=int(recipe.get("fps", 25)), loop=True, n_frames=n_frames)
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


def _write_poster(spec, poster_path: pathlib.Path) -> bool:
    """Write the movie's *final*-frame poster PNG (the ``<video poster>`` still).

    A still of the animated spec is its fully-evolved final field — the same image
    :mod:`figures` would render, but on the movie's grid / regime / colormap, so the
    poster matches the last movie frame.  Returns ``True`` on success.
    """
    try:
        spec.save(str(poster_path), dpi=_DPI, size=(_PX, _PX))
        return poster_path.exists()
    except Exception:  # noqa: BLE001 — a missing poster is non-fatal (video still plays)
        return False


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
    if movie.exists() and poster.exists():
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

                extra = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
                with mpl.rc_context({"animation.ffmpeg_args": extra}):
                    spec.save(str(tmp), fps=fps, dpi=_DPI, size=(_PX, _PX))
            else:
                spec.save(str(tmp), fps=fps, dpi=_DPI, size=(_PX, _PX))
        except Exception:  # noqa: BLE001 — encoder failure → soft-fail to static PNG
            with contextlib.suppress(OSError):
                tmp.unlink()
            return None
        if not tmp.exists():
            return None
        tmp.replace(movie)
        # The poster is the movie's fully-evolved final frame (a still of the spec).
        _write_poster(spec, poster)

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

    An **mp4** is a muted, looping, autoplaying, inline ``<video>`` (with the
    final-frame ``poster`` shown until it loads); a **gif** fallback is a plain
    ``<img>`` (a GIF autoplays and loops natively).
    """
    depth = uri.count("/") + 1
    prefix = "../" * depth
    src = prefix + movie_uri
    alt = html.escape(f"{entry.name} spatial-field movie")
    if ext == "gif":
        return f'<img class="ts-field-movie" src="{src}" alt="{alt}" loading="lazy" />'
    poster_attr = ""
    if poster_uri:
        poster_attr = f' poster="{prefix + poster_uri}"'
    # autoplay+muted+playsinline is the reliable "plays automatically" combination
    # across browsers (autoplay is only honoured for muted inline video).
    return (
        f'<video class="ts-field-movie" src="{src}"{poster_attr} '
        f'autoplay loop muted playsinline preload="metadata" '
        f'aria-label="{alt}"></video>'
    )
