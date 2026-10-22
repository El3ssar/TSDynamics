"""Render the animation guide's exact examples and verify their checked-in footage.

Run ``uv run --no-sync python docs/_tooling/make_animation_showcase.py`` after
changing a tagged example. Use ``--only animation-camera`` for one clip, or
``--check`` for the dependency-free source/asset freshness guard used by docs CI.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs/visualization/animation.md"
OUT = ROOT / "docs/assets/figures/viz"
MANIFEST = OUT / "animation-showcase.json"
REGENERATE = "uv run --no-sync python docs/_tooling/make_animation_showcase.py"
ENCODING = {
    "codec": "libx264",
    "crf": 21,
    "preset": "fast",
    "pixel_format": "yuv420p",
    "movflags": "+faststart",
    "dpi": 120,
}


@dataclass(frozen=True)
class Clip:
    """An authored example and its deliberate export geometry."""

    slug: str
    pixels: tuple[int, int]


CLIPS = (
    Clip("animation-reveal", (960, 600)),
    Clip("animation-trails", (1104, 432)),
    Clip("animation-camera", (1152, 480)),
    Clip("animation-overlay", (960, 560)),
    Clip("animation-linked", (1080, 480)),
    Clip("animation-nested", (1080, 720)),
    Clip("animation-map", (960, 600)),
    Clip("animation-field", (960, 600)),
    Clip("animation-pingpong", (960, 560)),
)


def example_source(slug: str) -> str:
    """Read one marked, executable documentation fence without injecting names."""
    matches = re.findall(
        rf"<!-- tsd-example: {re.escape(slug)} -->\s*```python\n(.*?)```",
        PAGE.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one executable example {slug!r}, found {len(matches)}.")
    return matches[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _RendererSemantics(ast.NodeTransformer):
    """Discard prose and release stamps while retaining actual drawing behavior."""

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            return None
        return self.generic_visit(node)


def _canonical(node):
    """Ignore empty AST fields added by newer supported Python parsers."""
    if isinstance(node, ast.AST):
        return type(node).__name__, tuple(
            (name, _canonical(value))
            for name, value in ast.iter_fields(node)
            if value is not None and value != []
        )
    if isinstance(node, list):
        return tuple(_canonical(value) for value in node)
    return node


def _renderer_sources():
    source = ROOT / "src"
    viz = source / "tsdynamics/viz"
    paths = [*sorted((viz / "render/mpl").rglob("*.py"))]
    paths += [viz / name for name in ("spec.py", "style.py", "compose.py")]
    return sorted((path.relative_to(source).as_posix(), path) for path in paths)


def renderer_fingerprint() -> str:
    """Hash the MP4 renderer only, using portable labels and semantic Python ASTs."""
    digest = hashlib.sha256()
    for label, path in _renderer_sources():
        tree = _RendererSemantics().visit(ast.parse(path.read_text(encoding="utf-8")))
        digest.update(label.encode())
        digest.update(repr(_canonical(tree)).encode())
    return digest.hexdigest()


def build_movie(slug: str):
    """Execute exactly the code the reader sees, and return its ``movie`` plot."""
    namespace = {}
    exec(compile(example_source(slug), f"animation.md:{slug}", "exec"), namespace)
    movie = namespace.get("movie")
    if movie is None or not getattr(movie, "is_animated", False):
        raise ValueError(f"Example {slug!r} must produce an animated plot named movie.")
    return movie


def _probe(path: Path) -> dict:
    """Decode every frame with FFmpeg, including installations without ffprobe."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-loglevel",
            "info",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            "showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    clock = re.search(r"config in time_base:\s*(\d+/\d+), frame_rate:\s*(\d+/\d+)", result.stderr)
    codec = re.search(r"Stream #[^\n]*?Video:\s*(\w+)", result.stderr)
    frames = re.findall(
        r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\S+.*?\bfmt:(\S+).*?\bs:(\d+)x(\d+)",
        result.stderr,
    )
    if not clock or not codec or not frames:
        raise ValueError(f"Could not inspect the decoded movie {path.name}.")
    if [int(frame[0]) for frame in frames] != list(range(len(frames))):
        raise ValueError(f"Non-contiguous decoded frames in {path.name}.")
    if len({frame[2:] for frame in frames}) != 1:
        raise ValueError(f"The decoded format changes during {path.name}.")
    fps = float(Fraction(clock[2]))
    return {
        "frames": len(frames),
        "fps": fps,
        "duration_seconds": len(frames) / fps,
        "width": int(frames[0][3]),
        "height": int(frames[0][4]),
        "codec": codec[1],
        "pixel_format": frames[0][2],
    }


def render_clip(clip: Clip) -> dict:
    """Encode one real movie, extract its two-thirds frame, and record evidence."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    source = example_source(clip.slug)
    renderer_hash = renderer_fingerprint()
    movie = build_movie(clip.slug)
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / f"{clip.slug}.mp4"
    poster = OUT / f"{clip.slug}-poster.png"
    with mpl.rc_context(
        {
            "animation.codec": ENCODING["codec"],
            "animation.ffmpeg_args": [
                "-crf",
                str(ENCODING["crf"]),
                "-preset",
                ENCODING["preset"],
                "-pix_fmt",
                ENCODING["pixel_format"],
                "-movflags",
                ENCODING["movflags"],
            ],
        }
    ):
        try:
            movie.save(str(video), size=clip.pixels, dpi=ENCODING["dpi"])
        finally:
            plt.close("all")
    info = _probe(video)
    animation = movie.animation
    forward = animation.frame_count(2)
    expected = forward + max(forward - 2, 0) if animation.pingpong else forward
    if info["frames"] != expected or info["fps"] != animation.fps:
        raise ValueError(f"Playback timing does not match the example for {clip.slug}: {info}")
    if (info["width"], info["height"]) != clip.pixels:
        raise ValueError(f"Unexpected encoded dimensions for {clip.slug}: {info}")
    if info["codec"] != "h264" or info["pixel_format"] != "yuv420p":
        raise ValueError(f"Expected web-compatible H.264/yuv420p for {clip.slug}: {info}")
    poster_time = info["duration_seconds"] * 2 / 3
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            str(poster_time),
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(poster),
        ],
        check=True,
        timeout=30,
    )
    if example_source(clip.slug) != source or renderer_fingerprint() != renderer_hash:
        raise ValueError(f"Source changed while rendering {clip.slug}; run the generator again.")
    return {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "renderer_sha256": renderer_hash,
        "encoding": ENCODING.copy(),
        "video": video.name,
        "video_sha256": _sha256(video),
        "poster": poster.name,
        "poster_sha256": _sha256(poster),
        "poster_seconds": poster_time,
        **info,
    }


def verify_manifest() -> None:
    """Reject stale/missing footage using only Python's standard library.

    This is the strict-build seam: import this module, then call this function.
    Tagged source, MP4 rendering semantics and encoding settings invalidate a
    clip; prose edits do not.
    Failures include the exact bounded regeneration command.
    """
    failures = []
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported animation showcase manifest schema.")
        records = manifest["clips"]
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(
            f"Missing or invalid animation showcase manifest. Run: {REGENERATE}"
        ) from exc
    if set(records) != {clip.slug for clip in CLIPS}:
        failures.append("manifest does not cover the nine declared examples")
    renderer_hash = renderer_fingerprint()
    for clip in CLIPS:
        record = records.get(clip.slug, {})
        source_hash = hashlib.sha256(example_source(clip.slug).encode()).hexdigest()
        if record.get("source_sha256") != source_hash:
            failures.append(f"{clip.slug}: example changed since its movie was rendered")
        if record.get("renderer_sha256") != renderer_hash:
            failures.append(f"{clip.slug}: MP4 renderer changed since its movie was rendered")
        if record.get("encoding") != ENCODING:
            failures.append(f"{clip.slug}: encoding settings changed")
        if (record.get("width"), record.get("height")) != clip.pixels:
            failures.append(f"{clip.slug}: encoded dimensions do not match its recipe")
        for kind in ("video", "poster"):
            asset = OUT / f"{clip.slug}{'.mp4' if kind == 'video' else '-poster.png'}"
            if record.get(kind) != asset.name:
                failures.append(f"{clip.slug}: incorrect {kind} filename")
            if not asset.is_file() or record.get(f"{kind}_sha256") != _sha256(asset):
                failures.append(f"{clip.slug}: missing or changed {kind}")
    if failures:
        raise ValueError(
            "Animation showcase is stale:\n- " + "\n- ".join(failures) + f"\nRun: {REGENERATE}"
        )


def validate_assets() -> list[str]:
    """Return actionable freshness errors, without importing plotting dependencies."""
    try:
        verify_manifest()
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        message = str(exc)
        return [message if REGENERATE in message else f"{message}\nRun: {REGENERATE}"]
    return []


def main() -> None:
    """Regenerate selected clips, or run the cheap source/asset freshness guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=[clip.slug for clip in CLIPS])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        verify_manifest()
        print("All nine animation examples and their checked-in assets are current.")
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Install FFmpeg on PATH before generating the animation showcase.")
    import matplotlib

    matplotlib.use("Agg")
    try:
        records = json.loads(MANIFEST.read_text(encoding="utf-8"))["clips"]
    except (OSError, ValueError, KeyError):
        records = {}
    for clip in CLIPS:
        if args.only and clip.slug not in args.only:
            continue
        started = time.perf_counter()
        records[clip.slug] = render_clip(clip)
        temporary_manifest = MANIFEST.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps({"schema_version": 1, "clips": records}, indent=2) + "\n", encoding="utf-8"
        )
        temporary_manifest.replace(MANIFEST)
        print(
            f"{clip.slug}: {records[clip.slug]['frames']} frames in {time.perf_counter() - started:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
