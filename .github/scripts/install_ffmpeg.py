# /// script
# requires-python = ">=3.12"
# dependencies = ["imageio-ffmpeg==0.6.0"]
# ///
"""Install and validate the pinned video encoder without system package mirrors."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


def main() -> None:
    """Publish a verified bundled FFmpeg to subsequent GitHub Actions steps."""
    bundled = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    binaries = (Path(imageio_ffmpeg.__file__).resolve().parent / "binaries").resolve()
    if not bundled.is_relative_to(binaries):
        raise RuntimeError(f"Expected the pinned wheel's bundled FFmpeg, got {bundled}")

    destination = Path(os.environ["RUNNER_TEMP"]) / "bin" / "ffmpeg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    # Keep the executable independent of the isolated environment/cache lifetime.
    shutil.copy2(bundled, destination)
    version = subprocess.check_output([destination, "-version"], text=True, timeout=15)
    print(version.splitlines()[0], flush=True)

    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        movie = Path(temporary) / "encoder-check.mp4"
        subprocess.run(
            [
                destination,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=teal:s=64x64:r=5",
                "-frames:v",
                "5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                movie,
            ],
            check=True,
            timeout=30,
        )
        decoded = subprocess.run(
            [
                destination,
                "-v",
                "error",
                "-i",
                movie,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        expected_bytes = 5 * 64 * 64 * 3
        if len(decoded.stdout) != expected_bytes:
            raise RuntimeError(
                f"Expected five decoded RGB frames ({expected_bytes} bytes), "
                f"got {len(decoded.stdout)} bytes"
            )

    with Path(os.environ["GITHUB_PATH"]).open("a", encoding="utf-8") as environment:
        environment.write(f"{destination.parent}\n")
    print("Validated five H.264/yuv420p frames; FFmpeg is ready for subsequent steps.")


if __name__ == "__main__":
    main()
