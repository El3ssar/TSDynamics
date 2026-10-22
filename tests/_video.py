"""Decode small test movies without requiring a separate ffprobe installation."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    """Describe the first video stream and every decoded presentation timestamp."""

    n_frames: int
    frame_rate: float
    width: int
    height: int
    codec_name: str
    pix_fmt: str
    frame_times: tuple[float, ...]


def probe_video(path: str | Path, *, prefer_ffprobe: bool = True) -> VideoInfo:
    """Decode all frames with ffprobe, or FFmpeg's showinfo filter when absent."""
    ffprobe = shutil.which("ffprobe") if prefer_ffprobe else None
    if ffprobe is not None:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,r_frame_rate,width,height,codec_name,pix_fmt:"
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        times = tuple(float(frame["best_effort_timestamp_time"]) for frame in payload["frames"])
        count = int(stream["nb_read_frames"])
        assert len(times) == count, "ffprobe did not report every decoded frame timestamp"
        return VideoInfo(
            count,
            float(Fraction(stream["r_frame_rate"])),
            int(stream["width"]),
            int(stream["height"]),
            stream["codec_name"],
            stream["pix_fmt"],
            times,
        )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Video verification requires FFmpeg or ffprobe on PATH.")
    result = subprocess.run(
        [
            ffmpeg,
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
        timeout=30,
    )
    output = result.stderr
    clock = re.search(r"config in time_base:\s*(\d+/\d+), frame_rate:\s*(\d+/\d+)", output)
    codec = re.search(r"Stream #[^\n]*?Video:\s*(\w+)", output)
    frames = re.findall(
        r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\S+.*?\bfmt:(\S+).*?\bs:(\d+)x(\d+)",
        output,
    )
    assert clock and codec and frames, f"FFmpeg did not report a decoded video stream:\n{output}"
    assert [int(frame[0]) for frame in frames] == list(range(len(frames)))
    assert len({frame[2:] for frame in frames}) == 1, "Video format changed during decoding"
    time_base = Fraction(clock[1])
    return VideoInfo(
        len(frames),
        float(Fraction(clock[2])),
        int(frames[0][3]),
        int(frames[0][4]),
        codec[1],
        frames[0][2],
        tuple(float(int(frame[1]) * time_base) for frame in frames),
    )
