"""Real spatial snapshots, portable encoding, and user-controlled documentation media."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _video import probe_video

import tsdynamics as ts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "_tooling"))
import catalog  # noqa: E402
import field_movies  # noqa: E402


class _DocFixedField(ts.ContinuousSystem):
    """A fixed-grid field whose exact solution is exponential decay."""

    dim = 4
    _field_shape = (2, 2)
    default_ic = (1.0, 2.0, 3.0, 4.0)

    def _equations(y, t):
        return [-y(i) for i in range(4)]


def _record(name):
    record = catalog.load_catalog().by_name(name)
    assert record is not None
    return record


def test_all_curated_spatial_models_have_a_movie_path(monkeypatch):
    monkeypatch.setenv("TSD_DOCS_FIGURES", "1")
    for name in ("Lorenz96", "KuramotoSivashinsky", "GrayScott", "SwiftHohenberg"):
        assert field_movies.eligible(_record(name)), name
    assert not field_movies.eligible(_record("Lorenz"))
    monkeypatch.setenv("TSD_DOCS_FIGURES", "0")
    assert not field_movies.eligible(_record("Lorenz96"))


@pytest.mark.parametrize("name,grid", [("Lorenz96", 8), ("KuramotoSivashinsky", 16)])
def test_profile_movie_contains_the_evolving_full_spatial_state(name, grid):
    pytest.importorskip("matplotlib")
    recipe = {"grid": grid, "profile": True, "params": {}, "final_time": 1.0, "dt": 0.25, "fps": 6}
    spec = field_movies._build_spec(_record(name), recipe)
    assert spec.title == ""
    frames = np.asarray(spec.layers[0].data["frames"])
    assert frames.shape == (5, grid)
    assert np.isfinite(frames).all()
    assert not np.allclose(frames[0], frames[-1])
    assert spec.animation.mode == "frames"
    assert spec.animation.n_frames == 5
    np.testing.assert_array_equal(spec.layers[0].data["x"], np.arange(grid))


def test_new_fixed_grid_field_needs_no_grid_constructor_parameter():
    pytest.importorskip("matplotlib")
    record = SimpleNamespace(
        name="FixedField", cls=_DocFixedField, family="ode", field_shape=(2, 2), params={}
    )
    recipe = {**field_movies._recipe(record), "final_time": 1.0, "dt": 0.25, "fps": 6}
    spec = field_movies._build_spec(record, recipe)
    frames = np.asarray(spec.layers[0].data["frames"])
    assert frames.shape == (5, 2, 2)
    np.testing.assert_allclose(frames[-1].ravel(), np.arange(1, 5) * np.exp(-1), rtol=1e-7)


def test_field_video_is_controllable_and_curated_settings_are_visible():
    html = field_movies.embed_html(
        _record("SwiftHohenberg"),
        "systems/ode/spatial-fields/SwiftHohenberg.md",
        movie_uri="assets/example.mp4",
        poster_uri="assets/example.png",
        ext="mp4",
    )
    assert "controls" in html and 'preload="none"' in html
    assert "autoplay" not in html
    assert 'src="../../../../assets/example.mp4"' in html
    assert "N=64" in html and "r=0.4" in html
    assert "forward/reverse playback" in html
    gif = field_movies.embed_html(
        _record("Lorenz96"),
        "systems/ode/chaotic-attractors/Lorenz96.md",
        movie_uri="assets/example.gif",
        poster_uri=None,
        ext="gif",
    )
    assert "<details" in gif and "<summary>" in gif
    assert " open" not in gif


def test_profile_mp4_encodes_all_spatial_snapshots(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    if not field_movies._ffmpeg_available():
        pytest.skip("requires FFmpeg")
    record = _record("Lorenz96")
    recipe = {
        "grid": 8,
        "profile": True,
        "params": {},
        "final_time": 1.0,
        "dt": 0.25,
        "fps": 6,
        "size": (320, 160),
    }
    monkeypatch.setattr(field_movies, "CACHE_DIR", tmp_path)
    monkeypatch.setitem(field_movies.MOVIE_RECIPES, record.name, recipe)
    result = field_movies._render_to_cache(record, recipe)
    assert result is not None
    movie, poster = result
    assert movie.stat().st_size > 1000 and poster.stat().st_size > 1000
    stream = probe_video(movie)
    assert stream.n_frames == 5
    assert (stream.width, stream.height) == (320, 160)
    assert stream.codec_name == "h264" and stream.pix_fmt == "yuv420p"
    assert stream.frame_rate == 6
    np.testing.assert_allclose(stream.frame_times, np.arange(5) / 6, atol=1e-6)


def test_ffmpeg_only_video_probe_decodes_the_same_frames_and_timing(tmp_path, monkeypatch):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("requires FFmpeg")
    movie = tmp_path / "probe.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:s=64x48:r=6",
            "-frames:v",
            "5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(movie),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    expected = probe_video(movie)
    # Exercise automatic selection when only the bundled FFmpeg is installed.
    which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "ffprobe" else which(name))
    fallback = probe_video(movie)
    assert fallback.n_frames == expected.n_frames == 5
    assert fallback.frame_rate == expected.frame_rate == 6
    assert (fallback.width, fallback.height) == (expected.width, expected.height) == (64, 48)
    assert fallback.codec_name == expected.codec_name == "h264"
    assert fallback.pix_fmt == expected.pix_fmt == "yuv420p"
    np.testing.assert_allclose(fallback.frame_times, np.arange(5) / 6, atol=1e-6)
    np.testing.assert_allclose(fallback.frame_times, expected.frame_times, atol=1e-6)
