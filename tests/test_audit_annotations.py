"""Regression tests: ``vline`` / ``hline`` annotation ``text`` is drawn on the axes.

Before the fix, ``_apply_annotations`` passed the annotation ``text`` only as the
line's *legend label* (``label=ann.text``) and never rendered it, so the label was
invisible unless a legend was shown.  It also forwarded the whole style dict to
``axvline`` / ``axhline``, so a text-only style key (e.g. ``fontsize``) raised.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import tsdynamics as ts
from tsdynamics.viz import Annotation


def _time_series_spec():
    traj = ts.Lorenz().integrate(final_time=2.0, dt=0.1)
    return traj.to_plot_spec(components=["x"])  # a simple 2-D time-series panel


def test_vline_text_is_drawn_as_an_axes_text_artist():
    spec = _time_series_spec()
    spec.annotations = [
        Annotation("vline", x=0.5, text="r₁", style={"color": "red", "linestyle": "--"})
    ]
    fig = spec.render("matplotlib")
    ax = fig.axes[0]
    drawn = [t.get_text() for t in ax.texts]
    assert "r₁" in drawn  # pre-fix: text lived only as a (hidden) legend label


def test_hline_text_is_drawn_as_an_axes_text_artist():
    spec = _time_series_spec()
    spec.annotations = [Annotation("hline", y=0.0, text="zero", style={"color": "green"})]
    fig = spec.render("matplotlib")
    ax = fig.axes[0]
    assert "zero" in [t.get_text() for t in ax.texts]


def test_vline_label_honours_a_text_only_style_key_without_breaking_the_line():
    # ``fontsize`` is a text-only key; pre-fix it was forwarded to axvline and raised.
    spec = _time_series_spec()
    spec.annotations = [
        Annotation(
            "vline", x=0.5, text="lbl", style={"color": "k", "linestyle": ":", "fontsize": 14}
        )
    ]
    fig = spec.render("matplotlib")  # must not raise
    ax = fig.axes[0]
    label = next(t for t in ax.texts if t.get_text() == "lbl")
    assert label.get_fontsize() == 14
    assert label.get_color() in ("k", "black", (0.0, 0.0, 0.0, 1.0))


def test_textless_vline_adds_no_text_artist():
    spec = _time_series_spec()
    spec.annotations = [Annotation("vline", x=0.5, style={"color": "red"})]
    fig = spec.render("matplotlib")
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts if t.get_text()] == []
