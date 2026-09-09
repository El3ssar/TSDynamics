"""Gates for density-aware line resolution (H1) and the constrained layout engine (F8).

Both defects were invisible to the existing suite because the **spec dict was
correct**: the failure was entirely in the drawn artifact.  These tests therefore
assert on rendered matplotlib artists, not on the IR.

H1 — the library's canonical object is a long chaotic trajectory, and at the
theme's constant 1.5 pt stroke a 100,000-sample Lorenz attractor rendered as a
solid blob with every trace of its laminar sheet structure destroyed.  The
resolution law lives in :func:`tsdynamics.viz.producers.autostyle_line`; these
tests pin its two contractual properties (non-regression below the pivot,
monotonicity above it), that an explicit user value always wins, and that the
escape hatch restores the old behaviour exactly.

F8 — no matplotlib layout engine was applied anywhere in ``viz/render``, so a
2x2 composite collided row-2 titles into row-1 tick labels and clipped the row-1
x-labels off the artifact entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.viz.producers import (
    AUTOSTYLE_MIN_ALPHA,
    AUTOSTYLE_MIN_LINEWIDTH,
    AUTOSTYLE_PIVOT,
    autostyle_line,
)
from tsdynamics.viz.spec import Layer, Layout, PlotKind, PlotSpec

pytest.importorskip("matplotlib")

_THEME_WIDTH = 1.5


# ---------------------------------------------------------------------------
# H1 — the law itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 100, 999, AUTOSTYLE_PIVOT])
def test_below_the_pivot_is_the_identity(n: int) -> None:
    """No existing figure moves: at or below the pivot the theme width is returned as-is."""
    assert autostyle_line(n, line_width=_THEME_WIDTH) == (_THEME_WIDTH, None)


def test_width_is_monotonically_non_increasing_and_clamped() -> None:
    counts = [AUTOSTYLE_PIVOT, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 10_000_000]
    widths = [autostyle_line(n, line_width=_THEME_WIDTH)[0] for n in counts]
    assert all(w is not None for w in widths)
    vals = [float(w) for w in widths if w is not None]
    assert vals == sorted(vals, reverse=True), vals
    assert all(AUTOSTYLE_MIN_LINEWIDTH <= w <= _THEME_WIDTH for w in vals), vals


def test_alpha_is_monotonically_non_increasing_and_clamped() -> None:
    counts = [5_000, 10_000, 100_000, 1_000_000, 100_000_000]
    alphas = [autostyle_line(n, line_width=_THEME_WIDTH)[1] for n in counts]
    vals = [float(a) for a in alphas if a is not None]
    assert len(vals) == len(counts)
    assert vals == sorted(vals, reverse=True), vals
    assert all(AUTOSTYLE_MIN_ALPHA <= a <= 1.0 for a in vals), vals


def test_calibration_target_at_100k() -> None:
    """The calibrated law lands on the hand-tuned ``lw=0.25, alpha=0.85`` reference.

    That pair is what a human picked to recover the Lorenz laminar sheets from the
    blob at n ~ 1e5; the exponent was fitted to reproduce it, and the contact
    sheet (rendered and eyeballed in the H1 deliverable) confirms it visually.
    """
    lw, alpha = autostyle_line(100_001, line_width=_THEME_WIDTH)
    assert lw == pytest.approx(0.25, abs=0.03)
    assert alpha == pytest.approx(0.85, abs=0.03)


def test_disabled_is_the_identity_at_any_density() -> None:
    assert autostyle_line(10_000_000, line_width=_THEME_WIDTH, enabled=False) == (
        _THEME_WIDTH,
        None,
    )


def test_no_theme_width_still_yields_an_alpha() -> None:
    lw, alpha = autostyle_line(100_000, line_width=None)
    assert lw is None
    assert alpha is not None and alpha < 1.0


# ---------------------------------------------------------------------------
# H1 — end to end through the renderers
# ---------------------------------------------------------------------------


def _line_spec(n: int, style: dict | None = None, meta: dict | None = None) -> PlotSpec:
    t = np.linspace(0.0, 1.0, n)
    return PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(kind=PlotKind.LINE, data={"x": t, "y": np.sin(50.0 * t)}, style=style or {})],
        meta=meta or {},
    )


def _only_line(fig):  # noqa: ANN001, ANN202 - test helper
    lines = [ln for ax in fig.axes for ln in ax.lines]
    assert len(lines) == 1
    return lines[0]


def test_dense_curve_is_thinned_on_matplotlib() -> None:
    sparse = _only_line(_line_spec(500).render(backend="matplotlib"))
    dense = _only_line(_line_spec(200_000).render(backend="matplotlib"))
    assert dense.get_linewidth() < sparse.get_linewidth()
    assert dense.get_alpha() is not None and dense.get_alpha() < 1.0
    assert sparse.get_alpha() is None


def test_explicit_linewidth_and_alpha_always_win() -> None:
    line = _only_line(
        _line_spec(500_000, {"linewidth": 3.0, "alpha": 1.0}).render(backend="matplotlib")
    )
    assert line.get_linewidth() == pytest.approx(3.0)
    assert line.get_alpha() == pytest.approx(1.0)


def test_meta_escape_hatch_restores_constant_width() -> None:
    """``spec.meta["autostyle"] = False`` reproduces the pre-H1 artifact exactly."""
    off = _only_line(_line_spec(200_000, meta={"autostyle": False}).render(backend="matplotlib"))
    ref = _only_line(_line_spec(500).render(backend="matplotlib"))
    assert off.get_linewidth() == pytest.approx(ref.get_linewidth())
    assert off.get_alpha() is None


def test_dense_curve_is_thinned_on_plotly() -> None:
    """The two drawing backends must agree — a density fix on one only is a new bug."""
    pytest.importorskip("plotly")
    sparse = _line_spec(500).render(backend="plotly").data[0]
    dense = _line_spec(200_000).render(backend="plotly").data[0]
    assert dense.line.width < sparse.line.width
    assert dense.opacity < 1.0


def test_dense_3d_curve_is_thinned() -> None:
    """3-D is 106 of the 136 catalogue ODEs — the majority path must be covered."""
    t = np.linspace(0.0, 1.0, 200_000)

    def spec(n: int) -> PlotSpec:
        u = np.linspace(0.0, 1.0, n)
        return PlotSpec(
            kind=PlotKind.PHASE_PORTRAIT_3D,
            ndim=3,
            layers=[
                Layer(
                    kind=PlotKind.LINE3D,
                    data={"x": u, "y": np.sin(u), "z": np.cos(u)},
                )
            ],
        )

    del t
    dense = _only_line(spec(200_000).render(backend="matplotlib"))
    sparse = _only_line(spec(500).render(backend="matplotlib"))
    assert dense.get_linewidth() < sparse.get_linewidth()


# ---------------------------------------------------------------------------
# F8 — the layout engine
# ---------------------------------------------------------------------------


def _panel(title: str) -> PlotSpec:
    t = np.linspace(0.0, 1.0, 50)
    return PlotSpec(
        kind=PlotKind.TIME_SERIES,
        title=title,
        layers=[Layer(kind=PlotKind.LINE, data={"x": t, "y": np.sin(t)})],
    )


@pytest.mark.parametrize(
    "spec_factory",
    [
        pytest.param(lambda: _line_spec(50), id="single-2d"),
        pytest.param(
            lambda: PlotSpec(
                kind=PlotKind.PHASE_PORTRAIT_3D,
                ndim=3,
                layers=[
                    Layer(
                        kind=PlotKind.LINE3D,
                        data={
                            "x": np.linspace(0, 1, 20),
                            "y": np.linspace(0, 1, 20),
                            "z": np.linspace(0, 1, 20),
                        },
                    )
                ],
            ),
            id="single-3d",
        ),
        pytest.param(
            lambda: PlotSpec(
                kind=PlotKind.COMPOSITE,
                panels=[_panel("a"), _panel("b"), _panel("c"), _panel("d")],
                layout=Layout(mode="grid"),
            ),
            id="composite-grid",
        ),
    ],
)
def test_every_figure_carries_a_layout_engine(spec_factory) -> None:  # noqa: ANN001
    """Without one, decorations are not merely cramped — labels are clipped away."""
    fig = spec_factory().render(backend="matplotlib")
    assert fig.get_layout_engine() is not None


def test_composite_row_one_x_label_survives_a_tight_figure() -> None:
    """The concrete F8 symptom: the top row's x-label used to fall outside the canvas."""
    spec = PlotSpec(
        kind=PlotKind.COMPOSITE,
        panels=[_panel("a"), _panel("b"), _panel("c"), _panel("d")],
        layout=Layout(mode="grid"),
    )
    for panel in spec.panels:
        panel.x.label = "a rather long x axis label"
        panel.y.label = "a rather long y axis label"
    fig = spec.render(backend="matplotlib", figsize=(6.4, 4.8))
    fig.canvas.draw()
    fig_h = fig.get_window_extent().height
    tops = [ax.xaxis.label.get_window_extent().y0 for ax in fig.axes]
    assert all(0.0 <= y <= fig_h for y in tops), (
        f"an x-axis label fell outside the figure canvas (height {fig_h}): {tops}"
    )
