"""The layout algebra composes: the parentheses you type are the layout you get.

``|`` (beside), ``/`` (below) and ``+`` (overlay) read as an algebra, and the
owner drove them by hand and asked for the one property an algebra must have —
that ``(plot(a) | plot(b)) / plot(c)`` be *a row of two with c spanning below*,
not three stacked rows.

Measured at HEAD before this module existed, every nesting collapsed::

    (plot(a) | plot(b)) / plot(c)  ->  panels=3  mode='stack'

so the grouping the parentheses expressed was discarded in silence — the figure
rendered fine, it was just a different figure.  Two properties carry the fix and
both are tested here:

- **structure** — a child arranged *differently* from its parent is kept as a
  nested panel; a child arranged the *same* way is absorbed, so ``a | b | c``
  stays one row of three whichever way it associates;
- **pixels** — the tree is only a claim until the renderer places it, so the
  nested layouts are rendered to an Agg buffer and the *positions* of the axes
  are read back (:mod:`tests._pixels`).
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

import _pixels  # noqa: E402

import tsdynamics as ts  # noqa: E402
import tsdynamics.viz as viz  # noqa: E402
from tsdynamics.errors import InvalidParameterError  # noqa: E402
from tsdynamics.viz.render.caps import VisualizationDegraded  # noqa: E402
from tsdynamics.viz.spec import PlotKind  # noqa: E402

_T = np.linspace(0.0, 10.0, 200)


def _panel(freq: float = 1.0, color: str = "crimson"):
    """One cheap, distinctly-coloured single-panel plot."""
    return viz.draw({"x": _T, "y": np.sin(freq * _T)}, "line", color=color)


def _shape(spec) -> str:
    """Render a composite tree as text: ``stack[row[leaf, leaf], leaf]``."""
    if not spec.is_composite:
        return "leaf"
    mode = spec.layout.mode if spec.layout is not None else "stack"
    return f"{mode}[" + ", ".join(_shape(p) for p in spec.panels) + "]"


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


class TestTheParenthesesAreTheLayout:
    """A grouped sub-expression stays grouped."""

    def test_a_row_inside_a_stack_stays_a_row(self):
        """The owner's line. Before: ``stack[leaf, leaf, leaf]`` — three stacked rows."""
        expr = (_panel(1) | _panel(2)) / _panel(3)
        assert _shape(expr) == "stack[row[leaf, leaf], leaf]"
        assert len(expr.panels) == 2  # a row, and c — not three peers
        assert expr.panels[0].kind is PlotKind.COMPOSITE

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda: (_panel(1) | _panel(2)) / _panel(3), "stack[row[leaf, leaf], leaf]"),
            (lambda: _panel(1) / (_panel(2) | _panel(3)), "stack[leaf, row[leaf, leaf]]"),
            (
                lambda: (_panel(1) | _panel(2)) / (_panel(3) | _panel(4)),
                "stack[row[leaf, leaf], row[leaf, leaf]]",
            ),
            (lambda: (_panel(1) / _panel(2)) | _panel(3), "row[stack[leaf, leaf], leaf]"),
            (
                lambda: ((_panel(1) | _panel(2)) / _panel(3)) | _panel(4),
                "row[stack[row[leaf, leaf], leaf], leaf]",
            ),
        ],
        ids=["row-over-one", "one-over-row", "two-by-two", "column-beside-one", "arbitrary-depth"],
    )
    def test_every_nesting_survives(self, build, expected):
        assert _shape(build()) == expected

    @pytest.mark.parametrize("reverse", [False, True])
    def test_the_same_operator_flattens_so_a_row_of_three_is_a_row_of_three(self, reverse):
        """``(a|b)|c`` and ``a|(b|c)`` are the SAME one row of three.

        Associativity within an operator is the other half of the contract: if a
        same-operator child nested too, ``a|b|c`` would draw a row containing a
        row of two, which is not what the reader typed either.
        """
        a, b, c = _panel(1), _panel(2), _panel(3)
        expr = a | (b | c) if reverse else (a | b) | c
        assert _shape(expr) == "row[leaf, leaf, leaf]"

    def test_stacks_chain_flat_too(self):
        assert _shape((_panel(1) / _panel(2)) / _panel(3)) == "stack[leaf, leaf, leaf]"

    def test_an_overlay_is_still_one_set_of_axes(self):
        """``+`` is unchanged — it merges layers, it does not make a panel."""
        expr = _panel(1) + _panel(2)
        assert not expr.is_composite
        assert len(expr.layers) == 2


class TestTheFlatDoorsAreUnchanged:
    """``ts.viz.grid`` and ``layout=`` keep their flat contract."""

    def test_grid_of_leaves_is_flat(self):
        g = viz.grid(_panel(1), _panel(2), _panel(3), _panel(4), rows=2, cols=2)
        assert _shape(g) == "grid[leaf, leaf, leaf, leaf]"
        assert (g.layout.rows, g.layout.cols) == (2, 2)

    def test_explicit_layout_of_leaves_is_flat(self):
        assert _shape(viz.plot(_panel(1), _panel(2), layout="stack")) == "stack[leaf, leaf]"

    def test_a_same_mode_child_is_still_absorbed_with_its_theme_pushed_down(self):
        """Absorption still inherits the child's figure context (it is not a new drop)."""
        inner = viz.plot(_panel(1), _panel(2), layout="stack").theme("dark")
        outer = viz.plot(inner, _panel(3), layout="stack")
        assert _shape(outer) == "stack[leaf, leaf, leaf]"
        assert all(p.resolved_theme.name == "dark" for p in outer.panels[:2])


class TestANestedTreeSurvivesTheRoundTrip:
    def test_json_round_trip_keeps_the_nesting(self):
        expr = (_panel(1) | _panel(2)) / (_panel(3) | _panel(4))
        back = viz.load(expr.to_json())
        assert _shape(back) == _shape(expr)
        assert back.to_dict() == expr.to_dict()

    def test_one_panels_title_is_not_hoisted_to_the_whole_figure(self):
        """A nested group's empty title is a considered answer, not an absence.

        Measured while building the nesting: ``(portrait | series) / psd`` came
        back suptitled ``"psd"``, because the untitled nested row was *skipped*
        and the lone remaining title looked unanimous — so the page carried the
        name of one of its three panels.
        """
        a, b = _panel(1).relabel(title="x-y"), _panel(2).relabel(title="x(t)")
        c = _panel(3).relabel(title="psd")
        assert ((a | b) / c).title == ""
        # ...and a genuinely unanimous title still carries.
        one, two = _panel(1).relabel(title="Lorenz"), _panel(2).relabel(title="Lorenz")
        assert ((one | two) / _panel(3).relabel(title="Lorenz")).title == "Lorenz"

    def test_the_repr_says_it_is_nested_and_counts_every_panel(self):
        """``2 panels in a 2x1 stack`` is true of ``(a|b)/c`` and describes two curves.

        There are three.  A repr that undercounts the picture on screen is the
        kind of small lie the whole result layer was rewritten to remove.
        """
        text = repr((_panel(1) | _panel(2)) / _panel(3))
        assert "nesting a row" in text
        assert "3 panels in all" in text
        # a flat composite says nothing extra
        assert "nesting" not in repr(_panel(1) | _panel(2) | _panel(3))

    def test_a_nested_child_still_inherits_the_composite_theme_at_render_time(self):
        expr = ((_panel(1) | _panel(2)) / _panel(3)).theme("dark")
        nested = expr.resolved_panels()[0]
        assert nested.resolved_theme.name == "dark"


# ---------------------------------------------------------------------------
# pixels — the tree is a claim until the renderer places it
# ---------------------------------------------------------------------------


def _boxes(expr):
    """Render and return the axes boxes, ordered left-to-right then top-to-bottom."""
    picture = _pixels.render(expr)
    return picture, sorted(picture.panels, key=lambda b: (b.top, b.left))


class TestTheRendererPlacesTheNesting:
    """Assert on where the ink is, not on what the spec says."""

    def test_a_two_by_two_puts_a_panel_in_every_quadrant(self):
        expr = (_panel(1, "crimson") | _panel(2, "navy")) / (
            _panel(3, "green") | _panel(4, "purple")
        )
        picture, boxes = _boxes(expr)
        assert len(boxes) == 4, "a 2x2 must make four axes"
        height, width = picture.rgba.shape[0], picture.rgba.shape[1]
        quadrants = {
            ((b.left + b.right) / 2 > width / 2, (b.top + b.bottom) / 2 > height / 2) for b in boxes
        }
        assert quadrants == {(False, False), (True, False), (False, True), (True, True)}
        # ...and every one of them is drawn in, not merely positioned.
        for i in range(4):
            assert picture.ink(i) > 0.01, f"panel {i} rendered blank"

    def test_a_single_panel_on_its_own_row_spans_the_full_width(self):
        """The owner's words: "the second c in the middle" — c gets the whole row."""
        expr = (_panel(1, "crimson") | _panel(2, "navy")) / _panel(3, "green")
        picture, boxes = _boxes(expr)
        assert len(boxes) == 3
        top_a, top_b, bottom = boxes  # sorted by (top, left)
        span = bottom.right - bottom.left
        assert span > 1.5 * (top_a.right - top_a.left), (
            f"the lone bottom panel is {span}px wide, barely more than the "
            f"{top_a.right - top_a.left}px of a panel in the row above it — it is not spanning"
        )
        # it starts at the left of the first top panel and ends at the right of the second
        assert bottom.left <= top_a.left + _pixels.INSET
        assert bottom.right >= top_b.right - _pixels.INSET
        assert all(picture.ink(i) > 0.01 for i in range(3))

    def test_a_column_beside_a_single_stacks_only_on_its_own_side(self):
        expr = (_panel(1, "crimson") / _panel(2, "navy")) | _panel(3, "green")
        picture, boxes = _boxes(expr)
        assert len(boxes) == 3
        width = picture.rgba.shape[1]
        left = [b for b in boxes if (b.left + b.right) / 2 < width / 2]
        right = [b for b in boxes if (b.left + b.right) / 2 > width / 2]
        assert len(left) == 2 and len(right) == 1, "two on the left, one tall one on the right"
        tall = right[0]
        assert (tall.bottom - tall.top) > 1.5 * (left[0].bottom - left[0].top)
        assert all(picture.ink(i) > 0.01 for i in range(3))


class TestPlotlyDeclinesWhatItCannotNest:
    """``make_subplots`` has no ``subgridspec``; drawing anyway was WRONG, not plain.

    Measured before the decline: ``(a|b)/c`` on plotly produced a two-cell figure
    carrying **one** of the three curves, the other two silently dropped.
    """

    def test_a_nested_composite_falls_back_to_matplotlib_and_says_so(self):
        pytest.importorskip("plotly")
        expr = (_panel(1) | _panel(2)) / _panel(3)
        with pytest.warns(VisualizationDegraded, match="plotly"):
            figure = expr.render(backend="plotly")
        assert type(figure).__module__.startswith("matplotlib")

    def test_a_flat_composite_still_renders_natively_on_plotly(self):
        pytest.importorskip("plotly")
        figure = (_panel(1) | _panel(2) | _panel(3)).render(backend="plotly")
        assert type(figure).__module__.startswith("plotly")
        assert len(figure.data) == 3


# ---------------------------------------------------------------------------
# the animation vocabulary: a wrong word is refused, never silently ignored
# ---------------------------------------------------------------------------


class TestAWrongAnimationWordIsRefused:
    """The owner: "if I change the 'time' tag for 'whatever' it does not yield an error".

    ``Animation``'s signatures already declared the closed vocabularies
    (``Literal["time", "steps"]``, ``Literal["reveal", "frames"]``); nothing
    enforced them, so the value was stored verbatim and the renderer quietly did
    something else.  The owner found the trail tag; this class sweeps the rest of
    the surface the same way.
    """

    def test_the_trail_unit_names_the_two_words_that_work(self):
        with pytest.raises(InvalidParameterError) as excinfo:
            _panel().animate().trail(length=("whatever", 3.0))
        message = str(excinfo.value)
        assert "'time'" in message and "'steps'" in message
        assert "samples" in message  # ...and what each one MEANS

    def test_a_bare_trail_length_is_refused_by_name_not_by_a_raw_unpack(self):
        """It used to raise ``TypeError: cannot unpack non-iterable float object``."""
        with pytest.raises(InvalidParameterError, match=r"trail\(length=3.0\)"):
            _panel().animate().trail(length=3.0)

    @pytest.mark.parametrize(
        ("build", "needle"),
        [
            (lambda p: p.animate(mode="whatever"), "'reveal'"),
            (lambda p: p.animate(fps=-5), "frames per second"),
            (lambda p: p.animate(fps=0), "frames per second"),
            (lambda p: p.animate(n_frames=0), "count of frames"),
            (lambda p: p.animate(duration=-1), "seconds of playback"),
            (lambda p: p.head(symbol="banana"), "marker"),
            (lambda p: p.head(size=-1), "marker size"),
            (lambda p: p.camera(spin="fast"), "full turns"),
            (lambda p: p.trail(backdrop_alpha=3.0), "opacity"),
        ],
        ids=[
            "mode",
            "negative-fps",
            "zero-fps",
            "zero-frames",
            "negative-duration",
            "head-symbol",
            "head-size",
            "camera-spin",
            "backdrop-alpha",
        ],
    )
    def test_every_animation_knob_refuses_a_value_it_would_have_ignored(self, build, needle):
        with pytest.raises(InvalidParameterError) as excinfo:
            build(_panel().animate())
        assert needle in str(excinfo.value)

    @pytest.mark.parametrize(
        "build",
        [
            lambda p: p.animate(mode="frames", fps=60, n_frames=12, duration=2.0),
            lambda p: p.trail(length=("time", 4.0), backdrop_alpha=0.5),
            lambda p: p.trail(length=("steps", 10)),
            lambda p: p.trail(length=None),
            lambda p: p.head(symbol="*", size=9.0),
            lambda p: p.camera(spin=2.0, elev=30.0, azim=45.0),
        ],
        ids=["animate", "trail-time", "trail-steps", "trail-persistent", "head", "camera"],
    )
    def test_the_values_that_do_work_still_work(self, build):
        assert build(_panel().animate()).animation is not None

    def test_the_door_refuses_an_animate_string_naming_the_three_spellings(self):
        """``animate="frames"`` used to be read as a bare truthy flag → a reveal movie."""
        with pytest.raises(InvalidParameterError) as excinfo:
            ts.plot(np.sin(_T), animate="frames")
        message = str(excinfo.value)
        assert "animate=True" in message and "Animation(" in message

    def test_the_dict_door_speaks_the_same_vocabulary(self):
        with pytest.raises(InvalidParameterError, match="'reveal'"):
            ts.plot(np.sin(_T), animate={"mode": "whatever"})

    def test_a_directive_rebuilt_from_json_is_validated_too(self):
        from tsdynamics.viz.spec import Animation

        with pytest.raises(InvalidParameterError, match="tail unit"):
            Animation.from_dict({**Animation().to_dict(), "trail_kind": "whatever"})
