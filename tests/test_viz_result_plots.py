"""Gates for the result-plot repairs: F4 (kwarg routing), F9 (the ``s``/``filled``
collision) and H4 (result plots that carry their result's meaning).

Every defect here shared one failure mode: the *spec dict* was well-formed, so no
existing test could see it, while the drawn artifact was wrong or the caller's
request was silently discarded.  These tests assert on the artifact, on the
normalized style, and on md5-distinctness of rendered PNGs — the honest form of
"the keyword actually reached the renderer".
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pytest

from tsdynamics.errors import InvalidParameterError

pytest.importorskip("matplotlib")


def _png(spec, tmp_path: Path, name: str) -> str:  # noqa: ANN001 - test helper
    """Hash the PNG a built spec writes.

    ``.plot()`` returns a :class:`PlotSpec` (v6), not a figure, so the artifact
    comes from the spec's own ``save`` — the verb that exists on every plottable
    object in the library.
    """
    p = tmp_path / f"{name}.png"
    spec.save(str(p))
    return hashlib.md5(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def orbit_diagram():  # noqa: ANN202
    import tsdynamics as ts

    return ts.analysis.orbit_diagram(
        ts.systems.Logistic(), "r", np.linspace(2.8, 4.0, 120), points_per_value=40, transient=200
    )


# ---------------------------------------------------------------------------
# F4 — ``.plot(**kwargs)`` must not swallow
# ---------------------------------------------------------------------------


def test_spec_keyword_reaches_the_artifact(orbit_diagram, tmp_path: Path) -> None:  # noqa: ANN001
    """``OrbitDiagram.to_plot_spec`` declares ``annotate=``; ``.plot()`` must forward it.

    Before this fix ``.plot()``, ``.plot(annotate=True)`` and
    ``.plot(totally_bogus_kwarg=42)`` produced **byte-identical** PNGs: every
    keyword went into ``backend_kw``, where the renderers' ``**_kw`` catch-all
    absorbed it.
    """
    plain = _png(orbit_diagram.plot(), tmp_path, "plain")
    annotated = _png(orbit_diagram.plot(annotate=True), tmp_path, "annotated")
    assert plain != annotated, "annotate=True did not reach the renderer"


def test_unknown_keyword_raises_and_names_both_accepted_sets(orbit_diagram) -> None:  # noqa: ANN001
    with pytest.raises(InvalidParameterError) as exc:
        orbit_diagram.plot(totally_bogus_kwarg=42)
    msg = str(exc.value)
    assert "totally_bogus_kwarg" in msg
    assert "annotate" in msg, "the message must name this result's own spec keywords"
    assert "figsize" in msg, "the message must name the backend keywords"


def test_backend_keyword_still_reaches_the_renderer(orbit_diagram, tmp_path: Path) -> None:  # noqa: ANN001
    """A renderer keyword still reaches the renderer, and still changes the artifact.

    ``.plot()`` hands back the spec rather than the figure (v6), so the figure is
    checked where it is produced — inside ``render`` — and the end-to-end effect
    is checked on the written PNG.
    """
    sizes = []
    small = orbit_diagram.plot(figsize=(3.0, 2.0))
    sizes.append(tuple(small.render("matplotlib", figsize=(3.0, 2.0)).get_size_inches()))
    big = orbit_diagram.plot(figsize=(8.0, 6.0))
    sizes.append(tuple(big.render("matplotlib", figsize=(8.0, 6.0)).get_size_inches()))
    assert sizes == [(3.0, 2.0), (8.0, 6.0)]


def test_typed_accessor_still_works(orbit_diagram) -> None:  # noqa: ANN001
    fig = orbit_diagram.plot.bifurcation()
    assert fig is not None


# ---------------------------------------------------------------------------
# F9 — the ``s`` / ``filled`` style collision
# ---------------------------------------------------------------------------


def test_stable_and_unstable_styles_survive_normalization() -> None:
    """They used to differ *only* in ``filled``, which ``normalize_style`` dropped.

    ``filled`` was not a canonical style key, so ``normalize_style(stable) ==
    normalize_style(unstable)`` was ``True`` and the whole stable/unstable
    distinction was gone before any renderer saw it.
    """
    from tsdynamics.analysis.fixedpoints.fixed import _fixed_point_style
    from tsdynamics.viz.style import normalize_style

    stable = normalize_style(_fixed_point_style(True), warn=False)
    unstable = normalize_style(_fixed_point_style(False), warn=False)
    assert stable != unstable
    # Encoded redundantly: fill *and* colour, so the distinction survives a
    # backend that does not honor ``filled`` and a greyscale print.
    assert stable["filled"] is True and unstable["filled"] is False
    assert stable["color"] != unstable["color"]


def test_fixed_point_marker_size_is_a_diameter_not_an_area() -> None:
    """The emitters spoke matplotlib's ``s`` (pt^2) into a key that means pt.

    ``{"s": 40.0}`` normalized to ``markersize=40``; ``_draw_scatter`` then squares
    it to ``s=1600`` — 40x too large linearly, 1600x by area.  That is the "three
    enormous discs" a Lorenz equilibrium plot used to be.
    """
    from tsdynamics.analysis.fixedpoints.fixed import _fixed_point_style

    size = _fixed_point_style(True)["markersize"]
    assert 3.0 <= float(size) <= 12.0, f"marker diameter {size} pt is not a plausible marker"


def test_no_bare_s_style_key_remains_in_the_library() -> None:
    """A grep gate: ``"s"`` is no longer a style key, and its trap is worth pinning.

    It collided head-on with matplotlib's ``s`` = *area* convention that every
    internal emitter assumed, **and** with the marker-shape *value* ``"s"`` =
    square (``{"s": 1, "marker": "s"}`` used one letter for both).
    """
    import ast

    root = Path(__file__).resolve().parents[1] / "src" / "tsdynamics"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "style" or not isinstance(kw.value, ast.Dict):
                    continue
                for key in kw.value.keys:
                    if isinstance(key, ast.Constant) and key.value == "s":
                        offenders.append(f"{path}:{key.lineno}")
    assert not offenders, f'style={{"s": ...}} is no longer a valid style key: {offenders}'


def test_stable_and_unstable_render_differently(tmp_path: Path) -> None:
    """The end of the chain: two differently-classified points must *look* different."""
    from tsdynamics.analysis.fixedpoints.fixed import _fixed_point_style
    from tsdynamics.viz.spec import Layer, PlotKind, PlotSpec

    def one(stable: bool):  # noqa: ANN202
        spec = PlotSpec(
            kind=PlotKind.FIXED_POINTS_OVERLAY,
            layers=[
                Layer(
                    kind=PlotKind.SCATTER,
                    data={"x": np.array([0.0]), "y": np.array([0.0])},
                    style=_fixed_point_style(stable),
                )
            ],
        )
        return spec

    assert _png(one(True), tmp_path, "stable") != _png(one(False), tmp_path, "unstable")


# ---------------------------------------------------------------------------
# H4 — result plots that carry their result's meaning
# ---------------------------------------------------------------------------


def test_fixed_points_label_axes_with_the_systems_variables() -> None:
    """``$x_0$`` / ``$x_1$`` throws away names the system already declares."""
    import tsdynamics as ts

    spec = ts.analysis.fixed_points(ts.systems.Lorenz()).to_plot_spec()
    assert spec.x.label == "$x$"
    assert spec.y.label == "$y$"


def test_axis_labels_fall_back_to_indices_without_variables() -> None:
    from tsdynamics.analysis import _plotbuilder as pb

    assert pb.axis_labels({}, (0, 2)) == ["$x_{0}$", "$x_{2}$"]
    assert pb.axis_labels({"variables": ("theta", "omega")}, (0, 1)) == ["$theta$", "$omega$"]
    # A short ``variables`` tuple must not index out of range.
    assert pb.axis_labels({"variables": ("u",)}, (0, 1)) == ["$u$", "$x_{1}$"]


def test_3d_specs_draw_their_annotations() -> None:
    """3-D silently dropped ``spec.annotations``: the helper was never called."""
    import tsdynamics as ts
    from tsdynamics.viz.spec import Annotation

    tr = ts.systems.Lorenz().run(final_time=5.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    spec = tr.to_plot_spec()
    before = len(spec.render(backend="matplotlib").axes[0].lines)
    spec.annotations.append(Annotation(kind="vline", x=0.0, text="x = 0"))
    after = len(spec.render(backend="matplotlib").axes[0].lines)
    assert after == before + 1, "the vline annotation did not reach the 3-D axes"


def test_dark_theme_themes_the_3d_panes() -> None:
    """Otherwise a dark page shows a light-grey 3-D box floating in it."""
    from matplotlib.colors import to_rgba

    import tsdynamics as ts

    tr = ts.systems.Lorenz().run(final_time=5.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    fig = tr.to_plot_spec().theme("dark").render(backend="matplotlib")
    ax = fig.axes[0]
    background = to_rgba(fig.get_facecolor())
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        assert to_rgba(axis.pane.get_facecolor()) == background


def test_basin_colorbar_is_a_categorical_legend() -> None:
    """A basin colour channel is an attractor *id*, not a quantity.

    It used to read ``0.5 / 1.5 / 2.5`` — the ``BoundaryNorm`` bin edges — which
    names nothing and implies an ordering the data does not have.
    """
    import tsdynamics as ts

    class _Duffing(ts.ContinuousSystem):
        params = {"delta": 0.15, "alpha": -1.0, "beta": 1.0}
        dim = 2
        variables = ("x", "v")
        default_ic = [0.5, 0.0]

        @staticmethod
        def _equations(y, t, delta, alpha, beta):  # noqa: ANN001, ANN205
            return [y(1), -delta * y(1) - alpha * y(0) - beta * y(0) ** 3]

    grid = ts.data.Grid(lo=[-2.0, -1.5], hi=[2.0, 1.5], counts=[16, 16])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ts.analysis.basins(_Duffing(), grid, dt=0.05, max_steps=2000)

    spec = res.to_plot_spec()
    labels = spec.meta["category_labels"]
    assert all(isinstance(v, str) for v in labels.values())
    assert any(v.startswith("attractor") for v in labels.values())
    # The system knows its coordinates are (x, v) — the plot must say so.
    assert spec.x.label == "$x$"
    assert spec.y.label == "$v$"

    fig = spec.render(backend="matplotlib")
    ticklabels = [t.get_text() for t in fig.axes[-1].get_yticklabels()]
    assert any(t.startswith("attractor") for t in ticklabels), ticklabels
    assert not any(t.strip().endswith(".5") for t in ticklabels), ticklabels


# ---------------------------------------------------------------------------
# Every result plot is a *plot*: it has axes, and they are labelled
# ---------------------------------------------------------------------------


def _result_specs():
    """One spec per analysis result type that owns a ``to_plot_spec``.

    Built once and shared, because several of these integrate a system.  Kept in
    one place so a new result type joins the sweep by being added here rather than
    by someone remembering to write a bespoke test for it.
    """
    import tsdynamics as ts

    lor = ts.systems.Lorenz()
    traj = lor.run(final_time=40.0, dt=0.01, ic=[1.0, 1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "fixed_points": ts.analysis.fixed_points(lor, seed=0).to_plot_spec(),
            "eigenvalue_plane": ts.analysis.fixed_points(lor, seed=0).eigenvalue_plane(),
            "orbit_diagram": ts.analysis.orbit_diagram(
                ts.systems.Logistic(),
                "r",
                np.linspace(2.8, 4.0, 60),
                points_per_value=30,
                transient=200,
            ).to_plot_spec(),
            "basins": ts.analysis.basins(
                ts.systems.Henon(), ts.data.Grid([-2.0, -2.0], [2.0, 2.0], (24, 24))
            ).to_plot_spec(),
            "recurrence": ts.analysis.recurrence_matrix(
                traj.y[:300], recurrence_rate=0.05
            ).to_plot_spec(),
            "dimension": ts.analysis.correlation_dimension(traj.y[::4]).to_plot_spec(),
            "poincare": ts.analysis.poincare_section(
                ts.systems.Rossler(), plane=("y", 0.0, "up"), crossings=120, seed=0
            ).to_plot_spec(),
            "gali": ts.analysis.gali(lor, k=2, final_time=40.0, ic=[1.0, 1.0, 1.0]).to_plot_spec(),
            "return_map": ts.analysis.return_map(traj, component="z", kind="max").to_plot_spec(),
            "lyapunov_from_data": ts.analysis.lyapunov_from_data(
                traj["x"][::4], dt=0.04, dimension=3
            ).to_plot_spec(),
        }


@pytest.fixture(scope="module")
def result_specs():  # noqa: ANN202
    return _result_specs()


@pytest.mark.parametrize(
    "name",
    [
        "fixed_points",
        "eigenvalue_plane",
        "orbit_diagram",
        "basins",
        "recurrence",
        "dimension",
        "poincare",
        "gali",
        "return_map",
        "lyapunov_from_data",
    ],
)
def test_every_result_plot_carries_labelled_axes_and_a_title(result_specs, name) -> None:  # noqa: ANN001
    """A result plot with an unlabelled axis is a plot the reader has to guess at.

    Cheap, but it is the check that catches a whole class of regression: a new
    result type, or a refactor of an existing one, that emits layers and forgets
    the axes carry the meaning.
    """
    spec = result_specs[name]
    assert spec.layers, f"{name} drew nothing"
    assert spec.title, f"{name} has no title"
    assert spec.x.label, f"{name} has no x-axis label"
    assert spec.y.label, f"{name} has no y-axis label"


def test_every_result_plot_renders_to_a_real_png(result_specs, tmp_path: Path) -> None:  # noqa: ANN001
    """…and every one of them survives an actual matplotlib render.

    A spec dict that validates is not evidence a figure draws; this is.
    """
    for name, spec in result_specs.items():
        out = tmp_path / f"{name}.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.save(out)
        assert out.exists() and out.stat().st_size > 5_000, f"{name} rendered an empty PNG"


def test_the_eigenvalue_plane_does_not_write_its_boundary_over_an_eigenvalue() -> None:
    """The stability criterion belongs in the title, not across the data.

    The ``Re λ = 0`` reference line used to carry an inline text label, which the
    matplotlib renderer draws rotated against the *top* of the axes — precisely
    where a complex conjugate pair near the imaginary axis sits.  On Lorenz it ran
    straight through the eigenvalue at ``+0.094 + 10.2i``.
    """
    import tsdynamics as ts

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = ts.analysis.fixed_points(ts.systems.Lorenz(), seed=0).eigenvalue_plane()

    assert all(not ann.text for ann in spec.annotations if ann.kind == "vline")
    assert "Re" in spec.title and "0" in spec.title
