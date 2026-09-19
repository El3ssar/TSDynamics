"""Structural gates for the ``PlotSpec`` IR (v6 Phase 5, Owner A).

Every defect this file guards was **invisible to the old suite**, because the old
suite asserted that a spec *dict* was right — which it always was.  The bugs were
that the dict never changed (a composite tweak was a silent no-op), that a
structurally impossible spec was constructible (a panel-less ``COMPOSITE``
rendered as a blank figure), and that ``save()`` reported success for a file it
had not written.

So the gates here are deliberately **structural** rather than example-based:

1. ``test_every_public_tweak_declares_a_scope`` — introspects ``PlotSpec`` and
   fails if any public fluent method (one returning a ``PlotSpec``) is neither
   panel-scoped nor figure-scoped.  A *new* tweak method cannot be added without
   making a forwarding decision, which is what stops F1 regressing.
2. ``test_panel_scoped_tweak_reaches_every_panel`` — the value test, parametrized
   over the panel-scoped tweaks: each must change **every** panel's ``to_dict()``.
   Before v6 all twelve left the panels byte-identical.
3. The ``COMPOSITE`` <-> ``panels`` construction invariant.
4. ``save()`` never returns a path it did not write.

Engine-free, fast tier: builds specs directly, renders only through matplotlib.
"""

from __future__ import annotations

import inspect
import typing
from pathlib import Path

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import InvalidParameterError
from tsdynamics.viz._tweaks import tweak_scopes
from tsdynamics.viz.render.caps import VisualizationDegraded
from tsdynamics.viz.spec import Layer, Layout, PlotKind, PlotSpec
from tsdynamics.viz.style import get_theme

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _panel(label: str = "a", *, three_d: bool = False) -> PlotSpec:
    """A minimal single-panel spec (2-D time series, or a 3-D portrait)."""
    t = np.linspace(0.0, 1.0, 16)
    if three_d:
        return PlotSpec(
            kind=PlotKind.PHASE_PORTRAIT_3D,
            layers=[Layer(PlotKind.LINE3D, {"x": t, "y": t**2, "z": np.cos(t)}, label=label)],
            ndim=3,
        )
    return PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(PlotKind.LINE, {"x": t, "y": np.sin(t)}, label=label)],
    )


def _composite(n: int = 2, *, three_d: bool = False) -> PlotSpec:
    """A COMPOSITE spec with ``n`` panels."""
    return PlotSpec(
        kind=PlotKind.COMPOSITE,
        panels=[_panel(f"p{i}", three_d=three_d) for i in range(n)],
        layout=Layout(mode="stack"),
    )


# ---------------------------------------------------------------------------
# 1 — the scope partition is total (the gate that stops F1 regressing)
# ---------------------------------------------------------------------------


def _public_fluent_methods() -> set[str]:
    """Public ``PlotSpec`` methods that return a ``PlotSpec`` — i.e. the tweaks."""
    out: set[str] = set()
    for name, attr in vars(PlotSpec).items():
        if name.startswith("_") or not callable(attr):
            continue
        try:
            ret = inspect.signature(attr).return_annotation
        except (TypeError, ValueError):  # pragma: no cover - builtins / slots
            continue
        # ``from __future__ import annotations`` makes every annotation a string;
        # accept the resolved class too, in case that ever changes.
        if (isinstance(ret, str) and ret.strip() in ("Plot", "PlotSpec")) or ret is PlotSpec:
            out.add(name)
    assert out, "found no fluent tweaks — the introspection broke, not the code"
    return out


def test_every_public_tweak_declares_a_scope() -> None:
    """No public fluent tweak may be unclassified as panel- or figure-scoped.

    This is the gate, not the fix.  A composite owns no axes and no layers, so a
    tweak that does not declare itself panel-scoped is a **silent no-op** there —
    the user chains it, sees a figure, and believes it landed.  Adding a new
    tweak method must therefore force a forwarding decision.
    """
    scopes = tweak_scopes(PlotSpec)
    unclassified = sorted(_public_fluent_methods() - set(scopes))
    assert not unclassified, (
        f"these PlotSpec tweaks declare no scope: {unclassified}. Decorate each with "
        "@panel_scoped() (it must recurse into composite panels) or @figure_scoped "
        "(it belongs to the figure) from tsdynamics.viz._tweaks."
    )
    assert set(scopes.values()) <= {"panel", "figure"}


def test_the_scope_partition_matches_the_reviewed_contract() -> None:
    """The panel/figure split is exactly the reviewed one (independent hard-coding).

    Deliberately **not** derived from the decorators — it is the second opinion,
    so flipping a decorator silently cannot pass.
    """
    expected_panel = {
        "autocolor",
        "camera",
        "colorize",
        "font",
        "gridlines",
        "hline",
        "limits",
        "palette",
        "recolor",
        "relabel",
        "rescale",
        "span",
        "style",
        "text",
        "ticks",
        "vline",
    }
    expected_figure = {
        "animate",
        "background",
        "clock",
        "head",
        # ``tweak`` forwards to the individual tweaks, which recurse themselves.
        "tweak",
        "size",
        "theme",
        "trail",
    }
    scopes = tweak_scopes(PlotSpec)
    assert {k for k, v in scopes.items() if v == "panel"} == expected_panel
    assert {k for k, v in scopes.items() if v == "figure"} == expected_figure


# ---------------------------------------------------------------------------
# 2 — F1: a panel-scoped tweak reaches every panel
# ---------------------------------------------------------------------------

#: name -> (apply, needs a 3-D composite).  One entry per panel-scoped tweak.
_PANEL_TWEAKS: dict[str, tuple[typing.Callable[[PlotSpec], PlotSpec], bool]] = {
    "relabel": (lambda s: s.relabel(x="time"), False),
    "rescale": (lambda s: s.rescale(y="log"), False),
    "limits": (lambda s: s.limits(x=(0.0, 0.5)), False),
    "ticks": (lambda s: s.ticks(x=[0.0, 1.0]), False),
    "style": (lambda s: s.style(linewidth=4.0), False),
    "recolor": (lambda s: s.recolor("red", "green"), False),
    "palette": (lambda s: s.palette(["#123456"]), False),
    "gridlines": (lambda s: s.gridlines(True), False),
    "font": (lambda s: s.font(size=22.0), False),
    "colorize": (lambda s: s.colorize(legend=True), False),
    "camera": (lambda s: s.camera(elev=12.0), True),
}


@pytest.mark.parametrize("name", sorted(_PANEL_TWEAKS))
def test_panel_scoped_tweak_reaches_every_panel(name: str) -> None:
    """Applying a panel-scoped tweak to a composite changes EVERY panel.

    The failing-first case: before v6 every one of these left
    ``composite.to_dict()["panels"]`` byte-identical.
    """
    apply, three_d = _PANEL_TWEAKS[name]
    spec = _composite(2, three_d=three_d)
    before = [p.to_dict() for p in spec.panels]
    apply(spec)
    after = [p.to_dict() for p in spec.panels]
    for i, (b, a) in enumerate(zip(before, after, strict=True)):
        assert b != a, f"{name}() did not reach panel {i} — it is a silent no-op there"


def test_autocolor_reaches_every_panel() -> None:
    """``autocolor`` forwards too (checked on panels that HAVE a color channel).

    Separated from the table above because ``autocolor`` is a deliberate no-op on
    a spec with no color dimension — so on a plain time-series panel "unchanged"
    is the correct answer and would not prove forwarding.
    """
    t = np.linspace(0.0, 1.0, 8)
    panels = [
        PlotSpec(
            kind=PlotKind.TIME_SERIES,
            layers=[Layer(PlotKind.LINE, {"x": t, "y": t, "c": t * (i + 1)})],
        )
        for i in range(2)
    ]
    spec = PlotSpec(kind=PlotKind.COMPOSITE, panels=panels, layout=Layout())
    spec.autocolor()
    for panel in spec.panels:
        assert panel.colorbar is not None
        assert panel.clim is not None


def test_figure_scoped_tweaks_do_not_leak_into_panels() -> None:
    """The dual: a figure-scoped tweak must NOT be pushed onto the panels.

    A panel that pinned its own theme would stop inheriting a later
    ``composite.theme(...)``, and per-panel ``Animation`` objects would
    desynchronise the lockstep master clock.
    """
    spec = _composite(2)
    spec.relabel(title="figure title").theme("dark").background("#000000").size(4.0, 3.0)
    spec.animate(fps=12.0).trail(("steps", 5)).head(True).clock(True)
    assert spec.title == "figure title"
    for panel in spec.panels:
        assert panel.title == ""
        assert panel._theme is None
        assert panel.animation is None
        assert "figsize" not in panel.meta


def test_recolor_addresses_panels_not_layers_on_a_composite() -> None:
    """On a composite, colour i goes to PANEL i (the only reading that can apply)."""
    spec = _composite(3)
    spec.recolor("red", "blue")
    assert [p.layers[0].style["color"] for p in spec.panels] == ["red", "blue", "red"]


def test_camera_spin_stays_on_the_figure_but_angle_forwards() -> None:
    """``camera(elev=)`` is per-panel; ``camera(spin=)`` is the figure's timeline."""
    spec = _composite(2, three_d=True)
    spec.camera(elev=15.0, spin=2.0)
    assert spec.animation is not None and spec.animation.spin == 2.0
    for panel in spec.panels:
        assert panel.meta["camera"]["elev"] == 15.0
        assert panel.animation is None, "spin must not create a per-panel animation"


def test_single_panel_tweaks_are_unchanged_by_the_forwarding_wrapper() -> None:
    """A non-composite spec behaves exactly as before (the wrapper is inert)."""
    spec = _panel()
    spec.relabel(x="t", title="T").rescale(y="log").limits(x=(0.0, 1.0)).style(linewidth=3.0)
    assert spec.x.label == "t"
    assert spec.title == "T"
    assert spec.y.scale == "log"
    assert spec.x.limits == (0.0, 1.0)
    assert spec.layers[0].style["linewidth"] == 3.0


def test_colorize_accepts_cmap() -> None:
    """``colorize(cmap=)`` works — the one kwarg a caller reaches for first.

    Before v6 it raised ``TypeError: unexpected keyword argument 'cmap'``.
    """
    spec = _panel()
    spec.colorize(cmap="viridis", norm="log", discrete=True)
    assert spec.colorbar is not None
    assert spec.colorbar.cmap == "viridis"
    assert spec.colorbar.norm == "log"
    assert spec.colorbar.discrete is True


# ---------------------------------------------------------------------------
# 3 — F2: the COMPOSITE <-> panels construction invariant
# ---------------------------------------------------------------------------


def test_composite_without_panels_is_rejected() -> None:
    """A panel-less COMPOSITE cannot be built (it used to render as a blank figure)."""
    with pytest.raises(InvalidParameterError, match="must carry at least one panel"):
        PlotSpec(kind=PlotKind.COMPOSITE)


def test_panels_without_composite_kind_are_rejected() -> None:
    """Panels on a non-COMPOSITE spec cannot be built (they would be ignored)."""
    with pytest.raises(InvalidParameterError, match="must have kind=COMPOSITE"):
        PlotSpec(kind=PlotKind.TIME_SERIES, panels=[_panel()])


def test_composite_invariant_holds_through_from_dict() -> None:
    """The invariant is enforced on deserialization too, not only on construction."""
    payload = _composite(2).to_dict()
    payload["panels"] = []
    with pytest.raises(InvalidParameterError):
        PlotSpec.from_dict(payload)


# ---------------------------------------------------------------------------
# 4 — H3: the hoisted spec-level helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "rows", "cols", "n", "expected"),
    [
        ("stack", None, None, 3, (3, 1)),
        ("row", None, None, 3, (1, 3)),
        ("grid", None, None, 4, (2, 2)),
        ("grid", None, None, 5, (2, 3)),
        ("grid", None, None, 1, (1, 1)),
        ("grid", 2, 3, 6, (2, 3)),
        ("grid", None, 2, 5, (3, 2)),
        ("grid", 3, None, 7, (3, 3)),
    ],
)
def test_layout_grid_shapes(mode, rows, cols, n, expected) -> None:
    """``Layout.grid`` reproduces the tiling arithmetic the three renderers duplicated."""
    assert Layout(mode=mode, rows=rows, cols=cols).grid(n) == expected


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 9, 12])
def test_layout_grid_always_fits_every_panel(n: int) -> None:
    """Whatever the mode / hints, ``rows * cols`` is large enough for every panel."""
    for layout in (
        Layout(mode="stack"),
        Layout(mode="row"),
        Layout(mode="grid"),
        Layout(mode="grid", cols=2),
        Layout(mode="grid", rows=2),
    ):
        rows, cols = layout.grid(n)
        assert rows * cols >= n, (layout, n, rows, cols)


def _legacy_is_three_d(spec: PlotSpec) -> bool:
    """The predicate as it was written (byte-identically) in all three backends."""
    if spec.ndim == 3 or spec.z is not None:
        return True
    return any(lyr.kind in (PlotKind.LINE3D, PlotKind.SURFACE3D) for lyr in spec.layers)


@pytest.mark.parametrize(
    "spec",
    [
        _panel(),
        _panel(three_d=True),
        PlotSpec(kind=PlotKind.IMAGE, layers=[Layer(PlotKind.IMAGE, {"c": np.zeros((2, 2))})]),
        PlotSpec(
            kind=PlotKind.PHASE_PORTRAIT_3D,
            layers=[Layer(PlotKind.SURFACE3D, {"z": np.zeros((2, 2))})],
            ndim=2,
        ),
    ],
    ids=["2d_line", "3d_line", "image", "surface3d_ndim2"],
)
def test_is_three_d_matches_the_three_copies_it_replaced(spec: PlotSpec) -> None:
    """The hoisted property answers exactly what the deleted duplicates answered."""
    assert spec.is_three_d == _legacy_is_three_d(spec)


def test_is_three_d_recurses_into_panels() -> None:
    """A composite is 3-D when any panel is (what the copies got by per-panel calls)."""
    assert _composite(2, three_d=True).is_three_d is True
    assert _composite(2).is_three_d is False
    mixed = PlotSpec(
        kind=PlotKind.COMPOSITE, panels=[_panel(), _panel(three_d=True)], layout=Layout()
    )
    assert mixed.is_three_d is True


def test_resolved_panels_pushes_theme_and_animation_down() -> None:
    """``resolved_panels`` applies the documented inheritance, without mutating."""
    spec = _composite(2).theme("dark").animate(fps=12.0)
    spec.panels[1].theme("publication")
    resolved = spec.resolved_panels()
    assert resolved[0].resolved_theme.name == "dark"  # inherited
    assert resolved[1].resolved_theme.name == "publication"  # its own wins
    assert all(p.animation is not None and p.animation.fps == 12.0 for p in resolved)
    # The originals are untouched — resolution returns copies.
    assert spec.panels[0]._theme is None
    assert spec.panels[0].animation is None


def test_resolved_panels_is_empty_for_a_single_panel_spec() -> None:
    """A non-composite has no panels to resolve."""
    assert _panel().resolved_panels() == []


@pytest.mark.parametrize(
    ("tweak", "args", "kwargs"),
    [
        ("palette", (("#ff9900", "#00ccff"),), {}),
        ("font", (), {"size": 13.0}),
        ("gridlines", (True,), {"color": "#888888"}),
    ],
)
def test_a_forwarded_theme_tweak_keeps_the_composites_theme(tweak, args, kwargs) -> None:
    """A panel-scoped tweak that pins a theme must build on the *inherited* one.

    ``palette`` / ``font`` / ``grid(color=)`` resolve their base theme as "mine,
    else the **global** default".  Forwarded into a panel that has no theme of its
    own, that discarded the composite's: ``plot(...).theme("dark").palette(...)``
    stamped a ``default``-based theme on every panel, so the figure rendered dark
    chrome (from the composite) around white axes with light-on-white tick labels
    (from the panels) — a visibly broken figure, and a regression, since before
    forwarding the tweak was simply a no-op and the figure stayed dark.
    """
    spec = _composite(2).theme("dark")
    getattr(spec, tweak)(*args, **kwargs)
    dark = spec.resolved_theme
    for panel in spec.panels:
        assert panel.resolved_theme.background == dark.background
        assert panel.resolved_theme.foreground == dark.foreground


def test_a_forwarded_theme_tweak_is_inert_when_the_composite_pins_no_theme() -> None:
    """The seeding only fires for a themed composite — otherwise nothing changes.

    A themeless composite's panels already resolve to the active global default,
    which is exactly what the tweak would have picked up on its own.
    """
    spec = _composite(2)
    spec.palette(("#ff9900", "#00ccff"))
    assert all(p.resolved_theme.palette[0] == "#ff9900" for p in spec.panels)
    assert all(p.resolved_theme.background == get_theme().background for p in spec.panels)


# ---------------------------------------------------------------------------
# 5 — F6: save() never returns a path it did not write
# ---------------------------------------------------------------------------


def test_save_rejects_an_unsupported_extension(tmp_path) -> None:
    """An extension the library cannot write raises instead of half-succeeding."""
    pytest.importorskip("matplotlib")
    target = tmp_path / "figure.xyz"
    with pytest.raises(InvalidParameterError, match="no installed backend writes"):
        _panel().save(str(target))
    assert not target.exists()


def test_save_rejects_an_animated_composite_to_html(tmp_path) -> None:
    """plotly's animation export is single-panel; the decline must be loud.

    This is the exact reported defect: it used to return the path and write
    nothing at all.
    """
    pytest.importorskip("plotly")
    target = tmp_path / "movie.html"
    spec = _composite(2).animate(n_frames=4)
    with pytest.raises(InvalidParameterError, match="cannot animate a composite"):
        spec.save(str(target))
    assert not target.exists()


@pytest.mark.parametrize(
    ("name", "backend", "match"),
    [
        ("f.html", "matplotlib", "does not write .html"),
        ("f.mp4", "plotly", "movie format and this Plot is not animated"),
        ("f.json", "matplotlib", "does not write .json"),
    ],
)
def test_save_rejects_an_impossible_extension_backend_pair(tmp_path, name, backend, match) -> None:
    """Every ``(extension, backend)`` pair with no writer raises, naming a way out."""
    pytest.importorskip("matplotlib")
    target = tmp_path / name
    with pytest.raises(InvalidParameterError, match=match):
        _panel().save(str(target), backend=backend)
    assert not target.exists()


@pytest.mark.parametrize("backend", ["json", "threejs"])
@pytest.mark.parametrize("ext", [".png", ".pdf", ".svg"])
def test_save_refuses_an_image_extension_for_a_data_export_backend(
    tmp_path, ext: str, backend: str
) -> None:
    """A serializing backend must not write its payload under an *image* extension.

    The mirror of ``save("x.html", backend="threejs")`` — which wrote a JSON
    document a browser cannot open — one extension over: ``json`` / ``threejs``
    write, they do not draw, so handed ``figure.png`` they used to dump their
    payload into it verbatim and report success.  A file whose bytes are JSON and
    whose name says PNG is the silent wrong-format write this contract exists to
    forbid, so the pair is refused up front and nothing is left behind.
    """
    pytest.importorskip("matplotlib")
    target = tmp_path / f"figure{ext}"
    with pytest.raises(InvalidParameterError, match="does not draw"):
        _panel().save(str(target), backend=backend)
    assert not target.exists()


def test_save_removes_an_image_file_that_turned_out_to_hold_json(tmp_path, monkeypatch) -> None:
    """The backstop: even an *unlisted* backend cannot leave JSON under ``.png``.

    ``_check_save_supported``'s table knows today's serializing backends; this
    guards the one after it.  A backend that writes a JSON payload to an image
    path is caught after the fact by the format sniff, and the misleading artifact
    is removed rather than returned as a successful save.
    """
    pytest.importorskip("matplotlib")
    target = tmp_path / "figure.png"

    def _write_json_instead(self, path, ext, backend, **kw):
        Path(path).write_text('{"geometries": []}', encoding="utf-8")

    monkeypatch.setattr(PlotSpec, "_write", _write_json_instead)
    with pytest.raises(InvalidParameterError, match="is a JSON document"):
        _panel().save(str(target))
    assert not target.exists()


@pytest.mark.parametrize("ext", [".png", ".pdf", ".svg"])
def test_save_writes_the_static_image_it_reports(tmp_path, ext: str) -> None:
    """The positive half: a supported pair really produces a non-empty file."""
    pytest.importorskip("matplotlib")
    target = tmp_path / f"figure{ext}"
    returned = _panel().save(str(target))
    assert returned == str(target)
    assert target.exists() and target.stat().st_size > 0


def test_save_html_does_not_inline_the_plotly_bundle(tmp_path) -> None:
    """A static ``.html`` is a light, CDN-referencing page — not a 4.9 MB bundle.

    ``save`` routes through the renderer's own writer (``render(path=...)``).
    Materialising a figure and calling plotly's default ``write_html`` inlines the
    whole plotly library into every page: 4.90 MB vs 0.05 MB for this spec (98x).
    """
    pytest.importorskip("plotly")
    target = tmp_path / "figure.html"
    _panel().save(str(target))
    assert target.exists()
    size_mb = target.stat().st_size / 1e6
    assert size_mb < 0.5, f"the plotly bundle looks inlined again ({size_mb:.2f} MB)"


def test_save_json_writes_the_plotspec_ir_envelope(tmp_path) -> None:
    """``.json`` (default backend) is the IR envelope — the thing from_dict reads."""
    import json

    target = tmp_path / "spec.json"
    _panel().save(str(target), backend="json")
    payload = json.loads(target.read_text())
    assert payload["spec"]["kind"] == PlotKind.TIME_SERIES.value


# ---------------------------------------------------------------------------
# 6 — F3: __plot_spec__(kind=) is validated against the ROUTING table
# ---------------------------------------------------------------------------


def _trajectory():
    """A bare 3-D Trajectory (no engine, no system needed for the front door)."""
    from tsdynamics.data import Trajectory

    t = np.linspace(0.0, 1.0, 64)
    y = np.column_stack([np.sin(t), np.cos(t), t])
    return Trajectory(t, y, system=None, meta={"dt": float(t[1] - t[0])})


#: Kinds the trajectory front door CAN build (the routing table, incl. recipes).
_BUILDABLE = [
    "time_series",
    "phase_portrait_2d",
    "phase_portrait_3d",
    "spacetime",
    "poincare_section",
    "delay",
    "delay_embedding",
]

#: Semantic kinds it cannot build — all of which it used to accept and mislabel.
_UNBUILDABLE = sorted(
    k.value
    for k in PlotKind
    if k.value not in {*_BUILDABLE, "spatial_field", "field"} and k not in PlotKind.layer_marks()
)


@pytest.mark.parametrize("kind", _BUILDABLE)
def test_plot_spec_accepts_every_buildable_kind(kind: str) -> None:
    """Each route the front door owns really builds a spec (not just passes a check)."""
    extra = {"delay_time": 0.1} if kind.startswith("delay") else {}
    spec = _trajectory().__plot_spec__(kind=kind, **extra)
    assert isinstance(spec, PlotSpec)
    assert spec.layers, f"kind={kind!r} produced a spec with no layers"


@pytest.mark.parametrize("kind", _UNBUILDABLE)
def test_plot_spec_rejects_a_kind_it_cannot_build(kind: str) -> None:
    """A kind this front door cannot build raises, naming the accepted set.

    Failing-first evidence: before v6 **all** of these were accepted and returned
    a spec whose ``.kind`` was the requested one but whose only layer was a plain
    ``LINE`` — a mislabelled plot rendered with that kind's preset.
    ``kind="composite"`` additionally yielded a zero-panel composite, silently
    discarding the trajectory and saving a blank PNG.
    """
    with pytest.raises(InvalidParameterError) as exc:
        _trajectory().__plot_spec__(kind=kind)
    assert "accepted kinds are" in str(exc.value)


@pytest.mark.parametrize("mark", sorted(k.value for k in PlotKind.layer_marks()))
def test_plot_spec_rejects_a_layer_mark_as_a_kind(mark: str) -> None:
    """A layer *mark* is not a semantic kind — ``kind="line"`` must not be accepted."""
    with pytest.raises(InvalidParameterError):
        _trajectory().__plot_spec__(kind=mark)


def test_explicit_poincare_section_builds_a_scatter_not_a_line_portrait() -> None:
    """``kind="poincare_section"`` builds a crossing point cloud, not a line portrait."""
    spec = _trajectory().__plot_spec__(kind="poincare_section")
    assert spec.kind is PlotKind.POINCARE_SECTION
    assert [lyr.kind for lyr in spec.layers] == [PlotKind.SCATTER]


def test_save_html_must_actually_be_an_html_document(tmp_path, monkeypatch) -> None:
    """A backend that writes the wrong *format* under ``.html`` is caught and cleaned up.

    The reported defect: ``save("x.html", backend="threejs")`` wrote a 142 KB JSON
    geometry payload under an ``.html`` name — a file no browser can open, reported
    as a success.  The guard is a content check rather than a per-backend
    declaration, so it keeps holding for whatever page a backend emits next.
    """
    pytest.importorskip("matplotlib")
    target = tmp_path / "payload.html"

    def _fake_write(self, path, *a, **k):  # noqa: ANN001, ARG001
        pathlib_write = open(path, "w", encoding="utf-8")  # noqa: SIM115
        pathlib_write.write('{"schema_version": 2, "kind": "phase_portrait_3d"}')
        pathlib_write.close()

    monkeypatch.setattr(PlotSpec, "_write", _fake_write)
    with pytest.raises(InvalidParameterError, match="not an HTML document"):
        _panel().save(str(target))
    assert not target.exists(), "the misleading artifact must be removed, not left behind"


def test_save_verifies_the_file_even_for_an_unknown_backend(tmp_path, monkeypatch) -> None:
    """A backend that returns without writing is caught by the post-write check.

    The table of impossible pairs can never be complete, so the guarantee rests on
    verification, not enumeration: ``save`` stats the file it is about to report.
    """
    pytest.importorskip("matplotlib")
    target = tmp_path / "figure.png"
    spec = _panel()
    monkeypatch.setattr(PlotSpec, "_write", lambda *a, **k: None)
    with pytest.raises(InvalidParameterError, match="did not write"):
        spec.save(str(target))


# ---------------------------------------------------------------------------
# 6 — v6: one type, one escape hatch, one figure vocabulary
# ---------------------------------------------------------------------------


def test_the_type_a_user_receives_is_called_plot() -> None:
    """``ts.plot`` returns a ``Plot`` — one type, no facade, no wrapper.

    ``PlotSpec`` stays bound as the *same class object* so 557 in-tree
    annotations keep working, but the name a user sees, types and reads in a
    repr is ``Plot``.  If these ever became two classes, "escalating from easy to
    expert is not a type change" would stop being true.
    """
    from tsdynamics.viz.spec import Plot

    assert Plot is PlotSpec
    assert Plot.__name__ == "Plot"
    assert type(_panel()).__name__ == "Plot"
    assert repr(_panel()).startswith("Plot(")


def test_the_repr_names_what_you_hold_and_the_verb_that_shows_it(monkeypatch) -> None:
    """The repr must answer both "what is this?" and "now what?" — in one line.

    The "now what?" half is **backend-aware** since v6 (see
    ``test_the_repr_does_not_offer_show_on_a_backend_that_cannot_display``), so
    the interactive case is pinned explicitly rather than inherited from whatever
    backend the session happens to be on.
    """
    import tsdynamics.viz.spec as spec_mod

    monkeypatch.setattr(spec_mod, "_mpl_backend_is_interactive", lambda: True)
    text = repr(_panel())
    assert "time_series" in text and "1 layer" in text
    assert ".show()" in text and ".save('f.png')" in text

    # Extension-aware: a .png of a movie is a still, so an animated plot must not
    # send the reader to the wrong verb.
    animated = repr(_panel().animate(fps=30))
    assert "animated 30 fps" in animated
    assert ".save('f.gif')" in animated and ".save('f.png')" not in animated

    # A composite names its arrangement rather than counting layers it has none of.
    grid = repr(
        PlotSpec(kind=PlotKind.COMPOSITE, panels=[_panel(), _panel()], layout=Layout(mode="row"))
    )
    assert "2 panels in a 1x2 row" in grid


def test_the_repr_names_the_producing_transforms_of_an_overlay() -> None:
    """An overlay reads as its sources, not as "3 layers"."""
    t = np.linspace(0.0, 1.0, 8)
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[
            Layer(PlotKind.IMAGE, {"x": t, "y": t, "z": np.outer(t, t)}, transform="flow_speed"),
            Layer(PlotKind.LINE, {"x": t, "y": t}, transform="streamlines"),
            Layer(PlotKind.LINE, {"x": t, "y": -t}, transform="nullclines"),
        ],
    )
    assert "3 layers: flow_speed, streamlines, nullclines" in repr(spec)


def test_fig_ax_axes_hand_you_matplotlib_and_the_plot_still_works() -> None:
    """The escape hatch: expert tier is one dot away and is not a type change."""
    plt = pytest.importorskip("matplotlib.pyplot")
    spec = _panel()
    fig = spec.fig
    assert fig.__class__.__name__ == "Figure"
    assert spec.fig is fig, "the figure is rendered once and cached"
    assert spec.ax is fig.axes[0]
    assert spec.axes == list(fig.axes)
    # ...and the library verbs still work on the very same object (the warning is
    # the "library tweaks first, matplotlib last" rule, gated separately below).
    with pytest.warns(Warning):
        assert spec.relabel(x="t") is spec
    plt.close("all")


def test_ax_on_a_composite_names_axes_instead_of_guessing() -> None:
    """A multi-panel figure has no single axes; the error says which name to use."""
    plt = pytest.importorskip("matplotlib.pyplot")
    spec = _composite(4)
    with pytest.raises(InvalidParameterError, match=r"4 panels; use \.axes"):
        _ = spec.ax
    assert len(spec.axes) == 4
    plt.close("all")


#: One benign call per mutating method — enough to prove the cache is dropped.
_MUTATOR_CALLS: dict[str, typing.Callable[[PlotSpec], object]] = {
    "add": lambda s: s.add(_panel("b")),
    "relabel": lambda s: s.relabel(x="t"),
    "rescale": lambda s: s.rescale(y="log"),
    "limits": lambda s: s.limits(x=(0.0, 1.0)),
    "ticks": lambda s: s.ticks(x=[0.0, 1.0]),
    "style": lambda s: s.style(color="red"),
    "recolor": lambda s: s.recolor("red"),
    "theme": lambda s: s.theme("dark"),
    "palette": lambda s: s.palette(["#123456"]),
    "gridlines": lambda s: s.gridlines(True),
    "font": lambda s: s.font(size=11.0),
    "background": lambda s: s.background("#101010"),
    "size": lambda s: s.size(width=4.0),
    "colorize": lambda s: s.colorize(legend=True),
    "autocolor": lambda s: s.autocolor(),
    "animate": lambda s: s.animate(fps=10),
    "trail": lambda s: s.trail(length=None),
    "head": lambda s: s.head(size=4.0),
    "camera": lambda s: s.camera(elev=10.0),
    "clock": lambda s: s.clock(True),
    "tweak": lambda s: s.tweak(title="x"),
    "vline": lambda s: s.vline(0.5),
    "hline": lambda s: s.hline(0.5),
    "span": lambda s: s.span(0.1, 0.2),
    "text": lambda s: s.text(0.1, 0.2, "hi"),
}


def test_every_mutating_plot_method_drops_the_figure_cache() -> None:
    """A cached figure that survives a tweak is a silent-wrong-answer generator.

    Structural, not example-based: the set of methods that must invalidate is
    **derived from the source** (every public method whose return annotation is
    ``Plot`` / ``PlotSpec`` / ``Self``), so adding a tweak that forgets to
    invalidate fails here rather than shipping a figure that disagrees with the
    object it came from.  ``add`` is annotated ``-> Self``, which is exactly why
    the derivation cannot filter on the class name alone.
    """
    derived = set()
    for name, attr in vars(PlotSpec).items():
        if name.startswith("_") or not callable(attr):
            continue
        try:
            ret = inspect.signature(attr).return_annotation
        except (TypeError, ValueError):  # pragma: no cover - builtins / slots
            continue
        text = ret.strip() if isinstance(ret, str) else getattr(ret, "__name__", "")
        if text in {"Plot", "PlotSpec", "Self"}:
            derived.add(name)
    assert "add" in derived, "the derivation must not miss the -> Self method"
    assert derived <= set(_MUTATOR_CALLS), (
        f"untested mutators: {sorted(derived - set(_MUTATOR_CALLS))} — add a call "
        "to _MUTATOR_CALLS so this gate can prove each drops the figure cache."
    )
    for name in sorted(derived):
        spec = _panel()
        spec._figure_cache = ("sentinel-figure", ["sentinel-axes"])  # type: ignore[assignment]
        _MUTATOR_CALLS[name](spec)
        assert spec._figure_cache is None, f"{name}() left a stale figure cached"


def test_mutating_after_handing_out_the_figure_warns_once() -> None:
    """Never silently lose your work: library tweaks first, matplotlib last."""
    plt = pytest.importorskip("matplotlib.pyplot")
    from tsdynamics.viz.render.caps import VisualizationDegraded

    spec = _panel()
    spec.ax.set_title("my hand edit")
    with pytest.warns(VisualizationDegraded, match="library tweaks first"):
        spec.style(color="red")
    # Exactly once: a second round of hand edit + tweak is a fresh situation, but
    # a second tweak on the same handed-out figure must not nag.
    assert spec.fig is not None
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        spec.style(color="blue")
    assert not [c for c in caught if issubclass(c.category, VisualizationDegraded)]
    plt.close("all")


def test_selecting_a_panel_returns_a_plot_so_it_chains() -> None:
    """``p[i]`` / ``p["name"]`` select; ``p.panels`` stays the list."""
    spec = _composite(3)
    spec.panels[1].title = "psd"
    assert spec[0] is spec.panels[0]
    assert spec["psd"] is spec.panels[1]
    assert spec[1].rescale(x="log") is spec.panels[1]
    # On a single-panel plot, p[0] is p — so grid code works on one panel too.
    panel = _panel()
    assert panel[0] is panel
    with pytest.raises(InvalidParameterError, match="no panel named"):
        _ = spec["nope"]
    # An out-of-range INT is a LookupError, which is what makes the sequence
    # protocol below work; a missing NAME stays a value error.
    with pytest.raises(IndexError, match="does not exist"):
        _ = spec[9]


def test_a_plot_is_a_complete_sequence_over_its_panels() -> None:
    """``len`` / ``iter`` / ``[]`` must agree — they did not, and iteration crashed.

    Measured before v6: ``len(p)`` was a ``TypeError``, ``hasattr(Plot,
    "__iter__")`` was ``False``, and ``list(p)`` yielded both panels and then
    raised ``InvalidParameterError`` — because Python's ``__getitem__``
    iteration fallback stops only on ``IndexError`` and this one raised a
    ``ValueError`` subclass.  So a ``Plot`` looked like a sequence, was indexable
    like a sequence, and could not be iterated.
    """
    spec = _composite(3)
    assert len(spec) == 3
    assert list(spec) == list(spec.panels)
    assert [p for p in spec] == [spec[i] for i in range(len(spec))]
    panel = _panel()
    assert len(panel) == 1 and list(panel) == [panel]


def test_style_by_name_addresses_one_source_inside_an_overlay() -> None:
    """``p.style("nullclines", ...)`` — the provenance stamp becomes usable."""
    t = np.linspace(0.0, 1.0, 8)
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[
            Layer(PlotKind.LINE, {"x": t, "y": t}, label="orbit", transform="streamlines"),
            Layer(PlotKind.LINE, {"x": t, "y": -t}, label="null", transform="nullclines"),
        ],
    )
    spec.style("nullclines", color="white", linewidth=1.2)
    assert spec.layers[1].style == {"color": "white", "linewidth": 1.2}
    assert spec.layers[0].style == {}, "a named restyle must not touch the others"
    spec.style("orbit", alpha=0.5)  # by legend label, too
    assert spec.layers[0].style == {"alpha": 0.5}
    with pytest.raises(InvalidParameterError, match="matched no layer"):
        spec.style("flow_speed", color="red")


def test_annotation_verbs_take_plain_python_and_survive_a_render() -> None:
    """``Annotation`` was exported because a signature demanded it (a C1 bug)."""
    plt = pytest.importorskip("matplotlib.pyplot")
    spec = _panel()
    returned = (
        spec.vline([0.25, 0.75], label="onsets", ls="--", color="crimson")
        .hline(0.0)
        .span(0.1, 0.2, alpha=0.2)
        .text(0.5, 0.5, "here")
    )
    assert returned is spec
    kinds = [a.kind for a in spec.annotations]
    assert kinds == ["vline", "vline", "hline", "span", "text"]
    # ``ls="--"`` went through the same normalizer ``.style()`` uses.
    assert spec.annotations[0].style == {"linestyle": "dashed", "color": "crimson"}
    spec.render("matplotlib")
    plt.close("all")


def test_a_plain_dict_annotation_can_no_longer_reach_a_renderer_raw() -> None:
    """The measured ``AttributeError: 'dict' object has no attribute 'style'``."""
    from tsdynamics.viz.spec import Annotation

    coerced = Annotation.from_mapping({"kind": "vline", "x": 1.0})
    assert isinstance(coerced, Annotation) and coerced.x == 1.0
    assert Annotation.from_mapping(coerced) is coerced


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ("t={:.1f}", "positional field"),
        ("{time}", "names 'time'"),
        ("no fields", "names no field"),
    ],
)
def test_clock_format_is_validated_at_the_call_not_at_save(bad: str, match: str) -> None:
    """A typo used to surface as a raw IndexError hundreds of frames later."""
    with pytest.raises(InvalidParameterError, match=match):
        _panel().clock(fmt=bad)


@pytest.mark.parametrize(("given", "want"), [("{}", "{t}"), ("t = {t:.1f}", "t = {t:.1f}")])
def test_clock_format_normalises_the_bare_field(given: str, want: str) -> None:
    """A bare ``{}`` means the time, and is rewritten so the callback can format it."""
    spec = _panel().clock(fmt=given)
    assert spec.animation is not None
    assert spec.animation.clock_format == want


def test_grid_names_gridlines_and_says_so() -> None:
    """One attribute cannot mean two things; the error is the migration guide."""
    with pytest.raises(AttributeError, match="gridlines"):
        _panel().grid(True)


def test_the_figure_vocabulary_is_derived_and_has_seventeen_names() -> None:
    """One definition of what a *figure* keyword is, shared by every door.

    Before v6 the front door carried a five-name copy of this set, so 12 of these
    17 raised at ``ts.plot(...)`` while all 17 worked at ``traj.plot(...)``.
    """
    from tsdynamics.viz.spec import FIGURE_KEYS, apply_figure_keywords, split_figure_keywords

    assert len(FIGURE_KEYS) == 17
    assert {"xscale", "xlim", "xticks", "clim", "colorbar", "legend", "theme"} <= FIGURE_KEYS

    kw = {
        "xlim": (0.0, 1.0),
        "yscale": "log",
        "title": "T",
        "legend": False,
        "theme": "dark",
        "final_time": 10.0,
    }
    figure = split_figure_keywords(kw)
    assert kw == {"final_time": 10.0}, "only figure keywords are peeled"

    spec = apply_figure_keywords(_panel(), figure)
    assert spec.x.limits == (0.0, 1.0)
    assert spec.y.scale == "log"
    assert spec.title == "T"
    assert spec.legend is None, "legend=False drops the legend"
    assert spec._theme is not None and spec._theme.name == "dark"


@pytest.mark.parametrize(
    ("name", "backend"),
    [
        ("f.png", "matplotlib"),
        ("f.pdf", "matplotlib"),
        ("f.svg", "matplotlib"),
        ("f.html", "plotly"),
        ("f.json", "json"),
    ],
)
def test_save_picks_the_backend_from_the_extension(name: str, backend: str) -> None:
    """``.png`` / ``.html`` / ``.json`` each route to the backend that writes them.

    The skip has to be about the EXPECTED backend, not about whether *some*
    backend answered.  ``.html`` is written by plotly AND by three.js, so with
    plotly absent the resolver correctly returns ``"threejs"`` — not ``None`` —
    and a "nothing was chosen" guard never fires.  Measured on the base CI job,
    that read as a routing bug when it was an uninstalled extra.
    """
    if backend in {"plotly", "matplotlib"}:
        pytest.importorskip(backend)
    chosen = _panel()._preferred_save_backend(name)
    if chosen is None:  # that backend is not installed in this environment
        pytest.skip(f"{backend} not installed")
    assert chosen == backend


def test_save_picks_a_movie_writer_only_for_an_animated_plot() -> None:
    """``.mp4`` / ``.gif`` route to matplotlib — and only when there is a movie."""
    pytest.importorskip("matplotlib")
    movie = _panel().animate(fps=10)
    assert movie._preferred_save_backend("f.mp4") == "matplotlib"
    assert movie._preferred_save_backend("f.gif") == "matplotlib"
    # A still of a movie is its final frame, so an image extension still resolves.
    assert movie._preferred_save_backend("f.png") == "matplotlib"
    # ...but a still plot has no movie writer at all, and says so.
    with pytest.raises(InvalidParameterError, match="movie format and this Plot is not animated"):
        _panel()._check_save_supported(".mp4", None)


def test_show_displays_and_returns_none_and_never_stays_silent() -> None:
    """A verb whose whole job is a side effect returns ``None``, like every ``show``.

    Measured before v6: ``.show()`` returned a ``Figure`` and, on a
    non-interactive backend, displayed nothing and said nothing — while the repr
    of **every** Plot pointed the reader at it.  In a notebook the returned
    figure was echoed, so the call looked like it worked; in a script it was a
    silent no-op.  Now: ``None``, plus one warning naming the backend and
    ``.save``.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    import matplotlib

    from tsdynamics.viz.render.caps import VisualizationDegraded

    previous = matplotlib.get_backend()
    matplotlib.use("Agg")
    try:
        with pytest.warns(VisualizationDegraded, match=r"has no window"):
            assert _panel().show() is None
    finally:
        matplotlib.use(previous)
        plt.close("all")


def test_the_repr_does_not_offer_show_on_a_backend_that_cannot_display(monkeypatch) -> None:
    """The first thing every user reads must not point at a verb that no-ops here.

    Backend-aware both ways: on a windowless backend the repr names ``.save`` and
    says why, and on one that can open a window it keeps the ``.show()`` hint.
    """
    pytest.importorskip("matplotlib.pyplot")
    import matplotlib

    import tsdynamics.viz.spec as spec_mod

    previous = matplotlib.get_backend()
    matplotlib.use("Agg")
    try:
        text = repr(_panel())
        assert ".save('f.png')" in text
        assert ".show()" not in text, text
        assert "has no window" in text, text
        monkeypatch.setattr(spec_mod, "_mpl_backend_is_interactive", lambda: True)
        assert ".show() to display" in repr(_panel())
    finally:
        matplotlib.use(previous)


def test_a_plot_without_a_backend_still_says_what_it_is_in_a_notebook() -> None:
    """``_repr_html_`` is the no-backend fallback; with a backend it stands aside."""
    import tsdynamics.viz.spec as spec_mod

    spec = _panel().relabel(x="t", y="x(t)", title="run 4")
    assert spec._repr_html_() is None, "a real backend draws instead"

    saved = spec_mod._resolve_renderers
    spec_mod._resolve_renderers = lambda: None  # type: ignore[assignment]
    try:
        html = spec._repr_html_()
    finally:
        spec_mod._resolve_renderers = saved  # type: ignore[assignment]
    assert html is not None
    assert "run 4" in html and "time_series" in html and "no drawing backend" in html


class TestShowActuallyDisplays:
    """``.show()`` opens a window, and it opens the window on YOUR figure.

    Two defects, both silent, both on the verb this library's own ``__repr__``
    tells every user to call.
    """

    def test_show_hands_the_figure_to_pyplot(self, monkeypatch):
        """A bare ``Figure`` has no manager, and ``plt.show()`` ignores it.

        The renderer builds ``matplotlib.figure.Figure(...)`` directly — right
        for a library, since pyplot's registry is global process state — so the
        figure carried no manager and ``Gcf.get_all_fig_managers()`` was empty.
        ``plt.show()`` displays the *managed* figures, of which there were none,
        so it returned instantly having drawn nothing.  And because the backend
        was genuinely interactive, the windowless warning did not fire either.
        """
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from matplotlib._pylab_helpers import Gcf

        monkeypatch.setattr("tsdynamics.viz.spec._mpl_backend_is_interactive", lambda: True)
        called: list[bool] = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: called.append(True))

        plt.close("all")
        plot = ts.plot(ts.systems.Lorenz().run(final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0]))
        assert Gcf.get_all_fig_managers() == [], "nothing managed before .show()"

        assert plot.show() is None, "show() is a side effect; it returns nothing"
        assert called, "plt.show() must be reached"
        managed = [m.canvas.figure for m in Gcf.get_all_fig_managers()]
        assert plot.fig in managed, ".show() must give the figure a pyplot manager"
        plt.close("all")

    def test_show_displays_the_figure_you_edited(self, monkeypatch):
        """``p.ax`` edits must survive ``p.show()``.

        ``show`` went through ``render()``, which draws afresh every call, so it
        displayed a figure *without* the hand edit while ``p.fig`` held one with
        it — the escape hatch and this verb disagreeing about which figure is
        yours, in silence.
        """
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from matplotlib._pylab_helpers import Gcf

        monkeypatch.setattr("tsdynamics.viz.spec._mpl_backend_is_interactive", lambda: True)
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)

        plt.close("all")
        plot = ts.plot(ts.systems.Lorenz().run(final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0]))
        plot.ax.set_facecolor("#112233")
        plot.show()

        shown = Gcf.get_all_fig_managers()[-1].canvas.figure
        assert shown is plot.fig, "show() must display the cached figure, not a re-render"
        assert shown.axes[0].get_facecolor() == pytest.approx(
            (0.06666666666666667, 0.13333333333333333, 0.2, 1.0)
        ), "the hand edit must be on the displayed figure"
        plt.close("all")

    def test_show_on_a_windowless_backend_still_says_so(self):
        """The headless path is unchanged: one warning naming the backend."""
        pytest.importorskip("matplotlib")
        import matplotlib
        import matplotlib.pyplot as plt

        if matplotlib.get_backend().lower() != "agg":  # pragma: no cover
            pytest.skip("this assertion is about the non-interactive path")
        plot = ts.plot(ts.systems.Lorenz().run(final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0]))
        with pytest.warns(VisualizationDegraded, match="has no window"):
            plot.show()
        plt.close("all")
