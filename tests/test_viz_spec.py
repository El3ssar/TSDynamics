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

from tsdynamics.errors import InvalidParameterError
from tsdynamics.viz._tweaks import tweak_scopes
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
        if (isinstance(ret, str) and ret.strip() == "PlotSpec") or ret is PlotSpec:
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
        "grid",
        "limits",
        "palette",
        "recolor",
        "relabel",
        "rescale",
        "style",
        "ticks",
    }
    expected_figure = {
        "animate",
        "background",
        "clock",
        "head",
        # ``plot`` forwards to the individual tweaks, which recurse themselves.
        "plot",
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
    "grid": (lambda s: s.grid(True), False),
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
        ("grid", (True,), {"color": "#888888"}),
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
    with pytest.raises(InvalidParameterError, match="unsupported format"):
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
        ("f.mp4", "plotly", "cannot write .mp4"),
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
# 6 — F3: to_plot_spec(kind=) is validated against the ROUTING table
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
def test_to_plot_spec_accepts_every_buildable_kind(kind: str) -> None:
    """Each route the front door owns really builds a spec (not just passes a check)."""
    extra = {"delay_time": 0.1} if kind.startswith("delay") else {}
    spec = _trajectory().to_plot_spec(kind=kind, **extra)
    assert isinstance(spec, PlotSpec)
    assert spec.layers, f"kind={kind!r} produced a spec with no layers"


@pytest.mark.parametrize("kind", _UNBUILDABLE)
def test_to_plot_spec_rejects_a_kind_it_cannot_build(kind: str) -> None:
    """A kind this front door cannot build raises, naming the accepted set.

    Failing-first evidence: before v6 **all** of these were accepted and returned
    a spec whose ``.kind`` was the requested one but whose only layer was a plain
    ``LINE`` — a mislabelled plot rendered with that kind's preset.
    ``kind="composite"`` additionally yielded a zero-panel composite, silently
    discarding the trajectory and saving a blank PNG.
    """
    with pytest.raises(InvalidParameterError) as exc:
        _trajectory().to_plot_spec(kind=kind)
    assert "accepted kinds are" in str(exc.value)


@pytest.mark.parametrize("mark", sorted(k.value for k in PlotKind.layer_marks()))
def test_to_plot_spec_rejects_a_layer_mark_as_a_kind(mark: str) -> None:
    """A layer *mark* is not a semantic kind — ``kind="line"`` must not be accepted."""
    with pytest.raises(InvalidParameterError):
        _trajectory().to_plot_spec(kind=mark)


def test_explicit_poincare_section_builds_a_scatter_not_a_line_portrait() -> None:
    """``kind="poincare_section"`` builds a crossing point cloud, not a line portrait."""
    spec = _trajectory().to_plot_spec(kind="poincare_section")
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
