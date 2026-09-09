"""Gate for F5 — plotly must honor ``path=`` / ``html=`` on **3-D** specs too.

The 3-D branch of ``plotly._core.render`` used to ``return`` the Figure straight
away, so ``path=`` wrote no file and ``html=True`` returned a Figure where the
caller was promised a ``str``.  3-D is 106 of the 136 catalogue ODEs, so the
majority of this library's plots could not be exported through the plotly
renderer's own API — silently: the call succeeded and returned *something*.

Also pins the write-capability declarations (:meth:`RendererCapabilities.can_save`)
that ``PlotSpec.save`` resolves ``(extension, backend)`` against, so a backend
cannot be handed an extension it does not write.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsdynamics.viz.spec import Layer, PlotKind, PlotSpec

pytest.importorskip("plotly")


def _spec_2d() -> PlotSpec:
    t = np.linspace(0.0, 1.0, 64)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[Layer(kind=PlotKind.LINE, data={"x": t, "y": np.sin(t)})],
    )


def _spec_3d() -> PlotSpec:
    t = np.linspace(0.0, 6.0, 64)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        ndim=3,
        layers=[
            Layer(kind=PlotKind.LINE3D, data={"x": np.sin(t), "y": np.cos(t), "z": t}),
        ],
    )


@pytest.mark.parametrize("factory", [_spec_2d, _spec_3d], ids=["2d", "3d"])
def test_path_writes_a_file_and_returns_a_path(factory, tmp_path: Path) -> None:  # noqa: ANN001
    out = tmp_path / "fig.html"
    result = factory().render(backend="plotly", path=out)
    assert out.exists(), "render(path=) returned without writing the file"
    assert isinstance(result, Path)
    assert out.stat().st_size > 1024
    assert out.read_text(encoding="utf-8").lstrip().lower().startswith("<html")


@pytest.mark.parametrize("factory", [_spec_2d, _spec_3d], ids=["2d", "3d"])
def test_html_true_returns_a_string(factory) -> None:  # noqa: ANN001
    result = factory().render(backend="plotly", html=True)
    assert isinstance(result, str), f"html=True returned {type(result).__name__}"
    assert "<div" in result


def test_exported_html_references_the_cdn_and_does_not_inline_the_bundle(tmp_path: Path) -> None:
    """A ~5 MB page for a 64-point curve is a bug, not a feature (F7's sibling)."""
    out = tmp_path / "fig.html"
    _spec_3d().render(backend="plotly", path=out)
    text = out.read_text(encoding="utf-8")
    assert "cdn.plot.ly" in text
    assert out.stat().st_size < 500_000


# ---------------------------------------------------------------------------
# The (extension, backend) write contract
# ---------------------------------------------------------------------------


def _caps(name: str):  # noqa: ANN202 - test helper
    from tsdynamics import registry
    from tsdynamics.viz.render import register_builtin_renderers

    register_builtin_renderers()
    return registry.renderers.get(name).capabilities


@pytest.mark.parametrize(
    ("backend", "ext", "expected"),
    [
        ("matplotlib", ".png", True),
        ("matplotlib", ".pdf", True),
        ("matplotlib", ".mp4", True),
        ("matplotlib", ".gif", True),
        ("matplotlib", ".html", False),
        ("plotly", ".html", True),
        ("plotly", "html", True),  # extension normalisation
        ("plotly", ".HTML", True),  # case-insensitive
        ("plotly", ".png", False),  # would need kaleido
        ("plotly", ".mp4", False),  # plotly cannot encode a video container
        ("json", ".json", True),
        ("json", ".png", False),
    ],
)
def test_declared_write_capability(backend: str, ext: str, expected: bool) -> None:
    pytest.importorskip("matplotlib")
    assert _caps(backend).can_save(ext) is expected


def test_plotly_declines_an_animated_composite() -> None:
    """Its animation core is single-panel, so an animated composite must fall back to mpl.

    Swallowing this decline is how ``save('*.html')`` on an animated composite
    came to return a path it had never written.
    """
    pytest.importorskip("matplotlib")
    from tsdynamics.viz.spec import Animation, Layout

    panel = _spec_2d()
    composite = PlotSpec(
        kind=PlotKind.COMPOSITE,
        panels=[panel, _spec_2d()],
        layout=Layout(mode="stack"),
        animation=Animation(),
    )
    assert _caps("plotly").can_render_spec(composite) is False
