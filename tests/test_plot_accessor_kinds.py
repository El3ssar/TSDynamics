"""Gate: every ``_PlotAccessor`` typed kind method names a valid ``PlotKind``.

Stream VIZ-FALLBACK-GATE (issue #274).  The visualization seam
:class:`tsdynamics.analysis._result._PlotAccessor` exposes typed convenience
methods — ``result.plot.scaling()``, ``.phase()``, ``.image()``, … — each of
which forces a particular *semantic* plot kind by passing a ``kind=`` string into
``to_plot_spec``.  That string has to be a member of the closed
:class:`tsdynamics.viz.spec.PlotKind` vocabulary, or the very first thing a real
renderer does (``PlotKind(kind)``) raises ``ValueError``.

This module is the genuine gate: it drives **each** typed method through the seam
with a fake renderer installed and captures the ``kind`` that reaches
``to_plot_spec``, then asserts that ``PlotKind(kind)`` resolves.  A method that
ships an invalid spelling (the historical ``"phase_portrait"`` bug) fails here.
It is engine-free — it never imports ``tsdynamics._rust`` and constructs no
system — so it stays in the fast tier.
"""

from __future__ import annotations

import inspect

import pytest

from tsdynamics.analysis._result import (
    AnalysisResult,
    ScalarResult,
    _PlotAccessor,
)
from tsdynamics.registry import renderers
from tsdynamics.viz.spec import PlotKind, PlotSpec


@pytest.fixture
def fake_renderer():
    """Install a no-op renderer as the *only* renderer for one test, then restore.

    The seam only attempts to render once ``registry.renderers`` is non-empty, so
    this fixture is what makes ``result.plot.<kind>()`` actually walk the
    ``_render`` path (and thus reach ``to_plot_spec(kind=...)``) instead of
    short-circuiting on ``VisualizationNotInstalled``.  It returns the spec it was
    handed so the test can inspect it.

    It snapshots and clears the global ``renderers`` registry so the fake is the
    sole backend dispatch can pick — otherwise a real renderer registered by an
    earlier test (e.g. matplotlib auto-registering on first render) would be
    selected instead and the fake would capture nothing, making this gate
    order-dependent.  The prior registry is restored verbatim on teardown.
    """
    captured: list[PlotSpec] = []

    def _render(spec: PlotSpec, **_kw: object) -> PlotSpec:
        captured.append(spec)
        return spec

    saved = renderers.all()
    renderers.clear()
    renderers.register("_fake_kinds_renderer", _render, replace=True)
    try:
        yield captured
    finally:
        renderers.clear()
        for entry in saved:
            renderers.register(entry.name, entry.obj, replace=True, **dict(entry.metadata))


class _KindCapturingResult(AnalysisResult):
    """A result whose ``to_plot_spec`` records the requested ``kind`` and validates it.

    Returns a real :class:`PlotSpec` (so the fake renderer is satisfied) and
    appends the resolved :class:`PlotKind` to :attr:`seen` — letting the test read
    back exactly which kind each typed accessor method forced.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "meta", {})
        object.__setattr__(self, "seen", [])

    def to_plot_spec(self, kind: str | None = None) -> PlotSpec:  # noqa: D102
        resolved = PlotKind(kind) if kind is not None else PlotKind.DIAGNOSTIC_CURVE
        self.seen.append(resolved)
        return PlotSpec(kind=resolved)


def _typed_kind_methods() -> list[str]:
    """Names of the typed ``_PlotAccessor`` kind methods (e.g. ``scaling``, ``phase``).

    The public, non-dunder methods other than ``__call__`` that force a specific
    kind: every one is supposed to route a valid ``PlotKind`` into ``to_plot_spec``.
    """
    skip = {"_render"}
    names = [
        n
        for n, _ in inspect.getmembers(_PlotAccessor, predicate=inspect.isfunction)
        if not n.startswith("__") and n not in skip
    ]
    return sorted(names)


_TYPED_METHODS = _typed_kind_methods()


def test_there_are_typed_kind_methods() -> None:
    """Sanity: the accessor exposes a non-trivial set of typed kind methods."""
    # If this drops to zero the parametrized gate below would vacuously pass, so
    # pin a floor (scaling/diagnostic/time_series/phase/image/bifurcation/…).
    assert len(_TYPED_METHODS) >= 8, _TYPED_METHODS


@pytest.mark.parametrize("method_name", _TYPED_METHODS)
def test_typed_kind_method_uses_valid_plotkind(method_name, fake_renderer) -> None:
    """Each typed ``.plot.<method>()`` forces a kind that is a real ``PlotKind``.

    Genuine gate: the kind string lives inside the method body, so we drive the
    method end-to-end (through ``_render`` → ``to_plot_spec(kind=...)``) and assert
    the captured kind resolves.  An invalid spelling raises ``ValueError`` from
    ``PlotKind(kind)`` here — the test fails rather than passing on a tautology.
    """
    result = _KindCapturingResult()
    method = getattr(result.plot, method_name)
    method()  # forces kind=<this method's kind> into to_plot_spec
    assert result.seen, f".plot.{method_name}() never reached to_plot_spec()"
    forced = result.seen[-1]
    assert isinstance(forced, PlotKind)
    # And the fake renderer was actually handed a spec carrying that kind.
    assert fake_renderer and fake_renderer[-1].kind == forced


def test_default_plot_call_resolves(fake_renderer) -> None:
    """``result.plot()`` (no typed kind) also resolves to a valid spec via the seam."""
    result = _KindCapturingResult()
    spec = result.plot()
    assert isinstance(spec, PlotSpec)
    assert isinstance(spec.kind, PlotKind)


def test_invalid_kind_would_fail(fake_renderer) -> None:
    """Guard against a tautology: an *invalid* kind genuinely raises through the seam.

    Proves the gate above has teeth — if a typed method ever passed a bogus kind
    like ``"phase_portrait"``, ``PlotKind(kind)`` rejects it exactly as here.
    """
    bad = ScalarResult(value=1.0)
    with pytest.raises(ValueError):
        # mimic what a broken typed method would do inside the seam
        bad.plot(kind="phase_portrait")  # not a PlotKind member
