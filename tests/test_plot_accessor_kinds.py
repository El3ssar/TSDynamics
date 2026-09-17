"""Gate: ``result.plot`` is a namespace of TRANSFORMS, never of forced kinds.

Originally (stream VIZ-FALLBACK-GATE, issue #274) this file proved that each of
the eight typed methods on
:class:`tsdynamics.analysis._result._PlotAccessor` — ``result.plot.scaling()``,
``.phase()``, ``.image()``, … — passed a *valid* ``PlotKind`` string into
``__plot_spec__``.  It did, and that was the problem: a valid kind is not a
correct picture.  Each method **relabelled** the result's own spec and changed
nothing else, so ``lyapunov_spectrum(lor).plot.section()`` handed back a bar
chart of exponents carrying the kind ``poincare_section``.

Measured across the 30 result fixtures before the removal: **not one of the
eight changed a single byte of layer data**, and four of them
(``.time_series()`` / ``.image()`` / ``.bifurcation()`` / ``.section()``) named
the result's own natural kind for **zero** results — they could only ever
mislabel.  The module that defined them had already deleted ``.histogram()`` and
``.spectrum()`` on exactly that reasoning.

So the gate inverted.  What it pins now:

1. the accessor exposes **no** kind-forcing method, and cannot grow one;
2. every name it *does* offer is a registered transform that admits this result,
   so ``result.plot.<TAB>`` means what ``traj.plot.<TAB>`` means;
3. a transform genuinely **computes** where a relabel only renamed;
4. a retired name is answered by name, with a line that runs;
5. ``kind=`` at this door is refused, naming the transform spelling.

It is engine-free — it never imports ``tsdynamics._rust`` and constructs no
system — so it stays in the fast tier.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tsdynamics.analysis._result import (
    AnalysisResult,
    ScalarResult,
    _PlotAccessor,
)
from tsdynamics.analysis._result_viz import _RETIRED_KIND_METHODS
from tsdynamics.errors import InvalidParameterError
from tsdynamics.registry import renderers
from tsdynamics.viz.spec import PlotKind, PlotSpec


@pytest.fixture
def fake_renderer():
    """Install a no-op renderer as the *only* renderer for one test, then restore.

    The seam only attempts to render once ``registry.renderers`` is non-empty, so
    this fixture is what makes ``result.plot(...)`` actually walk the ``_render``
    path instead of short-circuiting on ``VisualizationNotInstalled``.  It returns
    the spec it was handed so the test can inspect it.

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

    name = "_fake_kinds_renderer"
    saved = renderers.all()
    renderers.clear()
    renderers.register(name, _render, replace=True)
    try:
        # Yield (name, captured): matplotlib is the deterministic default backend,
        # so a no-arg render would route to mpl (and return a Figure); the seam
        # reaches a custom backend only *by name*, so callers must pass
        # ``backend=name`` to route a render through this fake and capture the spec.
        yield name, captured
    finally:
        renderers.clear()
        for entry in saved:
            renderers.register(entry.name, entry.obj, replace=True, **dict(entry.metadata))


class _KindCapturingResult(AnalysisResult):
    """A result whose ``__plot_spec__`` records any ``kind`` that reaches it.

    Returns a real :class:`PlotSpec` (so the fake renderer is satisfied).  If the
    accessor ever regrows a kind-forcing path, ``seen`` is how this file notices.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "meta", {})
        object.__setattr__(self, "seen", [])

    def __plot_spec__(self, kind: str | None = None) -> PlotSpec:  # noqa: D102
        resolved = PlotKind(kind) if kind is not None else PlotKind.DIAGNOSTIC_CURVE
        self.seen.append(resolved)
        return PlotSpec(kind=resolved)


def _public_accessor_methods() -> list[str]:
    """Public, non-dunder methods defined on the accessor class itself."""
    return sorted(
        n
        for n, _ in inspect.getmembers(_PlotAccessor, predicate=inspect.isfunction)
        if not n.startswith("_")
    )


# ---------------------------------------------------------------------------
# 1. No kind-forcing method exists, and none can come back
# ---------------------------------------------------------------------------


def test_the_accessor_exposes_no_kind_forcing_method() -> None:
    """``result.plot`` carries the verb and nothing else.

    A method here that named a ``PlotKind`` would be a second way to ask for a
    picture — and, as measured, a way that renames rather than redraws.
    """
    assert _public_accessor_methods() == [], _public_accessor_methods()
    for gone in sorted(_RETIRED_KIND_METHODS) + ["histogram", "spectrum"]:
        assert not hasattr(_PlotAccessor, gone), gone


def test_the_default_view_never_forces_a_kind(fake_renderer) -> None:
    """``result.plot()`` asks ``__plot_spec__`` for the result's OWN kind.

    The teeth: ``_KindCapturingResult`` records whatever ``kind`` arrives, so a
    seam that started forcing one again shows up here as a non-``None`` capture.
    """
    name, captured = fake_renderer
    result = _KindCapturingResult()
    spec = result.plot(backend=name)
    assert isinstance(spec, PlotSpec)
    assert result.seen == [PlotKind.DIAGNOSTIC_CURVE]  # the default branch, not a forced kind
    assert captured and captured[-1].kind is PlotKind.DIAGNOSTIC_CURVE


def test_kind_at_this_door_is_refused_and_names_the_transform_spelling() -> None:
    """``kind=`` was collapsed into the transform name; the refusal says so."""
    with pytest.raises(InvalidParameterError, match="collapsed the plot"):
        ScalarResult(value=1.0).plot(kind="phase_portrait_2d")


# ---------------------------------------------------------------------------
# 2-4. The namespace lists real transforms, which genuinely compute
# ---------------------------------------------------------------------------


def test_every_offered_name_is_a_transform_that_admits_this_result() -> None:
    """``result.plot.<TAB>`` is the same question ``traj.plot.<TAB>`` answers."""
    import tsdynamics as ts
    from tests._result_fixtures import build

    ts.viz.transforms.names()  # seed the registry
    offered_total = 0
    for result in build().values():
        names = dir(result.plot)
        offered_total += len(names)
        admitted = {getattr(t, "name", str(t)) for t in ts.viz.transforms.find(subject=result)}
        assert set(names) <= admitted, (type(result).__name__, names)
    # A floor, so the sweep above cannot pass vacuously on an empty registry.
    assert offered_total >= 10, offered_total


def test_a_transform_computes_where_a_relabel_only_renamed() -> None:
    """The replacement is strictly more capable, and that is measurable.

    ``dim.plot.scaling()`` used to hand back the result's own single layer under
    the name ``scaling_fit``.  The registered ``scaling_fit`` transform *builds*
    the scaling geometry — the curve, the fitted line, the window markers — so it
    returns strictly more layers than the default view.
    """
    from tests._result_fixtures import build

    dim = build()["DimensionResult"]
    default = dim.plot()
    computed = dim.plot.scaling_fit()
    assert len(computed.layers) > len(default.layers), (
        len(computed.layers),
        len(default.layers),
    )

    # ...and the extra layers carry data the default view does not have anywhere
    # in it — which is exactly what a relabel could never produce.
    def _fingerprints(spec):
        out = set()
        for layer in spec.layers:
            out.add(
                tuple(
                    (k, np.asarray(v).shape, float(np.nansum(np.asarray(v, dtype=float))))
                    for k, v in sorted(layer.data.items())
                )
            )
        return out

    assert _fingerprints(computed) - _fingerprints(default)


@pytest.mark.parametrize("retired", sorted(_RETIRED_KIND_METHODS))
def test_a_retired_kind_method_is_answered_by_name(retired) -> None:
    """The error IS the migration guide: it says what the method did, and what to type."""
    from tests._result_fixtures import build

    result = build()["DimensionResult"]
    with pytest.raises(AttributeError) as err:
        getattr(result.plot, retired)
    text = str(err.value)
    assert "forced a plot KIND" in text
    assert "result.plot()" in text  # a line that runs, for the default view


def test_a_plain_wrong_guess_is_still_an_ordinary_attribute_error() -> None:
    """``hasattr(result.plot, anything)`` must keep answering ``False``, not raise."""
    from tests._result_fixtures import build

    result = build()["DimensionResult"]
    assert not hasattr(result.plot, "totally_not_a_transform")


# ---------------------------------------------------------------------------
# 5. One vocabulary at all five plotting doors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"color": "red"},
        {"linewidth": 2.0},
        {"title": "T"},
        {"xscale": "log"},
        {"theme": "dark"},
    ],
)
def test_the_result_door_speaks_the_same_words_as_the_front_door(kwargs) -> None:
    """Measured pre-fix: every one of these raised here and worked at ``ts.plot``."""
    import tsdynamics as ts
    from tests._result_fixtures import build

    result = build()["DimensionResult"]
    assert isinstance(result.plot(**kwargs), PlotSpec)
    assert isinstance(ts.plot(result, **kwargs), PlotSpec)


def test_an_unknown_keyword_names_all_four_vocabularies() -> None:
    """A refusal has to say what the caller *may* draw from."""
    from tests._result_fixtures import build

    with pytest.raises(InvalidParameterError) as err:
        build()["DimensionResult"].plot(totally_bogus_kwarg=42)
    text = str(err.value)
    for heading in ("Spec keywords", "Style keywords", "Figure keywords", "Backend keywords"):
        assert heading in text, heading
