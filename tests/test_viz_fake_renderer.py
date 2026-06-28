"""Gate: every analysis result builds a valid ``PlotSpec`` (fake-renderer contract).

Stream VIZ-FALLBACK-GATE (issue #274), the contract counterpart of VIZ-TOPLOTSPEC
(issue #273).  Visualization in TSDynamics is an *intermediate representation*:
each result type produces a backend-agnostic :class:`tsdynamics.viz.spec.PlotSpec`
(``to_plot_spec``), and a future renderer consumes it — no result ever draws.
This module is the gate that keeps that contract honest:

1. **Every** concrete :class:`~tsdynamics.analysis._result.AnalysisResult`
   subclass reachable from the analysis package is constructed **synthetically**
   (tiny dummy arrays — no engine, no heavy analysis, fast tier) and its
   ``to_plot_spec()`` must either return a :class:`PlotSpec` whose ``kind`` is a
   real :class:`~tsdynamics.viz.spec.PlotKind` *or* raise the documented
   :class:`~tsdynamics.analysis._result.VisualizationNotInstalled` (the base
   fallback's "nothing numeric to draw" path).
2. The returned spec is driven through a **fake no-op renderer** to prove the
   whole seam (``.plot`` → ``PlotSpec.render`` → backend) resolves, and it
   round-trips losslessly through ``to_dict`` / ``from_dict``.
3. A registry-linked check ties it to ``registry.analyses``: every registered
   analysis whose annotated return type is an ``AnalysisResult`` subclass must be
   one of the classes this gate covers, so a new result type cannot ship without
   joining the sweep.

The gate has teeth: a result whose ``to_plot_spec`` returns ``None`` (or any
non-``PlotSpec``), an invalid kind, or that neither plots nor raises the
documented error fails here — see :func:`test_invalid_spec_is_rejected` for the
explicit negative control.  Engine-free by design (no ``tsdynamics._rust``
import).
"""

from __future__ import annotations

import contextlib
import typing

import numpy as np
import pytest

# Importing these subpackages registers their analyses AND defines every result
# subclass, so AnalysisResult.__subclasses__() below is fully populated.
import tsdynamics.analysis  # noqa: F401
from tsdynamics import registry
from tsdynamics.analysis._result import (
    AnalysisResult,
    ArrayResult,
    CollectionResult,
    CountResult,
    ScalarResult,
    ScalingResult,
    VisualizationNotInstalled,
)
from tsdynamics.analysis.basins.attractors import Attractor, AttractorSet
from tsdynamics.analysis.basins.basins import BasinFractions, BasinsResult
from tsdynamics.analysis.basins.continuation import ContinuationResult
from tsdynamics.analysis.basins.metrics import BasinEntropy, UncertaintyExponent, WadaResult
from tsdynamics.analysis.chaos.expansion import ExpansionEntropyResult
from tsdynamics.analysis.chaos.gali import GALIResult
from tsdynamics.analysis.chaos.zero_one import ZeroOneResult
from tsdynamics.analysis.dimensions._common import DimensionResult
from tsdynamics.analysis.embedding.delay import MutualInformation
from tsdynamics.analysis.embedding.dimension import EmbeddingDimension
from tsdynamics.analysis.embedding.embed import Embedding
from tsdynamics.analysis.fixedpoints.fixed import FixedPoint, FixedPointSet
from tsdynamics.analysis.fixedpoints.periodic import OrbitSet, PeriodicOrbit
from tsdynamics.analysis.lyapunov import LyapunovSpectrum
from tsdynamics.analysis.lyapunov.from_data import LyapunovFromData
from tsdynamics.analysis.orbits.orbit_diagram import OrbitDiagram
from tsdynamics.analysis.orbits.return_map import ReturnMap
from tsdynamics.analysis.recurrence.matrix import RecurrenceMatrix
from tsdynamics.analysis.recurrence.rqa import RQAResult
from tsdynamics.analysis.recurrence.windowed import WindowedRQA
from tsdynamics.analysis.surrogate.generators import SurrogateEnsemble
from tsdynamics.analysis.surrogate.hypothesis import SurrogateTest
from tsdynamics.data import Grid
from tsdynamics.viz.render.caps import VisualizationDegraded, style_honoring_gaps
from tsdynamics.viz.spec import PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# Fake no-op renderer
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_renderer():
    """Register a no-op renderer for one test, then remove it.

    The renderer just records and returns the spec it is handed — it draws
    nothing (the IR-groundwork contract: results describe, renderers draw later).
    Its presence is what makes the ``.plot`` seam actually walk through to
    ``PlotSpec.render`` instead of raising ``VisualizationNotInstalled``.
    """
    captured: list[PlotSpec] = []

    def _render(spec: PlotSpec, **_kw: object) -> PlotSpec:
        captured.append(spec)
        return spec

    name = "_fake_noop_renderer"
    registry.renderers.register(name, _render, replace=True)
    try:
        yield captured
    finally:
        if name in registry.renderers:
            registry.renderers.unregister(name)


# ---------------------------------------------------------------------------
# Synthetic result builders (tiny dummy data — engine-free, fast)
# ---------------------------------------------------------------------------


def _sparse_recurrence(n: int = 6):
    """A tiny symmetric sparse boolean recurrence matrix (no engine needed)."""
    from scipy import sparse

    i = np.array([0, 1, 2, 3], dtype=np.intp)
    j = np.array([2, 3, 4, 5], dtype=np.intp)
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.ones(rows.size, dtype=bool)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def _rqa_result() -> RQAResult:
    return RQAResult(
        recurrence_rate=0.1,
        determinism=0.8,
        laminarity=0.5,
        avg_diagonal_length=3.0,
        max_diagonal_length=7,
        divergence=1.0 / 7.0,
        diagonal_entropy=0.9,
        trapping_time=2.0,
        max_vertical_length=4,
        size=6,
        epsilon=0.5,
        theiler_window=0,
        min_diagonal=2,
        min_vertical=2,
        diagonal_lengths=np.array([2, 3, 5], dtype=float),
        vertical_lengths=np.array([2, 4], dtype=float),
    )


def _grid() -> Grid:
    return Grid(lo=np.array([-1.0, -1.0]), hi=np.array([1.0, 1.0]), counts=(4, 4))


def _attractor(aid: int = 1) -> Attractor:
    pts = np.array([[0.1, 0.2], [0.11, 0.19], [0.1, 0.2]], dtype=float)
    return Attractor(id=aid, points=pts, cells=3)


def _attractor_set() -> AttractorSet:
    return AttractorSet(attractors={1: _attractor(1), 2: _attractor(2)}, diverged=1, seeds=10)


def _basins_result() -> BasinsResult:
    labels = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1, -1, 2, 2], [1, 1, 2, 2]], dtype=int)
    return BasinsResult(labels=labels, attractors=_attractor_set(), grid=_grid())


def _continuation() -> ContinuationResult:
    values = np.linspace(0.0, 1.0, 5)
    return ContinuationResult(
        param="r",
        values=values,
        fractions={1: np.array([0.6, 0.6, 0.4, 0.2, 0.0]), 2: np.array([0.4, 0.4, 0.6, 0.8, 1.0])},
        attractors=[{} for _ in values],
        diverged=np.zeros(values.size),
    )


def _periodic_orbit(dim: int = 2, continuous: bool = False) -> PeriodicOrbit:
    pts = np.array([[0.1 * k, 0.2 * k, 0.3 * k][:dim] for k in range(4)], dtype=float)
    return PeriodicOrbit(
        points=pts,
        period=4 if not continuous else 6.6,
        multipliers=np.array([0.5, 1.5]),
        stable=False,
        continuous=continuous,
        residual=1e-9,
    )


def _fixed_point() -> FixedPoint:
    return FixedPoint(
        x=np.array([0.5, 0.5]), eigenvalues=np.array([0.4 + 0.0j, 0.9 + 0.0j]), stable=True
    )


def _scaling_curve():
    x = np.linspace(0.0, 1.0, 12)
    return x, -2.0 * x + 0.3


def _builders() -> dict[type, object]:
    """Map each result class to a zero-arg synthetic builder (tiny dummy data)."""
    x, y = _scaling_curve()
    return {
        # base wrappers
        ScalarResult: lambda: ScalarResult(value=0.42),
        CountResult: lambda: CountResult(7),
        ArrayResult: lambda: ArrayResult(values=np.array([0.42, -1.6])),
        CollectionResult: lambda: CollectionResult(items=(_fixed_point(),)),
        ScalingResult: lambda: ScalingResult(
            estimate=-2.0, stderr=0.1, abscissa=x, ordinate=y, fit_region=(2, 9), intercept=0.3
        ),
        # scaling-curve family
        DimensionResult: lambda: DimensionResult(
            estimate=2.05, stderr=0.03, abscissa=x, ordinate=y, fit_region=(2, 9), intercept=0.3
        ),
        ExpansionEntropyResult: lambda: ExpansionEntropyResult(
            estimate=0.46, stderr=0.02, abscissa=x, ordinate=y, fit_region=(2, 9), intercept=0.3
        ),
        LyapunovFromData: lambda: LyapunovFromData(
            estimate=0.9, stderr=0.05, abscissa=x, ordinate=y, fit_region=(2, 9), intercept=0.3
        ),
        # array results
        LyapunovSpectrum: lambda: LyapunovSpectrum(values=np.array([0.91, 0.0, -14.57])),
        SurrogateEnsemble: lambda: SurrogateEnsemble(
            values=np.array([[0.0, 1.0, 0.5, 0.2], [0.1, 0.9, 0.4, 0.3]]),
            meta={"method": "iaaft"},
        ),
        Embedding: lambda: Embedding(values=np.random.default_rng(0).random((20, 3))),
        # delay-selection diagnostic (mutual information first-minimum)
        MutualInformation: lambda: MutualInformation(
            values=np.array([1.5, 0.9, 0.6, 0.4, 0.5, 0.7]), meta={"analysis": "mutual_information"}
        ),
        # chaos / recurrence / surrogate dataclasses
        ZeroOneResult: lambda: ZeroOneResult(
            value=0.97,
            p=np.cumsum(np.random.default_rng(1).standard_normal(50)),
            q=np.cumsum(np.random.default_rng(2).standard_normal(50)),
        ),
        GALIResult: lambda: GALIResult(
            k=2, times=np.arange(1.0, 6.0), values=np.array([1.0, 0.6, 0.3, 0.1, 0.02])
        ),
        SurrogateTest: lambda: SurrogateTest(
            data_statistic=1.5,
            surrogate_statistics=np.array([0.1, 0.2, 0.3, 0.4]),
            p_value=0.05,
            z_score=2.1,
            rejected=True,
            statistic="time_reversal",
            method="iaaft",
            n_surrogates=4,
            tail="greater",
            alpha=0.05,
        ),
        RecurrenceMatrix: lambda: RecurrenceMatrix(matrix=_sparse_recurrence(), epsilon=0.5),
        RQAResult: _rqa_result,
        WindowedRQA: lambda: WindowedRQA(
            centers=np.array([2.5, 7.5]), results=(_rqa_result(), _rqa_result()), window=6, step=5
        ),
        # orbits
        OrbitDiagram: lambda: OrbitDiagram(
            param="r",
            values=np.linspace(2.8, 4.0, 5),
            points=[np.array([[0.5], [0.6]]) for _ in range(5)],
            components=(0,),
        ),
        ReturnMap: lambda: ReturnMap(
            current=np.array([0.1, 0.2, 0.3]),
            successor=np.array([0.2, 0.3, 0.1]),
            values=np.array([0.1, 0.2, 0.3, 0.1]),
            times=np.arange(4.0),
        ),
        # fixed points / periodic orbits
        FixedPoint: _fixed_point,
        FixedPointSet: lambda: FixedPointSet(items=(_fixed_point(),)),
        PeriodicOrbit: lambda: _periodic_orbit(dim=2, continuous=False),
        OrbitSet: lambda: OrbitSet(items=(_periodic_orbit(2, False), _periodic_orbit(2, True))),
        # embedding diagnostic
        EmbeddingDimension: lambda: EmbeddingDimension(
            dimension=3,
            dims=np.array([1, 2, 3, 4]),
            method="fnn",
            delay=1,
            fnn_fraction=np.array([0.8, 0.4, 0.05, 0.04]),
        ),
        # basins + metrics
        Attractor: _attractor,
        AttractorSet: _attractor_set,
        BasinsResult: _basins_result,
        BasinFractions: lambda: BasinFractions(
            fractions={1: 0.6, 2: 0.4}, diverged=0.0, n=100, attractors=_attractor_set()
        ),
        ContinuationResult: _continuation,
        BasinEntropy: lambda: BasinEntropy(
            sb=0.5,
            sbb=0.8,
            n_boxes=20,
            n_boundary_boxes=6,
            box_size=5,
            log_base=float(np.e),
            fractal_boundary=True,
        ),
        UncertaintyExponent: lambda: UncertaintyExponent(
            alpha=0.4,
            boundary_dimension=1.6,
            state_dimension=2,
            epsilons=np.array([0.1, 0.05, 0.025]),
            f=np.array([0.4, 0.3, 0.2]),
            r_squared=0.99,
        ),
        WadaResult: lambda: WadaResult(
            is_wada=True,
            n_basins=3,
            radii=np.array([1.0, 2.0, 3.0]),
            fractions=np.array([0.3, 0.6, 0.9]),
            n_boundary_cells=50,
            threshold=0.9,
        ),
    }


_BUILDERS = _builders()


# ---------------------------------------------------------------------------
# Coverage: every reachable AnalysisResult subclass has a synthetic builder
# ---------------------------------------------------------------------------


def _all_result_subclasses() -> set[type]:
    """Every concrete AnalysisResult subclass reachable after importing the analysis pkg."""
    seen: set[type] = set()

    def _walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                _walk(sub)

    _walk(AnalysisResult)
    # Drop private/test-only helper subclasses (names starting with "_") and any
    # defined inside test modules (their __module__ is a tests.* module).
    return {
        c for c in seen if not c.__name__.startswith("_") and not c.__module__.startswith("tests")
    }


def test_every_result_subclass_has_a_builder() -> None:
    """No AnalysisResult subclass escapes the gate (forces new results to enroll).

    If a new result type ships without a synthetic builder here, this fails — so
    the spec contract below genuinely covers the whole analysis layer rather than
    a frozen hand-list.
    """
    covered = set(_BUILDERS)
    reachable = _all_result_subclasses()
    missing = reachable - covered
    assert not missing, (
        "result subclasses with no synthetic builder in this gate: "
        f"{sorted(c.__name__ for c in missing)}"
    )


_CASES = sorted(_BUILDERS.items(), key=lambda kv: kv[0].__name__)


@pytest.mark.parametrize("cls,build", _CASES, ids=[c.__name__ for c, _ in _CASES])
def test_result_to_plot_spec_is_valid_or_documented(cls, build, fake_renderer) -> None:
    """Each result's ``to_plot_spec()`` returns a valid spec or raises the documented error.

    The real gate: build the result synthetically, call ``to_plot_spec()``, and
    require a :class:`PlotSpec` with a real :class:`PlotKind` (whose layers each
    carry a real layer-mark kind), *or* the documented
    :class:`VisualizationNotInstalled` for a result with nothing numeric to draw.
    Then drive the spec through the fake renderer and round-trip it, proving the
    full seam works without any plotting backend.
    """
    result = build()
    assert isinstance(result, cls)

    try:
        spec = result.to_plot_spec()
    except VisualizationNotInstalled:
        # The documented "no generic spec" path — acceptable per the contract.
        # But it must be a *real* raise, not a swallowed None: re-calling must
        # raise again (no hidden caching that turns it into a silent pass).
        with pytest.raises(VisualizationNotInstalled):
            result.to_plot_spec()
        return

    # Otherwise it MUST be a valid PlotSpec (not None, not some other object).
    assert isinstance(spec, PlotSpec), f"{cls.__name__}.to_plot_spec() returned {type(spec)}"
    assert isinstance(spec.kind, PlotKind)
    for layer in spec.layers:
        assert isinstance(layer.kind, PlotKind), f"{cls.__name__} layer kind invalid"
        for channel, arr in layer.data.items():
            assert isinstance(arr, np.ndarray), f"{cls.__name__} channel {channel!r} not an array"

    # The spec round-trips losslessly through the JSON-friendly dict form.
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert len(rebuilt.layers) == len(spec.layers)

    # And the whole seam resolves through a backend.  matplotlib is now the
    # deterministic default; route through the fake no-op renderer *by name* so
    # it actually receives the spec and the seam is exercised end-to-end.  The
    # fake backend declares no honored style knobs, so a spec carrying any
    # styled/palette-colored layer degrades — the dispatcher emits ONE
    # consolidated VisualizationDegraded naming the dropped knobs (which the
    # ``filterwarnings=["error"]`` suite would otherwise raise).  Expect the
    # warning exactly when the spec has gaps for this backend.
    expect_degraded = bool(style_honoring_gaps(spec, "_fake_noop_renderer"))
    ctx = pytest.warns(VisualizationDegraded) if expect_degraded else contextlib.nullcontext()
    with ctx:
        rendered = spec.render(backend="_fake_noop_renderer")
    assert rendered is fake_renderer[-1] is spec


def test_plot_seam_renders_through_fake_backend(fake_renderer) -> None:
    """``result.plot(backend=)`` routes a spec to the named backend (end-to-end seam)."""
    result = LyapunovSpectrum(values=np.array([0.91, 0.0, -14.57]))
    out = result.plot(backend="_fake_noop_renderer")
    assert isinstance(out, PlotSpec)
    assert fake_renderer and fake_renderer[-1] is out


def test_invalid_spec_is_rejected(fake_renderer) -> None:
    """Negative control: a result returning a non-spec / None must NOT pass the gate.

    Proves the gate is not a tautology — the same assertions used above reject a
    ``to_plot_spec`` that forgets to return a real :class:`PlotSpec`.
    """

    class _Broken(AnalysisResult):
        def __init__(self) -> None:
            object.__setattr__(self, "meta", {})

        def to_plot_spec(self, kind: str | None = None):  # noqa: D102
            return None  # the bug the gate must catch

    spec = _Broken().to_plot_spec()
    with pytest.raises(AssertionError):
        assert isinstance(spec, PlotSpec)


def test_base_fallback_raises_when_nothing_to_plot() -> None:
    """A result with no numeric field hits the documented base-fallback raise.

    This is the contract for "valid spec OR a documented error": a result that
    carries only non-numeric fields cannot be drawn generically, so the base
    ``to_plot_spec`` raises :class:`VisualizationNotInstalled` (caught as
    ``ImportError``) rather than returning ``None``.
    """

    from dataclasses import dataclass
    from typing import ClassVar

    @dataclass(frozen=True)
    class _LabelOnly(AnalysisResult):
        _repr_fields: ClassVar[tuple[str, ...]] = ("label",)
        label: str = "x"

    with pytest.raises(VisualizationNotInstalled):
        _LabelOnly().to_plot_spec()
    # VisualizationNotInstalled subclasses ImportError (the optional-dep idiom).
    assert issubclass(VisualizationNotInstalled, ImportError)


# ---------------------------------------------------------------------------
# Registry linkage: registered analyses returning a result are all covered
# ---------------------------------------------------------------------------


def _return_result_types(fn: object) -> tuple[type, ...]:
    """Resolved AnalysisResult subclasses in a callable's return annotation."""
    import types as _types

    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # pragma: no cover - defensive against exotic annotations
        return ()
    annotation = hints.get("return")
    if annotation is None:
        return ()
    flat: list[type] = []

    def _walk(node: object) -> None:
        origin = typing.get_origin(node)
        if origin in (typing.Union, _types.UnionType):
            for arg in typing.get_args(node):
                _walk(arg)
        elif isinstance(node, type):
            flat.append(node)

    _walk(annotation)
    return tuple(t for t in flat if issubclass(t, AnalysisResult))


def test_registered_analyses_result_types_are_covered() -> None:
    """Every registered analysis's result type is one the spec gate constructs.

    Ties the gate to ``registry.analyses``: a registered analysis that returns a
    new result class which this module does not build would slip the spec
    contract — this fails until the class is enrolled in ``_BUILDERS``.
    """
    covered = set(_BUILDERS)
    uncovered: dict[str, str] = {}
    for entry in registry.analyses.all():
        for rtype in _return_result_types(entry.obj):
            if rtype not in covered:
                uncovered[entry.name] = rtype.__name__
    assert not uncovered, f"registered analyses with uncovered result types: {uncovered}"
