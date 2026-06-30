"""Regression tests for WP1: result-class arithmetic + Lyapunov-from-data/spectrum.

Covers the audit findings:

- ``A12:A12-1`` — :class:`ScalingResult` (and its ``DimensionResult`` /
  ``LyapunovFromData`` / ``ExpansionEntropyResult`` subclasses) must be a drop-in
  for its ``float`` value: comparisons and arithmetic, not only ``float()``.
- ``A12:A12-2`` — the generic scalar ``to_plot_spec`` fallback must label each
  marker with its field name (``xcategories``), not anonymous integer ticks.
- ``A1:A1-1`` — ``max_lyapunov`` must thread ``seed`` into the map engine-kernel
  off-basin random-IC retry so a retried result is reproducible.
- ``A1:A1-4`` — the ``lyapunov_from_data`` ``eps`` default must document the
  comparably-scaled-coordinate assumption (diagnosis-only).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tsdynamics.analysis.chaos.expansion import ExpansionEntropyResult
from tsdynamics.analysis.dimensions._common import DimensionResult
from tsdynamics.analysis.lyapunov import _max_lyapunov_map, lyapunov_from_data, max_lyapunov
from tsdynamics.analysis.lyapunov.from_data import LyapunovFromData

# --------------------------------------------------------------------------
# A12:A12-1 — ScalingResult numeric protocol (comparisons + arithmetic)
# --------------------------------------------------------------------------


def test_scaling_result_comparisons_and_arithmetic() -> None:
    """A DimensionResult compares and combines like its float value.

    Pre-fix ``ScalingResult`` defined only ``__float__`` (no ``_NumericOps``), so
    ``r > 2.0`` / ``r + 1.0`` / ``2 * r`` raised ``TypeError`` — the advertised
    drop-in contract was false.
    """
    r = DimensionResult(estimate=2.05, stderr=0.03, kind="correlation")
    # comparisons (the natural `dim > 2` use)
    assert r > 2.0
    assert r >= 2.05
    assert r < 3.0
    assert not (r > 3.0)
    # arithmetic (the natural `0.5 * (d1 + d2)` use)
    assert r + 1.0 == pytest.approx(3.05)
    assert 2 * r == pytest.approx(4.10)
    assert r / 2.0 == pytest.approx(1.025)
    assert abs(DimensionResult(estimate=-1.5)) == pytest.approx(1.5)
    d1 = DimensionResult(estimate=2.0)
    d2 = DimensionResult(estimate=3.0)
    assert 0.5 * (d1 + d2) == pytest.approx(2.5)
    # pytest.approx still resolves (NotImplemented → reflected op)
    assert r == pytest.approx(2.05)


def test_scaling_result_equality_and_hash_are_value_based() -> None:
    """Equality/hash track the estimate (the _NumericOps contract), not the arrays."""
    a = DimensionResult(estimate=2.05, abscissa=np.array([1.0, 2.0]))
    b = DimensionResult(estimate=2.05, abscissa=np.array([9.0, 9.0, 9.0]))
    c = DimensionResult(estimate=2.06)
    assert a == b  # same estimate, differing curve arrays
    assert a != c
    assert hash(a) == hash(2.05)
    assert a == 2.05


def test_all_scaling_subclasses_are_numeric() -> None:
    """Lyapunov-from-data and expansion-entropy results also support arithmetic."""
    lyap = LyapunovFromData(estimate=0.42, method="kantz")
    assert lyap > 0.0
    assert lyap + 1.0 == pytest.approx(1.42)
    exp = ExpansionEntropyResult(estimate=np.log(2.0), n_samples=10, n_survivors=10)
    assert exp > 0.0
    assert 2 * exp == pytest.approx(2 * np.log(2.0))


# --------------------------------------------------------------------------
# A12:A12-2 — named x-axis categories in the scalar plot fallback
# --------------------------------------------------------------------------


def test_scalar_fallback_labels_fields() -> None:
    """The generic scalar-field plot fallback names each marker (xcategories).

    Pre-fix the field names were computed and discarded, leaving anonymous
    integer ticks 0, 1, 2, … with ``xcategories=None``.
    """
    from dataclasses import dataclass
    from typing import ClassVar

    from tsdynamics.analysis._result_base import AnalysisResult

    @dataclass(frozen=True)
    class _TwoScalarResult(AnalysisResult):
        _repr_fields: ClassVar[tuple[str, ...]] = ("Sb", "Sbb")
        Sb: float = 1.2
        Sbb: float = 0.8

    spec = _TwoScalarResult().to_plot_spec()
    assert spec.x.categories == ["Sb", "Sbb"]


# --------------------------------------------------------------------------
# A1:A1-1 — seed threaded into the map engine-kernel retry
# --------------------------------------------------------------------------


def _capture_reinit_ics(monkeypatch: Any) -> list[Any]:
    """Record every ``ic`` passed to ``DiscreteMap.reinit`` (None or array)."""
    from tsdynamics.families.discrete import DiscreteMap

    captured: list[Any] = []
    original = DiscreteMap.reinit

    def spy(self: Any, u: Any = None, **kw: Any) -> None:
        captured.append(None if u is None else np.asarray(u, dtype=float).copy())
        original(self, u, **kw)

    monkeypatch.setattr(DiscreteMap, "reinit", spy)
    return captured


def _fail_first_kernel_call(monkeypatch: Any) -> None:
    """Force the engine QR kernel to diverge on its first call, then run normally."""
    from tsdynamics.engine import run
    from tsdynamics.errors import ConvergenceError

    real = run.map_lyapunov
    state = {"calls": 0}

    def flaky(*args: Any, **kw: Any) -> Any:
        state["calls"] += 1
        if state["calls"] == 1:
            raise ConvergenceError("forced off-basin divergence (test)")
        return real(*args, **kw)

    monkeypatch.setattr(run, "map_lyapunov", flaky)


def test_map_retry_ic_is_seeded_and_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    """The off-basin retry draws its IC from the seeded RNG (deterministic).

    Pre-fix the map path forwarded neither ``d0`` nor ``seed``; the retry did
    ``reinit(None)`` — an *unseeded* random draw — so a retried result was not
    reproducible even with ``seed=`` set, breaking the determinism contract the
    two-trajectory path upholds.
    """
    pytest.importorskip("tsdynamics._rust")
    from tsdynamics.systems import Henon

    seed = 7
    sys = Henon()
    dim = int(sys.dim)
    expected_retry_ic = np.random.default_rng(seed).random(dim)

    _fail_first_kernel_call(monkeypatch)
    captured = _capture_reinit_ics(monkeypatch)

    _max_lyapunov_map(sys, n=200, steps_per=5, transient=100, ic=None, seed=seed)

    # First reinit is attempt 0 (ic=None); the second is the seeded retry.
    assert captured[0] is None
    retry_ic = captured[1]
    assert retry_ic is not None
    np.testing.assert_allclose(retry_ic, expected_retry_ic)


def test_map_retry_reproducible_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two seeded runs that both hit the retry give the same retry IC."""
    pytest.importorskip("tsdynamics._rust")
    from tsdynamics.systems import Henon

    def run_once() -> Any:
        _fail_first_kernel_call(monkeypatch)
        captured = _capture_reinit_ics(monkeypatch)
        _max_lyapunov_map(Henon(), n=200, steps_per=5, transient=100, ic=None, seed=11)
        return captured[1]

    ic_a = run_once()
    monkeypatch.undo()
    ic_b = run_once()
    np.testing.assert_allclose(ic_a, ic_b)


def test_max_lyapunov_map_seed_smoke() -> None:
    """The public max_lyapunov accepts seed for a map and returns a finite value."""
    pytest.importorskip("tsdynamics._rust")
    from tsdynamics.systems import Henon

    res = max_lyapunov(Henon(), n=200, steps_per=5, transient=200, ic=[0.1, 0.1], seed=3)
    assert np.isfinite(float(res))
    assert 0.3 < float(res) < 0.6  # Hénon MLE ≈ 0.42


# --------------------------------------------------------------------------
# A1:A1-4 — documented eps anisotropy caveat (diagnosis-only)
# --------------------------------------------------------------------------


def test_lyapunov_from_data_documents_eps_scale_assumption() -> None:
    """The eps default documents the comparably-scaled-coordinate assumption."""
    doc = lyapunov_from_data.__doc__ or ""
    assert "comparably scaled" in doc
    assert "standardize" in doc or "normalize" in doc
