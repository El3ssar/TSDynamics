"""Lorenz cross-validation — exercises the F4 xval scaffold (``tests/xval_harness.py``).

The scaffold runs on Lorenz in two layers:

* **reference self-consistency** — compiler-free, always runs (this is the
  "python job" in the split CI matrix). It proves the harness plumbing — two
  backends, grid alignment, the error metric, the report — and that the metric
  *fails* loudly when trajectories genuinely disagree.
* **Rust-vs-reference** — runs only when the ``tsdynamics-core`` accelerator is
  installed (the dedicated "cross-validation job"). This is the real Rust-vs-v2
  signal: pointwise RHS lowering to ~1e-9 and an early-window trajectory match.

I-XVAL grows this into the full-catalogue removal gate (ROADMAP §9, milestone
M3); the engine streams add the new engine as another backend in the harness.
"""

from __future__ import annotations

import numpy as np
import pytest
from xval_harness import (
    RustCore,
    ScipyReference,
    XValReport,
    crossvalidate,
    crossvalidate_rhs,
)

import tsdynamics as ts
from tsdynamics.engine import rustcore as rc

_LORENZ_IC = [1.0, 1.0, 1.0]
_T_EVAL = np.linspace(0.0, 5.0, 501)
# Two correct integrators of Lorenz stay close only while Lyapunov amplification
# is small; compare on this early window (matches the rustcore numeric tests).
_EARLY = (0.0, 2.0)

_needs_rustcore = pytest.mark.skipif(
    not rc.available(), reason="tsdynamics-core (Rust accelerator) is not installed"
)


# ---------------------------------------------------------------------------
# Harness plumbing — no compiler, no accelerator. These keep the scaffold honest.
# ---------------------------------------------------------------------------


def test_report_summary_is_informative() -> None:
    """``XValReport.summary`` carries the verdict, system, and metric."""
    rep = XValReport(
        system="Lorenz",
        reference="a",
        candidate="b",
        metric="trajectory",
        n_samples=3,
        max_abs_err=1e-9,
        max_rel_err=2e-9,
        atol=1e-3,
        rtol=0.0,
        passed=True,
        window=(0.0, 2.0),
    )
    text = rep.summary()
    assert "PASS" in text and "Lorenz" in text and "trajectory" in text
    assert str(rep) == text


def test_reference_self_consistency_lorenz() -> None:
    """Two correct reference integrations agree on the early window (harness sanity)."""
    tight = ScipyReference(method="DOP853", rtol=1e-12, atol=1e-14)
    loose = ScipyReference(method="RK45", rtol=1e-9, atol=1e-11)
    rep = crossvalidate(
        ts.Lorenz(),
        reference=tight,
        candidate=loose,
        ic=_LORENZ_IC,
        t_eval=_T_EVAL,
        window=_EARLY,
        atol=1e-3,
    )
    assert rep.passed, rep.summary()
    assert rep.metric == "trajectory"
    assert rep.n_samples > 0
    assert np.isfinite(rep.max_abs_err)


def test_metric_flags_genuine_divergence() -> None:
    """A validation scaffold must FAIL (not error) when trajectories truly disagree."""
    # Over a long, fully chaotic window two integrators at very different tolerance
    # diverge by O(system size) — the report must say so, not raise or pass.
    tight = ScipyReference(method="DOP853", rtol=1e-12, atol=1e-14)
    loose = ScipyReference(method="RK45", rtol=1e-4, atol=1e-6)
    rep = crossvalidate(
        ts.Lorenz(),
        reference=tight,
        candidate=loose,
        ic=_LORENZ_IC,
        t_eval=np.linspace(0.0, 40.0, 2001),
        atol=1e-6,
    )
    assert not rep.passed
    assert rep.max_abs_err > 1e-6


def test_empty_t_eval_is_a_trivial_pass() -> None:
    """An empty grid compares zero samples and passes — no spurious error."""
    ref = ScipyReference()
    rep = crossvalidate(
        ts.Lorenz(),
        reference=ref,
        candidate=ref,
        ic=_LORENZ_IC,
        t_eval=np.array([]),
    )
    assert rep.passed
    assert rep.n_samples == 0


# ---------------------------------------------------------------------------
# The real Rust-vs-v2 signal — only with the accelerator installed.
# ---------------------------------------------------------------------------


@_needs_rustcore
def test_rustcore_rhs_matches_symbolic_lorenz() -> None:
    """The Rust tape evaluates the exact symbolic Lorenz RHS (tolerance-tight)."""
    rep = crossvalidate_rhs(ts.Lorenz(), candidate=RustCore(), n=16, seed=7)
    assert rep.passed, rep.summary()
    assert rep.metric == "rhs"
    assert rep.max_abs_err < 1e-9


@_needs_rustcore
def test_rustcore_trajectory_matches_reference_lorenz() -> None:
    """The Rust RK45 trajectory tracks the SciPy reference on the early window."""
    ref = ScipyReference(method="DOP853", rtol=1e-11, atol=1e-13)
    cand = RustCore(method="RK45", rtol=1e-10, atol=1e-12)
    rep = crossvalidate(
        ts.Lorenz(),
        reference=ref,
        candidate=cand,
        ic=_LORENZ_IC,
        t_eval=_T_EVAL,
        window=_EARLY,
        atol=1e-3,
    )
    assert rep.passed, rep.summary()
    assert rep.max_abs_err < 1e-3
